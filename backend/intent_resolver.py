"""Domain-agnostic query intent resolver.

Runs a small GPT-4o-mini call on every public-mode query before retrieval to
extract a structured understanding of what the user is asking about. Every
downstream stage (search fan-out, embedding target for ranking, off-topic
filter) reads off the same dict — no topic, field, or entity is hardcoded.

On LLM error or when disabled, returns a fallback dict with `fallback=True`
so callers can fall back to the legacy keyword-based pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

from openai import OpenAI

from backend.utils.config import get_openai_api_key

logger = logging.getLogger(__name__)

INTENT_MODEL = os.getenv("INTENT_RESOLVER_MODEL", "gpt-4o-mini")
INTENT_TIMEOUT_SECONDS = float(os.getenv("INTENT_RESOLVER_TIMEOUT_SECONDS", "3.0"))
INTENT_MAX_OUTPUT_TOKENS = int(os.getenv("INTENT_RESOLVER_MAX_TOKENS", "300"))
INTENT_CACHE_TTL_SECONDS = int(os.getenv("INTENT_CACHE_TTL_SECONDS", "3600")) or 3600
INTENT_CACHE_MAX_ENTRIES = int(os.getenv("INTENT_CACHE_MAX_ENTRIES", "1000")) or 1000


def intent_resolver_enabled() -> bool:
    flag = os.getenv("INTENT_RESOLVER_ENABLED", "true").strip().lower()
    return flag not in {"0", "false", "no", "off"}


_INTENT_CACHE: Dict[str, tuple[float, Dict]] = {}


def _cache_get(key: str) -> Optional[Dict]:
    entry = _INTENT_CACHE.get(key)
    if not entry:
        return None
    ts, payload = entry
    if time.time() - ts > INTENT_CACHE_TTL_SECONDS:
        _INTENT_CACHE.pop(key, None)
        return None
    return dict(payload)


def _cache_put(key: str, payload: Dict) -> None:
    if len(_INTENT_CACHE) >= INTENT_CACHE_MAX_ENTRIES:
        oldest_key = min(_INTENT_CACHE.keys(), key=lambda k: _INTENT_CACHE[k][0])
        _INTENT_CACHE.pop(oldest_key, None)
    _INTENT_CACHE[key] = (time.time(), dict(payload))


def _fallback(query: str, *, reason: str, error: Optional[str] = None) -> Dict:
    return {
        "canonical_term": None,
        "domain": None,
        "scholarly_intent": True,
        "is_ambiguous": False,
        "alternative_senses": [],
        "disambiguation_hints": [],
        "search_queries": [],
        "raw_query": query,
        "fallback": True,
        "reason": reason,
        "error": error,
    }


_SYSTEM_PROMPT = (
    "You extract structured intent from scholarly search queries. The query may "
    "come from any academic field — humanities, social sciences, life sciences, "
    "physical sciences, engineering, medicine, law, business, or anything else. "
    "Do not assume a default field. Infer the most likely domain from the query text.\n\n"
    "Return strict JSON with exactly these keys:\n"
    "  canonical_term: string or null — the primary entity or concept the user is asking about.\n"
    "  domain: string or null — the inferred scholarly domain in lowercase, e.g. \"molecular biology\", \"medieval history\", \"macroeconomics\".\n"
    "  scholarly_intent: boolean — true when the query is asking about a concept, entity, method, or work that exists in scholarly literature.\n"
    "  is_ambiguous: boolean — true when canonical_term could plausibly refer to more than one real-world thing.\n"
    "  alternative_senses: array of strings — MUST contain 2 or more distinct meanings of canonical_term "
    "whenever is_ambiguous is true. Each entry is a short label like "
    "\"Transformer neural network architecture (NLP / deep learning)\" or "
    "\"Electrical power transformer (electrical engineering)\". "
    "Empty list is allowed only when is_ambiguous is false.\n"
    "  disambiguation_hints: array of 3 to 8 short domain-specific terms that, if present in a paper's title or abstract, confirm the intended scholarly sense. These guide downstream filtering.\n"
    "  search_queries: array of 2 to 4 concise search-ready query strings suitable for scholarly search APIs like Semantic Scholar or OpenAlex. Each should be 2–8 words, should not include stopwords like \"tell me about\", and should vary enough to give good recall.\n\n"
    "When is_ambiguous is true, search_queries must cover every sense listed in "
    "alternative_senses in a balanced way (at least one query per sense).\n\n"
    "Examples (keys illustrated across diverse fields):\n\n"
    "Query: \"the tanzimat reforms\"\n"
    "{\"canonical_term\":\"Tanzimat reforms\",\"domain\":\"modern history\",\"scholarly_intent\":true,\"is_ambiguous\":false,\"alternative_senses\":[],\"disambiguation_hints\":[\"Ottoman Empire\",\"19th century\",\"reorganization\",\"Abdulmejid\",\"modernization\"],\"search_queries\":[\"Tanzimat reforms Ottoman Empire\",\"Ottoman modernization 19th century\",\"Abdulmejid reform period\"]}\n\n"
    "Query: \"CRISPR-Cas9 off-target effects\"\n"
    "{\"canonical_term\":\"CRISPR-Cas9 off-target effects\",\"domain\":\"molecular biology\",\"scholarly_intent\":true,\"is_ambiguous\":false,\"alternative_senses\":[],\"disambiguation_hints\":[\"genome editing\",\"guide RNA\",\"DNA cleavage\",\"specificity\",\"Cas9 nuclease\"],\"search_queries\":[\"CRISPR-Cas9 off-target effects\",\"guide RNA specificity genome editing\",\"Cas9 nuclease mismatch tolerance\"]}\n\n"
    "Query: \"tell me about transformers\"\n"
    "{\"canonical_term\":\"transformer\",\"domain\":null,\"scholarly_intent\":true,\"is_ambiguous\":true,"
    "\"alternative_senses\":[\"Transformer neural-network architecture (NLP / deep learning)\",\"Electrical power transformer (power engineering)\",\"Transformers film or toy franchise (popular culture)\"],"
    "\"disambiguation_hints\":[\"self-attention\",\"encoder\",\"decoder\",\"BERT\",\"GPT\",\"Vaswani\",\"substation\",\"voltage\",\"power grid\"],"
    "\"search_queries\":[\"Transformer self-attention neural network\",\"Attention is all you need Vaswani 2017\",\"power transformer voltage grid\",\"transformer condition monitoring electrical\"]}\n\n"
    "Query: \"tell me about Mercury\"\n"
    "{\"canonical_term\":\"Mercury\",\"domain\":null,\"scholarly_intent\":true,\"is_ambiguous\":true,\"alternative_senses\":[\"the planet Mercury (astronomy)\",\"the element mercury Hg (chemistry)\",\"the Roman god Mercury (mythology)\",\"Freddie Mercury the musician (music / pop culture)\"],\"disambiguation_hints\":[\"planet\",\"solar system\",\"element\",\"toxicity\",\"Hg\",\"Roman god\",\"Freddie\"],\"search_queries\":[\"Mercury planet astronomy\",\"mercury element toxicity chemistry\",\"Mercury Roman mythology\",\"Freddie Mercury biography\"]}\n\n"
    "Return ONLY the JSON object. No prose, no markdown, no code fences."
)


def _client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def _parse_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    try:
        return json.loads(raw.strip())
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _coerce_str_list(value, *, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _derive_alt_senses_from_queries(
    canonical: Optional[str], search_queries: List[str]
) -> List[str]:
    """Synthesize alternative_senses from search_queries when the LLM omits them.

    Each search_query typically looks like "<canonical_term> <domain-hint-words>".
    We strip the canonical prefix and capitalize the remainder as a sense label,
    which is enough for the clarification UI to present distinct options.
    """
    if not search_queries:
        return []
    canon_lower = (canonical or "").strip().lower()
    canon_toks = {t for t in re.findall(r"[a-z0-9]+", canon_lower) if len(t) > 2}
    derived: List[str] = []
    seen: set[str] = set()
    for sq in search_queries:
        words = re.findall(r"[A-Za-z0-9\-]+", sq or "")
        remainder = [w for w in words if w.lower() not in canon_toks]
        if not remainder:
            continue
        label = " ".join(remainder).strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        pretty = f"{canonical} ({label})" if canonical else label
        derived.append(pretty)
        if len(derived) >= 4:
            break
    return derived


def _validate(payload: Dict, query: str) -> Dict:
    canonical = payload.get("canonical_term")
    if isinstance(canonical, str):
        canonical = canonical.strip() or None
    else:
        canonical = None

    domain = payload.get("domain")
    if isinstance(domain, str):
        domain = domain.strip().lower() or None
    else:
        domain = None

    scholarly_intent = bool(payload.get("scholarly_intent", True))
    is_ambiguous = bool(payload.get("is_ambiguous", False))
    alternative_senses = _coerce_str_list(payload.get("alternative_senses"), limit=8)
    disambiguation_hints = _coerce_str_list(payload.get("disambiguation_hints"), limit=8)
    search_queries = _coerce_str_list(payload.get("search_queries"), limit=4)

    # When the LLM flags ambiguity but forgets alternative_senses, derive them
    # from search_queries so the clarification UI can fire. Without this, an
    # ambiguous term like "transformers" silently collapses into whichever
    # domain happened to dominate the generated queries.
    if is_ambiguous and len(alternative_senses) < 2 and search_queries:
        derived = _derive_alt_senses_from_queries(canonical, search_queries)
        existing_keys = {s.lower() for s in alternative_senses}
        for d in derived:
            if d.lower() not in existing_keys:
                alternative_senses.append(d)
                existing_keys.add(d.lower())
        alternative_senses = alternative_senses[:8]

    return {
        "canonical_term": canonical,
        "domain": domain,
        "scholarly_intent": scholarly_intent,
        "is_ambiguous": is_ambiguous,
        "alternative_senses": alternative_senses,
        "disambiguation_hints": disambiguation_hints,
        "search_queries": search_queries,
        "raw_query": query,
        "fallback": False,
    }


def resolve_query_intent(query: str) -> Dict:
    """Return structured intent for `query`, or a fallback dict on error/disabled."""
    q = (query or "").strip()
    if not q:
        return _fallback(query or "", reason="empty-query")
    if not intent_resolver_enabled():
        return _fallback(q, reason="intent-resolver-disabled")

    cache_key = q.lower()
    cached = _cache_get(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    try:
        api_key = get_openai_api_key()
    except Exception as exc:
        return _fallback(q, reason="missing-openai-key", error=str(exc))
    if not api_key:
        return _fallback(q, reason="missing-openai-key")

    try:
        completion = _client().chat.completions.create(
            model=INTENT_MODEL,
            temperature=0.0,
            max_tokens=INTENT_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            timeout=INTENT_TIMEOUT_SECONDS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {q}"},
            ],
        )
    except Exception as exc:
        logger.warning("intent_resolver LLM call failed: %s", exc)
        return _fallback(q, reason="llm-error", error=str(exc))

    raw = (completion.choices[0].message.content or "") if completion.choices else ""
    payload = _parse_json(raw)
    if not isinstance(payload, dict):
        return _fallback(q, reason="unparseable-json", error=raw[:200])

    resolved = _validate(payload, q)
    _cache_put(cache_key, resolved)
    return resolved


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}")


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text or "") if len(tok) > 1}


def is_offtopic_by_intent(intent: Optional[Dict], citation: Dict) -> bool:
    """Off-topic filter driven by LLM-generated disambiguation hints.

    Returns True when the intent is ambiguous and the citation's title + snippet
    contains zero of the hint terms. For non-ambiguous intents we let ranking
    do its job — the relevance floor in public_search handles low-similarity
    hits.
    """
    if not isinstance(intent, dict):
        return False
    if not intent.get("is_ambiguous"):
        return False
    hints = intent.get("disambiguation_hints") or []
    if not hints:
        return False
    hay = f"{citation.get('title','')} {citation.get('snippet','')}".lower()
    if not hay.strip():
        return False
    hay_tokens = _tokenize(hay)
    for hint in hints:
        hint_norm = str(hint or "").strip().lower()
        if not hint_norm:
            continue
        if " " in hint_norm or "-" in hint_norm:
            if hint_norm in hay:
                return False
        else:
            if hint_norm in hay_tokens:
                return False
    return True
