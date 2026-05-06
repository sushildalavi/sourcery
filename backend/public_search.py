import logging
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from backend.utils.arxiv_utils import fetch_arxiv_candidates
from backend.utils.crossref_utils import fetch_from_crossref
from backend.utils.elsevier_utils import fetch_from_elsevier
from backend.utils.embedding_utils import embed_batch_cached, embed_query
from backend.utils.openalex_utils import fetch_candidates_from_openalex
from backend.utils.semanticscholar_utils import fetch_from_s2, fetch_seminal_from_s2
from backend.utils.springer_utils import fetch_from_springer

OPENALEX_LIMIT = int(os.getenv("PUBLIC_OPENALEX_LIMIT", "30")) or 30
ARXIV_LIMIT = int(os.getenv("PUBLIC_ARXIV_LIMIT", "30")) or 30
CROSSREF_LIMIT = int(os.getenv("PUBLIC_CROSSREF_LIMIT", "20")) or 20
S2_LIMIT = int(os.getenv("PUBLIC_S2_LIMIT", "25")) or 25
SPRINGER_LIMIT = int(os.getenv("PUBLIC_SPRINGER_LIMIT", "20")) or 20
ELSEVIER_LIMIT = int(os.getenv("PUBLIC_ELSEVIER_LIMIT", "20")) or 20
PUBLIC_SPARSE_WEIGHT = float(os.getenv("PUBLIC_SPARSE_WEIGHT", "0.35"))
PUBLIC_CORROB_MAX = float(os.getenv("PUBLIC_CORROB_MAX", "0.10"))
PUBLIC_CORROB_STEP = float(os.getenv("PUBLIC_CORROB_STEP", "0.035"))
PUBLIC_CITATION_MAX = float(os.getenv("PUBLIC_CITATION_MAX", "0.18"))
PUBLIC_CITATION_STEP = float(os.getenv("PUBLIC_CITATION_STEP", "0.02"))
PUBLIC_MIN_ABSTRACT_CHARS = int(os.getenv("PUBLIC_MIN_ABSTRACT_CHARS", "50"))
PUBLIC_MAX_PER_PROVIDER = int(os.getenv("PUBLIC_MAX_PER_PROVIDER", "3"))
PUBLIC_RELEVANCE_FLOOR_SIM = float(os.getenv("PUBLIC_RELEVANCE_FLOOR_SIM", "0.25"))
logger = logging.getLogger(__name__)
_DISABLED_PROVIDERS: set[str] = set()
_PUBLIC_SEARCH_CACHE: dict[tuple[str, str, int], tuple[float, dict]] = {}
PUBLIC_SEARCH_CACHE_TTL_SECONDS = int(os.getenv("PUBLIC_SEARCH_CACHE_TTL_SECONDS", "300")) or 300
PUBLIC_PROVIDER_MAX_WORKERS = int(os.getenv("PUBLIC_PROVIDER_MAX_WORKERS", "7")) or 7
PUBLIC_PROVIDERS = ("openalex", "arxiv", "crossref", "semanticscholar", "springer", "elsevier")
_PROVIDER_DISPLAY_PRIORITY = {
    "openalex": 0,
    "arxiv": 1,
    "springer": 2,
    "elsevier": 3,
    "semanticscholar": 4,
    "crossref": 5,
    "unknown_public": 6,
}


def _normalize_public_query(query: str) -> str:
    """
    Normalize chatty user prompts into search-friendly keyword queries.
    """
    q = (query or "").strip().lower()
    if not q:
        return ""

    # Remove common prompt wrappers/noise.
    noise_phrases = (
        "give me",
        "fetch",
        "please",
        "can you",
        "i want",
        "show me",
        "find me",
        "relevant",
        "research papers",
        "research paper",
        "papers",
        "paper",
        "from springer",
        "from elsevier",
        "from arxiv",
        "from openalex",
        "from semantic scholar",
        "only",
    )
    for p in noise_phrases:
        q = q.replace(p, " ")

    q = re.sub(r"[^a-z0-9\s-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        return ""

    stop = {
        "about",
        "info",
        "information",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "what",
        "who",
        "where",
        "when",
        "why",
        "how",
        "into",
        "using",
        "use",
    }
    toks = [t for t in q.split() if len(t) > 2 and t not in stop]
    if not toks:
        return q
    # Keep query focused but not too short.
    return " ".join(toks[:14])


def _query_variants(query: str) -> list[str]:
    """Produce search-ready variants of a user query.

    Variants (deduped, in priority order):
      1. Normalized core: stopwords stripped, noise phrases removed.
      2. Raw query as typed — preserves multi-word phrases and acronyms.
      3. Top-content-tokens only — aggressive keyword view for sparse engines.
    """
    core = _normalize_public_query(query)
    variants: list[str] = []
    if core:
        variants.append(core)
    raw = (query or "").strip()
    if raw and raw not in variants:
        variants.append(raw)
    # Keyword-only variant: top content tokens from the normalized query.
    # Helps recall on BM25/title-match providers (CrossRef, Springer).
    kw_tokens = _tokenize_for_sparse(core)
    if kw_tokens:
        keyword_variant = " ".join(kw_tokens[:8])
        if keyword_variant and keyword_variant not in variants:
            variants.append(keyword_variant)
    return variants[:3] if variants else []


def _tokenize_for_sparse(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]


def _sparse_overlap_score(query: str, text: str) -> float:
    q = _tokenize_for_sparse(query)
    if not q:
        return 0.0
    t = set(_tokenize_for_sparse(text))
    if not t:
        return 0.0
    return len({x for x in q if x in t}) / max(1, len(set(q)))


_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def _normalize_doi(raw) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    for prefix in _DOI_URL_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return s.strip(". \t\r\n")


def _normalize_title_key(raw) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    # Fold diacritics (Rényi -> Renyi) so Unicode-sensitive provider titles
    # collapse correctly: NFKD decomposes e.g. "é" into "e" + combining accent,
    # and the combining mark is then dropped by the non-word filter.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _NON_WORD_RE.sub(" ", s).strip()
    return " ".join(s.split())


def _dedupe_key_for(candidate: dict) -> str:
    """Stable cross-provider dedupe key.

    Preference order: normalized DOI -> normalized provider id -> normalized
    title. Normalization strips DOI URL prefixes, punctuation, and whitespace
    differences so e.g. "10.1109/xyz" and "https://doi.org/10.1109/xyz" collapse
    into the same key.
    """
    doi = _normalize_doi(candidate.get("doi"))
    if doi:
        return f"doi:{doi}"
    cid = str(candidate.get("id") or "").strip().lower()
    if cid:
        return f"id:{cid}"
    title_key = _normalize_title_key(candidate.get("title"))
    if title_key:
        return f"title:{title_key}"
    return ""


def _fetch_provider(provider: str, query: str, limit: int) -> list[dict]:
    if provider in _DISABLED_PROVIDERS:
        return []
    if provider == "openalex" and OPENALEX_LIMIT > 0:
        return fetch_candidates_from_openalex(query, limit=min(limit, OPENALEX_LIMIT))
    if provider == "arxiv" and ARXIV_LIMIT > 0:
        return fetch_arxiv_candidates(query, limit=min(limit, ARXIV_LIMIT))
    if provider == "crossref" and CROSSREF_LIMIT > 0:
        return fetch_from_crossref(query, limit=min(limit, CROSSREF_LIMIT))
    if provider == "semanticscholar" and S2_LIMIT > 0:
        return fetch_from_s2(query, limit=min(limit, S2_LIMIT))
    if provider == "springer" and SPRINGER_LIMIT > 0:
        return fetch_from_springer(query, limit=min(limit, SPRINGER_LIMIT))
    if provider == "elsevier" and ELSEVIER_LIMIT > 0:
        return fetch_from_elsevier(query, limit=min(limit, ELSEVIER_LIMIT))
    return []


def _provider_sort_key(provider: str) -> tuple[int, str]:
    p = (provider or "unknown_public").lower()
    return (_PROVIDER_DISPLAY_PRIORITY.get(p, 50), p)


def _provider_status_seed(provider: str) -> dict:
    needs_key = {
        "springer": "SPRINGER_API_KEY",
        "elsevier": "ELSEVIER_API_KEY",
        "semanticscholar": "SEMANTIC_SCHOLAR_API_KEY",
        "openalex": "OPENALEX_API_KEY",
    }
    key_name = needs_key.get(provider)
    available = True
    reason = None
    if key_name and not os.getenv(key_name, "").strip():
        available = False
        reason = f"missing_{key_name.lower()}"
    return {
        "available": available,
        "reason": reason,
        "queried": False,
        "variant": None,
        "fetched": 0,
        "selected": 0,
        "contributed": False,
    }


def _merge_public_candidate(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    incoming_source = (incoming.get("source") or "unknown_public").lower()
    existing_sources = merged.get("_source_providers") or [merged.get("source") or "unknown_public"]
    merged_sources = sorted({str(s).lower() for s in existing_sources} | {incoming_source}, key=_provider_sort_key)
    merged["_source_providers"] = merged_sources
    merged["source"] = merged_sources[0]

    # Keep richer text/url metadata when available.
    if not merged.get("abstract") and incoming.get("abstract"):
        merged["abstract"] = incoming.get("abstract")
    elif len(str(incoming.get("abstract") or "")) > len(str(merged.get("abstract") or "")):
        merged["abstract"] = incoming.get("abstract")
    if not merged.get("summary") and incoming.get("summary"):
        merged["summary"] = incoming.get("summary")
    if not merged.get("url") and incoming.get("url"):
        merged["url"] = incoming.get("url")
    if not merged.get("doi") and incoming.get("doi"):
        merged["doi"] = incoming.get("doi")
    if not merged.get("title") and incoming.get("title"):
        merged["title"] = incoming.get("title")
    if not merged.get("year") and incoming.get("year"):
        merged["year"] = incoming.get("year")
    # Preserve the MAX citation count across duplicate entries — different
    # providers report different counts for the same paper, and the hybrid
    # reranker's seminal-paper bump depends on this field being populated.
    for key in ("citation_count", "cited_by_count", "citationCount"):
        incoming_v = incoming.get(key)
        existing_v = merged.get(key)
        if isinstance(incoming_v, (int, float)) and incoming_v > (existing_v or 0):
            merged[key] = int(incoming_v)
    if isinstance(incoming.get("influential_citation_count"), (int, float)):
        merged["influential_citation_count"] = max(
            int(merged.get("influential_citation_count") or 0),
            int(incoming.get("influential_citation_count") or 0),
        )
    return merged


PUBLIC_MAX_VARIANTS = int(os.getenv("PUBLIC_MAX_VARIANTS", "3"))


def _variants_from_intent(intent: dict | None) -> list[str]:
    if not isinstance(intent, dict) or intent.get("fallback"):
        return []
    out: list[str] = []
    limit = max(1, PUBLIC_MAX_VARIANTS)
    for item in intent.get("search_queries", []) or []:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _seminal_seeds_from_intent(intent: dict | None, query_variants: list[str]) -> list[str]:
    """Choose seed queries for the seminal-paper S2 bulk calls.

    Returns up to 2 distinct seeds:
      1. The canonical_term from the intent resolver ("self-attention"), which
         catches papers whose title uses that exact phrasing.
      2. The first short search_query variant ("attention is all you need"),
         which catches canonical papers with a different surface form —
         Vaswani 2017 has no "self-attention" in its title, so a canonical-only
         seed misses it despite 100k+ citations.

    Empty list when nothing is specific enough to justify the extra calls.
    """
    seeds: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        if not candidate:
            return
        tokens = candidate.split()
        if not (1 <= len(tokens) <= 6):
            return
        key = candidate.lower()
        if key in seen:
            return
        seen.add(key)
        seeds.append(candidate)

    if isinstance(intent, dict) and not intent.get("fallback"):
        canonical = (intent.get("canonical_term") or "").strip()
        if canonical and len(canonical) >= 3:
            _add(canonical)
            # Hyphenated / compound canonicals often miss canonical papers
            # whose titles use the root term — "self-attention" should also
            # try "attention" (Vaswani 2017), "GAN" should also try
            # "adversarial network", etc. Take the last content-bearing
            # component so the broader term is used, not the qualifier.
            parts = re.split(r"[-\s]", canonical)
            parts = [p for p in parts if len(p) >= 4]
            if len(parts) >= 2:
                _add(parts[-1])
        for item in intent.get("search_queries", []) or []:
            if not isinstance(item, str):
                continue
            _add(item.strip())
            if len(seeds) >= 3:
                break
    if not seeds and query_variants:
        first = query_variants[0].strip()
        _add(first)

    return seeds[:3]


def _seminal_seed_from_intent(intent: dict | None, query_variants: list[str]) -> str:
    """Backward-compatible single-seed accessor. Prefer _seminal_seeds_from_intent()."""
    seeds = _seminal_seeds_from_intent(intent, query_variants)
    return seeds[0] if seeds else ""


def _embedding_query_from_intent(intent: dict | None, fallback: str) -> str:
    if isinstance(intent, dict) and not intent.get("fallback"):
        canonical = (intent.get("canonical_term") or "").strip()
        hints = [h for h in (intent.get("disambiguation_hints") or []) if isinstance(h, str) and h.strip()][:5]
        if canonical and hints:
            return f"{canonical} {' '.join(hints)}"
        if canonical:
            return canonical
        variants = _variants_from_intent(intent)
        if variants:
            return variants[0]
    return fallback


def public_live_search(
    query: str,
    k: int = 8,
    source_only: str | None = None,
    return_metadata: bool = False,
    intent: dict | None = None,
):
    """
    Fetch fresh candidates from external sources and rerank with embeddings + sparse overlap.

    When `intent` is supplied (from backend.intent_resolver.resolve_query_intent)
    and not in fallback, `intent["search_queries"]` drives provider fan-out and
    `canonical_term + disambiguation_hints` forms the embedding target used for
    ranking. When intent is missing or in fallback, the legacy `_query_variants`
    path runs unchanged.
    """
    # trivial chatty queries: skip external search
    qnorm = (query or "").strip().lower()
    if not qnorm or len(qnorm) < 3 or qnorm in {"hi", "hello", "hey", "thanks", "thank you"}:
        skip_reason = "empty_query" if not qnorm else ("query_too_short" if len(qnorm) < 3 else "greeting_only")
        return (
            {"results": [], "provider_status": {}, "skipped": {"reason": skip_reason, "normalized_query": qnorm}}
            if return_metadata
            else []
        )

    provider = (source_only or "").strip().lower()
    intent_variants = _variants_from_intent(intent)
    if intent_variants:
        query_variants = intent_variants
    else:
        query_variants = _query_variants(query)
    if not query_variants:
        return (
            {
                "results": [],
                "provider_status": {},
                "skipped": {"reason": "no_searchable_tokens", "normalized_query": qnorm},
            }
            if return_metadata
            else []
        )
    primary_query = query_variants[0]
    embedding_query = _embedding_query_from_intent(intent, primary_query)
    cache_key = (embedding_query, provider, int(k))
    cached = _PUBLIC_SEARCH_CACHE.get(cache_key)
    if cached and (time.time() - cached[0] <= PUBLIC_SEARCH_CACHE_TTL_SECONDS):
        payload = cached[1]
        if return_metadata:
            return {
                "results": list(payload.get("results", [])),
                "provider_status": dict(payload.get("provider_status", {})),
            }
        return list(payload.get("results", []))

    candidates = []
    provider_status: dict[str, dict] = {}
    if provider:
        provider_status[provider] = _provider_status_seed(provider)
        for qv in query_variants:
            rows = _fetch_provider(provider, qv, limit=max(k * 3, 12))
            provider_status[provider].update(
                {
                    "queried": True,
                    "variant": qv,
                    "fetched": max(int(provider_status[provider].get("fetched", 0) or 0), len(rows)),
                }
            )
            candidates += rows
            if len(candidates) >= max(k * 5, 20):
                break
    else:
        providers = PUBLIC_PROVIDERS
        for p in providers:
            provider_status[p] = _provider_status_seed(p)
        limit = max(k * 2, 10)
        tasks: list[tuple[str, str]] = [(p, qv) for p in providers for qv in query_variants]
        workers = min(PUBLIC_PROVIDER_MAX_WORKERS * 2, max(1, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_fetch_provider, p, qv, limit): (p, qv) for (p, qv) in tasks}
            for future in as_completed(future_map):
                p, qv = future_map[future]
                err = None
                try:
                    rows = future.result() or []
                except Exception as exc:
                    rows = []
                    err = f"{type(exc).__name__}: {exc}"
                    logger.warning("public_search provider=%s variant=%r error: %s", p, qv, err)
                prior_fetched = int(provider_status[p].get("fetched", 0) or 0)
                status_update = {
                    "queried": True,
                    "variant": qv,
                    "fetched": prior_fetched + len(rows),
                }
                if err:
                    status_update["error"] = err
                provider_status[p].update(status_update)
                candidates += rows

    # Seminal-paper boost: one extra S2 bulk call sorted by citation count on
    # the canonical term. Ensures foundational work (Vaswani 2017 for
    # "transformer", Goodfellow 2014 for "GAN", Kingma 2014 for "VAE") is in
    # the candidate pool even when relevance-ordered /search misses it by
    # returning recent preprints first. Skipped for provider-pinned queries
    # and when there's no clear canonical term to seed the query.
    seminal_seeds = _seminal_seeds_from_intent(intent, query_variants)
    if not provider and seminal_seeds:
        seminal_rows_total: list[dict] = []
        seminal_by_seed: dict[str, int] = {}
        # Run the 1-2 seminal calls in parallel; each hits S2's bulk endpoint
        # with a different title/abstract seed so a canonical paper whose title
        # doesn't match the canonical_term (e.g. "Attention Is All You Need"
        # for a "self-attention" query) still lands in the pool.
        with ThreadPoolExecutor(max_workers=len(seminal_seeds)) as seminal_pool:
            future_to_seed = {seminal_pool.submit(fetch_seminal_from_s2, seed, 4): seed for seed in seminal_seeds}
            for fut in as_completed(future_to_seed):
                seed = future_to_seed[fut]
                try:
                    rows = fut.result() or []
                except Exception as exc:
                    logger.warning("seminal S2 fetch failed for %r: %s", seed, exc)
                    rows = []
                seminal_by_seed[seed] = len(rows)
                seminal_rows_total += rows
        if seminal_rows_total:
            logger.info(
                "seminal S2 added %d candidates across seeds=%s (top cites: %s)",
                len(seminal_rows_total),
                list(seminal_by_seed.keys()),
                [r.get("citation_count") for r in seminal_rows_total[:3]],
            )
            candidates += seminal_rows_total
            ss = provider_status.setdefault("semanticscholar", _provider_status_seed("semanticscholar"))
            ss["queried"] = True
            ss["fetched"] = int(ss.get("fetched", 0) or 0) + len(seminal_rows_total)
            ss["seminal_seed"] = ", ".join(seminal_seeds)

    # dedupe by normalized DOI/id/title so the same paper from two providers
    # (e.g. Elsevier + OpenAlex) collapses into a single card instead of
    # rendering twice with different cross-provider URL styles.
    merged_by_key: dict[str, dict] = {}
    title_index: dict[str, str] = {}
    for c in candidates:
        key = _dedupe_key_for(c)
        if not key:
            continue
        current = dict(c)
        current["source"] = (current.get("source") or "unknown_public").lower()
        current["_source_providers"] = [current["source"]]
        current["_dedupe_key"] = key
        # Second-chance title match: if this candidate keyed by DOI/id but a
        # previous candidate was keyed by the same normalized title (e.g. two
        # providers where only one exposes the DOI), merge into that slot.
        title_key = _normalize_title_key(current.get("title"))
        if title_key and title_key in title_index and title_index[title_key] != key:
            existing_key = title_index[title_key]
            existing = merged_by_key.get(existing_key)
            if existing is not None:
                merged_by_key[existing_key] = _merge_public_candidate(existing, current)
                continue
        existing = merged_by_key.get(key)
        if existing is None:
            merged_by_key[key] = current
            if title_key:
                title_index.setdefault(title_key, key)
        else:
            merged_by_key[key] = _merge_public_candidate(existing, current)
    uniq = list(merged_by_key.values())

    # Keep rows with short/empty abstracts (they may still be valid metadata), but stamp
    # `_metadata_only` so downstream display can label them clearly instead of rendering
    # a phantom excerpt.
    filtered: list[dict] = []
    for c in uniq:
        body = str(c.get("abstract") or c.get("summary") or "")
        title = str(c.get("title") or "").strip()
        has_real_abstract = len(body.strip()) >= PUBLIC_MIN_ABSTRACT_CHARS
        if not has_real_abstract and not (c.get("doi") and len(title) >= 20):
            continue
        c["_metadata_only"] = not has_real_abstract
        filtered.append(c)
    uniq = filtered

    if not uniq:
        return {"results": [], "provider_status": provider_status} if return_metadata else []

    # Dynamic sparse weight for thin embedding queries: when the target reduces to a
    # single token, token-overlap dominates and rewards coincidental name matches
    # (e.g. "roberta" → any paper about a person named Roberta). Dial it down.
    effective_sparse_weight = PUBLIC_SPARSE_WEIGHT
    if len(_tokenize_for_sparse(embedding_query)) <= 1:
        effective_sparse_weight = PUBLIC_SPARSE_WEIGHT * 0.4

    texts = []
    ids = []
    sparse_vals = []
    for i, c in enumerate(uniq):
        title = (c.get("title") or "").strip()
        body = (c.get("abstract") or c.get("summary") or "").strip()
        # Double-weight the title so title-match queries score higher.
        text = f"{title}\n{title}\n{body}" if title else body
        texts.append(text)
        ids.append(i)
        sparse_vals.append(_sparse_overlap_score(embedding_query, f"{title} {body}"))

    emb_map = embed_batch_cached(list(zip([str(i) for i in ids], texts)))
    qv = embed_query(embedding_query)
    scored = []
    for i, c in enumerate(uniq):
        vec = emb_map.get(str(i))
        if vec is None:
            continue
        sim = float(np.dot(qv, vec.T)[0][0])
        sparse = float(sparse_vals[i])
        corroboration = min(PUBLIC_CORROB_MAX, PUBLIC_CORROB_STEP * max(0, len(c.get("_source_providers", [])) - 1))
        citations = 0
        for key in ("citation_count", "cited_by_count", "citationCount", "citations"):
            v = c.get(key)
            if isinstance(v, (int, float)) and v > 0:
                citations = int(v)
                break
        # Stepped citation bump — log-scaled is too gentle on the long tail.
        # A seminal paper with 50k+ citations should reliably beat a specific
        # but mid-relevance 200-citation survey, so we add a "seminal" tier.
        log_bump = min(PUBLIC_CITATION_MAX, PUBLIC_CITATION_STEP * float(np.log1p(citations)))
        if citations >= 50000:
            cite_bump = max(log_bump, 0.30)
        elif citations >= 10000:
            cite_bump = max(log_bump, 0.22)
        elif citations >= 1000:
            cite_bump = max(log_bump, 0.14)
        else:
            cite_bump = log_bump
        c["_sim"] = sim
        c["_sparse"] = sparse
        c["_cite_bump"] = round(cite_bump, 6)
        c["_citations"] = citations
        c["_hybrid"] = round(
            ((1.0 - effective_sparse_weight) * sim + effective_sparse_weight * sparse + corroboration + cite_bump),
            6,
        )
        if not c.get("source"):
            c["source"] = "unknown_public"
        scored.append(c)

    # Relevance floor: drop candidates whose semantic similarity to the embedding target
    # is below the configured threshold. Prevents thin-query lexical matches from surviving
    # when nothing in the result pool actually matches the concept.
    pre_floor_count = len(scored)
    scored = [c for c in scored if float(c.get("_sim") or 0.0) >= PUBLIC_RELEVANCE_FLOOR_SIM]
    if not scored:
        logger.info(
            "public_search relevance_floor dropped all %d candidates (embedding_query=%r, floor=%.2f)",
            pre_floor_count,
            embedding_query,
            PUBLIC_RELEVANCE_FLOOR_SIM,
        )
        skipped_meta = {
            "reason": "all_below_relevance_floor",
            "normalized_query": embedding_query,
            "floor": PUBLIC_RELEVANCE_FLOOR_SIM,
            "dropped": pre_floor_count,
        }
        if return_metadata:
            return {"results": [], "provider_status": provider_status, "skipped": skipped_meta}
        return []

    scored.sort(key=lambda x: x.get("_hybrid", x.get("_sim", 0.0)), reverse=True)
    if provider:
        final_results = scored[:k]
    else:
        # Relevance-first: pick top-scoring candidates, but soft-cap per provider so one
        # source can't monopolize the result list. If the cap starves the final pool,
        # a relaxation pass fills the remainder from the best remaining scores.
        final_results: list[dict] = []
        selected_keys: set[str] = set()
        per_provider_count: dict[str, int] = {}
        cap = max(1, PUBLIC_MAX_PER_PROVIDER)

        for row in scored:
            if len(final_results) >= k:
                break
            dkey = str(row.get("_dedupe_key") or "")
            if dkey in selected_keys:
                continue
            providers_for_row = row.get("_source_providers") or [row.get("source") or "unknown_public"]
            primary_prov = str(providers_for_row[0]).lower()
            if per_provider_count.get(primary_prov, 0) >= cap:
                continue
            final_results.append(row)
            selected_keys.add(dkey)
            per_provider_count[primary_prov] = per_provider_count.get(primary_prov, 0) + 1

        # Relaxation pass — fill any remaining slots from the top of the list without the cap.
        if len(final_results) < k:
            for row in scored:
                if len(final_results) >= k:
                    break
                dkey = str(row.get("_dedupe_key") or "")
                if dkey in selected_keys:
                    continue
                final_results.append(row)
                selected_keys.add(dkey)
    final_counts: dict[str, int] = {}
    for row in final_results:
        providers_for_row = row.get("_source_providers") or [row.get("source") or "unknown_public"]
        for src in providers_for_row:
            src_norm = str(src).lower()
            final_counts[src_norm] = final_counts.get(src_norm, 0) + 1
    for provider_name, meta in provider_status.items():
        meta["selected"] = final_counts.get(provider_name, 0)
        meta["contributed"] = meta["selected"] > 0
    logger.info(
        "public_search provider status query=%r status=%s",
        query_variants[0] if query_variants else query,
        provider_status,
    )
    _PUBLIC_SEARCH_CACHE[cache_key] = (
        time.time(),
        {"results": list(final_results), "provider_status": dict(provider_status)},
    )
    if return_metadata:
        return {"results": final_results, "provider_status": provider_status}
    return final_results
