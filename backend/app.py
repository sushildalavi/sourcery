# ------------------------------
# app.py — ScholarRAG Backend API
# ------------------------------

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from openai import OpenAI

from backend import agents, auth, chat, memory, pdf_ingest
from backend.confidence import build_confidence, score_percent
from backend.eval_metrics import aggregate_metrics
from backend.intent_resolver import is_offtopic_by_intent, resolve_query_intent
from backend.middleware import RequestIDMiddleware, SecurityHeadersMiddleware
from backend.schemas import (
    CalibrationResponse,
    EmbeddingHealthResponse,
    HealthFullResponse,
    LivenessResponse,
)
from backend.pdf_ingest import search_chunks as search_uploaded_chunks
from backend.public_search import public_live_search
from backend.public_web import public_web_search
from backend.sense_resolver import (
    expand_query_for_ml_sense,
    filter_citations_by_sense,
    is_offtopic_public_result,
    resolve_sense,
)
from backend.services.assistant_utils import (
    _append_public_source_links,
    _apply_usage_boost,
    _base_confidence,
    _build_evidence_id,
    _build_generation_prompt,
    _build_multi_doc_uploaded_summary,
    _build_public_source_listing_answer,
    _build_public_synthesis_fallback,
    _build_strict_grounded_answer,
    _build_uploaded_evidence_fallback,
    _build_uploaded_related_work_fallback,
    _chunk_query_overlap,
    _citation_coverage_stats,
    _citations_cover_specific_targets,
    _citations_support_entity_benchmark_pair,
    _citations_support_requested_metric,
    _clamp01,
    _classify_answer_mode,
    _compute_citation_msa,
    _confidence_breakdown,
    _extract_named_paper_reference,
    _has_official_company_docs,
    _humanize_answer_text,
    _is_company_intent_query,
    _is_doc_intent_query,
    _is_doc_visibility_query,
    _is_entity_level_query,
    _is_explicit_uploaded_summary_request,
    _is_general_knowledge_query,
    _is_related_work_query,
    _is_uploaded_doc_summary_query,
    _is_uploaded_key_concepts_query,
    _load_latest_calibration_weights,
    _named_paper_targets_supported,
    _needs_scope_limited_answer,
    _normalize_forward,
    _normalize_inline_citations,
    _normalize_source_url,
    _normalize_tokens,
    _primary_anchor_term,
    _prune_public_citations,
    _prune_uploaded_citations,
    _query_mentions_missing_uploaded_paper,
    _query_mentions_unseen_system,
    _query_mentions_unseen_terms,
    _query_overlap_strength,
    _query_requires_specific_grounding,
    _rank_and_trim_citations,
    _rebalance_uploaded_multi_doc_citations,
    _requested_public_source,
    _rerank_uploaded_by_query_prior,
    _resolve_effective_doc_id,
    _rewrite_ungrounded_claims,
    _scope_evidence_label,
    _scope_limited_answer,
    _source_breakdown,
    _source_scope,
    _specific_target_phrases,
    _uploaded_evidence_strength,
)
from backend.services.db import execute, fetchall, fetchone
from backend.services.embeddings import healthcheck_embeddings
from backend.services.judge import aggregate_judge_report, evaluate_faithfulness
from backend.services.nli import entailment_prob
from backend.services.research_feed import latest_research_feed
from backend.utils.config import get_openai_api_key
from backend.utils.logging_utils import log_json, setup_file_logger

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        _initialize_database_schema()
    except Exception as exc:
        logger.warning("Database schema initialization skipped: %s", exc)
    yield


# Initialize FastAPI app
app = FastAPI(
    title="ScholarRAG: Scholarly Retrieval-Augmented Generation System",
    description=(
        "Hybrid retrieval over uploaded PDFs + 7 live scholarly APIs, with "
        "MSA calibrated confidence and LLM-as-judge faithfulness scoring."
    ),
    version="1.0",
    lifespan=_lifespan,
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, and self-diagnostic endpoints."},
        {"name": "chat", "description": "Multi-turn assistant conversations."},
        {"name": "documents", "description": "Upload + manage the user's PDF corpus."},
        {"name": "search", "description": "Live aggregation across public scholarly APIs."},
        {"name": "evaluation", "description": "LLM judge runs and retrieval metrics."},
        {"name": "confidence", "description": "MSA calibration weights + scoring."},
        {"name": "metrics", "description": "Operational counters and request latencies."},
    ],
    contact={"name": "Sushil Dalavi", "email": "sushildalavi@gmail.com"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)

_cors_env = os.environ.get("CORS_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
    max_age=600,
)
# Order: outermost first. Security headers wrap everything. RequestID needs
# to wrap so the access log line includes status code from inner middlewares.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)

app.include_router(auth.router)
app.include_router(memory.router)
app.include_router(agents.router)
app.include_router(pdf_ingest.router)
app.include_router(chat.router)

RESEARCH_CHAT_MODEL = os.getenv("RESEARCH_CHAT_MODEL", "gpt-4o-mini")
OPENAI_CHAT_TIMEOUT_SECONDS = float(os.getenv("OPENAI_CHAT_TIMEOUT_SECONDS", "90") or 90)
ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "false").strip().lower() == "true"

# Fallback thresholds: only replace the LLM answer with a template when citation
# grounding is critically low.  Single-paragraph answers and answers with any inline
# citation are always preserved.
_FALLBACK_MIN_PARAGRAPHS = 3  # only enforce coverage on multi-paragraph answers
_FALLBACK_MIN_COVERAGE = 0.20  # < 20% of paragraphs cited → critically uncited


def _ensure_eval_schema() -> None:
    execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            id SERIAL PRIMARY KEY,
            name TEXT,
            scope TEXT DEFAULT 'uploaded',
            k INT DEFAULT 10,
            case_count INT DEFAULT 0,
            metrics_retrieval_only JSONB,
            metrics_retrieval_rerank JSONB,
            latency_breakdown JSONB,
            details JSONB,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )


def _ensure_msa_schema() -> None:
    execute(
        """
        CREATE TABLE IF NOT EXISTS confidence_calibration (
            id SERIAL PRIMARY KEY,
            model_name TEXT DEFAULT 'msa_logistic_v1',
            label TEXT DEFAULT 'default',
            weights JSONB NOT NULL,
            metrics JSONB,
            dataset_size INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_scores (
            id SERIAL PRIMARY KEY,
            request_id TEXT,
            sentence_id INT,
            citation_id INT,
            evidence_id TEXT,
            m_score REAL,
            s_score REAL,
            a_score REAL,
            score REAL,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_judge_runs (
            id SERIAL PRIMARY KEY,
            scope TEXT DEFAULT 'uploaded',
            query_count INT DEFAULT 0,
            metrics JSONB,
            details JSONB,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )


def _initialize_database_schema() -> None:
    pdf_ingest._ensure_doc_type_schema()
    _ensure_eval_schema()
    _ensure_msa_schema()


def _chat_answer(query: str) -> str:
    if client is None:
        return "I can help with questions, but the language model is not configured right now."
    completion = client.chat.completions.create(
        model=RESEARCH_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise assistant. Answer naturally in plain language. "
                    "Do not fabricate citations or claim to read hidden files."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.4,
        timeout=OPENAI_CHAT_TIMEOUT_SECONDS,
    )
    return (completion.choices[0].message.content or "").strip()


@app.get("/favicon.ico")
def favicon():
    """Avoid 404 spam from browsers requesting favicon."""
    return Response(status_code=204)


# ------------------------------
# Metrics endpoint (stub / cached)
# ------------------------------


@app.get("/metrics")
def metrics():
    """Real-time metrics from DB state and request logs."""
    now = datetime.utcnow().isoformat() + "Z"

    doc_count = (fetchone("SELECT count(*) AS c FROM documents WHERE status='ready'") or {}).get("c", 0)
    chunk_count = (fetchone("SELECT count(*) AS c FROM chunks") or {}).get("c", 0)
    eval_run_count = (fetchone("SELECT count(*) AS c FROM eval_runs") or {}).get("c", 0)

    latest_eval = fetchone(
        "SELECT metrics_retrieval_only, metrics_retrieval_rerank FROM eval_runs ORDER BY created_at DESC LIMIT 1"
    )
    retrieval = {}
    if latest_eval:
        m = latest_eval.get("metrics_retrieval_rerank") or latest_eval.get("metrics_retrieval_only") or {}
        if isinstance(m, str):
            m = json.loads(m)
        retrieval = {
            "recall_at_5": m.get("recall@5"),
            "ndcg_at_10": m.get("ndcg@10"),
            "mrr": m.get("mrr"),
        }

    log_path = LOG_DIR / "requests.jsonl"
    latencies: list[float] = []
    if log_path.exists():
        with log_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") == "assistant_answer" and rec.get("latency_ms") is not None:
                    latencies.append(float(rec["latency_ms"]))

    latency_stats = {}
    if latencies:
        s = sorted(latencies)
        latency_stats = {
            "p50": s[len(s) // 2],
            "p95": s[int(len(s) * 0.95)],
            "p99": s[int(len(s) * 0.99)],
        }

    return {
        "updated_at": now,
        "documents": doc_count,
        "chunks": chunk_count,
        "eval_runs": eval_run_count,
        "retrieval": retrieval,
        "latency_ms": latency_stats,
    }


# ------------------------------
# Assistant endpoint (RAG + GPT-4o-mini)
# ------------------------------


@app.post("/assistant/answer")
def assistant_answer(
    payload: dict = Body(
        ...,
        examples={
            "default": {
                "summary": "Default assistant query payload",
                "value": {
                    "query": "What does the paper really address?",
                    "scope": "uploaded",
                    "doc_id": None,
                    "k": 10,
                },
            }
        },
    ),
):
    """
    Unified QA endpoint for uploaded docs (chunk RAG) or public papers (FAISS/external).
    Returns answer plus lightweight citations.
    """
    started = time.time()
    t0 = time.perf_counter()
    query = payload.get("query") or ""
    scope = payload.get("scope") or "uploaded"
    doc_id = payload.get("doc_id")
    raw_doc_ids = payload.get("doc_ids")
    doc_ids: list[int] | None = None
    try:
        doc_id = int(doc_id) if doc_id is not None else None
    except Exception:
        doc_id = None
    if isinstance(raw_doc_ids, list):
        try:
            doc_ids = [int(x) for x in raw_doc_ids if x is not None]
        except Exception:
            doc_ids = None
    k = int(payload.get("k") or 10)
    multi_hop = bool(payload.get("multi_hop"))
    debug_confidence = bool(payload.get("debug_confidence"))
    run_judge = bool(payload.get("run_judge"))
    run_judge_llm = bool(payload.get("run_judge_llm", True))
    strict_grounding = bool(payload.get("strict_grounding"))
    allow_general_background = bool(payload.get("allow_general_background"))
    chosen_sense = payload.get("sense")
    compare_senses = bool(payload.get("compare_senses"))

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    qnorm = query.strip().lower()
    # Resolve structured query intent via the LLM resolver (domain-agnostic).
    # When the resolver is disabled or errors out, `query_intent["fallback"]`
    # is True and downstream stages revert to the legacy sense-expansion path.
    query_intent = resolve_query_intent(query)
    intent_usable = isinstance(query_intent, dict) and not query_intent.get("fallback")
    # Legacy sense expansion — kept for the fallback path and for uploaded-mode
    # effective_query derivation.
    sense_expansion = expand_query_for_ml_sense(query, scholarly_default=True)
    if intent_usable:
        canonical = (query_intent.get("canonical_term") or "").strip()
        hints = [h for h in (query_intent.get("disambiguation_hints") or []) if isinstance(h, str) and h.strip()][:5]
        if canonical and hints:
            effective_query = f"{canonical} {' '.join(hints)}"
        elif canonical:
            effective_query = canonical
        else:
            effective_query = sense_expansion["expanded_query"]
    else:
        effective_query = sense_expansion["expanded_query"]
    ambiguous_scholarly_term = bool(sense_expansion.get("term")) or bool(
        intent_usable and query_intent.get("is_ambiguous")
    )

    def _offtopic(citation: dict) -> bool:
        """Unified off-topic predicate: prefer intent-driven filter when available."""
        if intent_usable:
            return is_offtopic_by_intent(query_intent, citation)
        return is_offtopic_public_result(query, citation)

    answer_mode = _classify_answer_mode(query)
    doc_summary_intent = _is_uploaded_doc_summary_query(query)
    related_work_intent = _is_related_work_query(query)
    if not doc_ids:
        doc_id = _resolve_effective_doc_id(doc_id, scope, query)
    requested_public_source = _requested_public_source(query)
    if related_work_intent and scope == "uploaded":
        # Sense-comparison is not useful for related-work extraction from one paper.
        compare_senses = False
        chosen_sense = None
    # Heuristic routing: simple chat vs research
    small_talk_triggers = {
        "hi",
        "hello",
        "hey",
        "heyy",
        "sup",
        "ssup",
        "wassup",
        "what's up",
        "whats up",
        "yo",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "bye",
        "goodbye",
        "see ya",
        "later",
        "how are you",
        "how's it going",
        "hows it going",
        "good morning",
        "good evening",
        "good night",
        "gn",
        "gm",
        "ok",
        "okay",
        "cool",
        "nice",
        "lol",
        "lmao",
        "haha",
    }
    # Phrases that signal the user is chatting / asking for generic help rather than
    # posing a content question about the uploaded docs. "help me with my hw" is the
    # canonical failing case — no research cue, no doc reference, just vibes.
    generic_chat_phrases = (
        "help me",
        "can you help",
        "what can you do",
        "who are you",
        "what are you",
        "my hw",
        "my homework",
        "do my hw",
        "do my homework",
        "nothing much",
        "not much",
        "i'm fine",
        "im fine",
        "i am fine",
    )
    ui_help_actions = {"where", "how", "click", "button", "panel", "screen", "ui", "app"}
    ui_help_targets = {"upload", "uploaded", "document", "documents", "doc", "docs"}
    research_cues = {
        "paper",
        "study",
        "research",
        "citation",
        "doi",
        "arxiv",
        "openalex",
        "springer",
        "journal",
        "conference",
        "dataset",
        "method",
        "results",
        "conclusion",
        "abstract",
        "experiment",
    }
    # Widened small-talk detection: either (a) short + trigger word, or (b) contains
    # a canonical generic-chat phrase. "help me with my hw" (5 words, no doc terms,
    # no research cues) now correctly routes to chat.
    qtokens = qnorm.split()
    _is_small_trigger = any(t in qnorm for t in small_talk_triggers)
    _is_generic_chat = any(p in qnorm for p in generic_chat_phrases)
    is_small_talk = (len(qtokens) <= 6 and _is_small_trigger) or (len(qtokens) <= 8 and _is_generic_chat)
    # Only trigger canned UI guidance when question clearly asks "how/where to use the UI".
    # This avoids hijacking actual document questions like "can you see my attached docs?".
    is_ui_help = (
        any(t in qnorm for t in ui_help_actions)
        and any(t in qnorm for t in ui_help_targets)
        and not any(t in qnorm for t in research_cues)
    )
    is_research = any(t in qnorm for t in research_cues) or len(qnorm.split()) >= 6
    is_public_lookup = (
        _is_general_knowledge_query(query) or _is_related_work_query(query) or _is_company_intent_query(query)
    )

    if is_ui_help:
        answer = (
            "Yes, upload files in the left `Upload & Query Docs` panel using `+ Upload Source` "
            "or drag-and-drop. Wait until the file status changes from `Processing` to `Processed`, "
            "then ask in `Ask about my docs...` for doc-grounded answers. "
            "Use the right `AI Assistant` box for general/public questions."
        )
        return {"answer": answer, "citations": []}

    if scope == "uploaded" and _is_doc_visibility_query(qnorm):
        rows = fetchall("SELECT title, status FROM documents ORDER BY created_at DESC LIMIT 8")
        ready = [r for r in rows if (r.get("status") or "").lower() == "ready"]
        if ready:
            preview = ", ".join((r.get("title") or "Untitled") for r in ready[:3])
            more = f" (+{len(ready) - 3} more)" if len(ready) > 3 else ""
            answer = (
                f"Yes. I can use your uploaded documents for retrieval. "
                f"Currently processed: {preview}{more}. "
                "Ask a content question (for example: 'Summarize DES key schedule from CSCI531_Lec6.pdf')."
            )
        else:
            answer = (
                "I can’t use uploaded docs yet because none are in `Processed` state. "
                "Wait for processing to finish, then ask your question again."
            )
        return {"answer": answer, "citations": []}

    # Chat bypass: keep uploaded-doc questions in retrieval mode always, and
    # only bypass retrieval in public mode for true small-talk (greetings,
    # thanks, etc.). A non-chatty public-mode query — even a short one like
    # "tell me abut RGANs" — must go through retrieval; otherwise we end up
    # fabricating answers from the LLM's priors with no citations.
    if (
        not is_research
        and not is_public_lookup
        and scope != "uploaded"
        and not ambiguous_scholarly_term
        and (scope != "public" or is_small_talk)
    ):
        try:
            answer = _chat_answer(query)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

        log_json(
            REQUEST_LOG,
            {
                "ts": time.time(),
                "event": "assistant_answer_chat",
                "query": query,
                "scope": "chat",
                "citations": 0,
                "latency_ms": int((time.time() - started) * 1000),
            },
        )
        return {"answer": answer, "citations": []}

    if scope == "uploaded" and is_small_talk and not _is_doc_intent_query(qnorm):
        try:
            answer = _chat_answer(query)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc
        return {
            "answer": answer,
            "citations": [],
            "why_answer": {"rerank_changed_order": False, "top_chunks": []},
            "latency_breakdown_ms": {
                "retrieve": 0.0,
                "rerank": 0.0,
                "generate": 0.0,
                "total": int((time.time() - started) * 1000),
            },
            "retrieval_policy": {
                "mode": "chat-bypass",
                "uploaded_hits": 0,
                "public_hits": 0,
                "uploaded_strength": 0.0,
                "uploaded_overlap": 0.0,
                "used_public_fallback": False,
            },
        }

    # NOTE: we used to block here when scope=="uploaded" and no doc_id was selected.
    # That forced short/ambiguous queries ("tell me about Colbert") off the uploaded
    # corpus entirely, producing unrelated public-search hits. We now always search
    # the uploaded corpus across all docs and only surface a selection prompt when
    # the user really is asking for a UI-level doc action.

    # --- Query sense expansion (runs BEFORE retrieval).
    # Prevents "Colbert" / "RAG" / "BART" from being retrieved as the wrong sense.

    retrieval_ms = 0.0
    rerank_ms = 0.0
    generate_ms = 0.0

    def fetch_context(q: str, mode: str):
        local_citations = []
        if mode == "uploaded":
            # Over-fetch uploaded candidates, then rerank locally. This is critical for
            # concept-heavy queries where the right paper is not in the vector top-k
            # but is present slightly lower in the candidate list.
            candidate_k = max(int(k), min(120, int(k) * 10))
            results = search_uploaded_chunks({"q": q, "k": candidate_k, "doc_id": doc_id, "doc_ids": doc_ids})[
                "results"
            ]
            distances = [float(r.get("distance", 1.0) or 1.0) for r in results] or [1.0]
            cosines = [max(-1.0, min(1.0, 1.0 - d)) for d in distances] or [0.0]
            min_s, max_s = min(cosines), max(cosines)
            doc_counts = {}
            for r in results:
                did = r.get("document_id")
                doc_counts[did] = doc_counts.get(did, 0) + 1
            total = max(1, len(results))
            for rank, r in enumerate(results, start=1):
                dist = float(r.get("distance", 1.0) or 1.0)
                cosine = max(-1.0, min(1.0, 1.0 - dist))
                match_strength = _normalize_forward(cosine, min_s, max_s)
                support = _clamp01((doc_counts.get(r.get("document_id"), 1) - 1) / 3.0)
                conf = _base_confidence(match_strength, rank, total, support)
                conf_meta = _confidence_breakdown(match_strength, rank, total, support)
                local_citations.append(
                    {
                        "title": r.get("title") or f"Document {r.get('document_id')}",
                        "source": "uploaded",
                        "doc_id": r.get("document_id"),
                        "doc_type": r.get("doc_type") or "other",
                        "chunk_id": r.get("id"),
                        "page": r.get("page_no"),
                        "distance": r.get("distance"),
                        "sim_score": match_strength,
                        "sim_raw": round(cosine, 4),
                        "confidence": conf,
                        "_confidence_meta": conf_meta,
                        "snippet": r.get("text", ""),
                    }
                )
            local_citations = _rerank_uploaded_by_query_prior(q, local_citations)
            local_citations = _prune_uploaded_citations(q, local_citations, doc_ids=doc_ids)
            local_citations = _rerank_uploaded_by_query_prior(q, local_citations)
            for c in local_citations:
                c["source_origin"] = "uploaded"
        elif mode == "public":
            nonlocal public_provider_status
            public_resp = public_live_search(
                q,
                k=min(k, 8),
                source_only=requested_public_source,
                return_metadata=True,
                intent=query_intent if intent_usable else None,
            )
            docs = public_resp.get("results", [])
            skipped_meta = public_resp.get("skipped")
            if skipped_meta:
                public_provider_status["_skipped"] = skipped_meta
            for provider_name, meta in (public_resp.get("provider_status", {}) or {}).items():
                current = public_provider_status.get(provider_name, {})
                public_provider_status[provider_name] = {
                    "available": meta.get("available", current.get("available")),
                    "reason": meta.get("reason") or current.get("reason"),
                    "queried": True,
                    "variant": meta.get("variant") or current.get("variant"),
                    "fetched": max(int(current.get("fetched", 0) or 0), int(meta.get("fetched", 0) or 0)),
                    "selected": max(int(current.get("selected", 0) or 0), int(meta.get("selected", 0) or 0)),
                    "contributed": bool(current.get("contributed")) or bool(meta.get("contributed")),
                }
            source_count = len({(d.get("source") or d.get("venue") or "public").lower() for d in docs})
            sims = [float(d.get("_sim", 0.0) or 0.0) for d in docs] or [0.0]
            min_s, max_s = min(sims), max(sims)
            total = max(1, len(docs))
            for rank, d in enumerate(docs, start=1):
                sim = float(d.get("_sim", 0.0) or 0.0)
                match_strength = _normalize_forward(sim, min_s, max_s)
                agreement = _clamp01(source_count / 3.0)
                conf = _base_confidence(match_strength, rank, total, agreement)
                conf_meta = _confidence_breakdown(match_strength, rank, total, agreement)
                local_citations.append(
                    {
                        "title": d.get("title"),
                        "year": d.get("year"),
                        "source": d.get("source") or d.get("venue"),
                        "url": _normalize_source_url(d.get("url") or d.get("doi")),
                        "similarity": d.get("_sim"),
                        "sim_score": match_strength,
                        "sim_raw": round(sim, 4),
                        "confidence": conf,
                        "_confidence_meta": conf_meta,
                        "snippet": d.get("abstract") or d.get("summary") or "",
                        "metadata_only": bool(d.get("_metadata_only")),
                    }
                )
            if requested_public_source:
                local_citations = [
                    c for c in local_citations if (c.get("source") or "").lower() == requested_public_source
                ]
            for c in local_citations:
                c["source_origin"] = "public"
        elif mode == "web":
            docs = public_web_search(q, k=min(k, 8))
            sims = [float(d.get("_sim", 0.0) or 0.0) for d in docs] or [0.0]
            min_s, max_s = min(sims), max(sims)
            total = max(1, len(docs))
            for rank, d in enumerate(docs, start=1):
                sim = float(d.get("_sim", 0.0) or 0.0)
                match_strength = _normalize_forward(sim, min_s, max_s)
                agreement = _clamp01(1.0)
                conf = _base_confidence(match_strength, rank, total, agreement)
                conf_meta = _confidence_breakdown(match_strength, rank, total, agreement)
                local_citations.append(
                    {
                        "title": d.get("title"),
                        "year": None,
                        "source": d.get("source") or "web",
                        "url": _normalize_source_url(d.get("url")),
                        "similarity": d.get("_sim"),
                        "sim_score": match_strength,
                        "sim_raw": round(sim, 4),
                        "confidence": conf,
                        "_confidence_meta": conf_meta,
                        "snippet": d.get("snippet") or "",
                    }
                )
            for c in local_citations:
                c["source_origin"] = "web"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown retrieval mode: {mode}")
        return local_citations

    citations: list[dict] = []
    used_public_fallback = False
    uploaded_hits = 0
    public_hits = 0
    uploaded_strength = 0.0
    uploaded_overlap = 0.0
    public_provider_status: dict[str, dict] = {}

    if scope == "uploaded":
        if multi_hop and (" and " in query or ";" in query or "," in query):
            subqs = [q.strip() for q in re.split(r"and|;|,", query) if q.strip()]
            for sq in subqs:
                citations.extend(fetch_context(sq, "uploaded"))
        else:
            citations.extend(fetch_context(effective_query, "uploaded"))

        # Dedicated recall boost for "related/similar work" requests on uploaded docs.
        if related_work_intent:
            related_probe = (
                "related work prior work baseline comparison compared with previous studies "
                "limitations future work " + query
            )
            citations.extend(fetch_context(related_probe, "uploaded"))

        explicit_uploaded_summary = _is_explicit_uploaded_summary_request(query)
        if (
            allow_general_background
            and not explicit_uploaded_summary
            and (_is_general_knowledge_query(query) or _is_company_intent_query(query))
        ):
            public_citations = fetch_context(effective_query, "public")
            public_citations = _prune_public_citations(query, public_citations)
            public_citations = [c for c in public_citations if not _offtopic(c)]
            if not public_citations and ENABLE_WEB_FALLBACK:
                public_citations = fetch_context(effective_query, "web")
            if public_citations:
                used_public_fallback = True
                public_hits = len(public_citations)
                # In general-background mode, prefer public/web evidence as primary context.
                citations = public_citations + citations

        uploaded_hits = len(citations)
        uploaded_strength = _uploaded_evidence_strength(citations)
        uploaded_overlap = _query_overlap_strength(query, citations)
        strong_uploaded_match = uploaded_overlap >= 0.6 and uploaded_hits >= 1
        weak_uploaded = not strong_uploaded_match and (
            uploaded_overlap < 0.22 or (uploaded_hits < 3 and uploaded_strength < 0.52)
        )
        if (
            weak_uploaded
            and allow_general_background
            and not explicit_uploaded_summary
            and answer_mode in {"research_synthesis", "source_listing"}
        ):
            used_public_fallback = True
            public_citations = fetch_context(effective_query, "public")
            public_citations = _prune_public_citations(query, public_citations)
            public_citations = [c for c in public_citations if not _offtopic(c)]
            public_hits = len(public_citations)
            if public_hits > 0:
                # Keep uploaded citations first in uploaded mode.
                citations = citations + public_citations
    else:
        # Public scope with adaptive hybrid. The uploaded corpus is ALWAYS
        # probed when the user has not pinned a specific document. Each
        # uploaded chunk is kept only if it has plausible per-chunk overlap
        # with the query (>= UPLOADED_CHUNK_OVERLAP_FLOOR). The merged pool
        # (kept uploaded chunks + public chunks) is capped at k*2 candidates
        # and downstream scoring in _rank_and_trim_citations handles final
        # ordering via source prior + semantic sim + query overlap.
        #
        # Pinned-doc guard: if the user pinned a specific doc_id, we respect
        # the public scope switch and do NOT probe the pinned doc (it would
        # trivially beat any floor and flood the answer with irrelevant
        # citations from a paper the user did not ask about).
        UPLOADED_CHUNK_OVERLAP_FLOOR = 0.08

        probe_uploaded = not doc_id
        uploaded_probe: list[dict] = []
        if probe_uploaded:
            raw_uploaded = fetch_context(effective_query, "uploaded")
            # Per-chunk overlap floor: keep only chunks that plausibly match
            # the query. This replaces the prior aggregate-threshold gate
            # that silently dropped borderline-relevant uploaded papers.
            uploaded_probe = [c for c in raw_uploaded if _chunk_query_overlap(query, c) >= UPLOADED_CHUNK_OVERLAP_FLOOR]
            # Concept gate: when the intent resolver gave us a canonical term
            # and/or disambiguation hints, require the uploaded chunk to
            # mention at least one of them. Prevents a generic BERT/attention
            # chunk from leaking into an unrelated public query (e.g. RGAN).
            if intent_usable:
                canonical = (query_intent.get("canonical_term") or "").strip().lower()
                hints = [
                    str(h).strip().lower()
                    for h in (query_intent.get("disambiguation_hints") or [])
                    if isinstance(h, str) and h.strip()
                ]
                concept_terms: list[str] = []
                if canonical:
                    concept_terms.append(canonical)
                concept_terms.extend(hints)
                if concept_terms:
                    filtered = []
                    for c in uploaded_probe:
                        hay = f"{c.get('title','')} {c.get('snippet','')}".lower()
                        if any(term and term in hay for term in concept_terms):
                            filtered.append(c)
                    uploaded_probe = filtered
            uploaded_hits = len(uploaded_probe)
            uploaded_strength = _uploaded_evidence_strength(uploaded_probe)
            uploaded_overlap = _query_overlap_strength(query, uploaded_probe)

        if multi_hop and (" and " in query or ";" in query or "," in query):
            subqs = [q.strip() for q in re.split(r"and|;|,", query) if q.strip()]
            for sq in subqs:
                citations.extend(fetch_context(sq, "public"))
        else:
            citations.extend(fetch_context(effective_query, "public"))

        # Domain prior: drop public hits that are obviously the wrong sense
        # (e.g., person-named-Roberta papers when the user asked about RoBERTa
        # the language model). Uses the LLM-generated disambiguation_hints
        # when intent is available, else the curated sense-keyword table.
        citations = [c for c in citations if not _offtopic(c)]
        public_hits = len(citations)

        # Merge uploaded candidates into the pool. Cap at k*3 (was k*2) for
        # public scope so the downstream ranker has more candidates to work
        # with — public retrieval now fetches ~3x more per provider, and the
        # old k*2 cap was starving the reranker. Uploaded chunks are placed
        # first so ranker ties break toward the user's own corpus, but final
        # order comes from _rank_and_trim_citations.
        # Cap uploaded chunks at ~1/3 of k in public mode — otherwise a
        # permissive keyword-match on uploaded docs (e.g. any paper mentioning
        # "transformer") can crowd out the seminal public papers the user
        # explicitly switched to public mode to find.
        if uploaded_probe:
            uploaded_cap_public = max(1, int(k) // 3)
            uploaded_probe = uploaded_probe[:uploaded_cap_public]
            merged = uploaded_probe + citations
            pool_cap = max(int(k) * 3, len(uploaded_probe) + int(k) * 2)
            citations = merged[:pool_cap]
            used_public_fallback = True  # signals that we blended scopes

    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Abstention guard: if after sense-expansion + domain-prior filtering we have
    # either no citations, or only citations whose lexical overlap with the
    # query is vanishingly low, refuse instead of producing a confident
    # hallucination. This is the behavior we want for ambiguous queries like
    # "tell me about Colbert" when neither the uploaded corpus nor public
    # sources surface a ColBERT-the-model match.
    if citations:
        post_filter_overlap = _query_overlap_strength(query, citations)
    else:
        post_filter_overlap = 0.0

    # Short queries (e.g. "tell me about RGANs", "what is BERT") often reduce
    # to a single content token after stopword removal, making lexical-overlap
    # an unreliable abstention signal. When we have real retrieval hits and the
    # query is that short, trust the semantic retriever.
    query_content_tokens = _normalize_tokens(query)
    short_query_exempt = len(query_content_tokens) <= 1 and len(citations) > 0

    missing_named_paper = _query_mentions_missing_uploaded_paper(query)
    specific_targets = _specific_target_phrases(query)
    lacks_specific_support = (
        _query_requires_specific_grounding(query)
        and bool(specific_targets)
        and not _citations_cover_specific_targets(citations, specific_targets)
    )
    lacks_named_paper_target_support = bool(specific_targets) and not _named_paper_targets_supported(
        query, citations, specific_targets
    )
    missing_exact_metric = not _citations_support_requested_metric(query, citations)
    missing_entity_benchmark_pair = not _citations_support_entity_benchmark_pair(query, citations)
    unseen_system_reference = _query_mentions_unseen_system(query, citations)
    unseen_term_reference = _query_mentions_unseen_terms(query, citations)
    strict_exact_metric_query = "benchmark" in query.lower() and (
        "exact value" in query.lower() or "exact score" in query.lower()
    )

    if (
        not citations
        or (post_filter_overlap < 0.05 and not doc_id and not doc_ids and not short_query_exempt)
        or (missing_named_paper and not doc_id and not doc_ids)
        or (lacks_specific_support and not doc_id and not doc_ids and not short_query_exempt)
        or (lacks_named_paper_target_support and not doc_id and not doc_ids and not short_query_exempt)
        or (missing_exact_metric and not doc_id and not doc_ids)
        or (missing_entity_benchmark_pair and not doc_id and not doc_ids)
        or (strict_exact_metric_query and not doc_id and not doc_ids)
        or (unseen_system_reference and not doc_id and not doc_ids and not short_query_exempt)
        or (unseen_term_reference and not doc_id and not doc_ids and not short_query_exempt)
    ):
        evidence_label = _scope_evidence_label(scope)
        sense_hint = ""
        if sense_expansion.get("term") and sense_expansion.get("ml_sense"):
            sense_hint = (
                f" If you meant **{sense_expansion['ml_sense']}**, try asking with "
                f"more context (for example the paper title, model architecture, or a "
                f"related term like 'retrieval', 'BERT', or 'dual encoder')."
            )
        if requested_public_source and scope != "uploaded":
            provider = requested_public_source.upper()
            return {
                "answer": (
                    f"I couldn't find relevant material from {provider} for that query."
                    f"{sense_hint} Try a more specific topic (keywords, year range, or exact paper title)."
                ),
                "citations": [],
                "confidence": {
                    "score": 0.15,
                    "label": "Abstained (insufficient evidence)",
                    "needs_clarification": True,
                },
                "retrieval_policy": {
                    "mode": "abstention",
                    "reason": "no-relevant-match",
                    "uploaded_hits": uploaded_hits,
                    "public_hits": public_hits,
                    "post_filter_overlap": round(post_filter_overlap, 3),
                    "sense_expansion": sense_expansion,
                    "public_provider_status": public_provider_status,
                },
            }
        return {
            "answer": (
                f"I couldn't find reliable matching evidence in your {evidence_label} for that query."
                f"{sense_hint} Rather than guess, I'll wait for a clearer question."
            ),
            "citations": [],
            "confidence": {"score": 0.15, "label": "Abstained (insufficient evidence)", "needs_clarification": True},
            "retrieval_policy": {
                "mode": "abstention",
                "reason": (
                    "missing-named-paper"
                    if missing_named_paper
                    else "unseen-system-reference"
                    if unseen_system_reference
                    else "missing-exact-metric"
                    if missing_exact_metric
                    else "missing-named-paper-target-support"
                    if lacks_named_paper_target_support
                    else "missing-entity-benchmark-pair"
                    if missing_entity_benchmark_pair
                    else "strict-exact-metric-query"
                    if strict_exact_metric_query
                    else "missing-unseen-term-reference"
                    if unseen_term_reference
                    else "missing-specific-support"
                    if lacks_specific_support
                    else "no-relevant-match"
                    if not citations
                    else "low-lexical-overlap"
                ),
                "uploaded_hits": uploaded_hits,
                "public_hits": public_hits,
                "post_filter_overlap": round(post_filter_overlap, 3),
                "specific_targets": specific_targets,
                "named_paper_reference": _extract_named_paper_reference(query),
                "sense_expansion": sense_expansion,
                "public_provider_status": public_provider_status,
            },
        }

    entity_query = _is_entity_level_query(query) and not doc_summary_intent
    all_personal_resume = all(
        ((c.get("doc_type") in {"resume"}) or (_source_scope(c) == "personal_profile")) for c in citations
    )
    no_official_docs = not _has_official_company_docs()
    if entity_query and all_personal_resume and no_official_docs:
        entity = _primary_anchor_term(query) or "this entity"
        return {
            "answer": (
                f"I only found references to {entity} within a personal document (for example a resume). "
                "I do not have broader company-level documentation in your uploaded sources. "
                "Would you like a summary of the resume context or allow general background knowledge?"
            ),
            "citations": [],
            "needs_clarification": True,
            "clarification": {
                "question": "Choose answer scope:",
                "options": ["Resume context summary", "Public company overview"],
                "recommended_option": "Public company overview"
                if allow_general_background
                else "Resume context summary",
                "rationale": "Entity-level query with only personal-document evidence.",
                "term": entity,
            },
            "confidence": {
                "score": 0.18,
                "label": "Context-limited",
                "needs_clarification": True,
                "factors": {
                    "top_sim": 0.0,
                    "top_rerank_norm": 0.0,
                    "citation_coverage": 0.0,
                    "evidence_margin": 0.0,
                    "ambiguity_penalty": 0.0,
                    "insufficiency_penalty": 0.35,
                    "scope_penalty": 1.0,
                },
                "explanation": "Only personal-document context is available for an entity-level/company query.",
            },
            "answer_scope": "personal_document_context",
            "unsupported_claims": 0,
            "why_answer": {"rerank_changed_order": False, "top_chunks": []},
            "latency_breakdown_ms": {
                "retrieve": round(retrieval_ms, 2),
                "rerank": 0.0,
                "generate": 0.0,
                "total": int((time.time() - started) * 1000),
            },
            "retrieval_policy": {
                "mode": "entity-scope-guard",
                "uploaded_hits": uploaded_hits,
                "public_hits": public_hits,
                "uploaded_strength": uploaded_strength,
                "uploaded_overlap": uploaded_overlap,
                "used_public_fallback": used_public_fallback,
            },
        }

    if (
        scope == "uploaded"
        and allow_general_background
        and not _is_explicit_uploaded_summary_request(query)
        and _is_general_knowledge_query(query)
    ):
        primary = _primary_anchor_term(query)
        has_public_primary = any(
            (c.get("source") or "").lower() != "uploaded"
            and (primary in f"{c.get('title','')} {c.get('snippet','')}".lower() if primary else True)
            for c in citations
        )
        if not has_public_primary:
            return {
                "answer": (
                    "I couldn’t find reliable public/web evidence for that specific entity/topic. "
                    "Please refine the query (for example include official company name/ticker) or provide a trusted source."
                ),
                "citations": [],
                "needs_clarification": True,
                "clarification": {
                    "question": "Can you provide a more specific public identifier (official name, ticker, or domain)?",
                    "options": [],
                    "recommended_option": None,
                    "rationale": "No public/web evidence matched the primary entity anchor.",
                    "term": primary,
                },
                "confidence": {
                    "score": 0.1,
                    "label": "Low",
                    "needs_clarification": True,
                    "factors": {
                        "top_sim": 0.0,
                        "top_rerank_norm": 0.0,
                        "citation_coverage": 0.0,
                        "evidence_margin": 0.0,
                        "ambiguity_penalty": 0.0,
                        "insufficiency_penalty": 1.0,
                    },
                    "explanation": "No public/web source matched the primary entity anchor.",
                },
                "why_answer": {"rerank_changed_order": False, "top_chunks": []},
                "latency_breakdown_ms": {
                    "retrieve": round(retrieval_ms, 2),
                    "rerank": 0.0,
                    "generate": 0.0,
                    "total": int((time.time() - started) * 1000),
                },
                "retrieval_policy": {
                    "mode": "general-background-anchor-guard",
                    "uploaded_hits": uploaded_hits,
                    "public_hits": public_hits,
                    "uploaded_strength": uploaded_strength,
                    "uploaded_overlap": uploaded_overlap,
                    "used_public_fallback": used_public_fallback,
                },
            }

    rerank_start = time.perf_counter()
    prefer_public = (
        scope == "uploaded"
        and allow_general_background
        and (_is_general_knowledge_query(query) or _is_company_intent_query(query))
    )
    citations = _rank_and_trim_citations(
        query,
        citations,
        k,
        prefer_public=prefer_public,
        doc_ids=doc_ids if scope == "uploaded" else None,
    )
    if prefer_public:
        public_only = [c for c in citations if (c.get("source") or "").lower() != "uploaded"]
        if public_only:
            citations = public_only[:k]
    # For definition-style questions, don't get stuck on resume/course-only evidence:
    # automatically try public evidence once before forcing a scope-limited response.
    if (
        scope == "uploaded"
        and not _is_explicit_uploaded_summary_request(query)
        and _needs_scope_limited_answer(query, citations)
    ):
        public_citations = fetch_context(query, "public")
        public_citations = _prune_public_citations(query, public_citations)
        if not public_citations and ENABLE_WEB_FALLBACK:
            # For non-academic entity questions, fall back to general web summaries.
            public_citations = fetch_context(query, "web")
        if public_citations:
            used_public_fallback = True
            public_hits = len(public_citations)
            citations = _rank_and_trim_citations(
                query,
                public_citations + citations,
                k,
                prefer_public=prefer_public,
                doc_ids=doc_ids if scope == "uploaded" else None,
            )
    rerank_ms = (time.perf_counter() - rerank_start) * 1000

    # Intent-driven clarification: when the LLM resolver flags the query as
    # ambiguous (e.g. "tell me about Mercury" → planet / element / deity / musician),
    # surface the same clarification UI used by the legacy sense-resolver path.
    # This works for any domain, not just the hand-curated AMBIGUOUS_TERMS.
    if (
        intent_usable
        and query_intent.get("is_ambiguous")
        and not compare_senses
        and not chosen_sense
        and not related_work_intent
    ):
        canonical = (query_intent.get("canonical_term") or "").strip()
        alt_senses = [s for s in (query_intent.get("alternative_senses") or []) if isinstance(s, str) and s.strip()]
        domain = (query_intent.get("domain") or "").strip()
        primary_sense = f"{canonical} ({domain})" if canonical and domain else canonical

        # If the alternative_senses already include specific variants of the
        # canonical term (e.g. "Recurrent GAN", "Robust GAN" for canonical="RGAN"),
        # adding "RGAN (machine learning)" as a separate option is just "any of
        # the above" and adds noise. Drop the primary option in that case.
        def _shares_tokens(a: str, b: str) -> bool:
            tok_a = {t for t in re.findall(r"[a-z0-9]+", a.lower()) if len(t) > 2}
            tok_b = {t for t in re.findall(r"[a-z0-9]+", b.lower()) if len(t) > 2}
            return bool(tok_a & tok_b)

        primary_is_redundant = bool(
            primary_sense and alt_senses and all(_shares_tokens(primary_sense, alt) for alt in alt_senses)
        )
        options = (
            alt_senses if primary_is_redundant else (([primary_sense] + alt_senses) if primary_sense else alt_senses)
        )
        if len(options) >= 2:
            return {
                "answer": "",
                "citations": [],
                "confidence": {
                    "score": 0.2,
                    "label": "Low",
                    "needs_clarification": True,
                    "factors": {
                        "top_sim": 0.0,
                        "top_rerank_norm": 0.0,
                        "citation_coverage": 0.0,
                        "evidence_margin": 0.0,
                        "ambiguity_penalty": 1.0,
                        "insufficiency_penalty": 0.0,
                    },
                    "explanation": "Query is ambiguous; GPT-4o-mini intent resolver flagged multiple plausible senses.",
                },
                "why_answer": {"rerank_changed_order": False, "top_chunks": []},
                "needs_clarification": True,
                "clarification": {
                    "question": f'"{canonical or query}" could mean a few different things. Which do you mean?',
                    "options": options,
                    # Recommended option must be among the rendered buttons; when
                    # the primary sense was dropped as redundant, fall back to
                    # the first alternative.
                    "recommended_option": (options[0] if primary_is_redundant or not primary_sense else primary_sense),
                    "rationale": (
                        f"Intent resolver detected ambiguity. Primary sense inferred as {primary_sense!r}; "
                        f"alternatives: {alt_senses!r}."
                    ),
                    "term": canonical or None,
                },
                "latency_breakdown_ms": {
                    "retrieve": round(retrieval_ms, 2),
                    "rerank": round(rerank_ms, 2),
                    "generate": 0.0,
                    "total": int((time.time() - started) * 1000),
                },
                "retrieval_policy": {
                    "mode": "intent-clarification",
                    "uploaded_hits": uploaded_hits,
                    "public_hits": public_hits,
                    "uploaded_strength": uploaded_strength,
                    "uploaded_overlap": uploaded_overlap,
                    "used_public_fallback": used_public_fallback,
                    "query_intent": {
                        "canonical_term": query_intent.get("canonical_term"),
                        "domain": query_intent.get("domain"),
                        "is_ambiguous": True,
                        "alternative_senses": alt_senses,
                        "model": "gpt-4o-mini",
                    },
                },
            }

    sense = resolve_sense(query, citations, chosen_sense=chosen_sense)
    if sense.get("is_ambiguous") and not compare_senses and not chosen_sense and not related_work_intent:
        return {
            "answer": "",
            "citations": [],
            "confidence": {
                "score": 0.2,
                "label": "Low",
                "needs_clarification": True,
                "factors": {
                    "top_sim": 0.0,
                    "top_rerank_norm": 0.0,
                    "citation_coverage": 0.0,
                    "evidence_margin": 0.0,
                    "ambiguity_penalty": 1.0,
                    "insufficiency_penalty": 0.0,
                },
                "explanation": "Query needs sense clarification before a grounded answer can be generated.",
            },
            "why_answer": {"rerank_changed_order": False, "top_chunks": []},
            "needs_clarification": True,
            "clarification": {
                "question": f"Do you mean {', '.join(sense.get('options', []))}?",
                "options": sense.get("options", []),
                "recommended_option": sense.get("recommended_option"),
                "rationale": sense.get("rationale"),
                "term": sense.get("term"),
            },
            "latency_breakdown_ms": {
                "retrieve": round(retrieval_ms, 2),
                "rerank": round(rerank_ms, 2),
                "generate": 0.0,
                "total": int((time.time() - started) * 1000),
            },
            "retrieval_policy": {
                "mode": "sense-resolver",
                "uploaded_hits": uploaded_hits,
                "public_hits": public_hits,
                "uploaded_strength": uploaded_strength,
                "uploaded_overlap": uploaded_overlap,
                "used_public_fallback": used_public_fallback,
            },
        }
    if chosen_sense and not compare_senses:
        citations = filter_citations_by_sense(citations, chosen_sense)

    if _is_company_intent_query(query) and not doc_summary_intent:
        has_public = any((c.get("source") or "").lower() != "uploaded" for c in citations)
        has_profile = any((_source_scope(c) == "personal_profile") for c in citations)
        if has_profile and not has_public:
            return {
                "answer": (
                    "I only found company mentions in profile/resume context in your uploaded documents. "
                    "I don’t have reliable public evidence here to provide a company-level overview."
                ),
                "citations": [],
                "needs_clarification": True,
                "clarification": {
                    "question": "Do you want a profile-scoped summary from your docs, or a public company overview?",
                    "options": ["Profile-scoped summary", "Public company overview"],
                    "recommended_option": "Public company overview"
                    if allow_general_background
                    else "Profile-scoped summary",
                    "rationale": "Company intent detected but evidence is only personal profile context.",
                    "term": _primary_anchor_term(query),
                },
                "confidence": {
                    "score": 0.12,
                    "label": "Low",
                    "needs_clarification": True,
                    "factors": {
                        "top_sim": 0.0,
                        "top_rerank_norm": 0.0,
                        "citation_coverage": 0.0,
                        "evidence_margin": 0.0,
                        "ambiguity_penalty": 0.0,
                        "insufficiency_penalty": 1.0,
                    },
                    "explanation": "Detected company-intent query but only profile-scoped evidence was retrieved.",
                },
                "why_answer": {"rerank_changed_order": False, "top_chunks": []},
                "latency_breakdown_ms": {
                    "retrieve": round(retrieval_ms, 2),
                    "rerank": round(rerank_ms, 2),
                    "generate": 0.0,
                    "total": int((time.time() - started) * 1000),
                },
                "retrieval_policy": {
                    "mode": "company-intent-guard",
                    "uploaded_hits": uploaded_hits,
                    "public_hits": public_hits,
                    "uploaded_strength": uploaded_strength,
                    "uploaded_overlap": uploaded_overlap,
                    "used_public_fallback": used_public_fallback,
                },
            }

    context_lines = []
    for i, c in enumerate(citations, start=1):
        before_rank = int(c.get("initial_rank", i) or i)
        c["id"] = i
        c["rank_before"] = before_rank
        c["rank_after"] = i
        c["rank_delta"] = before_rank - i
        c["evidence_id"] = c.get("evidence_id") or _build_evidence_id(c)
        c["scope"] = _source_scope(c)
        c["rerank_raw"] = round(float(c.get("rerank_raw", _chunk_query_overlap(query, c)) or 0.0), 4)
        c["rerank_norm"] = round(float(c.get("rerank_norm", c.get("rerank_raw", 0.0)) or 0.0), 4)
        conf = float(c.get("confidence", 0.5))
        if c.get("source") == "uploaded":
            context_lines.append(
                f"[S{i}] doc {c.get('doc_id')} chunk {c.get('chunk_id')} page {c.get('page','?')} "
                f"(scope={c.get('scope')}, confidence={conf:.2f}): "
                f"{c.get('snippet','')}"
            )
        else:
            context_lines.append(
                f"[S{i}] {c.get('title','')} (scope={c.get('scope')}, confidence={conf:.2f}): {c.get('snippet','')}"
            )

    context = "\n\n".join(context_lines)
    compare_instruction = ""
    if compare_senses and sense.get("options"):
        compare_instruction = (
            "10) Compare senses mode is enabled. If multiple senses exist, write separate sections for each sense "
            f"from these options: {', '.join(sense.get('options', []))}. "
            "Do not merge senses in the same paragraph.\n"
        )
    if scope == "uploaded" and _is_uploaded_key_concepts_query(query):
        compare_instruction += (
            "11) This is a key-concepts extraction request over uploaded documents. Organize the answer by document "
            "using `##` headings. Under each document, extract only evidence-backed bullets for: core skills/topics, "
            "standout projects or claims, and tools/technologies if explicitly present. "
            "If multiple documents are selected, end with `## Shared themes` and `## Distinctive differences`.\n"
        )
    if scope == "uploaded" and doc_ids and len(doc_ids) > 1:
        compare_instruction += (
            "12) Multiple uploaded documents are selected. Synthesize across the selected documents and call out "
            "document-specific differences when the evidence supports them. Do not answer from only one document "
            "if multiple selected documents contain relevant evidence.\n"
        )

    prompt = _build_generation_prompt(
        query=query,
        context=context,
        answer_mode=answer_mode,
        allow_general_background=allow_general_background,
        compare_instruction=compare_instruction,
    )

    if strict_grounding:
        answer = _build_strict_grounded_answer(query, citations, scope, answer_mode)
        generate_ms = 0.0
    else:
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="OpenAI client not configured. Set OPENAI_API_KEY (and install python-dotenv if relying on .env).",
            )

        gen_start = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=RESEARCH_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=OPENAI_CHAT_TIMEOUT_SECONDS,
            )
            answer = completion.choices[0].message.content or ""
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc
        generate_ms = (time.perf_counter() - gen_start) * 1000

    answer = _normalize_inline_citations(answer)
    answer = _humanize_answer_text(answer)
    # Post-generation claim-grounding pass: hedge claims whose citation does
    # not actually entail them (and sentences with no citation). This lifts
    # effective faithfulness without dropping coverage: the answer still
    # addresses the same points, but confidently-ungrounded claims become
    # clearly-hedged claims. Recorded so the response surfaces the count.
    answer, hedged_count = _rewrite_ungrounded_claims(answer, citations)
    citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
    if scope == "uploaded" and _needs_scope_limited_answer(query, citations):
        answer = _scope_limited_answer(query, citations)
    if scope == "uploaded" and re.search(
        r"(cannot|can't|can not|not able to)\s+(access|see|view|read).*(documents|docs|files)",
        answer,
        flags=re.IGNORECASE,
    ):
        answer = (
            "I can use your uploaded documents through retrieval context. "
            "For this question, I need a bit more specific wording or clearer evidence in the indexed chunks. "
            "Insufficient evidence."
        )
    # Only fall back to template answers when the LLM output is critically uncited:
    # - zero inline [S#] citations anywhere in the answer, OR
    # - answer has _FALLBACK_MIN_PARAGRAPHS+ paragraphs but coverage < _FALLBACK_MIN_COVERAGE.
    # Intro sentences, transitions, and conclusions legitimately lack citations —
    # replacing a well-grounded answer just because one paragraph is citation-free
    # is the primary cause of retrieval-dump responses instead of coherent synthesis.
    # citation_coverage_par > 0 means _citation_coverage_stats already found at least one
    # cited paragraph — no need for a second regex pass over the answer.
    lacks_any_citations = citation_coverage_par == 0.0
    sparse_multi_para = paragraph_count >= _FALLBACK_MIN_PARAGRAPHS and citation_coverage_par < _FALLBACK_MIN_COVERAGE
    critically_uncited = lacks_any_citations or sparse_multi_para
    if critically_uncited:
        if scope != "uploaded" and citations:
            if answer_mode == "source_listing":
                answer = _build_public_source_listing_answer(citations)
            else:
                answer = _build_public_synthesis_fallback(citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
        elif scope == "uploaded" and related_work_intent and citations:
            answer = _build_uploaded_related_work_fallback(citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
        elif scope == "uploaded" and citations:
            answer = _build_uploaded_evidence_fallback(query, citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
        else:
            evidence_label = _scope_evidence_label(scope)
            answer = (
                f"I don’t have enough evidence in the selected {evidence_label} to support a grounded answer. "
                "Try rephrasing or uploading a more relevant source."
            )
            # Do not present misleading evidence cards when answer is explicitly blocked.
            citations = []
    citations = _apply_usage_boost(citations, answer)
    msa_by_citation: dict[int, dict] = {}
    unsupported_by_msa = 0
    # Compute MSA for any grounded prose answer (uploaded or public). The only
    # mode we skip is pure source_listing, where the "answer" is just a link list
    # and per-claim entailment is not meaningful. User expectation is that every
    # grounded answer — regardless of scope — surfaces per-citation support.
    should_compute_msa = (
        bool(citations)
        and bool((answer or "").strip())
        and (answer_mode in ("research_synthesis", "explanatory", "extractive") or run_judge)
    )
    if should_compute_msa:
        msa_by_citation, unsupported_by_msa = _compute_citation_msa(
            query,
            answer,
            citations,
            scope=scope,
            k=k,
            doc_id=doc_id,
            source_only=requested_public_source,
        )

    if msa_by_citation:
        for c in citations:
            cid = int(c.get("id") or 0)
            msa = msa_by_citation.get(cid)
            if not msa:
                continue
            msa_payload = {
                "M": float(msa.get("M", 0.0)),
                "S": float(msa.get("S", 0.0)),
                "A": float(msa.get("A", 0.0)),
                "weights": _load_latest_calibration_weights(scope),
            }
            c["msa"] = {
                **msa_payload,
                "msa_score": float(msa.get("msa_score", 0.0)),
                "score_percent": score_percent(float(msa.get("msa_score", 0.0))),
            }
            c["msa_supported"] = bool(float(msa.get("M", 0.0)) >= 0.5)
            # Keep `used_in_answer` as the LLM's own citation decision — the UI
            # separately surfaces msa_supported so weak-entailment citations are
            # visible as "Cited · Weak support" rather than silently demoted to
            # Candidate (which previously made "0 of N cited" render even when
            # the answer clearly used [S#] inline).
            c["confidence_obj"] = build_confidence(
                top_sim=float(c.get("sim_score", 0.0) or 0.0),
                top_rerank_norm=float(c.get("rerank_norm", 0.0) or 0.0),
                citation_coverage=1.0 if c.get("used_in_answer") else 0.0,
                evidence_margin=float(c.get("sim_score", 0.0) or 0.0),
                ambiguity_penalty=0.0,
                insufficiency_penalty=0.0,
                scope_penalty=0.0,
                msa=msa_payload,
            )

    if msa_by_citation and any(c.get("msa_supported") is False for c in citations):
        unsupported_claims = max(unsupported_claims, unsupported_by_msa or unsupported_claims)

    # Only replace the answer when MSA drops ALL citations AND the answer carries no
    # inline [S#] references — meaning it genuinely hallucinated without grounding.
    # citation_coverage_par reflects the most recently recomputed answer (post-fallback),
    # so citation_coverage_par == 0.0 is equivalent to "no inline citations" without
    # an extra regex scan.
    all_used_dropped = citations and all(not c.get("used_in_answer") for c in citations)
    if all_used_dropped and answer.strip() and citation_coverage_par == 0.0:
        if scope == "uploaded" and citations:
            answer = _build_uploaded_evidence_fallback(query, citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
        else:
            if answer_mode == "source_listing":
                answer = _build_public_source_listing_answer(citations)
            else:
                answer = _build_public_synthesis_fallback(citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)

    if scope == "uploaded" and doc_ids and len(doc_ids) > 1:
        citations = _rebalance_uploaded_multi_doc_citations(citations, doc_ids, k)
        for i, c in enumerate(citations, start=1):
            c["id"] = i
        # Detect the case where the LLM collapsed a multi-doc summary into a single-doc
        # narrative (e.g. asked about 4 papers, returned a BERT-only response). If the
        # answer only carries [S#] chips belonging to one doc, fall back to the
        # deterministic per-doc template so every selected document appears in the reply.
        llm_cited_docs: set[int] = set()
        for match in re.finditer(r"\[S(\d+)\]", answer or ""):
            try:
                sid = int(match.group(1))
            except ValueError:
                continue
            hit = next((c for c in citations if int(c.get("id") or 0) == sid), None)
            if hit and hit.get("doc_id") is not None:
                llm_cited_docs.add(int(hit["doc_id"]))
        total_docs_with_evidence = len({int(c.get("doc_id")) for c in citations if c.get("doc_id") is not None})
        llm_covers_single_doc = total_docs_with_evidence >= 2 and len(llm_cited_docs) <= 1
        is_summary_intent = _is_uploaded_doc_summary_query(query)
        force_multi_doc_template = is_summary_intent and (
            critically_uncited or not answer.strip() or llm_covers_single_doc
        )
        if force_multi_doc_template:
            answer = _build_multi_doc_uploaded_summary(citations, doc_ids)
            citations = _apply_usage_boost(citations, answer)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)

    if scope != "uploaded" and answer_mode == "source_listing":
        answer = _append_public_source_links(answer, citations)

    cited_count = sum(1 for c in citations if c.get("used_in_answer"))
    if run_judge:
        faithfulness = evaluate_faithfulness(query, answer, citations, use_llm=run_judge_llm)
    else:
        faithfulness = None

    # Avoid over-penalizing answers that cite the strongest subset of retrieved context.
    effective_pool = max(1, min(len(citations), max(3, paragraph_count * 2, cited_count)))
    citation_usage = cited_count / effective_pool if citations else 0.0
    citation_coverage = max(citation_coverage_par, citation_usage)
    sorted_by_sim = sorted(citations, key=lambda x: float(x.get("sim_score", 0.0) or 0.0), reverse=True)
    top_sim = float(sorted_by_sim[0].get("sim_score", 0.0) or 0.0) if sorted_by_sim else 0.0
    top_rerank_norm = float(citations[0].get("rerank_norm", 0.0) or 0.0) if citations else 0.0
    if len(sorted_by_sim) > 1:
        evidence_margin = max(0.0, top_sim - float(sorted_by_sim[1].get("sim_score", 0.0) or 0.0))
    else:
        evidence_margin = top_sim
    ambiguity_penalty = 0.35 if (sense.get("is_ambiguous") and compare_senses) else 0.0
    uncited_ratio = (unsupported_claims / max(1, paragraph_count)) if paragraph_count else 0.0
    insufficiency_penalty = min(0.5, 0.5 * uncited_ratio) if unsupported_claims > 0 else 0.0
    if isinstance(faithfulness, dict):
        try:
            judge_sentence_count = int(faithfulness.get("sentence_count") or 0)
            judge_unsupported = int(faithfulness.get("unsupported_count") or 0)
            if judge_sentence_count > 0:
                judge_unsupported_ratio = judge_unsupported / float(judge_sentence_count)
                insufficiency_penalty = max(insufficiency_penalty, min(0.65, 0.65 * judge_unsupported_ratio))
        except Exception:
            pass
    if unsupported_claims == 0 and "enough evidence" in (answer or "").lower():
        insufficiency_penalty = 0.25
    personal_only = bool(citations) and all((_source_scope(c) == "personal_profile") for c in citations)
    scope_penalty = 0.7 if (entity_query and personal_only) else 0.0
    has_uploaded_evidence = any((c.get("source") or "").lower() == "uploaded" for c in citations)
    answer_msa = None
    if msa_by_citation:
        weighted_rows: list[tuple[float, float, float, float]] = []
        for c in citations:
            msa_payload = c.get("msa")
            if not isinstance(msa_payload, dict):
                continue
            try:
                m = _clamp01(float(msa_payload.get("M", 0.0)))
                s = _clamp01(float(msa_payload.get("S", 0.0)))
                a = _clamp01(float(msa_payload.get("A", 0.0)))
            except Exception:
                continue
            rel = _clamp01(float(c.get("sim_score", 0.0) or 0.0))
            weight = 0.7 + (0.8 * rel) + (0.45 if c.get("used_in_answer") else 0.0)
            weighted_rows.append((m, s, a, weight))
        if weighted_rows:
            total_weight = sum(w for _, _, _, w in weighted_rows) or 1.0
            answer_msa = {
                "M": sum(m * w for m, _, _, w in weighted_rows) / total_weight,
                "S": sum(s * w for _, s, _, w in weighted_rows) / total_weight,
                "A": sum(a * w for _, _, a, w in weighted_rows) / total_weight,
                "weights": _load_latest_calibration_weights(scope),
            }
    grounded_minimum_score = 0.0
    if scope == "uploaded" and has_uploaded_evidence and citation_coverage >= 0.25 and unsupported_claims == 0:
        retrieval_strength = _clamp01((0.6 * top_sim) + (0.4 * top_rerank_norm))
        grounded_minimum_score = min(0.97, 0.62 + (0.2 * citation_coverage) + (0.16 * retrieval_strength))
    confidence = build_confidence(
        top_sim=top_sim,
        top_rerank_norm=top_rerank_norm,
        citation_coverage=citation_coverage,
        evidence_margin=evidence_margin,
        ambiguity_penalty=ambiguity_penalty,
        insufficiency_penalty=insufficiency_penalty,
        scope_penalty=scope_penalty,
        needs_clarification=False,
        msa=answer_msa,
        minimum_score=grounded_minimum_score,
    )
    trust = round(min(1.0, len(citations) / max(1, k)), 3)
    latency_ms = int((time.time() - started) * 1000)

    trace_chunks = []
    rerank_changed = False
    for c in citations:
        if (c.get("rank_delta") or 0) != 0:
            rerank_changed = True
        trace_chunks.append(
            {
                "id": c.get("id"),
                "title": c.get("title"),
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "page": c.get("page"),
                "snippet_preview": (c.get("snippet", "") or "")[:260],
                "sim_score": round(float(c.get("sim_score", 0.0) or 0.0), 4),
                "sim_raw": round(float(c.get("sim_raw", c.get("sim_score", 0.0)) or 0.0), 4),
                "rerank_raw": round(float(c.get("rerank_raw", 0.0) or 0.0), 4),
                "rerank_norm": round(float(c.get("rerank_norm", 0.0) or 0.0), 4),
                "rank_before": c.get("rank_before"),
                "rank_after": c.get("rank_after"),
                "rank_delta": c.get("rank_delta"),
                "cited": bool(c.get("used_in_answer")),
                "source": c.get("source"),
                "scope": c.get("scope"),
                "reranker_type": c.get("reranker_type"),
            }
        )

    if unsupported_claims > 0:
        trace_chunks = []
        rerank_changed = False

    log_json(
        REQUEST_LOG,
        {
            "ts": time.time(),
            "event": "assistant_answer",
            "query": query,
            "scope": scope,
            "doc_id": doc_id,
            "k": k,
            "multi_hop": multi_hop,
            "uploaded_hits": uploaded_hits,
            "public_hits": public_hits,
            "uploaded_strength": uploaded_strength,
            "uploaded_overlap": uploaded_overlap,
            "used_public_fallback": used_public_fallback,
            "context_count": len(citations),
            "citations": len(citations),
            "answer_mode": answer_mode,
            "public_provider_status": public_provider_status,
            "trust": trust,
            "confidence_score": confidence.get("score"),
            "latency_ms": latency_ms,
        },
    )

    citations_out = [dict(c) for c in citations]
    for c in citations_out:
        if "confidence_obj" not in c:
            # Per-card Grounding uses the retrieval-heuristic composite
            # (sim · rerank · coverage · margin). We intentionally do NOT thread
            # msa= here: the MSA-calibrated probability was fit against human
            # labels where ~97% of uploaded-mode claim-evidence pairs were marked
            # "unsupported", so the calibrated output compresses into [0.2, 0.5]
            # and conflates "this retrieval was good" with "this claim is literally
            # entailed". The retrieval composite rewards good retrievals at
            # 70-90%, which is what the per-card badge should reflect. The MSA
            # signal is still surfaced via the "Weak support" chip and the
            # answer-level grounding pill.
            c["confidence_obj"] = build_confidence(
                top_sim=float(c.get("sim_score", 0.0) or 0.0),
                top_rerank_norm=float(c.get("rerank_norm", 0.0) or 0.0),
                citation_coverage=1.0 if c.get("used_in_answer") else 0.0,
                evidence_margin=float(c.get("sim_score", 0.0) or 0.0),
                ambiguity_penalty=0.0,
                insufficiency_penalty=0.0,
                scope_penalty=0.0,
                needs_clarification=False,
            )
        c["confidence"] = float(c.get("confidence", c["confidence_obj"].get("score", 0.0)) or 0.0)
        c["confidence_percent"] = score_percent(float(c["confidence"]))
        # Sync the usage-boosted confidence back into confidence_obj so the
        # per-card Grounding badge in the UI sees the same number the top-of-
        # answer pill and confidence_percent report. Otherwise the badge shows
        # the pre-boost heuristic value and looks lower than the reported
        # confidence_percent.
        if c.get("confidence_obj") and "score" in c["confidence_obj"]:
            boosted = float(c.get("confidence") or 0.0)
            if boosted > c["confidence_obj"]["score"]:
                c["confidence_obj"]["score"] = round(boosted, 4)
                label_bucket = "High" if boosted >= 0.75 else "Med" if boosted >= 0.5 else "Low"
                c["confidence_obj"]["label"] = label_bucket
    if not debug_confidence:
        for c in citations_out:
            c.pop("_confidence_meta", None)
            c.pop("base_confidence", None)
            c.pop("usage_boost", None)

    # Re-sort citations by their displayed Confidence so the UI order matches
    # the confidence pill the user sees. Retrieval order is tracked via
    # rank_before / rank_after for analytics; visual order now prioritizes
    # high-confidence cited sources first, then high-confidence candidates,
    # with sim_score as a tiebreaker.
    def _confidence_sort_key(c: dict) -> tuple[float, float, float]:
        conf_score = float((c.get("confidence_obj") or {}).get("score", 0.0))
        used = 1.0 if c.get("used_in_answer") else 0.0
        sim = float(c.get("sim_score", 0.0) or 0.0)
        return (-conf_score, -used, -sim)

    if citations_out:
        # Assign stable pre-sort ids (match the [S#] refs the LLM emitted).
        for old_idx, c in enumerate(citations_out, start=1):
            c["_old_id"] = old_idx
        citations_out.sort(key=_confidence_sort_key)
        # Map old_id -> new_id so we can rewrite inline [S#] refs in the answer.
        id_remap: dict[int, int] = {}
        for new_idx, c in enumerate(citations_out, start=1):
            old_id = int(c.pop("_old_id"))
            id_remap[old_id] = new_idx
            c["id"] = new_idx
            c["rank_after"] = new_idx
            c["rank_delta"] = c.get("rank_before", new_idx) - new_idx
        if id_remap and answer:

            def _remap_ref(m: re.Match) -> str:
                prefix = m.group(1) or ""
                old_id = int(m.group(2))
                new_id = id_remap.get(old_id, old_id)
                return f"[{prefix}{new_id}]"

            answer = re.sub(r"\[(S?)(\d+)\]", _remap_ref, answer)

    if citations:
        if all((_source_scope(c) == "personal_profile") for c in citations):
            scope_label = "personal_document_context"
        elif any((c.get("source") or "").lower() != "uploaded" for c in citations):
            scope_label = "official_document_context"
        else:
            scope_label = "uploaded_document_context"
    else:
        scope_label = "retrieved_context"

    return {
        "answer": answer,
        "citations": citations_out,
        "confidence": confidence,
        "needs_clarification": False,
        "clarification": None,
        "answer_scope": chosen_sense or ("compare_senses" if compare_senses else scope_label),
        "unsupported_claims": unsupported_claims,
        "faithfulness": faithfulness if faithfulness is not None else None,
        "trust": trust,
        "latency_ms": latency_ms,
        "confidence_note": (
            "Confidence is heuristic/derived from evidence retrieval, retrieval stability, and optional MSA calibration. "
            "Use debug_confidence=true for per-source breakdown."
        ),
        "why_answer": {
            "rerank_changed_order": rerank_changed,
            "top_chunks": trace_chunks,
        },
        "scoring": {
            "similarity_metric": "cosine_similarity",
            "reranker_used": True,
            "reranker_type": "lexical_overlap",
            "rerank_score_fields": ["rerank_raw", "rerank_norm"],
        },
        "latency_breakdown_ms": {
            "retrieve": round(retrieval_ms, 2),
            "rerank": round(rerank_ms, 2),
            "generate": round(generate_ms, 2),
            "total": latency_ms,
        },
        "retrieval_policy": {
            "mode": "uploaded-first" if scope == "uploaded" else "public-only",
            "answer_mode": answer_mode,
            "uploaded_hits": uploaded_hits,
            "public_hits": public_hits,
            "uploaded_strength": uploaded_strength,
            "uploaded_overlap": uploaded_overlap,
            "used_public_fallback": used_public_fallback,
            "source_breakdown": _source_breakdown(citations),
            "public_provider_status": public_provider_status,
            "query_intent": (
                {
                    "canonical_term": query_intent.get("canonical_term"),
                    "domain": query_intent.get("domain"),
                    "is_ambiguous": bool(query_intent.get("is_ambiguous")),
                    "alternative_senses": list(query_intent.get("alternative_senses") or []),
                    "disambiguation_hints": list(query_intent.get("disambiguation_hints") or []),
                    "search_queries": list(query_intent.get("search_queries") or []),
                    "model": "gpt-4o-mini",
                }
                if intent_usable
                else None
            ),
        },
    }


@app.post("/assistant/resolve_sense")
def assistant_resolve_sense(payload: dict = Body(...)):
    query = (payload.get("query") or "").strip()
    scope = payload.get("scope") or "uploaded"
    k = int(payload.get("k") or 8)
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    chunks = []
    if scope == "uploaded":
        rows = search_uploaded_chunks(query, k=k, doc_id=payload.get("doc_id"))["results"]
        for r in rows:
            chunks.append(
                {
                    "title": r.get("title"),
                    "snippet": r.get("text", ""),
                    "doc_id": r.get("document_id"),
                    "chunk_id": r.get("id"),
                    "page": r.get("page_no"),
                }
            )
    sense = resolve_sense(query, chunks, chosen_sense=payload.get("sense"))
    return sense


@app.get("/metrics/requests")
def metrics_requests():
    """
    Lightweight aggregation over logs/requests.jsonl (assistant_answer events).
    """
    import json

    path = LOG_DIR / "requests.jsonl"
    if not path.exists():
        return {"count": 0, "avg_latency_ms": None, "avg_trust": None, "avg_citations": None}

    latencies = []
    trusts = []
    cits = []
    count = 0
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") != "assistant_answer":
                continue
            count += 1
            if rec.get("latency_ms") is not None:
                latencies.append(rec["latency_ms"])
            if rec.get("trust") is not None:
                trusts.append(rec["trust"])
            if rec.get("citations") is not None:
                cits.append(rec["citations"])

    def avg(arr):
        return sum(arr) / len(arr) if arr else None

    return {
        "count": count,
        "avg_latency_ms": avg(latencies),
        "avg_trust": avg(trusts),
        "avg_citations": avg(cits),
    }


def _eval_candidates_for_query(
    query: str,
    k: int,
    doc_id: int | None = None,
    doc_ids: list[int] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    t_retrieve = time.perf_counter()
    raw = search_uploaded_chunks({"q": query, "k": max(10, k), "doc_id": doc_id, "doc_ids": doc_ids})["results"]
    retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

    retrieval_only = []
    for idx, r in enumerate(raw, start=1):
        retrieval_only.append(
            {
                "doc_id": r.get("document_id"),
                "chunk_id": r.get("id"),
                "page": r.get("page_no"),
                "title": r.get("title") or f"Document {r.get('document_id')}",
                "distance": float(r.get("distance", 1.0) or 1.0),
                "snippet": r.get("text", ""),
                "initial_rank": idx,
                "confidence": 1.0 - min(1.0, float(r.get("distance", 1.0) or 1.0)),
            }
        )

    t_rerank = time.perf_counter()
    reranked = _rank_and_trim_citations(query, retrieval_only, k=max(10, k), doc_ids=doc_ids)
    rerank_ms = (time.perf_counter() - t_rerank) * 1000
    for i, c in enumerate(reranked, start=1):
        c["rank_after"] = i

    return retrieval_only[:k], reranked[:k], {"retrieve_ms": round(retrieve_ms, 2), "rerank_ms": round(rerank_ms, 2)}


def _judge_label_to_binary(label: object) -> int | None:
    if label is None:
        return None
    v = str(label).strip().lower()
    if not v:
        return None
    if v in {"strong", "high", "positive", "yes", "1", "true", "supported"}:
        return 1
    if v in {"weak", "moderate", "medium", "low", "negative", "0", "false", "unsupported"}:
        return 0
    return None


def _sigmoid(x: float) -> float:
    # Clamp for numerical stability
    x = max(-60.0, min(60.0, float(x)))
    return 1.0 / (1.0 + np.exp(-x))


def _fit_logistic_weights(
    records: list[tuple[float, float, float, int]], iters: int = 2200
) -> tuple[dict[str, float], dict[str, float]]:
    # records: [(m,s,a,label_int)]
    if not records:
        return (
            {"w1": 0.58, "w2": 0.22, "w3": 0.20, "b": 0.0},
            {"status": "empty"},
        )

    n = len(records)
    w1 = 0.58
    w2 = 0.22
    w3 = 0.20
    b = 0.0
    lr = 0.38
    l2 = 0.001

    for _ in range(max(1, iters)):
        g1 = g2 = g3 = gb = 0.0
        correct = 0
        brier = 0.0
        for m, s, a, y in records:
            z = b + w1 * m + w2 * s + w3 * a
            p = _sigmoid(z)
            y_f = float(y)
            diff = p - y_f
            g1 += diff * m
            g2 += diff * s
            g3 += diff * a
            gb += diff
            brier += (p - y_f) ** 2
            if (p >= 0.5) == bool(y_f):
                correct += 1

        g1 = g1 / n + l2 * w1
        g2 = g2 / n + l2 * w2
        g3 = g3 / n + l2 * w3
        gb = gb / n + l2 * b

        w1 -= lr * g1
        w2 -= lr * g2
        w3 -= lr * g3
        b -= lr * gb

    accuracy = correct / n if n else 0.0
    brier = brier / n if n else 0.0
    weights = {"w1": round(w1, 6), "w2": round(w2, 6), "w3": round(w3, 6), "b": round(b, 6)}
    metrics = {
        "n": n,
        "accuracy": round(accuracy, 4),
        "brier": round(brier, 4),
        "method": "gradient_logistic",
    }
    return weights, metrics


def _build_msa_records(payload: dict) -> list[tuple[float, float, float, int]]:
    rows: list[tuple[float, float, float, int]] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        msa = item.get("msa") or {}
        if isinstance(msa, dict) and all(k in msa for k in ("M", "S", "A")):
            m = float(msa.get("M", 0.0))
            s = float(msa.get("S", 0.0))
            a = float(msa.get("A", 0.0))
        else:
            sentence = (item.get("sentence") or "").strip()
            evidence_text = item.get("evidence") or item.get("evidence_text") or item.get("evidence_snippet") or ""
            if sentence and evidence_text:
                m = entailment_prob(sentence, str(evidence_text))
                s = float(item.get("S", 0.5))
                a = float(item.get("A", 0.5))
            else:
                continue

        label = _judge_label_to_binary(item.get("label"))
        if label is None and "answer_supported" in item:
            label = 1 if bool(item.get("answer_supported")) else 0
        if label is None:
            continue
        rows.append((_clamp01(m), _clamp01(s), _clamp01(a), label))
    return rows


@app.post("/eval/run")
def run_eval(payload: dict = Body(...)):
    name = (payload.get("name") or "Eval run").strip()
    scope = payload.get("scope") or "uploaded"
    k = int(payload.get("k") or 10)
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise HTTPException(status_code=400, detail="cases must be a non-empty list")
    if scope != "uploaded":
        raise HTTPException(status_code=400, detail="Eval currently supports uploaded scope only")

    retrieval_rows = []
    rerank_rows = []
    details = []
    lat_retrieve = []
    lat_rerank = []
    lat_generate = []

    for case in cases:
        query = (case.get("query") or "").strip()
        gold_doc_id = case.get("expected_doc_id")
        case_doc_id = case.get("doc_id")
        raw_case_doc_ids = case.get("doc_ids")
        case_doc_ids = None
        try:
            gold_doc_id = int(gold_doc_id) if gold_doc_id is not None else None
        except Exception:
            gold_doc_id = None
        try:
            case_doc_id = int(case_doc_id) if case_doc_id is not None else None
        except Exception:
            case_doc_id = None
        if isinstance(raw_case_doc_ids, list):
            try:
                case_doc_ids = [int(x) for x in raw_case_doc_ids if x is not None]
            except Exception:
                case_doc_ids = None
        if not query:
            continue

        base, reranked, lat = _eval_candidates_for_query(query, k, doc_id=case_doc_id, doc_ids=case_doc_ids)
        lat_retrieve.append(lat["retrieve_ms"])
        lat_rerank.append(lat["rerank_ms"])
        lat_generate.append(0.0)

        retrieval_pred = [int(x.get("doc_id")) for x in base if x.get("doc_id") is not None]
        rerank_pred = [int(x.get("doc_id")) for x in reranked if x.get("doc_id") is not None]
        retrieval_rows.append({"pred_doc_ids": retrieval_pred, "gold_doc_id": gold_doc_id})
        rerank_rows.append({"pred_doc_ids": rerank_pred, "gold_doc_id": gold_doc_id})
        details.append(
            {
                "query": query,
                "gold_doc_id": gold_doc_id,
                "doc_id": case_doc_id,
                "doc_ids": case_doc_ids,
                "retrieval_only_top": base[:5],
                "rerank_top": reranked[:5],
                "latency_ms": lat,
            }
        )

    metrics_retrieval_only = aggregate_metrics(retrieval_rows)
    metrics_retrieval_rerank = aggregate_metrics(rerank_rows)
    latency_breakdown = {
        "retrieve_ms_avg": round(sum(lat_retrieve) / max(1, len(lat_retrieve)), 2),
        "rerank_ms_avg": round(sum(lat_rerank) / max(1, len(lat_rerank)), 2),
        "generate_ms_avg": round(sum(lat_generate) / max(1, len(lat_generate)), 2),
    }

    row = fetchone(
        """
        INSERT INTO eval_runs
        (name, scope, k, case_count, metrics_retrieval_only, metrics_retrieval_rerank, latency_breakdown, details)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
        RETURNING id, created_at
        """,
        [
            name,
            scope,
            k,
            len(details),
            json.dumps(metrics_retrieval_only),
            json.dumps(metrics_retrieval_rerank),
            json.dumps(latency_breakdown),
            json.dumps(details),
        ],
    )

    return {
        "run_id": row.get("id") if row else None,
        "created_at": row.get("created_at").isoformat() if row and row.get("created_at") else None,
        "name": name,
        "scope": scope,
        "k": k,
        "case_count": len(details),
        "metrics_retrieval_only": metrics_retrieval_only,
        "metrics_retrieval_rerank": metrics_retrieval_rerank,
        "latency_breakdown": latency_breakdown,
        "details": details,
    }


@app.get("/eval/runs")
def list_eval_runs(limit: int = 20):
    rows = fetchall(
        """
        SELECT id, name, scope, k, case_count, metrics_retrieval_only, metrics_retrieval_rerank, latency_breakdown, created_at
        FROM eval_runs
        ORDER BY id DESC
        LIMIT %s
        """,
        [max(1, min(limit, 100))],
    )
    out = []
    for r in rows:
        row_id = r.get("id")
        out.append(
            {
                "id": row_id,
                "run_id": row_id,
                "name": r.get("name"),
                "scope": r.get("scope"),
                "k": r.get("k"),
                "case_count": r.get("case_count"),
                "metrics_retrieval_only": r.get("metrics_retrieval_only"),
                "metrics_retrieval_rerank": r.get("metrics_retrieval_rerank"),
                "latency_breakdown": r.get("latency_breakdown"),
                "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
            }
        )
    return {"runs": out}


@app.post("/eval/judge")
def run_judge(payload: dict = Body(...)):
    scope = payload.get("scope") or "uploaded"
    k = int(payload.get("k") or 10)
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise HTTPException(status_code=400, detail="cases must be a non-empty list")
    if scope not in {"uploaded", "public"}:
        raise HTTPException(status_code=400, detail="scope must be uploaded or public")

    run_judge_llm = bool(payload.get("run_judge_llm", True))
    details = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        query = (case.get("query") or "").strip()
        if not query:
            continue

        case_doc_id = case.get("doc_id")
        raw_case_doc_ids = case.get("doc_ids")
        case_doc_ids = None
        try:
            case_doc_id = int(case_doc_id) if case_doc_id is not None else None
        except Exception:
            case_doc_id = None
        if isinstance(raw_case_doc_ids, list):
            try:
                case_doc_ids = [int(x) for x in raw_case_doc_ids if x is not None]
            except Exception:
                case_doc_ids = None

        answer = (case.get("answer") or "").strip()
        citations = case.get("citations")
        if not answer or not isinstance(citations, list):
            result = assistant_answer(
                {
                    "query": query,
                    "scope": scope,
                    "doc_id": case_doc_id,
                    "doc_ids": case_doc_ids,
                    "k": k,
                    "run_judge": False,
                    "allow_general_background": bool(case.get("allow_general_background", False)),
                }
            )
            answer = (result.get("answer") or "").strip()
            citations = result.get("citations") or []
        report = evaluate_faithfulness(
            query, answer, citations if isinstance(citations, list) else [], use_llm=run_judge_llm
        )
        details.append(
            {
                "query": query,
                "answer": answer,
                "citations": citations if isinstance(citations, list) else [],
                "faithfulness": report,
                "doc_id": case_doc_id,
                "doc_ids": case_doc_ids,
                "scope": scope,
            }
        )

    if not details:
        raise HTTPException(status_code=400, detail="No valid cases found")

    aggregate = aggregate_judge_report([d.get("faithfulness", {}) for d in details])
    row = fetchone(
        """
        INSERT INTO evaluation_judge_runs
        (scope, query_count, metrics, details)
        VALUES (%s, %s, %s::jsonb, %s::jsonb)
        RETURNING id, created_at
        """,
        [
            scope,
            len(details),
            json.dumps(aggregate),
            json.dumps(details),
        ],
    )

    return {
        "run_id": row.get("id") if row else None,
        "created_at": row.get("created_at").isoformat() if row and row.get("created_at") else None,
        "scope": scope,
        "query_count": len(details),
        "metrics": aggregate,
        "details": details,
    }


@app.get("/eval/judge/runs")
def list_judge_runs(limit: int = 20):
    rows = fetchall(
        """
        SELECT id, scope, query_count, metrics, details, created_at
        FROM evaluation_judge_runs
        ORDER BY id DESC
        LIMIT %s
        """,
        [max(1, min(limit, 100))],
    )
    out = []
    for r in rows:
        row_id = r.get("id")
        out.append(
            {
                "id": row_id,
                "run_id": row_id,
                "scope": r.get("scope"),
                "query_count": r.get("query_count"),
                "metrics": r.get("metrics"),
                "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
            }
        )
    return {"runs": out}


@app.post("/confidence/calibrate")
def calibrate_confidence(payload: dict = Body(...)):
    records = payload.get("records") or []
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=400, detail="records must be a non-empty list")
    msarecords = _build_msa_records(records)
    if len(msarecords) < 5:
        raise HTTPException(status_code=400, detail="At least 5 labeled records required to fit calibration.")
    model_name = (payload.get("model_name") or "msa_logistic_v1").strip() or "msa_logistic_v1"
    label = (payload.get("label") or "default").strip() or "default"
    weights, metrics = _fit_logistic_weights(msarecords)
    row = fetchone(
        """
        INSERT INTO confidence_calibration
        (model_name, label, weights, metrics, dataset_size)
        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)
        RETURNING id, created_at
        """,
        [model_name, label, json.dumps(weights), json.dumps(metrics), len(msarecords)],
    )
    return {
        "run_id": row.get("id") if row else None,
        "created_at": row.get("created_at").isoformat() if row and row.get("created_at") else None,
        "model_name": model_name,
        "label": label,
        "records_used": len(msarecords),
        "weights": weights,
        "metrics": metrics,
    }


@app.get(
    "/confidence/calibration",
    response_model=CalibrationResponse,
    tags=["confidence"],
)
def get_latest_calibration(label: str | None = None):
    if label:
        row = fetchone(
            """
            SELECT id, model_name, label, weights, metrics, dataset_size, created_at
            FROM confidence_calibration
            WHERE label = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (label,),
        )
    else:
        row = fetchone(
            """
            SELECT id, model_name, label, weights, metrics, dataset_size, created_at
            FROM confidence_calibration
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
    if not row:
        return {
            "model_name": "msa_logistic_v1",
            "label": "default",
            "weights": {"w1": 0.58, "w2": 0.22, "w3": 0.20, "b": 0.0},
            "metrics": None,
            "dataset_size": 0,
            "created_at": None,
        }
    created = row.get("created_at")
    return {
        "id": row.get("id"),
        "model_name": row.get("model_name"),
        "label": row.get("label"),
        "weights": row.get("weights") or {"w1": 0.58, "w2": 0.22, "w3": 0.20, "b": 0.0},
        "metrics": row.get("metrics"),
        "dataset_size": row.get("dataset_size"),
        "created_at": created.isoformat() if created else None,
    }


LOG_DIR = Path("logs")
RETRIEVAL_LOG = LOG_DIR / "retrieval.log"

# ------------------------------
# OpenAI Generation Config
# ------------------------------
try:
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
except RuntimeError as err:
    logging.getLogger(__name__).warning("OpenAI client init failed: %s", err)
    client = None

# Logger for observability
REQUEST_LOG = setup_file_logger(LOG_DIR / "requests.jsonl")


def log_request(entry: dict) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with RETRIEVAL_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def trust_score(sim: float, has_doi: bool) -> float:
    # Simple heuristic trust: similarity plus DOI bonus
    base = max(sim, 0.0)
    bonus = 0.05 if has_doi else 0.0
    return round(min(base + bonus, 1.0), 3)


# ------------------------------
# Endpoints
# ------------------------------

_BOOT_TS = time.time()


@app.get("/", response_model=LivenessResponse, tags=["health"])
def home():
    return {
        "message": "ScholarRAG backend is live!",
        "service": "scholarrag-backend",
        "version": app.version,
        "uptime_seconds": round(time.time() - _BOOT_TS, 1),
    }


@app.get("/health/full", response_model=HealthFullResponse, tags=["health"])
def health_full():
    """Aggregated readiness: db reachable, embedding provider live.

    Returns 200 with `status: degraded` when any dependency is down — the
    monitor reads `status` instead of relying on HTTP code.
    """
    checks: dict[str, dict] = {}

    try:
        ping = fetchone("SELECT 1 AS ok") or {}
        checks["db"] = {"ok": ping.get("ok") == 1}
    except Exception as exc:
        checks["db"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    embed = healthcheck_embeddings()
    checks["embeddings"] = {"ok": bool(embed.get("ok")), **{k: v for k, v in embed.items() if k != "ok"}}

    overall = all(c.get("ok") for c in checks.values())
    return {
        "status": "ok" if overall else "degraded",
        "service": "scholarrag-backend",
        "version": app.version,
        "uptime_seconds": round(time.time() - _BOOT_TS, 1),
        "checks": checks,
    }


@app.get("/health/embeddings", response_model=EmbeddingHealthResponse, tags=["health"])
def embeddings_health():
    return healthcheck_embeddings()


@app.get("/research/latest")
def research_latest(
    topic: str | None = Query(default=None, description="Optional research topic"),
    limit: int = Query(default=8, ge=1, le=24),
    days: int = Query(default=45, ge=1, le=365),
    sort: str = Query(default="latest", description="latest | trending | top_cited"),
):
    return latest_research_feed(topic=topic, limit=limit, days=days, sort=sort)


@app.get("/feed/latest")
def latest_papers(limit: int = 10):
    """Compatibility alias for the landing/latest research feed."""
    return latest_research_feed(limit=limit)


@app.get("/search")
def search_papers(query: str = Query(..., description="Search query text"), k: int = 5):
    """Return top-k live public scholarly results for a given query."""
    public_resp = public_live_search(query, k=max(1, k), return_metadata=True)
    results = []
    for rank, row in enumerate(public_resp.get("results", []), start=1):
        results.append(
            {
                "rank": rank,
                "title": row.get("title", "Unknown Title"),
                "year": row.get("year"),
                "doi": row.get("doi", ""),
                "source": row.get("source"),
                "url": row.get("url") or row.get("source_url"),
                "similarity": row.get("score") or row.get("similarity"),
                "summary": (row.get("abstract") or row.get("summary") or "")[:320],
            }
        )
    return {
        "query": query,
        "results": results,
        "provider_status": public_resp.get("provider_status", {}),
    }


@app.get("/summarize")
def summarize(query: str = Query(..., description="Topic to summarize")):
    """Summarize top live scholarly results for a given query using GPT."""
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")

    public_resp = public_live_search(query, k=5, return_metadata=True)
    rows = public_resp.get("results", [])
    if not rows:
        return {
            "query": query,
            "summary": "No relevant public sources found.",
            "provider_status": public_resp.get("provider_status", {}),
        }

    numbered_sources = []
    for i, row in enumerate(rows, start=1):
        snippet = (row.get("abstract") or row.get("summary") or "").strip()
        if snippet:
            snippet = snippet[:400]
        numbered_sources.append(
            f"[P{i}] {row.get('title', 'Unknown Title')} ({row.get('source', 'unknown')}, {row.get('year', 'n/a')})\n{snippet}"
        )
    top_titles = "\n\n".join(numbered_sources)

    prompt = (
        f"Summarize key themes and insights for '{query}' using ONLY the papers below.\n"
        "Add inline citations for each claim using [P#].\n\n"
        f"Papers:\n{top_titles}"
    )
    completion = client.chat.completions.create(
        model=RESEARCH_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    summary = completion.choices[0].message.content
    return {"query": query, "summary": summary, "provider_status": public_resp.get("provider_status", {})}


@app.post("/ask")
def ask(payload: dict = Body(...)):
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")

    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'.")

    k = int(payload.get("k", 10))
    start = time.perf_counter()
    public_resp = public_live_search(query, k=max(1, k), return_metadata=True)
    rows = public_resp.get("results", [])
    if not rows:
        return {
            "answer": "No relevant public sources found.",
            "sources": [],
            "candidate_counts": {},
            "metrics": {"latency_ms": round((time.perf_counter() - start) * 1000, 2)},
            "provider_status": public_resp.get("provider_status", {}),
        }

    context_blocks = []
    for i, row in enumerate(rows, start=1):
        snippet = (row.get("abstract") or row.get("summary") or "").strip()
        context_blocks.append(
            f"[P{i}] {row.get('title', 'Unknown Title')} | source={row.get('source', 'unknown')} | year={row.get('year', 'n/a')}\n{snippet}"
        )
    prompt = (
        "Answer the user's scholarly question using ONLY the provided sources.\n"
        "Every factual sentence must include an inline citation like [P1].\n"
        "If the evidence is weak or incomplete, say so explicitly.\n\n"
        f"Question: {query}\n\nSources:\n" + "\n\n".join(context_blocks)
    )
    completion = client.chat.completions.create(
        model=RESEARCH_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content or ""
    latency_ms = (time.perf_counter() - start) * 1000

    # Shape sources
    sources = []
    for d in rows:
        sim = round(float(d.get("score") or d.get("similarity") or 0.0), 3)
        t_score = trust_score(sim, bool(d.get("doi")))
        sources.append(
            {
                "title": d.get("title", "Unknown Title"),
                "year": d.get("year", "Unknown Year"),
                "doi": d.get("doi", ""),
                "openalex_id": d.get("id") or d.get("paper_id"),
                "arxiv_id": d.get("arxiv_id"),
                "concepts": (d.get("concepts") or [])[:5],
                "why_relevant": d.get("why_relevant", ""),
                "snippet": (d.get("abstract") or d.get("summary") or "")[:900],
                "similarity": sim,
                "trust_score": t_score,
                "authors": d.get("authors", []),
                "url": d.get("url") or d.get("source_url"),
                "source": d.get("source"),
            }
        )

    similarities = [s["similarity"] for s in sources if s.get("similarity") is not None]
    metrics = {
        "latency_ms": round(latency_ms, 2),
        "fallback_used": False,
        "pool_size": len(rows),
        "ranked": len(rows),
        "max_similarity": max(similarities) if similarities else None,
        "mean_similarity": round(sum(similarities) / len(similarities), 3) if similarities else None,
        "token_prompt": completion.usage.prompt_tokens if completion.usage else None,
        "token_completion": completion.usage.completion_tokens if completion.usage else None,
        "token_total": completion.usage.total_tokens if completion.usage else None,
    }

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "k": k,
        "metrics": metrics,
        "candidate_counts": {"public_results": len(rows)},
        "fallback_used": False,
        "public_provider_status": public_resp.get("provider_status", {}),
        "sources": [
            {
                "title": s.get("title"),
                "openalex_id": s.get("openalex_id"),
                "similarity": s.get("similarity"),
                "trust_score": s.get("trust_score"),
            }
            for s in sources
        ],
    }
    log_request(log_entry)

    return {
        "answer": answer,
        "sources": sources,
        "fallback_used": False,
        "candidate_counts": {"public_results": len(rows)},
        "metrics": metrics,
        "provider_status": public_resp.get("provider_status", {}),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
