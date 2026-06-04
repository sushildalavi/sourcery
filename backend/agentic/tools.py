from __future__ import annotations

from typing import Any, Iterable

from backend.pdf_ingest import search_chunks as search_uploaded_chunks
from backend.public_search import public_live_search
from backend.public_web import public_web_search
from backend.services.assistant_utils import _build_evidence_id, _rank_and_trim_citations
from backend.services.judge import evaluate_faithfulness

from .schemas import EvidenceItem


def _to_uploaded_evidence(row: dict[str, Any], *, index: int) -> EvidenceItem:
    citation = {
        "source": "uploaded",
        "doc_id": row.get("document_id"),
        "chunk_id": row.get("id"),
        "page": row.get("page_no"),
        "title": row.get("title") or f"Document {row.get('document_id')}",
    }
    score = 0.0
    distance = row.get("distance")
    if isinstance(distance, (int, float)):
        score = max(0.0, 1.0 - float(distance))
    return EvidenceItem(
        source_id=_build_evidence_id(citation),
        title=row.get("title") or f"Document {row.get('document_id')}",
        snippet=(row.get("text") or "").strip(),
        score=round(score, 4),
        citation=f"[S{index}]",
        source="uploaded",
        doc_id=row.get("document_id"),
        chunk_id=row.get("id"),
        page=row.get("page_no"),
        metadata={
            "doc_type": row.get("doc_type"),
            "chunk_index": row.get("chunk_index"),
        },
    )


def _to_public_evidence(row: dict[str, Any], *, index: int) -> EvidenceItem:
    citation = {
        "source": row.get("source") or row.get("venue") or "public",
        "title": row.get("title"),
        "year": row.get("year"),
        "url": row.get("url") or row.get("doi"),
    }
    score = row.get("_sim")
    if score is None:
        score = row.get("similarity")
    if score is None:
        score = row.get("citation_count", 0)
    if isinstance(score, (int, float)):
        score = float(score)
    else:
        score = 0.0
    return EvidenceItem(
        source_id=_build_evidence_id(citation),
        title=row.get("title") or "Source",
        snippet=(row.get("abstract") or row.get("summary") or "").strip(),
        url=row.get("url") or row.get("doi"),
        score=round(score, 4),
        citation=f"[S{index}]",
        source=(row.get("source") or row.get("venue") or "public").lower(),
        metadata={
            "year": row.get("year"),
            "venue": row.get("venue"),
            "provider": row.get("provider"),
            "citation_count": row.get("citation_count"),
            "why_relevant": row.get("why_relevant"),
        },
    )


def search_uploaded_docs(
    query: str,
    limit: int = 8,
    *,
    workspace_id: str = "default",
    doc_id: int | None = None,
    doc_ids: list[int] | None = None,
) -> list[EvidenceItem]:
    payload: dict[str, Any] = {"q": query, "k": int(limit)}
    if doc_ids:
        payload["doc_ids"] = doc_ids
    elif doc_id is not None:
        payload["doc_id"] = doc_id
    rows = search_uploaded_chunks(payload=payload, workspace_id=workspace_id).get("results", [])
    return [_to_uploaded_evidence(row, index=i) for i, row in enumerate(rows, start=1)]


def search_scholarly_sources(
    query: str,
    limit: int = 8,
    *,
    intent: dict | None = None,
    return_metadata: bool = False,
) -> list[EvidenceItem] | dict[str, Any]:
    payload: dict[str, Any]
    try:
        payload = public_live_search(query, k=int(limit), return_metadata=True, intent=intent)
    except RuntimeError as exc:
        web_results = public_web_search(query, k=int(limit))
        payload = {
            "results": web_results,
            "provider_status": {},
            "skipped": {"reason": "public_db_unavailable", "error": str(exc)},
        }
    results = payload.get("results", []) if isinstance(payload, dict) else []
    items = [_to_public_evidence(row, index=i) for i, row in enumerate(results, start=1)]
    if return_metadata:
        return {
            "results": items,
            "provider_status": payload.get("provider_status", {}) if isinstance(payload, dict) else {},
            "skipped": payload.get("skipped") if isinstance(payload, dict) else None,
        }
    return items


def rerank_evidence(
    query: str,
    items: list[EvidenceItem],
    limit: int = 6,
    *,
    doc_ids: list[int] | None = None,
    prefer_public: bool = False,
) -> list[EvidenceItem]:
    if not items:
        return []

    rank_input: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        rank_input.append(
            {
                "source": item.source,
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "title": item.title,
                "page": item.page,
                "snippet": item.snippet,
                "sim_score": item.score,
                "confidence": item.score,
                "url": item.url,
                "id": idx,
            }
        )

    ranked = _rank_and_trim_citations(
        query,
        rank_input,
        int(limit),
        prefer_public=prefer_public,
        doc_ids=doc_ids,
    )

    out: list[EvidenceItem] = []
    for idx, row in enumerate(ranked, start=1):
        source = (row.get("source") or "public").lower()
        citation = {
            "source": source,
            "doc_id": row.get("doc_id"),
            "chunk_id": row.get("chunk_id"),
            "page": row.get("page"),
            "title": row.get("title"),
            "url": row.get("url"),
        }
        out.append(
            EvidenceItem(
                source_id=_build_evidence_id(citation),
                title=row.get("title") or "Source",
                snippet=(row.get("snippet") or "").strip(),
                url=row.get("url"),
                score=float(row.get("rerank_norm", row.get("sim_score", row.get("confidence", 0.0))) or 0.0),
                citation=f"[S{idx}]",
                source=source,
                doc_id=row.get("doc_id"),
                chunk_id=row.get("chunk_id"),
                page=row.get("page"),
                metadata={
                    "rerank_raw": row.get("rerank_raw"),
                    "rerank_norm": row.get("rerank_norm"),
                    "reranker_type": row.get("reranker_type"),
                    "used_in_answer": row.get("used_in_answer", False),
                },
            )
        )
    return out[: max(1, int(limit))]


def judge_answer_support(
    query: str,
    answer: str,
    evidence: Iterable[EvidenceItem],
    *,
    use_llm: bool = True,
) -> dict[str, Any]:
    evidence_list = list(evidence)
    citations = []
    for idx, item in enumerate(evidence_list, start=1):
        citations.append(
            {
                "id": idx,
                "source": item.source,
                "title": item.title,
                "snippet": item.snippet,
                "url": item.url,
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "page": item.page,
            }
        )

    try:
        report = evaluate_faithfulness(query, answer, citations, use_llm=use_llm)
    except Exception:
        report = {
            "overall_score": 0.0,
            "citation_coverage": 0.0,
            "supported_count": 0,
            "unsupported_count": 1,
            "sentence_count": max(1, len(answer.split("."))),
            "claims": [],
            "unsupported": [{"sentence": "Unable to verify answer."}],
            "method": "fallback",
        }

    unsupported = []
    for item in report.get("unsupported", []) or []:
        if isinstance(item, dict) and item.get("sentence"):
            unsupported.append(str(item["sentence"]))
    confidence = float(report.get("overall_score", 0.0) or 0.0)
    coverage = float(report.get("citation_coverage", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, (0.7 * confidence) + (0.3 * coverage)))
    return {
        "confidence": round(confidence, 4),
        "unsupported_claims": unsupported,
        "needs_human_review": confidence < 0.70 or bool(unsupported),
        "report": report,
    }
