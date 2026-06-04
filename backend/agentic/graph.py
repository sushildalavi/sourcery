from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

try:  # optional dependency; the manual runner remains the fallback.
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - fallback when langgraph is unavailable
    END = START = StateGraph = None

from backend.intent_resolver import resolve_query_intent
from backend.services.assistant_utils import (
    _build_generation_prompt,
    _build_strict_grounded_answer,
    _classify_answer_mode,
    _normalize_inline_citations,
    _rewrite_ungrounded_claims,
)
from backend.utils.config import get_openai_api_key

from .guardrails import has_sufficient_evidence, should_require_human_review
from .schemas import AgentAnswer, EvidenceItem, QueryIntent, RetrievalPlan, RetrievalSource
from .state import ResearchAgentState
from .tools import judge_answer_support, rerank_evidence, search_scholarly_sources, search_uploaded_docs
from .traces import write_trace


def _use_llm() -> bool:
    flag = os.getenv("AGENTIC_USE_LLM", "true").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def _detect_intent(query: str, resolved: dict[str, Any] | None) -> QueryIntent:
    lower = (query or "").lower()
    if "compare" in lower or "vs" in lower or "versus" in lower or "difference" in lower:
        return QueryIntent.COMPARISON
    if "summar" in lower or "overview" in lower or "survey" in lower:
        return QueryIntent.SUMMARY
    if "evidence" in lower or "cite" in lower or "support" in lower or "prove" in lower:
        return QueryIntent.EVIDENCE_SEEKING
    if isinstance(resolved, dict) and resolved.get("is_ambiguous"):
        return QueryIntent.WEAK_EVIDENCE_RISK
    if isinstance(resolved, dict) and resolved.get("fallback"):
        return QueryIntent.EVIDENCE_SEEKING
    return QueryIntent.FACTUAL


def _detect_source_strategy(scope: str, doc_id: int | None, doc_ids: list[int]) -> RetrievalSource:
    scope_norm = (scope or "both").strip().lower()
    if scope_norm == "uploaded" or doc_id is not None or doc_ids:
        return RetrievalSource.UPLOADED_DOCS
    if scope_norm == "public":
        return RetrievalSource.SCHOLARLY_WEB
    return RetrievalSource.BOTH


def _scope_hint(strategy: RetrievalSource) -> str:
    if strategy == RetrievalSource.UPLOADED_DOCS:
        return "uploaded"
    if strategy == RetrievalSource.SCHOLARLY_WEB:
        return "public"
    return "both"


def _build_context(evidence: list[EvidenceItem]) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for idx, item in enumerate(evidence, start=1):
        item.citation = f"[S{idx}]"
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
        meta_parts = [item.title]
        if item.page is not None:
            meta_parts.append(f"p.{item.page}")
        if item.url:
            meta_parts.append(item.url)
        header = " · ".join(part for part in meta_parts if part)
        lines.append(f"[S{idx}] {header}\n{item.snippet}".strip())
    return "\n\n".join(lines), citations


def _generate_answer(
    query: str,
    evidence: list[EvidenceItem],
    *,
    use_llm: bool,
    allow_general_background: bool,
) -> str:
    answer_mode = _classify_answer_mode(query)
    scope = "public" if any(item.source != "uploaded" for item in evidence) else "uploaded"
    context, _ = _build_context(evidence)
    if not evidence:
        return (
            "I do not have enough reliable evidence to answer this confidently."
        )

    if use_llm and _use_llm():
        try:
            prompt = _build_generation_prompt(
                query,
                context,
                answer_mode,
                allow_general_background,
            )
            completion = _client().chat.completions.create(
                model=os.getenv("AGENTIC_CHAT_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = (completion.choices[0].message.content or "").strip()
            if answer:
                answer = _normalize_inline_citations(answer)
                answer, _hedged = _rewrite_ungrounded_claims(answer, [
                    {
                        "id": idx,
                        "source": item.source,
                        "title": item.title,
                        "snippet": item.snippet,
                        "doc_id": item.doc_id,
                        "chunk_id": item.chunk_id,
                        "page": item.page,
                    }
                    for idx, item in enumerate(evidence, start=1)
                ])
                return answer
        except Exception:
            pass

    return _build_strict_grounded_answer(query, [
        {
            "id": idx,
            "source": item.source,
            "title": item.title,
            "snippet": item.snippet,
            "doc_id": item.doc_id,
            "chunk_id": item.chunk_id,
            "page": item.page,
        }
        for idx, item in enumerate(evidence, start=1)
    ], scope, answer_mode)


def _plan_retrieval(state: ResearchAgentState) -> ResearchAgentState:
    query = (state.get("query") or "").strip()
    resolved = resolve_query_intent(query)
    scope = state.get("scope") or "both"
    doc_id = state.get("doc_id")
    doc_ids = state.get("doc_ids") or []
    source_strategy = _detect_source_strategy(scope, doc_id, doc_ids)
    intent = _detect_intent(query, resolved)
    risk_notes: list[str] = []
    if resolved.get("fallback"):
        risk_notes.append("intent_resolver_fallback")
    if resolved.get("is_ambiguous"):
        risk_notes.append("ambiguous_query")
    if len(query.split()) < 3:
        risk_notes.append("short_query")
    plan = RetrievalPlan(
        query=query,
        intent=intent,
        source_strategy=source_strategy,
        required_evidence_count=max(3, int(state.get("limit") or 6)),
        require_citations=True,
        risk_notes=risk_notes,
        allow_general_background=bool(state.get("allow_general_background", False)),
        scope_hint=_scope_hint(source_strategy),
        doc_id=doc_id,
        doc_ids=list(doc_ids),
    )
    trace_id = state.get("trace_id") or uuid.uuid4().hex
    next_state = {
        **state,
        "trace_id": trace_id,
        "plan": plan,
        "retrieval_metadata": {
            "resolved_intent": resolved,
            "scope": scope,
        },
    }
    write_trace(trace_id, "plan", {"plan": plan.model_dump(), "resolved": resolved})
    return next_state


def _retrieve_uploaded_docs(state: ResearchAgentState) -> ResearchAgentState:
    plan = state["plan"]
    trace_id = state["trace_id"]
    if plan.source_strategy == RetrievalSource.SCHOLARLY_WEB:
        write_trace(trace_id, "retrieve_uploaded_docs", {"skipped": True, "reason": "scholarly_only"})
        return {**state, "uploaded_doc_results": []}
    results = search_uploaded_docs(
        plan.query,
        limit=max(plan.required_evidence_count, 6),
        workspace_id=state.get("workspace_id") or "default",
        doc_id=plan.doc_id,
        doc_ids=plan.doc_ids or None,
    )
    write_trace(trace_id, "retrieve_uploaded_docs", {"count": len(results)})
    return {**state, "uploaded_doc_results": results}


def _retrieve_scholarly_sources(state: ResearchAgentState) -> ResearchAgentState:
    plan = state["plan"]
    trace_id = state["trace_id"]
    if plan.source_strategy == RetrievalSource.UPLOADED_DOCS:
        write_trace(trace_id, "retrieve_scholarly_sources", {"skipped": True, "reason": "uploaded_only"})
        return {**state, "scholarly_results": []}
    payload = search_scholarly_sources(
        plan.query,
        limit=max(plan.required_evidence_count, 6),
        intent=state.get("retrieval_metadata", {}).get("resolved_intent"),
        return_metadata=True,
    )
    results = payload.get("results", []) if isinstance(payload, dict) else []
    metadata = dict(state.get("retrieval_metadata", {}))
    metadata["public_provider_status"] = payload.get("provider_status", {}) if isinstance(payload, dict) else {}
    if isinstance(payload, dict) and payload.get("skipped"):
        metadata["skipped"] = payload["skipped"]
    write_trace(trace_id, "retrieve_scholarly_sources", {"count": len(results)})
    return {**state, "scholarly_results": results, "retrieval_metadata": metadata}


def _combine_and_rerank(state: ResearchAgentState) -> ResearchAgentState:
    plan = state["plan"]
    trace_id = state["trace_id"]
    all_items = list(state.get("uploaded_doc_results", [])) + list(state.get("scholarly_results", []))
    prefer_public = plan.source_strategy == RetrievalSource.SCHOLARLY_WEB
    reranked = rerank_evidence(
        plan.query,
        all_items,
        limit=plan.required_evidence_count,
        doc_ids=plan.doc_ids or None,
        prefer_public=prefer_public,
    )
    write_trace(trace_id, "rerank", {"count": len(reranked), "sources": [item.source for item in reranked]})
    return {**state, "reranked_evidence": reranked}


def _draft_answer(state: ResearchAgentState) -> ResearchAgentState:
    trace_id = state["trace_id"]
    evidence = state.get("reranked_evidence", [])
    query = state["query"]
    if not has_sufficient_evidence(evidence):
        answer = AgentAnswer(
            answer="I do not have enough reliable evidence to answer this confidently.",
            citations=[],
            confidence=0.35,
            unsupported_claims=["Insufficient evidence retrieved."],
            needs_human_review=True,
        )
        write_trace(trace_id, "draft_answer", {"answer": answer.model_dump(), "reason": "insufficient_evidence"})
        return {**state, "draft_answer": answer}

    allow_general_background = bool(state.get("allow_general_background", False))
    answer_text = _generate_answer(
        query,
        evidence,
        use_llm=bool(state.get("use_llm", True)),
        allow_general_background=allow_general_background,
    )
    citations = [item.citation or f"[S{idx}]" for idx, item in enumerate(evidence, start=1)]
    answer = AgentAnswer(
        answer=answer_text,
        citations=citations,
        confidence=0.75,
        unsupported_claims=[],
        needs_human_review=False,
    )
    write_trace(trace_id, "draft_answer", {"answer": answer.model_dump()})
    return {**state, "draft_answer": answer}


def _verify_answer(state: ResearchAgentState) -> ResearchAgentState:
    trace_id = state["trace_id"]
    draft = state["draft_answer"]
    evidence = state.get("reranked_evidence", [])
    report = judge_answer_support(
        state["query"],
        draft.answer,
        evidence,
        use_llm=bool(state.get("use_llm", True)),
    )
    final = AgentAnswer(
        answer=draft.answer,
        citations=draft.citations,
        confidence=report["confidence"],
        unsupported_claims=report["unsupported_claims"],
        needs_human_review=should_require_human_review(
            report["confidence"], report["unsupported_claims"]
        ),
    )
    write_trace(
        trace_id,
        "verify_answer",
        {
            "confidence": report["confidence"],
            "unsupported_claims": report["unsupported_claims"],
            "needs_human_review": final.needs_human_review,
        },
    )
    return {**state, "final_answer": final, "judge_report": report}


def _human_review(state: ResearchAgentState) -> ResearchAgentState:
    trace_id = state["trace_id"]
    final = state["final_answer"]
    safe_answer = final.answer
    if final.unsupported_claims or final.confidence < 0.70:
        safe_answer = (
            "I do not have enough reliable evidence to answer this confidently. "
            "Human review is recommended."
        )
    updated = final.model_copy(
        update={
            "answer": safe_answer,
            "needs_human_review": True,
        }
    )
    write_trace(trace_id, "human_review", {"answer": updated.model_dump()})
    return {**state, "final_answer": updated}


def _build_langgraph_agent():
    if StateGraph is None:
        return None
    try:
        graph = StateGraph(ResearchAgentState)
        graph.add_node("plan_retrieval", _plan_retrieval)
        graph.add_node("retrieve_uploaded_docs", _retrieve_uploaded_docs)
        graph.add_node("retrieve_scholarly_sources", _retrieve_scholarly_sources)
        graph.add_node("combine_and_rerank", _combine_and_rerank)
        graph.add_node("draft_answer", _draft_answer)
        graph.add_node("verify_answer", _verify_answer)
        graph.add_node("human_review", _human_review)

        graph.add_edge(START, "plan_retrieval")
        graph.add_edge("plan_retrieval", "retrieve_uploaded_docs")
        graph.add_edge("retrieve_uploaded_docs", "retrieve_scholarly_sources")
        graph.add_edge("retrieve_scholarly_sources", "combine_and_rerank")
        graph.add_edge("combine_and_rerank", "draft_answer")
        graph.add_edge("draft_answer", "verify_answer")

        def _route_after_verify(state: ResearchAgentState):
            final = state.get("final_answer")
            if final and final.needs_human_review:
                return "human_review"
            return END

        graph.add_conditional_edges("verify_answer", _route_after_verify)
        graph.add_edge("human_review", END)
        return graph.compile()
    except Exception:
        return None


@dataclass
class ResearchAgent:
    """Lightweight agent runner with LangGraph-compatible node semantics."""

    compiled: Any | None = None

    def __post_init__(self) -> None:
        if self.compiled is None:
            self.compiled = _build_langgraph_agent()

    def invoke(self, payload: dict[str, Any]) -> ResearchAgentState:
        state: ResearchAgentState = {
            "query": (payload.get("query") or "").strip(),
            "scope": (payload.get("scope") or "both").strip().lower(),
            "doc_id": payload.get("doc_id"),
            "doc_ids": [int(x) for x in (payload.get("doc_ids") or []) if x is not None],
            "limit": int(payload.get("limit") or 6),
            "use_llm": bool(payload.get("use_llm", True)),
            "allow_general_background": bool(payload.get("allow_general_background", False)),
            "workspace_id": (payload.get("workspace_id") or "default").strip() or "default",
            "trace_id": payload.get("trace_id") or uuid.uuid4().hex,
        }
        if not state["query"]:
            raise ValueError("query is required")

        if self.compiled is not None:
            try:
                return self.compiled.invoke(state)
            except Exception:
                # Fall back to the explicit runner if the optional LangGraph
                # build is unavailable or the compiled graph errors out.
                pass

        state = _plan_retrieval(state)
        state = _retrieve_uploaded_docs(state)
        state = _retrieve_scholarly_sources(state)
        state = _combine_and_rerank(state)
        state = _draft_answer(state)
        state = _verify_answer(state)
        if state["final_answer"].needs_human_review:
            state = _human_review(state)
        return state


def build_research_agent() -> ResearchAgent:
    return ResearchAgent()
