from __future__ import annotations

from typing import TypedDict

from .schemas import AgentAnswer, EvidenceItem, RetrievalPlan


class ResearchAgentState(TypedDict, total=False):
    query: str
    scope: str
    doc_id: int | None
    doc_ids: list[int]
    limit: int
    use_llm: bool
    allow_general_background: bool
    workspace_id: str
    trace_id: str
    plan: RetrievalPlan
    uploaded_doc_results: list[EvidenceItem]
    scholarly_results: list[EvidenceItem]
    reranked_evidence: list[EvidenceItem]
    draft_answer: AgentAnswer
    final_answer: AgentAnswer
    judge_report: dict
    retrieval_metadata: dict
    error: str
