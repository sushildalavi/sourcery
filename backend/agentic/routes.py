from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.middleware import current_workspace

from .graph import build_research_agent

router = APIRouter(prefix="/agent", tags=["agentic-rag"])

_agent = build_research_agent()


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    scope: str = Field(default="both", description="uploaded | public | both")
    doc_id: int | None = None
    doc_ids: list[int] = Field(default_factory=list)
    limit: int = Field(default=6, ge=1, le=12)
    use_llm: bool = True
    allow_general_background: bool = False
    trace_id: str | None = None


@router.post("/research")
def research_agent_route(request: Request, payload: AgentQueryRequest):
    ws = current_workspace(request)
    try:
        result = _agent.invoke(
            {
                "query": payload.query,
                "scope": payload.scope,
                "doc_id": payload.doc_id,
                "doc_ids": payload.doc_ids,
                "limit": payload.limit,
                "use_llm": payload.use_llm,
                "allow_general_background": payload.allow_general_background,
                "trace_id": payload.trace_id,
                "workspace_id": ws,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    final = result["final_answer"]
    evidence = [item.model_dump() for item in result.get("reranked_evidence", [])]
    return {
        "trace_id": result.get("trace_id"),
        "workspace_id": ws,
        "plan": result.get("plan").model_dump() if result.get("plan") else None,
        "answer": final.answer,
        "citations": final.citations,
        "confidence": final.confidence,
        "unsupported_claims": final.unsupported_claims,
        "needs_human_review": final.needs_human_review,
        "evidence": evidence,
        "judge_report": result.get("judge_report"),
        "retrieval_metadata": result.get("retrieval_metadata"),
    }
