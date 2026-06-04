from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    EVIDENCE_SEEKING = "evidence_seeking"
    WEAK_EVIDENCE_RISK = "weak_evidence_risk"


class RetrievalSource(str, Enum):
    UPLOADED_DOCS = "uploaded_docs"
    SCHOLARLY_WEB = "scholarly_web"
    BOTH = "both"


class RetrievalPlan(BaseModel):
    query: str
    intent: QueryIntent
    source_strategy: RetrievalSource
    required_evidence_count: int = Field(default=4, ge=1, le=12)
    require_citations: bool = True
    risk_notes: list[str] = Field(default_factory=list)
    allow_general_background: bool = False
    scope_hint: str = "both"
    doc_id: int | None = None
    doc_ids: list[int] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    source_id: str
    title: str
    snippet: str
    url: Optional[str] = None
    score: float = 0.0
    citation: Optional[str] = None
    source: str = "uploaded"
    doc_id: int | None = None
    chunk_id: int | None = None
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentAnswer(BaseModel):
    answer: str
    citations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    needs_human_review: bool = False

