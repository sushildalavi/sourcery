from __future__ import annotations

from typing import List

from .schemas import EvidenceItem


def has_sufficient_evidence(evidence: List[EvidenceItem], min_items: int = 3) -> bool:
    return len(evidence) >= max(1, int(min_items))


def should_require_human_review(confidence: float, unsupported_claims: List[str]) -> bool:
    return float(confidence) < 0.70 or len(unsupported_claims) > 0


def redact_sensitive_text(text: str) -> str:
    return (text or "").replace("@", "[at]")
