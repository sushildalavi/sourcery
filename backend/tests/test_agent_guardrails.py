from __future__ import annotations

from backend.agentic.guardrails import has_sufficient_evidence, should_require_human_review
from backend.agentic.schemas import EvidenceItem


def test_insufficient_evidence_fails_guardrail():
    assert not has_sufficient_evidence([])


def test_evidence_threshold_passes_guardrail():
    items = [
        EvidenceItem(source_id="a", title="A", snippet="x"),
        EvidenceItem(source_id="b", title="B", snippet="y"),
        EvidenceItem(source_id="c", title="C", snippet="z"),
    ]
    assert has_sufficient_evidence(items)


def test_low_confidence_requires_human_review():
    assert should_require_human_review(0.4, [])


def test_unsupported_claim_requires_human_review():
    assert should_require_human_review(0.9, ["unsupported claim"])
