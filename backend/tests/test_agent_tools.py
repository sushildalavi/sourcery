from __future__ import annotations

from backend.agentic.schemas import EvidenceItem
from backend.agentic.tools import rerank_evidence


def test_rerank_limits_results():
    items = [
        EvidenceItem(source_id="a", title="A", snippet="retrieval augmented generation", score=0.1),
        EvidenceItem(source_id="b", title="B", snippet="dense retrieval and sparse overlap", score=0.9),
    ]

    out = rerank_evidence("dense retrieval", items, limit=1)
    assert len(out) == 1
    assert out[0].title == "B"

