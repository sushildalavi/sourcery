from __future__ import annotations

from backend.citation_quality import score_citation_quality


def test_citation_quality_counts_cited_claims():
    score = score_citation_quality("The paper improves retrieval with hybrid search [1]. It also calibrates confidence [2].")
    assert score.num_answer_claims == 2
    assert score.num_cited_claims == 2
    assert score.unsupported_claim_count == 0
    assert score.citation_coverage_ratio == 1.0
    assert score.abstention_triggered is False


def test_citation_quality_flags_unsupported_claims():
    score = score_citation_quality("The paper improves retrieval with hybrid search. It also calibrates confidence [2].")
    assert score.num_answer_claims == 2
    assert score.num_cited_claims == 1
    assert score.unsupported_claim_count == 1
    assert score.citation_coverage_ratio == 0.5


def test_citation_quality_tracks_abstention():
    score = score_citation_quality("Insufficient evidence to answer.", abstained=True)
    assert score.abstention_triggered is True
    assert score.unsupported_claim_count == 0

