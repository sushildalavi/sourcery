from __future__ import annotations

import re
from dataclasses import dataclass, asdict

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION_MARKER = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitationQualityScore:
    num_answer_claims: int
    num_cited_claims: int
    unsupported_claim_count: int
    citation_coverage_ratio: float
    abstention_triggered: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _extract_claims(answer: str) -> list[str]:
    text = answer.strip()
    if not text:
        return []
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    claims = [sentence for sentence in sentences if len(sentence.split()) >= 4]
    return claims or [text]


def score_citation_quality(answer: str, *, abstained: bool = False) -> CitationQualityScore:
    claims = _extract_claims(answer)
    cited_claims = sum(1 for claim in claims if _CITATION_MARKER.search(claim))
    unsupported_claims = 0 if abstained else max(0, len(claims) - cited_claims)
    coverage = 0.0 if not claims else round(cited_claims / len(claims), 4)
    return CitationQualityScore(
        num_answer_claims=len(claims),
        num_cited_claims=cited_claims,
        unsupported_claim_count=unsupported_claims,
        citation_coverage_ratio=coverage,
        abstention_triggered=abstained,
    )

