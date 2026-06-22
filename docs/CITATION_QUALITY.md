# Citation Quality

This repository now has a deterministic citation-quality utility at `backend/citation_quality.py`.

## Scored fields

- `num_answer_claims`
- `num_cited_claims`
- `unsupported_claim_count`
- `citation_coverage_ratio`
- `abstention_triggered`

## How it works

- Splits the answer into simple sentence-level claims.
- Counts claims that contain explicit citation markers like `[1]`.
- Treats uncited claims as unsupported unless the answer abstains.

## What it is good for

- Fast local evaluation
- Regression checks on cite-or-abstain behavior
- Portfolio proof that the system prefers grounded answers

## Limitations

- It is intentionally deterministic and lightweight.
- It is not a semantic claim verifier.

