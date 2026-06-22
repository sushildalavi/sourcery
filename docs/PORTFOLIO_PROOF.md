# Portfolio Proof

## What the project does

Sourcery is a local scholarly QA system with hybrid retrieval, citations, calibrated confidence, and cite-or-abstain behavior.

## Why it is technically impressive

- Hybrid dense and sparse retrieval.
- Calibration artifacts and retrieval evaluation are already checked into the repo.
- The system is designed to abstain instead of hallucinating when evidence is weak.

## Architecture summary

- Query -> retrieval -> evidence selection -> answer generation -> confidence/citation handling.
- Evaluation lives under `evaluation/`, backend logic under `backend/`, and UI under `frontend/`.

## How to run locally

- `make quick-start`
- Frontend dev: `npm run dev` in `frontend/`

## How to test

- Backend tests via `pytest`
- Frontend build via `npm run build`
- Linting via `ruff` if configured in the local environment

## How to benchmark or evaluate

- Review `evaluation/data/calibration/cv_metrics.json`
- Review `evaluation/data/retrieval_eval_120.json`

## Verified metrics only

- Brier mean: 0.1632
- Brier std: 0.015
- AUC mean: 0.8446
- AUC std: 0.0277
- Recall@5: 0.9917
- MRR: 0.9812
- nDCG@10: 0.9857

## Current limitations

- Citation-quality scoring is not yet documented as a standalone artifact.
- Local embedding mode guidance is still scattered across the repo.

## Future improvements

- Add a deterministic citation-quality utility and tests.
- Add a calibration report and local retrieval mode guide.
- Surface verified metrics in a compact recruiter-friendly table.

## Resume bullets

- Built a citation-grounded QA system with hybrid retrieval and abstention on weak evidence.
- Documented calibration and retrieval quality with verified evaluation artifacts.
- Designed an offline-first scholarly assistant with calibrated confidence rather than blind generation.

## Verification Log

- `python3 -m pytest /Users/sushildalavi/Desktop/Github/sourcery/backend/tests/test_citation_quality.py` - pass - 2026-06-16 - Verified deterministic citation scoring behavior.
- `python3 -m compileall /Users/sushildalavi/Desktop/Github/sourcery/backend/citation_quality.py` - pass - 2026-06-16 - Verified syntax.
- Artifacts generated: none beyond pytest output.
