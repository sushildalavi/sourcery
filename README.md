# sourcery

sourcery is a local scholarly answer system with citations, confidence scoring, and offline-first operation.

## What it does

- Answers questions using retrieved evidence
- Refuses to answer when evidence is weak
- Tracks calibrated confidence
- Combines dense and sparse retrieval
- Runs locally with optional external APIs
- Includes benchmarks, evaluation, and Helm deployment support

## Quick start

```bash
make quick-start
```

## Highlights

- Cite-or-abstain behavior
- Calibrated confidence score
- Hybrid retrieval
- Offline support
- Prometheus and Grafana observability

## Notes

- The project is tuned for local use first.
- External services improve results when available, but they are optional.
- The repository includes evaluation and deployment material for reviewers.

## Portfolio Proof

- Architecture and evaluation: [docs/PORTFOLIO_PROOF.md](/Users/sushildalavi/Desktop/Github/sourcery/docs/PORTFOLIO_PROOF.md)
- Verified metrics: [evaluation/data/calibration/cv_metrics.json](/Users/sushildalavi/Desktop/Github/sourcery/evaluation/data/calibration/cv_metrics.json) and [evaluation/data/retrieval_eval_120.json](/Users/sushildalavi/Desktop/Github/sourcery/evaluation/data/retrieval_eval_120.json)
- Demo and local mode: use `make quick-start`
- Test commands: backend pytest, frontend `npm run build`
