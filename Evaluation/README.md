# ScholarRAG evaluation

Human-calibrated grounding confidence for the ScholarRAG system.

## Structure

- Corpus manifest and source downloads
- Query sets and claim/evidence pairs
- Calibration labels and feature workbooks
- Reliability outputs and fitted weights

## Reproduce

1. Download the corpus.
2. Generate queries.
3. Build claim/evidence workbooks.
4. Compute inter-annotator agreement and gold labels.
5. Extract M, S, and A features.
6. Fit the calibration model.
7. Deploy the fitted weights.

## Headline numbers

- 15-paper corpus
- 120 queries
- 530 claim/evidence pairs
- 3 coders
- Brier score 0.160
- AUC-ROC 0.852
