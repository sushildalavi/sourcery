# ScholarRAG Evaluation

Human-calibrated grounding confidence for the ScholarRAG RAG system, fit on a
15-paper diverse corpus with a 3-coder labeling protocol.

## Structure

```
Evaluation/
├── README.md
├── papers/                         # 15-paper corpus (gitignored PDFs)
│   ├── download_corpus.sh          # Reproducible multi-source downloader
│   └── MANIFEST.md                 # Corpus manifest with paper links
├── queries/
│   ├── queries_120.json            # 120 LLM-generated queries (4 types × 2 × 15 papers)
│   └── claim_evidence_pairs.json   # Extracted (claim, evidence) pairs from live system runs
└── data/
    └── calibration/
        ├── gold_labels.xlsx            # 530 majority-voted human labels (stratified sample, seed=42)
        ├── iaa_report.json            # Pairwise Cohen's kappa + distributions
        ├── label_distribution.json    # Per-coder + gold breakdown
        ├── features.xlsx               # M / S / A features per pair
        ├── calibration_fit.json       # Fitted logistic weights + ablation
        └── reliability_diagram.xlsx    # 10-bucket reliability diagram
```

## Reproducing the calibration end-to-end

Runs below assume PostgreSQL is reachable via `backend.services.db` and
`OPENAI_API_KEY` is set for NLI + embedding calls.

### 1. Corpus

```bash
# Download the 15 PDFs
cd Evaluation/papers && bash download_corpus.sh

# Ingest into the documents table (idempotent; skips already-ingested SHAs)
python -m backend.scripts.ingest_corpus
```

### 2. Queries

```bash
# Generate 120 GPT-4o-mini queries grounded on real chunks
python -m backend.scripts.generate_queries
```

### 3. Claim-evidence pair extraction + coder workbooks

```bash
# Run assistant_answer for each query, extract (claim, evidence) pairs,
# emit 3 identical coder xlsx files at /Users/…/Desktop/HUMAN EVAL/
CODEBOOK_MAX_QUERIES=80 CODEBOOK_INCLUDE_PUBLIC=true \
  python -m backend.scripts.build_codebooks
```

Coders independently fill dropdowns on the `Labeling` sheet of each workbook.

### 4. IAA + gold labels

```bash
# Compute pairwise Cohen's kappa + majority-vote resolve → gold_labels.xlsx
python -m backend.scripts.compute_iaa_majority
```

### 5. M / S / A feature extraction

```bash
# For each gold-labeled pair, compute:
#   M = NLI support probability (entailment + 0.3·neutral)
#   S = uploaded-retrieval stability via query perturbation
#   A = lexical multi-source corroboration (orthogonal to M)
python -m backend.scripts.extract_msa_features
```

### 6. Unified logistic fit + per-mode ablation

```bash
# Fits sigmoid(b + w1·M + w2·S + w3·A); runs pooled vs per-mode ablation;
# writes winning weights to confidence_calibration under label='unified'.
python -m backend.scripts.fit_unified_calibration --write-db
```

### 7. Deploy

```bash
# Switch the live backend from uniform-prior defaults to the fitted row
export CONFIDENCE_USE_FITTED_WEIGHTS=true
```

## Headline numbers

| Metric | Value |
|---|---|
| Corpus | 15 diverse papers · 7 subfields · 1997–2023 |
| Queries | 120 (definitional, methodology, factual, limitations) |
| Claim-evidence pairs | 530 (stratified sample by mode × query_type from 740 labelled; seed=42) |
| Coders | 3 independent |
| Cohen's κ (pairwise avg) | **0.47 (moderate)** — A-B 0.37, A-C 0.44, B-C 0.59 |
| Unanimous agreement | 59.8% |
| Gold label distribution | 50.4% supported / 49.6% unsupported |
| Calibration weights | `w₁(M)=3.81`, `w₂(S)=−0.29`, `w₃(A)=3.35`, `b=−4.86` |
| **Brier score** | **0.160** (lower = better; random = 0.25) |
| **AUC-ROC** | **0.852** |
| Unified-vs-per-mode Δ Brier | **0.003** → unified model empirically validated |
| 5-fold CV Brier (held-out) | **0.163 ± 0.015** |
| 5-fold CV AUC (held-out) | **0.845 ± 0.028** |
| Retrieval Recall@5 / MRR / nDCG@10 | **0.992 / 0.981 / 0.986** (120 queries) |

## Methodology notes

- **M / S / A orthogonality**: `A` is computed via lexical token overlap across
  distinct document sources — *not* via NLI — so it remains statistically
  independent of `M`. This prevents label leakage where the agreement feature
  would otherwise duplicate the entailment signal the logistic is trying to fit
  against.
- **Unified across scopes**: the same weights apply to both uploaded-mode
  (user's own PDFs) and public-mode (arXiv / Semantic Scholar / OpenAlex /
  Elsevier retrieval). Ablation confirmed pooled-fit Brier is within 0.003 of
  the per-mode-average Brier, so a single unified logistic is justified.
- **S weight ≈ 0**: retrieval stability did not discriminate in this dataset
  because the corpus's retrieval was uniformly stable across query variants.
  Kept in the framework for robustness against noisier retrievals.
- **κ = 0.47 (moderate)**: the diverse corpus and harder query types
  (methodology, limitations) require nuanced semantic judgment. Majority-vote
  resolution over the 530-pair sample yields 59.8% unanimous agreement.
- **Why 530 and not 740?** The three coders each labelled 740 pairs; the
  calibration fit uses a deterministic stratified sample (seed=42; strata =
  mode × query_type) drawn to N=530, preserving the labelled proportions.
  This matches the pre-registered dataset size; the larger 740-pair fit is
  within 0.005 Brier, so the downsample does not materially change results.
  The full 740 labels remain in the three coder workbooks for auditability.
