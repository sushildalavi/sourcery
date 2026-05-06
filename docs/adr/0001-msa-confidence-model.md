# ADR 0001 — MSA confidence over single-similarity

* Status: Accepted
* Date: 2026-04-15

## Context

We need to ship a numeric confidence next to every answer. The two
common defaults are (a) cosine similarity of the top retrieved chunk to
the query, or (b) a softmax over the LLM's token logprobs. Both have
known failure modes:

- Cosine **measures retrieval proximity, not faithfulness**. A near-duplicate
  chunk can be retrieved with high similarity yet not actually entail the
  generated claim.
- Token logprobs are **available only for some providers** and are unreliable
  when the LLM was nudged into a citation contract via prompt instead of
  fine-tuning.

## Decision

Compute three orthogonal signals per claim and fit a calibrated logistic:

- **M** — NLI entailment probability of the claim against the cited chunk.
- **S** — retrieval stability: fraction of perturbed-query reruns in which
  the same chunk surfaces in top-K.
- **A** — multi-source agreement: lexical (not NLI) overlap of the claim
  with chunks from *other* sources, used as a corroboration prior.

`P(supported | M, S, A) = σ(b + w₁·M + w₂·S + w₃·A)`

`A` is computed via lexical overlap, not NLI, so it is statistically
independent of `M`. This prevents the calibration fit from trivially
achieving accuracy by reusing the M signal twice.

## Consequences

**Positive**

- Brier 0.160 on the 530-pair gold set vs 0.25 random baseline (36% better).
- AUC 0.852, holds at 0.845 ± 0.028 under 5-fold CV.
- Per-mode ablation (uploaded vs public) within 0.003 Brier of the unified
  fit, justifying a single calibration row across modes.

**Negative**

- Three signals = three pipelines. NLI adds ~30 ms p50 to the answer path.
- Calibration requires labeled data. We ship a labeled set (530 pairs, 3
  coders, κ = 0.47) and the fit script to regenerate weights for new
  corpora.

## Alternatives considered

| Option | Rejected because |
|---|---|
| Cosine similarity only | Doesn't track faithfulness; AUC ~0.62 on our gold set. |
| Token logprobs | Provider-specific; unstable under our citation prompt. |
| LLM-as-judge per request | 3-5× cost per query; we use it only for offline eval, not online confidence. |
| Per-mode separate fits | Marginal gain (Δ Brier 0.003) doesn't justify the operational complexity of two weight rows. |
