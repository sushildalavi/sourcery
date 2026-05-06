from __future__ import annotations

import math
from typing import Dict, Optional


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def confidence_label(score: float) -> str:
    s = clamp01(score)
    if s >= 0.75:
        return "High"
    if s >= 0.5:
        return "Med"
    return "Low"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_msa_score(m: float, s: float, a: float, weights: Optional[Dict[str, float]] = None) -> float:
    """Compute calibrated confidence from M/S/A in [0,1] using logistic combination."""
    m = clamp01(m)
    s = clamp01(s)
    a = clamp01(a)
    w = weights or {}
    base = (
        float(w.get("b", 0.0))
        + float(w.get("w1", 0.58)) * m
        + float(w.get("w2", 0.22)) * s
        + float(w.get("w3", 0.20)) * a
    )
    return clamp01(sigmoid(base))


def build_confidence(
    *,
    top_sim: float,
    top_rerank_norm: float,
    citation_coverage: float,
    evidence_margin: float,
    ambiguity_penalty: float,
    insufficiency_penalty: float,
    scope_penalty: float = 0.0,
    needs_clarification: bool = False,
    msa: Optional[Dict[str, float]] = None,
    minimum_score: float = 0.0,
    scope: Optional[str] = None,
) -> Dict:
    """Compute a per-citation confidence score (MSA-only).

    When MSA features (M/S/A) are provided, confidence = `sigmoid(b + w1·M + w2·S + w3·A)`
    using the currently-loaded calibration weights. This matches what the logistic
    is fit against — the previous `0.62·retrieval_score + 0.38·msa_score` blend
    used an uncalibrated fixed coefficient and is no longer applied.

    If no MSA is supplied (legacy callers, smoke tests), the function falls back
    to the pre-existing retrieval-only score computed from sim, rerank, coverage,
    margin and penalised by ambiguity/insufficiency/scope. This fallback is NOT
    a validated calibration — it exists only so callers without MSA features
    still receive a numeric score.

    `scope` is accepted for symmetry with `_load_latest_calibration_weights(scope)`
    upstream; it is no longer used for branching here now that both modes are
    MSA-only. `needs_clarification` and `minimum_score` still apply as global
    sanity bounds.
    """
    sim = clamp01(top_sim)
    rerank = clamp01(top_rerank_norm)
    coverage = clamp01(citation_coverage)
    margin = clamp01(evidence_margin)
    amb_pen = clamp01(ambiguity_penalty)
    ins_pen = clamp01(insufficiency_penalty)
    scope_pen = clamp01(scope_penalty)

    # Retrieval-side score is kept only as a legacy fallback when no MSA is supplied.
    base = clamp01((0.35 * sim) + (0.25 * rerank) + (0.25 * coverage) + (0.15 * margin))
    retrieval_score = clamp01(base - (0.45 * amb_pen) - (0.4 * ins_pen) - (0.5 * scope_pen))

    msa_score = None
    if isinstance(msa, dict):
        weights = msa.get("weights")
        # `weights` may legitimately be a dict, None, or (in legacy callers)
        # a scalar — coerce to the typed shape compute_msa_score expects.
        weights_arg: dict | None = weights if isinstance(weights, dict) else None
        msa_score = compute_msa_score(msa.get("M", 0.0), msa.get("S", 0.0), msa.get("A", 0.0), weights_arg)

    if msa_score is not None:
        score = clamp01(msa_score)
    else:
        score = retrieval_score

    min_score = clamp01(minimum_score)
    if min_score > 0.0 and not needs_clarification:
        score = max(score, min_score)

    if needs_clarification:
        score = min(score, 0.25)

    explanation = (
        "Per-citation confidence is the calibrated MSA logistic: "
        "sigmoid(b + w1·M + w2·S + w3·A). "
        "M = entailment probability between claim and evidence, "
        "S = retrieval-stability rate, "
        "A = multi-source lexical corroboration."
    )

    factors = {
        "top_sim": round(sim, 4),
        "top_rerank_norm": round(rerank, 4),
        "citation_coverage": round(coverage, 4),
        "evidence_margin": round(margin, 4),
        "ambiguity_penalty": round(amb_pen, 4),
        "insufficiency_penalty": round(ins_pen, 4),
        "scope_penalty": round(scope_pen, 4),
        "msa_source": "calibrated" if msa_score is not None else "heuristic_fallback",
    }
    if min_score > 0.0:
        factors["minimum_score"] = round(min_score, 4)
    if msa is not None:
        factors["msa"] = {
            "M": round(clamp01(msa.get("M", 0.0)), 4),
            "S": round(clamp01(msa.get("S", 0.0)), 4),
            "A": round(clamp01(msa.get("A", 0.0)), 4),
            "msa_score": round(msa_score, 4) if msa_score is not None else 0.0,
            "weights": msa.get("weights", {}),
        }

    return {
        "score": round(score, 4),
        "label": confidence_label(score),
        "needs_clarification": bool(needs_clarification),
        "factors": factors,
        "explanation": explanation,
    }


def score_percent(probability: float) -> float:
    return round(clamp01(probability) * 100.0, 2)
