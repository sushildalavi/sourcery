"""Pydantic response schemas for the public-facing API surface.

These power FastAPI's OpenAPI documentation and runtime response validation.
Request bodies are still accepted as flexible dicts so we don't break in-flight
clients; response shapes are pinned so consumers can rely on them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ──────────────────────────── health + ops ─────────────────────────────────


class LivenessResponse(BaseModel):
    """`GET /` — process is alive and the FastAPI app is reachable."""

    message: str = Field(..., examples=["ScholarRAG backend is live!"])
    service: str = Field(..., examples=["scholarrag-backend"])
    version: str = Field(..., examples=["1.0"])
    uptime_seconds: float = Field(..., ge=0)


class ProbeStatus(BaseModel):
    ok: bool
    error: str | None = None


class HealthFullResponse(BaseModel):
    """`GET /health/full` — aggregated readiness across deps."""

    status: str = Field(..., description="`ok` or `degraded`", examples=["ok"])
    service: str
    version: str
    uptime_seconds: float
    checks: dict[str, dict[str, Any]]


class EmbeddingHealthResponse(BaseModel):
    """`GET /health/embeddings` — embedding-provider self-test."""

    provider: str = Field(..., examples=["ollama", "openai", "stub"])
    model: str
    embedding_version: str
    raw_dim: int
    vector_store_dim: int
    base_url: str | None = None
    openai_dimensions: int | None = None
    max_query_words: int
    max_doc_words: int
    ok: bool
    returned_dim: int | None = None
    error: str | None = None


# ──────────────────────────── metrics ──────────────────────────────────────


class LatencySummary(BaseModel):
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class MetricsResponse(BaseModel):
    """`GET /metrics` — operational counters + rolling latency."""

    updated_at: str
    documents: int = 0
    chunks: int = 0
    eval_runs: int = 0
    retrieval: dict[str, Any] = Field(default_factory=dict)
    latency_ms: LatencySummary = Field(default_factory=LatencySummary)


# ──────────────────────────── confidence ───────────────────────────────────


class CalibrationWeights(BaseModel):
    w1: float
    w2: float
    w3: float
    b: float


class CalibrationResponse(BaseModel):
    """`GET /confidence/calibration` — active MSA logistic weights."""

    model_name: str
    label: str
    weights: CalibrationWeights
    metrics: dict[str, Any] | None = None
    dataset_size: int = 0
    created_at: str | None = None


# ──────────────────────────── error envelope ───────────────────────────────


class ErrorResponse(BaseModel):
    """Uniform error envelope returned on 4xx / 5xx by FastAPI."""

    detail: str
    request_id: str | None = Field(
        default=None,
        description="Echo of the X-Request-ID header on the failing request.",
    )
