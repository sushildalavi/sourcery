"""Tests for the second-stage reranker."""

from __future__ import annotations

import importlib
import os

import pytest

from backend.services import reranker


@pytest.fixture(autouse=True)
def _enable_rerank(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    importlib.reload(reranker)
    yield


def test_rerank_returns_scored_candidates_in_descending_order():
    out = reranker.rerank_candidates(
        "graph neural network molecular property prediction",
        [
            {"id": 1, "text": "completely unrelated text about cooking", "score": 0.8},
            {"id": 2, "text": "graph neural network molecular property prediction", "score": 0.5},
            {"id": 3, "text": "neural network for property prediction", "score": 0.6},
        ],
    )
    assert [c.chunk_id for c in out][0] == 2
    assert all(out[i].final_score >= out[i + 1].final_score for i in range(len(out) - 1))


def test_rerank_exact_phrase_promotes_candidate():
    query = "deep residual learning"
    out = reranker.rerank_candidates(
        query,
        [
            {"id": 1, "text": "an unrelated discussion of cats", "score": 0.5},
            {"id": 2, "text": "a paper introducing deep residual learning", "score": 0.5},
        ],
    )
    assert out[0].chunk_id == 2


def test_top_k_truncates():
    out = reranker.rerank_candidates(
        "x",
        [{"id": i, "text": f"x {i}", "score": 0.1 * i} for i in range(10)],
        top_k=3,
    )
    assert len(out) == 3


def test_disabled_rerank_passes_through_stage1_order(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "false")
    importlib.reload(reranker)
    try:
        out = reranker.rerank_candidates(
            "anything",
            [
                {"id": 1, "text": "perfectly matching anything word", "score": 0.1},
                {"id": 2, "text": "no overlap whatsoever", "score": 0.9},
            ],
        )
        # With rerank disabled, order should be by stage1 (0.9 > 0.1).
        assert out[0].chunk_id == 2
        assert out[0].stage2_score == 0.0
    finally:
        os.environ.pop("RERANK_ENABLED", None)
        importlib.reload(reranker)
