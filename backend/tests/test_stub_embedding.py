"""Tests for the stub embedding provider used by tests + CI."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _force_stub(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "stub")
    # Reload to pick up the env change in module-level constants.
    import importlib

    import backend.services.embeddings as emb

    importlib.reload(emb)
    yield emb
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    importlib.reload(emb)


def test_stub_is_deterministic(_force_stub):
    emb = _force_stub
    a = emb._stub_embedding("hello world")
    b = emb._stub_embedding("hello world")
    assert a == b


def test_stub_changes_with_input(_force_stub):
    emb = _force_stub
    a = emb._stub_embedding("hello world")
    b = emb._stub_embedding("hello mars")
    assert a != b


def test_stub_is_unit_norm(_force_stub):
    emb = _force_stub
    v = emb._stub_embedding("anything goes here")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_stub_returns_correct_dimension(_force_stub):
    emb = _force_stub
    v = emb._stub_embedding("dimension probe")
    assert len(v) == emb._STUB_DIM == 1024
