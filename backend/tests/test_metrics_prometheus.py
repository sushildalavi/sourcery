"""Tests for the Prometheus exposition endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_prom_returns_prometheus_text(monkeypatch):
    """Stub the metrics() impl so we don't need a live DB for this assertion."""
    from backend import app as app_module

    def _fake_metrics():
        return {
            "updated_at": "2026-05-06T12:00:00Z",
            "documents": 42,
            "chunks": 1337,
            "eval_runs": 7,
            "retrieval": {"recall_at_5": 0.99, "ndcg_at_10": 0.97, "mrr": 0.95},
            "latency_ms": {"p50": 420.0, "p95": 980.0, "p99": 1600.0},
        }

    monkeypatch.setattr(app_module, "metrics", _fake_metrics)

    client = TestClient(app_module.app)
    resp = client.get("/metrics/prom")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text

    # Spot-check the wire format.
    assert "# HELP sourcery_documents_total" in body
    assert "# TYPE sourcery_documents_total gauge" in body
    assert "sourcery_documents_total 42.0" in body
    assert "sourcery_chunks_total 1337.0" in body
    assert "sourcery_retrieval_recall_at_5 0.99" in body
    assert "sourcery_assistant_latency_ms_p99 1600.0" in body


def test_metrics_prom_skips_missing_values(monkeypatch):
    """If a metric is None, it should be omitted (not emitted as `nan`)."""
    from backend import app as app_module

    monkeypatch.setattr(
        app_module,
        "metrics",
        lambda: {"documents": None, "chunks": 5, "retrieval": {}, "latency_ms": {}},
    )

    client = TestClient(app_module.app)
    body = client.get("/metrics/prom").text

    assert "sourcery_documents_total" not in body
    assert "sourcery_chunks_total 5.0" in body
