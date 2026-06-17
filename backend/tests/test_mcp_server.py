from __future__ import annotations

from backend.agentic import mcp_server
from backend.agentic.mcp_server import tool_registry


def test_mcp_registry_exposes_allowlisted_tools():
    tools = tool_registry()
    assert set(tools) == {
        "search_uploaded_documents",
        "search_scholarly_web",
        "rerank_research_evidence",
        "evaluate_answer_support",
        "get_document_metadata",
        "list_recent_documents",
        "list_recent_eval_runs",
        "list_recent_judge_runs",
        "get_latest_calibration",
    }


def test_document_metadata_tool_is_workspace_scoped(monkeypatch):
    calls = {}

    def fake_fetchone(query, params=None):
        calls["query"] = query
        calls["params"] = params
        return {
            "id": 7,
            "title": "Doc",
            "doc_type": "research_paper",
            "status": "ready",
            "pages": 12,
            "bytes": 1024,
            "hash_sha256": "abc",
            "mime_type": "application/pdf",
            "created_at": "2025-06-04",
            "chunk_count": 3,
            "embedding_count": 3,
        }

    monkeypatch.setattr(mcp_server, "fetchone", fake_fetchone)
    out = mcp_server._document_metadata(7, workspace_id="alpha")
    assert out["found"] is True
    assert calls["params"] == [7, "alpha"]


def test_latest_calibration_falls_back_to_default_workspace(monkeypatch):
    calls = []

    def fake_fetchone(query, params=None):
        calls.append((query, params))
        if len(calls) == 1:
            return None
        return {
            "id": 3,
            "model_name": "msa_logistic_v1",
            "label": "unified",
            "weights": {"w1": 1.0, "w2": 2.0, "w3": 3.0, "b": 4.0},
            "metrics": {"brier": 0.16},
            "dataset_size": 530,
            "created_at": "2025-06-04",
        }

    monkeypatch.setattr(mcp_server, "fetchone", fake_fetchone)
    out = mcp_server._latest_calibration(label="unified", workspace_id="tenant-a")
    assert out["found"] is True
    assert len(calls) == 2
