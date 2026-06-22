from __future__ import annotations

from backend.agentic.mcp_server import tool_registry


def test_mcp_registry_exposes_requested_read_only_aliases():
    tools = tool_registry()
    assert "search_documents" in tools
    assert "retrieve_passages" in tools
    assert "score_citation_quality" in tools
    assert "get_latest_calibration" in tools
