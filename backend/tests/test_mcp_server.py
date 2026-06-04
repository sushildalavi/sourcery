from __future__ import annotations

from backend.agentic.mcp_server import tool_registry


def test_mcp_registry_exposes_allowlisted_tools():
    tools = tool_registry()
    assert set(tools) == {
        "search_uploaded_documents",
        "search_scholarly_web",
        "rerank_research_evidence",
        "evaluate_answer_support",
    }

