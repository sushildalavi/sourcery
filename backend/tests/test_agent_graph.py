from __future__ import annotations

from backend.agentic.graph import build_research_agent
from backend.agentic.schemas import EvidenceItem


def test_agent_returns_structured_answer(monkeypatch):
    from backend.agentic import graph as graph_module

    monkeypatch.setattr(
        graph_module,
        "search_uploaded_docs",
        lambda query, limit=8, workspace_id="default", doc_id=None, doc_ids=None: [
            EvidenceItem(
                source_id="uploaded:1",
                title="Retrieval-Augmented Generation",
                snippet="RAG combines retrieval with generation to ground answers.",
                score=0.9,
                source="uploaded",
                doc_id=1,
                chunk_id=11,
                page=2,
            )
        ],
    )
    monkeypatch.setattr(graph_module, "search_scholarly_sources", lambda *args, **kwargs: [])
    monkeypatch.setattr(graph_module, "rerank_evidence", lambda query, items, limit=6, **kwargs: items[:limit])

    agent = build_research_agent()
    result = agent.invoke(
        {
            "query": "What evidence supports retrieval-augmented generation?",
            "scope": "uploaded",
            "use_llm": False,
            "workspace_id": "default",
        }
    )

    final = result["final_answer"]
    assert result["plan"].query.startswith("What evidence")
    assert final.answer
    assert isinstance(final.citations, list)
    assert 0.0 <= final.confidence <= 1.0
    assert isinstance(final.unsupported_claims, list)


def test_agent_routes_weak_evidence_to_review(monkeypatch):
    from backend.agentic import graph as graph_module

    monkeypatch.setattr(graph_module, "search_uploaded_docs", lambda *args, **kwargs: [])
    monkeypatch.setattr(graph_module, "search_scholarly_sources", lambda *args, **kwargs: [])
    monkeypatch.setattr(graph_module, "rerank_evidence", lambda query, items, limit=6, **kwargs: [])

    agent = build_research_agent()
    result = agent.invoke(
        {
            "query": "What is retrieval augmented generation?",
            "scope": "both",
            "use_llm": False,
            "workspace_id": "default",
        }
    )

    final = result["final_answer"]
    assert final.needs_human_review is True
    assert "reliable evidence" in final.answer.lower()
