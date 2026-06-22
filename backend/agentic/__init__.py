"""Agentic RAG workflow, evaluation helpers, and MCP tool exposure."""

from __future__ import annotations

__all__ = ["ResearchAgent", "build_research_agent"]


def __getattr__(name: str):
    if name in {"ResearchAgent", "build_research_agent"}:
        from .graph import ResearchAgent, build_research_agent

        return {"ResearchAgent": ResearchAgent, "build_research_agent": build_research_agent}[name]
    raise AttributeError(name)
