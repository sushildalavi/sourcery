from __future__ import annotations

from typing import Any, Callable

from .tools import judge_answer_support, rerank_evidence, search_scholarly_sources, search_uploaded_docs


def tool_registry() -> dict[str, Callable[..., Any]]:
    return {
        "search_uploaded_documents": search_uploaded_docs,
        "search_scholarly_web": search_scholarly_sources,
        "rerank_research_evidence": rerank_evidence,
        "evaluate_answer_support": judge_answer_support,
    }


def build_mcp_app():
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        return None

    mcp = FastMCP("sourcery-agent-tools")

    @mcp.tool()
    def search_uploaded_documents(
        query: str,
        limit: int = 5,
        workspace_id: str = "default",
        doc_id: int | None = None,
        doc_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump()
            for item in search_uploaded_docs(
                query,
                limit=limit,
                workspace_id=workspace_id,
                doc_id=doc_id,
                doc_ids=doc_ids,
            )
        ]

    @mcp.tool()
    def search_scholarly_web(query: str, limit: int = 5) -> list[dict[str, Any]]:
        return [item.model_dump() for item in search_scholarly_sources(query, limit=limit)]

    @mcp.tool()
    def rerank_research_evidence(query: str, evidence: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
        from .schemas import EvidenceItem

        items = [EvidenceItem.model_validate(row) for row in evidence]
        return [item.model_dump() for item in rerank_evidence(query, items, limit=limit)]

    @mcp.tool()
    def evaluate_answer_support(query: str, answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        from .schemas import EvidenceItem

        items = [EvidenceItem.model_validate(row) for row in evidence]
        return judge_answer_support(query, answer, items)

    return mcp


def main() -> None:
    mcp = build_mcp_app()
    if mcp is None:
        raise RuntimeError("mcp is not installed. Install requirements.txt to run the MCP server.")
    mcp.run()


if __name__ == "__main__":
    main()

