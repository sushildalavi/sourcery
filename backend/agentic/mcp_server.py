from __future__ import annotations

from typing import Any, Callable


def _tools_module():
    from . import tools as tools_module

    return tools_module


def _fetchone(query: str, params: list[Any] | None = None):
    from backend.services.db import fetchone

    return fetchone(query, params)


def _fetchall(query: str, params: list[Any] | None = None):
    from backend.services.db import fetchall

    return fetchall(query, params)


def _search_documents_proxy(*args: Any, **kwargs: Any):
    return _tools_module().search_documents(*args, **kwargs)


def _retrieve_passages_proxy(*args: Any, **kwargs: Any):
    return _tools_module().retrieve_passages(*args, **kwargs)


def _search_uploaded_docs_proxy(*args: Any, **kwargs: Any):
    return _tools_module().search_uploaded_docs(*args, **kwargs)


def _search_scholarly_sources_proxy(*args: Any, **kwargs: Any):
    return _tools_module().search_scholarly_sources(*args, **kwargs)


def _rerank_evidence_proxy(*args: Any, **kwargs: Any):
    return _tools_module().rerank_evidence(*args, **kwargs)


def _judge_answer_support_proxy(*args: Any, **kwargs: Any):
    return _tools_module().judge_answer_support(*args, **kwargs)


def _score_citation_quality_proxy(*args: Any, **kwargs: Any):
    return _tools_module().score_citation_quality(*args, **kwargs)


def _require_workspace(workspace_id: str | None) -> str:
    ws = (workspace_id or "").strip()
    return ws or "default"


def _document_metadata(document_id: int, workspace_id: str = "default") -> dict[str, Any]:
    ws = _require_workspace(workspace_id)
    row = _fetchone(
        """
        SELECT
            d.id,
            d.title,
            d.doc_type,
            d.status,
            d.pages,
            d.bytes,
            d.hash_sha256,
            d.mime_type,
            d.created_at,
            COUNT(DISTINCT c.id) AS chunk_count,
            COUNT(DISTINCT ce.id) AS embedding_count
        FROM documents d
        LEFT JOIN chunks c
          ON c.document_id = d.id
         AND c.workspace_id = d.workspace_id
        LEFT JOIN chunk_embeddings ce
          ON ce.chunk_id = c.id
        WHERE d.id = %s
          AND d.workspace_id = %s
        GROUP BY d.id
        """,
        [document_id, ws],
    )
    if not row:
        return {"found": False, "workspace_id": ws, "document_id": document_id}
    return {
        "found": True,
        "workspace_id": ws,
        "document": row,
    }


def _recent_documents(workspace_id: str = "default", limit: int = 10) -> list[dict[str, Any]]:
    ws = _require_workspace(workspace_id)
    rows = _fetchall(
        """
        SELECT id, title, doc_type, status, pages, bytes, created_at
        FROM documents
        WHERE workspace_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        [ws, max(1, min(int(limit), 25))],
    )
    return rows


def _recent_eval_runs(limit: int = 10) -> list[dict[str, Any]]:
    rows = _fetchall(
        """
        SELECT id, name, scope, k, case_count, metrics_retrieval_only, metrics_retrieval_rerank, latency_breakdown, created_at
        FROM eval_runs
        ORDER BY created_at DESC
        LIMIT %s
        """,
        [max(1, min(int(limit), 25))],
    )
    return rows


def _recent_judge_runs(limit: int = 10) -> list[dict[str, Any]]:
    rows = _fetchall(
        """
        SELECT id, scope, query_count, metrics, details, created_at
        FROM evaluation_judge_runs
        ORDER BY created_at DESC
        LIMIT %s
        """,
        [max(1, min(int(limit), 25))],
    )
    return rows


def _latest_calibration(label: str = "unified", workspace_id: str = "default") -> dict[str, Any]:
    ws = _require_workspace(workspace_id)
    row = _fetchone(
        """
        SELECT id, model_name, label, weights, metrics, dataset_size, created_at
        FROM confidence_calibration
        WHERE label = %s AND workspace_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [label, ws],
    )
    if not row and ws != "default":
        row = _fetchone(
            """
            SELECT id, model_name, label, weights, metrics, dataset_size, created_at
            FROM confidence_calibration
            WHERE label = %s AND workspace_id = 'default'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [label],
        )
    if not row:
        return {"found": False, "label": label, "workspace_id": ws}
    return {"found": True, "workspace_id": ws, "calibration": row}


def tool_registry() -> dict[str, Callable[..., Any]]:
    return {
        "search_documents": _search_documents_proxy,
        "retrieve_passages": _retrieve_passages_proxy,
        "search_uploaded_documents": _search_uploaded_docs_proxy,
        "search_scholarly_web": _search_scholarly_sources_proxy,
        "rerank_research_evidence": _rerank_evidence_proxy,
        "evaluate_answer_support": _judge_answer_support_proxy,
        "score_citation_quality": _score_citation_quality_proxy,
        "get_document_metadata": _document_metadata,
        "list_recent_documents": _recent_documents,
        "list_recent_eval_runs": _recent_eval_runs,
        "list_recent_judge_runs": _recent_judge_runs,
        "get_latest_calibration": _latest_calibration,
    }


def build_mcp_app():
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        return None

    mcp = FastMCP("sourcery-agent-tools")

    @mcp.tool()
    def search_documents(
        query: str,
        limit: int = 5,
        workspace_id: str = "default",
        scope: str = "both",
        doc_id: int | None = None,
        doc_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump()
            for item in _search_documents_proxy(
                query,
                limit=limit,
                workspace_id=workspace_id,
                scope=scope,
                doc_id=doc_id,
                doc_ids=doc_ids,
            )
        ]

    @mcp.tool()
    def retrieve_passages(
        query: str,
        limit: int = 5,
        workspace_id: str = "default",
        doc_id: int | None = None,
        doc_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump()
            for item in _retrieve_passages_proxy(
                query,
                limit=limit,
                workspace_id=workspace_id,
                doc_id=doc_id,
                doc_ids=doc_ids,
            )
        ]

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
            for item in _search_uploaded_docs_proxy(
                query,
                limit=limit,
                workspace_id=workspace_id,
                doc_id=doc_id,
                doc_ids=doc_ids,
            )
        ]

    @mcp.tool()
    def search_scholarly_web(query: str, limit: int = 5) -> list[dict[str, Any]]:
        return [item.model_dump() for item in _search_scholarly_sources_proxy(query, limit=limit)]

    @mcp.tool()
    def rerank_research_evidence(query: str, evidence: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
        from .schemas import EvidenceItem

        items = [EvidenceItem.model_validate(row) for row in evidence]
        return [item.model_dump() for item in _rerank_evidence_proxy(query, items, limit=limit)]

    @mcp.tool()
    def evaluate_answer_support(query: str, answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        from .schemas import EvidenceItem

        items = [EvidenceItem.model_validate(row) for row in evidence]
        return _judge_answer_support_proxy(query, answer, items)

    @mcp.tool()
    def score_citation_quality(answer: str, abstained: bool = False) -> dict[str, Any]:
        return _score_citation_quality_proxy(answer, abstained=abstained)

    @mcp.tool()
    def get_document_metadata(document_id: int, workspace_id: str = "default") -> dict[str, Any]:
        return _document_metadata(document_id, workspace_id=workspace_id)

    @mcp.tool()
    def list_recent_documents(workspace_id: str = "default", limit: int = 10) -> list[dict[str, Any]]:
        return _recent_documents(workspace_id=workspace_id, limit=limit)

    @mcp.tool()
    def list_recent_eval_runs(limit: int = 10) -> list[dict[str, Any]]:
        return _recent_eval_runs(limit=limit)

    @mcp.tool()
    def list_recent_judge_runs(limit: int = 10) -> list[dict[str, Any]]:
        return _recent_judge_runs(limit=limit)

    @mcp.tool()
    def get_latest_calibration(label: str = "unified", workspace_id: str = "default") -> dict[str, Any]:
        return _latest_calibration(label=label, workspace_id=workspace_id)

    return mcp


def main() -> None:
    mcp = build_mcp_app()
    if mcp is None:
        raise RuntimeError("mcp is not installed. Install requirements.txt to run the MCP server.")
    mcp.run()


if __name__ == "__main__":
    main()
