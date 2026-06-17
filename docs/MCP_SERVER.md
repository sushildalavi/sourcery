# Sourcery MCP Server

Sourcery exposes its retrieval and evaluation helpers through a minimal MCP tool surface.

## Tools

- `search_uploaded_documents`
- `search_scholarly_web`
- `rerank_research_evidence`
- `evaluate_answer_support`
- `get_document_metadata`
- `list_recent_documents`
- `list_recent_eval_runs`
- `list_recent_judge_runs`
- `get_latest_calibration`

## PostgreSQL tool surface

The extra tools above are read-only PostgreSQL helpers. They expose workspace-scoped
document metadata, recent evaluation runs, judge runs, and calibration summaries.
They do not accept arbitrary SQL and do not expose writes.

## Startup

Run the server with:

```bash
python -m backend.agentic.mcp_server
```

## Local config example

Add this to your MCP client config if you want to expose the Sourcery tool surface locally:

```json
{
  "mcpServers": {
    "sourcery-agent-tools": {
      "command": "python",
      "args": ["-m", "backend.agentic.mcp_server"]
    }
  }
}
```

## Safety model

The server only exposes allowlisted retrieval, evaluation, and read-only database tools.
It does not expose arbitrary shell execution, unrestricted filesystem access, arbitrary SQL, or generic network access.
