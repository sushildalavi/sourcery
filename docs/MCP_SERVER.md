# Sourcery MCP Server

Sourcery exposes its retrieval and evaluation helpers through a minimal MCP tool surface.

## Tools

- `search_uploaded_documents`
- `search_scholarly_web`
- `rerank_research_evidence`
- `evaluate_answer_support`

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

The server only exposes allowlisted retrieval and evaluation tools.
It does not expose arbitrary shell execution, unrestricted filesystem access, or generic network access.
