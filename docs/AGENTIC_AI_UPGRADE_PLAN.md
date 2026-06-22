# Sourcery Agentic AI Upgrade Plan

## Current relevant capabilities
- Existing agentic research workflow already exists in `backend/agentic/graph.py`, `backend/agentic/routes.py`, `backend/agentic/tools.py`, `backend/agentic/mcp_server.py`, `backend/agentic/traces.py`, and `backend/agentic/evaluation.py`.
- Deterministic citation-quality scoring exists in `backend/citation_quality.py`.
- Confidence calibration and abstention logic already exist in `backend/confidence.py` and `backend/services/judge.py`.
- MCP surface is read-only and already exposes retrieval and evaluation helpers.
- Frontend includes an `AgenticResearch` route and supporting chat/research UI.

## Safest agentic extension points
- Extend the current agentic graph instead of replacing it.
- Add a thin, traceable workflow wrapper that reuses `search_uploaded_docs`, `search_scholarly_sources`, `rerank_evidence`, and `judge_answer_support`.
- Keep MCP additions read-only and limited to data lookup / scoring helpers.
- Keep local/demo mode fixture-backed and deterministic by default.

## Proposed files to change
- `backend/agentic/trace.py` or `backend/agentic/traces.py`
- `backend/agentic/tools.py`
- `backend/agentic/research_workflow.py` if a wrapper is added
- `backend/agentic/routes.py`
- `backend/agentic/mcp_server.py`
- `backend/tests/test_agentic_research.py`
- `docs/AGENTIC_RESEARCH_WORKFLOW.md`
- `docs/MCP_TOOLS.md`
- `docs/AGENT_TRACE_SCHEMA.md`

## Tests to add
- Planner creates subquestions or plan metadata from representative queries.
- Retrieval tool returns fixture-backed evidence.
- Citation judge rejects unsupported claims.
- Workflow abstains when evidence is insufficient.
- Trace records every step with stable fields.
- MCP helpers remain read-only and do not expose arbitrary SQL or writes.

## Local demo command
- `python -m backend.agentic.mcp_server`
- `curl -X POST http://localhost:8000/agent/research ...`

## Risks / unknowns
- The current workflow already depends on several backend services; over-wiring a new agent layer could duplicate logic.
- Optional LLM usage may require API keys, so the default path should remain local and deterministic.
- The exact state schema should be checked before introducing new trace fields.

## What not to claim
- Do not claim autonomous research or production-grade autonomy.
- Do not claim semantic claim verification beyond the current deterministic support checks.
- Do not claim paid APIs are required for the default demo.
- Do not claim all answers are fully grounded if the workflow may abstain.
