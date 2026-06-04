# Sourcery Agentic RAG

Sourcery now includes an agentic research workflow that plans retrieval, searches the uploaded corpus and/or scholarly web, reranks evidence, generates a citation-grounded answer, and then verifies support before returning the final result.

## Workflow

1. Resolve the query intent and choose a retrieval strategy.
2. Retrieve candidate evidence from uploaded documents, public scholarly sources, or both.
3. Rerank the evidence with the existing lexical + source-prior stack.
4. Build a grounded answer with inline citations.
5. Judge answer support against the retrieved evidence.
6. Route low-confidence or unsupported answers to human review.

## API

`POST /agent/research`

Request shape:

```json
{
  "query": "What evidence supports retrieval-augmented generation?",
  "scope": "both",
  "limit": 6,
  "use_llm": true
}
```

Response includes:

- `answer`
- `citations`
- `confidence`
- `unsupported_claims`
- `needs_human_review`
- `evidence`
- `judge_report`
- `retrieval_metadata`

## Safety and guardrails

- Structured Pydantic models for workflow state and outputs.
- Citation-required answer formatting.
- Human-review routing for unsupported or weakly supported answers.
- JSONL trace logs under `artifacts/agent_traces/`.
- Fallback behavior when the LLM is unavailable.

