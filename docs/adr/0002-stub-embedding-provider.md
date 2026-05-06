# ADR 0002 — Stub embedding provider for tests + CI

* Status: Accepted
* Date: 2026-05-05

## Context

The default embedding provider is Ollama (`mxbai-embed-large`) for local
work and OpenAI as a paid alternative. CI shouldn't require either:

- Ollama isn't installed on standard GitHub runners, and pulling
  `mxbai-embed-large` adds ~700 MB to every job.
- OpenAI calls cost money per CI run and would leak the API key into
  workflow logs if mishandled.

## Decision

Add a third provider value: `EMBEDDING_PROVIDER=stub`.

The stub embedder hashes the input with SHA-256, expands the digest into
a fixed-length float vector, and L2-normalises it. Properties:

- **Deterministic** — same text always produces the same vector.
- **Distinct** — different text produces different vectors (collision
  probability ≈ 2⁻²⁵⁶).
- **Unit norm** — cosine similarity behaves like a real embedding.
- **Zero deps** — no network, no model files.

CI sets `EMBEDDING_PROVIDER=stub` and the full 170-test suite runs in ~2 s.

## Consequences

**Positive**

- CI runs offline. No Ollama install, no OpenAI bill, no rate limits.
- Tests of retrieval glue can assert on actual vector outputs without
  mocking the embedder.
- New contributors can `pytest backend/tests` immediately without the
  Ollama setup.

**Negative**

- The stub doesn't capture semantic relationships. Tests that assert
  "synonym should rank highly" must be marked as integration tests and
  run against Ollama / OpenAI manually.
- An accidental `EMBEDDING_PROVIDER=stub` in production would silently
  serve garbage retrieval. Mitigation: `/health/embeddings` exposes the
  active provider in its body so dashboards can alert if production
  flips to `stub`.

## Alternatives considered

| Option | Rejected because |
|---|---|
| Mock the embedder per-test | Spreads ~30 mock declarations across the suite; brittle and easy to forget. |
| Run Ollama inside a CI service container | Cold pull of the embedding model adds 90+ seconds per CI run. |
| Use OpenAI in CI | Real money per run; key-leak risk. |
