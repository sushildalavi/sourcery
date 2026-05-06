# Architecture — citelens

This document describes the **why** behind the major design decisions. The
**what** lives in the [README architecture diagrams](../README.md#architecture).

## Goals

1. **Cite or abstain.** Every claim returned to the user must point at a chunk
   the system can re-render. If we can't ground a claim, we say so instead of
   hallucinating.
2. **Calibrated, not just confident.** Confidence numbers must track empirical
   accuracy on a held-out set; vibes don't ship.
3. **Local-first, cloud-optional.** A laptop with Docker + Ollama can run the
   full stack offline. Cloud (OpenAI, scholarly APIs) is additive, not
   load-bearing.
4. **Reproducible benchmarks.** Anyone can re-run the calibration pipeline and
   replicate the numbers in the README.

## Layered architecture

```
┌─ Frontend ───────────── React + Vite + Tailwind, lazy-loaded routes
├─ API ────────────────── FastAPI, Pydantic schemas, OWASP headers
├─ Domain ─────────────── chat · intent · sense · confidence · ingest
├─ Services ───────────── embeddings · db · NLI · LLM judge · feed
├─ Providers ──────────── arxiv · openalex · s2 · crossref · springer · elsevier
└─ Storage ────────────── PostgreSQL + pgvector · embedding cache
```

## Major design decisions

| # | Decision | Why |
|---|---|---|
| 1 | Hybrid dense + sparse retrieval (pgvector ANN + token overlap) | Pure dense recall on the calibration corpus underperformed BM25-style keyword overlap on factual / acronym queries; combining both boosted Recall@10 from 0.71 → 0.82. |
| 2 | M / S / A confidence (not single similarity) | Cosine similarity correlates with retrieval quality, not faithfulness. M (entailment), S (stability under query perturbation), A (cross-source agreement) are statistically distinct and jointly hit AUC 0.852 on a 530-pair gold set. |
| 3 | Calibration uses `A = lexical agreement`, **not** NLI agreement | Avoids label leakage — A and M (NLI) must be statistically independent for the logistic fit to be honest. |
| 4 | Embedding versioning contract in the DB | Each chunk row carries `(provider, model, version, dim)`. The query path filters by the active contract, so silent vector mixing across model upgrades is impossible. |
| 5 | Pluggable embedding providers (`ollama` / `openai` / `stub`) | The `stub` provider hashes input to a deterministic L2-normalised vector — tests and CI run zero-dependency. |
| 6 | LLM-as-Judge for faithfulness, with heuristic fallback | GPT-4o-mini extracts claims and labels them supported/unsupported. When the LLM is unavailable, a sparse-overlap heuristic fills in so the metric pipeline never stalls. |
| 7 | Uploaded-first hybrid routing | Always consult the user's corpus first; public results are only blended when they add signal. Off-topic public hits are dropped by a domain prior — this is what `sense_resolver` does. |
| 8 | Abstention guard | When post-filter overlap is below threshold and no document is pinned, the system returns "insufficient evidence" instead of generating. |
| 9 | Three independent k8s probes | `/health/live` only checks the process. `/health/ready` checks deps and returns 503 when degraded — load balancers should delist, not restart. `/health/full` returns the same body but always 200 for status dashboards. |
| 10 | OWASP-aligned default headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, strict referrer/permissions/COOP/CORP. HSTS is opt-in via `ENABLE_HSTS=true` so dev environments don't trip browsers into HTTPS-only mode. |

## Data flow

A `POST /assistant/answer` walks roughly:

1. `RequestIDMiddleware` mints / preserves `X-Request-ID`, starts a timer.
2. `SecurityHeadersMiddleware` reserves the OWASP headers on the eventual response.
3. `intent_resolver.resolve_query_intent` extracts canonical term + sub-queries via GPT-4o-mini (or the 57-term lexicon fallback).
4. `pdf_ingest.search_chunks` runs hybrid retrieval over uploaded corpus.
5. If scope = `public`: `public_search.public_live_search` fans out concurrently to 6 scholarly APIs, deduplicates by DOI / title, applies the sense filter.
6. `assistant_utils._build_strict_grounded_answer` composes a citation-strict prompt and calls the LLM.
7. `confidence.build_confidence` computes M / S / A on the cited chunks, applies the calibrated logistic.
8. Response goes back through middleware; access log line + `X-Request-ID` header.

## Where to read the code

| Concern | File |
|---|---|
| Routes / orchestration | [`backend/app.py`](../backend/app.py) |
| PDF ingest + chunking | [`backend/pdf_ingest.py`](../backend/pdf_ingest.py) |
| Public scholarly aggregator | [`backend/public_search.py`](../backend/public_search.py) |
| Confidence model | [`backend/confidence.py`](../backend/confidence.py) |
| Sense / off-topic filter | [`backend/sense_resolver.py`](../backend/sense_resolver.py) |
| LLM-as-judge | [`backend/services/judge.py`](../backend/services/judge.py) |
| NLI service | [`backend/services/nli.py`](../backend/services/nli.py) |
| Embedding providers | [`backend/services/embeddings.py`](../backend/services/embeddings.py) |
| Middleware | [`backend/middleware.py`](../backend/middleware.py) |
| Pydantic schemas | [`backend/schemas.py`](../backend/schemas.py) |

## Decision records

Architectural decisions that aren't obvious from the code live in
[`docs/adr/`](adr/) — one short file per decision in MADR format.
