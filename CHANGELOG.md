# Changelog

All notable changes to this repo are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
roughly tracks [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Mermaid versions of every benchmark chart in the README — calibration weights, IAA κ, gold distribution, per-mode ablation Brier/AUC, retrieval quality, provider share pie, latency p50/p95/p99 stages.
- `backend/middleware.py` — `RequestIDMiddleware` that mints / preserves `X-Request-ID` and emits one structured access log line per request. Header is exposed via CORS for browser fetches.
- `GET /health/full` — aggregated readiness check covering DB reachability + embedding provider, returns `status: degraded` (not 5xx) when anything is down.
- `EMBEDDING_PROVIDER=stub` — deterministic offline provider that hashes input into an L2-normalised pseudo-vector. Lets the test suite + CI run without Ollama or a network OpenAI call.
- Frontend `<ErrorBoundary>` wrapping the route tree with a retry / go-home fallback.
- Lazy-loaded `/analytics` route via `React.lazy` + `<Suspense>` with a CSS-only skeleton fallback. Initial JS bundle dropped **453 KB → 78 KB**.
- Vendor chunk splitting (`react`, `framer-motion`, `lucide-react`) — eliminated the >500 KB chunk warning.
- `SECURITY.md` for vulnerability disclosure policy.
- `frontend/Dockerfile` (multi-stage `node:20-alpine` → `nginx:1.27-alpine`) + `frontend` profile in `docker-compose.yml`.
- Compose: db & backend healthchecks, `depends_on: condition: service_healthy`.
- `Makefile`: `compose-up`, `stack-up`, `stack-down`, `health`, `frontend-build`, `frontend-lint`, `frontend-typecheck`, `pre-commit-install`.
- `.pre-commit-config.yaml` (ruff, ruff-format, trailing-whitespace, large-file guard, secret detector, frontend typecheck).
- `.editorconfig` and frontend `.prettierrc.json` / `.prettierignore`.
- `backend/.env.example` documenting all 28 env vars referenced by the backend.
- CI: pytest coverage emitted as `coverage.xml` and uploaded as a workflow artifact.

### Changed
- FastAPI app now uses the `lifespan` async context manager — eliminates the `on_event` deprecation warning at startup.
- `/health/embeddings` returns HTTP 200 with `{ok: false, error: "..."}` when the provider is unreachable, so uptime monitors get a clean degraded signal instead of noisy 5xx.
- `GET /` now exposes `service`, `version`, `uptime_seconds`.
- `pyproject.toml`: explicit `asyncio_default_fixture_loop_scope = "function"` (silences pytest-asyncio future-default warning).
- README: added "Quality Gates" section reflecting actual current state (160/160 tests, 32 routes).
- `pypdf` replaces deprecated `PyPDF2` (drop-in `PdfReader`).

### Fixed
- Frontend fast-refresh warning by splitting `useDefaultPaletteActions` into its own file.
- README pointed at the wrong GitHub repo slug (`Final-Project-ScholarRAG` → `citelens`).

## [0.1.0] — initial commit history

- FastAPI backend with chat, multi-doc upload, public scholarly aggregation across 7 APIs.
- React + Vite frontend with chat shell, command palette, analytics dashboard.
- PostgreSQL + pgvector schema (papers, documents, chunks, chunk_embeddings, embedding_cache, evaluation_judge_runs).
- M / S / A confidence calibration pipeline + 530-pair gold set + 5-fold CV.
- Retrieval evaluation harness (Recall@K, MRR, nDCG@K) on 120 queries.
- Docker Compose with Postgres + Adminer + Ollama profiles.
- 160-test pytest suite.
