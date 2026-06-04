.PHONY: test lint lint-fix format format-check typecheck run run-frontend install install-dev clean help \
        compose-up compose-down compose-logs compose-rebuild stack-up stack-down \
        health frontend-build frontend-lint frontend-typecheck pre-commit-install ci-local \
        agent-eval

# ── Environment ───────────────────────────────────────────────────────────────
PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
UVICORN := $(VENV)/bin/uvicorn
PYTHONPATH_ENV := PYTHONPATH=.

# ── Setup ─────────────────────────────────────────────────────────────────────

install: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

# ── Test ──────────────────────────────────────────────────────────────────────

test:
	$(PYTEST) backend/tests/ -v --tb=short

test-fast:
	$(PYTEST) backend/tests/ -x --tb=short -q

test-coverage:
	$(PYTEST) backend/tests/ --cov=backend --cov-report=term-missing --cov-report=html

# Run the cross-tenant isolation tests against a throwaway pgvector container
# on port 5433 (so it doesn't fight whatever's already on 5432). Idempotent —
# safe to re-run; the container is recreated each time.
test-isolation:
	@echo "→ tearing down any previous sourcery-iso-db"
	@docker rm -f sourcery-iso-db 2>/dev/null || true
	@echo "→ booting pgvector on :5433"
	@docker run -d --name sourcery-iso-db \
		-e POSTGRES_USER=scholarrag -e POSTGRES_PASSWORD=scholarrag \
		-e POSTGRES_DB=scholarrag -p 5433:5432 \
		pgvector/pgvector:pg16 >/dev/null
	@sleep 6
	@docker exec -i sourcery-iso-db psql -U scholarrag -d scholarrag < db/init.sql >/dev/null
	@echo "→ running isolation tests"
	@DATABASE_URL=postgresql://scholarrag:scholarrag@127.0.0.1:5433/scholarrag \
		PGHOST=127.0.0.1 PGPORT=5433 PGUSER=scholarrag PGPASSWORD=scholarrag PGDATABASE=scholarrag \
		EMBEDDING_PROVIDER=stub OPENAI_API_KEY=test \
		$(PYTEST) backend/tests/test_workspace_isolation.py -v --tb=short ; \
		EXITCODE=$$? ; \
		echo "→ tearing down" ; \
		docker rm -f sourcery-iso-db >/dev/null 2>&1 || true ; \
		exit $$EXITCODE

# ── Lint ──────────────────────────────────────────────────────────────────────

lint:
	$(RUFF) check backend/ scripts/

lint-fix:
	$(RUFF) check backend/ scripts/ --fix

format:
	$(RUFF) format backend/ scripts/

format-check:
	$(RUFF) format --check backend/ scripts/

typecheck:
	$(VENV)/bin/pyright backend/confidence.py backend/eval_metrics.py backend/services/nli.py

# Run the EXACT same gates CI runs. If this passes locally, CI passes.
ci-local: lint format-check test
	@echo "✓ ci-local clean — pushing should succeed"

# ── Run ───────────────────────────────────────────────────────────────────────

run:
	$(UVICORN) backend.app:app --reload --host 127.0.0.1 --port 8000

run-prod:
	$(UVICORN) backend.app:app --host 0.0.0.0 --port 8000 --workers 4

run-frontend:
	cd frontend && npm run dev

# ── Database ──────────────────────────────────────────────────────────────────

db-up:
	docker compose up -d db adminer

db-down:
	docker compose down

# ── Compose / full stack ────────────────────────────────────────────────────

compose-up:
	docker compose up -d db adminer

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f --tail=200

compose-rebuild:
	docker compose build --no-cache backend frontend

stack-up:
	docker compose --profile backend --profile frontend up -d

stack-down:
	docker compose --profile backend --profile frontend down

health:
	@echo "→ db";       docker compose exec db pg_isready -U scholarrag || true
	@echo "→ backend";  curl -fsS http://127.0.0.1:8000/ || echo "  (backend not reachable)"
	@echo "→ embed";    curl -fsS http://127.0.0.1:8000/health/embeddings || true
	@echo "→ frontend"; curl -fsS -o /dev/null -w "  http %{http_code}\n" http://127.0.0.1:5173/ || true

# ── Frontend dev tasks ─────────────────────────────────────────────────────

frontend-lint:
	cd frontend && npx eslint .

frontend-typecheck:
	cd frontend && npx tsc --noEmit

frontend-build:
	cd frontend && npm run build

# ── Hooks ──────────────────────────────────────────────────────────────────

pre-commit-install:
	$(PIP) install pre-commit
	$(VENV)/bin/pre-commit install

# ── Reindex ───────────────────────────────────────────────────────────────────

reindex:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/reindex_embeddings.py --purge-all

eval:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/eval_retrieval.py \
		--k 10 \
		--output Evaluation/data/retrieval_run_$(shell date +%Y%m%d).json

agent-eval:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/eval_agentic_rag.py

# ── Calibration pipeline (full reproduction in 6 steps) ─────────────────────
#
# 1. make ingest-corpus      — pull the 15 PDFs into the documents table
# 2. make generate-queries   — emit 120 LLM-generated queries
# 3. make build-codebooks    — run assistant_answer, emit 3 coder xlsx files
# 4. Coders fill supported/unsupported via the dropdown, hand files back
# 5. make compute-iaa        — Cohen's kappa + majority-vote gold labels
# 6. make extract-features   — M/S/A per gold pair
# 7. make fit-calibration    — fit unified logistic + write DB row
#
# Then set CONFIDENCE_USE_FITTED_WEIGHTS=true to use the fitted row in prod.

ingest-corpus:
	$(PYTHONPATH_ENV) $(PYTHON) -m backend.scripts.ingest_corpus

generate-queries:
	$(PYTHONPATH_ENV) $(PYTHON) -m backend.scripts.generate_queries

build-codebooks:
	CODEBOOK_MAX_QUERIES=80 CODEBOOK_INCLUDE_PUBLIC=true PUBLIC_IEEE_LIMIT=0 \
		$(PYTHONPATH_ENV) $(PYTHON) -m backend.scripts.build_codebooks

compute-iaa:
	$(PYTHONPATH_ENV) $(PYTHON) -m backend.scripts.compute_iaa_majority

extract-features:
	PUBLIC_IEEE_LIMIT=0 \
		$(PYTHONPATH_ENV) $(PYTHON) -m backend.scripts.extract_msa_features

fit-calibration:
	$(PYTHONPATH_ENV) $(PYTHON) -m backend.scripts.fit_unified_calibration --write-db

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "ScholarRAG Makefile targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install       Install runtime dependencies"
	@echo "    make install-dev   Install runtime + dev dependencies"
	@echo ""
	@echo "  Test:"
	@echo "    make test          Run full test suite"
	@echo "    make test-fast     Run tests, stop on first failure"
	@echo "    make test-coverage Run tests with coverage report"
	@echo ""
	@echo "  Lint:"
	@echo "    make lint          Run ruff linter"
	@echo "    make lint-fix      Run ruff with auto-fix"
	@echo "    make typecheck     Run pyright on core modules"
	@echo ""
	@echo "  Run:"
	@echo "    make run           Start backend (dev, auto-reload)"
	@echo "    make run-prod      Start backend (production, 4 workers)"
	@echo "    make run-frontend  Start frontend dev server"
	@echo ""
	@echo "  Stack (docker compose):"
	@echo "    make compose-up    Bring up db + adminer"
	@echo "    make stack-up      Bring up everything (db, backend, frontend)"
	@echo "    make stack-down    Stop the full stack"
	@echo "    make health        Probe db / backend / embeddings / frontend"
	@echo ""
	@echo "  Calibration pipeline:"
	@echo "    make reindex           Rebuild all chunk embeddings"
	@echo "    make ingest-corpus     Ingest the 15-paper PDF corpus"
	@echo "    make generate-queries  Generate 120 GPT-4o-mini queries"
	@echo "    make build-codebooks   Run assistant_answer + emit 3 coder xlsx files"
	@echo "    make compute-iaa       Pairwise Cohen's kappa + majority-vote gold labels"
	@echo "    make extract-features  Compute M / S / A per gold pair"
	@echo "    make fit-calibration   Fit unified logistic + ablation + DB write"
	@echo "    make agent-eval        Run the agentic RAG smoke evaluation"
