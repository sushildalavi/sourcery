# Contributing

Thanks for considering a contribution.

## Quick local setup

```bash
git clone https://github.com/sushildalavi/citelens.git
cd citelens
make install-dev
make compose-up                  # Postgres + Adminer
cp backend/.env.example backend/.env
# fill in OPENAI_API_KEY, etc.
make run                         # backend on :8000
make run-frontend                # SPA on :5173
```

## Before you push

```bash
make lint                        # ruff (errors fail CI)
make test                        # pytest, requires DB up
make frontend-typecheck
make frontend-lint
make frontend-build
```

If you have `pre-commit` installed: `make pre-commit-install` once, and the
hooks run on every commit.

## Commit conventions

- Short, lowercase, imperative subject (`fix retry backoff`, `add x-request-id`).
- One logical change per commit. Bundle small refactors into the parent commit only when they're truly inseparable.
- Don't co-author or sign commits with bot identities — keep author = you.

## PRs

- One concern per PR. If the diff touches retrieval and the frontend, that's two PRs.
- For changes that affect retrieval quality, include `Recall@5 / MRR / nDCG@10` numbers from `make eval` in the PR body.
- For changes that affect calibration, re-run `make fit-calibration` and attach the new `cv_metrics.json` summary.

## Code style

- **Python**: 4-space indent, ruff-formatted, type hints on public functions, no bare `except:`. Module-level docstring on each file.
- **TypeScript**: 2-space indent, `tsc --noEmit` clean, no `any` unless adapting an external lib that has none.
- **SQL**: keep migrations idempotent (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`). New tables get matching indexes in the same migration.

## Reporting bugs / requesting features

Use the issue templates. For bugs include:
- exact `git rev-parse HEAD`,
- the request that triggered it,
- backend log lines around the failure,
- whether it reproduces against a fresh `db/init.sql`.

## Security

Don't open a public issue for security findings — see [SECURITY.md](SECURITY.md).
