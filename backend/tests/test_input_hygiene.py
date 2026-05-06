"""Repo-wide guards for two recurring bugs:

1. Whitespace-only inputs slipping past `if not query` validation.
2. Raw `OpenAI()` instantiation that bypasses the centralized config helper.

These are AST-level grep tests so they catch the bug at any call site, in
any file, without each route needing its own dedicated test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


def _python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "tests" not in p.parts and "__pycache__" not in p.parts]


# ────────────────────── whitespace-only input guard ──────────────────────


# Match `payload.get("query")` or `payload.get("q")` not followed by `.strip()`
# within ~40 chars (catches `payload.get("query") or ""` too).
_UNGUARDED = re.compile(
    r'payload\.get\(\s*"(?:query|q)"\s*\)(?![^\n]{0,40}\.strip\(\))',
)


def test_no_route_reads_query_without_strip():
    """Every JSON route that accepts a `query` / `q` field must strip it
    before validating with `if not value`. Otherwise `"   "` slips through."""
    offenders: list[tuple[Path, int, str]] = []
    for f in _python_files(BACKEND):
        for lineno, line in enumerate(f.read_text().splitlines(), start=1):
            if _UNGUARDED.search(line):
                offenders.append((f, lineno, line.strip()))
    if offenders:
        report = "\n".join(f"  {p.relative_to(BACKEND.parent)}:{n}: {s}" for p, n, s in offenders)
        pytest.fail(
            "These call sites read `payload.get('query'|'q')` without "
            "an adjacent `.strip()` — whitespace-only input will slip "
            "past `if not value` validation:\n" + report
        )


# ────────────────────── centralized OpenAI client guard ──────────────────


# Allow `OpenAI(api_key=...)` (intentional) but reject bare `OpenAI()`.
_RAW_CLIENT = re.compile(r"\bOpenAI\(\s*\)")


def test_no_raw_openai_client_construction():
    """All `OpenAI()` calls must pass an `api_key=` argument so that the
    `.env` / Streamlit / AWS Secrets Manager fallback chain in
    `backend.utils.config.get_openai_api_key()` is honored uniformly.
    """
    offenders: list[tuple[Path, int, str]] = []
    for f in _python_files(BACKEND):
        for lineno, line in enumerate(f.read_text().splitlines(), start=1):
            if _RAW_CLIENT.search(line):
                offenders.append((f, lineno, line.strip()))
    if offenders:
        report = "\n".join(f"  {p.relative_to(BACKEND.parent)}:{n}: {s}" for p, n, s in offenders)
        pytest.fail("Raw `OpenAI()` construction found — pass `api_key=get_openai_api_key()` instead:\n" + report)


# ────────────────────── import-time DB side-effect guard ─────────────────


# Match anything that looks like a top-level (column 0) call to `_ensure_*`
# or `ensure_*_table` — these are DB DDL helpers that must only run from
# FastAPI lifespan, never at import.
_IMPORT_TIME_DB_CALL = re.compile(r"^(?:_ensure_\w+|ensure_\w+_table)\(\)\s*$")


def test_no_import_time_db_calls():
    """Modules under `backend/` must be import-safe so unit tests, IDE
    introspection, and `--help` invocations don't require a live Postgres.
    """
    offenders: list[tuple[Path, int, str]] = []
    for f in _python_files(BACKEND):
        for lineno, line in enumerate(f.read_text().splitlines(), start=1):
            if _IMPORT_TIME_DB_CALL.match(line):
                offenders.append((f, lineno, line.strip()))
    if offenders:
        report = "\n".join(f"  {p.relative_to(BACKEND.parent)}:{n}: {s}" for p, n, s in offenders)
        pytest.fail("Import-time DB DDL calls found (must be moved into FastAPI lifespan startup):\n" + report)
