# Security Policy

## Reporting a vulnerability

Email **sushildalavi@gmail.com** with:
- a clear description of the issue,
- the commit hash you tested against,
- a minimal reproduction or proof-of-concept,
- the impact you believe it has.

Please do not open a public GitHub issue for security findings.

You can expect:
- acknowledgement within **3 business days**,
- a triage decision (accepted / needs more info / not in scope) within **7 business days**,
- a fix or mitigation timeline once triaged.

## In scope

- Backend (`backend/`) — auth bypass, query/SQL injection, SSRF in scholarly fetchers, prompt injection that escapes the citation contract, RCE, secret exposure.
- Frontend (`frontend/`) — XSS, open redirect, dependency CVEs that are reachable in production builds.
- Container build (`Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`) — privileged escape, exposed credentials, vulnerable base images.
- CI (`.github/workflows/`) — token leakage, supply-chain risks.

## Out of scope

- Findings that require a compromised developer machine, leaked OpenAI key, or local DB access — those are pre-conditions, not vulnerabilities.
- Self-XSS that needs the victim to paste attacker-controlled JS into their own browser console.
- Rate-limit absence on local dev endpoints.

## Hardening already in place

- No secrets in repo (verified by `detect-private-key` pre-commit hook).
- pgvector queries use parameterized SQL via `psycopg2` `mogrify`/`execute_values`.
- CORS allowlist driven by `CORS_ORIGINS` env var; defaults to localhost only.
- `/health/embeddings` returns failure as data, never as 5xx — no info-leak via stack traces.
