# Support

## How to get help

**1. Read the docs first.** The [README](README.md) covers architecture, the [Quick Start](README.md#quick-start), and benchmark methodology. The [Evaluation README](Evaluation/README.md) covers the calibration pipeline. [`docs/architecture.md`](docs/architecture.md) explains the deeper design decisions.

**2. Check existing issues.** Search [open and closed issues](https://github.com/sushildalavi/citelens/issues?q=is%3Aissue) before opening a new one — your question may already have an answer.

**3. Open the right issue type.**
- **Bug** — something doesn't behave the way the README / code claims. Use the [bug template](.github/ISSUE_TEMPLATE/bug_report.yml).
- **Feature request** — propose a new capability or behavior change. Use the [feature template](.github/ISSUE_TEMPLATE/feature_request.yml).
- **Security finding** — do **not** open a public issue. Email per [SECURITY.md](SECURITY.md).

## What we respond to

| Channel | Response time |
|---|---|
| Bug reports with a reproducible curl/log | 3 business days |
| Bug reports without a repro | best-effort |
| Feature requests | reviewed monthly |
| Security findings | acknowledged within 3 business days |

## What's out of scope for free support

- Help integrating with proprietary scholarly APIs whose ToS prohibits aggregation.
- "Your model gave a wrong answer" without a citable evidence chunk we can replay.
- Customizing the calibration weights for a private corpus we have no access to.

## Versioning policy

Citelens roughly tracks SemVer:

- **Patch** (1.0.x) — bug fixes, dependency bumps, no API surface change.
- **Minor** (1.x.0) — new endpoints, new features, fully backwards-compatible.
- **Major** (x.0.0) — breaking API or DB schema change. Migration notes in [CHANGELOG.md](CHANGELOG.md).

The current version is in `app.version` (exposed at `GET /`) and at the top of [`CHANGELOG.md`](CHANGELOG.md).
