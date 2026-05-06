<!--
Keep PRs single-concern. If this PR touches retrieval AND the frontend, split it.
-->

## What

<!-- one or two lines -->

## Why

<!-- the user/dev/ops problem this solves; link an issue if there is one -->

## How to verify

```bash
# the commands a reviewer should run to confirm the change
```

## Checklist

- [ ] `make lint` clean
- [ ] `make test` passes locally (DB up)
- [ ] If retrieval changed: ran `make eval` and pasted Recall@5 / MRR / nDCG@10 below
- [ ] If calibration changed: re-ran `make fit-calibration`, attached cv summary
- [ ] Frontend: `make frontend-typecheck && make frontend-lint && make frontend-build`
- [ ] No new secrets / API keys in the diff
