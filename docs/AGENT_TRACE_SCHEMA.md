# Agent Trace Schema

Sourcery traces agent runs as JSONL under `artifacts/agent_traces/`.

## Record shape

- `ts`
- `trace_id`
- `event`
- `payload`

## Notes

- The schema is append-only.
- One trace file is created per `trace_id`.
- Step order is represented by append order.
