# Curl recipes

All commands assume `http://localhost:8000`. Replace with your deployed host.
Every response carries an `X-Request-ID` header you can grep for in the
backend access log.

## Health probes (k8s-style)

```bash
# liveness — process is up. Use as k8s livenessProbe.
curl -fsS http://localhost:8000/health/live | jq

# readiness — db + embeddings ready. Returns 503 when degraded.
# Use as k8s readinessProbe.
curl -i http://localhost:8000/health/ready | head -1

# verbose readiness body (always 200 — for status dashboards)
curl -fsS http://localhost:8000/health/full | jq
```

## Operational metrics

```bash
curl -fsS http://localhost:8000/metrics | jq
```

## Calibration weights (active MSA logistic)

```bash
curl -fsS http://localhost:8000/confidence/calibration | jq
```

## Ask a question over uploaded corpus

```bash
curl -X POST http://localhost:8000/assistant/answer \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: trace-readme-demo-001' \
  -d '{
        "query": "What does the chinchilla paper say about scaling laws?",
        "scope": "uploaded",
        "k": 8
      }' | jq
```

## Ask a question across the public corpus (6-API fan-out)

```bash
curl -X POST http://localhost:8000/assistant/answer \
  -H 'Content-Type: application/json' \
  -d '{
        "query": "Recent advances in retrieval-augmented generation",
        "scope": "public",
        "k": 10
      }' | jq
```

## Upload a PDF

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F 'file=@/path/to/paper.pdf' | jq
```

## List uploaded documents

```bash
curl -fsS http://localhost:8000/documents | jq
```

## Run retrieval evaluation against a JSON eval set

```bash
curl -X POST http://localhost:8000/eval/run \
  -H 'Content-Type: application/json' \
  -d '{
        "eval_set": "/abs/path/to/queries_120.json",
        "k": 10
      }' | jq
```

## Trace a request end-to-end

Mint a request id, send it, then grep the backend log:

```bash
RID=$(uuidgen)
curl -s -H "X-Request-ID: $RID" http://localhost:8000/metrics > /dev/null
docker compose logs backend | grep "$RID"
```
