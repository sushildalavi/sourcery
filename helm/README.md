# Helm chart

Deploy citelens to Kubernetes.

```bash
# 1. create the secret with your real values (NEVER commit it)
kubectl create secret generic citelens-secrets \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=PGPASSWORD=$(openssl rand -hex 24) \
  --from-literal=DATABASE_URL=postgresql://scholarrag:<the-password>@citelens-db:5432/scholarrag

# 2. install
helm install citelens ./helm/citelens

# 3. apply schema (one-shot, after pods are healthy)
kubectl exec -it citelens-db-0 -- psql -U scholarrag -d scholarrag \
  -f - < db/init.sql

# 4. for the multi-tenant migration
kubectl exec -it citelens-db-0 -- psql -U scholarrag -d scholarrag \
  -f - < db/migrations/002_workspace_isolation.sql
```

## Values worth tuning

| Key | Default | Why you'd change it |
|---|---|---|
| `backend.replicaCount` | 2 | Bump for higher RPS. The app is stateless. |
| `postgres.enabled` | true | Set false if you use RDS / Cloud SQL — point `DATABASE_URL` at it instead. |
| `postgres.persistence.size` | 20Gi | Embeddings dominate; ~10 KB per chunk × 1.5KB JSON. |
| `serviceMonitor.enabled` | false | Set true if Prometheus Operator runs in your cluster. Scrapes `/metrics/prom`. |
| `ingress.enabled` | false | Most users front this with their own ingress. |
| `backend.env.ENABLE_HSTS` | "true" | Only safe behind TLS. |

## Probes

The chart wires k8s-style probes against the backend:

- `livenessProbe` → `GET /health/live` — restart on failure (process dead).
- `readinessProbe` → `GET /health/ready` — remove from service when deps degraded; does **not** restart.
