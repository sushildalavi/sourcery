# Helm chart

Deploy sourcery to Kubernetes.

## Install

```bash
kubectl create secret generic sourcery-secrets \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=PGPASSWORD=$(openssl rand -hex 24) \
  --from-literal=DATABASE_URL=postgresql://scholarrag:<the-password>@sourcery-db:5432/scholarrag

helm install sourcery <chart-directory>
```

## Tuning

- Scale backend replicas for higher throughput.
- Disable the bundled database if you already have a managed Postgres.
- Increase storage if embeddings dominate your dataset.
- Enable metrics scraping if your cluster has Prometheus Operator.
- Turn on ingress only when you have your own ingress path.

## Probes

- Liveness restarts the backend when the process is unhealthy.
- Readiness removes the service from rotation when dependencies degrade.
