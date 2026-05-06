FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY db/ db/
COPY scripts/ scripts/

RUN chown -R app:app /app
USER app

EXPOSE 8000

# Liveness only — `/health/live` is intentionally DB-free so a transient
# Postgres blip doesn't mark the API container unhealthy. For deps health,
# use `/health/ready` (returns 503 when degraded) from your orchestrator.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
