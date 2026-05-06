"""End-to-end isolation tests for tenant data.

These tests boot the FastAPI app against a live Postgres, write data into
two different workspaces via the same routes, and assert that one tenant
cannot read the other's documents, chunks, memory, or digests.

These ARE integration tests — they need a real DB. They run automatically:
  - in CI (workflow sets `DATABASE_URL`).
  - locally via `make test-isolation` (brings up Postgres on port 5433).

When DATABASE_URL isn't set they skip with a clear message rather than
failing. To run by hand:

    docker run -d --name citelens-iso-db \\
      -e POSTGRES_USER=scholarrag -e POSTGRES_PASSWORD=scholarrag \\
      -e POSTGRES_DB=scholarrag -p 5433:5432 pgvector/pgvector:pg16
    sleep 5
    docker exec -i citelens-iso-db psql -U scholarrag -d scholarrag < db/init.sql
    DATABASE_URL=postgresql://scholarrag:scholarrag@127.0.0.1:5433/scholarrag \\
      EMBEDDING_PROVIDER=stub OPENAI_API_KEY=test \\
      pytest backend/tests/test_workspace_isolation.py -v
"""

from __future__ import annotations

import os
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

DB_URL = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DB_URL,
    reason=(
        "Integration test — needs a live Postgres. "
        "Run `make test-isolation` (auto-spins a container) "
        "or set DATABASE_URL to a pgvector-enabled DB."
    ),
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Force-pin the stub provider AND reload the embeddings module so the
    # module-level `EMBEDDING_PROVIDER` constant picks it up. Other tests
    # in the suite (`test_stub_embedding.py`) intentionally delete the env
    # var on teardown and reload to default "ollama"; without this re-pin
    # we'd inherit that and try to hit the real Ollama at 11434.
    import importlib

    os.environ["EMBEDDING_PROVIDER"] = "stub"
    os.environ.setdefault("OPENAI_API_KEY", "test")
    import backend.services.embeddings as emb

    importlib.reload(emb)
    from backend import app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


def _ws(name: str) -> dict:
    return {"X-Workspace-Id": name}


def test_document_upload_isolated_per_workspace(client: TestClient):
    """A doc uploaded to workspace A must not appear in workspace B's list."""
    a_payload = ("test-isolation-A.txt", BytesIO(b"alpha tenant content"), "text/plain")
    b_payload = ("test-isolation-B.txt", BytesIO(b"beta tenant content"), "text/plain")

    ra = client.post("/documents/upload", files={"file": a_payload}, headers=_ws("alpha-tenant"))
    rb = client.post("/documents/upload", files={"file": b_payload}, headers=_ws("beta-tenant"))
    assert ra.status_code == 200, ra.text
    assert rb.status_code == 200, rb.text
    a_id = ra.json()["document_id"]
    b_id = rb.json()["document_id"]
    assert a_id != b_id

    list_a = client.get("/documents", headers=_ws("alpha-tenant")).json()["documents"]
    list_b = client.get("/documents", headers=_ws("beta-tenant")).json()["documents"]

    a_ids = {d["id"] for d in list_a}
    b_ids = {d["id"] for d in list_b}
    assert a_id in a_ids, "alpha doesn't see its own doc"
    assert b_id in b_ids, "beta doesn't see its own doc"
    assert b_id not in a_ids, "alpha can see beta's doc — ISOLATION BROKEN"
    assert a_id not in b_ids, "beta can see alpha's doc — ISOLATION BROKEN"


def test_memory_log_isolated_per_workspace(client: TestClient):
    """`/memory/log` writes per workspace; `/memory/history` reads per workspace."""
    client.post(
        "/memory/log",
        json={"user_id": "ada", "query": "alpha-only-question", "answer": "x"},
        headers=_ws("alpha-tenant"),
    )
    client.post(
        "/memory/log",
        json={"user_id": "ada", "query": "beta-only-question", "answer": "y"},
        headers=_ws("beta-tenant"),
    )

    a = client.get("/memory/history?user_id=ada", headers=_ws("alpha-tenant")).json()["history"]
    b = client.get("/memory/history?user_id=ada", headers=_ws("beta-tenant")).json()["history"]

    a_qs = {row["query"] for row in a}
    b_qs = {row["query"] for row in b}
    assert "alpha-only-question" in a_qs
    assert "beta-only-question" in b_qs
    assert "beta-only-question" not in a_qs
    assert "alpha-only-question" not in b_qs


def test_digest_isolated_per_workspace(client: TestClient):
    client.post(
        "/agents/digest",
        json={"user_id": "ada", "query": "alpha-digest-keyword", "frequency": "weekly"},
        headers=_ws("alpha-tenant"),
    )
    client.post(
        "/agents/digest",
        json={"user_id": "ada", "query": "beta-digest-keyword", "frequency": "weekly"},
        headers=_ws("beta-tenant"),
    )

    a = client.get("/agents/digest?user_id=ada", headers=_ws("alpha-tenant")).json()["digests"]
    b = client.get("/agents/digest?user_id=ada", headers=_ws("beta-tenant")).json()["digests"]
    assert any(d["query"] == "alpha-digest-keyword" for d in a)
    assert any(d["query"] == "beta-digest-keyword" for d in b)
    assert not any(d["query"] == "beta-digest-keyword" for d in a)
    assert not any(d["query"] == "alpha-digest-keyword" for d in b)


def test_doc_delete_blocked_across_workspaces(client: TestClient):
    """Workspace alpha cannot delete a document that lives in workspace beta,
    even when alpha calls DELETE with beta's doc id."""
    payload = ("test-cross-delete.txt", BytesIO(b"beta-only doc body for cross-delete"), "text/plain")
    rb = client.post("/documents/upload", files={"file": payload}, headers=_ws("beta-cross-delete"))
    assert rb.status_code == 200
    beta_id = rb.json()["document_id"]

    # alpha attempts to delete beta's doc — must be silently a no-op
    res = client.delete(f"/documents/{beta_id}", headers=_ws("alpha-cross-delete"))
    assert res.status_code == 200
    assert res.json()["deleted_ids"] == [], "alpha was able to delete beta's doc"

    # beta still sees its doc
    listing = client.get("/documents", headers=_ws("beta-cross-delete")).json()["documents"]
    assert beta_id in {d["id"] for d in listing}, "beta lost its own doc to a cross-tenant delete"


def test_doc_type_update_blocked_across_workspaces(client: TestClient):
    """PUT /documents/{id}/type must not update a doc owned by another tenant."""
    payload = ("test-cross-update.txt", BytesIO(b"beta type-update target"), "text/plain")
    rb = client.post("/documents/upload", files={"file": payload}, headers=_ws("beta-cross-update"))
    beta_id = rb.json()["document_id"]

    res = client.put(
        f"/documents/{beta_id}/type",
        json={"doc_type": "resume"},
        headers=_ws("alpha-cross-update"),
    )
    # Endpoint returns 200 either way; isolation is enforced via WHERE clause.
    assert res.status_code == 200

    listing = client.get("/documents", headers=_ws("beta-cross-update")).json()["documents"]
    beta_row = next((d for d in listing if d["id"] == beta_id), None)
    assert beta_row is not None
    assert beta_row.get("doc_type") != "resume", "alpha mutated beta's doc_type"


def test_calibration_isolated_per_workspace(client: TestClient):
    """`/confidence/calibration` returns the tenant's own row when present;
    otherwise falls back to the global `default` workspace's row."""
    # Default tenant should always resolve (defaults shipped with the app).
    default_resp = client.get("/confidence/calibration").json()
    assert default_resp["weights"]["w1"] is not None

    # Unknown tenant falls back to default — no row planted here, no leak.
    novel = client.get("/confidence/calibration", headers=_ws("never-calibrated-tenant")).json()
    assert novel["weights"]["w1"] is not None
