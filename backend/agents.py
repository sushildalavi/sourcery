"""
Agentic workflow endpoints: register scheduled digests and list them.
Backed by PostgreSQL for persistence across restarts.

Tenant-scoped: every digest carries the request's workspace_id.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from backend.middleware import current_workspace
from backend.services.db import execute, fetchall, fetchone

router = APIRouter()


def ensure_digests_table() -> None:
    """Create the `digests` table if missing.

    Called from FastAPI lifespan startup, NEVER at import time — modules
    must be import-safe so unit tests and tooling can load them without a
    live Postgres.
    """
    execute("""
        CREATE TABLE IF NOT EXISTS digests (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'guest',
            query TEXT NOT NULL,
            frequency TEXT NOT NULL DEFAULT 'weekly',
            created_at TIMESTAMP DEFAULT now()
        )
    """)


@router.post("/agents/digest")
def create_digest(payload: dict, request: Request):
    ws = current_workspace(request)
    user_id = payload.get("user_id") or "guest"
    query = (payload.get("query") or "").strip()
    frequency = payload.get("frequency", "weekly")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    row = fetchone(
        "INSERT INTO digests (user_id, query, frequency, workspace_id) "
        "VALUES (%s, %s, %s, %s) "
        "RETURNING id, user_id, query, frequency, created_at",
        [user_id, query, frequency, ws],
    )
    row["created_at"] = row["created_at"].isoformat() + "Z" if row.get("created_at") else None
    return row


@router.get("/agents/digest")
def list_digests(request: Request, user_id: str = None):
    ws = current_workspace(request)
    if user_id:
        items = fetchall(
            "SELECT id, user_id, query, frequency, created_at FROM digests "
            "WHERE user_id = %s AND workspace_id = %s ORDER BY id DESC",
            [user_id, ws],
        )
    else:
        items = fetchall(
            "SELECT id, user_id, query, frequency, created_at FROM digests WHERE workspace_id = %s ORDER BY id DESC",
            [ws],
        )
    for item in items:
        if item.get("created_at"):
            item["created_at"] = item["created_at"].isoformat() + "Z"
    return {"digests": items}
