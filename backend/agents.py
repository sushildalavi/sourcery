"""
Agentic workflow endpoints: register scheduled digests and list them.
Backed by PostgreSQL for persistence across restarts.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from backend.services.db import execute, fetchall, fetchone

router = APIRouter()


def _ensure_digests_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS digests (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'guest',
            query TEXT NOT NULL,
            frequency TEXT NOT NULL DEFAULT 'weekly',
            created_at TIMESTAMP DEFAULT now()
        )
    """)


_ensure_digests_table()


@router.post("/agents/digest")
def create_digest(payload: dict):
    user_id = payload.get("user_id") or "guest"
    query = payload.get("query")
    frequency = payload.get("frequency", "weekly")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    row = fetchone(
        "INSERT INTO digests (user_id, query, frequency) VALUES (%s, %s, %s) RETURNING id, user_id, query, frequency, created_at",
        [user_id, query, frequency],
    )
    row["created_at"] = row["created_at"].isoformat() + "Z" if row.get("created_at") else None
    return row


@router.get("/agents/digest")
def list_digests(user_id: str = None):
    if user_id:
        items = fetchall(
            "SELECT id, user_id, query, frequency, created_at FROM digests WHERE user_id = %s ORDER BY id DESC",
            [user_id],
        )
    else:
        items = fetchall("SELECT id, user_id, query, frequency, created_at FROM digests ORDER BY id DESC")
    for item in items:
        if item.get("created_at"):
            item["created_at"] = item["created_at"].isoformat() + "Z"
    return {"digests": items}
