"""
User memory/history storage backed by PostgreSQL.

Tenant-scoped: every row carries a workspace_id sourced from
`X-Workspace-Id` (resolved by `WorkspaceMiddleware`). Cross-workspace
read/write is impossible because the SQL filters always include the
workspace.
"""

from fastapi import APIRouter, HTTPException, Request

from backend.middleware import current_workspace
from backend.services.db import execute, fetchall, fetchone

router = APIRouter()


def ensure_memory_table() -> None:
    """Create the `user_memory` table + index if missing.

    Called from FastAPI lifespan startup, NEVER at import time.
    """
    execute("""
        CREATE TABLE IF NOT EXISTS user_memory (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'guest',
            query TEXT,
            answer TEXT,
            notes TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT now()
        )
    """)
    execute("CREATE INDEX IF NOT EXISTS idx_user_memory_user ON user_memory(user_id)")


@router.post("/memory/log")
def log_interaction(payload: dict, request: Request):
    ws = current_workspace(request)
    user_id = payload.get("user_id") or "guest"
    query = (payload.get("query") or "").strip()
    answer = payload.get("answer")
    notes = payload.get("notes", "")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    execute(
        "INSERT INTO user_memory (user_id, query, answer, notes, workspace_id) VALUES (%s, %s, %s, %s, %s)",
        [user_id, query, answer, notes, ws],
    )
    count_row = fetchone(
        "SELECT count(*) AS c FROM user_memory WHERE user_id = %s AND workspace_id = %s",
        [user_id, ws],
    )
    return {"ok": True, "count": (count_row or {}).get("c", 0)}


@router.get("/memory/history")
def get_history(request: Request, user_id: str = "guest", limit: int = 20):
    ws = current_workspace(request)
    rows = fetchall(
        "SELECT query, answer, notes, created_at FROM user_memory "
        "WHERE user_id = %s AND workspace_id = %s ORDER BY id DESC LIMIT %s",
        [user_id, ws, max(1, min(limit, 200))],
    )
    for row in rows:
        if row.get("created_at"):
            row["created_at"] = row["created_at"].isoformat() + "Z"
    return {"user_id": user_id, "history": list(reversed(rows))}
