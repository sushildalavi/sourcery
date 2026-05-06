import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.middleware import current_workspace
from backend.pdf_ingest import (
    _chunk_text,
    _embed_and_store_chunks,
    _extract_pdf_text,
    _hash_bytes,
    _is_supported_upload,
    _sanitize_text,
)
from backend.pdf_ingest import search_chunks as search_uploaded_chunks
from backend.public_search import public_live_search
from backend.services.db import execute, fetchall, fetchone
from backend.utils.config import get_openai_api_key

router = APIRouter(prefix="/assistant/chat", tags=["chat"])

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
CHAT_UPLOAD_DIR = STORAGE_DIR / "chat_uploads"


def _create_session(workspace_id: str = "default") -> int:
    _ensure_chat_tables()
    row = fetchone(
        "INSERT INTO chat_sessions (workspace_id) VALUES (%s) RETURNING id",
        [workspace_id],
    )
    return row["id"]


def _ensure_chat_tables():
    # Lightweight safeguard in case init.sql was not reapplied
    execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id SERIAL PRIMARY KEY,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            created_at TIMESTAMP DEFAULT now(),
            updated_at TIMESTAMP DEFAULT now()
        );
        """
    )
    # In case the table existed before workspace_id was added.
    execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default'")
    execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id SERIAL PRIMARY KEY,
            session_id INT REFERENCES chat_sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT,
            citations JSONB,
            created_at TIMESTAMP DEFAULT now()
        );
        """
    )
    execute(
        """
        CREATE TABLE IF NOT EXISTS chat_uploads (
            id SERIAL PRIMARY KEY,
            session_id INT REFERENCES chat_sessions(id) ON DELETE CASCADE,
            doc_id INT REFERENCES documents(id) ON DELETE SET NULL,
            file_path TEXT,
            mime_type TEXT,
            created_at TIMESTAMP DEFAULT now()
        );
        """
    )


def _store_message(session_id: int, role: str, content: str, citations: Optional[Any] = None):
    payload = citations
    if citations is not None:
        payload = json.dumps(citations)
    execute(
        """
        INSERT INTO chat_messages (session_id, role, content, citations)
        VALUES (%s, %s, %s, %s)
        """,
        [session_id, role, content, payload],
    )


def _get_history(session_id: int, workspace_id: str = "default"):
    """Fetch a session's history, joining through chat_sessions to enforce
    that the requesting workspace owns the session — otherwise an attacker
    who guessed a session_id could read another tenant's chat."""
    rows = fetchall(
        """
        SELECT m.id, m.role, m.content, m.citations, m.created_at
        FROM chat_messages m
        JOIN chat_sessions s ON s.id = m.session_id
        WHERE m.session_id = %s AND s.workspace_id = %s
        ORDER BY m.id ASC
        """,
        [session_id, workspace_id],
    )
    return rows


def _ingest_upload(session_id: int, upload: UploadFile, workspace_id: str = "default") -> Optional[int]:
    data = upload.file.read()
    if not data:
        return None
    mime = upload.content_type or "application/octet-stream"
    if not _is_supported_upload(upload.filename or "", mime):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF, TXT, or Markdown files.")
    fname = f"{int(time.time())}_{upload.filename}"
    CHAT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fpath = CHAT_UPLOAD_DIR / fname
    fpath.write_bytes(data)

    sha = _hash_bytes(data)

    doc_row = fetchone(
        """
        INSERT INTO documents (title, source_path, mime_type, bytes, hash_sha256, status, workspace_id)
        VALUES (%s, %s, %s, %s, %s, 'processing', %s)
        RETURNING id
        """,
        [upload.filename, str(fpath), mime, len(data), sha, workspace_id],
    )
    doc_id = doc_row["id"]

    pages = []
    if mime == "application/pdf" or upload.filename.lower().endswith(".pdf"):
        pages = _extract_pdf_text(data)
    else:
        try:
            pages = [(1, _sanitize_text(data.decode("utf-8", errors="ignore")))]
        except Exception:
            pages = []

    chunk_tuples: List[Any] = []
    for page_no, text in pages:
        for idx, chunk in enumerate(_chunk_text(text)):
            chunk_tuples.append((page_no, idx, chunk))

    if chunk_tuples:
        inserted = _embed_and_store_chunks(doc_id, chunk_tuples, workspace_id=workspace_id)
        if inserted > 0:
            execute("UPDATE documents SET pages=%s, status='ready' WHERE id=%s", [len(pages), doc_id])
        else:
            execute("UPDATE documents SET status='error' WHERE id=%s", [doc_id])
    else:
        execute("UPDATE documents SET status='error' WHERE id=%s", [doc_id])

    execute(
        "INSERT INTO chat_uploads (session_id, doc_id, file_path, mime_type) VALUES (%s, %s, %s, %s)",
        [session_id, doc_id, str(fpath), mime],
    )
    return doc_id


@router.get("/{session_id}")
def get_chat(session_id: int, request: Request):
    _ensure_chat_tables()
    ws = current_workspace(request)
    history = _get_history(session_id, workspace_id=ws)
    return {"session_id": session_id, "messages": history}


@router.post("/{session_id}/upload")
async def upload_to_chat(session_id: int, request: Request, file: UploadFile = File(...)):
    _ensure_chat_tables()
    ws = current_workspace(request)
    doc_id = _ingest_upload(session_id, file, workspace_id=ws)
    return {"session_id": session_id, "doc_id": doc_id}


@router.post("")
def chat(request: Request, payload: dict = None):
    _ensure_chat_tables()
    ws = current_workspace(request)
    if payload is None:
        payload = {}
    if isinstance(payload, str):
        payload = {"message": payload}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    # Support creating a chat session without sending a message (for uploads)
    if payload.get("session_only"):
        session_id = _create_session(workspace_id=ws)
        return {"session_id": session_id, "messages": []}

    session_id = payload.get("session_id") or _create_session(workspace_id=ws)
    message = payload.get("message") or ""
    scope = payload.get("scope") or "public"
    doc_id = payload.get("doc_id")
    k = int(payload.get("k") or 8)

    if not message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    _store_message(session_id, "user", message, None)

    # Retrieve context
    context_blocks = []
    citations: List[Dict[str, Any]] = []
    if scope == "uploaded":
        results = search_uploaded_chunks(
            payload={"q": message, "k": k, "doc_id": doc_id},
            workspace_id=ws,
        )["results"]
        for r in results:
            citations.append(
                {
                    "title": f"Document {r.get('document_id')}",
                    "source": "uploaded",
                    "doc_id": r.get("document_id"),
                    "chunk_id": r.get("id"),
                    "page": r.get("page_no"),
                }
            )
            context_blocks.append(
                f"[doc {r.get('document_id')} chunk {r.get('id')} page {r.get('page_no', '?')}] {r.get('text', '')}"
            )
    else:
        docs = public_live_search(message, k=min(k, 8))
        for d in docs:
            citations.append(
                {
                    "title": d.get("title"),
                    "year": d.get("year"),
                    "source": d.get("source") or d.get("venue"),
                    "url": d.get("url") or d.get("doi"),
                }
            )
            context_blocks.append(f"[{d.get('title', '')}] {d.get('abstract') or d.get('summary') or ''}")

    context = "\n\n".join(context_blocks) if context_blocks else "No context found."
    prompt = (
        "You are a research assistant. Use the provided context to answer. "
        "Respond with a detailed answer and cite sources inline like [1], [2]. "
        "If context is weak, say so. Do not invent citations.\n\n"
        f"Question:\n{message}\n\nContext:\n{context}\n"
    )

    # Centralized client construction so this route honors the same
    # `.env` / Streamlit / AWS Secrets Manager fallback chain as
    # `/assistant/answer`. Imported lazily to keep this module
    # import-safe when the openai package is unavailable.
    from openai import OpenAI

    client = OpenAI(api_key=get_openai_api_key())
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content
    _store_message(session_id, "assistant", answer, citations)
    history = _get_history(session_id, workspace_id=ws)
    return {"session_id": session_id, "messages": history}
