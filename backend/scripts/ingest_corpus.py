"""Ingest the 15-paper corpus into the documents table.

Deletes any stale documents whose title matches the 15-entry list in
`_LEGACY_TITLES` (unrelated papers that might be left over from previous
corpora), then ingests the 15 canonical corpus PDFs from evaluation/papers/
(01_ResNet.pdf … 15_PageRank.pdf).

Safe to rerun — existing ingested docs are skipped (dedup by SHA).

Usage:
    python -m backend.scripts.ingest_corpus

Requires:
    - PostgreSQL reachable via backend.services.db.
    - OPENAI_API_KEY (for embeddings via backend.pdf_ingest).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from backend.pdf_ingest import (
    STORAGE_DIR,
    _hash_bytes,
    _infer_doc_type,
    _ingest_document,
    delete_document,
)
from backend.services.db import execute, fetchall, fetchone

# Stale filenames to purge from the DB if they linger from other corpora.
_LEGACY_TITLES = {
    "01_DPR.pdf",
    "02_ColBERT.pdf",
    "03_RAG.pdf",
    "04_BEIR.pdf",
    "05_SQuAD.pdf",
    "06_NaturalQuestions.pdf",
    "07_DrQA.pdf",
    "08_PEGASUS.pdf",
    "09_BART.pdf",
    "10_FActScore.pdf",
    "11_BERT.pdf",
    "12_Attention.pdf",
    "13_ChainOfThought.pdf",
    "14_InstructGPT.pdf",
    "15_LLMasJudge.pdf",
}

# Canonical corpus. Files must exist in evaluation/papers/.
CORPUS_FILES = [
    "01_ResNet.pdf",
    "02_GAN.pdf",
    "03_Word2Vec.pdf",
    "04_LLaMA2.pdf",
    "05_Chinchilla.pdf",
    "06_ConstitutionalAI.pdf",
    "07_AlphaGo.pdf",
    "08_CLIP.pdf",
    "09_AlphaFold.pdf",
    "10_StableDiffusion.pdf",
    "11_LSTM.pdf",
    "12_DQN.pdf",
    "13_VAE.pdf",
    "14_SwinTransformer.pdf",
    "15_PageRank.pdf",
]

PAPERS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "papers"


def purge_legacy() -> int:
    """Delete any stale documents still sitting in the DB."""
    purged = 0
    rows = fetchall("SELECT id, title FROM documents ORDER BY id")
    for row in rows:
        title = (row.get("title") or "").strip()
        if title in _LEGACY_TITLES:
            try:
                delete_document(int(row["id"]))
                print(f"  purged legacy doc {row['id']}: {title}")
                purged += 1
            except Exception as exc:
                print(f"  WARN could not purge {row['id']} / {title}: {exc}")
    return purged


def ingest_one(filename: str) -> tuple[int | None, str]:
    """Ingest a single PDF via the same code path that /upload uses."""
    src = PAPERS_DIR / filename
    if not src.exists():
        return None, f"missing-file ({src})"
    data = src.read_bytes()
    if not data:
        return None, "empty-file"
    mime = "application/pdf"
    sha = _hash_bytes(data)

    existing = fetchone(
        """
        SELECT id, status
        FROM documents
        WHERE hash_sha256=%s AND status IN ('processing','ready')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [sha],
    )
    if existing:
        return int(existing["id"]), f"already-ingested (status={existing.get('status')})"

    # Copy into the canonical storage dir the running backend reads from.
    storage_path = STORAGE_DIR / f"{int(time.time())}_{filename}"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_bytes(data)

    inferred_doc_type = _infer_doc_type(filename)
    doc_row = fetchone(
        """
        INSERT INTO documents (title, source_path, mime_type, bytes, hash_sha256, status, doc_type)
        VALUES (%s, %s, %s, %s, %s, 'processing', %s)
        RETURNING id
        """,
        [filename, str(storage_path), mime, len(data), sha, inferred_doc_type],
    )
    doc_id = int(doc_row["id"])
    try:
        _ingest_document(doc_id, data, mime, filename, filename)
    except Exception as exc:
        execute("UPDATE documents SET status='error' WHERE id=%s", [doc_id])
        return doc_id, f"error during ingest: {exc}"
    return doc_id, "ready"


def main() -> int:
    print("=== Purging stale legacy documents ===")
    purged = purge_legacy()
    print(f"  purged {purged} legacy document(s).\n")

    print("=== Ingesting corpus ===")
    errors: list[str] = []
    for filename in CORPUS_FILES:
        doc_id, status = ingest_one(filename)
        marker = "OK " if status in {"ready"} or status.startswith("already") else "ERR"
        print(f"  [{marker}] {filename:26s} -> doc_id={doc_id} status={status}")
        if marker == "ERR":
            errors.append(f"{filename}: {status}")

    print()
    if errors:
        print(f"=== {len(errors)} error(s) ===")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("=== Final documents state ===")
    rows = fetchall("SELECT id, title, status FROM documents ORDER BY id")
    for r in rows:
        print(f"  {r['id']:>3}  [{r.get('status'):<10}]  {r.get('title')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
