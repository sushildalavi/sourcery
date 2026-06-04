from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

TRACE_DIR = Path("artifacts/agent_traces")


def write_trace(trace_id: str, event: str, payload: dict[str, Any]) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "trace_id": trace_id,
        "event": event,
        "payload": payload,
    }
    path = TRACE_DIR / f"{trace_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
