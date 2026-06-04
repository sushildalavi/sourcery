from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agentic.evaluation import EvalCase, run_agent_eval

if __name__ == "__main__":
    cases = [
        EvalCase(
            query="retrieval augmented generation",
            expected_citation_keyword="retrieval",
            min_confidence=0.8,
        ),
        EvalCase(
            query="machine learning",
            expected_citation_keyword="machine",
            min_confidence=0.8,
        ),
    ]
    result = run_agent_eval(cases, scope="public")
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / "agentic_rag_eval.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "summary": result}, indent=2, sort_keys=True))
