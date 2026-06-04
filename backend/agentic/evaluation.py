from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .graph import build_research_agent


@dataclass
class EvalCase:
    query: str
    expected_citation_keyword: str
    min_confidence: float


def run_agent_eval(cases: Iterable[EvalCase], *, scope: str = "public") -> dict:
    agent = build_research_agent()
    results = []
    passed = 0

    for case in cases:
        output = agent.invoke({"query": case.query, "scope": scope, "limit": 6, "use_llm": False})
        final = output["final_answer"]
        evidence_text = " ".join(
            f"{item.title} {item.snippet}" for item in output.get("reranked_evidence", [])
        ).lower()
        citation_hit = case.expected_citation_keyword.lower() in evidence_text
        confidence_ok = float(final.confidence) >= float(case.min_confidence)
        no_unsupported = len(final.unsupported_claims) == 0
        ok = citation_hit and confidence_ok and no_unsupported
        passed += int(ok)
        results.append(
            {
                "query": case.query,
                "ok": ok,
                "confidence": final.confidence,
                "citation_hit": citation_hit,
                "unsupported_claims": final.unsupported_claims,
                "needs_human_review": final.needs_human_review,
            }
        )

    total = len(results)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / max(total, 1),
        "results": results,
    }
