#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pdf_ingest import search_chunks


def _dcg(relevances: list[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        score += rel / math.log2(i + 1)
    return score


def _ndcg(results: list[int], ideal_count: int, k: int) -> float:
    ideal = [1] * min(ideal_count, k)
    denom = _dcg(ideal, k)
    if denom == 0:
        return 0.0
    return _dcg(results, k) / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate uploaded-doc retrieval against a labeled eval set.")
    parser.add_argument("--eval-set", required=True, help="Path to labeled eval JSON")
    parser.add_argument("--k", type=int, default=10, help="Top-k retrieval cutoff")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    path = Path(args.eval_set)
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        if isinstance(raw.get("cases"), list):
            cases = raw["cases"]
        elif isinstance(raw.get("queries"), list):
            cases = raw["queries"]
        else:
            raise SystemExit("Eval set JSON object must contain `cases` or `queries` list.")
    elif isinstance(raw, list):
        cases = raw
    else:
        raise SystemExit("Eval set must be a JSON list or object with `cases`/`queries`.")

    recall5 = []
    recall10 = []
    mrrs = []
    ndcgs = []

    for case in cases:
        query = case["query"]
        doc_id = case.get("doc_id")
        doc_ids = case.get("doc_ids")
        relevant_chunk_ids = {int(x) for x in case.get("relevant_chunk_ids", [])}
        relevant_doc_ids = {int(x) for x in case.get("relevant_doc_ids", [])}
        payload = {"q": query, "k": max(args.k, 10)}
        if doc_id is not None:
            payload["doc_id"] = int(doc_id)
        if doc_ids:
            payload["doc_ids"] = [int(x) for x in doc_ids]

        results = search_chunks(payload=payload)["results"]

        hit5 = False
        hit10 = False
        reciprocal_rank = 0.0
        rel_vector = []
        # Known-item semantics: each relevant chunk / doc counts at most once
        # so nDCG stays bounded in [0, 1].
        seen_chunks: set[int] = set()
        seen_docs: set[int] = set()

        for rank, row in enumerate(results[:10], start=1):
            row_chunk_id = int(row["id"])
            row_doc_id = int(row["document_id"])
            chunk_match = row_chunk_id in relevant_chunk_ids and row_chunk_id not in seen_chunks
            doc_match = (not chunk_match) and row_doc_id in relevant_doc_ids and row_doc_id not in seen_docs
            relevant = chunk_match or doc_match
            if chunk_match:
                seen_chunks.add(row_chunk_id)
            if doc_match:
                seen_docs.add(row_doc_id)
            rel_vector.append(1 if relevant else 0)
            if relevant and reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
            if rank <= 5 and relevant:
                hit5 = True
            if rank <= 10 and relevant:
                hit10 = True

        recall5.append(1.0 if hit5 else 0.0)
        recall10.append(1.0 if hit10 else 0.0)
        mrrs.append(reciprocal_rank)
        ndcgs.append(_ndcg(rel_vector, max(len(relevant_chunk_ids), len(relevant_doc_ids), 1), 10))

    report = {
        "case_count": len(cases),
        "embedding_provider": None,
        "embedding_model": None,
        "recall@5": round(sum(recall5) / len(recall5), 4) if recall5 else 0.0,
        "recall@10": round(sum(recall10) / len(recall10), 4) if recall10 else 0.0,
        "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
        "ndcg@10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "input_format": {
            "query": "string",
            "doc_id": "optional int",
            "doc_ids": "optional list[int]",
            "relevant_chunk_ids": "optional list[int]",
            "relevant_doc_ids": "optional list[int]",
        },
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote retrieval report to {out}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
