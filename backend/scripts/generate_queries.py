"""Generate 120 fresh queries targeted at the corpus via GPT-4o-mini.

Produces Evaluation/queries/queries_120.json with the schema:
    [
      {
        "query_id": "q1",
        "query": "...",
        "target_doc_id": 17,         # primary paper the query is about
        "target_doc_title": "01_ResNet.pdf",
        "query_type": "definitional" | "methodology" | "factual" | ...,
        "mode_affinity": "uploaded" | "public"   # suggested default scope
      },
      ...
    ]

Breakdown (all divide cleanly into 120):
    - 8 queries per paper × 15 papers = 120
    - Each paper gets 2 queries per type across 4 canonical types
      (definitional, methodology, factual, limitations)

Usage:
    python -m backend.scripts.generate_queries
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from backend.services.db import fetchall
from backend.utils.config import get_openai_api_key

MODEL = "gpt-4o-mini"
OUT = Path("Evaluation/queries/queries_120.json")

QUERY_TYPES: List[str] = ["definitional", "methodology", "factual", "limitations"]

CORPUS_TITLES = {
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
}

HUMAN_NAMES = {
    "01_ResNet.pdf": "Deep Residual Learning for Image Recognition (ResNet)",
    "02_GAN.pdf": "Generative Adversarial Networks (GAN)",
    "03_Word2Vec.pdf": "Efficient Estimation of Word Representations in Vector Space (Word2Vec)",
    "04_LLaMA2.pdf": "LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
    "05_Chinchilla.pdf": "Training Compute-Optimal Large Language Models (Chinchilla)",
    "06_ConstitutionalAI.pdf": "Constitutional AI: Harmlessness from AI Feedback",
    "07_AlphaGo.pdf": "Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo)",
    "08_CLIP.pdf": "Learning Transferable Visual Models From Natural Language Supervision (CLIP)",
    "09_AlphaFold.pdf": "Highly Accurate Protein Structure Prediction with AlphaFold",
    "10_StableDiffusion.pdf": "High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)",
    "11_LSTM.pdf": "Long Short-Term Memory (LSTM)",
    "12_DQN.pdf": "Human-level Control Through Deep Reinforcement Learning (DQN)",
    "13_VAE.pdf": "Auto-Encoding Variational Bayes (VAE)",
    "14_SwinTransformer.pdf": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
    "15_PageRank.pdf": "The Anatomy of a Large-Scale Hypertextual Web Search Engine (PageRank)",
}

TYPE_PROMPTS: Dict[str, str] = {
    "definitional": (
        "Write 2 natural-language questions a graduate student would ask to learn "
        "the CORE IDEA / DEFINITION of the paper. Good patterns: 'What is X?', "
        "'How does X work?', 'What problem does X solve?'."
    ),
    "methodology": (
        "Write 2 questions about the paper's METHOD or technique. Good patterns: "
        "'How is X trained?', 'What architecture does X use?', 'What is the objective "
        "function of X?', 'What inputs does X take?'."
    ),
    "factual": (
        "Write 2 questions about SPECIFIC factual details or results the paper "
        "reports. Good patterns: 'What dataset does X use?', 'What score does X achieve "
        "on benchmark Y?', 'How many parameters does X have?'. Questions should be "
        "answerable from the paper's own numbers/experiments."
    ),
    "limitations": (
        "Write 2 questions about the paper's LIMITATIONS, failure modes, or trade-offs. "
        "Good patterns: 'What are the limitations of X?', 'In which scenarios does X "
        "underperform?', 'What assumptions does X rely on?'."
    ),
}


def client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def sample_chunks_for(doc_id: int, n: int = 6) -> str:
    """Grab a few diverse chunks from the paper to ground the LLM's query generation."""
    rows = fetchall(
        """
        SELECT text
        FROM chunks
        WHERE document_id = %s
        ORDER BY page_no ASC, chunk_index ASC
        """,
        [doc_id],
    )
    if not rows:
        return ""
    pool = [(r.get("text") or "").strip() for r in rows if (r.get("text") or "").strip()]
    if not pool:
        return ""
    # Pick n evenly-spaced chunks across the paper.
    step = max(1, len(pool) // n)
    selected = pool[::step][:n]
    return "\n\n---\n\n".join(selected)[:6000]


def generate_for(doc_id: int, title: str, qtype: str, starting_index: int) -> List[Dict]:
    context = sample_chunks_for(doc_id)
    human_name = HUMAN_NAMES.get(title, title)
    system = (
        "You produce research-quality evaluation queries over a given scientific paper. "
        "Each query must be something a knowledgeable graduate student would naturally "
        "ask. Do NOT include citation markers or mention 'the paper' — write the question "
        "as if the student already knows the topic. Output strict JSON only."
    )
    user = (
        f"Paper: {human_name}\n\n"
        f"Query type: {qtype}\n\n"
        f"Excerpts (for grounding):\n{context or '(no excerpts available)'}\n\n"
        f"{TYPE_PROMPTS[qtype]}\n\n"
        'Return strict JSON: {"queries": ["...", "..."]} with exactly 2 entries.'
    )
    resp = client().chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        timeout=30,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        payload = json.loads(raw)
        questions = [q.strip() for q in payload.get("queries", []) if isinstance(q, str) and q.strip()]
    except Exception:
        matches = re.findall(r'"([^"]{10,})"', raw)
        questions = [m.strip() for m in matches][:2]
    out = []
    for i, q in enumerate(questions[:2]):
        idx = starting_index + i
        out.append(
            {
                "query_id": f"q{idx}",
                "query": q,
                "target_doc_id": doc_id,
                "target_doc_title": title,
                "query_type": qtype,
                "mode_affinity": "uploaded",
            }
        )
    return out


def main() -> int:
    rows = fetchall(
        """
        SELECT id, title
        FROM documents
        WHERE title = ANY(%s)
        ORDER BY title ASC
        """,
        [list(CORPUS_TITLES)],
    )
    if len(rows) != 15:
        print(f"[!] Expected 15 corpus docs, got {len(rows)}. Titles found:")
        for r in rows:
            print("  -", r.get("title"))
        return 1

    all_queries: List[Dict] = []
    next_idx = 1
    for row in rows:
        doc_id = int(row["id"])
        title = row["title"]
        print(f"=> {title} (doc_id={doc_id})")
        for qtype in QUERY_TYPES:
            batch = generate_for(doc_id, title, qtype, next_idx)
            for q in batch:
                print(f"     [{qtype}] {q['query']}")
            all_queries.extend(batch)
            next_idx += len(batch)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_queries, indent=2, ensure_ascii=False))
    print()
    print(f"Wrote {len(all_queries)} queries -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
