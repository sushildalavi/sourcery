"""Second-stage reranker over first-stage retrieval results.

Stage 1 (pgvector ANN + sparse overlap) optimizes for recall — it casts a wide
net so the right chunk is somewhere in the top-K. Stage 2 (this module) re-scores
the shortlist with signals that are too expensive to apply to every chunk in
the corpus but cheap on the top-50: lexical bigram overlap, exact-phrase hits,
title-position weighting.

The reranker is intentionally pluggable — a future cross-encoder (e.g.
`cross-encoder/ms-marco-MiniLM-L-6-v2`) can be wired behind the same interface
without touching the call sites in `backend/app.py`.

Public surface
--------------
- `rerank_candidates(query, candidates, top_k=None) -> list[ScoredCandidate]`
- `RERANK_ENABLED` env flag (default: True). Disable to A/B against stage-1 only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "50") or 50)

# Tunable blend: stage-1 score vs stage-2 lexical signal.
# Higher alpha = more weight on the original retrieval similarity;
# 0.6 leaves enough room for the lexical signal to flip ties without
# demoting strong dense matches.
_BLEND_ALPHA = float(os.getenv("RERANK_ALPHA", "0.6") or 0.6)


@dataclass
class ScoredCandidate:
    chunk_id: int | str
    text: str
    stage1_score: float
    stage2_score: float
    final_score: float
    title: str | None = None


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _content_token_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    qset = {t for t in query_tokens if len(t) > 2}
    if not qset:
        return 0.0
    dset = set(doc_tokens)
    return len(qset & dset) / max(1, len(qset))


def _bigram_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    qb = _bigrams(query_tokens)
    if not qb:
        return 0.0
    db = _bigrams(doc_tokens)
    if not db:
        return 0.0
    return len(qb & db) / max(1, len(qb))


def _exact_phrase_bonus(query: str, text: str) -> float:
    q = (query or "").strip().lower()
    if len(q.split()) < 2:
        return 0.0
    return 0.10 if q in (text or "").lower() else 0.0


def _title_match_bonus(query_tokens: list[str], title: str | None) -> float:
    if not title:
        return 0.0
    title_tokens = set(_tokens(title))
    if not title_tokens:
        return 0.0
    qset = {t for t in query_tokens if len(t) > 2}
    if not qset:
        return 0.0
    overlap = len(qset & title_tokens) / max(1, len(qset))
    return 0.05 * overlap


def _lexical_score(query: str, text: str, title: str | None) -> float:
    qt = _tokens(query)
    dt = _tokens(text)
    score = 0.0
    score += 0.60 * _content_token_overlap(qt, dt)
    score += 0.30 * _bigram_overlap(qt, dt)
    score += _exact_phrase_bonus(query, text)
    score += _title_match_bonus(qt, title)
    return min(1.0, score)


def rerank_candidates(
    query: str,
    candidates: Iterable[dict],
    *,
    top_k: int | None = None,
) -> list[ScoredCandidate]:
    """Rerank a stage-1 candidate list.

    `candidates` is an iterable of dicts with at least `chunk_id` (or `id`)
    and `text`. Optional fields: `score` (stage-1), `title`.

    Returns the same items wrapped in `ScoredCandidate`, sorted by
    `final_score` descending. If reranking is disabled, returns the
    original order with stage-1 scores propagated to `final_score`.
    """
    items: list[ScoredCandidate] = []
    for c in candidates:
        cid = c.get("chunk_id") if "chunk_id" in c else c.get("id")
        text = c.get("text") or c.get("snippet") or ""
        stage1 = float(c.get("score") or c.get("sim_score") or 0.0)
        title = c.get("title")
        if RERANK_ENABLED:
            stage2 = _lexical_score(query, text, title)
            final = _BLEND_ALPHA * stage1 + (1.0 - _BLEND_ALPHA) * stage2
        else:
            stage2 = 0.0
            final = stage1
        items.append(
            ScoredCandidate(
                chunk_id=cid,
                text=text,
                stage1_score=round(stage1, 6),
                stage2_score=round(stage2, 6),
                final_score=round(final, 6),
                title=title,
            )
        )
    items.sort(key=lambda x: x.final_score, reverse=True)
    if top_k is not None:
        items = items[:top_k]
    return items
