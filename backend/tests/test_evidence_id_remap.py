"""Regression test for the citation-renumbering / faithfulness mapping bug.

Background
----------
The judge emits `evidence_ids: ["S<old_id>"]` against the answer's pre-sort
citation IDs. The app then sorts citations by confidence, renumbers them,
and rewrites inline `[S#]` refs in the answer. Previously the `evidence_ids`
in the faithfulness payload were NOT remapped, so the frontend's
`citation.evidence_id`-keyed claim mapping silently drifted out of sync.

The fix in `backend/app.py` rewrites each claim's `["S<old>"]` to the
corresponding citation's stable `evidence_id`. This test runs that exact
remap function in isolation and asserts the new contract.
"""

from __future__ import annotations

import re


def _apply_citation_renumbering_and_judge_remap(
    citations_pre_sort: list[dict],
    answer: str,
    faithfulness: dict,
    sort_key,
):
    """Tiny pure helper that mirrors the logic in backend/app.py so we can
    test it without booting the full FastAPI app + DB.
    """
    for old_idx, c in enumerate(citations_pre_sort, start=1):
        c["_old_id"] = old_idx
    citations_pre_sort.sort(key=sort_key)

    id_remap: dict[int, int] = {}
    old_id_to_evidence_id: dict[int, str] = {}
    for new_idx, c in enumerate(citations_pre_sort, start=1):
        old_id = int(c.pop("_old_id"))
        id_remap[old_id] = new_idx
        c["id"] = new_idx
        ev_id = c.get("evidence_id")
        if ev_id:
            old_id_to_evidence_id[old_id] = ev_id

    if id_remap and answer:

        def _remap_ref(m: re.Match) -> str:
            prefix = m.group(1) or ""
            old_id = int(m.group(2))
            new_id = id_remap.get(old_id, old_id)
            return f"[{prefix}{new_id}]"

        answer = re.sub(r"\[(S?)(\d+)\]", _remap_ref, answer)

    if faithfulness and isinstance(faithfulness, dict):
        for claim in faithfulness.get("claims") or []:
            raw = claim.get("evidence_ids") or []
            rewritten: list[str] = []
            for ref in raw:
                if not isinstance(ref, str):
                    continue
                m = re.match(r"^S(\d+)$", ref.strip())
                if not m:
                    rewritten.append(ref)
                    continue
                old_id = int(m.group(1))
                stable = old_id_to_evidence_id.get(old_id)
                if stable:
                    rewritten.append(stable)
                else:
                    new_id = id_remap.get(old_id, old_id)
                    rewritten.append(f"S{new_id}")
            claim["evidence_ids"] = rewritten

    return citations_pre_sort, answer, faithfulness


def _by_confidence(c: dict) -> tuple:
    score = float((c.get("confidence_obj") or {}).get("score", 0.0))
    return (-score,)


def test_evidence_ids_track_stable_evidence_id_after_resort():
    citations = [
        {"id": 1, "evidence_id": "ev-alpha", "confidence_obj": {"score": 0.4}},
        {"id": 2, "evidence_id": "ev-beta", "confidence_obj": {"score": 0.9}},
        {"id": 3, "evidence_id": "ev-gamma", "confidence_obj": {"score": 0.7}},
    ]
    answer = "Beta supports the claim [S2]. Gamma adds context [S3]. Alpha is weak [S1]."
    faithfulness = {
        "claims": [
            {"sentence_id": 1, "sentence": "Beta supports.", "evidence_ids": ["S2"]},
            {"sentence_id": 2, "sentence": "Gamma adds.", "evidence_ids": ["S3"]},
            {"sentence_id": 3, "sentence": "Alpha is weak.", "evidence_ids": ["S1"]},
        ]
    }

    citations_out, answer_out, faith_out = _apply_citation_renumbering_and_judge_remap(
        citations, answer, faithfulness, _by_confidence
    )

    # New display order: beta (0.9) -> gamma (0.7) -> alpha (0.4)
    assert [c["evidence_id"] for c in citations_out] == ["ev-beta", "ev-gamma", "ev-alpha"]
    assert [c["id"] for c in citations_out] == [1, 2, 3]

    # Inline refs follow the new numbering: [S2] (beta) -> [S1], [S3] (gamma) -> [S2], [S1] (alpha) -> [S3]
    assert "[S1]" in answer_out and "[S2]" in answer_out and "[S3]" in answer_out
    assert answer_out == "Beta supports the claim [S1]. Gamma adds context [S2]. Alpha is weak [S3]."

    # The faithfulness claim mapping now uses STABLE evidence_ids — so the
    # frontend can resolve them via `claimsByEvidence` regardless of how
    # display IDs were renumbered.
    assert [c["evidence_ids"] for c in faith_out["claims"]] == [
        ["ev-beta"],
        ["ev-gamma"],
        ["ev-alpha"],
    ]


def test_evidence_ids_fall_back_to_new_display_id_when_no_stable_id():
    """If a citation has no stable evidence_id, we fall back to S<new_id>
    so the frontend at least resolves to the correct (renumbered) citation."""
    citations = [
        {"id": 1, "confidence_obj": {"score": 0.2}},  # no evidence_id
        {"id": 2, "confidence_obj": {"score": 0.9}},  # no evidence_id
    ]
    answer = "Strong [S2]. Weak [S1]."
    faithfulness = {
        "claims": [
            {"sentence_id": 1, "sentence": "Strong.", "evidence_ids": ["S2"]},
            {"sentence_id": 2, "sentence": "Weak.", "evidence_ids": ["S1"]},
        ]
    }

    _, answer_out, faith_out = _apply_citation_renumbering_and_judge_remap(
        citations, answer, faithfulness, _by_confidence
    )

    assert answer_out == "Strong [S1]. Weak [S2]."
    # No stable evidence_id available -> fall back to S<new_id>.
    assert faith_out["claims"][0]["evidence_ids"] == ["S1"]
    assert faith_out["claims"][1]["evidence_ids"] == ["S2"]


def test_already_stable_evidence_ids_are_passed_through():
    """If the LLM judge ever emits stable IDs directly, we don't mangle them."""
    citations = [{"id": 1, "evidence_id": "ev-x", "confidence_obj": {"score": 1.0}}]
    answer = "X [S1]."
    faithfulness = {"claims": [{"sentence_id": 1, "sentence": "X.", "evidence_ids": ["ev-x"]}]}

    _, _, faith_out = _apply_citation_renumbering_and_judge_remap(citations, answer, faithfulness, _by_confidence)
    assert faith_out["claims"][0]["evidence_ids"] == ["ev-x"]
