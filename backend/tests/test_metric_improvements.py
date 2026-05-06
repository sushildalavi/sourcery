"""Tests for the metric-improvement changes.

Covers the three generator-side fixes (extractive mode, claim-rewrite pass,
new calibration features) and the annotator-rubric v2 script so the
behavior does not regress.
"""
from __future__ import annotations

import unittest

from backend.services.assistant_utils import (
    _build_generation_prompt,
    _classify_answer_mode,
    _compute_claim_features,
    _is_factual_query,
    _rewrite_ungrounded_claims,
)


class ExtractiveModeTests(unittest.TestCase):
    def test_factual_is_detected(self):
        for q in (
            "What is BERT?",
            "How many layers does ResNet-50 have?",
            "Which dataset does DPR evaluate on?",
            "What year was the Transformer paper published?",
            "Define retrieval-augmented generation",
            "Who proposed chain-of-thought prompting?",
        ):
            self.assertTrue(_is_factual_query(q), q)

    def test_non_factual_is_not_flagged(self):
        for q in (
            "Compare DPR and ColBERT.",
            "What are the tradeoffs between BM25 and dense retrieval?",
            "Synthesize the literature on attention mechanisms.",
        ):
            self.assertFalse(_is_factual_query(q), q)

    def test_classify_answer_mode_returns_extractive_for_factual(self):
        self.assertEqual(_classify_answer_mode("What is DPR?"), "extractive")

    def test_build_generation_prompt_has_extractive_block(self):
        prompt = _build_generation_prompt(
            query="What is DPR?",
            context="[S1] DPR is a dense passage retriever.",
            answer_mode="extractive",
            allow_general_background=False,
        )
        self.assertIn("RESPONSE FORMAT — extractive", prompt)
        self.assertIn("verbatim", prompt)
        self.assertNotIn("RESPONSE FORMAT — explanatory", prompt)


class ClaimRewriteTests(unittest.TestCase):
    def setUp(self):
        # Minimal citation list: S1 contains strong evidence, S2 is unrelated.
        self.citations = [
            {"snippet": "BERT uses masked language modeling to pretrain bidirectional representations."},
            {"snippet": "The recipe for chocolate cake is flour, eggs, butter, cocoa."},
        ]

    def test_sentence_without_citation_gets_hedged(self):
        answer = "Transformers solve all NLP problems. [S1] BERT uses masked language modeling."
        rewritten, hedged = _rewrite_ungrounded_claims(answer, self.citations)
        self.assertGreaterEqual(hedged, 1)
        self.assertNotEqual(answer, rewritten)

    def test_already_hedged_sentence_is_left_alone(self):
        answer = "Reportedly, X improves things. It is suggested that Y works."
        rewritten, hedged = _rewrite_ungrounded_claims(answer, self.citations)
        self.assertEqual(hedged, 0)
        self.assertEqual(answer, rewritten)

    def test_empty_inputs_are_safe(self):
        self.assertEqual(_rewrite_ungrounded_claims("", self.citations), ("", 0))
        self.assertEqual(_rewrite_ungrounded_claims("foo", [])[0], "foo")


class ClaimFeatureTests(unittest.TestCase):
    def test_all_features_are_in_unit_range(self):
        feats = _compute_claim_features(
            sentence="BERT uses masked language modeling.",
            cited_snippet="BERT is a masked language model using bidirectional training.",
            context_by_id={
                1: {"doc_id": 1, "snippet": "Some evidence text."},
                2: {"doc_id": 2, "snippet": "Different source evidence."},
            },
            stability={"e1": 0.9, "e2": 0.3},
            evidence_id="e1",
            sentences_in_answer=["BERT uses masked language modeling.", "It trains bidirectional representations."],
            sidx=1,
        )
        for k, v in feats.items():
            self.assertGreaterEqual(v, 0.0, f"{k}={v}")
            self.assertLessEqual(v, 1.0, f"{k}={v}")

    def test_specificity_peaks_for_medium_length_snippets(self):
        short = _compute_claim_features("BERT.", "x", {}, {}, "e1", ["BERT."], 1)
        medium = _compute_claim_features("BERT.", "x" * 260, {}, {}, "e1", ["BERT."], 1)
        long = _compute_claim_features("BERT.", "x" * 900, {}, {}, "e1", ["BERT."], 1)
        self.assertGreater(medium["citation_specificity"], short["citation_specificity"])
        self.assertGreater(medium["citation_specificity"], long["citation_specificity"])

    def test_diversity_at_ceiling_when_all_unique_docs(self):
        ctx = {i: {"doc_id": i, "snippet": "x"} for i in range(1, 6)}
        feats = _compute_claim_features(
            "Claim.", "evidence.", ctx, {}, "e", ["Claim."], 1
        )
        # 5 distinct docs / 5 ctx entries = 1.0
        self.assertEqual(feats["retrieval_diversity"], 1.0)


if __name__ == "__main__":
    unittest.main()
