import unittest
from unittest.mock import patch

from backend.services.assistant_utils import (
    _citations_support_entity_benchmark_pair,
    _query_mentions_unseen_terms,
    _uploaded_title_prior_boost,
)
from backend.services.judge import evaluate_faithfulness


class UploadedPriorTests(unittest.TestCase):
    def test_natural_questions_prior(self):
        q = "Which dataset pairs real Google search queries with long-form answers from Wikipedia?"
        self.assertGreater(_uploaded_title_prior_boost(q, "06_NaturalQuestions.pdf"), 0.5)
        self.assertLess(_uploaded_title_prior_boost(q, "05_SQuAD.pdf"), 0.0)

    def test_sparse_bm25_prior_prefers_drqa(self):
        q = "Which paper's retrieval step relies on sparse BM25-style term matching rather than dense vectors?"
        self.assertGreater(_uploaded_title_prior_boost(q, "07_DrQA.pdf"), 0.5)
        self.assertLess(_uploaded_title_prior_boost(q, "01_DPR.pdf"), 0.0)

    def test_dual_encoder_beats_bm25_prior_prefers_dpr(self):
        q = "Which retrieval paper argues that a single fine-tuned dual encoder beats BM25 across open-domain QA benchmarks?"
        self.assertGreater(_uploaded_title_prior_boost(q, "01_DPR.pdf"), 0.5)
        self.assertLess(_uploaded_title_prior_boost(q, "04_BEIR.pdf"), 0.0)


class AbstentionGuardTests(unittest.TestCase):
    def test_fabricated_lora_terms_flagged(self):
        q = "What is the difference between Alpha-LoRA and Beta-LoRA as described in the corpus?"
        self.assertTrue(_query_mentions_unseen_terms(q, citations=[]))

    def test_common_hyphenation_not_flagged(self):
        q = "Which retrieval paper argues that a single fine-tuned dual encoder beats BM25?"
        self.assertFalse(_query_mentions_unseen_terms(q, citations=[]))

    def test_entity_benchmark_pair_requires_joint_evidence(self):
        q = "In DPR, what is the exact value of the top-20 retrieval accuracy on the WebQuestions benchmark?"
        citations = [
            {"title": "01_DPR.pdf", "snippet": "Dense Passage Retrieval improves top-20 accuracy on QA tasks."},
            {"title": "04_BEIR.pdf", "snippet": "WebQuestions is one benchmark in open-domain QA."},
        ]
        self.assertFalse(_citations_support_entity_benchmark_pair(q, citations))

    def test_entity_benchmark_pair_passes_when_jointly_present(self):
        q = "In DPR, what is the exact value of the top-20 retrieval accuracy on the WebQuestions benchmark?"
        citations = [
            {
                "title": "01_DPR.pdf",
                "snippet": "DPR reports top-20 retrieval accuracy of 79.4 on WebQuestions.",
            }
        ]
        self.assertTrue(_citations_support_entity_benchmark_pair(q, citations))


class JudgeCoverageFallbackTests(unittest.TestCase):
    @patch("backend.services.judge._client")
    def test_llm_empty_claims_falls_back_to_sentence_labels(self, mock_client):
        mock_completion = type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Msg",
                                (),
                                {
                                    "content": (
                                        '{"overall_score":0.0,"citation_coverage":0.0,'
                                        '"supported_count":0,"unsupported_count":0,'
                                        '"sentence_count":2,"claims":[]}'
                                    )
                                },
                            )()
                        },
                    )()
                ]
            },
        )()
        mock_client.return_value.chat.completions.create.return_value = mock_completion

        report = evaluate_faithfulness(
            query="What is BERT?",
            answer="BERT is a language model. It is widely used.",
            citations=[],
            use_llm=True,
        )
        self.assertEqual(report.get("method"), "llm_empty_fallback")
        self.assertEqual(report.get("sentence_count"), 2)
        self.assertEqual(len(report.get("unsupported", [])), 2)


if __name__ == "__main__":
    unittest.main()
