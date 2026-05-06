import unittest
from types import SimpleNamespace
from unittest.mock import patch

import backend.app as app_module


class _DummyCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def _dummy_client(content: str):
    return SimpleNamespace(chat=SimpleNamespace(completions=_DummyCompletions(content)))


class AssistantAnswerRoutingTests(unittest.TestCase):
    def test_public_scope_short_ambiguous_query_uses_retrieval_not_chat_bypass(self):
        uploaded_results = {
            "results": [
                {
                    "id": 101,
                    "document_id": 2,
                    "title": "02_ColBERT.pdf",
                    "doc_type": "research_paper",
                    "text": (
                        "ColBERT is a late interaction retrieval model over BERT for "
                        "passage ranking in information retrieval."
                    ),
                    "page_no": 1,
                    "distance": 0.08,
                }
            ]
        }
        public_results = {
            "results": [
                {
                    "title": "Stephen Colbert and political satire",
                    "source": "semanticscholar",
                    "abstract": "Late-night television comedy and satire.",
                    "_sim": 0.92,
                    "url": "https://example.com/colbert-tv",
                }
            ],
            "provider_status": {},
        }

        with (
            patch.object(app_module, "_chat_answer", side_effect=AssertionError("chat bypass should not run")),
            patch.object(app_module, "search_uploaded_chunks", return_value=uploaded_results),
            patch.object(app_module, "public_live_search", return_value=public_results),
            patch.object(
                app_module, "_rank_and_trim_citations", side_effect=lambda query, citations, k, **kwargs: citations[:k]
            ),
            patch.object(app_module, "_build_generation_prompt", return_value="prompt"),
            patch.object(app_module, "_compute_citation_msa", return_value=({}, 0)),
            patch.object(app_module, "_has_official_company_docs", return_value=True),
            patch.object(app_module, "client", _dummy_client("ColBERT is a retrieval model [S1].")),
            patch.object(app_module, "log_json", return_value=None),
        ):
            resp = app_module.assistant_answer({"query": "tell me about Colbert", "scope": "public", "k": 4})

        self.assertTrue(resp["citations"])
        self.assertEqual(resp["citations"][0]["source"], "uploaded")
        self.assertIn("colbert", resp["citations"][0]["title"].lower())
        self.assertNotIn("stephen colbert", " ".join(c["title"] for c in resp["citations"]).lower())

    def test_public_scope_offtopic_ambiguous_query_abstains_after_filtering(self):
        public_results = {
            "results": [
                {
                    "title": "Stephen Colbert and the Late Show",
                    "source": "semanticscholar",
                    "abstract": "Political satire and television studies.",
                    "_sim": 0.94,
                    "url": "https://example.com/late-show",
                }
            ],
            "provider_status": {},
        }

        with (
            patch.object(app_module, "_chat_answer", side_effect=AssertionError("chat bypass should not run")),
            patch.object(app_module, "search_uploaded_chunks", return_value={"results": []}),
            patch.object(app_module, "public_live_search", return_value=public_results),
            patch.object(app_module, "log_json", return_value=None),
        ):
            resp = app_module.assistant_answer({"query": "tell me about Colbert", "scope": "public", "k": 4})

        self.assertEqual(resp["citations"], [])
        self.assertEqual(resp["retrieval_policy"]["mode"], "abstention")
        self.assertIn("Abstained", resp["confidence"]["label"])

    def test_public_scope_plain_small_talk_still_uses_chat_bypass(self):
        with (
            patch.object(app_module, "_chat_answer", return_value="hello there"),
            patch.object(app_module, "log_json", return_value=None),
        ):
            resp = app_module.assistant_answer({"query": "hello", "scope": "public"})

        self.assertEqual(resp["answer"], "hello there")
        self.assertEqual(resp["citations"], [])

    def test_public_scope_short_non_ambiguous_query_uses_retrieval_not_chat_bypass(self):
        """Regression: 'tell me abut RGANs' (4 words, typo, unknown acronym) used
        to short-circuit to _chat_answer because no research cue matched and the
        legacy sense_resolver didn't flag it as ambiguous. The LLM then
        fabricated a detailed answer from its priors with zero citations.
        Public-mode non-chatty queries must always route through retrieval so
        the abstention guard can fire when evidence is lacking.
        """
        public_results = {
            "results": [],
            "provider_status": {},
        }

        with (
            patch.object(
                app_module,
                "_chat_answer",
                side_effect=AssertionError("chat bypass should not run in public mode for non-chatty queries"),
            ),
            patch.object(app_module, "search_uploaded_chunks", return_value={"results": []}),
            patch.object(app_module, "public_live_search", return_value=public_results),
            patch.object(app_module, "log_json", return_value=None),
        ):
            resp = app_module.assistant_answer({"query": "tell me abut RGANs", "scope": "public", "k": 4})

        self.assertEqual(resp["citations"], [])
        self.assertIn("evidence", resp["answer"].lower())

    def test_public_scope_with_pinned_doc_does_not_leak_pinned_doc_citations(self):
        """Regression: user pinned 15_LLMasJudge.pdf, switched to public
        scope, and asked about RAG. Previously we probed the uploaded
        corpus with doc_id pinned and flooded the answer with chunks from
        the pinned paper. Now the probe must be skipped entirely.
        """
        search_calls: list[dict] = []

        def _spy_search(payload):
            search_calls.append(dict(payload))
            return {
                "results": [
                    {
                        "id": 300,
                        "document_id": 15,
                        "title": "15_LLMasJudge.pdf",
                        "doc_type": "research_paper",
                        "text": "This paper is about LLM-as-a-Judge evaluation on MT-Bench.",
                        "page_no": 1,
                        "distance": 0.10,
                    }
                ]
            }

        public_results = {
            "results": [
                {
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
                    "source": "arxiv",
                    "abstract": "RAG combines a parametric seq2seq model with a DPR retriever.",
                    "_sim": 0.88,
                    "url": "https://example.com/rag",
                }
            ],
            "provider_status": {},
        }

        with (
            patch.object(app_module, "_chat_answer", side_effect=AssertionError("chat bypass should not run")),
            patch.object(app_module, "search_uploaded_chunks", side_effect=_spy_search),
            patch.object(app_module, "public_live_search", return_value=public_results),
            patch.object(
                app_module, "_rank_and_trim_citations", side_effect=lambda query, citations, k, **kwargs: citations[:k]
            ),
            patch.object(app_module, "_build_generation_prompt", return_value="prompt"),
            patch.object(app_module, "_compute_citation_msa", return_value=({}, 0)),
            patch.object(app_module, "_has_official_company_docs", return_value=True),
            patch.object(app_module, "client", _dummy_client("RAG is a retrieval-augmented generation approach [S1].")),
            patch.object(app_module, "log_json", return_value=None),
        ):
            resp = app_module.assistant_answer(
                {
                    "query": "Find recent papers on retrieval-augmented generation",
                    "scope": "public",
                    "doc_id": 15,
                    "k": 4,
                }
            )

        # The uploaded corpus must NOT be probed when a doc is pinned + public scope.
        self.assertEqual(
            search_calls,
            [],
            f"Uploaded probe should be skipped when doc_id is pinned in public scope. Got calls: {search_calls}",
        )
        self.assertTrue(resp["citations"], "expected public citations")
        for c in resp["citations"]:
            self.assertNotEqual(
                (c.get("source") or "").lower(),
                "uploaded",
                f"Pinned doc leaked into citations: {c}",
            )
            self.assertNotIn(
                "llmasjudge",
                (c.get("title") or "").lower().replace(" ", "").replace("-", ""),
                f"Pinned doc title leaked: {c.get('title')}",
            )


if __name__ == "__main__":
    unittest.main()
