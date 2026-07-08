from __future__ import annotations

import unittest

from rag.web_search import TavilyWebSearch
from scripts.run_eval import extract_retrieved_titles

try:
    from rag.chatbot import FoodChatbot
except ModuleNotFoundError:
    FoodChatbot = None


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def answer(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"answer": "ok", "sources": [], "debug": {}}


class ProjectContractTests(unittest.TestCase):
    @unittest.skipIf(FoodChatbot is None, "Project runtime dependencies are not available in this interpreter.")
    def test_food_chatbot_forwards_runtime_options(self) -> None:
        chatbot = FoodChatbot.__new__(FoodChatbot)
        chatbot.pipeline = _FakePipeline()

        chatbot.answer(
            "bun bo o dau",
            top_k=3,
            mode="baseline",
            retrieval_strategy="lexical",
            generation_mode="template",
            conversation_messages=[{"role": "user", "content": "bun bo"}],
            search_mode="local_only",
        )

        self.assertEqual(
            chatbot.pipeline.calls[-1],
            {
                "question": "bun bo o dau",
                "top_k": 3,
                "mode": "baseline",
                "retrieval_strategy": "lexical",
                "generation_mode": "template",
                "conversation_messages": [{"role": "user", "content": "bun bo"}],
                "search_mode": "local_only",
            },
        )

    def test_extract_retrieved_titles_reads_debug_payload(self) -> None:
        result = {
            "answer": "ok",
            "sources": [],
            "debug": {
                "retrieved_chunks": [
                    {"title": "Quan A"},
                    {"title": "Quan B"},
                    {"title": ""},
                ]
            },
        }

        self.assertEqual(extract_retrieved_titles(result), ["Quan A", "Quan B"])

    def test_tavily_wrapper_fails_with_clear_message_when_unavailable(self) -> None:
        searcher = TavilyWebSearch(api_key="")
        searcher.client = None

        with self.assertRaises(RuntimeError) as context:
            searcher.search("bun bo da nang")

        self.assertTrue(str(context.exception))


if __name__ == "__main__":
    unittest.main()
