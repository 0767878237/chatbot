from __future__ import annotations

from rag.adaptive_rag import AdaptiveRAGPipeline
from rag.retriever import TfidfRetriever


class FoodChatbot:
    def __init__(self, retriever: TfidfRetriever):
        self.pipeline = AdaptiveRAGPipeline(retriever)

    def answer(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "agentic",
        retrieval_strategy: str = "hybrid",
        generation_mode: str = "template",
        conversation_messages: list[dict] | None = None,
        search_mode: str = "adaptive",
    ) -> dict:
        return self.pipeline.answer(
            question=question,
            top_k=5,
            mode="agentic",
            retrieval_strategy="hybrid",
            generation_mode="template",
            conversation_messages=conversation_messages,
            search_mode="adaptive",
        )
