from __future__ import annotations

from rag.agent import RetrievalAgent
from rag.generator import OllamaAnswerGenerator, TemplateAnswerGenerator
from rag.retriever import TfidfRetriever
from rag.types import AgentStep, QueryAnalysis, RetrievalResult


class FoodChatbot:
    def __init__(self, retriever: TfidfRetriever):
        self.retriever = retriever
        self.agent = RetrievalAgent(retriever)
        self.template_generator = TemplateAnswerGenerator()
        self.ollama_generator = OllamaAnswerGenerator()

    def answer(
        self,
        question: str,
        top_k: int = 4,
        mode: str = "baseline",
        retrieval_strategy: str = "hybrid",
        generation_mode: str = "template",
    ) -> dict:
        if mode == "agentic":
            agent_run = self.agent.run(question, top_k=top_k, strategy=retrieval_strategy)
            results = agent_run.results
            analysis = agent_run.analysis
            agent_steps = agent_run.steps
        else:
            results = self.retriever.search(question, top_k=top_k, strategy=retrieval_strategy)
            analysis = None
            agent_steps = []

        if not results:
            return {
                "answer": (
                    "Minh chua tim thay thong tin phu hop trong bo du lieu hien tai. "
                    "Ban hay thu hoi ro hon ve mon an, kieu quan hoac ten dia diem o TP.HCM."
                ),
                "sources": [],
                "debug": {
                    "retrieved_chunks": [],
                    "prompt_preview": self.build_prompt(question, [], analysis=analysis, mode=mode),
                    "agent_steps": self.serialize_agent_steps(agent_steps),
                    "query_analysis": self.serialize_query_analysis(analysis),
                    "retrieval_strategy": retrieval_strategy,
                    "generation_mode": generation_mode,
                },
            }

        prompt_preview = self.build_prompt(question, results, analysis=analysis, mode=mode)
        answer, generation_debug = self.generate_answer(
            question=question,
            results=results,
            mode=mode,
            analysis=analysis,
            prompt_preview=prompt_preview,
            generation_mode=generation_mode,
        )

        return {
            "answer": answer,
            "sources": results,
            "debug": {
                "retrieved_chunks": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "title": item.chunk.document.title,
                        "score": round(item.score, 4),
                        "matched_terms": item.matched_terms,
                        "score_breakdown": item.score_breakdown,
                        "content": item.chunk.text,
                    }
                    for item in results
                ],
                "prompt_preview": prompt_preview,
                "agent_steps": self.serialize_agent_steps(agent_steps),
                "query_analysis": self.serialize_query_analysis(analysis),
                "retrieval_strategy": retrieval_strategy,
                "generation_mode": generation_mode,
                "generation_debug": generation_debug,
            },
        }

    def generate_answer(
        self,
        question: str,
        results: list[RetrievalResult],
        mode: str,
        analysis: QueryAnalysis | None,
        prompt_preview: str,
        generation_mode: str,
    ) -> tuple[str, dict]:
        if generation_mode == "ollama":
            return self.ollama_generator.generate(
                question=question,
                results=results,
                mode=mode,
                prompt=prompt_preview,
                analysis=analysis,
            )
        return self.template_generator.generate(
            question=question,
            results=results,
            mode=mode,
            prompt=prompt_preview,
            analysis=analysis,
        )

    def build_prompt(
        self,
        question: str,
        results: list[RetrievalResult],
        analysis: QueryAnalysis | None = None,
        mode: str = "baseline",
    ) -> str:
        context_lines = []
        for item in results:
            document = item.chunk.document
            addresses = "; ".join(document.addresses) if document.addresses else "Khong co dia chi"
            context_lines.append(
                f"- {item.chunk.chunk_id} | {document.title} | {addresses} | {item.chunk.text}"
            )

        context_block = "\n".join(context_lines) if context_lines else "- Khong co ngu canh phu hop"
        return (
            "Ban la chatbot am thuc TP.HCM.\n"
            "Chi tra loi dua tren ngu canh da truy xuat.\n"
            f"Che do: {mode}\n"
            f"Cau hoi: {question}\n"
            f"Phan tich truy van: {self.describe_analysis(analysis) if analysis else 'Khong co'}\n"
            f"Ngu canh:\n{context_block}\n"
            "Tra loi ngan gon, co neu duoc thi nen dua goi y chinh, dia chi va nhac ro khi du lieu chua du."
        )

    def describe_analysis(self, analysis: QueryAnalysis) -> str:
        parts = [
            f"category={', '.join(analysis.categories) or 'khong ro'}",
            f"mon={', '.join(analysis.cuisine_terms) or 'khong ro'}",
            f"vibe={', '.join(analysis.vibe_terms) or 'khong ro'}",
            f"dia_diem={', '.join(analysis.location_terms) or 'khong ro'}",
            f"intent={', '.join(analysis.intents) or 'khong ro'}",
        ]
        return "; ".join(parts)

    def serialize_query_analysis(self, analysis: QueryAnalysis | None) -> dict | None:
        if analysis is None:
            return None
        return {
            "normalized_query": analysis.normalized_query,
            "categories": analysis.categories,
            "location_terms": analysis.location_terms,
            "cuisine_terms": analysis.cuisine_terms,
            "vibe_terms": analysis.vibe_terms,
            "intents": analysis.intents,
            "query_variants": analysis.query_variants,
        }

    def serialize_agent_steps(self, steps: list[AgentStep]) -> list[dict]:
        return [
            {
                "name": step.name,
                "detail": step.detail,
                "payload": step.payload,
            }
            for step in steps
        ]
