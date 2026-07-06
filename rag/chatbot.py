from __future__ import annotations

from rag.agent import RetrievalAgent
from rag.retriever import TfidfRetriever
from rag.types import AgentStep, QueryAnalysis, RetrievalResult


class FoodChatbot:
    def __init__(self, retriever: TfidfRetriever):
        self.retriever = retriever
        self.agent = RetrievalAgent(retriever)

    def answer(self, question: str, top_k: int = 4, mode: str = "baseline") -> dict:
        if mode == "agentic":
            agent_run = self.agent.run(question, top_k=top_k)
            results = agent_run.results
            analysis = agent_run.analysis
            agent_steps = agent_run.steps
        else:
            results = self.retriever.search(question, top_k=top_k)
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
                },
            }

        answer = self.build_answer(question, results, mode=mode, analysis=analysis)
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
                        "content": item.chunk.text,
                    }
                    for item in results
                ],
                "prompt_preview": self.build_prompt(question, results, analysis=analysis, mode=mode),
                "agent_steps": self.serialize_agent_steps(agent_steps),
                "query_analysis": self.serialize_query_analysis(analysis),
            },
        }

    def build_answer(
        self,
        question: str,
        results: list[RetrievalResult],
        mode: str = "baseline",
        analysis: QueryAnalysis | None = None,
    ) -> str:
        top_chunk = results[0].chunk
        top_document = top_chunk.document
        recommendation_lines = [
            f"Neu ban dang hoi ve '{question.strip()}', minh goi y ban bat dau voi {top_document.title}.",
        ]

        if top_document.addresses:
            recommendation_lines.append(
                "Dia chi noi bat: " + "; ".join(top_document.addresses) + "."
            )

        recommendation_lines.append(
            "Doan ngu canh phu hop nhat:\n" + top_chunk.text
        )

        companion_chunks = [
            item.chunk for item in results[1:] if item.chunk.document.doc_id == top_document.doc_id
        ]
        if companion_chunks:
            recommendation_lines.append(
                "Thong tin bo sung cung quan:\n"
                + "\n".join(f"- {chunk.text}" for chunk in companion_chunks[:2])
            )

        if len(results) > 1:
            alternative_titles = ", ".join(
                list(dict.fromkeys(item.chunk.document.title for item in results[1:]))
            )
            recommendation_lines.append(
                f"Ban cung co the tham khao them: {alternative_titles}."
            )

        if mode == "agentic" and analysis is not None:
            recommendation_lines.append(
                "Agent da thu nhieu bien the truy van, loc theo y dinh va chon lai cac chunk phu hop nhat truoc khi tra loi."
            )
            recommendation_lines.append(
                "Tom tat hieu truy van: " + self.describe_analysis(analysis) + "."
            )
        else:
            recommendation_lines.append(
                "Cau tra loi nay dang duoc tao theo dang template tu cac doan du lieu truy xuat duoc de ban de quan sat flow RAG."
            )
        return "\n\n".join(recommendation_lines)

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
            "Neu thieu du lieu thi noi ro la chua co thong tin trong bo du lieu."
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
