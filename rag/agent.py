from __future__ import annotations

from dataclasses import dataclass

from rag.query_router import analyze_query
from rag.retriever import TfidfRetriever
from rag.types import AgentStep, QueryAnalysis, RetrievalResult


@dataclass
class AgentRun:
    results: list[RetrievalResult]
    analysis: QueryAnalysis
    steps: list[AgentStep]


class RetrievalAgent:
    def __init__(self, retriever: TfidfRetriever):
        self.retriever = retriever

    def run(self, question: str, top_k: int = 4, strategy: str = "hybrid") -> AgentRun:
        analysis = analyze_query(question)
        steps: list[AgentStep] = [
            AgentStep(
                name="analyze_query",
                detail="Phan tich y dinh, category, mon an va dia diem trong cau hoi.",
                payload={
                    "categories": analysis.categories,
                    "cuisine_terms": analysis.cuisine_terms,
                    "vibe_terms": analysis.vibe_terms,
                    "location_terms": analysis.location_terms,
                    "intents": analysis.intents,
                },
            )
        ]

        candidate_map: dict[str, RetrievalResult] = {}
        for variant in analysis.query_variants[:4]:
            variant_results = self.retriever.search_filtered(
                variant,
                top_k=top_k,
                category_filters=analysis.categories,
                location_terms=analysis.location_terms,
                strategy=strategy,
            )
            steps.append(
                AgentStep(
                    name="search_chunks",
                    detail=f"Truy xuat voi bien the truy van: '{variant}'.",
                    payload={
                        "query_variant": variant,
                        "hits": [
                            {
                                "chunk_id": item.chunk.chunk_id,
                                "title": item.chunk.document.title,
                                "score": round(item.score, 4),
                                "score_breakdown": item.score_breakdown,
                            }
                            for item in variant_results
                        ],
                    },
                )
            )
            for item in variant_results:
                key = item.chunk.chunk_id
                candidate_map[key] = self._pick_better(candidate_map.get(key), item, question, analysis)

        reranked = sorted(
            candidate_map.values(),
            key=lambda item: self._final_score(item, question, analysis),
            reverse=True,
        )[:top_k]

        steps.append(
            AgentStep(
                name="rerank_candidates",
                detail="Tong hop cac ket qua retrieve va cham diem lai theo overlap, category va y dinh.",
                payload={
                    "final_hits": [
                        {
                            "chunk_id": item.chunk.chunk_id,
                            "title": item.chunk.document.title,
                            "score": round(item.score, 4),
                            "category": item.chunk.document.category,
                            "score_breakdown": item.score_breakdown,
                        }
                        for item in reranked
                    ]
                },
            )
        )

        return AgentRun(results=reranked, analysis=analysis, steps=steps)

    def _pick_better(
        self,
        previous: RetrievalResult | None,
        current: RetrievalResult,
        question: str,
        analysis: QueryAnalysis,
    ) -> RetrievalResult:
        if previous is None:
            return current
        previous_score = self._final_score(previous, question, analysis)
        current_score = self._final_score(current, question, analysis)
        return current if current_score > previous_score else previous

    def _final_score(
        self,
        item: RetrievalResult,
        question: str,
        analysis: QueryAnalysis,
    ) -> float:
        lexical_bonus = self.retriever.lexical_overlap_score(question, item.chunk.text)
        category_bonus = 0.12 if item.chunk.document.category in analysis.categories else 0.0
        cuisine_bonus = 0.08 if any(term in item.chunk.text.lower() for term in analysis.cuisine_terms) else 0.0
        vibe_bonus = 0.05 if any(term in item.chunk.text.lower() for term in analysis.vibe_terms) else 0.0
        return item.score + lexical_bonus + category_bonus + cuisine_bonus + vibe_bonus
