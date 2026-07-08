from __future__ import annotations

try:
    from langchain_core.runnables import RunnableLambda
except ImportError:  # pragma: no cover - exercised only in reduced environments
    class RunnableLambda:
        def __init__(self, func):
            self.func = func

        def __or__(self, other):
            if isinstance(other, RunnableLambda):
                return _RunnableChain([self.func, other.func])
            return _RunnableChain([self.func, other])

        def invoke(self, payload):
            return self.func(payload)


    class _RunnableChain:
        def __init__(self, steps):
            self.steps = list(steps)

        def __or__(self, other):
            if isinstance(other, RunnableLambda):
                return _RunnableChain(self.steps + [other.func])
            return _RunnableChain(self.steps + [other])

        def invoke(self, payload):
            state = payload
            for step in self.steps:
                state = step(state)
            return state

from rag.agent import RetrievalAgent
from rag.generator import OllamaAnswerGenerator, TemplateAnswerGenerator
from rag.memory import (
    build_memory_before_current_turn,
    rewrite_with_memory,
    serialize_memory,
    update_memory_with_results,
)
from rag.query_router import analyze_query, is_food_domain_query, is_greeting_query
from rag.retriever import TfidfRetriever
from rag.scope_guard import check_scope
from rag.types import AdaptiveRouteDecision, QueryAnalysis, RetrievalResult, WebSearchResult
from rag.web_search import TavilyWebSearch


class AdaptiveRAGPipeline:
    def __init__(self, retriever: TfidfRetriever):
        self.retriever = retriever
        self.agent = RetrievalAgent(retriever)
        self.template_generator = TemplateAnswerGenerator()
        self.ollama_generator = OllamaAnswerGenerator()
        self.web_search = TavilyWebSearch()
        self.chain = (
            RunnableLambda(self._prepare_state)
            | RunnableLambda(self._route)
            | RunnableLambda(self._retrieve)
            | RunnableLambda(self._synthesize)
        )

    def answer(
        self,
        question: str,
        top_k: int = 4,
        mode: str = "agentic",
        retrieval_strategy: str = "hybrid",
        generation_mode: str = "template",
        conversation_messages: list[dict] | None = None,
        search_mode: str = "adaptive",
    ) -> dict:
        payload = {
            "question": question,
            "top_k": top_k,
            "mode": mode,
            "retrieval_strategy": retrieval_strategy,
            "generation_mode": generation_mode,
            "conversation_messages": conversation_messages or [],
            "search_mode": search_mode,
        }
        return self.chain.invoke(payload)

    def _prepare_state(self, payload: dict) -> dict:
        question = str(payload["question"])
        memory = build_memory_before_current_turn(payload.get("conversation_messages", []), question)
        rewritten_question, memory_debug = rewrite_with_memory(question, memory)
        analysis = analyze_query(rewritten_question)
        scope_result = check_scope(rewritten_question, self.retriever.supported_locations)
        normalized_question = analysis.normalized_query
        guard_response = self._guard_non_retrieval_question(
            question=question,
            normalized_question=normalized_question,
            analysis=analysis,
        )

        payload.update(
            {
                "memory": memory,
                "memory_debug": memory_debug,
                "memory_snapshot": serialize_memory(memory),
                "rewritten_question": rewritten_question,
                "analysis": analysis,
                "scope_result": scope_result,
                "guard_response": guard_response,
            }
        )
        return payload

    def _route(self, payload: dict) -> dict:
        if payload.get("guard_response"):
            payload["route_decision"] = AdaptiveRouteDecision(
                route="guard",
                reason="non_retrieval_guard",
                local_confidence=0.0,
                used_memory=bool(payload["memory_debug"].get("used_memory")),
            )
            return payload

        question = str(payload["question"])
        rewritten_question = str(payload["rewritten_question"])
        search_mode = str(payload.get("search_mode", "adaptive"))
        scope_result = payload["scope_result"]
        analysis: QueryAnalysis = payload["analysis"]
        local_peek = self.retriever.search(
            rewritten_question,
            top_k=1,
            strategy=str(payload["retrieval_strategy"]),
        )
        local_confidence = local_peek[0].score if local_peek else 0.0

        if search_mode == "local_only":
            decision = AdaptiveRouteDecision(
                route="local",
                reason="user_forced_local_only",
                local_confidence=local_confidence,
                used_memory=bool(payload["memory_debug"].get("used_memory")),
            )
        elif search_mode == "web_only":
            decision = AdaptiveRouteDecision(
                route="web",
                reason="user_forced_web_only",
                local_confidence=local_confidence,
                used_memory=bool(payload["memory_debug"].get("used_memory")),
            )
        elif not scope_result.allowed:
            decision = AdaptiveRouteDecision(
                route="web",
                reason=f"scope_guard:{scope_result.reason}",
                local_confidence=local_confidence,
                used_memory=bool(payload["memory_debug"].get("used_memory")),
            )
        elif self._should_blend_local_and_web(question, rewritten_question, analysis, local_confidence):
            decision = AdaptiveRouteDecision(
                route="hybrid",
                reason="local_signal_weak_or_open_world_query",
                local_confidence=local_confidence,
                used_memory=bool(payload["memory_debug"].get("used_memory")),
            )
        else:
            decision = AdaptiveRouteDecision(
                route="local",
                reason="local_signal_strong",
                local_confidence=local_confidence,
                used_memory=bool(payload["memory_debug"].get("used_memory")),
            )

        payload["route_decision"] = decision
        return payload

    def _retrieve(self, payload: dict) -> dict:
        route_decision: AdaptiveRouteDecision = payload["route_decision"]
        if route_decision.route == "guard":
            payload.update(
                {
                    "agent_steps": [],
                    "local_results": [],
                    "web_results": [],
                    "web_error": "",
                }
            )
            return payload

        rewritten_question = str(payload["rewritten_question"])
        top_k = int(payload["top_k"])
        strategy = str(payload["retrieval_strategy"])
        mode = str(payload["mode"])

        local_results: list[RetrievalResult] = []
        analysis = payload["analysis"]
        agent_steps: list[dict] = []

        if route_decision.route in {"local", "hybrid"}:
            if mode == "agentic":
                agent_run = self.agent.run(rewritten_question, top_k=top_k, strategy=strategy)
                local_results = agent_run.results
                analysis = agent_run.analysis
                agent_steps = [
                    {
                        "name": step.name,
                        "detail": step.detail,
                        "payload": step.payload,
                    }
                    for step in agent_run.steps
                ]
            else:
                local_results = self.retriever.search(
                    rewritten_question,
                    top_k=top_k,
                    strategy=strategy,
                )

        web_results: list[WebSearchResult] = []
        web_error = ""
        if route_decision.route in {"web", "hybrid"}:
            try:
                web_results = self.web_search.search(rewritten_question, max_results=top_k)
            except Exception as exc:
                web_error = str(exc)

        payload.update(
            {
                "analysis": analysis,
                "agent_steps": agent_steps,
                "local_results": local_results,
                "web_results": web_results,
                "web_error": web_error,
            }
        )
        return payload

    def _synthesize(self, payload: dict) -> dict:
        if payload.get("guard_response"):
            return self._build_guard_response(payload)

        question = str(payload["question"])
        rewritten_question = str(payload["rewritten_question"])
        mode = str(payload["mode"])
        generation_mode = str(payload["generation_mode"])
        route_decision: AdaptiveRouteDecision = payload["route_decision"]
        local_results: list[RetrievalResult] = payload["local_results"]
        web_results: list[WebSearchResult] = payload["web_results"]
        analysis: QueryAnalysis = payload["analysis"]
        memory = payload["memory"]

        if route_decision.route == "local" and local_results:
            updated_memory = update_memory_with_results(memory, rewritten_question, local_results)
            prompt_preview = self._build_local_prompt(
                question=question,
                effective_question=rewritten_question,
                results=local_results,
                analysis=analysis,
                route_decision=route_decision,
            )
            answer, generation_debug = self._generate_local_answer(
                question=question,
                results=local_results,
                mode=mode,
                analysis=analysis,
                prompt_preview=prompt_preview,
                generation_mode=generation_mode,
            )
            return self._build_response(
                answer=answer,
                prompt_preview=prompt_preview,
                local_results=local_results,
                web_results=web_results,
                payload=payload,
                generation_debug=generation_debug,
                updated_memory=updated_memory,
            )

        if route_decision.route == "hybrid" and (local_results or web_results):
            updated_memory = update_memory_with_results(memory, rewritten_question, local_results)
            prompt_preview = self._build_hybrid_prompt(
                question=question,
                effective_question=rewritten_question,
                local_results=local_results,
                web_results=web_results,
                analysis=analysis,
                route_decision=route_decision,
            )
            answer = self._compose_hybrid_answer(local_results, web_results, question)
            return self._build_response(
                answer=answer,
                prompt_preview=prompt_preview,
                local_results=local_results,
                web_results=web_results,
                payload=payload,
                generation_debug={
                    "generator": "adaptive_hybrid_template",
                    "fallback_used": False,
                },
                updated_memory=updated_memory,
            )

        if route_decision.route == "web" and web_results:
            prompt_preview = self._build_web_prompt(
                question=question,
                effective_question=rewritten_question,
                web_results=web_results,
                route_decision=route_decision,
            )
            answer = self._compose_web_answer(web_results, question)
            return self._build_response(
                answer=answer,
                prompt_preview=prompt_preview,
                local_results=[],
                web_results=web_results,
                payload=payload,
                generation_debug={
                    "generator": "adaptive_web_template",
                    "fallback_used": False,
                },
                updated_memory=memory,
            )

        return self._build_empty_response(payload)

    def _generate_local_answer(
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

    def _build_response(
        self,
        answer: str,
        prompt_preview: str,
        local_results: list[RetrievalResult],
        web_results: list[WebSearchResult],
        payload: dict,
        generation_debug: dict,
        updated_memory,
    ) -> dict:
        route_decision: AdaptiveRouteDecision = payload["route_decision"]
        scope_result = payload["scope_result"]
        return {
            "answer": answer,
            "sources": [],
            "debug": {
                "route": {
                    "selected": route_decision.route,
                    "reason": route_decision.reason,
                    "local_confidence": round(route_decision.local_confidence, 4),
                    "used_memory": route_decision.used_memory,
                },
                "retrieved_chunks": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "title": item.chunk.document.title,
                        "score": round(item.score, 4),
                        "matched_terms": item.matched_terms,
                        "score_breakdown": item.score_breakdown,
                        "content": item.chunk.text,
                    }
                    for item in local_results
                ],
                "web_results": [
                    {
                        "title": item.title,
                        "snippet": item.snippet,
                        "url": item.url,
                    }
                    for item in web_results
                ],
                "prompt_preview": prompt_preview,
                "agent_steps": payload.get("agent_steps", []),
                "query_analysis": self._serialize_query_analysis(payload["analysis"]),
                "retrieval_strategy": payload["retrieval_strategy"],
                "generation_mode": payload["generation_mode"],
                "search_mode": payload["search_mode"],
                "generation_debug": generation_debug,
                "memory_debug": payload["memory_debug"],
                "memory_snapshot": serialize_memory(updated_memory),
                "scope_check": {
                    "allowed": scope_result.allowed,
                    "reason": scope_result.reason,
                    "matched_locations": scope_result.matched_locations,
                    "unsupported_locations": scope_result.unsupported_locations,
                },
                "web_error": payload.get("web_error", ""),
            },
        }

    def _build_empty_response(self, payload: dict) -> dict:
        route_decision: AdaptiveRouteDecision = payload["route_decision"]
        question = str(payload["question"])
        rewritten_question = str(payload["rewritten_question"])

        if route_decision.route == "web" and payload.get("web_error"):
            area_hint = self._describe_area_hint(payload["scope_result"].unsupported_locations, question)
            answer = (
                f"Minh da nhan ra ban dang hoi khu vuc {area_hint} va da chuyen sang tim kiem web, "
                "nhung hien chua lay duoc ket qua ben ngoai du lieu noi bo. "
                "Ban thu hoi cu the hon theo mon an hoac khu vuc nho hon de minh tim lai."
            )
        elif route_decision.route == "web":
            area_hint = self._describe_area_hint(payload["scope_result"].unsupported_locations, question)
            answer = (
                f"Minh da mo rong tim kiem ngoai bo du lieu cho khu vuc {area_hint} nhung chua thay ket qua du hop ly. "
                "Ban thu hoi cu the hon, vi du kieu 'bun bo o Son Tra, Da Nang' hoac 'hai san o Da Nang'."
            )
        elif route_decision.route == "local":
            answer = (
                "Minh chua tim thay noi dung phu hop trong bo du lieu noi bo. "
                "Ban co the hoi ro hon theo mon an, khu vuc hoac bat che do adaptive/web de mo rong pham vi tim."
            )
        else:
            answer = (
                "Minh da thu ca local va web nhung van chua du co so de tra loi hop ly. "
                "Ban thu bo sung ten dia diem, thanh pho hoac dac diem quan an cu the nhe."
            )

        return {
            "answer": answer,
            "sources": [],
            "debug": {
                "route": {
                    "selected": route_decision.route,
                    "reason": route_decision.reason,
                    "local_confidence": round(route_decision.local_confidence, 4),
                    "used_memory": route_decision.used_memory,
                },
                "retrieved_chunks": [],
                "web_results": [],
                "prompt_preview": self._build_web_prompt(
                    question=question,
                    effective_question=rewritten_question,
                    web_results=[],
                    route_decision=route_decision,
                ),
                "agent_steps": payload.get("agent_steps", []),
                "query_analysis": self._serialize_query_analysis(payload["analysis"]),
                "retrieval_strategy": payload["retrieval_strategy"],
                "generation_mode": payload["generation_mode"],
                "search_mode": payload["search_mode"],
                "generation_debug": {"generator": "none", "fallback_used": False},
                "memory_debug": payload["memory_debug"],
                "memory_snapshot": payload["memory_snapshot"],
                "scope_check": {
                    "allowed": payload["scope_result"].allowed,
                    "reason": payload["scope_result"].reason,
                    "matched_locations": payload["scope_result"].matched_locations,
                    "unsupported_locations": payload["scope_result"].unsupported_locations,
                },
                "web_error": payload.get("web_error", ""),
            },
        }

    def _build_guard_response(self, payload: dict) -> dict:
        route_decision: AdaptiveRouteDecision = payload["route_decision"]
        return {
            "answer": str(payload["guard_response"]),
            "sources": [],
            "debug": {
                "route": {
                    "selected": route_decision.route,
                    "reason": route_decision.reason,
                    "local_confidence": 0.0,
                    "used_memory": route_decision.used_memory,
                },
                "retrieved_chunks": [],
                "web_results": [],
                "prompt_preview": "",
                "agent_steps": [],
                "query_analysis": self._serialize_query_analysis(payload["analysis"]),
                "retrieval_strategy": payload["retrieval_strategy"],
                "generation_mode": payload["generation_mode"],
                "search_mode": payload["search_mode"],
                "generation_debug": {"generator": "guard", "fallback_used": False},
                "memory_debug": payload["memory_debug"],
                "memory_snapshot": payload["memory_snapshot"],
                "scope_check": {
                    "allowed": payload["scope_result"].allowed,
                    "reason": payload["scope_result"].reason,
                    "matched_locations": payload["scope_result"].matched_locations,
                    "unsupported_locations": payload["scope_result"].unsupported_locations,
                },
                "web_error": "",
            },
        }

    def _build_local_prompt(
        self,
        question: str,
        effective_question: str,
        results: list[RetrievalResult],
        analysis: QueryAnalysis | None,
        route_decision: AdaptiveRouteDecision,
    ) -> str:
        context_lines = []
        for item in results:
            document = item.chunk.document
            addresses = "; ".join(document.addresses) if document.addresses else "Khong co dia chi"
            context_lines.append(
                f"- {document.title} | {addresses} | {item.chunk.text}"
            )
        context_block = "\n".join(context_lines) if context_lines else "- Khong co ngu canh"
        analysis_text = self._describe_analysis(analysis) if analysis else "Khong co"
        return (
            "Adaptive RAG local route\n"
            f"Cau hoi goc: {question}\n"
            f"Cau hoi sau rewrite: {effective_question}\n"
            f"Ly do route: {route_decision.reason}\n"
            f"Phan tich truy van: {analysis_text}\n"
            f"Ngu canh local:\n{context_block}\n"
            "Tra loi ngan gon, uu tien de xuat quan va dia chi neu co."
        )

    def _build_web_prompt(
        self,
        question: str,
        effective_question: str,
        web_results: list[WebSearchResult],
        route_decision: AdaptiveRouteDecision,
    ) -> str:
        result_lines = [
            f"- {item.title} | {item.snippet} | {item.url}"
            for item in web_results
        ]
        result_block = "\n".join(result_lines) if result_lines else "- Khong co ket qua web"
        return (
            "Adaptive RAG web route\n"
            f"Cau hoi goc: {question}\n"
            f"Cau hoi sau rewrite: {effective_question}\n"
            f"Ly do route: {route_decision.reason}\n"
            f"Ket qua web:\n{result_block}"
        )

    def _build_hybrid_prompt(
        self,
        question: str,
        effective_question: str,
        local_results: list[RetrievalResult],
        web_results: list[WebSearchResult],
        analysis: QueryAnalysis | None,
        route_decision: AdaptiveRouteDecision,
    ) -> str:
        local_block = "\n".join(
            f"- {item.chunk.document.title} | {item.chunk.text}" for item in local_results
        ) or "- Khong co local result"
        web_block = "\n".join(
            f"- {item.title} | {item.snippet}" for item in web_results
        ) or "- Khong co web result"
        analysis_text = self._describe_analysis(analysis) if analysis else "Khong co"
        return (
            "Adaptive RAG hybrid route\n"
            f"Cau hoi goc: {question}\n"
            f"Cau hoi sau rewrite: {effective_question}\n"
            f"Ly do route: {route_decision.reason}\n"
            f"Phan tich truy van: {analysis_text}\n"
            f"Local context:\n{local_block}\n"
            f"Web context:\n{web_block}"
        )

    def _compose_web_answer(self, web_results: list[WebSearchResult], question: str) -> str:
        lines = [f"Minh da mo rong tim kiem web cho cau hoi '{question.strip()}'."]
        if web_results:
            top_results = web_results[:3]
            for item in top_results:
                line = f"- {item.title}: {item.snippet}"
                lines.append(line)
        lines.append("Neu ban muon, minh co the tiep tuc loc theo mon an, muc gia hoac khu vuc cu the hon.")
        return "\n\n".join([lines[0], "\n".join(lines[1:])])

    def _compose_hybrid_answer(
        self,
        local_results: list[RetrievalResult],
        web_results: list[WebSearchResult],
        question: str,
    ) -> str:
        paragraphs = [f"Minh da ket hop du lieu noi bo va tim kiem web cho cau hoi '{question.strip()}'."]

        if local_results:
            top_local = local_results[0].chunk.document
            text = f"Trong bo du lieu noi bo, goi y noi bat la {top_local.title}."
            if top_local.addresses:
                text += " Dia chi: " + "; ".join(top_local.addresses) + "."
            paragraphs.append(text)

        if web_results:
            web_lines = ["Ngoai bo du lieu, minh tim thay them:"]
            for item in web_results[:2]:
                line = f"- {item.title}: {item.snippet}"
                web_lines.append(line)
            paragraphs.append("\n".join(web_lines))

        paragraphs.append("Neu ban muon, minh co the chot lai theo tieu chi uu tien cua ban de de chon hon.")
        return "\n\n".join(paragraphs)

    def _should_blend_local_and_web(
        self,
        question: str,
        rewritten_question: str,
        analysis: QueryAnalysis,
        local_confidence: float,
    ) -> bool:
        normalized = rewritten_question.lower()
        if any(marker in normalized for marker in ["ha noi", "da nang", "can tho", "nha trang"]):
            return False
        if any(term in normalized for term in ["o dau", "dia chi", "khu vuc khac", "thanh pho khac"]):
            return local_confidence < 0.28
        if not analysis.location_terms and any(token in normalized for token in ["gan day", "khu vuc", "ngoai data"]):
            return True
        return local_confidence < 0.18

    def _describe_analysis(self, analysis: QueryAnalysis) -> str:
        parts = [
            f"category={', '.join(analysis.categories) or 'khong ro'}",
            f"mon={', '.join(analysis.cuisine_terms) or 'khong ro'}",
            f"vibe={', '.join(analysis.vibe_terms) or 'khong ro'}",
            f"dia_diem={', '.join(analysis.location_terms) or 'khong ro'}",
            f"intent={', '.join(analysis.intents) or 'khong ro'}",
        ]
        return "; ".join(parts)

    def _serialize_query_analysis(self, analysis: QueryAnalysis | None) -> dict | None:
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

    def _guard_non_retrieval_question(
        self,
        question: str,
        normalized_question: str,
        analysis: QueryAnalysis,
    ) -> str | None:
        if is_greeting_query(normalized_question):
            return (
                "Chao ban, minh la chatbot goi y am thuc. "
                "Ban co the hoi minh ve mon an, quan an, khu vuc hoac phong cach quan ma ban quan tam."
            )

        if not is_food_domain_query(analysis):
            return (
                "Minh dang tap trung vao bai toan goi y am thuc va dia diem an uong. "
                "Ban hay hoi theo kieu nhu 'quan nao o Binh Thanh', 'an toi o dau', hoac 'goi y mon bun bo'."
            )

        if (
            len(normalized_question.split()) <= 2
            and not analysis.location_terms
            and not analysis.cuisine_terms
            and not analysis.categories
            and not analysis.intents
        ):
            return (
                f"Minh chua du hieu cau hoi '{question.strip()}'. "
                "Ban hay noi ro them mon an, khu vuc hoac tieu chi ban muon tim de minh goi y chinh xac hon."
            )

        return None

    def _describe_area_hint(self, unsupported_locations: list[str], question: str) -> str:
        if unsupported_locations:
            return ", ".join(unsupported_locations)
        return question.strip() or "ban vua hoi"
