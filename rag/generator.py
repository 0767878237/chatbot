from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from rag.types import QueryAnalysis, RetrievalResult


class AnswerGenerator:
    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
        mode: str,
        prompt: str,
        analysis: QueryAnalysis | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError


class TemplateAnswerGenerator(AnswerGenerator):
    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
        mode: str,
        prompt: str,
        analysis: QueryAnalysis | None = None,
    ) -> tuple[str, dict]:
        top_chunk = results[0].chunk
        top_document = top_chunk.document
        recommendation_lines = [
            f"Neu ban dang hoi ve '{question.strip()}', minh goi y ban bat dau voi {top_document.title}.",
        ]

        if top_document.addresses:
            recommendation_lines.append(
                "Dia chi noi bat: " + "; ".join(top_document.addresses) + "."
            )

        recommendation_lines.append("Doan ngu canh phu hop nhat:\n" + top_chunk.text)

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
            recommendation_lines.append(f"Ban cung co the tham khao them: {alternative_titles}.")

        if mode == "agentic" and analysis is not None:
            recommendation_lines.append(
                "Agent da thu nhieu bien the truy van, loc theo y dinh va chon lai cac chunk phu hop nhat truoc khi tra loi."
            )
        else:
            recommendation_lines.append(
                "Cau tra loi nay dang duoc tao theo dang template tu cac doan du lieu truy xuat duoc de ban de quan sat flow RAG."
            )

        return "\n\n".join(recommendation_lines), {
            "generator": "template",
            "fallback_used": False,
        }


class OllamaAnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "gemma3:1b"
        self.timeout_seconds = timeout_seconds
        self.template_generator = TemplateAnswerGenerator()

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
        mode: str,
        prompt: str,
        analysis: QueryAnalysis | None = None,
    ) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ban la chatbot am thuc TP.HCM. "
                        "Chi duoc tra loi dua tren context duoc cung cap. "
                        "Neu thieu du lieu, hay noi ro la chua co thong tin trong bo du lieu."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                message = body.get("message", {})
                answer = str(message.get("content", "")).strip()
                if not answer:
                    raise ValueError("Empty Ollama response")
                return answer, {
                    "generator": "ollama",
                    "model": self.model,
                    "fallback_used": False,
                }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            fallback_answer, fallback_meta = self.template_generator.generate(
                question=question,
                results=results,
                mode=mode,
                prompt=prompt,
                analysis=analysis,
            )
            fallback_meta.update(
                {
                    "generator": "template",
                    "fallback_used": True,
                    "fallback_reason": str(exc),
                    "requested_generator": "ollama",
                    "model": self.model,
                }
            )
            return fallback_answer, fallback_meta
