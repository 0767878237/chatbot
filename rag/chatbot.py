from __future__ import annotations

from rag.retriever import TfidfRetriever
from rag.types import RetrievalResult


class FoodChatbot:
    def __init__(self, retriever: TfidfRetriever):
        self.retriever = retriever

    def answer(self, question: str, top_k: int = 4) -> dict:
        results = self.retriever.search(question, top_k=top_k)
        if not results:
            return {
                "answer": (
                    "Mình chưa tìm thấy thông tin phù hợp trong bộ dữ liệu hiện tại. "
                    "Bạn hãy thử hỏi rõ hơn về món ăn, kiểu quán hoặc tên địa điểm ở TP.HCM."
                ),
                "sources": [],
                "debug": {
                    "retrieved_chunks": [],
                    "prompt_preview": self.build_prompt(question, []),
                },
            }

        answer = self.build_answer(question, results)
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
                "prompt_preview": self.build_prompt(question, results),
            },
        }

    def build_answer(self, question: str, results: list[RetrievalResult]) -> str:
        top_chunk = results[0].chunk
        top_document = top_chunk.document
        recommendation_lines = [
            f"Nếu bạn đang hỏi về “{question.strip()}”, mình gợi ý bạn bắt đầu với {top_document.title}.",
        ]

        if top_document.addresses:
            recommendation_lines.append(
                "Địa chỉ nổi bật: " + "; ".join(top_document.addresses) + "."
            )

        recommendation_lines.append(
            "Đoạn ngữ cảnh phù hợp nhất:\n" + top_chunk.text
        )

        companion_chunks = [
            item.chunk for item in results[1:] if item.chunk.document.doc_id == top_document.doc_id
        ]
        if companion_chunks:
            recommendation_lines.append(
                "Thông tin bổ sung cùng quán:\n"
                + "\n".join(f"- {chunk.text}" for chunk in companion_chunks[:2])
            )

        if len(results) > 1:
            alternative_titles = ", ".join(
                list(dict.fromkeys(item.chunk.document.title for item in results[1:]))
            )
            recommendation_lines.append(
                f"Bạn cũng có thể tham khảo thêm: {alternative_titles}."
            )

        recommendation_lines.append(
            "Câu trả lời này đang được tạo theo dạng template từ các đoạn dữ liệu truy xuất được, để bạn dễ quan sát flow RAG."
        )
        return "\n\n".join(recommendation_lines)

    def build_prompt(self, question: str, results: list[RetrievalResult]) -> str:
        context_lines = []
        for item in results:
            document = item.chunk.document
            addresses = "; ".join(document.addresses) if document.addresses else "Không có địa chỉ"
            context_lines.append(
                f"- {item.chunk.chunk_id} | {document.title} | {addresses} | {item.chunk.text}"
            )

        context_block = "\n".join(context_lines) if context_lines else "- Không có ngữ cảnh phù hợp"
        return (
            "Bạn là chatbot ẩm thực TP.HCM.\n"
            "Chỉ trả lời dựa trên ngữ cảnh đã truy xuất.\n"
            f"Câu hỏi: {question}\n"
            f"Ngữ cảnh:\n{context_block}\n"
            "Nếu thiếu dữ liệu thì nói rõ là chưa có thông tin trong bộ dữ liệu."
        )
