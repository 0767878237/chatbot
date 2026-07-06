from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.types import Chunk, RetrievalResult


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.corpus = [chunk.text for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def search(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]

        query_terms = normalize_terms(query)
        results: list[RetrievalResult] = []

        for index in ranked_indices[:top_k]:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self.chunks[index]
            matched_terms = [
                term for term in query_terms if term in normalize_text(self.corpus[index])
            ]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    matched_terms=matched_terms,
                )
            )

        return results

    def search_filtered(
        self,
        query: str,
        top_k: int = 4,
        category_filters: list[str] | None = None,
        location_terms: list[str] | None = None,
    ) -> list[RetrievalResult]:
        base_results = self.search(query, top_k=max(top_k * 3, top_k))
        filtered: list[RetrievalResult] = []

        for item in base_results:
            document = item.chunk.document
            if category_filters and document.category not in category_filters:
                continue

            if location_terms:
                haystack = normalize_text(
                    " ".join([document.title, document.content, " ".join(document.addresses)])
                )
                if not any(term in haystack for term in location_terms):
                    continue

            filtered.append(item)
            if len(filtered) >= top_k:
                break

        return filtered or base_results[:top_k]

    def lexical_overlap_score(self, query: str, text: str) -> float:
        query_terms = set(normalize_terms(query))
        text_terms = set(normalize_terms(text))
        if not query_terms or not text_terms:
            return 0.0
        return len(query_terms & text_terms) / max(len(query_terms), 1)

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        chunks_payload = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.document.doc_id,
                "title": chunk.document.title,
                "addresses": chunk.document.addresses,
                "category": chunk.document.category,
                "order": chunk.order,
                "text": chunk.text,
            }
            for chunk in self.chunks
        ]
        (output_path / "chunks.json").write_text(
            json.dumps(chunks_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "vocabulary.json").write_text(
            json.dumps(self.vectorizer.vocabulary_, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def normalize_text(text: str) -> str:
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_terms(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [term for term in re.split(r"[^\wàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+", normalized) if len(term) > 1]
