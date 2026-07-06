from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.types import Chunk, RetrievalResult


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk], semantic_weight: float = 0.35):
        self.chunks = chunks
        self.semantic_weight = semantic_weight
        self.supported_locations: set[str] = set()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.corpus = [chunk.text for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(self.corpus)

        semantic_components = max(2, min(64, self.matrix.shape[0] - 1, self.matrix.shape[1] - 1))
        if semantic_components >= 2:
            self.svd = TruncatedSVD(n_components=semantic_components, random_state=42)
            self.semantic_matrix = self.svd.fit_transform(self.matrix)
        else:
            self.svd = None
            self.semantic_matrix = None

    def search(self, query: str, top_k: int = 4, strategy: str = "hybrid") -> list[RetrievalResult]:
        return self._search_internal(query=query, top_k=top_k, strategy=strategy)

    def search_filtered(
        self,
        query: str,
        top_k: int = 4,
        category_filters: list[str] | None = None,
        location_terms: list[str] | None = None,
        strategy: str = "hybrid",
    ) -> list[RetrievalResult]:
        base_results = self.search(query, top_k=max(top_k * 3, top_k), strategy=strategy)
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

    def _search_internal(self, query: str, top_k: int, strategy: str) -> list[RetrievalResult]:
        query_vector = self.vectorizer.transform([query])
        lexical_scores = cosine_similarity(query_vector, self.matrix).flatten()
        semantic_scores = self._semantic_scores(query_vector)
        hybrid_scores = self._combine_scores(lexical_scores, semantic_scores, strategy)
        ranked_indices = np.argsort(hybrid_scores)[::-1]

        query_terms = normalize_terms(query)
        results: list[RetrievalResult] = []

        for index in ranked_indices[:top_k]:
            score = float(hybrid_scores[index])
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
                    score_breakdown={
                        "hybrid": round(float(hybrid_scores[index]), 4),
                        "lexical": round(float(lexical_scores[index]), 4),
                        "semantic": round(float(semantic_scores[index]), 4),
                    },
                )
            )

        return results

    def _semantic_scores(self, query_vector) -> np.ndarray:
        if self.svd is None or self.semantic_matrix is None:
            return np.zeros(self.matrix.shape[0], dtype=float)
        query_semantic = self.svd.transform(query_vector)
        return cosine_similarity(query_semantic, self.semantic_matrix).flatten()

    def _combine_scores(
        self,
        lexical_scores: np.ndarray,
        semantic_scores: np.ndarray,
        strategy: str,
    ) -> np.ndarray:
        if strategy == "lexical":
            return lexical_scores
        if strategy == "semantic":
            return semantic_scores
        return ((1 - self.semantic_weight) * lexical_scores) + (
            self.semantic_weight * semantic_scores
        )


def normalize_text(text: str) -> str:
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_terms(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [
        term
        for term in re.split(
            r"[^\wÃ Ã¡áº£Ã£áº¡Äƒáº¯áº±áº³áºµáº·Ã¢áº¥áº§áº©áº«áº­Ã¨Ã©áº»áº½áº¹Ãªáº¿á»á»ƒá»…á»‡Ã¬Ã­á»‰Ä©á»‹Ã²Ã³á»Ãµá»Ã´á»‘á»“á»•á»—á»™Æ¡á»›á»á»Ÿá»¡á»£Ã¹Ãºá»§Å©á»¥Æ°á»©á»«á»­á»¯á»±á»³Ã½á»·á»¹á»µÄ‘]+",
            normalized,
        )
        if len(term) > 1
    ]
