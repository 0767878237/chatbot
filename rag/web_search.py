from __future__ import annotations

from ddgs import DDGS

from rag.types import WebSearchResult


class DuckDuckGoWebSearch:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str, max_results: int | None = None) -> list[WebSearchResult]:
        limit = max_results or self.max_results
        results: list[WebSearchResult] = []

        with DDGS() as ddgs:
            items = ddgs.text(
                keywords=query,
                max_results=limit,
                safesearch="moderate",
            )
            for item in items:
                title = str(item.get("title", "")).strip()
                snippet = str(item.get("body", "")).strip()
                url = str(item.get("href", "")).strip()
                if not (title or snippet):
                    continue
                results.append(
                    WebSearchResult(
                        title=title or "Khong ro tieu de",
                        snippet=snippet,
                        url=url,
                    )
                )

        return results
