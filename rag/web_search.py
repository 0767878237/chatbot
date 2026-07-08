from __future__ import annotations

import os
from pathlib import Path

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - depends on optional runtime package
    TavilyClient = None

from rag.query_router import normalize_text
from rag.types import WebSearchResult


class TavilyWebSearch:
    def __init__(self, max_results: int = 5, api_key: str | None = None):
        self.max_results = max_results
        self.api_key = api_key or _load_tavily_api_key(required=False)
        self.client = self._build_client()

    def search(self, query: str, max_results: int | None = None) -> list[WebSearchResult]:
        if self.client is None:
            raise RuntimeError(_build_unavailable_reason(self.api_key))

        limit = max_results or self.max_results
        errors: list[str] = []

        for candidate_query in self._build_query_candidates(query):
            try:
                results = self._search_once(candidate_query, limit)
            except Exception as exc:
                errors.append(f"{candidate_query}: {exc}")
                continue
            if results:
                return results

        if errors:
            raise RuntimeError("Tavily search failed. " + " | ".join(errors[:3]))

        return []

    def _build_client(self):
        if TavilyClient is None or not self.api_key:
            return None
        return TavilyClient(api_key=self.api_key)

    def _search_once(self, query: str, limit: int) -> list[WebSearchResult]:
        response = self.client.search(
            query=query,
            topic="general",
            search_depth="advanced",
            max_results=limit,
            include_answer=False,
            include_raw_content=False,
        )

        items = response.get("results", [])
        results: list[WebSearchResult] = []
        for item in items:
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("content", "")).strip()
            url = str(item.get("url", "")).strip()
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

    def _build_query_candidates(self, query: str) -> list[str]:
        normalized = normalize_text(query)
        candidates = [
            query.strip(),
            f"{query.strip()} quan an",
            f"{query.strip()} am thuc",
        ]

        if "mon ngon" in normalized:
            candidates.append(f"{query.strip()} dia diem an uong")
        elif not any(term in normalized for term in ["quan", "nha hang", "am thuc", "an", "uong"]):
            candidates.append(f"{query.strip()} mon ngon")

        deduped: list[str] = []
        for candidate in candidates:
            cleaned = candidate.strip()
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        return deduped


def _load_tavily_api_key(required: bool = True) -> str:
    env_value = os.getenv("TAVILY_API_KEY", "").strip().strip('"').strip("'")
    if env_value:
        return env_value

    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "TAVILY_API_KEY":
                cleaned = value.strip().strip('"').strip("'")
                if cleaned:
                    return cleaned

    if required:
        raise RuntimeError(_build_unavailable_reason(""))
    return ""


def _build_unavailable_reason(api_key: str | None) -> str:
    if TavilyClient is None:
        return "Chua cai dat goi 'tavily-python', nen khong the tim web bang Tavily."
    if not api_key:
        return "TAVILY_API_KEY chua duoc cau hinh. Hay them khoa vao bien moi truong hoac file .env."
    return "Tavily search tam thoi khong kha dung."
