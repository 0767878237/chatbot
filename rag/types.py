from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    title: str
    addresses: list[str]
    content: str
    category: str


@dataclass
class Chunk:
    chunk_id: str
    document: Document
    text: str
    order: int


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    matched_terms: list[str]


@dataclass
class QueryAnalysis:
    normalized_query: str
    categories: list[str] = field(default_factory=list)
    location_terms: list[str] = field(default_factory=list)
    cuisine_terms: list[str] = field(default_factory=list)
    vibe_terms: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)


@dataclass
class AgentStep:
    name: str
    detail: str
    payload: dict[str, object] = field(default_factory=dict)
