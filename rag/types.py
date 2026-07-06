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
    score_breakdown: dict[str, float] = field(default_factory=dict)


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


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    turns: list[ConversationTurn] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    cuisines: list[str] = field(default_factory=list)
    vibes: list[str] = field(default_factory=list)
    last_recommendations: list[str] = field(default_factory=list)
    last_user_query: str = ""


@dataclass
class ScopeCheckResult:
    allowed: bool
    reason: str
    matched_locations: list[str] = field(default_factory=list)
    unsupported_locations: list[str] = field(default_factory=list)


@dataclass
class WebSearchResult:
    title: str
    snippet: str
    url: str


@dataclass
class AdaptiveRouteDecision:
    route: str
    reason: str
    local_confidence: float = 0.0
    used_memory: bool = False
