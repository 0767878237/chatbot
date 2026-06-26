from dataclasses import dataclass


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
