from __future__ import annotations

from pathlib import Path

from rag.ingest import chunk_documents, load_documents, serialize_chunks, serialize_documents
from rag.retriever import TfidfRetriever


def build_retriever(data_dir: str = "data", artifacts_dir: str = "artifacts") -> TfidfRetriever:
    documents = load_documents(data_dir)
    chunks = chunk_documents(documents)
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    serialize_documents(documents, artifacts_path / "documents.normalized.json")
    serialize_chunks(chunks, artifacts_path / "chunks.normalized.json")
    retriever = TfidfRetriever(chunks)
    retriever.save(artifacts_path)
    return retriever
