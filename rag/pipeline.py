from __future__ import annotations

from pathlib import Path

from rag.ingest import chunk_documents, load_documents, serialize_chunks, serialize_documents
from rag.retriever import TfidfRetriever
from rag.scope_guard import build_supported_location_terms


def build_retriever(
    data_dir: str = "data",
    artifacts_dir: str = "artifacts",
    persist_artifacts: bool = True,
) -> TfidfRetriever:
    documents = load_documents(data_dir)
    chunks = chunk_documents(documents)
    retriever = TfidfRetriever(chunks)
    retriever.supported_locations = build_supported_location_terms(
        [
            {
                "title": document.title,
                "content": document.content,
                "addresses": document.addresses,
            }
            for document in documents
        ]
    )

    if persist_artifacts:
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        serialize_documents(documents, artifacts_path / "documents.normalized.json")
        serialize_chunks(chunks, artifacts_path / "chunks.normalized.json")
        retriever.save(artifacts_path)

    return retriever
