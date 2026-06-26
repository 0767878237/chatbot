from __future__ import annotations

import json
import re
from pathlib import Path

from rag.types import Chunk, Document


ENTRY_PATTERN = re.compile(r"^\s*(\d+)\.\s*(.+)$", re.MULTILINE)
CATEGORY_KEYWORDS = {
    "lang_man": ["lãng mạn", "bờ sông", "hiện đại"],
    "an_toi": ["ăn tối", "buổi tối", "đêm sài gòn"],
    "an_vat": ["ăn vặt", "lót bụng", "bánh tráng"],
    "lau_nuong": ["lẩu", "nướng", "sườn nướng"],
    "gia_dinh": ["gia đình", "cơm mẹ nấu", "ấm áp"],
}


def load_documents(data_dir: str | Path) -> list[Document]:
    data_path = Path(data_dir)
    documents: list[Document] = []

    for txt_file in sorted(data_path.glob("*.txt")):
        raw_text = txt_file.read_text(encoding="utf-8")
        documents.extend(parse_documents(raw_text, source_name=txt_file.stem))

    return documents


def parse_documents(raw_text: str, source_name: str) -> list[Document]:
    matches = list(ENTRY_PATTERN.finditer(raw_text))
    documents: list[Document] = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        chunk = raw_text[start:end].strip()
        title, addresses, content = parse_entry(chunk)
        category = infer_category(f"{title}\n{content}")
        doc_id = f"{source_name}-{index + 1}"
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                addresses=addresses,
                content=content,
                category=category,
            )
        )

    return documents


def parse_entry(entry_text: str) -> tuple[str, list[str], str]:
    lines = [line.strip() for line in entry_text.splitlines() if line.strip()]
    title_line = re.sub(r"^\d+\.\s*", "", lines[0]).strip()

    addresses: list[str] = []
    description_lines: list[str] = []

    for line in lines[1:]:
        if line.lower().startswith("địa chỉ:"):
            addresses.append(line.split(":", 1)[1].strip())
        else:
            description_lines.append(line)

    content = " ".join(description_lines)
    return title_line, addresses, content


def infer_category(text: str) -> str:
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "tong_hop"


def serialize_documents(documents: list[Document], output_path: str | Path) -> None:
    payload = [
        {
            "doc_id": document.doc_id,
            "title": document.title,
            "addresses": document.addresses,
            "content": document.content,
            "category": document.category,
        }
        for document in documents
    ]
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def chunk_documents(
    documents: list[Document],
    max_sentences: int = 2,
    sentence_overlap: int = 1,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        sentences = split_sentences(document.content)
        if not sentences:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-chunk-1",
                    document=document,
                    text=build_chunk_text(document, document.content),
                    order=1,
                )
            )
            continue

        step = max(1, max_sentences - sentence_overlap)
        chunk_order = 1
        for start in range(0, len(sentences), step):
            sentence_group = sentences[start : start + max_sentences]
            if not sentence_group:
                continue

            chunk_body = " ".join(sentence_group).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-chunk-{chunk_order}",
                    document=document,
                    text=build_chunk_text(document, chunk_body),
                    order=chunk_order,
                )
            )
            chunk_order += 1

    return chunks


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def build_chunk_text(document: Document, chunk_body: str) -> str:
    address_text = "; ".join(document.addresses)
    prefix_parts = [document.title]
    if address_text:
        prefix_parts.append(f"Địa chỉ: {address_text}")
    prefix_parts.append(f"Chủ đề: {document.category}")
    prefix = ". ".join(prefix_parts)
    return f"{prefix}. {chunk_body}".strip()


def serialize_chunks(chunks: list[Chunk], output_path: str | Path) -> None:
    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.document.doc_id,
            "title": chunk.document.title,
            "addresses": chunk.document.addresses,
            "category": chunk.document.category,
            "order": chunk.order,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
