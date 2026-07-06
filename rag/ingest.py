from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from rag.types import Chunk, Document


ENTRY_PATTERN = re.compile(r"^\s*(\d+)\.\s*(.+)$", re.MULTILINE)
CATEGORY_KEYWORDS = {
    "lang_man": ["lang man", "bo song", "view dep", "romantic", "hen ho"],
    "an_toi": ["an toi", "bua toi", "dem sai gon", "toi"],
    "an_vat": ["an vat", "lot bung", "banh trang", "snack", "pha lau"],
    "lau_nuong": ["lau", "nuong", "suon nuong", "bbq"],
    "gia_dinh": ["gia dinh", "com me nau", "am ap", "nhieu nguoi"],
}


def load_documents(data_dir: str | Path) -> list[Document]:
    data_path = Path(data_dir)
    documents: list[Document] = []

    for txt_file in sorted(data_path.glob("*.txt")):
        raw_text = txt_file.read_text(encoding="utf-8")
        documents.extend(parse_documents(raw_text, source_name=txt_file.stem))

    for csv_file in sorted(data_path.glob("*.csv")):
        documents.extend(load_csv_documents(csv_file))

    for json_file in sorted(data_path.glob("*.json")):
        documents.extend(load_json_documents(json_file))

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


def load_csv_documents(csv_file: Path) -> list[Document]:
    documents: list[Document] = []
    with csv_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            title = clean_field(row.get("title") or row.get("name"))
            content = clean_field(row.get("content") or row.get("description"))
            if not title or not content:
                continue

            addresses = parse_addresses(row.get("addresses") or row.get("address"))
            explicit_category = clean_field(row.get("category"))
            category = explicit_category or infer_category(f"{title}\n{content}")
            doc_id = clean_field(row.get("doc_id")) or f"{csv_file.stem}-{index}"
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


def load_json_documents(json_file: Path) -> list[Document]:
    payload = json.loads(json_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("items", [])
    else:
        records = payload

    documents: list[Document] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue

        question = clean_field(record.get("question"))
        answer = clean_field(record.get("answer") or record.get("content"))
        title = clean_field(record.get("title")) or question
        content = answer or clean_field(record.get("description"))
        if not title or not content:
            continue

        addresses = parse_addresses(record.get("addresses") or record.get("address"))
        explicit_category = clean_field(record.get("category"))
        combined_text = " ".join(part for part in [title, question, content] if part)
        category = explicit_category or infer_category(combined_text)
        doc_id = clean_field(record.get("doc_id")) or f"{json_file.stem}-{index}"
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
        normalized_line = line.lower()
        if normalized_line.startswith("dia chi:") or normalized_line.startswith("địa chỉ:"):
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
        prefix_parts.append(f"Dia chi: {address_text}")
    prefix_parts.append(f"Chu de: {document.category}")
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


def parse_addresses(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_field(item) for item in value if clean_field(item)]
    raw = clean_field(value)
    if not raw:
        return []
    separators = ["|", ";", "\n"]
    addresses = [raw]
    for separator in separators:
        next_addresses: list[str] = []
        for item in addresses:
            next_addresses.extend(item.split(separator))
        addresses = next_addresses
    return [item.strip() for item in addresses if item.strip()]


def clean_field(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
