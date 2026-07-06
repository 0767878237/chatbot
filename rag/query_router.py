from __future__ import annotations

import re

from rag.types import QueryAnalysis


CATEGORY_SYNONYMS = {
    "lang_man": ["lang man", "hen ho", "bo song", "view dep", "romantic", "date"],
    "an_toi": ["an toi", "bua toi", "toi nay", "toi", "khuya"],
    "an_vat": ["an vat", "lot bung", "snack", "banh trang", "pha lau"],
    "lau_nuong": ["lau", "nuong", "bbq", "suon", "do nuong"],
    "gia_dinh": ["gia dinh", "com nha", "am cung", "nhieu nguoi", "me nau"],
    "tong_hop": ["quan ngon", "goi y", "nen thu", "noi bat"],
}

CUISINE_KEYWORDS = [
    "chao",
    "bun mam",
    "bun bo",
    "com tam",
    "mi cay",
    "lau",
    "nuong",
    "ga",
    "bo",
    "banh trang",
    "spaghetti",
    "sup cua",
    "banh canh",
    "pha lau",
]

VIBE_KEYWORDS = [
    "lang man",
    "view dep",
    "gia dinh",
    "nhau",
    "sang trong",
    "am ap",
    "yen tinh",
    "thoang mat",
    "re",
]

LOCATION_PATTERNS = [
    r"quan\s+\d+",
    r"thu duc",
    r"binh thanh",
    r"go vap",
    r"phu nhuan",
    r"quan \d+",
    r"nguyen trai",
    r"xo viet nghe tinh",
    r"bui vien",
]


def normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^0-9a-zA-Z\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def analyze_query(question: str) -> QueryAnalysis:
    normalized_query = normalize_text(question)
    categories = [
        category
        for category, synonyms in CATEGORY_SYNONYMS.items()
        if any(term in normalized_query for term in synonyms)
    ]
    if not categories:
        categories = ["tong_hop"]

    cuisine_terms = [term for term in CUISINE_KEYWORDS if term in normalized_query]
    vibe_terms = [term for term in VIBE_KEYWORDS if term in normalized_query]
    location_terms = sorted(
        {
            match.group(0)
            for pattern in LOCATION_PATTERNS
            for match in re.finditer(pattern, normalized_query)
        }
    )

    intents = infer_intents(normalized_query)
    query_variants = build_query_variants(
        normalized_query=normalized_query,
        categories=categories,
        cuisine_terms=cuisine_terms,
        vibe_terms=vibe_terms,
        location_terms=location_terms,
    )

    return QueryAnalysis(
        normalized_query=normalized_query,
        categories=categories,
        location_terms=location_terms,
        cuisine_terms=cuisine_terms,
        vibe_terms=vibe_terms,
        intents=intents,
        query_variants=query_variants,
    )


def infer_intents(normalized_query: str) -> list[str]:
    intents: list[str] = []
    if any(token in normalized_query for token in ["o dau", "quan nao", "goi y", "nen an"]):
        intents.append("recommend")
    if any(token in normalized_query for token in ["so sanh", "khac nhau", "chon"]):
        intents.append("compare")
    if any(token in normalized_query for token in ["gan", "khu vuc", "dia chi", "duong"]):
        intents.append("locate")
    if not intents:
        intents.append("recommend")
    return intents


def build_query_variants(
    normalized_query: str,
    categories: list[str],
    cuisine_terms: list[str],
    vibe_terms: list[str],
    location_terms: list[str],
) -> list[str]:
    variants = [normalized_query]

    if cuisine_terms:
        variants.append(" ".join(cuisine_terms))
    if vibe_terms:
        variants.append(" ".join(vibe_terms))
    if location_terms:
        variants.append(" ".join(location_terms))
    if categories:
        variants.append(" ".join(categories))

    hybrid_variant = " ".join(
        part for part in [normalized_query, " ".join(cuisine_terms), " ".join(vibe_terms)] if part
    ).strip()
    if hybrid_variant:
        variants.append(hybrid_variant)

    deduped: list[str] = []
    for item in variants:
        if item and item not in deduped:
            deduped.append(item)
    return deduped
