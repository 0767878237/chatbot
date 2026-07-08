from __future__ import annotations

import re
import unicodedata

from rag.types import QueryAnalysis


CATEGORY_SYNONYMS = {
    "lang_man": ["lang man", "hen ho", "bo song", "view dep", "romantic", "date"],
    "an_toi": ["an toi", "bua toi", "toi nay", "toi", "khuya"],
    "an_vat": ["an vat", "lot bung", "snack", "banh trang", "pha lau"],
    "lau_nuong": ["lau", "nuong", "bbq", "suon", "do nuong"],
    "gia_dinh": ["gia dinh", "com nha", "am cung", "nhieu nguoi", "me nau"],
    "tong_hop": ["quan ngon", "goi y", "nen thu", "noi bat", "mon ngon"],
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
    r"tan binh",
    r"tan phu",
    r"nguyen trai",
    r"xo viet nghe tinh",
    r"bui vien",
    r"tphcm",
    r"tp hcm",
    r"ho chi minh",
    r"ha noi",
    r"da nang",
    r"son tra",
]

GREETING_PATTERNS = [
    "xin chao",
    "chao ban",
    "chao",
    "hello",
    "hi",
    "hey",
]

FOOD_DOMAIN_HINTS = [
    "quan",
    "quan an",
    "nha hang",
    "mon",
    "mon ngon",
    "an",
    "uong",
    "am thuc",
    "goi y",
    "o dau",
    "dia chi",
    "quan nao",
]


def normalize_text(text: str) -> str:
    lowered = unicodedata.normalize("NFKC", text.lower()).replace("đ", "d")
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(character)
    )
    cleaned = re.sub(r"[^0-9a-z\s]", " ", without_accents)
    return re.sub(r"\s+", " ", cleaned).strip()


def analyze_query(question: str) -> QueryAnalysis:
    normalized_query = normalize_text(question)
    categories = [
        category
        for category, synonyms in CATEGORY_SYNONYMS.items()
        if any(contains_phrase(normalized_query, term) for term in synonyms)
    ]

    cuisine_terms = [term for term in CUISINE_KEYWORDS if matches_cuisine_term(normalized_query, term)]
    vibe_terms = [term for term in VIBE_KEYWORDS if contains_phrase(normalized_query, term)]
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
    if any(contains_phrase(normalized_query, token) for token in ["o dau", "quan nao", "goi y", "nen an", "mon ngon"]):
        intents.append("recommend")
    if any(contains_phrase(normalized_query, token) for token in ["so sanh", "khac nhau", "chon"]):
        intents.append("compare")
    if any(contains_phrase(normalized_query, token) for token in ["gan", "khu vuc", "dia chi", "duong"]):
        intents.append("locate")
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


def is_greeting_query(normalized_query: str) -> bool:
    return any(
        normalized_query == pattern or normalized_query.startswith(pattern + " ")
        for pattern in GREETING_PATTERNS
    )


def is_food_domain_query(analysis: QueryAnalysis) -> bool:
    if analysis.cuisine_terms or analysis.vibe_terms or analysis.location_terms:
        return True
    if analysis.categories or analysis.intents:
        return True
    return any(contains_phrase(analysis.normalized_query, hint) for hint in FOOD_DOMAIN_HINTS)


def contains_phrase(normalized_query: str, phrase: str) -> bool:
    escaped = re.escape(phrase)
    pattern = r"(?<![0-9a-z])" + escaped.replace(r"\ ", r"\s+") + r"(?![0-9a-z])"
    return re.search(pattern, normalized_query) is not None


def matches_cuisine_term(normalized_query: str, term: str) -> bool:
    if not contains_phrase(normalized_query, term):
        return False
    if term != "chao":
        return True
    if is_greeting_query(normalized_query):
        return False
    return any(
        contains_phrase(normalized_query, token)
        for token in ["mon", "an", "quan", "goi y", "tim", "dia chi"]
    )
