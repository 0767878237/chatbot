from __future__ import annotations

from rag.query_router import normalize_text
from rag.types import ScopeCheckResult


SUPPORTED_HCMC_MARKERS = [
    "tp hcm",
    "tp.hcm",
    "hcm",
    "sai gon",
    "tphcm",
    "ho chi minh",
]

UNSUPPORTED_LOCATION_MARKERS = [
    "ha noi",
    "da nang",
    "son tra",
    "hai chau",
    "ngu hanh son",
    "can tho",
    "nha trang",
    "vung tau",
    "bien hoa",
    "binh duong",
    "dong nai",
    "hue",
    "hai phong",
]


def build_supported_location_terms(documents_payload: list[dict]) -> set[str]:
    supported: set[str] = set()
    for item in documents_payload:
        title = normalize_text(str(item.get("title", "")))
        content = normalize_text(str(item.get("content", "")))
        supported.update(extract_known_locations(title))
        supported.update(extract_known_locations(content))
        for address in item.get("addresses", []):
            supported.update(extract_known_locations(normalize_text(str(address))))
    return {term for term in supported if term}


def check_scope(question: str, supported_locations: set[str]) -> ScopeCheckResult:
    normalized = normalize_text(question)
    unsupported_locations = [
        marker for marker in UNSUPPORTED_LOCATION_MARKERS if marker in normalized
    ]
    if unsupported_locations:
        return ScopeCheckResult(
            allowed=False,
            reason="outside_supported_city",
            unsupported_locations=unsupported_locations,
        )

    requested_locations = extract_known_locations(normalized)
    matched_locations = sorted(location for location in requested_locations if location in supported_locations)
    unresolved_locations = sorted(
        location
        for location in requested_locations
        if location not in supported_locations and location not in SUPPORTED_HCMC_MARKERS
    )

    if unresolved_locations:
        return ScopeCheckResult(
            allowed=False,
            reason="location_not_in_dataset",
            matched_locations=matched_locations,
            unsupported_locations=unresolved_locations,
        )

    return ScopeCheckResult(
        allowed=True,
        reason="allowed",
        matched_locations=matched_locations,
        unsupported_locations=[],
    )


def extract_known_locations(normalized_text_value: str) -> set[str]:
    candidates = {
        "quan 1",
        "quan 3",
        "quan 5",
        "quan 7",
        "quan 10",
        "binh thanh",
        "go vap",
        "phu nhuan",
        "thu duc",
        "tan binh",
        "tan phu",
        "nguyen trai",
        "xo viet nghe tinh",
        "bui vien",
        "nguyen hue",
        "son tra",
        "hai chau",
        "ngu hanh son",
    }
    return {candidate for candidate in candidates if candidate in normalized_text_value}
