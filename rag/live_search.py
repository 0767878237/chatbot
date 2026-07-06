from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from rag.query_router import analyze_query, normalize_text


KNOWN_LOCATIONS = [
    "ha noi",
    "da nang",
    "can tho",
    "nha trang",
    "vung tau",
    "bien hoa",
    "binh duong",
    "dong nai",
    "hue",
    "hai phong",
    "thu duc",
    "binh thanh",
    "go vap",
    "phu nhuan",
    "tan binh",
    "tan phu",
    "quan 1",
    "quan 2",
    "quan 3",
    "quan 4",
    "quan 5",
    "quan 6",
    "quan 7",
    "quan 8",
    "quan 9",
    "quan 10",
    "quan 11",
    "quan 12",
]


class LiveRestaurantSearch:
    def __init__(self, user_agent: str = "SaiGonFoodChatbot/1.0"):
        self.user_agent = user_agent
        self.nominatim_url = "https://nominatim.openstreetmap.org/search"
        self.overpass_url = "https://overpass-api.de/api/interpreter"

    def search(self, question: str, limit: int = 5, original_question: str | None = None) -> dict:
        original_question = original_question or question
        area_query = self.extract_area_query(original_question)
        geocode = self.geocode_area(area_query)
        if not geocode:
            return {
                "ok": False,
                "reason": "area_not_found",
                "area_query": area_query,
                "results": [],
            }

        places = self.search_places(
            lat=float(geocode["lat"]),
            lon=float(geocode["lon"]),
            question=original_question,
            limit=limit,
        )
        if not places:
            return {
                "ok": False,
                "reason": "no_places_found",
                "area_query": area_query,
                "resolved_area": geocode.get("display_name", area_query),
                "results": [],
            }

        return {
            "ok": True,
            "reason": "live_results_found",
            "area_query": area_query,
            "resolved_area": geocode.get("display_name", area_query),
            "results": places,
        }

    def extract_area_query(self, question: str) -> str:
        normalized = normalize_text(question)
        explicit = self.detect_known_location(normalized)
        if explicit:
            return explicit + ", viet nam"

        explicit_pattern = re.search(
            r"(o|tai|khu vuc|gan)\s+([a-z0-9\s]+?)(?:\?|$|,|\.)",
            normalized,
        )
        if explicit_pattern:
            candidate = explicit_pattern.group(2).strip()
            if candidate:
                return candidate + ", viet nam"

        return question + ", viet nam"

    def geocode_area(self, area_query: str) -> dict | None:
        params = urllib.parse.urlencode(
            {
                "q": area_query,
                "format": "jsonv2",
                "limit": 1,
            }
        )
        request = urllib.request.Request(
            url=f"{self.nominatim_url}?{params}",
            headers={"User-Agent": self.user_agent},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload:
            return None
        return payload[0]

    def search_places(self, lat: float, lon: float, question: str, limit: int) -> list[dict]:
        analysis = analyze_query(question)
        query = self.build_overpass_query(lat=lat, lon=lon, radius=6000)
        request = urllib.request.Request(
            url=self.overpass_url,
            data=query.encode("utf-8"),
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))

        elements = payload.get("elements", [])
        ranked = sorted(
            [self.transform_place(element, analysis.cuisine_terms, analysis.vibe_terms) for element in elements],
            key=lambda item: item["score"],
            reverse=True,
        )
        deduped: list[dict] = []
        seen_keys: set[str] = set()
        for item in ranked:
            key = item["name"] + "|" + item["address"]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def build_overpass_query(self, lat: float, lon: float, radius: int) -> str:
        return f"""
[out:json][timeout:20];
(
  node["amenity"~"restaurant|fast_food|food_court|cafe"](around:{radius},{lat},{lon});
  way["amenity"~"restaurant|fast_food|food_court|cafe"](around:{radius},{lat},{lon});
  rel["amenity"~"restaurant|fast_food|food_court|cafe"](around:{radius},{lat},{lon});
);
out center 50;
""".strip()

    def transform_place(
        self,
        element: dict,
        cuisine_terms: list[str],
        vibe_terms: list[str],
    ) -> dict:
        tags = element.get("tags", {})
        name = tags.get("name") or "Dia diem an uong"
        address = self.build_address(tags)
        haystack = normalize_text(
            " ".join(
                [
                    str(name),
                    str(tags.get("cuisine", "")),
                    str(tags.get("amenity", "")),
                    str(tags.get("description", "")),
                ]
            )
        )

        score = 0.15
        amenity = tags.get("amenity", "")
        if amenity == "restaurant":
            score += 0.25
        elif amenity in {"fast_food", "cafe", "food_court"}:
            score += 0.12

        if address:
            score += 0.15
        if tags.get("opening_hours"):
            score += 0.05
        if tags.get("cuisine"):
            score += 0.08
        if cuisine_terms and any(term in haystack for term in cuisine_terms):
            score += 0.3
        if vibe_terms and any(term in haystack for term in vibe_terms):
            score += 0.08

        return {
            "source_type": "live",
            "name": name,
            "address": address or "Khong ro dia chi",
            "amenity": amenity or "restaurant",
            "cuisine": tags.get("cuisine", "Khong ro"),
            "opening_hours": tags.get("opening_hours", ""),
            "score": round(score, 4),
        }

    def build_address(self, tags: dict) -> str:
        parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:suburb", ""),
            tags.get("addr:city", ""),
        ]
        return ", ".join(part for part in parts if part).strip(", ")

    def detect_known_location(self, normalized_question: str) -> str | None:
        detected = sorted(
            {hint for hint in KNOWN_LOCATIONS if hint in normalized_question},
            key=len,
            reverse=True,
        )
        return detected[0] if detected else None
