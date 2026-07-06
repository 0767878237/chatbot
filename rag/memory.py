from __future__ import annotations

from rag.query_router import analyze_query, normalize_text
from rag.types import ConversationMemory, ConversationTurn, RetrievalResult


FOLLOW_UP_MARKERS = [
    "con",
    "khac",
    "them",
    "so sanh",
    "quan do",
    "cho nay",
    "noi do",
    "o do",
    "ben do",
    "the con",
    "vay con",
]


def build_memory_from_messages(messages: list[dict]) -> ConversationMemory:
    memory = ConversationMemory()
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            continue
        memory.turns.append(ConversationTurn(role=role, content=content))
        if role == "user":
            memory.last_user_query = content
            analysis = analyze_query(content)
            memory.locations = merge_unique(memory.locations, analysis.location_terms)
            memory.cuisines = merge_unique(memory.cuisines, analysis.cuisine_terms)
            memory.vibes = merge_unique(memory.vibes, analysis.vibe_terms)
            memory.preferences = merge_unique(
                memory.preferences,
                analysis.cuisine_terms + analysis.vibe_terms + analysis.location_terms,
            )
        elif role == "assistant":
            titles = extract_titles_from_debug(message.get("debug"))
            memory.last_recommendations = merge_unique(memory.last_recommendations, titles)[:5]

    return memory


def build_memory_before_current_turn(messages: list[dict], current_question: str) -> ConversationMemory:
    trimmed_messages = list(messages)
    while trimmed_messages:
        last_message = trimmed_messages[-1]
        role = str(last_message.get("role", "")).strip()
        content = str(last_message.get("content", "")).strip()
        if role == "user" and content == current_question.strip():
            trimmed_messages = trimmed_messages[:-1]
            break
        trimmed_messages = trimmed_messages[:-1]
    return build_memory_from_messages(trimmed_messages)


def rewrite_with_memory(question: str, memory: ConversationMemory) -> tuple[str, dict]:
    normalized = normalize_text(question)
    if not is_follow_up(normalized):
        return question, {
            "used_memory": False,
            "reason": "query_is_self_contained",
        }

    parts = [question.strip()]
    injected_context: list[str] = []
    current_analysis = analyze_query(question)
    has_explicit_location = bool(current_analysis.location_terms)
    has_explicit_cuisine = bool(current_analysis.cuisine_terms)
    has_explicit_vibe = bool(current_analysis.vibe_terms)

    if memory.last_user_query:
        injected_context.append(f"ngu canh truoc: {memory.last_user_query}")
    if memory.cuisines and not has_explicit_cuisine:
        injected_context.append("mon dang quan tam: " + ", ".join(memory.cuisines[-3:]))
    if memory.vibes and not has_explicit_vibe:
        injected_context.append("vibe dang quan tam: " + ", ".join(memory.vibes[-3:]))
    if memory.locations and not has_explicit_location:
        injected_context.append("dia diem dang quan tam: " + ", ".join(memory.locations[-3:]))
    if memory.last_recommendations:
        injected_context.append(
            "quan da duoc goi y: " + ", ".join(memory.last_recommendations[-3:])
        )

    if injected_context:
        parts.append(". " + ". ".join(injected_context))

    rewritten = "".join(parts).strip()
    return rewritten, {
        "used_memory": True,
        "reason": "follow_up_detected",
        "rewritten_query": rewritten,
        "memory_snapshot": serialize_memory(memory),
    }


def update_memory_with_results(
    memory: ConversationMemory,
    question: str,
    results: list[RetrievalResult],
) -> ConversationMemory:
    updated = ConversationMemory(
        turns=list(memory.turns),
        preferences=list(memory.preferences),
        locations=list(memory.locations),
        cuisines=list(memory.cuisines),
        vibes=list(memory.vibes),
        last_recommendations=list(memory.last_recommendations),
        last_user_query=question,
    )

    analysis = analyze_query(question)
    updated.locations = merge_unique(updated.locations, analysis.location_terms)
    updated.cuisines = merge_unique(updated.cuisines, analysis.cuisine_terms)
    updated.vibes = merge_unique(updated.vibes, analysis.vibe_terms)
    updated.preferences = merge_unique(
        updated.preferences,
        analysis.location_terms + analysis.cuisine_terms + analysis.vibe_terms,
    )
    updated.last_recommendations = merge_unique(
        updated.last_recommendations,
        [item.chunk.document.title for item in results],
    )[:5]
    return updated


def serialize_memory(memory: ConversationMemory) -> dict:
    return {
        "preferences": memory.preferences,
        "locations": memory.locations,
        "cuisines": memory.cuisines,
        "vibes": memory.vibes,
        "last_recommendations": memory.last_recommendations,
        "last_user_query": memory.last_user_query,
        "turn_count": len(memory.turns),
    }


def is_follow_up(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in FOLLOW_UP_MARKERS)


def merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    for item in incoming:
        if item and item not in merged:
            merged.append(item)
    return merged


def extract_titles_from_debug(debug_data: object) -> list[str]:
    if not isinstance(debug_data, dict):
        return []
    chunks = debug_data.get("retrieved_chunks", [])
    if not isinstance(chunks, list):
        return []
    titles: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        title = str(chunk.get("title", "")).strip()
        if title and title not in titles:
            titles.append(title)
    return titles
