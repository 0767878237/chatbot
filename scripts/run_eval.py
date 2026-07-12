from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_eval_set(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_retrieved_titles(result: dict) -> list[str]:
    debug = result.get("debug", {})
    retrieved_chunks = debug.get("retrieved_chunks", [])
    titles: list[str] = []
    for item in retrieved_chunks:
        title = str(item.get("title", "")).strip()
        if title:
            titles.append(title)
    return titles


def run_eval(
    eval_set: list[dict],
    mode: str,
    retrieval_strategy: str,
    top_k: int = 4,
) -> dict:
    from rag.chatbot import FoodChatbot
    from rag.pipeline import build_retriever
    from rag.retriever import normalize_text

    retriever = build_retriever(persist_artifacts=False)
    chatbot = FoodChatbot(retriever)

    total = len(eval_set)
    top1_hits = 0
    topk_hits = 0
    reciprocal_rank_sum = 0.0
    rows: list[dict] = []

    for item in eval_set:
        result = chatbot.answer(
            item["question"],
            top_k=top_k,
            mode=mode,
            retrieval_strategy=retrieval_strategy,
            search_mode="local_only",
        )
        retrieved_titles = extract_retrieved_titles(result)
        normalized_expected = [normalize_text(title) for title in item["expected_titles"]]
        normalized_titles = [normalize_text(title) for title in retrieved_titles]

        hit_rank = None
        for index, title in enumerate(normalized_titles, start=1):
            if any(expected in title or title in expected for expected in normalized_expected):
                hit_rank = index
                break

        if hit_rank == 1:
            top1_hits += 1
        if hit_rank is not None:
            topk_hits += 1
            reciprocal_rank_sum += 1 / hit_rank

        rows.append(
            {
                "question": item["question"],
                "expected_titles": item["expected_titles"],
                "retrieved_titles": retrieved_titles,
                "hit_rank": hit_rank,
            }
        )

    return {
        "mode": mode,
        "retrieval_strategy": retrieval_strategy,
        "total_questions": total,
        "top1_accuracy": round(top1_hits / total, 4) if total else 0.0,
        "topk_accuracy": round(topk_hits / total, 4) if total else 0.0,
        "mrr": round(reciprocal_rank_sum / total, 4) if total else 0.0,
        "details": rows,
    }


def main() -> None:
    eval_path = PROJECT_ROOT / "evals" / "sample_queries.json"
    eval_set = load_eval_set(eval_path)
    reports = [
        run_eval(eval_set, mode="baseline", retrieval_strategy="lexical"),
        run_eval(eval_set, mode="baseline", retrieval_strategy="hybrid"),
        run_eval(eval_set, mode="agentic", retrieval_strategy="hybrid"),
    ]

    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
