from __future__ import annotations

import re
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def collect_text_entries(payload: Any, prefix: str = "") -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            field_name = f"{prefix}.{key}" if prefix else key
            entries.extend(collect_text_entries(value, field_name))
        return entries
    if isinstance(payload, list):
        for index, value in enumerate(payload):
            field_name = f"{prefix}[{index}]"
            entries.extend(collect_text_entries(value, field_name))
        return entries
    if isinstance(payload, str):
        cleaned = normalize_text(payload)
        if cleaned:
            entries.append({"path": prefix or "$", "text": cleaned})
    return entries


def search_task_text(payload: dict[str, Any], query: str, limit: int = 5) -> list[dict[str, str]]:
    query_tokens = tokenize(query)
    scored: list[tuple[int, int, dict[str, str]]] = []
    for index, entry in enumerate(collect_text_entries(payload)):
        text = entry["text"]
        tokens = tokenize(text)
        overlap = len(tokens & query_tokens)
        coverage = sum(1 for token in query_tokens if token in text.lower())
        score = overlap + coverage
        scored.append((score, -index, entry))

    ranked = [entry for score, _, entry in sorted(scored, reverse=True) if score > 0]
    if ranked:
        return ranked[:limit]
    return collect_text_entries(payload)[:limit]
