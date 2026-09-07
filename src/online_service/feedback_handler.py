"""Feedback recording helpers for the online service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_jsonl_record(file_path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL log file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_query_log_record(
    query_id: str,
    query: str,
    context: str,
    kg_path: str,
    response: str,
    verification: list[dict[str, Any]],
    linked_entities: list[str],
    citations: list[dict[str, Any]],
    status: str,
    persona: str = "student",
    persona_mode: str = "retrieval",
    persona_profile_version: str = "v1",
) -> dict[str, Any]:
    """Create a structured query log record."""
    return {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "context": context,
        "kg_path": kg_path,
        "response": response,
        "verification": verification,
        "linked_entities": linked_entities,
        "citations": citations,
        "status": status,
        "persona": persona,
        "persona_mode": persona_mode,
        "persona_profile_version": persona_profile_version,
    }


def build_feedback_log_record(
    query_id: str,
    is_helpful: bool,
    comment: str | None = None,
) -> dict[str, Any]:
    """Create a structured feedback log record."""
    return {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_helpful": is_helpful,
        "comment": comment,
    }
