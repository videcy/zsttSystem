"""Feedback recording helpers for the online service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_citations(selected_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a lightweight citation payload from selected retrieval results."""
    citations: list[dict[str, Any]] = []
    for document in selected_docs:
        citation = {
            "source": document.get("source"),
            "id": document.get("id"),
            "metadata": document.get("metadata", {}),
        }
        if document.get("source") == "graph":
            citation["path"] = document.get("nodes", [])
        citations.append(citation)
    return citations


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
