"""Offline corpus statistics backing BM25 reranking.

The vector index is rebuilt whenever chunks change, so document frequencies
are computed in the same stage and persisted next to the other pipeline
artifacts.  Retrieval degrades to plain term overlap when the file is absent,
which keeps the service usable on a fresh checkout.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.utils.lexical import term_set, tokenize

STATS_VERSION = "lexical-stats-v1"


def build_lexical_stats(
    chunks: Iterable[dict[str, Any]],
    *,
    min_document_frequency: int = 2,
) -> dict[str, Any]:
    """Count document frequencies and mean length over the chunk corpus.

    Terms rarer than ``min_document_frequency`` are dropped: an unlisted term
    is scored with the maximum IDF anyway, so keeping them only inflates the
    artifact.
    """
    document_frequency: Counter[str] = Counter()
    document_count = 0
    total_length = 0
    for chunk in chunks:
        text = str(chunk.get("text", "") or "")
        if not text.strip():
            continue
        document_count += 1
        tokens = tokenize(text)
        total_length += len(tokens)
        document_frequency.update(term_set(text))
    kept = {
        term: count
        for term, count in document_frequency.items()
        if count >= min_document_frequency
    }
    return {
        "version": STATS_VERSION,
        "document_count": document_count,
        "average_length": (total_length / document_count) if document_count else 0.0,
        "vocabulary_size": len(document_frequency),
        "min_document_frequency": min_document_frequency,
        "document_frequency": kept,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_lexical_stats(
    chunks: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    min_document_frequency: int = 2,
) -> dict[str, Any]:
    """Build statistics and persist them as JSON, returning a summary."""
    stats = build_lexical_stats(
        chunks,
        min_document_frequency=min_document_frequency,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stats, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        key: value
        for key, value in stats.items()
        if key != "document_frequency"
    }


def load_lexical_stats(path: str | Path) -> dict[str, Any] | None:
    """Load persisted statistics, or ``None`` when unavailable/corrupt."""
    stats_path = Path(path)
    if not stats_path.exists():
        return None
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "document_frequency" not in payload:
        return None
    return payload
