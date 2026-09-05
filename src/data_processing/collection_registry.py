"""Versioned Chroma collections with an alias pointer.

Chroma has no native alias concept, so the pointer lives in a small JSON file
next to the other pipeline artifacts::

    {"alias": "zstt_chunks", "active": "zstt_chunks_v3", "history": [...]}

Readers resolve ``alias -> active`` at startup; a rebuild writes a new
``<alias>_v<n>`` collection and only repoints the alias once it is verified,
which makes a bad retraining round recoverable by pointing back at ``v<n-1>``
instead of reparsing the corpus.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import config

_VERSION_SUFFIX = re.compile(r"^(?P<alias>.+)_v(?P<version>\d+)$")


def parse_version(collection_name: str) -> int | None:
    """Return the ``_v<n>`` suffix of a collection name, if it has one."""
    match = _VERSION_SUFFIX.match(collection_name)
    return int(match.group("version")) if match else None


def version_name(alias: str, version: int) -> str:
    return f"{alias}_v{int(version)}"


def read_alias_record(alias_path: str | Path | None = None) -> dict[str, Any]:
    """Load the alias pointer, returning ``{}`` when it does not exist."""
    path = Path(alias_path or config.collection_alias_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_alias_record(
    active: str,
    *,
    alias: str | None = None,
    alias_path: str | Path | None = None,
    note: str = "",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Point the alias at ``active`` and append to its switch history."""
    path = Path(alias_path or config.collection_alias_path)
    record = read_alias_record(path)
    resolved_alias = alias or record.get("alias") or config.chroma_collection
    history = list(record.get("history") or [])
    history.append(
        {
            "active": active,
            "previous": record.get("active"),
            "note": note,
            "metrics": metrics or {},
            "switched_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    payload = {
        "alias": resolved_alias,
        "active": active,
        "history": history[-50:],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def resolve_active_collection(
    alias: str | None = None,
    *,
    alias_path: str | Path | None = None,
    available: list[str] | None = None,
) -> str:
    """Resolve the collection a reader should open.

    Falls back to the plain alias name when no pointer exists, or when the
    pointer names a collection that is not present in ``available``.
    """
    resolved_alias = alias or config.chroma_collection
    record = read_alias_record(alias_path)
    if record.get("alias") not in (None, resolved_alias):
        return resolved_alias
    active = str(record.get("active") or "").strip()
    if not active:
        return resolved_alias
    if available is not None and active not in available:
        return resolved_alias
    return active


def next_version_name(
    alias: str | None = None,
    *,
    existing: list[str] | None = None,
    alias_path: str | Path | None = None,
) -> str:
    """Name the next version after the highest one seen in ``existing``."""
    resolved_alias = alias or config.chroma_collection
    versions = [0]
    for name in existing or []:
        match = _VERSION_SUFFIX.match(name)
        if match and match.group("alias") == resolved_alias:
            versions.append(int(match.group("version")))
    record = read_alias_record(alias_path)
    active_version = parse_version(str(record.get("active") or ""))
    if active_version is not None:
        versions.append(active_version)
    return version_name(resolved_alias, max(versions) + 1)
