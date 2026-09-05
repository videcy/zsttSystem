"""Gold question schema shared by every evaluation entry point.

One item is one question plus everything needed to grade an answer to it
without re-reading the corpus:

``question``            the user-facing query, verbatim
``expected_route``      ``fact`` | ``content`` | ``dependency`` | ``catalog`` |
                        ``hybrid`` -- the path that *should* answer it
``answerable``          ``False`` for the deliberately unanswerable items that
                        measure the refusal mechanism
``answer_keys``         substrings that must appear in a correct answer; an
                        item is scored correct when every group is covered
``gold_chunk_ids``      chunk ids that contain the answer (human-labelled)
``gold_course_codes``   course-level fallback used when chunk ids are absent
``gold_section_types``  optional section constraint for the fallback
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

ROUTES = ("fact", "content", "dependency", "catalog", "hybrid")


@dataclass
class GoldItem:
    id: str
    question: str
    expected_route: str
    answerable: bool = True
    answer_keys: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_course_codes: list[str] = field(default_factory=list)
    gold_section_types: list[str] = field(default_factory=list)
    persona: str = "student"
    source: str = "human"
    notes: str = ""

    def validate(self) -> list[str]:
        """Return human-readable problems with this item."""
        problems: list[str] = []
        if not self.id:
            problems.append("missing id")
        if not self.question.strip():
            problems.append(f"{self.id}: empty question")
        if self.expected_route not in ROUTES:
            problems.append(
                f"{self.id}: expected_route {self.expected_route!r} "
                f"not in {ROUTES}"
            )
        if self.answerable and not (
            self.answer_keys or self.gold_chunk_ids or self.gold_course_codes
        ):
            problems.append(
                f"{self.id}: an answerable item needs answer_keys, "
                "gold_chunk_ids or gold_course_codes"
            )
        if not self.answerable and self.answer_keys:
            problems.append(
                f"{self.id}: an unanswerable item must not carry answer_keys"
            )
        return problems

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_items(records: Iterable[dict[str, Any]]) -> list[GoldItem]:
    known = set(GoldItem.__dataclass_fields__)
    items: list[GoldItem] = []
    for record in records:
        payload = {key: value for key, value in record.items() if key in known}
        items.append(GoldItem(**payload))
    return items


def load_dataset(path: str | Path) -> list[GoldItem]:
    """Load a gold set, raising on schema problems rather than scoring junk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("items", payload) if isinstance(payload, dict) else payload
    items = to_items(records)
    problems = [problem for item in items for problem in item.validate()]
    identifiers = [item.id for item in items]
    duplicates = {value for value in identifiers if identifiers.count(value) > 1}
    if duplicates:
        problems.append(f"duplicate ids: {sorted(duplicates)}")
    if problems:
        raise ValueError(
            "gold dataset is invalid:\n  " + "\n  ".join(problems)
        )
    return items


def save_dataset(
    items: Iterable[GoldItem],
    path: str | Path,
    *,
    description: str = "",
) -> Path:
    """Write a gold set with a stable key order for reviewable diffs."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": description,
        "items": [item.as_dict() for item in items],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def dataset_summary(items: Iterable[GoldItem]) -> dict[str, Any]:
    """Counts per route and answerability, for the report header."""
    materialised = list(items)
    by_route: dict[str, int] = {}
    for item in materialised:
        by_route[item.expected_route] = by_route.get(item.expected_route, 0) + 1
    return {
        "total": len(materialised),
        "by_route": dict(sorted(by_route.items())),
        "answerable": sum(1 for item in materialised if item.answerable),
        "unanswerable": sum(1 for item in materialised if not item.answerable),
        "human_labelled": sum(
            1 for item in materialised if item.source != "auto-seed"
        ),
    }
