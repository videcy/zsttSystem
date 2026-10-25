#!/usr/bin/env python3
"""Export accepted concept dependencies as course-level Mermaid graphs.

The verified edge file contains both accepted and rejected LLM verdicts.  This
script keeps only ``requires=true`` relations, lifts them to course pairs using
the concept registry, and counts unique concept relations per course pair.

Examples:

    python scripts/export_concept_mermaid.py --list-programs
    python scripts/export_concept_mermaid.py --program 图书馆学
    python scripts/export_concept_mermaid.py --program 图书馆学 \
        --output-dir outputs/mermaid
    python scripts/export_concept_mermaid.py --all --output-dir outputs/mermaid
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.online_service.program_index import core_program_name  # noqa: E402

DEFAULT_REGISTRY = REPO_ROOT / "outputs" / "concept_registry.json"
DEFAULT_EDGES = REPO_ROOT / "outputs" / "concept_verified_edges.json"
DEFAULT_COURSES = REPO_ROOT / "outputs" / "courses.json"
DEFAULT_TOP = 30


@dataclass(frozen=True)
class CourseFlow:
    """One aggregated course-to-course knowledge flow."""

    source: str
    target: str
    count: int


def load_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    """Load a JSON array and fail with a useful message for bad artifacts."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{label} must contain a JSON array: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _course_codes(concept: Mapping[str, Any]) -> set[str]:
    raw = concept.get("source_course_codes") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return set()
    codes = set()
    for code in raw:
        text = str(code or "").strip()
        if text:
            codes.add(text)
    return codes


def aggregate_course_flows(
    registry: Iterable[Mapping[str, Any]],
    verified_edges: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    """Count unique accepted concept relations for every course pair.

    Duplicate verification rows for the same concept pair are collapsed before
    counting, so repeated votes or repeated fixture rows cannot inflate a flow.
    """
    concepts = {
        str(item.get("id") or ""): item
        for item in registry
        if str(item.get("id") or "").strip()
    }
    relations: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

    for edge in verified_edges:
        if edge.get("requires") is not True:
            continue
        source_id = str(edge.get("source_id") or "").strip()
        target_id = str(edge.get("target_id") or "").strip()
        source = concepts.get(source_id)
        target = concepts.get(target_id)
        if not source or not target or source_id == target_id:
            continue
        relation = (source_id, target_id)
        for source_course in _course_codes(source):
            for target_course in _course_codes(target):
                if source_course != target_course:
                    relations[(source_course, target_course)].add(relation)

    return {pair: len(items) for pair, items in relations.items() if items}


def major_course_sets(
    courses: Iterable[Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Map each core major name to the course codes in any of its plans."""
    offerings: list[tuple[str, str, str]] = []
    for course in courses:
        code = str(course.get("course_code") or "").strip()
        if not code:
            continue
        course_offerings = [
            item for item in course.get("offerings") or [] if isinstance(item, Mapping)
        ]
        if course_offerings:
            for offering in course_offerings:
                name = str(offering.get("program_name") or "").strip()
                if name:
                    offerings.append(
                        (code, name, str(offering.get("program_type") or "").strip())
                    )
        else:
            for name in course.get("program_names") or []:
                if str(name).strip():
                    offerings.append((code, str(name).strip(), ""))

    program_types = sorted(
        {program_type for _, _, program_type in offerings if program_type},
        key=len,
        reverse=True,
    )
    majors: dict[str, set[str]] = defaultdict(set)
    for code, program_name, _ in offerings:
        major = core_program_name(program_name, program_types) or program_name
        majors[major].add(code)
    return dict(sorted(majors.items()))


def course_name_map(courses: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Return the display name for every course code in the catalogue."""
    return {
        str(course.get("course_code") or "").strip(): str(
            course.get("course_name") or course.get("name") or ""
        ).strip()
        for course in courses
        if str(course.get("course_code") or "").strip()
    }


def top_course_flows(
    counts: Mapping[tuple[str, str], int],
    allowed_courses: Iterable[str],
    *,
    limit: int = DEFAULT_TOP,
) -> list[CourseFlow]:
    """Return the heaviest flows wholly inside one major, at most ``limit``."""
    allowed = {str(code) for code in allowed_courses}
    flows = [
        CourseFlow(source=source, target=target, count=count)
        for (source, target), count in counts.items()
        if source in allowed and target in allowed
    ]
    flows.sort(key=lambda flow: (-flow.count, flow.source, flow.target))
    return flows[: max(limit, 0)]


def _escape_label(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', "&quot;")
        .replace("\r", " ")
        .replace("\n", "<br/>")
    )


def render_mermaid(
    flows: Sequence[CourseFlow],
    names: Mapping[str, str],
) -> str:
    """Render one readable left-to-right Mermaid flowchart."""
    codes = sorted({flow.source for flow in flows} | {flow.target for flow in flows})
    node_ids = {code: f"course_{index}" for index, code in enumerate(codes)}
    lines = ["```mermaid", "flowchart LR"]
    for code in codes:
        name = names.get(code, "").strip()
        label = f"{code}<br/>{name}" if name else code
        lines.append(f'  {node_ids[code]}["{_escape_label(label)}"]')
    for flow in flows:
        lines.append(
            f"  {node_ids[flow.source]} -->|{flow.count}| {node_ids[flow.target]}"
        )
    lines.append("```")
    return "\n".join(lines)


def render_markdown(
    major: str,
    flows: Sequence[CourseFlow],
    names: Mapping[str, str],
    *,
    limit: int = DEFAULT_TOP,
) -> str:
    """Wrap a major's Mermaid graph with a short report-friendly heading."""
    return "\n".join(
        [
            f"# {major}：课程级知识流向",
            "",
            f"按课程对聚合已验收概念关系，按关系数降序排列，最多 {limit} 条。",
            "",
            render_mermaid(flows, names),
            "",
        ]
    )


def resolve_majors(
    available: Mapping[str, set[str]],
    requested: Sequence[str],
) -> list[str]:
    """Resolve exact core names; reject typos instead of silently exporting all."""
    if not requested:
        return list(available)
    selected: list[str] = []
    for name in requested:
        if name not in available:
            choices = "、".join(available)
            raise ValueError(f"unknown major {name!r}; choose from: {choices}")
        if name not in selected:
            selected.append(name)
    return selected


def _safe_filename(name: str, index: int) -> str:
    stem = re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("._")
    return f"{index:02d}-{stem or 'major'}.md"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--courses", type=Path, default=DEFAULT_COURSES)
    parser.add_argument(
        "--program",
        action="append",
        default=[],
        help="专业核心名，可重复；不传则导出全部专业",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="导出全部专业（默认行为，保留该参数便于显式表达）",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="每个专业写一个 .md；不传则输出到标准输出",
    )
    parser.add_argument("--list-programs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.top < 1:
        parser.error("--top must be at least 1")

    try:
        courses = load_json_list(arguments.courses, "courses")
        majors = major_course_sets(courses)
        if arguments.list_programs:
            for name in majors:
                print(name)
            return 0

        requested = [] if arguments.all else arguments.program
        selected = resolve_majors(majors, requested)
        registry = load_json_list(arguments.registry, "concept registry")
        edges = load_json_list(arguments.edges, "verified concept edges")
    except ValueError as exc:
        parser.error(str(exc))

    counts = aggregate_course_flows(registry, edges)
    names = course_name_map(courses)
    rendered = []
    for major in selected:
        flows = top_course_flows(counts, majors[major], limit=arguments.top)
        rendered.append(
            render_markdown(major, flows, names, limit=arguments.top).strip()
        )

    if arguments.output_dir is None:
        print("\n\n".join(rendered))
        return 0

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (major, markdown) in enumerate(zip(selected, rendered), start=1):
        path = arguments.output_dir / _safe_filename(major, index)
        path.write_text(markdown + "\n", encoding="utf-8")
        print(f"[mermaid] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
