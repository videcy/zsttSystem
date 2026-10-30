"""Program-level curriculum network for the 专业课程图谱 page.

The page shows one training plan as a semester-ordered network: every course
the plan lists, the hard prerequisite edges between them, and -- optionally --
course-to-course links inferred from the verified concept graph.

Two kinds of edge, deliberately kept apart:

* ``prerequisite`` edges come from the plan's 先修课程 column or the
  syllabus 先修课程 field after exact alias resolution (the same records the
  graph stage writes as ``PREREQUISITE_OF``).  They are facts.
* ``knowledge`` edges aggregate verified concept dependencies
  (``requires=true`` from an LLM vote or a human review) up to the courses that
  teach each concept.  They are inferences, and the API labels them so.

Everything is computed from the pipeline's JSON artefacts, so the page works
without Neo4j -- the same promise the content path makes.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.online_service.program_index import core_program_name

# Placeholders the training plans use for "no value"; mirrors the router's
# notion of a missing field so the two never disagree about a teacher.
_MISSING = {"", "/", "／", "无", "-", "—", "暂无", "none", "null"}

_COLLEGE_PATTERN = re.compile(r"^(.*?学院)")

# Display order of course groups inside one semester column.
GROUP_ORDER: tuple[str, ...] = (
    "public",
    "foundation",
    "core",
    "practice",
    "elective",
    "minor",
    "other",
)

GROUP_LABELS: dict[str, str] = {
    "public": "公共必修课",
    "foundation": "基础课（大类 / 专业基础）",
    "core": "专业核心课",
    "practice": "专业实践课",
    "elective": "专业选修课",
    "minor": "辅修课程",
    "other": "其他",
}

# Plan types ordered the way a student meets them.
_PLAN_TYPE_ORDER = ("主修专业类", "主修专业", "辅修专业", "辅修微专业")

# Inferred links beyond this many are dropped (heaviest kept): past it the
# picture stops being readable and the long tail is single-concept noise.
MAX_KNOWLEDGE_EDGES = 150


def _clean(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.casefold() in _MISSING else text


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def parse_semester(value: Any) -> tuple[int | None, str]:
    """Return ``(column, label)`` for a plan's 开课学期 cell.

    A single semester places the course in that column.  A range such as
    ``1-6`` means "any semester in that window": it has no fixed column and
    goes to the flexible column, keeping its original label.
    """
    number = _number(value)
    if number is not None and float(number).is_integer() and 1 <= number <= 12:
        return int(number), f"第 {int(number)} 学期"
    text = _clean(value)
    if not text:
        return None, "未标注"
    match = re.fullmatch(r"(\d+)\s*[-–~～至]\s*(\d+)", text)
    if match:
        return None, f"第 {match.group(1)}–{match.group(2)} 学期（弹性）"
    return None, text


def course_group(category: str, subcategory: str) -> str:
    text = f"{category} {subcategory}"
    if "公共" in text:
        return "public"
    if "核心" in text:
        return "core"
    if "实践" in text or "实习" in text:
        return "practice"
    if "基础" in text:
        return "foundation"
    if "选修" in text:
        return "elective"
    if "辅修" in text:
        return "minor"
    return "other"


def college_name(program_name: str) -> str:
    match = _COLLEGE_PATTERN.match(str(program_name or "").strip())
    return match.group(1) if match else "未标注学院"


def _plan_sort_key(plan_type: str) -> tuple[int, str]:
    try:
        return _PLAN_TYPE_ORDER.index(plan_type), plan_type
    except ValueError:
        return len(_PLAN_TYPE_ORDER), plan_type


def program_catalog(courses: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Group every training plan by college and major for the selector."""
    plans: dict[str, dict[str, Any]] = {}
    for course in courses:
        for offering in course.get("offerings") or []:
            name = str(offering.get("program_name") or "").strip()
            if not name:
                continue
            plan = plans.setdefault(
                name,
                {
                    "program_name": name,
                    "program_type": str(offering.get("program_type") or ""),
                    "course_codes": set(),
                },
            )
            plan["course_codes"].add(str(course.get("course_code") or ""))

    types = sorted({plan["program_type"] for plan in plans.values()}, key=len, reverse=True)
    colleges: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for plan in plans.values():
        major = core_program_name(plan["program_name"], types) or plan["program_name"]
        colleges[college_name(plan["program_name"])][major].append(
            {
                "program_name": plan["program_name"],
                "program_type": plan["program_type"],
                "course_count": len(plan["course_codes"] - {""}),
            }
        )

    result = []
    for college in sorted(colleges):
        majors = []
        for major, items in colleges[college].items():
            items.sort(key=lambda item: _plan_sort_key(item["program_type"]))
            majors.append({"name": major, "plans": items})
        # The broad-category plan (大类) first, then majors by size.
        majors.sort(
            key=lambda major: (
                0 if any(p["program_type"] == "主修专业类" for p in major["plans"]) else 1,
                -max(p["course_count"] for p in major["plans"]),
                major["name"],
            )
        )
        result.append({"name": college, "majors": majors})
    return {"colleges": result, "program_count": len(plans)}


# ---------------------------------------------------------------------------
# Inferred knowledge links
# ---------------------------------------------------------------------------

_CONCEPT_LINK_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def aggregate_concept_links(
    registry: Iterable[Mapping[str, Any]],
    verified_edges: Iterable[Mapping[str, Any]],
    *,
    min_confidence: float,
) -> dict[str, Any]:
    """Lift verified concept dependencies to course pairs.

    The same gate as the graph stage: ``requires`` must be literally ``True``,
    the verification must be a full LLM vote or a human review, and the
    confidence must reach ``min_confidence``.
    """
    concepts = {
        str(item.get("id") or item.get("concept_id") or ""): item
        for item in registry
    }
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    accepted = 0
    for edge in verified_edges:
        if edge.get("requires") is not True:
            continue
        if str(edge.get("verification_source") or "").casefold() not in {"llm", "human"}:
            continue
        try:
            confidence = float(edge.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue
        source = concepts.get(str(edge.get("source_id") or ""))
        target = concepts.get(str(edge.get("target_id") or ""))
        if not source or not target:
            continue
        accepted += 1
        example = (
            f"{source.get('canonical_name') or source.get('name')}"
            f" → {target.get('canonical_name') or target.get('name')}"
        )
        for source_code in source.get("source_course_codes") or []:
            for target_code in target.get("source_course_codes") or []:
                source_code, target_code = str(source_code), str(target_code)
                if source_code and target_code and source_code != target_code:
                    if example not in pairs[(source_code, target_code)]:
                        pairs[(source_code, target_code)].append(example)
    return {
        "pairs": dict(pairs),
        "concept_count": len(concepts),
        "verified_relation_count": accepted,
    }


def load_concept_links(
    registry_path: Path,
    edges_path: Path,
    *,
    min_confidence: float,
    loader: Callable[[Path], Any],
) -> dict[str, Any] | None:
    """Aggregate once per artefact version; ``None`` when either is missing."""
    registry_mtime, edges_mtime = _mtime(registry_path), _mtime(edges_path)
    if registry_mtime is None or edges_mtime is None:
        return None
    key = (str(registry_path), registry_mtime, str(edges_path), edges_mtime, min_confidence)
    cached = _CONCEPT_LINK_CACHE.get(key)
    if cached is None:
        registry = loader(registry_path)
        edges = loader(edges_path)
        if not isinstance(registry, list) or not isinstance(edges, list):
            return None
        cached = aggregate_concept_links(registry, edges, min_confidence=min_confidence)
        _CONCEPT_LINK_CACHE.clear()
        _CONCEPT_LINK_CACHE[key] = cached
    return cached


def _safe_load_json(path: Path) -> Any:
    """只读、容错地读取 JSON；缺失或损坏都返回 None。"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_configured_concept_links() -> dict[str, Any] | None:
    """按配置的产物路径与阈值加载课程对知识关联；`/programs/graph` 与路由层共用。"""
    from src.config import config

    return load_concept_links(
        config.concept_registry_path,
        config.concept_verified_edge_path,
        min_confidence=config.concept_verified_min_confidence,
        loader=_safe_load_json,
    )


# ---------------------------------------------------------------------------
# Program graph
# ---------------------------------------------------------------------------


def _offering_for(course: Mapping[str, Any], program_name: str) -> dict[str, Any] | None:
    for offering in course.get("offerings") or []:
        if str(offering.get("program_name") or "") == program_name:
            return dict(offering)
    return None


def _home_offering(course: Mapping[str, Any]) -> dict[str, Any]:
    """Where a course outside the plan normally sits: its earliest fixed slot."""
    offerings = list(course.get("offerings") or [])
    fixed = [o for o in offerings if parse_semester(o.get("semester"))[0] is not None]
    pool = fixed or offerings
    if not pool:
        return {}
    return dict(min(pool, key=lambda o: parse_semester(o.get("semester"))[0] or 99))


def _node(course: Mapping[str, Any], offering: Mapping[str, Any], *, external: bool) -> dict[str, Any]:
    semester, semester_label = parse_semester(offering.get("semester"))
    category = _clean(offering.get("course_category"))
    subcategory = _clean(offering.get("course_subcategory"))
    instructor = _clean(offering.get("instructor")) or _clean(course.get("instructor"))
    return {
        "course_code": str(course.get("course_code") or ""),
        "course_name": str(course.get("course_name") or ""),
        "english_name": _clean(course.get("english_name")),
        "semester": semester,
        "semester_label": semester_label,
        "category": category,
        "subcategory": subcategory,
        "group": course_group(category, subcategory),
        "credits": _number(offering.get("credits")),
        "hours": _number(offering.get("hours")),
        "instructor": instructor,
        "external": external,
        "home_program": str(offering.get("program_name") or "") if external else "",
    }


def _longest_chain(codes: Iterable[str], edges: list[dict[str, Any]]) -> list[str]:
    """Longest hard-prerequisite path; cycles are cut rather than followed."""
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[edge["source"]].append(edge["target"])
    best: dict[str, list[str]] = {}
    visiting: set[str] = set()

    def walk(code: str) -> list[str]:
        if code in best:
            return best[code]
        if code in visiting:
            return [code]
        visiting.add(code)
        tail: list[str] = []
        for nxt in successors.get(code, []):
            candidate = walk(nxt)
            if len(candidate) > len(tail):
                tail = candidate
        visiting.discard(code)
        best[code] = [code, *tail]
        return best[code]

    longest: list[str] = []
    for code in codes:
        path = walk(code)
        if len(path) > len(longest):
            longest = path
    return longest if len(longest) > 1 else []


def build_program_graph(
    courses: Iterable[Mapping[str, Any]],
    program_name: str,
    *,
    concept_links: Mapping[str, Any] | None = None,
    include_knowledge: bool = True,
    max_knowledge_edges: int = MAX_KNOWLEDGE_EDGES,
) -> dict[str, Any] | None:
    """Return the network of one plan, or ``None`` when the plan is unknown."""
    catalog = {str(course.get("course_code") or ""): course for course in courses}
    nodes: dict[str, dict[str, Any]] = {}
    program_type = ""
    for code, course in catalog.items():
        offering = _offering_for(course, program_name)
        if offering is None or not code:
            continue
        program_type = program_type or str(offering.get("program_type") or "")
        nodes[code] = _node(course, offering, external=False)
    if not nodes:
        return None
    in_plan = set(nodes)

    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for target in sorted(in_plan):
        course = catalog[target]
        evidence = {
            str(item.get("course_code") or ""): item
            for item in course.get("prerequisite_evidence") or []
            if isinstance(item, Mapping)
        }
        for source in course.get("prerequisites") or []:
            source = str(source)
            if source not in catalog or source == target or (source, target) in seen_pairs:
                continue
            if source not in nodes:
                nodes[source] = _node(
                    catalog[source], _home_offering(catalog[source]), external=True
                )
            item = evidence.get(source, {})
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "type": "prerequisite",
                    "source_file": str(item.get("source_file") or "").rsplit("/", 1)[-1]
                    or None,
                    "raw_name": item.get("raw_name"),
                    "evidence_type": item.get("source_type") or "training_plan",
                }
            )
            seen_pairs.add((source, target))

    knowledge_edges: list[dict[str, Any]] = []
    knowledge_total = 0
    if include_knowledge and concept_links:
        candidates: dict[tuple[str, str], list[str]] = {}
        for (source, target), examples in (concept_links.get("pairs") or {}).items():
            if source not in in_plan or target not in in_plan:
                continue
            if (source, target) in seen_pairs:
                continue
            source_sem = nodes[source]["semester"]
            target_sem = nodes[target]["semester"]
            # Knowledge flows forward in time: a link that points from a later
            # course to an earlier one contradicts the plan, so it is dropped.
            if source_sem is not None and target_sem is not None and source_sem > target_sem:
                continue
            candidates[(source, target)] = list(examples)
        for (source, target), examples in candidates.items():
            # Both directions surviving means the shared concepts carry no
            # order at all -- that is overlap, not dependency.
            if (target, source) in candidates:
                continue
            knowledge_edges.append(
                {
                    "source": source,
                    "target": target,
                    "type": "knowledge",
                    "weight": len(examples),
                    "examples": examples[:3],
                }
            )
        knowledge_total = len(knowledge_edges)
        knowledge_edges.sort(key=lambda e: (-e["weight"], e["source"], e["target"]))
        knowledge_edges = knowledge_edges[:max_knowledge_edges]

    chain = _longest_chain(sorted(nodes), edges)
    linked = {e["source"] for e in edges} | {e["target"] for e in edges}
    knowledge_linked = {e["source"] for e in knowledge_edges} | {
        e["target"] for e in knowledge_edges
    }
    semesters = sorted({n["semester"] for n in nodes.values() if n["semester"] is not None})
    group_counts: dict[str, int] = defaultdict(int)
    for node in nodes.values():
        if not node["external"]:
            group_counts[node["group"]] += 1
    return {
        "program_name": program_name,
        "program_type": program_type,
        "college": college_name(program_name),
        "semesters": semesters,
        "groups": [
            {"key": key, "label": GROUP_LABELS[key], "count": group_counts.get(key, 0)}
            for key in GROUP_ORDER
            if group_counts.get(key)
        ],
        "nodes": sorted(
            nodes.values(),
            key=lambda n: (
                n["semester"] if n["semester"] is not None else 99,
                GROUP_ORDER.index(n["group"]),
                n["course_code"],
            ),
        ),
        "edges": edges,
        "knowledge_edges": knowledge_edges,
        "knowledge_available": concept_links is not None,
        "stats": {
            "course_count": len(in_plan),
            "external_count": len(nodes) - len(in_plan),
            "prerequisite_edge_count": len(edges),
            "linked_course_count": len(linked & in_plan),
            "knowledge_edge_count": len(knowledge_edges),
            "knowledge_edge_total": knowledge_total,
            "knowledge_linked_count": len(knowledge_linked),
            "longest_chain": chain,
        },
    }


def corpus_overview(
    courses: list[Mapping[str, Any]],
    *,
    concept_links: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Headline numbers for the overview page, all derived from artefacts."""
    codes = {str(course.get("course_code") or "") for course in courses}
    prerequisite_edges = {
        (str(source), str(course.get("course_code")))
        for course in courses
        for source in course.get("prerequisites") or []
        if str(source) in codes
    }
    catalog = program_catalog(courses)
    majors = {
        major["name"]
        for college in catalog["colleges"]
        for major in college["majors"]
    }
    return {
        "course_count": len(codes - {""}),
        "program_count": catalog["program_count"],
        "major_count": len(majors),
        "college_count": len(catalog["colleges"]),
        "prerequisite_edge_count": len(prerequisite_edges),
        "concept_count": concept_links.get("concept_count") if concept_links else None,
        "verified_relation_count": (
            concept_links.get("verified_relation_count") if concept_links else None
        ),
    }
