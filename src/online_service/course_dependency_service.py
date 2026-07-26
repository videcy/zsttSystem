"""Deterministic course-prerequisite neighborhood queries and pruning."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


class CourseDependencyNotFoundError(LookupError):
    """Raised when the requested course does not exist in Neo4j."""


_PATH_QUERIES = {
    depth: f"""
        MATCH p=(center:Course {{course_code: $course_code}})
            -[:PREREQUISITE_OF*1..{depth}]-(neighbor:Course)
        RETURN
            [node IN nodes(p) | {{
                course_code: node.course_code,
                course_name: node.course_name,
                semester: node.semester,
                offerings: node.offerings
            }}] AS nodes,
            [rel IN relationships(p) | {{
                source: startNode(rel).course_code,
                target: endNode(rel).course_code,
                relation: 'PREREQUISITE_OF',
                confidence: coalesce(rel.confidence, 1.0),
                verified: coalesce(rel.verified, false),
                source_file: rel.source_file,
                section: rel.section
            }}] AS edges
        LIMIT 500
    """
    for depth in (1, 2, 3)
}

_ALL_PREREQUISITES_QUERY = """
    MATCH (source:Course)-[rel:PREREQUISITE_OF]->(target:Course)
    RETURN
        source {
            .course_code,
            .course_name,
            .semester,
            .offerings
        } AS source_node,
        target {
            .course_code,
            .course_name,
            .semester,
            .offerings
        } AS target_node,
        {
            source: source.course_code,
            target: target.course_code,
            relation: 'PREREQUISITE_OF',
            confidence: coalesce(rel.confidence, 1.0),
            verified: coalesce(rel.verified, false),
            source_file: rel.source_file,
            section: rel.section
        } AS edge
"""


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "data"):
        return dict(value.data())
    return dict(value)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _offerings(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = node.get("offerings") or []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _official_semester(
    node: Mapping[str, Any],
    program_name: str | None,
) -> Any:
    if program_name:
        for offering in _offerings(node):
            if offering.get("program_name") == program_name:
                return offering.get("semester")
        return None
    return node.get("semester")


def _find_cycle(
    codes: set[str],
    adjacency: Mapping[str, set[str]],
) -> list[str]:
    state: dict[str, int] = {}
    stack: list[str] = []
    positions: dict[str, int] = {}

    def visit(code: str) -> list[str]:
        state[code] = 1
        positions[code] = len(stack)
        stack.append(code)
        for target in sorted(adjacency.get(code, set())):
            if target not in codes:
                continue
            if state.get(target, 0) == 0:
                cycle = visit(target)
                if cycle:
                    return cycle
            elif state.get(target) == 1:
                return [*stack[positions[target] :], target]
        stack.pop()
        positions.pop(code, None)
        state[code] = 2
        return []

    for code in sorted(codes):
        if state.get(code, 0) == 0:
            cycle = visit(code)
            if cycle:
                return cycle
    return []


def build_prerequisite_plan(
    target_course: str,
    nodes: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
    *,
    program_name: str | None = None,
) -> dict[str, Any]:
    """Build prerequisite layers with Kahn sorting; never infer missing edges."""
    node_by_code = {
        str(node.get("course_code", "")): dict(node)
        for node in nodes
        if node.get("course_code")
    }
    target_course = target_course.strip()
    if target_course not in node_by_code:
        raise ValueError("target course must exist in planning nodes")

    hard_edges = [
        dict(edge)
        for edge in edges
        if edge.get("relation") == "PREREQUISITE_OF"
        and edge.get("source") in node_by_code
        and edge.get("target") in node_by_code
    ]
    incoming: dict[str, set[str]] = {}
    adjacency: dict[str, set[str]] = {}
    for edge in hard_edges:
        source = str(edge["source"])
        target = str(edge["target"])
        incoming.setdefault(target, set()).add(source)
        adjacency.setdefault(source, set()).add(target)

    relevant = {target_course}
    frontier = [target_course]
    while frontier:
        target = frontier.pop()
        for source in incoming.get(target, set()):
            if source not in relevant:
                relevant.add(source)
                frontier.append(source)

    relevant_edges = [
        edge
        for edge in hard_edges
        if edge["source"] in relevant and edge["target"] in relevant
    ]
    indegree = {code: 0 for code in relevant}
    for edge in relevant_edges:
        indegree[str(edge["target"])] += 1
    queue = sorted(code for code, count in indegree.items() if count == 0)
    topological: list[str] = []
    while queue:
        code = queue.pop(0)
        topological.append(code)
        for target in sorted(adjacency.get(code, set())):
            if target not in indegree:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()

    if len(topological) != len(relevant):
        cycle = _find_cycle(relevant, adjacency)
        return {
            "status": "cycle_detected",
            "is_dag": False,
            "program_name": program_name,
            "available_programs": [],
            "layers": [],
            "warnings": ["检测到硬先修环，已停止生成选课顺序。"],
            "cycle": cycle,
        }

    target_programs = sorted(
        {
            str(offering.get("program_name"))
            for offering in _offerings(node_by_code[target_course])
            if offering.get("program_name")
        }
    )
    if not relevant_edges:
        return {
            "status": "no_prerequisites",
            "is_dag": True,
            "program_name": None,
            "available_programs": target_programs,
            "layers": [],
            "warnings": ["未找到明确的硬先修课程。"],
            "cycle": [],
        }
    if program_name and target_programs and program_name not in target_programs:
        return {
            "status": "program_not_found",
            "is_dag": True,
            "program_name": program_name,
            "available_programs": target_programs,
            "layers": [],
            "warnings": ["目标课程不属于所选培养方案。"],
            "cycle": [],
        }
    if not program_name and len(target_programs) > 1:
        return {
            "status": "program_required",
            "is_dag": True,
            "program_name": None,
            "available_programs": target_programs,
            "layers": [],
            "warnings": ["目标课程属于多个培养方案，请先选择方案再规划。"],
            "cycle": [],
        }
    selected_program = (
        program_name
        or (target_programs[0] if len(target_programs) == 1 else None)
    )

    layer_by_code: dict[str, int] = {}
    for code in topological:
        predecessors = incoming.get(code, set()) & relevant
        layer_by_code[code] = (
            max(layer_by_code[source] for source in predecessors) + 1
            if predecessors
            else 1
        )

    warnings: list[str] = []
    for edge in relevant_edges:
        source = str(edge["source"])
        target = str(edge["target"])
        source_semester = _number(
            _official_semester(node_by_code[source], selected_program)
        )
        target_semester = _number(
            _official_semester(node_by_code[target], selected_program)
        )
        if (
            source_semester is not None
            and target_semester is not None
            and source_semester >= target_semester
        ):
            warnings.append(
                f"{source} 的开课学期不早于 {target}，请核对培养方案。"
            )

    layers = []
    for stage in sorted(set(layer_by_code.values())):
        stage_codes = sorted(
            code for code, value in layer_by_code.items() if value == stage
        )
        layers.append(
            {
                "stage": stage,
                "courses": [
                    {
                        "course_code": code,
                        "label": node_by_code[code].get("course_name") or code,
                        "official_semester": _official_semester(
                            node_by_code[code],
                            selected_program,
                        ),
                    }
                    for code in stage_codes
                ],
            }
        )

    return {
        "status": "ok",
        "is_dag": True,
        "program_name": selected_program,
        "available_programs": target_programs,
        "layers": layers,
        "warnings": warnings,
        "cycle": [],
    }


def build_course_dependency_subgraph(
    center: Mapping[str, Any],
    path_rows: Iterable[Mapping[str, Any]],
    *,
    depth: int = 2,
    max_nodes: int = 30,
    planning_nodes: Iterable[Mapping[str, Any]] | None = None,
    planning_edges: Iterable[Mapping[str, Any]] | None = None,
    program_name: str | None = None,
) -> dict[str, Any]:
    """Deduplicate and deterministically prune path records returned by Neo4j."""
    if depth not in _PATH_QUERIES:
        raise ValueError("depth must be one of 1, 2, or 3")
    if not 1 <= max_nodes <= 30:
        raise ValueError("max_nodes must be between 1 and 30")

    center = dict(center)
    center_code = str(center.get("course_code", ""))
    if not center_code:
        raise ValueError("center course_code is required")

    nodes: dict[str, dict[str, Any]] = {center_code: center}
    distances: dict[str, int] = {center_code: 0}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in path_rows:
        row_nodes = [_as_dict(item) for item in row.get("nodes", [])]
        for distance, node in enumerate(row_nodes):
            code = str(node.get("course_code", ""))
            if not code:
                continue
            nodes.setdefault(code, node)
            distances[code] = min(distances.get(code, distance), distance)
        for raw_edge in row.get("edges", []):
            edge = _as_dict(raw_edge)
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            relation = str(edge.get("relation") or "PREREQUISITE_OF")
            if source and target:
                edges[(source, target, relation)] = edge

    incoming: dict[str, set[str]] = {}
    for source, target, _relation in edges:
        incoming.setdefault(target, set()).add(source)
    prerequisite_codes: set[str] = set()
    frontier = [center_code]
    while frontier:
        target = frontier.pop()
        for source in incoming.get(target, set()):
            if source not in prerequisite_codes:
                prerequisite_codes.add(source)
                frontier.append(source)

    verified_count: dict[str, int] = {}
    best_confidence: dict[str, float] = {}
    for edge in edges.values():
        confidence = _number(edge.get("confidence")) or 0.0
        for code in (str(edge.get("source", "")), str(edge.get("target", ""))):
            if edge.get("verified"):
                verified_count[code] = verified_count.get(code, 0) + 1
            best_confidence[code] = max(
                best_confidence.get(code, 0.0),
                confidence,
            )

    target_semester = _number(center.get("semester"))

    def rank(code: str) -> tuple[Any, ...]:
        semester = _number(nodes[code].get("semester"))
        semester_gap = (
            abs(semester - target_semester)
            if semester is not None and target_semester is not None
            else float("inf")
        )
        return (
            0 if code == center_code else 1,
            distances.get(code, depth + 1),
            0 if code in prerequisite_codes else 1,
            -verified_count.get(code, 0),
            -best_confidence.get(code, 0.0),
            semester_gap,
            code,
        )

    ordered_codes = sorted(nodes, key=rank)
    retained_codes = ordered_codes[:max_nodes]
    retained = set(retained_codes)
    output_nodes = [
        {
            "id": code,
            "course_code": code,
            "label": nodes[code].get("course_name") or code,
            "semester": nodes[code].get("semester"),
            "distance": distances.get(code, 0 if code == center_code else None),
        }
        for code in retained_codes
    ]
    output_edges = [
        {
            key: value
            for key, value in edge.items()
            if value is not None
        }
        for edge in edges.values()
        if str(edge.get("source")) in retained
        and str(edge.get("target")) in retained
    ]
    output_edges.sort(
        key=lambda edge: (
            distances.get(str(edge.get("target")), depth + 1),
            str(edge.get("source")),
            str(edge.get("target")),
        )
    )
    plan = build_prerequisite_plan(
        center_code,
        planning_nodes if planning_nodes is not None else nodes.values(),
        planning_edges if planning_edges is not None else edges.values(),
        program_name=program_name,
    )
    return {
        "target_course": center_code,
        "nodes": output_nodes,
        "edges": output_edges,
        "depth": depth,
        "max_nodes": max_nodes,
        "truncated": len(nodes) > max_nodes,
        "total_nodes": len(nodes),
        "plan": plan,
    }


def get_course_dependency_subgraph(
    driver: Any,
    course_code: str,
    *,
    depth: int = 2,
    max_nodes: int = 30,
    program_name: str | None = None,
) -> dict[str, Any]:
    """Fetch a bounded k-hop course neighborhood from Neo4j."""
    if depth not in _PATH_QUERIES:
        raise ValueError("depth must be one of 1, 2, or 3")
    if not 1 <= max_nodes <= 30:
        raise ValueError("max_nodes must be between 1 and 30")

    canonical_code = course_code.strip().upper()
    with driver.session() as session:
        record = session.run(
            """
            MATCH (course:Course {course_code: $course_code})
            RETURN course {
                .course_code,
                .course_name,
                .semester,
                .offerings
            } AS course
            """,
            course_code=canonical_code,
        ).single()
        if record is None:
            raise CourseDependencyNotFoundError(canonical_code)
        center = _as_dict(record["course"])
        rows = [
            {
                "nodes": list(row["nodes"]),
                "edges": list(row["edges"]),
            }
            for row in session.run(
                _PATH_QUERIES[depth],
                course_code=canonical_code,
            )
        ]
        planning_nodes: dict[str, dict[str, Any]] = {
            canonical_code: center
        }
        planning_edges: list[dict[str, Any]] = []
        for row in session.run(_ALL_PREREQUISITES_QUERY):
            source_node = _as_dict(row["source_node"])
            target_node = _as_dict(row["target_node"])
            planning_nodes[str(source_node["course_code"])] = source_node
            planning_nodes[str(target_node["course_code"])] = target_node
            planning_edges.append(_as_dict(row["edge"]))
    return build_course_dependency_subgraph(
        center,
        rows,
        depth=depth,
        max_nodes=max_nodes,
        planning_nodes=planning_nodes.values(),
        planning_edges=planning_edges,
        program_name=program_name,
    )
