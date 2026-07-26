from __future__ import annotations

import pytest

from src.online_service.course_dependency_service import (
    build_course_dependency_subgraph,
    build_prerequisite_plan,
)


def test_dependency_subgraph_is_pruned_deterministically() -> None:
    center = {
        "course_code": "IM300",
        "course_name": "目标课程",
        "semester": 5,
    }
    path_rows = [
        {
            "nodes": [
                center,
                {
                    "course_code": "IM200",
                    "course_name": "近先修",
                    "semester": 3,
                },
                {
                    "course_code": "IM100",
                    "course_name": "远先修",
                    "semester": 1,
                },
            ],
            "edges": [
                {
                    "source": "IM200",
                    "target": "IM300",
                    "relation": "PREREQUISITE_OF",
                    "confidence": 1.0,
                    "verified": True,
                },
                {
                    "source": "IM100",
                    "target": "IM200",
                    "relation": "PREREQUISITE_OF",
                    "confidence": 1.0,
                    "verified": True,
                },
            ],
        },
        {
            "nodes": [
                center,
                {
                    "course_code": "IM250",
                    "course_name": "后续课程",
                    "semester": 6,
                },
            ],
            "edges": [
                {
                    "source": "IM300",
                    "target": "IM250",
                    "relation": "PREREQUISITE_OF",
                    "confidence": 1.0,
                    "verified": True,
                }
            ],
        },
    ]

    payload = build_course_dependency_subgraph(
        center,
        path_rows,
        depth=2,
        max_nodes=3,
    )

    assert [node["course_code"] for node in payload["nodes"]] == [
        "IM300",
        "IM200",
        "IM250",
    ]
    assert all(
        edge["source"] in {"IM300", "IM200", "IM250"}
        and edge["target"] in {"IM300", "IM200", "IM250"}
        for edge in payload["edges"]
    )
    assert payload["truncated"] is True
    assert payload["total_nodes"] == 4
    assert [
        [course["course_code"] for course in layer["courses"]]
        for layer in payload["plan"]["layers"]
    ] == [["IM100"], ["IM200"], ["IM300"]]


def test_prerequisite_plan_keeps_parallel_courses_in_the_same_layer() -> None:
    nodes = [
        {"course_code": "A", "course_name": "基础甲", "semester": 1},
        {"course_code": "B", "course_name": "基础乙", "semester": 1},
        {"course_code": "C", "course_name": "中间课程", "semester": 2},
        {"course_code": "T", "course_name": "目标课程", "semester": 3},
    ]
    edges = [
        {"source": "A", "target": "C", "relation": "PREREQUISITE_OF"},
        {"source": "B", "target": "C", "relation": "PREREQUISITE_OF"},
        {"source": "C", "target": "T", "relation": "PREREQUISITE_OF"},
    ]

    plan = build_prerequisite_plan("T", nodes, edges)

    assert plan["is_dag"] is True
    assert plan["status"] == "ok"
    assert [
        [course["course_code"] for course in layer["courses"]]
        for layer in plan["layers"]
    ] == [["A", "B"], ["C"], ["T"]]
    assert all("official_semester" in course for layer in plan["layers"] for course in layer["courses"])


def test_prerequisite_plan_stops_when_a_cycle_is_present() -> None:
    nodes = [
        {"course_code": "A", "course_name": "课程 A"},
        {"course_code": "B", "course_name": "课程 B"},
        {"course_code": "T", "course_name": "目标课程"},
    ]
    edges = [
        {"source": "A", "target": "B", "relation": "PREREQUISITE_OF"},
        {"source": "B", "target": "A", "relation": "PREREQUISITE_OF"},
        {"source": "B", "target": "T", "relation": "PREREQUISITE_OF"},
    ]

    plan = build_prerequisite_plan("T", nodes, edges)

    assert plan["is_dag"] is False
    assert plan["status"] == "cycle_detected"
    assert plan["layers"] == []
    assert plan["cycle"][0] == plan["cycle"][-1]


def test_prerequisite_plan_does_not_mix_multiple_programs() -> None:
    nodes = [
        {
            "course_code": "A",
            "course_name": "基础课",
            "offerings": [
                {"program_name": "方案一", "semester": 1},
                {"program_name": "方案二", "semester": 2},
            ],
        },
        {
            "course_code": "T",
            "course_name": "目标课程",
            "offerings": [
                {"program_name": "方案一", "semester": 3},
                {"program_name": "方案二", "semester": 4},
            ],
        },
    ]
    edges = [
        {"source": "A", "target": "T", "relation": "PREREQUISITE_OF"},
    ]

    pending = build_prerequisite_plan("T", nodes, edges)
    selected = build_prerequisite_plan(
        "T",
        nodes,
        edges,
        program_name="方案二",
    )

    assert pending["status"] == "program_required"
    assert pending["layers"] == []
    assert pending["available_programs"] == ["方案一", "方案二"]
    assert [
        course["official_semester"]
        for layer in selected["layers"]
        for course in layer["courses"]
    ] == [2, 4]


def test_prerequisite_plan_warns_about_semester_conflicts() -> None:
    plan = build_prerequisite_plan(
        "T",
        [
            {"course_code": "A", "course_name": "先修课", "semester": 4},
            {"course_code": "T", "course_name": "目标课", "semester": 3},
        ],
        [{"source": "A", "target": "T", "relation": "PREREQUISITE_OF"}],
    )

    assert plan["is_dag"] is True
    assert any("开课学期" in warning for warning in plan["warnings"])


def test_prerequisite_plan_reports_no_dependencies_without_program_choice() -> None:
    plan = build_prerequisite_plan(
        "T",
        [
            {
                "course_code": "T",
                "course_name": "目标课",
                "offerings": [
                    {"program_name": "方案一", "semester": 1},
                    {"program_name": "方案二", "semester": 2},
                ],
            }
        ],
        [],
    )

    assert plan["status"] == "no_prerequisites"
    assert plan["is_dag"] is True
    assert plan["layers"] == []


@pytest.mark.parametrize(
    ("depth", "max_nodes"),
    [(0, 30), (4, 30), (2, 0), (2, 31)],
)
def test_dependency_subgraph_rejects_unsafe_limits(
    depth: int,
    max_nodes: int,
) -> None:
    with pytest.raises(ValueError):
        build_course_dependency_subgraph(
            {"course_code": "IM300", "course_name": "目标课程"},
            [],
            depth=depth,
            max_nodes=max_nodes,
        )
