from __future__ import annotations

from typing import Any

import pytest

from src.online_service.program_index import ProgramIndex, core_program_name
from src.online_service.query_router import QueryRouter


def _offering(program_name: str, program_type: str, **extra: Any) -> dict[str, Any]:
    return {"program_name": program_name, "program_type": program_type, **extra}


# A different university entirely: no 信息管理学院, no 档案学, different plan
# vocabulary. Nothing about it appears anywhere in the source tree.
OTHER_SCHOOL = [
    {
        "course_code": "ME101",
        "course_name": "工程制图",
        "offerings": [
            _offering(
                "机械工程学院2026级机械设计制造及其自动化专业培养方案",
                "主修专业",
                course_category="专业必修课",
                course_subcategory="专业核心课",
                semester=1,
            )
        ],
    },
    {
        "course_code": "ME220",
        "course_name": "机器人学导论",
        "offerings": [
            _offering(
                "机械工程学院2026级机械设计制造及其自动化专业辅修专业培养方案",
                "辅修专业",
                course_category="辅修课程",
                semester=3,
            )
        ],
    },
    {
        "course_code": "CE110",
        "course_name": "土木工程材料",
        "offerings": [
            _offering(
                "土木工程学院2026级土木工程专业培养方案",
                "主修专业",
                course_category="专业必修课",
                course_subcategory="专业核心课",
                semester=2,
            )
        ],
    },
]


def test_core_name_strips_plan_title_affixes() -> None:
    types = ("辅修微专业", "辅修专业", "主修专业")

    assert (
        core_program_name("信息管理学院2025级档案学专业培养方案", types) == "档案学"
    )
    assert (
        core_program_name(
            "信息管理学院2025级档案学专业辅修微专业培养方案", types
        )
        == "档案学"
    )
    assert (
        core_program_name("机械工程学院2026级机械设计制造及其自动化专业培养方案", types)
        == "机械设计制造及其自动化"
    )
    # A title without the usual affixes survives unchanged.
    assert core_program_name("图书情报与档案管理类", types) == "图书情报与档案管理类"


def test_index_learns_names_and_types_from_the_data() -> None:
    index = ProgramIndex(OTHER_SCHOOL)

    assert set(index.core_names) == {"机械设计制造及其自动化", "土木工程"}
    assert set(index.program_types) == {"主修专业", "辅修专业"}
    assert index.match_keyword("土木工程专业有哪些课") == "土木工程"
    assert index.match_keyword("机器人相关的课") == ""


def test_alias_beats_the_derived_names() -> None:
    index = ProgramIndex(OTHER_SCHOOL, aliases={"机自": "机械设计制造及其自动化"})

    assert index.match_keyword("机自的核心课程") == "机械设计制造及其自动化"


def test_full_plan_title_is_always_matchable() -> None:
    index = ProgramIndex(OTHER_SCHOOL)

    resolved = index.match_keyword("土木工程学院2026级土木工程专业培养方案有哪些课")

    assert resolved in {"土木工程", "土木工程学院2026级土木工程专业培养方案"}


def test_named_plan_type_wins_over_the_majority_type() -> None:
    index = ProgramIndex(OTHER_SCHOOL)
    offerings = [
        offering
        for course in OTHER_SCHOOL
        for offering in course["offerings"]
    ]

    assert index.match_type("机械设计制造及其自动化辅修专业的课", offerings) == "辅修专业"
    # Unqualified: the main plan is whichever type carries the most offerings.
    assert index.match_type("机械设计制造及其自动化的课", offerings) == "主修专业"


def test_match_type_without_offerings_is_unconstrained() -> None:
    assert ProgramIndex([]).match_type("任意问题") == ""


def _router(courses: list[dict[str, Any]]) -> QueryRouter:
    router = QueryRouter.__new__(QueryRouter)
    router.courses = courses
    router.vector_retriever = None
    return router


def test_catalog_answers_for_a_school_that_appears_nowhere_in_the_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PROGRAM_ALIASES", raising=False)
    router = _router(OTHER_SCHOOL)

    result = router._handle_catalog("土木工程专业的核心课程有哪些")

    assert "土木工程材料（CE110）" in result.answer
    assert "工程制图" not in result.answer


def test_catalog_respects_a_minor_plan_of_another_school(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PROGRAM_ALIASES", raising=False)
    router = _router(OTHER_SCHOOL)

    main_plan = router._handle_catalog("机械设计制造及其自动化专业的核心课程有哪些")
    minor_plan = router._handle_catalog("机械设计制造及其自动化辅修专业有哪些课程")

    assert "工程制图（ME101）" in main_plan.answer
    assert "机器人学导论" not in main_plan.answer
    assert "机器人学导论（ME220）" in minor_plan.answer


def test_fact_offering_selection_follows_the_named_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PROGRAM_ALIASES", raising=False)
    course = {
        "course_code": "ME101",
        "course_name": "工程制图",
        "offerings": [
            _offering("机械工程学院2026级机械设计制造及其自动化专业培养方案", "主修专业", credits=4),
            _offering(
                "机械工程学院2026级机械设计制造及其自动化专业辅修专业培养方案",
                "辅修专业",
                credits=2,
            ),
        ],
    }
    router = _router([course])

    main_offering = router._select_offering(course, "工程制图有几学分")
    minor_offering = router._select_offering(course, "辅修专业的工程制图有几学分")

    assert main_offering["credits"] == 4
    assert minor_offering["credits"] == 2


def test_route_rebuilds_the_index_when_courses_are_replaced() -> None:
    router = _router([])
    assert router.program_index.core_names == ()

    router.courses = OTHER_SCHOOL

    assert "土木工程" in router.program_index.core_names


def test_original_corpus_behaviour_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The four names that used to be literals still resolve from the data."""
    monkeypatch.delenv("PROGRAM_ALIASES", raising=False)
    courses = [
        {
            "course_code": "IM104",
            "course_name": "档案学概论",
            "offerings": [
                _offering("信息管理学院2025级档案学专业培养方案", "主修专业"),
                _offering(
                    "信息管理学院2025级档案学专业辅修微专业培养方案", "辅修微专业"
                ),
            ],
        }
    ]
    index = ProgramIndex(courses)

    assert index.match_keyword("档案学专业的核心课程") == "档案学"
    assert index.match_type("档案学辅修微专业的课程", []) == "辅修微专业"
    assert index.match_type("档案学专业的课程", courses[0]["offerings"]) in {
        "主修专业",
        "辅修微专业",
    }


def test_alias_resolution_is_case_of_configuration_not_code() -> None:
    """No school-specific program name may reappear as a literal in the router."""
    from pathlib import Path

    source = Path("src/online_service/query_router.py").read_text(encoding="utf-8")

    for literal in ("信息管理与信息系统", "图书情报与档案管理类", "图书馆学", "档案学"):
        assert literal not in source
