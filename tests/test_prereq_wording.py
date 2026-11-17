from __future__ import annotations

import asyncio

import pytest

from src.online_service.query_router import QueryRouter

COURSES = [
    {"course_code": "IM152", "course_name": "数据科学方法与应用", "prerequisites": ["IM129"]},
    {"course_code": "IM230", "course_name": "数据可视化", "prerequisites": ["IM152"]},
]


def _router() -> QueryRouter:
    router = QueryRouter(vector_retriever=None)
    router.courses = list(COURSES)
    return router


@pytest.mark.parametrize("query", ["学数据可视化要先学什么", "数据可视化要先修什么"])
def test_prerequisite_wording_uses_programme_prerequisites(query: str) -> None:
    router = _router()
    assert router.classify_intent(query).labels == (router.classify_intent(query).primary,)
    assert router.classify_intent(query).primary.value == "dependency"

    result = asyncio.run(router._handle_dependency(query, None, None))

    assert result.answer == "数据可视化的先修课程为：数据科学方法与应用（IM152）。"


def test_why_question_still_goes_to_concept_graph() -> None:
    result = asyncio.run(
        _router()._handle_dependency("为什么要先学数据科学方法与应用再学数据可视化", None, None)
    )

    assert result.metadata.get("error_code") == "NEO4J_UNAVAILABLE"
