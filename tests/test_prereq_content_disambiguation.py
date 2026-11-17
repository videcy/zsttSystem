from __future__ import annotations

import asyncio

import pytest

from src.online_service.query_router import QueryRouter, QueryType, RouteResult

COURSES = [
    {"course_code": "IM152", "course_name": "数据科学方法与应用", "prerequisites": ["IM129"]},
    {"course_code": "IM230", "course_name": "数据可视化", "prerequisites": ["IM152"]},
]


def _router() -> QueryRouter:
    router = QueryRouter(vector_retriever=None)
    router.courses = list(COURSES)
    return router


@pytest.mark.parametrize(
    "query",
    ["学数据可视化之前学什么", "学数据可视化之前先学什么", "数据可视化之前要学什么"],
)
def test_prerequisite_cue_routes_to_dependency_only(query: str) -> None:
    router = _router()
    assert router.classify_intent(query).labels == (QueryType.DEPENDENCY,)

    result = asyncio.run(router._handle_dependency(query, None, None))
    assert result.answer == "数据可视化的先修课程为：数据科学方法与应用（IM152）。"


def test_content_question_is_unchanged() -> None:
    assert QueryRouter.classify_intent("数据可视化要学什么").labels == (QueryType.CONTENT,)


def test_both_candidates_route_cleanly() -> None:
    assert QueryRouter.classify_intent("《数据可视化》的先修课程是什么？").labels == (
        QueryType.DEPENDENCY,
    )
    assert QueryRouter.classify_intent("《数据可视化》的主要内容是什么？").labels == (
        QueryType.CONTENT,
    )


def test_dependency_plus_content_answer_suggests_both_readings() -> None:
    router = _router()

    async def fake_single(query_type, query, **kwargs):
        return RouteResult(f"{query_type.value} 部分", query_type=query_type.value)

    router._handle_single_intent = fake_single
    result = asyncio.run(
        router._handle_multi_intent(
            "数据可视化讲什么，需要什么基础",
            (QueryType.DEPENDENCY, QueryType.CONTENT),
            neo4j_driver=None,
            llm_client=None,
            persona="student",
        )
    )

    assert result.metadata["clarify"] == "suggest"
    assert result.metadata["clarify_candidates"] == [
        "《数据可视化》的先修课程是什么？",
        "《数据可视化》的主要内容是什么？",
    ]
