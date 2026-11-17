from __future__ import annotations

import asyncio

import pytest

from src.online_service.query_router import QueryRouter, _clarify_candidates

COURSES = [{"course_code": "X1", "course_name": "数据结构"}]


class _NoRetriever:
    def search(self, *args, **kwargs):
        raise AssertionError("澄清追问不应检索")


def _router(retriever=None) -> QueryRouter:
    router = QueryRouter(vector_retriever=retriever or _NoRetriever())
    router.courses = list(COURSES)
    return router


@pytest.mark.parametrize("query", ["这门课难吗", "要学编程吗", "老师怎么样"])
def test_vague_query_without_course_is_blocked(query: str) -> None:
    result = asyncio.run(_router()._handle_hybrid(query, None))

    assert result.metadata["status"] == "clarify"
    assert result.metadata["clarify"] == "block"
    assert result.metadata["llm_used"] is False
    assert len(result.metadata["clarify_candidates"]) == 3
    assert all("《课程名》" in c for c in result.metadata["clarify_candidates"])


def test_teacher_question_leads_with_teacher_candidate() -> None:
    assert _clarify_candidates("老师怎么样", "数据结构")[0] == "《数据结构》的任课老师是谁？"


def test_candidates_route_to_answerable_paths() -> None:
    router = _router()
    for text in _clarify_candidates("数据结构难吗", "数据结构") + _clarify_candidates(
        "老师怎么样", "数据结构"
    ):
        assert router.classify_intent(text).primary.value != "hybrid", text


def test_program_level_question_without_course_is_not_blocked() -> None:
    class EmptyRetriever:
        def search(self, *args, **kwargs):
            return []

    router = _router(EmptyRetriever())
    result = asyncio.run(router._handle_hybrid("本专业的培养目标是什么", None))

    assert result.metadata.get("status") != "clarify"
    assert "clarify_candidates" not in result.metadata


def test_linked_course_answers_and_suggests() -> None:
    from tests.test_generation_layer import FakeRetriever, _fake_client

    router = _router(FakeRetriever())
    result = asyncio.run(router._handle_hybrid("数据结构难吗", _fake_client("")))

    assert result.metadata.get("status") != "clarify"
    assert result.answer
    assert result.metadata["clarify"] == "suggest"
    assert result.metadata["clarify_candidates"][0] == "《数据结构》的主要内容是什么？"
