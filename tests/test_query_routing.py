from __future__ import annotations

import asyncio
from typing import Any

from src.online_service.query_router import (
    LOW_CONFIDENCE_THRESHOLD,
    QueryRouter,
    QueryType,
    RouteResult,
)


def test_single_intent_keeps_full_confidence() -> None:
    prediction = QueryRouter.classify_intent("数据库原理多少学分？")

    assert prediction.primary is QueryType.FACT
    assert prediction.labels == (QueryType.FACT,)
    assert prediction.confidence == 1.0
    assert prediction.is_multi_intent is False


def test_unmatched_query_falls_through_to_hybrid() -> None:
    prediction = QueryRouter.classify_intent("比较数据库与档案管理的差异")

    assert prediction.primary is QueryType.HYBRID
    assert prediction.confidence == 0.0


def test_multi_intent_query_keeps_both_labels() -> None:
    prediction = QueryRouter.classify_intent("管理运筹学几学分，有哪些先修课？")

    assert prediction.primary is QueryType.DEPENDENCY
    assert set(prediction.labels) == {QueryType.DEPENDENCY, QueryType.FACT}
    assert prediction.confidence < LOW_CONFIDENCE_THRESHOLD
    assert prediction.as_metadata()["intent_labels"] == ["dependency", "fact"]


def test_program_scoped_question_routes_to_catalog_not_fact() -> None:
    # "必修" alone matches the fact patterns; the program cue must win.
    assert QueryRouter.classify("信管专业要修哪些专业必修课？") is QueryType.CATALOG
    assert QueryRouter.classify("数据库原理是必修课吗") is QueryType.FACT


def test_natural_semester_phrasings_route_to_fact() -> None:
    for query in (
        "《信息检索》在第几学期开课？",
        "文献计量学什么学期上？",
        "管理运筹学哪个学期开",
    ):
        assert QueryRouter.classify(query) is QueryType.FACT


class _StubRouter(QueryRouter):
    """Router with every backend handler replaced by a recorded stub."""

    def __init__(self) -> None:  # noqa: D107 - test double
        self.courses = []
        self.calls: list[str] = []
        self.answers = {
            QueryType.FACT: "学分为 2",
            QueryType.DEPENDENCY: "先修课程：信息管理学基础",
            QueryType.CONTENT: "课程内容概述",
            QueryType.CATALOG: "培养方案课程列表",
            QueryType.HYBRID: "综合回答",
        }

    async def _handle_single_intent(  # type: ignore[override]
        self,
        query_type: QueryType,
        query: str,
        **_kwargs: Any,
    ) -> RouteResult:
        self.calls.append(query_type.value)
        return RouteResult(
            answer=self.answers[query_type],
            citations=[{"chunk_id": f"{query_type.value}-1"}],
            query_type=query_type.value,
            metadata={"handler": query_type.value},
        )


def test_route_merges_a_fact_and_dependency_question() -> None:
    router = _StubRouter()

    result = asyncio.run(router.route("管理运筹学几学分，有哪些先修课？", "q-1"))

    assert router.calls == ["dependency", "fact"]
    assert "【先修关系】" in result.answer and "【课程基本信息】" in result.answer
    assert result.query_type == "dependency+fact"
    assert len(result.citations) == 2
    assert result.metadata["answered_intents"] == ["dependency", "fact"]
    assert result.metadata["intent_confidence"] < LOW_CONFIDENCE_THRESHOLD


def test_route_leaves_single_intent_queries_untouched() -> None:
    router = _StubRouter()

    result = asyncio.run(router.route("数据库原理多少学分？", "q-2"))

    assert router.calls == ["fact"]
    assert result.answer == "学分为 2"
    assert result.query_type == "fact"
    assert result.metadata["intent_confidence"] == 1.0


def test_unmergeable_pair_above_threshold_uses_the_primary_intent() -> None:
    router = _StubRouter()

    # fact + content is not a mergeable pair (the content handler already
    # surfaces basic information), and the evidence splits evenly, so the
    # winning intent answers alone.
    prediction = QueryRouter.classify_intent("这门课的教材是什么，主要内容包括哪些")
    assert set(prediction.labels) == {QueryType.FACT, QueryType.CONTENT}

    result = asyncio.run(
        router.route("这门课的教材是什么，主要内容包括哪些", "q-3")
    )

    assert router.calls == ["fact"]
    assert "low_confidence_fallback" not in result.metadata


def test_unmergeable_pair_below_threshold_falls_back_to_hybrid() -> None:
    router = _StubRouter()
    query = "信管专业的必修课有几学分，学时多少？"

    prediction = QueryRouter.classify_intent(query)
    assert prediction.labels[:2] == (QueryType.CATALOG, QueryType.FACT)
    assert prediction.confidence < LOW_CONFIDENCE_THRESHOLD

    result = asyncio.run(router.route(query, "q-5"))

    assert router.calls == ["hybrid"]
    assert result.metadata["low_confidence_fallback"] is True


def test_multi_intent_drops_empty_sections() -> None:
    router = _StubRouter()
    router.answers[QueryType.FACT] = "   "

    result = asyncio.run(router.route("管理运筹学几学分，有哪些先修课？", "q-4"))

    assert result.metadata["answered_intents"] == ["dependency"]
    assert "【课程基本信息】" not in result.answer


class _FallbackRetriever:
    """Retriever whose top hit is unrelated to the question."""

    connected = True

    def __init__(self, lexical_score: float | None) -> None:
        self.lexical_score = lexical_score

    @property
    def count(self) -> int:
        return 1

    def search(self, _query: str, _top_k: int = 5, **_kwargs: Any) -> list[dict[str, Any]]:
        hit = {
            "chunk_id": "unrelated",
            "text": "3. 当代保护理论. 上海：同济大学出版社，2012.12",
            "score": 0.4,
            "vector_score": 0.4,
            "metadata": {"course_code": "IM214", "source_type": "syllabus"},
        }
        if self.lexical_score is not None:
            hit["lexical_score"] = self.lexical_score
        return [hit]


def _fact_router(lexical_score: float | None) -> QueryRouter:
    router = QueryRouter.__new__(QueryRouter)
    router.courses = []
    router.vector_retriever = _FallbackRetriever(lexical_score)
    return router


def test_fact_fallback_refuses_when_evidence_shares_no_terms() -> None:
    result = asyncio.run(_fact_router(0.0).route("《火星种植学》有几学分？", "q-6"))

    assert "未找到足够相关" in result.answer
    assert result.metadata["error_code"] == "FACT_NO_RELEVANT_EVIDENCE"
    assert result.citations == []


def test_fact_fallback_answers_when_evidence_overlaps() -> None:
    result = asyncio.run(_fact_router(0.4).route("《保护理论》有几学分？", "q-7"))

    assert "当代保护理论" in result.answer
    assert result.citations


def test_fact_fallback_stays_permissive_without_a_lexical_score() -> None:
    # A retriever that reports no lexical score is not evidence of irrelevance.
    result = asyncio.run(_fact_router(None).route("《保护理论》有几学分？", "q-8"))

    assert "当代保护理论" in result.answer
