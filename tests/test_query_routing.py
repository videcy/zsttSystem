from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

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


def test_assessment_phrasings_route_to_fact() -> None:
    for query in (
        "《中外档案事业史》怎么考核？",
        "《信息管理学基础》如何考核？",
        "《信息管理学基础》怎么考试？",
        "《信息管理学基础》考核形式是什么？",
        "《信息管理学基础》考核办法是什么？",
    ):
        prediction = QueryRouter.classify_intent(query)
        assert prediction.primary is QueryType.FACT, query
        assert prediction.labels == (QueryType.FACT,), query


def test_course_names_do_not_trigger_dependency_patterns() -> None:
    for query in (
        "《信息组织基础》主要讲什么内容？",
        "《核聚变工程基础》主要讲什么内容？",
        "《档案学基础》有哪些内容？",
    ):
        prediction = QueryRouter.classify_intent(query)
        assert prediction.primary is QueryType.CONTENT, query
        assert prediction.labels == (QueryType.CONTENT,), query


def test_basis_outside_book_title_still_routes_to_dependency() -> None:
    for query in (
        "学《信息组织基础》需要什么基础？",
        "学《X》需要什么基础？",
    ):
        assert QueryRouter.classify(query) is QueryType.DEPENDENCY, query


def test_routing_regressions_keep_their_original_precedence() -> None:
    expected = {
        "信管专业要修哪些专业必修课？": QueryType.CATALOG,
        "管理运筹学多少学分？": QueryType.FACT,
        "《信息素养与创新思维》第三章讲什么？": QueryType.STRUCTURE,
        "管理运筹学几学分，有哪些先修课？": QueryType.DEPENDENCY,
        "《信息组织基础》主要讲什么内容？": QueryType.CONTENT,
    }

    for query, query_type in expected.items():
        assert QueryRouter.classify(query) is query_type, query


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


class _VerificationRouter(_StubRouter):
    """Stub whose handlers carry NLI metadata like the real ones do."""

    def __init__(self, per_intent: dict[QueryType, dict[str, Any]]) -> None:
        super().__init__()
        self.per_intent = per_intent

    async def _handle_single_intent(  # type: ignore[override]
        self,
        query_type: QueryType,
        query: str,
        **_kwargs: Any,
    ) -> RouteResult:
        self.calls.append(query_type.value)
        return RouteResult(
            answer=self.answers[query_type],
            citations=[],
            query_type=query_type.value,
            metadata=dict(self.per_intent.get(query_type, {})),
        )


def test_merged_answer_lifts_verification_metadata_to_the_top_level() -> None:
    router = _VerificationRouter(
        {
            QueryType.DEPENDENCY: {
                "nli_status": "passed",
                "nli_verified": True,
                "nli_attempts": 1,
                "nli_details": [{"sentence": "先修", "label": "Entailment"}],
                "status": "ok",
            },
            QueryType.FACT: {
                "nli_status": "pruned",
                "nli_verified": True,
                "nli_attempts": 2,
                "nli_details": [{"sentence": "学分", "label": "Neutral"}],
                "status": "ok",
            },
        }
    )

    result = asyncio.run(router.route("管理运筹学几学分，有哪些先修课？", "q-9"))

    # main._log_query and the active-learning sampler read these keys from the
    # top level; nesting them per section would drop merged answers out of the
    # review queue entirely.
    assert result.metadata["nli_status"] == "pruned"
    assert result.metadata["nli_attempts"] == 3
    assert len(result.metadata["nli_details"]) == 2
    assert result.metadata["nli_verified"] is True
    assert result.metadata["status"] == "ok"


def test_merged_answer_reports_the_worst_section_status() -> None:
    router = _VerificationRouter(
        {
            QueryType.DEPENDENCY: {"status": "degraded", "error_code": "NO_NEO4J"},
            QueryType.FACT: {"status": "ok", "nli_status": "passed"},
        }
    )

    result = asyncio.run(router.route("管理运筹学几学分，有哪些先修课？", "q-10"))

    assert result.metadata["status"] == "ok"
    assert result.metadata["partial_status"] == ["degraded"]


def test_merged_answer_stays_degraded_when_every_section_failed() -> None:
    router = _VerificationRouter(
        {
            QueryType.DEPENDENCY: {"status": "degraded", "error_code": "NO_NEO4J"},
            QueryType.FACT: {"status": "refused"},
        }
    )

    result = asyncio.run(router.route("管理运筹学几学分，有哪些先修课？", "q-11"))

    assert result.metadata["status"] == "degraded"
    assert "partial_status" not in result.metadata


# --------------------------------------------------------------------------- #
# P1-6：structure 路由样本（改造方案 §4 M4.1 / §7）。
# 只追加，上面既有样本与断言一律不动。
# --------------------------------------------------------------------------- #

_STRUCTURE_SAMPLES = (
    "《信息素养与创新思维》第三章讲什么？",
    "这门课一共几章？",
    "第 3 周学什么？",
    "课程的教学进度安排是什么？",
)


def test_structure_samples_route_to_structure() -> None:
    for query in _STRUCTURE_SAMPLES:
        assert QueryRouter.classify(query) is QueryType.STRUCTURE, query


def test_structure_patterns_do_not_steal_fact_samples() -> None:
    # 考核方式 / 学分属于 FACT；「第 3 周」这类结构词面不能把它们抢走。
    assert QueryRouter.classify("这门课的考核方式是什么？") is QueryType.FACT
    assert QueryRouter.classify("管理运筹学多少学分？") is QueryType.FACT


def test_structure_patterns_leave_the_original_170_routes_intact() -> None:
    """新增 STRUCTURE 不能把原有评测样本抢走（§7「确认原 170 题路由结果不变」）。

    ``gold_seed.json`` 是 ``eval/`` 的产物，不进仓库时跳过；只要存在就断言：
    没有任何 expected_route ≠ structure 的样本被新结构词面劫持到 structure。
    """
    dataset = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "datasets"
        / "gold_seed.json"
    )
    if not dataset.is_file():
        pytest.skip("gold_seed.json 不随仓库分发")

    items = json.loads(dataset.read_text(encoding="utf-8"))["items"]
    stolen = [
        item["id"]
        for item in items
        if QueryRouter.classify(item["question"]) is QueryType.STRUCTURE
        and item.get("expected_route") != "structure"
    ]
    assert stolen == []


def test_teacher_phrasings_route_to_fact() -> None:
    for query in ("管理运筹学的老师是谁", "信息计量学哪位老师上", "数据库原理的主讲教师"):
        assert QueryRouter.classify(query) is QueryType.FACT, query


def test_chapter_summary_drops_heading_path_and_table_labels() -> None:
    from types import SimpleNamespace

    node = SimpleNamespace(
        title="第1章 数字信息组织理论：概述",
        text=(
            "（一）教学进度表 / 第1章 数字信息组织理论：概述\n"
            "章节：第1章 数字信息组织理论：概述 内容：1数字信息环境变化 2数字信息组织理论 "
            "学时：2 育人元素：无"
        ),
    )

    summary = QueryRouter._chapter_summary(None, node)

    assert summary == "1数字信息环境变化 2数字信息组织理论"
