from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from src.online_service.query_router import QueryRouter, QueryType, RouteResult


_DATASET_PATH = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "gold_seed.json"
_TEXTBOOK_PATTERN = ("教材", "参考书", "课本", "教科书", "指定用书")

pytestmark = pytest.mark.skipif(
    not _DATASET_PATH.exists(),
    reason="Q3 audit requires the generated gold_seed dataset",
)


class _ExplodingRetriever:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def count(self) -> int:
        return 0

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        self.calls += 1
        raise AssertionError("Q3 audit must not retrieve")


class _Q3AuditRouter(QueryRouter):
    def __init__(self) -> None:
        self.courses = [
            {
                "course_code": "IM3132",
                "course_name": "关联数据及应用",
            }
        ]
        self.calls: list[str] = []
        self.vector_retriever = _ExplodingRetriever()

    async def _handle_single_intent(  # type: ignore[override]
        self,
        query_type: QueryType,
        query: str,
        **_kwargs: Any,
    ) -> RouteResult:
        self.calls.append(query_type.value)
        return RouteResult("正常回答", query_type=query_type.value)


def _items() -> list[dict[str, Any]]:
    return json.loads(_DATASET_PATH.read_text(encoding="utf-8"))["items"]


def test_current_dataset_has_no_formal_textbook_question() -> None:
    questions = [str(item.get("question") or "") for item in _items()]

    assert questions
    assert not [
        question
        for question in questions
        if any(pattern in question for pattern in _TEXTBOOK_PATTERN)
    ]


def test_noanswer_023_is_now_a_classroom_question() -> None:
    item = next(item for item in _items() if item["id"] == "noanswer-023")

    assert item["question"] == "《关联数据及应用》的上课教室在哪里？"
    assert item["answerable"] is False
    assert item["notes"] == "课程存在但语料不记录该字段"


def test_noanswer_023_deterministically_refuses_without_textbook_answer() -> None:
    item = next(item for item in _items() if item["id"] == "noanswer-023")
    question = str(item["question"])
    router = _Q3AuditRouter()

    first = asyncio.run(router.route(question, "q3-repro-1"))
    second = asyncio.run(router.route(question, "q3-repro-2"))

    assert QueryRouter.classify(question) is QueryType.FACT
    for result in (first, second):
        assert result.answer == "培养方案与教学大纲均未记录《关联数据及应用》的上课地点/教室。"
        assert "课程教材" not in result.answer
        assert result.metadata["status"] == "refused"
        assert result.metadata["error_code"] == "FIELD_NOT_RECORDED"
        assert result.metadata["field"] == "上课地点/教室"
        assert result.metadata["vector_hits"] == 0
    assert first.answer == second.answer
    assert router.calls == []
    assert router.vector_retriever.calls == 0
