from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.online_service.query_router import QueryRouter, QueryType, RouteResult


_COURSES: list[dict[str, Any]] = [
    {
        "course_code": "IM121",
        "course_name": "信息管理学基础",
        "english_name": "Foundations of Information Management",
        "aliases": ["信管基础"],
    },
    {"course_code": "IM129", "course_name": "信息管理编程基础"},
    {"course_code": "IM2105", "course_name": "信息组织基础"},
    {"course_code": "IM282", "course_name": "信息检索"},
    {"course_code": "IM397", "course_name": "文书学"},
    {"course_code": "IM390", "course_name": "公共图书馆与公共文化服务"},
    {"course_code": "IM439", "course_name": "信息用户与参考咨询服务"},
    {"course_code": "IM230", "course_name": "数据可视化"},
    {"course_code": "IM3132", "course_name": "关联数据及应用"},
    {"course_code": "IM255", "course_name": "公共文化服务导论"},
    {"course_code": "IM394", "course_name": "目录学"},
    {
        "course_code": "IM999",
        "course_name": "工匠书缘：古籍的修复与文化传承 转自DOC",
    },
]


class _ExplodingRetriever:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def count(self) -> int:
        return 0

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        self.calls += 1
        raise AssertionError("safety gates must not retrieve")


class _SafetyRouter(QueryRouter):
    def __init__(self) -> None:
        self.courses = _COURSES
        self.calls: list[str] = []
        self.vector_retriever = _ExplodingRetriever()

    async def _handle_single_intent(  # type: ignore[override]
        self,
        query_type: QueryType,
        query: str,
        **_kwargs: Any,
    ) -> RouteResult:
        self.calls.append(query_type.value)
        return RouteResult(
            "正常回答",
            query_type=query_type.value,
            metadata={"status": "ok"},
        )


def test_absent_course_is_refused_before_dispatch() -> None:
    router = _SafetyRouter()

    result = asyncio.run(router.route("《火星种植学》有几学分？", "s1-refused"))

    assert result.answer == "培养方案中没有名为《火星种植学》的课程。"
    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "COURSE_NOT_FOUND"
    assert router.calls == []
    assert router.vector_retriever.calls == 0


def test_similar_course_name_asks_for_clarification() -> None:
    router = _SafetyRouter()

    result = asyncio.run(router.route("《信息管理基础》有几学分？", "s1-clarify"))

    assert result.answer.startswith("你是不是想问")
    assert "《信息管理学基础》" in result.answer
    assert result.metadata["status"] == "clarify"
    assert result.metadata["error_code"] == "COURSE_AMBIGUOUS"
    assert 1 <= len(result.metadata["candidates"]) <= 3
    assert router.calls == []
    assert router.vector_retriever.calls == 0


def test_no_book_title_does_not_trigger_course_gate() -> None:
    router = _SafetyRouter()

    result = asyncio.run(router.route("火星种植学有几学分？", "s1-no-title"))

    assert result.answer == "正常回答"
    assert router.calls == ["fact"]
    assert "error_code" not in result.metadata


@pytest.mark.parametrize(
    "query",
    [
        "《信息管理学基础》有几学分？",
        "《信管基础》有几学分？",
        "《IM121》有几学分？",
        "《工匠书缘：古籍的修复与文化传承》有几学分？",
    ],
)
def test_known_course_alias_code_and_long_punctuated_name_are_not_rejected(
    query: str,
) -> None:
    router = _SafetyRouter()

    result = asyncio.run(router.route(query, "s1-known"))

    assert result.answer == "正常回答"
    assert router.calls == ["fact"]
    assert result.metadata.get("status") == "ok"


def test_similarity_threshold_is_environment_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COURSE_NAME_SIMILARITY_THRESHOLD", "0.99")
    router = _SafetyRouter()

    result = asyncio.run(router.route("《信息管理基础》有几学分？", "s1-threshold"))

    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "COURSE_NOT_FOUND"


_MISSING_FIELD_CASES = (
    (
        "《文书学》的上课教室在哪里？",
        "文书学",
        "上课地点/教室",
    ),
    (
        "《公共图书馆与公共文化服务》的期末考试具体日期是哪天？",
        "公共图书馆与公共文化服务",
        "考试具体日期/时间",
    ),
    (
        "《信息用户与参考咨询服务》去年的平均分是多少？",
        "信息用户与参考咨询服务",
        "历年成绩/平均分/及格率",
    ),
    ("《信息管理学基础》的上课教室在哪里？", "信息管理学基础", "上课地点/教室"),
    ("《信息检索》的期末考试具体日期是哪天？", "信息检索", "考试具体日期/时间"),
    ("《数据可视化》去年的平均分是多少？", "数据可视化", "历年成绩/平均分/及格率"),
    ("《关联数据及应用》的上课教室在哪里？", "关联数据及应用", "上课地点/教室"),
    (
        "《公共文化服务导论》的期末考试具体日期是哪天？",
        "公共文化服务导论",
        "考试具体日期/时间",
    ),
    ("《目录学》去年的平均分是多少？", "目录学", "历年成绩/平均分/及格率"),
)


@pytest.mark.parametrize("query,course_name,field", _MISSING_FIELD_CASES)
def test_nine_noanswer_field_questions_are_deterministic_refusals(
    query: str,
    course_name: str,
    field: str,
) -> None:
    assert QueryRouter.classify(query) is QueryType.FACT
    router = _SafetyRouter()

    result = asyncio.run(router.route(query, "s2-refused"))

    assert result.answer == f"培养方案与教学大纲均未记录《{course_name}》的{field}。"
    assert result.query_type == "fact"
    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "FIELD_NOT_RECORDED"
    assert result.metadata["field"] == field
    assert result.metadata["vector_hits"] == 0
    assert router.calls == []
    assert router.vector_retriever.calls == 0


def test_unrecorded_field_gate_runs_after_course_validation() -> None:
    router = _SafetyRouter()

    result = asyncio.run(
        router.route("《火星种植学》的上课教室在哪里？", "s1-before-s2")
    )

    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "COURSE_NOT_FOUND"
    assert router.calls == []


def test_unrecorded_field_gate_also_handles_unbracketed_known_course() -> None:
    router = _SafetyRouter()

    result = asyncio.run(router.route("文书学上课教室在哪里？", "s2-no-title"))

    assert result.answer == "培养方案与教学大纲均未记录《文书学》的上课地点/教室。"
    assert result.metadata["error_code"] == "FIELD_NOT_RECORDED"
    assert router.calls == []


@pytest.mark.parametrize(
    "query,field",
    [
        ("《文书学》上课地点在哪？", "上课地点/教室"),
        ("《文书学》期末考试时间是什么时候？", "考试具体日期/时间"),
        ("《文书学》历年的及格率是多少？", "历年成绩/平均分/及格率"),
        ("《文书学》的选课人数是多少？", "选课人数"),
    ],
)
def test_closed_unrecorded_field_list_is_covered(query: str, field: str) -> None:
    router = _SafetyRouter()

    result = asyncio.run(router.route(query, "s2-closed-list"))

    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "FIELD_NOT_RECORDED"
    assert result.metadata["field"] == field
    assert router.calls == []
    assert router.vector_retriever.calls == 0
