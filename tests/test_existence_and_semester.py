"""R4 / Q4 路由部分：存在性问法、学期问法与学期展示、「这门课」不触发章节补齐。

全部离线：课程、文档树都是内存里构造的；检索器一旦被调用就报错，用来证明
这些问题走的是确定性路径（不检索、不调 LLM）。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.data_processing.doc_tree import build_doc_tree
from src.online_service.doc_tree import DocTree
from src.online_service.query_router import (
    QueryRouter,
    QueryType,
    RouteResult,
    _semester_display,
)

_LIB = "信息管理学院2025级图书馆学专业培养方案"
_ARC = "信息管理学院2025级档案学专业培养方案"
_ARC_MINOR = "信息管理学院2025级档案学专业辅修专业培养方案"
_ARC_MICRO = "信息管理学院2025级档案学专业辅修微专业培养方案"
_IMIS = "信息管理学院2025级信息管理与信息系统专业培养方案"


def _offering(
    program: str,
    program_type: str,
    semester: Any,
    *,
    credits: Any = 3,
    hours: Any = 54,
) -> dict[str, Any]:
    return {
        "program_name": program,
        "program_type": program_type,
        "semester": semester,
        "credits": credits,
        "hours": hours,
        "source_file": "02：主修专业使用.xlsx",
    }


_COURSES: list[dict[str, Any]] = [
    {
        "course_code": "IM214",
        "course_name": "电子文件管理",
        "credits": 3,
        "hours": 54,
        "semester": 4,
        "offerings": [
            _offering(_LIB, "主修专业", 4),
            _offering(_ARC, "主修专业", 4),
            _offering(_ARC_MICRO, "辅修微专业", 4),
            _offering(_ARC_MINOR, "辅修专业", 4),
        ],
    },
    {
        "course_code": "IM121",
        "course_name": "信息管理学基础",
        "offerings": [_offering(_IMIS, "主修专业", 1, credits=3, hours=48)],
    },
    {
        "course_code": "PE101",
        "course_name": "体育",
        "offerings": [
            _offering(_IMIS, "主修专业", "1-6", credits=1, hours=36),
            _offering(_LIB, "主修专业", "1-6", credits=1, hours=36),
        ],
    },
    {
        "course_code": "IM777",
        "course_name": "档案馆建筑",
        "offerings": [
            _offering(_LIB, "主修专业", 3),
            _offering(_ARC, "主修专业", 5),
        ],
    },
    {
        # 学分 / 学时缺失占位符：回答里不能出现「None」或「/ 学分」。
        "course_code": "IM888",
        "course_name": "文献计量",
        "offerings": [_offering(_IMIS, "主修专业", 6, credits="/", hours=None)],
    },
    {"course_code": "IM999", "course_name": "工匠书缘 转自DOC"},
]


class _ExplodingRetriever:
    calls = 0
    count = 0

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        type(self).calls += 1
        raise AssertionError("确定性路径不能检索")


class _Router(QueryRouter):
    """真实分发；hybrid 只记录被调用，用来验证「回退」而不连后端。"""

    def __init__(self) -> None:
        self.courses = _COURSES
        self.vector_retriever = _ExplodingRetriever()
        self.hybrid_calls: list[str] = []

    async def _handle_hybrid(self, query: str, *_args: Any, **_kwargs: Any):  # type: ignore[override]
        self.hybrid_calls.append(query)
        return RouteResult("hybrid", query_type="hybrid")


@pytest.fixture(autouse=True)
def _aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROGRAM_ALIASES", "信管=信息管理与信息系统")


def _route(query: str) -> tuple[RouteResult, _Router]:
    router = _Router()
    result = asyncio.run(router.route(query, "r4"))
    assert router.vector_retriever.calls == 0
    return result, router


# --------------------------------------------------------------------------- #
# 1) 存在性意图
# --------------------------------------------------------------------------- #

_EXISTENCE_QUERIES = (
    "有电子文件管理这门课吗",
    "本科生有电子文件管理这门课吗",
    "有没有电子文件管理课",
    "有没有电子文件管理这门课",
    "开不开电子文件管理",
    "电子文件管理这门课存在吗",
    "电子文件管理开不开",
    "有《电子文件管理》这门课吗",
    "请问有电子文件管理这门课么？",
)


@pytest.mark.parametrize("query", _EXISTENCE_QUERIES)
def test_existence_phrasings_route_to_existence(query: str) -> None:
    assert QueryRouter.classify(query) is QueryType.EXISTENCE


@pytest.mark.parametrize("query", _EXISTENCE_QUERIES)
def test_existing_course_is_answered_deterministically(query: str) -> None:
    result, _router = _route(query)

    assert result.answer.startswith("有。")
    assert "《电子文件管理》（IM214）" in result.answer
    assert "图书馆学、档案学专业培养方案" in result.answer
    assert "档案学辅修专业、辅修微专业也开设" in result.answer
    assert "第 4 学期（大二下学期）" in result.answer
    assert "3 学分" in result.answer
    assert "54 学时" in result.answer
    assert result.query_type == "existence"
    assert result.metadata["llm_used"] is False
    assert result.metadata["vector_hits"] == 0
    assert result.metadata["course_code"] == "IM214"
    assert result.citations and result.citations[0]["course_code"] == "IM214"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("电子文件管理有哪些先修课", QueryType.DEPENDENCY),
        ("电子文件管理和信息检索有什么关系", QueryType.DEPENDENCY),
        ("有没有先修课", QueryType.DEPENDENCY),
        ("电子文件管理有几个学分", QueryType.FACT),
        ("有哪些课程", None),
        ("信管专业有哪些必修课", QueryType.CATALOG),
        ("专业有哪些选修课", QueryType.CATALOG),
        ("有没有选修课", QueryType.FACT),
        ("档案学专业有多少门核心课", QueryType.CATALOG),
        ("这门课有什么特点", None),
    ],
)
def test_existence_patterns_do_not_steal_other_questions(
    query: str, expected: QueryType | None
) -> None:
    prediction = QueryRouter.classify_intent(query)

    assert prediction.primary is not QueryType.EXISTENCE
    if expected is not None:
        assert prediction.primary is expected


def test_attribute_question_about_a_known_course_falls_back_to_hybrid() -> None:
    """「X 有实验课吗」问的是课程属性，不是课程是否存在。"""
    result, router = _route("电子文件管理有实验课吗")

    assert result.answer == "hybrid"
    assert router.hybrid_calls == ["电子文件管理有实验课吗"]


def test_program_clue_without_the_course_names_where_it_is_offered() -> None:
    result, _router = _route("信管专业有电子文件管理这门课吗")

    assert not result.answer.startswith("有。")
    assert result.answer.startswith(
        "信息管理与信息系统专业培养方案中没有《电子文件管理》（IM214）这门课；它开设在"
    )
    assert "图书馆学、档案学专业培养方案中" in result.answer
    assert "第 4 学期（大二下学期）" in result.answer
    assert result.metadata["llm_used"] is False
    assert result.metadata["course_code"] == "IM214"


def test_program_clue_with_the_course_answers_for_that_program_only() -> None:
    result, _router = _route("档案学专业有电子文件管理这门课吗")

    assert result.answer.startswith("有。")
    assert "档案学专业培养方案中的课程" in result.answer
    assert "图书馆学" not in result.answer


def test_program_clue_alias_matching_the_course_says_yes() -> None:
    result, _router = _route("信管有信息管理学基础这门课吗")

    assert result.answer.startswith("有。")
    assert "信息管理与信息系统专业培养方案中的课程" in result.answer
    assert "第 1 学期（大一上学期）" in result.answer


def test_unknown_course_without_book_title_is_refused() -> None:
    result, _router = _route("有机器学习这门课吗")

    assert result.answer == "培养方案中没有名为《机器学习》的课程。"
    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "COURSE_NOT_FOUND"
    assert result.metadata["llm_used"] is False
    assert result.metadata["vector_hits"] == 0


def test_unknown_course_with_program_prefix_is_refused_by_course_name() -> None:
    result, _router = _route("信管专业有机器学习这门课吗")

    assert result.answer == "培养方案中没有名为《机器学习》的课程。"


def test_similar_course_name_without_book_title_asks_for_clarification() -> None:
    result, _router = _route("有没有信息管理基础这门课")

    assert result.answer.startswith("你是不是想问")
    assert "《信息管理学基础》" in result.answer
    assert result.metadata["status"] == "clarify"
    assert result.metadata["error_code"] == "COURSE_AMBIGUOUS"


def test_course_without_offerings_is_not_claimed_to_be_in_a_plan() -> None:
    result, _router = _route("有工匠书缘这门课吗")

    assert result.answer.startswith("有。")
    assert "培养方案里没有找到" in result.answer


def test_missing_credit_and_hour_values_are_skipped() -> None:
    result, _router = _route("有文献计量这门课吗")

    assert result.answer.startswith("有。")
    assert "第 6 学期（大三下学期）" in result.answer
    assert "学分" not in result.answer
    assert "学时" not in result.answer
    assert "None" not in result.answer


def test_existence_answer_lists_each_value_when_plans_disagree() -> None:
    result, _router = _route("有档案馆建筑这门课吗")

    assert "第 3 学期（大二上学期）" in result.answer
    assert "第 5 学期（大三上学期）" in result.answer


# --------------------------------------------------------------------------- #
# 2) 学期问法
# --------------------------------------------------------------------------- #

_SEMESTER_QUERIES = (
    "电子文件管理这门课是大几才开",
    "电子文件管理在大几开设",
    "电子文件管理这门课在大几开设",
    "电子文件管理什么时候上",
    "电子文件管理什么时候开",
    "电子文件管理这门课哪一年开",
    "电子文件管理开设时间是什么",
    "电子文件管理的开课时间",
    "电子文件管理一般几年级学",
    "电子文件管理何时开",
    "电子文件管理在第几学期开",
)


@pytest.mark.parametrize("query", _SEMESTER_QUERIES)
def test_semester_phrasings_route_to_fact_and_hit_the_semester_field(
    query: str,
) -> None:
    assert QueryRouter.classify(query) is QueryType.FACT

    result, _router = _route(query)

    assert result.query_type == "fact"
    assert "第 4 学期（大二下学期）" in result.answer
    assert result.metadata["llm_used"] is False


@pytest.mark.parametrize(
    "query",
    ["电子文件管理考试什么时候", "电子文件管理期末考试什么时候", "电子文件管理考试时间"],
)
def test_exam_time_is_still_refused_by_s2(query: str) -> None:
    result, _router = _route(query)

    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "FIELD_NOT_RECORDED"
    assert "考试具体日期/时间" in result.answer


def test_semester_patterns_do_not_hijack_exam_phrasing() -> None:
    # 「什么时候考试」经 S2 的封闭字段进入 fact，但不能被当成问开课学期。
    import re

    from src.online_service.query_router import _FACT_FIELD_PATTERNS

    semester = dict(_FACT_FIELD_PATTERNS)["semester"]
    assert not re.search(semester, "电子文件管理什么时候考试")


# --------------------------------------------------------------------------- #
# 3) 学期展示
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "第 1 学期（大一上学期）"),
        (2, "第 2 学期（大一下学期）"),
        (3, "第 3 学期（大二上学期）"),
        (4, "第 4 学期（大二下学期）"),
        (5, "第 5 学期（大三上学期）"),
        (6, "第 6 学期（大三下学期）"),
        (7, "第 7 学期（大四上学期）"),
        (8, "第 8 学期（大四下学期）"),
        (4.0, "第 4 学期（大二下学期）"),
        ("4", "第 4 学期（大二下学期）"),
        ("1-6", "第 1–6 学期均可修读（大一至大三）"),
        ("1-4", "第 1–4 学期均可修读（大一至大二）"),
        ("1-8", "第 1–8 学期均可修读（大一至大四）"),
        ("1-2", "第 1–2 学期均可修读（大一）"),
        ("春季", "春季"),
    ],
)
def test_semester_display(value: Any, expected: str) -> None:
    assert _semester_display(value) == expected


def test_range_semester_is_shown_as_flexible_window() -> None:
    result, _router = _route("体育什么时候上")

    assert "第 1–6 学期均可修读（大一至大三）" in result.answer


def test_inconsistent_semesters_are_listed_with_each_converted() -> None:
    result, _router = _route("档案馆建筑在大几开设")

    assert "各培养方案不一致" in result.answer
    assert "第 3 学期（大二上学期）" in result.answer
    assert "第 5 学期（大三上学期）" in result.answer


def test_program_clue_picks_one_semester() -> None:
    result, _router = _route("档案学专业的档案馆建筑什么时候上")

    assert "第 5 学期（大三上学期）" in result.answer
    assert "第 3 学期" not in result.answer


# --------------------------------------------------------------------------- #
# 4) Q4：「这门课」不再触发整门课补齐
# --------------------------------------------------------------------------- #


def _doc_tree(course_code: str) -> DocTree:
    schedule = ["四、教学进度"]
    chunks = [
        {
            "chunk_id": f"c{index}",
            "document_hash": "hash",
            "chunk_index": index,
            "heading_path": schedule + [f"第{name}章 章节{index}"],
            "text": f"章节次序及名称：第{name}章 章节{index}\n学时：2",
            "source_file": "x.docx",
            "metadata": {
                "source_type": "syllabus",
                "section_type": "teaching_schedule",
                "course_code": course_code,
                "course_name": "电子文件管理",
                "source_year": "2025",
            },
        }
        for index, name in enumerate("一二三", 1)
    ]
    return DocTree(build_doc_tree(chunks))


@pytest.mark.parametrize(
    "query",
    [
        "有电子文件管理这门课吗",
        "电子文件管理这门课是大几才开",
        "电子文件管理这门课几个学分",
        "这门课的考核方式是什么",
    ],
)
def test_this_course_alone_no_longer_triggers_chapter_coverage(query: str) -> None:
    course = _COURSES[0]

    items, coverage = QueryRouter._apply_chapter_coverage(
        query, [], course, _doc_tree("IM214")
    )

    assert coverage is None
    assert not [item for item in items if item.get("coverage_filler")]


@pytest.mark.parametrize(
    "query",
    [
        "电子文件管理这门课讲什么",
        "介绍一下这门课",
        "概括一下这门课",
        "介绍一下电子文件管理",
        "电子文件管理的课程内容",
    ],
)
def test_whole_course_phrasings_still_get_chapter_coverage(query: str) -> None:
    course = _COURSES[0]

    items, coverage = QueryRouter._apply_chapter_coverage(
        query, [], course, _doc_tree("IM214")
    )

    assert coverage == "0/3"
    assert len([item for item in items if item.get("coverage_filler")]) == 3


def test_existence_question_never_reaches_evidence_assembly() -> None:
    """存在性问题在 route 内就答完了：不检索、不组装证据，自然没有 coverage_filler。"""
    result, router = _route("有电子文件管理这门课吗")

    assert result.query_type == "existence"
    assert "chapter_coverage" not in result.metadata
    assert router.hybrid_calls == []


def test_exam_time_in_question_first_order_is_refused_by_s2():
    """「什么时候考试」与「考试什么时候」同属未记录字段，语序不影响拒答。"""
    router = _Router()
    for query in ("电子文件管理什么时候考试", "电子文件管理何时期末考试"):
        gate = router._unrecorded_field_gate(query)
        assert gate is not None, query
        assert gate.metadata["error_code"] == "FIELD_NOT_RECORDED"
