"""S3 课程对关系闸门 + R3 关系问法路由 + S4 _handle_dependency 拒答透传。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.online_service import dependency_explainer
from src.online_service.query_router import QueryRouter, QueryType, RouteResult


def _course(code: str, name: str, prerequisites: list[str] | None = None, **extra: Any):
    return {
        "course_code": code,
        "course_name": name,
        "prerequisites": prerequisites or [],
        "offerings": [{"program_name": "测试方案", "source_file": "plan.xlsx"}],
        **extra,
    }


_COURSES: list[dict[str, Any]] = [
    _course("IM3137", "新媒体"),
    _course("IM3129", "大数据分析"),
    _course("IM3100", "大数据"),
    _course("IM101", "程序设计基础"),
    _course("IM102", "数据结构", ["IM101"]),
    _course("IM103", "算法分析", ["IM102"]),
    _course("IM104", "数据库原理", ["IM103"]),
    _course("IM105", "信息系统开发", ["IM104"]),
    _course("IM230", "数据可视化"),
    _course("IM152", "数据科学方法与应用"),
    # 「转自DOC」双胞胎：同名，没有 offerings。
    {"course_code": "IM900", "course_name": "数据结构 转自DOC", "prerequisites": []},
    _course("IM2105", "信息组织基础", english_name="Information Organization"),
]


class _ExplodingRetriever:
    calls = 0

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        type(self).calls += 1
        raise AssertionError("课程对关系不应触发检索")


class _Router(QueryRouter):
    def __init__(self, links: dict[str, Any] | None = None) -> None:
        self.courses = _COURSES
        self.vector_retriever = _ExplodingRetriever()
        self.links = links
        self.dispatched: list[str] = []

    def _concept_links(self) -> dict[str, Any] | None:
        return self.links

    async def _handle_single_intent(  # type: ignore[override]
        self, query_type: QueryType, query: str, **_kwargs: Any
    ) -> RouteResult:
        self.dispatched.append(query_type.value)
        return RouteResult("正常回答", query_type=query_type.value)


def _names(router: QueryRouter, query: str) -> list[str]:
    return [c["course_name"] for c in router._link_courses(query)]


# ---------------------------------------------------------------------------
# _link_courses / _link_course
# ---------------------------------------------------------------------------

def test_link_courses_longest_name_claims_its_span() -> None:
    router = _Router()
    # 「大数据」被「大数据分析」覆盖，不再单独命中。
    assert _names(router, "新媒体与大数据分析") == ["新媒体", "大数据分析"]
    # 覆盖区间之外的「大数据」照样命中。
    assert _names(router, "大数据和大数据分析") == ["大数据", "大数据分析"]


def test_link_courses_orders_by_appearance_and_matches_codes() -> None:
    router = _Router()
    assert _names(router, "大数据分析和新媒体") == ["大数据分析", "新媒体"]
    assert _names(router, "im3129 与新媒体") == ["大数据分析", "新媒体"]
    # 代码被更长的代码覆盖时，不再命中较短的代码。
    assert _names(router, "IM3129") == ["大数据分析"]


def test_link_courses_resolves_bracketed_alias() -> None:
    router = _Router()
    assert _names(router, "《Information Organization》和新媒体") == [
        "信息组织基础",
        "新媒体",
    ]


def test_link_courses_dedupes_twins_and_prefers_offerings() -> None:
    router = _Router()
    courses = router._link_courses("数据结构和数据结构有什么区别")
    assert [c["course_code"] for c in courses] == ["IM102"]


def test_link_course_keeps_single_course_semantics() -> None:
    router = _Router()
    assert router._link_course("新媒体与大数据分析")["course_name"] == "大数据分析"
    assert router._link_course("数据结构")["course_code"] == "IM102"
    assert router._link_course("《Information Organization》")["course_code"] == "IM2105"
    assert router._link_course("今天天气不错") is None


# ---------------------------------------------------------------------------
# 关系闸门
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "新媒体与大数据分析之间有什么关系",
        "新媒体和大数据分析有关系吗",
        "《新媒体》与《大数据分析》有什么联系",
    ],
)
def test_unrelated_courses_are_refused_without_retrieval(query: str) -> None:
    router = _Router()

    result = asyncio.run(router.route(query, "rel-none"))

    assert result.query_type == "dependency"
    assert result.answer == (
        "根据培养方案与课程图谱，《新媒体》与《大数据分析》之间没有先修关系，"
        "也没有经过验证的知识关联，无法说明两者存在关系。"
    )
    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "COURSE_RELATION_NOT_FOUND"
    assert result.metadata["course_codes"] == ["IM3137", "IM3129"]
    assert result.metadata["llm_used"] is False
    assert result.metadata["vector_hits"] == 0
    assert "persona" in result.metadata and "intent_primary" in result.metadata
    assert router.dispatched == []
    assert _ExplodingRetriever.calls == 0


def test_direct_prerequisite_direction_is_correct_both_ways() -> None:
    router = _Router()

    forward = asyncio.run(router.route("程序设计基础和数据结构有什么关系", "d1"))
    backward = asyncio.run(router.route("数据结构和程序设计基础有什么关系", "d2"))

    for result in (forward, backward):
        assert result.answer == "《程序设计基础》是《数据结构》的先修课程。"
        assert result.metadata["llm_used"] is False
        assert result.citations[0]["course_code"] == "IM102"
        assert "status" not in result.metadata


def test_indirect_prerequisite_shows_shortest_chain() -> None:
    router = _Router()

    result = asyncio.run(router.route("数据库原理与程序设计基础有什么关系", "ind"))

    assert "《程序设计基础》是《数据库原理》的间接先修课程" in result.answer
    assert "《程序设计基础》→《数据结构》→《算法分析》→《数据库原理》" in result.answer


def test_chain_deeper_than_three_edges_is_not_a_relation() -> None:
    router = _Router()

    # 程序设计基础 -> ... -> 信息系统开发 有 4 条边。
    result = asyncio.run(router.route("程序设计基础和信息系统开发有什么关系", "deep"))

    assert result.metadata["error_code"] == "COURSE_RELATION_NOT_FOUND"


def test_knowledge_link_only_lists_at_most_three_pairs() -> None:
    links = {
        "pairs": {
            ("IM230", "IM152"): ["a → b", "c → d", "e → f"],
            ("IM152", "IM230"): ["g → h"],
        }
    }
    router = _Router(links)

    result = asyncio.run(router.route("数据可视化和数据科学方法与应用有什么关系", "k"))

    assert "没有先修要求，但图谱中有经过验证的知识关联" in result.answer
    assert "a → b、c → d、e → f" in result.answer
    assert "g → h" not in result.answer
    assert result.answer.endswith("（由 LLM 验证的概念依赖汇总得出）。")
    assert result.metadata["relation_type"] == "knowledge"
    assert "status" not in result.metadata


def test_knowledge_link_in_reverse_direction_is_found() -> None:
    router = _Router({"pairs": {("IM152", "IM230"): ["g → h"]}})

    result = asyncio.run(router.route("数据可视化和数据科学方法与应用有联系吗", "kr"))

    assert "《数据科学方法与应用》到《数据可视化》：g → h" in result.answer


def test_prerequisite_takes_precedence_over_knowledge_link() -> None:
    router = _Router({"pairs": {("IM101", "IM102"): ["x → y"]}})

    result = asyncio.run(router.route("程序设计基础和数据结构有关联吗", "p"))

    assert result.answer == "《程序设计基础》是《数据结构》的先修课程。"


def test_relation_wording_alone_with_one_course_does_not_trigger_gate() -> None:
    router = _Router()

    result = asyncio.run(router.route("数据可视化和数据清洗有关系吗", "one"))

    assert result.query_type == "dependency"
    assert router.dispatched == ["dependency"]


def test_two_courses_without_relation_wording_skip_the_gate() -> None:
    router = _Router()

    asyncio.run(router.route("新媒体和大数据分析各有几学分", "plain"))

    assert router.dispatched  # 走原来的意图分发，没有被关系闸门截走


# ---------------------------------------------------------------------------
# R3
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "数据清洗和可视化有关系吗",
        "数据清洗和可视化有没有关系",
        "数据清洗和可视化有联系吗",
        "数据清洗和可视化有关联吗",
    ],
)
def test_relation_questions_classify_as_dependency(query: str) -> None:
    assert QueryRouter.classify_intent(query).primary is QueryType.DEPENDENCY


# ---------------------------------------------------------------------------
# S4: _handle_dependency 的 status 透传
# ---------------------------------------------------------------------------

class _Driver:
    def session(self) -> Any:
        class _Session:
            def __enter__(self_inner) -> Any:
                return object()

            def __exit__(self_inner, *_exc: Any) -> None:
                return None

        return _Session()


def test_dependency_not_found_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dependency_explainer, "build_dependency_answer", lambda *a, **k: None)

    result = asyncio.run(_Router()._handle_dependency("某问题", _Driver(), None))

    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "DEPENDENCY_NOT_FOUND"


def test_dependency_not_connected_passes_status_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dependency_explainer,
        "build_dependency_answer",
        lambda *a, **k: {
            "status": "refused",
            "error_code": "DEPENDENCY_NOT_CONNECTED",
            "explanation": "图谱中没有找到《X》与《Y》之间的依赖路径，无法说明两者存在关系。",
        },
    )

    result = asyncio.run(_Router()._handle_dependency("某问题", _Driver(), None))

    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "DEPENDENCY_NOT_CONNECTED"
    assert "无法说明两者存在关系" in result.answer


def test_dependency_service_failures_stay_degraded() -> None:
    no_driver = asyncio.run(_Router()._handle_dependency("某问题", None, None))
    assert no_driver.metadata["status"] == "degraded"

    class _Broken:
        def session(self) -> Any:
            raise RuntimeError("down")

    broken = asyncio.run(_Router()._handle_dependency("某问题", _Broken(), None))
    assert broken.metadata["status"] == "degraded"
    assert broken.metadata["error_code"] == "DEPENDENCY_QUERY_FAILED"
