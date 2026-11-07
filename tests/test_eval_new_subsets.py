"""M5.1 / M5.2 / P0-5 评测扩展的离线单元测试。

全部使用构造产物（写到 ``tmp_path``）与假 router，不依赖 ``data/``、
``outputs/``、ChromaDB 或网络，因此可以在无外网环境下快速运行。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from eval import build_seed_dataset as seed
from eval import metrics, run_eval
from eval.schema import GoldItem, load_dataset, save_dataset


# ---------------------------------------------------------------------------
# 构造产物
# ---------------------------------------------------------------------------

COURSES = [
    {"course_code": "IM101", "course_name": "数据结构", "instructor": "/", "credits": 3},
    {"course_code": "IM102", "course_name": "数据库原理", "instructor": "张三", "credits": 2},
    {"course_code": "IM104", "course_name": "计算机网络", "instructor": None, "credits": 4},
]

CHUNKS = [
    {"chunk_id": "c-IM101", "metadata": {"source_type": "syllabus", "course_code": "IM101"}},
    {"chunk_id": "c-IM102", "metadata": {"source_type": "syllabus", "course_code": "IM102"}},
    {"chunk_id": "c-IM104", "metadata": {"source_type": "syllabus", "course_code": "IM104"}},
]


def _chapter_node(
    number: Any, title: str, chunk_ids: list[str], kind: str = "chapter"
) -> dict[str, Any]:
    return {
        "level": 1,
        "title": title,
        "kind": kind,
        "chapter_no": number,
        "children": [],
        "chunk_ids": chunk_ids,
        "hours": 2,
        "text": f"{title} 的内容",
        "section_type": None,
    }


DOC_TREE = {
    "version": 1,
    "documents": {
        "h1": {
            "course_code": "IM101",
            "course_name": "数据结构",
            "source_file": "a.docx",
            "source_year": "2024",
            "chunk_ids": ["c-IM101-0", "c-IM101-1", "c-IM101-2"],
            "nodes": {
                "n0": _chapter_node(None, "教学进度", ["c-IM101-0"], kind="schedule"),
                "n1": _chapter_node(1, "第一章 绪论", ["c-IM101-1"]),
                "n2": _chapter_node(2, "第二章 线性表", ["c-IM101-2"]),
            },
            "roots": ["n0"],
            "chapter_count": 2,
            "non_chapter_rows": [],
            "header_chunk_ids": [],
        },
        "h2": {
            "course_code": "IM102",
            "course_name": "数据库原理",
            "source_file": "b.docx",
            "source_year": "2024",
            "chunk_ids": ["c-IM102-1", "c-IM102-2", "c-IM102-3"],
            "nodes": {
                "m1": _chapter_node(1, "第一章 数据库概论", ["c-IM102-1"]),
                "m2": _chapter_node(2, "第二章 关系模型", ["c-IM102-2"]),
                "m3": _chapter_node(3, "第三章 SQL", ["c-IM102-3"]),
            },
            "roots": ["m1"],
            "chapter_count": 3,
            "non_chapter_rows": [],
            "header_chunk_ids": [],
        },
    },
}

PROFILES = {
    "IM101|2024": {
        "course_code": "IM101",
        "course_name": "数据结构",
        "course_owner": "李四",
        "lecturers": ["王五"],
        "assessment_mode": "闭卷考试",
        "field_chunk_ids": {"course_owner": "c-IM101", "assessment_mode": "c-IM101"},
    },
    "IM102|2024": {
        "course_code": "IM102",
        "course_name": "数据库原理",
        "course_owner": "张三",
        "lecturers": [],
        "assessment_mode": "课程论文",
        "field_chunk_ids": {},
    },
}


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _point_env_at_fixtures(
    monkeypatch: Any,
    tmp_path: Path,
    *,
    doc_tree: bool = True,
    profiles: bool = True,
) -> None:
    monkeypatch.setenv(
        "COURSES_OUTPUT_PATH", str(_write_json(tmp_path / "courses.json", COURSES))
    )
    monkeypatch.setenv(
        "CHUNKS_OUTPUT_PATH", str(_write_json(tmp_path / "chunks.json", CHUNKS))
    )
    doc_path = (
        _write_json(tmp_path / "doc_tree.json", DOC_TREE)
        if doc_tree
        else tmp_path / "missing_doc_tree.json"
    )
    profile_path = (
        _write_json(tmp_path / "profiles.json", PROFILES)
        if profiles
        else tmp_path / "missing_profiles.json"
    )
    monkeypatch.setenv("DOC_TREE_PATH", str(doc_path))
    monkeypatch.setenv("SYLLABUS_PROFILES_PATH", str(profile_path))


# ---------------------------------------------------------------------------
# M5.1 子集生成与优雅降级
# ---------------------------------------------------------------------------

def test_structure_route_is_valid() -> None:
    problems = GoldItem(
        id="s-0",
        question="《X》第 1 章讲什么？",
        expected_route="structure",
        answer_keys=["绪论"],
    ).validate()

    assert problems == []


def test_build_generates_new_subsets(monkeypatch: Any, tmp_path: Path) -> None:
    _point_env_at_fixtures(monkeypatch, tmp_path)
    output = tmp_path / "gold.json"

    seed.build(output)
    items = load_dataset(output)

    by_subset: dict[str, list[GoldItem]] = {}
    for item in items:
        by_subset.setdefault(item.subset, []).append(item)

    assert len(by_subset["structure-chapter"]) == 5
    assert len(by_subset["structure-outline"]) == 2
    assert len(by_subset["fact-instructor-syllabus"]) == 2
    assert len(by_subset["fact-instructor-missing"]) == 1
    assert len(by_subset["fact-assessment"]) == 2
    assert len(by_subset["noanswer-chapter"]) == 2

    chapter = by_subset["structure-chapter"][0]
    assert chapter.expected_route == "structure"
    assert chapter.gold_chunk_ids
    assert len(chapter.gold_chapter_titles) == 1

    outline = {
        item.gold_course_codes[0]: item for item in by_subset["structure-outline"]
    }
    assert outline["IM101"].gold_chapter_titles == ["第一章 绪论", "第二章 线性表"]
    assert outline["IM102"].gold_chapter_titles == [
        "第一章 数据库概论",
        "第二章 关系模型",
        "第三章 SQL",
    ]
    assert outline["IM102"].gold_chunk_ids == ["c-IM102-1", "c-IM102-2", "c-IM102-3"]

    instructors = {
        item.gold_course_codes[0]: item
        for item in by_subset["fact-instructor-syllabus"]
    }
    assert instructors["IM101"].expected_instructors == ["李四", "王五"]
    assert instructors["IM101"].gold_chunk_ids == ["c-IM101"]
    assert instructors["IM102"].expected_instructors == ["张三"]

    missing = by_subset["fact-instructor-missing"][0]
    assert missing.answerable is False
    assert missing.gold_course_codes == ["IM104"]

    # structure-chapter(5) + structure-outline(2) + 教师字段(1) 都带 chunk 级金标。
    assert sum(1 for item in items if item.gold_chunk_ids) >= 8
    # 既有子集没有被新逻辑改动：种子集里仍有 fact 题。
    assert any(item.id.startswith("fact-") for item in items)


def test_build_degrades_when_artifacts_missing(monkeypatch: Any, tmp_path: Path) -> None:
    _point_env_at_fixtures(monkeypatch, tmp_path, doc_tree=False, profiles=False)
    output = tmp_path / "gold.json"

    summary = seed.build(output)  # 不得抛异常
    items = load_dataset(output)

    assert summary["total"] == len(items)
    # 产物缺失时，M5.1 新子集全部跳过，只保留既有五类（subset 为空）；
    # E5 子集只依赖 courses.json（no-brackets 由既有题派生），不受影响。
    e5_subsets = {"relation-neg", "relation-pos", "existence", "semester-phrasing", "no-brackets"}
    assert all(item.subset in ({""} | e5_subsets) for item in items)


# ---------------------------------------------------------------------------
# M5.2 新指标（手算例子）
# ---------------------------------------------------------------------------

def test_structure_metrics_on_hand_example() -> None:
    titles = ["第一章 绪论", "第二章 线性表", "第三章 栈"]

    assert metrics.chapter_coverage(
        "本课程共三章：第一章 绪论、第二章 线性表。", titles
    ) == round(2 / 3, 4)
    assert metrics.chapter_coverage("完全没有提到章节", titles) == 0.0
    assert metrics.chapter_coverage("任意", []) is None

    ordered = "第一章 绪论 ... 第二章 线性表 ... 第三章 栈"
    reversed_answer = "第三章 栈 ... 第二章 线性表 ... 第一章 绪论"
    assert metrics.chapter_order_correct(ordered, titles) is True
    assert metrics.chapter_order_correct(reversed_answer, titles) is False
    assert metrics.chapter_order_correct("第一章 绪论", titles) is None


def test_instructor_and_refusal_subset_metrics() -> None:
    known = ["李四", "王五", "张三"]

    assert (
        metrics.instructor_set_correct(
            "任课教师：李四、王五。", ["李四", "王五"], known_names=known
        )
        is True
    )
    assert (
        metrics.instructor_set_correct(
            "任课教师：李四。", ["李四", "王五"], known_names=known
        )
        is False
    )
    assert (
        metrics.instructor_set_correct(
            "任课教师：李四、张三。", ["李四"], known_names=known
        )
        is False
    )
    assert metrics.instructor_set_correct("无", []) is None

    results = [
        {"subset": "content", "answerable": True, "refused": True},
        {"subset": "content", "answerable": True, "refused": False},
        {"subset": "noanswer-chapter", "answerable": False, "refused": True},
        {"subset": "noanswer-chapter", "answerable": False, "refused": False},
    ]
    by_subset = metrics.refusal_report_by_subset(results)

    assert by_subset["content"]["false_refusal_rate"] == 0.5
    assert by_subset["noanswer-chapter"]["correct_refusal_rate"] == 0.5


def test_nli_attribution_priority_buckets() -> None:
    results = [
        {"nli_details": [{"label": "Contradiction"}, {"label": "Entailment"}]},
        {"nli_details": [{"label": "Unknown"}]},
        {"nli_details": [{"label": "Neutral"}]},
        {"nli_details": [{"label": "Entailment"}]},
        {"nli_details": []},
    ]

    report = metrics.nli_attribution(results)

    assert report["contradiction"] == 1
    assert report["unknown"] == 1
    assert report["non_entailment"] == 1
    assert report["all_entailed"] == 1
    assert report["no_details"] == 1
    assert report["total_items"] == 5


def test_subset_and_gold_fields_round_trip(tmp_path: Path) -> None:
    items = [
        GoldItem(
            id="s1",
            question="《X》第 1 章讲什么？",
            expected_route="structure",
            answer_keys=["第一章 绪论"],
            gold_chapter_titles=["第一章 绪论"],
            subset="structure-chapter",
        )
    ]

    loaded = load_dataset(save_dataset(items, tmp_path / "g.json"))

    assert loaded[0].subset == "structure-chapter"
    assert loaded[0].gold_chapter_titles == ["第一章 绪论"]


# ---------------------------------------------------------------------------
# P0-5 --dump-nli：默认关闭不改变既有结构
# ---------------------------------------------------------------------------

class _FakeRouteResult:
    def __init__(self, answer: str, metadata: dict[str, Any] | None = None) -> None:
        self.answer = answer
        self.citations: list[dict[str, Any]] = []
        self.query_type = "content"
        self.metadata = metadata or {}


class _FakeRouter:
    def __init__(self, result: _FakeRouteResult) -> None:
        self._result = result

    async def route(self, query: str, item_id: str, **kwargs: Any) -> _FakeRouteResult:
        return self._result


def _answer_item() -> GoldItem:
    return GoldItem(
        id="content-001",
        question="《数据结构》主要讲什么内容？",
        expected_route="content",
        answer_keys=["要点"],
        gold_course_codes=["IM101"],
        gold_chapter_titles=["第一章 绪论"],
        subset="structure-chapter",
    )


def test_dump_nli_off_keeps_legacy_result_shape() -> None:
    result = _FakeRouteResult(
        "要点：第一章 绪论。",
        {
            "nli_status": "passed",
            "nli_details": [{"sentence": "要点", "label": "Entailment", "score": 0.9}],
        },
    )

    report = asyncio.run(
        run_eval.grade_answers([_answer_item()], _FakeRouter(result), dump_nli=False)
    )
    row = report["results"][0]

    assert "nli_details" not in row
    assert row["chapter_coverage"] == 1.0
    assert row["refused"] is False
    assert "answer_preview" in row


def test_dump_nli_on_exports_claims_and_retention() -> None:
    result = _FakeRouteResult(
        "要点：第一章 绪论。",
        {
            "nli_status": "fallback",
            "nli_verified": False,
            "nli_attempts": 1,
            "nli_details": [
                {"sentence": "概括句", "label": "Neutral", "score": 0.3},
                {"sentence": "细节", "label": "Entailment", "score": 0.9},
            ],
        },
    )

    report = asyncio.run(
        run_eval.grade_answers([_answer_item()], _FakeRouter(result), dump_nli=True)
    )
    row = report["results"][0]

    assert [claim["verdict"] for claim in row["nli_details"]] == ["Neutral", "Entailment"]
    assert row["nli_details"][0]["retained"] is False
    assert row["nli_details"][1]["retained"] is True
    assert metrics.nli_attribution(report["results"])["non_entailment"] == 1


def test_evaluate_default_omits_nli_attribution() -> None:
    item = GoldItem(
        id="r1",
        question="《数据库原理》有几学分？",
        expected_route="fact",
        answer_keys=["2"],
    )

    report = run_eval.evaluate([item], stages=("routing",), dump_nli=False)

    assert report["routing"]["count"] == 1
    assert "nli_attribution" not in report
