"""E5 补充评测子集与「答案切题率」的离线单元测试。

全部使用内存里构造的小课程表与假 router，不依赖 ``outputs/``、ChromaDB 或网络。
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import pytest

from eval import build_seed_dataset as seed
from eval import metrics, run_eval
from eval.schema import GoldItem, load_dataset

IS_PLAN = "信息管理学院2025级信息管理与信息系统专业培养方案"
IS_MINOR = "信息管理学院2025级信息管理与信息系统专业辅修专业培养方案"
LIB_PLAN = "信息管理学院2025级图书馆学专业培养方案"
ARC_MINOR = "信息管理学院2025级档案学专业辅修专业培养方案"


def _offering(program: str, semester: Any, program_type: str = "主修专业") -> dict[str, Any]:
    return {"program_name": program, "program_type": program_type, "semester": semester}


def _course(
    code: str,
    name: str,
    *,
    prerequisites: list[str] | None = None,
    offerings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "course_code": code,
        "course_name": name,
        "prerequisites": prerequisites or [],
        "offerings": offerings if offerings is not None else [_offering(IS_PLAN, 3)],
    }


# 先修链 A -> B -> C -> D -> I（A 到 I 距离 4，超出深度 3）；
# E 与 G 只有知识关联；F 与其余课程无关。
COURSES = [
    _course("A101", "数据结构", offerings=[_offering(IS_PLAN, 1)]),
    _course("B101", "算法设计", prerequisites=["A101"], offerings=[_offering(IS_PLAN, 2)]),
    _course("C101", "高级算法", prerequisites=["B101"], offerings=[_offering(IS_PLAN, 3)]),
    _course("D101", "数据挖掘", prerequisites=["C101"], offerings=[_offering(IS_PLAN, 4)]),
    _course("I101", "深度学习", prerequisites=["D101"], offerings=[_offering(IS_PLAN, 5)]),
    _course("E101", "图书馆学基础", offerings=[_offering(LIB_PLAN, 6)]),
    _course("F101", "档案学概论", offerings=[_offering(LIB_PLAN, 7)]),
    _course("G101", "知识图谱", offerings=[_offering(IS_PLAN, 8)]),
    # 课名带连接词：关系子集不选。
    _course("H101", "数据与分析"),
    # 学期在各方案间不一致 / 弹性区间 / 没有 offerings / 同名：学期、关系子集都不选。
    _course("J101", "信息伦理", offerings=[_offering(IS_PLAN, 3), _offering(LIB_PLAN, 4)]),
    _course("K101", "大学外语", offerings=[_offering(IS_PLAN, "1-4")]),
    _course("L101", "无开课记录", offerings=[]),
    _course("M101", "重名课"),
    _course("M102", "重名课"),
    # 只在档案学辅修方案里出现：带专业线索的存在性题不用它。
    _course("N101", "档案编研", offerings=[_offering(ARC_MINOR, 6, "辅修专业")]),
]

CONCEPT_PAIRS = {frozenset({"E101", "G101"})}


def _pair(item: GoldItem) -> frozenset[str]:
    return frozenset(item.gold_course_codes)


# ---------------------------------------------------------------------------
# relation-neg / relation-pos
# ---------------------------------------------------------------------------

def test_prerequisite_distances_respect_depth_limit() -> None:
    distances = seed._prerequisite_distances(COURSES)

    assert distances[frozenset({"A101", "B101"})] == 1
    assert distances[frozenset({"A101", "D101"})] == 3
    assert frozenset({"A101", "I101"}) not in distances


def test_relation_negatives_exclude_every_related_pair() -> None:
    items = seed._relation_items(COURSES, CONCEPT_PAIRS, random.Random(1))
    negatives = [item for item in items if item.subset == "relation-neg"]
    distances = seed._prerequisite_distances(COURSES)

    assert negatives
    for item in negatives:
        assert item.answerable is False
        assert item.expected_route == "dependency"
        assert not item.answer_keys
        assert _pair(item) not in distances
        assert _pair(item) not in CONCEPT_PAIRS
        assert "COURSE_RELATION_NOT_FOUND" in item.notes
    # 距离 4 的 A101 / I101 超出深度，应当属于无关系的一侧。
    all_negative_pairs = {
        frozenset((a, b))
        for a, b, _ in _all_negative_candidates(CONCEPT_PAIRS)
    }
    assert frozenset({"A101", "I101"}) in all_negative_pairs


def _all_negative_candidates(concept_pairs: Any) -> list[tuple[str, str, str]]:
    """把 quota 调大，拿到全部负样本候选（只为断言边界）。"""
    original = dict(seed.E5_QUOTAS)
    seed.E5_QUOTAS["relation-neg"] = 10_000
    try:
        items = seed._relation_items(COURSES, concept_pairs, random.Random(1))
    finally:
        seed.E5_QUOTAS.update(original)
    return [
        (item.gold_course_codes[0], item.gold_course_codes[1], "none")
        for item in items
        if item.subset == "relation-neg"
    ]


def test_relation_positives_cover_direct_indirect_and_concept() -> None:
    items = seed._relation_items(COURSES, CONCEPT_PAIRS, random.Random(1))
    positives = {
        item.notes: item for item in items if item.subset == "relation-pos"
    }
    kinds = {item.notes for item in items if item.subset == "relation-pos"}

    assert kinds == {"relation=direct", "relation=indirect", "relation=concept"}
    for item in items:
        if item.subset != "relation-pos":
            continue
        assert item.answerable is True
        assert item.expected_route == "dependency"
        # 答案要同时含两门课的课名。
        assert len(item.answer_keys) == 2
        assert all(name in item.question for name in item.answer_keys)
        if item.notes == "relation=concept":
            assert _pair(item) == frozenset({"E101", "G101"})
        else:
            assert _pair(item) in seed._prerequisite_distances(COURSES)
    assert positives


def test_relation_skips_connector_names_duplicates_and_substrings() -> None:
    items = seed._relation_items(COURSES, CONCEPT_PAIRS, random.Random(1))
    used = {code for item in items for code in item.gold_course_codes}

    assert not used & {"H101", "L101", "M101", "M102"}
    for item in items:
        assert item.validate() == []


def test_relation_templates_cycle_and_names_are_plain() -> None:
    items = [
        item
        for item in seed._relation_items(COURSES, CONCEPT_PAIRS, random.Random(1))
        if item.subset == "relation-pos"
    ]
    questions = " ".join(item.question for item in items)

    assert "之间有什么关系" in questions
    assert "有关系吗" in questions
    assert "有什么联系" in questions
    # 只有第三种问法带书名号。
    assert all(("《" in item.question) == item.question.endswith("有什么联系？") for item in items)


def test_relation_without_concept_artifacts_uses_prerequisites_only() -> None:
    items = seed._relation_items(COURSES, None, random.Random(1))
    positives = [item for item in items if item.subset == "relation-pos"]
    negatives = [item for item in items if item.subset == "relation-neg"]

    assert positives
    assert all(item.notes != "relation=concept" for item in positives)
    assert all("concept 产物缺失" in item.notes for item in negatives)
    # 没有知识关联时 E101 / G101 被当作无关系的一对。
    assert frozenset({"E101", "G101"}) in {_pair(item) for item in negatives}


def test_relation_items_are_reproducible() -> None:
    first = seed._relation_items(COURSES, CONCEPT_PAIRS, random.Random(7))
    second = seed._relation_items(COURSES, CONCEPT_PAIRS, random.Random(7))

    assert [item.as_dict() for item in first] == [item.as_dict() for item in second]


def test_load_concept_pairs_uses_config_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = [
        {"id": "c1", "canonical_name": "排序", "source_course_codes": ["A101"]},
        {"id": "c2", "canonical_name": "查找", "source_course_codes": ["F101"]},
    ]
    edges = [
        {
            "source_id": "c1",
            "target_id": "c2",
            "requires": True,
            "verification_source": "llm",
            "confidence": 0.9,
        },
        {
            "source_id": "c2",
            "target_id": "c1",
            "requires": True,
            "verification_source": "llm",
            "confidence": 0.1,
        },
    ]
    (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    (tmp_path / "edges.json").write_text(json.dumps(edges), encoding="utf-8")
    monkeypatch.setenv("CONCEPT_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("CONCEPT_VERIFIED_EDGE_PATH", str(tmp_path / "edges.json"))

    assert seed._load_concept_pairs() == {frozenset({"A101", "F101"})}

    monkeypatch.setenv("CONCEPT_REGISTRY_PATH", str(tmp_path / "missing.json"))
    assert seed._load_concept_pairs() is None


# ---------------------------------------------------------------------------
# existence
# ---------------------------------------------------------------------------

def test_existence_items_shape_and_gold() -> None:
    items = seed._existence_items(COURSES, random.Random(3))
    by_note = {item.id: item for item in items}

    assert len(by_note) == len(items)
    absent = [item for item in items if not item.answerable]
    assert len(absent) == 5
    for item in items:
        assert item.expected_route == "existence"
        assert item.subset == "existence"
        assert "《" not in item.question and "》" not in item.question
        assert item.validate() == []
        assert item.focus_keys
    for item in absent:
        assert item.focus_keys == [seed.ABSENT_FOCUS]
        assert any(name in item.question for name in seed.ABSENT_COURSES)
    for item in items:
        if item.answerable and item.focus_keys == [seed.EXISTS_FOCUS]:
            assert item.answer_keys == ["有。"]


def test_existence_program_cue_gold_follows_offerings() -> None:
    items = seed._existence_items(COURSES, random.Random(3))
    cue_items = [item for item in items if "program_cue=" in item.notes]
    by_code = {course["course_code"]: course for course in COURSES}

    assert cue_items
    for item in cue_items:
        keyword = item.notes.split("program_cue=")[1].split(";")[0]
        code = item.gold_course_codes[0]
        programs = [o["program_name"] for o in by_code[code]["offerings"]]
        has_course = any(keyword in program for program in programs)
        assert ("program_has_course=True" in item.notes) is has_course
        if has_course:
            assert item.answer_keys == ["有。"]
        else:
            assert item.answer_keys == [f"中没有《{by_code[code]['course_name']}》"]
        # 只在档案学辅修方案里出现的课，不能拿来问档案学专业有没有。
        assert not (code == "N101" and keyword == "档案学")


def test_program_status_ignores_minor_only_courses() -> None:
    by_code = {course["course_code"]: course for course in COURSES}

    assert seed._program_status(by_code["A101"], "信息管理与信息系统") == "present"
    assert seed._program_status(by_code["A101"], "图书馆学") == "absent"
    assert seed._program_status(by_code["N101"], "档案学") == "unclear"


def test_existence_skipped_when_catalogue_has_no_usable_course() -> None:
    assert seed._existence_items([_course("X1", "只有一门", offerings=[])], random.Random(0)) == []


# ---------------------------------------------------------------------------
# semester-phrasing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("semester", "grade"),
    [(1, "大一上"), (2, "大一下"), (3, "大二上"), (4, "大二下"), (7, "大四上"), (8, "大四下")],
)
def test_semester_to_grade(semester: int, grade: str) -> None:
    assert seed.semester_to_grade(semester) == grade


def test_semester_items_only_use_consistent_single_semesters() -> None:
    items = seed._semester_items(COURSES, random.Random(5))
    codes = {item.gold_course_codes[0] for item in items}
    by_code = {course["course_code"]: course for course in COURSES}

    assert items
    # 学期不一致（J101）、弹性区间（K101）、无 offerings（L101）、同名课都不选。
    assert not codes & {"J101", "K101", "L101", "M101", "M102"}
    # 每门课 3 种问法，且互不相同。
    for code in codes:
        questions = [i.question for i in items if i.gold_course_codes[0] == code]
        assert len(questions) == 3 and len(set(questions)) == 3
    for item in items:
        semester = item.expected_semester
        assert semester == by_code[item.gold_course_codes[0]]["offerings"][0]["semester"]
        assert item.expected_route == "fact"
        assert item.answer_keys[1] == seed.semester_to_grade(semester)
        assert "《" not in item.question
        assert item.validate() == []


# ---------------------------------------------------------------------------
# no-brackets
# ---------------------------------------------------------------------------

def test_no_bracket_items_strip_marks_and_keep_route() -> None:
    base = [
        GoldItem(
            id=f"fact-{index:03d}",
            question=f"《课程{index}》有几学分？",
            expected_route="fact",
            answer_keys=["3"],
            notes="field=credits",
        )
        for index in range(1, 15)
    ] + [
        GoldItem(
            id=f"dependency-{index:03d}",
            question=f"学《课程{index}》之前需要先修哪些课程？",
            expected_route="dependency",
            answer_keys=["数据结构"],
        )
        for index in range(1, 15)
    ]
    base.append(
        GoldItem(
            id="noanswer-001",
            question="《火星种植学》有几学分？",
            expected_route="fact",
            answerable=False,
        )
    )

    items = seed._no_bracket_items(base, random.Random(2))
    sources = {item.id: item for item in base}

    assert len(items) == 20
    assert sum(1 for item in items if item.expected_route == "fact") == 10
    for item in items:
        source = sources[item.notes.split("源题=")[1]]
        assert source.id.startswith(("fact-", "dependency-"))
        assert item.subset == "no-brackets"
        assert "《" not in item.question and "》" not in item.question
        assert item.question == source.question.replace("《", "").replace("》", "")
        assert item.expected_route == source.expected_route
        assert item.answer_keys == source.answer_keys
    assert len({item.id for item in items}) == 20


# ---------------------------------------------------------------------------
# build 集成：新子集追加在最后，不改变既有子集
# ---------------------------------------------------------------------------

def test_build_appends_e5_subsets_without_touching_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chunks = [
        {"chunk_id": f"c-{course['course_code']}", "metadata": {
            "source_type": "syllabus", "course_code": course["course_code"]}}
        for course in COURSES
    ]
    for name, payload in (("courses.json", COURSES), ("chunks.json", chunks)):
        (tmp_path / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("COURSES_OUTPUT_PATH", str(tmp_path / "courses.json"))
    monkeypatch.setenv("CHUNKS_OUTPUT_PATH", str(tmp_path / "chunks.json"))
    for env in ("DOC_TREE_PATH", "SYLLABUS_PROFILES_PATH", "CONCEPT_REGISTRY_PATH",
                "CONCEPT_VERIFIED_EDGE_PATH"):
        monkeypatch.setenv(env, str(tmp_path / "missing.json"))

    seed.build(tmp_path / "gold.json")
    items = load_dataset(tmp_path / "gold.json")

    subsets = [item.subset for item in items]
    legacy = [item for item in items if item.subset == ""]
    first_e5 = next(index for index, value in enumerate(subsets) if value)
    assert all(value for value in subsets[first_e5:])
    assert len(legacy) == first_e5
    assert {"relation-neg", "relation-pos", "existence", "semester-phrasing",
            "no-brackets"} <= set(subsets)


# ---------------------------------------------------------------------------
# 指标：REFUSAL_MARKERS / 切题率
# ---------------------------------------------------------------------------

def test_relation_refusal_text_counts_as_refusal() -> None:
    answer = (
        "根据培养方案与课程图谱，《新媒体》与《大数据分析》之间没有先修关系，"
        "也没有经过验证的知识关联，无法说明两者存在关系。"
    )

    assert metrics.is_refusal(answer)


def test_first_sentence_keeps_terminator_and_splits_on_newline() -> None:
    assert metrics.first_sentence("有。《电子文件管理》开设在第 4 学期。") == "有。"
    assert metrics.first_sentence("\n  开设在第 4 学期\n第二行。") == "开设在第 4 学期"
    assert metrics.first_sentence("") == ""


def test_answer_on_topic_for_existence() -> None:
    exists = ["有。"]
    absent = [seed.ABSENT_FOCUS]

    assert metrics.answer_on_topic("有。《电子文件管理》（IM214）第 4 学期开设。", focus_keys=exists)
    assert not metrics.answer_on_topic("电子文件管理主要讲档案的电子化。有。", focus_keys=exists)
    assert metrics.answer_on_topic("培养方案中没有名为《机器学习》的课程。", focus_keys=absent)
    assert metrics.answer_on_topic("未找到该课程，你是不是想问《机器学习基础》？", focus_keys=absent)
    assert metrics.answer_on_topic("信息管理与信息系统专业培养方案中没有《X》。", focus_keys=["中没有《X》"])


def test_answer_on_topic_for_semester_uses_whitespace_tolerant_regex() -> None:
    assert metrics.answer_on_topic("《X》在第 4 学期（大二下学期）开设。", expected_semester=4)
    assert metrics.answer_on_topic("《X》在第4学期开设。", expected_semester=4)
    assert not metrics.answer_on_topic("《X》在第 5 学期开设。", expected_semester=4)
    # 第 4 学期出现在第二句：第一句没有直接回答。
    assert not metrics.answer_on_topic("《X》是一门专业课。第 4 学期开设。", expected_semester=4)


def test_chapter_heading_makes_answer_off_topic() -> None:
    answer = "有。《电子文件管理》第 4 学期开设。\n第一章 绪论（3 学时）\n第二章 管理（4 学时）"

    assert metrics.has_chapter_heading(answer)
    assert not metrics.answer_on_topic(answer, focus_keys=["有。"])
    assert metrics.has_chapter_heading("- 第三章 标准（2 学时）")
    # 句中提到「第三章」不是标题行。
    assert not metrics.has_chapter_heading("这门课的重点见第三章内容。")


def test_answer_on_topic_none_without_expectation() -> None:
    assert metrics.answer_on_topic("任意答案") is None


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("第 4 学期（大二下学期）", True),
        ("第4学期，大二下", True),
        ("第 4 学期（大二上学期）", False),  # 年级换算错
        ("第 4 学期", False),  # 缺年级
        ("第 3 学期（大二上学期）", False),
    ],
)
def test_semester_answer_correct(answer: str, expected: bool) -> None:
    assert metrics.semester_answer_correct(answer, 4) is expected


def test_answer_relevance_report_aggregates_by_subset() -> None:
    results = [
        {"subset": "existence", "on_topic": True, "chapter_heading": False},
        {"subset": "existence", "on_topic": False, "chapter_heading": True},
        {"subset": "semester-phrasing", "on_topic": True, "chapter_heading": False},
        {"subset": "semester-phrasing", "on_topic": True, "chapter_heading": False},
        {"subset": "fact-assessment", "on_topic": None, "chapter_heading": None},
    ]

    report = metrics.answer_relevance_report(results)

    assert report["on_topic_items"] == 4
    assert report["answer_on_topic_rate"] == 0.75
    assert report["chapter_heading_rate"] == 0.25
    assert report["existence_on_topic_rate"] == 0.5
    assert report["semester-phrasing_on_topic_rate"] == 1.0
    assert metrics.answer_relevance_report([]) == {"on_topic_items": 0}


# ---------------------------------------------------------------------------
# run_eval 接入
# ---------------------------------------------------------------------------

class _FakeRouteResult:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.citations: list[dict[str, Any]] = []
        self.query_type = "fact"
        self.metadata: dict[str, Any] = {"error_code": "COURSE_RELATION_NOT_FOUND"}


class _FakeRouter:
    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    async def route(self, query: str, item_id: str, **kwargs: Any) -> _FakeRouteResult:
        return _FakeRouteResult(self._answers[item_id])


def test_grade_answers_scores_e5_subsets() -> None:
    items = [
        GoldItem(
            id="semester-phrasing-001",
            question="数据挖掘什么时候上？",
            expected_route="fact",
            answer_keys=["第4学期|第 4 学期", "大二下"],
            focus_keys=["第4学期|第 4 学期"],
            expected_semester=4,
            subset="semester-phrasing",
        ),
        GoldItem(
            id="existence-001",
            question="有数据挖掘这门课吗？",
            expected_route="existence",
            answer_keys=["有。"],
            focus_keys=["有。"],
            subset="existence",
        ),
        GoldItem(
            id="existence-002",
            question="有火星种植学这门课吗？",
            expected_route="existence",
            answerable=False,
            focus_keys=[seed.ABSENT_FOCUS],
            subset="existence",
        ),
        GoldItem(
            id="relation-neg-001",
            question="数据挖掘与劳动教育有关系吗？",
            expected_route="dependency",
            answerable=False,
            subset="relation-neg",
        ),
    ]
    router = _FakeRouter(
        {
            "semester-phrasing-001": "数据挖掘在第 4 学期（大二下学期）开设。",
            "existence-001": "数据挖掘课程包括：\n第一章 绪论（2 学时）",
            "existence-002": "培养方案中没有名为《火星种植学》的课程。",
            "relation-neg-001": "两门课之间没有先修关系，无法说明两者存在关系。",
        }
    )

    report = asyncio.run(run_eval.grade_answers(items, router))
    rows = {row["id"]: row for row in report["results"]}

    assert rows["semester-phrasing-001"]["answer_correct"] is True
    assert rows["semester-phrasing-001"]["on_topic"] is True
    assert rows["existence-001"]["on_topic"] is False
    assert rows["existence-001"]["chapter_heading"] is True
    assert rows["existence-002"]["on_topic"] is True
    assert rows["existence-002"]["refused"] is True
    # 非 E5 切题子集不计算切题率。
    assert rows["relation-neg-001"]["on_topic"] is None
    assert rows["relation-neg-001"]["refused"] is True
    assert rows["relation-neg-001"]["error_code"] == "COURSE_RELATION_NOT_FOUND"

    relevance = metrics.answer_relevance_report(report["results"])
    assert relevance["on_topic_items"] == 3
    assert relevance["answer_on_topic_rate"] == pytest.approx(0.6667, abs=1e-4)


def test_existence_route_is_valid_and_rendered_in_report() -> None:
    assert "existence" in run_eval.ROUTE_LABELS
    assert GoldItem(
        id="e", question="有X这门课吗", expected_route="existence", answer_keys=["有。"]
    ).validate() == []

    report = {
        "tag": "t",
        "created_at": "now",
        "dataset": "d",
        "dataset_summary": {"total": 1, "unanswerable": 0},
        "environment": {
            "embedding_model": "m",
            "collection": None,
            "rerank_weights": {},
            "llm_available": False,
            "neo4j_available": False,
        },
        "answer_relevance": {"on_topic_items": 3, "answer_on_topic_rate": 0.5},
    }

    assert "答案切题率" in run_eval.render_markdown(report)
