from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import metrics
from eval.schema import GoldItem, dataset_summary, load_dataset, save_dataset


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_gold_item_requires_evidence_for_answerable_questions() -> None:
    problems = GoldItem(
        id="x-1",
        question="《数据库原理》有几学分？",
        expected_route="fact",
    ).validate()

    assert any("answer_keys" in problem for problem in problems)


def test_unanswerable_item_must_not_carry_answer_keys() -> None:
    problems = GoldItem(
        id="x-2",
        question="《火星种植学》有几学分？",
        expected_route="fact",
        answerable=False,
        answer_keys=["2"],
    ).validate()

    assert any("unanswerable" in problem for problem in problems)


def test_unknown_route_is_rejected() -> None:
    problems = GoldItem(
        id="x-3",
        question="随便问问",
        expected_route="chitchat",
        answer_keys=["x"],
    ).validate()

    assert any("expected_route" in problem for problem in problems)


def test_dataset_round_trip_and_summary(tmp_path: Path) -> None:
    items = [
        GoldItem(
            id="fact-001",
            question="《数据库原理》有几学分？",
            expected_route="fact",
            answer_keys=["学分为 3"],
            source="auto-seed",
        ),
        GoldItem(
            id="noanswer-001",
            question="《火星种植学》讲什么？",
            expected_route="content",
            answerable=False,
        ),
    ]
    path = save_dataset(items, tmp_path / "gold.json", description="test")

    loaded = load_dataset(path)
    summary = dataset_summary(loaded)

    assert [item.id for item in loaded] == ["fact-001", "noanswer-001"]
    assert summary == {
        "total": 2,
        "by_route": {"content": 1, "fact": 1},
        "answerable": 1,
        "unanswerable": 1,
        "human_labelled": 1,
    }


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    item = {
        "id": "dup",
        "question": "问题",
        "expected_route": "fact",
        "answer_keys": ["值"],
    }
    path.write_text(json.dumps({"items": [item, item]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate ids"):
        load_dataset(path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_refusal_detection_matches_the_generator_wording() -> None:
    assert metrics.is_refusal(
        "根据当前知识库，未找到足够相关且可靠的教学证据来回答这个问题。"
    )
    assert metrics.is_refusal("未找到明确的硬先修课程。")
    assert metrics.is_refusal(
        "课程依赖查询需要 Neo4j 图数据库支持，当前服务未连接 Neo4j。"
    )
    assert not metrics.is_refusal("管理运筹学：学分为 2，总学时为 36。")


def test_routing_report_counts_accuracy_and_label_recall() -> None:
    results = [
        {
            "expected_route": "fact",
            "predicted_route": "fact",
            "predicted_labels": ["fact"],
            "confidence": 1.0,
        },
        {
            "expected_route": "fact",
            "predicted_route": "dependency",
            "predicted_labels": ["dependency", "fact"],
            "confidence": 0.33,
        },
        {
            "expected_route": "catalog",
            "predicted_route": "content",
            "predicted_labels": ["content"],
            "confidence": 1.0,
        },
    ]

    report = metrics.routing_report(results, ("fact", "content", "dependency", "catalog"))

    assert report["accuracy"] == round(1 / 3, 4)
    # The second item is recovered by the secondary label, the third is not.
    assert report["label_recall"] == round(2 / 3, 4)
    assert report["confusion_matrix"]["fact"]["dependency"] == 1
    assert report["per_class"]["fact"]["support"] == 2


def test_hit_positions_prefers_chunk_ids_over_the_course_proxy() -> None:
    hits = [
        {"chunk_id": "a", "metadata": {"course_code": "IM121"}},
        {"chunk_id": "b", "metadata": {"course_code": "IM121"}},
    ]

    exact = metrics.hit_positions(hits, ["b"], gold_course_codes=["IM121"])
    proxy = metrics.hit_positions(hits, [], gold_course_codes=["IM121"])
    filtered = metrics.hit_positions(
        hits,
        [],
        gold_course_codes=["IM121"],
        gold_section_types=["course_objectives"],
    )

    assert exact == [2]
    assert proxy == [1, 2]
    assert filtered == []


def test_retrieval_report_computes_recall_and_mrr() -> None:
    results = [
        {"gradable": True, "grading": "chunk", "positions": [1]},
        {"gradable": True, "grading": "course", "positions": [7]},
        {"gradable": True, "grading": "chunk", "positions": []},
        {"gradable": False, "grading": "course", "positions": []},
    ]

    report = metrics.retrieval_report(results, cutoffs=(1, 5, 10))

    assert report["count"] == 3
    assert report["recall@1"] == round(1 / 3, 4)
    assert report["recall@5"] == round(1 / 3, 4)
    assert report["recall@10"] == round(2 / 3, 4)
    assert report["mrr"] == round((1.0 + 1 / 7) / 3, 4)
    assert report["chunk_level_items"] == 2


def test_answer_key_alternatives_and_conjunction() -> None:
    answer = "管理运筹学：学分为 2，总学时为 36。"

    assert metrics.answer_covers_keys(answer, ["学分为 2|2学分"])
    assert metrics.answer_covers_keys(answer, ["学分为 2", "总学时为 36"])
    assert not metrics.answer_covers_keys(answer, ["学分为 2", "任课教师"])
    assert not metrics.answer_covers_keys(answer, [])


def test_citation_precision_is_none_without_ground_truth() -> None:
    citations = [{"chunk_id": "a", "course_code": "IM121"}]

    assert metrics.citation_precision(citations) is None
    assert metrics.citation_precision([], gold_chunk_ids=["a"]) is None
    assert metrics.citation_precision(citations, gold_chunk_ids=["a"]) == 1.0
    assert metrics.citation_precision(citations, gold_chunk_ids=["b"]) == 0.0
    assert metrics.citation_precision(citations, gold_course_codes=["IM121"]) == 1.0


def test_refusal_report_separates_correct_and_false_refusals() -> None:
    results = [
        {"answerable": False, "refused": True},
        {"answerable": False, "refused": False},
        {"answerable": True, "refused": True},
        {"answerable": True, "refused": False},
        {"answerable": True, "refused": False},
    ]

    report = metrics.refusal_report(results)

    assert report["correct_refusal_rate"] == 0.5
    assert report["hallucination_rate"] == 0.5
    assert report["false_refusal_rate"] == round(1 / 3, 4)


def test_generation_report_averages_only_scored_items() -> None:
    results = [
        {
            "answerable": True,
            "answer_keys": ["x"],
            "answer_correct": True,
            "citation_precision": 1.0,
            "citation_count": 2,
            "refused": False,
        },
        {
            "answerable": True,
            "answer_keys": [],
            "answer_correct": False,
            "citation_precision": None,
            "citation_count": 0,
            "refused": False,
        },
    ]

    report = metrics.generation_report(results)

    assert report["scored_answers"] == 1
    assert report["answer_key_coverage"] == 1.0
    assert report["citation_precision"] == 1.0
    assert report["uncited_answer_rate"] == 0.5


def test_confusion_matrix_renders_as_markdown() -> None:
    matrix = metrics.confusion_matrix([("fact", "fact"), ("fact", "hybrid")], ("fact", "hybrid"))

    rendered = metrics.render_confusion_matrix(matrix, ("fact", "hybrid"))

    assert "| **fact** | 1 | 1 | 2 |" in rendered
