from __future__ import annotations

from eval import metrics
from src.online_service.query_router import (
    _COURSE_CLARIFY_PREFIX,
    _COURSE_NOT_FOUND_TEMPLATE,
    _NO_INSTRUCTOR_RECORDED,
    _UNRECORDED_FIELD_TEMPLATE,
)


def test_instructor_missing_constant_counts_as_refusal() -> None:
    assert metrics.is_refusal(_NO_INSTRUCTOR_RECORDED)


def test_course_gate_messages_count_as_refusal() -> None:
    assert metrics.is_refusal(_COURSE_NOT_FOUND_TEMPLATE.format(name="火星种植学"))
    assert metrics.is_refusal(f"{_COURSE_CLARIFY_PREFIX}《信息管理学基础》？")
    assert metrics.is_refusal(
        _UNRECORDED_FIELD_TEMPLATE.format(name="文书学", field="上课地点/教室")
    )


def test_citation_precision_matches_each_citation_independently() -> None:
    citations = [
        {"chunk_id": "gold-chunk"},
        {"chunk_id": "other-chunk", "course_code": "IM121"},
        {"course_code": "IM121"},
        {"course_code": None, "course_codes": ["IM122"]},
        {"course_code": "", "course_codes": ["IM999"]},
    ]

    precision = metrics.citation_precision(
        citations,
        gold_chunk_ids=["gold-chunk"],
        gold_course_codes=["IM121", "IM122"],
    )

    assert precision == 0.6


def test_citation_precision_uses_course_codes_when_course_code_is_empty() -> None:
    citations = [{"course_code": None, "course_codes": ["IM121", "IM999"]}]

    assert metrics.citation_precision(
        citations,
        gold_course_codes=["IM121"],
    ) == 1.0


def test_citation_precision_chunk_uses_only_chunk_bearing_citations() -> None:
    citations = [
        {"chunk_id": "gold-chunk"},
        {"chunk_id": "other-chunk", "course_code": "IM121"},
        {"course_code": "IM121"},
    ]

    assert metrics.citation_precision(
        citations,
        gold_chunk_ids=["gold-chunk"],
        gold_course_codes=["IM121"],
    ) == round(2 / 3, 4)
    assert metrics.citation_precision_chunk(
        citations,
        gold_chunk_ids=["gold-chunk"],
    ) == 0.5
    assert metrics.citation_precision_chunk(
        [{"course_code": "IM121"}],
        gold_chunk_ids=["gold-chunk"],
    ) is None
    assert metrics.citation_precision_chunk(
        [{"chunk_id": "gold-chunk"}],
    ) is None


def test_generation_report_keeps_chunk_precision_separate() -> None:
    results = [
        {
            "answerable": True,
            "answer_keys": [],
            "citation_precision": 1.0,
            "citation_precision_chunk": 1.0,
        },
        {
            "answerable": True,
            "answer_keys": [],
            "citation_precision": 0.0,
            "citation_precision_chunk": 0.0,
        },
        {
            "answerable": True,
            "answer_keys": [],
            "citation_precision": None,
            "citation_precision_chunk": None,
        },
    ]

    report = metrics.generation_report(results)

    assert report["citation_precision"] == 0.5
    assert report["citation_precision_chunk"] == 0.5
    assert report["scored_citations"] == 2
    assert report["scored_chunk_citations"] == 2


def test_routing_accuracy_accepts_any_acceptable_route() -> None:
    results = [
        {
            "expected_route": "fact",
            "predicted_route": "hybrid",
            "predicted_labels": ["hybrid"],
            "acceptable_routes": ["fact", "hybrid"],
        },
        {
            "expected_route": "content",
            "predicted_route": "dependency",
            "predicted_labels": ["dependency"],
            "acceptable_routes": ["dependency"],
        },
        {
            "expected_route": "fact",
            "predicted_route": "catalog",
            "predicted_labels": ["catalog"],
        },
    ]

    report = metrics.routing_report(
        results,
        ("fact", "content", "dependency", "catalog", "hybrid"),
    )

    assert report["accuracy"] == round(2 / 3, 4)
    assert report["confusion_matrix"]["fact"]["hybrid"] == 1
    assert report["confusion_matrix"]["fact"]["catalog"] == 1
    assert report["confusion_matrix"]["content"]["dependency"] == 1
