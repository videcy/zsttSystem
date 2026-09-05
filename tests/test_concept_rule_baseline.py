from __future__ import annotations

import json
from pathlib import Path

from src.data_processing.concept_extractor import (
    PROMPT_VERSION,
    extract_course_concepts,
)


def _chunk(course_code: str, text: str, document_hash: str = "h1") -> dict:
    return {
        "text": text,
        "document_hash": document_hash,
        "metadata": {"course_code": course_code},
    }


def test_teaching_boilerplate_is_filtered_out(tmp_path: Path) -> None:
    chunks = [
        _chunk("IM121", "本课程教学内容包括信息组织与信息检索，要求学生掌握信息计量学"),
    ]

    names = {row["name"] for row in extract_course_concepts(chunks, tmp_path / "c.json")}

    assert "信息组织" in names
    assert "信息计量学" in names
    for noise in ("本课程", "教学内容", "学生"):
        assert noise not in names
    # "要求学生掌握" carries an instruction fragment, so the whole run is dropped.
    assert not any("掌握" in name for name in names)


def test_structural_headings_are_not_concepts(tmp_path: Path) -> None:
    chunks = [_chunk("IM104", "第一章 档案学概论，第二节 复习与答疑，绪论部分")]

    names = {row["name"] for row in extract_course_concepts(chunks, tmp_path / "c.json")}

    assert "档案学概论" in names
    assert not any(name.startswith("第") for name in names)
    assert "复习" not in names and "绪论" not in names


def test_cross_course_boilerplate_is_cut_by_document_frequency(tmp_path: Path) -> None:
    # "信息素养" appears in every course, "信息栈" only in one.
    chunks = [
        _chunk(f"IM{index}", "信息素养、信息栈" if index == 1 else "信息素养、其他主题")
        for index in range(1, 5)
    ]

    concepts = extract_course_concepts(chunks, tmp_path / "c.json", max_document_ratio=0.5)
    names = {row["name"] for row in concepts}

    assert "信息栈" in names
    assert "信息素养" not in names


def test_cache_is_written_once_and_reused(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    chunks = [_chunk("IM121", "信息组织与信息检索")]

    first = extract_course_concepts(chunks, cache_path)
    cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    written_at = cache_path.stat().st_mtime_ns

    second = extract_course_concepts(chunks, cache_path)

    assert first == second
    assert cached_payload["IM121"]["prompt_version"] == PROMPT_VERSION
    # An unchanged corpus must not rewrite the cache file.
    assert cache_path.stat().st_mtime_ns == written_at


def test_changed_document_hash_invalidates_the_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    extract_course_concepts([_chunk("IM121", "信息组织")], cache_path)

    updated = extract_course_concepts(
        [_chunk("IM121", "信息计量学", document_hash="h2")],
        cache_path,
    )

    assert {row["name"] for row in updated} == {"信息计量学"}


def test_ranking_is_deterministic_and_bounded(tmp_path: Path) -> None:
    text = "、".join(f"概念{index}" for index in range(60))
    chunks = [_chunk("IM999", text)]

    first = extract_course_concepts(chunks, tmp_path / "a.json", max_concepts=10)
    second = extract_course_concepts(chunks, tmp_path / "b.json", max_concepts=10)

    assert len(first) == 10
    assert [row["name"] for row in first] == [row["name"] for row in second]
    assert all(row["extraction_source"] == "rule" for row in first)
