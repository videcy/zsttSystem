"""M1 data-set fixes: acceptable routes and course-name cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval import build_seed_dataset as seed
from eval.schema import GoldItem, load_dataset, save_dataset


COURSES = [
    {
        "course_code": "IM101",
        "course_name": "工匠书缘：古籍的修复与文化传承",
        "instructor": "/",
        "credits": 3,
        "hours": 48,
        "semester": 2,
        "prerequisites": [],
        "offerings": [],
    },
    {
        "course_code": "IM102",
        "course_name": "信息组织基础",
        "instructor": "李四",
        "credits": 2,
        "hours": 32,
        "semester": 1,
        "prerequisites": ["IM101"],
        "offerings": [],
    },
]

CHUNKS = [
    {
        "chunk_id": "c-IM101",
        "metadata": {"source_type": "syllabus", "course_code": "IM101"},
    },
    {
        "chunk_id": "c-IM102",
        "metadata": {"source_type": "syllabus", "course_code": "IM102"},
    },
]


def _chapter(number: int, title: str, chunk_id: str) -> dict[str, Any]:
    return {
        "level": 1,
        "title": title,
        "kind": "chapter",
        "chapter_no": number,
        "children": [],
        "chunk_ids": [chunk_id],
        "hours": 2,
        "text": title,
        "section_type": None,
    }


DOC_TREE = {
    "documents": {
        "h1": {
            "course_code": "IM101",
            "course_name": "工匠书缘 古籍的修复与文化传承 转自DOC",
            "nodes": {
                "n1": _chapter(1, "第一章 古籍修复", "c-IM101"),
                "n2": _chapter(2, "第二章 文化传承", "c-IM101-2"),
            },
        },
        "h2": {
            "course_code": "IM102",
            "course_name": "信息组织基础_转自_DOC",
            "nodes": {
                "n1": _chapter(1, "第一章 信息组织", "c-IM102"),
                "n2": _chapter(2, "第二章 分类法", "c-IM102-2"),
            },
        },
    }
}

PROFILES = {
    "IM101|2024": {
        "course_code": "IM101",
        "course_name": "工匠书缘 古籍的修复与文化传承 转自DOC",
        "course_owner": "王五",
        "assessment_mode": "课程论文",
        "field_chunk_ids": {"assessment_mode": "c-IM101"},
    },
    "IM102|2024": {
        "course_code": "IM102",
        "course_name": "信息组织基础_转自_DOC",
        "course_owner": "赵六",
        "assessment_mode": "闭卷考试",
        "field_chunk_ids": {"assessment_mode": "c-IM102"},
    },
}


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _point_config_at_fixtures(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "COURSES_OUTPUT_PATH", str(_write_json(tmp_path / "courses.json", COURSES))
    )
    monkeypatch.setenv(
        "CHUNKS_OUTPUT_PATH", str(_write_json(tmp_path / "chunks.json", CHUNKS))
    )
    monkeypatch.setenv(
        "DOC_TREE_PATH", str(_write_json(tmp_path / "doc_tree.json", DOC_TREE))
    )
    monkeypatch.setenv(
        "SYLLABUS_PROFILES_PATH",
        str(_write_json(tmp_path / "profiles.json", PROFILES)),
    )


def test_acceptable_routes_validate_and_round_trip(tmp_path: Path) -> None:
    valid = GoldItem(
        id="noanswer-001",
        question="《信息组织基础》的上课教室在哪里？",
        expected_route="fact",
        acceptable_routes=["fact", "hybrid"],
        answerable=False,
    )
    assert valid.acceptable_routes == ["fact", "hybrid"]
    assert valid.validate() == []

    invalid = GoldItem(
        id="noanswer-002",
        question="问题",
        expected_route="fact",
        acceptable_routes=["fact", "unknown"],
        answerable=False,
    )
    assert any("acceptable_routes" in problem for problem in invalid.validate())

    loaded = load_dataset(save_dataset([valid], tmp_path / "round-trip.json"))
    assert loaded[0].acceptable_routes == ["fact", "hybrid"]


def test_load_dataset_defaults_acceptable_routes_for_legacy_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "legacy-001",
                        "question": "《信息组织基础》有几学分？",
                        "expected_route": "fact",
                        "answer_keys": ["2"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = load_dataset(path)

    assert loaded[0].acceptable_routes == []


def test_missing_field_noanswer_items_allow_fact_and_hybrid(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _point_config_at_fixtures(monkeypatch, tmp_path)
    output = tmp_path / "gold.json"

    seed.build(output)
    items = load_dataset(output)

    missing_field = [
        item
        for item in items
        if item.notes == "课程存在但语料不记录该字段"
    ]
    assert missing_field
    assert all(item.expected_route == "fact" for item in missing_field)
    assert all(
        item.acceptable_routes == ["fact", "hybrid"] for item in missing_field
    )

    # Only the documented missing-field family gets the alternative route.
    absent = [item for item in items if item.notes == "课程不存在于语料中"]
    assert absent
    assert all(item.acceptable_routes == [] for item in absent)
    assert all(
        item.acceptable_routes == [] for item in items if item.answerable
    )


def test_generated_questions_prefer_catalogue_names_and_clean_doc_suffix(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _point_config_at_fixtures(monkeypatch, tmp_path)
    output = tmp_path / "gold.json"

    seed.build(output)
    questions = [item.question for item in load_dataset(output)]

    assert all("转自DOC" not in question for question in questions)
    assert any("《工匠书缘：古籍的修复与文化传承》" in q for q in questions)
    assert any("《信息组织基础》" in q for q in questions)

    # When courses.json has no name, the cleaned doc-tree title is the fallback.
    document = DOC_TREE["documents"]["h2"]
    assert seed._document_name(document) == "信息组织基础"
    assert seed._clean_course_name("课程名 转自 DOC") == "课程名"
    assert seed._clean_course_name("课程名_转自_DOC") == "课程名"
