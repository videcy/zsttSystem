"""M1/M2 E2b-E2c citation field tests.

These tests pin the citation dictionaries produced by the catalog and
structure routes.  They are fully offline: catalogue rows, document chunks,
and doc-tree lookups are constructed in-process.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.data_processing.doc_tree import build_doc_tree
from src.online_service.doc_tree import DocTree
from src.online_service.query_router import QueryRouter


COURSE_CODE = "IM127"
COURSE_NAME = "信息素养与创新思维"
DOC_HASH = "hashIM127"
SOURCE_FILE = "2025-1教学大纲汇总/IM127信息素养与创新思维.docx"
PROGRAM_NAME = "信息管理学院2025级档案学专业培养方案"


def _catalog_course(
    course_code: str,
    source_file: str,
    *,
    semester: int,
) -> dict[str, Any]:
    return {
        "course_code": course_code,
        "course_name": f"课程{course_code}",
        "offerings": [
            {
                "program_name": PROGRAM_NAME,
                "program_type": "主修专业",
                "course_category": "专业必修课",
                "course_subcategory": "",
                "source_file": source_file,
                "semester": semester,
            }
        ],
    }


def test_catalog_citation_aggregates_course_codes_by_source_file() -> None:
    router = QueryRouter.__new__(QueryRouter)
    router.courses = [
        _catalog_course("IM001", "plans/档案学培养方案.xlsx", semester=1),
        _catalog_course("IM002", "plans/档案学培养方案.xlsx", semester=2),
        _catalog_course("IM003", "plans/档案学培养方案-补充.xlsx", semester=3),
    ]

    result = router._handle_catalog("档案学专业培养方案有哪些专业必修课？")

    assert result.query_type == "catalog"
    assert len(result.citations) == 2
    by_source = {
        citation["source_file"]: citation for citation in result.citations
    }
    primary = by_source["档案学培养方案.xlsx"]
    assert primary["course_code"] is None
    assert primary["course_codes"] == ["IM001", "IM002"]
    assert by_source["档案学培养方案-补充.xlsx"]["course_codes"] == ["IM003"]


def _chunk(
    chunk_id: str,
    chunk_index: int,
    heading_path: list[str],
    text: str,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_hash": DOC_HASH,
        "chunk_index": chunk_index,
        "heading_path": heading_path,
        "text": text,
        "source_file": SOURCE_FILE,
        "metadata": {
            "source_type": "syllabus",
            "section_type": "teaching_schedule",
            "course_code": COURSE_CODE,
            "course_name": COURSE_NAME,
            "source_year": "2025",
        },
    }


def _doc_tree() -> DocTree:
    schedule = ["四、教学进度"]
    return DocTree(
        build_doc_tree(
            [
                _chunk(
                    "h2",
                    2,
                    schedule + ["第一章 课程导论"],
                    "章节次序及名称：第一章 课程导论\n学时：2",
                ),
                _chunk(
                    "h3",
                    3,
                    schedule + ["第二章 搜寻信息资源"],
                    "章节次序及名称：第二章 搜寻信息资源\n学时：4",
                ),
                _chunk(
                    "h4",
                    4,
                    schedule + ["第三章 信息检索"],
                    "章节次序及名称：第三章 信息检索\n学时：4",
                ),
            ]
        )
    )


def _structure_router() -> QueryRouter:
    router = QueryRouter.__new__(QueryRouter)
    router.courses = [
        {"course_code": COURSE_CODE, "course_name": COURSE_NAME, "credits": 2}
    ]
    router._doc_tree_override = _doc_tree()
    return router


def test_chapter_citation_uses_the_chapter_nodes_first_chunk_id() -> None:
    result = asyncio.run(
        _structure_router().route("《信息素养与创新思维》第二章讲什么？", "chapter-2")
    )

    assert result.query_type == "structure"
    assert result.citations[0]["section"] == "第二章 搜寻信息资源"
    assert result.citations[0]["chunk_id"] == "h3"


def test_outline_citation_uses_first_chunk_id_in_document_order() -> None:
    result = asyncio.run(
        _structure_router().route("《信息素养与创新思维》一共有几章？", "outline")
    )

    assert result.query_type == "structure"
    assert result.citations[0]["section"] == "教学进度"
    assert result.citations[0]["chunk_id"] == "h2"


def test_citation_omits_chunk_id_when_nodes_have_no_chunk_ids() -> None:
    doc_tree = _doc_tree()
    chapter = doc_tree.chapter(COURSE_CODE, 2)
    assert chapter is not None
    chapter.chunk_ids = []

    _, chapter_citations = QueryRouter._render_chapter_answer(
        {"course_code": COURSE_CODE, "course_name": COURSE_NAME},
        doc_tree,
        chapter,
    )
    assert "chunk_id" not in chapter_citations[0]

    outline = doc_tree.outline(COURSE_CODE)
    for node in outline:
        node.chunk_ids = []
    _, outline_citations, _ = QueryRouter._render_outline_answer(
        {"course_code": COURSE_CODE, "course_name": COURSE_NAME},
        doc_tree,
        outline,
    )
    assert "chunk_id" not in outline_citations[0]


def test_frontend_template_still_does_not_render_item_chunk_id() -> None:
    template = (
        Path(__file__).resolve().parents[1] / "src" / "templates" / "demo.html"
    ).read_text(encoding="utf-8")

    assert "item.chunk_id" not in template
