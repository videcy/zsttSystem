from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_processing.parser_chunker import SyllabusChunker
from src.data_processing.graph_builder import build_graph_records, neo4j_properties


def test_training_plan_parser_extracts_fields_and_stops_at_invalid_rows() -> None:
    dataframe = pd.DataFrame(
        [
            ["信息管理学院2025级信息管理与信息系统专业培养方案"],
            [
                "课程细类",
                "",
                "课程编码",
                "课程中文名称/英文名称",
                "",
                "",
                "学分情况",
                "",
                "",
                "学时情况",
                "",
                "",
                "开课学期",
                "课程负责人",
            ],
            [
                "",
                "",
                "",
                "",
                "",
                "",
                "总学分",
                "理论学分",
                "实验实践学分",
                "总学时",
                "理论学时",
                "实验实践学时",
                "",
                "",
            ],
            [
                "专业必修课",
                "专业核心课",
                "IM399",
                "管理运筹学\nOperations Research",
                "",
                "",
                3,
                3,
                0,
                54,
                54,
                0,
                4,
                "朱侯，聂卉",
            ],
            ["", "", "IM400", "决策分析", "", "", 2, 2, 0, 36, 36, 0, 5, "教师"],
            ["", "", "大一下", "9", "", "", "", "", "", "", "", "", "", ""],
        ]
    )

    catalog = SyllabusChunker()._extract_sheet_course_catalog(
        dataframe,
        Path("02：主修专业使用-25级信息管理与信息系统.xlsx"),
        "主修专业培养方案",
    )

    assert set(catalog) == {"IM399", "IM400"}
    course = catalog["IM399"]
    assert course["english_name"] == "Operations Research"
    assert course["credits"] == 3
    assert course["hours"] == 54
    assert course["semester"] == 4
    assert course["instructor"] == "朱侯，聂卉"
    assert course["course_category"] == "专业必修课"
    assert course["course_subcategory"] == "专业核心课"


def test_training_plan_memberships_are_merged_and_indexed_separately() -> None:
    chunker = SyllabusChunker()
    first = {
        "IM399": {
            "course_code": "IM399",
            "course_name": "管理运筹学",
            "english_name": "Operations Research",
            "credits": 3,
            "prerequisites": [],
            "program_names": ["主修方案"],
            "offerings": [
                {
                    "program_name": "主修方案",
                    "program_type": "主修专业",
                    "course_category": "专业必修课",
                    "course_subcategory": "专业基础课",
                    "credits": 3,
                    "source_file": "main.xlsx",
                    "sheet_name": "Sheet1",
                }
            ],
        }
    }
    second = {
        "IM399": {
            **first["IM399"],
            "program_names": ["辅修方案"],
            "offerings": [
                {
                    "program_name": "辅修方案",
                    "program_type": "辅修专业",
                    "course_category": "辅修课程",
                    "course_subcategory": "",
                    "credits": 3,
                    "source_file": "minor.xlsx",
                    "sheet_name": "Sheet1",
                }
            ],
        }
    }

    chunker._merge_catalog(first, second)
    chunks = chunker.build_training_plan_chunks(first)

    assert first["IM399"]["program_names"] == ["主修方案", "辅修方案"]
    assert len(first["IM399"]["offerings"]) == 2
    assert len(chunks) == 2
    assert all(chunk["metadata"]["source_type"] == "training_plan" for chunk in chunks)


def test_filename_course_code_is_not_replaced_by_fuzzy_name_match() -> None:
    chunker = SyllabusChunker()
    metadata = chunker._match_course_metadata(
        Path("IM113信息素养与信息检索通用教程.docx"),
        [{"section_title": "课程简介", "content": "信息检索基础"}],
        {
            "IM282": {
                "course_code": "IM282",
                "course_name": "信息检索",
                "credits": 2,
            }
        },
    )

    assert metadata["course_code"] == "IM113"


def test_nested_course_memberships_are_neo4j_compatible() -> None:
    properties = neo4j_properties(
        {
            "course_code": "IM399",
            "program_names": ["主修方案", "辅修方案"],
            "offerings": [{"program_name": "主修方案"}],
        }
    )

    assert properties["program_names"] == ["主修方案", "辅修方案"]
    assert '"program_name": "主修方案"' in properties["offerings"]


def test_im399_syllabus_is_split_into_objectives_and_schedule_rows() -> None:
    project_root = Path(__file__).resolve().parents[1]
    syllabus_root = project_root / "data" / "syllabi"
    syllabus_path = next(syllabus_root.rglob("IM399*.docx"))
    chunker = SyllabusChunker()
    catalog = {
        "IM399": {
            "course_code": "IM399",
            "course_name": "管理运筹学",
            "credits": 3,
            "prerequisites": [],
        }
    }

    chunks = chunker.build_syllabus_chunks(
        syllabus_path,
        syllabus_root,
        catalog,
    )

    objective = [
        chunk
        for chunk in chunks
        if chunk["metadata"]["section_type"] == "course_objectives"
    ]
    linear_programming = [
        chunk
        for chunk in chunks
        if chunk["metadata"]["section_type"] == "teaching_schedule"
        and "第一章 线性规划" in chunk["text"]
    ]
    assert objective and "单纯" in objective[0]["text"]
    assert linear_programming and "数学模型" in linear_programming[0]["text"]
    assert max(len(chunk["text"]) for chunk in chunks) <= 750
    assert all(chunk["metadata"]["parent_document"] for chunk in chunks)
    assert all(chunk["metadata"]["source_year"] == "2025" for chunk in chunks)
    assert "要求有一定的字数" not in "\n".join(
        chunk["text"] for chunk in chunks
    )


def test_syllabus_prerequisite_is_resolved_with_traceable_evidence(
    tmp_path: Path,
) -> None:
    from docx import Document

    syllabus_root = tmp_path / "syllabi"
    syllabus_root.mkdir()
    syllabus_path = syllabus_root / "IM2105信息组织基础.docx"
    document = Document()
    table = document.add_table(rows=3, cols=4)
    values = [
        ("课程代码", "IM2105", "课程名称", "信息组织基础"),
        ("学分", "3", "课程类别", "专业课"),
        ("先修课程", "《信息管理学基础》", "课程目标", "掌握信息组织方法"),
    ]
    for row, row_values in zip(table.rows, values):
        for cell, value in zip(row.cells, row_values):
            cell.text = value
    document.save(syllabus_path)

    catalog = {
        "IM121": {
            "course_code": "IM121",
            "course_name": "信息管理学基础",
            "english_name": "Foundations of Information Management",
            "prerequisites": [],
        },
        "IM2105": {
            "course_code": "IM2105",
            "course_name": "信息组织基础",
            "prerequisites": [],
        },
    }

    chunks = SyllabusChunker().build_syllabus_chunks(
        syllabus_path,
        syllabus_root,
        catalog,
    )
    basic_info = next(
        chunk
        for chunk in chunks
        if chunk["metadata"]["section_type"] == "basic_info"
    )
    metadata = basic_info["metadata"]

    assert metadata["prerequisites"] == ["IM121"]
    assert metadata["unresolved_prerequisites"] == []
    assert metadata["prerequisite_evidence"] == [
        {
            "raw_name": "《信息管理学基础》",
            "course_code": "IM121",
            "course_name": "信息管理学基础",
            "relation": "PREREQUISITE_OF",
            "confidence": 1.0,
            "source_type": "syllabus",
            "source_file": "IM2105信息组织基础.docx",
            "section": "课程基本信息",
            "source_year": "",
            "verified": True,
        }
    ]


def test_unmatched_prerequisite_is_not_promoted_to_hard_edge() -> None:
    chunker = SyllabusChunker()
    resolved, unresolved = chunker.resolve_prerequisites(
        ["高等数学"],
        {
            "IM2105": {
                "course_code": "IM2105",
                "course_name": "信息组织基础",
            }
        },
        dependent_course_code="IM2105",
        source_file="IM2105.docx",
        section="课程基本信息",
        source_year="2025",
    )

    assert resolved == []
    assert unresolved == [
        {
            "raw_name": "高等数学",
            "dependent_course_code": "IM2105",
            "source_file": "IM2105.docx",
            "section": "课程基本信息",
            "source_year": "2025",
            "reason": "not_found",
        }
    ]


def test_prerequisite_graph_edge_preserves_evidence() -> None:
    courses = [
        {"course_code": "IM121", "course_name": "信息管理学基础"},
        {
            "course_code": "IM2105",
            "course_name": "信息组织基础",
            "prerequisites": ["IM121"],
            "prerequisite_evidence": [
                {
                    "raw_name": "信息管理学基础",
                    "course_code": "IM121",
                    "source_file": "IM2105.docx",
                    "section": "课程基本信息",
                    "confidence": 1.0,
                    "verified": True,
                }
            ],
        },
    ]

    graph = build_graph_records(courses, [], [])

    assert graph["edges"] == [
        {
            "source": "IM121",
            "target": "IM2105",
            "type": "PREREQUISITE_OF",
            "raw_name": "信息管理学基础",
            "source_file": "IM2105.docx",
            "section": "课程基本信息",
            "confidence": 1.0,
            "verified": True,
        }
    ]
