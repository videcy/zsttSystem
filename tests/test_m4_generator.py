from __future__ import annotations

import pytest

from src.online_service.generator import build_fallback_answer


def _evidence(
    excerpt: str,
    *,
    section: str = "课程目标",
    section_type: str = "course_objectives",
) -> dict[str, str]:
    return {
        "course_name": "数据库原理",
        "section": section,
        "section_type": section_type,
        "source_file": "数据库原理.docx",
        "source_type": "syllabus",
        "excerpt": excerpt,
    }


def test_fallback_filters_schedule_total_row() -> None:
    evidence = [
        _evidence(
            "教学进度表\n合计:36\n第一章介绍关系数据库与SQL查询。",
            section="教学进度表",
            section_type="teaching_schedule",
        )
    ]

    answer = build_fallback_answer("数据库原理学什么", evidence, "student")

    assert "合计" not in answer
    assert "关系数据库与SQL查询" in answer


def test_fallback_skips_section_with_only_total_rows() -> None:
    evidence = [
        _evidence(
            "教学进度表\n合计：36",
            section="教学进度表",
            section_type="teaching_schedule",
        )
    ]

    answer = build_fallback_answer("数据库原理学什么", evidence, "student")

    assert "合计" not in answer
    assert "- 教学进度表：" not in answer
    assert "暂无可提炼的具体内容" in answer


@pytest.mark.parametrize("colon", ["：", ":"])
def test_fallback_defers_colon_lead_in(colon: str) -> None:
    evidence = [
        _evidence(
            f"课程目标{colon}\n教学目标与任务{colon}\n掌握关系数据库设计与SQL查询。"
        )
    ]

    answer = build_fallback_answer("数据库原理的课程目标是什么", evidence, "student")

    assert "- 课程目标：掌握关系数据库设计与SQL查询。" in answer
    assert "教学目标与任务" not in answer


def test_fallback_skips_section_when_all_candidates_are_lead_ins() -> None:
    evidence = [
        _evidence("教学目标与任务：\n课程要求："),
    ]

    answer = build_fallback_answer("数据库原理的课程目标是什么", evidence, "student")

    assert "- 课程目标：" not in answer
    assert "暂无可提炼的具体内容" in answer


def test_fallback_learning_summary_unchanged() -> None:
    evidence = [
        _evidence("本课程介绍数据库基础。\n学生应掌握SQL查询与事务处理。")
    ]

    answer = build_fallback_answer("数据库原理学什么", evidence, "student")

    assert "- 课程目标：学生应掌握SQL查询与事务处理。" in answer
