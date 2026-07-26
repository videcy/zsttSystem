"""Fixed persona profiles shared by retrieval and answer generation."""

from __future__ import annotations

from typing import Any, Literal


Persona = Literal["student", "teacher", "visitor"]
DEFAULT_PERSONA: Persona = "student"
PERSONA_MODE = "retrieval"
PERSONA_PROFILE_VERSION = "v1"

PERSONA_PROFILES: dict[Persona, dict[str, Any]] = {
    "student": {
        "syllabus_top_k": 10,
        "plan_top_k": 2,
        "preferred_sections": (
            "course_objectives",
            "teaching_schedule",
            "prerequisites",
        ),
        "source_boosts": {"syllabus": 0.04},
        "prompt": (
            "面向学生回答。优先解释课程主要内容、先修基础、学习难点和建议顺序。"
            "第一次出现的专业术语用一句通俗语言解释。"
        ),
        "fallback_heading": "核心内容：",
        "fallback_limit": 4,
    },
    "teacher": {
        "syllabus_top_k": 8,
        "plan_top_k": 5,
        "preferred_sections": (
            "course_objectives",
            "assessment",
            "teaching_schedule",
        ),
        "source_boosts": {"training_plan": 0.06},
        "prompt": (
            "面向教师回答。优先说明课程目标、教学内容、考核方式及其与培养方案的对应。"
            "保留必要教学设计术语，并明确证据来源。"
        ),
        "fallback_heading": "教学要点：",
        "fallback_limit": 5,
    },
    "visitor": {
        "syllabus_top_k": 4,
        "plan_top_k": 5,
        "preferred_sections": (
            "overview",
            "basic_info",
            "course_objectives",
        ),
        "source_boosts": {"training_plan": 0.04},
        "prompt": (
            "面向非专业访客回答。先说明课程或专业定位，减少未解释术语，"
            "不展开内部教学细节。"
        ),
        "fallback_heading": "课程概览：",
        "fallback_limit": 3,
    },
}

