from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.main import QueryRequest
from src.online_service.chroma_retriever import ChromaRetriever
from src.online_service.feedback_handler import build_query_log_record
from src.online_service.generator import build_fallback_answer, generate_answer_once
from src.online_service.persona import PERSONA_PROFILES
from src.online_service.query_router import QueryRouter


class RecordingRetriever:
    connected = True

    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, Any]]] = []

    def search(
        self,
        _query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append((top_k, kwargs))
        return []


@pytest.mark.parametrize(
    ("persona", "syllabus_top_k", "plan_top_k"),
    [
        ("student", 10, 2),
        ("teacher", 8, 5),
        ("visitor", 4, 5),
    ],
)
def test_persona_controls_retrieval_profile(
    persona: str,
    syllabus_top_k: int,
    plan_top_k: int,
) -> None:
    retriever = RecordingRetriever()
    router = QueryRouter(vector_retriever=retriever)
    router.courses = []

    asyncio.run(router._retrieve_course_evidence("课程介绍", persona))

    assert [call[0] for call in retriever.calls] == [
        syllabus_top_k,
        plan_top_k,
    ]
    assert (
        retriever.calls[0][1]["preferred_section_types"]
        == PERSONA_PROFILES[persona]["preferred_sections"]
    )
    assert (
        retriever.calls[1][1]["source_boosts"]
        == PERSONA_PROFILES[persona]["source_boosts"]
    )


def test_persona_is_validated_and_defaults_to_student() -> None:
    assert QueryRequest(query="数据库学什么").persona == "student"
    with pytest.raises(ValidationError):
        QueryRequest(query="数据库学什么", persona="admin")


def test_route_metadata_records_default_and_selected_persona() -> None:
    router = QueryRouter(vector_retriever=RecordingRetriever())
    router.courses = []

    default_result = asyncio.run(router.route("课程介绍", "default"))
    teacher_result = asyncio.run(
        router.route("课程介绍", "teacher", persona="teacher")
    )

    assert default_result.metadata["persona"] == "student"
    assert teacher_result.metadata["persona"] == "teacher"
    assert teacher_result.metadata["persona_mode"] == "retrieval"


def test_all_personas_change_generation_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_generate_text(
        _client: Any,
        _model: str,
        prompt: str,
        **_kwargs: Any,
    ) -> str:
        prompts.append(prompt)
        return "有证据支持的回答"

    monkeypatch.setattr(
        "src.online_service.generator.generate_text",
        fake_generate_text,
    )
    evidence = [
        {
            "course_name": "数据库原理",
            "section": "课程目标",
            "source_file": "数据库原理.docx",
            "excerpt": "掌握关系数据库基础。",
        }
    ]

    for persona in ("student", "teacher", "visitor"):
        generate_answer_once("数据库原理学什么", evidence, object(), persona)

    assert "面向学生回答" in prompts[0]
    assert "面向教师回答" in prompts[1]
    assert "面向非专业访客回答" in prompts[2]


def test_fallback_summary_is_persona_aware() -> None:
    evidence = [
        {
            "course_name": "数据库原理",
            "section": "课程目标",
            "section_type": "course_objectives",
            "source_file": "数据库原理.docx",
            "source_type": "syllabus",
            "excerpt": "掌握关系数据库基础。",
        }
    ]

    student = build_fallback_answer("数据库原理学什么", evidence, "student")
    teacher = build_fallback_answer("数据库原理学什么", evidence, "teacher")
    visitor = build_fallback_answer("数据库原理学什么", evidence, "visitor")

    assert "核心内容：" in student
    assert "教学要点：" in teacher
    assert "课程概览：" in visitor


class FakeCollection:
    metadata: dict[str, Any] = {}

    def count(self) -> int:
        return 2

    def query(self, **_kwargs: Any) -> dict[str, list[list[Any]]]:
        return {
            "ids": [["syllabus", "plan"]],
            "documents": [["课程资料", "培养方案资料"]],
            "metadatas": [
                [
                    {"source_type": "syllabus", "section_type": "basic_info"},
                    {"source_type": "training_plan", "section_type": "overview"},
                ]
            ],
            "distances": [[0.2, 0.2]],
        }


class FakeEncoder:
    def encode(self, _texts: list[str]) -> list[list[float]]:
        return [[0.0]]


def test_source_boost_changes_rerank_order() -> None:
    retriever = ChromaRetriever.__new__(ChromaRetriever)
    retriever.collection = FakeCollection()
    retriever.encoder = FakeEncoder()

    hits = retriever.search(
        "无词面重合",
        2,
        source_boosts={"training_plan": 0.06},
    )

    assert [hit["chunk_id"] for hit in hits] == ["plan", "syllabus"]


def test_query_log_contains_persona_experiment_fields() -> None:
    record = build_query_log_record(
        query_id="query-1",
        query="课程介绍",
        context="",
        kg_path="content",
        response="回答",
        verification=[],
        linked_entities=[],
        citations=[],
        status="ok",
        persona="teacher",
        persona_mode="retrieval",
        persona_profile_version="v1",
    )

    assert record["persona"] == "teacher"
    assert record["persona_mode"] == "retrieval"
    assert record["persona_profile_version"] == "v1"


def test_demo_submits_explicit_persona() -> None:
    template = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "templates"
        / "demo.html"
    ).read_text(encoding="utf-8")

    assert 'id="persona"' in template
    assert '<option value="student" selected>学生</option>' in template
    assert "persona: personaInput.value" in template
