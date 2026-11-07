"""生成层：thinking 开关、空输出不兜底、prompt 形态（Q4 / Q5 / Q7）。"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from src.online_service.generator import (
    EmptyGenerationError,
    _rewrite_answer,
    _split_sentences,
    generate_answer_once,
    generate_answer_with_verification,
)
from src.online_service.persona import PERSONA_PROFILE_VERSION, PERSONA_PROFILES
from src.online_service.query_router import QueryRouter
from src.utils.deepseek_client import generate_text


class FakeCompletions:
    """记录请求；按顺序返回预置内容，用完后重复最后一个。"""

    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.contents) - 1)
        message = SimpleNamespace(
            content=self.contents[index],
            reasoning_content="思考过程" if not self.contents[index] else None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="length")]
        )


def _fake_client(*contents: str) -> Any:
    completions = FakeCompletions(list(contents))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.completions = completions
    return client


class FakeRetriever:
    connected = True
    count = 1

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.hits = [
            {
                "chunk_id": "chunk-1",
                "text": "电子文件管理开设于第五学期。",
                "source_file": "IM127电子文件管理.docx",
                "section": "课程基本信息",
                "score": 0.9,
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM127",
                    "course_name": "电子文件管理",
                },
            }
        ]

    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[dict]:
        self.queries.append(query)
        return self.hits[:top_k]


EVIDENCE = [{"excerpt": "证据正文。", "source_file": "course.docx"}]


# ---------------------------------------------------------------------------
# Q7：thinking 开关
# ---------------------------------------------------------------------------


def test_text_request_disables_thinking_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LLM_THINKING", raising=False)
    client = _fake_client("回答。")
    generate_text(client, "m", "prompt")
    assert client.completions.requests[0]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert "response_format" not in client.completions.requests[0]


def test_text_request_omits_thinking_field_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("LLM_THINKING", "enabled")
    client = _fake_client("回答。")
    generate_text(client, "m", "prompt")
    assert "extra_body" not in client.completions.requests[0]


def test_json_request_always_disables_thinking(monkeypatch) -> None:
    monkeypatch.setenv("LLM_THINKING", "enabled")
    client = _fake_client('{"a": 1}')
    generate_text(client, "m", "prompt", json_mode=True)
    request = client.completions.requests[0]
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert request["response_format"] == {"type": "json_object"}


def test_empty_text_content_logs_finish_reason(monkeypatch, caplog) -> None:
    monkeypatch.delenv("LLM_THINKING", raising=False)
    client = _fake_client("")
    with caplog.at_level(logging.WARNING, logger="src.utils.deepseek_client"):
        assert generate_text(client, "m", "prompt") == ""
    assert "finish_reason=length" in caplog.text
    assert "has_reasoning_content=True" in caplog.text


# ---------------------------------------------------------------------------
# Q7：空输出不再静默兜底
# ---------------------------------------------------------------------------


def test_generate_answer_once_raises_on_empty_content() -> None:
    with pytest.raises(EmptyGenerationError):
        generate_answer_once("有这门课吗", EVIDENCE, _fake_client("   "))


def test_generate_answer_once_without_client_still_falls_back() -> None:
    answer = generate_answer_once("有这门课吗", EVIDENCE, None)
    assert "证据正文" in answer


def test_verification_propagates_first_generation_error() -> None:
    with patch("src.online_service.generator.verify_answer_with_nli") as verify:
        with pytest.raises(EmptyGenerationError):
            generate_answer_with_verification(
                "有这门课吗", EVIDENCE, _fake_client("")
            )
    verify.assert_not_called()


def test_rewrite_empty_content_is_caught_as_rewrite_failure(monkeypatch) -> None:
    monkeypatch.setenv("NLI_MAX_RETRIES", "1")
    neutral = [{"sentence": "旧回答。", "label": "Neutral", "score": 0.1}]
    with (
        patch(
            "src.online_service.generator.generate_answer_once",
            return_value="旧回答。",
        ),
        patch(
            "src.online_service.generator.verify_answer_with_nli",
            return_value=(False, neutral),
        ),
        patch("src.online_service.generator.generate_text", return_value=""),
    ):
        _, metadata = generate_answer_with_verification(
            "有这门课吗", EVIDENCE, object()
        )
    assert metadata["error_code"] == "NLI_REWRITE_FAILED"
    assert metadata["nli_status"] == "unavailable"

    with patch("src.online_service.generator.generate_text", return_value=" "):
        with pytest.raises(EmptyGenerationError):
            _rewrite_answer("q", "a", EVIDENCE, neutral, object(), "student")


@pytest.mark.parametrize("route", ["hybrid", "content"])
def test_router_marks_empty_output_as_degraded(route: str) -> None:
    router = QueryRouter(vector_retriever=FakeRetriever())
    client = _fake_client("")
    handler = getattr(router, f"_handle_{route}")
    result = asyncio.run(handler("电子文件管理这门课是大几才开", client))

    assert result.metadata["status"] == "degraded"
    assert result.metadata["error_code"] == "LLM_EMPTY_OUTPUT"
    assert result.metadata.get("nli_status") != "passed"
    assert "nli_verified" not in result.metadata
    assert result.answer  # 兜底摘要照旧展示
    # 只调了一次生成，没有进入 NLI（NLI 也会走 client）
    assert len(client.completions.requests) == 1


def test_router_keeps_llm_unavailable_for_other_errors() -> None:
    router = QueryRouter(vector_retriever=FakeRetriever())
    with patch(
        "src.online_service.generator.generate_answer_once",
        side_effect=RuntimeError("provider unavailable"),
    ):
        result = asyncio.run(router._handle_hybrid("比较两门课", object()))
    assert result.metadata["error_code"] == "LLM_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Q4：章节顺序约束只在补齐时下发
# ---------------------------------------------------------------------------


def _capture_prompt(call) -> str:
    captured: list[str] = []

    def fake_generate_text(_client, _model, prompt, **_kwargs):
        captured.append(prompt)
        return "回答。"

    with patch("src.online_service.generator.generate_text", fake_generate_text):
        call()
    return captured[0]


def test_order_constraint_only_with_coverage_filler() -> None:
    plain = [{"excerpt": "证据正文。"}]
    filled = [*plain, {"excerpt": "第二章", "coverage_filler": True}]

    assert "证据按原文顺序排列" not in _capture_prompt(
        lambda: generate_answer_once("q", plain, object())
    )
    assert "证据按原文顺序排列" in _capture_prompt(
        lambda: generate_answer_once("q", filled, object())
    )

    def rewrite(items):
        return lambda: _rewrite_answer(
            "q", "旧。", items, [{"sentence": "旧。", "label": "Neutral"}],
            object(), "student",
        )

    assert "证据按原文顺序排列" not in _capture_prompt(rewrite(plain))
    assert "证据按原文顺序排列" in _capture_prompt(rewrite(filled))


# ---------------------------------------------------------------------------
# Q5：回答形态
# ---------------------------------------------------------------------------


def test_generation_prompt_no_longer_forces_three_part_format() -> None:
    prompt = _capture_prompt(lambda: generate_answer_once("q", EVIDENCE, object()))
    assert "回答格式必须是" not in prompt
    assert "第一句必须直接回答问题本身" in prompt
    assert "不列核心内容" in prompt
    assert "与问题无关的内容不要写进回答" in prompt
    assert "无法确认" in prompt
    assert "资料来源由系统自动追加" in prompt

    rewrite_prompt = _capture_prompt(
        lambda: _rewrite_answer(
            "q", "旧。", EVIDENCE, [{"sentence": "旧。", "label": "Neutral"}],
            object(), "student",
        )
    )
    assert "仍需包含直接回答、核心内容" not in rewrite_prompt
    assert "第一句必须直接回答问题本身" in rewrite_prompt


def test_persona_prompts_scope_to_course_content_and_bump_version() -> None:
    assert PERSONA_PROFILE_VERSION == "v2"
    for profile in PERSONA_PROFILES.values():
        assert "当问题涉及课程内容时" in profile["prompt"]


def test_plan_query_includes_user_query_when_course_linked() -> None:
    retriever = FakeRetriever()
    router = QueryRouter(vector_retriever=retriever)
    course = {"course_name": "电子文件管理", "course_code": "IM127"}
    with patch.object(router, "_link_course", return_value=course):
        asyncio.run(router._retrieve_course_evidence("这门课几个学分"))
    assert "电子文件管理 IM127 这门课几个学分" in retriever.queries

    retriever.queries.clear()
    with patch.object(router, "_link_course", return_value=None):
        asyncio.run(router._retrieve_course_evidence("这门课几个学分"))
    assert retriever.queries == ["这门课几个学分", "这门课几个学分"]


def test_sentence_split_without_core_content_label() -> None:
    # 非内容题只有一两句话，没有「核心内容：」标签；资料来源仍被排除。
    answer = (
        "电子文件管理开设在第五学期。该课程为专业必修课。\n\n"
        "资料来源：\n- 电子文件管理 · 课程基本信息 · IM127.docx"
    )
    assert _split_sentences(answer) == [
        "电子文件管理开设在第五学期。",
        "该课程为专业必修课。",
    ]
    # 带标签的内容题切分不变
    labelled = "是的，开设该课程。\n核心内容：\n- 学习文件鉴定。"
    assert _split_sentences(labelled) == ["是的，开设该课程。", "学习文件鉴定。"]
