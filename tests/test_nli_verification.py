from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from src.online_service.generator import (
    generate_answer_with_verification,
    select_generation_evidence,
    verify_answer_with_nli,
)
from src.online_service.query_router import QueryRouter


def _result(label: str, score: float = 0.9) -> dict[str, Any]:
    return {"sentence": "claim", "label": label, "score": score}


def test_nli_threshold_override_and_contradictions() -> None:
    answer = "第一句。第二句。第三句。"
    with patch(
        "src.online_service.generator._run_single_nli",
        side_effect=[_result("Entailment"), _result("Entailment"), _result("Neutral")],
    ):
        assert verify_answer_with_nli(answer, "context", object(), threshold=0.6)[0]
    with patch(
        "src.online_service.generator._run_single_nli",
        side_effect=[_result("Entailment"), _result("Entailment"), _result("Neutral")],
    ):
        assert not verify_answer_with_nli(answer, "context", object(), threshold=0.8)[0]
    with patch(
        "src.online_service.generator._run_single_nli",
        side_effect=[
            _result("Entailment"),
            _result("Entailment"),
            _result("Contradiction"),
        ],
    ):
        assert not verify_answer_with_nli(answer, "context", object(), threshold=0.6)[0]


def test_nli_ignores_headings_and_source_listing() -> None:
    checked: list[str] = []

    def record(sentence, _context, _client, _model):
        checked.append(sentence)
        return _result("Entailment")

    answer = (
        "数据库课程介绍关系模型。\n\n"
        "核心内容：\n- 学习SQL查询。\n\n"
        "资料来源：\n- 数据库原理 · 第一章 · database.docx"
    )
    with patch("src.online_service.generator._run_single_nli", side_effect=record):
        verified, _details = verify_answer_with_nli(answer, "context", object())

    assert verified
    assert checked == ["数据库课程介绍关系模型。", "学习SQL查询。"]


def test_nli_rejects_unknown_labels_and_invalid_scores() -> None:
    with patch(
        "src.online_service.generator.generate_json",
        return_value={"label": "Not Entailment", "score": "invalid"},
    ):
        verified, details = verify_answer_with_nli("一条陈述。", "context", object())

    assert verified is False
    assert details == [
        {"sentence": "一条陈述。", "label": "Unknown", "score": 0.0}
    ]


def test_generation_rewrites_once_and_then_passes(monkeypatch) -> None:
    monkeypatch.setenv("NLI_MAX_RETRIES", "1")
    evidence = [{"excerpt": "课程学习线性规划。", "source_file": "course.docx"}]
    first_details = [{"sentence": "学习量子计算。", "label": "Neutral", "score": 0.2}]
    second_details = [{"sentence": "学习线性规划。", "label": "Entailment", "score": 0.9}]

    with (
        patch("src.online_service.generator.generate_answer_once", return_value="学习量子计算。"),
        patch(
            "src.online_service.generator.verify_answer_with_nli",
            side_effect=[(False, first_details), (True, second_details)],
        ),
        patch("src.online_service.generator._rewrite_answer", return_value="学习线性规划。") as rewrite,
    ):
        answer, metadata = generate_answer_with_verification(
            "课程学什么",
            evidence,
            object(),
        )

    assert answer.startswith("学习线性规划。")
    assert "资料来源：\n- course.docx" in answer
    assert metadata["nli_verified"] is True
    assert metadata["nli_status"] == "rewritten"
    assert metadata["nli_attempts"] == 2
    rewrite.assert_called_once()


def test_source_heading_cannot_hide_a_trailing_claim() -> None:
    checked: list[str] = []

    def record(sentence, _context, _client, _model):
        checked.append(sentence)
        return _result("Entailment")

    answer = (
        "数据库课程介绍关系模型。\n"
        "资料来源：\n"
        "- database.docx\n"
        "该课程还教授量子计算。"
    )
    with patch("src.online_service.generator._run_single_nli", side_effect=record):
        verified, _details = verify_answer_with_nli(answer, "context", object())

    assert verified
    assert checked == ["数据库课程介绍关系模型。", "该课程还教授量子计算。"]


def test_unknown_judge_results_retry_without_rewriting(monkeypatch) -> None:
    monkeypatch.setenv("NLI_MAX_RETRIES", "1")
    unknown = [{"sentence": "学习线性规划。", "label": "Unknown", "score": 0.0}]
    with (
        patch(
            "src.online_service.generator.generate_answer_once",
            return_value="学习线性规划。",
        ),
        patch(
            "src.online_service.generator.verify_answer_with_nli",
            side_effect=[(False, unknown), (False, unknown)],
        ) as verify,
        patch("src.online_service.generator._rewrite_answer") as rewrite,
    ):
        answer, metadata = generate_answer_with_verification(
            "课程学什么",
            [{"excerpt": "课程学习线性规划。"}],
            object(),
        )

    assert "无法给出可靠答案" in answer
    assert metadata["nli_status"] == "unavailable"
    assert metadata["error_code"] == "NLI_UNAVAILABLE"
    assert metadata["nli_attempts"] == 2
    assert verify.call_count == 2
    rewrite.assert_not_called()


def test_rewrite_failure_returns_a_specific_safe_fallback(monkeypatch) -> None:
    monkeypatch.setenv("NLI_MAX_RETRIES", "1")
    neutral = [{"sentence": "未知事实。", "label": "Neutral", "score": 0.2}]
    with (
        patch(
            "src.online_service.generator.generate_answer_once",
            return_value="未知事实。",
        ),
        patch(
            "src.online_service.generator.verify_answer_with_nli",
            return_value=(False, neutral),
        ),
        patch(
            "src.online_service.generator._rewrite_answer",
            side_effect=RuntimeError("provider unavailable"),
        ),
    ):
        answer, metadata = generate_answer_with_verification(
            "课程学什么",
            [{"excerpt": "已知事实。"}],
            object(),
        )

    assert "无法给出可靠答案" in answer
    assert metadata["nli_status"] == "unavailable"
    assert metadata["error_code"] == "NLI_REWRITE_FAILED"
    assert metadata["nli_verification_target"] == "discarded_generated_answer"


def test_generation_refuses_after_contradictory_retries(monkeypatch) -> None:
    monkeypatch.setenv("NLI_MAX_RETRIES", "1")
    details = [
        {"sentence": "课程不学习线性规划。", "label": "Contradiction", "score": 0.95}
    ]
    with (
        patch("src.online_service.generator.generate_answer_once", return_value="课程不学习线性规划。"),
        patch(
            "src.online_service.generator.verify_answer_with_nli",
            side_effect=[(False, details), (False, details)],
        ),
        patch("src.online_service.generator._rewrite_answer", return_value="仍然错误。"),
    ):
        answer, metadata = generate_answer_with_verification(
            "课程学什么",
            [{"excerpt": "课程学习线性规划。"}],
            object(),
        )

    assert "无法给出可靠答案" in answer
    assert metadata["nli_verified"] is False
    assert metadata["nli_status"] == "refused"
    assert metadata["error_code"] == "NLI_VERIFICATION_FAILED"


def test_generation_prunes_neutral_claims_after_verification(monkeypatch) -> None:
    monkeypatch.setenv("NLI_MAX_RETRIES", "0")
    details = [
        {"sentence": "课程学习线性规划。", "label": "Entailment", "score": 0.95},
        {"sentence": "课程学习量子计算。", "label": "Neutral", "score": 0.2},
    ]
    evidence = [
        {
            "excerpt": "课程学习线性规划。",
            "course_name": "管理运筹学",
            "section": "课程目标",
            "source_file": "course.docx",
        }
    ]
    with (
        patch(
            "src.online_service.generator.generate_answer_once",
            return_value="课程学习线性规划。课程学习量子计算。",
        ),
        patch(
            "src.online_service.generator.verify_answer_with_nli",
            return_value=(True, details),
        ),
    ):
        answer, metadata = generate_answer_with_verification(
            "课程学什么",
            evidence,
            object(),
        )

    assert "课程学习线性规划。" in answer
    assert "量子计算" not in answer
    assert "管理运筹学 · 课程目标 · course.docx" in answer
    assert metadata["nli_status"] == "pruned"
    assert metadata["nli_verification_target"] == "retained_claims"


def test_generation_skips_nli_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("NLI_VERIFICATION_ENABLED", "false")
    with (
        patch(
            "src.online_service.generator.generate_answer_once",
            return_value="课程学习线性规划。",
        ),
        patch("src.online_service.generator.verify_answer_with_nli") as verify,
    ):
        answer, metadata = generate_answer_with_verification(
            "课程学什么",
            [{"excerpt": "课程学习线性规划。", "source_file": "course.docx"}],
            object(),
        )

    assert "资料来源：\n- course.docx" in answer
    assert metadata["nli_status"] == "skipped"
    assert metadata["nli_verification_target"] == "returned_answer"
    verify.assert_not_called()


def test_generation_sources_and_api_citations_share_the_same_evidence_budget(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NLI_VERIFICATION_ENABLED", "false")
    evidence = [
        {
            "excerpt": f"证据{i}",
            "course_name": f"课程{i}",
            "section": f"章节{i}",
            "source_file": f"source-{i}.docx",
        }
        for i in range(11)
    ]
    received: list[dict[str, Any]] = []

    def generate(_query, selected, _client, _persona):
        received.extend(selected)
        return "可验证回答。"

    with patch(
        "src.online_service.generator.generate_answer_once",
        side_effect=generate,
    ):
        answer, _metadata = generate_answer_with_verification(
            "问题",
            evidence,
            object(),
        )

    selected = select_generation_evidence(evidence)
    citations = QueryRouter._evidence_citations(selected)
    assert received == selected
    assert len(selected) == len(citations) == 10
    assert "source-9.docx" in answer
    assert "source-10.docx" not in answer


def test_evidence_budget_reserves_a_slot_for_hybrid_graph_context() -> None:
    evidence = [
        {
            "excerpt": f"向量证据{i}",
            "source_file": f"vector-{i}.docx",
            "source_type": "syllabus",
        }
        for i in range(10)
    ]
    evidence.append(
        {
            "excerpt": "线性规划 → 整数规划",
            "source_file": "Neo4j",
            "source_type": "knowledge_graph",
            "section": "知识图谱关联",
        }
    )

    selected = select_generation_evidence(evidence)

    assert len(selected) == 10
    assert selected[-1]["source_type"] == "knowledge_graph"
    assert QueryRouter._evidence_citations(selected)[-1]["source_file"] == "Neo4j"


class _Retriever:
    connected = True

    def search(self, _query: str, top_k: int = 5, **kwargs: Any):
        if kwargs.get("source_types") == ("training_plan",):
            return []
        return [
            {
                "chunk_id": "chunk-1",
                "text": "课程学习线性规划。",
                "source_file": "course.docx",
                "section": "课程目标",
                "score": 0.9,
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM399",
                    "course_name": "管理运筹学",
                },
            }
        ][:top_k]


@pytest.mark.parametrize("query", ["课程内容是什么", "比较课程知识与实际应用"])
def test_content_and_hybrid_routes_publish_nli_metadata(query: str) -> None:
    router = QueryRouter(vector_retriever=_Retriever())
    router.courses = []
    verification = {
        "nli_verified": True,
        "nli_status": "passed",
        "nli_details": [{"sentence": "课程学习线性规划。", "label": "Entailment"}],
        "nli_attempts": 1,
    }
    with patch(
        "src.online_service.generator.generate_answer_with_verification",
        return_value=("课程学习线性规划。", verification),
    ):
        result = asyncio.run(router.route(query, "query-1", llm_client=object()))

    assert result.metadata["nli_verified"] is True
    assert result.metadata["nli_status"] == "passed"
    assert result.metadata["nli_details"][0]["label"] == "Entailment"


def test_route_clears_citations_when_verification_refuses_answer() -> None:
    router = QueryRouter(vector_retriever=_Retriever())
    router.courses = []
    verification = {
        "nli_verified": False,
        "nli_status": "refused",
        "nli_details": [{"sentence": "错误。", "label": "Contradiction"}],
        "nli_attempts": 2,
        "status": "fallback",
        "error_code": "NLI_VERIFICATION_FAILED",
    }
    with patch(
        "src.online_service.generator.generate_answer_with_verification",
        return_value=("根据资料无法给出可靠答案。", verification),
    ):
        result = asyncio.run(
            router.route("课程内容是什么", "query-2", llm_client=object())
        )

    assert result.citations == []
    assert result.metadata["status"] == "fallback"
    assert result.metadata["error_code"] == "NLI_VERIFICATION_FAILED"
