"""Answer generation and hallucination interception helpers."""

from __future__ import annotations

import os
import re
from json import JSONDecodeError
from typing import Any

from src.utils.deepseek_client import generate_json, generate_text

NLI_ENTAILMENT_THRESHOLD = float(os.getenv("NLI_ENTAILMENT_THRESHOLD", "0.6"))
NLI_MAX_RETRIES = int(os.getenv("NLI_MAX_RETRIES", "1"))


def assemble_prompt(query: str, context: str, kg_path: str) -> str:
    """Assemble a grounded generation prompt."""
    return f"""
你是一名严格遵循证据的教学问答助手。请基于给定材料回答问题。

【用户问题】
{query}

【参考资料-教材原文】
{context or "无可用教材原文"}

【参考资料-知识关联路径】
{kg_path or "无可用知识关联路径"}

【生成要求】
1. 必须严格基于参考资料作答，不得使用未提供的外部知识。
2. 回答时尽量条理清晰，适合教学问答场景。
3. 必须明确引用依据，引用格式可使用"依据：……"。
4. 如果参考资料不足以支持完整回答，必须明确回答"我不知道"或"资料不足"。
5. 不要编造教师、教材、定义、公式或推导过程。
"""


def assemble_retry_prompt(query: str, context: str, kg_path: str, prev_answer: str) -> str:
    """Assemble a shorter, more constrained retry prompt."""
    return f"""
你是一名严格遵循证据的教学问答助手。请仅基于给定材料，用 **不超过3句话** 直接回答问题。

【用户问题】
{query}

【参考资料-教材原文】
{context or "无可用教材原文"}

【参考资料-知识关联路径】
{kg_path or "无可用知识关联路径"}

【生成要求】
1. 回答必须用 3 句以内完成。
2. 每句话必须能从参考资料中找到直接支撑。
3. 如果不能基于资料回答，输出"资料不足"。

上一轮回答因部分陈述未被资料支撑而被拒绝，请确保本轮回答更保守、更精确。

上一轮回答（供参考，不要重复）：
{prev_answer[:200]}
"""


def generate_answer(prompt: str, llm_client: Any) -> str:
    """Generate a low-temperature answer with the LLM."""
    answer = generate_text(
        llm_client,
        os.getenv("TEXT_MODEL", "deepseek-v4-flash"),
        prompt,
        temperature=0.1,
        max_output_tokens=512,
    )
    return clean_markdown_format(answer)


def clean_markdown_format(text: str) -> str:
    """Normalise Markdown formatting for plain-text display."""
    cleaned = text.replace("**", "")
    cleaned = re.sub(r"^-\s", "", cleaned, flags=re.MULTILINE)
    return cleaned


def _split_sentences(answer: str) -> list[str]:
    """Split answer text into sentence-like units."""
    parts = re.split(r"(?<=[。！？!?；;])\s*", answer)
    return [part.strip() for part in parts if part.strip()]


def _classify_verdict(entailment_ratio: float, contradiction_count: int, total: int) -> dict[str, Any]:
    """Classify the overall verification result with graded severity."""
    if total == 0:
        return {"verified": False, "level": "empty", "reason": "No sentences to verify."}

    has_contradiction = contradiction_count > 0

    if entailment_ratio >= NLI_ENTAILMENT_THRESHOLD and not has_contradiction:
        return {"verified": True, "level": "passed", "reason": ""}
    if entailment_ratio >= NLI_ENTAILMENT_THRESHOLD and has_contradiction:
        return {
            "verified": False,
            "level": "partial",
            "reason": f"{contradiction_count}/{total} sentences contradict the context.",
        }
    if entailment_ratio >= 0.4 and not has_contradiction:
        return {
            "verified": False,
            "level": "weak",
            "reason": f"Only {entailment_ratio:.0%} sentences verified (threshold {NLI_ENTAILMENT_THRESHOLD:.0%}).",
        }
    return {
        "verified": False,
        "level": "failed",
        "reason": f"{entailment_ratio:.0%} entailment with {contradiction_count} contradictions.",
    }


def _run_single_nli(sentence: str, context: str, nli_client: Any, judge_model: str) -> dict[str, Any]:
    """Run NLI check for one sentence."""
    prompt = f"""
你是一名事实校验助手。请判断下面句子是否能被给定上下文支持。

上下文：
{context}

待验证句子：
{sentence}

请严格输出 JSON：
{{
  "label": "Entailment" | "Neutral" | "Contradiction",
  "score": 0.0
}}
"""
    try:
        result = generate_json(
            nli_client,
            judge_model,
            prompt,
            temperature=0.0,
            max_output_tokens=120,
        )
    except (ValueError, JSONDecodeError):
        return {
            "sentence": sentence,
            "label": "Unknown",
            "score": 0.0,
            "is_unknown": True,
        }

    label = str(result.get("label", "Neutral"))
    score = float(result.get("score", 0.0))
    return {
        "sentence": sentence,
        "label": label,
        "score": score,
        "is_unknown": False,
    }


def verify_answer_with_nli(
    answer: str,
    context: str,
    nli_model: Any,
    *,
    threshold: float | None = None,
) -> tuple[bool, list[dict[str, str]]]:
    """Check whether answer sentences are supported by context.

    Uses proportion-based verification: an answer passes when
    >= threshold (default 0.6) sentences are entailed AND there are
    zero Contradiction sentences.  Unknown (LLM parse error) sentences
    count as non-entailed.
    """
    threshold = threshold if threshold is not None else NLI_ENTAILMENT_THRESHOLD
    sentences = _split_sentences(answer)
    if not sentences:
        return False, []

    judge_model = os.getenv("JUDGE_MODEL", os.getenv("TEXT_MODEL", "deepseek-v4-flash"))
    details: list[dict[str, str]] = []
    entailed_count = 0
    contradiction_count = 0
    total = 0

    for sentence in sentences:
        result = _run_single_nli(sentence, context, nli_model, judge_model)
        total += 1
        label = result["label"]
        if "entail" in label.lower():
            entailed_count += 1
        elif "contradict" in label.lower():
            contradiction_count += 1

        details.append({
            "sentence": sentence,
            "label": label,
            "score": f"{result['score']:.4f}",
        })

    entailment_ratio = entailed_count / total if total > 0 else 0.0
    verdict = _classify_verdict(entailment_ratio, contradiction_count, total)
    return verdict["verified"], details


def retry_generate_with_expanded_context(
    query: str,
    context: str,
    kg_path: str,
    llm_client: Any,
    prev_answer: str = "",
) -> str | None:
    """Attempt a shorter, more conservative generation when the first answer fails NLI."""
    prompt = assemble_retry_prompt(query, context, kg_path, prev_answer)
    try:
        answer = generate_text(
            llm_client,
            os.getenv("TEXT_MODEL", "deepseek-v4-flash"),
            prompt,
            temperature=0.0,
            max_output_tokens=256,
        )
        return clean_markdown_format(answer) if answer else None
    except Exception:
        return None


def get_fallback_response() -> dict[str, Any]:
    """Return a safe fallback answer when verification fails."""
    return {
        "answer": "根据当前检索到的资料，我暂时无法给出可靠答案。建议补充更明确的课程材料后再提问。",
        "citations": [],
        "status": "fallback",
    }
