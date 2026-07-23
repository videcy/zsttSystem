"""Grounded answer generation and optional NLI verification."""

from __future__ import annotations

import re
from json import JSONDecodeError
from typing import Any

from src.config import config
from src.utils.deepseek_client import generate_json, generate_text


def generate_answer_once(query: str, evidence: str, llm_client: Any = None) -> str:
    """Single-pass grounded answer generation used by the lightweight router."""
    if llm_client is None:
        return evidence[:1200]
    prompt = ("仅根据给定证据回答问题，不要补充证据之外的事实。\n"
              f"问题：{query}\n证据：{evidence}")
    return generate_text(llm_client, config.text_model, prompt, temperature=0.1, max_output_tokens=512) or evidence[:1200]


def _split_sentences(answer: str) -> list[str]:
    """Split answer text into sentence-like units."""
    parts = re.split(r"(?<=[。！？!?；;])\s*", answer)
    return [part.strip() for part in parts if part.strip()]


def _classify_verdict(entailment_ratio: float, contradiction_count: int, total: int) -> dict[str, Any]:
    """Classify the overall verification result with graded severity."""
    if total == 0:
        return {"verified": False, "level": "empty", "reason": "No sentences to verify."}

    threshold = config.nli_entailment_threshold
    has_contradiction = contradiction_count > 0

    if entailment_ratio >= threshold and not has_contradiction:
        return {"verified": True, "level": "passed", "reason": ""}
    if entailment_ratio >= threshold and has_contradiction:
        return {
            "verified": False,
            "level": "partial",
            "reason": f"{contradiction_count}/{total} sentences contradict the context.",
        }
    if entailment_ratio >= 0.4 and not has_contradiction:
        return {
            "verified": False,
            "level": "weak",
            "reason": f"Only {entailment_ratio:.0%} sentences verified (threshold {threshold:.0%}).",
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
    threshold = threshold if threshold is not None else config.nli_entailment_threshold
    sentences = _split_sentences(answer)
    if not sentences:
        return False, []

    judge_model = config.judge_model
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


def get_fallback_response() -> dict[str, Any]:
    """Return a safe fallback answer when verification fails."""
    return {
        "answer": "根据当前检索到的资料，我暂时无法给出可靠答案。建议补充更明确的课程材料后再提问。",
        "citations": [],
        "status": "fallback",
    }
