"""Grounded answer generation and optional NLI verification."""

from __future__ import annotations

import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from src.config import config
from src.online_service.persona import DEFAULT_PERSONA, PERSONA_PROFILES, Persona
from src.utils.deepseek_client import generate_json, generate_text


def _source_name(value: Any) -> str:
    normalized = str(value or "").replace("\\", "/").split("#", 1)[0]
    return Path(normalized).name or "本地知识库"


def _format_evidence_for_prompt(evidence_items: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(evidence_items[:10], 1):
        blocks.append(
            "\n".join(
                [
                    f"证据 {index}",
                    f"课程：{item.get('course_name') or '未标注'}"
                    f"（{item.get('course_code') or '无代码'}）",
                    f"章节：{item.get('section') or '未标注'}",
                    f"文件：{_source_name(item.get('source_file'))}",
                    f"内容：{str(item.get('excerpt') or '').strip()}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_fallback_answer(
    query: str,
    evidence_items: list[dict[str, Any]],
    persona: Persona = DEFAULT_PERSONA,
) -> str:
    """Build a readable summary without exposing raw retrieval payloads."""
    profile = PERSONA_PROFILES[persona]
    if not evidence_items:
        if persona == "student":
            return "根据当前知识库，未找到足够相关且可靠的学习资料来回答这个问题。"
        if persona == "teacher":
            return "根据当前知识库，未找到足够相关且可靠的教学证据来回答这个问题。"
        return "根据当前知识库，未找到足够相关且可靠的资料来回答这个问题。"

    course_name = next(
        (
            str(item.get("course_name") or "").strip()
            for item in evidence_items
            if str(item.get("course_name") or "").strip()
        ),
        "",
    )
    if course_name and re.search(r"学什么|主要学习|主要讲|讲什么|课程内容", query):
        opening = f"根据课程大纲，{course_name}主要涉及以下内容："
    elif course_name:
        opening = f"根据现有课程资料，{course_name}的相关信息如下："
    else:
        opening = "根据现有资料，可确认以下信息："

    def item_priority(item: dict[str, Any]) -> tuple[int, str]:
        section = str(item.get("section") or "")
        section_type = str(item.get("section_type") or "")
        source_type = str(item.get("source_type") or "")
        preferred_sections = profile["preferred_sections"]
        if section_type in preferred_sections:
            return preferred_sections.index(section_type), section
        if "课程目标" in section:
            return 0, section
        if re.search(r"第一章|第1章|^1[ .、]", section):
            return 1, section
        if source_type == "training_plan":
            return 2, section
        if "教学进度表" in section:
            return 3, section
        return 4, section

    ordered_items = sorted(evidence_items, key=item_priority)
    bullets: list[str] = []
    selected_items: list[dict[str, Any]] = []
    seen_sections: set[str] = set()
    for item in ordered_items:
        excerpt = str(item.get("excerpt") or "").replace("\\n", "\n").strip()
        section = str(item.get("section") or "课程资料").strip()
        if section in seen_sections:
            continue
        passages = [
            passage.strip()
            for passage in re.split(r"(?<=[。！？；])|\n+", excerpt)
            if passage.strip() and passage.strip() != section
        ]
        if not passages:
            continue
        if re.search(r"学什么|主要学习|主要讲|讲什么|课程内容", query):
            summary = max(
                enumerate(passages),
                key=lambda value: (
                    3 * ("掌握" in value[1])
                    + 2 * ("学习" in value[1] or "包括" in value[1])
                    + ("介绍" in value[1] or "内容" in value[1]),
                    -value[0],
                ),
            )[1]
        else:
            summary = passages[0]
        if len(summary) > 140:
            summary = summary[:137].rstrip("，,；;：: ") + "……"
        bullet = f"- {section}：{summary}"
        if bullet not in bullets:
            bullets.append(bullet)
            selected_items.append(item)
            seen_sections.add(section)
        if len(bullets) == profile["fallback_limit"]:
            break

    sources: list[str] = []
    for item in selected_items:
        label = " · ".join(
            part
            for part in (
                str(item.get("course_name") or "").strip(),
                str(item.get("section") or "").strip(),
                _source_name(item.get("source_file")),
            )
            if part
        )
        if label and label not in sources:
            sources.append(label)
        if len(sources) == 4:
            break

    lines = [
        opening,
        "",
        profile["fallback_heading"],
        *(bullets or ["- 暂无可提炼的具体内容。"]),
    ]
    if sources:
        lines.extend(["", "资料来源：", *[f"- {source}" for source in sources]])
    return "\n".join(lines)


def generate_answer_once(
    query: str,
    evidence_items: list[dict[str, Any]],
    llm_client: Any = None,
    persona: Persona = DEFAULT_PERSONA,
) -> str:
    """Generate a grounded answer from structured evidence items."""
    if llm_client is None:
        return build_fallback_answer(query, evidence_items, persona)
    evidence = _format_evidence_for_prompt(evidence_items)
    profile = PERSONA_PROFILES[persona]
    prompt = (
        "你是课程知识库问答助手。仅根据下面的证据回答，不得补充证据之外的事实。\n"
        f"{profile['prompt']}\n"
        "回答格式必须是：\n"
        "1. 第一段直接回答问题；\n"
        "2. 使用“核心内容：”列出简洁要点；\n"
        "3. 使用“资料来源：”列出课程、章节和文件名。\n"
        "不要输出 JSON、查询 ID、分数、chunk_id、内部字段或代码块。"
        "证据不足时明确说明无法确认。\n"
        f"问题：{query}\n\n"
        f"{evidence}"
    )
    answer = generate_text(
        llm_client,
        config.text_model,
        prompt,
        temperature=0.1,
        max_output_tokens=512,
    )
    return answer.strip() if answer and answer.strip() else build_fallback_answer(
        query,
        evidence_items,
        persona,
    )


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
