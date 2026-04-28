"""Answer generation and hallucination interception helpers."""

from __future__ import annotations

import os
import re
from typing import Any

from src.utils.glm_client import generate_json, generate_text


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
3. 必须明确引用依据，引用格式可使用“依据：……”。
4. 如果参考资料不足以支持完整回答，必须明确回答“我不知道”或“资料不足”。
5. 不要编造教师、教材、定义、公式或推导过程。
"""


def generate_answer(prompt: str, llm_client: Any) -> str:
    """Generate a low-temperature answer with the LLM."""
    return generate_text(
        llm_client,
        os.getenv("GLM_TEXT_MODEL", "glm-5"),
        prompt,
        temperature=0.1,
        max_output_tokens=512,
    )


def _split_sentences(answer: str) -> list[str]:
    """Split answer text into sentence-like units."""
    parts = re.split(r"(?<=[。！？!?；;])\s*", answer)
    return [part.strip() for part in parts if part.strip()]


def verify_answer_with_nli(answer: str, context: str, nli_model: Any) -> tuple[bool, list[dict[str, str]]]:
    """Check whether each answer sentence is supported by context using an API judge."""
    sentences = _split_sentences(answer)
    verification_details: list[dict[str, str]] = []

    for sentence in sentences:
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
        result = generate_json(
            nli_model,
            os.getenv("GLM_JUDGE_MODEL", os.getenv("GLM_TEXT_MODEL", "glm-5")),
            prompt,
            temperature=0.0,
            max_output_tokens=120,
        )
        label = str(result.get("label", "Neutral"))
        score = float(result.get("score", 0.0))
        verification_details.append(
            {
                "sentence": sentence,
                "label": label,
                "score": f"{score:.4f}",
            }
        )
        if "entail" not in label.lower():
            return False, verification_details

    return True, verification_details


def get_fallback_response() -> dict[str, Any]:
    """Return a safe fallback answer when verification fails."""
    return {
        "answer": "根据当前检索到的资料，我暂时无法给出可靠答案。建议补充更明确的课程材料后再提问。",
        "citations": [],
        "status": "fallback",
    }
