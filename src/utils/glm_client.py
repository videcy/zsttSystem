"""Helpers for GLM-compatible API calls used across the project."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


def create_glm_client() -> OpenAI:
    """Create a GLM client via the OpenAI-compatible interface."""
    api_key = os.getenv("ZAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ZAI_API_KEY is required for GLM API inference.")

    base_url = os.getenv("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/").strip()
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(1))

    raise ValueError("Model output does not contain a JSON object.")


def generate_text(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.1,
    max_output_tokens: int = 512,
) -> str:
    """Generate plain text with GLM chat completions."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        ).strip()
    return str(content or "").strip()


def generate_json(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.1,
    max_output_tokens: int = 800,
) -> dict[str, Any]:
    """Generate JSON by prompting the model and extracting the object."""
    text = generate_text(
        client,
        model,
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return extract_json_object(text)


def embed_texts(
    client: OpenAI,
    model: str,
    texts: list[str],
    *,
    batch_size: int = 64,
    max_chars: int | None = None,
) -> list[list[float]]:
    """Create embeddings for one or more texts."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if max_chars is None:
        configured = os.getenv("GLM_EMBEDDING_MAX_CHARS", "").strip()
        max_chars = int(configured) if configured else 3000

    sanitized = []
    for text in texts:
        value = text if text.strip() else " "
        if max_chars is not None and max_chars > 0:
            value = value[:max_chars]
        sanitized.append(value)

    embeddings: list[list[float]] = []

    for start in range(0, len(sanitized), batch_size):
        batch = sanitized[start:start + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
        )
        embeddings.extend(item.embedding for item in response.data)

    return embeddings
