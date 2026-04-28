"""Helpers for OpenAI-compatible API calls used across the project."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


def create_openai_client() -> OpenAI:
    """Create an OpenAI client using environment configuration."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for API-based inference.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
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
    """Generate plain text with the Responses API."""
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return (response.output_text or "").strip()


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
) -> list[list[float]]:
    """Create embeddings for one or more texts."""
    sanitized = [text if text.strip() else " " for text in texts]
    response = client.embeddings.create(
        model=model,
        input=sanitized,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]
