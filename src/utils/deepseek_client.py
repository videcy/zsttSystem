"""Helpers for DeepSeek chat inference and configurable embeddings."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from functools import lru_cache
from hashlib import md5
from typing import Any

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.config import config


logger = logging.getLogger(__name__)


def create_deepseek_client() -> OpenAI:
    """Create a DeepSeek client via the OpenAI-compatible interface."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek API inference.")

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_json_value(text: str) -> Any:
    """Extract the first JSON array or object from model output."""
    fenced_match = re.search(r"```json\s*(.+?)\s*```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    for candidate in _iter_json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("Model output does not contain a JSON object or array.")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    value = extract_json_value(text)
    if not isinstance(value, dict):
        raise ValueError("Model output does not contain a JSON object.")
    return value


def _iter_json_candidates(text: str) -> list[str]:
    """Yield balanced JSON object/array candidates from text."""
    candidates: list[str] = []
    start = -1
    stack: list[str] = []
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if start == -1:
            if char in "{[":
                start = index
                stack = [char]
                in_string = False
                escape = False
            continue

        if escape:
            escape = False
            continue

        if char == "\\" and in_string:
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                start = -1
                continue
            opener = stack[-1]
            if (opener == "{" and char == "}") or (opener == "[" and char == "]"):
                stack.pop()
            else:
                start = -1
                stack = []
                continue
            if not stack:
                candidates.append(text[start:index + 1])
                start = -1

    return candidates


def generate_text(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.1,
    max_output_tokens: int = 512,
    json_mode: bool = False,
) -> str:
    """Generate plain text with DeepSeek chat completions."""
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if json_mode:
        request["response_format"] = {"type": "json_object"}
        # DeepSeek enables thinking by default for current chat models. JSON
        # extraction needs the final answer in ``content`` rather than an
        # optional reasoning-only response, so disable thinking explicitly.
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = client.chat.completions.create(**request)
    choice = response.choices[0]
    content = choice.message.content
    if isinstance(content, list):
        result = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        ).strip()
    else:
        result = str(content or "").strip()
    if json_mode and not result:
        logger.warning(
            "[deepseek_client] empty JSON content "
            "(finish_reason=%s, has_reasoning_content=%s)",
            getattr(choice, "finish_reason", None),
            bool(getattr(choice.message, "reasoning_content", None)),
        )
    return result


def generate_json(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.1,
    max_output_tokens: int = 800,
) -> dict[str, Any]:
    """Generate JSON by prompting the model and extracting the object."""
    value = generate_json_value(
        client,
        model,
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if not isinstance(value, dict):
        raise ValueError("Model output does not contain a JSON object.")
    return value


def generate_json_value(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.1,
    max_output_tokens: int = 800,
) -> Any:
    """Generate JSON by prompting the model and extracting the first JSON value."""
    last_error: ValueError | None = None
    for attempt in range(2):
        retry_instruction = (
            "\n上一次响应为空或不是合法 JSON。请只返回一个合法 JSON 对象。"
            if attempt
            else ""
        )
        text = generate_text(
            client,
            model,
            prompt + retry_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_mode=True,
        )
        try:
            return extract_json_value(text)
        except ValueError as exc:
            last_error = exc
            logger.warning(
                "[deepseek_client] invalid JSON response "
                "(attempt=%d, content_length=%d)",
                attempt + 1,
                len(text),
            )
    if last_error is not None:
        raise ValueError(
            "DeepSeek returned empty or invalid JSON after two JSON-mode "
            "attempts (thinking disabled)."
        ) from last_error
    raise ValueError("Model output does not contain valid JSON.")


@lru_cache(maxsize=1)
def _load_local_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and cache the local sentence-transformers model."""
    return SentenceTransformer(model_name)


def _simple_tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text into simple lexical units."""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())


def _simple_embed_texts(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    """Build deterministic local embeddings without external downloads.

    This is a lightweight hashed bag-of-tokens fallback used when the
    sentence-transformers model cannot be downloaded or loaded.
    """
    if dimensions <= 0:
        raise ValueError("dimensions must be greater than 0.")

    embeddings: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        tokens = _simple_tokenize(text)
        if not tokens:
            embeddings.append(vector)
            continue

        for token in tokens:
            digest = md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        embeddings.append(vector)

    return embeddings


def embed_texts(
    client: OpenAI | None,
    model: str,
    texts: list[str],
    *,
    batch_size: int = 64,
    max_chars: int | None = None,
    allow_hash_fallback: bool = True,
) -> list[list[float]]:
    """Create embeddings with a configurable backend.

    DeepSeek's official docs currently list chat/reasoning models, so this project
    defaults to local sentence-transformers embeddings unless EMBEDDING_PROVIDER is
    explicitly set to an OpenAI-compatible endpoint.
    """
    del client
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if max_chars is None:
        configured = os.getenv("EMBEDDING_MAX_CHARS", "").strip()
        max_chars = int(configured) if configured else 3000

    sanitized = []
    for text in texts:
        max_chars_val = max_chars if max_chars is not None else config.embedding_max_chars
        value = text if text.strip() else " "
        if max_chars_val is not None and max_chars_val > 0:
            value = value[:max_chars_val]
        sanitized.append(value)

    provider = config.embedding_provider
    if provider == "hash":
        return _simple_embed_texts(
            sanitized,
            dimensions=config.simple_embedding_dimensions,
        )
    if provider == "local":
        local_model_name = model or config.local_embedding_model
        try:
            encoder = _load_local_embedding_model(local_model_name)
            matrix = encoder.encode(
                sanitized,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [row.tolist() for row in matrix]
        except Exception as exc:
            if not allow_hash_fallback:
                raise RuntimeError(
                    f"local embedding model unavailable: {local_model_name}"
                ) from exc
            fallback_dimensions = config.simple_embedding_dimensions
            print(
                "[embedding] Failed to load local sentence-transformers model "
                f"'{local_model_name}'. Falling back to deterministic local embeddings. "
                f"Reason: {exc}"
            )
            return _simple_embed_texts(sanitized, dimensions=fallback_dimensions)

    if provider != "api":
        raise ValueError(
            "EMBEDDING_PROVIDER must be one of: local, hash, api."
        )

    api_key = config.embedding_api_key
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=api.")

    base_url = config.embedding_base_url or None
    embedding_client = OpenAI(api_key=api_key, base_url=base_url)

    embeddings: list[list[float]] = []
    for start in range(0, len(sanitized), batch_size):
        batch = sanitized[start:start + batch_size]
        response = embedding_client.embeddings.create(
            model=model,
            input=batch,
        )
        embeddings.extend(item.embedding for item in response.data)

    return embeddings
