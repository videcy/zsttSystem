"""Query preprocessing utilities for the online RAG-KG pipeline."""

from __future__ import annotations

import os
import re
from typing import Any

from src.utils.deepseek_client import embed_texts, generate_text


def HyDE(
    query: str,
    llm_client: Any,
    embedding_client: Any,
    *,
    text_model: str | None = None,
    embedding_model: str | None = None,
) -> list[float]:
    """Generate a hypothetical answer and embed it for retrieval."""
    prompt = f"""
你是一名课程问答助教。请针对下面的用户问题，生成一段简洁、可信、偏教材风格的假设性参考答案，
用于后续检索相关资料。不要输出解释，不要编造超出教学范围的内容。

用户问题：{query}
"""
    hypothetical_answer = generate_text(
        llm_client,
        text_model or os.getenv("TEXT_MODEL", "deepseek-v4-flash"),
        prompt,
        temperature=0.1,
        max_output_tokens=256,
    )
    embedding = embed_texts(
        embedding_client,
        embedding_model
        or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        [hypothetical_answer or query],
    )[0]
    return embedding


def _normalize_query_text(text: str) -> str:
    """Normalize query text for exact and fuzzy matching."""
    return re.sub(r"\s+", " ", text).strip()


def link_entities(query: str, neo4j_session: Any) -> list[str]:
    """Link user query text to canonical node names in the knowledge graph."""
    normalized_query = _normalize_query_text(query)
    result = neo4j_session.run(
        """
        MATCH (n)
        WHERE $query_text CONTAINS n.name OR n.name CONTAINS $query_text
        RETURN n.name AS name
        LIMIT 10
        """,
        query_text=normalized_query,
    )
    matched_names = [record["name"] for record in result if record.get("name")]
    if matched_names:
        return matched_names

    tokens = [token for token in re.split(r"[\s,，。；;、]+", normalized_query) if len(token) >= 2]
    fuzzy_matches: list[str] = []
    for token in tokens:
        token_result = neo4j_session.run(
            """
            MATCH (n)
            WHERE n.name CONTAINS $token
            RETURN n.name AS name
            LIMIT 5
            """,
            token=token,
        )
        for record in token_result:
            name = record.get("name")
            if name and name not in fuzzy_matches:
                fuzzy_matches.append(name)

    return fuzzy_matches
