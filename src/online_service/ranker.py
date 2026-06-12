"""Fusion, reranking, and context compression utilities."""

from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any

import tiktoken

from src.utils.deepseek_client import generate_json


def fuse_results(
    vector_results: list[dict[str, Any]],
    graph_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fuse vector and graph retrieval results with Reciprocal Rank Fusion."""
    fused_scores: dict[str, dict[str, Any]] = {}
    k = 60

    for rank, item in enumerate(vector_results, start=1):
        item_id = str(item.get("id") or f"vector::{rank}")
        fused_scores[item_id] = {
            **item,
            "rrf_score": 1.0 / (k + rank),
        }

    for rank, item in enumerate(graph_results, start=1):
        item_id = json.dumps(
            {
                "nodes": item.get("nodes", []),
                "relations": item.get("relations", []),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if item_id not in fused_scores:
            fused_scores[item_id] = {
                **item,
                "rrf_score": 0.0,
            }
        fused_scores[item_id]["rrf_score"] += 1.0 / (k + rank)

    return sorted(
        fused_scores.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def rerank_with_cross_encoder(
    query: str,
    documents: list[dict[str, Any]],
    reranker_model: Any,
) -> list[dict[str, Any]]:
    """Rerank the top fused results with an API judge."""
    top_documents = documents[:5]
    if not top_documents:
        return []

    candidates = []
    for index, document in enumerate(top_documents):
        candidate_text = document.get("text")
        if not candidate_text:
            candidate_text = " -> ".join(document.get("nodes", []))
        candidates.append(
            {
                "index": index,
                "content": str(candidate_text),
            }
        )

    prompt = f"""
你是一名检索重排序助手。请根据用户问题，为每个候选文档打一个 0 到 1 的相关性分数。

用户问题：
{query}

候选文档：
{json.dumps(candidates, ensure_ascii=False)}

请严格输出 JSON：
{{
  "scores": [
    {{"index": 0, "score": 0.95}}
  ]
}}
"""
    try:
        response = generate_json(
            reranker_model,
            os.getenv("RERANK_MODEL", os.getenv("TEXT_MODEL", "deepseek-v4-flash")),
            prompt,
            temperature=0.0,
            max_output_tokens=300,
        )
    except (ValueError, JSONDecodeError):
        return top_documents
    score_map = {
        int(item["index"]): float(item["score"])
        for item in response.get("scores", [])
        if isinstance(item, dict) and "index" in item and "score" in item
    }

    reranked = []
    for index, document in enumerate(top_documents):
        enriched = dict(document)
        enriched["rerank_score"] = float(score_map.get(index, document.get("rrf_score", 0.0)))
        reranked.append(enriched)

    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)


def compress_context(
    reranked_docs: list[dict[str, Any]],
    token_limit: int = 4096,
) -> tuple[str, list[dict[str, Any]]]:
    """Assemble a context string within a fixed token budget.

    Graph paths are prioritised before vector documents because they
    are compact and carry structural dependency information that is
    critical for reasoning.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    graph_docs = [d for d in reranked_docs if d.get("source") == "graph"]
    vector_docs = [d for d in reranked_docs if d.get("source") != "graph"]

    context_sections: list[str] = []
    selected_docs: list[dict[str, Any]] = []
    used_tokens = 0

    for document in graph_docs + vector_docs:
        if document.get("source") == "graph":
            candidate_text = (
                f"知识路径: {' -> '.join(document.get('nodes', []))} "
                f"| 关系: {', '.join(document.get('relations', []))}"
            )
        else:
            metadata = document.get("metadata", {})
            section_name = metadata.get("syllabus_section", "")
            candidate_text = (
                f"章节: {section_name}\n"
                f"内容: {document.get('text', '')}"
            ).strip()

        token_count = len(encoding.encode(candidate_text))
        if used_tokens + token_count > token_limit:
            break
        context_sections.append(candidate_text)
        selected_docs.append(document)
        used_tokens += token_count

    return "\n\n".join(context_sections), selected_docs
