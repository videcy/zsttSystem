"""
LightRAG HTTP API client wrapper.

Encapsulates communication with a LightRAG server instance, combining
zsttSystem's query enhancement logic (HyDE, concept normalisation) with
LightRAG's multi-mode retrieval and generation capabilities.

Reference: https://github.com/HKUDS/LightRAG
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import requests

from src.config import config


class LightRAGClient:
    """LightRAG REST API client.

    Communicates with a LightRAG server (default http://127.0.0.1:9621).
    Supports inserting documents and querying with multiple retrieval modes.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = (base_url or config.lightrag_base_url).rstrip("/")
        self.api_key = api_key or config.lightrag_api_key
        self.headers: dict[str, str] = {}
        if self.api_key:
            # LightRAG 的 --key 鉴权校验 X-API-Key 头；
            # Authorization: Bearer 走的是 OAuth2/JWT 登录令牌逻辑，会被当成无效 JWT → 401
            self.headers["X-API-Key"] = self.api_key

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if the LightRAG server is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        mode: str = "mix",
        enable_rerank: bool = True,
        only_need_context: bool = False,
    ) -> dict[str, Any]:
        """Send a query to LightRAG.

        Args:
            question: Natural-language query (may be HyDE-enhanced).
            mode: Retrieval mode – one of ``local``, ``global``, ``hybrid``,
                  ``naive``, or ``mix``.
            enable_rerank: Whether to run the built-in reranker.
            only_need_context: If True, return retrieved context without
                               generation.

        Returns:
            LightRAG JSON response with keys ``response``, ``context``, etc.
        """
        return self._post(
            "/query",
            {
                "query": question,
                "mode": mode,
                "enable_rerank": enable_rerank,
                "only_need_context": only_need_context,
            },
        )

    def query_with_hyde(
        self,
        original_question: str,
        hyde_answer: str,
        mode: str = "mix",
    ) -> dict[str, Any]:
        """Query LightRAG using a HyDE-expanded prompt.

        The hypothetical answer is prefixed as domain-hint context so that
        LightRAG's mix mode can leverage both concrete entities and global
        relationships.
        """
        enriched = (
            f"原始问题: {original_question}\n\n"
            f"以下是一份假设性参考回答，请利用其中的术语和知识点辅助检索与生成:"
            f"\n{hyde_answer}"
        )
        return self.query(enriched, mode=mode)

    def query_with_concepts(
        self,
        question: str,
        standardized_concepts: list[str],
        mode: str = "mix",
    ) -> dict[str, Any]:
        """Query LightRAG with explicit concept annotations.

        Standardised concept names help LightRAG's local mode match the
        correct entity nodes in the knowledge graph.
        """
        concept_str = "、".join(standardized_concepts) if standardized_concepts else ""
        enriched = f"问题: {question}"
        if concept_str:
            enriched += f"\n相关知识概念: {concept_str}"
        return self.query(enriched, mode=mode)

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
    def insert_text(self, text: str, description: str = "") -> dict[str, Any]:
        """Insert a plain-text document.

        LightRAG v1.5+ requires ``file_source`` for text insertions.
        The *description* is reused as the source label.
        """
        return self._post(
            "/documents/text",
            {
                "text": text,
                "description": description,
                "file_source": description or "zstt_sync",
            },
        )

    def insert_file(self, file_path: str | Path) -> dict[str, Any]:
        """Upload a file to LightRAG."""
        file_path = Path(file_path)
        with open(file_path, "rb") as fh:
            resp = requests.post(
                f"{self.base_url}/documents/upload",
                files={"file": (file_path.name, fh)},
                headers=self.headers,
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()

    def insert_batch(
        self,
        texts: list[dict[str, str]],
        *,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Insert multiple documents with per-item exponential-backoff retries.

        Each dict should contain ``"text"`` and optionally ``"description"``.
        Returns a summary dict with ``success``/``failed`` counts and any
        ``errors`` encountered.
        """
        results: dict[str, Any] = {"success": 0, "failed": 0, "errors": []}

        for i, item in enumerate(texts):
            last_exc: Exception | None = None
            delay = 1.0
            for attempt in range(max_retries + 1):
                try:
                    self.insert_text(
                        item.get("text", ""),
                        description=item.get("description", ""),
                    )
                    results["success"] += 1
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= 2

            if last_exc is not None:
                results["failed"] += 1
                results["errors"].append({"index": i, "error": str(last_exc)})

            if (i + 1) % 10 == 0:
                print(
                    f"  [LightRAG sync] {i+1}/{len(texts)}  "
                    f"success={results['success']}  failed={results['failed']}"
                )

        return results

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    def list_documents(self) -> dict[str, Any]:
        """Return the list of indexed documents."""
        resp = requests.get(
            f"{self.base_url}/documents",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        """Delete a document by its identifier."""
        resp = requests.delete(
            f"{self.base_url}/documents/{doc_id}",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _post(
        self,
        endpoint: str,
        json_data: dict[str, Any],
        timeout: int = 120,
    ) -> dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}{endpoint}",
            json=json_data,
            headers=self.headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
