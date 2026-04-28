"""Simple local JSON vector store used for Windows demo deployments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class LocalVectorCollection:
    """Persist embeddings and query them with cosine similarity."""

    def __init__(self, db_path: str | Path, name: str = "scholar_collection") -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.store_path = self.db_path / f"{name}.json"
        self.records: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.store_path.exists():
            return []
        with self.store_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []

    def _save(self) -> None:
        with self.store_path.open("w", encoding="utf-8") as file:
            json.dump(self.records, file, ensure_ascii=False)

    def reset(self) -> None:
        self.records = []
        self._save()

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for chunk_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            self.records.append(
                {
                    "id": chunk_id,
                    "document": document,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            )
        self._save()

    def get(
        self,
        *,
        ids: list[str],
        include: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        include = include or []
        matched = [record for record in self.records if record["id"] in ids]
        results: dict[str, list[Any]] = {"ids": [record["id"] for record in matched]}
        if "documents" in include:
            results["documents"] = [record["document"] for record in matched]
        if "metadatas" in include:
            results["metadatas"] = [record["metadata"] for record in matched]
        if "embeddings" in include:
            results["embeddings"] = [record["embedding"] for record in matched]
        return results

    def update(
        self,
        *,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        metadata_map = {}
        if metadatas is not None:
            metadata_map = {
                chunk_id: metadata
                for chunk_id, metadata in zip(ids, metadatas)
            }

        updated = False
        for record in self.records:
            chunk_id = record["id"]
            if chunk_id in metadata_map:
                record["metadata"] = metadata_map[chunk_id]
                updated = True

        if updated:
            self._save()

    def count(self) -> int:
        return len(self.records)

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        include: list[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        del include
        if not query_embeddings:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        query_embedding = query_embeddings[0]
        scored_records: list[tuple[float, dict[str, Any]]] = []
        for record in self.records:
            similarity = self._cosine_similarity(query_embedding, record["embedding"])
            scored_records.append((similarity, record))

        scored_records.sort(key=lambda item: item[0], reverse=True)
        top_records = scored_records[:n_results]

        return {
            "ids": [[item[1]["id"] for item in top_records]],
            "documents": [[item[1]["document"] for item in top_records]],
            "metadatas": [[item[1]["metadata"] for item in top_records]],
            "distances": [[1.0 - item[0] for item in top_records]],
        }

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0

        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)
