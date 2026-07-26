"""ChromaDB retrieval for the online service."""

from __future__ import annotations

import json
import re
from typing import Any

from chromadb.errors import ChromaError, NotFoundError

from src.config import config
from src.data_processing.chroma_index import TextEncoder, create_chroma_client


class ChromaRetriever:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        client=None,
        collection_name: str | None = None,
    ):
        self.model_name = model_name or config.local_embedding_model
        self.collection_name = collection_name or config.chroma_collection
        self.client = None
        self.collection = None
        self.encoder = None
        self.connected = False
        try:
            self.client = client or create_chroma_client()
            self.collection = self.client.get_collection(
                self.collection_name,
                embedding_function=None,
            )
            indexed_model = str(
                (self.collection.metadata or {}).get("embedding_model", "")
            )
            if indexed_model and indexed_model != self.model_name:
                raise ValueError(
                    "embedding model mismatch: "
                    f"index={indexed_model}, query={self.model_name}"
                )
            dimensions = int(
                (self.collection.metadata or {}).get("dimensions", 384)
            )
            self.encoder = TextEncoder(self.model_name, dimensions)
            self.connected = True
        except NotFoundError:
            self.connected = True
        except (ChromaError, OSError, ValueError):
            self.connected = False

    @property
    def count(self) -> int:
        if self.collection is None:
            return 0
        return self.collection.count()

    @staticmethod
    def _where_filter(
        course_code: str | None,
        source_types: list[str] | tuple[str, ...] | None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if course_code:
            clauses.append({"course_code": course_code})
        if source_types:
            clauses.append({"source_type": {"$in": list(source_types)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        lowered = text.lower()
        words = set(re.findall(r"[a-z][a-z0-9]+", lowered))
        chinese = re.sub(r"[^\u4e00-\u9fff]", "", lowered)
        words.update(chinese[index : index + 2] for index in range(len(chinese) - 1))
        return words

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        course_code: str | None = None,
        source_types: list[str] | tuple[str, ...] | None = None,
        min_score: float | None = None,
        preferred_section_types: list[str] | tuple[str, ...] | None = None,
        source_boosts: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        if self.collection is None or self.encoder is None or self.count == 0:
            return []
        candidate_count = min(max(1, top_k * 4), self.count)
        where = self._where_filter(course_code, source_types)
        query_args: dict[str, Any] = {
            "query_embeddings": self.encoder.encode([query]),
            "n_results": candidate_count,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where
        result = self.collection.query(
            **query_args,
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            stored = dict(metadata or {})
            try:
                domain_metadata = json.loads(stored.pop("metadata_json", "{}"))
            except json.JSONDecodeError:
                domain_metadata = {}
            vector_score = 1.0 - float(distance)
            if min_score is not None and vector_score < min_score:
                continue
            query_terms = self._query_terms(query)
            document_terms = self._query_terms(document or "")
            lexical_score = (
                len(query_terms & document_terms) / len(query_terms)
                if query_terms
                else 0.0
            )
            section_type = (
                stored.get("section_type")
                or domain_metadata.get("section_type")
                or ""
            )
            section_boost = (
                0.08
                if preferred_section_types
                and section_type in preferred_section_types
                else 0.0
            )
            source_type = (
                stored.get("source_type")
                or domain_metadata.get("source_type")
                or ""
            )
            source_boost = (source_boosts or {}).get(str(source_type), 0.0)
            rerank_score = (
                vector_score * 0.75
                + lexical_score * 0.15
                + section_boost
                + source_boost
            )
            hits.append(
                {
                    **stored,
                    "chunk_id": chunk_id,
                    "text": document or "",
                    "metadata": domain_metadata,
                    "score": vector_score,
                    "vector_score": vector_score,
                    "rerank_score": rerank_score,
                }
            )
        hits.sort(key=lambda hit: hit["rerank_score"], reverse=True)
        return hits[:top_k]

    retrieve = search
