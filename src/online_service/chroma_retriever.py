"""ChromaDB retrieval for the online service."""

from __future__ import annotations

import json
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

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.collection is None or self.encoder is None or self.count == 0:
            return []
        result = self.collection.query(
            query_embeddings=self.encoder.encode([query]),
            n_results=min(max(1, top_k), self.count),
            include=["documents", "metadatas", "distances"],
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
            hits.append(
                {
                    **stored,
                    "chunk_id": chunk_id,
                    "text": document or "",
                    "metadata": domain_metadata,
                    "score": 1.0 - float(distance),
                }
            )
        return hits

    retrieve = search
