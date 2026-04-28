"""Vectorize chunked text data and persist embeddings into ChromaDB."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.online_service.local_vector_store import LocalVectorCollection
from src.utils.glm_client import create_glm_client, embed_texts


class VectorIndexer:
    """Load processed chunks, embed them in batches, and store them in ChromaDB."""

    def __init__(
        self,
        model_name: str | None = None,
        db_path: str = "vector_store/",
        embedding_batch_size: int = 64,
        upsert_batch_size: int = 64,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "GLM_EMBEDDING_MODEL",
            "embedding-3",
        )
        self.embedding_batch_size = embedding_batch_size
        self.upsert_batch_size = upsert_batch_size
        self.db_path = Path(db_path)
        self.api_client = create_glm_client()
        self.collection = LocalVectorCollection(db_path=self.db_path, name="scholar_collection")

    def load_chunks(self, json_path: str = "outputs/chunked_data.json") -> list[dict[str, Any]]:
        """Load chunk records from a JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunk JSON file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        if not isinstance(chunks, list):
            raise ValueError("Chunk JSON must contain a list of chunk objects.")
        return chunks

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert metadata values into Chroma-compatible scalar types."""
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            else:
                normalized[key] = json.dumps(value, ensure_ascii=False)
        return normalized

    def create_index(self, chunks: list[dict[str, Any]]) -> None:
        """Batch-encode all chunks and add them to the Chroma collection."""
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        self.collection.reset()

        for start in range(0, len(chunks), self.upsert_batch_size):
            batch_chunks = chunks[start:start + self.upsert_batch_size]
            texts = [str(chunk.get("text", "")).strip() for chunk in batch_chunks]
            ids = [str(chunk["chunk_id"]) for chunk in batch_chunks]
            metadatas = [
                self._normalize_metadata(chunk.get("metadata", {}))
                for chunk in batch_chunks
            ]

            embeddings = embed_texts(
                self.api_client,
                self.model_name,
                texts,
                batch_size=self.embedding_batch_size,
            )

            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def run(self, json_path: str) -> None:
        """Load chunks from disk and build the vector index."""
        chunks = self.load_chunks(json_path=json_path)
        self.create_index(chunks)
        print(f"Indexed {len(chunks)} documents into scholar_collection.")


def main() -> None:
    """Provide a simple CLI for building the vector index."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vectorize chunked data and store it in ChromaDB."
    )
    parser.add_argument(
        "--json-path",
        default="outputs/chunked_data.json",
        help="Path to the chunk JSON file.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Embedding model name used for the API provider.",
    )
    parser.add_argument(
        "--db-path",
        default="vector_store/",
        help="Path to the persistent ChromaDB directory.",
    )
    args = parser.parse_args()

    indexer = VectorIndexer(model_name=args.model_name, db_path=args.db_path)
    indexer.run(json_path=args.json_path)


if __name__ == "__main__":
    main()
