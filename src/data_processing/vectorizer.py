"""Vectorize chunked text data and persist embeddings into ChromaDB."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.data_processing.concept_normalizer import ConceptNormalizer
from src.online_service.local_vector_store import LocalVectorCollection
from src.utils.deepseek_client import create_deepseek_client, embed_texts


class VectorIndexer:
    """Load processed chunks, embed them in batches, and store them in ChromaDB."""

    def __init__(
        self,
        model_name: str | None = None,
        db_path: str = "vector_store/",
        embedding_batch_size: int = 64,
        upsert_batch_size: int = 64,
        preprocessed_chunks_path: str = "outputs/chunked_data_with_concepts.json",
        concept_registry_path: str = "outputs/concept_registry.json",
        concept_alias_map_path: str = "outputs/concept_alias_map.json",
        concept_candidate_path: str = "outputs/concept_candidates.json",
        verified_edge_path: str = "outputs/concept_verified_edges.json",
    ) -> None:
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.embedding_batch_size = embedding_batch_size
        self.upsert_batch_size = upsert_batch_size
        self.db_path = Path(db_path)
        self.preprocessed_chunks_path = Path(preprocessed_chunks_path)
        self.concept_registry_path = Path(concept_registry_path)
        self.concept_alias_map_path = Path(concept_alias_map_path)
        self.concept_candidate_path = Path(concept_candidate_path)
        self.verified_edge_path = Path(verified_edge_path)
        self.api_client = create_deepseek_client()
        self.concept_normalizer = ConceptNormalizer()
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

    def preprocess_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
        """Extract and canonicalize concepts before embedding."""
        return self.concept_normalizer.preprocess_chunks(
            chunks,
            registry_output_path=self.concept_registry_path,
            alias_output_path=self.concept_alias_map_path,
            enriched_chunks_output_path=self.preprocessed_chunks_path,
            candidate_output_path=self.concept_candidate_path,
            verified_output_path=self.verified_edge_path,
        )

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

        enriched_chunks, concept_registry, _ = self.preprocess_chunks(chunks)
        self.collection.reset()

        for start in range(0, len(enriched_chunks), self.upsert_batch_size):
            batch_chunks = enriched_chunks[start:start + self.upsert_batch_size]
            embedding_texts = [
                str(chunk.get("embedding_text") or chunk.get("text", "")).strip()
                for chunk in batch_chunks
            ]
            documents = [str(chunk.get("text", "")).strip() for chunk in batch_chunks]
            ids = [str(chunk["chunk_id"]) for chunk in batch_chunks]
            metadatas = [
                self._normalize_metadata(chunk.get("metadata", {}))
                for chunk in batch_chunks
            ]

            embeddings = embed_texts(
                self.api_client,
                self.model_name,
                embedding_texts,
                batch_size=self.embedding_batch_size,
            )

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        print(
            "[concept] Extracted and canonicalized "
            f"{len(concept_registry)} concepts into {self.concept_registry_path}."
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
