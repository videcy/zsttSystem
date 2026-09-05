"""ChromaDB-backed vector index for syllabus chunks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import chromadb
import numpy as np
from chromadb.config import Settings

from src.config import config


def _bypass_proxy_for_loopback(host: str) -> None:
    """Keep local Chroma traffic out of system HTTP proxies."""
    normalized_host = host.strip().strip("[]").lower()
    if normalized_host not in {"127.0.0.1", "localhost", "::1"}:
        return
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy", "")
    entries = [entry.strip() for entry in existing.split(",") if entry.strip()]
    if normalized_host not in {entry.lower() for entry in entries}:
        entries.append(normalized_host)
    os.environ["NO_PROXY"] = ",".join(entries)


def _hash_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    """Deterministic embedding used by tests and explicit hash mode."""
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class TextEncoder:
    """Encode text with SentenceTransformer or deterministic hash mode."""

    def __init__(self, model_name: str, dimensions: int = 384):
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = None
        if model_name != "hash":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                if hasattr(self._model, "get_embedding_dimension"):
                    self.dimensions = self._model.get_embedding_dimension()
                else:
                    self.dimensions = (
                        self._model.get_sentence_embedding_dimension()
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"embedding model unavailable: {model_name}"
                ) from exc

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if self._model is not None:
            return np.asarray(
                self._model.encode(values, normalize_embeddings=True),
                dtype=np.float32,
            )
        if not values:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.vstack(
            [_hash_embedding(text, self.dimensions) for text in values]
        )


def create_chroma_client():
    """Create the configured local persistent or HTTP Chroma client."""
    settings = Settings(anonymized_telemetry=False)
    if config.chroma_mode == "http":
        _bypass_proxy_for_loopback(config.chroma_host)
        return chromadb.HttpClient(
            host=config.chroma_host,
            port=config.chroma_port,
            ssl=config.chroma_ssl,
            settings=settings,
        )
    if config.chroma_mode != "local":
        raise ValueError("CHROMA_MODE must be 'local' or 'http'")
    config.vector_db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=config.vector_db_path,
        settings=settings,
    )


def _chunk_id(chunk: dict[str, Any]) -> str:
    value = str(chunk.get("chunk_id", "")).strip()
    if value:
        return value
    payload = "|".join(
        [
            str(chunk.get("source_file", "")),
            str(chunk.get("section", "")),
            str(chunk.get("text", "")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    domain_metadata = chunk.get("metadata") or {}
    metadata: dict[str, Any] = {
        "chunk_id": _chunk_id(chunk),
        "course_code": domain_metadata.get("course_code")
        or chunk.get("course_code"),
        "course_name": domain_metadata.get("course_name"),
        "source_type": domain_metadata.get("source_type"),
        "program_name": domain_metadata.get("program_name"),
        "program_type": domain_metadata.get("program_type"),
        "course_category": domain_metadata.get("course_category"),
        "course_subcategory": domain_metadata.get("course_subcategory"),
        "credits": domain_metadata.get("credits"),
        "hours": domain_metadata.get("hours"),
        "semester": domain_metadata.get("semester"),
        "instructor": domain_metadata.get("instructor"),
        "section_type": domain_metadata.get("section_type"),
        "source_year": domain_metadata.get("source_year"),
        "parent_document": domain_metadata.get("parent_document"),
        "parent_section": domain_metadata.get("parent_section"),
        "chunk_part": domain_metadata.get("chunk_part"),
        "section": chunk.get("section")
        or domain_metadata.get("syllabus_section"),
        "source_file": chunk.get("source_file"),
        "metadata_json": json.dumps(
            domain_metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) and value != ""
    }


class ChromaVectorIndex:
    """Build a fresh Chroma collection from parsed chunks."""

    def __init__(
        self,
        model_name: str,
        dimensions: int = 384,
        *,
        client=None,
        collection_name: str | None = None,
    ):
        self.client = client or create_chroma_client()
        self.collection_name = collection_name or config.chroma_collection
        self.encoder = TextEncoder(model_name, dimensions)

    def build(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        existing = {collection.name for collection in self.client.list_collections()}
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
        collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.encoder.model_name,
                "dimensions": self.encoder.dimensions,
            },
        )

        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            documents = [str(chunk.get("text", "")) for chunk in batch]
            embedding_inputs = [
                str(chunk.get("embedding_text") or document)
                for chunk, document in zip(batch, documents)
            ]
            embeddings = self.encoder.encode(embedding_inputs)
            collection.upsert(
                ids=[_chunk_id(chunk) for chunk in batch],
                embeddings=embeddings,
                documents=documents,
                metadatas=[_chroma_metadata(chunk) for chunk in batch],
            )
        return {
            "collection": self.collection_name,
            "count": collection.count(),
            "model": self.encoder.model_name,
            "dimensions": self.encoder.dimensions,
        }


def build_index(
    chunks: list[dict[str, Any]],
    _output_dir: str | Path | None = None,
    *,
    model_name: str,
    dimensions: int = 384,
    client=None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Compatibility entry point used by the pipeline and maintenance task."""
    return ChromaVectorIndex(
        model_name,
        dimensions,
        client=client,
        collection_name=collection_name,
    ).build(chunks)
