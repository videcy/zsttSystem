"""Abstract base class for vector-store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Protocol that any vector-store backend must implement."""

    @abstractmethod
    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert records with their embeddings and metadata."""

    @abstractmethod
    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        include: list[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        """Return the top-K records by cosine similarity.

        Returns a dict with keys: ids, documents, metadatas, distances.
        Each value is a list-of-lists where the outer list corresponds to
        query embeddings and the inner list to ranked results.
        """

    @abstractmethod
    def get(
        self,
        *,
        ids: list[str],
        include: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        """Retrieve records by their ids."""

    @abstractmethod
    def update(
        self,
        *,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update metadata for existing records."""

    @abstractmethod
    def reset(self) -> None:
        """Delete all records in the store."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored records."""
