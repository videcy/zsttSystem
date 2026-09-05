"""ChromaDB retrieval for the online service."""

from __future__ import annotations

import json
from typing import Any

from chromadb.errors import ChromaError, NotFoundError

from src.config import config
from src.data_processing.chroma_index import TextEncoder, create_chroma_client
from src.data_processing.collection_registry import resolve_active_collection
from src.data_processing.lexical_stats import load_lexical_stats
from src.utils.lexical import BM25Scorer, overlap_score, term_set, tokenize


class RerankWeights:
    """Linear rerank coefficients, resolved from config unless overridden.

    Held in one object so that ablation runs (``eval/ablation.py``) and the
    weight sweep (``eval/tune_rerank.py``) can vary them without touching the
    environment of a running service.
    """

    __slots__ = ("vector", "lexical", "section_boost", "lexical_scheme", "k1", "b")

    def __init__(
        self,
        *,
        vector: float | None = None,
        lexical: float | None = None,
        section_boost: float | None = None,
        lexical_scheme: str | None = None,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        self.vector = config.rerank_weight_vector if vector is None else float(vector)
        self.lexical = (
            config.rerank_weight_lexical if lexical is None else float(lexical)
        )
        self.section_boost = (
            config.rerank_section_boost
            if section_boost is None
            else float(section_boost)
        )
        self.lexical_scheme = (
            config.rerank_lexical_scheme
            if lexical_scheme is None
            else str(lexical_scheme).strip().lower()
        )
        self.k1 = config.rerank_bm25_k1 if k1 is None else float(k1)
        self.b = config.rerank_bm25_b if b is None else float(b)

    def as_dict(self) -> dict[str, Any]:
        return {
            "vector": self.vector,
            "lexical": self.lexical,
            "section_boost": self.section_boost,
            "lexical_scheme": self.lexical_scheme,
            "bm25_k1": self.k1,
            "bm25_b": self.b,
        }


class ChromaRetriever:
    # Class-level defaults so that a bare ``__new__`` instance -- the shape
    # used by the retrieval tests -- can still score without a full __init__.
    _weights: RerankWeights | None = None
    _bm25: BM25Scorer | None = None
    _bm25_loaded: bool = False
    _lexical_stats_path: Any = None

    def __init__(
        self,
        model_name: str | None = None,
        *,
        client=None,
        collection_name: str | None = None,
        weights: RerankWeights | None = None,
        lexical_stats_path: str | None = None,
    ):
        self.model_name = model_name or config.local_embedding_model
        self._weights = weights
        self._lexical_stats_path = lexical_stats_path
        self._bm25 = None
        self._bm25_loaded = False
        self.collection_name = collection_name or config.chroma_collection
        self.client = None
        self.collection = None
        self.encoder = None
        self.connected = False
        try:
            self.client = client or create_chroma_client()
            self.collection_name = collection_name or self._resolve_collection_name()
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

    def _resolve_collection_name(self) -> str:
        """Follow the alias pointer written by the retraining updater."""
        try:
            available = [
                collection.name for collection in self.client.list_collections()
            ]
        except (ChromaError, OSError, AttributeError, TypeError):
            available = None
        return resolve_active_collection(available=available)

    @property
    def count(self) -> int:
        if self.collection is None:
            return 0
        return self.collection.count()

    @property
    def weights(self) -> RerankWeights:
        """Rerank coefficients, resolved from config on first use."""
        if self._weights is None:
            self._weights = RerankWeights()
        return self._weights

    @property
    def bm25(self) -> BM25Scorer | None:
        """Lazily loaded corpus statistics; ``None`` when unavailable."""
        if not self._bm25_loaded:
            self._bm25_loaded = True
            stats = load_lexical_stats(
                self._lexical_stats_path or config.lexical_stats_path
            )
            if stats:
                scorer = BM25Scorer(
                    stats,
                    k1=self.weights.k1,
                    b=self.weights.b,
                )
                self._bm25 = scorer if scorer.usable else None
        return self._bm25

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
        return term_set(text)

    def _lexical_scorer(self, query: str):
        """Return a ``document -> lexical score`` callable for this query.

        BM25 needs corpus-wide document frequencies; without the statistics
        file the retriever falls back to the historical term-overlap ratio so
        that a fresh checkout still ranks sensibly.
        """
        scheme = self.weights.lexical_scheme
        if scheme == "none":
            return lambda _document: 0.0
        scorer = self.bm25 if scheme == "bm25" else None
        if scorer is not None:
            query_tokens = tokenize(query)
            return lambda document: scorer.normalized_score(
                query_tokens,
                tokenize(document),
            )
        query_terms = term_set(query)
        return lambda document: overlap_score(query_terms, term_set(document))

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
        candidate_count = min(
            max(1, top_k * config.rerank_candidate_multiplier),
            self.count,
        )
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
        lexical_score_of = self._lexical_scorer(query)
        weights = self.weights
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
            lexical_score = lexical_score_of(document or "")
            section_type = (
                stored.get("section_type")
                or domain_metadata.get("section_type")
                or ""
            )
            section_boost = (
                weights.section_boost
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
                vector_score * weights.vector
                + lexical_score * weights.lexical
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
                    "lexical_score": lexical_score,
                    "rerank_score": rerank_score,
                }
            )
        hits.sort(key=lambda hit: hit["rerank_score"], reverse=True)
        return hits[:top_k]

    retrieve = search
