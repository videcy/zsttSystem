from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_processing.collection_registry import (
    next_version_name,
    parse_version,
    read_alias_record,
    resolve_active_collection,
    write_alias_record,
)
from src.data_processing.lexical_stats import build_lexical_stats, load_lexical_stats
from src.online_service.chroma_retriever import ChromaRetriever, RerankWeights
from src.utils.lexical import BM25Scorer, idf, overlap_score, term_set, tokenize


def test_tokenize_emits_latin_words_and_chinese_bigrams() -> None:
    tokens = tokenize("数据库 SQL")

    assert "sql" in tokens
    assert "数据" in tokens and "据库" in tokens
    # No cross-boundary bigram: punctuation and latin text are stripped first.
    assert "库s" not in tokens


def test_idf_collapses_for_terms_in_every_document() -> None:
    universal = idf(document_frequency=100, document_count=100)
    rare = idf(document_frequency=1, document_count=100)

    assert universal < 0.01
    assert rare > 1.0
    assert idf(document_frequency=0, document_count=0) == 0.0


def test_bm25_prefers_the_rare_term_over_the_boilerplate_term() -> None:
    stats = {
        "document_count": 100,
        "average_length": 20,
        "document_frequency": {"运筹": 3, "筹学": 3, "教学": 98, "学内": 98},
    }
    scorer = BM25Scorer(stats)
    query = tokenize("运筹学")

    specific = scorer.normalized_score(query, tokenize("运筹学的教学内容"))
    boilerplate = scorer.normalized_score(query, tokenize("教学内容与教学安排"))

    assert specific > boilerplate
    assert boilerplate == 0.0


def test_build_lexical_stats_counts_documents_and_prunes_rare_terms() -> None:
    chunks = [
        {"text": "信息检索与信息组织"},
        {"text": "信息检索导论"},
        {"text": ""},
    ]

    stats = build_lexical_stats(chunks, min_document_frequency=2)

    assert stats["document_count"] == 2
    assert stats["document_frequency"]["信息"] == 2
    assert "组织" not in stats["document_frequency"]


def test_load_lexical_stats_returns_none_for_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_lexical_stats(path) is None
    assert load_lexical_stats(tmp_path / "missing.json") is None


class _Collection:
    metadata: dict[str, Any] = {}

    def __init__(self, documents: list[str], distances: list[float]) -> None:
        self.documents = documents
        self.distances = distances

    def count(self) -> int:
        return len(self.documents)

    def query(self, **_kwargs: Any) -> dict[str, list[list[Any]]]:
        return {
            "ids": [[f"chunk-{index}" for index in range(len(self.documents))]],
            "documents": [self.documents],
            "metadatas": [[{} for _ in self.documents]],
            "distances": [self.distances],
        }


class _Encoder:
    def encode(self, _texts: list[str]) -> list[list[float]]:
        return [[0.0]]


def _retriever(
    documents: list[str],
    distances: list[float],
    *,
    weights: RerankWeights,
    stats_path: str | None = None,
) -> ChromaRetriever:
    retriever = ChromaRetriever.__new__(ChromaRetriever)
    retriever.collection = _Collection(documents, distances)
    retriever.encoder = _Encoder()
    retriever._weights = weights
    retriever._lexical_stats_path = stats_path
    retriever._bm25 = None
    retriever._bm25_loaded = False
    return retriever


def test_retriever_uses_bm25_when_stats_are_available(tmp_path: Path) -> None:
    stats_path = tmp_path / "lexical_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "document_count": 50,
                "average_length": 10,
                "document_frequency": {"教学": 49, "学内": 49, "运筹": 2, "筹学": 2},
            }
        ),
        encoding="utf-8",
    )
    documents = ["教学内容教学内容", "运筹学与教学内容"]
    retriever = _retriever(
        documents,
        [0.5, 0.5],
        weights=RerankWeights(lexical_scheme="bm25", lexical=0.5),
        stats_path=str(stats_path),
    )

    hits = retriever.search("运筹学", 2)

    # Equal vector scores, so the IDF-weighted lexical term decides the order.
    assert [hit["chunk_id"] for hit in hits] == ["chunk-1", "chunk-0"]
    assert hits[0]["lexical_score"] > hits[1]["lexical_score"] == 0.0


def test_retriever_falls_back_to_overlap_without_stats(tmp_path: Path) -> None:
    retriever = _retriever(
        ["教学内容", "运筹学"],
        [0.5, 0.5],
        weights=RerankWeights(lexical_scheme="bm25", lexical=0.5),
        stats_path=str(tmp_path / "absent.json"),
    )

    hits = retriever.search("运筹学", 2)

    assert retriever.bm25 is None
    assert hits[0]["chunk_id"] == "chunk-1"


def test_lexical_scheme_none_zeroes_the_lexical_term() -> None:
    retriever = _retriever(
        ["运筹学"],
        [0.4],
        weights=RerankWeights(lexical_scheme="none", lexical=0.5),
    )

    hit = retriever.search("运筹学", 1)[0]

    assert hit["lexical_score"] == 0.0
    assert hit["rerank_score"] == hit["vector_score"] * retriever.weights.vector


def test_overlap_score_matches_the_legacy_formula() -> None:
    query = term_set("运筹学")
    assert overlap_score(query, term_set("运筹学导论")) == 1.0
    assert overlap_score(query, term_set("档案学")) == 0.0
    assert overlap_score(set(), term_set("任何文本")) == 0.0


def test_collection_alias_round_trip(tmp_path: Path) -> None:
    alias_path = tmp_path / "collection_alias.json"

    write_alias_record("zstt_chunks_v1", alias="zstt_chunks", alias_path=alias_path)
    write_alias_record("zstt_chunks_v2", alias="zstt_chunks", alias_path=alias_path)
    record = read_alias_record(alias_path)

    assert record["active"] == "zstt_chunks_v2"
    assert record["history"][-1]["previous"] == "zstt_chunks_v1"
    assert (
        resolve_active_collection(
            "zstt_chunks",
            alias_path=alias_path,
            available=["zstt_chunks_v1", "zstt_chunks_v2"],
        )
        == "zstt_chunks_v2"
    )


def test_resolve_falls_back_when_the_pointed_collection_is_gone(tmp_path: Path) -> None:
    alias_path = tmp_path / "collection_alias.json"
    write_alias_record("zstt_chunks_v9", alias="zstt_chunks", alias_path=alias_path)

    resolved = resolve_active_collection(
        "zstt_chunks",
        alias_path=alias_path,
        available=["zstt_chunks"],
    )

    assert resolved == "zstt_chunks"


def test_next_version_name_advances_past_existing_and_active(tmp_path: Path) -> None:
    alias_path = tmp_path / "collection_alias.json"
    write_alias_record("zstt_chunks_v4", alias="zstt_chunks", alias_path=alias_path)

    assert parse_version("zstt_chunks_v4") == 4
    assert parse_version("zstt_chunks") is None
    assert (
        next_version_name(
            "zstt_chunks",
            existing=["zstt_chunks", "zstt_chunks_v2"],
            alias_path=alias_path,
        )
        == "zstt_chunks_v5"
    )
