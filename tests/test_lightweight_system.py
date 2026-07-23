from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest
import httpx
from neo4j.exceptions import ServiceUnavailable

import maintenance.retraining_updater as retraining_module
import src.main as main_module
from maintenance.retraining_updater import RetrainingUpdater
from run_pipeline import build_stage_map
from src.data_processing.chroma_index import ChromaVectorIndex
from src.online_service.chroma_retriever import ChromaRetriever
from src.online_service.query_router import QueryRouter, QueryType


class FakeRetriever:
    def __init__(self, hits: list[dict] | None = None) -> None:
        self.hits = hits or []
        self.connected = True

    def search(self, _query: str, _top_k: int = 5) -> list[dict]:
        return self.hits

    @property
    def count(self) -> int:
        return len(self.hits)


def test_chroma_index_round_trip() -> None:
    client = chromadb.EphemeralClient()
    chunks = [
        {
            "chunk_id": "database",
            "text": "database index transaction",
            "source_file": "database.docx",
            "metadata": {"course_code": "IM001"},
        },
        {
            "chunk_id": "archive",
            "text": "archive preservation catalog",
            "source_file": "archive.docx",
            "metadata": {"course_code": "IM002"},
        },
    ]
    summary = ChromaVectorIndex(
        "hash",
        dimensions=32,
        client=client,
        collection_name="test_chunks",
    ).build(chunks)

    retriever = ChromaRetriever(
        "hash",
        client=client,
        collection_name="test_chunks",
    )

    hits = retriever.search("database index", top_k=1)
    assert summary["count"] == 2
    assert hits[0]["chunk_id"] == "database"
    assert hits[0]["score"] > 0


def test_chroma_retriever_rejects_model_mismatch() -> None:
    client = chromadb.EphemeralClient()
    ChromaVectorIndex(
        "hash",
        dimensions=16,
        client=client,
        collection_name="model_test",
    ).build([{"chunk_id": "one", "text": "one"}])

    retriever = ChromaRetriever(
        "different-model",
        client=client,
        collection_name="model_test",
    )

    assert retriever.connected is False
    assert retriever.search("query") == []


def test_query_router_classifies_and_handles_fact_queries() -> None:
    router = QueryRouter(vector_retriever=FakeRetriever())
    router.courses = [
        {
            "course_code": "IM001",
            "course_name": "数据库原理",
            "credits": 3,
            "source_file": "database.docx",
        }
    ]

    assert router.classify("数据库原理多少学分？") is QueryType.FACT
    result = asyncio.run(router.route("数据库原理多少学分？", "query-1"))

    assert result.query_type == "fact"
    assert "3" in result.answer
    assert result.metadata["llm_used"] is False


def test_hybrid_query_degrades_when_llm_fails() -> None:
    hits = [
        {
            "chunk_id": "chunk-1",
            "text": "可信课程证据",
            "source_file": "course.docx",
            "metadata": {"course_code": "IM001"},
        }
    ]
    router = QueryRouter(vector_retriever=FakeRetriever(hits))

    with patch(
        "src.online_service.generator.generate_answer_once",
        side_effect=RuntimeError("provider unavailable"),
    ):
        result = asyncio.run(
            router.route("比较数据库与档案管理的差异", "query-2", llm_client=object())
        )

    assert result.answer.startswith("可信课程证据")
    assert result.metadata["status"] == "degraded"
    assert result.metadata["error_code"] == "LLM_UNAVAILABLE"


def test_course_graph_returns_course_specific_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    courses_path = tmp_path / "courses.json"
    chunks_path = tmp_path / "chunks.json"
    concepts_path = tmp_path / "concepts.json"
    courses_path.write_text(
        json.dumps(
            [
                {
                    "course_code": "IM001",
                    "course_name": "数据库原理",
                    "prerequisites": ["IM000"],
                },
                {"course_code": "IM999", "course_name": "其他课程"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "chunk-1",
                    "source_file": "database.docx",
                    "metadata": {"course_code": "IM001"},
                },
                {
                    "chunk_id": "chunk-999",
                    "source_file": "other.docx",
                    "metadata": {"course_code": "IM999"},
                },
            ]
        ),
        encoding="utf-8",
    )
    concepts_path.write_text(
        json.dumps(
            [
                {
                    "concept_id": "concept-1",
                    "name": "事务",
                    "course_code": "IM001",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COURSES_OUTPUT_PATH", str(courses_path))
    monkeypatch.setenv("CHUNKS_OUTPUT_PATH", str(chunks_path))
    monkeypatch.setenv(
        "CONCEPT_CACHE_PATH",
        str(tmp_path / "concept_cache.json"),
    )

    payload = main_module._build_course_graph("im001")

    assert payload["course_code"] == "IM001"
    assert {node["id"] for node in payload["nodes"]} == {
        "IM001",
        "IM000",
        "concept-1",
        "chunk-1",
    }
    assert payload["summary"] == {"node_count": 4, "edge_count": 3}


def test_retraining_updates_chunks_and_rebuilds_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks_path = tmp_path / "chunks.json"
    chunked_path = tmp_path / "chunked_data.json"
    original = [
        {
            "chunk_id": "chunk-1",
            "text": "old",
            "metadata": {"course_code": "IM001"},
        }
    ]
    for path in (chunks_path, chunked_path):
        path.write_text(json.dumps(original), encoding="utf-8")

    build_calls: list[list[dict]] = []
    monkeypatch.setattr(
        retraining_module,
        "CHUNKS_PATHS",
        (chunks_path, chunked_path),
    )
    monkeypatch.setattr(retraining_module, "VECTOR_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        retraining_module,
        "build_index",
        lambda chunks, *_args, **_kwargs: build_calls.append(chunks),
    )

    updater = RetrainingUpdater()
    updater.apply_updates(
        [
            {
                "issue_type": "chunking",
                "chunk_id": "chunk-1",
                "corrected_text": "new",
                "corrected_metadata": {"reviewed": True},
            }
        ]
    )

    for path in (chunks_path, chunked_path):
        updated = json.loads(path.read_text(encoding="utf-8"))
        assert updated[0]["text"] == "new"
        assert updated[0]["metadata"]["reviewed"] is True
    assert len(build_calls) == 1


def test_pipeline_stage_map_contains_only_supported_stages() -> None:
    stages = build_stage_map()
    assert set(stages) == {"baseline", "parse", "concept", "graph", "embed", "all"}
    assert len(stages["all"]) == 4


def test_api_lifespan_reports_degraded_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnavailableDriver:
        closed = False

        def verify_connectivity(self) -> None:
            raise ServiceUnavailable("offline")

        def close(self) -> None:
            self.closed = True

    unavailable_driver = UnavailableDriver()
    fake_retriever = FakeRetriever(
        [{"chunk_id": "chunk-1", "text": "evidence"}]
    )
    monkeypatch.setattr(main_module, "create_deepseek_client", lambda: object())
    monkeypatch.setattr(
        main_module,
        "ChromaRetriever",
        lambda *_args, **_kwargs: fake_retriever,
    )
    monkeypatch.setattr(
        main_module.GraphDatabase,
        "driver",
        lambda *_args, **_kwargs: unavailable_driver,
    )

    async def run_healthcheck() -> httpx.Response:
        async with main_module.lifespan(main_module.app):
            transport = httpx.ASGITransport(app=main_module.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                return await client.get("/health")

    response = asyncio.run(run_healthcheck())

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["chroma"] == "connected"
    assert response.json()["vector_index"] == "loaded"
    assert unavailable_driver.closed is True
