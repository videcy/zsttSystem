from __future__ import annotations

import asyncio
import json
import os
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
from src.data_processing.chroma_index import ChromaVectorIndex, create_chroma_client
from src.online_service.chroma_retriever import ChromaRetriever
from src.online_service.query_router import QueryRouter, QueryType
from src.utils.deepseek_client import embed_texts


class FakeRetriever:
    def __init__(self, hits: list[dict] | None = None) -> None:
        self.hits = hits or []
        self.connected = True
        self.calls: list[dict] = []

    def search(self, _query: str, _top_k: int = 5, **kwargs) -> list[dict]:
        self.calls.append(kwargs)
        source_types = kwargs.get("source_types")
        course_code = kwargs.get("course_code")
        hits = [
            hit
            for hit in self.hits
            if (
                not course_code
                or (hit.get("metadata") or {}).get("course_code") == course_code
                or hit.get("course_code") == course_code
            )
            and (
                not source_types
                or (hit.get("metadata") or {}).get("source_type") in source_types
                or hit.get("source_type") in source_types
            )
        ]
        return hits[:_top_k]

    @property
    def count(self) -> int:
        return len(self.hits)


def test_local_chroma_client_bypasses_system_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHROMA_MODE", "http")
    monkeypatch.setenv("CHROMA_HOST", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "example.test")

    with patch("src.data_processing.chroma_index.chromadb.HttpClient") as client:
        create_chroma_client()

    assert os.environ["NO_PROXY"] == "example.test,127.0.0.1"
    client.assert_called_once()


def test_chroma_index_round_trip() -> None:
    client = chromadb.EphemeralClient()
    chunks = [
        {
            "chunk_id": "database",
            "text": "database index transaction",
            "source_file": "database.docx",
            "metadata": {
                "source_type": "syllabus",
                "course_code": "IM001",
            },
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


def test_chroma_indexes_embedding_text_but_returns_original_text() -> None:
    client = chromadb.EphemeralClient()
    chunks = [
        {
            "chunk_id": "enriched",
            "text": "原始课程片段",
            "embedding_text": "稀有概念xyz",
        },
        {
            "chunk_id": "plain",
            "text": "稀有概念xyz",
            "embedding_text": "完全不同的检索词",
        },
    ]
    ChromaVectorIndex(
        "hash",
        dimensions=256,
        client=client,
        collection_name="embedding_text_test",
    ).build(chunks)
    retriever = ChromaRetriever(
        "hash",
        client=client,
        collection_name="embedding_text_test",
    )

    hit = retriever.search("稀有概念xyz", top_k=1)[0]

    assert hit["chunk_id"] == "enriched"
    assert hit["text"] == "原始课程片段"


def test_shared_embedding_helper_supports_explicit_hash_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("SIMPLE_EMBEDDING_DIMENSIONS", "16")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    embeddings = embed_texts(None, "unused", ["线性规划", "整数规划"])

    assert len(embeddings) == 2
    assert all(len(vector) == 16 for vector in embeddings)


def test_research_embedding_mode_does_not_silently_fall_back_to_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    with (
        patch(
            "src.utils.deepseek_client._load_local_embedding_model",
            side_effect=OSError("model unavailable"),
        ),
        pytest.raises(RuntimeError, match="local embedding model unavailable"),
    ):
        embed_texts(
            None,
            "research-model",
            ["线性规划"],
            allow_hash_fallback=False,
        )


def test_chroma_retriever_filters_by_course_and_source_type() -> None:
    client = chromadb.EphemeralClient()
    chunks = [
        {
            "chunk_id": "im399-objective",
            "text": "管理运筹学课程目标包括线性规划和单纯形法",
            "source_file": "im399.docx",
            "metadata": {
                "source_type": "syllabus",
                "course_code": "IM399",
                "section_type": "course_objectives",
            },
        },
        {
            "chunk_id": "im441-internship",
            "text": "图书馆学专业实习课程安排",
            "source_file": "im441.docx",
            "metadata": {
                "source_type": "syllabus",
                "course_code": "IM441",
            },
        },
    ]
    ChromaVectorIndex(
        "hash",
        dimensions=32,
        client=client,
        collection_name="course_filter",
    ).build(chunks)
    retriever = ChromaRetriever(
        "hash",
        client=client,
        collection_name="course_filter",
    )

    hits = retriever.search(
        "管理运筹学主要学什么",
        course_code="IM399",
        source_types=("syllabus",),
        preferred_section_types=("course_objectives",),
    )

    assert [hit["chunk_id"] for hit in hits] == ["im399-objective"]


def test_content_query_uses_course_filtered_multisource_evidence() -> None:
    retriever = FakeRetriever(
        [
            {
                "chunk_id": "objective",
                "text": "课程目标\n管理运筹学学习线性规划、整数规划和网络分析。",
                "source_file": "IM399.docx",
                "section": "课程目标",
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM399",
                    "course_name": "管理运筹学",
                    "section_type": "course_objectives",
                },
            },
            {
                "chunk_id": "schedule",
                "text": "教学进度 - 第一章 线性规划\n讲授数学模型与单纯形法。",
                "source_file": "IM399.docx",
                "section": "教学进度 - 第一章 线性规划",
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM399",
                    "course_name": "管理运筹学",
                    "section_type": "teaching_schedule",
                },
            },
            {
                "chunk_id": "plan",
                "text": "培养方案课程信息\n课程：管理运筹学（IM399）\n学分：3",
                "source_file": "training_plans/plan.xlsx",
                "section": "培养方案课程信息",
                "metadata": {
                    "source_type": "training_plan",
                    "course_code": "IM399",
                    "course_name": "管理运筹学",
                },
            },
            {
                "chunk_id": "internship",
                "text": "专业实习内容",
                "source_file": "IM441.docx",
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM441",
                },
            },
        ]
    )
    router = QueryRouter(vector_retriever=retriever)
    router.courses = [
        {"course_code": "IM399", "course_name": "管理运筹学"}
    ]

    result = asyncio.run(
        router.route("管理运筹学主要学什么", "content-query")
    )

    assert result.query_type == "content"
    assert "课程目标" in result.answer
    assert "第一章 线性规划" in result.answer
    assert "培养方案课程信息" in result.answer
    assert "专业实习" not in result.answer
    assert all(call["course_code"] == "IM399" for call in retriever.calls)


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


def test_catalog_query_uses_training_plan_memberships() -> None:
    router = QueryRouter(vector_retriever=FakeRetriever())
    router.courses = [
        {
            "course_code": "IM249",
            "course_name": "高级程序设计",
            "offerings": [
                {
                    "program_name": "信息管理学院2025级信息管理与信息系统专业培养方案",
                    "program_type": "主修专业",
                    "course_category": "专业必修课",
                    "course_subcategory": "专业核心课",
                    "semester": 3,
                    "source_file": "信息管理与信息系统.xlsx",
                }
            ],
        },
        {
            "course_code": "IM399",
            "course_name": "管理运筹学",
            "offerings": [
                {
                    "program_name": "信息管理学院2025级信息管理与信息系统专业培养方案",
                    "program_type": "主修专业",
                    "course_category": "专业必修课",
                    "course_subcategory": "专业基础课",
                    "semester": 4,
                    "source_file": "信息管理与信息系统.xlsx",
                }
            ],
        },
    ]

    result = asyncio.run(
        router.route(
            "信息管理与信息系统专业核心课程有哪些",
            "catalog-query",
        )
    )

    assert result.query_type == "catalog"
    assert "高级程序设计（IM249）" in result.answer
    assert "管理运筹学" not in result.answer
    assert result.metadata["source_type"] == "training_plan"


def test_demo_renders_answer_and_citations_without_raw_json() -> None:
    template = (
        Path(__file__).resolve().parents[1] / "src" / "templates" / "demo.html"
    ).read_text(encoding="utf-8")

    assert 'lines.join("\\\\n")' not in template
    assert "JSON.stringify(data.citations" not in template
    assert 'result.textContent = data.answer' in template
    assert 'id="citation-list"' in template
    assert 'id="debug-details"' in template
    assert 'src="/static/vendor/mermaid.min.js"' in template
    assert "cdn.jsdelivr.net" not in template
    assert 'securityLevel: "strict"' in template
    assert 'id="dependency-panel"' in template
    assert 'id="dependency-edge-list"' in template
    assert 'id="learning-plan"' in template
    assert "buildMermaidDefinition" in template
    assert "renderDependencyInfo" in template
    assert "data.nodes.length > 30" in template
    assert "关系图渲染失败，请使用下方课程边列表。" in template
    assert "[hidden]" in template

    vendor_root = Path(__file__).resolve().parents[1] / "src" / "static" / "vendor"
    assert (vendor_root / "mermaid.min.js").stat().st_size > 1_000_000
    assert (vendor_root / "MERMAID-LICENSE").exists()


def test_start_script_detects_an_existing_api_before_launching() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "start_all.bat"
    ).read_text(encoding="utf-8")

    health_check = script.index("http://127.0.0.1:8000/health")
    uvicorn_launch = script.index("start \"zsttSystem\"")
    assert health_check < uvicorn_launch
    assert "Get-NetTCPConnection -LocalPort 8000" in script
    assert "API is already running" in script
    assert "Port 8000 is already occupied" in script


def test_hybrid_query_degrades_when_llm_fails() -> None:
    hits = [
        {
            "chunk_id": "chunk-1",
            "text": "可信课程证据",
            "source_file": "course.docx",
            "metadata": {
                "source_type": "syllabus",
                "course_code": "IM001",
            },
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

    assert "核心内容：" in result.answer
    assert "可信课程证据" in result.answer
    assert "chunk_id" not in result.answer
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
    monkeypatch.setenv(
        "CONCEPT_REGISTRY_PATH",
        str(tmp_path / "missing_registry.json"),
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
    original = [
        {
            "chunk_id": "chunk-1",
            "text": "old",
            "metadata": {"course_code": "IM001"},
        }
    ]
    chunks_path.write_text(json.dumps(original), encoding="utf-8")

    build_calls: list[list[dict]] = []
    monkeypatch.setattr(
        retraining_module,
        "CHUNKS_PATHS",
        (chunks_path,),
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

    updated = json.loads(chunks_path.read_text(encoding="utf-8"))
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


def test_api_lifespan_starts_without_a_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableDriver:
        def verify_connectivity(self) -> None:
            raise ServiceUnavailable("offline")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        main_module,
        "create_deepseek_client",
        lambda: (_ for _ in ()).throw(ValueError("missing key")),
    )
    monkeypatch.setattr(
        main_module,
        "ChromaRetriever",
        lambda *_args, **_kwargs: FakeRetriever(),
    )
    monkeypatch.setattr(
        main_module.GraphDatabase,
        "driver",
        lambda *_args, **_kwargs: UnavailableDriver(),
    )

    async def run_lifespan() -> None:
        async with main_module.lifespan(main_module.app):
            assert main_module.app.state.llm_client is None

    asyncio.run(run_lifespan())


def test_query_log_marks_error_metadata_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[dict] = []
    monkeypatch.setattr(
        main_module,
        "append_jsonl_record",
        lambda _path, record: records.append(record),
    )

    asyncio.run(
        main_module._log_query(
            "query-id",
            "问题",
            "回答",
            [],
            {"error_code": "DEPENDENCY_QUERY_FAILED"},
            "dependency",
        )
    )

    assert records[0]["status"] == "fallback"
