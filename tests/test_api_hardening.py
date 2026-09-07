from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src import main as main_module


class _StubRouter:
    async def route(self, query: str, _query_id: str, **_kwargs: Any):
        from src.online_service.query_router import RouteResult

        return RouteResult(
            answer=f"回答：{query}",
            citations=[],
            query_type="fact",
            metadata={"status": "ok"},
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    """App instance with the backends stubbed and logging redirected."""
    monkeypatch.setattr(main_module, "QUERY_LOG_PATH", tmp_path / "query.jsonl")
    monkeypatch.setattr(main_module, "FEEDBACK_LOG_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(main_module, "create_deepseek_client", lambda: None)
    monkeypatch.setattr(main_module, "ChromaRetriever", lambda *_a, **_k: None)
    monkeypatch.setattr(main_module, "QueryRouter", lambda *_a, **_k: _StubRouter())
    monkeypatch.setattr(
        main_module.GraphDatabase,
        "driver",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("no neo4j")),
    )
    main_module._REQUEST_TIMES.clear()
    with TestClient(main_module.app) as test_client:
        yield test_client
    main_module._REQUEST_TIMES.clear()


def test_query_accepts_a_normal_question(client: TestClient) -> None:
    response = client.post("/query", json={"query": "管理运筹学多少学分？"})

    assert response.status_code == 200
    assert response.json()["answer"].startswith("回答：")


def test_query_rejects_an_oversized_prompt(client: TestClient) -> None:
    from src.config import config

    response = client.post("/query", json={"query": "长" * (config.api_max_query_chars + 1)})

    assert response.status_code == 422


def test_query_rejects_an_empty_prompt(client: TestClient) -> None:
    assert client.post("/query", json={"query": ""}).status_code == 422


def test_rate_limiter_returns_429_with_retry_after(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        type(main_module.config),
        "api_rate_limit_per_minute",
        property(lambda _self: 3),
    )

    statuses = [
        client.post("/query", json={"query": "管理运筹学多少学分？"}).status_code
        for _ in range(4)
    ]

    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429


def test_unlimited_paths_are_not_throttled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        type(main_module.config),
        "api_rate_limit_per_minute",
        property(lambda _self: 1),
    )

    assert all(client.get("/health").status_code == 200 for _ in range(5))


def test_course_graph_ignores_the_legacy_concepts_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A stale outputs/concepts.json must not be served as a knowledge graph."""
    from src.config import config

    courses = tmp_path / "courses.json"
    courses.write_text(
        '[{"course_code": "IM121", "course_name": "信息管理学基础"}]',
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]", encoding="utf-8")
    # The legacy projection sits next to the (absent) registry and holds the
    # rule extractor's noise.
    legacy = tmp_path / "concepts.json"
    legacy.write_text(
        '[{"id": "c1", "name": "中山大学", "course_code": "IM121"}]',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        type(config), "courses_output_path", property(lambda _s: courses)
    )
    monkeypatch.setattr(
        type(config), "chunks_output_path", property(lambda _s: chunks)
    )
    monkeypatch.setattr(
        type(config),
        "concept_registry_path",
        property(lambda _s: tmp_path / "concept_registry.json"),
    )

    graph = main_module._build_course_graph("IM121")

    assert [node["label"] for node in graph["nodes"]] == ["Course"]
    assert graph["summary"]["concepts_available"] is False


def test_feedback_endpoint_records_the_verdict(
    client: TestClient,
    tmp_path,
) -> None:
    """The demo page's 有用/没用 buttons land in the review log."""
    response = client.post(
        "/feedback",
        json={"query_id": "query-1", "is_helpful": False},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "logged", "query_id": "query-1"}
    record = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8"))
    assert record["query_id"] == "query-1"
    assert record["is_helpful"] is False


def test_feedback_rejects_an_empty_query_id(client: TestClient) -> None:
    response = client.post("/feedback", json={"query_id": "  ", "is_helpful": True})

    assert response.status_code == 400
