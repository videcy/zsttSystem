from __future__ import annotations

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
