"""
FastAPI entry point for the zsttSystem v2.0 online RAG-KG service.

Architecture (Plan C):
  zsttSystem domain layer  ──HTTP API──>  LightRAG retrieval engine
  (concept normalisation, dependency reasoning, NLI verification)

Endpoints:
  GET  /              Demo page
  GET  /health        Health check
  POST /query         Main Q&A endpoint (routed via QueryRouter)
  POST /feedback      Log user feedback
  GET  /dependency    Course dependency reasoning (zsttSystem-native)
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from pydantic import BaseModel

from src.config import config
from src.online_service.feedback_handler import (
    append_jsonl_record,
    build_feedback_log_record,
    build_query_log_record,
)
from src.online_service.lightrag_adapter import LightRAGClient
from src.online_service.query_router import QueryRouter
from src.utils.deepseek_client import create_deepseek_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

QUERY_LOG_PATH = config.query_log_path
FEEDBACK_LOG_PATH = config.feedback_log_path
NEO4J_URI = config.neo4j_uri
NEO4J_USER = config.neo4j_user
NEO4J_PASSWORD = config.neo4j_password


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


class FeedbackRequest(BaseModel):
    query_id: str
    is_helpful: bool
    comment: str | None = None


# ---------------------------------------------------------------------------
# App lifespan – initialise shared clients
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # LLM client (DeepSeek) – used for HyDE, dependency reasoning, NLI
    llm_client = create_deepseek_client()

    # LightRAG client
    lightrag = LightRAGClient()
    lightrag_available = lightrag.health_check()
    if lightrag_available:
        print("[lifespan] LightRAG server is reachable.")
    else:
        print("[lifespan] WARNING: LightRAG server is NOT reachable – "
              "retrieval-dependent queries will return fallback responses.")

    # Query router
    router = QueryRouter(lightrag)

    # Neo4j driver
    neo4j_driver = None
    try:
        candidate = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        candidate.verify_connectivity()
        neo4j_driver = candidate
        print("[lifespan] Neo4j connection established.")
    except (AuthError, ServiceUnavailable, Neo4jError):
        print("[lifespan] WARNING: Neo4j is not available – "
              "dependency queries will return fallback responses.")

    # Store on app.state
    app.state.llm_client = llm_client
    app.state.lightrag = lightrag
    app.state.lightrag_available = lightrag_available
    app.state.router = router
    app.state.neo4j_driver = neo4j_driver

    try:
        yield
    finally:
        if neo4j_driver is not None:
            neo4j_driver.close()


app = FastAPI(lifespan=lifespan, title="zsttSystem v2.0 RAG-KG API")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def demo_home() -> str:
    """Serve the demo page."""
    template_path = PROJECT_ROOT / "src" / "templates" / "demo.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "<html><body><h1>Demo page not found</h1></body></html>"


@app.get("/health")
async def healthcheck() -> dict[str, Any]:
    """Health check – reports backend availability."""
    return {
        "status": "ok",
        "lightrag": "connected" if app.state.lightrag_available else "unavailable",
        "neo4j": "connected" if app.state.neo4j_driver else "unavailable",
    }


@app.post("/query")
async def process_query(
    request: QueryRequest, fastapi_request: Request
) -> dict[str, Any]:
    """Main Q&A endpoint.

    Routes to the appropriate backend based on query intent:
    - dependency queries → Neo4j concept graph
    - simple fact lookups → LightRAG naive mode
    - complex questions  → HyDE expansion + LightRAG mix mode
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    query_id = str(uuid.uuid4())
    router: QueryRouter = fastapi_request.app.state.router
    llm_client = fastapi_request.app.state.llm_client
    neo4j_driver = fastapi_request.app.state.neo4j_driver

    # Dispatch
    result = await router.route(
        query,
        query_id,
        neo4j_driver=neo4j_driver,
        llm_client=llm_client,
    )

    # Build response
    response: dict[str, Any] = {
        "query_id": query_id,
        "answer": result.answer,
        "citations": result.citations,
        "query_type": result.query_type,
        "metadata": result.metadata,
    }
    if result.dependency_info:
        response["dependency_info"] = result.dependency_info

    # Log
    await _log_query(
        query_id, query, result.answer, result.citations,
        result.metadata, result.query_type,
    )

    return response


@app.get("/dependency")
async def dependency_query(
    query: str, fastapi_request: Request
) -> dict[str, Any]:
    """Dedicated dependency reasoning endpoint (Neo4j-native)."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    router: QueryRouter = fastapi_request.app.state.router
    llm_client = fastapi_request.app.state.llm_client
    neo4j_driver = fastapi_request.app.state.neo4j_driver

    result = await router.route(
        query, str(uuid.uuid4()),
        neo4j_driver=neo4j_driver,
        llm_client=llm_client,
    )

    return {
        "query": query,
        "answer": result.answer,
        "dependency_info": result.dependency_info,
        "metadata": result.metadata,
    }


@app.post("/feedback")
async def handle_feedback(feedback: FeedbackRequest) -> dict[str, Any]:
    """Persist user feedback."""
    query_id = feedback.query_id.strip()
    if not query_id:
        raise HTTPException(status_code=400, detail="query_id cannot be empty")

    await _log_feedback(query_id, feedback.is_helpful, feedback.comment)
    return {"status": "logged", "query_id": query_id}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _log_query(
    query_id: str,
    query: str,
    answer: str,
    citations: list[dict[str, Any]],
    metadata: dict[str, Any],
    query_type: str,
) -> None:
    import asyncio
    await asyncio.to_thread(
        append_jsonl_record,
        QUERY_LOG_PATH,
        build_query_log_record(
            query_id=query_id,
            query=query,
            context=str({k: v for k, v in metadata.items() if k != "nli_details"}),
            kg_path=query_type,
            response=answer,
            verification=metadata.get("nli_details", []),
            linked_entities=[],
            citations=citations,
            status="fallback" if "error" in metadata else "ok",
        ),
    )


async def _log_feedback(
    query_id: str, is_helpful: bool, comment: str | None
) -> None:
    import asyncio
    await asyncio.to_thread(
        append_jsonl_record,
        FEEDBACK_LOG_PATH,
        build_feedback_log_record(
            query_id=query_id,
            is_helpful=is_helpful,
            comment=comment,
        ),
    )
