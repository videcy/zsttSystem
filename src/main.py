"""
FastAPI entry point for the zsttSystem v2.0 online RAG-KG service.

Architecture (Plan C):
  ChromaDB vector retrieval + Neo4j dependency graph + DeepSeek generation

Endpoints:
  GET  /              Demo page
  GET  /health        Health check
  POST /query         Main Q&A endpoint (routed via QueryRouter)
  POST /feedback      Log user feedback
  GET  /dependency    Course dependency reasoning (zsttSystem-native)
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from pydantic import BaseModel

from src.config import config
from src.online_service.feedback_handler import (
    append_jsonl_record,
    build_feedback_log_record,
    build_query_log_record,
)
from src.online_service.course_dependency_service import (
    CourseDependencyNotFoundError,
    get_course_dependency_subgraph,
)
from src.online_service.query_router import QueryRouter
from src.online_service.chroma_retriever import ChromaRetriever
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
    persona: Literal["student", "teacher", "visitor"] = "student"


class CitationResponse(BaseModel):
    source_file: str | None = None
    course_code: str | None = None
    course_name: str | None = None
    section: str | None = None


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[CitationResponse]
    query_type: str
    status: str
    metadata: dict[str, Any]
    graph_paths: list[dict[str, Any]] | None = None
    dependency_info: dict[str, Any] | None = None


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

    # Query router
    vector_retriever = ChromaRetriever(config.local_embedding_model)
    router = QueryRouter(vector_retriever)

    # Neo4j driver
    neo4j_driver = None
    candidate = None
    try:
        candidate = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), connection_timeout=2, connection_acquisition_timeout=2)
        await asyncio.to_thread(candidate.verify_connectivity)
        neo4j_driver = candidate
        print("[lifespan] Neo4j connection established.")
    except (AuthError, ServiceUnavailable, Neo4jError):
        if candidate is not None:
            candidate.close()
        print("[lifespan] WARNING: Neo4j is not available – "
              "dependency queries will return fallback responses.")

    # Store on app.state
    app.state.llm_client = llm_client
    app.state.router = router
    app.state.neo4j_driver = neo4j_driver
    app.state.vector_retriever = vector_retriever

    try:
        yield
    finally:
        if neo4j_driver is not None:
            neo4j_driver.close()


app = FastAPI(lifespan=lifespan, title="zsttSystem v2.0 RAG-KG API")
app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "src" / "static"),
    name="static",
)


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
    retriever = getattr(app.state, "vector_retriever", None)
    neo4j_driver = getattr(app.state, "neo4j_driver", None)
    chroma_connected = bool(retriever and retriever.connected)
    chunk_count = retriever.count if retriever else 0
    return {
        "status": "ok" if chroma_connected and chunk_count > 0 else "degraded",
        "neo4j": "connected" if neo4j_driver else "unavailable",
        "chroma": "connected" if chroma_connected else "unavailable",
        "vector_index": "loaded" if chunk_count > 0 else "empty",
        "embedding_model": "loaded" if chunk_count > 0 else "unavailable",
        "chunk_count": chunk_count,
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    response_model_exclude_none=True,
)
async def process_query(
    request: QueryRequest, fastapi_request: Request
) -> dict[str, Any]:
    """Main Q&A endpoint.

    Routes to the appropriate backend based on query intent:
    - dependency queries → Neo4j concept graph
    - simple fact lookups → ChromaDB vector retrieval
    - complex questions  → ChromaDB + Neo4j + grounded generation
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
        persona=request.persona,
    )

    # Build response
    response: dict[str, Any] = {
        "query_id": query_id,
        "answer": result.answer,
        "citations": result.citations,
        "query_type": result.query_type,
        "metadata": result.metadata,
    }
    response["status"] = result.metadata.get("status", "ok")
    if result.dependency_info and "graph_paths" not in response:
        response["graph_paths"] = result.dependency_info.get("paths", [])
    if result.dependency_info:
        response["dependency_info"] = result.dependency_info

    # Log
    await _log_query(
        query_id, query, result.answer, result.citations,
        result.metadata, result.query_type,
    )

    return response


@app.get("/courses/{course_code}")
async def course_info(course_code: str) -> dict[str, Any]:
    return await asyncio.to_thread(_find_course, course_code)


@app.get("/courses/{course_code}/graph")
async def course_graph(course_code: str) -> dict[str, Any]:
    return await asyncio.to_thread(_build_course_graph, course_code)


@app.get("/courses/{course_code}/dependencies")
async def course_dependencies(
    course_code: str,
    fastapi_request: Request,
    depth: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(30, ge=1, le=30),
    program_name: str | None = Query(None, min_length=1, max_length=200),
) -> dict[str, Any]:
    """Return a bounded hard-prerequisite neighborhood for one course."""
    neo4j_driver = fastapi_request.app.state.neo4j_driver
    if neo4j_driver is None:
        raise HTTPException(
            status_code=503,
            detail="Neo4j is unavailable",
        )
    try:
        return await asyncio.to_thread(
            get_course_dependency_subgraph,
            neo4j_driver,
            course_code,
            depth=depth,
            max_nodes=max_nodes,
            program_name=program_name,
        )
    except CourseDependencyNotFoundError as exc:
        raise HTTPException(status_code=404, detail="course not found") from exc
    except (Neo4jError, ServiceUnavailable, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="course dependency graph is unavailable",
        ) from exc


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

def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"invalid local data file: {path.name}",
        ) from exc


def _find_course(course_code: str) -> dict[str, Any]:
    courses = _load_json(config.courses_output_path, [])
    for course in courses:
        if str(course.get("course_code", "")).casefold() == course_code.casefold():
            return course
    raise HTTPException(status_code=404, detail="course not found")


def _build_course_graph(course_code: str) -> dict[str, Any]:
    course = _find_course(course_code)
    canonical_code = str(course["course_code"])
    concepts = [
        concept
        for concept in _load_json(config.concept_cache_path.with_name("concepts.json"), [])
        if str(concept.get("course_code", "")).casefold()
        == canonical_code.casefold()
    ]
    chunks = [
        chunk
        for chunk in _load_json(config.chunks_output_path, [])
        if str(
            (chunk.get("metadata") or {}).get("course_code")
            or chunk.get("course_code", "")
        ).casefold()
        == canonical_code.casefold()
    ]

    nodes: list[dict[str, Any]] = [
        {
            "id": canonical_code,
            "label": "Course",
            **course,
        }
    ]
    edges: list[dict[str, str]] = []
    for prerequisite in course.get("prerequisites", []) or []:
        prerequisite = str(prerequisite)
        nodes.append(
            {
                "id": prerequisite,
                "label": "Course",
                "course_code": prerequisite,
            }
        )
        edges.append(
            {
                "source": prerequisite,
                "target": canonical_code,
                "type": "PREREQUISITE_OF",
            }
        )
    for concept in concepts:
        concept_id = str(concept.get("concept_id", ""))
        if not concept_id:
            continue
        nodes.append({"id": concept_id, "label": "Concept", **concept})
        edges.append(
            {
                "source": canonical_code,
                "target": concept_id,
                "type": "TEACHES",
            }
        )
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        if not chunk_id:
            continue
        nodes.append(
            {
                "id": chunk_id,
                "label": "Chunk",
                "chunk_id": chunk_id,
                "section": chunk.get("section"),
                "source_file": chunk.get("source_file"),
            }
        )
        edges.append(
            {
                "source": canonical_code,
                "target": chunk_id,
                "type": "CONTAINS",
            }
        )
    return {
        "course_code": canonical_code,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }


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
            persona=metadata.get("persona", "student"),
            persona_mode=metadata.get("persona_mode", "retrieval"),
            persona_profile_version=metadata.get("persona_profile_version", "v1"),
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
