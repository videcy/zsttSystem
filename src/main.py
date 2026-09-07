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
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from pydantic import BaseModel, Field

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
from src.online_service.data_import import ImportRejected, inventory, store_upload
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
    # /query reaches the LLM, so an unbounded field is an open cost amplifier.
    query: str = Field(min_length=1, max_length=config.api_max_query_chars)
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
    try:
        llm_client = create_deepseek_client()
    except ValueError:
        llm_client = None
        print(
            "[lifespan] WARNING: DEEPSEEK_API_KEY is not configured – "
            "generation and NLI will use deterministic fallbacks."
        )

    # Query router
    retrieval_model = (
        config.local_embedding_model
        if config.embedding_provider == "local"
        else "hash"
    )
    vector_retriever = ChromaRetriever(retrieval_model)
    router = QueryRouter(vector_retriever)

    # Neo4j driver
    neo4j_driver = None
    candidate = None
    try:
        candidate = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), connection_timeout=2, connection_acquisition_timeout=2)
        await asyncio.to_thread(candidate.verify_connectivity)
        neo4j_driver = candidate
        print("[lifespan] Neo4j connection established.")
    # OSError covers DNS and socket failures, which reach here as-is and would
    # otherwise abort startup instead of degrading.
    except (AuthError, ServiceUnavailable, Neo4jError, OSError):
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

if config.api_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

# /feedback appends a line per request and is now reachable from the demo
# page by every visitor, so it is throttled alongside the LLM endpoints.
RATE_LIMITED_PATHS = ("/query", "/dependency", "/admin", "/feedback")
_REQUEST_TIMES: dict[str, deque[float]] = defaultdict(deque)

# One reindex at a time, tracked in-process; a restart forgets it, which is
# the honest behaviour for a job whose output lives in Chroma anyway.
_REINDEX_STATE: dict[str, Any] = {"status": "idle"}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Per-client sliding window over the endpoints that reach the LLM.

    In-process and single-worker by design: it exists to stop a demo laptop
    from being drained by a loop, not to survive a distributed attack.
    """
    limit = config.api_rate_limit_per_minute
    if limit <= 0 or not request.url.path.startswith(RATE_LIMITED_PATHS):
        return await call_next(request)

    client_id = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _REQUEST_TIMES[client_id]
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= limit:
        retry_after = max(1, int(60.0 - (now - window[0])))
        return JSONResponse(
            status_code=429,
            content={"detail": "rate limit exceeded"},
            headers={"Retry-After": str(retry_after)},
        )
    window.append(now)
    return await call_next(request)


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
    response["status"] = result.metadata.get("status") or (
        "fallback"
        if result.metadata.get("error") or result.metadata.get("error_code")
        else "ok"
    )
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
    fastapi_request: Request,
    query: str = Query(
        ...,
        min_length=1,
        max_length=config.api_max_query_chars,
    ),
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
# Data import (admin)
# ---------------------------------------------------------------------------

def _require_admin(request: Request) -> None:
    """Gate the endpoints that write to disk or start a reindex.

    No token configured means the whole admin surface is off.  Defaulting to
    "open" would hand anyone who finds the host a file-upload endpoint.
    """
    token = config.admin_token
    if not token:
        raise HTTPException(
            status_code=503,
            detail="数据导入接口未启用：请先设置 ADMIN_TOKEN",
        )
    scheme, _, presented = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        presented.strip(),
        token,
    ):
        raise HTTPException(status_code=401, detail="invalid admin token")


@app.get("/admin/data")
async def list_data(fastapi_request: Request) -> dict[str, Any]:
    """Report what the pipeline would parse right now."""
    _require_admin(fastapi_request)
    return await asyncio.to_thread(
        inventory,
        syllabus_dir=config.syllabus_dir,
        training_plan_dir=config.training_plan_dir,
    )


@app.post("/admin/data/import")
async def import_data(
    fastapi_request: Request,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Store uploaded .docx syllabi and .xlsx training plans.

    Rejections are per file and reported rather than raised, so a batch with
    one bad file still imports the rest.
    """
    _require_admin(fastapi_request)
    stored: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for upload in files:
        payload = await upload.read()
        try:
            record = await asyncio.to_thread(
                store_upload,
                upload.filename or "",
                payload,
                syllabus_dir=config.syllabus_dir,
                training_plan_dir=config.training_plan_dir,
                max_bytes=config.import_max_bytes,
            )
        except ImportRejected as exc:
            rejected.append(
                {"filename": upload.filename or "", "reason": str(exc)}
            )
            continue
        stored.append(record.as_dict())

    return {
        "stored": stored,
        "rejected": rejected,
        "next": "POST /admin/data/reindex 使导入生效",
    }


@app.get("/admin/data/reindex")
async def reindex_status(fastapi_request: Request) -> dict[str, Any]:
    """Current state of the one reindex job this process tracks."""
    _require_admin(fastapi_request)
    return dict(_REINDEX_STATE)


@app.post("/admin/data/reindex", status_code=202)
async def start_reindex(
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Re-parse the data directory and rebuild the vector index.

    Runs in the background: a full rebuild takes minutes and would otherwise
    hold the request open past every sensible client timeout.
    """
    _require_admin(fastapi_request)
    if _REINDEX_STATE.get("status") == "running":
        raise HTTPException(status_code=409, detail="reindex already running")

    _REINDEX_STATE.update(
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        detail="",
        stages=[],
    )
    background_tasks.add_task(_run_reindex, fastapi_request.app)
    return dict(_REINDEX_STATE)


def _run_reindex(app: FastAPI) -> None:
    """Parse + embed, then swap the freshly built index into the live app.

    Rebuilding replaces the Chroma collection, which invalidates the handle
    the running retriever holds -- without the swap the service would answer
    from a deleted collection until someone restarted it.
    """
    # Imported here: the pipeline pulls in the parsing stack, which the API
    # does not otherwise need at startup.
    from run_pipeline import run_embed_stage, run_parse_stage

    stages: list[str] = []
    try:
        run_parse_stage(incremental=True)
        stages.append("parse")
        run_embed_stage()
        stages.append("embed")

        retrieval_model = (
            config.local_embedding_model
            if config.embedding_provider == "local"
            else "hash"
        )
        retriever = ChromaRetriever(retrieval_model)
        router: QueryRouter = app.state.router
        app.state.vector_retriever = retriever
        router.vector_retriever = retriever
        if config.courses_output_path.exists():
            router.courses = json.loads(
                config.courses_output_path.read_text(encoding="utf-8")
            )
        stages.append("reload")
        _REINDEX_STATE.update(
            status="succeeded",
            finished_at=datetime.now(timezone.utc).isoformat(),
            detail=f"索引重建完成，共 {retriever.count} 个片段",
            stages=stages,
        )
    except Exception as exc:  # noqa: BLE001 - reported through the status endpoint
        _REINDEX_STATE.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            detail=f"{type(exc).__name__}: {exc}",
            stages=stages,
        )


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
    # Only the verified registry counts as concepts.  The old fallback read
    # outputs/concepts.json, a compatibility projection that can predate the
    # current extractor -- serving its noise ("中山大学", "课程名称") as a
    # knowledge graph is worse than serving no concepts at all.
    concepts_available = config.concept_registry_path.exists()
    concepts = [
        concept
        for concept in _load_json(config.concept_registry_path, [])
        if canonical_code.casefold()
        in {
            str(code).casefold()
            for code in (
                concept.get("source_course_codes")
                or [concept.get("course_code", "")]
            )
        }
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
        concept_id = str(concept.get("id") or concept.get("concept_id", ""))
        if not concept_id:
            continue
        nodes.append(
            {
                **concept,
                "id": concept_id,
                "label": "Concept",
                "concept_id": concept_id,
                "name": concept.get("canonical_name") or concept.get("name", ""),
            }
        )
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
            # Tells a caller apart: a course with no concepts, versus a
            # concept layer that was never built on this deployment.
            "concepts_available": concepts_available,
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
    status = metadata.get("status") or (
        "fallback"
        if metadata.get("error") or metadata.get("error_code")
        else "ok"
    )
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
            status=str(status),
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
