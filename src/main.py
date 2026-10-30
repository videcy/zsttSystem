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
from src.data_processing.collection_registry import (
    parse_version,
    resolve_active_collection,
)
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
from src.online_service.persona import (
    DEFAULT_PERSONA,
    PERSONA_PROFILE_VERSION,
    PERSONA_PROFILES,
)
from src.online_service.program_graph import (
    build_program_graph,
    corpus_overview,
    load_configured_concept_links,
    program_catalog,
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
    retrieval_model = config.retrieval_model_name
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
    # 文档树（P1-3）与 BM25 倒排（P2 预留）：缺失 / 未落地时退化为 None，
    # 服务照常启动，/health 与既有接口不受影响。
    doc_tree, doc_tree_info = _load_doc_tree()
    lexical_index = _build_lexical_index()
    app.state.doc_tree = doc_tree
    app.state.doc_tree_info = doc_tree_info
    app.state.lexical_index = lexical_index
    _attach_live_objects(router, doc_tree, lexical_index)

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
    doc_tree = _doc_tree_health()
    return {
        "status": "ok" if chroma_connected and chunk_count > 0 else "degraded",
        "neo4j": "connected" if neo4j_driver else "unavailable",
        "chroma": "connected" if chroma_connected else "unavailable",
        "vector_index": "loaded" if chunk_count > 0 else "empty",
        "embedding_model": "loaded" if chunk_count > 0 else "unavailable",
        "chunk_count": chunk_count,
        # doc_tree.json 不存在时为 None，但 /health 仍返回 200（§8 风险表）。
        "doc_tree_version": doc_tree["version"],
        "doc_tree": doc_tree,
        "collection": _collection_health(chunk_count),
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


def _concept_links() -> dict[str, Any] | None:
    return load_configured_concept_links()


@app.get("/programs")
async def list_programs() -> dict[str, Any]:
    """Training plans grouped by college and major, for the graph selector."""
    courses = await asyncio.to_thread(_load_json, config.courses_output_path, [])
    return program_catalog(courses)


@app.get("/programs/graph")
async def program_graph(
    program_name: str = Query(..., min_length=1, max_length=200),
    knowledge: bool = Query(True),
) -> dict[str, Any]:
    """Semester-ordered prerequisite network of one training plan.

    Built from ``courses.json`` and the verified concept artefacts, so it
    stays available when Neo4j is down.
    """

    def build() -> dict[str, Any] | None:
        courses = _load_json(config.courses_output_path, [])
        return build_program_graph(
            courses,
            program_name,
            concept_links=_concept_links() if knowledge else None,
            include_knowledge=knowledge,
        )

    graph = await asyncio.to_thread(build)
    if graph is None:
        raise HTTPException(status_code=404, detail="program not found")
    return graph


@app.get("/overview")
async def overview(fastapi_request: Request) -> dict[str, Any]:
    """Corpus scale for the landing page; every number comes from artefacts."""

    def build() -> dict[str, Any]:
        courses = _load_json(config.courses_output_path, [])
        return corpus_overview(courses, concept_links=_concept_links())

    summary = await asyncio.to_thread(build)
    retriever = getattr(fastapi_request.app.state, "vector_retriever", None)
    tree = _doc_tree_health()
    summary.update(
        chunk_count=retriever.count if retriever and retriever.connected else None,
        syllabus_count=tree.get("documents"),
        neo4j="connected" if fastapi_request.app.state.neo4j_driver else "unavailable",
        llm="configured" if fastapi_request.app.state.llm_client else "unavailable",
    )
    return summary


@app.get("/personas")
async def personas() -> dict[str, Any]:
    """The retrieval and style profile behind each persona, as configured."""
    return {
        "version": PERSONA_PROFILE_VERSION,
        "default": DEFAULT_PERSONA,
        "profiles": {
            key: {
                "syllabus_top_k": profile["syllabus_top_k"],
                "plan_top_k": profile["plan_top_k"],
                "preferred_sections": list(profile["preferred_sections"]),
                "source_boosts": dict(profile["source_boosts"]),
                "prompt": profile["prompt"],
            }
            for key, profile in PERSONA_PROFILES.items()
        },
    }


@app.get("/eval/latest")
async def latest_eval_report() -> dict[str, Any]:
    """Aggregate metrics of the newest evaluation report (no per-item rows)."""
    reports = sorted(config.eval_report_dir.glob("*.json"))
    for path in reversed(reports):
        report = await asyncio.to_thread(_safe_load_json, path)
        if isinstance(report, dict) and report.get("run_id"):
            return {
                key: value
                for key, value in report.items()
                if not key.endswith("_details")
            }
    raise HTTPException(status_code=404, detail="no evaluation report")


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

        retrieval_model = config.retrieval_model_name
        retriever = ChromaRetriever(retrieval_model)
        router: QueryRouter = app.state.router
        app.state.vector_retriever = retriever
        router.vector_retriever = retriever
        if config.courses_output_path.exists():
            router.courses = json.loads(
                config.courses_output_path.read_text(encoding="utf-8")
            )
        stages.append("reload")
        # 文档树 / 倒排必须和向量库在同一个回调里刷新，否则会长期停在
        # 「新向量库 + 旧文档树」的错位状态（§8 风险表）。
        doc_tree_info = _refresh_live_artifacts(app, router)
        stages.append("doc_tree")
        _REINDEX_STATE.update(
            status="succeeded",
            finished_at=datetime.now(timezone.utc).isoformat(),
            detail=f"索引重建完成，共 {retriever.count} 个片段",
            stages=stages,
            doc_tree_version=doc_tree_info.get("version"),
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


def _safe_load_json(path: Path) -> Any:
    """只读、容错地读取 JSON；缺失或损坏都返回 None（health 路径不能 500）。"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _iso_mtime(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _count_tree_chunks(documents: Any) -> int | None:
    """文档树里引用的 chunk id 总数；结构不符时返回 None。"""
    if not isinstance(documents, dict):
        return None
    total = 0
    for document in documents.values():
        if isinstance(document, dict) and isinstance(document.get("chunk_ids"), list):
            total += len(document["chunk_ids"])
    return total


def _load_doc_tree() -> tuple[Any | None, dict[str, Any]]:
    """构造 ``DocTree`` 查询对象并读取版本信息（M2.3）。

    查询模块或 ``doc_tree.json`` 缺失时必须退化：构造不抛异常、查询返回空，
    在线问答与 ``/health`` 都不受影响。
    """
    path = config.doc_tree_path
    raw = _safe_load_json(path)
    doc_tree = None
    try:
        from src.online_service.doc_tree import DocTree
    except ImportError as exc:  # pragma: no cover - 依赖并行交付
        print(f"[doc_tree] 查询模块不可用（已退化）：{exc}")
    else:
        try:
            doc_tree = DocTree(path)
        except Exception as exc:  # noqa: BLE001 - 加载失败只降级，不阻断启动
            print(f"[doc_tree] 加载失败（已退化）：{type(exc).__name__}: {exc}")
            doc_tree = None
    documents = raw.get("documents") if isinstance(raw, dict) else None
    version = getattr(doc_tree, "version", None)
    if version is None and isinstance(raw, dict):
        version = raw.get("version")
    return doc_tree, {
        "version": version,
        "document_count": len(documents) if isinstance(documents, dict) else None,
        "chunk_count": _count_tree_chunks(documents),
        "modified_at": _file_mtime(path),
    }


def _build_lexical_index() -> Any | None:
    """构建 P2 的 BM25 内存倒排（M5.1）。

    ``chunks.json`` 缺失或损坏时返回 ``None``：调用方（lifespan / 重建回调 /
    ``/health``）都把它当成「可能为空」的对象，检索器自身也会惰性重建一份。
    """
    try:
        from src.data_processing.lexical_stats import load_lexical_stats
        from src.utils.lexical import build_inverted_index
    except ImportError as exc:  # pragma: no cover - 依赖并行交付
        print(f"[lexical] 倒排模块不可用（已退化）：{exc}")
        return None
    try:
        raw = json.loads(config.chunks_output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[lexical] 倒排语料不可用（已退化）：{exc}")
        return None
    documents = (
        [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, list)
        else []
    )
    if not documents:
        return None
    index = build_inverted_index(
        documents,
        k1=config.rerank_bm25_k1,
        b=config.rerank_bm25_b,
        stats=load_lexical_stats(config.lexical_stats_path),
    )
    return index if getattr(index, "usable", False) else None


def _attach_live_objects(target: Any, doc_tree: Any, lexical_index: Any) -> None:
    """把文档树 / 倒排挂到路由层；路由层不允许挂载额外属性时跳过。"""
    if target is None:
        return
    # ``QueryRouter`` 通过 ``set_doc_tree()`` 注入（内部写 ``_doc_tree_override``），
    # 直接 ``setattr(target, "doc_tree", ...)`` 是读不到的；有 setter 时优先用它，
    # 否则（例如测试里的 stub router）退回属性挂载。
    setter = getattr(target, "set_doc_tree", None)
    if callable(setter):
        try:
            setter(doc_tree)
        except Exception as exc:  # noqa: BLE001 - 路由层实现可能不兼容
            print(f"[doc_tree] 路由层拒收文档树：{exc}")
    for attribute, value in (
        ("doc_tree", doc_tree),
        ("lexical_index", lexical_index),
    ):
        try:
            setattr(target, attribute, value)
        except Exception as exc:  # noqa: BLE001 - 路由层实现可能用 __slots__
            print(f"[doc_tree] 无法把 {attribute} 挂到路由层：{exc}")


def _refresh_live_artifacts(app: FastAPI, router: Any | None = None) -> dict[str, Any]:
    """在重建回调里原子替换 ``DocTree`` 与倒排，使其与向量库同版本。

    先构造好新对象再逐个写 ``app.state``：读者要么全看到旧的一份、要么全看到新
    的一份，不会读到「新向量库 + 旧文档树」的中间态。
    """
    doc_tree, info = _load_doc_tree()
    lexical_index = _build_lexical_index()
    app.state.doc_tree = doc_tree
    app.state.doc_tree_info = info
    app.state.lexical_index = lexical_index
    _attach_live_objects(
        router if router is not None else getattr(app.state, "router", None),
        doc_tree,
        lexical_index,
    )
    return info


def _doc_tree_health() -> dict[str, Any]:
    """文档树版本 + 与 ``chunks.json`` 的新鲜度比对（缺失时字段为 None）。

    热重建如果漏刷文档树，``stale`` 会是 True，对应 §8「热重建后文档树 /
    倒排未刷新，与向量库不一致」这条风险，让运维一眼看得出来。
    """
    info = getattr(app.state, "doc_tree_info", None)
    info = info if isinstance(info, dict) else {}
    modified_at = info.get("modified_at")
    chunks_mtime = _file_mtime(config.chunks_output_path)
    stale: bool | None = None
    if isinstance(modified_at, (int, float)) and chunks_mtime is not None:
        stale = modified_at < chunks_mtime
    return {
        "version": info.get("version"),
        "documents": info.get("document_count"),
        "chunk_ids": info.get("chunk_count"),
        "modified_at": _iso_mtime(modified_at),
        "stale": stale,
        "stale_reason": (
            "doc_tree.json 早于 chunks.json，可能未随上次重建刷新" if stale else None
        ),
    }


def _collection_health(chunk_count: int) -> dict[str, Any]:
    """当前服务集合的版本信息，用于和 ``doc_tree_version`` 对照。"""
    retriever = getattr(app.state, "vector_retriever", None)
    name = str(getattr(retriever, "collection_name", "") or "").strip()
    if not name:
        try:
            name = resolve_active_collection()
        except Exception:  # noqa: BLE001 - 别名记录缺失不该影响 /health
            name = ""
    return {
        "name": name or None,
        "version": parse_version(name) if name else None,
        "chunk_count": chunk_count,
    }


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
