"""FastAPI entry point for the zsttsystem online RAG-KG service."""

from __future__ import annotations

import asyncio
import os
import re
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
    build_citations,
    build_feedback_log_record,
    build_query_log_record,
)
from src.online_service.dependency_explainer import build_dependency_answer
from src.online_service.generator import (
    assemble_prompt,
    generate_answer,
    get_fallback_response,
    retry_generate_with_expanded_context,
    verify_answer_with_nli,
)
from src.online_service.local_vector_store import LocalVectorCollection
from src.online_service.query_processor import HyDE, link_entities
from src.online_service.ranker import compress_context, fuse_results, rerank_with_cross_encoder
from src.online_service.retriever import retrieve_graph_path, retrieve_vectors
from src.utils.deepseek_client import create_deepseek_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

VECTOR_DB_PATH = config.vector_db_path
NEO4J_URI = config.neo4j_uri
NEO4J_USER = config.neo4j_user
NEO4J_PASSWORD = config.neo4j_password
TEXT_MODEL = config.text_model
EMBEDDING_MODEL = config.embedding_model
RERANK_MODEL = config.rerank_model
JUDGE_MODEL = config.judge_model
QUERY_LOG_PATH = config.query_log_path
FEEDBACK_LOG_PATH = config.feedback_log_path


class QueryRequest(BaseModel):
    """Request schema for query processing."""

    query: str


class FeedbackRequest(BaseModel):
    """Request schema for user feedback logging."""

    query_id: str
    is_helpful: bool
    comment: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared models and database clients once for the app lifetime."""
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    llm_client = create_deepseek_client()
    chroma_collection = LocalVectorCollection(
        db_path=VECTOR_DB_PATH,
        name=COLLECTION_NAME,
    )
    neo4j_driver = None
    try:
        candidate_driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        candidate_driver.verify_connectivity()
        neo4j_driver = candidate_driver
    except (AuthError, ServiceUnavailable, Neo4jError):
        neo4j_driver = None

    app.state.llm_client = llm_client
    app.state.chroma_collection = chroma_collection
    app.state.neo4j_driver = neo4j_driver
    try:
        yield
    finally:
        if neo4j_driver is not None:
            neo4j_driver.close()


app = FastAPI(lifespan=lifespan, title="zsttsystem RAG-KG API")


@app.get("/", response_class=HTMLResponse)
async def demo_home() -> str:
    """Serve the local demo page from the template file."""
    template_path = PROJECT_ROOT / "src" / "templates" / "demo.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "<html><body><h1>Demo page not found</h1></body></html>"


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Return a simple health response for local demo checks."""
    return {"status": "ok"}


@app.post("/query")
async def process_query(request: QueryRequest, fastapi_request: Request) -> dict[str, Any]:
    """Execute the end-to-end RAG-KG query pipeline with dependency context injection."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")
    query_id = str(uuid.uuid4())
    query_intent = classify_query_intent(query)

    llm_client = fastapi_request.app.state.llm_client
    chroma_collection = fastapi_request.app.state.chroma_collection
    neo4j_driver = fastapi_request.app.state.neo4j_driver

    hyde_embedding, linked_entities, dependency_answer = await asyncio.gather(
        asyncio.to_thread(
            HyDE, query, llm_client, llm_client,
            text_model=TEXT_MODEL, embedding_model=EMBEDDING_MODEL,
        ),
        asyncio.to_thread(_link_entities, query, neo4j_driver),
        asyncio.to_thread(_build_dependency_answer, query, neo4j_driver, llm_client),
    )

    graph_tasks = [
        asyncio.to_thread(_retrieve_graph_for_entity, entity_name, neo4j_driver)
        for entity_name in linked_entities[:5]
    ]
    vector_task = asyncio.to_thread(
        retrieve_vectors, hyde_embedding, chroma_collection, 10,
    )
    graph_future = (
        asyncio.gather(*graph_tasks, return_exceptions=False)
        if graph_tasks
        else asyncio.sleep(0, result=[])
    )
    vector_results, graph_batches = await asyncio.gather(vector_task, graph_future)
    graph_results = [item for batch in graph_batches for item in batch]

    if dependency_answer is not None:
        for dep_path in dependency_answer.get("paths", []) or []:
            nodes = dep_path.get("nodes", []) or []
            relations = dep_path.get("relations", []) or []
            if not nodes:
                continue
            graph_results.append({
                "nodes": [n.get("name", "") if isinstance(n, dict) else str(n) for n in nodes],
                "relations": [
                    r.get("type", "") if isinstance(r, dict) else str(r)
                    for r in relations
                ],
                "score": float(dep_path.get("avg_confidence", 0.5)),
                "source": "graph",
                "entity": "dependency_path",
            })

    fused_results = fuse_results(vector_results, graph_results)
    reranked_docs = await asyncio.to_thread(
        rerank_with_cross_encoder, query, fused_results, llm_client,
    )
    context, selected_docs = compress_context(reranked_docs, token_limit=4096)

    if dependency_answer is not None:
        dep_context = _build_dependency_context_prompt(dependency_answer)
        context = dep_context + "\n\n" + context

    kg_path = format_kg_paths_clean(graph_results)
    prompt = assemble_prompt(query, context, kg_path)
    answer = await asyncio.to_thread(generate_answer, prompt, llm_client)
    verified, verification_details = await asyncio.to_thread(
        verify_answer_with_nli, answer, context, llm_client,
    )
    if not verified:
        retry_answer = await asyncio.to_thread(
            retry_generate_with_expanded_context,
            query, context, kg_path, llm_client, answer,
        )
        if retry_answer:
            verified, verification_details = await asyncio.to_thread(
                verify_answer_with_nli, retry_answer, context, llm_client,
            )
            if verified:
                answer = retry_answer
    citations = build_citations(selected_docs)

    has_dependency_info = dependency_answer is not None

    if not verified:
        fallback = get_fallback_response()
        response_payload = {
            "query_id": query_id,
            "answer": fallback["answer"],
            "dependency_answer": dependency_answer,
            "citations": [],
            "linked_entities": linked_entities,
            "verification": verification_details,
            "status": "fallback",
            "query_intent": query_intent,
        }
        await asyncio.to_thread(
            append_jsonl_record,
            QUERY_LOG_PATH,
            build_query_log_record(
                query_id=query_id,
                query=query,
                context=context,
                kg_path=kg_path,
                response=str(fallback["answer"]),
                verification=verification_details,
                linked_entities=linked_entities,
                citations=[],
                status="fallback",
            ),
        )
        return response_payload

    response_payload: dict[str, Any] = {
        "query_id": query_id,
        "answer": answer,
        "citations": citations,
        "linked_entities": linked_entities,
        "verification": verification_details,
        "status": "ok",
        "query_intent": query_intent,
    }
    if has_dependency_info:
        response_payload["dependency_answer"] = dependency_answer

    await asyncio.to_thread(
        append_jsonl_record,
        QUERY_LOG_PATH,
        build_query_log_record(
            query_id=query_id,
            query=query,
            context=context,
            kg_path=kg_path,
            response=answer,
            verification=verification_details,
            linked_entities=linked_entities,
            citations=citations,
            status="ok",
        ),
    )
    return response_payload


@app.post("/feedback")
async def handle_feedback(feedback: FeedbackRequest) -> dict[str, Any]:
    """Persist user feedback to the feedback log."""
    query_id = feedback.query_id.strip()
    if not query_id:
        raise HTTPException(status_code=400, detail="query_id cannot be empty")

    await asyncio.to_thread(
        append_jsonl_record,
        FEEDBACK_LOG_PATH,
        build_feedback_log_record(
            query_id=query_id,
            is_helpful=feedback.is_helpful,
            comment=feedback.comment,
        ),
    )
    return {"status": "logged", "query_id": query_id}


def _retrieve_graph_for_entity(entity_name: str, neo4j_driver: Any) -> list[dict[str, Any]]:
    """Fetch graph paths for one linked entity within a fresh Neo4j session."""
    if neo4j_driver is None:
        return []
    try:
        with neo4j_driver.session() as neo4j_session:
            return retrieve_graph_path(entity_name, neo4j_session, max_hops=2)
    except (AuthError, ServiceUnavailable, Neo4jError):
        return []


def _link_entities(query: str, neo4j_driver: Any) -> list[str]:
    """Link query entities within a fresh Neo4j session."""
    if neo4j_driver is None:
        return []
    try:
        with neo4j_driver.session() as neo4j_session:
            return link_entities(query, neo4j_session)
    except (AuthError, ServiceUnavailable, Neo4jError):
        return []


def _build_dependency_answer(query: str, neo4j_driver: Any, llm_client: Any) -> dict[str, Any] | None:
    """Build a structured concept dependency answer using a fresh Neo4j session."""
    if neo4j_driver is None:
        return None
    try:
        with neo4j_driver.session() as neo4j_session:
            return build_dependency_answer(query, neo4j_session, llm_client)
    except (AuthError, ServiceUnavailable, Neo4jError):
        return None


def _build_dependency_context_prompt(dependency_answer: dict[str, Any]) -> str:
    """Build a prompt-friendly context block from dependency answer prerequisites."""
    prerequisites = dependency_answer.get("prerequisites", []) or []
    if not prerequisites:
        return ""

    lines = ["【先修知识依赖关系】"]
    for item in prerequisites:
        course = item.get("course", "")
        chapter = item.get("chapter", "")
        concepts = item.get("concepts", []) or []
        concept_names = "、".join(c.get("name", "") for c in concepts if isinstance(c, dict))
        if concept_names:
            lines.append(f"- {course} · {chapter}：{concept_names}")

    explanation = dependency_answer.get("explanation", "")
    if explanation:
        lines.append(f"\n先修关系说明：{explanation}")

    return "\n".join(lines)


def format_kg_paths_clean(graph_results: list[dict[str, Any]]) -> str:
    """Format graph retrieval results into prompt-friendly text."""
    lines: list[str] = []
    for item in graph_results[:5]:
        path_text = " -> ".join(item.get("nodes", []))
        relation_text = ", ".join(item.get("relations", []))
        score = float(item.get("score", 0))
        lines.append(f"PATH: {path_text} | RELATIONS: {relation_text} | SCORE: {score:.2f}")
    return "\n".join(lines)


def classify_query_intent(query: str) -> str:
    """Classify query as 'dependency', 'definition', or 'general'."""
    dep_patterns = [
        r"(为什么|为何).*(学|需要|先修|前置|依赖|前提)",
        r"(依赖|先修|前置|前提|基础).*(什么|哪些|哪个)",
        r"(课程|概念|知识).*(顺序|路径|依赖|关联)",
        r"(先学|后学|前置课程|先修课|预修课)",
        r"(有什么|有哪些).*(依赖|联系|关系)",
    ]
    for pattern in dep_patterns:
        if re.search(pattern, query):
            return "dependency"

    def_patterns = [
        r"(什么是|什么叫|什么是).*",
        r".*(定义|概念|含义|指的是|是什么)",
    ]
    for pattern in def_patterns:
        if re.search(pattern, query):
            return "definition"

    return "general"
