"""FastAPI entry point for the zsttsystem online RAG-KG service."""

from __future__ import annotations

import asyncio
import os
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

from src.online_service.feedback_handler import (
    append_jsonl_record,
    build_citations,
    build_feedback_log_record,
    build_query_log_record,
)
from src.online_service.generator import (
    assemble_prompt,
    generate_answer,
    get_fallback_response,
    verify_answer_with_nli,
)
from src.online_service.local_vector_store import LocalVectorCollection
from src.online_service.query_processor import HyDE, link_entities
from src.online_service.ranker import compress_context, fuse_results, rerank_with_cross_encoder
from src.online_service.retriever import retrieve_graph_path, retrieve_vectors
from src.utils.glm_client import create_glm_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", str(PROJECT_ROOT / "vector_store")))
COLLECTION_NAME = "scholar_collection"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
GLM_TEXT_MODEL = os.getenv("GLM_TEXT_MODEL", "glm-5")
GLM_EMBEDDING_MODEL = os.getenv("GLM_EMBEDDING_MODEL", "embedding-3")
GLM_RERANK_MODEL = os.getenv("GLM_RERANK_MODEL", GLM_TEXT_MODEL)
GLM_JUDGE_MODEL = os.getenv("GLM_JUDGE_MODEL", GLM_TEXT_MODEL)
QUERY_LOG_PATH = Path(
    os.getenv("QUERY_LOG_PATH", str(PROJECT_ROOT / "outputs" / "query_log.jsonl"))
)
FEEDBACK_LOG_PATH = Path(
    os.getenv("FEEDBACK_LOG_PATH", str(PROJECT_ROOT / "outputs" / "feedback_log.jsonl"))
)


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

    glm_client = create_glm_client()
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

    app.state.glm_client = glm_client
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
    """Serve a lightweight local demo page."""
    return """
    <!doctype html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>zsttsystem Local Demo</title>
        <style>
            :root {
                color-scheme: light;
                --bg: #f6f8fb;
                --panel: #ffffff;
                --border: #d8e0ea;
                --text: #1f2937;
                --muted: #6b7280;
                --accent: #14532d;
                --accent-soft: #e8f5ec;
            }
            body {
                margin: 0;
                font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
                background: linear-gradient(180deg, #eef6ff 0%, var(--bg) 100%);
                color: var(--text);
            }
            .wrap {
                max-width: 960px;
                margin: 40px auto;
                padding: 24px;
            }
            .panel {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 24px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            }
            h1 {
                margin-top: 0;
                font-size: 30px;
            }
            p {
                color: var(--muted);
                line-height: 1.7;
            }
            textarea {
                width: 100%;
                min-height: 120px;
                border-radius: 12px;
                border: 1px solid var(--border);
                padding: 14px;
                font: inherit;
                box-sizing: border-box;
                resize: vertical;
            }
            button {
                margin-top: 14px;
                border: 0;
                border-radius: 10px;
                padding: 12px 18px;
                font: inherit;
                cursor: pointer;
                background: var(--accent);
                color: #fff;
            }
            .meta {
                margin-top: 18px;
                padding: 14px;
                border-radius: 12px;
                background: var(--accent-soft);
                color: var(--accent);
                font-size: 14px;
            }
            .result {
                margin-top: 18px;
                padding: 18px;
                border-radius: 12px;
                background: #f8fafc;
                border: 1px solid var(--border);
                white-space: pre-wrap;
                line-height: 1.7;
            }
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="panel">
                <h1>zsttsystem Local Demo</h1>
                <p>Run the offline pipeline first, then enter a question here. This page calls the local <code>/query</code> endpoint directly.</p>
                <textarea id="query" placeholder="Example: What are the core courses in the major?"></textarea>
                <button id="submit">Ask</button>
                <div class="meta" id="status">Status: waiting for input</div>
                <div class="result" id="result">Answer will appear here.</div>
            </div>
        </div>
        <script>
            const submit = document.getElementById("submit");
            const queryInput = document.getElementById("query");
            const status = document.getElementById("status");
            const result = document.getElementById("result");

            submit.addEventListener("click", async () => {
                const query = queryInput.value.trim();
                if (!query) {
                    status.textContent = "Status: enter a question";
                    return;
                }

                submit.disabled = true;
                status.textContent = "Status: querying";
                result.textContent = "Please wait...";

                try {
                    const response = await fetch("/query", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query })
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(data.detail || "Request failed");
                    }

                    const lines = [
                        `Status: ${data.status || "ok"}`,
                        `Query ID: ${data.query_id || ""}`,
                        "",
                        "Answer:",
                        data.answer || "",
                    ];

                    if (Array.isArray(data.linked_entities) && data.linked_entities.length) {
                        lines.push("", "Linked entities:", data.linked_entities.join(", "));
                    }

                    if (Array.isArray(data.citations) && data.citations.length) {
                        lines.push("", "Citations:", JSON.stringify(data.citations, null, 2));
                    }

                    result.textContent = lines.join("\\n");
                    status.textContent = "Status: done";
                } catch (error) {
                    status.textContent = "Status: failed";
                    result.textContent = String(error);
                } finally {
                    submit.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Return a simple health response for local demo checks."""
    return {"status": "ok"}


@app.post("/query")
async def process_query(request: QueryRequest, fastapi_request: Request) -> dict[str, Any]:
    """Execute the end-to-end RAG-KG query pipeline."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")
    query_id = str(uuid.uuid4())

    glm_client = fastapi_request.app.state.glm_client
    chroma_collection = fastapi_request.app.state.chroma_collection
    neo4j_driver = fastapi_request.app.state.neo4j_driver

    hyde_embedding = await asyncio.to_thread(
        HyDE,
        query,
        glm_client,
        glm_client,
        text_model=GLM_TEXT_MODEL,
        embedding_model=GLM_EMBEDDING_MODEL,
    )

    linked_entities = await asyncio.to_thread(_link_entities, query, neo4j_driver)

    graph_tasks = [
        asyncio.to_thread(_retrieve_graph_for_entity, entity_name, neo4j_driver)
        for entity_name in linked_entities[:5]
    ]
    vector_task = asyncio.to_thread(
        retrieve_vectors,
        hyde_embedding,
        chroma_collection,
        10,
    )
    graph_future = (
        asyncio.gather(*graph_tasks, return_exceptions=False)
        if graph_tasks
        else asyncio.sleep(0, result=[])
    )
    vector_results, graph_batches = await asyncio.gather(vector_task, graph_future)
    graph_results = [item for batch in graph_batches for item in batch]

    fused_results = fuse_results(vector_results, graph_results)
    reranked_docs = await asyncio.to_thread(
        rerank_with_cross_encoder,
        query,
        fused_results,
        glm_client,
    )
    context, selected_docs = compress_context(reranked_docs, token_limit=4096)
    kg_path = format_kg_paths_clean(graph_results)
    prompt = assemble_prompt(query, context, kg_path)
    answer = await asyncio.to_thread(generate_answer, prompt, glm_client)
    verified, verification_details = await asyncio.to_thread(
        verify_answer_with_nli,
        answer,
        context,
        glm_client,
    )
    citations = build_citations(selected_docs)

    if not verified:
        fallback = get_fallback_response()
        fallback["query_id"] = query_id
        fallback["verification"] = verification_details
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
        return fallback

    response_payload = {
        "query_id": query_id,
        "answer": answer,
        "citations": citations,
        "linked_entities": linked_entities,
        "verification": verification_details,
        "status": "ok",
    }
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


def _format_kg_paths(graph_results: list[dict[str, Any]]) -> str:
    """Format graph retrieval results into prompt-friendly text."""
    lines: list[str] = []
    for item in graph_results[:5]:
        path_text = " -> ".join(item.get("nodes", []))
        relation_text = ", ".join(item.get("relations", []))
        score = float(item.get("score", 0))
        lines.append(f"路径: {path_text} | 关系: {relation_text} | 强度: {score:.2f}")
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
