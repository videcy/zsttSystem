"""
Query router for the zsttSystem online service.

Routes user queries to the most appropriate backend based on intent
classification:
- **dependency**  → Neo4j concept dependency reasoning (zsttSystem native)
- **fact/content** → ChromaDB vector retrieval
- **hybrid**      → ChromaDB + Neo4j + grounded LLM generation
"""
from __future__ import annotations

import asyncio
import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.config import config
from src.online_service.chroma_retriever import ChromaRetriever


class QueryType(Enum):
    DEPENDENCY = "dependency"
    FACT = "fact"
    CONTENT = "content"
    HYBRID = "hybrid"
    SIMPLE = "fact"
    COMPLEX = "hybrid"


# ---------------------------------------------------------------------------
# Intent classification patterns
# ---------------------------------------------------------------------------

_DEPENDENCY_PATTERNS: list[str] = [
    r"(为什么|为何).{0,4}(学|需要|先修|前置|依赖|前提|先上)",
    r"(依赖|先修|前置|前提|基础).{0,4}(什么|哪些|哪个)",
    r"(课程|概念|知识).{0,4}(顺序|路径|依赖|关联)",
    r"(先学|后学|前置课程|先修课|预修课|前导课)",
    r"(有什么|有哪些).{0,4}(依赖|联系|关系|关联)",
    r"(怎么|如何).{0,4}(选修|选课|规划|安排)",
]

_FACT_PATTERNS: list[str] = [
    r"(学分|学时|开课学期|考核方式|任课教师|教材|上课地点|教室)",
    r"(课程代码|课程编号|课号)",
    r"(几学分|多少学时|哪个老师|谁教)",
    r"(必修|选修|任选|公选|通识)",
    r"(期末考试|期中|平时成绩|考核|评分).{0,4}(方式|办法|比例|占比)",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    """Unified return type for all query paths."""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    query_type: str = "complex"
    metadata: dict[str, Any] = field(default_factory=dict)
    dependency_info: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """Intent classifier + dispatcher for the zsttSystem query pipeline."""

    def __init__(self, vector_retriever: ChromaRetriever | None = None):
        self.vector_retriever = vector_retriever or ChromaRetriever()
        self.courses = []
        if config.courses_output_path.exists():
            try: self.courses = json.loads(config.courses_output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError): pass

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------
    @staticmethod
    def classify(query: str) -> QueryType:
        """Classify a user query into one of three intent types."""
        textual = query.strip().lower()

        for pattern in _DEPENDENCY_PATTERNS:
            if re.search(pattern, textual):
                return QueryType.DEPENDENCY

        for pattern in _FACT_PATTERNS:
            if re.search(pattern, textual):
                return QueryType.FACT

        if re.search(r"(主要讲|内容|介绍|包括)", textual):
            return QueryType.CONTENT
        return QueryType.HYBRID

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------
    async def route(
        self,
        query: str,
        query_id: str,
        *,
        neo4j_driver: Any = None,
        llm_client: Any = None,
    ) -> RouteResult:
        """Route a query to the appropriate backend.

        Required callbacks for zsttSystem-native paths:
        * ``neo4j_driver`` – Neo4j driver for dependency reasoning.
        * ``llm_client`` – LLM client for grounded hybrid generation.

        Returns a ``RouteResult`` with the final answer, citations, and
        metadata indicating which backends were used.
        """
        query_type = self.classify(query)

        # ---- Path 1: Dependency reasoning (zsttSystem-native) ----
        if query_type == QueryType.DEPENDENCY:
            return await self._handle_dependency(query, neo4j_driver, llm_client)

        # ---- Path 2: ChromaDB retrieval ----
        if query_type == QueryType.FACT:
            return await self._handle_fact(query)
        if query_type == QueryType.CONTENT:
            return await self._handle_content(query)

        # ---- Path 3: ChromaDB + Neo4j + grounded generation ----
        return await self._handle_hybrid(
            query,
            llm_client,
            neo4j_driver=neo4j_driver,
        )

    # ------------------------------------------------------------------
    # Path handlers
    # ------------------------------------------------------------------
    async def _handle_fact(self, query: str) -> RouteResult:
        course = self._link_course(query)
        if course:
            asked = "credits" if re.search(r"学分", query) else "hours" if re.search(r"学时|课时", query) else None
            if asked:
                value = course.get(asked)
                return RouteResult(f"{course.get('course_name') or course.get('course_code')}：{asked == 'credits' and '学分' or '学时'}为 {value}。", [{"source_file": course.get("source_file"), "course_code": course.get("course_code"), "course_name": course.get("course_name"), "score": 1.0}], "fact", {"vector_hits": 0, "graph_nodes": 1, "llm_used": False})
        hits = await asyncio.to_thread(self.vector_retriever.search, query, 5)
        if not hits:
            return RouteResult("本地知识库暂无可用证据。", query_type="fact", metadata={"status": "degraded", "error_code": "VECTOR_INDEX_UNAVAILABLE"})
        return RouteResult(hits[0].get("text", "")[:500], self._vector_citations(hits), "fact", {"vector_hits": len(hits), "graph_nodes": 0, "llm_used": False})

    def _link_course(self, query: str) -> dict[str, Any] | None:
        lowered = query.lower()
        matches = [c for c in self.courses if str(c.get("course_code", "")).lower() in lowered or str(c.get("course_name", "")) in query]
        return max(matches, key=lambda c: len(str(c.get("course_name", "")))) if matches else None

    async def _handle_content(self, query: str) -> RouteResult:
        hits = await asyncio.to_thread(self.vector_retriever.search, query, 5)
        if not hits:
            return RouteResult("本地向量索引当前不可用。", query_type="content", metadata={"status": "degraded", "error_code": "VECTOR_INDEX_UNAVAILABLE"})
        answer = "\n\n".join(h.get("text", "")[:300] for h in hits[:3])
        return RouteResult(answer, self._vector_citations(hits), "content", {"vector_hits": len(hits), "graph_nodes": 0, "llm_used": False})

    async def _handle_hybrid(self, query: str, llm_client: Any, neo4j_driver: Any = None, **kwargs: Any) -> RouteResult:
        hits = await asyncio.to_thread(self.vector_retriever.search, query, 5)
        graph_paths = []
        if neo4j_driver is not None:
            try:
                graph_paths = await asyncio.to_thread(
                    _query_graph_paths,
                    neo4j_driver,
                    query,
                )
            except Exception:
                graph_paths = []
        if not hits and not graph_paths:
            return RouteResult("当前无法获取足够的图谱或文本证据。", query_type="hybrid", metadata={"status": "degraded", "error_code": "NO_EVIDENCE"})
        evidence = "\n\n".join(h.get("text", "")[:250] for h in hits[:3]) + "\n图谱路径：" + json.dumps(graph_paths, ensure_ascii=False)
        from src.online_service.generator import generate_answer_once
        metadata = {
            "vector_hits": len(hits),
            "graph_nodes": len(graph_paths),
            "llm_used": llm_client is not None,
        }
        try:
            answer = await asyncio.to_thread(
                generate_answer_once,
                query,
                evidence,
                llm_client,
            )
        except Exception:
            answer = evidence[:1200]
            metadata.update(
                status="degraded",
                error_code="LLM_UNAVAILABLE",
            )
        return RouteResult(
            answer,
            self._vector_citations(hits),
            "hybrid",
            metadata,
            {"paths": graph_paths},
        )

    @staticmethod
    def _vector_citations(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"source_file": h.get("source_file"), "course_code": (h.get("metadata") or {}).get("course_code") or h.get("course_code"), "course_name": (h.get("metadata") or {}).get("course_name"), "section": h.get("section") or (h.get("metadata") or {}).get("syllabus_section"), "chunk_id": h.get("chunk_id"), "score": h.get("score")} for h in hits]

    async def _handle_dependency(
        self,
        query: str,
        neo4j_driver: Any,
        llm_client: Any,
    ) -> RouteResult:
        """Dependency reasoning via Neo4j concept graph."""
        from src.online_service.dependency_explainer import build_dependency_answer

        if neo4j_driver is None:
            return RouteResult(
                answer="课程依赖查询需要 Neo4j 图数据库支持，当前服务未连接 Neo4j。",
                query_type="dependency",
                metadata={"backend": "zsttSystem_Neo4j", "status": "neo4j_unavailable"},
            )

        try:
            with neo4j_driver.session() as session:
                dep_result = await asyncio.to_thread(
                    build_dependency_answer, query, session, llm_client
                )
        except Exception as exc:
            return RouteResult(
                answer="课程依赖查询暂时不可用，请稍后重试。",
                query_type="dependency",
                metadata={"backend": "zsttSystem_Neo4j", "error": str(exc)},
            )

        if dep_result is None:
            return RouteResult(
                answer="未找到与该问题相关的课程依赖关系。",
                query_type="dependency",
                metadata={"backend": "zsttSystem_Neo4j"},
            )

        return RouteResult(
            answer=dep_result.get("explanation", ""),
            query_type="dependency",
            dependency_info=dep_result,
            metadata={"backend": "zsttSystem_Neo4j"},
        )

# ---------------------------------------------------------------------------
# Internal helpers (used by the router)
# ---------------------------------------------------------------------------

def _query_graph_paths(neo4j_driver: Any, query: str) -> list[dict[str, Any]]:
    """Fetch nearby graph paths without blocking the async event loop."""
    cypher = """
    MATCH p=(c:Course)-[*1..2]-(n)
    WHERE toLower($query) CONTAINS toLower(c.course_name)
       OR toLower($query) CONTAINS toLower(c.course_code)
    RETURN [x IN nodes(p) |
        coalesce(x.course_name, x.name, x.course_code)
    ] AS path
    LIMIT 20
    """
    with neo4j_driver.session() as session:
        rows = session.run(cypher, query=query)
        return [
            {
                "path": row["path"],
                "relationship": "NEIGHBOR",
                "confidence": 1.0,
            }
            for row in rows
        ]
