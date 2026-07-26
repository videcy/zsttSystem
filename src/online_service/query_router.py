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
from src.online_service.persona import (
    DEFAULT_PERSONA,
    PERSONA_MODE,
    PERSONA_PROFILES,
    PERSONA_PROFILE_VERSION,
    Persona,
)


class QueryType(Enum):
    DEPENDENCY = "dependency"
    FACT = "fact"
    CONTENT = "content"
    CATALOG = "catalog"
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
    r"(学分|学时|开课学期|考核方式|任课教师|课程负责人|谁负责|负责人|教材|上课地点|教室)",
    r"(课程代码|课程编号|课号)",
    r"(几学分|多少学时|哪个老师|谁教)",
    r"(必修|选修|任选|公选|通识)",
    r"(期末考试|期中|平时成绩|考核|评分).{0,4}(方式|办法|比例|占比)",
]

_CATALOG_PATTERNS: list[str] = [
    r"(培养方案|课程设置|课程体系)",
    r"(专业|主修|辅修).{0,8}(核心课|核心课程|基础课|必修课|选修课|课程)",
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
            try:
                self.courses = json.loads(
                    config.courses_output_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                pass

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

        for pattern in _CATALOG_PATTERNS:
            if re.search(pattern, textual):
                return QueryType.CATALOG

        if re.search(r"(主要讲|主要学习|学什么|讲什么|内容|介绍|包括)", textual):
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
        persona: Persona = DEFAULT_PERSONA,
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
            result = await self._handle_dependency(query, neo4j_driver, llm_client)
        elif query_type == QueryType.FACT:
            # ---- Path 2: ChromaDB retrieval ----
            result = await self._handle_fact(query)
        elif query_type == QueryType.CONTENT:
            result = await self._handle_content(
                query,
                llm_client,
                neo4j_driver=neo4j_driver,
                persona=persona,
            )
        elif query_type == QueryType.CATALOG:
            result = self._handle_catalog(query)
        else:
            # ---- Path 3: ChromaDB + Neo4j + grounded generation ----
            result = await self._handle_hybrid(
                query,
                llm_client,
                neo4j_driver=neo4j_driver,
                persona=persona,
            )
        result.metadata.update(
            persona=persona,
            persona_mode=PERSONA_MODE,
            persona_profile_version=PERSONA_PROFILE_VERSION,
        )
        return result

    # ------------------------------------------------------------------
    # Path handlers
    # ------------------------------------------------------------------
    async def _handle_fact(self, query: str) -> RouteResult:
        course = self._link_course(query)
        if course:
            offering = self._select_offering(course, query)
            requested_fields = [
                field
                for field, pattern in (
                    ("credits", r"学分"),
                    ("hours", r"学时|课时"),
                    ("semester", r"学期"),
                    (
                        "instructor",
                        r"任课教师|课程负责人|谁负责|负责人|谁教|哪个老师",
                    ),
                )
                if re.search(pattern, query)
            ]
            if requested_fields:
                labels = {
                    "credits": "学分",
                    "hours": "总学时",
                    "semester": "开课学期",
                    "instructor": "课程负责人",
                }
                facts = []
                for field in requested_fields:
                    value = (
                        offering.get(field)
                        if offering and offering.get(field) not in (None, "")
                        else course.get(field)
                    )
                    facts.append(f"{labels[field]}为 {value}")
                citation = self._catalog_citation(course, offering)
                return RouteResult(
                    f"{course.get('course_name') or course.get('course_code')}："
                    f"{'，'.join(facts)}。",
                    [citation],
                    "fact",
                    {
                        "vector_hits": 0,
                        "graph_nodes": 0,
                        "llm_used": False,
                        "source_type": "training_plan",
                    },
                )
        hits = await asyncio.to_thread(self.vector_retriever.search, query, 5)
        if not hits:
            return RouteResult("本地知识库暂无可用证据。", query_type="fact", metadata={"status": "degraded", "error_code": "VECTOR_INDEX_UNAVAILABLE"})
        return RouteResult(hits[0].get("text", "")[:500], self._vector_citations(hits), "fact", {"vector_hits": len(hits), "graph_nodes": 0, "llm_used": False})

    def _link_course(self, query: str) -> dict[str, Any] | None:
        lowered = query.lower()
        matches = [c for c in self.courses if str(c.get("course_code", "")).lower() in lowered or str(c.get("course_name", "")) in query]
        return max(matches, key=lambda c: len(str(c.get("course_name", "")))) if matches else None

    @staticmethod
    def _program_keyword(query: str) -> str:
        if "信管" in query:
            return "信息管理与信息系统"
        for keyword in (
            "信息管理与信息系统",
            "图书情报与档案管理类",
            "图书馆学",
            "档案学",
        ):
            if keyword in query:
                return keyword
        return ""

    @staticmethod
    def _program_type_matches(program_type: str, query: str) -> bool:
        if "辅修微专业" in query or "微专业" in query:
            return program_type == "辅修微专业"
        if "辅修" in query:
            return program_type in {"辅修专业", "辅修微专业"}
        return program_type in {"主修专业", "主修专业类"}

    def _select_offering(
        self,
        course: dict[str, Any],
        query: str,
    ) -> dict[str, Any] | None:
        offerings = list(course.get("offerings") or [])
        if not offerings:
            return None
        program_keyword = self._program_keyword(query)
        matches = [
            offering
            for offering in offerings
            if (
                not program_keyword
                or program_keyword in str(offering.get("program_name", ""))
            )
            and self._program_type_matches(
                str(offering.get("program_type", "")),
                query,
            )
        ]
        return (matches or offerings)[0]

    @staticmethod
    def _catalog_citation(
        course: dict[str, Any],
        offering: dict[str, Any] | None,
    ) -> dict[str, Any]:
        offering = offering or {}
        source_file = str(
            offering.get("source_file") or course.get("source_file") or ""
        ).replace("\\", "/")
        return {
            "source_file": source_file.rsplit("/", 1)[-1] or None,
            "course_code": course.get("course_code"),
            "course_name": course.get("course_name"),
            "section": "培养方案课程信息",
        }

    def _handle_catalog(self, query: str) -> RouteResult:
        program_keyword = self._program_keyword(query)
        category_keyword = (
            "核心"
            if "核心" in query
            else "基础"
            if "基础" in query
            else "必修"
            if "必修" in query
            else "选修"
            if "选修" in query
            else ""
        )
        matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for course in self.courses:
            for offering in course.get("offerings") or []:
                program_name = str(offering.get("program_name", ""))
                category = " ".join(
                    [
                        str(offering.get("course_category", "")),
                        str(offering.get("course_subcategory", "")),
                    ]
                )
                if program_keyword and program_keyword not in program_name:
                    continue
                if not self._program_type_matches(
                    str(offering.get("program_type", "")),
                    query,
                ):
                    continue
                if category_keyword and category_keyword not in category:
                    continue
                matches.append((course, offering))
                break

        if not matches:
            return RouteResult(
                "培养方案中未找到符合条件的课程。",
                query_type="catalog",
                metadata={
                    "status": "degraded",
                    "error_code": "CATALOG_NO_MATCH",
                    "llm_used": False,
                },
            )

        def sort_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple:
            course, offering = item
            semester = offering.get("semester")
            semester_key = float(semester) if isinstance(semester, (int, float)) else 99
            return semester_key, str(course.get("course_code", ""))

        matches.sort(key=sort_key)
        lines = [
            f"{course.get('course_name')}（{course.get('course_code')}）"
            for course, _offering in matches
        ]
        description = (
            f"{program_keyword or '目标专业'}的"
            f"{category_keyword or ''}课程包括：\n"
            + "\n".join(f"{index}. {line}" for index, line in enumerate(lines, 1))
        )
        citations: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for course, offering in matches:
            citation = self._catalog_citation(course, offering)
            citation["course_code"] = None
            citation["course_name"] = offering.get("program_name")
            citation["section"] = (
                f"{offering.get('course_category', '')}"
                f" / {offering.get('course_subcategory', '')}"
            ).strip(" /")
            source = str(citation.get("source_file", ""))
            if source and source not in seen_sources:
                citations.append(citation)
                seen_sources.add(source)
        return RouteResult(
            description,
            citations,
            "catalog",
            {
                "catalog_matches": len(matches),
                "llm_used": False,
                "source_type": "training_plan",
            },
        )

    async def _retrieve_course_evidence(
        self,
        query: str,
        persona: Persona = DEFAULT_PERSONA,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        profile = PERSONA_PROFILES[persona]
        course = self._link_course(query)
        course_code = str(course.get("course_code", "")) if course else None
        syllabus_min_score = 0.12 if course else 0.5
        plan_min_score = 0.08 if course else 0.5
        preferred = profile["preferred_sections"]
        syllabus_hits = await asyncio.to_thread(
            self.vector_retriever.search,
            query,
            profile["syllabus_top_k"],
            course_code=course_code,
            source_types=("syllabus",),
            min_score=syllabus_min_score,
            preferred_section_types=preferred,
            source_boosts=profile["source_boosts"],
        )
        plan_query = (
            f"{course.get('course_name', '')} {course_code} 培养方案 课程信息"
            if course
            else query
        )
        plan_hits = await asyncio.to_thread(
            self.vector_retriever.search,
            plan_query,
            profile["plan_top_k"],
            course_code=course_code,
            source_types=("training_plan",),
            min_score=plan_min_score,
            preferred_section_types=preferred,
            source_boosts=profile["source_boosts"],
        )
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in [*syllabus_hits, *plan_hits]:
            identity = str(hit.get("chunk_id") or hit.get("text", ""))
            if identity and identity not in seen:
                hits.append(hit)
                seen.add(identity)
        hits.sort(
            key=lambda hit: float(
                hit.get("rerank_score", hit.get("score", 0.0))
            ),
            reverse=True,
        )
        return course, hits

    @staticmethod
    def _evidence_terms(query: str) -> set[str]:
        chinese = re.sub(r"[^\u4e00-\u9fff]", "", query)
        terms = {
            chinese[index : index + 2]
            for index in range(max(0, len(chinese) - 1))
        }
        terms.update(re.findall(r"[a-z][a-z0-9]+", query.lower()))
        return terms

    @classmethod
    def _relevant_excerpt(cls, query: str, text: str, limit: int = 700) -> str:
        """Keep complete, query-relevant passages instead of a fixed prefix."""
        text = text.strip()
        if len(text) <= limit:
            return text
        terms = cls._evidence_terms(query)
        passages = [
            passage.strip()
            for passage in re.split(r"(?<=[。！？；])|\n+", text)
            if passage.strip()
        ]
        ranked = sorted(
            enumerate(passages),
            key=lambda item: (
                -sum(term in item[1].lower() for term in terms),
                item[0],
            ),
        )
        selected: list[tuple[int, str]] = []
        used = 0
        for index, passage in ranked:
            if used + len(passage) > limit and selected:
                continue
            selected.append((index, passage))
            used += len(passage)
            if used >= limit:
                break
        return "\n".join(passage for _, passage in sorted(selected))

    @classmethod
    def _build_evidence_items(
        cls,
        query: str,
        hits: list[dict[str, Any]],
        graph_paths: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        budget = 4200
        for hit in hits:
            metadata = hit.get("metadata") or {}
            section = (
                hit.get("section")
                or metadata.get("syllabus_section")
                or "课程资料"
            )
            excerpt = cls._relevant_excerpt(query, str(hit.get("text", "")))
            if (
                not excerpt
                or sum(len(str(item.get("excerpt", ""))) for item in items)
                + len(excerpt)
                > budget
            ):
                continue
            items.append(
                {
                    "course_code": metadata.get("course_code")
                    or hit.get("course_code"),
                    "course_name": metadata.get("course_name")
                    or hit.get("course_name"),
                    "section": section,
                    "source_file": hit.get("source_file"),
                    "source_type": metadata.get("source_type")
                    or hit.get("source_type"),
                    "section_type": hit.get("section_type")
                    or metadata.get("section_type"),
                    "excerpt": excerpt,
                }
            )
        if graph_paths:
            readable_paths = [
                " → ".join(str(node) for node in path.get("path", []) if node)
                for path in graph_paths
            ]
            readable_paths = [path for path in readable_paths if path]
            if readable_paths:
                items.append(
                    {
                        "course_code": None,
                        "course_name": None,
                        "section": "知识图谱关联",
                        "source_file": "Neo4j",
                        "source_type": "knowledge_graph",
                        "excerpt": "\n".join(readable_paths[:8]),
                    }
                )
        return items

    async def _handle_content(
        self,
        query: str,
        llm_client: Any,
        *,
        neo4j_driver: Any = None,
        persona: Persona = DEFAULT_PERSONA,
    ) -> RouteResult:
        _course, hits = await self._retrieve_course_evidence(query, persona)
        graph_paths: list[dict[str, Any]] = []
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
            from src.online_service.generator import build_fallback_answer

            return RouteResult(build_fallback_answer(query, [], persona), query_type="content", metadata={"status": "degraded", "error_code": "NO_RELEVANT_EVIDENCE"})
        evidence_items = self._build_evidence_items(query, hits, graph_paths)
        from src.online_service.generator import (
            build_fallback_answer,
            generate_answer_once,
        )

        try:
            answer = await asyncio.to_thread(
                generate_answer_once,
                query,
                evidence_items,
                llm_client,
                persona,
            )
        except Exception:
            answer = build_fallback_answer(query, evidence_items, persona)
        return RouteResult(answer, self._vector_citations(hits), "content", {"vector_hits": len(hits), "graph_nodes": len(graph_paths), "llm_used": llm_client is not None})

    async def _handle_hybrid(
        self,
        query: str,
        llm_client: Any,
        neo4j_driver: Any = None,
        persona: Persona = DEFAULT_PERSONA,
        **kwargs: Any,
    ) -> RouteResult:
        _course, hits = await self._retrieve_course_evidence(query, persona)
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
            from src.online_service.generator import build_fallback_answer

            return RouteResult(build_fallback_answer(query, [], persona), query_type="hybrid", metadata={"status": "degraded", "error_code": "NO_EVIDENCE"})
        evidence_items = self._build_evidence_items(query, hits, graph_paths)
        from src.online_service.generator import (
            build_fallback_answer,
            generate_answer_once,
        )
        metadata = {
            "vector_hits": len(hits),
            "graph_nodes": len(graph_paths),
            "llm_used": llm_client is not None,
        }
        try:
            answer = await asyncio.to_thread(
                generate_answer_once,
                query,
                evidence_items,
                llm_client,
                persona,
            )
        except Exception:
            answer = build_fallback_answer(query, evidence_items, persona)
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
        citations: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for hit in hits:
            metadata = hit.get("metadata") or {}
            source_file = str(hit.get("source_file") or "").replace("\\", "/")
            citation = {
                "source_file": source_file.rsplit("/", 1)[-1].split("#", 1)[0]
                or None,
                "course_code": metadata.get("course_code")
                or hit.get("course_code"),
                "course_name": metadata.get("course_name")
                or hit.get("course_name"),
                "section": hit.get("section")
                or metadata.get("syllabus_section"),
            }
            identity = tuple(citation.values())
            if identity not in seen:
                citations.append(citation)
                seen.add(identity)
        return citations

    async def _handle_dependency(
        self,
        query: str,
        neo4j_driver: Any,
        llm_client: Any,
    ) -> RouteResult:
        """Dependency reasoning via Neo4j concept graph."""
        from src.online_service.course_dependency_service import (
            get_course_dependency_subgraph,
        )
        from src.online_service.dependency_explainer import build_dependency_answer

        course = self._link_course(query)
        if course and re.search(r"先修课程|先修课|前置课程|预修课", query):
            prerequisites = [
                str(value).strip()
                for value in course.get("prerequisites") or []
                if str(value).strip()
            ]
            hits: list[dict[str, Any]] = []
            supporting_hits: list[dict[str, Any]] = []
            if not prerequisites:
                hits = await asyncio.to_thread(
                    self.vector_retriever.search,
                    query,
                    10,
                    course_code=str(course.get("course_code", "")),
                    source_types=("syllabus",),
                    min_score=0.2,
                    preferred_section_types=("prerequisites", "basic_info"),
                )
                for hit in hits:
                    matched = False
                    for value in re.findall(
                        r"先修课程\s*[：:]\s*([^\n。；]+)",
                        str(hit.get("text", "")),
                    ):
                        normalized = value.strip()
                        if normalized and normalized not in prerequisites:
                            prerequisites.append(normalized)
                        matched = True
                    if matched:
                        supporting_hits.append(hit)
            if prerequisites:
                course_label = course.get("course_name") or course.get("course_code")
                course_by_code = {
                    str(item.get("course_code", "")): item
                    for item in self.courses
                }
                prerequisite_labels = [
                    (
                        f"{course_by_code[value].get('course_name')}（{value}）"
                        if value in course_by_code
                        else value
                    )
                    for value in prerequisites
                ]
                dependency_info = None
                if neo4j_driver is not None:
                    try:
                        dependency_info = await asyncio.to_thread(
                            get_course_dependency_subgraph,
                            neo4j_driver,
                            str(course.get("course_code", "")),
                            depth=2,
                            max_nodes=30,
                        )
                    except Exception:
                        dependency_info = None
                return RouteResult(
                    answer=(
                        f"{course_label}的先修课程为："
                        f"{'、'.join(prerequisite_labels)}。"
                    ),
                    citations=self._vector_citations(supporting_hits)
                    if supporting_hits
                    else [self._catalog_citation(course, None)],
                    query_type="dependency",
                    dependency_info=dependency_info,
                    metadata={
                        "backend": (
                            "syllabus+Neo4j"
                            if dependency_info
                            else "syllabus"
                        ),
                        "llm_used": False,
                        "prerequisite_count": len(prerequisites),
                    },
                )

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
