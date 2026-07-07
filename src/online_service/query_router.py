"""
Query router for the zsttSystem online service.

Routes user queries to the most appropriate backend based on intent
classification:
- **dependency**  → Neo4j concept dependency reasoning (zsttSystem native)
- **simple**      → LightRAG naive/local mode (fast, low-LLM-cost)
- **complex**     → zsttSystem HyDE + concept normalisation → LightRAG mix mode
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.config import config
from src.online_service.lightrag_adapter import LightRAGClient


class QueryType(Enum):
    DEPENDENCY = "dependency"
    SIMPLE = "simple"
    COMPLEX = "complex"


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

_SIMPLE_PATTERNS: list[str] = [
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

    def __init__(self, lightrag: LightRAGClient):
        self.lightrag = lightrag

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

        for pattern in _SIMPLE_PATTERNS:
            if re.search(pattern, textual):
                return QueryType.SIMPLE

        return QueryType.COMPLEX

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
        enable_hyde: Optional[bool] = None,
        enable_nli: Optional[bool] = None,
    ) -> RouteResult:
        """Route a query to the appropriate backend.

        Required callbacks for zsttSystem-native paths:
        * ``neo4j_driver`` – Neo4j driver for dependency reasoning.
        * ``llm_client`` – LLM client for HyDE expansion and NLI verification.

        Returns a ``RouteResult`` with the final answer, citations, and
        metadata indicating which backends were used.
        """
        enable_hyde = config.enable_hyde_expansion if enable_hyde is None else enable_hyde
        enable_nli = config.enable_nli_verification if enable_nli is None else enable_nli
        query_type = self.classify(query)

        # ---- Path 1: Dependency reasoning (zsttSystem-native) ----
        if query_type == QueryType.DEPENDENCY:
            return await self._handle_dependency(query, neo4j_driver, llm_client)

        # ---- Path 2: Simple factual query (LightRAG naive) ----
        if query_type == QueryType.SIMPLE:
            return await self._handle_simple(query)

        # ---- Path 3: Complex query (HyDE + LightRAG mix) ----
        return await self._handle_complex(
            query, llm_client, enable_hyde=enable_hyde, enable_nli=enable_nli
        )

    # ------------------------------------------------------------------
    # Path handlers
    # ------------------------------------------------------------------
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

    async def _handle_simple(self, query: str) -> RouteResult:
        """Simple fact lookup via LightRAG naive mode."""
        try:
            result = await asyncio.to_thread(
                self.lightrag.query, query, "naive", True
            )
        except Exception as naive_exc:
            # Fallback: try local mode with explicit entity matching
            logging.getLogger(__name__).warning(
                "[query_router] naive mode failed, falling back to local: %s", naive_exc
            )
            try:
                result = await asyncio.to_thread(
                    self.lightrag.query, query, "local", True
                )
            except Exception as exc:
                return RouteResult(
                    answer="检索服务暂时不可用，请检查 LightRAG 服务是否正常运行。",
                    query_type="simple",
                    metadata={"backend": "LightRAG", "error": str(exc)},
                )

        return RouteResult(
            answer=result.get("response", ""),
            citations=self._extract_citations(result),
            query_type="simple",
            metadata={"backend": "LightRAG_naive"},
        )

    async def _handle_complex(
        self,
        query: str,
        llm_client: Any,
        *,
        enable_hyde: bool = True,
        enable_nli: bool = False,
    ) -> RouteResult:
        """Complex QA: HyDE + concept normalisation → LightRAG mix mode."""
        metadata: dict[str, Any] = {"backend": "zsttSystem_HyDE + LightRAG_mix"}

        # Step 1: Standardise concepts in the query
        concepts: list[str] = []
        if config.enable_concept_normalization:
            concepts = await asyncio.to_thread(_normalize_query_concepts, query)

        # Step 2: HyDE expansion (generate hypothetical answer for embedding)
        hyde_answer = ""
        if enable_hyde and llm_client is not None:
            hyde_answer = await asyncio.to_thread(_hyde_expand, query, llm_client)

        # Step 3: Build enhanced query
        enhanced = _build_enhanced_query(query, hyde_answer, concepts)

        # Step 4: LightRAG mix-mode retrieval + generation
        try:
            result = await asyncio.to_thread(
                self.lightrag.query, enhanced, "mix", True
            )
        except Exception as exc:
            # Fallback: try without HyDE enrichment
            try:
                result = await asyncio.to_thread(
                    self.lightrag.query, query, "mix", True
                )
                metadata["fallback"] = "removed_hyde"
            except Exception:
                return RouteResult(
                    answer="检索服务暂时不可用，请检查 LightRAG 服务是否正常运行。",
                    query_type="complex",
                    metadata={**metadata, "error": str(exc)},
                )

        answer = result.get("response", "")
        context = result.get("context", [])

        # Step 5: Optional NLI verification
        if enable_nli and llm_client is not None and answer and context:
            verified, nli_details = await asyncio.to_thread(
                _nli_verify, answer, context, llm_client
            )
            metadata["nli_status"] = "verified" if verified else "failed"
            metadata["nli_details"] = nli_details
            if not verified:
                # One retry with stricter prompt
                retry_query = (
                    f"请严格仅基于以下资料回答，不要推测: {query}\n"
                    f"上一轮回答已通过事实核查标注为需修正。"
                )
                try:
                    retry = await asyncio.to_thread(
                        self.lightrag.query, retry_query, "mix", True
                    )
                    retry_answer = retry.get("response", "")
                    if retry_answer:
                        re_verified, retry_details = await asyncio.to_thread(
                            _nli_verify, retry_answer, context, llm_client
                        )
                        if re_verified:
                            answer = retry_answer
                            metadata["nli_status"] = "verified_after_retry"
                            metadata["nli_details"] = retry_details
                except Exception as nli_retry_exc:
                    logging.getLogger(__name__).warning(
                        "[query_router] NLI retry failed: %s", nli_retry_exc
                    )
        else:
            metadata["nli_status"] = "disabled"

        return RouteResult(
            answer=answer,
            citations=self._extract_citations(result),
            query_type="complex",
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_citations(lightrag_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert LightRAG context entries to zsttSystem-style citations."""
        citations: list[dict[str, Any]] = []
        for i, ctx in enumerate(lightrag_result.get("context", []) or []):
            citations.append({
                "index": i + 1,
                "content": str(ctx.get("content", ""))[:300],
                "source": str(ctx.get("source", ctx.get("source_id", "unknown"))),
            })
        return citations


# ---------------------------------------------------------------------------
# Internal helpers (used by the router)
# ---------------------------------------------------------------------------

def _hyde_expand(query: str, llm_client: Any) -> str:
    """Generate a short hypothetical answer for HyDE embedding."""
    from src.utils.deepseek_client import generate_text

    prompt = (
        "你是一名课程问答助教。请针对下面的用户问题，生成一段简洁、可信、偏教材风格的"
        "假设性参考答案，用于后续检索相关资料。不要输出解释，不要编造超出教学范围的内容。\n\n"
        f"用户问题：{query}"
    )
    return generate_text(
        llm_client,
        config.text_model,
        prompt,
        temperature=0.1,
        max_output_tokens=256,
    ) or ""


def _normalize_query_concepts(query: str) -> list[str]:
    """Extract and normalise concept names from the query text.

    Lightweight version that uses simple tokenisation and a few heuristics
    rather than the full LLM-based pipeline used during offline processing.
    """
    tokens = re.findall(r"[\u4e00-\u9fff\w]{2,20}", query)
    # Filter out trivial stopwords-like tokens
    stop = {"什么", "如何", "怎么", "哪些", "哪个", "为什么", "是不是", "有没有",
            "请问", "问题", "回答", "用户", "帮我", "可以", "这个", "那个",
            "的", "了", "是", "在", "有", "和", "与", "或", "等", "及",
            "吗", "呢", "啊", "吧", "的是", "关于", "对于"}
    return [t for t in tokens if t not in stop][:8]


def _build_enhanced_query(
    query: str,
    hyde_answer: str,
    concepts: list[str],
) -> str:
    """Build the enriched query string for LightRAG."""
    parts = [f"问题: {query}"]
    if concepts:
        parts.append(f"相关概念: {'、'.join(concepts)}")
    if hyde_answer:
        parts.append(
            f"参考信息（以下内容仅做术语参考，请以实际检索到的资料为准）:\n{hyde_answer}"
        )
    return "\n\n".join(parts)


def _nli_verify(
    answer: str,
    context: list[dict[str, Any]],
    llm_client: Any,
) -> tuple[bool, dict[str, Any]]:
    """Run per-sentence NLI entailment check against retrieved context."""
    from src.online_service.generator import verify_answer_with_nli

    context_text = "\n".join(
        str(ctx.get("content", "")) for ctx in (context or [])[:10]
    )
    if not context_text.strip():
        return True, {"status": "no_context"}

    return verify_answer_with_nli(answer, context_text, llm_client)
