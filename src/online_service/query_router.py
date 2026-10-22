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
import difflib
import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.config import config
from src.online_service.chroma_retriever import ChromaRetriever
from src.utils.lexical import term_set
from src.online_service.program_index import ProgramIndex
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
    STRUCTURE = "structure"
    CONTENT = "content"
    CATALOG = "catalog"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Intent classification patterns
# ---------------------------------------------------------------------------

_DEPENDENCY_PATTERNS: list[str] = [
    r"(为什么|为何).{0,4}(学|需要|先修|前置|依赖|前提|先上)",
    r"(依赖|先修|前置|前提|基础).{0,4}(什么|哪些|哪个)",
    # 书名号内的课名会在分类前被屏蔽，因此「学《X》需要什么基础」
    # 需要由书名号外的问法本身命中 dependency。
    r"(需要|需|要求|具备).{0,4}(什么|哪些|哪个).{0,2}(基础|前提|先修|前置|知识)",
    r"(课程|概念|知识).{0,4}(顺序|路径|依赖|关联)",
    r"(先学|后学|前置课程|先修课|预修课|前导课)",
    r"(有什么|有哪些).{0,4}(依赖|联系|关系|关联)",
    r"(怎么|如何).{0,4}(选修|选课|规划|安排)",
]

_ASSESSMENT_MODE_PATTERN = (
    r"怎么考核|如何考核|怎么考试|考核形式|考核办法"
    r"|考核方式|怎么考|怎么评"
)

# 语料不记录字段（S2）。顺序即命中的优先级；只有课程已识别时才拒答。
# 这里是封闭清单：不要把教材、考核方式等部分课程有记录的字段放进来。
_UNRECORDED_FIELD_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "上课地点/教室",
        r"上课地点|上课教室|教室|在哪(?:里|儿)?上课|哪里上课",
    ),
    (
        "考试具体日期/时间",
        r"(?:期末|期中|课程|结课)?考试.{0,8}(?:具体)?"
        r"(?:日期|时间|哪天|哪一天|什么时候|何时|几号)"
        r"|(?:具体)?(?:日期|时间).{0,4}(?:期末|期中|课程)?考试",
    ),
    (
        "历年成绩/平均分/及格率",
        r"(?:历年|往年|去年|往届|上一届|上一学年).{0,8}"
        r"(?:成绩|平均分|及格率)|平均分|及格率",
    ),
    (
        "选课人数",
        r"选课人数|多少人选(?:这|该|这门)?课|选这门课的有多少人",
    ),
)

_FACT_PATTERNS: list[str] = [
    r"(学分|学时|开课学期|考核方式|任课教师|课程负责人|谁负责|负责人|教材|上课地点|教室)",
    r"(课程代码|课程编号|课号)",
    r"(几学分|多少学时|哪个老师|谁教)",
    r"(老师是谁|哪位老师|主讲教师|授课教师|任课老师|授课老师|主讲老师)",
    r"(必修|选修|任选|公选|通识)",
    r"(期末考试|期中|平时成绩|考核|评分).{0,4}(方式|办法|比例|占比)",
    _ASSESSMENT_MODE_PATTERN,
    # Natural phrasings of the semester field: routing evaluation showed that
    # ``开课学期`` alone missed every question a student actually types.
    r"(第几学期|什么学期|哪个学期|哪一学期|几年级).{0,3}(开|上|学|修|开课)?",
    # S2 的封闭字段也必须稳定进入 fact，避免落到 hybrid 后做整体摘要。
    *(pattern for _label, pattern in _UNRECORDED_FIELD_PATTERNS),
]

_CATALOG_PATTERNS: list[str] = [
    r"(培养方案|课程设置|课程体系)",
    r"(专业|主修|辅修).{0,8}(核心课|核心课程|基础课|必修课|选修课|课程)",
]

_CONTENT_PATTERNS: list[str] = [
    r"(主要讲|主要学习|学什么|讲什么|内容|介绍|包括)",
]

# 结构类问题的正则（大创方案 §4 M4.1）。优先级放在 FACT 之后、CONTENT
# 之前：「第三章讲什么」同时命中 content 的「讲什么」，必须先归 structure。
# `目录(?!学)` 里的负向断言是刻意的：课程《目录学》属于事实/内容题，
# 被「目录」误吞会让既有 170 条路由样本回退（见 P1-6 回归要求）。
_STRUCTURE_PATTERNS: list[str] = [
    r"第\s*[一二三四五六七八九十百0-9]+\s*(章|节|讲|周|部分|单元|模块)",
    r"(有|共|分)(几|多少)(章|讲|部分|个单元|个模块)",
    r"(章节|目录(?!学)|大纲结构|教学进度|教学安排|课程安排|进度安排)",
    r"(每周|哪一?周|第几周).{0,6}(学|讲|上)",
]

# Ordered by precedence.  A query that matches several groups is answered by
# the earliest one here, which keeps single-intent behaviour identical to the
# original short-circuit chain; the remaining matches survive as secondary
# labels so that ``route`` can merge them instead of dropping them.
# CATALOG precedes FACT: a program-scoped question ("信管专业要修哪些专业必修
# 课") also matches the 必修/选修 fact pattern, and answering it from a single
# course record instead of the plan was the largest routing error in the
# baseline confusion matrix.  A course-scoped question carries no program cue,
# so it is unaffected.
_INTENT_PATTERNS: tuple[tuple[QueryType, list[str]], ...] = (
    (QueryType.DEPENDENCY, _DEPENDENCY_PATTERNS),
    (QueryType.CATALOG, _CATALOG_PATTERNS),
    (QueryType.FACT, _FACT_PATTERNS),
    (QueryType.STRUCTURE, _STRUCTURE_PATTERNS),
    (QueryType.CONTENT, _CONTENT_PATTERNS),
)

_BOOK_TITLE_PATTERN = re.compile(r"《[^》]*》")


def _mask_book_titles(text: str) -> str:
    """Replace book-title contents before intent regex matching.

    Course names such as 《信息组织基础》 can otherwise lend their own
    substrings to dependency or other intent patterns.  Only the contents are
    masked; the surrounding punctuation remains so phrases outside the title
    keep their original boundaries.
    """
    return _BOOK_TITLE_PATTERN.sub("《》", text)

# Below this confidence a single-label prediction is treated as unreliable and
# falls through to the hybrid path, which retrieves more broadly.
LOW_CONFIDENCE_THRESHOLD = 0.34

# Intent pairs worth answering jointly.  ``content`` is deliberately excluded
# from merging with ``fact``: the content handler already surfaces the basic
# information block, so merging duplicates evidence without adding facts.
_MERGEABLE_INTENTS: frozenset[frozenset[QueryType]] = frozenset(
    {
        frozenset({QueryType.FACT, QueryType.DEPENDENCY}),
        frozenset({QueryType.CONTENT, QueryType.DEPENDENCY}),
        frozenset({QueryType.CATALOG, QueryType.DEPENDENCY}),
        frozenset({QueryType.STRUCTURE, QueryType.DEPENDENCY}),
    }
)

# How much verification backing each NLI outcome carries, worst last.  A
# merged answer reports the highest severity among its sections.
_NLI_STATUS_SEVERITY: dict[str, int] = {
    "passed": 0,
    "rewritten": 1,
    "pruned": 2,
    "skipped": 3,
    "unavailable": 4,
    "refused": 5,
}

_INTENT_HEADINGS: dict[QueryType, str] = {
    QueryType.FACT: "课程基本信息",
    QueryType.STRUCTURE: "教学进度",
    QueryType.DEPENDENCY: "先修关系",
    QueryType.CONTENT: "课程内容",
    QueryType.CATALOG: "培养方案课程",
}


# ---------------------------------------------------------------------------
# 统一判空与字段归一（大创方案 §4 M4.3 / P0-1）
# ---------------------------------------------------------------------------

# 培养方案与大纲用这些占位符表示「没有记录」，必须当作缺失值而不是答案。
_MISSING_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "/",
        "\\",
        "无",
        "-",
        "—",
        "–",
        "－",
        "暂无",
        "无。",
        "n/a",
        "na",
        "none",
        "null",
        "...",
        "…",
    }
)


def _is_missing(value: Any) -> bool:
    """判断字段是否为缺失占位符。

    ``None``、空白、``/``、``无``、``-``、``—``、``暂无`` 都算缺失；列表 /
    元组 / 集合按「无有效元素」判定，字典按空字典判定。改造前只排除 ``None``
    和 ``""``，于是培养方案里的 ``/`` 被当成答案，出现「课程负责人为 /」
    （§2 D6 / P0-1）。
    """
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return not any(not _is_missing(item) for item in value)
    if isinstance(value, dict):
        return not value
    return str(value).strip() in _MISSING_TOKENS


def _clean_names(value: Any) -> list[str]:
    """把「张三、李四」这类教师字段拆成姓名列表，并丢掉占位符。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        raw = [str(item) for item in value]
    else:
        raw = re.split(r"[、,，;；/\s]+", str(value))
    names: list[str] = []
    for item in raw:
        name = str(item).strip().strip("。；;：:")
        if name and not _is_missing(name) and name not in names:
            names.append(name)
    return names


def _coalesce(*values: Any) -> Any:
    """返回第一个非缺失值；全部缺失时返回 ``None``。"""
    for value in values:
        if not _is_missing(value):
            return value
    return None


def _safe_tree_call(
    tree: Any,
    method: str,
    *args: Any,
    default: Any = None,
    **kwargs: Any,
) -> Any:
    """安全调用 ``DocTree`` 方法。

    ``doc_tree.json`` 可能不存在、``DocTree`` 也可能还没落地（P1-3 由另一个
    agent 并行实现）。任何异常都退化成 ``default``，保证改造前的行为不变。
    """
    if tree is None:
        return default
    function = getattr(tree, method, None)
    if not callable(function):
        return default
    try:
        return function(*args, **kwargs)
    except Exception:
        return default


def _generate_with_route(
    query: str,
    evidence_items: list[dict[str, Any]],
    llm_client: Any,
    persona: Persona,
    route_type: str,
) -> tuple[str, dict[str, Any]]:
    """按冻结签名调用生成器，并把路由名透传下去（P0-4 / M6.5）。

    ``route_type`` 是另一个 agent 在 ``generator.py`` 上新增的可选参数；对方
    的改动落地前先探测一次签名，避免本文件改完、集成还没完成时把已有的
    content / hybrid 路径打挂。
    """
    import inspect

    from src.online_service.generator import generate_answer_with_verification

    try:
        signature = inspect.signature(generate_answer_with_verification)
    except (TypeError, ValueError):
        return generate_answer_with_verification(query, evidence_items, llm_client, persona)
    if "route_type" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return generate_answer_with_verification(
            query,
            evidence_items,
            llm_client,
            persona,
            route_type=route_type,
        )
    return generate_answer_with_verification(query, evidence_items, llm_client, persona)


# ---------------------------------------------------------------------------
# fact 路由的字段映射（大创方案 §4 M4.3 / P1-7）
# ---------------------------------------------------------------------------

# 顺序即输出顺序；先命中的字段先答。
_FACT_FIELD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("credits", r"学分"),
    ("hours", r"学时|课时"),
    ("semester", r"学期"),
    (
        "instructor",
        r"任课教师|课程负责人|主讲教师|授课教师|谁负责|负责人|谁教|哪个老师|哪位老师|老师是谁",
    ),
    ("assessment_mode", _ASSESSMENT_MODE_PATTERN),
    (
        "grading_mode",
        r"成绩录入|成绩构成|成绩评定|成绩记载|评分方式|成绩比例|平时成绩占比",
    ),
    ("grade", r"授课年级|开课年级|面向年级"),
    ("audience", r"面向专业|面向对象|授课对象|适用专业"),
    ("department", r"开课单位|开课学院|开课院系|开课部门"),
    ("textbook", r"教材|参考书|参考文献|指定用书"),
)

_FACT_FIELD_LABELS: dict[str, str] = {
    "credits": "学分",
    "hours": "总学时",
    "semester": "开课学期",
    "instructor": "任课教师",
    "assessment_mode": "考核方式",
    "grading_mode": "成绩录入方式",
    "grade": "授课年级",
    "audience": "面向专业",
    "department": "开课单位",
    "textbook": "教材",
}

# 字段 → 大纲 section_type：课程已识别但结构化字段缺失时，用它在本课程范围内
# 做定向检索，而不是退回全库（§2 D7 / P0-2）。
_FACT_SECTION_TYPES: dict[str, tuple[str, ...]] = {
    "credits": ("basic_info",),
    "hours": ("basic_info",),
    "semester": ("basic_info",),
    "instructor": ("basic_info", "assessment"),
    "assessment_mode": ("assessment",),
    "grading_mode": ("assessment",),
    "grade": ("basic_info",),
    "audience": ("basic_info",),
    "department": ("basic_info",),
    "textbook": ("textbook", "references"),
}

# 「任课教师 / 谁教」与「课程负责人 / 谁负责」是两类问法（M4.3 第 3 条表格）。
_INSTRUCTOR_TEACHING_PATTERN = (
    r"任课教师|主讲教师|授课教师|谁教|哪个老师|哪位老师|老师是谁"
)
_INSTRUCTOR_OWNER_PATTERN = r"课程负责人|负责人|谁负责"

# 两源都没有记录时的固定文案：按拒答计，不自己编。
_NO_INSTRUCTOR_RECORDED = "培养方案与教学大纲均未记录该课程的任课教师"
_COURSE_NOT_FOUND_TEMPLATE = "培养方案中没有名为《{name}》的课程。"
_COURSE_CLARIFY_PREFIX = "你是不是想问"
_UNRECORDED_FIELD_TEMPLATE = "培养方案与教学大纲均未记录《{name}》的{field}。"


_COURSE_NAME_PUNCTUATION = re.compile(
    r"[\s\u3000·・_\-—–:：/\\（）()【】\[\]《》〈〉“”\"'、，,。.；;]+"
)
_COURSE_SOURCE_SUFFIX = re.compile(r"转自\s*doc", re.IGNORECASE)


def _normalize_course_name(value: Any) -> str:
    """Normalize a course name/alias for exact and character-level matching."""
    text = str(value or "").strip().casefold()
    text = _COURSE_SOURCE_SUFFIX.sub("", text)
    return _COURSE_NAME_PUNCTUATION.sub("", text)


def _course_display_name(course: dict[str, Any]) -> str:
    """Return a user-facing course name without pipeline source suffixes."""
    name = str(course.get("course_name") or course.get("course_code") or "").strip()
    name = re.sub(r"\s*转自\s*doc$", "", name, flags=re.IGNORECASE).strip()
    return name or str(course.get("course_code") or "").strip()


def _course_alias_values(course: dict[str, Any]) -> list[str]:
    aliases = course.get("aliases") or course.get("course_aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    values = [
        course.get("course_code"),
        course.get("course_name"),
        course.get("english_name"),
        *aliases,
    ]
    return [str(value).strip() for value in values if str(value or "").strip()]


def _course_similarity_values(course: dict[str, Any]) -> list[str]:
    """Aliases used for fuzzy matching; bare course codes score poorly and are excluded."""
    aliases = course.get("aliases") or course.get("course_aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    values = [course.get("course_name"), course.get("english_name"), *aliases]
    return [str(value).strip() for value in values if str(value or "").strip()]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentPrediction:
    """Multi-label intent decision with an evidence-share confidence.

    ``confidence`` is the share of matched patterns owned by ``primary``: a
    query matching only fact patterns scores 1.0, while ``管理运筹学几学分，
    有哪些先修课？`` splits its evidence between ``fact`` and ``dependency``
    and scores below :data:`LOW_CONFIDENCE_THRESHOLD`.  The value is logged so
    that the routing confusion matrix in ``eval/`` can be sliced by it.
    """

    primary: QueryType
    labels: tuple[QueryType, ...]
    confidence: float
    scores: dict[str, int]

    @property
    def is_multi_intent(self) -> bool:
        return len(self.labels) > 1

    def as_metadata(self) -> dict[str, Any]:
        return {
            "intent_primary": self.primary.value,
            "intent_labels": [label.value for label in self.labels],
            "intent_confidence": round(self.confidence, 4),
            "intent_scores": dict(self.scores),
        }


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

    # Class-level defaults so that a bare ``__new__`` router still resolves
    # programs; the index is rebuilt whenever ``courses`` is replaced.
    _program_index: ProgramIndex | None = None
    _program_courses: Any = None

    # 文档树惰性缓存：P1-3 产物 / 模块缺失时为 None，行为与改造前一致。
    _doc_tree_cache: Any = None

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
        # 允许注入 / 热替换的文档树；为 None 时回退到类级惰性加载。
        self._doc_tree_override: Any = None

    # ------------------------------------------------------------------
    # S1/S2 safety gates
    # ------------------------------------------------------------------
    def _course_alias_index(self) -> dict[str, list[dict[str, Any]]]:
        """Build a normalized exact index over course names, aliases and codes."""
        index: dict[str, list[dict[str, Any]]] = {}
        for course in self.courses:
            for value in _course_alias_values(course):
                normalized = _normalize_course_name(value)
                if not normalized:
                    continue
                bucket = index.setdefault(normalized, [])
                if not any(item is course for item in bucket):
                    bucket.append(course)
        return index

    @staticmethod
    def _bracketed_course_names(query: str) -> list[str]:
        return [
            match.strip()
            for match in re.findall(r"《([^《》]+)》", query)
            if match.strip()
        ]

    def _course_from_bracketed_name(self, name: str) -> dict[str, Any] | None:
        matches = self._course_alias_index().get(_normalize_course_name(name), [])
        if not matches:
            return None
        return max(matches, key=lambda course: len(_course_display_name(course)))

    def _course_name_gate(self, query: str) -> RouteResult | None:
        """S1: validate every 《课程名》 before any backend is dispatched."""
        names = self._bracketed_course_names(query)
        if not names or not self.courses:
            return None

        alias_index = self._course_alias_index()
        unknown = next(
            (
                name
                for name in names
                if _normalize_course_name(name) not in alias_index
            ),
            None,
        )
        if unknown is None:
            return None

        normalized_unknown = _normalize_course_name(unknown)
        threshold = config.course_name_similarity_threshold
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for course in self.courses:
            best_score = max(
                (
                    difflib.SequenceMatcher(
                        None,
                        normalized_unknown,
                        _normalize_course_name(value),
                    ).ratio()
                    for value in _course_similarity_values(course)
                ),
                default=0.0,
            )
            if best_score >= threshold:
                scored.append((best_score, _course_display_name(course), course))

        # 同一门课可能同时命中课程名和英文名/别名，按课程身份去重。
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        seen: set[tuple[str, str]] = set()
        for row in sorted(scored, key=lambda item: (-item[0], item[1])):
            _score, display, course = row
            identity = (
                str(course.get("course_code") or ""),
                _normalize_course_name(display),
            )
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(row)
            if len(candidates) == 3:
                break

        if not candidates:
            return RouteResult(
                _COURSE_NOT_FOUND_TEMPLATE.format(name=unknown),
                query_type="refused",
                metadata={
                    "status": "refused",
                    "error_code": "COURSE_NOT_FOUND",
                    "course_name": unknown,
                },
            )

        candidate_names = [display for _score, display, _course in candidates]
        question = "/".join(f"《{name}》" for name in candidate_names)
        return RouteResult(
            f"{_COURSE_CLARIFY_PREFIX}{question}？",
            query_type="clarify",
            metadata={
                "status": "clarify",
                "error_code": "COURSE_AMBIGUOUS",
                "course_name": unknown,
                "candidates": candidate_names,
            },
        )

    def _unrecorded_field_gate(self, query: str) -> RouteResult | None:
        """S2: refuse known courses' never-recorded fields without retrieval."""
        if not self.courses:
            return None
        course = self._link_course(query)
        if course is None:
            return None
        matched_field = next(
            (
                label
                for label, pattern in _UNRECORDED_FIELD_PATTERNS
                if re.search(pattern, query)
            ),
            None,
        )
        if matched_field is None:
            return None
        display_name = _course_display_name(course)
        return RouteResult(
            _UNRECORDED_FIELD_TEMPLATE.format(
                name=display_name,
                field=matched_field,
            ),
            query_type="fact",
            metadata={
                "status": "refused",
                "error_code": "FIELD_NOT_RECORDED",
                "course_code": course.get("course_code"),
                "course_name": display_name,
                "field": matched_field,
                "vector_hits": 0,
                "llm_used": False,
            },
        )

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------
    @staticmethod
    def classify_intent(query: str) -> IntentPrediction:
        """Score every intent group and return a multi-label prediction.

        The regex chain stays as the high-precision, explainable front end;
        what changes is that a query is allowed to carry more than one intent
        and to report how confident the winning one is.
        """
        textual = _mask_book_titles(query.strip().lower())
        scores: dict[str, int] = {}
        matched: list[QueryType] = []
        for query_type, patterns in _INTENT_PATTERNS:
            hits = sum(1 for pattern in patterns if re.search(pattern, textual))
            if hits:
                scores[query_type.value] = hits
                matched.append(query_type)

        total = sum(scores.values())
        if not matched:
            return IntentPrediction(
                primary=QueryType.HYBRID,
                labels=(QueryType.HYBRID,),
                confidence=0.0,
                scores=scores,
            )
        primary = matched[0]
        confidence = scores[primary.value] / total if total else 0.0
        return IntentPrediction(
            primary=primary,
            labels=tuple(matched),
            confidence=confidence,
            scores=scores,
        )

    @classmethod
    def classify(cls, query: str) -> QueryType:
        """Legacy single-label entry point: the winning intent only."""
        return cls.classify_intent(query).primary

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
        intent = self.classify_intent(query)
        safety_gate = self._course_name_gate(query)
        if safety_gate is None:
            safety_gate = self._unrecorded_field_gate(query)
        if safety_gate is not None:
            safety_gate.query_type = intent.primary.value
            safety_gate.metadata.update(
                persona=persona,
                persona_mode=PERSONA_MODE,
                persona_profile_version=PERSONA_PROFILE_VERSION,
                **intent.as_metadata(),
            )
            return safety_gate
        merge_labels = self._mergeable_labels(intent)

        if merge_labels:
            result = await self._handle_multi_intent(
                query,
                merge_labels,
                neo4j_driver=neo4j_driver,
                llm_client=llm_client,
                persona=persona,
            )
        else:
            # Several incompatible intents matched and none dominates: the
            # specialised handlers would each answer half the question, so
            # retrieve broadly instead.
            fallback = (
                intent.is_multi_intent
                and intent.confidence < LOW_CONFIDENCE_THRESHOLD
            )
            result = await self._handle_single_intent(
                QueryType.HYBRID if fallback else intent.primary,
                query,
                neo4j_driver=neo4j_driver,
                llm_client=llm_client,
                persona=persona,
            )
            if fallback:
                result.metadata["low_confidence_fallback"] = True
        result.metadata.update(
            persona=persona,
            persona_mode=PERSONA_MODE,
            persona_profile_version=PERSONA_PROFILE_VERSION,
            **intent.as_metadata(),
        )
        return result

    @staticmethod
    def _mergeable_labels(intent: IntentPrediction) -> tuple[QueryType, ...]:
        """Return the intents to answer jointly, or ``()`` for a single path.

        Only the two strongest labels are merged: answering three intents in
        one response costs a second LLM call for evidence a user rarely asked
        for, and the citation list stops being readable.
        """
        if not intent.is_multi_intent:
            return ()
        pair = intent.labels[:2]
        if frozenset(pair) not in _MERGEABLE_INTENTS:
            return ()
        return pair

    async def _handle_single_intent(
        self,
        query_type: QueryType,
        query: str,
        *,
        neo4j_driver: Any,
        llm_client: Any,
        persona: Persona,
    ) -> RouteResult:
        # ---- Path 1: Dependency reasoning (zsttSystem-native) ----
        if query_type == QueryType.DEPENDENCY:
            return await self._handle_dependency(query, neo4j_driver, llm_client)
        if query_type == QueryType.FACT:
            # ---- Path 2: ChromaDB retrieval ----
            return await self._handle_fact(query, llm_client, persona=persona)
        if query_type == QueryType.STRUCTURE:
            # ---- Path 2b: 文档树确定性回答（M4.1 / M4.2）----
            return await self._handle_structure(
                query,
                llm_client,
                neo4j_driver=neo4j_driver,
                persona=persona,
            )
        if query_type == QueryType.CONTENT:
            return await self._handle_content(
                query,
                llm_client,
                neo4j_driver=neo4j_driver,
                persona=persona,
            )
        if query_type == QueryType.CATALOG:
            return self._handle_catalog(query)
        # ---- Path 3: ChromaDB + Neo4j + grounded generation ----
        return await self._handle_hybrid(
            query,
            llm_client,
            neo4j_driver=neo4j_driver,
            persona=persona,
        )

    async def _handle_multi_intent(
        self,
        query: str,
        labels: tuple[QueryType, ...],
        *,
        neo4j_driver: Any,
        llm_client: Any,
        persona: Persona,
    ) -> RouteResult:
        """Answer a query that carries two intents, one section per intent.

        Each sub-answer keeps its own citations, and a section whose handler
        declined is kept as its own refusal rather than dropped: the user
        asked two things, and silently answering one of them would hide that
        the other has no evidence.

        Per-section metadata is nested under ``<intent>_metadata``, but the
        verification keys are also aggregated to the top level, because that
        is where ``main._log_query`` and the active-learning sampler look for
        them.  The merged answer inherits the *weakest* section's NLI status:
        it is only as verified as its least supported half.
        """
        sections: list[str] = []
        citations: list[dict[str, Any]] = []
        seen_citations: set[str] = set()
        metadata: dict[str, Any] = {}
        dependency_info: dict[str, Any] | None = None
        answered: list[str] = []
        part_metadata: list[dict[str, Any]] = []

        for label in labels:
            part = await self._handle_single_intent(
                label,
                query,
                neo4j_driver=neo4j_driver,
                llm_client=llm_client,
                persona=persona,
            )
            answer = (part.answer or "").strip()
            if not answer:
                continue
            answered.append(label.value)
            heading = _INTENT_HEADINGS.get(label, label.value)
            sections.append(f"【{heading}】\n{answer}")
            for citation in part.citations:
                identity = json.dumps(citation, ensure_ascii=False, sort_keys=True)
                if identity not in seen_citations:
                    seen_citations.add(identity)
                    citations.append(citation)
            metadata[f"{label.value}_metadata"] = part.metadata
            part_metadata.append(part.metadata)
            if part.dependency_info and dependency_info is None:
                dependency_info = part.dependency_info

        if not sections:
            return await self._handle_hybrid(
                query,
                llm_client,
                neo4j_driver=neo4j_driver,
                persona=persona,
            )

        metadata["answered_intents"] = answered
        metadata.update(self._merged_verification(part_metadata))
        return RouteResult(
            answer="\n\n".join(sections),
            citations=citations,
            query_type="+".join(answered),
            metadata=metadata,
            dependency_info=dependency_info,
        )

    @staticmethod
    def _merged_verification(parts: list[dict[str, Any]]) -> dict[str, Any]:
        """Roll each section's verification and status up to the top level."""
        merged: dict[str, Any] = {}
        statuses = [
            str(part["nli_status"]) for part in parts if part.get("nli_status")
        ]
        if statuses:
            merged["nli_status"] = max(
                statuses,
                key=lambda status: _NLI_STATUS_SEVERITY.get(status, 0),
            )
            merged["nli_attempts"] = sum(
                int(part.get("nli_attempts") or 0) for part in parts
            )
            details: list[Any] = []
            for part in parts:
                details.extend(part.get("nli_details") or [])
            merged["nli_details"] = details
            verified = [
                part["nli_verified"]
                for part in parts
                if part.get("nli_verified") is not None
            ]
            merged["nli_verified"] = all(verified) if verified else None

        # A section that degraded or refused must not be hidden behind an
        # overall "ok": the response status reports the worst section.
        degraded = [
            str(part["status"])
            for part in parts
            if part.get("status") and str(part["status"]) != "ok"
        ]
        merged["status"] = degraded[0] if len(degraded) == len(parts) else "ok"
        if degraded and len(degraded) < len(parts):
            merged["partial_status"] = degraded
        return merged

    # ------------------------------------------------------------------
    # Path handlers
    # ------------------------------------------------------------------
    async def _handle_fact(
        self,
        query: str,
        llm_client: Any = None,
        *,
        persona: Persona = DEFAULT_PERSONA,
    ) -> RouteResult:
        """结构化字段优先；字段没有结构化来源时只在本课程内检索（P0-1/P0-2/P1-7）。"""
        course = self._link_course(query)
        if course:
            offering = self._select_offering(course, query)
            requested_fields = [
                field
                for field, pattern in _FACT_FIELD_PATTERNS
                if re.search(pattern, query)
            ]
            if requested_fields:
                structured = self._structured_fact_result(
                    course,
                    offering,
                    requested_fields,
                    query,
                )
                if structured is not None:
                    return structured
            # 课程已识别、字段却没有结构化来源：只在本课程范围内检索（§2 D7）。
            return await self._answer_fact_from_course(
                query,
                course,
                requested_fields,
                llm_client,
                persona,
            )
        hits = await asyncio.to_thread(self.vector_retriever.search, query, 5)
        if not hits:
            return RouteResult("本地知识库暂无可用证据。", query_type="fact", metadata={"status": "degraded", "error_code": "VECTOR_INDEX_UNAVAILABLE"})
        # 课程没识别出来，答案完全依赖 Top-1；若连一个判别性词都不共享，这条
        # chunk 就是无关的——把它当事实返回，正是「不存在的课程被别门课的教材
        # 清单回答」的来源。
        lexical_score = hits[0].get("lexical_score")
        if lexical_score is not None and float(lexical_score) <= 0.0:
            return RouteResult(
                "根据当前知识库，未找到足够相关且可靠的资料来回答这个问题。",
                query_type="fact",
                metadata={
                    "status": "refused",
                    "error_code": "FACT_NO_RELEVANT_EVIDENCE",
                    "vector_hits": len(hits),
                    "llm_used": False,
                },
            )
        return RouteResult(hits[0].get("text", "")[:500], self._vector_citations(hits), "fact", {"vector_hits": len(hits), "graph_nodes": 0, "llm_used": False})

    def _structured_fact_result(
        self,
        course: dict[str, Any],
        offering: dict[str, Any] | None,
        requested_fields: list[str],
        query: str,
    ) -> RouteResult | None:
        """用培养方案 / ``syllabus_profile`` 直接回答，无任何结构化值时返回 ``None``。

        返回 ``None`` 表示「这门课的事实字段没有结构化来源」，调用方会改成限定
        本课程的定向检索，而不是退回全库（§2 D7）。
        """
        profile_value = course.get("syllabus_profile")
        profile: dict[str, Any] = (
            profile_value if isinstance(profile_value, dict) else {}
        )
        facts: list[str] = []
        notes: list[str] = []
        conflicts = self._course_conflicts(course, profile)
        from_profile = False
        for fact_field in requested_fields:
            if fact_field == "semester":
                variants = self._semester_variants(course, query)
                if len(variants) > 1:
                    facts.append(
                        "开课学期各培养方案不一致："
                        + "；".join(
                            f"{program or '未标注方案'} 为 {value}"
                            for program, value in variants
                        )
                    )
                    continue
            if fact_field == "instructor":
                text, refused = self._instructor_answer(
                    course,
                    offering,
                    profile,
                    query,
                )
                if refused:
                    notes.append(text)
                else:
                    facts.append(text)
                continue
            if fact_field == "textbook":
                # 教材通常只存在于大纲块里，交给定向检索。
                continue
            conflict = conflicts.get(fact_field)
            if conflict and not (
                _is_missing(conflict[0]) or _is_missing(conflict[1])
            ):
                # M3.3：培养方案与大纲冲突时不覆盖，两者都给并注明来源。
                facts.append(
                    f"{_FACT_FIELD_LABELS[fact_field]}培养方案为 {conflict[0]}，"
                    f"教学大纲为 {conflict[1]}"
                )
                continue
            value = _coalesce(
                offering.get(fact_field) if offering else None,
                course.get(fact_field),
                profile.get(fact_field),
            )
            if value is None:
                continue
            if not _is_missing(profile.get(fact_field)):
                from_profile = True
            facts.append(f"{_FACT_FIELD_LABELS[fact_field]}为 {value}")
        if not facts:
            if notes:
                # 两源都没有记录：按拒答计（走既有拒答结构，不自己编）。
                return RouteResult(
                    notes[0],
                    query_type="fact",
                    metadata={
                        "status": "refused",
                        "error_code": "FACT_INSTRUCTOR_NOT_RECORDED",
                        "vector_hits": 0,
                        "llm_used": False,
                    },
                )
            return None
        course_label = course.get("course_name") or course.get("course_code")
        answer = f"{course_label}：{'，'.join(facts)}。"
        if notes:
            answer += "；".join(notes)
        return RouteResult(
            answer,
            [self._catalog_citation(course, offering)],
            "fact",
            {
                "vector_hits": 0,
                "graph_nodes": 0,
                "llm_used": False,
                "source_type": (
                    "syllabus_profile" if from_profile else "training_plan"
                ),
            },
        )

    @staticmethod
    def _course_conflicts(
        course: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, tuple[Any, Any]]:
        """收集培养方案与大纲的字段冲突（M3.3）。

        ``run_pipeline`` 把 ``[{"field", "plan_value", "syllabus_value"}]`` 同时写在
        ``course["conflicts"]`` 和 ``course["syllabus_profile"]["conflicts"]``，
        这里两处都读，任一处可用即可。
        """
        merged: dict[str, tuple[Any, Any]] = {}
        for container in (course.get("conflicts"), profile.get("conflicts")):
            if not isinstance(container, list):
                continue
            for item in container:
                if not isinstance(item, dict):
                    continue
                conflict_field = str(item.get("field") or "")
                if not conflict_field or conflict_field in merged:
                    continue
                merged[conflict_field] = (
                    item.get("plan_value"),
                    item.get("syllabus_value"),
                )
        return merged

    def _semester_variants(
        self,
        course: dict[str, Any],
        query: str,
    ) -> list[tuple[str, Any]]:
        """同一门课在多个培养方案里的学期取值（P0-6）。

        问题里带了专业线索时沿用 :meth:`_select_offering` 的选择，不列举；只有
        「没有专业线索且各方案取值不一致」时才列出全部取值，避免 fact-031 那类
        题静默选错学期。
        """
        offerings = list(course.get("offerings") or [])
        if not offerings:
            value = course.get("semester")
            return [] if _is_missing(value) else [("", value)]
        if self.program_index.match_keyword(query):
            return []
        seen: set[tuple[str, str]] = set()
        variants: list[tuple[str, Any]] = []
        for offering in offerings:
            value = offering.get("semester")
            if _is_missing(value):
                continue
            program = str(offering.get("program_name") or "").strip()
            key = (program, str(value))
            if key in seen:
                continue
            seen.add(key)
            variants.append((program, value))
        if len({str(value) for _, value in variants}) <= 1:
            return []
        return variants

    @classmethod
    def _instructor_answer(
        cls,
        course: dict[str, Any],
        offering: dict[str, Any] | None,
        profile: dict[str, Any],
        query: str,
    ) -> tuple[str, bool]:
        """教师回答策略（M4.3 第 3 条表格），返回 ``(文案, 是否未记录)``。"""
        lecturers = _clean_names(profile.get("lecturers")) or _clean_names(
            course.get("lecturers")
        )
        owner = _coalesce(
            profile.get("course_owner"),
            offering.get("instructor") if offering else None,
            course.get("instructor"),
            course.get("course_owner"),
        )
        owner = None if owner is None else str(owner).strip()
        asks_teaching = bool(re.search(_INSTRUCTOR_TEACHING_PATTERN, query))
        asks_owner = bool(re.search(_INSTRUCTOR_OWNER_PATTERN, query))
        if asks_teaching and not asks_owner:
            # 「任课教师 / 谁教 / 哪个老师」
            if lecturers:
                return f"主讲教师为 {'、'.join(lecturers)}", False
            if owner:
                return f"大纲未单列任课教师；课程负责人为 {owner}", False
            return _NO_INSTRUCTOR_RECORDED, True
        # 「课程负责人 / 谁负责」：优先负责人，同时附主讲教师。
        if owner and lecturers:
            return (
                f"课程负责人为 {owner}，主讲教师为 {'、'.join(lecturers)}",
                False,
            )
        if owner:
            return f"课程负责人为 {owner}", False
        if lecturers:
            return f"主讲教师为 {'、'.join(lecturers)}", False
        return _NO_INSTRUCTOR_RECORDED, True

    async def _answer_fact_from_course(
        self,
        query: str,
        course: dict[str, Any],
        requested_fields: list[str],
        llm_client: Any,
        persona: Persona,
    ) -> RouteResult:
        """课程已识别但字段无结构化来源：只在该课程范围内检索并生成（P0-2）。

        改造前这里调用的是不带 ``course_code`` 过滤的全库检索，答案里混进别的
        课程内容（§2 D7）；这里显式限定课程与 section_type，并且不退回全库。
        """
        course_code = str(course.get("course_code") or "").strip()
        preferred: list[str] = []
        for fact_field in requested_fields:
            for section_type in _FACT_SECTION_TYPES.get(fact_field, ()):
                if section_type not in preferred:
                    preferred.append(section_type)
        hits = await asyncio.to_thread(
            self.vector_retriever.search,
            query,
            5,
            course_code=course_code or None,
            source_types=("syllabus",),
            min_score=0.12 if course_code else 0.5,
            preferred_section_types=tuple(preferred) or None,
        )
        if not hits:
            return RouteResult(
                "根据当前知识库，未找到足够相关且可靠的资料来回答这个问题。",
                query_type="fact",
                metadata={
                    "status": "refused",
                    "error_code": "FACT_NO_RELEVANT_EVIDENCE",
                    "course_code": course_code or None,
                    "vector_hits": 0,
                    "llm_used": False,
                },
            )
        doc_tree = self._get_doc_tree()
        evidence_items = self._build_evidence_items(
            query,
            hits,
            [],
            course=course,
            doc_tree=doc_tree,
        )
        from src.online_service.generator import (
            build_fallback_answer,
            select_generation_evidence,
        )

        evidence_items = select_generation_evidence(evidence_items)
        metadata: dict[str, Any] = {
            "vector_hits": len(hits),
            "graph_nodes": 0,
            "llm_used": llm_client is not None,
            "course_code": course_code or None,
            "route_type": "fact",
        }
        try:
            answer, verification_metadata = await asyncio.to_thread(
                _generate_with_route,
                query,
                evidence_items,
                llm_client,
                persona,
                "fact",
            )
            metadata.update(verification_metadata)
        except Exception:
            answer = build_fallback_answer(query, evidence_items, persona)
            metadata.update(status="degraded", error_code="LLM_UNAVAILABLE")
        citations = (
            []
            if metadata.get("nli_status") in {"refused", "unavailable"}
            else self._evidence_citations(evidence_items)
        )
        return RouteResult(answer, citations, "fact", metadata)

    # ------------------------------------------------------------------
    # 结构路由（M4.1 / M4.2）：文档树确定性回答
    # ------------------------------------------------------------------

    _CHAPTER_NO_PATTERN = (
        r"第\s*([一二三四五六七八九十百两0-9]+)\s*(?:章|讲|部分|单元|模块|节)"
    )
    _WEEK_NO_PATTERN = r"第\s*([一二三四五六七八九十百两0-9]+)\s*周"
    _CHAPTER_COUNT_PATTERN = r"几|多少"
    _STRUCTURE_REFUSAL = "根据当前知识库，未找到足够相关且可靠的资料来回答这个问题。"
    _CN_DIGITS: dict[str, int] = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "百": 100,
    }

    @classmethod
    def _parse_number(cls, token: str) -> int | None:
        """把「三」「十一」「12」解析成整数；解析不了返回 ``None``。"""
        token = str(token or "").strip()
        if not token:
            return None
        if token.isdigit():
            return int(token)
        section = 0
        number = 0
        for char in token:
            if char in ("零", "〇"):
                number = 0
                continue
            digit = cls._CN_DIGITS.get(char)
            if digit is None:
                return None
            if digit >= 10:
                section += (number or 1) * digit
                number = 0
            else:
                number = digit
        return (section + number) or None

    @classmethod
    def _extract_number(cls, query: str, pattern: str) -> int | None:
        match = re.search(pattern, query)
        if not match:
            return None
        return cls._parse_number(match.group(1))

    def _get_doc_tree(self) -> Any:
        """返回文档树；不可用时返回 ``None``，调用方需退化成改造前行为。"""
        override = self.__dict__.get("_doc_tree_override")
        if override is not None:
            return override
        return type(self)._load_doc_tree()

    @classmethod
    def _load_doc_tree(cls) -> Any:
        if cls._doc_tree_cache is not None:
            return cls._doc_tree_cache
        try:
            # P1-3 由另一个 agent 并行实现；模块不存在时静默退化为 None。
            from src.online_service.doc_tree import DocTree
        except Exception:
            return None
        try:
            cls._doc_tree_cache = DocTree(config.doc_tree_path)
        except Exception:
            return None
        return cls._doc_tree_cache

    def set_doc_tree(self, tree: Any) -> None:
        """注入 / 热替换文档树（``/admin/data/reindex`` 成功后由 main 调用）。"""
        self._doc_tree_override = tree

    @classmethod
    def reload_doc_tree(cls) -> Any:
        """清空类级缓存并重新加载 ``doc_tree.json``。"""
        cls._doc_tree_cache = None
        return cls._load_doc_tree()

    @classmethod
    def _doc_source(cls, doc_tree: Any, course_code: str) -> str:
        document = _safe_tree_call(doc_tree, "latest_document", course_code)
        source_file = str(getattr(document, "source_file", "") or "")
        return source_file.replace("\\", "/").rsplit("/", 1)[-1]

    @staticmethod
    def _chapter_for_week(doc_tree: Any, outline: list[Any], week_no: int) -> Any:
        """按进度表周次列定位章节；定位不到返回 ``None``。"""
        pattern = re.compile(rf"第\s*0?{week_no}\s*周")
        for node in outline:
            text = QueryRouter._chapter_text(doc_tree, node)
            if pattern.search(text):
                return node
        return None

    @classmethod
    def _chapter_text(cls, doc_tree: Any, node: Any) -> str:
        """取章节正文。

        已交付的 ``ChapterNode`` 只有 ``chunk_ids``，没有 ``text`` 字段，所以正文
        通过 ``parent_block(chunk_id, kind="chapter")`` 取 chapter 父块全文；若
        将来 dataclass 补了 ``text`` 就直接用。
        """
        direct = str(getattr(node, "text", "") or "").strip()
        if direct:
            return direct
        for chunk_id in list(getattr(node, "chunk_ids", None) or [])[:1]:
            block = _safe_tree_call(
                doc_tree, "parent_block", str(chunk_id), kind="chapter"
            )
            text = str(getattr(block, "text", "") or "").strip()
            if text:
                return text
        return ""

    @classmethod
    def _chapter_summary(cls, doc_tree: Any, node: Any, limit: int = 80) -> str:
        text = cls._chapter_text(doc_tree, node)
        title = str(getattr(node, "title", "") or "").strip()
        lines = text.splitlines()
        # 正文首行是标题路径（「（一）教学进度表 / 第1章 …」），摘要里不要重复。
        if lines and " / " in lines[0]:
            text = "\n".join(lines[1:]).strip()
        if title and text.startswith(title):
            text = text[len(title):].lstrip("：: \n")
        # 进度表行是「章节：… 内容：… 学时：…」，目录里只要内容那一栏。
        content = re.search(r"内容[:：]\s*(.*?)\s*(?:学时[:：]|$)", text, re.S)
        if content and content.group(1).strip():
            text = content.group(1)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        return text[:limit] + ("…" if len(text) > limit else "")

    @staticmethod
    def _first_chunk_id(node: Any) -> str | None:
        """Return the node's first usable chunk id, if it has one."""
        for chunk_id in getattr(node, "chunk_ids", None) or []:
            value = str(chunk_id or "").strip()
            if value:
                return value
        return None

    @classmethod
    def _first_outline_chunk_id(cls, outline: list[Any]) -> str | None:
        """Return the first usable chunk id in document-tree node order."""
        for node in outline:
            chunk_id = cls._first_chunk_id(node)
            if chunk_id:
                return chunk_id
        return None

    @classmethod
    def _render_chapter_answer(
        cls,
        course: dict[str, Any],
        doc_tree: Any,
        node: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        """第 N 章的确定性回答：章节序号 + 标题 + 内容 + 学时 + 资料来源。"""
        name = course.get("course_name") or course.get("course_code")
        course_code = str(course.get("course_code") or "")
        title = str(getattr(node, "title", "") or "").strip()
        number = getattr(node, "chapter_no", None)
        if not title:
            title = f"第{number}章" if number is not None else "章节"
        body = cls._chapter_text(doc_tree, node)
        if title and body.startswith(title):
            body = body[len(title):].lstrip("：: \n")
        lines = [f"《{name}》{title}："]
        if body:
            lines.append(body)
        hours = getattr(node, "hours", None)
        if not _is_missing(hours):
            lines.append(f"（共 {hours} 学时）")
        source = cls._doc_source(doc_tree, course_code)
        if source:
            lines.append(f"资料来源：{name} · {title} · {source}")
        citation = {
            "source_file": source or None,
            "course_code": course.get("course_code"),
            "course_name": course.get("course_name"),
            "section": title,
        }
        chunk_id = cls._first_chunk_id(node)
        if chunk_id:
            citation["chunk_id"] = chunk_id
        return "\n".join(lines), [citation]

    @classmethod
    def _render_outline_answer(
        cls,
        course: dict[str, Any],
        doc_tree: Any,
        outline: list[Any],
    ) -> tuple[str, list[dict[str, Any]], int]:
        """整门课教学进度：有序章节目录（含非章节行）+ 资料来源。"""
        name = course.get("course_name") or course.get("course_code")
        course_code = str(course.get("course_code") or "")
        document = _safe_tree_call(doc_tree, "latest_document", course_code)
        non_chapter = [
            str(row) for row in (getattr(document, "non_chapter_rows", None) or [])
        ]
        chapters = [
            node
            for node in outline
            if getattr(node, "chapter_no", None) is not None
        ]
        count = len(chapters) or len(outline)
        head = f"《{name}》教学进度共 {count} 章"
        if non_chapter:
            head += f"（另有{'、'.join(non_chapter[:6])}）"
        lines = [head + "："]
        for index, node in enumerate(outline, 1):
            title = (
                str(getattr(node, "title", "") or "").strip() or f"第{index}部分"
            )
            summary = cls._chapter_summary(doc_tree, node)
            hours = getattr(node, "hours", None)
            tail = f"（{hours} 学时）" if not _is_missing(hours) else ""
            if summary:
                lines.append(f"{index}. {title}——{summary}{tail}")
            else:
                lines.append(f"{index}. {title}{tail}")
        source = cls._doc_source(doc_tree, course_code)
        if source:
            lines.append(f"资料来源：{name} · 教学进度 · {source}")
        citation = {
            "source_file": source or None,
            "course_code": course.get("course_code"),
            "course_name": course.get("course_name"),
            "section": "教学进度",
        }
        chunk_id = cls._first_outline_chunk_id(outline)
        if chunk_id:
            citation["chunk_id"] = chunk_id
        return "\n".join(lines), [citation], count

    async def _structure_fallback(
        self,
        query: str,
        llm_client: Any,
        neo4j_driver: Any,
        persona: Persona,
        reason: str,
    ) -> RouteResult:
        """未识别课程 / 文档树不可用：退回带面包屑的 content 向量检索。"""
        result = await self._handle_content(
            query,
            llm_client,
            neo4j_driver=neo4j_driver,
            persona=persona,
        )
        result.query_type = "structure"
        result.metadata["structure_fallback"] = reason
        return result

    async def _handle_structure(
        self,
        query: str,
        llm_client: Any,
        *,
        neo4j_driver: Any = None,
        persona: Persona = DEFAULT_PERSONA,
    ) -> RouteResult:
        """结构类问题的确定性回答（M4.2）：不调 LLM、不过 NLI，与 fact 一致。"""
        if config.structure_mode != "deterministic":
            # A6 对照臂：不走文档树，退回生成式 content 路径。
            return await self._structure_fallback(
                query, llm_client, neo4j_driver, persona, "ablation_llm_mode"
            )
        course = self._link_course(query)
        doc_tree = self._get_doc_tree()
        if course is None:
            return await self._structure_fallback(
                query, llm_client, neo4j_driver, persona, "no_course"
            )
        if doc_tree is None:
            return await self._structure_fallback(
                query, llm_client, neo4j_driver, persona, "no_doc_tree"
            )
        course_code = str(course.get("course_code") or "")
        outline = list(
            _safe_tree_call(doc_tree, "outline", course_code, default=None) or []
        )
        if not outline:
            return await self._structure_fallback(
                query, llm_client, neo4j_driver, persona, "empty_outline"
            )
        asks_count = bool(re.search(self._CHAPTER_COUNT_PATTERN, query))
        chapter_no = self._extract_number(query, self._CHAPTER_NO_PATTERN)
        week_no = self._extract_number(query, self._WEEK_NO_PATTERN)
        if chapter_no is not None and not asks_count:
            node = _safe_tree_call(
                doc_tree, "chapter", course_code, chapter_no, default=None
            )
            if node is None:
                # 第 N 章不存在 → 既有拒答文案
                return RouteResult(
                    self._STRUCTURE_REFUSAL,
                    query_type="structure",
                    metadata={
                        "status": "refused",
                        "error_code": "STRUCTURE_CHAPTER_NOT_FOUND",
                        "llm_used": False,
                        "chapter_no": chapter_no,
                    },
                )
            answer, citations = self._render_chapter_answer(course, doc_tree, node)
            return RouteResult(
                answer,
                citations,
                "structure",
                {
                    "llm_used": False,
                    "structure_source": "doc_tree",
                    "chapter_no": chapter_no,
                },
            )
        if week_no is not None and not asks_count:
            node = self._chapter_for_week(doc_tree, outline, week_no)
            if node is not None:
                answer, citations = self._render_chapter_answer(
                    course, doc_tree, node
                )
                return RouteResult(
                    answer,
                    citations,
                    "structure",
                    {
                        "llm_used": False,
                        "structure_source": "doc_tree",
                        "week_no": week_no,
                    },
                )
            return await self._structure_fallback(
                query, llm_client, neo4j_driver, persona, "week_not_located"
            )
        answer, citations, chapter_count = self._render_outline_answer(
            course, doc_tree, outline
        )
        return RouteResult(
            answer,
            citations,
            "structure",
            {
                "llm_used": False,
                "structure_source": "doc_tree",
                "chapter_count": chapter_count,
                "chapter_coverage": f"{chapter_count}/{chapter_count}",
            },
        )

    def _link_course(self, query: str) -> dict[str, Any] | None:
        lowered = query.lower()
        # Syllabus-only courses are stored as "X 转自DOC"; users type just "X".
        matches = [
            c
            for c in self.courses
            if str(c.get("course_code", "")).lower() in lowered
            or str(c.get("course_name", "")) in query
            or _course_display_name(c) in query
        ]
        if matches:
            return max(matches, key=lambda c: len(_course_display_name(c)))
        for name in self._bracketed_course_names(query):
            course = self._course_from_bracketed_name(name)
            if course is not None:
                return course
        return None

    @property
    def program_index(self) -> ProgramIndex:
        """Program vocabulary derived from the loaded catalogue.

        Rebuilt when ``courses`` is replaced, which the tests do after
        constructing a router.
        """
        if self._program_index is None or self._program_courses is not self.courses:
            self._program_index = ProgramIndex(
                self.courses,
                aliases=config.program_aliases,
            )
            self._program_courses = self.courses
        return self._program_index

    def _select_offering(
        self,
        course: dict[str, Any],
        query: str,
    ) -> dict[str, Any] | None:
        offerings = list(course.get("offerings") or [])
        if not offerings:
            return None
        index = self.program_index
        program_keyword = index.match_keyword(query)
        candidates = [
            offering
            for offering in offerings
            if not program_keyword
            or program_keyword in str(offering.get("program_name", ""))
        ] or offerings
        program_type = index.match_type(query, candidates)
        matches = [
            offering
            for offering in candidates
            if not program_type
            or str(offering.get("program_type", "")) == program_type
        ]
        return (matches or candidates)[0]

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

    # Only some plans label a "专业核心课" subcategory.  Where that label is
    # absent the question still has an answer, so the filter widens one step at
    # a time: labelled core courses, then the required major courses, then the
    # whole plan (the micro-minor plans file everything under "辅修课程").  A
    # plan that does label its core courses is unaffected.
    _CATEGORY_FALLBACKS: dict[str, tuple[tuple[str, ...], ...]] = {
        "核心": (("专业必修",), ()),
        "基础": (("专业必修",), ()),
    }

    def _catalog_matches(
        self,
        program_keyword: str,
        program_type: str,
        accepted_categories: tuple[str, ...],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Offerings of one program whose category matches the question."""
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
                if (
                    program_type
                    and str(offering.get("program_type", "")) != program_type
                ):
                    continue
                if accepted_categories and not any(
                    token in category for token in accepted_categories
                ):
                    continue
                matches.append((course, offering))
                break
        return matches

    def _handle_catalog(self, query: str) -> RouteResult:
        index = self.program_index
        program_keyword = index.match_keyword(query)
        # Resolve the plan type once against every offering of that program:
        # an unqualified question means the main plan, and the main plan is
        # whichever type carries the most courses.
        program_type = index.match_type(
            query,
            [
                offering
                for course in self.courses
                for offering in course.get("offerings") or []
                if not program_keyword
                or program_keyword in str(offering.get("program_name", ""))
            ],
        )
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
        accepted_categories = (category_keyword,) if category_keyword else ()
        matches = self._catalog_matches(
            program_keyword,
            program_type,
            accepted_categories,
        )
        for fallback in self._CATEGORY_FALLBACKS.get(category_keyword, ()):
            if matches:
                break
            matches = self._catalog_matches(program_keyword, program_type, fallback)

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
        citations_by_source: dict[str, dict[str, Any]] = {}
        for course, offering in matches:
            citation = self._catalog_citation(course, offering)
            citation["course_code"] = None
            citation["course_name"] = offering.get("program_name")
            citation["section"] = (
                f"{offering.get('course_category', '')}"
                f" / {offering.get('course_subcategory', '')}"
            ).strip(" /")
            source = str(citation.get("source_file", ""))
            if not source:
                continue
            grouped = citations_by_source.get(source)
            if grouped is None:
                citation["course_codes"] = []
                citations_by_source[source] = citation
                grouped = citation
            course_code = str(course.get("course_code") or "").strip()
            if course_code and course_code not in grouped["course_codes"]:
                grouped["course_codes"].append(course_code)
        citations = list(citations_by_source.values())
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

    @classmethod
    def _relevant_excerpt(cls, query: str, text: str, limit: int = 700) -> str:
        """Keep complete, query-relevant passages instead of a fixed prefix."""
        text = text.strip()
        if len(text) <= limit:
            return text
        terms = term_set(query)
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

    # 证据排序兜底顺序（DocTree 不可用时，P0-3）
    _SECTION_TYPE_ORDER: dict[str, int] = {
        "basic_info": 0,
        "overview": 1,
        "course_objectives": 2,
        "teaching_content": 3,
        "teaching_schedule": 4,
        "assessment": 5,
        "textbook": 6,
        "references": 7,
        "prerequisites": 8,
    }

    # 「整门课内容」问法：M6.4 完备性检查的触发条件。
    _WHOLE_COURSE_PATTERN = (
        r"整门课|这门课|课程内容|主要讲|主要学|讲什么|学什么|包括哪些|有哪些内容"
        r"|课程安排|教学安排|教学进度"
    )

    @classmethod
    def _header_chunk_ids(
        cls,
        course: dict[str, Any] | None,
        hits: list[dict[str, Any]],
        doc_tree: Any,
    ) -> set[str]:
        """课程的头部块（``is_header``）集合（M4.4）。

        ``DocTree.header_chunks()`` 可用时以它为准，否则退化成按 ``is_header``
        标量过滤已有命中。未识别课程时不做头部直通。
        """
        if not course:
            return set()
        course_code = str(course.get("course_code") or "")
        ids = list(
            _safe_tree_call(doc_tree, "header_chunks", course_code, default=None)
            or []
        )
        if not ids:
            for hit in hits:
                metadata = hit.get("metadata") or {}
                if hit.get("is_header") or metadata.get("is_header"):
                    chunk_id = str(hit.get("chunk_id") or "")
                    if chunk_id:
                        ids.append(chunk_id)
        return {str(chunk_id) for chunk_id in ids if chunk_id}

    @classmethod
    def _evidence_item_from_hit(
        cls,
        query: str,
        hit: dict[str, Any],
        metadata: dict[str, Any],
        header_ids: set[str],
    ) -> dict[str, Any] | None:
        section = (
            hit.get("section")
            or metadata.get("syllabus_section")
            or "课程资料"
        )
        excerpt = cls._relevant_excerpt(query, str(hit.get("text", "")))
        if not excerpt:
            return None
        chunk_id = str(hit.get("chunk_id") or "")
        chapter_no = hit.get("chapter_no")
        if chapter_no is None:
            chapter_no = metadata.get("chapter_no")
        return {
            "course_code": metadata.get("course_code") or hit.get("course_code"),
            "course_name": metadata.get("course_name")
            or hit.get("course_name"),
            "section": section,
            "source_file": hit.get("source_file"),
            "source_type": metadata.get("source_type")
            or hit.get("source_type"),
            "section_type": hit.get("section_type")
            or metadata.get("section_type"),
            "excerpt": excerpt,
            "chunk_id": chunk_id or None,
            "chapter_no": chapter_no,
            "is_header": bool(chunk_id and chunk_id in header_ids),
        }

    @classmethod
    def _chapter_block(cls, doc_tree: Any, chunk_id: str) -> Any:
        """取包含该 chunk 的 chapter 父块（含全文）。

        同时用于 small-to-big（M6.1）与「chunk 属于第几章」（M6.4 完备性检查）。
        """
        if not chunk_id or doc_tree is None:
            return None
        return _safe_tree_call(doc_tree, "parent_block", chunk_id, kind="chapter")

    @classmethod
    def _parent_excerpt(
        cls,
        query: str,
        text: str,
        title: str,
        limit: int = 1200,
    ) -> str:
        """父块摘要；截取时必须保留章节标题行与章节号（M6.1）。"""
        text = str(text or "").strip()
        header = str(title or "").strip()
        if not header:
            header = next(
                (line.strip() for line in text.splitlines() if line.strip()),
                "",
            )
        body = (
            text
            if len(text) <= limit
            else cls._relevant_excerpt(query, text, limit=limit)
        )
        if header and header not in body:
            body = f"{header}\n{body}" if body else header
        return body

    @classmethod
    def _order_evidence(cls, items: list[dict[str, Any]], doc_tree: Any) -> None:
        """先选后排（M6.2）；A4 消融下 ``EVIDENCE_ORDER=score`` 保持相关度顺序。"""
        if config.evidence_order == "original":
            items.sort(key=cls._evidence_order_key(doc_tree))

    @classmethod
    def _evidence_order_key(cls, doc_tree: Any):
        """先选后排（M6.2）。

        培养方案证据排在同课程大纲证据之前，图谱证据始终最后；同源证据按
        ``DocTree.order_key()``（``(course_code, chunk_index)``）排序，文档树
        不可用时退回 ``(course_code, section_type 序, chunk_part)``。
        """

        def key(item: dict[str, Any]) -> tuple[int, int, str, int, str]:
            source_type = str(item.get("source_type") or "")
            if source_type == "knowledge_graph":
                return (3, 0, "", 0, "")
            plan_first = 0 if source_type == "training_plan" else 1
            course_code = str(item.get("course_code") or "")
            if item.get("coverage_filler"):
                return (
                    plan_first,
                    # 完备性补齐项排在真实证据之后（组内 0=文档树有序证据、
                    # 1=无文档树时的回退序、2=补齐项），否则当章节数超过
                    # 10 条证据预算时，补齐的章节标题会把命中的正文挤出去。
                    2,
                    course_code,
                    int(item.get("chapter_no") or 0),
                    "",
                )
            order = _safe_tree_call(doc_tree, "order_key", item.get("chunk_id"))
            # ``order_key`` 对文档树里没有的 chunk 返回 ``("", -1)``；那说明
            # DocTree 覆盖不到它（培养方案块、或文档树为空），退回 P0-3 顺序。
            if (
                isinstance(order, (tuple, list))
                and len(order) == 2
                and str(order[0])
            ):
                try:
                    chunk_index = int(order[1])
                except (TypeError, ValueError):
                    chunk_index = 0
                return (plan_first, 0, str(order[0]), chunk_index, "")
            section_type = str(item.get("section_type") or "")
            return (
                plan_first,
                1,
                course_code,
                cls._SECTION_TYPE_ORDER.get(section_type, 99),
                str(item.get("section") or ""),
            )

        return key

    @classmethod
    def _build_evidence_items(
        cls,
        query: str,
        hits: list[dict[str, Any]],
        graph_paths: list[dict[str, Any]],
        *,
        course: dict[str, Any] | None = None,
        doc_tree: Any = None,
    ) -> list[dict[str, Any]]:
        """组装证据：头部直通（M4.4）→ small-to-big（M6.1）→ 先选后排（M6.2）。"""
        budget = 4200
        header_ids = cls._header_chunk_ids(course, hits, doc_tree)
        # 头部块固定在场，优先占预算，不参与相关度竞争。
        ordered_hits = sorted(
            hits,
            key=lambda hit: 0
            if str(hit.get("chunk_id") or "") in header_ids
            else 1,
        )
        items: list[dict[str, Any]] = []
        replaced_parents: set[str] = set()

        def used() -> int:
            return sum(len(str(item.get("excerpt", ""))) for item in items)

        for hit in ordered_hits:
            metadata = hit.get("metadata") or {}
            chunk_id = str(hit.get("chunk_id") or "")
            item = cls._evidence_item_from_hit(query, hit, metadata, header_ids)
            if item is None:
                continue
            # A3 消融：``EVIDENCE_SMALL_TO_BIG=off`` 时回到行级证据。
            chapter_block = (
                cls._chapter_block(doc_tree, chunk_id)
                if config.evidence_small_to_big
                else None
            )
            if item.get("chapter_no") is None and chapter_block is not None:
                item["chapter_no"] = getattr(chapter_block, "chapter_no", None)
            parent_id = str(
                hit.get("parent_id")
                or metadata.get("parent_id")
                or getattr(chapter_block, "node_id", "")
                or ""
            )
            if parent_id and parent_id in replaced_parents:
                continue
            parent_text = str(getattr(chapter_block, "text", "") or "").strip()
            parent_title = str(getattr(chapter_block, "title", "") or "").strip()
            if parent_id and parent_text:
                excerpt = cls._parent_excerpt(query, parent_text, parent_title)
                if used() + len(excerpt) <= budget:
                    replaced_parents.add(parent_id)
                    item["excerpt"] = excerpt
                    item["parent_id"] = parent_id
                    if parent_title:
                        item["section"] = parent_title
                    item["parent_replacement"] = True
                    items.append(item)
                    continue
            if used() + len(item["excerpt"]) > budget:
                continue
            items.append(item)
        cls._order_evidence(items, doc_tree)
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

    @classmethod
    def _apply_chapter_coverage(
        cls,
        query: str,
        items: list[dict[str, Any]],
        course: dict[str, Any] | None,
        doc_tree: Any,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """确定性完备性检查（M6.4）。

        问题涉及「整门课内容」且识别到课程时，把文档树里证据没覆盖到的章节
        标题行补进来，并回报 ``chapter_no`` 覆盖率（如 ``"8/8"``）。
        """
        if course is None or doc_tree is None:
            return items, None
        if not re.search(cls._WHOLE_COURSE_PATTERN, query):
            return items, None
        if re.search(r"第\s*[一二三四五六七八九十百0-9]+\s*(章|讲|节)", query):
            return items, None
        course_code = str(course.get("course_code") or "")
        outline = list(
            _safe_tree_call(doc_tree, "outline", course_code, default=None) or []
        )
        expected: list[tuple[int, Any]] = []
        for node in outline:
            try:
                expected.append((int(getattr(node, "chapter_no", None)), node))
            except (TypeError, ValueError):
                continue
        if not expected:
            return items, None
        present: set[int] = set()
        for item in items:
            try:
                present.add(int(item.get("chapter_no")))
            except (TypeError, ValueError):
                continue
        missing = [
            (number, node) for number, node in expected if number not in present
        ]
        document = _safe_tree_call(doc_tree, "latest_document", course_code)
        source_file = getattr(document, "source_file", None)
        for number, node in missing:
            title = str(getattr(node, "title", "") or f"第{number}章")
            hours = getattr(node, "hours", None)
            excerpt = title if _is_missing(hours) else f"{title}（{hours} 学时）"
            items.append(
                {
                    "course_code": course.get("course_code"),
                    "course_name": course.get("course_name"),
                    "section": title,
                    "source_file": source_file,
                    "source_type": "syllabus",
                    "section_type": "teaching_schedule",
                    "excerpt": excerpt,
                    "chunk_id": None,
                    "chapter_no": number,
                    "is_header": False,
                    "coverage_filler": True,
                }
            )
        if missing:
            cls._order_evidence(items, doc_tree)
        coverage = f"{len(expected) - len(missing)}/{len(expected)}"
        return items, coverage

    async def _handle_content(
        self,
        query: str,
        llm_client: Any,
        *,
        neo4j_driver: Any = None,
        persona: Persona = DEFAULT_PERSONA,
    ) -> RouteResult:
        course, hits = await self._retrieve_course_evidence(query, persona)
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
        doc_tree = self._get_doc_tree()
        evidence_items = self._build_evidence_items(
            query,
            hits,
            graph_paths,
            course=course,
            doc_tree=doc_tree,
        )
        evidence_items, chapter_coverage = self._apply_chapter_coverage(
            query, evidence_items, course, doc_tree
        )
        from src.online_service.generator import (
            build_fallback_answer,
            select_generation_evidence,
        )
        evidence_items = select_generation_evidence(evidence_items)

        metadata = {
            "vector_hits": len(hits),
            "graph_nodes": len(graph_paths),
            "llm_used": llm_client is not None,
        }
        if chapter_coverage:
            metadata["chapter_coverage"] = chapter_coverage
        try:
            answer, verification_metadata = await asyncio.to_thread(
                _generate_with_route,
                query,
                evidence_items,
                llm_client,
                persona,
                "content",
            )
            metadata.update(verification_metadata)
        except Exception:
            answer = build_fallback_answer(query, evidence_items, persona)
            metadata.update(status="degraded", error_code="LLM_UNAVAILABLE")
        citations = (
            []
            if metadata.get("nli_status") in {"refused", "unavailable"}
            else self._evidence_citations(evidence_items)
        )
        return RouteResult(answer, citations, "content", metadata)

    async def _handle_hybrid(
        self,
        query: str,
        llm_client: Any,
        neo4j_driver: Any = None,
        persona: Persona = DEFAULT_PERSONA,
        **kwargs: Any,
    ) -> RouteResult:
        course, hits = await self._retrieve_course_evidence(query, persona)
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
        doc_tree = self._get_doc_tree()
        evidence_items = self._build_evidence_items(
            query,
            hits,
            graph_paths,
            course=course,
            doc_tree=doc_tree,
        )
        evidence_items, chapter_coverage = self._apply_chapter_coverage(
            query, evidence_items, course, doc_tree
        )
        from src.online_service.generator import (
            build_fallback_answer,
            select_generation_evidence,
        )
        evidence_items = select_generation_evidence(evidence_items)
        metadata = {
            "vector_hits": len(hits),
            "graph_nodes": len(graph_paths),
            "llm_used": llm_client is not None,
        }
        if chapter_coverage:
            metadata["chapter_coverage"] = chapter_coverage
        try:
            answer, verification_metadata = await asyncio.to_thread(
                _generate_with_route,
                query,
                evidence_items,
                llm_client,
                persona,
                "hybrid",
            )
            metadata.update(verification_metadata)
        except Exception:
            answer = build_fallback_answer(query, evidence_items, persona)
            metadata.update(
                status="degraded",
                error_code="LLM_UNAVAILABLE",
            )
        citations = (
            []
            if metadata.get("nli_status") in {"refused", "unavailable"}
            else self._evidence_citations(evidence_items)
        )
        return RouteResult(
            answer,
            citations,
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

    @staticmethod
    def _evidence_citations(
        evidence_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Project the exact vector/graph evidence supplied to generation."""
        citations: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for item in evidence_items:
            source_file = str(item.get("source_file") or "").replace("\\", "/")
            citation = {
                "source_file": source_file.rsplit("/", 1)[-1].split("#", 1)[0]
                or None,
                "course_code": item.get("course_code"),
                "course_name": item.get("course_name"),
                "section": item.get("section"),
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
                metadata={
                    "backend": "zsttSystem_Neo4j",
                    "status": "degraded",
                    "error_code": "NEO4J_UNAVAILABLE",
                },
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
                metadata={
                    "backend": "zsttSystem_Neo4j",
                    "status": "degraded",
                    "error_code": "DEPENDENCY_QUERY_FAILED",
                    "error": str(exc),
                },
            )

        if dep_result is None:
            return RouteResult(
                answer="未找到与该问题相关的课程依赖关系。",
                query_type="dependency",
                metadata={
                    "backend": "zsttSystem_Neo4j",
                    "status": "degraded",
                    "error_code": "DEPENDENCY_NOT_FOUND",
                },
            )

        nli_metadata = {
            key: dep_result[key]
            for key in (
                "nli_verified",
                "nli_status",
                "nli_details",
                "nli_attempts",
                "nli_verification_target",
                "status",
                "error_code",
            )
            if key in dep_result
        }
        return RouteResult(
            answer=dep_result.get("explanation", ""),
            query_type="dependency",
            dependency_info=dep_result,
            metadata={"backend": "zsttSystem_Neo4j", **nli_metadata},
        )

# ---------------------------------------------------------------------------
# Internal helpers (used by the router)
# ---------------------------------------------------------------------------

def _query_graph_paths(neo4j_driver: Any, query: str) -> list[dict[str, Any]]:
    """Fetch nearby graph paths without blocking the async event loop."""
    cypher = """
    MATCH p=(c:ZSTT_Course)-[*1..2]-(n)
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
