"""Structured explanations for verified concept dependency paths."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.config import config
from src.data_processing.graph_builder import CONCEPT_DEPENDENCY_TYPES
from src.online_service.generator import verify_answer_with_nli
from src.utils.deepseek_client import generate_json_value

logger = logging.getLogger(__name__)


# Read the dependency edges straight from the writer's definition. A second
# hand-maintained copy here had drifted: it listed a REQUIRES type nothing ever
# writes, which Neo4j reported as an unknown relationship type on every query.
# Whether an edge is a prerequisite is the rel.requires property, checked in the
# traversal below, not a relationship type of its own.
# Sorted so the generated Cypher is byte-stable across processes.
DEPENDENCY_RELATION_TYPES = tuple(sorted(CONCEPT_DEPENDENCY_TYPES))


def extract_query_entities(question: str, llm_client: Any) -> dict[str, list[str]]:
    """Extract course and concept mentions from a natural-language question."""
    prompt = (
        "从用户问题中提取明确提到的课程名和概念名。只输出 JSON："
        '{"courses": ["..."], "concepts": ["..."]}。'
        "如果没有对应实体，输出空数组。不要输出额外解释。\n"
        f"用户问题：{question}"
    )
    try:
        payload = generate_json_value(
            llm_client,
            config.text_model,
            prompt,
            temperature=0.0,
            max_output_tokens=200,
        )
    except Exception as exc:
        logger.warning(
            "[dependency_explainer] LLM entity extraction failed, using fallback: %s", exc
        )
        payload = _fallback_extract_query_entities(question)

    if not isinstance(payload, dict):
        payload = _fallback_extract_query_entities(question)

    entities = {
        "courses": _normalize_string_list(payload.get("courses", [])),
        "concepts": _normalize_string_list(payload.get("concepts", [])),
    }
    fallback = _fallback_extract_query_entities(question)
    entities["courses"] = _merge_unique(entities["courses"], fallback["courses"])
    entities["concepts"] = _merge_unique(entities["concepts"], fallback["concepts"])
    return entities


def build_dependency_answer(
    question: str,
    neo4j_session: Any,
    llm_client: Any,
    *,
    max_depth: int = 3,
) -> dict[str, Any] | None:
    """Build a structured dependency answer from the concept dependency subgraph."""
    entities = extract_query_entities(question, llm_client)
    names = _merge_unique(entities["courses"], entities["concepts"])
    # 多实体时按实体分组：问题问的是实体之间的关系，只有「起点属于一组、
    # 终点属于另一组」的路径才算数，否则会把各自内部的路径当成答案。
    groups: list[list[dict[str, Any]]] = []
    if len(names) >= 2:
        for name in names:
            group = locate_target_concepts(
                {
                    "courses": [name] if name in entities["courses"] else [],
                    "concepts": [name] if name in entities["concepts"] else [],
                },
                neo4j_session,
            )
            if group:
                groups.append(group)
        target_nodes = _merge_target_groups(groups)
    else:
        target_nodes = locate_target_concepts(entities, neo4j_session)
    if not target_nodes:
        return None

    paths = retrieve_dependency_paths(target_nodes, neo4j_session, max_depth=max_depth)
    # 有实体没在图谱里定位到时，只有「关系 / 联系」类问法才拒答；
    # 「需要什么基础」类问法保持原行为，避免多抽出的杂项实体误伤。
    multi = len(names) >= 2 and (len(groups) >= 2 or _is_relation_question(question))
    if multi:
        paths = _connecting_paths(paths, groups)
        if not paths:
            return _not_connected_result(entities, names)
    if not paths:
        return None

    prerequisites = aggregate_prerequisites(paths)
    explanation = generate_dependency_explanation(
        question, paths, prerequisites, llm_client, connected=multi
    )
    return {
        "linked_entities": entities,
        "target_concepts": target_nodes,
        "prerequisites": prerequisites,
        "prerequisite_table": explanation.get("prerequisite_table", ""),
        "explanation": explanation.get("explanation", ""),
        "mermaid": explanation.get("mermaid") or build_mermaid_graph(paths),
        "paths": paths,
        "nli_verified": explanation.get("nli_verified"),
        "nli_status": explanation.get("nli_status", "skipped"),
        "nli_details": explanation.get("nli_details", []),
        "nli_attempts": explanation.get("nli_attempts", 0),
        "nli_verification_target": explanation.get(
            "nli_verification_target",
            "returned_answer",
        ),
        **{
            key: explanation[key]
            for key in ("status", "error_code")
            if key in explanation
        },
    }


def locate_target_concepts(
    entities: dict[str, list[str]],
    neo4j_session: Any,
) -> list[dict[str, Any]]:
    """Locate concept nodes from mentioned concepts and courses."""
    targets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for concept in entities.get("concepts", []):
        result = neo4j_session.run(
            """
            MATCH (n:ZSTT_Concept)
            WHERE n.concept_id IS NOT NULL
              AND n.source = 'concept_dependency'
              AND (
                n.name CONTAINS $term
                OR $term CONTAINS n.name
                OR $term IN coalesce(n.aliases, [])
              )
            RETURN n.concept_id AS concept_id,
                   n.name AS name,
                   n.discipline AS discipline,
                   n.bloom_level AS bloom_level,
                   n.source_courses AS source_courses,
                   n.source_course_codes AS source_course_codes,
                   n.source_chapters AS source_chapters,
                   n.source_occurrences AS source_occurrences
            LIMIT 30
            """,
            term=concept,
        )
        _append_unique_nodes(targets, seen_ids, result)

    for course in entities.get("courses", []):
        result = neo4j_session.run(
            """
            MATCH (n:ZSTT_Concept)
            WHERE n.concept_id IS NOT NULL
              AND n.source = 'concept_dependency'
              AND (
                any(c IN coalesce(n.source_courses, [])
                    WHERE c CONTAINS $course OR $course CONTAINS c)
                OR any(code IN coalesce(n.source_course_codes, [])
                       WHERE toUpper(code) = toUpper($course))
              )
            RETURN n.concept_id AS concept_id,
                   n.name AS name,
                   n.discipline AS discipline,
                   n.bloom_level AS bloom_level,
                   n.source_courses AS source_courses,
                   n.source_course_codes AS source_course_codes,
                   n.source_chapters AS source_chapters,
                   n.source_occurrences AS source_occurrences
            LIMIT 120
            """,
            course=course,
        )
        _append_unique_nodes(targets, seen_ids, result)

    return targets


def retrieve_dependency_paths(
    target_nodes: list[dict[str, Any]],
    neo4j_session: Any,
    *,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """Traverse source-to-target dependency edges for the requested targets."""
    target_ids = [node["concept_id"] for node in target_nodes if node.get("concept_id")]
    if not target_ids:
        return []

    relation_filter = "|".join(DEPENDENCY_RELATION_TYPES)
    query = f"""
    MATCH p=(source:ZSTT_Concept)-[rels:{relation_filter}*1..{max_depth}]->(target:ZSTT_Concept)
    WHERE target.concept_id IN $target_ids
      AND source.concept_id IS NOT NULL
      AND target.source = 'concept_dependency'
      AND all(rel IN rels WHERE coalesce(rel.requires, true) = true)
    RETURN
      [node IN nodes(p) | {{
        concept_id: node.concept_id,
        name: node.name,
        discipline: node.discipline,
        bloom_level: node.bloom_level,
        source_courses: node.source_courses,
        source_course_codes: node.source_course_codes,
        source_chapters: node.source_chapters,
        source_occurrences: node.source_occurrences
      }}] AS nodes,
      [rel IN relationships(p) | {{
        type: type(rel),
        confidence: rel.confidence,
        reason: rel.reason
      }}] AS relations,
      reduce(total = 0.0, rel IN rels | total + coalesce(rel.confidence, 0.5)) / size(rels) AS avg_confidence
    ORDER BY avg_confidence DESC
    LIMIT 100
    """
    result = neo4j_session.run(query, target_ids=target_ids)

    paths: list[dict[str, Any]] = []
    for record in result:
        nodes = [dict(node) for node in record["nodes"]]
        relations = [dict(rel) for rel in record["relations"]]
        paths.append(
            {
                "nodes": nodes,
                "relations": relations,
                "avg_confidence": float(record["avg_confidence"]),
                "source": nodes[0] if nodes else {},
                "target": nodes[-1] if nodes else {},
            }
        )
    return paths


def aggregate_prerequisites(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group prerequisite concepts by course and chapter."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        nodes = path.get("nodes", [])
        if len(nodes) < 2:
            continue
        source = nodes[0]
        occurrences = _normalize_source_occurrences(
            source.get("source_occurrences", [])
        )
        if occurrences:
            course_chapter_pairs = [
                (
                    occurrence.get("course")
                    or occurrence.get("course_code")
                    or "未知课程",
                    occurrence.get("chapter") or "未知章节",
                )
                for occurrence in occurrences
            ]
        else:
            courses = _normalize_string_list(source.get("source_courses", [])) or ["未知课程"]
            chapters = _normalize_string_list(source.get("source_chapters", [])) or ["未知章节"]
            course_chapter_pairs = [
                (course, chapter)
                for course in courses
                for chapter in chapters
            ]
        for course, chapter in course_chapter_pairs:
            key = (course, chapter)
            item = grouped.setdefault(
                key,
                {
                    "course": course,
                    "chapter": chapter,
                    "concepts": [],
                    "max_confidence": 0.0,
                },
            )
            concept = {
                "concept_id": source.get("concept_id", ""),
                "name": source.get("name", ""),
                "discipline": source.get("discipline", ""),
                "bloom_level": source.get("bloom_level", ""),
            }
            if concept not in item["concepts"]:
                item["concepts"].append(concept)
            item["max_confidence"] = max(
                float(item["max_confidence"]),
                float(path.get("avg_confidence", 0.0)),
            )

    return sorted(
        grouped.values(),
        key=lambda item: (-float(item["max_confidence"]), item["course"], item["chapter"]),
    )


def generate_dependency_explanation(
    question: str,
    paths: list[dict[str, Any]],
    prerequisites: list[dict[str, Any]],
    llm_client: Any,
    *,
    connected: bool = False,
) -> dict[str, Any]:
    """Generate a grounded Chinese explanation and Mermaid graph from paths.

    ``connected`` 表示 paths 已经过「两端分属不同实体」的过滤（多实体问题）。
    """
    # 问「关系 / 联系」时不能预设关系存在，只让模型陈述路径体现的依赖方向。
    instruction = (
        "请仅根据下列路径说明两者的依赖方向；路径不能说明的关系不要推断。\n"
        if _is_relation_question(question)
        else "请仅使用这些依赖关系，用通俗中文解释为什么必须先学这些先修知识。\n"
    )
    prompt = (
        f"用户问题：{question}\n"
        f"系统检索到的依赖路径：{json.dumps(_compact_paths_for_prompt(paths), ensure_ascii=False)}\n"
        f"先修知识聚合：{json.dumps(prerequisites, ensure_ascii=False)}\n"
        f"{instruction}"
        "只输出 JSON 对象 {\"explanation\": \"...\"}，explanation 不超过200字。"
        "不要输出图、表格或 JSON 以外的内容。"
    )
    payload: dict[str, Any] = {}
    if llm_client is not None:
        try:
            generated = generate_json_value(
                llm_client,
                config.text_model,
                prompt,
                temperature=0.1,
                max_output_tokens=300,
            )
            if isinstance(generated, dict):
                payload = generated
        except Exception as exc:
            logger.warning(
                "[dependency_explainer] LLM explanation generation failed, using fallback: %s", exc
            )

    explanation = str(payload.get("explanation", "")).strip()
    nli_metadata: dict[str, Any] = {
        "nli_verified": None,
        "nli_status": "skipped",
        "nli_details": [],
        "nli_attempts": 0,
        "nli_verification_target": "returned_answer",
    }
    if explanation and llm_client is not None and config.nli_verification_enabled:
        context = json.dumps(
            {
                "paths": _compact_paths_for_prompt(paths),
                "prerequisites": prerequisites,
            },
            ensure_ascii=False,
        )
        verified, details = verify_answer_with_nli(
            explanation,
            context,
            llm_client,
        )
        verified = verified and bool(details) and all(
            item.get("label") == "Entailment" for item in details
        )
        # 句级剪枝：没有矛盾句时，保留通过校验的句子，只剪掉证据不足的那几句。
        retained = [
            str(item.get("sentence", "")).strip()
            for item in details
            if item.get("label") == "Entailment" and str(item.get("sentence", "")).strip()
        ]
        if (
            not verified
            and retained
            and not any(item.get("label") == "Contradiction" for item in details)
        ):
            return {
                "prerequisite_table": _build_prerequisite_table(prerequisites),
                "explanation": "".join(retained),
                "mermaid": build_mermaid_graph(paths),
                "nli_verified": True,
                "nli_status": "pruned",
                "nli_details": details,
                "nli_attempts": 1,
                "nli_verification_target": "retained_claims",
                "status": "degraded",
                "error_code": "NLI_UNSUPPORTED_CLAIMS_REMOVED",
            }
        nli_metadata = {
            "nli_verified": verified,
            "nli_status": "passed" if verified else "fallback",
            "nli_details": details,
            "nli_attempts": 1,
            "nli_verification_target": (
                "returned_answer" if verified else "discarded_generated_answer"
            ),
            **(
                {}
                if verified
                else {
                    "status": "degraded",
                    "error_code": "NLI_VERIFICATION_FAILED",
                }
            ),
        }
        if not verified:
            explanation = ""

    return {
        # These structured views are deterministic projections of verified
        # graph paths; the LLM is only allowed to phrase the explanation.
        "prerequisite_table": _build_prerequisite_table(prerequisites),
        "explanation": explanation
        or (
            _path_summary(paths)
            if connected
            else _fallback_explanation(prerequisites)
        ),
        "mermaid": build_mermaid_graph(paths),
        **nli_metadata,
    }


def build_mermaid_graph(paths: list[dict[str, Any]]) -> str:
    """Convert dependency paths to Mermaid graph TD format."""
    lines = ["graph TD"]
    edges: set[tuple[str, str]] = set()
    labels: dict[str, str] = {}

    for path in paths:
        nodes = path.get("nodes", [])
        if len(nodes) < 2:
            continue
        for node in nodes:
            node_id = _mermaid_node_id(str(node.get("concept_id") or node.get("name", "")))
            labels[node_id] = _mermaid_label(node)
        for left, right in zip(nodes, nodes[1:]):
            left_id = _mermaid_node_id(str(left.get("concept_id") or left.get("name", "")))
            right_id = _mermaid_node_id(str(right.get("concept_id") or right.get("name", "")))
            edges.add((left_id, right_id))

    for left_id, right_id in sorted(edges):
        lines.append(f"    {left_id}[{labels[left_id]}] --> {right_id}[{labels[right_id]}]")
    return "\n".join(lines)


_RELATION_QUESTION = re.compile(r"关系|联系|关联|区别")


def _is_relation_question(question: str) -> bool:
    """问「X 和 Y 有什么关系」而不是「为什么要先学 / 需要什么基础」。"""
    return bool(_RELATION_QUESTION.search(question))


def _merge_target_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Union of per-entity concept groups, unique by concept_id."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for node in group:
            concept_id = str(node.get("concept_id") or "")
            if concept_id and concept_id not in seen:
                seen.add(concept_id)
                merged.append(node)
    return merged


def _connecting_paths(
    paths: list[dict[str, Any]],
    groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Keep paths whose start and end belong to two different entity groups."""
    group_ids = [{str(node.get("concept_id")) for node in group} for group in groups]
    kept = []
    for path in paths:
        source = str((path.get("source") or {}).get("concept_id"))
        target = str((path.get("target") or {}).get("concept_id"))
        if any(
            source in left and target in right
            for i, left in enumerate(group_ids)
            for j, right in enumerate(group_ids)
            if i != j
        ):
            kept.append(path)
    return kept


def _not_connected_result(entities: dict[str, list[str]], names: list[str]) -> dict[str, Any]:
    """Refusal for a multi-entity question whose entities share no dependency path."""
    return {
        "status": "refused",
        "error_code": "DEPENDENCY_NOT_CONNECTED",
        "explanation": (
            f"图谱中没有找到《{names[0]}》与《{names[1]}》之间的依赖路径，"
            "无法说明两者存在关系。"
        ),
        "linked_entities": entities,
        "target_concepts": [],
        "prerequisites": [],
        "prerequisite_table": "",
        "mermaid": "",
        "paths": [],
        "nli_verified": None,
        "nli_status": "skipped",
        "nli_details": [],
        "nli_attempts": 0,
        "nli_verification_target": "returned_answer",
    }


def _path_summary(paths: list[dict[str, Any]]) -> str:
    """Deterministic wording for connected paths when the LLM explanation is dropped."""
    lines = []
    for path in paths[:3]:
        names = " → ".join(str(node.get("name", "")) for node in path.get("nodes", []))
        if names:
            lines.append(names)
    return "图谱中检索到的依赖路径：" + "；".join(lines) + "。"


def _fallback_extract_query_entities(question: str) -> dict[str, list[str]]:
    """Heuristic entity extraction for quoted course names and dependency wording."""
    courses = re.findall(r"《([^》]+)》", question)
    stripped = re.sub(
        r"为什么|为何|需要|依赖|哪些|什么|先修|前置|基础|知识点|知识|概念|"
        r"如何|怎么|关系|关联|掌握|学习|才能|有什么|有哪些|课程",
        " ",
        question,
    )
    concepts = [
        fragment.strip("《》")
        for fragment in re.split(r"[，。？！?、\s]+", stripped)
        if 2 <= len(fragment.strip("《》")) <= 30
        and fragment.strip("《》") not in courses
    ]
    # The remaining phrase may be either a course or a concept. Searching both
    # fields keeps dependency lookup useful when no entity-extraction LLM exists.
    return {
        "courses": _merge_unique(courses, concepts),
        "concepts": concepts,
    }


def _append_unique_nodes(targets: list[dict[str, Any]], seen_ids: set[str], records: Any) -> None:
    """Append Neo4j records as unique concept node dictionaries."""
    for record in records:
        node = {
            "concept_id": record.get("concept_id"),
            "name": record.get("name"),
            "discipline": record.get("discipline"),
            "bloom_level": record.get("bloom_level"),
            "source_courses": list(record.get("source_courses") or []),
            "source_course_codes": list(record.get("source_course_codes") or []),
            "source_chapters": list(record.get("source_chapters") or []),
            "source_occurrences": _normalize_source_occurrences(
                record.get("source_occurrences")
            ),
        }
        concept_id = str(node.get("concept_id") or "")
        if concept_id and concept_id not in seen_ids:
            seen_ids.add(concept_id)
            targets.append(node)


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize string or list input to a clean list[str]."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return [str(item).strip() for item in values if str(item).strip()]


def _normalize_source_occurrences(value: Any) -> list[dict[str, str]]:
    """Decode paired course/chapter provenance stored as a Neo4j JSON property."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    occurrences: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        occurrence = {
            "course": str(item.get("course") or "").strip(),
            "course_code": str(item.get("course_code") or "").strip(),
            "chapter": str(item.get("chapter") or "").strip(),
        }
        key = (
            occurrence["course"],
            occurrence["course_code"],
            occurrence["chapter"],
        )
        if any(key) and key not in seen:
            occurrences.append(occurrence)
            seen.add(key)
    return occurrences


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    """Merge two lists preserving order."""
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


def _compact_paths_for_prompt(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce path payload for LLM context."""
    compact: list[dict[str, Any]] = []
    for path in paths[:20]:
        compact.append(
            {
                "nodes": [
                    {
                        "name": node.get("name", ""),
                        "course": ", ".join(_normalize_string_list(node.get("source_courses", []))),
                        "chapter": ", ".join(_normalize_string_list(node.get("source_chapters", []))),
                    }
                    for node in path.get("nodes", [])
                ],
                "relations": path.get("relations", []),
                "avg_confidence": path.get("avg_confidence", 0.0),
            }
        )
    return compact


def _build_prerequisite_table(prerequisites: list[dict[str, Any]]) -> str:
    """Build a Markdown prerequisite table."""
    rows = ["| 课程 | 章节 | 先修概念 | 置信度 |", "|---|---|---|---|"]
    for item in prerequisites:
        concept_names = "、".join(concept.get("name", "") for concept in item.get("concepts", []))
        rows.append(
            f"| {item.get('course', '')} | {item.get('chapter', '')} | "
            f"{concept_names} | {float(item.get('max_confidence', 0.0)):.2f} |"
        )
    return "\n".join(rows)


def _fallback_explanation(prerequisites: list[dict[str, Any]]) -> str:
    """Build a short deterministic explanation when LLM generation fails."""
    if not prerequisites:
        return "当前图谱中没有检索到足够的先修依赖路径。"
    # 解释被拦下时，至少把图谱里查到的先修知识列出来：这些是图谱事实，不是模型生成。
    parts = []
    for item in prerequisites[:5]:
        names = "、".join(
            concept.get("name", "") for concept in item.get("concepts", []) if concept.get("name")
        )
        if not names:
            continue
        where = f"《{item['course']}》" if item.get("course") else ""
        if item.get("chapter"):
            where += f"{item['chapter']}的"
        parts.append(f"{where}{names}")
    if not parts:
        return "这些先修知识位于已验证的依赖路径上，先理解它们有助于掌握目标课程中的后续概念。"
    return "图谱中检索到的先修知识：" + "；".join(parts) + "。先理解它们有助于掌握目标课程中的后续概念。"


def _mermaid_node_id(value: str) -> str:
    """Create a Mermaid-safe node id."""
    safe = re.sub(r"[^0-9A-Za-z_]", "_", value)
    if not safe or safe[0].isdigit():
        safe = f"N_{safe}"
    return safe


def _mermaid_label(node: dict[str, Any]) -> str:
    """Build Mermaid node label as course plus concept."""
    courses = _normalize_string_list(node.get("source_courses", []))
    course = courses[0] if courses else "未知课程"
    name = str(node.get("name", "")).strip() or "未知概念"
    return f"{course}·{name}"
