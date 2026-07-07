"""Extract teaching entities and relations from chunks into a Neo4j graph."""

from __future__ import annotations

import json
import os
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from openai import BadRequestError
from dotenv import load_dotenv

from src.utils.deepseek_client import create_deepseek_client, generate_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class KnowledgeGraphBuilder:
    """Build a teaching knowledge graph from chunked syllabus data."""

    ALLOWED_ENTITY_TYPES = {
        "Course",
        "Knowledge_Point",
        "Instructor",
        "Textbook",
    }
    ALLOWED_RELATION_TYPES = {
        "COVERS_KNOWLEDGE",
        "TAUGHT_BY",
        "USES_TEXTBOOK",
        "HAS_PREREQUISITE",
        "FOUNDATION_OF",
        "METHOD_ANALOGY",
        "TOOL_PREREQ",
        "CONCEPTUAL_BASIS",
    }

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str | None = None,
        llm_model_name: str = "deepseek-v4-flash",
    ) -> None:
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.llm_model_name = llm_model_name or os.getenv("TEXT_MODEL", "deepseek-v4-flash")
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
        )
        self.llm = None
        self.graph_enabled = True

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def load_chunks(self, json_path: str = "outputs/chunked_data.json") -> list[dict[str, Any]]:
        """Load chunked syllabus data from disk."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunk JSON file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        if not isinstance(chunks, list):
            raise ValueError("Chunk JSON must contain a list of chunk objects.")
        return chunks

    def _load_json_list(self, json_path: str | Path) -> list[dict[str, Any]]:
        """Load a JSON list payload from disk."""
        path = Path(json_path)
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        return payload if isinstance(payload, list) else []

    def _load_existing_records(
        self,
        output_path: str | Path,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Load existing extraction results to support resumable runs."""
        path = Path(output_path)
        if not path.exists() or path.stat().st_size == 0:
            return [], {}

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, list):
            raise ValueError("KG output JSON must contain a list of extraction records.")

        indexed_records: dict[str, dict[str, Any]] = {}
        ordered_records: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id in indexed_records:
                continue
            indexed_records[chunk_id] = item
            ordered_records.append(item)

        return ordered_records, indexed_records

    def _persist_records(
        self,
        output_path: str | Path,
        records: list[dict[str, Any]],
    ) -> None:
        """Persist extraction records incrementally for crash-safe resumes."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _course_display_name(self, metadata: dict[str, Any]) -> str:
        """Create a canonical display name for course nodes and relations."""
        course_code = str(metadata.get("course_code", "")).strip()
        course_name = str(metadata.get("course_name", "")).strip()
        return f"{course_code} {course_name}".strip() or course_name or course_code

    def _extract_json_block(self, text: str) -> str:
        """Extract the first JSON object from an LLM response."""
        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)

        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1)

        raise ValueError("LLM response does not contain a JSON object.")

    def _normalize_entity(self, entity: dict[str, Any]) -> dict[str, Any] | None:
        """Validate and normalize an extracted entity."""
        entity_name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("type", "")).strip()
        properties = entity.get("properties", {})

        if not entity_name or entity_type not in self.ALLOWED_ENTITY_TYPES:
            return None
        if not isinstance(properties, dict):
            properties = {}

        normalized_properties = {
            str(key): value
            for key, value in properties.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        return {
            "name": entity_name,
            "type": entity_type,
            "properties": normalized_properties,
        }

    def _normalize_relation(self, relation: dict[str, Any]) -> dict[str, str] | None:
        """Validate and normalize an extracted relation."""
        source = str(relation.get("source", "")).strip()
        target = str(relation.get("target", "")).strip()
        relation_type = str(relation.get("type", "")).strip()

        if not source or not target or relation_type not in self.ALLOWED_RELATION_TYPES:
            return None
        return {
            "source": source,
            "target": target,
            "type": relation_type,
        }

    def _merge_course_and_prerequisites(
        self,
        entities: list[dict[str, Any]],
        relations: list[dict[str, str]],
        metadata: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Inject the course entity and prerequisite links from metadata."""
        course_name = str(metadata.get("course_name", "")).strip()
        course_code = str(metadata.get("course_code", "")).strip()
        course_display = self._course_display_name(metadata)

        if course_name or course_code:
            if not any(
                entity["type"] == "Course" and entity["name"] == course_display
                for entity in entities
            ):
                entities.append(
                    {
                        "name": course_display,
                        "type": "Course",
                        "properties": {
                            "code": course_code,
                            "course_name": course_name,
                            "credits": metadata.get("credits"),
                        },
                    }
                )

        prerequisites = metadata.get("prerequisites", [])
        if not isinstance(prerequisites, list):
            prerequisites = []

        for prerequisite in prerequisites:
            prerequisite_name = str(prerequisite).strip()
            if not prerequisite_name:
                continue

            if not any(
                entity["type"] == "Course" and entity["name"] == prerequisite_name
                for entity in entities
            ):
                entities.append(
                    {
                        "name": prerequisite_name,
                        "type": "Course",
                        "properties": {},
                    }
                )

            prerequisite_relation = {
                "source": course_display,
                "target": prerequisite_name,
                "type": "HAS_PREREQUISITE",
            }
            if prerequisite_relation not in relations and course_display:
                relations.append(prerequisite_relation)

        return entities, relations

    def llm_extract(self, chunk_text: str, metadata: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Extract teaching entities and relations from a chunk with an LLM."""
        if self.llm is None:
            self.llm = create_deepseek_client()
        prompt = f"""
你是一位教育知识图谱构建专家。请从以下课程大纲的片段中，抽取出教学相关的实体和关系。

课程元数据: {json.dumps(metadata, ensure_ascii=False)}

文本片段: "{chunk_text}"

约束条件:
1. 实体类型必须是 'Course', 'Knowledge_Point', 'Instructor', 'Textbook'。
2. 关系类型必须是 'COVERS_KNOWLEDGE', 'TAUGHT_BY', 'USES_TEXTBOOK', 'HAS_PREREQUISITE'。
3. 'Course' 实体必须使用元数据中的 course_name 和 course_code。
4. 'HAS_PREREQUISITE' 关系可以直接从元数据中的 prerequisites 字段获得，无需从文本中抽取。
5. 只抽取文本中明确出现的信息，不要猜测。
6. 输出必须是严格的 JSON 格式，包含 'entities' 和 'relations' 两个列表。

输出 JSON 示例（按照示例的模板来写）:
{{
  "entities": [
    {{"name": "CS101 计算机科学导论", "type": "Course", "properties": {{"code": "CS101"}}}},
    {{"name": "递归", "type": "Knowledge_Point", "properties": {{}}}},
    {{"name": "张三", "type": "Instructor", "properties": {{}}}},
    {{"name": "C++ Primer", "type": "Textbook", "properties": {{}}}}
  ],
  "relations": [
    {{"source": "CS101 计算机科学导论", "target": "递归", "type": "COVERS_KNOWLEDGE"}},
    {{"source": "CS101 计算机科学导论", "target": "张三", "type": "TAUGHT_BY"}},
    {{"source": "CS101 计算机科学导论", "target": "C++ Primer", "type": "USES_TEXTBOOK"}},
    {{"source": "CS101 计算机科学导论", "target": "MATH101 高等数学", "type": "HAS_PREREQUISITE"}}
  ]
}}
"""
        try:
            parsed = generate_json(
                self.llm,
                self.llm_model_name,
                prompt,
                temperature=0.0,
                max_output_tokens=700,
            )
        except BadRequestError as exc:
            if "模型不存在" in str(exc):
                return self._fallback_extract(chunk_text, metadata)
            raise
        except (ValueError, JSONDecodeError):
            return self._fallback_extract(chunk_text, metadata)

        raw_entities = parsed.get("entities", [])
        raw_relations = parsed.get("relations", [])
        if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
            raise ValueError("LLM output JSON must contain list fields 'entities' and 'relations'.")

        entities: list[dict[str, Any]] = []
        relations: list[dict[str, str]] = []

        for entity in raw_entities:
            if isinstance(entity, dict):
                normalized_entity = self._normalize_entity(entity)
                if normalized_entity is not None:
                    entities.append(normalized_entity)

        for relation in raw_relations:
            if isinstance(relation, dict):
                normalized_relation = self._normalize_relation(relation)
                if normalized_relation is not None:
                    relations.append(normalized_relation)

        entities, relations = self._merge_course_and_prerequisites(
            entities=entities,
            relations=relations,
            metadata=metadata,
        )

        return {
            "entities": entities,
            "relations": relations,
        }

    def _fallback_extract(
        self,
        chunk_text: str,
        metadata: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build a minimal extraction when the configured LLM model is unavailable."""
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, str]] = []

        textbook_patterns = [
            r"教材[:：]\s*([^\n]+)",
            r"主要教材[:：]\s*([^\n]+)",
        ]
        for pattern in textbook_patterns:
            for match in re.findall(pattern, chunk_text):
                textbook = str(match).strip(" ;；。")
                if textbook:
                    entities.append(
                        {
                            "name": textbook,
                            "type": "Textbook",
                            "properties": {},
                        }
                    )

        instructor_patterns = [
            r"任课教师[:：]\s*([^\n]+)",
            r"授课教师[:：]\s*([^\n]+)",
            r"主讲教师[:：]\s*([^\n]+)",
        ]
        for pattern in instructor_patterns:
            for match in re.findall(pattern, chunk_text):
                for name in re.split(r"[、,，/ ]+", str(match).strip()):
                    cleaned = name.strip()
                    if cleaned:
                        entities.append(
                            {
                                "name": cleaned,
                                "type": "Instructor",
                                "properties": {},
                            }
                        )

        knowledge_points = []
        for line in chunk_text.splitlines():
            text = line.strip()
            if not text:
                continue
            if len(text) <= 30 and any(token in text for token in ("第一", "第二", "第", "概论", "基础", "方法", "技术")):
                knowledge_points.append(text)
        for name in knowledge_points[:8]:
            entities.append(
                {
                    "name": name,
                    "type": "Knowledge_Point",
                    "properties": {},
                }
            )

        deduped_entities: list[dict[str, Any]] = []
        seen_entities: set[tuple[str, str]] = set()
        for entity in entities:
            key = (entity["type"], entity["name"])
            if key not in seen_entities:
                seen_entities.add(key)
                deduped_entities.append(entity)
        entities = deduped_entities

        entities, relations = self._merge_course_and_prerequisites(
            entities=entities,
            relations=relations,
            metadata=metadata,
        )

        course_display = self._course_display_name(metadata)
        if course_display:
            for entity in entities:
                if entity["type"] == "Knowledge_Point":
                    relations.append(
                        {
                            "source": course_display,
                            "target": entity["name"],
                            "type": "COVERS_KNOWLEDGE",
                        }
                    )
                elif entity["type"] == "Instructor":
                    relations.append(
                        {
                            "source": course_display,
                            "target": entity["name"],
                            "type": "TAUGHT_BY",
                        }
                    )
                elif entity["type"] == "Textbook":
                    relations.append(
                        {
                            "source": course_display,
                            "target": entity["name"],
                            "type": "USES_TEXTBOOK",
                        }
                    )

        deduped_relations: list[dict[str, str]] = []
        seen_relations: set[tuple[str, str, str]] = set()
        for relation in relations:
            key = (relation["source"], relation["target"], relation["type"])
            if key not in seen_relations:
                seen_relations.add(key)
                deduped_relations.append(relation)

        return {
            "entities": entities,
            "relations": deduped_relations,
        }

    def update_graph(self, extraction: dict[str, list[dict[str, Any]]]) -> None:
        """Write extracted entities and relations into Neo4j."""
        if not self.graph_enabled:
            return

        entity_type_queries = {
            "Course": """
                MERGE (n:Course {name: $name})
                SET n += $properties
            """,
            "Knowledge_Point": """
                MERGE (n:Knowledge_Point {name: $name})
                SET n += $properties
                SET n.source = coalesce(n.source, 'chunk_extraction')
            """,
            "Instructor": """
                MERGE (n:Instructor {name: $name})
                SET n += $properties
            """,
            "Textbook": """
                MERGE (n:Textbook {name: $name})
                SET n += $properties
            """,
        }

        relation_queries = {
            "COVERS_KNOWLEDGE": """
                MATCH (source {name: $source}), (target {name: $target})
                MERGE (source)-[:COVERS_KNOWLEDGE]->(target)
            """,
            "TAUGHT_BY": """
                MATCH (source {name: $source}), (target {name: $target})
                MERGE (source)-[:TAUGHT_BY]->(target)
            """,
            "USES_TEXTBOOK": """
                MATCH (source {name: $source}), (target {name: $target})
                MERGE (source)-[:USES_TEXTBOOK]->(target)
            """,
            "HAS_PREREQUISITE": """
                MATCH (source {name: $source}), (target {name: $target})
                MERGE (source)-[:HAS_PREREQUISITE]->(target)
            """,
            "FOUNDATION_OF": """
                MATCH (source {name: $source}), (target {name: $target})
                MERGE (source)-[:FOUNDATION_OF]->(target)
            """,
            "METHOD_ANALOGY": """
                MATCH (source {name: $source}), (target {name: $target})
                MERGE (source)-[:METHOD_ANALOGY]->(target)
            """,
            "TOOL_PREREQ": """
                MATCH (source {name: $source}), (target {name: $target})
                MERGE (source)-[:TOOL_PREREQ]->(target)
            """,
            "CONCEPTUAL_BASIS": """
                MATCH (source {name: $source}), (target {name: $target})
                MERGE (source)-[:CONCEPTUAL_BASIS]->(target)
            """,
        }

        try:
            with self.driver.session() as session:
                for entity in extraction.get("entities", []):
                    entity_type = entity["type"]
                    query = entity_type_queries.get(entity_type)
                    if query is None:
                        continue
                    session.run(
                        query,
                        name=entity["name"],
                        properties=entity.get("properties", {}),
                    )

                for relation in extraction.get("relations", []):
                    query = relation_queries.get(relation["type"])
                    if query is None:
                        continue
                    session.run(
                        query,
                        source=relation["source"],
                        target=relation["target"],
                    )
        except (AuthError, ServiceUnavailable, Neo4jError):
            self.graph_enabled = False

    def update_concept_dependency_graph(
        self,
        concept_registry: list[dict[str, Any]],
        concept_edges: list[dict[str, Any]],
    ) -> int:
        """Write canonical concept nodes and verified dependency edges into Neo4j."""
        if not self.graph_enabled:
            return 0
        if not concept_registry and not concept_edges:
            return 0

        node_query = """
            MERGE (n:Knowledge_Point {name: $name})
            SET n.concept_id = $concept_id,
                n.canonical_name = $canonical_name,
                n.type = $concept_type,
                n.bloom_level = $bloom_level,
                n.discipline = $discipline,
                n.source_courses = $source_courses,
                n.source_course_codes = $source_course_codes,
                n.source_chapters = $source_chapters,
                n.source = 'concept_dependency'
        """

        _SET_CLAUSE = (
            "SET r.confidence = $confidence, r.reason = $reason, "
            "r.requires = $requires, r.candidate_confidence = $candidate_confidence"
        )
        _MATCH_CLAUSE = (
            "MATCH (source:Knowledge_Point {concept_id: $source_id}) "
            "MATCH (target:Knowledge_Point {concept_id: $target_id}) "
        )
        concept_relation_queries: dict[str, str] = {
            "FOUNDATION_OF": (
                f"{_MATCH_CLAUSE} MERGE (source)-[r:FOUNDATION_OF]->(target) {_SET_CLAUSE}"
            ),
            "METHOD_ANALOGY": (
                f"{_MATCH_CLAUSE} MERGE (source)-[r:METHOD_ANALOGY]->(target) {_SET_CLAUSE}"
            ),
            "TOOL_PREREQ": (
                f"{_MATCH_CLAUSE} MERGE (source)-[r:TOOL_PREREQ]->(target) {_SET_CLAUSE}"
            ),
            "CONCEPTUAL_BASIS": (
                f"{_MATCH_CLAUSE} MERGE (source)-[r:CONCEPTUAL_BASIS]->(target) {_SET_CLAUSE}"
            ),
        }

        written_edges = 0
        try:
            with self.driver.session() as session:
                for concept in concept_registry:
                    session.run(
                        node_query,
                        name=str(concept.get("canonical_name", "")).strip(),
                        concept_id=str(concept.get("id", "")).strip(),
                        canonical_name=str(concept.get("canonical_name", "")).strip(),
                        concept_type=str(concept.get("type", "")).strip(),
                        bloom_level=str(concept.get("bloom_level", "")).strip(),
                        discipline=str(concept.get("discipline", "")).strip(),
                        source_courses=list(concept.get("source_courses", [])),
                        source_course_codes=list(concept.get("source_course_codes", [])),
                        source_chapters=list(concept.get("source_chapters", [])),
                    )

                for edge in concept_edges:
                    if not bool(edge.get("requires", False)):
                        continue
                    relation_type = str(edge.get("relation_type", "")).strip()
                    source_id = str(edge.get("source_id", "")).strip()
                    target_id = str(edge.get("target_id", "")).strip()
                    if not source_id or not target_id:
                        continue

                    query = concept_relation_queries.get(relation_type)
                    if query is None:
                        continue

                    session.run(
                        query,
                        source_id=source_id,
                        target_id=target_id,
                        confidence=float(edge.get("confidence", 0.0)),
                        reason=str(edge.get("reason", "")).strip(),
                        requires=True,
                        candidate_confidence=float(edge.get("candidate_confidence", 0.0)),
                    )
                    written_edges += 1
        except (AuthError, ServiceUnavailable, Neo4jError):
            self.graph_enabled = False
            return 0

        return written_edges

    def reset_concept_dependency_subgraph(self) -> None:
        """Delete only canonical concept dependency nodes/edges managed by this pipeline."""
        if not self.graph_enabled:
            return

        delete_queries = [
            """
                MATCH ()-[r:FOUNDATION_OF|METHOD_ANALOGY|TOOL_PREREQ|CONCEPTUAL_BASIS]->()
                DELETE r
            """,
            """
                MATCH (n:Knowledge_Point)
                WHERE n.concept_id IS NOT NULL
                DETACH DELETE n
            """,
        ]

        try:
            with self.driver.session() as session:
                for query in delete_queries:
                    session.run(query)
        except (AuthError, ServiceUnavailable, Neo4jError):
            self.graph_enabled = False

    def run(
        self,
        json_path: str = "outputs/chunked_data.json",
        output_path: str = "outputs/kg_extracted_data.json",
        concept_registry_path: str = "outputs/concept_registry.json",
        concept_edge_path: str = "outputs/concept_verified_edges.json",
        *,
        resume: bool = True,
        reset_concept_subgraph: bool = False,
    ) -> list[dict[str, Any]]:
        """Extract graph facts from each chunk, persist them, and save extraction logs."""
        chunks = self.load_chunks(json_path=json_path)
        if resume:
            extraction_records, existing_by_chunk_id = self._load_existing_records(output_path)
        else:
            extraction_records, existing_by_chunk_id = [], {}

        if reset_concept_subgraph:
            self.reset_concept_dependency_subgraph()

        total_chunks = len(chunks)
        skipped_count = 0
        processed_count = 0
        _PERSIST_INTERVAL = 50

        try:
            for index, chunk in enumerate(chunks, start=1):
                chunk_id = str(chunk.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                if resume and chunk_id in existing_by_chunk_id:
                    skipped_count += 1
                    if skipped_count == 1 or skipped_count % 50 == 0 or index == total_chunks:
                        print(f"[kg] Resume skip {skipped_count} existing chunks ({index}/{total_chunks}).")
                    continue

                chunk_text = str(chunk.get("text", "")).strip()
                metadata = chunk.get("metadata", {})
                if not chunk_text or not isinstance(metadata, dict):
                    continue

                extraction = self.llm_extract(chunk_text, metadata)
                self.update_graph(extraction)
                record = {
                    "chunk_id": chunk_id,
                    "metadata": metadata,
                    "entities": extraction.get("entities", []),
                    "relations": extraction.get("relations", []),
                }
                extraction_records.append(record)
                existing_by_chunk_id[chunk_id] = record
                processed_count += 1

                if processed_count % _PERSIST_INTERVAL == 0:
                    self._persist_records(output_path, extraction_records)

                print(f"[kg] Processed {index}/{total_chunks} (new: {processed_count}, skipped: {skipped_count}).")
        finally:
            if processed_count > 0:
                self._persist_records(output_path, extraction_records)

        concept_registry = self._load_json_list(concept_registry_path)
        concept_edges = self._load_json_list(concept_edge_path)
        if not concept_registry:
            print(f"[kg] No concept registry found at {concept_registry_path}; skipped concept nodes.")
        if not concept_edges:
            print(f"[kg] No verified concept edges found at {concept_edge_path}; skipped concept dependency edges.")
        written_edges = self.update_concept_dependency_graph(concept_registry, concept_edges)

        print(
            f"Knowledge graph updated from {len(chunks)} chunks and saved to {Path(output_path)}."
        )
        if written_edges:
            print(f"[kg] Wrote {written_edges} verified concept dependency edges to Neo4j.")
        if not self.graph_enabled:
            print("Neo4j is unavailable. Saved local KG extraction results without remote graph writes.")
        return extraction_records


def main() -> None:
    """Provide a simple CLI for graph extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract a teaching knowledge graph from chunked data."
    )
    parser.add_argument(
        "--json-path",
        default="outputs/chunked_data.json",
        help="Path to the chunk JSON file.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/kg_extracted_data.json",
        help="Path to save chunk-level KG extraction results.",
    )
    parser.add_argument(
        "--concept-registry-path",
        default="outputs/concept_registry.json",
        help="Path to canonical concept registry JSON.",
    )
    parser.add_argument(
        "--concept-edge-path",
        default="outputs/concept_verified_edges.json",
        help="Path to verified concept dependency edge JSON.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI.",
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username.",
    )
    parser.add_argument(
        "--neo4j-password",
        default=None,
        help="Neo4j password. Falls back to NEO4J_PASSWORD env var when omitted.",
    )
    parser.add_argument(
        "--llm-model-name",
        default="deepseek-v4-flash",
        help="Instruction-tuned generation model used for extraction.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rebuild KG extraction from scratch instead of resuming from existing output.",
    )
    parser.add_argument(
        "--reset-concept-subgraph",
        action="store_true",
        help="Delete only the canonical concept dependency subgraph before writing new verified concept edges.",
    )
    args = parser.parse_args()

    builder = KnowledgeGraphBuilder(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        llm_model_name=args.llm_model_name,
    )
    try:
        builder.run(
            json_path=args.json_path,
            output_path=args.output_path,
            concept_registry_path=args.concept_registry_path,
            concept_edge_path=args.concept_edge_path,
            resume=not args.no_resume,
            reset_concept_subgraph=args.reset_concept_subgraph,
        )
    finally:
        builder.close()


if __name__ == "__main__":
    main()
