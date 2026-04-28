"""Extract teaching entities and relations from chunks into a Neo4j graph."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from openai import BadRequestError

from src.utils.glm_client import create_glm_client, generate_json


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
    }

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str | None = None,
        llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    ) -> None:
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.llm_model_name = llm_model_name or os.getenv("GLM_TEXT_MODEL", "glm-5")
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
        )
        self.llm = create_glm_client()
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

    def run(
        self,
        json_path: str = "outputs/chunked_data.json",
        output_path: str = "outputs/kg_extracted_data.json",
    ) -> list[dict[str, Any]]:
        """Extract graph facts from each chunk, persist them, and save extraction logs."""
        chunks = self.load_chunks(json_path=json_path)
        extraction_records: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_text = str(chunk.get("text", "")).strip()
            metadata = chunk.get("metadata", {})
            if not chunk_text or not isinstance(metadata, dict):
                continue

            extraction = self.llm_extract(chunk_text, metadata)
            self.update_graph(extraction)
            extraction_records.append(
                {
                    "chunk_id": str(chunk.get("chunk_id", "")).strip(),
                    "metadata": metadata,
                    "entities": extraction.get("entities", []),
                    "relations": extraction.get("relations", []),
                }
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(extraction_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Knowledge graph updated from {len(chunks)} chunks and saved to {output_file}."
        )
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
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Instruction-tuned generation model used for extraction.",
    )
    args = parser.parse_args()

    builder = KnowledgeGraphBuilder(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        llm_model_name=args.llm_model_name,
    )
    try:
        builder.run(json_path=args.json_path, output_path=args.output_path)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
