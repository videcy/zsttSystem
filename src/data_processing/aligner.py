"""Link vector-store chunks with graph nodes for bidirectional retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

from src.online_service.local_vector_store import LocalVectorCollection


class BimodalAligner:
    """Align ChromaDB chunks with Neo4j graph nodes and run basic consistency checks."""

    def __init__(
        self,
        chroma_db_path: str = "vector_store/",
        collection_name: str = "scholar_collection",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str | None = None,
    ) -> None:
        self.collection = LocalVectorCollection(
            db_path=chroma_db_path,
            name=collection_name,
        )
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password or os.getenv("NEO4J_PASSWORD", "")),
        )
        self.graph_enabled = True

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def _load_json(self, path: str | Path) -> Any:
        """Load JSON content from disk."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _normalize_node_name(self, entity: dict[str, Any]) -> str:
        """Extract a stable node name from an entity-like record."""
        return str(entity.get("name", "")).strip()

    def _extract_linked_kg_nodes(
        self,
        kg_record: dict[str, Any],
        fallback_course_name: str = "",
    ) -> list[str]:
        """Get the list of KG node names referenced by a chunk-level KG extraction result."""
        linked_nodes: list[str] = []

        entities = kg_record.get("entities", [])
        if isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, dict):
                    node_name = self._normalize_node_name(entity)
                    if node_name and node_name not in linked_nodes:
                        linked_nodes.append(node_name)

        relations = kg_record.get("relations", [])
        if isinstance(relations, list):
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                for key in ("source", "target"):
                    node_name = str(relation.get(key, "")).strip()
                    if node_name and node_name not in linked_nodes:
                        linked_nodes.append(node_name)

        if fallback_course_name and fallback_course_name not in linked_nodes:
            linked_nodes.append(fallback_course_name)

        return linked_nodes

    def _build_chunk_lookup(self, chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Create a lookup from chunk_id to chunk payload."""
        lookup: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if chunk_id:
                lookup[chunk_id] = chunk
        return lookup

    def _build_chunk_to_kg_mapping(
        self,
        chunks: list[dict[str, Any]],
        kg_results: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Map each chunk_id to the list of extracted KG node names."""
        chunk_lookup = self._build_chunk_lookup(chunks)
        mapping: dict[str, list[str]] = {}

        for kg_record in kg_results:
            if not isinstance(kg_record, dict):
                continue

            chunk_id = str(kg_record.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id not in chunk_lookup:
                continue

            chunk_metadata = chunk_lookup[chunk_id].get("metadata", {})
            course_code = ""
            course_name = ""
            if isinstance(chunk_metadata, dict):
                course_code = str(chunk_metadata.get("course_code", "")).strip()
                course_name = str(chunk_metadata.get("course_name", "")).strip()
            fallback_course_name = f"{course_code} {course_name}".strip() or course_name or course_code

            mapping[chunk_id] = self._extract_linked_kg_nodes(
                kg_record=kg_record,
                fallback_course_name=fallback_course_name,
            )

        return mapping

    def link_vector_to_kg(
        self,
        chunk_data_path: str = "outputs/chunked_data.json",
        extracted_kg_data_path: str = "outputs/kg_extracted_data.json",
    ) -> dict[str, list[str]]:
        """Attach linked KG node names to the metadata of each vector chunk."""
        chunks = self._load_json(chunk_data_path)
        kg_results = self._load_json(extracted_kg_data_path)

        if not isinstance(chunks, list) or not isinstance(kg_results, list):
            raise ValueError("Chunk data and KG extraction data must both be JSON lists.")

        chunk_to_kg_nodes = self._build_chunk_to_kg_mapping(chunks, kg_results)

        for chunk_id, linked_kg_nodes in chunk_to_kg_nodes.items():
            existing = self.collection.get(ids=[chunk_id], include=["metadatas"])
            existing_metadatas = existing.get("metadatas", [])
            existing_metadata = existing_metadatas[0] if existing_metadatas else {}
            if existing_metadata is None:
                existing_metadata = {}

            merged_metadata = dict(existing_metadata)
            merged_metadata["linked_kg_nodes"] = json.dumps(linked_kg_nodes, ensure_ascii=False)
            self.collection.update(ids=[chunk_id], metadatas=[merged_metadata])

        return chunk_to_kg_nodes

    def link_kg_to_vector(
        self,
        chunk_data_path: str = "outputs/chunked_data.json",
        extracted_kg_data_path: str = "outputs/kg_extracted_data.json",
    ) -> dict[str, list[str]]:
        """Write source chunk IDs back onto KG nodes in Neo4j."""
        if not self.graph_enabled:
            return {}

        chunks = self._load_json(chunk_data_path)
        kg_results = self._load_json(extracted_kg_data_path)

        if not isinstance(chunks, list) or not isinstance(kg_results, list):
            raise ValueError("Chunk data and KG extraction data must both be JSON lists.")

        chunk_to_kg_nodes = self._build_chunk_to_kg_mapping(chunks, kg_results)
        node_to_chunk_ids: dict[str, list[str]] = {}
        for chunk_id, node_names in chunk_to_kg_nodes.items():
            for node_name in node_names:
                node_to_chunk_ids.setdefault(node_name, [])
                if chunk_id not in node_to_chunk_ids[node_name]:
                    node_to_chunk_ids[node_name].append(chunk_id)

        try:
            with self.driver.session() as session:
                for node_name, chunk_ids in node_to_chunk_ids.items():
                    session.run(
                        """
                        MATCH (n {name: $node_name})
                        SET n.source_chunks = $chunk_ids
                        """,
                        node_name=node_name,
                        chunk_ids=chunk_ids,
                    )
        except (AuthError, ServiceUnavailable, Neo4jError):
            self.graph_enabled = False
            return {}

        return node_to_chunk_ids

    def consistency_check(self) -> dict[str, Any]:
        """Check whether DEPENDS_ON relations have textual support in source chunks."""
        report: dict[str, Any] = {
            "checked_relations": 0,
            "warnings": [],
        }
        if not self.graph_enabled:
            report["status"] = "skipped"
            report["reason"] = "Neo4j unavailable during local demo run."
            return report

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (a)-[r:DEPENDS_ON]->(b)
                    RETURN a.name AS source_name,
                           b.name AS target_name,
                           coalesce(r.strength, 0) AS strength,
                           coalesce(a.source_chunks, []) AS source_chunks,
                           coalesce(b.source_chunks, []) AS target_chunks
                    """
                )
                records = list(result)
        except (AuthError, ServiceUnavailable, Neo4jError):
            self.graph_enabled = False
            report["status"] = "skipped"
            report["reason"] = "Neo4j unavailable during local demo run."
            return report

        for record in records:
            report["checked_relations"] += 1
            source_name = str(record["source_name"])
            target_name = str(record["target_name"])
            strength = int(record["strength"])
            source_chunk_ids = list(record["source_chunks"] or [])
            target_chunk_ids = list(record["target_chunks"] or [])
            candidate_ids = list(dict.fromkeys(source_chunk_ids + target_chunk_ids))

            textual_support_found = False
            if candidate_ids:
                chunk_payload = self.collection.get(ids=candidate_ids, include=["documents"])
                documents = chunk_payload.get("documents", []) or []
                for document in documents:
                    text = str(document or "")
                    if source_name in text and target_name in text:
                        textual_support_found = True
                        break

            if strength >= 5 and not textual_support_found:
                report["warnings"].append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "strength": strength,
                        "status": "待人工审核",
                        "reason": "Strong DEPENDS_ON relation lacks direct textual evidence in linked chunks.",
                    }
                )

        return report

    def run(
        self,
        chunk_data_path: str = "outputs/chunked_data.json",
        extracted_kg_data_path: str = "outputs/kg_extracted_data.json",
    ) -> dict[str, Any]:
        """Run vector-to-KG linking, KG-to-vector linking, and consistency checks."""
        chunk_to_kg_nodes = self.link_vector_to_kg(
            chunk_data_path=chunk_data_path,
            extracted_kg_data_path=extracted_kg_data_path,
        )
        node_to_chunk_ids = self.link_kg_to_vector(
            chunk_data_path=chunk_data_path,
            extracted_kg_data_path=extracted_kg_data_path,
        )
        report = self.consistency_check()

        summary = {
            "linked_chunks": len(chunk_to_kg_nodes),
            "linked_nodes": len(node_to_chunk_ids),
            "consistency_report": report,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary


def main() -> None:
    """Provide a simple CLI for bimodal alignment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Align ChromaDB chunks with Neo4j knowledge graph nodes."
    )
    parser.add_argument(
        "--chunk-data-path",
        default="outputs/chunked_data.json",
        help="Path to chunked chunk JSON data.",
    )
    parser.add_argument(
        "--kg-data-path",
        default="outputs/kg_extracted_data.json",
        help="Path to saved KG extraction results.",
    )
    parser.add_argument(
        "--chroma-db-path",
        default="vector_store/",
        help="Path to the persistent ChromaDB directory.",
    )
    parser.add_argument(
        "--collection-name",
        default="scholar_collection",
        help="ChromaDB collection name.",
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
    args = parser.parse_args()

    aligner = BimodalAligner(
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )
    try:
        aligner.run(
            chunk_data_path=args.chunk_data_path,
            extracted_kg_data_path=args.kg_data_path,
        )
    finally:
        aligner.close()


if __name__ == "__main__":
    main()
