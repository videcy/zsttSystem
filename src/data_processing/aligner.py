"""Link parsed chunk metadata to Neo4j knowledge-graph nodes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable


class BimodalAligner:
    """Align chunks with Neo4j graph nodes (Neo4j metadata only)."""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str | None = None,
    ) -> None:
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

    def link_kg_to_chunks(
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

    DEPENDENCY_RELATION_TYPES = [
        "FOUNDATION_OF",
        "METHOD_ANALOGY",
        "TOOL_PREREQ",
        "CONCEPTUAL_BASIS",
    ]

    def _consistency_check_cypher(self) -> str:
        """Build Cypher that checks all actual dependency relation types."""
        conditions = " OR ".join(
            f"type(r) = '{rel}'" for rel in self.DEPENDENCY_RELATION_TYPES
        )
        return f"""
            MATCH (a:Knowledge_Point)-[r]->(b:Knowledge_Point)
            WHERE a.concept_id IS NOT NULL
              AND b.concept_id IS NOT NULL
              AND ({conditions})
            RETURN a.canonical_name AS source_name,
                   b.canonical_name AS target_name,
                   type(r) AS relation_type,
                   coalesce(r.confidence, 0) AS confidence,
                   coalesce(a.source_chunks, []) AS source_chunks,
                   coalesce(b.source_chunks, []) AS target_chunks
        """

    def consistency_check(self) -> dict[str, Any]:
        """Check that knowledge-graph relation metadata is present."""
        report: dict[str, Any] = {
            "checked_relations": 0,
            "warnings": [],
            "status": "skipped",
            "reason": "Relation metadata checked; retrieval evidence lives in ChromaDB.",
        }
        if not self.graph_enabled:
            return report

        try:
            with self.driver.session() as session:
                result = session.run(self._consistency_check_cypher())
                records = list(result)
                report["checked_relations"] = len(records)
        except (AuthError, ServiceUnavailable, Neo4jError):
            self.graph_enabled = False
            return report

        # ChromaDB handles retrieval; skip per-relation text verification here.
        return report

    def align_concept_nodes_to_chunks(
        self,
        chunk_data_path: str = "outputs/chunked_data.json",
        concept_edge_path: str = "outputs/concept_verified_edges.json",
        concept_registry_path: str = "outputs/concept_registry.json",
    ) -> dict[str, list[str]]:
        """Write source_chunk_ids onto concept dependency Knowledge_Point nodes in Neo4j."""
        if not self.graph_enabled:
            return {}

        # Concept files are produced by ConceptNormalizer; if that stage was
        # skipped they won't exist.  Degrade gracefully like kg_builder /
        # module_dependency instead of crashing the whole pipeline.
        if not Path(concept_edge_path).exists():
            print(f"[alignment] No verified concept edges at {concept_edge_path}; "
                  "skipped concept node alignment.")
            return {}
        if not Path(concept_registry_path).exists():
            print(f"[alignment] No concept registry at {concept_registry_path}; "
                  "skipped concept node alignment.")
            return {}

        chunks = self._load_json(chunk_data_path)
        concept_edges = self._load_json(concept_edge_path)
        concept_registry = self._load_json(concept_registry_path)

        if not isinstance(chunks, list):
            return {}

        concept_lookup: dict[str, dict[str, Any]] = {}
        if isinstance(concept_registry, list):
            concept_lookup = {
                str(c.get("id", "")): c
                for c in concept_registry
                if isinstance(c, dict)
            }

        concept_to_chunk_ids: dict[str, set[str]] = {}
        if isinstance(concept_edges, list):
            for edge in concept_edges:
                if not isinstance(edge, dict) or not edge.get("requires"):
                    continue
                for role in ("source_id", "target_id"):
                    concept_id = str(edge.get(role, ""))
                    if not concept_id:
                        continue
                    concept_to_chunk_ids.setdefault(concept_id, set())

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id", ""))
            metadata = chunk.get("metadata", {})
            if not chunk_id or not isinstance(metadata, dict):
                continue
            core_concepts = metadata.get("core_concepts", [])
            if not isinstance(core_concepts, list):
                continue
            for concept in core_concepts:
                if not isinstance(concept, dict):
                    continue
                concept_id = str(concept.get("id", ""))
                if concept_id in concept_to_chunk_ids:
                    concept_to_chunk_ids[concept_id].add(chunk_id)

        node_to_chunk_ids: dict[str, list[str]] = {}
        for concept_id, chunk_ids in concept_to_chunk_ids.items():
            concept = concept_lookup.get(concept_id, {})
            canonical_name = str(concept.get("canonical_name", ""))
            if not canonical_name:
                continue
            sorted_ids = sorted(chunk_ids)
            node_to_chunk_ids[canonical_name] = sorted_ids
            try:
                with self.driver.session() as session:
                    session.run(
                        """
                        MATCH (n:Knowledge_Point {concept_id: $concept_id})
                        SET n.source_chunks = $chunk_ids
                        """,
                        concept_id=concept_id,
                        chunk_ids=sorted_ids,
                    )
            except (AuthError, ServiceUnavailable, Neo4jError):
                pass

        return node_to_chunk_ids

    def run(
        self,
        chunk_data_path: str = "outputs/chunked_data.json",
        extracted_kg_data_path: str = "outputs/kg_extracted_data.json",
        concept_edge_path: str = "outputs/concept_verified_edges.json",
        concept_registry_path: str = "outputs/concept_registry.json",
    ) -> dict[str, Any]:
        """Run KG-to-chunk linking, concept node alignment, and consistency checks."""
        node_to_chunk_ids = self.link_kg_to_chunks(
            chunk_data_path=chunk_data_path,
            extracted_kg_data_path=extracted_kg_data_path,
        )
        concept_alignment = self.align_concept_nodes_to_chunks(
            chunk_data_path=chunk_data_path,
            concept_edge_path=concept_edge_path,
            concept_registry_path=concept_registry_path,
        )
        report = self.consistency_check()

        summary = {
            "linked_nodes": len(node_to_chunk_ids),
            "aligned_concept_nodes": len(concept_alignment),
            "consistency_report": report,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary


def main() -> None:
    """Provide a simple CLI for bimodal alignment (Neo4j metadata only)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Align chunks with Neo4j knowledge graph nodes (Neo4j metadata only)."
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
