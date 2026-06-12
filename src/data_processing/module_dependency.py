"""Build course-level dependency edges from verified concept dependency edges.

Aggregates concept-level FOUNDATION_OF / CONCEPTUAL_BASIS / METHOD_ANALOGY /
TOOL_PREREQ edges into module-level MODULE_DEPENDS_ON edges in Neo4j.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable


class ModuleDependencyBuilder:
    """Aggregate concept dependency edges into course-level module dependencies."""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str | None = None,
        min_concept_edge_count: int = 1,
        min_avg_confidence: float = 0.3,
    ) -> None:
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.min_concept_edge_count = min_concept_edge_count
        self.min_avg_confidence = min_avg_confidence
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password),
        )
        self.graph_enabled = True

    def close(self) -> None:
        self.driver.close()

    def _load_json(self, path: str | Path) -> list[dict[str, Any]]:
        file_path = Path(path)
        if not file_path.exists():
            return []
        with file_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, list) else []

    def _get_concept_kind(self, relation_type: str) -> str:
        if relation_type in ("FOUNDATION_OF", "CONCEPTUAL_BASIS"):
            return "foundational"
        if relation_type in ("METHOD_ANALOGY", "TOOL_PREREQ"):
            return "methodological"
        return "other"

    def build_module_edges_from_files(
        self,
        concept_registry_path: str = "outputs/concept_registry.json",
        concept_edge_path: str = "outputs/concept_verified_edges.json",
    ) -> list[dict[str, Any]]:
        registry = self._load_json(concept_registry_path)
        edges = self._load_json(concept_edge_path)

        if not registry or not edges:
            return []

        concept_courses: dict[str, set[str]] = {}
        for concept in registry:
            if not isinstance(concept, dict):
                continue
            concept_id = str(concept.get("id", ""))
            courses = set(
                str(c).strip() for c in (concept.get("source_courses", []) or [])
                if str(c).strip()
            )
            if concept_id and courses:
                concept_courses[concept_id] = courses

        module_pairs: dict[tuple[str, str], dict[str, Any]] = {}
        for edge in edges:
            if not isinstance(edge, dict) or not edge.get("requires"):
                continue
            source_id = str(edge.get("source_id", ""))
            target_id = str(edge.get("target_id", ""))
            if not source_id or not target_id:
                continue

            source_courses = concept_courses.get(source_id, set())
            target_courses = concept_courses.get(target_id, set())
            if not source_courses or not target_courses:
                continue

            confidence = float(edge.get("confidence", 0.0))
            relation_type = str(edge.get("relation_type", "FOUNDATION_OF"))
            kind = self._get_concept_kind(relation_type)

            for src_course in source_courses:
                for tgt_course in target_courses:
                    if src_course == tgt_course:
                        continue
                    key = (src_course, tgt_course)
                    item = module_pairs.setdefault(key, {
                        "source_course": src_course,
                        "target_course": tgt_course,
                        "edge_count": 0,
                        "total_confidence": 0.0,
                        "max_confidence": 0.0,
                        "concept_pairs": [],
                        "relation_kinds": set(),
                    })
                    item["edge_count"] += 1
                    item["total_confidence"] += confidence
                    item["max_confidence"] = max(item["max_confidence"], confidence)
                    item["relation_kinds"].add(kind)
                    item["concept_pairs"].append({
                        "source_concept": str(edge.get("source_name", "")),
                        "target_concept": str(edge.get("target_name", "")),
                        "relation_type": relation_type,
                        "confidence": confidence,
                    })

        module_edges: list[dict[str, Any]] = []
        for key, item in sorted(module_pairs.items()):
            if item["edge_count"] < self.min_concept_edge_count:
                continue
            avg_confidence = item["total_confidence"] / item["edge_count"]
            if avg_confidence < self.min_avg_confidence:
                continue
            module_edges.append({
                **item,
                "avg_confidence": round(avg_confidence, 6),
                "relation_kinds": sorted(item["relation_kinds"]),
            })

        module_edges.sort(
            key=lambda e: (-e["avg_confidence"], -e["edge_count"],
                          e["source_course"], e["target_course"]),
        )
        return module_edges

    def write_module_edges_to_neo4j(
        self,
        module_edges: list[dict[str, Any]],
        *,
        reset: bool = False,
    ) -> int:
        if not self.graph_enabled or not module_edges:
            return 0

        written = 0
        try:
            with self.driver.session() as session:
                if reset:
                    session.run(
                        "MATCH ()-[r:MODULE_DEPENDS_ON]->() DELETE r"
                    )

                for edge in module_edges:
                    session.run(
                        """
                        MERGE (src:Course {name: $source_course})
                        MERGE (tgt:Course {name: $target_course})
                        MERGE (src)-[r:MODULE_DEPENDS_ON]->(tgt)
                        SET r.edge_count = $edge_count,
                            r.avg_confidence = $avg_confidence,
                            r.max_confidence = $max_confidence,
                            r.relation_kinds = $relation_kinds,
                            r.concept_pairs = $concept_pairs
                        """,
                        source_course=edge["source_course"],
                        target_course=edge["target_course"],
                        edge_count=edge["edge_count"],
                        avg_confidence=edge["avg_confidence"],
                        max_confidence=edge["max_confidence"],
                        relation_kinds=edge["relation_kinds"],
                        concept_pairs=json.dumps(
                            edge["concept_pairs"][:20],
                            ensure_ascii=False,
                        ),
                    )
                    written += 1
        except (AuthError, ServiceUnavailable, Neo4jError):
            self.graph_enabled = False
            return 0

        return written

    def build_module_graph_from_neo4j(self) -> list[dict[str, Any]]:
        """Aggregate concept dependency edges already in Neo4j into module edges."""
        if not self.graph_enabled:
            return []

        relation_types = ["FOUNDATION_OF", "METHOD_ANALOGY",
                          "TOOL_PREREQ", "CONCEPTUAL_BASIS"]
        module_pairs: dict[tuple[str, str], dict[str, Any]] = {}

        try:
            with self.driver.session() as session:
                for rel in relation_types:
                    result = session.run(
                        f"""
                        MATCH (src:Knowledge_Point)-[r:{rel}]->(tgt:Knowledge_Point)
                        WHERE src.concept_id IS NOT NULL
                          AND tgt.concept_id IS NOT NULL
                          AND coalesce(r.requires, true) = true
                        UNWIND coalesce(src.source_courses, []) AS src_course
                        UNWIND coalesce(tgt.source_courses, []) AS tgt_course
                        WHERE src_course <> tgt_course
                        RETURN src_course, tgt_course,
                               count(r) AS edge_count,
                               avg(coalesce(r.confidence, 0.5)) AS avg_conf,
                               max(coalesce(r.confidence, 0.5)) AS max_conf,
                               $rel_type AS relation_type
                        """,
                        rel_type=rel,
                    )
                    for record in result:
                        src = str(record["src_course"])
                        tgt = str(record["tgt_course"])
                        key = (src, tgt)
                        item = module_pairs.setdefault(key, {
                            "source_course": src,
                            "target_course": tgt,
                            "edge_count": 0,
                            "total_confidence": 0.0,
                            "max_confidence": 0.0,
                            "relation_kinds": set(),
                        })
                        cnt = int(record["edge_count"])
                        item["edge_count"] += cnt
                        item["total_confidence"] += float(record["avg_conf"]) * cnt
                        item["max_confidence"] = max(
                            item["max_confidence"], float(record["max_conf"])
                        )
                        item["relation_kinds"].add(
                            self._get_concept_kind(str(record["relation_type"]))
                        )
        except (AuthError, ServiceUnavailable, Neo4jError):
            self.graph_enabled = False
            return []

        module_edges: list[dict[str, Any]] = []
        for key, item in sorted(module_pairs.items()):
            if item["edge_count"] < self.min_concept_edge_count:
                continue
            avg_conf = item["total_confidence"] / item["edge_count"]
            if avg_conf < self.min_avg_confidence:
                continue
            module_edges.append({
                **item,
                "avg_confidence": round(avg_conf, 6),
                "relation_kinds": sorted(item["relation_kinds"]),
            })

        module_edges.sort(
            key=lambda e: (-e["avg_confidence"], -e["edge_count"],
                          e["source_course"], e["target_course"]),
        )
        return module_edges

    def run(
        self,
        concept_registry_path: str = "outputs/concept_registry.json",
        concept_edge_path: str = "outputs/concept_verified_edges.json",
        *,
        use_neo4j_aggregation: bool = False,
        reset: bool = False,
    ) -> dict[str, Any]:
        if use_neo4j_aggregation:
            module_edges = self.build_module_graph_from_neo4j()
        else:
            module_edges = self.build_module_edges_from_files(
                concept_registry_path=concept_registry_path,
                concept_edge_path=concept_edge_path,
            )

        written = self.write_module_edges_to_neo4j(module_edges, reset=reset)

        summary: dict[str, Any] = {
            "module_edge_count": len(module_edges),
            "written_to_neo4j": written,
            "graph_available": self.graph_enabled,
            "source": "neo4j" if use_neo4j_aggregation else "json_files",
        }
        if module_edges:
            top_edges = module_edges[:5]
            summary["top_edges"] = [
                f"{e['source_course']} -> {e['target_course']}"
                f" (edges:{e['edge_count']}, conf:{e['avg_confidence']:.3f})"
                for e in top_edges
            ]

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build course-level module dependency graph from concept edges."
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
        help="Neo4j password.",
    )
    parser.add_argument(
        "--min-edge-count",
        type=int,
        default=1,
        help="Minimum concept edge count to create a module-level edge.",
    )
    parser.add_argument(
        "--min-avg-confidence",
        type=float,
        default=0.3,
        help="Minimum average confidence for a module-level edge.",
    )
    parser.add_argument(
        "--use-neo4j-aggregation",
        action="store_true",
        help="Aggregate edges already in Neo4j instead of reading JSON files.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing MODULE_DEPENDS_ON edges before writing.",
    )
    args = parser.parse_args()

    builder = ModuleDependencyBuilder(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        min_concept_edge_count=args.min_edge_count,
        min_avg_confidence=args.min_avg_confidence,
    )
    try:
        builder.run(
            concept_registry_path=args.concept_registry_path,
            concept_edge_path=args.concept_edge_path,
            use_neo4j_aggregation=args.use_neo4j_aggregation,
            reset=args.reset,
        )
    finally:
        builder.close()


if __name__ == "__main__":
    main()
