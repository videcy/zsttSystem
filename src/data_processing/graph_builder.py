"""Unified, idempotent graph preparation for the lightweight pipeline."""

from __future__ import annotations

import hashlib
import json
import uuid


def stable_id(label: str, value: str) -> str:
    return hashlib.sha256(f"{label}|{value}".encode("utf-8")).hexdigest()[:24]


def neo4j_properties(values: dict) -> dict:
    """Convert nested values into Neo4j-compatible property values."""
    properties = {}
    for key, value in values.items():
        if isinstance(value, dict) or (
            isinstance(value, list)
            and any(isinstance(item, (dict, list)) for item in value)
        ):
            properties[key] = json.dumps(value, ensure_ascii=False)
        else:
            properties[key] = value
    return properties


def build_graph_records(
    courses: list[dict],
    concepts: list[dict],
    chunks: list[dict],
) -> dict:
    nodes = []
    seen = set()
    for course in courses:
        code = course.get("course_code", "")
        if code and ("Course", code) not in seen:
            nodes.append(
                {
                    "id": stable_id("Course", code),
                    "label": "Course",
                    "course_code": code,
                    **course,
                }
            )
            seen.add(("Course", code))
    for concept in concepts:
        name = concept.get("name", "")
        if name and ("Concept", name) not in seen:
            nodes.append(
                {
                    "id": concept.get("concept_id")
                    or stable_id("Concept", name),
                    "label": "Concept",
                    **concept,
                }
            )
            seen.add(("Concept", name))
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        if cid and ("Chunk", cid) not in seen:
            nodes.append(
                {
                    "id": cid,
                    "label": "Chunk",
                    "chunk_id": cid,
                    "source_file": chunk.get("source_file"),
                }
            )
            seen.add(("Chunk", cid))
    edges = []
    for concept in concepts:
        code = concept.get("course_code")
        if code and concept.get("name"):
            edges.append({"source": code, "target": concept.get("concept_id") or stable_id("Concept", concept["name"]), "type": "TEACHES"})
    for chunk in chunks:
        code = (chunk.get("metadata") or {}).get("course_code") or chunk.get("course_code")
        if code and chunk.get("chunk_id"):
            edges.append({"source": code, "target": chunk["chunk_id"], "type": "CONTAINS"})
    for course in courses:
        target = course.get("course_code", "")
        evidence_by_code = {
            item.get("course_code"): item
            for item in course.get("prerequisite_evidence", []) or []
            if item.get("course_code")
        }
        for prereq in course.get("prerequisites", []) or []:
            evidence = evidence_by_code.get(prereq, {})
            edges.append(
                {
                    "source": prereq,
                    "target": target,
                    "type": "PREREQUISITE_OF",
                    **{
                        key: evidence[key]
                        for key in (
                            "raw_name",
                            "source_file",
                            "section",
                            "source_year",
                            "confidence",
                            "verified",
                        )
                        if key in evidence
                    },
                }
            )
    return {"nodes": nodes, "edges": edges}


def ensure_constraints(session) -> None:
    for label, key in (
        ("Course", "course_code"),
        ("Concept", "concept_id"),
        ("Chunk", "chunk_id"),
    ):
        try:
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_{key}_unique "
                f"IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
            ).consume()
        except Exception:
            pass


def _replace_managed_graph(
    tx,
    courses: list[dict],
    concepts: list[dict],
    chunks: list[dict],
    edges: list[dict],
    build_id: str,
) -> None:
    """Replace the three labels owned by this pipeline as one snapshot."""
    tx.run(
        """
        MATCH (n)
        WHERE n:Course OR n:Concept OR n:Chunk
        DETACH DELETE n
        """
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (n:Course {course_code: row.key})
        SET n += row.props, n.build_id = $build_id
        """,
        rows=[
            {
                "key": course.get("course_code"),
                "props": neo4j_properties(course),
            }
            for course in courses
            if course.get("course_code")
        ],
        build_id=build_id,
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (n:Concept {concept_id: row.key})
        SET n += row.props, n.build_id = $build_id
        """,
        rows=[
            {
                "key": concept.get("concept_id")
                or stable_id("Concept", concept.get("name", "")),
                "props": neo4j_properties(concept),
            }
            for concept in concepts
            if concept.get("name")
        ],
        build_id=build_id,
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (n:Chunk {chunk_id: row.key})
        SET n.text = row.text,
            n.source_file = row.source_file,
            n.build_id = $build_id
        """,
        rows=[
            {
                "key": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
                "source_file": chunk.get("source_file", ""),
            }
            for chunk in chunks
            if chunk.get("chunk_id")
        ],
        build_id=build_id,
    ).consume()

    match_by_type = {
        "PREREQUISITE_OF": (
            "MATCH (a:Course {course_code: row.source}), "
            "(b:Course {course_code: row.target}) "
            "MERGE (a)-[r:PREREQUISITE_OF]->(b) "
            "SET r += row.props, r.build_id = $build_id"
        ),
        "TEACHES": (
            "MATCH (a:Course {course_code: row.source}), "
            "(b:Concept {concept_id: row.target}) "
            "MERGE (a)-[r:TEACHES]->(b) "
            "SET r.build_id = $build_id"
        ),
        "CONTAINS": (
            "MATCH (a:Course {course_code: row.source}), "
            "(b:Chunk {chunk_id: row.target}) "
            "MERGE (a)-[r:CONTAINS]->(b) "
            "SET r.build_id = $build_id"
        ),
    }
    for edge_type, match_clause in match_by_type.items():
        rows = []
        for edge in edges:
            if edge.get("type") != edge_type:
                continue
            properties = {
                key: value
                for key, value in edge.items()
                if key not in {"source", "target", "type"} and value is not None
            }
            rows.append(
                {
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "props": neo4j_properties(properties),
                }
            )
        tx.run(
            f"UNWIND $rows AS row {match_clause}",
            rows=rows,
            build_id=build_id,
        ).consume()


def write_neo4j(
    driver,
    graph: dict,
    courses: list[dict],
    concepts: list[dict],
    chunks: list[dict],
) -> dict:
    """Write one transactionally consistent snapshot of the managed graph."""
    build_id = uuid.uuid4().hex
    with driver.session() as session:
        ensure_constraints(session)
        session.execute_write(
            _replace_managed_graph,
            courses,
            concepts,
            chunks,
            graph["edges"],
            build_id,
        )
    return {
        "build_id": build_id,
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
    }
