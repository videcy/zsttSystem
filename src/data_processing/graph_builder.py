"""Unified, idempotent graph preparation for the lightweight pipeline."""

from __future__ import annotations

import hashlib
import json
import uuid


CONCEPT_DEPENDENCY_TYPES = {
    "FOUNDATION_OF",
    "METHOD_ANALOGY",
    "TOOL_PREREQ",
    "CONCEPTUAL_BASIS",
}
MANAGED_BY = "zsttSystem"


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


def _would_create_cycle(
    adjacency: dict[str, set[str]],
    source_id: str,
    target_id: str,
) -> bool:
    """Return whether adding source -> target would close a directed cycle."""
    pending = [target_id]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == source_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency.get(current, ()))
    return False


def build_graph_records(
    courses: list[dict],
    concepts: list[dict],
    chunks: list[dict],
    *,
    concept_dependencies: list[dict] | None = None,
    verified_min_confidence: float = 0.6,
    include_chunk_nodes: bool = True,
) -> dict:
    """Assemble the managed node/edge records for one graph build.

    ``include_chunk_nodes`` mirrors the chunk layer (``Course-[:CONTAINS]->
    Chunk-[:MENTIONS]->Concept``) into the graph.  The chunk text itself lives
    in Chroma and no online Cypher path reads these nodes, so the pipeline
    turns them off by default (``GRAPH_INCLUDE_CHUNK_NODES``); when they are
    on, they carry the metadata a graph-side query would need to be useful.
    """
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
    valid_course_codes = {
        str(node["course_code"])
        for node in nodes
        if node.get("label") == "Course" and node.get("course_code")
    }
    for concept in concepts:
        concept_id = concept.get("id") or concept.get("concept_id")
        name = concept.get("canonical_name") or concept.get("name", "")
        if concept_id and name and ("Concept", concept_id) not in seen:
            nodes.append(
                {
                    **concept,
                    "id": concept_id,
                    "label": "Concept",
                    "concept_id": concept_id,
                    "name": name,
                    "source": "concept_dependency",
                }
            )
            seen.add(("Concept", concept_id))
    valid_concept_ids = {
        str(node["concept_id"])
        for node in nodes
        if node.get("label") == "Concept" and node.get("concept_id")
    }
    for chunk in chunks if include_chunk_nodes else ():
        cid = chunk.get("chunk_id")
        if cid and ("Chunk", cid) not in seen:
            chunk_metadata = chunk.get("metadata") or {}
            nodes.append(
                {
                    "id": cid,
                    "label": "Chunk",
                    "chunk_id": cid,
                    "source_file": chunk.get("source_file"),
                    **{
                        key: value
                        for key, value in (
                            ("course_code", chunk_metadata.get("course_code")),
                            ("course_name", chunk_metadata.get("course_name")),
                            ("section", chunk.get("section")),
                            ("section_type", chunk_metadata.get("section_type")),
                            ("source_type", chunk_metadata.get("source_type")),
                            ("source_year", chunk_metadata.get("source_year")),
                        )
                        if value not in (None, "")
                    },
                }
            )
            seen.add(("Chunk", cid))
    edges = []
    for concept in concepts:
        concept_id = concept.get("id") or concept.get("concept_id")
        course_codes = concept.get("source_course_codes") or [concept.get("course_code")]
        for code in dict.fromkeys(str(value).strip() for value in course_codes if value):
            if (
                code in valid_course_codes
                and str(concept_id) in valid_concept_ids
            ):
                edges.append(
                    {"source": code, "target": str(concept_id), "type": "TEACHES"}
                )
    for chunk in chunks if include_chunk_nodes else ():
        code = (chunk.get("metadata") or {}).get("course_code") or chunk.get("course_code")
        if code in valid_course_codes and chunk.get("chunk_id"):
            edges.append({"source": code, "target": chunk["chunk_id"], "type": "CONTAINS"})
        for concept_id in (chunk.get("metadata") or {}).get("core_concept_ids", []) or []:
            if chunk.get("chunk_id") and str(concept_id) in valid_concept_ids:
                edges.append(
                    {
                        "source": chunk["chunk_id"],
                        "target": str(concept_id),
                        "type": "MENTIONS",
                    }
                )
    for course in courses:
        target = course.get("course_code", "")
        evidence_by_code = {
            item.get("course_code"): item
            for item in course.get("prerequisite_evidence", []) or []
            if item.get("course_code")
        }
        for prereq in course.get("prerequisites", []) or []:
            if prereq not in valid_course_codes or target not in valid_course_codes:
                continue
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
    dependency_edges: list[dict] = []
    for dependency in concept_dependencies or []:
        relation_type = str(dependency.get("relation_type", "")).upper()
        verification_source = str(
            dependency.get("verification_source") or ""
        ).casefold()
        if (
            dependency.get("requires") is not True
            or relation_type not in CONCEPT_DEPENDENCY_TYPES
            or verification_source not in {"llm", "human"}
        ):
            continue
        source_id = str(dependency.get("source_id") or "")
        target_id = str(dependency.get("target_id") or "")
        try:
            confidence = float(dependency.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if (
            not source_id
            or not target_id
            or source_id == target_id
            or source_id not in valid_concept_ids
            or target_id not in valid_concept_ids
            or not 0.0 <= confidence <= 1.0
            or confidence < verified_min_confidence
        ):
            continue
        dependency_edges.append(
            {
                "source": source_id,
                "target": target_id,
                "type": relation_type,
                **{
                    key: dependency[key]
                    for key in (
                        "requires",
                        "confidence",
                        "reason",
                        "candidate_confidence",
                        "fusion_score",
                        "evidence",
                        "verification_source",
                    )
                    if key in dependency
                },
            }
        )
    dependency_edges.sort(
        key=lambda edge: (
            -float(edge.get("confidence", 0.0)),
            edge["source"],
            edge["target"],
            edge["type"],
        )
    )
    adjacency: dict[str, set[str]] = {}
    accepted_pairs: set[tuple[str, str]] = set()
    for edge in dependency_edges:
        pair = (edge["source"], edge["target"])
        if pair in accepted_pairs or _would_create_cycle(
            adjacency,
            edge["source"],
            edge["target"],
        ):
            continue
        edges.append(edge)
        accepted_pairs.add(pair)
        adjacency.setdefault(edge["source"], set()).add(edge["target"])
    return {"nodes": nodes, "edges": edges}


def ensure_constraints(session) -> None:
    for label, key in (
        ("ZSTT_Course", "course_code"),
        ("ZSTT_Concept", "concept_id"),
        ("ZSTT_Chunk", "chunk_id"),
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
    reset_existing: bool = False,
) -> None:
    """Replace the three labels owned by this pipeline as one snapshot."""
    if reset_existing:
        tx.run(
            """
            MATCH (n)
            WHERE n:Course OR n:Concept OR n:Chunk
            DETACH DELETE n
            """
        ).consume()
    else:
        tx.run(
            """
            MATCH (n {managed_by: $managed_by})
            WHERE n:Course OR n:Concept OR n:Chunk
            DETACH DELETE n
            """,
            managed_by=MANAGED_BY,
        ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (n:ZSTT_Course {course_code: row.key})
        SET n:Course,
            n += row.props,
            n.build_id = $build_id,
            n.managed_by = $managed_by
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
        managed_by=MANAGED_BY,
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (n:ZSTT_Concept {concept_id: row.key})
        SET n:Concept:Knowledge_Point,
            n += row.props,
            n.build_id = $build_id,
            n.managed_by = $managed_by
        """,
        rows=[
            {
                "key": concept.get("id") or concept.get("concept_id"),
                "props": neo4j_properties(
                    {
                        **concept,
                        "concept_id": concept.get("id") or concept.get("concept_id"),
                        "name": concept.get("canonical_name") or concept.get("name", ""),
                        "source": "concept_dependency",
                    }
                ),
            }
            for concept in concepts
            if (concept.get("id") or concept.get("concept_id"))
            and (concept.get("canonical_name") or concept.get("name"))
        ],
        build_id=build_id,
        managed_by=MANAGED_BY,
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (n:ZSTT_Chunk {chunk_id: row.key})
        SET n:Chunk,
            n.text = row.text,
            n.source_file = row.source_file,
            n.section = row.section,
            n.course_code = row.course_code,
            n.build_id = $build_id,
            n.managed_by = $managed_by
        """,
        rows=[
            {
                "key": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
                "source_file": chunk.get("source_file", ""),
                "section": chunk.get("section")
                or (chunk.get("metadata") or {}).get("syllabus_section", ""),
                "course_code": (chunk.get("metadata") or {}).get("course_code")
                or chunk.get("course_code", ""),
            }
            for chunk in chunks
            if chunk.get("chunk_id")
        ],
        build_id=build_id,
        managed_by=MANAGED_BY,
    ).consume()

    match_by_type = {
        "PREREQUISITE_OF": (
            "MATCH (a:ZSTT_Course {course_code: row.source}), "
            "(b:ZSTT_Course {course_code: row.target}) "
            "MERGE (a)-[r:PREREQUISITE_OF]->(b) "
            "SET r += row.props, r.build_id = $build_id, "
            "r.managed_by = $managed_by"
        ),
        "TEACHES": (
            "MATCH (a:ZSTT_Course {course_code: row.source}), "
            "(b:ZSTT_Concept {concept_id: row.target}) "
            "MERGE (a)-[r:TEACHES]->(b) "
            "SET r.build_id = $build_id, r.managed_by = $managed_by"
        ),
        "CONTAINS": (
            "MATCH (a:ZSTT_Course {course_code: row.source}), "
            "(b:ZSTT_Chunk {chunk_id: row.target}) "
            "MERGE (a)-[r:CONTAINS]->(b) "
            "SET r.build_id = $build_id, r.managed_by = $managed_by"
        ),
        "MENTIONS": (
            "MATCH (a:ZSTT_Chunk {chunk_id: row.source}), "
            "(b:ZSTT_Concept {concept_id: row.target}) "
            "MERGE (a)-[r:MENTIONS]->(b) "
            "SET r.build_id = $build_id, r.managed_by = $managed_by"
        ),
    }
    for relation_type in CONCEPT_DEPENDENCY_TYPES:
        match_by_type[relation_type] = (
            "MATCH (a:ZSTT_Concept {concept_id: row.source}), "
            "(b:ZSTT_Concept {concept_id: row.target}) "
            f"MERGE (a)-[r:{relation_type}]->(b) "
            "SET r += row.props, r.build_id = $build_id, "
            "r.managed_by = $managed_by"
        )
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
            managed_by=MANAGED_BY,
        ).consume()


def write_neo4j(
    driver,
    graph: dict,
    courses: list[dict],
    concepts: list[dict],
    chunks: list[dict],
    *,
    reset_existing: bool = False,
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
            reset_existing,
        )
    return {
        "build_id": build_id,
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
    }
