"""Unified, idempotent graph preparation for the lightweight pipeline."""
from __future__ import annotations
import hashlib

def stable_id(label: str, value: str) -> str:
    return hashlib.sha256(f"{label}|{value}".encode("utf-8")).hexdigest()[:24]

def build_graph_records(courses: list[dict], concepts: list[dict], chunks: list[dict]) -> dict:
    nodes = []
    seen = set()
    for course in courses:
        code = course.get("course_code", "")
        if code and ("Course", code) not in seen:
            nodes.append({"id": stable_id("Course", code), "label": "Course", "course_code": code, **course}); seen.add(("Course", code))
    for concept in concepts:
        name = concept.get("name", "")
        if name and ("Concept", name) not in seen:
            nodes.append({"id": concept.get("concept_id") or stable_id("Concept", name), "label": "Concept", **concept}); seen.add(("Concept", name))
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        if cid and ("Chunk", cid) not in seen:
            nodes.append({"id": cid, "label": "Chunk", "chunk_id": cid, "source_file": chunk.get("source_file")}); seen.add(("Chunk", cid))
    edges = []
    course_by_code = {c.get("course_code"): c for c in courses}
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
        for prereq in course.get("prerequisites", []) or []:
            edges.append({"source": prereq, "target": target, "type": "PREREQUISITE_OF"})
    return {"nodes": nodes, "edges": edges}

def ensure_constraints(session) -> None:
    for label, key in (("Course", "course_code"), ("Concept", "concept_id"), ("Chunk", "chunk_id")):
        try:
            session.run(f"CREATE CONSTRAINT {label.lower()}_{key}_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE").consume()
        except Exception:
            pass

def write_neo4j(driver, graph: dict, courses: list[dict], concepts: list[dict], chunks: list[dict]) -> dict:
    """Idempotently MERGE nodes and relationships in one Neo4j transaction."""
    with driver.session() as session:
        ensure_constraints(session)
        for c in courses:
            session.run("MERGE (n:Course {course_code:$code}) SET n += $props", code=c.get("course_code"), props=c).consume()
        for c in concepts:
            session.run("MERGE (n:Concept {concept_id:$id}) SET n += $props", id=c.get("concept_id") or stable_id("Concept", c.get("name", "")), props=c).consume()
        for c in chunks:
            session.run("MERGE (n:Chunk {chunk_id:$id}) SET n.text=$text, n.source_file=$source", id=c.get("chunk_id"), text=c.get("text", ""), source=c.get("source_file", "")).consume()
        for edge in graph["edges"]:
            if edge["type"] == "PREREQUISITE_OF":
                session.run("MATCH (a:Course {course_code:$source}), (b:Course {course_code:$target}) MERGE (a)-[:PREREQUISITE_OF]->(b)", source=edge["source"], target=edge["target"]).consume()
            elif edge["type"] == "TEACHES":
                session.run("MATCH (a:Course {course_code:$source}), (b:Concept {concept_id:$target}) MERGE (a)-[:TEACHES]->(b)", source=edge["source"], target=edge["target"]).consume()
            elif edge["type"] == "CONTAINS":
                session.run("MATCH (a:Course {course_code:$source}), (b:Chunk {chunk_id:$target}) MERGE (a)-[:CONTAINS]->(b)", source=edge["source"], target=edge["target"]).consume()
    return {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])}
