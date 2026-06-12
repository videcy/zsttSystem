"""Hybrid retrieval utilities across vector and graph stores."""

from __future__ import annotations

from typing import Any

CONCEPT_DEPENDENCY_RELATION_TYPES = (
    "FOUNDATION_OF",
    "METHOD_ANALOGY",
    "TOOL_PREREQ",
    "CONCEPTUAL_BASIS",
)


def retrieve_vectors(
    query_embedding: list[float],
    chromadb_collection: Any,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve top-K vector candidates and filter low-similarity results."""
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted_results: list[dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        similarity = 1.0 - float(distance)
        if similarity < 0.65:
            continue
        formatted_results.append(
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata or {},
                "score": similarity,
                "source": "vector",
            }
        )
    return formatted_results


def retrieve_graph_path(
    entity_name: str,
    neo4j_session: Any,
    max_hops: int = 2,
) -> list[dict[str, Any]]:
    """Retrieve graph paths anchored on a named entity.

    Matches Knowledge_Point nodes (concept dependency graph) first,
    then falls back to course-level or other entity traversal.
    Uses confidence-based filtering for concept dependency edges.
    """
    concept_paths = _retrieve_concept_dependency_paths(entity_name, neo4j_session, max_hops)
    if concept_paths:
        return concept_paths

    paths = _retrieve_entity_neighbourhood(entity_name, neo4j_session, max_hops)
    return paths


def _retrieve_concept_dependency_paths(
    entity_name: str,
    neo4j_session: Any,
    max_hops: int = 2,
) -> list[dict[str, Any]]:
    """Traverse verified concept dependency edges anchored on entity_name."""
    relation_filter = "|".join(CONCEPT_DEPENDENCY_RELATION_TYPES)
    query = f"""
        MATCH (start:Knowledge_Point)
        WHERE start.concept_id IS NOT NULL
          AND (start.name CONTAINS $entity_name
               OR $entity_name CONTAINS start.name
               OR $entity_name IN coalesce(start.aliases, []))
        MATCH p=(start)-[rels:{relation_filter}*1..{max_hops}]->(neighbor:Knowledge_Point)
        WHERE neighbor.concept_id IS NOT NULL
          AND all(rel IN rels WHERE coalesce(rel.requires, true) = true)
        WITH p, nodes(p) AS path_nodes, relationships(p) AS path_rels,
             reduce(total = 0.0, rel IN rels | total + coalesce(rel.confidence, 0.5)) AS total_conf,
             size(rels) AS hop_count
        WHERE total_conf / hop_count >= 0.4
        RETURN
            [node IN path_nodes | {{
                concept_id: node.concept_id,
                name: node.name,
                discipline: node.discipline,
                source_courses: node.source_courses,
                source_chapters: node.source_chapters
            }}] AS nodes,
            [rel IN path_rels | {{
                type: type(rel),
                confidence: rel.confidence,
                reason: rel.reason
            }}] AS relations,
            total_conf / hop_count AS avg_confidence
        ORDER BY avg_confidence DESC
        LIMIT 10
        """
    result = neo4j_session.run(query, entity_name=entity_name)

    paths: list[dict[str, Any]] = []
    for record in result:
        paths.append(
            {
                "entity": entity_name,
                "nodes": [
                    {"name": node.get("name", ""), "concept_id": node.get("concept_id", "")}
                    for node in record["nodes"]
                ],
                "relations": [
                    rel.get("type", "") for rel in record["relations"]
                ],
                "score": float(record["avg_confidence"]),
                "source": "graph",
            }
        )
    return paths


def _retrieve_entity_neighbourhood(
    entity_name: str,
    neo4j_session: Any,
    max_hops: int = 2,
) -> list[dict[str, Any]]:
    """Fallback: traverse label-qualified neighbourhood for non-concept entities."""
    query = f"""
        MATCH (start)
        WHERE (start:Course OR start:Instructor OR start:Textbook OR start:Knowledge_Point)
          AND start.name CONTAINS $entity_name
        MATCH p=(start)-[*1..{max_hops}]-(neighbor)
        WHERE neighbor:Course OR neighbor:Knowledge_Point
        WITH p, nodes(p) AS path_nodes, relationships(p) AS path_rels,
             size(path_rels) AS hop_count
        RETURN
            [node IN path_nodes | node.name] AS nodes,
            [rel IN path_rels | type(rel)] AS relations,
            0.5 / hop_count AS avg_score
        ORDER BY avg_score DESC
        LIMIT 10
        """
    result = neo4j_session.run(query, entity_name=entity_name)

    paths: list[dict[str, Any]] = []
    for record in result:
        paths.append(
            {
                "entity": entity_name,
                "nodes": list(record["nodes"]),
                "relations": list(record["relations"]),
                "score": float(record["avg_score"]),
                "source": "graph",
            }
        )
    return paths
