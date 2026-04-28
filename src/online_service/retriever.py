"""Hybrid retrieval utilities across vector and graph stores."""

from __future__ import annotations

from typing import Any


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
    """Retrieve related graph paths around an entity with strength filtering."""
    query = f"""
    MATCH p=(start {{name: $entity_name}})-[rels*1..{max_hops}]-(neighbor)
    WITH p, neighbor,
         reduce(total = 0, rel IN rels | total + coalesce(rel.strength, 3)) AS total_strength,
         size(rels) AS hop_count
    WHERE total_strength / hop_count >= 3
    RETURN [node IN nodes(p) | node.name] AS nodes,
           [rel IN relationships(p) | type(rel)] AS relations,
           total_strength / hop_count AS avg_strength
    ORDER BY avg_strength DESC
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
                "score": float(record["avg_strength"]),
                "source": "graph",
            }
        )
    return paths
