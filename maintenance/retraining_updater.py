"""Apply reviewed corrections to the vector store and knowledge graph."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTED_SAMPLES_PATH = PROJECT_ROOT / "outputs" / "corrected_samples.json"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "scholar_collection"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
EMBEDDING_FINETUNE_DATASET_PATH = PROJECT_ROOT / "outputs" / "embedding_finetune_samples.jsonl"


class RetrainingUpdater:
    """Update retrieval and graph assets from manually corrected samples."""

    def __init__(self) -> None:
        self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )

    def close(self) -> None:
        """Close external clients."""
        self.neo4j_driver.close()

    def load_corrected_samples(self, path: str | Path = CORRECTED_SAMPLES_PATH) -> list[dict[str, Any]]:
        """Load corrected samples exported by human reviewers."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Corrected samples file not found: {file_path}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("corrected_samples.json must contain a list.")
        return data

    def handle_chunking_issue(self, sample: dict[str, Any]) -> None:
        """Update chunk text/metadata in ChromaDB after manual chunk corrections."""
        chunk_id = str(sample.get("chunk_id", "")).strip()
        corrected_text = str(sample.get("corrected_text", "")).strip()
        corrected_metadata = sample.get("corrected_metadata", {})
        if not chunk_id or not corrected_text:
            return
        if not isinstance(corrected_metadata, dict):
            corrected_metadata = {}
        self.collection.update(
            ids=[chunk_id],
            documents=[corrected_text],
            metadatas=[corrected_metadata],
        )

    def handle_kg_issue(self, sample: dict[str, Any]) -> None:
        """Apply Cypher delete/create fixes for corrected KG relations."""
        delete_relations = sample.get("delete_relations", [])
        create_relations = sample.get("create_relations", [])
        with self.neo4j_driver.session() as session:
            for relation in delete_relations:
                if not isinstance(relation, dict):
                    continue
                session.run(
                    """
                    MATCH (a {name: $source})-[r]->(b {name: $target})
                    WHERE type(r) = $relation_type
                    DELETE r
                    """,
                    source=relation.get("source", ""),
                    target=relation.get("target", ""),
                    relation_type=relation.get("type", ""),
                )
            for relation in create_relations:
                if not isinstance(relation, dict):
                    continue
                relation_type = str(relation.get("type", "")).strip()
                if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", relation_type):
                    continue
                session.run(
                    f"""
                    MATCH (a {{name: $source}}), (b {{name: $target}})
                    MERGE (a)-[:{relation_type}]->(b)
                    """,
                    source=relation.get("source", ""),
                    target=relation.get("target", ""),
                )

    def handle_embedding_issue(self, sample: dict[str, Any]) -> None:
        """Append problematic query-response pairs to a future embedding fine-tune set."""
        record = {
            "query": sample.get("query", ""),
            "positive_text": sample.get("corrected_text", ""),
            "reason": sample.get("issue_type", "embedding"),
        }
        EMBEDDING_FINETUNE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EMBEDDING_FINETUNE_DATASET_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def apply_updates(self, corrected_samples: list[dict[str, Any]]) -> None:
        """Dispatch corrected samples to the corresponding update handlers."""
        for sample in corrected_samples:
            if not isinstance(sample, dict):
                continue
            issue_type = str(sample.get("issue_type", "")).strip().lower()
            if issue_type == "chunking":
                self.handle_chunking_issue(sample)
            elif issue_type == "kg":
                self.handle_kg_issue(sample)
            elif issue_type == "embedding":
                self.handle_embedding_issue(sample)


def main() -> None:
    """Apply updates from corrected samples."""
    updater = RetrainingUpdater()
    try:
        corrected_samples = updater.load_corrected_samples()
        updater.apply_updates(corrected_samples)
    finally:
        updater.close()
    print(f"Applied updates for {len(corrected_samples)} corrected samples.")


if __name__ == "__main__":
    main()
