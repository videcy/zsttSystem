"""Apply reviewed corrections to the local vector index and knowledge graph."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from src.config import config
from src.data_processing.chroma_index import build_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTED_SAMPLES_PATH = PROJECT_ROOT / "outputs" / "corrected_samples.json"
CHUNKS_PATHS = (
    PROJECT_ROOT / "outputs" / "chunks.json",
    PROJECT_ROOT / "outputs" / "chunked_data.json",
)
VECTOR_OUTPUT_DIR = PROJECT_ROOT / "outputs"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
EMBEDDING_FINETUNE_DATASET_PATH = PROJECT_ROOT / "outputs" / "embedding_finetune_samples.jsonl"


class RetrainingUpdater:
    """Update retrieval and graph assets from manually corrected samples."""

    def __init__(self) -> None:
        self.neo4j_driver = None

    def close(self) -> None:
        """Close external clients."""
        if self.neo4j_driver is not None:
            self.neo4j_driver.close()

    def _get_neo4j_driver(self):
        if self.neo4j_driver is None:
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
            )
        return self.neo4j_driver

    def load_corrected_samples(self, path: str | Path = CORRECTED_SAMPLES_PATH) -> list[dict[str, Any]]:
        """Load corrected samples exported by human reviewers."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Corrected samples file not found: {file_path}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("corrected_samples.json must contain a list.")
        return data

    def handle_chunking_issue(self, sample: dict[str, Any]) -> bool:
        """Update persisted chunk text/metadata after manual corrections."""
        chunk_id = str(sample.get("chunk_id", "")).strip()
        corrected_text = str(sample.get("corrected_text", "")).strip()
        corrected_metadata = sample.get("corrected_metadata", {})
        if not chunk_id or not corrected_text:
            return False
        if not isinstance(corrected_metadata, dict):
            corrected_metadata = {}

        updated = False
        for path in CHUNKS_PATHS:
            if not path.exists():
                continue
            chunks = json.loads(path.read_text(encoding="utf-8"))
            path_updated = False
            for chunk in chunks:
                if str(chunk.get("chunk_id", "")) != chunk_id:
                    continue
                chunk["text"] = corrected_text
                chunk["metadata"] = {
                    **(chunk.get("metadata") or {}),
                    **corrected_metadata,
                }
                path_updated = True
            if path_updated:
                path.write_text(
                    json.dumps(chunks, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                updated = True
        return updated

    def handle_kg_issue(self, sample: dict[str, Any]) -> None:
        """Apply Cypher delete/create fixes for corrected KG relations."""
        delete_relations = sample.get("delete_relations", [])
        create_relations = sample.get("create_relations", [])
        with self._get_neo4j_driver().session() as session:
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
        chunks_changed = False
        for sample in corrected_samples:
            if not isinstance(sample, dict):
                continue
            issue_type = str(sample.get("issue_type", "")).strip().lower()
            if issue_type == "chunking":
                chunks_changed = self.handle_chunking_issue(sample) or chunks_changed
            elif issue_type == "kg":
                self.handle_kg_issue(sample)
            elif issue_type == "embedding":
                self.handle_embedding_issue(sample)

        if chunks_changed:
            source_path = next((path for path in CHUNKS_PATHS if path.exists()), None)
            if source_path is not None:
                chunks = json.loads(source_path.read_text(encoding="utf-8"))
                model_name = (
                    config.local_embedding_model
                    if config.embedding_provider == "local"
                    else "hash"
                )
                build_index(
                    chunks,
                    VECTOR_OUTPUT_DIR,
                    model_name=model_name,
                    dimensions=config.simple_embedding_dimensions,
                )


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
