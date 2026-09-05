"""Apply reviewed corrections to the local vector index and knowledge graph.

Corrections are never applied to the collection the service is reading.  A
round builds ``<alias>_v<n+1>``, optionally scores it against the gold set,
and only then repoints the alias -- so a bad correction is undone by
``--rollback`` (one pointer write) instead of by reparsing the corpus, and the
before/after numbers needed to claim the human-in-the-loop cycle *works* fall
out of the same run.

    python maintenance/retraining_updater.py --evaluate
    python maintenance/retraining_updater.py --rollback
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from chromadb.errors import ChromaError
from neo4j import GraphDatabase

from src.config import config
from src.data_processing.chroma_index import build_index, create_chroma_client
from src.data_processing.collection_registry import (
    next_version_name,
    read_alias_record,
    resolve_active_collection,
    write_alias_record,
)
from src.data_processing.lexical_stats import write_lexical_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTED_SAMPLES_PATH = PROJECT_ROOT / "outputs" / "corrected_samples.json"
CHUNKS_PATHS = (
    PROJECT_ROOT / "outputs" / "chunks.json",
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

    def rebuild_index(self) -> dict[str, Any] | None:
        """Build the corrected chunks into a new collection version.

        The alias is left alone: the caller decides whether the new version is
        good enough to serve.
        """
        source_path = next((path for path in CHUNKS_PATHS if path.exists()), None)
        if source_path is None:
            return None
        chunks = json.loads(source_path.read_text(encoding="utf-8"))
        model_name = (
            config.local_embedding_model
            if config.embedding_provider == "local"
            else "hash"
        )
        client = None
        existing: list[str] = []
        try:
            client = create_chroma_client()
            existing = [collection.name for collection in client.list_collections()]
        except (ChromaError, ValueError, OSError, RuntimeError) as exc:
            # Without a reachable store the version number still advances from
            # the alias record, and build_index reports the real failure.
            print(f"[retrain] could not list collections ({exc})")
        target = next_version_name(config.chroma_collection, existing=existing)
        summary = build_index(
            chunks,
            VECTOR_OUTPUT_DIR,
            model_name=model_name,
            dimensions=config.simple_embedding_dimensions,
            client=client,
            collection_name=target,
        )
        # Document frequencies shift with the corrected text, so BM25 stats
        # are rebuilt alongside the vectors rather than left stale.
        write_lexical_stats(
            chunks,
            Path(VECTOR_OUTPUT_DIR) / config.lexical_stats_path.name,
        )
        return summary

    def apply_updates(self, corrected_samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Dispatch corrected samples and rebuild the index when needed."""
        chunks_changed = False
        counts = {"chunking": 0, "kg": 0, "embedding": 0, "skipped": 0}
        for sample in corrected_samples:
            if not isinstance(sample, dict):
                counts["skipped"] += 1
                continue
            issue_type = str(sample.get("issue_type", "")).strip().lower()
            if issue_type == "chunking":
                chunks_changed = self.handle_chunking_issue(sample) or chunks_changed
                counts["chunking"] += 1
            elif issue_type == "kg":
                self.handle_kg_issue(sample)
                counts["kg"] += 1
            elif issue_type == "embedding":
                self.handle_embedding_issue(sample)
                counts["embedding"] += 1
            else:
                counts["skipped"] += 1

        summary: dict[str, Any] = {"counts": counts, "rebuilt": None}
        if chunks_changed:
            summary["rebuilt"] = self.rebuild_index()
        return summary


def _score_collection(collection_name: str | None, dataset: str | None) -> dict[str, Any]:
    """Retrieval metrics for one collection, or an explanation of why not."""
    try:
        from eval.run_eval import evaluate, load_dataset, resolve_dataset_path
    except ImportError as exc:  # pragma: no cover - depends on checkout layout
        return {"error": f"eval package unavailable: {exc}"}
    try:
        items = load_dataset(resolve_dataset_path(dataset))
    except SystemExit as exc:
        return {"error": str(exc)}
    report = evaluate(
        items,
        stages=["retrieval"],
        tag="retraining",
        collection_name=collection_name,
        keep_details=False,
    )
    return report.get("retrieval", {"error": "retrieval stage did not run"})


def _print_comparison(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Print a before/after table; return whether the new version regressed."""
    keys = ("recall@1", "recall@5", "recall@10", "mrr")
    print("\n指标对比（修正前 -> 修正后）")
    print("| 指标 | before | after | delta |")
    print("| --- | --- | --- | --- |")
    regressed = False
    for key in keys:
        old = before.get(key)
        new = after.get(key)
        if old is None or new is None:
            print(f"| {key} | {old} | {new} | - |")
            continue
        delta = round(new - old, 4)
        regressed = regressed or delta < 0
        print(f"| {key} | {old} | {new} | {delta:+} |")
    return regressed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="score the old and new collections on the gold set before promoting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="promote the new version even if the metrics regress",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="point the alias back at the previous collection and exit",
    )
    parser.add_argument("--dataset", default=None)
    arguments = parser.parse_args()

    if arguments.rollback:
        record = read_alias_record()
        history = record.get("history") or []
        previous = next(
            (
                entry.get("previous")
                for entry in reversed(history)
                if entry.get("previous")
            ),
            None,
        )
        if not previous:
            raise SystemExit("no previous collection recorded; nothing to roll back to")
        write_alias_record(previous, note="rollback")
        print(f"alias now points at {previous}")
        return

    updater = RetrainingUpdater()
    try:
        corrected_samples = updater.load_corrected_samples()
        summary = updater.apply_updates(corrected_samples)
    finally:
        updater.close()

    print(f"Applied updates for {len(corrected_samples)} corrected samples.")
    print(json.dumps(summary["counts"], ensure_ascii=False))

    rebuilt = summary.get("rebuilt")
    if not rebuilt:
        print("No chunk corrections applied; the served collection is unchanged.")
        return

    candidate = rebuilt["collection"]
    active = resolve_active_collection()
    print(f"Built {candidate} ({rebuilt['count']} chunks); active alias is {active}.")

    metrics_payload: dict[str, Any] = {}
    if arguments.evaluate:
        before = _score_collection(active, arguments.dataset)
        after = _score_collection(candidate, arguments.dataset)
        metrics_payload = {"before": before, "after": after}
        if "error" in before or "error" in after:
            print(f"[retrain] evaluation skipped: {before.get('error') or after.get('error')}")
        elif _print_comparison(before, after) and not arguments.force:
            print(
                f"\n新版本在金标集上出现指标下降，未切换别名。"
                f"确认无误后可执行：\n"
                f"  python -c \"from src.data_processing.collection_registry import "
                f"write_alias_record; write_alias_record('{candidate}', note='manual')\""
            )
            return

    write_alias_record(
        candidate,
        note="retraining round",
        metrics=metrics_payload,
    )
    print(f"alias now points at {candidate}")


if __name__ == "__main__":
    main()
