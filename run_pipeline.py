"""Command-line entry point for the offline zsttSystem data pipeline.

Use ``--incremental`` to skip unchanged syllabus files based on a SHA256
manifest. The ``all`` stage builds local parsing, concept, graph, and vector
artifacts in ChromaDB and optional Neo4j.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError

from src.config import config
from src.data_processing.parser_chunker import SyllabusChunker
from src.data_processing.chroma_index import build_index
from src.data_processing.concept_extractor import extract_course_concepts
from src.data_processing.graph_builder import build_graph_records, write_neo4j
from src.utils.file_manifest import FileManifest

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
TRAINING_PLAN_DIR = config.training_plan_dir
SYLLABUS_DIR = config.syllabus_dir
CHUNKED_OUTPUT_PATH = config.chunked_output_path
OUTPUT_DIR = CHUNKED_OUTPUT_PATH.parent
NEO4J_URI = config.neo4j_uri
NEO4J_USER = config.neo4j_user
NEO4J_PASSWORD = config.neo4j_password

PipelineStep = Callable[[], None]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_parsing(*, incremental: bool = False, force: bool = False) -> None:
    """Parse XLSX training plans + DOCX syllabi into structured chunks.

    When *incremental* is True and *force* is False, only re-parses syllabus
    files whose content has changed (detected via SHA256 manifest).  Chunks
    from unchanged files are preserved from the previous run.
    """
    chunker = SyllabusChunker()

    if incremental and not force and CHUNKED_OUTPUT_PATH.exists():
        _incremental_parse(chunker)
    else:
        results = chunker.run(
            plan_dir=TRAINING_PLAN_DIR,
            syllabus_dir=SYLLABUS_DIR,
            output_path=CHUNKED_OUTPUT_PATH,
        )
        tag = "force" if force else "full"
        print(f"[parsing] ({tag}) Generated {len(results)} chunks → {CHUNKED_OUTPUT_PATH}")


def _incremental_parse(chunker: SyllabusChunker) -> None:
    """Detect changed syllabus files and patch chunked_data.json in-place."""
    manifest = FileManifest()
    syllabus_files = sorted(
        p for p in SYLLABUS_DIR.rglob("*.docx") if not p.name.startswith("~$")
    )

    added, changed, removed = manifest.check(syllabus_files)

    if not added and not changed and not removed:
        print("[parsing] (incremental) No changes detected — skipping.")
        return

    # Load existing chunks
    existing = json.loads(CHUNKED_OUTPUT_PATH.read_text(encoding="utf-8"))
    existing_by_file: dict[str, list[dict]] = {}
    other_chunks: list[dict] = []
    for ch in existing:
        src = ch.get("source_file", "")
        if src:
            existing_by_file.setdefault(src, []).append(ch)
        else:
            other_chunks.append(ch)

    # Remove chunks from deleted files
    for rel in removed:
        removed_count = len(existing_by_file.pop(rel, []))
        print(f"[parsing] (incremental) Removed: {rel} ({removed_count} chunks)")

    # Re-parse changed / added files
    files_to_parse = added + changed
    if files_to_parse:
        # Load training plan metadata (read once)
        all_course_metadata: dict[str, dict] = {}
        for xlsx_file in sorted(TRAINING_PLAN_DIR.glob("*.xlsx")):
            all_course_metadata.update(chunker.parse_training_plan(xlsx_file))

        for f in files_to_parse:
            rel = str(f.relative_to(SYLLABUS_DIR).as_posix())
            syllabus_chunks = chunker.parse_syllabus(f)
            if not syllabus_chunks:
                continue

            course_metadata = chunker._match_course_metadata(
                f, syllabus_chunks, all_course_metadata,
            )
            new_chunks: list[dict] = []
            import hashlib
            for chk in syllabus_chunks:
                text_parts = [chk.get("section_title", ""), chk.get("content", "")]
                text = chunker._normalize_text("\n\n".join(part for part in text_parts if part))
                if not text:
                    continue
                section = chunker._normalize_text(chk.get("section_title", ""))
                document_hash = hashlib.sha256(f.read_bytes()).hexdigest()
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                new_chunks.append({
                    "chunk_id": hashlib.sha256(f"{rel}|{section}|{text_hash}".encode()).hexdigest(),
                    "text": text,
                    "source_file": rel,
                    "document_hash": document_hash,
                    "section": section,
                    "metadata": {
                        "course_code": chunker._normalize_text(course_metadata.get("course_code", "")),
                        "course_name": chunker._normalize_text(course_metadata.get("course_name", "")),
                        "syllabus_section": chunker._normalize_text(chk.get("section_title", "")),
                        "prerequisites": list(course_metadata.get("prerequisites", [])),
                        "credits": course_metadata.get("credits"),
                    },
                })

            old_count = len(existing_by_file.get(rel, []))
            existing_by_file[rel] = new_chunks
            status = "Added" if f in added else "Changed"
            print(f"[parsing] (incremental) {status}: {rel} "
                  f"({old_count}→{len(new_chunks)} chunks)")

    # Merge and write
    merged = list(other_chunks)
    for chunks in existing_by_file.values():
        merged.extend(chunks)

    CHUNKED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKED_OUTPUT_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[parsing] (incremental) Merged {len(merged)} chunks → {CHUNKED_OUTPUT_PATH}")

    # Update manifest
    manifest.update(syllabus_files)
    manifest.save()


def run_concept_stage() -> None:
    chunks = json.loads(CHUNKED_OUTPUT_PATH.read_text(encoding="utf-8"))
    concepts = extract_course_concepts(chunks, OUTPUT_DIR / "concept_cache.json", config.text_model)
    (OUTPUT_DIR / "concepts.json").write_text(json.dumps(concepts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[concept] {len(concepts)} course-level concepts")

def run_embed_stage() -> None:
    chunks = json.loads((OUTPUT_DIR / "chunks.json").read_text(encoding="utf-8")) if (OUTPUT_DIR / "chunks.json").exists() else json.loads(CHUNKED_OUTPUT_PATH.read_text(encoding="utf-8"))
    summary = build_index(
        chunks,
        OUTPUT_DIR,
        model_name=(
            config.local_embedding_model
            if config.embedding_provider == "local"
            else "hash"
        ),
        dimensions=config.simple_embedding_dimensions,
    )
    print(
        f"[embed] ChromaDB collection {summary['collection']} "
        f"contains {summary['count']} chunks"
    )

def run_parse_stage(incremental: bool = False, force: bool = False) -> None:
    run_parsing(incremental=incremental, force=force)
    chunks = json.loads(CHUNKED_OUTPUT_PATH.read_text(encoding="utf-8"))
    courses = {}
    for c in chunks:
        m = c.get("metadata", {}); code = m.get("course_code") or c.get("course_code")
        if code: courses.setdefault(code, {"course_code": code, "course_name": m.get("course_name", ""), "credits": m.get("credits"), "prerequisites": m.get("prerequisites", []), "source_file": c.get("source_file"), "document_hash": c.get("document_hash")})
    (OUTPUT_DIR / "courses.json").write_text(json.dumps(list(courses.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

def run_graph_stage() -> None:
    """Write a deterministic graph manifest; Neo4j writing remains optional."""
    chunks = json.loads((OUTPUT_DIR / "chunks.json").read_text(encoding="utf-8"))
    concepts = json.loads((OUTPUT_DIR / "concepts.json").read_text(encoding="utf-8")) if (OUTPUT_DIR / "concepts.json").exists() else []
    courses = json.loads((OUTPUT_DIR / "courses.json").read_text(encoding="utf-8")) if (OUTPUT_DIR / "courses.json").exists() else []
    graph = build_graph_records(courses, concepts, chunks)
    neo4j_status = "unavailable"
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), connection_timeout=2, connection_acquisition_timeout=2, max_transaction_retry_time=1); driver.verify_connectivity()
        try:
            write_neo4j(driver, graph, courses, concepts, chunks); neo4j_status = "written"
        finally: driver.close()
    except (Neo4jError, ServiceUnavailable, AuthError, OSError) as exc:
        print(f"[graph] Neo4j unavailable: {exc}")
    manifest = {"courses": sorted({c.get("course_code") for c in courses}), "concepts": [c["concept_id"] for c in concepts], "nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "neo4j": neo4j_status, "updated_at": __import__('datetime').datetime.now().isoformat()}
    (OUTPUT_DIR / "graph_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[graph] manifest written ({len(manifest['courses'])} courses)")

def run_baseline_stage() -> None:
    """Freeze a reproducible baseline without requiring external services."""
    baseline = OUTPUT_DIR / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in OUTPUT_DIR.glob("*"):
        if src.is_file() and src.name != "baseline.json":
            dst = baseline / src.name; shutil.copy2(src, dst); copied.append(src.name)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "outputs": copied, "chunk_count": 0, "api_calls": 0}
    if CHUNKED_OUTPUT_PATH.exists():
        try: summary["chunk_count"] = len(json.loads(CHUNKED_OUTPUT_PATH.read_text(encoding="utf-8")))
        except json.JSONDecodeError: pass
    (baseline / "baseline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[baseline] saved to {baseline}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_stage_map(incremental: bool = False, force: bool = False) -> dict[str, list[PipelineStep]]:
    """Build the stage-to-functions mapping.

    When *incremental* is True, the parsing stage acts as a patch rather
    than a full re-parse.
    """
    return {
        "parse": [lambda: run_parse_stage(incremental=incremental, force=force)],
        "baseline": [run_baseline_stage],
        "concept": [run_concept_stage],
        "all": [
            lambda: run_parse_stage(incremental=incremental, force=force), run_concept_stage, run_graph_stage, run_embed_stage,
        ],
        "graph": [run_graph_stage],
        "embed": [run_embed_stage],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="zsttSystem offline data pipeline (v2.0)")
    parser.add_argument(
        "--stage",
        choices=["baseline", "parse", "concept", "graph", "embed", "all"],
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: only process changed syllabus files (parsing stage).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full reprocess even in incremental mode.",
    )
    return parser.parse_args()


def main() -> None:
    CHUNKED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    stage_map = build_stage_map(
        incremental=args.incremental,
        force=args.force,
    )

    for step in stage_map[args.stage]:
        step()


if __name__ == "__main__":
    main()
