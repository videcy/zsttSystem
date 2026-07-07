"""Command-line entry point for the offline zsttSystem data pipeline.

With Plan C, the vectorisation stage is delegated to LightRAG.
Use ``--sync`` (or ``--stage sync``) after parsing + kg to push enriched chunks.
Use ``--incremental`` to skip unchanged syllabus files based on a SHA256 manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from src.config import config
from src.data_processing.aligner import BimodalAligner
from src.data_processing.concept_normalizer import ConceptNormalizer
from src.data_processing.data_bridge import run_sync
from src.data_processing.kg_builder import KnowledgeGraphBuilder
from src.data_processing.module_dependency import ModuleDependencyBuilder
from src.data_processing.parser_chunker import SyllabusChunker
from src.utils.file_manifest import FileManifest

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
TRAINING_PLAN_DIR = config.training_plan_dir
SYLLABUS_DIR = config.syllabus_dir
CHUNKED_OUTPUT_PATH = config.chunked_output_path
KG_OUTPUT_PATH = config.kg_output_path
CONCEPT_REGISTRY_PATH = config.concept_registry_path
CONCEPT_VERIFIED_EDGE_PATH = config.concept_verified_edge_path
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
            # Import uuid here for chunk_id generation
            import uuid
            for chk in syllabus_chunks:
                text_parts = [chk.get("section_title", ""), chk.get("content", "")]
                text = chunker._normalize_text("\n\n".join(part for part in text_parts if part))
                if not text:
                    continue
                new_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": text,
                    "source_file": rel,
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


def run_concept_normalization() -> None:
    """Extract & canonicalise teaching concepts, then write the concept
    registry + verified dependency edges and inject ``core_concepts`` back
    into ``chunked_data.json`` so the kg / alignment / module stages can use them.

    Requires the LightRAG embedding stack? No — it calls DeepSeek directly for
    concept extraction + pairwise dependency verification, so it can be slow and
    consumes API quota proportional to the number of chunks and concept pairs.
    """
    chunks = json.loads(CHUNKED_OUTPUT_PATH.read_text(encoding="utf-8"))
    normalizer = ConceptNormalizer()
    enriched_chunks, registry, _alias = normalizer.preprocess_chunks(
        chunks,
        registry_output_path=CONCEPT_REGISTRY_PATH,
        verified_output_path=CONCEPT_VERIFIED_EDGE_PATH,
        enriched_chunks_output_path=CHUNKED_OUTPUT_PATH,
    )
    print(
        f"[concept] {len(registry)} canonical concepts; "
        f"enriched {len(enriched_chunks)} chunks → {CHUNKED_OUTPUT_PATH}\n"
        f"[concept] registry → {CONCEPT_REGISTRY_PATH}\n"
        f"[concept] verified edges → {CONCEPT_VERIFIED_EDGE_PATH}"
    )


def run_kg_building() -> None:
    """Extract entities and relations from chunks → Neo4j knowledge graph."""
    builder = KnowledgeGraphBuilder(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
    )
    try:
        records = builder.run(
            json_path=str(CHUNKED_OUTPUT_PATH),
            output_path=str(KG_OUTPUT_PATH),
            concept_registry_path=str(CONCEPT_REGISTRY_PATH),
            concept_edge_path=str(CONCEPT_VERIFIED_EDGE_PATH),
            reset_concept_subgraph=config.reset_concept_subgraph,
        )
    finally:
        builder.close()
    print(f"[kg] {len(records)} chunk-level KG records → {KG_OUTPUT_PATH}")


def run_alignment() -> None:
    """Link chunk metadata to KG nodes in Neo4j."""
    aligner = BimodalAligner(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
    )
    try:
        summary = aligner.run(
            chunk_data_path=str(CHUNKED_OUTPUT_PATH),
            extracted_kg_data_path=str(KG_OUTPUT_PATH),
            concept_edge_path=str(CONCEPT_VERIFIED_EDGE_PATH),
            concept_registry_path=str(CONCEPT_REGISTRY_PATH),
        )
    finally:
        aligner.close()
    print(
        f"[alignment] {summary['linked_nodes']} nodes, "
        f"{summary.get('aligned_concept_nodes', 0)} concept nodes linked."
    )


def run_module_dependency() -> None:
    """Aggregate concept-level edges → course-level dependency edges in Neo4j."""
    builder = ModuleDependencyBuilder(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
    )
    try:
        summary = builder.run(
            concept_registry_path=str(CONCEPT_REGISTRY_PATH),
            concept_edge_path=str(CONCEPT_VERIFIED_EDGE_PATH),
        )
    finally:
        builder.close()
    print(
        f"[module] {summary['module_edge_count']} module dependency edges "
        f"(Neo4j: {summary.get('written_to_neo4j', 0)})."
    )


def run_lightrag_sync() -> None:
    """Sync chunked + concept-normalised data to LightRAG."""
    result = run_sync(
        chunk_path=str(CHUNKED_OUTPUT_PATH),
        concept_registry_path=str(CONCEPT_REGISTRY_PATH),
    )
    print(f"[sync] LightRAG sync done. success={result['success']}, failed={result['failed']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_stage_map(incremental: bool = False, force: bool = False) -> dict[str, list[PipelineStep]]:
    """Build the stage-to-functions mapping.

    When *incremental* is True, the parsing stage acts as a patch rather
    than a full re-parse.
    """
    def _parse():
        run_parsing(incremental=incremental, force=force)

    return {
        "parsing": [_parse],
        "concept": [run_concept_normalization],
        "kg": [run_kg_building],
        "alignment": [run_alignment],
        "module": [run_module_dependency],
        "sync": [run_lightrag_sync],
        "all": [
            _parse,
            run_concept_normalization,
            run_kg_building,
            run_alignment,
            run_module_dependency,
            run_lightrag_sync,
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="zsttSystem offline data pipeline (v2.0)")
    parser.add_argument(
        "--stage",
        choices=["parsing", "concept", "kg", "alignment", "module", "sync", "all"],
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
    KG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    stage_map = build_stage_map(
        incremental=args.incremental,
        force=args.force,
    )

    for step in stage_map[args.stage]:
        step()


if __name__ == "__main__":
    main()
