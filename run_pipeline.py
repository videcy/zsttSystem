"""Command-line entry point for the offline zsttsystem data pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from src.data_processing.aligner import BimodalAligner
from src.data_processing.kg_builder import KnowledgeGraphBuilder
from src.data_processing.parser_chunker import SyllabusChunker
from src.data_processing.vectorizer import VectorIndexer


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

TRAINING_PLAN_DIR = Path(
    os.getenv("TRAINING_PLAN_DIR", str(PROJECT_ROOT / "data" / "training_plans"))
)
SYLLABUS_DIR = Path(
    os.getenv("SYLLABUS_DIR", str(PROJECT_ROOT / "data" / "syllabi"))
)
CHUNKED_OUTPUT_PATH = Path(
    os.getenv("CHUNKED_OUTPUT_PATH", str(PROJECT_ROOT / "outputs" / "chunked_data.json"))
)
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_DB_PATH", str(PROJECT_ROOT / "vector_store")))
KG_OUTPUT_PATH = Path(
    os.getenv("KG_OUTPUT_PATH", str(PROJECT_ROOT / "outputs" / "kg_extracted_data.json"))
)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

PipelineStep = Callable[[], None]


def run_parsing() -> None:
    """Run document parsing and chunking."""
    chunker = SyllabusChunker()
    results = chunker.run(
        plan_dir=TRAINING_PLAN_DIR,
        syllabus_dir=SYLLABUS_DIR,
        output_path=CHUNKED_OUTPUT_PATH,
    )
    print(f"[parsing] Generated {len(results)} chunks at {CHUNKED_OUTPUT_PATH}.")


def run_vectorization() -> None:
    """Run embedding generation and vector index building."""
    indexer = VectorIndexer(db_path=str(VECTOR_STORE_PATH))
    indexer.run(json_path=str(CHUNKED_OUTPUT_PATH))
    print(f"[vectorization] Indexed chunks from {CHUNKED_OUTPUT_PATH} into {VECTOR_STORE_PATH}.")


def run_kg_building() -> None:
    """Run knowledge extraction and graph construction."""
    builder = KnowledgeGraphBuilder(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
    )
    try:
        records = builder.run(
            json_path=str(CHUNKED_OUTPUT_PATH),
            output_path=str(KG_OUTPUT_PATH),
        )
    finally:
        builder.close()
    print(f"[kg] Extracted and saved {len(records)} chunk-level KG records to {KG_OUTPUT_PATH}.")


def run_alignment() -> None:
    """Run alignment between chunks and graph entities."""
    aligner = BimodalAligner(
        chroma_db_path=str(VECTOR_STORE_PATH),
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
    )
    try:
        summary = aligner.run(
            chunk_data_path=str(CHUNKED_OUTPUT_PATH),
            extracted_kg_data_path=str(KG_OUTPUT_PATH),
        )
    finally:
        aligner.close()
    print(f"[alignment] Linked {summary['linked_chunks']} chunks and {summary['linked_nodes']} nodes.")


def build_stage_map() -> dict[str, list[PipelineStep]]:
    """Return the executable pipeline stages."""
    return {
        "parsing": [run_parsing],
        "vectorization": [run_vectorization],
        "kg": [run_kg_building],
        "alignment": [run_alignment],
        "all": [
            run_parsing,
            run_vectorization,
            run_kg_building,
            run_alignment,
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stage-based pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run the offline pipeline for the zsttsystem RAG project."
    )
    parser.add_argument(
        "--stage",
        choices=["parsing", "vectorization", "kg", "alignment", "all"],
        default="all",
        help="Specify which pipeline stage to execute.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the selected pipeline stage."""
    CHUNKED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    KG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    stage_map = build_stage_map()

    for step in stage_map[args.stage]:
        step()


if __name__ == "__main__":
    main()
