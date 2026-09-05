"""Centralised typed configuration for zsttsystem.

All environment-variable reading happens here.  Import ``config`` from this
module and access its attributes wherever a setting is needed.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _path(value: str) -> Path:
    return PROJECT_ROOT / value


class Config:
    """Typed, lazily-evaluated configuration loaded from environment vars."""

    # -- LLM -----------------------------------------------------------------
    @property
    def deepseek_api_key(self) -> str:
        return os.getenv("DEEPSEEK_API_KEY", "").strip()

    @property
    def deepseek_base_url(self) -> str:
        return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()

    @property
    def text_model(self) -> str:
        return os.getenv("TEXT_MODEL", "deepseek-v4-flash").strip()

    @property
    def rerank_model(self) -> str:
        return os.getenv("RERANK_MODEL", self.text_model).strip()

    @property
    def judge_model(self) -> str:
        return os.getenv("JUDGE_MODEL", self.text_model).strip()

    # -- Embedding -----------------------------------------------------------
    @property
    def embedding_provider(self) -> str:
        return os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

    @property
    def embedding_model(self) -> str:
        return os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ).strip()

    @property
    def local_embedding_model(self) -> str:
        return os.getenv("LOCAL_EMBEDDING_MODEL", self.embedding_model).strip()

    @property
    def embedding_max_chars(self) -> int:
        val = os.getenv("EMBEDDING_MAX_CHARS", "").strip()
        return int(val) if val else 3000

    @property
    def simple_embedding_dimensions(self) -> int:
        return int(os.getenv("SIMPLE_EMBEDDING_DIMENSIONS", "384"))

    @property
    def embedding_api_key(self) -> str:
        return os.getenv("EMBEDDING_API_KEY", "").strip()

    @property
    def embedding_base_url(self) -> str:
        return os.getenv("EMBEDDING_BASE_URL", "").strip()

    # -- Neo4j ---------------------------------------------------------------
    @property
    def neo4j_uri(self) -> str:
        return os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()

    @property
    def neo4j_user(self) -> str:
        return os.getenv("NEO4J_USER", "neo4j").strip()

    @property
    def neo4j_password(self) -> str:
        return os.getenv("NEO4J_PASSWORD", "").strip()

    # -- Paths ---------------------------------------------------------------
    @property
    def vector_db_path(self) -> Path:
        return _path(os.getenv("VECTOR_DB_PATH", "chroma_data"))

    @property
    def chroma_mode(self) -> str:
        return os.getenv("CHROMA_MODE", "local").strip().lower()

    @property
    def chroma_host(self) -> str:
        return os.getenv("CHROMA_HOST", "127.0.0.1").strip()

    @property
    def chroma_port(self) -> int:
        return int(os.getenv("CHROMA_PORT", "8001"))

    @property
    def chroma_ssl(self) -> bool:
        return _bool(os.getenv("CHROMA_SSL", "false"))

    @property
    def chroma_collection(self) -> str:
        return os.getenv("CHROMA_COLLECTION", "zstt_chunks").strip()

    @property
    def query_log_path(self) -> Path:
        return _path(os.getenv("QUERY_LOG_PATH", "outputs/query_log.jsonl"))

    @property
    def feedback_log_path(self) -> Path:
        return _path(os.getenv("FEEDBACK_LOG_PATH", "outputs/feedback_log.jsonl"))

    @property
    def training_plan_dir(self) -> Path:
        return _path(os.getenv("TRAINING_PLAN_DIR", "data/training_plans"))

    @property
    def syllabus_dir(self) -> Path:
        return _path(os.getenv("SYLLABUS_DIR", "data/syllabi"))

    @property
    def courses_output_path(self) -> Path:
        return _path(os.getenv("COURSES_OUTPUT_PATH", "outputs/courses.json"))

    @property
    def chunks_output_path(self) -> Path:
        return _path(os.getenv("CHUNKS_OUTPUT_PATH", "outputs/chunks.json"))

    @property
    def concept_cache_path(self) -> Path:
        return _path(os.getenv("CONCEPT_CACHE_PATH", "outputs/concept_cache.json"))

    @property
    def graph_manifest_path(self) -> Path:
        return _path(os.getenv("GRAPH_MANIFEST_PATH", "outputs/graph_manifest.json"))

    @property
    def concept_registry_path(self) -> Path:
        return _path(os.getenv("CONCEPT_REGISTRY_PATH", "outputs/concept_registry.json"))

    @property
    def concept_verified_edge_path(self) -> Path:
        return _path(os.getenv("CONCEPT_VERIFIED_EDGE_PATH", "outputs/concept_verified_edges.json"))

    @property
    def concept_alias_path(self) -> Path:
        return _path(os.getenv("CONCEPT_ALIAS_PATH", "outputs/concept_aliases.json"))

    @property
    def concept_candidate_edge_path(self) -> Path:
        return _path(os.getenv("CONCEPT_CANDIDATE_EDGE_PATH", "outputs/concept_candidate_edges.json"))

    @property
    def concept_extraction_cache_path(self) -> Path:
        return _path(os.getenv("CONCEPT_EXTRACTION_CACHE_PATH", "outputs/concept_extraction_cache.json"))

    @property
    def concept_validation_cache_path(self) -> Path:
        return _path(os.getenv("CONCEPT_VALIDATION_CACHE_PATH", "outputs/concept_validation_cache.json"))

    @property
    def concept_normalization_model(self) -> str:
        return os.getenv("CONCEPT_NORMALIZATION_MODEL", "BAAI/bge-large-zh-v1.5").strip()

    @property
    def concept_retrieval_model(self) -> str:
        return os.getenv("CONCEPT_RETRIEVAL_MODEL", "BAAI/bge-large-zh-v1.5").strip()

    @property
    def lexical_stats_path(self) -> Path:
        return _path(os.getenv("LEXICAL_STATS_PATH", "outputs/lexical_stats.json"))

    @property
    def collection_alias_path(self) -> Path:
        return _path(
            os.getenv("COLLECTION_ALIAS_PATH", "outputs/collection_alias.json")
        )

    @property
    def eval_dataset_path(self) -> Path:
        return _path(
            os.getenv("EVAL_DATASET_PATH", "eval/datasets/gold_questions.json")
        )

    @property
    def eval_report_dir(self) -> Path:
        return _path(os.getenv("EVAL_REPORT_DIR", "eval/reports"))

    # -- Retrieval reranking -------------------------------------------------
    # Weights are read from the environment so that the grid search in
    # ``eval/tune_rerank.py`` can sweep them without editing code.  The
    # defaults reproduce the hand-tuned values used before the sweep existed.
    @property
    def rerank_weight_vector(self) -> float:
        return float(os.getenv("RERANK_WEIGHT_VECTOR", "0.75"))

    @property
    def rerank_weight_lexical(self) -> float:
        return float(os.getenv("RERANK_WEIGHT_LEXICAL", "0.15"))

    @property
    def rerank_section_boost(self) -> float:
        return float(os.getenv("RERANK_SECTION_BOOST", "0.08"))

    @property
    def rerank_lexical_scheme(self) -> str:
        """``bm25`` (IDF-weighted, default) or ``overlap`` (legacy baseline)."""
        return os.getenv("RERANK_LEXICAL_SCHEME", "bm25").strip().lower()

    @property
    def rerank_bm25_k1(self) -> float:
        return float(os.getenv("RERANK_BM25_K1", "1.5"))

    @property
    def rerank_bm25_b(self) -> float:
        return float(os.getenv("RERANK_BM25_B", "0.75"))

    @property
    def rerank_candidate_multiplier(self) -> int:
        return max(1, int(os.getenv("RERANK_CANDIDATE_MULTIPLIER", "4")))

    # -- Graph ---------------------------------------------------------------
    @property
    def graph_include_chunk_nodes(self) -> bool:
        """Mirror chunks into Neo4j.

        Chunk nodes carry no properties beyond an id and a filename -- the
        text lives in Chroma -- so no Cypher path uses them.  Off by default;
        enable when experimenting with graph-side community summarisation.
        """
        return _bool(os.getenv("GRAPH_INCLUDE_CHUNK_NODES", "false"))

    # -- API surface ---------------------------------------------------------
    @property
    def api_max_query_chars(self) -> int:
        return max(1, int(os.getenv("API_MAX_QUERY_CHARS", "500")))

    @property
    def api_rate_limit_per_minute(self) -> int:
        """Per-client requests allowed each minute; 0 disables the limiter."""
        return max(0, int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60")))

    @property
    def api_cors_origins(self) -> list[str]:
        raw = os.getenv("API_CORS_ORIGINS", "").strip()
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    # -- Concept Normalization -----------------------------------------------
    @property
    def concept_cluster_threshold(self) -> float:
        return float(os.getenv("CONCEPT_CLUSTER_THRESHOLD", "0.84"))

    @property
    def concept_top_k(self) -> int:
        return int(os.getenv("CONCEPT_TOP_K", "5"))

    @property
    def concept_candidate_min_confidence(self) -> float:
        return float(os.getenv("CONCEPT_CANDIDATE_MIN_CONFIDENCE", "0.45"))

    @property
    def concept_verified_min_confidence(self) -> float:
        return float(os.getenv("CONCEPT_VERIFIED_MIN_CONFIDENCE", "0.6"))

    @property
    def concept_max_verification_candidates(self) -> int:
        return max(0, int(os.getenv("CONCEPT_MAX_VERIFICATION_CANDIDATES", "500")))

    @property
    def concept_min_extraction_coverage(self) -> float:
        return float(os.getenv("CONCEPT_MIN_EXTRACTION_COVERAGE", "0.1"))

    @property
    def concept_course_order_bonus(self) -> float:
        return float(os.getenv("CONCEPT_COURSE_ORDER_BONUS", "0.3"))

    @property
    def concept_cross_discipline_decay(self) -> float:
        return float(os.getenv("CONCEPT_CROSS_DISCIPLINE_DECAY", "0.85"))

    @property
    def concept_score_weight_vector(self) -> float:
        return float(os.getenv("CONCEPT_SCORE_WEIGHT_VECTOR", "0.2"))

    @property
    def concept_score_weight_structure(self) -> float:
        return float(os.getenv("CONCEPT_SCORE_WEIGHT_STRUCTURE", "0.2"))

    @property
    def concept_score_weight_rule(self) -> float:
        return float(os.getenv("CONCEPT_SCORE_WEIGHT_RULE", "0.2"))

    @property
    def concept_wikipedia_enabled(self) -> bool:
        return _bool(os.getenv("CONCEPT_WIKIPEDIA_ENABLED", "false"))

    @property
    def concept_llm_vote_count(self) -> int:
        return int(os.getenv("CONCEPT_LLM_VOTE_COUNT", "3"))

    @property
    def concept_api_concurrency(self) -> int:
        return max(1, int(os.getenv("CONCEPT_API_CONCURRENCY", "4")))

    @property
    def concept_llm_vote_temperature(self) -> float:
        return float(os.getenv("CONCEPT_LLM_VOTE_TEMPERATURE", "0.2"))

    @property
    def reset_concept_subgraph(self) -> bool:
        return _bool(os.getenv("RESET_CONCEPT_SUBGRAPH", ""))

    # -- NLI -----------------------------------------------------------------
    @property
    def nli_verification_enabled(self) -> bool:
        return _bool(os.getenv("NLI_VERIFICATION_ENABLED", "true"))

    @property
    def nli_entailment_threshold(self) -> float:
        return float(os.getenv("NLI_ENTAILMENT_THRESHOLD", "0.6"))

    @property
    def nli_max_retries(self) -> int:
        return int(os.getenv("NLI_MAX_RETRIES", "1"))

    # -- OpenSearch / Alternate keys ----------------------------------------
    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "").strip()

    @property
    def openai_base_url(self) -> str:
        return os.getenv("OPENAI_BASE_URL", "").strip()

config = Config()
