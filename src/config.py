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
    def embedding_max_chars(self) -> int | None:
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
        return _path(os.getenv("VECTOR_DB_PATH", "vector_store"))

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
    def chunked_output_path(self) -> Path:
        return _path(os.getenv("CHUNKED_OUTPUT_PATH", "outputs/chunked_data.json"))

    @property
    def kg_output_path(self) -> Path:
        return _path(os.getenv("KG_OUTPUT_PATH", "outputs/kg_extracted_data.json"))

    @property
    def concept_registry_path(self) -> Path:
        return _path(os.getenv("CONCEPT_REGISTRY_PATH", "outputs/concept_registry.json"))

    @property
    def concept_verified_edge_path(self) -> Path:
        return _path(os.getenv("CONCEPT_VERIFIED_EDGE_PATH", "outputs/concept_verified_edges.json"))

    @property
    def concept_normalization_model(self) -> str:
        return os.getenv("CONCEPT_NORMALIZATION_MODEL", "BAAI/bge-large-zh-v1.5").strip()

    @property
    def concept_retrieval_model(self) -> str:
        return os.getenv("CONCEPT_RETRIEVAL_MODEL", "BAAI/bge-large-zh-v1.5").strip()

    # -- Concept Normalization -----------------------------------------------
    @property
    def concept_cluster_threshold(self) -> float:
        return float(os.getenv("CONCEPT_CLUSTER_THRESHOLD", "0.84"))

    @property
    def concept_top_k(self) -> int:
        return int(os.getenv("CONCEPT_TOP_K", "5"))

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
    def concept_score_weight_external(self) -> float:
        return float(os.getenv("CONCEPT_SCORE_WEIGHT_EXTERNAL", "0.4"))

    @property
    def concept_wikipedia_enabled(self) -> bool:
        return _bool(os.getenv("CONCEPT_WIKIPEDIA_ENABLED", "true"))

    @property
    def concept_llm_vote_count(self) -> int:
        return int(os.getenv("CONCEPT_LLM_VOTE_COUNT", "3"))

    @property
    def concept_llm_vote_temperature(self) -> float:
        return float(os.getenv("CONCEPT_LLM_VOTE_TEMPERATURE", "0.2"))

    @property
    def reset_concept_subgraph(self) -> bool:
        return _bool(os.getenv("RESET_CONCEPT_SUBGRAPH", ""))

    # -- NLI -----------------------------------------------------------------
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
