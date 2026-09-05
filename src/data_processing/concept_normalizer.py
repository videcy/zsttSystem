"""Extract, normalize, score, and validate teaching concepts before embedding."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.deepseek_client import create_deepseek_client, embed_texts, generate_json_value
from src.config import config

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT_VERSION = "teaching-concepts-v4"
DEPENDENCY_PROMPT_VERSION = "concept-dependency-v2"

_CONCEPT_NOISE = {
    "中山大学",
    "信息管理学院",
    "课程名称",
    "课程编码",
    "课程代码",
    "课程类别",
    "课程负责人",
    "课程目标",
    "教学大纲",
    "本科课程教学大纲",
    "开课单位",
    "授课年级",
    "面向专业",
    "编写日期",
    "学分",
    "学时",
}


class ConceptNormalizer:
    """Use LLMs and embeddings to canonicalize and validate concept dependencies."""

    ALLOWED_TYPES = {
        "algorithm",
        "theorem",
        "operator",
        "method",
        "model",
        "framework",
        "data_structure",
        "paradigm",
        "concept",
    }
    ALLOWED_BLOOM_LEVELS = {"understand", "apply", "analyze"}
    TYPE_ALIASES = {
        "算法": "algorithm",
        "定理": "theorem",
        "算子": "operator",
        "方法": "method",
        "模型": "model",
        "框架": "framework",
        "数据结构": "data_structure",
        "data structure": "data_structure",
        "data-structure": "data_structure",
        "范式": "paradigm",
        "概念": "concept",
    }
    BLOOM_ALIASES = {
        "理解": "understand",
        "理解层次": "understand",
        "comprehension": "understand",
        "应用": "apply",
        "应用层次": "apply",
        "application": "apply",
        "分析": "analyze",
        "分析层次": "analyze",
        "analysis": "analyze",
    }
    BLOOM_ORDER = {"understand": 0, "apply": 1, "analyze": 2}
    ALLOWED_RELATION_TYPES = {
        "FOUNDATION_OF",
        "METHOD_ANALOGY",
        "TOOL_PREREQ",
        "CONCEPTUAL_BASIS",
        "NO_RELATION",
    }

    def __init__(
        self,
        *,
        llm_model_name: str | None = None,
        embedding_model_name: str | None = None,
        retrieval_model_name: str | None = None,
        similarity_threshold: float | None = None,
        wikipedia_enabled: bool | None = None,
        candidate_top_k: int | None = None,
        candidate_min_confidence: float | None = None,
        max_verification_candidates: int | None = None,
        min_extraction_coverage: float | None = None,
        course_order_bonus: float | None = None,
        cross_discipline_decay: float | None = None,
        score_weight_vector: float | None = None,
        score_weight_structure: float | None = None,
        score_weight_rule: float | None = None,
        llm_vote_count: int | None = None,
        llm_vote_temperature: float | None = None,
        api_concurrency: int | None = None,
        allow_rule_fallback: bool = True,
        require_complete_llm_validation: bool = False,
        allow_embedding_fallback: bool = True,
    ) -> None:
        self.llm_model_name = llm_model_name or config.text_model
        self.embedding_model_name = embedding_model_name or config.concept_normalization_model
        self.retrieval_model_name = retrieval_model_name or config.concept_retrieval_model
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else config.concept_cluster_threshold
        )
        self.wikipedia_enabled = (
            wikipedia_enabled
            if wikipedia_enabled is not None
            else config.concept_wikipedia_enabled
        )
        self.candidate_top_k = (
            candidate_top_k if candidate_top_k is not None else config.concept_top_k
        )
        self.candidate_min_confidence = (
            candidate_min_confidence
            if candidate_min_confidence is not None
            else config.concept_candidate_min_confidence
        )
        self.max_verification_candidates = max(
            0,
            max_verification_candidates
            if max_verification_candidates is not None
            else config.concept_max_verification_candidates,
        )
        self.min_extraction_coverage = (
            min_extraction_coverage
            if min_extraction_coverage is not None
            else config.concept_min_extraction_coverage
        )
        if not 0.0 <= self.min_extraction_coverage <= 1.0:
            raise ValueError("min_extraction_coverage must be between 0 and 1")
        self.course_order_bonus = (
            course_order_bonus if course_order_bonus is not None
            else config.concept_course_order_bonus
        )
        self.cross_discipline_decay = (
            cross_discipline_decay if cross_discipline_decay is not None
            else config.concept_cross_discipline_decay
        )
        self.score_weight_vector = (
            score_weight_vector if score_weight_vector is not None
            else config.concept_score_weight_vector
        )
        self.score_weight_structure = (
            score_weight_structure if score_weight_structure is not None
            else config.concept_score_weight_structure
        )
        self.score_weight_rule = (
            score_weight_rule if score_weight_rule is not None
            else config.concept_score_weight_rule
        )
        configured_vote_count = (
            llm_vote_count
            if llm_vote_count is not None
            else config.concept_llm_vote_count
        )
        self.llm_vote_count = max(1, int(configured_vote_count))
        if self.llm_vote_count % 2 == 0:
            self.llm_vote_count += 1
        self.llm_vote_temperature = (
            llm_vote_temperature if llm_vote_temperature is not None
            else config.concept_llm_vote_temperature
        )
        self.api_concurrency = max(
            1,
            api_concurrency if api_concurrency is not None else config.concept_api_concurrency,
        )
        self.allow_rule_fallback = allow_rule_fallback
        self.require_complete_llm_validation = require_complete_llm_validation
        self.allow_embedding_fallback = allow_embedding_fallback
        self._normalize_score_weights()
        try:
            self.api_client = create_deepseek_client()
        except ValueError:
            # The project supports an offline mode. Concept extraction may use
            # deterministic rules there, but relation verification must remain
            # fail-closed and never promote a heuristic guess to a hard edge.
            self.api_client = None
        self.embedding_client = self.api_client
        self._wiki_cache: dict[str, str] = {}

    def _normalize_score_weights(self) -> None:
        """Ensure score component weights sum to 1.0."""
        total = (
            self.score_weight_vector
            + self.score_weight_structure
            + self.score_weight_rule
        )
        if total <= 0:
            return
        self.score_weight_vector /= total
        self.score_weight_structure /= total
        self.score_weight_rule /= total

    def preprocess_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        registry_output_path: str | Path | None = None,
        alias_output_path: str | Path | None = None,
        enriched_chunks_output_path: str | Path | None = None,
        candidate_output_path: str | Path | None = None,
        verified_output_path: str | Path | None = None,
        extraction_cache_path: str | Path | None = None,
        validation_cache_path: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
        """Extract concepts and write canonical, candidate, and verified artifacts."""
        if not chunks:
            return [], [], {}

        extraction_cache = self._load_extraction_cache(extraction_cache_path)

        def extract_chunk(chunk: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
            metadata = chunk.get("metadata", {})
            chunk_text = str(chunk.get("text", "")).strip()
            normalized_metadata = metadata if isinstance(metadata, dict) else {}
            cache_key = self._extraction_cache_key(chunk_text, normalized_metadata)
            cached = extraction_cache.get(cache_key)
            cached_is_verified_llm_output = (
                isinstance(cached, list)
                and bool(cached)
                and all(
                    isinstance(item, dict)
                    and item.get("extraction_source") == "llm"
                    for item in cached
                )
            )
            if isinstance(cached, list) and (
                self.api_client is None or cached_is_verified_llm_output
            ):
                return cache_key, cached
            concepts = self.extract_core_concepts(
                chunk_text,
                normalized_metadata,
            )
            return cache_key, concepts

        with ThreadPoolExecutor(max_workers=self.api_concurrency) as executor:
            # executor.map preserves input order, keeping generated artifacts deterministic.
            extracted_rows = list(executor.map(extract_chunk, chunks))

        extracted_by_chunk: list[list[dict[str, str]]] = []
        for cache_key, concepts in extracted_rows:
            extraction_cache[cache_key] = concepts
            extracted_by_chunk.append(concepts)
        if extraction_cache_path is not None:
            self._write_json(extraction_cache_path, extraction_cache)

        all_concepts: list[dict[str, str]] = []
        for concepts in extracted_by_chunk:
            all_concepts.extend(concepts)
        extraction_coverage = sum(bool(items) for items in extracted_by_chunk) / len(
            extracted_by_chunk
        )
        if not self.allow_rule_fallback and (
            not all_concepts
            or extraction_coverage < self.min_extraction_coverage
        ):
            raise RuntimeError(
                "concept extraction coverage was too low "
                f"({extraction_coverage:.1%} < {self.min_extraction_coverage:.1%}); "
                "existing artifacts must be preserved"
            )

        alias_table, canonical_registry = self._canonicalize_concepts(all_concepts)
        concept_lookup = {concept["canonical_name"]: concept for concept in canonical_registry}

        enriched_chunks: list[dict[str, Any]] = []
        for chunk, raw_concepts in zip(chunks, extracted_by_chunk):
            enriched_chunks.append(
                self._enrich_chunk(chunk, raw_concepts, alias_table, concept_lookup)
            )

        candidate_links = self.build_candidate_links(canonical_registry)
        verified_edges = self.verify_candidate_links(
            candidate_links,
            canonical_registry,
            validation_cache_path=validation_cache_path,
        )

        if registry_output_path is not None:
            self._write_json(registry_output_path, canonical_registry)
        if alias_output_path is not None:
            self._write_json(alias_output_path, alias_table)
        if enriched_chunks_output_path is not None:
            self._write_json(enriched_chunks_output_path, enriched_chunks)
        if candidate_output_path is not None:
            self._write_json(candidate_output_path, candidate_links)
        if verified_output_path is not None:
            self._write_json(verified_output_path, verified_edges)

        return enriched_chunks, canonical_registry, alias_table

    def extract_core_concepts(
        self,
        chunk_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """Call DeepSeek to extract core concepts from a teaching module chunk."""
        normalized_text = str(chunk_text or "").strip()
        if not normalized_text:
            return []

        metadata = metadata or {}
        prompt = (
            "提取该模块中的所有核心概念，并判断每个概念的类型"
            "(algorithm/theorem/operator/method/model/framework/data_structure/paradigm/concept)、"
            "Bloom层级(understand/apply/analyze)、所属学科。"
            "结果输出 JSON 对象："
            '{"concepts": [{"name": "...", "type": "concept", '
            '"bloom_level": "understand", "discipline": "...", '
            '"source_course": "...", "source_chapter": "..."}]}。\n'
            "约束：\n"
            "1. 只保留文本中明确出现或能直接对应的核心概念，不要输出课程名、教师名、教材名。\n"
            "2. type 只能是 algorithm、theorem、operator、method、model、framework、data_structure、paradigm、concept 之一。\n"
            "3. bloom_level 只能是 understand、apply、analyze 之一。\n"
            "4. discipline 用简洁中文或英文学科名。\n"
            "5. source_course 填该概念所在课程名；source_chapter 填该概念所在章节名。\n"
            "6. 优先使用给定模块元数据中的 course_name 和 syllabus_section 作为 source 字段。\n"
            "7. 顶层必须是只含 concepts 字段的 JSON 对象，不要额外解释。\n"
            f"模块元数据: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"模块文本:\n{normalized_text}"
        )
        try:
            response = generate_json_value(
                self.api_client,
                self.llm_model_name,
                prompt,
                temperature=0.0,
                max_output_tokens=1600,
            )
        except Exception as exc:
            action = (
                "using fallback"
                if self.allow_rule_fallback
                else "aborting authoritative run"
            )
            logger.warning(
                "[concept_normalizer] LLM concept extraction failed, %s: %s",
                action,
                exc,
            )
            return self._fallback_or_raise(normalized_text, metadata, exc)

        if isinstance(response, dict):
            response = response.get("concepts")
        if not isinstance(response, list):
            return self._fallback_or_raise(
                normalized_text,
                metadata,
                ValueError(
                    "concept extractor did not return a JSON object with a concepts array"
                ),
            )

        normalized_concepts: list[dict[str, str]] = []
        for item in response:
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_extracted_concept(item, metadata)
            if normalized is not None:
                normalized["extraction_source"] = "llm"
                normalized_concepts.append(normalized)

        if response and not normalized_concepts:
            rejection_counts = self._concept_rejection_counts(response, metadata)
            # A well-formed response containing only template/course-name noise
            # is a legitimate empty extraction for this chunk. The aggregate
            # coverage gate later decides whether enough chunks were useful.
            if (
                set(rejection_counts) == {"noise_name"}
                and rejection_counts["noise_name"] == len(response)
            ):
                return []
            rejection_summary = self._format_rejection_counts(rejection_counts)
            return self._fallback_or_raise(
                normalized_text,
                metadata,
                ValueError(
                    "concept extractor returned no schema-valid entries "
                    f"(rejections: {rejection_summary})"
                ),
            )
        return normalized_concepts

    def _fallback_or_raise(
        self,
        chunk_text: str,
        metadata: dict[str, Any],
        error: Exception,
    ) -> list[dict[str, str]]:
        if not self.allow_rule_fallback:
            cause = self._safe_extraction_error_summary(error)
            raise RuntimeError(
                "verified concept extraction was unavailable "
                f"(cause: {cause}); existing artifacts must be preserved"
            ) from error
        return self._fallback_extract_core_concepts(chunk_text, metadata)

    @staticmethod
    def _safe_extraction_error_summary(error: Exception) -> str:
        """Describe extraction failures without exposing prompts or credentials."""
        error_type = type(error).__name__
        details: list[str] = []

        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            details.append(f"HTTP {status_code}")

        # These messages originate inside this module/client and never contain
        # course text. Third-party exception messages are deliberately omitted.
        message = str(error).strip()
        safe_prefixes = (
            "DeepSeek returned empty or invalid JSON",
            "Model output does not contain",
            "concept extractor did not return",
            "concept extractor returned no schema-valid entries",
        )
        if any(message.startswith(prefix) for prefix in safe_prefixes):
            details.append(message)

        return ": ".join([error_type, *details])

    def build_candidate_links(self, canonical_registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build and merge cross-course source concept candidates."""
        if not canonical_registry:
            return []

        descriptions = [self._build_concept_description(item) for item in canonical_registry]
        embeddings = embed_texts(
            self.embedding_client,
            self.retrieval_model_name,
            descriptions,
            batch_size=min(32, max(1, len(descriptions))),
            allow_hash_fallback=self.allow_embedding_fallback,
        )
        pair_candidates = self._build_pair_candidates(canonical_registry, embeddings)
        merged_candidates = self._merge_candidate_pairs(pair_candidates)
        merged_candidates.sort(
            key=lambda item: (-item["initial_confidence"], item["target_id"], item["source_id"])
        )
        return merged_candidates

    def verify_candidate_links(
        self,
        candidate_links: list[dict[str, Any]],
        canonical_registry: list[dict[str, Any]],
        *,
        validation_cache_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Validate merged candidate links with LLM self-consistency voting."""
        if not candidate_links:
            return []

        concept_lookup = {item["id"]: item for item in canonical_registry}
        validation_cache = self._load_validation_cache(validation_cache_path)
        cached_results: list[tuple[str, dict[str, Any]]] = []
        uncached_candidates: list[dict[str, Any]] = []
        for candidate in candidate_links:
            source = concept_lookup.get(candidate.get("source_id"))
            target = concept_lookup.get(candidate.get("target_id"))
            if source is None or target is None:
                continue
            cache_key = self._validation_cache_key(candidate, source, target)
            cached = validation_cache.get(cache_key)
            if (
                isinstance(cached, dict)
                and cached.get("verification_source") == "llm"
                and cached.get("source_id") == candidate.get("source_id")
                and cached.get("target_id") == candidate.get("target_id")
            ):
                cached_results.append((cache_key, cached))
            else:
                uncached_candidates.append(candidate)
        selected_candidates = uncached_candidates[
            : self.max_verification_candidates
        ]

        def verify_candidate(
            candidate: dict[str, Any],
        ) -> tuple[str, dict[str, Any] | None]:
            source = concept_lookup.get(candidate["source_id"])
            target = concept_lookup.get(candidate["target_id"])
            if source is None or target is None:
                return "", None
            cache_key = self._validation_cache_key(candidate, source, target)
            return cache_key, self._llm_verify_candidate(candidate, source, target)

        with ThreadPoolExecutor(max_workers=self.api_concurrency) as executor:
            verified_results = [
                *cached_results,
                *executor.map(verify_candidate, selected_candidates),
            ]

        verified_edges: list[dict[str, Any]] = []
        for cache_key, edge in verified_results:
            if edge is None:
                continue
            verified_edges.append(edge)
            # A transient provider failure must not become a durable validation
            # result. Only complete LLM votes are safe to reuse.
            if edge.get("verification_source") == "llm":
                validation_cache[cache_key] = edge
        if validation_cache_path is not None:
            self._write_json(validation_cache_path, validation_cache)

        verified_edges.sort(
            key=lambda item: (-item["confidence"], item["target_id"], item["source_id"])
        )
        return verified_edges

    def _build_pair_candidates(
        self,
        canonical_registry: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Generate candidate rows before pair-wise merge."""
        candidates: list[dict[str, Any]] = []
        for target_index, target in enumerate(canonical_registry):
            per_target: list[dict[str, Any]] = []
            for source_index, source in enumerate(canonical_registry):
                if target_index == source_index:
                    continue
                candidate = self._score_candidate(
                    target=target,
                    source=source,
                    target_embedding=embeddings[target_index],
                    source_embedding=embeddings[source_index],
                )
                if candidate is not None:
                    per_target.append(candidate)

            per_target.sort(
                key=lambda item: (
                    -item["initial_confidence"],
                    -item["score_components"]["S_vector"],
                    item["source_id"],
                )
            )
            candidates.extend(per_target[: self.candidate_top_k])
        return candidates

    def _score_candidate(
        self,
        *,
        target: dict[str, Any],
        source: dict[str, Any],
        target_embedding: list[float],
        source_embedding: list[float],
    ) -> dict[str, Any] | None:
        """Score one candidate source concept for a target concept."""
        if not self._bloom_allowed(source.get("bloom_level", ""), target.get("bloom_level", "")):
            return None
        if self._share_any_value(
            target.get("source_course_codes", []),
            source.get("source_course_codes", []),
        ):
            return None

        cosine_score = max(0.0, self._cosine_similarity(target_embedding, source_embedding))
        structure_signal = self._structure_signal(source, target)
        rule_signal = self._rule_signal(source, target)
        same_discipline = source.get("discipline") == target.get("discipline")
        domain_factor = 1.0 if same_discipline else self.cross_discipline_decay

        raw_score = (
            self.score_weight_vector * cosine_score
            + self.score_weight_structure * structure_signal
            + self.score_weight_rule * rule_signal
        )
        initial_confidence = self._clamp01(raw_score * domain_factor)
        if initial_confidence < self.candidate_min_confidence:
            return None

        evidence = [
            {
                "type": "vector_similarity",
                "value": round(cosine_score, 6),
                "detail": f"{source['canonical_name']} -> {target['canonical_name']}",
            },
            {
                "type": "course_order",
                "value": round(structure_signal, 6),
                "detail": self._course_order_evidence(source, target),
            },
            {
                "type": "rule_filter",
                "value": round(rule_signal, 6),
                "detail": (
                    f"same_discipline={same_discipline}, "
                    f"source_bloom={source['bloom_level']}, target_bloom={target['bloom_level']}"
                ),
            },
            {
                "type": "domain_decay",
                "value": round(domain_factor, 6),
                "detail": "same_discipline" if same_discipline else "cross_discipline_decay_applied",
            },
        ]

        return {
            "source_id": source["id"],
            "source_name": source["canonical_name"],
            "source_courses": source.get("source_courses", []),
            "source_course_codes": source.get("source_course_codes", []),
            "source_bloom_level": source["bloom_level"],
            "source_discipline": source["discipline"],
            "target_id": target["id"],
            "target_name": target["canonical_name"],
            "target_courses": target.get("source_courses", []),
            "target_course_codes": target.get("source_course_codes", []),
            "target_bloom_level": target["bloom_level"],
            "target_discipline": target["discipline"],
            "score_components": {
                "S_vector": round(cosine_score, 6),
                "S_structure": round(structure_signal, 6),
                "S_rule": round(rule_signal, 6),
            },
            "weights": {
                "w1_vector": round(self.score_weight_vector, 6),
                "w2_structure": round(self.score_weight_structure, 6),
                "w3_rule": round(self.score_weight_rule, 6),
            },
            "domain_decay_factor": round(domain_factor, 6),
            "fusion_score": round(raw_score, 6),
            "initial_confidence": round(initial_confidence, 6),
            "evidence": evidence,
        }

    def _merge_candidate_pairs(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge duplicate candidate pairs by (source_id, target_id)."""
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            grouped[(candidate["source_id"], candidate["target_id"])].append(candidate)

        merged: list[dict[str, Any]] = []
        for (source_id, target_id), items in grouped.items():
            best_item = max(items, key=lambda item: item["initial_confidence"])
            merged.append(
                {
                    "source_id": source_id,
                    "source_name": best_item["source_name"],
                    "source_courses": self._merge_string_lists(items, "source_courses"),
                    "source_course_codes": self._merge_string_lists(items, "source_course_codes"),
                    "source_bloom_level": best_item["source_bloom_level"],
                    "source_discipline": best_item["source_discipline"],
                    "target_id": target_id,
                    "target_name": best_item["target_name"],
                    "target_courses": self._merge_string_lists(items, "target_courses"),
                    "target_course_codes": self._merge_string_lists(items, "target_course_codes"),
                    "target_bloom_level": best_item["target_bloom_level"],
                    "target_discipline": best_item["target_discipline"],
                    "fusion_score": round(max(item["fusion_score"] for item in items), 6),
                    "initial_confidence": round(max(item["initial_confidence"] for item in items), 6),
                    "score_formula": "Score = w1*S_vector + w2*S_structure + w3*S_rule",
                    "weights": best_item["weights"],
                    "domain_decay_factor": round(min(item["domain_decay_factor"] for item in items), 6),
                    "score_components": {
                        "S_vector": round(max(item["score_components"]["S_vector"] for item in items), 6),
                        "S_structure": round(max(item["score_components"]["S_structure"] for item in items), 6),
                        "S_rule": round(max(item["score_components"]["S_rule"] for item in items), 6),
                    },
                    "evidence": self._merge_evidence_lists(items),
                    "duplicate_count": len(items),
                }
            )
        return merged

    def _llm_verify_candidate(
        self,
        candidate: dict[str, Any],
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, Any]:
        """Run self-consistency LLM validation on one merged candidate pair."""
        votes: list[dict[str, Any]] = []
        for _ in range(self.llm_vote_count):
            if self.api_client is None:
                if self.require_complete_llm_validation:
                    raise RuntimeError("concept dependency validator is unavailable")
                votes.append(self._fallback_dependency_vote(candidate))
                continue
            try:
                vote = self._call_dependency_validator(candidate, source, target)
            except Exception as exc:
                if self.require_complete_llm_validation:
                    raise RuntimeError(
                        "concept dependency validation was incomplete; existing "
                        "artifacts must be preserved"
                    ) from exc
                logger.warning(
                    "[concept_normalizer] LLM dependency vote failed, using fallback: %s", exc
                )
                vote = self._fallback_dependency_vote(candidate)
            votes.append(vote)

        requires_counter = Counter(vote.get("requires") is True for vote in votes)
        requires = requires_counter.most_common(1)[0][0]
        relation_votes = [
            str(vote.get("relation_type", "NO_RELATION")).strip()
            for vote in votes
            if vote.get("requires") is True
            and vote.get("relation_type")
            and str(vote.get("relation_type")).strip() != "NO_RELATION"
        ]
        relation_type = Counter(relation_votes).most_common(1)[0][0] if relation_votes else "NO_RELATION"
        if relation_type not in self.ALLOWED_RELATION_TYPES:
            relation_type = "NO_RELATION"
        supporting_votes = [vote for vote in votes if vote.get("requires") is True]
        if requires and relation_type == "NO_RELATION":
            requires = False
        confidence_votes = supporting_votes if requires else votes
        average_confidence = sum(
            self._safe_float(vote.get("confidence", 0.0)) for vote in confidence_votes
        ) / max(len(confidence_votes), 1)
        reason = self._choose_majority_reason(
            supporting_votes if requires else votes
        )

        return {
            "source_id": candidate["source_id"],
            "target_id": candidate["target_id"],
            "requires": requires,
            "relation_type": relation_type if requires else "NO_RELATION",
            "confidence": round(self._clamp01(average_confidence), 6),
            "reason": reason,
            "candidate_confidence": candidate["initial_confidence"],
            "fusion_score": candidate["fusion_score"],
            "evidence": candidate["evidence"],
            "llm_votes": votes,
            "verification_source": (
                "llm"
                if all(vote.get("verification_source") == "llm" for vote in votes)
                else "fallback"
            ),
        }

    def _call_dependency_validator(
        self,
        candidate: dict[str, Any],
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, Any]:
        """Call DeepSeek with a structured dependency-validation prompt."""
        prompt = (
            "你是一名课程知识依赖判定专家。请判断 source concept 是否构成 target concept 的真实教学依赖，"
            "而不是仅仅语义相关。\n"
            "请从以下维度分析：\n"
            "1. 定义依赖：目标概念的定义中是否显式或隐含使用了源概念？\n"
            "2. 推导依赖：理解目标概念的核心公式/算法是否需要源概念？\n"
            "3. 可替代性：是否容易用别的概念替代源概念来理解目标？\n"
            "4. Bloom层级比较：源概念认知要求是否 <= 目标概念认知要求？\n"
            "5. 教学实践：典型教材是否常将源概念列为前置知识？\n"
            "允许的 relation_type: FOUNDATION_OF, METHOD_ANALOGY, TOOL_PREREQ, CONCEPTUAL_BASIS, NO_RELATION。\n"
            "若不构成依赖，requires=false 且 relation_type=NO_RELATION。\n"
            "输出 JSON 对象，字段必须包含：\n"
            "{requires, relation_type, confidence, reason, dimensions}\n"
            "其中 dimensions 为对象，字段包括：\n"
            "{definition_dependency, derivation_dependency, substitutable, bloom_compatible, teaching_practice}\n"
            "布尔字段用 true/false；confidence 为 0~1 小数；reason 简洁说明。\n"
            "不要输出 JSON 以外的内容。\n"
            f"source concept: {json.dumps(source, ensure_ascii=False)}\n"
            f"target concept: {json.dumps(target, ensure_ascii=False)}\n"
            f"candidate evidence: {json.dumps(candidate, ensure_ascii=False)}"
        )
        raw = generate_json_value(
            self.api_client,
            self.llm_model_name,
            prompt,
            temperature=self.llm_vote_temperature,
            max_output_tokens=900,
        )
        if not isinstance(raw, dict):
            raise ValueError("Dependency validator did not return a JSON object.")
        normalized = self._normalize_dependency_vote(raw, candidate)
        normalized["verification_source"] = "llm"
        return normalized

    def _normalize_dependency_vote(
        self,
        vote: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize one validator vote."""
        relation_type = str(vote.get("relation_type", "NO_RELATION")).strip().upper()
        if relation_type not in self.ALLOWED_RELATION_TYPES:
            relation_type = "NO_RELATION"
        requires = vote.get("requires") is True
        confidence = self._clamp01(self._safe_float(vote.get("confidence", candidate["initial_confidence"])))
        reason = str(vote.get("reason", "")).strip() or "No reason provided."
        dimensions = vote.get("dimensions", {})
        if not isinstance(dimensions, dict):
            dimensions = {}

        normalized_dimensions = {
            "definition_dependency": dimensions.get("definition_dependency") is True,
            "derivation_dependency": dimensions.get("derivation_dependency") is True,
            "substitutable": dimensions.get("substitutable") is True,
            "bloom_compatible": dimensions.get("bloom_compatible", True) is True,
            "teaching_practice": dimensions.get("teaching_practice") is True,
        }
        if not requires:
            relation_type = "NO_RELATION"

        return {
            "requires": requires,
            "relation_type": relation_type,
            "confidence": round(confidence, 6),
            "reason": reason,
            "dimensions": normalized_dimensions,
        }

    def _fallback_dependency_vote(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Fail closed when the dependency validator is unavailable."""
        return {
            "requires": False,
            "relation_type": "NO_RELATION",
            "confidence": 0.0,
            "reason": "Dependency validator unavailable; candidate kept unverified.",
            "dimensions": {
                "definition_dependency": False,
                "derivation_dependency": False,
                "substitutable": True,
                "bloom_compatible": True,
                "teaching_practice": False,
            },
            "verification_source": "fallback",
        }

    def _choose_majority_reason(self, votes: list[dict[str, Any]]) -> str:
        """Pick the most frequent reason text, preferring longer explanations on ties."""
        reasons = [str(vote.get("reason", "")).strip() for vote in votes if str(vote.get("reason", "")).strip()]
        if not reasons:
            return ""
        counts = Counter(reasons)
        return sorted(reasons, key=lambda item: (-counts[item], -len(item), item))[0]

    def _structure_signal(self, source: dict[str, Any], target: dict[str, Any]) -> float:
        """Normalize the course ordering prior to [0, 1]."""
        return self._clamp01(self._course_order_bonus_for_pair(source, target))

    def _rule_signal(self, source: dict[str, Any], target: dict[str, Any]) -> float:
        """Combine rule-based signals into a normalized score."""
        same_discipline = source.get("discipline") == target.get("discipline")
        bloom_ok = self._bloom_allowed(source.get("bloom_level", ""), target.get("bloom_level", ""))
        if not bloom_ok:
            return 0.0
        return 1.0 if same_discipline else 0.5

    def _bloom_allowed(self, source_level: str, target_level: str) -> bool:
        """Keep only candidates whose Bloom level is not above the target.

        When the target level is unrecognised (empty string, LLM parse error,
        etc.) we default to 999 so the pair is accepted rather than silently
        dropped — a strict filter on unknown data would eliminate all candidates.
        """
        return self.BLOOM_ORDER.get(source_level, 999) <= self.BLOOM_ORDER.get(target_level, 999)

    def _course_order_bonus_for_pair(self, source: dict[str, Any], target: dict[str, Any]) -> float:
        """Add bonus when the source concept comes from an earlier course."""
        source_orders = [
            self._extract_course_order(code)
            for code in source.get("source_course_codes", [])
        ]
        target_orders = [
            self._extract_course_order(code)
            for code in target.get("source_course_codes", [])
        ]
        source_orders = [value for value in source_orders if value is not None]
        target_orders = [value for value in target_orders if value is not None]
        if not source_orders or not target_orders:
            return 0.0

        for source_order in source_orders:
            for target_order in target_orders:
                if source_order < target_order:
                    return self.course_order_bonus
        return 0.0

    def _course_order_evidence(self, source: dict[str, Any], target: dict[str, Any]) -> str:
        """Describe the course-order relation used as prior."""
        source_codes = ",".join(source.get("source_course_codes", []))
        target_codes = ",".join(target.get("source_course_codes", []))
        if self._course_order_bonus_for_pair(source, target) > 0:
            return f"{source_codes} precedes {target_codes}"
        return f"no prerequisite advantage between {source_codes} and {target_codes}"

    def _build_concept_description(self, concept: dict[str, Any]) -> str:
        """Build a dense retrieval description for one canonical concept."""
        aliases = ", ".join(concept.get("aliases", []))
        courses = ", ".join(concept.get("source_courses", []))
        chapters = ", ".join(concept.get("source_chapters", []))
        course_codes = ", ".join(concept.get("source_course_codes", []))
        return (
            f"concept:{concept['canonical_name']}\n"
            f"aliases:{aliases}\n"
            f"type:{concept['type']}\n"
            f"bloom:{concept['bloom_level']}\n"
            f"discipline:{concept['discipline']}\n"
            f"courses:{courses}\n"
            f"course_codes:{course_codes}\n"
            f"chapters:{chapters}"
        )

    def _fallback_extract_core_concepts(
        self,
        chunk_text: str,
        metadata: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Use simple heuristics when the LLM is unavailable."""
        discipline = str(metadata.get("course_name", "")).strip() or "unknown"
        source_course = self._default_source_course(metadata)
        source_chapter = self._default_source_chapter(metadata)
        source_course_code = self._default_source_course_code(metadata)
        concepts: list[dict[str, str]] = []
        seen_names: set[str] = set()

        type_keywords = [
            (r"[《""]?([A-Za-z\u4e00-\u9fff0-9_+/-]{2,40}(?:算法))", "algorithm"),
            (r"[《""]?([A-Za-z\u4e00-\u9fff0-9_+/-]{2,40}(?:定理))", "theorem"),
            (r"[《""]?([A-Za-z\u4e00-\u9fff0-9_+/-]{1,40}(?:算子|operator))", "operator"),
            (r"[《""]?([A-Za-z\u4e00-\u9fff0-9_+/-]{2,40}(?:方法|方法学))", "method"),
            (r"[《""]?([A-Za-z\u4e00-\u9fff0-9_+/-]{2,40}(?:模型))", "model"),
            (r"[《""]?([A-Za-z\u4e00-\u9fff0-9_+/-]{2,40}(?:框架))", "framework"),
            (r"[《""]?([A-Za-z\u4e00-\u9fff0-9_+/-]{2,40}(?:结构|数据结构))", "data_structure"),
        ]
        for pattern, concept_type in type_keywords:
            for match in re.findall(pattern, chunk_text, flags=re.IGNORECASE):
                name = str(match).strip("《》“”\"'` ")
                if not name or self._is_noise_concept(name, metadata):
                    continue
                normalized_key = self._normalize_alias_key(name)
                if normalized_key in seen_names:
                    continue
                seen_names.add(normalized_key)
                concepts.append(
                    {
                        "name": name,
                        "type": concept_type,
                        "bloom_level": "understand",
                        "discipline": discipline,
                        "source_course": source_course,
                        "source_chapter": source_chapter,
                        "source_course_code": source_course_code,
                        "extraction_source": "rule",
                    }
                )

        return concepts

    def _normalize_extracted_concept(
        self,
        concept: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, str] | None:
        """Validate and normalize a raw LLM concept entry."""
        name = self._cleanup_term(
            self._first_present(concept, "name", "concept_name", "概念名称")
        )
        concept_type = self._normalize_enum(
            self._first_present(concept, "type", "concept_type", "概念类型"),
            self.TYPE_ALIASES,
        )
        bloom_level = self._normalize_enum(
            self._first_present(
                concept,
                "bloom_level",
                "bloom",
                "Bloom层级",
                "认知层级",
            ),
            self.BLOOM_ALIASES,
        )
        # Preserve a valid concept when the model invents a non-empty subtype.
        # ``concept`` is the ontology's explicit catch-all; coercing to a more
        # specific category would overstate what the response established.
        if concept_type and concept_type not in self.ALLOWED_TYPES:
            concept_type = "concept"
        discipline = self._cleanup_term(
            self._first_present(concept, "discipline", "domain", "学科")
        )
        source_course = self._default_source_course(metadata) or self._cleanup_term(
            concept.get("source_course", "")
        )
        source_chapter = self._default_source_chapter(metadata) or self._cleanup_term(
            concept.get("source_chapter", "")
        )
        source_course_code = self._default_source_course_code(metadata)

        if (
            not name
            or self._is_noise_concept(name, metadata)
            or not concept_type
            or bloom_level not in self.ALLOWED_BLOOM_LEVELS
        ):
            return None
        if not discipline:
            discipline = "unknown"

        return {
            "name": name,
            "type": concept_type,
            "bloom_level": bloom_level,
            "discipline": discipline,
            "source_course": source_course,
            "source_chapter": source_chapter,
            "source_course_code": source_course_code,
        }

    def _concept_rejection_counts(
        self,
        concepts: list[Any],
        metadata: dict[str, Any],
    ) -> Counter[str]:
        """Count schema rejection reasons without logging generated content."""
        reasons: Counter[str] = Counter()
        for concept in concepts:
            if not isinstance(concept, dict):
                reasons["not_object"] += 1
                continue
            name = self._cleanup_term(
                self._first_present(concept, "name", "concept_name", "概念名称")
            )
            concept_type = self._normalize_enum(
                self._first_present(concept, "type", "concept_type", "概念类型"),
                self.TYPE_ALIASES,
            )
            bloom_level = self._normalize_enum(
                self._first_present(
                    concept,
                    "bloom_level",
                    "bloom",
                    "Bloom层级",
                    "认知层级",
                ),
                self.BLOOM_ALIASES,
            )
            if not name:
                reasons["missing_name"] += 1
            elif self._is_noise_concept(name, metadata):
                reasons["noise_name"] += 1
            if not concept_type:
                reasons["missing_type"] += 1
            if bloom_level not in self.ALLOWED_BLOOM_LEVELS:
                reasons["invalid_bloom_level"] += 1
        return reasons

    @staticmethod
    def _format_rejection_counts(reasons: Counter[str]) -> str:
        return ", ".join(f"{key}={value}" for key, value in sorted(reasons.items()))

    @staticmethod
    def _first_present(concept: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = concept.get(key)
            if value is not None and str(value).strip():
                return value
        return ""

    @staticmethod
    def _normalize_enum(value: Any, aliases: dict[str, str]) -> str:
        normalized = str(value or "").strip().lower()
        return aliases.get(normalized, normalized)

    def _canonicalize_concepts(
        self,
        concepts: list[dict[str, str]],
    ) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
        """Cluster aliases and produce canonical concept records with unique IDs."""
        if not concepts:
            return {}, []

        concept_occurrences: dict[str, list[dict[str, str]]] = defaultdict(list)
        unique_names: list[str] = []
        for concept in concepts:
            name = concept["name"]
            if name not in concept_occurrences:
                unique_names.append(name)
            concept_occurrences[name].append(concept)

        groups = self._cluster_names(unique_names)
        alias_table: dict[str, list[str]] = {}
        canonical_registry: list[dict[str, Any]] = []

        for group in groups:
            aliases = sorted(group, key=lambda item: (len(item), item.lower()))
            wiki_candidates = [self._lookup_wikipedia_title(alias) for alias in aliases]
            canonical_name = self._choose_canonical_name(aliases, wiki_candidates)
            member_concepts: list[dict[str, str]] = []
            for alias in aliases:
                member_concepts.extend(concept_occurrences[alias])

            alias_table[canonical_name] = aliases
            concept_key = self._normalize_alias_key(canonical_name)
            concept_id = "concept_" + hashlib.sha256(
                concept_key.encode("utf-8")
            ).hexdigest()[:16]
            canonical_registry.append(
                {
                    "id": concept_id,
                    "canonical_name": canonical_name,
                    "aliases": aliases,
                    "type": self._majority_value(member_concepts, "type", default="algorithm"),
                    "bloom_level": self._majority_value(member_concepts, "bloom_level", default="understand"),
                    "discipline": self._majority_value(member_concepts, "discipline", default="unknown"),
                    "source_courses": self._unique_values(member_concepts, "source_course"),
                    "source_chapters": self._unique_values(member_concepts, "source_chapter"),
                    "source_course_codes": self._unique_values(member_concepts, "source_course_code"),
                    "source_occurrences": self._unique_source_occurrences(member_concepts),
                    "extraction_sources": self._unique_values(member_concepts, "extraction_source"),
                }
            )

        canonical_registry.sort(key=lambda item: (item["canonical_name"].casefold(), item["id"]))
        return alias_table, canonical_registry

    def _load_extraction_cache(
        self,
        extraction_cache_path: str | Path | None,
    ) -> dict[str, list[dict[str, str]]]:
        if extraction_cache_path is None:
            return {}
        path = Path(extraction_cache_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_validation_cache(
        self,
        validation_cache_path: str | Path | None,
    ) -> dict[str, dict[str, Any]]:
        if validation_cache_path is None:
            return {}
        path = Path(validation_cache_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _validation_cache_key(
        self,
        candidate: dict[str, Any],
        source: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "prompt_version": DEPENDENCY_PROMPT_VERSION,
            "model": self.llm_model_name,
            "vote_count": self.llm_vote_count,
            "vote_temperature": self.llm_vote_temperature,
            "source_id": candidate.get("source_id"),
            "target_id": candidate.get("target_id"),
            "fusion_score": candidate.get("fusion_score"),
            "initial_confidence": candidate.get("initial_confidence"),
            "evidence": candidate.get("evidence", []),
            "source": source or {},
            "target": target or {},
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _extraction_cache_key(
        self,
        chunk_text: str,
        metadata: dict[str, Any],
    ) -> str:
        payload = {
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "model": self.llm_model_name,
            "mode": "llm" if self.api_client is not None else "fallback",
            "text": chunk_text,
            "course_code": metadata.get("course_code", ""),
            "course_name": metadata.get("course_name", ""),
            "section": metadata.get("syllabus_section", ""),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _is_noise_concept(self, name: str, metadata: dict[str, Any]) -> bool:
        normalized = self._normalize_alias_key(name)
        course_name = self._normalize_alias_key(metadata.get("course_name", ""))
        noise = {self._normalize_alias_key(item) for item in _CONCEPT_NOISE}
        return normalized in noise or bool(course_name and normalized == course_name)

    def _cluster_names(self, names: list[str]) -> list[list[str]]:
        """Group names using exact normalization and embedding similarity."""
        if not names:
            return []

        parent = list(range(len(names)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        normalized_buckets: dict[str, list[int]] = defaultdict(list)
        for index, name in enumerate(names):
            normalized_buckets[self._normalize_alias_key(name)].append(index)
        for bucket in normalized_buckets.values():
            for index in bucket[1:]:
                union(bucket[0], index)

        embeddings = embed_texts(
            self.embedding_client,
            self.embedding_model_name,
            names,
            batch_size=min(32, max(1, len(names))),
            allow_hash_fallback=self.allow_embedding_fallback,
        )
        if embeddings:
            # Vectorised cosine similarity via numpy: embeddings are already L2-normalised
            # so cosine_sim(a, b) == dot(a, b).  One BLAS call replaces n*(n-1)/2 Python calls.
            mat = np.array(embeddings, dtype=np.float32)
            sim_matrix = np.dot(mat, mat.T)
            rows, cols = np.where(np.triu(sim_matrix, k=1) >= self.similarity_threshold)
            for left, right in zip(rows.tolist(), cols.tolist()):
                union(int(left), int(right))

        grouped: dict[int, list[str]] = defaultdict(list)
        for index, name in enumerate(names):
            grouped[find(index)].append(name)
        return sorted(grouped.values(), key=lambda item: min(name.lower() for name in item))

    def _enrich_chunk(
        self,
        chunk: dict[str, Any],
        raw_concepts: list[dict[str, str]],
        alias_table: dict[str, list[str]],
        concept_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Attach canonicalized concept metadata and embedding text to a chunk."""
        enriched_chunk = dict(chunk)
        metadata = dict(chunk.get("metadata", {}))

        canonical_concepts: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for concept in raw_concepts:
            canonical_name = self._resolve_canonical_name(concept["name"], alias_table)
            canonical_record = concept_lookup.get(canonical_name)
            if canonical_record is None or canonical_record["id"] in seen_ids:
                continue
            seen_ids.add(canonical_record["id"])
            canonical_concepts.append(
                {
                    "id": canonical_record["id"],
                    "name": canonical_record["canonical_name"],
                    "type": canonical_record["type"],
                    "bloom_level": canonical_record["bloom_level"],
                    "discipline": canonical_record["discipline"],
                    "source_course": concept.get("source_course", ""),
                    "source_chapter": concept.get("source_chapter", ""),
                    "source_course_code": concept.get("source_course_code", ""),
                }
            )

        metadata["core_concepts_raw"] = raw_concepts
        metadata["core_concepts"] = canonical_concepts
        metadata["core_concept_ids"] = [item["id"] for item in canonical_concepts]
        metadata["core_concept_names"] = [item["name"] for item in canonical_concepts]
        enriched_chunk["metadata"] = metadata
        enriched_chunk["embedding_text"] = self._build_embedding_text(str(chunk.get("text", "")), canonical_concepts)
        return enriched_chunk

    def _build_embedding_text(self, text: str, canonical_concepts: list[dict[str, str]]) -> str:
        """Append canonical concept descriptors to the embedding text."""
        normalized_text = str(text).strip()
        if not canonical_concepts:
            return normalized_text

        concept_lines = [
            (
                f"{concept['id']}|{concept['name']}|{concept['type']}|"
                f"{concept['bloom_level']}|{concept['discipline']}|"
                f"{concept.get('source_course', '')}|{concept.get('source_course_code', '')}|"
                f"{concept.get('source_chapter', '')}"
            )
            for concept in canonical_concepts
        ]
        return f"{normalized_text}\n\n核心概念:\n" + "\n".join(concept_lines)

    def _resolve_canonical_name(self, alias: str, alias_table: dict[str, list[str]]) -> str:
        """Resolve an alias back to its canonical name."""
        normalized_alias = self._normalize_alias_key(alias)
        for canonical_name, aliases in alias_table.items():
            if any(self._normalize_alias_key(item) == normalized_alias for item in aliases):
                return canonical_name
        return alias

    def _choose_canonical_name(self, aliases: list[str], wiki_candidates: list[str]) -> str:
        """Choose a canonical term from alias and Wikipedia evidence."""
        for candidate in wiki_candidates:
            if candidate:
                return candidate
        counts = Counter(aliases)
        return sorted(aliases, key=lambda item: (-counts[item], -self._contains_cjk(item), len(item), item.lower()))[0]

    def _lookup_wikipedia_title(self, term: str) -> str:
        """Query Wikipedia and return the best-matching title when available."""
        cleaned_term = self._cleanup_term(term)
        if not cleaned_term or not self.wikipedia_enabled:
            return ""
        if cleaned_term in self._wiki_cache:
            return self._wiki_cache[cleaned_term]

        title = ""
        for language in ("zh", "en"):
            title = self._request_wikipedia_title(cleaned_term, language)
            if title:
                break
        self._wiki_cache[cleaned_term] = title
        return title

    def _request_wikipedia_title(self, term: str, language: str) -> str:
        """Call the MediaWiki search API and prefer near-exact title matches."""
        params = urllib.parse.urlencode({"action": "query", "list": "search", "format": "json", "srlimit": 5, "srsearch": term})
        url = f"https://{language}.wikipedia.org/w/api.php?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": "zsttSystem/1.1 concept-normalizer"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return ""

        search_results = payload.get("query", {}).get("search", [])
        if not isinstance(search_results, list):
            return ""

        normalized_term = self._normalize_alias_key(term)
        for item in search_results:
            title = self._cleanup_term(item.get("title", ""))
            if title and self._normalize_alias_key(title) == normalized_term:
                return title
        return self._cleanup_term(search_results[0].get("title", "")) if search_results else ""

    def _majority_value(self, concepts: list[dict[str, str]], key: str, *, default: str) -> str:
        """Return the most common value for a concept attribute."""
        values = [concept.get(key, "").strip() for concept in concepts if concept.get(key, "").strip()]
        return Counter(values).most_common(1)[0][0] if values else default

    def _unique_values(self, concepts: list[dict[str, str]], key: str) -> list[str]:
        """Collect unique non-empty values while preserving insertion order."""
        values: list[str] = []
        for concept in concepts:
            value = concept.get(key, "").strip()
            if value and value not in values:
                values.append(value)
        return values

    def _unique_source_occurrences(
        self,
        concepts: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Keep course/chapter provenance paired instead of forming a cross product."""
        occurrences: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for concept in concepts:
            occurrence = {
                "course": str(concept.get("source_course", "")).strip(),
                "course_code": str(concept.get("source_course_code", "")).strip(),
                "chapter": str(concept.get("source_chapter", "")).strip(),
            }
            key = (
                occurrence["course"],
                occurrence["course_code"],
                occurrence["chapter"],
            )
            if any(key) and key not in seen:
                occurrences.append(occurrence)
                seen.add(key)
        return occurrences

    def _merge_string_lists(self, items: list[dict[str, Any]], key: str) -> list[str]:
        """Merge list[str] fields from multiple candidate rows."""
        merged: list[str] = []
        for item in items:
            for value in item.get(key, []):
                if value not in merged:
                    merged.append(value)
        return merged

    def _merge_evidence_lists(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate evidence entries across duplicate candidate rows."""
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, float]] = set()
        for item in items:
            for evidence in item.get("evidence", []):
                key = (str(evidence.get("type", "")), str(evidence.get("detail", "")), float(evidence.get("value", 0.0)))
                if key not in seen:
                    seen.add(key)
                    merged.append(evidence)
        return merged

    def _default_source_course(self, metadata: dict[str, Any]) -> str:
        """Derive the default course label from chunk metadata."""
        return self._cleanup_term(metadata.get("course_name", ""))

    def _default_source_chapter(self, metadata: dict[str, Any]) -> str:
        """Derive the default chapter label from chunk metadata."""
        return self._cleanup_term(metadata.get("syllabus_section", ""))

    def _default_source_course_code(self, metadata: dict[str, Any]) -> str:
        """Derive the default course code from chunk metadata."""
        return self._cleanup_term(metadata.get("course_code", ""))

    def _extract_course_order(self, course_code: str) -> int | None:
        """Extract the first digit from a course code as the study-order prior."""
        match = re.search(r"\d", str(course_code or ""))
        return int(match.group(0)) if match is not None else None

    def _share_any_value(self, left: list[str], right: list[str]) -> bool:
        """Return whether two string lists overlap."""
        return bool(set(left) & set(right))

    def _normalize_alias_key(self, text: str) -> str:
        """Normalize aliases for case-insensitive and punctuation-insensitive matching."""
        normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
        normalized = normalized.replace("（", "(").replace("）", ")")
        return re.sub(r"[\s_\-–—,，.。:：;；/\\()\[\]{}'\"`]+", "", normalized)

    def _cleanup_term(self, value: Any) -> str:
        """Trim concept strings and collapse internal whitespace."""
        text = unicodedata.normalize("NFKC", str(value or ""))
        return re.sub(r"\s+", " ", text).strip()

    def _contains_cjk(self, text: str) -> int:
        """Return 1 when the term contains CJK characters, else 0."""
        return 1 if re.search(r"[\u4e00-\u9fff]", text) else 0

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        """Compute cosine similarity for two dense vectors."""
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        return 0.0 if left_norm == 0 or right_norm == 0 else numerator / (left_norm * right_norm)

    def _safe_float(self, value: Any) -> float:
        """Convert a value to float safely."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _clamp01(self, value: float) -> float:
        """Clamp a score into [0, 1]."""
        return max(0.0, min(1.0, value))

    def _write_json(self, output_path: str | Path, payload: Any) -> None:
        """Persist JSON artifacts."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
