"""Extract, normalize, score, and validate teaching concepts before embedding."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.deepseek_client import create_deepseek_client, embed_texts, generate_json_value
from src.config import config

logger = logging.getLogger(__name__)


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
        course_order_bonus: float | None = None,
        cross_discipline_decay: float | None = None,
        score_weight_vector: float | None = None,
        score_weight_structure: float | None = None,
        score_weight_rule: float | None = None,
        llm_vote_count: int | None = None,
        llm_vote_temperature: float | None = None,
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
        self.score_weight_external = config.concept_score_weight_external
        self.llm_vote_count = (
            llm_vote_count if llm_vote_count is not None else config.concept_llm_vote_count
        )
        self.llm_vote_temperature = (
            llm_vote_temperature if llm_vote_temperature is not None
            else config.concept_llm_vote_temperature
        )
        self._normalize_score_weights()
        self.api_client = create_deepseek_client()
        self.embedding_client = self.api_client
        self._wiki_cache: dict[str, str] = {}

    def _normalize_score_weights(self) -> None:
        """Ensure score component weights sum to 1.0."""
        total = (
            self.score_weight_vector
            + self.score_weight_structure
            + self.score_weight_rule
            + self.score_weight_external
        )
        if total <= 0:
            return
        self.score_weight_vector /= total
        self.score_weight_structure /= total
        self.score_weight_rule /= total
        self.score_weight_external /= total

    def preprocess_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        registry_output_path: str | Path | None = None,
        alias_output_path: str | Path | None = None,
        enriched_chunks_output_path: str | Path | None = None,
        candidate_output_path: str | Path | None = None,
        verified_output_path: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
        """Extract concepts and write canonical, candidate, and verified artifacts."""
        if not chunks:
            return [], [], {}

        extracted_by_chunk: list[list[dict[str, str]]] = []
        all_concepts: list[dict[str, str]] = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            chunk_text = str(chunk.get("text", "")).strip()
            concepts = self.extract_core_concepts(
                chunk_text,
                metadata if isinstance(metadata, dict) else {},
            )
            extracted_by_chunk.append(concepts)
            all_concepts.extend(concepts)

        alias_table, canonical_registry = self._canonicalize_concepts(all_concepts)
        concept_lookup = {concept["canonical_name"]: concept for concept in canonical_registry}

        enriched_chunks: list[dict[str, Any]] = []
        for chunk, raw_concepts in zip(chunks, extracted_by_chunk):
            enriched_chunks.append(
                self._enrich_chunk(chunk, raw_concepts, alias_table, concept_lookup)
            )

        candidate_links = self.build_candidate_links(canonical_registry)
        verified_edges = self.verify_candidate_links(candidate_links, canonical_registry)

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
            "结果输出JSON数组："
            "[{name, type, bloom_level, discipline, source_course, source_chapter}]。\n"
            "约束：\n"
            "1. 只保留文本中明确出现或能直接对应的核心概念，不要输出课程名、教师名、教材名。\n"
            "2. type 只能是 algorithm、theorem、operator、method、model、framework、data_structure、paradigm、concept 之一。\n"
            "3. bloom_level 只能是 understand、apply、analyze 之一。\n"
            "4. discipline 用简洁中文或英文学科名。\n"
            "5. source_course 填该概念所在课程名；source_chapter 填该概念所在章节名。\n"
            "6. 优先使用给定模块元数据中的 course_name 和 syllabus_section 作为 source 字段。\n"
            "7. 只输出 JSON 数组，不要额外解释。\n"
            f"模块元数据: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"模块文本:\n{normalized_text}"
        )
        try:
            response = generate_json_value(
                self.api_client,
                self.llm_model_name,
                prompt,
                temperature=0.0,
                max_output_tokens=1000,
            )
        except Exception as exc:
            logger.warning(
                "[concept_normalizer] LLM concept extraction failed, using fallback: %s", exc
            )
            return self._fallback_extract_core_concepts(normalized_text, metadata)

        if not isinstance(response, list):
            return self._fallback_extract_core_concepts(normalized_text, metadata)

        normalized_concepts: list[dict[str, str]] = []
        for item in response:
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_extracted_concept(item, metadata)
            if normalized is not None:
                normalized_concepts.append(normalized)

        return normalized_concepts or self._fallback_extract_core_concepts(normalized_text, metadata)

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
    ) -> list[dict[str, Any]]:
        """Validate merged candidate links with LLM self-consistency voting."""
        if not candidate_links:
            return []

        concept_lookup = {item["id"]: item for item in canonical_registry}
        verified_edges: list[dict[str, Any]] = []
        for candidate in candidate_links:
            source = concept_lookup.get(candidate["source_id"])
            target = concept_lookup.get(candidate["target_id"])
            if source is None or target is None:
                continue
            verified_edges.append(self._llm_verify_candidate(candidate, source, target))

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
                "w4_external": round(self.score_weight_external, 6),
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
            try:
                vote = self._call_dependency_validator(candidate, source, target)
            except Exception as exc:
                logger.warning(
                    "[concept_normalizer] LLM dependency vote failed, using fallback: %s", exc
                )
                vote = self._fallback_dependency_vote(candidate)
            votes.append(vote)

        requires_counter = Counter(bool(vote.get("requires", False)) for vote in votes)
        requires = requires_counter.most_common(1)[0][0]
        relation_votes = [
            str(vote.get("relation_type", "NO_RELATION")).strip()
            for vote in votes
            if vote.get("relation_type")
        ]
        relation_type = Counter(relation_votes).most_common(1)[0][0] if relation_votes else "NO_RELATION"
        if relation_type not in self.ALLOWED_RELATION_TYPES:
            relation_type = "NO_RELATION"
        average_confidence = sum(self._safe_float(vote.get("confidence", 0.0)) for vote in votes) / max(len(votes), 1)
        reason = self._choose_majority_reason(votes)

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
        return self._normalize_dependency_vote(raw, candidate)

    def _normalize_dependency_vote(
        self,
        vote: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize one validator vote."""
        relation_type = str(vote.get("relation_type", "NO_RELATION")).strip().upper()
        if relation_type not in self.ALLOWED_RELATION_TYPES:
            relation_type = "NO_RELATION"
        requires = bool(vote.get("requires", False))
        confidence = self._clamp01(self._safe_float(vote.get("confidence", candidate["initial_confidence"])))
        reason = str(vote.get("reason", "")).strip() or "No reason provided."
        dimensions = vote.get("dimensions", {})
        if not isinstance(dimensions, dict):
            dimensions = {}

        normalized_dimensions = {
            "definition_dependency": bool(dimensions.get("definition_dependency", False)),
            "derivation_dependency": bool(dimensions.get("derivation_dependency", False)),
            "substitutable": bool(dimensions.get("substitutable", False)),
            "bloom_compatible": bool(dimensions.get("bloom_compatible", True)),
            "teaching_practice": bool(dimensions.get("teaching_practice", False)),
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
        """Fallback validator when the LLM is unavailable."""
        requires = candidate["initial_confidence"] >= 0.45
        return {
            "requires": requires,
            "relation_type": "CONCEPTUAL_BASIS" if requires else "NO_RELATION",
            "confidence": round(candidate["initial_confidence"], 6),
            "reason": "Fallback decision based on fused candidate score.",
            "dimensions": {
                "definition_dependency": False,
                "derivation_dependency": False,
                "substitutable": True,
                "bloom_compatible": True,
                "teaching_practice": False,
            },
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
        return 1.0 if self._course_order_bonus_for_pair(source, target) > 0 else 0.0

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
                if not name:
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
                    }
                )

        return concepts

    def _normalize_extracted_concept(
        self,
        concept: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, str] | None:
        """Validate and normalize a raw LLM concept entry."""
        name = self._cleanup_term(concept.get("name", ""))
        concept_type = str(concept.get("type", "")).strip().lower()
        bloom_level = str(concept.get("bloom_level", "")).strip().lower()
        discipline = self._cleanup_term(concept.get("discipline", ""))
        source_course = self._cleanup_term(concept.get("source_course", "")) or self._default_source_course(metadata)
        source_chapter = self._cleanup_term(concept.get("source_chapter", "")) or self._default_source_chapter(metadata)
        source_course_code = self._default_source_course_code(metadata)

        if not name or concept_type not in self.ALLOWED_TYPES or bloom_level not in self.ALLOWED_BLOOM_LEVELS:
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

        for index, group in enumerate(groups, start=1):
            aliases = sorted(group, key=lambda item: (len(item), item.lower()))
            wiki_candidates = [self._lookup_wikipedia_title(alias) for alias in aliases]
            canonical_name = self._choose_canonical_name(aliases, wiki_candidates)
            member_concepts: list[dict[str, str]] = []
            for alias in aliases:
                member_concepts.extend(concept_occurrences[alias])

            alias_table[canonical_name] = aliases
            canonical_registry.append(
                {
                    "id": f"concept_{index:05d}",
                    "canonical_name": canonical_name,
                    "aliases": aliases,
                    "type": self._majority_value(member_concepts, "type", default="algorithm"),
                    "bloom_level": self._majority_value(member_concepts, "bloom_level", default="understand"),
                    "discipline": self._majority_value(member_concepts, "discipline", default="unknown"),
                    "source_courses": self._unique_values(member_concepts, "source_course"),
                    "source_chapters": self._unique_values(member_concepts, "source_chapter"),
                    "source_course_codes": self._unique_values(member_concepts, "source_course_code"),
                }
            )

        canonical_registry.sort(key=lambda item: item["id"])
        return alias_table, canonical_registry

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
