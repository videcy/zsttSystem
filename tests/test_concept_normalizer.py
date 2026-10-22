from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.config import Config
from src.data_processing.concept_normalizer import (
    ALIAS_TRAILING_MODIFIERS,
    ConceptNormalizer,
)
from src.utils.deepseek_client import extract_json_value, generate_json_value


def test_alias_modifier_table_covers_required_suffixes() -> None:
    assert set(ALIAS_TRAILING_MODIFIERS) >= {
        "定义",
        "概念",
        "的概念",
        "基本概念",
        "方法",
    }


def test_alias_normalization_strips_particles_and_trailing_modifiers() -> None:
    normalize = ConceptNormalizer._normalize_alias_key

    assert normalize("数组的定义") == "数组"
    assert normalize("数组定义") == "数组"
    assert normalize("数组的概念") == "数组"
    assert normalize("数组基本概念") == "数组"
    assert normalize("数组的方法") == "数组"
    assert normalize("数组概念方法") == "数组"


def test_alias_normalization_only_strips_real_suffixes() -> None:
    normalize = ConceptNormalizer._normalize_alias_key

    assert normalize("方法论") == "方法论"
    assert normalize("方法学") == "方法学"
    assert normalize("定义域") == "定义域"
    assert normalize("概念图") == "概念图"


def test_cluster_threshold_is_read_from_config(monkeypatch) -> None:
    monkeypatch.setenv("CONCEPT_CLUSTER_THRESHOLD", "0.87")

    with patch(
        "src.data_processing.concept_normalizer.create_deepseek_client",
        return_value=object(),
    ):
        normalizer = ConceptNormalizer(wikipedia_enabled=False)

    assert normalizer.similarity_threshold == pytest.approx(0.87)


def test_cluster_threshold_default_is_conservative(monkeypatch) -> None:
    monkeypatch.delenv("CONCEPT_CLUSTER_THRESHOLD", raising=False)

    with patch(
        "src.data_processing.concept_normalizer.create_deepseek_client",
        return_value=object(),
    ):
        normalizer = ConceptNormalizer(wikipedia_enabled=False)

    assert normalizer.similarity_threshold == pytest.approx(0.94)


class DeepSeekJsonValueTests(unittest.TestCase):
    def test_extract_json_value_supports_array_payload(self) -> None:
        payload = '```json\n[{"name":"gradient descent","type":"algorithm"}]\n```'
        value = extract_json_value(payload)
        self.assertIsInstance(value, list)
        self.assertEqual(value[0]["name"], "gradient descent")

    def test_generate_json_value_enables_json_mode_and_retries_invalid_output(self) -> None:
        requests: list[dict] = []
        contents = iter(["not json", '{"concepts": []}'])

        class FakeCompletions:
            def create(self, **kwargs):
                requests.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=next(contents))
                        )
                    ]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )

        payload = generate_json_value(
            client,
            "deepseek-v4-flash",
            "请输出 JSON 对象",
        )

        self.assertEqual(payload, {"concepts": []})
        self.assertEqual(len(requests), 2)
        self.assertTrue(
            all(
                request["response_format"] == {"type": "json_object"}
                for request in requests
            )
        )
        self.assertTrue(
            all(
                request["extra_body"] == {"thinking": {"type": "disabled"}}
                for request in requests
            )
        )


class ConceptNormalizerTests(unittest.TestCase):
    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    @patch("src.data_processing.concept_normalizer.embed_texts")
    def test_candidate_links_merge_pairs_and_apply_fusion(self, mock_embed_texts, _mock_client) -> None:
        mock_embed_texts.return_value = [
            [1.0, 0.0],
            [0.78, 0.6257795139],
            [0.74, 0.6726068688],
            [0.7, 0.7141428429],
            [0.6, 0.8],
        ]
        normalizer = ConceptNormalizer(
            wikipedia_enabled=False,
            candidate_top_k=4,
            candidate_min_confidence=0.0,
            course_order_bonus=0.3,
            cross_discipline_decay=0.85,
            score_weight_vector=0.2,
            score_weight_structure=0.2,
            score_weight_rule=0.2,
            course_semesters={
                "IM2105": 4,
                "IM112": 1,
                "IM212": 2,
                "IM3101": 5,
                "LIS101": 2,
            },
        )
        registry = [
            {
                "id": "concept_00001",
                "canonical_name": "index",
                "aliases": ["index"],
                "type": "operator",
                "bloom_level": "apply",
                "discipline": "database",
                "source_courses": ["database"],
                "source_chapters": ["index structures"],
                "source_course_codes": ["IM2105"],
            },
            {
                "id": "concept_00002",
                "canonical_name": "B+ tree",
                "aliases": ["B+ tree"],
                "type": "operator",
                "bloom_level": "understand",
                "discipline": "database",
                "source_courses": ["data structure", "advanced data structure"],
                "source_chapters": ["trees", "advanced indexes"],
                "source_course_codes": ["IM112", "IM212"],
            },
            {
                "id": "concept_00003",
                "canonical_name": "hash index",
                "aliases": ["hash index"],
                "type": "operator",
                "bloom_level": "apply",
                "discipline": "database",
                "source_courses": ["data structure"],
                "source_chapters": ["hashing"],
                "source_course_codes": ["IM112"],
            },
            {
                "id": "concept_00004",
                "canonical_name": "transaction scheduling",
                "aliases": ["transaction scheduling"],
                "type": "algorithm",
                "bloom_level": "analyze",
                "discipline": "database",
                "source_courses": ["database systems"],
                "source_chapters": ["concurrency control"],
                "source_course_codes": ["IM3101"],
            },
            {
                "id": "concept_00005",
                "canonical_name": "library cataloging",
                "aliases": ["library cataloging"],
                "type": "operator",
                "bloom_level": "understand",
                "discipline": "library science",
                "source_courses": ["cataloging"],
                "source_chapters": ["subject analysis"],
                "source_course_codes": ["LIS101"],
            },
        ]

        links = normalizer.build_candidate_links(registry)

        bptree_to_index = next(
            item
            for item in links
            if item["source_id"] == "concept_00002" and item["target_id"] == "concept_00001"
        )
        self.assertAlmostEqual(bptree_to_index["score_components"]["S_vector"], 0.78, places=6)
        self.assertAlmostEqual(bptree_to_index["score_components"]["S_structure"], 0.3, places=6)
        self.assertAlmostEqual(bptree_to_index["score_components"]["S_rule"], 1.0, places=6)
        self.assertAlmostEqual(bptree_to_index["fusion_score"], 0.693333, places=6)
        self.assertAlmostEqual(bptree_to_index["initial_confidence"], 0.693333, places=6)
        self.assertTrue(any(e["type"] == "course_order" for e in bptree_to_index["evidence"]))

        cross_domain = next(
            item
            for item in links
            if item["source_id"] == "concept_00005" and item["target_id"] == "concept_00001"
        )
        self.assertAlmostEqual(cross_domain["score_components"]["S_vector"], 0.6, places=6)
        self.assertAlmostEqual(cross_domain["domain_decay_factor"], 0.85, places=6)
        self.assertAlmostEqual(cross_domain["fusion_score"], 0.466667, places=6)
        self.assertAlmostEqual(cross_domain["initial_confidence"], 0.396667, places=6)

        self.assertFalse(
            any(
                item["source_id"] == "concept_00004" and item["target_id"] == "concept_00001"
                for item in links
            )
        )

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_verify_candidate_links_uses_majority_vote_and_average_confidence(self, _mock_client) -> None:
        normalizer = ConceptNormalizer(
            wikipedia_enabled=False,
            llm_vote_count=3,
            require_unanimous=False,
        )
        candidate_links = [
            {
                "source_id": "concept_00002",
                "source_name": "B+ tree",
                "source_courses": ["data structure"],
                "source_course_codes": ["IM112"],
                "source_bloom_level": "understand",
                "source_discipline": "database",
                "target_id": "concept_00001",
                "target_name": "inverted index",
                "target_courses": ["database"],
                "target_course_codes": ["IM2105"],
                "target_bloom_level": "apply",
                "target_discipline": "database",
                "fusion_score": 0.556,
                "initial_confidence": 0.556,
                "evidence": [{"type": "vector_similarity", "value": 0.78, "detail": "B+ tree -> inverted index"}],
            }
        ]
        registry = [
            {
                "id": "concept_00001",
                "canonical_name": "inverted index",
                "aliases": ["inverted index"],
                "type": "operator",
                "bloom_level": "apply",
                "discipline": "database",
                "source_courses": ["database"],
                "source_chapters": ["index structures"],
                "source_course_codes": ["IM2105"],
                "source_occurrences": [
                    {
                        "course": "database",
                        "course_code": "IM2105",
                        "chapter": "index structures",
                        "evidence_sentence": "An inverted index maps terms to documents.",
                    }
                ],
            },
            {
                "id": "concept_00002",
                "canonical_name": "B+ tree",
                "aliases": ["B+ tree"],
                "type": "operator",
                "bloom_level": "understand",
                "discipline": "database",
                "source_courses": ["data structure"],
                "source_chapters": ["trees"],
                "source_course_codes": ["IM112"],
                "source_occurrences": [
                    {
                        "course": "data structure",
                        "course_code": "IM112",
                        "chapter": "trees",
                        "evidence_sentence": "B+ tree is a balanced tree index.",
                    }
                ],
            },
        ]

        votes = [
            {
                "requires": True,
                "relation_type": "FOUNDATION_OF",
                "confidence": 0.9,
                "reason": "Vote A",
                "dimensions": {
                    "definition_dependency": True,
                    "derivation_dependency": True,
                    "substitutable": False,
                    "bloom_compatible": True,
                    "teaching_practice": True,
                },
            },
            {
                "requires": True,
                "relation_type": "FOUNDATION_OF",
                "confidence": 0.8,
                "reason": "Vote A",
                "dimensions": {
                    "definition_dependency": True,
                    "derivation_dependency": False,
                    "substitutable": False,
                    "bloom_compatible": True,
                    "teaching_practice": True,
                },
            },
            {
                "requires": False,
                "relation_type": "NO_RELATION",
                "confidence": 0.4,
                "reason": "Vote B",
                "dimensions": {
                    "definition_dependency": False,
                    "derivation_dependency": False,
                    "substitutable": True,
                    "bloom_compatible": True,
                    "teaching_practice": False,
                },
            },
        ]

        with patch.object(normalizer, "_call_dependency_validator", side_effect=votes):
            verified = normalizer.verify_candidate_links(candidate_links, registry)

        self.assertEqual(len(verified), 1)
        edge = verified[0]
        self.assertTrue(edge["requires"])
        self.assertEqual(edge["relation_type"], "FOUNDATION_OF")
        self.assertAlmostEqual(edge["confidence"], 0.85, places=6)
        self.assertEqual(edge["reason"], "Vote A")
        self.assertEqual(len(edge["llm_votes"]), 3)


class ConceptNormalizerGateTests(unittest.TestCase):
    def test_concept_gate_defaults_are_unanimous_and_expanded(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
            self.assertEqual(cfg.concept_max_verification_candidates, 3000)
            self.assertTrue(cfg.concept_require_unanimous)
            self.assertEqual(cfg.concept_verified_min_confidence, 0.6)

    @staticmethod
    def _concept(concept_id: str, name: str, code: str) -> dict:
        return {
            "id": concept_id,
            "canonical_name": name,
            "aliases": [name],
            "type": "concept",
            "bloom_level": "understand",
            "discipline": "test",
            "source_courses": [code],
            "source_chapters": ["chapter"],
            "source_course_codes": [code],
            "source_occurrences": [
                {
                    "course": code,
                    "course_code": code,
                    "chapter": "chapter",
                    "evidence_sentence": f"{name} is taught in this course.",
                }
            ],
        }

    @staticmethod
    def _vote(requires: bool, confidence: float, reason: str) -> dict:
        return {
            "requires": requires,
            "relation_type": "FOUNDATION_OF" if requires else "NO_RELATION",
            "confidence": confidence,
            "reason": reason,
            "dimensions": {},
        }

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_mar_course_concepts_are_removed_before_candidate_generation(self, _mock_client) -> None:
        normalizer = ConceptNormalizer(
            wikipedia_enabled=False,
            candidate_top_k=3,
            candidate_min_confidence=0.0,
            course_semesters={"IM101": 1, "IM201": 2},
        )
        registry = [
            self._concept("im-1", "information retrieval", "IM101"),
            self._concept("im-2", "index structure", "IM201"),
            self._concept("mar-1", "事物普遍联系", "MAR112"),
        ]
        with patch(
            "src.data_processing.concept_normalizer.embed_texts",
            return_value=[[1.0, 0.0], [0.9, 0.1]],
        ):
            links = normalizer.build_candidate_links(registry)

        self.assertTrue(links)
        self.assertTrue(
            all(
                "mar-1" not in (link["source_id"], link["target_id"])
                for link in links
            )
        )

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_unanimous_vote_and_confidence_threshold_accept_edge(self, _mock_client) -> None:
        normalizer = ConceptNormalizer(
            wikipedia_enabled=False,
            require_unanimous=True,
            verified_min_confidence=0.7,
            llm_vote_count=3,
        )
        candidate = {
            "source_id": "source",
            "target_id": "target",
            "initial_confidence": 0.9,
            "fusion_score": 0.9,
            "evidence": [],
        }
        source = self._concept("source", "线性规划", "IM101")
        target = self._concept("target", "整数规划", "IM201")
        votes = [
            self._vote(True, 0.9, "same reason"),
            self._vote(True, 0.8, "same reason"),
            self._vote(True, 0.75, "same reason"),
        ]

        edge = normalizer._llm_verify_candidate(
            candidate,
            source,
            target,
            votes=votes,
        )

        self.assertTrue(edge["requires"])
        self.assertTrue(edge["unanimous"])
        self.assertAlmostEqual(edge["confidence"], 0.816667, places=6)

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_non_unanimous_vote_rejects_edge_when_gate_enabled(self, _mock_client) -> None:
        normalizer = ConceptNormalizer(
            wikipedia_enabled=False,
            require_unanimous=True,
            verified_min_confidence=0.6,
            llm_vote_count=3,
        )
        candidate = {
            "source_id": "source",
            "target_id": "target",
            "initial_confidence": 0.9,
            "fusion_score": 0.9,
            "evidence": [],
        }
        votes = [
            self._vote(True, 0.9, "yes"),
            self._vote(True, 0.9, "yes"),
            self._vote(False, 0.9, "no"),
        ]

        edge = normalizer._llm_verify_candidate(
            candidate,
            {"id": "source", "canonical_name": "线性规划"},
            {"id": "target", "canonical_name": "整数规划"},
            votes=votes,
        )

        self.assertFalse(edge["requires"])
        self.assertFalse(edge["unanimous"])
        self.assertEqual(edge["relation_type"], "NO_RELATION")

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_unanimous_vote_below_confidence_threshold_is_rejected(self, _mock_client) -> None:
        normalizer = ConceptNormalizer(
            wikipedia_enabled=False,
            require_unanimous=True,
            verified_min_confidence=0.8,
            llm_vote_count=3,
        )
        candidate = {
            "source_id": "source",
            "target_id": "target",
            "initial_confidence": 0.9,
            "fusion_score": 0.9,
            "evidence": [],
        }
        votes = [self._vote(True, 0.7, "weak") for _ in range(3)]

        edge = normalizer._llm_verify_candidate(
            candidate,
            {"id": "source", "canonical_name": "线性规划"},
            {"id": "target", "canonical_name": "整数规划"},
            votes=votes,
        )

        self.assertFalse(edge["requires"])
        self.assertTrue(edge["unanimous"])
        self.assertLess(edge["confidence"], 0.8)

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_validation_cache_key_changes_with_unanimity_and_threshold(self, _mock_client) -> None:
        strict = ConceptNormalizer(
            wikipedia_enabled=False,
            require_unanimous=True,
            verified_min_confidence=0.8,
        )
        legacy = ConceptNormalizer(
            wikipedia_enabled=False,
            require_unanimous=False,
            verified_min_confidence=0.6,
        )
        candidate = {"source_id": "a", "target_id": "b", "evidence": []}
        self.assertNotEqual(
            strict._validation_cache_key(candidate),
            legacy._validation_cache_key(candidate),
        )

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_selected_edge_gets_both_evidence_sentences(self, _mock_client) -> None:
        normalizer = ConceptNormalizer(wikipedia_enabled=False)
        candidate = {
            "source_id": "source",
            "target_id": "target",
            "initial_confidence": 0.9,
            "fusion_score": 0.9,
            "evidence": [],
        }
        source = self._concept("source", "线性规划", "IM101")
        target = self._concept("target", "整数规划", "IM201")

        with patch.object(
            normalizer,
            "_llm_verify_candidate",
            return_value={
                "source_id": "source",
                "target_id": "target",
                "requires": True,
                "relation_type": "FOUNDATION_OF",
                "confidence": 0.9,
                "verification_source": "llm",
            },
        ):
            edges = normalizer.verify_candidate_links(
                [candidate],
                [source, target],
            )

        self.assertEqual(len(edges), 1)
        self.assertEqual(
            edges[0]["source_evidence"],
            "线性规划 is taught in this course.",
        )
        self.assertEqual(
            edges[0]["target_evidence"],
            "整数规划 is taught in this course.",
        )

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_selected_edge_without_evidence_is_rejected(self, _mock_client) -> None:
        normalizer = ConceptNormalizer(wikipedia_enabled=False)
        candidate = {
            "source_id": "source",
            "target_id": "target",
            "initial_confidence": 0.9,
            "fusion_score": 0.9,
            "evidence": [],
        }
        registry = [
            {"id": "source", "canonical_name": "线性规划", "source_course_codes": ["IM101"]},
            {"id": "target", "canonical_name": "整数规划", "source_course_codes": ["IM201"]},
        ]

        with patch.object(
            normalizer,
            "_llm_verify_candidate",
            return_value={
                "source_id": "source",
                "target_id": "target",
                "requires": True,
                "relation_type": "FOUNDATION_OF",
                "confidence": 0.9,
                "verification_source": "llm",
            },
        ):
            edges = normalizer.verify_candidate_links([candidate], registry)

        self.assertEqual(edges, [])

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_evidence_falls_back_to_syllabus_chunks(self, _mock_client) -> None:
        normalizer = ConceptNormalizer(wikipedia_enabled=False)
        concept = {
            "id": "source",
            "canonical_name": "线性规划",
            "source_course_codes": ["IM101"],
        }
        chunks = [
            {
                "chunk_id": "chunk-1",
                "text": "第一章介绍模型。线性规划用于资源分配。",
                "metadata": {"source_type": "syllabus", "course_code": "IM101"},
            }
        ]

        evidence = normalizer._find_concept_evidence(
            concept,
            normalizer._chunk_evidence_index(chunks),
        )

        self.assertEqual(evidence, "线性规划用于资源分配。")

    @patch("src.data_processing.concept_normalizer.create_deepseek_client", return_value=object())
    def test_verify_candidate_links_also_rejects_mar_endpoints(self, _mock_client) -> None:
        normalizer = ConceptNormalizer(wikipedia_enabled=False)
        candidate = {
            "source_id": "source",
            "target_id": "target",
            "initial_confidence": 0.9,
            "fusion_score": 0.9,
            "evidence": [],
        }
        registry = [
            self._concept("source", "线性规划", "MAR112"),
            self._concept("target", "整数规划", "IM201"),
        ]
        with patch.object(
            normalizer,
            "_llm_verify_candidate",
            side_effect=AssertionError("MAR endpoints must be filtered before validation"),
        ):
            edges = normalizer.verify_candidate_links([candidate], registry)

        self.assertEqual(edges, [])


@pytest.mark.parametrize(
    ("course_semesters", "expected_pairs"),
    [
        ({"A100": 1, "B200": 2}, {("source", "target")}),
        ({"A100": 2, "B200": 2}, {("source", "target"), ("target", "source")}),
        ({"A100": "1-6", "B200": 2}, {("source", "target"), ("target", "source")}),
        ({"B200": 2}, {("source", "target"), ("target", "source")}),
        ({"A100": 1}, {("source", "target"), ("target", "source")}),
        ({"A100": 3, "B200": 2}, {("target", "source")}),
    ],
)
def test_candidate_generation_rejects_only_known_reverse_semesters(
    course_semesters: dict,
    expected_pairs: set[tuple[str, str]],
) -> None:
    registry = [
        {
            "id": "source",
            "canonical_name": "source concept",
            "aliases": ["source concept"],
            "type": "concept",
            "bloom_level": "understand",
            "discipline": "information science",
            "source_courses": ["course A"],
            "source_chapters": ["chapter A"],
            "source_course_codes": ["A100"],
        },
        {
            "id": "target",
            "canonical_name": "target concept",
            "aliases": ["target concept"],
            "type": "concept",
            "bloom_level": "understand",
            "discipline": "information science",
            "source_courses": ["course B"],
            "source_chapters": ["chapter B"],
            "source_course_codes": ["B200"],
        },
    ]

    with patch(
        "src.data_processing.concept_normalizer.create_deepseek_client",
        return_value=object(),
    ):
        normalizer = ConceptNormalizer(
            wikipedia_enabled=False,
            candidate_min_confidence=0.0,
            course_semesters=course_semesters,
        )
    with patch(
        "src.data_processing.concept_normalizer.embed_texts",
        return_value=[[1.0, 0.0], [0.95, 0.3122498999]],
    ):
        links = normalizer.build_candidate_links(registry)

    assert {(link["source_id"], link["target_id"]) for link in links} == expected_pairs


def test_concept_order_unknown_when_any_source_course_semester_is_unknown() -> None:
    with patch(
        "src.data_processing.concept_normalizer.create_deepseek_client",
        return_value=object(),
    ):
        normalizer = ConceptNormalizer(
            wikipedia_enabled=False,
            course_semesters={"A100": 1, "B200": 2},
        )

    assert normalizer._concept_semester(
        {"source_course_codes": ["A100", "MISSING"]}
    ) is None
    assert normalizer._concept_semester(
        {"source_course_codes": ["A100", "B200"]}
    ) == 1


def test_course_semesters_use_earliest_fixed_offering_across_plans_and_cache(
    tmp_path: Path,
) -> None:
    courses_path = tmp_path / "courses.json"
    courses_path.write_text(
        json.dumps(
            [
                {
                    "course_code": "A100",
                    "offerings": [
                        {"program_name": "plan 1", "semester": "1-6"},
                        {"program_name": "plan 2", "semester": 4},
                        {"program_name": "plan 3", "semester": 2},
                    ],
                },
                {
                    "course_code": "B200",
                    "offerings": [
                        {"program_name": "plan 1", "semester": "1-6"},
                        {"program_name": "plan 2", "semester": "/"},
                    ],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with patch(
        "src.data_processing.concept_normalizer.create_deepseek_client",
        return_value=object(),
    ):
        normalizer = ConceptNormalizer(
            wikipedia_enabled=False,
            courses_output_path=courses_path,
        )

    assert normalizer._get_course_semesters() == {"a100": 2, "b200": None}

    courses_path.write_text(
        json.dumps(
            [{"course_code": "A100", "offerings": [{"semester": 5}]}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert normalizer._get_course_semesters() == {"a100": 2, "b200": None}


if __name__ == "__main__":
    unittest.main()
