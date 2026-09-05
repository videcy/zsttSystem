from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.data_processing.concept_normalizer import ConceptNormalizer
from src.utils.deepseek_client import extract_json_value, generate_json_value


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
                "source_course_codes": ["MAR112", "MAR212"],
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
                "source_course_codes": ["MAR112"],
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
        normalizer = ConceptNormalizer(wikipedia_enabled=False, llm_vote_count=3)
        candidate_links = [
            {
                "source_id": "concept_00002",
                "source_name": "B+ tree",
                "source_courses": ["data structure"],
                "source_course_codes": ["MAR112"],
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
                "source_course_codes": ["MAR112"],
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


if __name__ == "__main__":
    unittest.main()
