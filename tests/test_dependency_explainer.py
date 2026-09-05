from __future__ import annotations

import unittest
from unittest.mock import patch

from src.online_service.dependency_explainer import (
    _fallback_extract_query_entities,
    aggregate_prerequisites,
    build_mermaid_graph,
    generate_dependency_explanation,
)


class DependencyExplainerTests(unittest.TestCase):
    def test_fallback_entity_extraction_keeps_unquoted_target(self) -> None:
        entities = _fallback_extract_query_entities("整数规划需要哪些先修知识？")

        self.assertIn("整数规划", entities["concepts"])
        self.assertIn("整数规划", entities["courses"])

    def test_aggregate_prerequisites_groups_by_course_and_chapter(self) -> None:
        paths = [
            {
                "nodes": [
                    {
                        "concept_id": "source_1",
                        "name": "B+ tree",
                        "discipline": "database",
                        "bloom_level": "understand",
                        "source_courses": ["Data Structures"],
                        "source_chapters": ["Trees"],
                    },
                    {
                        "concept_id": "target_1",
                        "name": "Inverted index",
                        "source_courses": ["Information Retrieval"],
                        "source_chapters": ["Indexing"],
                    },
                ],
                "avg_confidence": 0.87,
            }
        ]

        prerequisites = aggregate_prerequisites(paths)

        self.assertEqual(len(prerequisites), 1)
        self.assertEqual(prerequisites[0]["course"], "Data Structures")
        self.assertEqual(prerequisites[0]["chapter"], "Trees")
        self.assertEqual(prerequisites[0]["concepts"][0]["name"], "B+ tree")
        self.assertAlmostEqual(prerequisites[0]["max_confidence"], 0.87)

    def test_aggregate_prerequisites_preserves_course_chapter_pairs(self) -> None:
        paths = [
            {
                "nodes": [
                    {
                        "concept_id": "source_1",
                        "name": "线性规划",
                        "source_courses": ["课程A", "课程B"],
                        "source_chapters": ["第一章", "第三章"],
                        "source_occurrences": [
                            {
                                "course": "课程A",
                                "course_code": "A001",
                                "chapter": "第一章",
                            },
                            {
                                "course": "课程B",
                                "course_code": "B001",
                                "chapter": "第三章",
                            },
                        ],
                    },
                    {"concept_id": "target_1", "name": "整数规划"},
                ],
                "avg_confidence": 0.9,
            }
        ]

        prerequisites = aggregate_prerequisites(paths)

        self.assertEqual(
            {(item["course"], item["chapter"]) for item in prerequisites},
            {("课程A", "第一章"), ("课程B", "第三章")},
        )

    def test_build_mermaid_graph_uses_dependency_direction(self) -> None:
        paths = [
            {
                "nodes": [
                    {
                        "concept_id": "source_1",
                        "name": "B+ tree",
                        "source_courses": ["Data Structures"],
                    },
                    {
                        "concept_id": "middle_1",
                        "name": "Database index",
                        "source_courses": ["Database"],
                    },
                    {
                        "concept_id": "target_1",
                        "name": "Inverted index",
                        "source_courses": ["Information Retrieval"],
                    },
                ],
                "relations": [],
            }
        ]

        mermaid = build_mermaid_graph(paths)

        self.assertIn("graph TD", mermaid)
        self.assertIn("source_1[Data Structures·B+ tree] --> middle_1[Database·Database index]", mermaid)
        self.assertIn("middle_1[Database·Database index] --> target_1[Information Retrieval·Inverted index]", mermaid)

    def test_failed_dependency_nli_is_reported_as_degraded_fallback(self) -> None:
        paths = [
            {
                "nodes": [
                    {"concept_id": "source", "name": "线性规划"},
                    {"concept_id": "target", "name": "整数规划"},
                ],
                "relations": [],
                "avg_confidence": 0.9,
            }
        ]
        prerequisites = [
            {
                "course": "管理运筹学",
                "chapter": "第一章",
                "concepts": [{"name": "线性规划"}],
                "max_confidence": 0.9,
            }
        ]
        with (
            patch.dict("os.environ", {"NLI_VERIFICATION_ENABLED": "true"}),
            patch(
                "src.online_service.dependency_explainer.generate_json_value",
                return_value={"explanation": "错误解释。"},
            ),
            patch(
                "src.online_service.dependency_explainer.verify_answer_with_nli",
                return_value=(
                    False,
                    [
                        {
                            "sentence": "错误解释。",
                            "label": "Contradiction",
                            "score": 0.95,
                        }
                    ],
                ),
            ),
        ):
            result = generate_dependency_explanation(
                "整数规划需要哪些先修知识？",
                paths,
                prerequisites,
                object(),
            )

        self.assertEqual(result["nli_status"], "fallback")
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["error_code"], "NLI_VERIFICATION_FAILED")
        self.assertNotEqual(result["explanation"], "错误解释。")


if __name__ == "__main__":
    unittest.main()
