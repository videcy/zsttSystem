from __future__ import annotations

import unittest

from src.online_service.dependency_explainer import aggregate_prerequisites, build_mermaid_graph


class DependencyExplainerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
