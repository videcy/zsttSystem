from __future__ import annotations

import unittest
from unittest.mock import patch

from src.online_service.dependency_explainer import (
    _fallback_extract_query_entities,
    aggregate_prerequisites,
    build_dependency_answer,
    build_mermaid_graph,
    generate_dependency_explanation,
)


def _concept(concept_id: str, name: str, course: str) -> dict:
    return {
        "concept_id": concept_id,
        "name": name,
        "discipline": None,
        "bloom_level": None,
        "source_courses": [course],
        "source_course_codes": [],
        "source_chapters": [],
        "source_occurrences": [],
    }


class _FakeRecord(dict):
    pass


class _FakeSession:
    """按课程名返回概念，按边表返回「终点在目标集合里」的路径。"""

    def __init__(self, concepts: list[dict], edges: list[tuple[str, str]]) -> None:
        self.concepts = {c["concept_id"]: c for c in concepts}
        self.edges = edges

    def run(self, query: str, **params):
        if "$course" in query:
            course = params["course"]
            return [
                _FakeRecord(c)
                for c in self.concepts.values()
                if course in c["source_courses"]
            ]
        if "$term" in query:
            return [
                _FakeRecord(c) for c in self.concepts.values() if params["term"] == c["name"]
            ]
        targets = set(params["target_ids"])
        rows = []
        for source, target in self.edges:
            if target in targets:
                rows.append(
                    {
                        "nodes": [self.concepts[source], self.concepts[target]],
                        "relations": [{"type": "REQUIRES", "confidence": 0.9}],
                        "avg_confidence": 0.9,
                    }
                )
        return rows


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
        self.assertIn("《管理运筹学》第一章的线性规划", result["explanation"])

    def test_dependency_nli_prunes_unsupported_sentences(self) -> None:
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
                return_value={"explanation": "整数规划建立在线性规划之上。学完会更顺。"},
            ),
            patch(
                "src.online_service.dependency_explainer.verify_answer_with_nli",
                return_value=(
                    False,
                    [
                        {"sentence": "整数规划建立在线性规划之上。", "label": "Entailment", "score": 0.9},
                        {"sentence": "学完会更顺。", "label": "Neutral", "score": 0.3},
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

        self.assertEqual(result["nli_status"], "pruned")
        self.assertEqual(result["explanation"], "整数规划建立在线性规划之上。")


    def _two_course_graph(self, extra_edges: list[tuple[str, str]] = ()) -> _FakeSession:
        concepts = [
            _concept("a1", "新媒体概念一", "新媒体"),
            _concept("a2", "新媒体概念二", "新媒体"),
            _concept("b1", "大数据概念一", "大数据分析"),
            _concept("b2", "大数据概念二", "大数据分析"),
            _concept("x1", "外部概念", "统计学"),
        ]
        # 两门课各自有内部路径，另有外部课程指向它们，彼此不连通。
        edges = [("a1", "a2"), ("b1", "b2"), ("x1", "a2"), ("x1", "b2"), *extra_edges]
        return _FakeSession(concepts, edges)

    def test_unconnected_entities_are_refused_without_llm(self) -> None:
        session = self._two_course_graph()

        with (
            patch(
                "src.online_service.dependency_explainer.extract_query_entities",
                return_value={"courses": ["新媒体", "大数据分析"], "concepts": []},
            ),
            patch(
                "src.online_service.dependency_explainer.generate_dependency_explanation"
            ) as explain,
        ):
            result = build_dependency_answer("新媒体与大数据分析有什么关系", session, object())

        explain.assert_not_called()
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["error_code"], "DEPENDENCY_NOT_CONNECTED")
        self.assertEqual(
            result["explanation"],
            "图谱中没有找到《新媒体》与《大数据分析》之间的依赖路径，无法说明两者存在关系。",
        )
        self.assertEqual(result["paths"], [])

    def test_unconnected_entities_do_not_use_fallback_explanation(self) -> None:
        session = self._two_course_graph()

        with (
            patch(
                "src.online_service.dependency_explainer.extract_query_entities",
                return_value={"courses": ["新媒体", "大数据分析"], "concepts": []},
            ),
            patch(
                "src.online_service.dependency_explainer._fallback_explanation"
            ) as fallback,
        ):
            build_dependency_answer("新媒体和大数据分析有关系吗", session, None)

        fallback.assert_not_called()

    def test_connected_entities_keep_only_cross_group_paths(self) -> None:
        session = self._two_course_graph([("a1", "b2")])

        with (
            patch(
                "src.online_service.dependency_explainer.extract_query_entities",
                return_value={"courses": ["新媒体", "大数据分析"], "concepts": []},
            ),
            patch.dict("os.environ", {"NLI_VERIFICATION_ENABLED": "true"}),
            patch(
                "src.online_service.dependency_explainer.generate_json_value",
                return_value={"explanation": "新媒体概念一是大数据概念二的先修。"},
            ),
            patch(
                "src.online_service.dependency_explainer.verify_answer_with_nli",
                return_value=(False, [{"sentence": "x。", "label": "Contradiction"}]),
            ),
        ):
            result = build_dependency_answer("新媒体和大数据分析有什么关系", session, object())

        # 解释被 NLI 拦下时，多实体问题退回到连通路径的确定性描述。
        self.assertIn("新媒体概念一 → 大数据概念二", result["explanation"])
        self.assertEqual(
            [(p["source"]["concept_id"], p["target"]["concept_id"]) for p in result["paths"]],
            [("a1", "b2")],
        )

    def test_single_entity_keeps_all_paths_and_fallback(self) -> None:
        session = self._two_course_graph()

        with (
            patch(
                "src.online_service.dependency_explainer.extract_query_entities",
                return_value={"courses": ["新媒体"], "concepts": []},
            ),
            patch(
                "src.online_service.dependency_explainer.generate_json_value",
                return_value={},
            ),
        ):
            result = build_dependency_answer("新媒体需要哪些先修知识", session, object())

        self.assertEqual(len(result["paths"]), 2)
        self.assertIn("图谱中检索到的先修知识", result["explanation"])

    def test_relation_question_uses_neutral_prompt(self) -> None:
        paths = [
            {
                "nodes": [
                    {"concept_id": "s", "name": "线性规划"},
                    {"concept_id": "t", "name": "整数规划"},
                ],
                "relations": [],
                "avg_confidence": 0.9,
            }
        ]
        prompts: list[str] = []

        def capture(_client, _model, prompt, **_kwargs):
            prompts.append(prompt)
            return {}

        with patch(
            "src.online_service.dependency_explainer.generate_json_value", side_effect=capture
        ):
            generate_dependency_explanation("线性规划和整数规划有什么关系", paths, [], object())
            generate_dependency_explanation("整数规划需要哪些先修知识？", paths, [], object())

        self.assertIn("路径不能说明的关系不要推断", prompts[0])
        self.assertNotIn("为什么必须先学", prompts[0])
        self.assertIn("为什么必须先学这些先修知识", prompts[1])


if __name__ == "__main__":
    unittest.main()
