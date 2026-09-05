from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import run_pipeline
from src.data_processing.concept_normalizer import ConceptNormalizer
from src.data_processing.graph_builder import _replace_managed_graph, build_graph_records


def _concept(name: str, course_code: str) -> dict[str, str]:
    return {
        "name": name,
        "type": "concept",
        "bloom_level": "understand",
        "discipline": "信息管理",
        "source_course": "信息管理学",
        "source_chapter": "第一章",
        "source_course_code": course_code,
    }


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_concept_cleaning_and_ids_are_stable(_mock_client) -> None:
    normalizer = ConceptNormalizer(
        wikipedia_enabled=False,
        candidate_min_confidence=0.0,
    )
    metadata = {"course_name": "信息管理学", "course_code": "IM121"}

    assert normalizer._normalize_extracted_concept(
        _concept("课程编码", "IM121"), metadata
    ) is None
    assert normalizer._normalize_extracted_concept(
        _concept("信息管理学", "IM121"), metadata
    ) is None
    assert normalizer._normalize_extracted_concept(
        _concept("信息生命周期", "IM121"), metadata
    ) is not None

    def distinct_embeddings(_client, _model, names, **_kwargs):
        return [
            [1.0 if row == column else 0.0 for column in range(len(names))]
            for row in range(len(names))
        ]

    with patch(
        "src.data_processing.concept_normalizer.embed_texts",
        side_effect=distinct_embeddings,
    ):
        _, first = normalizer._canonicalize_concepts(
            [_concept("信息生命周期", "IM121"), _concept("信息组织", "IM2105")]
        )
        _, second = normalizer._canonicalize_concepts(
            [
                _concept("信息生命周期", "IM121"),
                _concept("信息组织", "IM2105"),
                _concept("知识管理", "IM317"),
            ]
        )

    first_ids = {item["canonical_name"]: item["id"] for item in first}
    second_ids = {item["canonical_name"]: item["id"] for item in second}
    assert second_ids["信息生命周期"] == first_ids["信息生命周期"]
    assert second_ids["信息组织"] == first_ids["信息组织"]


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_canonical_registry_keeps_exact_source_occurrences(_mock_client) -> None:
    normalizer = ConceptNormalizer(wikipedia_enabled=False)
    occurrences = [
        {
            **_concept("线性规划", "A001"),
            "source_course": "课程A",
            "source_chapter": "第一章",
        },
        {
            **_concept("线性规划", "B001"),
            "source_course": "课程B",
            "source_chapter": "第三章",
        },
    ]
    with patch(
        "src.data_processing.concept_normalizer.embed_texts",
        return_value=[[1.0]],
    ):
        _aliases, registry = normalizer._canonicalize_concepts(occurrences)

    assert registry[0]["source_occurrences"] == [
        {"course": "课程A", "course_code": "A001", "chapter": "第一章"},
        {"course": "课程B", "course_code": "B001", "chapter": "第三章"},
    ]


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_dependency_validation_cache_reuses_only_complete_llm_results(
    _mock_client,
    tmp_path: Path,
) -> None:
    normalizer = ConceptNormalizer(
        wikipedia_enabled=False,
        max_verification_candidates=10,
    )
    candidate = {
        "source_id": "source",
        "target_id": "target",
        "initial_confidence": 0.9,
        "fusion_score": 0.9,
        "evidence": [],
    }
    registry = [
        {"id": "source", "canonical_name": "线性规划"},
        {"id": "target", "canonical_name": "整数规划"},
    ]
    cached_edge = {
        "source_id": "source",
        "target_id": "target",
        "requires": True,
        "relation_type": "FOUNDATION_OF",
        "confidence": 0.9,
        "verification_source": "llm",
    }
    cache_path = tmp_path / "validation-cache.json"

    with patch.object(
        normalizer,
        "_llm_verify_candidate",
        return_value=cached_edge,
    ) as validator:
        first = normalizer.verify_candidate_links(
            [candidate],
            registry,
            validation_cache_path=cache_path,
        )
    with patch.object(
        normalizer,
        "_llm_verify_candidate",
        side_effect=AssertionError("cache should be reused"),
    ):
        second = normalizer.verify_candidate_links(
            [candidate],
            registry,
            validation_cache_path=cache_path,
        )

    assert first == second == [cached_edge]
    validator.assert_called_once()


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_cached_edges_are_reused_before_new_validation_budget_is_applied(
    _mock_client,
    tmp_path: Path,
) -> None:
    normalizer = ConceptNormalizer(
        wikipedia_enabled=False,
        max_verification_candidates=1,
    )
    candidates = [
        {
            "source_id": "a",
            "target_id": "b",
            "initial_confidence": 0.9,
            "fusion_score": 0.9,
            "evidence": [],
        },
        {
            "source_id": "b",
            "target_id": "c",
            "initial_confidence": 0.8,
            "fusion_score": 0.8,
            "evidence": [],
        },
    ]
    registry = [
        {"id": value, "canonical_name": value.upper()}
        for value in ("a", "b", "c")
    ]
    cached_edge = {
        "source_id": "b",
        "target_id": "c",
        "requires": True,
        "relation_type": "FOUNDATION_OF",
        "confidence": 0.8,
        "verification_source": "llm",
    }
    cache_path = tmp_path / "validation-cache.json"
    cache_path.write_text(
        json.dumps(
            {
                normalizer._validation_cache_key(
                    candidates[1],
                    registry[1],
                    registry[2],
                ): cached_edge
            }
        ),
        encoding="utf-8",
    )
    new_edge = {
        "source_id": "a",
        "target_id": "b",
        "requires": True,
        "relation_type": "FOUNDATION_OF",
        "confidence": 0.9,
        "verification_source": "llm",
    }

    with patch.object(
        normalizer,
        "_llm_verify_candidate",
        return_value=new_edge,
    ) as validator:
        edges = normalizer.verify_candidate_links(
            candidates,
            registry,
            validation_cache_path=cache_path,
        )

    assert [(edge["source_id"], edge["target_id"]) for edge in edges] == [
        ("a", "b"),
        ("b", "c"),
    ]
    validator.assert_called_once()


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    side_effect=ValueError("missing key"),
)
def test_dependency_verification_fails_closed_without_llm(_mock_client) -> None:
    normalizer = ConceptNormalizer(wikipedia_enabled=False, llm_vote_count=3)
    candidate = {
        "source_id": "source",
        "target_id": "target",
        "initial_confidence": 0.99,
        "fusion_score": 0.99,
        "evidence": [],
    }
    source = {"id": "source", "canonical_name": "线性规划"}
    target = {"id": "target", "canonical_name": "整数规划"}

    edge = normalizer._llm_verify_candidate(candidate, source, target)

    assert edge["requires"] is False
    assert edge["relation_type"] == "NO_RELATION"
    assert edge["confidence"] == 0.0


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_schema_invalid_concept_array_fails_closed_in_authoritative_mode(
    _mock_client,
) -> None:
    normalizer = ConceptNormalizer(
        wikipedia_enabled=False,
        allow_rule_fallback=False,
    )
    with (
        patch(
            "src.data_processing.concept_normalizer.generate_json_value",
            return_value=[{}],
        ),
        pytest.raises(
            RuntimeError,
            match=(
                "verified concept extraction.*"
                "concept extractor returned no schema-valid entries"
            ),
        ),
    ):
        normalizer.extract_core_concepts(
            "线性规划",
            {"course_name": "管理运筹学"},
        )


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_noise_only_concept_response_is_a_valid_empty_chunk(_mock_client) -> None:
    normalizer = ConceptNormalizer(
        wikipedia_enabled=False,
        allow_rule_fallback=False,
    )
    response = {
        "concepts": [
            {
                "name": "管理运筹学",
                "type": "concept",
                "bloom_level": "understand",
                "discipline": "运筹学",
            }
        ]
    }
    with patch(
        "src.data_processing.concept_normalizer.generate_json_value",
        return_value=response,
    ):
        concepts = normalizer.extract_core_concepts(
            "管理运筹学课程介绍",
            {"course_name": "管理运筹学"},
        )

    assert concepts == []


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_concept_extractor_accepts_json_object_wrapper(_mock_client) -> None:
    normalizer = ConceptNormalizer(
        wikipedia_enabled=False,
        allow_rule_fallback=False,
    )
    response = {
        "concepts": [
            {
                "name": "线性规划",
                "type": "model",
                "bloom_level": "understand",
                "discipline": "运筹学",
                "source_course": "管理运筹学",
                "source_chapter": "第一章",
            }
        ]
    }
    with patch(
        "src.data_processing.concept_normalizer.generate_json_value",
        return_value=response,
    ) as generate:
        concepts = normalizer.extract_core_concepts(
            "线性规划",
            {
                "course_name": "管理运筹学",
                "syllabus_section": "第一章",
            },
        )

    assert concepts[0]["name"] == "线性规划"
    assert concepts[0]["extraction_source"] == "llm"
    assert '"concepts"' in generate.call_args.args[2]


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_low_extraction_coverage_cannot_publish_an_empty_snapshot(
    _mock_client,
) -> None:
    normalizer = ConceptNormalizer(
        wikipedia_enabled=False,
        allow_rule_fallback=False,
        min_extraction_coverage=0.5,
    )
    chunks = [
        {"chunk_id": "one", "text": "文本一", "metadata": {}},
        {"chunk_id": "two", "text": "文本二", "metadata": {}},
    ]
    with (
        patch.object(normalizer, "extract_core_concepts", return_value=[]),
        pytest.raises(RuntimeError, match="coverage was too low"),
    ):
        normalizer.preprocess_chunks(chunks)


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_dependency_vote_requires_a_real_json_boolean(_mock_client) -> None:
    normalizer = ConceptNormalizer(wikipedia_enabled=False)
    candidate = {"initial_confidence": 0.9}

    for value in ("false", "true", 1, 0, None):
        vote = normalizer._normalize_dependency_vote(
            {
                "requires": value,
                "relation_type": "FOUNDATION_OF",
                "confidence": 0.9,
            },
            candidate,
        )
        assert vote["requires"] is False
        assert vote["relation_type"] == "NO_RELATION"

    assert normalizer._normalize_dependency_vote(
        {
            "requires": True,
            "relation_type": "FOUNDATION_OF",
            "confidence": 0.9,
        },
        candidate,
    )["requires"] is True


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_concept_provenance_comes_from_chunk_metadata(_mock_client) -> None:
    normalizer = ConceptNormalizer(wikipedia_enabled=False)
    normalized = normalizer._normalize_extracted_concept(
        {
            "name": "线性规划",
            "type": "model",
            "bloom_level": "understand",
            "discipline": "运筹学",
            "source_course": "伪造课程",
            "source_chapter": "伪造章节",
        },
        {
            "course_name": "管理运筹学",
            "course_code": "IM399",
            "syllabus_section": "第一章 线性规划",
        },
    )

    assert normalized is not None
    assert normalized["source_course"] == "管理运筹学"
    assert normalized["source_chapter"] == "第一章 线性规划"
    assert normalized["source_course_code"] == "IM399"


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_concept_schema_normalizes_common_chinese_aliases(_mock_client) -> None:
    normalizer = ConceptNormalizer(wikipedia_enabled=False)
    normalized = normalizer._normalize_extracted_concept(
        {
            "concept_name": "线性规划",
            "concept_type": "模型",
            "bloom": "理解",
            "domain": "运筹学",
        },
        {"course_name": "管理运筹学"},
    )

    assert normalized is not None
    assert normalized["name"] == "线性规划"
    assert normalized["type"] == "model"
    assert normalized["bloom_level"] == "understand"
    assert normalized["discipline"] == "运筹学"


@patch(
    "src.data_processing.concept_normalizer.create_deepseek_client",
    return_value=object(),
)
def test_unknown_nonempty_concept_type_uses_generic_concept(_mock_client) -> None:
    normalizer = ConceptNormalizer(wikipedia_enabled=False)
    normalized = normalizer._normalize_extracted_concept(
        {
            "name": "信息生命周期",
            "type": "theory",
            "bloom_level": "understand",
            "discipline": "信息管理",
        },
        {"course_name": "信息管理学"},
    )

    assert normalized is not None
    assert normalized["type"] == "concept"


def test_graph_records_include_traceable_verified_concept_edges() -> None:
    concepts = [
        {
            "id": "source",
            "canonical_name": "线性规划",
            "source_course_codes": ["IM121", "IM399"],
        },
        {
            "id": "target",
            "canonical_name": "整数规划",
            "source_course_codes": ["IM399"],
        },
    ]
    chunks = [
        {
            "chunk_id": "chunk-1",
            "metadata": {
                "course_code": "IM399",
                "core_concept_ids": ["source", "target"],
            },
        }
    ]
    dependencies = [
        {
            "source_id": "source",
            "target_id": "target",
            "requires": True,
            "relation_type": "FOUNDATION_OF",
            "confidence": 0.91,
            "verification_source": "llm",
            "reason": "先掌握线性模型",
            "evidence": [{"type": "course_order", "value": 1.0}],
        },
        {
            "source_id": "target",
            "target_id": "source",
            "requires": False,
            "relation_type": "NO_RELATION",
        },
    ]

    graph = build_graph_records(
        [{"course_code": "IM121"}, {"course_code": "IM399"}],
        concepts,
        chunks,
        concept_dependencies=dependencies,
    )

    source_node = next(node for node in graph["nodes"] if node["id"] == "source")
    assert source_node["concept_id"] == "source"
    assert source_node["name"] == "线性规划"
    assert source_node["source"] == "concept_dependency"
    assert {
        (edge["source"], edge["target"])
        for edge in graph["edges"]
        if edge["type"] == "TEACHES"
    } >= {("IM121", "source"), ("IM399", "source")}
    assert any(edge["type"] == "MENTIONS" for edge in graph["edges"])
    concept_edges = [
        edge for edge in graph["edges"] if edge["type"] == "FOUNDATION_OF"
    ]
    assert len(concept_edges) == 1
    assert concept_edges[0]["confidence"] == 0.91
    assert concept_edges[0]["evidence"][0]["type"] == "course_order"


def test_graph_rejects_unverified_dangling_low_confidence_and_cyclic_edges() -> None:
    concepts = [
        {"id": value, "canonical_name": value.upper()}
        for value in ("a", "b", "c")
    ]
    dependencies = [
        {
            "source_id": "a",
            "target_id": "b",
            "requires": True,
            "relation_type": "FOUNDATION_OF",
            "confidence": 0.9,
            "verification_source": "llm",
        },
        {
            "source_id": "b",
            "target_id": "a",
            "requires": True,
            "relation_type": "FOUNDATION_OF",
            "confidence": 0.8,
            "verification_source": "llm",
        },
        {
            "source_id": "a",
            "target_id": "c",
            "requires": True,
            "relation_type": "FOUNDATION_OF",
            "confidence": 0.2,
            "verification_source": "llm",
        },
        {
            "source_id": "missing",
            "target_id": "c",
            "requires": True,
            "relation_type": "FOUNDATION_OF",
            "confidence": 0.99,
            "verification_source": "llm",
        },
        {
            "source_id": "c",
            "target_id": "b",
            "requires": "false",
            "relation_type": "FOUNDATION_OF",
            "confidence": 0.99,
            "verification_source": "llm",
        },
    ]

    graph = build_graph_records(
        [],
        concepts,
        [{"chunk_id": "chunk", "metadata": {"core_concept_ids": ["missing"]}}],
        concept_dependencies=dependencies,
        verified_min_confidence=0.6,
    )

    concept_edges = [
        (edge["source"], edge["target"])
        for edge in graph["edges"]
        if edge["type"] == "FOUNDATION_OF"
    ]
    assert concept_edges == [("a", "b")]
    assert not any(edge["type"] == "MENTIONS" for edge in graph["edges"])


class _Consumed:
    def consume(self) -> None:
        return None


class _RecordingTransaction:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        return _Consumed()


def test_neo4j_schema_matches_dependency_query_contract() -> None:
    transaction = _RecordingTransaction()
    _replace_managed_graph(
        transaction,
        [{"course_code": "IM399"}],
        [
            {
                "id": "concept-1",
                "canonical_name": "线性规划",
                "source_course_codes": ["IM399"],
            },
            {
                "id": "concept-2",
                "canonical_name": "整数规划",
                "source_course_codes": ["IM399"],
            },
        ],
        [],
        [
            {
                "source": "concept-1",
                "target": "concept-2",
                "type": "FOUNDATION_OF",
                "requires": True,
                "confidence": 0.9,
            }
        ],
        "build-1",
    )

    queries = "\n".join(query for query, _kwargs in transaction.calls)
    assert "MATCH (n {managed_by: $managed_by})" in queries
    assert "MERGE (n:ZSTT_Concept" in queries
    assert "SET n:Concept:Knowledge_Point" in queries
    assert "MERGE (a)-[r:FOUNDATION_OF]->(b)" in queries
    concept_call = next(
        kwargs
        for query, kwargs in transaction.calls
        if "ZSTT_Concept" in query and "MERGE" in query
    )
    assert concept_call["rows"][0]["props"]["source"] == "concept_dependency"
    assert concept_call["managed_by"] == "zsttSystem"


def test_concept_stage_enriches_only_teaching_chunks_and_preserves_all_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunks_path = tmp_path / "chunks.json"
    chunks = [
        {
            "chunk_id": "teaching",
            "text": "线性规划与单纯形法",
            "metadata": {
                "source_type": "syllabus",
                "section_type": "teaching_schedule",
                "course_code": "IM399",
            },
        },
        {
            "chunk_id": "overview",
            "text": "中山大学本科课程教学大纲",
            "metadata": {
                "source_type": "syllabus",
                "section_type": "overview",
                "course_code": "IM399",
            },
        },
        {
            "chunk_id": "plan",
            "text": "培养方案课程信息",
            "metadata": {"source_type": "training_plan", "course_code": "IM399"},
        },
    ]
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(run_pipeline, "CHUNKS_OUTPUT_PATH", chunks_path)
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setenv("CONCEPT_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("CONCEPT_ALIAS_PATH", str(tmp_path / "aliases.json"))
    monkeypatch.setenv("CONCEPT_CANDIDATE_EDGE_PATH", str(tmp_path / "candidates.json"))
    monkeypatch.setenv("CONCEPT_VERIFIED_EDGE_PATH", str(tmp_path / "verified.json"))
    monkeypatch.setenv("CONCEPT_EXTRACTION_CACHE_PATH", str(tmp_path / "cache.json"))

    class FakeNormalizer:
        def preprocess_chunks(self, selected, **paths):
            assert [chunk["chunk_id"] for chunk in selected] == ["teaching"]
            for key in (
                "registry_output_path",
                "alias_output_path",
                "candidate_output_path",
                "verified_output_path",
                "extraction_cache_path",
            ):
                Path(paths[key]).write_text("[]", encoding="utf-8")
            enriched = [
                {
                    **selected[0],
                    "embedding_text": "线性规划与单纯形法\n\n核心概念: 单纯形法",
                    "metadata": {
                        **selected[0]["metadata"],
                        "core_concept_ids": ["concept-1"],
                    },
                }
            ]
            registry = [
                {
                    "id": "concept-1",
                    "canonical_name": "单纯形法",
                    "source_course_codes": ["IM399"],
                }
            ]
            return enriched, registry, {"单纯形法": ["单纯形法"]}

    monkeypatch.setattr(run_pipeline, "ConceptNormalizer", FakeNormalizer)

    run_pipeline.run_concept_stage()

    merged = json.loads(chunks_path.read_text(encoding="utf-8"))
    assert [chunk["chunk_id"] for chunk in merged] == ["teaching", "overview", "plan"]
    assert "embedding_text" in merged[0]
    assert "embedding_text" not in merged[1]
    assert "embedding_text" not in merged[2]
    compatibility = json.loads((tmp_path / "concepts.json").read_text(encoding="utf-8"))
    assert compatibility[0]["concept_id"] == "concept-1"


def test_concept_stage_without_llm_preserves_existing_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunks_path = tmp_path / "chunks.json"
    chunks = [
        {
            "chunk_id": "teaching",
            "text": "线性规划与单纯形法",
            "metadata": {
                "source_type": "syllabus",
                "section_type": "teaching_schedule",
            },
        }
    ]
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text('[{"id":"trusted"}]', encoding="utf-8")
    monkeypatch.setattr(run_pipeline, "CHUNKS_OUTPUT_PATH", chunks_path)
    monkeypatch.setenv("CONCEPT_REGISTRY_PATH", str(registry_path))

    class OfflineNormalizer:
        api_client = None

        def preprocess_chunks(self, *_args, **_kwargs):
            raise AssertionError("offline main pipeline must not rebuild concept artifacts")

    monkeypatch.setattr(run_pipeline, "ConceptNormalizer", OfflineNormalizer)

    run_pipeline.run_concept_stage()

    assert json.loads(chunks_path.read_text(encoding="utf-8")) == chunks
    assert json.loads(registry_path.read_text(encoding="utf-8")) == [
        {"id": "trusted"}
    ]


def test_concept_stage_preserves_artifacts_after_transient_llm_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "teaching",
                    "text": "线性规划",
                    "metadata": {
                        "source_type": "syllabus",
                        "section_type": "teaching_content",
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text('[{"id":"trusted"}]', encoding="utf-8")
    monkeypatch.setattr(run_pipeline, "CHUNKS_OUTPUT_PATH", chunks_path)
    monkeypatch.setenv("CONCEPT_REGISTRY_PATH", str(registry_path))

    class FailingNormalizer:
        api_client = object()
        allow_rule_fallback = True
        require_complete_llm_validation = False
        allow_embedding_fallback = True

        def preprocess_chunks(self, *_args, **_kwargs):
            assert self.allow_rule_fallback is False
            assert self.require_complete_llm_validation is True
            assert self.allow_embedding_fallback is False
            raise RuntimeError("temporary provider failure; existing artifacts preserved")

    monkeypatch.setattr(run_pipeline, "ConceptNormalizer", FailingNormalizer)

    run_pipeline.run_concept_stage()

    assert json.loads(registry_path.read_text(encoding="utf-8")) == [
        {"id": "trusted"}
    ]


def test_graph_stage_loads_canonical_registry_and_verified_edges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunks_path = tmp_path / "chunks.json"
    registry_path = tmp_path / "registry.json"
    verified_path = tmp_path / "verified.json"
    manifest_path = tmp_path / "manifest.json"
    chunks_path.write_text("[]", encoding="utf-8")
    (tmp_path / "courses.json").write_text("[]", encoding="utf-8")
    registry_path.write_text(
        json.dumps(
            [
                {"id": "source", "canonical_name": "线性规划"},
                {"id": "target", "canonical_name": "整数规划"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    verified_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "source",
                    "target_id": "target",
                    "requires": True,
                    "relation_type": "FOUNDATION_OF",
                    "confidence": 0.9,
                    "verification_source": "llm",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_pipeline, "CHUNKS_OUTPUT_PATH", chunks_path)
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setenv("CONCEPT_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("CONCEPT_VERIFIED_EDGE_PATH", str(verified_path))
    monkeypatch.setenv("GRAPH_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("CONCEPT_VERIFIED_MIN_CONFIDENCE", "0.6")
    monkeypatch.setattr(
        run_pipeline.GraphDatabase,
        "driver",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    run_pipeline.run_graph_stage()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["concepts"] == ["source", "target"]
    assert manifest["concept_dependencies"] == 1
    assert manifest["verified_min_confidence"] == 0.6
    assert manifest["neo4j"] == "unavailable"


def test_graph_stage_does_not_clear_neo4j_when_canonical_artifacts_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")
    (tmp_path / "concepts.json").write_text(
        '[{"concept_id":"legacy","name":"旧概念"}]',
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"build_id":"trusted"}', encoding="utf-8")
    monkeypatch.setattr(run_pipeline, "CHUNKS_OUTPUT_PATH", chunks_path)
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setenv("CONCEPT_REGISTRY_PATH", str(tmp_path / "missing-registry.json"))
    monkeypatch.setenv("CONCEPT_VERIFIED_EDGE_PATH", str(tmp_path / "missing-edges.json"))
    monkeypatch.setenv("GRAPH_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setattr(
        run_pipeline.GraphDatabase,
        "driver",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Neo4j must not be touched")
        ),
    )

    run_pipeline.run_graph_stage()

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "build_id": "trusted"
    }


def test_all_stage_propagates_a_failed_concept_snapshot_to_graph(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        run_pipeline,
        "run_parse_stage",
        lambda **_kwargs: calls.append(("parse", None)),
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_concept_stage",
        lambda: calls.append(("concept", False)) or False,
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_graph_stage",
        lambda **kwargs: calls.append(
            ("graph", kwargs.get("concept_stage_succeeded"))
        ),
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_embed_stage",
        lambda: calls.append(("embed", None)),
    )

    for step in run_pipeline.build_stage_map()["all"]:
        step()

    assert calls == [
        ("parse", None),
        ("concept", False),
        ("graph", False),
        ("embed", None),
    ]
