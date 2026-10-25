from __future__ import annotations

from collections import Counter

from eval.concept_edges_eval import (
    build_template,
    evaluate_annotations,
    load_selected_edges,
    stratified_sample,
)


def _edge(index: int, pair: str) -> dict:
    return {
        "id": f"edge-{index:03d}",
        "source_id": f"s-{index}",
        "target_id": f"t-{index}",
        "requires": True,
        "relation_type": "FOUNDATION_OF",
        "confidence": 0.9,
        "course_pair": pair,
        "source_evidence": f"source evidence {index}",
        "target_evidence": f"target evidence {index}",
    }


def test_stratified_sample_is_proportional_and_deterministic() -> None:
    edges = [
        *[_edge(index, "IM101->IM201") for index in range(60)],
        *[_edge(index, "IM102->IM202") for index in range(30, 60)],
        *[_edge(index, "IM103->IM203") for index in range(50, 60)],
    ]

    first = stratified_sample(edges, sample_size=50, seed=7)
    second = stratified_sample(edges, sample_size=50, seed=7)

    assert [edge["id"] for edge in first] == [edge["id"] for edge in second]
    counts = Counter(edge["course_pair"] for edge in first)
    assert len(first) == 50
    assert counts == {
        "IM101->IM201": 30,
        "IM102->IM202": 15,
        "IM103->IM203": 5,
    }


def test_template_records_sampling_method_and_evidence() -> None:
    edges = [_edge(index, "IM101->IM201") for index in range(5)]
    payload = build_template(edges, sample_size=3, seed=11, source="edges.json")

    assert payload["sampling"]["method"] == "stratified_by_course_pair"
    assert payload["sampling"]["population_size"] == 5
    assert payload["sampling"]["sample_size"] == 3
    assert payload["sampling"]["random_seed"] == 11
    assert payload["sampling"]["strata"]["IM101->IM201"] == {
        "population": 5,
        "sampled": 3,
    }
    assert all(item["label"] == "" for item in payload["items"])
    assert all(item["source_evidence"] for item in payload["items"])
    assert all(item["target_evidence"] for item in payload["items"])


def test_accuracy_counts_only_accepted_edges_as_correct() -> None:
    payload = {
        "sampling": {
            "method": "stratified_by_course_pair",
            "population_size": 120,
            "sample_size": 4,
        },
        "items": [
            {"id": "1", "label": "成立", "course_pair": "A->B"},
            {"id": "2", "label": "成立", "course_pair": "A->B"},
            {"id": "3", "label": "不成立", "course_pair": "B->C"},
            {"id": "4", "label": "勉强", "course_pair": "B->C"},
        ],
    }

    report = evaluate_annotations(payload)

    assert report["accuracy"] == 0.5
    assert report["acceptable_accuracy"] == 0.75
    assert report["sample_size"] == 4
    assert report["labelled_size"] == 4
    assert report["label_counts"] == {"成立": 2, "不成立": 1, "勉强": 1}
    assert report["sampling_method"] == "stratified_by_course_pair"


def test_load_selected_edges_filters_rejected_edges_and_marks_course_pair(tmp_path) -> None:
    path = tmp_path / "edges.json"
    path.write_text(
        """
        [
          {"source_id":"s","target_id":"t","requires":true,
           "source_course_codes":["IM101"],"target_course_codes":["IM201"]},
          {"source_id":"x","target_id":"y","requires":false,
           "source_course_codes":["IM101"],"target_course_codes":["IM201"]}
        ]
        """,
        encoding="utf-8",
    )

    edges = load_selected_edges(path)

    assert len(edges) == 1
    assert edges[0]["course_pair"] == "IM101->IM201"
