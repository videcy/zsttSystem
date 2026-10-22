"""Stratified human review for verified concept-dependency edges.

    python eval/concept_edges_eval.py --make-template
    python eval/concept_edges_eval.py

The template samples up to 50 accepted edges proportionally from each course
pair.  Annotators fill ``label`` with one of ``成立 / 不成立 / 勉强`` and keep
the file at ``eval/datasets/concept_edges_gold.json`` (the one dataset file
explicitly un-ignored for this workflow).  Accuracy is strict: only ``成立``
counts as correct; ``acceptable_accuracy`` additionally reports ``成立+勉强``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LABELS = ("成立", "不成立", "勉强")
DEFAULT_EDGES_PATH = Path("outputs/concept_verified_edges.json")
DEFAULT_REGISTRY_PATH = Path("outputs/concept_registry.json")
DEFAULT_GOLD_PATH = Path("eval/datasets/concept_edges_gold.json")
DEFAULT_REPORT_DIR = Path("eval/reports")
SAMPLING_METHOD = "stratified_by_course_pair"


def _load_json(path: str | Path, default: Any) -> Any:
    source = Path(path)
    if not source.exists():
        return default
    return json.loads(source.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def registry_by_id(registry: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(concept.get("id") or concept.get("concept_id")): dict(concept)
        for concept in registry
        if concept.get("id") or concept.get("concept_id")
    }


def _first_course_code(
    edge: Mapping[str, Any],
    concept: Mapping[str, Any] | None,
    *,
    side: str,
) -> str:
    for value in _as_list(edge.get(f"{side}_course_codes")):
        code = str(value or "").strip()
        if code:
            return code
    for value in _as_list(edge.get(f"{side}_course_code")):
        code = str(value or "").strip()
        if code:
            return code
    if concept:
        for value in _as_list(concept.get("source_course_codes")):
            code = str(value or "").strip()
            if code:
                return code
        for value in _as_list(concept.get("source_occurrences")):
            if isinstance(value, Mapping):
                code = str(value.get("course_code") or "").strip()
                if code:
                    return code
    return ""


def course_pair_for_edge(
    edge: Mapping[str, Any],
    concepts: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return a stable, directed course-pair label for stratification."""
    explicit = str(edge.get("course_pair") or "").strip()
    if explicit:
        return explicit
    source = concepts.get(str(edge.get("source_id") or ""))
    target = concepts.get(str(edge.get("target_id") or ""))
    source_code = _first_course_code(edge, source, side="source")
    target_code = _first_course_code(edge, target, side="target")
    if not source_code:
        source_code = str(edge.get("source_name") or edge.get("source_id") or "unknown")
    if not target_code:
        target_code = str(edge.get("target_name") or edge.get("target_id") or "unknown")
    return f"{source_code}->{target_code}"


def load_selected_edges(
    path: str | Path = DEFAULT_EDGES_PATH,
    *,
    concepts: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load accepted (``requires=true``) edges and attach their course pair."""
    payload = _load_json(path, [])
    rows = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"concept edge file must contain a list: {path}")
    lookup = concepts or {}
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("requires") is not True:
            continue
        edge = dict(row)
        edge["course_pair"] = course_pair_for_edge(edge, lookup)
        selected.append(edge)
    return selected


def stratified_sample(
    edges: Sequence[dict[str, Any]],
    *,
    sample_size: int = 50,
    seed: int = 20261022,
) -> list[dict[str, Any]]:
    """Sample proportionally by course pair, with deterministic tie-breaking."""
    if sample_size <= 0 or not edges:
        return []
    materialised = list(edges)
    if sample_size >= len(materialised):
        return sorted(materialised, key=lambda edge: str(edge.get("id") or ""))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in materialised:
        groups[str(edge.get("course_pair") or "unknown")].append(edge)
    names = sorted(groups)
    rng = random.Random(seed)

    if sample_size <= len(names):
        chosen = sorted(rng.sample(names, sample_size))
        return [
            rng.choice(groups[name])
            for name in chosen
        ]

    quotas = {name: 1 for name in names}
    remaining = sample_size - len(names)
    total = len(materialised)
    while remaining > 0:
        available = [name for name in names if quotas[name] < len(groups[name])]
        if not available:
            break
        # Hamilton largest-remainder allocation, with one seat per stratum
        # reserved so small course pairs are still represented.
        name = max(
            available,
            key=lambda item: (
                sample_size * len(groups[item]) / total - quotas[item],
                len(groups[item]),
                item,
            ),
        )
        quotas[name] += 1
        remaining -= 1

    sampled: list[dict[str, Any]] = []
    for name in names:
        pool = sorted(groups[name], key=lambda edge: str(edge.get("id") or ""))
        sampled.extend(rng.sample(pool, quotas[name]))
    return sorted(sampled, key=lambda edge: str(edge.get("id") or ""))


def build_template(
    edges: Sequence[dict[str, Any]],
    *,
    sample_size: int = 50,
    seed: int = 20261022,
    source: str = "",
) -> dict[str, Any]:
    """Build the human annotation payload for a stratified edge sample."""
    sampled = stratified_sample(edges, sample_size=sample_size, seed=seed)
    population = Counter(str(edge.get("course_pair") or "unknown") for edge in edges)
    selected = Counter(str(edge.get("course_pair") or "unknown") for edge in sampled)
    items: list[dict[str, Any]] = []
    for index, edge in enumerate(sampled, start=1):
        items.append(
            {
                "id": str(edge.get("id") or f"concept-edge-{index:03d}"),
                "label": "",
                "source_id": edge.get("source_id"),
                "source_name": edge.get("source_name"),
                "target_id": edge.get("target_id"),
                "target_name": edge.get("target_name"),
                "course_pair": edge.get("course_pair"),
                "relation_type": edge.get("relation_type"),
                "confidence": edge.get("confidence"),
                "reason": edge.get("reason"),
                "source_evidence": edge.get("source_evidence", ""),
                "target_evidence": edge.get("target_evidence", ""),
            }
        )
    return {
        "description": "概念依赖边人工抽检金标（成立/不成立/勉强）",
        "labels": list(LABELS),
        "sampling": {
            "method": SAMPLING_METHOD,
            "population_size": len(edges),
            "sample_size": len(items),
            "requested_sample_size": sample_size,
            "random_seed": seed,
            "source": source,
            "strata": {
                name: {
                    "population": population[name],
                    "sampled": selected[name],
                }
                for name in sorted(population)
            },
        },
        "items": items,
    }


def evaluate_annotations(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute strict accuracy, acceptable accuracy, and sample metadata."""
    if isinstance(payload, Mapping):
        items = payload.get("items", [])
        sampling = payload.get("sampling") or {}
    else:
        items = payload
        sampling = {}
    rows = [item for item in items if isinstance(item, Mapping)]

    label_counts: Counter[str] = Counter()
    by_pair: dict[str, Counter[str]] = defaultdict(Counter)
    for item in rows:
        label = str(item.get("label") or "").strip()
        if label not in LABELS:
            continue
        label_counts[label] += 1
        by_pair[str(item.get("course_pair") or "unknown")][label] += 1

    labelled_size = sum(label_counts.values())
    correct = label_counts["成立"]
    acceptable = correct + label_counts["勉强"]
    accuracy = correct / labelled_size if labelled_size else 0.0
    acceptable_accuracy = acceptable / labelled_size if labelled_size else 0.0
    pair_accuracy = {}
    for pair, counts in sorted(by_pair.items()):
        pair_total = sum(counts.values())
        pair_accuracy[pair] = {
            "labelled": pair_total,
            "accuracy": round(counts["成立"] / pair_total, 6) if pair_total else 0.0,
        }

    return {
        "accuracy": round(accuracy, 6),
        "acceptable_accuracy": round(acceptable_accuracy, 6),
        "sample_size": len(rows),
        "labelled_size": labelled_size,
        "unlabelled_size": len(rows) - labelled_size,
        "correct": correct,
        "acceptable": acceptable,
        "label_counts": {label: label_counts[label] for label in LABELS},
        "accuracy_by_course_pair": pair_accuracy,
        "sampling_method": sampling.get("method", SAMPLING_METHOD),
        "sampling": dict(sampling),
    }


def evaluate_gold(path: str | Path = DEFAULT_GOLD_PATH) -> dict[str, Any]:
    return evaluate_annotations(_load_json(path, {}))


def render_markdown(report: Mapping[str, Any]) -> str:
    sampling = report.get("sampling") or {}
    lines = [
        "# 概念依赖边人工抽检",
        "",
        f"- 准确率（仅“成立”）：{report.get('accuracy', 0.0):.2%}",
        f"- 可接受率（“成立+勉强”）：{report.get('acceptable_accuracy', 0.0):.2%}",
        f"- 样本量：{report.get('sample_size', 0)}",
        f"- 已标注：{report.get('labelled_size', 0)}；未标注：{report.get('unlabelled_size', 0)}",
        f"- 抽样方式：`{report.get('sampling_method', SAMPLING_METHOD)}`",
        f"- 随机种子：{sampling.get('random_seed', '-')}；总体：{sampling.get('population_size', '-')}",
        "",
        "| 标注 | 数量 |",
        "| --- | --- |",
    ]
    for label, count in (report.get("label_counts") or {}).items():
        lines.append(f"| {label} | {count} |")
    return "\n".join(lines)


def write_evaluation_report(
    report: Mapping[str, Any],
    directory: str | Path = DEFAULT_REPORT_DIR,
    *,
    stem: str = "concept-edges",
) -> tuple[Path, Path]:
    output = Path(directory)
    json_path = _write_json(output / f"{stem}.json", report)
    markdown_path = output / f"{stem}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", default=str(DEFAULT_EDGES_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH))
    parser.add_argument("--make-template", action="store_true")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    registry = registry_by_id(_load_json(arguments.registry, []))
    if arguments.make_template:
        edges = load_selected_edges(arguments.edges, concepts=registry)
        payload = build_template(
            edges,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
            source=arguments.edges,
        )
        gold_path = Path(arguments.gold)
        if gold_path.exists() and not arguments.force:
            raise SystemExit(f"{gold_path} already exists; pass --force to overwrite")
        _write_json(gold_path, payload)
        print(f"[concept-edges] wrote {len(payload['items'])} samples to {gold_path}")
        return

    report = evaluate_gold(arguments.gold)
    json_path, markdown_path = write_evaluation_report(report, arguments.report_dir)
    print(render_markdown(report))
    print(f"\n[concept-edges] wrote {json_path}\n[concept-edges] wrote {markdown_path}")


if __name__ == "__main__":
    main()
