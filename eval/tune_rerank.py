"""Grid search over the rerank weights, plus a sensitivity curve.

    python eval/tune_rerank.py --metric recall@5

Replaces "0.75 / 0.15 / 0.08 because they looked right" with a swept surface
over the gold set.  Two outputs:

* the best configuration, ready to paste into ``.env``
* a one-factor-at-a-time sensitivity table -- how much each weight moves the
  metric while the others stay at the best value, which is the figure a
  reviewer asks for when a paper reports tuned weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import (  # noqa: E402
    build_retriever,
    grade_retrieval,
    load_dataset,
    resolve_dataset_path,
)
from src.config import config  # noqa: E402
from src.online_service.chroma_retriever import RerankWeights  # noqa: E402

DEFAULT_VECTOR = (0.55, 0.65, 0.75, 0.85, 1.0)
DEFAULT_LEXICAL = (0.0, 0.1, 0.15, 0.25, 0.35)
DEFAULT_SECTION = (0.0, 0.04, 0.08, 0.16)


def _score(report: dict[str, Any], metric: str) -> float:
    value = report.get(metric)
    return float(value) if value is not None else 0.0


def sweep(
    items: Sequence[Any],
    *,
    metric: str,
    top_k: int,
    vector_values: Sequence[float],
    lexical_values: Sequence[float],
    section_values: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vector, lexical, section in product(
        vector_values,
        lexical_values,
        section_values,
    ):
        weights = RerankWeights(
            vector=vector,
            lexical=lexical,
            section_boost=section,
        )
        # One retriever per configuration: the weights are read at scoring
        # time, and a fresh instance also resets the BM25 cache.
        retriever = build_retriever(weights)
        if retriever is None:
            raise SystemExit(
                "vector index unavailable; run `python run_pipeline.py embed`"
            )
        report = grade_retrieval(items, retriever, top_k=top_k)["report"]
        rows.append(
            {
                "vector": vector,
                "lexical": lexical,
                "section_boost": section,
                "score": _score(report, metric),
                "metrics": report,
            }
        )
    rows.sort(key=lambda row: -row["score"])
    return rows


def sensitivity(
    rows: Sequence[dict[str, Any]],
    best: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Vary one weight at a time, holding the other two at their best value."""
    axes = ("vector", "lexical", "section_boost")
    curves: dict[str, list[dict[str, Any]]] = {}
    for axis in axes:
        held = [other for other in axes if other != axis]
        points = [
            {"value": row[axis], "score": row["score"]}
            for row in rows
            if all(row[other] == best[other] for other in held)
        ]
        curves[axis] = sorted(points, key=lambda point: point["value"])
    return curves


def render_markdown(report: dict[str, Any]) -> str:
    best = report["best"]
    lines = [
        "# 重排权重网格搜索",
        "",
        f"- 数据集：`{report['dataset']}`（{report['item_count']} 题）",
        f"- 优化指标：`{report['metric']}`，共 {report['configurations']} 组配置",
        "",
        "## 最优配置",
        "",
        "```dotenv",
        f"RERANK_WEIGHT_VECTOR={best['vector']}",
        f"RERANK_WEIGHT_LEXICAL={best['lexical']}",
        f"RERANK_SECTION_BOOST={best['section_boost']}",
        "```",
        "",
        f"`{report['metric']}` = {best['score']}"
        f"（改动前默认配置：{report['baseline']['score']}）",
        "",
        "## Top 10 配置",
        "",
        "| vector | lexical | section_boost | " + report["metric"] + " | MRR |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["top"]:
        lines.append(
            f"| {row['vector']} | {row['lexical']} | {row['section_boost']} | "
            f"{row['score']} | {row['metrics'].get('mrr', '-')} |"
        )
    lines.extend(["", "## 权重敏感性（其余权重固定在最优值）", ""])
    for axis, points in report["sensitivity"].items():
        rendered = "，".join(
            f"{point['value']}→{point['score']}" for point in points
        )
        lines.append(f"- **{axis}**：{rendered}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--metric", default="recall@5")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="halve the grid for a quick pass",
    )
    arguments = parser.parse_args()

    dataset_path = resolve_dataset_path(arguments.dataset)
    items = load_dataset(dataset_path)
    if arguments.limit:
        items = items[: arguments.limit]

    vector_values = DEFAULT_VECTOR[::2] if arguments.coarse else DEFAULT_VECTOR
    lexical_values = DEFAULT_LEXICAL[::2] if arguments.coarse else DEFAULT_LEXICAL
    section_values = DEFAULT_SECTION[::2] if arguments.coarse else DEFAULT_SECTION

    rows = sweep(
        items,
        metric=arguments.metric,
        top_k=arguments.top_k,
        vector_values=vector_values,
        lexical_values=lexical_values,
        section_values=section_values,
    )
    best = rows[0]
    default = RerankWeights()
    baseline = next(
        (
            row
            for row in rows
            if row["vector"] == default.vector
            and row["lexical"] == default.lexical
            and row["section_boost"] == default.section_boost
        ),
        {"score": None},
    )
    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "item_count": len(items),
        "metric": arguments.metric,
        "configurations": len(rows),
        "best": best,
        "baseline": baseline,
        "top": rows[:10],
        "sensitivity": sensitivity(rows, best),
    }
    directory = config.eval_report_dir
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{report['run_id']}-rerank-grid"
    (directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (directory / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
