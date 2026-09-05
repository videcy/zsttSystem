"""Measure how much the three personas actually change retrieval.

    python eval/persona_overlap.py --top-k 5

The persona layer only adjusts ``top_k``, section preference and a 0.04-0.06
source boost, all applied on top of a 0.75-weighted vector score.  Whether
that is enough to change what the user sees is an empirical question, and the
answer decides how the paper may describe it:

* high overlap (>0.9 at Top-5) -- persona is evidence *ordering*, and the
  wording must say so, not claim personalised generation;
* meaningful divergence -- the claim is supported, and this table is the
  evidence for it.

Overlap is measured on the persona-aware path the service really uses
(``QueryRouter._retrieve_course_evidence``), not on a reimplementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import build_retriever, load_dataset, resolve_dataset_path  # noqa: E402
from src.config import config  # noqa: E402
from src.online_service.persona import PERSONA_PROFILES  # noqa: E402
from src.online_service.query_router import QueryRouter  # noqa: E402

PERSONAS = tuple(PERSONA_PROFILES)


def _overlap(left: Sequence[str], right: Sequence[str]) -> tuple[float, float]:
    """Return ``(overlap@k, jaccard)`` for two ranked id lists."""
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0, 1.0
    shared = len(left_set & right_set)
    smaller = min(len(left_set), len(right_set)) or 1
    union = len(left_set | right_set) or 1
    return shared / smaller, shared / union


def _top_rank_changed(left: Sequence[str], right: Sequence[str]) -> bool:
    return bool(left) and bool(right) and left[0] != right[0]


async def collect(
    router: QueryRouter,
    questions: Sequence[str],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        per_persona: dict[str, list[str]] = {}
        for persona in PERSONAS:
            # Private on purpose: this is the persona-sensitive stage, and
            # measuring anything else would measure a different system.
            _course, hits = await router._retrieve_course_evidence(question, persona)
            per_persona[persona] = [
                str(hit.get("chunk_id") or hit.get("text", ""))[:64]
                for hit in hits[:top_k]
            ]
        rows.append({"question": question, "hits": per_persona})
    return rows


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs = list(combinations(PERSONAS, 2))
    totals = {
        f"{left}-{right}": {"overlap": 0.0, "jaccard": 0.0, "top1_changed": 0}
        for left, right in pairs
    }
    empty = 0
    for row in rows:
        if not any(row["hits"].values()):
            empty += 1
            continue
        for left, right in pairs:
            key = f"{left}-{right}"
            overlap, jaccard = _overlap(row["hits"][left], row["hits"][right])
            totals[key]["overlap"] += overlap
            totals[key]["jaccard"] += jaccard
            totals[key]["top1_changed"] += int(
                _top_rank_changed(row["hits"][left], row["hits"][right])
            )
    scored = max(1, len(rows) - empty)
    return {
        "questions": len(rows),
        "questions_without_hits": empty,
        "pairs": {
            key: {
                "mean_overlap": round(value["overlap"] / scored, 4),
                "mean_jaccard": round(value["jaccard"] / scored, 4),
                "top1_change_rate": round(value["top1_changed"] / scored, 4),
            }
            for key, value in totals.items()
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 角色（persona）检索差异测试",
        "",
        f"- 数据集：`{report['dataset']}`，评测问题 {summary['questions']} 条"
        f"（其中 {summary['questions_without_hits']} 条无命中）",
        f"- Top-K：{report['top_k']}",
        "",
        "| 角色对 | Top-K 重合率 | Jaccard | Top-1 变化率 |",
        "| --- | --- | --- | --- |",
    ]
    for key, value in summary["pairs"].items():
        lines.append(
            f"| {key} | {value['mean_overlap']} | {value['mean_jaccard']} | "
            f"{value['top1_change_rate']} |"
        )
    highest = max(
        value["mean_overlap"] for value in summary["pairs"].values()
    )
    lines.extend(
        [
            "",
            "## 结论判据",
            "",
            "- 重合率 > 0.9：角色差异实质上不改变证据集合，论文应表述为"
            "**证据优先级调度**，并把"
            "“不改变课程事实”写成设计选择而非局限；",
            "- 重合率 < 0.7 或 Top-1 变化率 > 0.3：角色确实改变了证据，"
            "可以支撑更强的角色感知表述。",
            "",
            f"本次最高重合率 = {highest}。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=60)
    arguments = parser.parse_args()

    dataset_path = resolve_dataset_path(arguments.dataset)
    items = [item for item in load_dataset(dataset_path) if item.answerable]
    if arguments.limit:
        items = items[: arguments.limit]

    retriever = build_retriever()
    if retriever is None:
        raise SystemExit(
            "vector index unavailable; run `python run_pipeline.py embed` first"
        )
    router = QueryRouter(retriever)
    rows = asyncio.run(
        collect(router, [item.question for item in items], top_k=arguments.top_k)
    )
    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "top_k": arguments.top_k,
        "summary": summarise(rows),
        "personas": {
            persona: {
                key: value
                for key, value in profile.items()
                if key != "prompt"
            }
            for persona, profile in PERSONA_PROFILES.items()
        },
    }
    directory = config.eval_report_dir
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{report['run_id']}-persona-overlap"
    (directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (directory / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
