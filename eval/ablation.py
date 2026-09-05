"""Ablation runner: same gold set, one component removed at a time.

    python eval/ablation.py                     # retrieval arms only (fast)
    python eval/ablation.py --with-answers      # also grade answers per arm
    python eval/ablation.py --arms full,no-rerank

Arms differ only in the retrieval configuration, so any metric gap is
attributable to the ablated component rather than to sampling.  The embedding
arm is the exception: swapping ``hash`` for the sentence-transformer model
requires rebuilding the collection, so it is run as a separate pass with
``EMBEDDING_PROVIDER=hash python run_pipeline.py embed`` and compared by tag.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import (  # noqa: E402
    evaluate,
    load_dataset,
    resolve_dataset_path,
)
from src.config import config  # noqa: E402
from src.online_service.chroma_retriever import RerankWeights  # noqa: E402

# name -> (description, RerankWeights kwargs)
ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "full": ("完整重排（向量 + BM25 词面 + 章节/来源加权）", {}),
    "no-rerank": (
        "仅向量相似度，去掉词面分与章节加权",
        {"lexical": 0.0, "section_boost": 0.0, "lexical_scheme": "none"},
    ),
    "no-lexical": (
        "保留章节加权，去掉词面分",
        {"lexical": 0.0, "lexical_scheme": "none"},
    ),
    "overlap-lexical": (
        "词面分用改造前的 bigram 重合率（无 IDF）",
        {"lexical_scheme": "overlap"},
    ),
    "no-section-boost": ("去掉章节偏好加权", {"section_boost": 0.0}),
    "lexical-heavy": ("词面分权重提到 0.35", {"vector": 0.55, "lexical": 0.35}),
}


def run_arms(
    arm_names: list[str],
    *,
    dataset_path: Path,
    with_answers: bool,
    top_k: int,
    limit: int = 0,
) -> dict[str, Any]:
    items = load_dataset(dataset_path)
    if limit:
        items = items[:limit]
    stages = ["retrieval", "answer"] if with_answers else ["retrieval"]

    arms: dict[str, Any] = {}
    for name in arm_names:
        description, overrides = ARMS[name]
        print(f"[ablation] running arm {name}: {description}")
        report = evaluate(
            items,
            stages=stages,
            tag=f"ablation-{name}",
            dataset_path=str(dataset_path),
            top_k=top_k,
            weights=RerankWeights(**overrides),
            keep_details=False,
        )
        arms[name] = {
            "description": description,
            "weights": report["environment"]["rerank_weights"],
            "retrieval": report.get("retrieval", {}),
            "generation": report.get("generation", {}),
            "refusal": report.get("refusal", {}),
        }
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "item_count": len(items),
        "arms": arms,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 检索消融实验",
        "",
        f"- 数据集：`{report['dataset']}`（{report['item_count']} 题）",
        f"- 生成时间：{report['created_at']}",
        "",
        "| 实验组 | 说明 | Recall@1 | Recall@5 | Recall@10 | MRR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, arm in report["arms"].items():
        retrieval = arm.get("retrieval") or {}
        lines.append(
            f"| `{name}` | {arm['description']} | "
            f"{retrieval.get('recall@1', '-')} | {retrieval.get('recall@5', '-')} | "
            f"{retrieval.get('recall@10', '-')} | {retrieval.get('mrr', '-')} |"
        )
    answer_arms = {
        name: arm
        for name, arm in report["arms"].items()
        if arm.get("generation")
    }
    if answer_arms:
        lines.extend(
            [
                "",
                "| 实验组 | 答案要点命中率 | 引用正确率 | 正确拒答率 | 误拒率 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for name, arm in answer_arms.items():
            generation = arm["generation"]
            refusal = arm.get("refusal") or {}
            lines.append(
                f"| `{name}` | {generation.get('answer_key_coverage', '-')} | "
                f"{generation.get('citation_precision', '-')} | "
                f"{refusal.get('correct_refusal_rate', '-')} | "
                f"{refusal.get('false_refusal_rate', '-')} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--with-answers", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()

    arm_names = [name.strip() for name in arguments.arms.split(",") if name.strip()]
    unknown = [name for name in arm_names if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; available: {sorted(ARMS)}")

    dataset_path = resolve_dataset_path(arguments.dataset)
    report = run_arms(
        arm_names,
        dataset_path=dataset_path,
        with_answers=arguments.with_answers,
        top_k=arguments.top_k,
        limit=arguments.limit,
    )
    directory = config.eval_report_dir
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{report['run_id']}-ablation"
    (directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (directory / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\n[ablation] wrote {directory / f'{stem}.md'}")


if __name__ == "__main__":
    main()
