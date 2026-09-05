"""Run the gold set against the live system and write a metrics report.

    python eval/run_eval.py                      # every stage that can run
    python eval/run_eval.py --stages routing     # no index or API key needed
    python eval/run_eval.py --persona teacher --tag teacher-run

Three stages, each degrading independently so that a partial environment
still produces numbers:

``routing``    pure regex/classifier grading -- no backend required
``retrieval``  Recall@k and MRR -- requires a built Chroma collection
``answer``     end-to-end answers, citations and refusals -- also uses the
               LLM and Neo4j when they are configured, and falls back to the
               deterministic templates when they are not (which is itself a
               measurable arm, recorded as ``llm_available: false``)

Reports land in ``eval/reports/<timestamp>-<tag>.{json,md}``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

# Entry points outside src/ do not otherwise read .env, and scoring a
# different collection than the service serves would be worse than useless.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from eval import metrics  # noqa: E402
from eval.schema import GoldItem, dataset_summary, load_dataset  # noqa: E402
from src.config import config  # noqa: E402
from src.online_service.chroma_retriever import ChromaRetriever, RerankWeights  # noqa: E402
from src.online_service.query_router import QueryRouter  # noqa: E402

ROUTE_LABELS = ("fact", "content", "dependency", "catalog", "hybrid")
DEFAULT_STAGES = ("routing", "retrieval", "answer")


def resolve_dataset_path(explicit: str | None = None) -> Path:
    """Prefer the human-curated set, fall back to the generated seed."""
    if explicit:
        return Path(explicit)
    curated = config.eval_dataset_path
    if curated.exists():
        return curated
    seed = curated.with_name("gold_seed.json")
    if seed.exists():
        return seed
    raise SystemExit(
        f"no gold dataset at {curated} or {seed}\n"
        "run `python eval/build_seed_dataset.py` first"
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def grade_routing(items: Sequence[GoldItem]) -> dict[str, Any]:
    results = []
    for item in items:
        prediction = QueryRouter.classify_intent(item.question)
        results.append(
            {
                "id": item.id,
                "expected_route": item.expected_route,
                "predicted_route": prediction.primary.value,
                "predicted_labels": [label.value for label in prediction.labels],
                "confidence": prediction.confidence,
            }
        )
    report = metrics.routing_report(results, ROUTE_LABELS)
    return {"report": report, "results": results}


def grade_retrieval(
    items: Sequence[GoldItem],
    retriever: ChromaRetriever,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    results = []
    for item in items:
        if not item.answerable:
            continue
        gradable = bool(item.gold_chunk_ids or item.gold_course_codes)
        hits = retriever.search(item.question, top_k) if gradable else []
        positions = metrics.hit_positions(
            hits,
            item.gold_chunk_ids,
            gold_course_codes=item.gold_course_codes,
            gold_section_types=item.gold_section_types,
        )
        results.append(
            {
                "id": item.id,
                "gradable": gradable,
                "grading": "chunk" if item.gold_chunk_ids else "course",
                "positions": positions,
                "retrieved": len(hits),
            }
        )
    return {"report": metrics.retrieval_report(results), "results": results}


async def grade_answers(
    items: Sequence[GoldItem],
    router: QueryRouter,
    *,
    llm_client: Any = None,
    neo4j_driver: Any = None,
    persona: str | None = None,
) -> dict[str, Any]:
    results = []
    for item in items:
        route_result = await router.route(
            item.question,
            item.id,
            neo4j_driver=neo4j_driver,
            llm_client=llm_client,
            persona=persona or item.persona,
        )
        answer = route_result.answer or ""
        refused = metrics.is_refusal(answer)
        results.append(
            {
                "id": item.id,
                "answerable": item.answerable,
                "answer_keys": item.answer_keys,
                "route": route_result.query_type,
                "refused": refused,
                "answer_correct": metrics.answer_covers_keys(answer, item.answer_keys),
                "citation_count": len(route_result.citations),
                "citation_precision": metrics.citation_precision(
                    route_result.citations,
                    gold_chunk_ids=item.gold_chunk_ids,
                    gold_course_codes=item.gold_course_codes,
                ),
                "answer_preview": answer[:160],
            }
        )
    return {
        "generation": metrics.generation_report(results),
        "refusal": metrics.refusal_report(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def build_retriever(
    weights: RerankWeights | None = None,
    *,
    collection_name: str | None = None,
) -> ChromaRetriever | None:
    """Return a retriever with a populated index, or ``None``.

    ``collection_name`` bypasses the alias pointer, which is how the
    retraining updater scores a candidate collection before promoting it.
    """
    model = (
        config.local_embedding_model
        if config.embedding_provider == "local"
        else "hash"
    )
    try:
        retriever = ChromaRetriever(
            model,
            weights=weights,
            collection_name=collection_name,
        )
    except Exception as exc:  # noqa: BLE001 - environment probe
        print(f"[eval] retriever unavailable: {exc}")
        return None
    if not retriever.connected or retriever.count == 0:
        print(
            "[eval] vector index is empty -- run `python run_pipeline.py embed` "
            "to enable the retrieval and answer stages"
        )
        return None
    return retriever


def build_llm_client() -> Any:
    from src.utils.deepseek_client import create_deepseek_client

    try:
        return create_deepseek_client()
    except (ValueError, RuntimeError) as exc:
        print(f"[eval] LLM unavailable ({exc}); scoring the fallback path")
        return None


def build_neo4j_driver() -> Any:
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

    driver = None
    try:
        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
            connection_timeout=2,
            connection_acquisition_timeout=2,
        )
        driver.verify_connectivity()
        return driver
    except (AuthError, ServiceUnavailable, Neo4jError, OSError) as exc:
        if driver is not None:
            driver.close()
        print(f"[eval] Neo4j unavailable ({exc}); dependency answers use fallbacks")
        return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# zsttSystem 评测报告 · {report['tag']}",
        "",
        f"- 生成时间：{report['created_at']}",
        f"- 数据集：`{report['dataset']}`（{report['dataset_summary']['total']} 题，"
        f"其中无答案题 {report['dataset_summary']['unanswerable']} 题）",
        f"- 检索模型：`{report['environment']['embedding_model']}`，"
        f"collection `{report['environment']['collection']}`",
        f"- 重排权重：{report['environment']['rerank_weights']}",
        f"- LLM 可用：{report['environment']['llm_available']}；"
        f"Neo4j 可用：{report['environment']['neo4j_available']}",
        "",
    ]
    routing = report.get("routing")
    if routing:
        lines.append(metrics.render_metric_table(routing, "路由（Routing）"))
        lines.extend(
            [
                "",
                "#### 混淆矩阵",
                "",
                metrics.render_confusion_matrix(
                    routing["confusion_matrix"],
                    ROUTE_LABELS,
                ),
                "",
                "#### 分类别 P/R/F1",
                "",
                "| 类别 | precision | recall | f1 | support |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for label, scores in routing["per_class"].items():
            lines.append(
                f"| {label} | {scores['precision']} | {scores['recall']} | "
                f"{scores['f1']} | {scores['support']} |"
            )
        lines.append("")
    for key, title in (
        ("retrieval", "检索（Retrieval）"),
        ("generation", "生成与引用（Generation）"),
        ("refusal", "拒答（Refusal）"),
    ):
        section = report.get(key)
        if section:
            lines.extend([metrics.render_metric_table(section, title), ""])
    return "\n".join(lines)


def write_report(report: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{report['run_id']}-{report['tag']}"
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def evaluate(
    items: Sequence[GoldItem],
    *,
    stages: Iterable[str] = DEFAULT_STAGES,
    tag: str = "run",
    dataset_path: str = "",
    top_k: int = 10,
    persona: str | None = None,
    weights: RerankWeights | None = None,
    use_llm: bool = True,
    use_neo4j: bool = True,
    keep_details: bool = True,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Run the requested stages and assemble one report dictionary."""
    stages = tuple(stages)
    retriever = (
        build_retriever(weights, collection_name=collection_name)
        if {"retrieval", "answer"} & set(stages)
        else None
    )
    llm_client = build_llm_client() if use_llm and "answer" in stages else None
    neo4j_driver = build_neo4j_driver() if use_neo4j and "answer" in stages else None

    report: dict[str, Any] = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "dataset": dataset_path,
        "dataset_summary": dataset_summary(items),
        "stages": list(stages),
        "environment": {
            "embedding_model": (
                config.local_embedding_model
                if config.embedding_provider == "local"
                else "hash"
            ),
            "collection": retriever.collection_name if retriever else None,
            "indexed_chunks": retriever.count if retriever else 0,
            "bm25_available": bool(retriever and retriever.bm25),
            "rerank_weights": (weights or RerankWeights()).as_dict(),
            "llm_available": llm_client is not None,
            "neo4j_available": neo4j_driver is not None,
            "persona": persona or "per-item",
        },
    }

    try:
        if "routing" in stages:
            routing = grade_routing(items)
            report["routing"] = routing["report"]
            if keep_details:
                report["routing_details"] = routing["results"]

        if "retrieval" in stages and retriever is not None:
            retrieval = grade_retrieval(items, retriever, top_k=top_k)
            report["retrieval"] = retrieval["report"]
            if keep_details:
                report["retrieval_details"] = retrieval["results"]

        if "answer" in stages and retriever is not None:
            router = QueryRouter(retriever)
            answers = asyncio.run(
                grade_answers(
                    items,
                    router,
                    llm_client=llm_client,
                    neo4j_driver=neo4j_driver,
                    persona=persona,
                )
            )
            report["generation"] = answers["generation"]
            report["refusal"] = answers["refusal"]
            if keep_details:
                report["answer_details"] = answers["results"]
    finally:
        if neo4j_driver is not None:
            neo4j_driver.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help="comma-separated subset of routing,retrieval,answer",
    )
    parser.add_argument("--tag", default="run")
    parser.add_argument("--limit", type=int, default=0, help="grade the first N items")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--persona",
        default=None,
        choices=("student", "teacher", "visitor"),
        help="override every item's persona",
    )
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-neo4j", action="store_true")
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()

    dataset_path = resolve_dataset_path(arguments.dataset)
    items = load_dataset(dataset_path)
    if arguments.limit:
        items = items[: arguments.limit]

    report = evaluate(
        items,
        stages=[stage.strip() for stage in arguments.stages.split(",") if stage.strip()],
        tag=arguments.tag,
        dataset_path=str(dataset_path),
        top_k=arguments.top_k,
        persona=arguments.persona,
        use_llm=not arguments.no_llm,
        use_neo4j=not arguments.no_neo4j,
    )
    directory = Path(arguments.output_dir) if arguments.output_dir else config.eval_report_dir
    json_path, markdown_path = write_report(report, directory)
    print(render_markdown(report))
    print(f"\n[eval] wrote {json_path}\n[eval] wrote {markdown_path}")


if __name__ == "__main__":
    main()
