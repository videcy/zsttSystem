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
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

ROUTE_LABELS = ("fact", "content", "dependency", "catalog", "hybrid", "structure")
DEFAULT_STAGES = ("routing", "retrieval", "answer")


def resolve_git_commit(
    *,
    env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> str:
    """Return the build/runtime commit without making eval depend on git."""
    environment = os.environ if env is None else env
    from_env = str(environment.get("GIT_COMMIT") or "").strip()
    if from_env:
        return from_env

    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = str(result.stdout or "").strip()
    return commit or "unknown"


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
                "acceptable_routes": list(
                    getattr(item, "acceptable_routes", ()) or ()
                ),
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


def _nli_claim_rows(route_result: Any) -> dict[str, Any]:
    """把一条回答的 NLI 校验明细整理成可导出的 claim 列表。

    每条 claim 记录 verdict / 相似度 / 是否被保留；保留规则与
    ``generator._retain_entailed_claims`` 一致：出现矛盾则整题拒答（全部不保留），
    否则只保留判为 ``Entailment`` 的句子。
    """
    metadata = getattr(route_result, "metadata", None) or {}
    details = metadata.get("nli_details") or []
    has_contradiction = any(
        isinstance(entry, dict) and str(entry.get("label")) == "Contradiction"
        for entry in details
    )
    claims = [
        {
            "sentence": str(entry.get("sentence", "")),
            "verdict": str(entry.get("label", "")),
            "score": entry.get("score"),
            "retained": (not has_contradiction) and str(entry.get("label")) == "Entailment",
        }
        for entry in details
        if isinstance(entry, dict)
    ]
    return {
        "nli_status": metadata.get("nli_status", "skipped"),
        "nli_verified": metadata.get("nli_verified"),
        "nli_attempts": metadata.get("nli_attempts", 0),
        "nli_details": claims,
    }


async def grade_answers(
    items: Sequence[GoldItem],
    router: QueryRouter,
    *,
    llm_client: Any = None,
    neo4j_driver: Any = None,
    persona: str | None = None,
    dump_nli: bool = False,
    known_instructor_names: Sequence[str] = (),
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
        row: dict[str, Any] = {
            "id": item.id,
            "subset": item.subset,
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
            "citation_precision_chunk": metrics.citation_precision_chunk(
                route_result.citations,
                gold_chunk_ids=item.gold_chunk_ids,
            ),
            # M5.2 新指标：章节覆盖率 / 顺序正确率 / 教师字段准确率。
            "chapter_coverage": metrics.chapter_coverage(
                answer, item.gold_chapter_titles
            ),
            "order_correct": metrics.chapter_order_correct(
                answer, item.gold_chapter_titles
            ),
            "instructor_correct": metrics.instructor_set_correct(
                answer,
                item.expected_instructors,
                known_names=known_instructor_names,
            ),
            "answer_preview": answer[:160],
        }
        if dump_nli:
            row.update(_nli_claim_rows(route_result))
        results.append(row)
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
    model = config.retrieval_model_name
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
            "[eval] vector index is empty -- run `python run_pipeline.py --stage embed` "
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
    structure = report.get("structure")
    if structure:
        lines.extend([metrics.render_metric_table(structure, "结构（Structure）"), ""])
    instructor = report.get("instructor")
    if instructor:
        lines.extend(
            [metrics.render_metric_table(instructor, "教师字段（Instructor）"), ""]
        )
    refusal_by_subset = report.get("refusal_by_subset")
    if refusal_by_subset:
        lines.extend(
            [
                "### 拒答（按子集）",
                "",
                "| 子集 | 无答案题 | 正确拒答率 | 幻觉率 | 误拒率 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for subset, section in refusal_by_subset.items():
            lines.append(
                f"| `{subset}` | {section.get('unanswerable_items', '-')} | "
                f"{section.get('correct_refusal_rate', '-')} | "
                f"{section.get('hallucination_rate', '-')} | "
                f"{section.get('false_refusal_rate', '-')} |"
            )
        lines.append("")
    nli = report.get("nli_attribution")
    if nli:
        lines.extend([metrics.render_metric_table(nli, "NLI 归因（--dump-nli）"), ""])
    concept_edges = report.get("concept_edges")
    if concept_edges:
        lines.extend(
            [
                "### 概念依赖边人工抽检",
                "",
                "| 准确率 | 可接受率 | 样本量 | 已标注 | 抽样方式 |",
                "| --- | --- | --- | --- | --- |",
                f"| {concept_edges.get('accuracy', 0.0):.2%} | "
                f"{concept_edges.get('acceptable_accuracy', 0.0):.2%} | "
                f"{concept_edges.get('sample_size', 0)} | "
                f"{concept_edges.get('labelled_size', 0)} | "
                f"`{concept_edges.get('sampling_method', '')}` |",
                "",
            ]
        )
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
    dump_nli: bool = False,
    concept_edges_gold: str | Path | None = None,
) -> dict[str, Any]:
    """Run the requested stages and assemble one report dictionary."""
    stages = tuple(stages)
    known_instructor_names = sorted(
        {name for item in items for name in item.expected_instructors if name}
    )
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
            "git_commit": resolve_git_commit(),
            "embedding_model": config.retrieval_model_name,
            "collection": retriever.collection_name if retriever else None,
            "indexed_chunks": retriever.count if retriever else 0,
            "bm25_available": bool(retriever and retriever.bm25),
            "rerank_weights": (weights or RerankWeights()).as_dict(),
            "llm_available": llm_client is not None,
            "neo4j_available": neo4j_driver is not None,
            "persona": persona or "per-item",
        },
    }
    gold_path = Path(concept_edges_gold) if concept_edges_gold else Path(
        "eval/datasets/concept_edges_gold.json"
    )
    if gold_path.exists():
        from eval.concept_edges_eval import evaluate_gold

        report["concept_edges"] = evaluate_gold(gold_path)

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
                    dump_nli=dump_nli,
                    known_instructor_names=known_instructor_names,
                )
            )
            report["generation"] = answers["generation"]
            report["refusal"] = answers["refusal"]
            # M5.2 新指标：拒答按子集切片、章节覆盖/顺序、教师字段准确率。
            report["refusal_by_subset"] = metrics.refusal_report_by_subset(
                answers["results"]
            )
            report["structure"] = metrics.structure_report(answers["results"])
            report["instructor"] = metrics.instructor_report(answers["results"])
            if dump_nli:
                # P0-5 归因表：只有显式开启 --dump-nli 时才写入报告。
                report["nli_attribution"] = metrics.nli_attribution(answers["results"])
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
    parser.add_argument(
        "--dump-nli",
        action="store_true",
        help="导出逐条 claim 的 NLI 明细（verdict/相似度/是否保留）与归因表（默认关闭）",
    )
    parser.add_argument(
        "--concept-edges-gold",
        default=None,
        help="概念依赖边人工抽检金标；默认读取 eval/datasets/concept_edges_gold.json（存在时）",
    )
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
        dump_nli=arguments.dump_nli,
        concept_edges_gold=arguments.concept_edges_gold,
    )
    directory = Path(arguments.output_dir) if arguments.output_dir else config.eval_report_dir
    json_path, markdown_path = write_report(report, directory)
    print(render_markdown(report))
    print(f"\n[eval] wrote {json_path}\n[eval] wrote {markdown_path}")


if __name__ == "__main__":
    main()
