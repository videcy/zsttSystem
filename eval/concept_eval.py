"""Precision / recall / F1 for concept extraction: rule baseline vs LLM.

    python eval/concept_eval.py --make-template --courses 20   # 生成待标注模板
    python eval/concept_eval.py                                # 评测

The gold file is a flat mapping of course code to the concepts a human
accepts for that course::

    {"IM121": ["信息组织", "信息检索", "信息计量学"], ...}

Annotating from scratch is slow, so ``--make-template`` writes a file that
already contains every candidate both extractors proposed, with a
``_candidates`` block per course: the annotator deletes the noise and adds
what was missed, which is 10x faster than typing concepts from the syllabus.

Two matching modes are reported: ``strict`` (normalised exact match) and
``lenient`` (either string contains the other), because Chinese concept names
differ mostly by qualifier -- ``信息计量`` vs ``信息计量学``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

# Entry points outside src/ do not otherwise read .env, and scoring a
# different collection than the service serves would be worse than useless.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.config import config  # noqa: E402
from src.data_processing.concept_extractor import extract_course_concepts  # noqa: E402

GOLD_PATH = Path("eval/datasets/concept_gold.json")
_PUNCTUATION = re.compile(r"[\s·、,，。.:：;；()（）《》\"'\-_/]+")


def normalize(name: str) -> str:
    return _PUNCTUATION.sub("", str(name or "")).lower()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def syllabus_chunks() -> list[dict[str, Any]]:
    chunks = _load_json(config.chunks_output_path, [])
    return [
        chunk
        for chunk in chunks
        if (chunk.get("metadata") or {}).get("source_type") == "syllabus"
    ]


def rule_predictions(chunks: list[dict[str, Any]], cache_path: Path) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for row in extract_course_concepts(chunks, cache_path):
        predictions.setdefault(str(row["course_code"]), []).append(str(row["name"]))
    return predictions


def llm_predictions() -> dict[str, list[str]]:
    """Read the canonical registry produced by ConceptNormalizer.

    ``outputs/concepts.json`` is only a compatibility projection and may still
    hold output from an older pipeline, so falling back to it is announced --
    scoring stale rule output as if it were the LLM arm would invalidate the
    comparison this script exists to make.
    """
    registry = _load_json(config.concept_registry_path, [])
    if not registry:
        print(
            f"[concept-eval] WARNING: {config.concept_registry_path.name} is "
            "missing; falling back to outputs/concepts.json, which may predate "
            "the current extractor. Run `python run_pipeline.py concept` for a "
            "valid LLM arm."
        )
        registry = _load_json(config.concept_cache_path.with_name("concepts.json"), [])
    predictions: dict[str, list[str]] = {}
    for concept in registry:
        name = str(concept.get("canonical_name") or concept.get("name") or "")
        if not name:
            continue
        codes = concept.get("source_course_codes") or [concept.get("course_code")]
        for code in codes:
            if code:
                predictions.setdefault(str(code), []).append(name)
    return predictions


def _matches(predicted: str, gold_names: Iterable[str], *, lenient: bool) -> bool:
    predicted_key = normalize(predicted)
    for gold in gold_names:
        gold_key = normalize(gold)
        if predicted_key == gold_key:
            return True
        if lenient and predicted_key and gold_key and (
            predicted_key in gold_key or gold_key in predicted_key
        ):
            return True
    return False


def score_arm(
    gold: dict[str, list[str]],
    predictions: dict[str, list[str]],
    *,
    lenient: bool,
) -> dict[str, Any]:
    """Micro (pooled) and macro (per-course mean) P/R/F1."""
    true_positive = predicted_total = gold_total = 0
    per_course: list[dict[str, Any]] = []
    for code, gold_names in gold.items():
        predicted_names = predictions.get(code, [])
        hits = sum(
            1
            for name in predicted_names
            if _matches(name, gold_names, lenient=lenient)
        )
        covered = sum(
            1
            for name in gold_names
            if _matches(name, predicted_names, lenient=lenient)
        )
        true_positive += hits
        predicted_total += len(predicted_names)
        gold_total += len(gold_names)
        precision = hits / len(predicted_names) if predicted_names else 0.0
        recall = covered / len(gold_names) if gold_names else 0.0
        per_course.append(
            {
                "course_code": code,
                "predicted": len(predicted_names),
                "gold": len(gold_names),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )
    micro_precision = true_positive / predicted_total if predicted_total else 0.0
    micro_recall = (
        sum(
            1
            for code, names in gold.items()
            for name in names
            if _matches(name, predictions.get(code, []), lenient=lenient)
        )
        / gold_total
        if gold_total
        else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    macro_precision = (
        sum(row["precision"] for row in per_course) / len(per_course)
        if per_course
        else 0.0
    )
    macro_recall = (
        sum(row["recall"] for row in per_course) / len(per_course)
        if per_course
        else 0.0
    )
    return {
        "courses": len(gold),
        "predicted_concepts": predicted_total,
        "gold_concepts": gold_total,
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "per_course": per_course,
    }


def make_template(courses: int, output: Path) -> Path:
    """Emit a gold template pre-filled with both arms' candidates."""
    chunks = syllabus_chunks()
    rule = rule_predictions(chunks, config.concept_cache_path.with_name("rule_eval_cache.json"))
    llm = llm_predictions()
    codes = sorted(set(rule) | set(llm))[:courses]
    payload: dict[str, Any] = {
        "_instructions": (
            "为每门课保留真正的学科知识点，删除教学组织词；"
            "可自行补充候选中没有的概念。标注完成后删除 _candidates 字段。"
        ),
    }
    for code in codes:
        candidates = list(dict.fromkeys([*llm.get(code, []), *rule.get(code, [])]))
        payload[code] = []
        payload[f"{code}_candidates"] = candidates[:60]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def load_gold(path: Path) -> dict[str, list[str]]:
    payload = _load_json(path, {})
    if not payload:
        raise SystemExit(
            f"no concept gold set at {path}\n"
            "run `python eval/concept_eval.py --make-template` and annotate it"
        )
    gold = {
        str(code): [str(name) for name in names]
        for code, names in payload.items()
        if not code.startswith("_") and not code.endswith("_candidates")
        and isinstance(names, list) and names
    }
    if not gold:
        raise SystemExit(f"{path} contains no annotated courses yet")
    return gold


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 概念抽取评测（规则基线 vs LLM 抽取）",
        "",
        f"- 金标课程数：{report['gold_courses']}，金标概念数：{report['gold_concepts']}",
        f"- 生成时间：{report['created_at']}",
        "",
        "| 方法 | 匹配 | 概念数 | micro-P | micro-R | micro-F1 | macro-P | macro-R |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, modes in report["arms"].items():
        for mode, scores in modes.items():
            lines.append(
                f"| {arm} | {mode} | {scores['predicted_concepts']} | "
                f"{scores['micro_precision']} | {scores['micro_recall']} | "
                f"{scores['micro_f1']} | {scores['macro_precision']} | "
                f"{scores['macro_recall']} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(GOLD_PATH))
    parser.add_argument("--make-template", action="store_true")
    parser.add_argument("--courses", type=int, default=20)
    arguments = parser.parse_args()

    gold_path = Path(arguments.gold)
    if arguments.make_template:
        written = make_template(arguments.courses, gold_path.with_name("concept_gold.template.json"))
        print(f"[concept-eval] annotation template written to {written}")
        return

    gold = load_gold(gold_path)
    chunks = syllabus_chunks()
    arms = {
        "rule-baseline": rule_predictions(
            chunks,
            config.concept_cache_path.with_name("rule_eval_cache.json"),
        ),
        "llm-normalizer": llm_predictions(),
    }
    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gold_path": str(gold_path),
        "gold_courses": len(gold),
        "gold_concepts": sum(len(names) for names in gold.values()),
        "arms": {
            name: {
                "strict": score_arm(gold, predictions, lenient=False),
                "lenient": score_arm(gold, predictions, lenient=True),
            }
            for name, predictions in arms.items()
        },
    }
    directory = config.eval_report_dir
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{report['run_id']}-concept-eval"
    (directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (directory / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
