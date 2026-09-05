"""Metric primitives for the zsttSystem evaluation harness.

Everything here is pure: it takes gold items plus recorded predictions and
returns numbers, so the same functions grade a live run, a replayed log, and
an ablation arm.

Four metric families, matching the four things the system claims to do:

* **routing** -- accuracy, per-class precision/recall/F1, confusion matrix
* **retrieval** -- Recall@k and MRR against gold chunks (or their
  course-level proxy)
* **generation** -- answer-key coverage and citation correctness
* **refusal** -- correct refusal on unanswerable questions versus false
  refusal on answerable ones
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

# Phrases the service emits when it declines to answer.  Kept in one place so
# that a wording change in the generator surfaces here as a failing test
# rather than as a silently wrong refusal rate.
REFUSAL_MARKERS: tuple[str, ...] = (
    "未找到足够相关",
    "无法获取足够",
    "未找到符合条件",
    "未找到与该问题相关",
    "未找到明确的硬先修课程",
    "证据不足",
    "无法确认",
    "无法回答",
    # Degraded-backend messages: not an answer either, and counting them as
    # answers would inflate the hallucination rate whenever Neo4j is down.
    "未连接 Neo4j",
    "本地知识库暂无可用证据",
)


def is_refusal(answer: str) -> bool:
    """Whether an answer is the system declining rather than answering."""
    text = answer or ""
    return any(marker in text for marker in REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def confusion_matrix(
    pairs: Iterable[tuple[str, str]],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Rows are gold labels, columns are predictions."""
    matrix = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    for gold, predicted in pairs:
        if gold not in matrix:
            matrix[gold] = {predicted_label: 0 for predicted_label in labels}
        if predicted not in matrix[gold]:
            matrix[gold][predicted] = 0
        matrix[gold][predicted] += 1
    return matrix


def per_class_scores(
    matrix: Mapping[str, Mapping[str, int]],
    labels: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Precision / recall / F1 / support for each routing label."""
    scores: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = matrix.get(label, {}).get(label, 0)
        gold_total = sum(matrix.get(label, {}).values())
        predicted_total = sum(
            matrix.get(other, {}).get(label, 0) for other in matrix
        )
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / gold_total if gold_total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        scores[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": gold_total,
        }
    return scores


def routing_report(
    results: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Grade routing decisions.

    Each result carries ``expected_route``, ``predicted_route`` and -- for the
    multi-label classifier -- ``predicted_labels`` and ``confidence``.
    ``label_recall`` credits a prediction whose secondary labels contain the
    gold route, which is the metric that shows what multi-intent routing buys
    over the single-label chain.
    """
    if not results:
        return {"count": 0}
    pairs = [
        (str(item["expected_route"]), str(item["predicted_route"]))
        for item in results
    ]
    matrix = confusion_matrix(pairs, labels)
    correct = sum(1 for gold, predicted in pairs if gold == predicted)
    in_labels = sum(
        1
        for item in results
        if str(item["expected_route"])
        in [str(label) for label in item.get("predicted_labels", [])]
        or str(item["expected_route"]) == str(item["predicted_route"])
    )
    confidences = [
        float(item.get("confidence", 0.0))
        for item in results
        if item.get("confidence") is not None
    ]
    return {
        "count": len(results),
        "accuracy": round(correct / len(results), 4),
        "label_recall": round(in_labels / len(results), 4),
        "mean_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "confusion_matrix": matrix,
        "per_class": per_class_scores(matrix, labels),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def hit_positions(
    retrieved: Sequence[Mapping[str, Any]],
    gold_chunk_ids: Sequence[str],
    *,
    gold_course_codes: Sequence[str] = (),
    gold_section_types: Sequence[str] = (),
) -> list[int]:
    """1-based ranks of the retrieved hits that count as relevant.

    Chunk ids win when the item has them.  Otherwise the course-level proxy
    applies: a hit is relevant when it belongs to a gold course (and, if the
    item constrains them, to a gold section type).  The proxy is looser than
    chunk labels, so reports state which rule was used.
    """
    gold_ids = {str(value) for value in gold_chunk_ids}
    gold_courses = {str(value) for value in gold_course_codes}
    gold_sections = {str(value) for value in gold_section_types}
    positions: list[int] = []
    for rank, hit in enumerate(retrieved, start=1):
        metadata = hit.get("metadata") or {}
        chunk_id = str(hit.get("chunk_id", ""))
        if gold_ids:
            if chunk_id in gold_ids:
                positions.append(rank)
            continue
        course_code = str(
            hit.get("course_code") or metadata.get("course_code") or ""
        )
        section_type = str(
            hit.get("section_type") or metadata.get("section_type") or ""
        )
        if gold_courses and course_code not in gold_courses:
            continue
        if gold_sections and section_type not in gold_sections:
            continue
        if gold_courses or gold_sections:
            positions.append(rank)
    return positions


def retrieval_report(
    results: Sequence[Mapping[str, Any]],
    *,
    cutoffs: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Recall@k and MRR over items that carry retrieval ground truth."""
    graded = [item for item in results if item.get("gradable")]
    if not graded:
        return {"count": 0}
    report: dict[str, Any] = {"count": len(graded)}
    for cutoff in cutoffs:
        hits = sum(
            1
            for item in graded
            if any(rank <= cutoff for rank in item["positions"])
        )
        report[f"recall@{cutoff}"] = round(hits / len(graded), 4)
    reciprocal = [
        1.0 / min(item["positions"]) if item["positions"] else 0.0
        for item in graded
    ]
    report["mrr"] = round(sum(reciprocal) / len(graded), 4)
    report["chunk_level_items"] = sum(
        1 for item in graded if item.get("grading") == "chunk"
    )
    report["course_level_items"] = sum(
        1 for item in graded if item.get("grading") == "course"
    )
    return report


# ---------------------------------------------------------------------------
# Generation and citations
# ---------------------------------------------------------------------------

def answer_covers_keys(answer: str, answer_keys: Sequence[str]) -> bool:
    """Every key must appear; keys are ``|``-separated alternatives."""
    if not answer_keys:
        return False
    text = answer or ""
    return all(
        any(alternative and alternative in text for alternative in key.split("|"))
        for key in answer_keys
    )


def citation_precision(
    citations: Sequence[Mapping[str, Any]],
    *,
    gold_chunk_ids: Sequence[str] = (),
    gold_course_codes: Sequence[str] = (),
) -> float | None:
    """Share of citations pointing at evidence the gold item accepts.

    ``None`` when the item has no citation ground truth, so that unlabelled
    items neither help nor hurt the average.
    """
    if not citations:
        return None
    gold_ids = {str(value) for value in gold_chunk_ids}
    gold_courses = {str(value) for value in gold_course_codes}
    if not gold_ids and not gold_courses:
        return None
    correct = 0
    for citation in citations:
        chunk_id = str(citation.get("chunk_id", ""))
        course_code = str(citation.get("course_code", ""))
        if gold_ids and chunk_id in gold_ids:
            correct += 1
        elif not gold_ids and gold_courses and course_code in gold_courses:
            correct += 1
    return round(correct / len(citations), 4)


def generation_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Answer correctness, citation precision, and empty-citation rate."""
    answerable = [
        item
        for item in results
        if item.get("answerable") and item.get("answer_keys")
    ]
    citation_scores = [
        item["citation_precision"]
        for item in results
        if item.get("citation_precision") is not None
    ]
    with_answer = [item for item in results if item.get("answerable")]
    report: dict[str, Any] = {
        "scored_answers": len(answerable),
        "answer_key_coverage": (
            round(
                sum(1 for item in answerable if item["answer_correct"])
                / len(answerable),
                4,
            )
            if answerable
            else None
        ),
        "citation_precision": (
            round(sum(citation_scores) / len(citation_scores), 4)
            if citation_scores
            else None
        ),
        "scored_citations": len(citation_scores),
    }
    if with_answer:
        report["uncited_answer_rate"] = round(
            sum(
                1
                for item in with_answer
                if not item.get("refused") and not item.get("citation_count")
            )
            / len(with_answer),
            4,
        )
    return report


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------

def refusal_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The refusal trade-off: correct refusals against false refusals."""
    unanswerable = [item for item in results if not item.get("answerable")]
    answerable = [item for item in results if item.get("answerable")]
    report: dict[str, Any] = {
        "unanswerable_items": len(unanswerable),
        "answerable_items": len(answerable),
    }
    if unanswerable:
        report["correct_refusal_rate"] = round(
            sum(1 for item in unanswerable if item.get("refused"))
            / len(unanswerable),
            4,
        )
        report["hallucination_rate"] = round(
            1
            - sum(1 for item in unanswerable if item.get("refused"))
            / len(unanswerable),
            4,
        )
    if answerable:
        report["false_refusal_rate"] = round(
            sum(1 for item in answerable if item.get("refused")) / len(answerable),
            4,
        )
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_confusion_matrix(
    matrix: Mapping[str, Mapping[str, int]],
    labels: Sequence[str],
) -> str:
    """Markdown table, gold on rows and predictions on columns."""
    header = "| gold \\ pred | " + " | ".join(labels) + " | 合计 |"
    divider = "| --- " * (len(labels) + 2) + "|"
    lines = [header, divider]
    for gold in labels:
        row = matrix.get(gold, {})
        counts = [str(row.get(label, 0)) for label in labels]
        lines.append(
            f"| **{gold}** | " + " | ".join(counts) + f" | {sum(row.values())} |"
        )
    return "\n".join(lines)


def render_metric_table(report: Mapping[str, Any], title: str) -> str:
    """Flat ``key | value`` markdown table for the scalar metrics."""
    lines = [f"### {title}", "", "| 指标 | 值 |", "| --- | --- |"]
    for key, value in report.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)
