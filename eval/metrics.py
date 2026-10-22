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
    "培养方案中没有名为",
    "你是不是想问",
    "均未记录",
    # Degraded-backend messages: not an answer either, and counting them as
    # answers would inflate the hallucination rate whenever Neo4j is down.
    "未连接 Neo4j",
    "本地知识库暂无可用证据",
    # NLI verification failure (generator.get_fallback_response): the model
    # produced text but nothing survived grounding, so the service declined.
    # Missing this marker counted every such refusal as an uncited answer.
    "无法给出可靠答案",
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
    correct = 0
    for item, (_, predicted) in zip(results, pairs):
        accepted = {str(item["expected_route"])}
        acceptable = item.get("acceptable_routes") or ()
        if isinstance(acceptable, str):
            acceptable = (acceptable,)
        accepted.update(str(route) for route in acceptable)
        if predicted in accepted:
            correct += 1
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
        chunk_id = str(citation.get("chunk_id") or "")
        if gold_ids and chunk_id:
            if chunk_id in gold_ids:
                correct += 1
            continue
        if not gold_courses:
            continue
        course_code = str(citation.get("course_code") or "")
        if course_code:
            citation_courses = {course_code}
        else:
            course_codes = citation.get("course_codes") or ()
            if isinstance(course_codes, str):
                course_codes = (course_codes,)
            citation_courses = {str(code) for code in course_codes if code}
        if citation_courses & gold_courses:
            correct += 1
    return round(correct / len(citations), 4)


def citation_precision_chunk(
    citations: Sequence[Mapping[str, Any]],
    *,
    gold_chunk_ids: Sequence[str] = (),
) -> float | None:
    """Share of chunk-bearing citations whose chunk id is in the gold set."""
    chunk_citations = [
        citation
        for citation in citations
        if str(citation.get("chunk_id") or "")
    ]
    gold_ids = {str(value) for value in gold_chunk_ids}
    if not chunk_citations or not gold_ids:
        return None
    correct = sum(
        1
        for citation in chunk_citations
        if str(citation.get("chunk_id")) in gold_ids
    )
    return round(correct / len(chunk_citations), 4)


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
    chunk_citation_scores = [
        item["citation_precision_chunk"]
        for item in results
        if item.get("citation_precision_chunk") is not None
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
        "citation_precision_chunk": (
            round(sum(chunk_citation_scores) / len(chunk_citation_scores), 4)
            if chunk_citation_scores
            else None
        ),
        "scored_chunk_citations": len(chunk_citation_scores),
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


# ---------------------------------------------------------------------------
# M5.2 结构类指标：章节覆盖率 / 顺序正确率
# ---------------------------------------------------------------------------

#: 章节标题模糊匹配时取标题前 N 个字（改动时需同步论文口径）。
CHAPTER_TITLE_MATCH_CHARS = 8


def _compact(text: Any) -> str:
    """去掉全部空白，让「第 3 章」与「第3章」能够互相匹配。"""
    return "".join(str(text or "").split())


def _title_key(title: str, match_chars: int) -> str:
    return _compact(title)[:match_chars]


def _title_first_position(answer: str, title: str, match_chars: int) -> int | None:
    key = _title_key(title, match_chars)
    if not key:
        return None
    position = answer.find(key)
    return position if position >= 0 else None


def chapter_coverage(
    answer: str,
    chapter_titles: Sequence[str],
    *,
    match_chars: int = CHAPTER_TITLE_MATCH_CHARS,
) -> float | None:
    """答案命中的章节标题数 / 金标章节总数（模糊匹配标题前 ``match_chars`` 字）。

    没有章节金标时返回 ``None``，让这条题不参与平均。
    """
    titles = [str(title) for title in chapter_titles if _compact(title)]
    if not titles:
        return None
    text = _compact(answer)
    hit = sum(
        1 for title in titles if _title_first_position(text, title, match_chars) is not None
    )
    return round(hit / len(titles), 4)


def chapter_order_correct(
    answer: str,
    chapter_titles: Sequence[str],
    *,
    match_chars: int = CHAPTER_TITLE_MATCH_CHARS,
) -> bool | None:
    """答案中章节出现顺序是否与文档树一致（Kendall τ = 1 记为正确）。

    少于两个可定位章节时无法判断，返回 ``None``；重复命中同一位置也判为错。
    """
    titles = [str(title) for title in chapter_titles if _compact(title)]
    if len(titles) < 2:
        return None
    text = _compact(answer)
    found: list[tuple[int, int]] = []
    for index, title in enumerate(titles):
        position = _title_first_position(text, title, match_chars)
        if position is not None:
            found.append((position, index))
    if len(found) < 2:
        return None
    found.sort()
    order = [index for _, index in found]
    return order == sorted(order) and len(order) == len(set(order))


def structure_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """structure 子集的章节覆盖率与顺序正确率（逐题字段由 run_eval 填充）。"""
    coverages = [
        float(item["chapter_coverage"])
        for item in results
        if item.get("chapter_coverage") is not None
    ]
    orders = [
        bool(item["order_correct"])
        for item in results
        if item.get("order_correct") is not None
    ]
    report: dict[str, Any] = {"coverage_items": len(coverages)}
    if coverages:
        report["chapter_coverage"] = round(sum(coverages) / len(coverages), 4)
    report["order_items"] = len(orders)
    if orders:
        report["order_correct_rate"] = round(
            sum(1 for value in orders if value) / len(orders), 4
        )
    return report


# ---------------------------------------------------------------------------
# M5.2 教师字段准确率
# ---------------------------------------------------------------------------

def instructor_set_correct(
    answer: str,
    expected_instructors: Sequence[str],
    *,
    known_names: Sequence[str] = (),
) -> bool | None:
    """教师姓名集合是否与金标完全一致。

    传入 ``known_names``（全数据集出现过的教师姓名池）时做真正的集合相等：
    答案里出现池中任何非金标姓名都算错。否则退化为「金标姓名全部命中」。
    """
    expected = {_compact(name) for name in expected_instructors if _compact(name)}
    if not expected:
        return None
    text = _compact(answer)
    pool = {
        _compact(name)
        for name in (known_names or expected_instructors)
        if _compact(name)
    }
    appeared = {name for name in pool if name in text}
    if known_names:
        return appeared == expected
    return expected <= appeared


def instructor_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """教师字段准确率；「未记录」题（两源都缺）单独统计拒答率。"""
    scored = [item for item in results if item.get("instructor_correct") is not None]
    missing = [
        item
        for item in results
        if "instructor-missing" in str(item.get("subset") or "")
    ]
    report: dict[str, Any] = {"instructor_items": len(scored)}
    if scored:
        report["instructor_accuracy"] = round(
            sum(1 for item in scored if item["instructor_correct"]) / len(scored), 4
        )
    report["instructor_missing_items"] = len(missing)
    if missing:
        report["instructor_missing_refusal_rate"] = round(
            sum(1 for item in missing if item.get("refused")) / len(missing), 4
        )
    return report


# ---------------------------------------------------------------------------
# M5.2 拒答按子集切片
# ---------------------------------------------------------------------------

def refusal_report_by_subset(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """把拒答指标按子集切片，避免 structure 高分掩盖 content 的变化。"""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in results:
        subset = str(item.get("subset") or "")
        if subset:
            groups.setdefault(subset, []).append(item)
    return {subset: refusal_report(items) for subset, items in sorted(groups.items())}


# ---------------------------------------------------------------------------
# P0-5 NLI 归因：概括句非蕴含 / 有矛盾 / 判官 Unknown
# ---------------------------------------------------------------------------

def nli_attribution(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """按主导原因归类 NLI 校验结果。

    优先级：有矛盾 > 判官 Unknown > 概括句被判非蕴含 > 全部蕴含。
    每条结果需带 ``nli_details``（由 run_eval ``--dump-nli`` 填充）。
    """
    buckets = {
        "contradiction": 0,
        "unknown": 0,
        "non_entailment": 0,
        "all_entailed": 0,
        "no_details": 0,
    }
    for item in results:
        details = item.get("nli_details") or []
        labels = [str(entry.get("label", "")) for entry in details if isinstance(entry, Mapping)]
        if not labels:
            buckets["no_details"] += 1
            continue
        if "Contradiction" in labels:
            buckets["contradiction"] += 1
        elif "Unknown" in labels:
            buckets["unknown"] += 1
        elif any(label != "Entailment" for label in labels):
            buckets["non_entailment"] += 1
        else:
            buckets["all_entailed"] += 1
    buckets["total_items"] = len(results)
    return buckets
