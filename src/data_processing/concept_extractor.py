"""Rule-based course concept extraction -- the ablation baseline.

The production concept layer is
:meth:`src.data_processing.concept_normalizer.ConceptNormalizer.extract_core_concepts`:
an LLM extractor with a constrained schema, majority voting and dependency
validation.  This module is kept as the *rule-only* baseline it always was, so
that ``eval/concept_eval.py`` can report precision/recall for both arms of the
same gold set instead of comparing the LLM against nothing.

Being a baseline is not an excuse for being a strawman, so the naive
"every 2-16 character run is a concept" rule is filtered two ways:

* an explicit stop list of teaching-organisation vocabulary, and
* a cross-course document-frequency cut -- a phrase occurring in more than
  ``max_document_ratio`` of the courses in the batch is boilerplate
  (``本课程``, ``教学内容``, ``考核方式``), not a knowledge point.

Surviving candidates are ranked by TF-IDF over courses, which is also what
makes the baseline reproducible: the same batch always yields the same list.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROMPT_VERSION = "course-concepts-rule-v2"

# Teaching-organisation vocabulary: words that describe how a course is run
# rather than what it teaches.
_NOISE: frozenset[str] = frozenset(
    {
        "教学进度表", "成绩评定", "参考书目", "参考书", "文档概览",
        "课程基本信息", "考核方式", "教材", "本课程", "课程", "学生",
        "教学内容", "教学目标", "课程目标", "课程简介", "教学方法",
        "教学要求", "教学安排", "学时", "学分", "章节", "作业",
        "考试", "期末", "期中", "平时成绩", "课堂", "讲授", "实验",
        "掌握", "了解", "熟悉", "理解", "能够", "要求", "培养",
        "重点", "难点", "内容", "方法", "主要", "基本", "相关",
        "教师", "专业", "学院", "大学", "教研室", "先修课程",
    }
)

# Fragments that make a candidate an instruction rather than a concept.
_NOISE_FRAGMENTS: tuple[str, ...] = (
    "掌握", "了解", "熟悉", "能够", "要求", "重点", "难点",
    "学时", "学分", "考核", "成绩", "作业", "本章", "本节",
)

# Document structure markers: chapter/section headings and their neighbours.
_STRUCTURAL = re.compile(
    r"^(第[一二三四五六七八九十百千零\d]+[章节讲课周部单元次]"
    r"|复习|答疑|绪论|导论|小结|总结|习题|案例分析|课堂讨论)"
)

_CANDIDATE = re.compile(r"[一-鿿A-Za-z][一-鿿A-Za-z0-9]{1,15}")

# Without segmentation a 16-character run of prose becomes one "concept", so
# candidates are cut at punctuation, whitespace, the coordinating particles
# that join list items, and the instructional verbs that introduce them.
_SPLITTER = re.compile(
    r"[\s、,，;；。!！?？:：/|()（）\[\]【】]+"
    r"|(?:以及|包括|涉及|掌握|了解|熟悉|理解|能够|要求|运用|应用于|等|与|和|及|的|并|或)"
)


def _course_key(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    return str(
        metadata.get("course_code")
        or chunk.get("course_code")
        or chunk.get("source_file", "unknown")
    )


def _is_noise(candidate: str) -> bool:
    if len(candidate) < 2 or candidate in _NOISE:
        return True
    if _STRUCTURAL.match(candidate):
        return True
    return any(fragment in candidate for fragment in _NOISE_FRAGMENTS)


def _candidate_counts(items: Iterable[dict[str, Any]]) -> Counter[str]:
    """Count candidate phrases occurring in one course's chunks."""
    counts: Counter[str] = Counter()
    for chunk in items:
        for segment in _SPLITTER.split(str(chunk.get("text", "") or "")):
            for candidate in _CANDIDATE.findall(segment):
                if _is_noise(candidate):
                    continue
                counts[candidate] += 1
    return counts


def _rank_concepts(
    counts: Counter[str],
    document_frequency: Counter[str],
    course_count: int,
    *,
    max_document_ratio: float,
    max_concepts: int,
) -> list[tuple[str, float]]:
    """TF-IDF rank a course's candidates after the boilerplate cut."""
    ranked: list[tuple[str, float]] = []
    for candidate, frequency in counts.items():
        document_count = document_frequency.get(candidate, 1)
        if course_count > 1 and document_count / course_count > max_document_ratio:
            continue
        inverse = math.log(course_count / document_count) + 1.0
        ranked.append((candidate, frequency * inverse))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:max_concepts]


def extract_course_concepts(
    chunks: list[dict],
    cache_path: str | Path,
    model: str = "rule-baseline",
    *,
    max_concepts: int = 40,
    max_document_ratio: float = 0.5,
) -> list[dict]:
    """Extract per-course concepts, reusing cached results by content hash.

    The cache is keyed on the concatenated ``document_hash`` of a course's
    chunks together with ``model`` and :data:`PROMPT_VERSION`, so an unchanged
    course is never re-extracted and a rule change invalidates every entry.
    """
    path = Path(cache_path)
    cache: dict[str, Any] = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[_course_key(chunk)].append(chunk)

    # Candidate statistics are computed over the whole batch so that the
    # document-frequency cut sees every course, not just the uncached ones.
    counts_by_course = {
        code: _candidate_counts(items) for code, items in grouped.items()
    }
    document_frequency: Counter[str] = Counter()
    for counts in counts_by_course.values():
        document_frequency.update(counts.keys())
    course_count = max(1, len(counts_by_course))

    concepts: list[dict] = []
    cache_dirty = False
    for code, items in grouped.items():
        document_hash = hashlib.sha256(
            "".join(str(chunk.get("document_hash", "")) for chunk in items).encode()
        ).hexdigest()
        cached = cache.get(code, {})
        if (
            cached.get("document_hash") == document_hash
            and cached.get("model") == model
            and cached.get("prompt_version") == PROMPT_VERSION
        ):
            concepts.extend(cached.get("concepts", []))
            continue

        ranked = _rank_concepts(
            counts_by_course[code],
            document_frequency,
            course_count,
            max_document_ratio=max_document_ratio,
            max_concepts=max_concepts,
        )
        rows = [
            {
                "concept_id": hashlib.sha256(f"{code}|{name}".encode()).hexdigest()[:20],
                "name": name,
                "course_code": code,
                "aliases": [],
                "score": round(score, 4),
                "extraction_source": "rule",
            }
            for name, score in ranked
        ]
        cache[code] = {
            "document_hash": document_hash,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "concepts": rows,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_dirty = True
        concepts.extend(rows)

    # Written once per call: the previous per-course write re-serialised the
    # whole cache N times for N courses.
    if cache_dirty:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return concepts
