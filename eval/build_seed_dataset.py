"""Generate the seed gold set from the parsed course catalogue.

Hand-writing 170 questions is the slow part of building an evaluation set, but
most of them are mechanical: the training plan already states the credits,
hours, semester, instructor and prerequisites of every course, so the question
*and* its answer key can be derived from the same record the service answers
from.  This script emits those, plus deliberately unanswerable questions, and
leaves the judgement-heavy items (content questions, chunk-level labels) for a
human pass -- see ``eval/README.md``.

    python eval/build_seed_dataset.py

Everything is derived deterministically from a fixed seed, so regenerating
after a corpus update produces a reviewable diff instead of a new file.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.schema import GoldItem, dataset_summary, save_dataset  # noqa: E402
from src.config import config  # noqa: E402

SEED = 20260905

QUOTAS = {
    "fact": 40,
    "content": 40,
    "dependency": 40,
    "catalog": 25,
    "unanswerable": 25,
}

FACT_FIELDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("credits", "学分", ("《{name}》有几学分？", "{name}这门课是多少学分？")),
    ("hours", "总学时", ("《{name}》的总学时是多少？", "{name}一共多少学时？")),
    ("semester", "开课学期", ("《{name}》在第几学期开课？", "{name}什么学期上？")),
    (
        "instructor",
        "课程负责人",
        ("《{name}》的任课教师是谁？", "{name}这门课谁负责？"),
    ),
)

CONTENT_TEMPLATES = (
    "《{name}》主要讲什么内容？",
    "{name}这门课主要学习哪些内容？",
    "介绍一下《{name}》的课程目标。",
)

DEPENDENCY_TEMPLATES = (
    "学《{name}》之前需要先修哪些课程？",
    "《{name}》的先修课是什么？",
    "为什么要先学{prerequisite}再学{name}？",
)

CATALOG_TEMPLATES = (
    "{program}的核心课程有哪些？",
    "{program}要修哪些专业必修课？",
    "{program}的课程体系包括哪些课程？",
    "{program}的培养方案里有哪些课程？",
)

# Courses that do not exist anywhere in the corpus: the answer to any question
# about them is a refusal.
ABSENT_COURSES = (
    "火星种植学",
    "量子航天导论",
    "星际物流管理",
    "深海考古学导论",
    "神经芯片伦理",
    "古气象重建技术",
    "核聚变工程基础",
    "太空法与治理",
)

# Real courses, but asking for a field the syllabi and plans never record.
MISSING_FIELD_TEMPLATES = (
    "《{name}》的上课教室在哪里？",
    "《{name}》的期末考试具体日期是哪天？",
    "《{name}》去年的平均分是多少？",
)

CONTENT_SECTIONS = (
    "course_objectives",
    "teaching_content",
    "teaching_schedule",
    "overview",
)

#: 文档标题末尾可能带来源标记（如 ``_转自DOC`` / `` 转自 DOC``）。
_DOC_SOURCE_SUFFIX = re.compile(r"[\s_]*转自[\s_]*DOC\s*$", re.IGNORECASE)


def _load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(
            f"missing pipeline artifact: {path}\n"
            "run `python run_pipeline.py parse` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _syllabus_course_codes(chunks: list[dict[str, Any]]) -> set[str]:
    codes: set[str] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        if metadata.get("source_type") == "syllabus" and metadata.get("course_code"):
            codes.add(str(metadata["course_code"]))
    return codes


def _format_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _clean_course_name(value: Any) -> str:
    """Return a user-facing course name without a ``转自DOC`` title suffix."""
    name = str(value or "").strip()
    if not name:
        return ""
    return _DOC_SOURCE_SUFFIX.sub("", name).strip()


def _fact_items(
    courses: list[dict[str, Any]],
    documented: set[str],
    rng: random.Random,
) -> list[GoldItem]:
    """One question per (course, field), spread across as many courses as possible."""
    candidates: list[tuple[dict[str, Any], str, str, tuple[str, ...]]] = []
    for course in courses:
        if str(course.get("course_code")) not in documented:
            continue
        for field, label, templates in FACT_FIELDS:
            value = course.get(field)
            if value in (None, "", "/"):
                continue
            candidates.append((course, field, label, templates))
    rng.shuffle(candidates)

    items: list[GoldItem] = []
    used_courses: set[str] = set()
    # Two passes so that the quota covers distinct courses before doubling up.
    for pass_index in (0, 1):
        for course, field, label, templates in candidates:
            if len(items) >= QUOTAS["fact"]:
                break
            code = str(course["course_code"])
            if pass_index == 0 and code in used_courses:
                continue
            used_courses.add(code)
            name = _clean_course_name(course.get("course_name")) or code
            value = _format_value(course.get(field))
            items.append(
                GoldItem(
                    id=f"fact-{len(items) + 1:03d}",
                    question=templates[len(items) % len(templates)].format(name=name),
                    expected_route="fact",
                    answer_keys=[f"{label}为 {value}|{value}"],
                    gold_course_codes=[code],
                    source="auto-seed",
                    notes=f"field={field}",
                )
            )
    return items


def _content_items(
    courses: list[dict[str, Any]],
    documented: set[str],
    rng: random.Random,
) -> list[GoldItem]:
    """Content questions carry retrieval ground truth only.

    What counts as a correct *answer* for ``主要讲什么`` is a judgement call, so
    ``answer_keys`` is left empty for a human pass; the course and section
    constraints already make Recall@k and citation precision gradable.
    """
    pool = [
        course
        for course in courses
        if str(course.get("course_code")) in documented and course.get("course_name")
    ]
    rng.shuffle(pool)
    items: list[GoldItem] = []
    for course in pool[: QUOTAS["content"]]:
        name = _clean_course_name(course["course_name"])
        items.append(
            GoldItem(
                id=f"content-{len(items) + 1:03d}",
                question=CONTENT_TEMPLATES[len(items) % len(CONTENT_TEMPLATES)].format(
                    name=name
                ),
                expected_route="content",
                gold_course_codes=[str(course["course_code"])],
                gold_section_types=list(CONTENT_SECTIONS),
                source="auto-seed",
                notes="answer_keys 待人工标注",
            )
        )
    return items


def _dependency_items(
    courses: list[dict[str, Any]],
    rng: random.Random,
) -> list[GoldItem]:
    by_code = {
        str(course.get("course_code")): course
        for course in courses
        if course.get("course_code")
    }
    pool = [course for course in courses if course.get("prerequisites")]
    rng.shuffle(pool)
    items: list[GoldItem] = []
    template_index = 0
    while pool and len(items) < QUOTAS["dependency"]:
        exhausted = True
        for course in pool:
            if len(items) >= QUOTAS["dependency"]:
                break
            template = DEPENDENCY_TEMPLATES[template_index % len(DEPENDENCY_TEMPLATES)]
            name = _clean_course_name(course.get("course_name")) or str(
                course["course_code"]
            )
            prerequisites = [str(code) for code in course.get("prerequisites") or []]
            prerequisite_names = [
                _clean_course_name(by_code.get(code, {}).get("course_name")) or code
                for code in prerequisites
            ]
            if not prerequisite_names:
                continue
            question = template.format(
                name=name,
                prerequisite=prerequisite_names[0],
            )
            items.append(
                GoldItem(
                    id=f"dependency-{len(items) + 1:03d}",
                    question=question,
                    expected_route="dependency",
                    answer_keys=["|".join(prerequisite_names)],
                    gold_course_codes=[str(course["course_code"]), *prerequisites],
                    source="auto-seed",
                    notes=f"prerequisites={','.join(prerequisites)}",
                )
            )
            exhausted = False
        template_index += 1
        if exhausted or template_index > len(DEPENDENCY_TEMPLATES):
            break
    return items


def _catalog_items(courses: list[dict[str, Any]], rng: random.Random) -> list[GoldItem]:
    # Required courses make the strongest answer keys, but a plan that lists
    # none (the micro-minors) still deserves catalogue questions, so fall back
    # to whatever it does offer.
    required: dict[str, dict[str, dict[str, Any]]] = {}
    offered: dict[str, dict[str, dict[str, Any]]] = {}
    for course in courses:
        code = str(course.get("course_code") or "")
        if not code:
            continue
        for offering in course.get("offerings") or []:
            program = str(offering.get("program_name") or "")
            if not program:
                continue
            # A course appears once per offering; key by code so a plan that
            # lists it several times contributes one answer key, not three.
            offered.setdefault(program, {})[code] = course
            # "核心课程" means the major's own required courses, not the
            # university-wide ones (foreign language, politics) that every
            # plan repeats first.
            if "专业必修" in str(offering.get("course_category") or ""):
                required.setdefault(program, {})[code] = course
    by_program = {
        program: list((required.get(program) or members).values())
        for program, members in offered.items()
    }

    programs = sorted(by_program)
    rng.shuffle(programs)
    items: list[GoldItem] = []
    template_index = 0
    while programs and len(items) < QUOTAS["catalog"]:
        for program in programs:
            if len(items) >= QUOTAS["catalog"]:
                break
            members = by_program[program]
            keys = list(
                dict.fromkeys(
                    _clean_course_name(course.get("course_name"))
                    for course in members
                    if course.get("course_name")
                )
            )[:3]
            if not keys:
                continue
            template = CATALOG_TEMPLATES[template_index % len(CATALOG_TEMPLATES)]
            items.append(
                GoldItem(
                    id=f"catalog-{len(items) + 1:03d}",
                    question=template.format(program=_program_alias(program)),
                    expected_route="catalog",
                    answer_keys=["|".join(keys)],
                    gold_course_codes=[
                        str(course.get("course_code"))
                        for course in members[:10]
                        if course.get("course_code")
                    ],
                    source="auto-seed",
                    notes=f"program={program}",
                )
            )
        template_index += 1
        if template_index > len(CATALOG_TEMPLATES):
            break
    return items


def _program_alias(program_name: str) -> str:
    """Shorten a plan title into what a user would actually type."""
    for keyword in (
        "信息管理与信息系统",
        "图书情报与档案管理类",
        "图书馆学",
        "档案学",
    ):
        if keyword in program_name:
            suffix = "辅修微专业" if "微专业" in program_name else (
                "辅修专业" if "辅修" in program_name else "专业"
            )
            return f"{keyword}{suffix}"
    return program_name


def _unanswerable_items(
    courses: list[dict[str, Any]],
    documented: set[str],
    rng: random.Random,
) -> list[GoldItem]:
    items: list[GoldItem] = []
    fabricated_templates = (
        ("《{name}》主要讲什么内容？", "content"),
        ("《{name}》有几学分？", "fact"),
        ("学《{name}》之前要先修哪些课？", "dependency"),
    )
    for index, name in enumerate(ABSENT_COURSES * 2):
        if len(items) >= QUOTAS["unanswerable"] - 9:
            break
        template, route = fabricated_templates[index % len(fabricated_templates)]
        items.append(
            GoldItem(
                id=f"noanswer-{len(items) + 1:03d}",
                question=template.format(name=name),
                expected_route=route,
                answerable=False,
                source="auto-seed",
                notes="课程不存在于语料中",
            )
        )

    pool = [
        course
        for course in courses
        if str(course.get("course_code")) in documented and course.get("course_name")
    ]
    rng.shuffle(pool)
    for index, course in enumerate(pool):
        if len(items) >= QUOTAS["unanswerable"]:
            break
        template = MISSING_FIELD_TEMPLATES[index % len(MISSING_FIELD_TEMPLATES)]
        items.append(
            GoldItem(
                id=f"noanswer-{len(items) + 1:03d}",
                question=template.format(
                    name=_clean_course_name(course["course_name"])
                ),
                expected_route="fact",
                acceptable_routes=["fact", "hybrid"],
                answerable=False,
                gold_course_codes=[str(course["course_code"])],
                source="auto-seed",
                notes="课程存在但语料不记录该字段",
            )
        )
    return items


# ---------------------------------------------------------------------------
# M5.1 新题型：structure / 教师 / 考核 / 无答案章节
# 金标全部从 doc_tree.json / syllabus_profiles.json 自动生成；产物缺失时跳过。
# ---------------------------------------------------------------------------

#: 新增子集的目标条数；产物不足时取可用最大值并在日志中说明。
STRUCTURE_QUOTAS: dict[str, int] = {
    "structure-chapter": 30,
    "structure-outline": 20,
    "fact-instructor-syllabus": 20,
    "fact-instructor-missing": 10,
    "fact-assessment": 15,
    "noanswer-chapter": 10,
}

#: 占位符与空值都视为「未记录」。
_MISSING_VALUES = (None, "", "/", "／", "-", "—", "无", "N/A", "n/a")

#: 任课教师字段的候选键名（防御式读取，兼容并行实现的键名差异）。
_INSTRUCTOR_FIELDS = (
    "course_owner",
    "lecturers",
    "instructor",
    "teacher",
    "任课教师",
    "主讲教师",
    "课程负责人",
)


def _is_missing(value: Any) -> bool:
    """占位符（``/`` 等）与空值都算缺失。"""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in _MISSING_VALUES
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _load_optional(path: Path) -> Any | None:
    """读取可选产物；缺失或损坏时告警并返回 ``None``（优雅降级）。"""
    if not path.exists():
        print(f"[seed] 产物缺失：{path}；相关子集跳过", file=sys.stderr)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[seed] 产物无法解析：{path}（{exc}）；相关子集跳过", file=sys.stderr)
        return None


def _documents(doc_tree: Any) -> list[dict[str, Any]]:
    if not isinstance(doc_tree, dict):
        return []
    documents = doc_tree.get("documents")
    if not isinstance(documents, dict):
        return []
    return [doc for doc in documents.values() if isinstance(doc, dict)]


def _chapter_nodes(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """按章号返回去重后的 chapter 节点（稳定顺序）。"""
    nodes = document.get("nodes")
    if not isinstance(nodes, dict):
        return []
    chapters: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()
    for node in nodes.values():
        if not isinstance(node, dict) or str(node.get("kind")) != "chapter":
            continue
        key = (node.get("chapter_no"), str(node.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        chapters.append(node)

    def sort_key(node: Mapping[str, Any]) -> tuple[int, str]:
        try:
            number = int(node.get("chapter_no"))
        except (TypeError, ValueError):
            number = 1 << 30
        return (number, str(node.get("title") or ""))

    chapters.sort(key=sort_key)
    return chapters


def _node_chunk_ids(node: Mapping[str, Any]) -> list[str]:
    ids = node.get("chunk_ids")
    if not isinstance(ids, (list, tuple)):
        return []
    return [str(cid) for cid in ids if str(cid)]


def _document_name(
    document: Mapping[str, Any],
    course_names: Mapping[str, str] | None = None,
) -> str:
    """Resolve a display name, preferring the course catalogue over doc titles."""
    code = str(document.get("course_code") or "").strip()
    catalogue_name = _clean_course_name((course_names or {}).get(code))
    if catalogue_name:
        return catalogue_name
    return _clean_course_name(
        document.get("course_name") or document.get("title") or code
    )


def _course_name_map(courses: list[dict[str, Any]], doc_tree: Any) -> dict[str, str]:
    """Build code -> name from doc titles, then override with courses.json."""
    names: dict[str, str] = {}
    for document in _documents(doc_tree):
        code = str(document.get("course_code") or "").strip()
        name = _document_name(document)
        if code and name:
            names[code] = name
    for course in courses:
        code = str(course.get("course_code") or "").strip()
        name = _clean_course_name(course.get("course_name"))
        if code and name:
            names[code] = name
    return names


def _profile_records(profiles: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(profiles, dict):
        return []
    return [(str(key), value) for key, value in profiles.items() if isinstance(value, dict)]


def _profile_code(profile: Mapping[str, Any], key: str) -> str:
    """取课程代码；键名兼容 ``course_code::year`` 与 ``course_code|year``。"""
    code = str(profile.get("course_code") or "").strip()
    if code:
        return code
    for separator in ("::", "|"):
        if separator in key:
            return key.split(separator, 1)[0].strip()
    return key.strip()


def _profile_names(profile: Mapping[str, Any]) -> list[str]:
    """汇总大纲里的教师姓名（course_owner / lecturers / 别名键）。"""
    names: list[str] = []
    for field in _INSTRUCTOR_FIELDS:
        value = profile.get(field)
        if value is None or isinstance(value, Mapping):
            continue
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, (list, tuple, set)):
            names.extend(str(part) for part in value)
    cleaned: list[str] = []
    for name in names:
        name = name.strip()
        if _is_missing(name) or name in cleaned:
            continue
        cleaned.append(name)
    return cleaned


def _profile_chunk_ids(profile: Mapping[str, Any], tokens: tuple[str, ...]) -> list[str]:
    """从 ``field_chunk_ids`` 里挑出与给定字段名匹配的 chunk 引用。"""
    mapping = profile.get("field_chunk_ids")
    if not isinstance(mapping, dict):
        return []
    ids: list[str] = []
    for field, value in mapping.items():
        if not any(token in str(field) for token in tokens):
            continue
        if isinstance(value, str) and value:
            ids.append(value)
        elif isinstance(value, (list, tuple)):
            ids.extend(str(entry) for entry in value if entry)
    return list(dict.fromkeys(ids))


def _spread(candidates: list[Any], quota: int, group_key: Any) -> list[Any]:
    """先每门课各取一条，再回头补足，保证子集覆盖尽量多的课程。"""
    picked: list[Any] = []
    seen: set[Any] = set()
    for candidate in candidates:
        if len(picked) >= quota:
            return picked
        group = group_key(candidate)
        if group in seen:
            continue
        seen.add(group)
        picked.append(candidate)
    for candidate in candidates:
        if len(picked) >= quota:
            break
        if candidate in picked:
            continue
        picked.append(candidate)
    return picked


def _structure_chapter_items(
    doc_tree: Any,
    rng: random.Random,
    course_names: Mapping[str, str] | None = None,
) -> list[GoldItem]:
    """《X》第 N 章讲什么？——chunk 级金标 = 该章 chunk_ids。"""
    candidates: list[tuple[Any, ...]] = []
    for document in _documents(doc_tree):
        name = _document_name(document, course_names)
        code = str(document.get("course_code") or "").strip()
        if not name:
            continue
        for node in _chapter_nodes(document):
            title = str(node.get("title") or "").strip()
            chunk_ids = _node_chunk_ids(node)
            number = node.get("chapter_no")
            if not title or not chunk_ids or number is None:
                continue
            candidates.append((code, number, name, title, chunk_ids))
    rng.shuffle(candidates)
    selected = _spread(candidates, STRUCTURE_QUOTAS["structure-chapter"], lambda c: c[0])
    items: list[GoldItem] = []
    for index, (code, number, name, title, chunk_ids) in enumerate(selected, start=1):
        items.append(
            GoldItem(
                id=f"structure-chapter-{index:03d}",
                question=f"《{name}》第 {number} 章讲什么？",
                expected_route="structure",
                answer_keys=[title],
                gold_chunk_ids=chunk_ids,
                gold_course_codes=[code] if code else [],
                gold_chapter_titles=[title],
                subset="structure-chapter",
                source="auto-seed",
                notes=f"chapter_no={number}（answer_keys 暂用章标题，内容关键词待人工补充）",
            )
        )
    return items


def _structure_outline_items(
    doc_tree: Any,
    rng: random.Random,
    course_names: Mapping[str, str] | None = None,
) -> list[GoldItem]:
    """《X》一共有几章？分别是什么？——整表金标 = 全部章节 chunk_ids。"""
    candidates: list[tuple[str, str, list[str], list[str]]] = []
    for document in _documents(doc_tree):
        name = _document_name(document, course_names)
        code = str(document.get("course_code") or "").strip()
        chapters = _chapter_nodes(document)
        titles = [str(node.get("title") or "").strip() for node in chapters]
        titles = [title for title in titles if title]
        chunk_ids: list[str] = []
        for node in chapters:
            for chunk_id in _node_chunk_ids(node):
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
        if not name or len(titles) < 2 or not chunk_ids:
            continue
        candidates.append((code, name, titles, chunk_ids))
    rng.shuffle(candidates)
    selected = _spread(candidates, STRUCTURE_QUOTAS["structure-outline"], lambda c: c[0])
    items: list[GoldItem] = []
    for index, (code, name, titles, chunk_ids) in enumerate(selected, start=1):
        count = len(titles)
        answer_keys = [f"{count}章|{count} 章|共{count}章|共 {count} 章"]
        answer_keys.extend(titles)
        items.append(
            GoldItem(
                id=f"structure-outline-{index:03d}",
                question=f"《{name}》一共有几章？分别是什么？",
                expected_route="structure",
                answer_keys=answer_keys,
                gold_chunk_ids=chunk_ids,
                gold_course_codes=[code] if code else [],
                gold_chapter_titles=titles,
                subset="structure-outline",
                source="auto-seed",
                notes=f"chapter_count={count}",
            )
        )
    return items


def _fact_instructor_items(
    profiles: Any,
    courses: list[dict[str, Any]],
    rng: random.Random,
    course_names: Mapping[str, str] | None = None,
) -> list[GoldItem]:
    """《X》的任课教师是谁？——金标取大纲结构化字段，优先培养方案为 ``/`` 的课。"""
    plan_instructor = {
        str(course.get("course_code")): course.get("instructor")
        for course in courses
        if course.get("course_code")
    }
    candidates: list[tuple[int, str, str, list[str], list[str]]] = []
    for key, profile in _profile_records(profiles):
        names = _profile_names(profile)
        code = _profile_code(profile, key)
        name = _clean_course_name((course_names or {}).get(code))
        if not name:
            name = _clean_course_name(profile.get("course_name"))
        if not names or not name:
            continue
        priority = 0 if _is_missing(plan_instructor.get(code)) else 1
        chunk_ids = _profile_chunk_ids(
            profile, ("course_owner", "lectur", "instructor", "teacher", "教师", "负责")
        )
        candidates.append((priority, code, name, names, chunk_ids))
    rng.shuffle(candidates)
    candidates.sort(key=lambda candidate: candidate[0])
    selected = _spread(
        candidates, STRUCTURE_QUOTAS["fact-instructor-syllabus"], lambda c: c[1]
    )
    items: list[GoldItem] = []
    for index, (_, code, name, names, chunk_ids) in enumerate(selected, start=1):
        items.append(
            GoldItem(
                id=f"instr-syllabus-{index:03d}",
                question=f"《{name}》的任课教师是谁？",
                expected_route="fact",
                answer_keys=["|".join(names)],
                gold_chunk_ids=chunk_ids,
                gold_course_codes=[code] if code else [],
                expected_instructors=names,
                subset="fact-instructor-syllabus",
                source="auto-seed",
                notes="source=syllabus_profile",
            )
        )
    return items


def _fact_instructor_missing_items(
    courses: list[dict[str, Any]],
    profiles: Any,
) -> list[GoldItem]:
    """两源（培养方案 + 大纲）都无教师记录：期望明确回答「未记录」，按拒答计。

    只要培养方案未记录教师、且大纲侧也没有该课（或大纲没有教师字段），就算
    「两源都缺」——不要求这门口课一定有大纲，否则口径过窄会漏掉无大纲的课。
    """
    if profiles is None:
        print(
            "[seed] 缺 syllabus_profiles，fact-instructor-missing 子集跳过",
            file=sys.stderr,
        )
        return []
    with_syllabus = {
        _profile_code(profile, key)
        for key, profile in _profile_records(profiles)
        if _profile_names(profile)
    }
    items: list[GoldItem] = []
    for course in courses:
        if len(items) >= STRUCTURE_QUOTAS["fact-instructor-missing"]:
            break
        code = str(course.get("course_code") or "")
        name = _clean_course_name(course.get("course_name"))
        if not code or not name:
            continue
        if not _is_missing(course.get("instructor")):
            continue
        if code in with_syllabus:
            continue
        items.append(
            GoldItem(
                id=f"instr-missing-{len(items) + 1:03d}",
                question=f"《{name}》的任课教师是谁？",
                expected_route="fact",
                answerable=False,
                gold_course_codes=[code],
                subset="fact-instructor-missing",
                source="auto-seed",
                notes="培养方案与大纲均未记录任课教师",
            )
        )
    return items


def _fact_assessment_items(
    profiles: Any,
    rng: random.Random,
    course_names: Mapping[str, str] | None = None,
) -> list[GoldItem]:
    """《X》怎么考核？——金标取 syllabus_profile.assessment_mode。"""
    candidates: list[tuple[str, str, str, list[str]]] = []
    for key, profile in _profile_records(profiles):
        mode = profile.get("assessment_mode")
        if isinstance(mode, (list, tuple)):
            mode = "；".join(str(part) for part in mode if not _is_missing(part))
        if _is_missing(mode):
            continue
        code = _profile_code(profile, key)
        name = _clean_course_name((course_names or {}).get(code))
        if not name:
            name = _clean_course_name(profile.get("course_name"))
        if not name:
            continue
        chunk_ids = _profile_chunk_ids(profile, ("assessment", "grading", "考核", "成绩"))
        candidates.append((code, name, str(mode).strip(), chunk_ids))
    rng.shuffle(candidates)
    selected = _spread(candidates, STRUCTURE_QUOTAS["fact-assessment"], lambda c: c[0])
    items: list[GoldItem] = []
    for index, (code, name, mode, chunk_ids) in enumerate(selected, start=1):
        items.append(
            GoldItem(
                id=f"assessment-{index:03d}",
                question=f"《{name}》怎么考核？",
                expected_route="fact",
                answer_keys=[mode],
                gold_chunk_ids=chunk_ids,
                gold_course_codes=[code] if code else [],
                subset="fact-assessment",
                source="auto-seed",
                notes="field=assessment_mode",
            )
        )
    return items


def _noanswer_chapter_items(
    doc_tree: Any,
    rng: random.Random,
    course_names: Mapping[str, str] | None = None,
) -> list[GoldItem]:
    """《X》第 99 章讲什么？——章号不存在于文档树，期望拒答。"""
    candidates: list[tuple[str, str]] = []
    for document in _documents(doc_tree):
        name = _document_name(document, course_names)
        code = str(document.get("course_code") or "").strip()
        if not name or not _chapter_nodes(document):
            continue
        candidates.append((code, name))
    rng.shuffle(candidates)
    selected = _spread(candidates, STRUCTURE_QUOTAS["noanswer-chapter"], lambda c: c[0])
    items: list[GoldItem] = []
    for index, (code, name) in enumerate(selected, start=1):
        items.append(
            GoldItem(
                id=f"noanswer-chapter-{index:03d}",
                question=f"《{name}》第 99 章讲什么？",
                expected_route="structure",
                answerable=False,
                gold_course_codes=[code] if code else [],
                subset="noanswer-chapter",
                source="auto-seed",
                notes="章号 99 不存在于文档树",
            )
        )
    return items


def _format_subset_counts(items: list[GoldItem]) -> None:
    """日志：说明各子集实际产出条数（产物不足时取到多少就多少）。"""
    counts: dict[str, int] = {}
    for item in items:
        key = item.subset or "(既有子集)"
        counts[key] = counts.get(key, 0) + 1
    print("[seed] 子集条数：", file=sys.stderr)
    for key in sorted(counts):
        print(f"[seed]   {key}: {counts[key]}", file=sys.stderr)


def build(output: Path) -> dict[str, Any]:
    courses = _load(config.courses_output_path)
    chunks = _load(config.chunks_output_path)
    documented = _syllabus_course_codes(chunks)
    rng = random.Random(SEED)

    # 既有五类子集的生成逻辑、顺序与随机种子保持不变；新题型一律追加在后。
    items = [
        *_fact_items(courses, documented, rng),
        *_content_items(courses, documented, rng),
        *_dependency_items(courses, rng),
        *_catalog_items(courses, rng),
        *_unanswerable_items(courses, documented, rng),
    ]

    # M5.1 新题型：从并行交付的产物自动生成；产物缺失时跳过并告警（不抛异常）。
    doc_tree = _load_optional(config.doc_tree_path)
    profiles = _load_optional(config.syllabus_profiles_path)
    course_names = _course_name_map(courses, doc_tree)
    items.extend(_structure_chapter_items(doc_tree, rng, course_names))
    items.extend(_structure_outline_items(doc_tree, rng, course_names))
    items.extend(_fact_instructor_items(profiles, courses, rng, course_names))
    items.extend(_fact_instructor_missing_items(courses, profiles))
    items.extend(_fact_assessment_items(profiles, rng, course_names))
    items.extend(_noanswer_chapter_items(doc_tree, rng, course_names))

    problems = [problem for item in items for problem in item.validate()]
    if problems:
        raise SystemExit("generated items failed validation:\n  " + "\n  ".join(problems))

    _format_subset_counts(items)
    chunk_level = sum(1 for item in items if item.gold_chunk_ids)
    print(f"[seed] chunk 级金标题目：{chunk_level} 条", file=sys.stderr)

    save_dataset(
        items,
        output,
        description=(
            "自动生成的种子评测集：既有事实/内容/依赖/目录/无答案五类，"
            "新增 structure-chapter / structure-outline / fact-instructor-syllabus / "
            "fact-instructor-missing / fact-assessment / noanswer-chapter 六类。"
            "新题型由 doc_tree.json 与 syllabus_profiles.json 自动生成（产物缺失时跳过）。"
            "内容题答案要点与部分 chunk 级标注仍需人工补充。"
        ),
    )
    return dataset_summary(items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path("eval/datasets/gold_seed.json")),
        help="dataset path to write",
    )
    arguments = parser.parse_args()
    summary = build(Path(arguments.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written to {arguments.output}")


if __name__ == "__main__":
    main()
