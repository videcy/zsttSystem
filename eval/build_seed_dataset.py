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
import sys
from pathlib import Path
from typing import Any

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
            name = str(course.get("course_name") or code)
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
        name = str(course["course_name"])
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
            name = str(course.get("course_name") or course["course_code"])
            prerequisites = [str(code) for code in course.get("prerequisites") or []]
            prerequisite_names = [
                str(by_code.get(code, {}).get("course_name") or code)
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
                    str(course.get("course_name"))
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
                question=template.format(name=str(course["course_name"])),
                expected_route="fact",
                answerable=False,
                gold_course_codes=[str(course["course_code"])],
                source="auto-seed",
                notes="课程存在但语料不记录该字段",
            )
        )
    return items


def build(output: Path) -> dict[str, Any]:
    courses = _load(config.courses_output_path)
    chunks = _load(config.chunks_output_path)
    documented = _syllabus_course_codes(chunks)
    rng = random.Random(SEED)

    items = [
        *_fact_items(courses, documented, rng),
        *_content_items(courses, documented, rng),
        *_dependency_items(courses, rng),
        *_catalog_items(courses, rng),
        *_unanswerable_items(courses, documented, rng),
    ]
    problems = [problem for item in items for problem in item.validate()]
    if problems:
        raise SystemExit("generated items failed validation:\n  " + "\n  ".join(problems))

    save_dataset(
        items,
        output,
        description=(
            "自动生成的种子评测集：事实/内容/依赖/目录/无答案五类。"
            "答案要点由培养方案字段推导，chunk 级标注与内容题答案要点需人工补充。"
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
