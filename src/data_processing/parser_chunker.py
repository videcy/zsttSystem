"""Parse training plans and course syllabi into structure-aware chunks."""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import docx
import pandas as pd
from docx.document import Document as DocxDocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph


@dataclass(frozen=True)
class CourseOffering:
    """One course occurrence in one training plan."""

    program_name: str
    program_type: str
    course_category: str
    course_subcategory: str
    credits: int | float | None
    theory_credits: int | float | None
    practice_credits: int | float | None
    hours: int | float | str | None
    theory_hours: int | float | str | None
    practice_hours: int | float | str | None
    semester: int | float | str | None
    instructor: str
    is_honors: str
    graduation_requirements: str
    source_file: str
    sheet_name: str


@dataclass
class CourseCatalogEntry:
    """Canonical course plus all of its training-plan memberships."""

    course_code: str
    course_name: str
    english_name: str = ""
    prerequisites: list[str] = field(default_factory=list)
    offerings: list[CourseOffering] = field(default_factory=list)

    def add_offering(self, offering: CourseOffering) -> None:
        if offering not in self.offerings:
            self.offerings.append(offering)

    def to_dict(self) -> dict[str, Any]:
        primary = self.offerings[0] if self.offerings else None
        return {
            "course_code": self.course_code,
            "course_name": self.course_name,
            "english_name": self.english_name,
            "credits": primary.credits if primary else None,
            "hours": primary.hours if primary else None,
            "semester": primary.semester if primary else None,
            "instructor": primary.instructor if primary else "",
            "course_category": primary.course_category if primary else "",
            "course_subcategory": primary.course_subcategory if primary else "",
            "prerequisites": self.prerequisites,
            "program_names": list(
                dict.fromkeys(offering.program_name for offering in self.offerings)
            ),
            "offerings": [asdict(offering) for offering in self.offerings],
        }


class SyllabusChunker:
    """Parse XLSX training plans and DOCX syllabi into teaching-oriented chunks."""

    def __init__(self) -> None:
        self.section_keywords = (
            "课程目标",
            "教学内容",
            "课程基本内容",
            "教学要求",
            "教学安排",
            "教学进度",
            "课程简介",
            "课程性质",
            "课程任务",
            "实践环节",
            "实验内容",
            "课程考核",
            "考核方式",
            "成绩评定",
            "教材",
            "参考书",
            "先修课程",
            "毕业要求",
        )
        self.course_code_pattern = re.compile(r"\b[A-Za-z]{2,}\d{2,}|\b\d{5,}\b")
        self.week_title_pattern = re.compile(
            r"^\s*(第[一二三四五六七八九十百0-9]+周.*|第[一二三四五六七八九十百0-9]+讲.*)$"
        )
        self.max_chunk_chars = 700
        self.chunk_overlap_chars = 80
        self.template_noise_patterns = (
            re.compile(r"^注[：:].*(A4|教学大纲|打印|纸张)", re.I),
            re.compile(r"^[（(].*(要求有一定的字数|对各种教学环节的安排|包括课堂讲授|推荐若干参考书).*[）)]$"),
        )

    def _normalize_text(self, value: Any) -> str:
        """Normalize cell or paragraph text for matching and output."""
        if value is None:
            return ""
        text = str(value).replace("\r", "\n")
        text = re.sub(r"[ \t\u3000]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_key(self, value: str) -> str:
        """Normalize header names for flexible column matching."""
        return re.sub(r"[\s_（）()\-:/：]+", "", value).lower()

    def _find_column(self, columns: list[str], candidates: tuple[str, ...]) -> str | None:
        """Resolve the best matching column for a logical field."""
        normalized_map = {column: self._normalize_key(column) for column in columns}
        normalized_candidates = [self._normalize_key(candidate) for candidate in candidates]

        for column, normalized in normalized_map.items():
            if normalized in normalized_candidates:
                return column

        for column, normalized in normalized_map.items():
            if any(candidate in normalized for candidate in normalized_candidates):
                return column

        return None

    def _find_column_index(
        self,
        columns: list[str],
        candidates: tuple[str, ...],
    ) -> int | None:
        """Resolve a logical field to its column position."""
        matched = self._find_column(columns, candidates)
        return columns.index(matched) if matched is not None else None

    def _looks_like_course_table_header(self, row_values: list[str]) -> bool:
        """Detect the first header row of a course table inside a worksheet."""
        normalized = [self._normalize_key(value) for value in row_values if self._normalize_text(value)]
        if not normalized:
            return False

        has_code = any(
            token in item
            for item in normalized
            for token in ("课程编码", "课程代码", "课程编号")
        )
        has_name = any(
            token in item
            for item in normalized
            for token in ("课程中文名称", "课程名称", "课程")
        )
        return has_code and has_name

    def _build_header_row(
        self,
        dataframe: pd.DataFrame,
        header_row_index: int,
    ) -> tuple[list[str], int]:
        """Flatten one-row or two-row worksheet headers into a single header list."""
        primary = [
            self._normalize_text(value)
            for value in dataframe.iloc[header_row_index].tolist()
        ]

        data_start_index = header_row_index + 1
        if header_row_index + 1 >= len(dataframe):
            return primary, data_start_index

        secondary = [
            self._normalize_text(value)
            for value in dataframe.iloc[header_row_index + 1].tolist()
        ]
        secondary_normalized = [self._normalize_key(value) for value in secondary if value]

        if any(
            token in item
            for item in secondary_normalized
            for token in ("总学分", "理论学分", "实验实践学分")
        ):
            merged = []
            for first, second in zip(primary, secondary):
                if first and second:
                    merged.append(f"{first}-{second}")
                else:
                    merged.append(first or second)
            return merged, header_row_index + 2

        return primary, data_start_index

    def _extract_program_name(self, dataframe: pd.DataFrame, xlsx_path: Path) -> str:
        """Infer a program name from the first few rows of a worksheet."""
        for _, row in dataframe.head(12).iterrows():
            for value in row.tolist():
                text = self._normalize_text(value)
                if text and "培养方案" in text:
                    return text
        return xlsx_path.stem

    def _extract_sheet_course_catalog(
        self,
        dataframe: pd.DataFrame,
        xlsx_path: Path,
        sheet_name: str = "",
    ) -> dict[str, dict[str, Any]]:
        """Extract course metadata from a single worksheet."""
        dataframe = dataframe.fillna("")

        header_row_index: int | None = None
        for idx in range(len(dataframe)):
            row_values = [self._normalize_text(value) for value in dataframe.iloc[idx].tolist()]
            if self._looks_like_course_table_header(row_values):
                header_row_index = idx
                break

        if header_row_index is None:
            return {}

        columns, data_start_index = self._build_header_row(dataframe, header_row_index)
        if not any(self._normalize_text(column) for column in columns):
            return {}

        code_index = self._find_column_index(
            columns,
            ("课程代码", "课程编码", "课程编号"),
        )
        name_index = self._find_column_index(
            columns,
            ("课程名称", "课程中文名称", "课程中文名称/英文名称"),
        )
        credits_index = self._find_column_index(
            columns,
            ("总学分", "学分情况-总学分", "学分", "课程学分"),
        )
        theory_credits_index = self._find_column_index(
            columns,
            ("理论学分", "学分情况-理论学分"),
        )
        practice_credits_index = self._find_column_index(
            columns,
            ("实验实践学分", "学分情况-实验实践学分"),
        )
        hours_index = self._find_column_index(
            columns,
            ("总学时", "学时情况-总学时", "学时", "课程学时"),
        )
        theory_hours_index = self._find_column_index(
            columns,
            ("理论学时", "学时情况-理论学时"),
        )
        practice_hours_index = self._find_column_index(
            columns,
            ("实验实践学时", "学时情况-实验实践学时"),
        )
        semester_index = self._find_column_index(columns, ("开课学期", "学期"))
        instructor_index = self._find_column_index(
            columns,
            ("课程负责人", "任课教师", "教师"),
        )
        honors_index = self._find_column_index(columns, ("是否荣誉课程",))
        graduation_index = self._find_column_index(
            columns,
            ("对应毕业要求", "毕业要求"),
        )
        prereq_index = self._find_column_index(
            columns,
            ("先修课程代码", "先修课程", "前置课程", "先修"),
        )
        category_index = self._find_column_index(
            columns,
            ("课程细类", "课程类别", "课程类型"),
        )
        subcategory_index = None
        if (
            category_index is not None
            and code_index is not None
            and category_index + 1 < code_index
            and not self._normalize_text(columns[category_index + 1])
        ):
            subcategory_index = category_index + 1

        if code_index is None or name_index is None:
            return {}

        program_name = self._extract_program_name(dataframe, xlsx_path)
        program_type = self._extract_program_type(program_name, xlsx_path)
        entries: dict[str, CourseCatalogEntry] = {}
        current_category = ""
        current_subcategory = ""

        for row_index in range(data_start_index, len(dataframe)):
            values = [
                self._normalize_text(value)
                for value in dataframe.iloc[row_index].tolist()
            ]
            values += [""] * (len(columns) - len(values))

            category = self._value_at(values, category_index)
            if category:
                current_category = category
                current_subcategory = ""
            subcategory = self._value_at(values, subcategory_index)
            if subcategory:
                current_subcategory = subcategory

            raw_code = self._value_at(values, code_index)
            raw_name = self._value_at(values, name_index)
            course_codes = self._extract_course_codes_from_text(raw_code)
            if not course_codes or not raw_name:
                continue

            course_name = raw_name.splitlines()[0].strip()
            english_name = (
                raw_name.splitlines()[1].strip()
                if len(raw_name.splitlines()) > 1
                else ""
            )
            if not course_name:
                continue

            offering = CourseOffering(
                program_name=program_name,
                program_type=program_type,
                course_category=current_category,
                course_subcategory=current_subcategory,
                credits=self._number_or_text(self._value_at(values, credits_index)),
                theory_credits=self._number_or_text(
                    self._value_at(values, theory_credits_index)
                ),
                practice_credits=self._number_or_text(
                    self._value_at(values, practice_credits_index)
                ),
                hours=self._number_or_text(self._value_at(values, hours_index)),
                theory_hours=self._number_or_text(
                    self._value_at(values, theory_hours_index)
                ),
                practice_hours=self._number_or_text(
                    self._value_at(values, practice_hours_index)
                ),
                semester=self._number_or_text(
                    self._value_at(values, semester_index)
                ),
                instructor=self._value_at(values, instructor_index),
                is_honors=self._value_at(values, honors_index),
                graduation_requirements=self._value_at(
                    values,
                    graduation_index,
                ),
                source_file=xlsx_path.name,
                sheet_name=sheet_name,
            )
            prerequisites = self._split_prerequisites(
                self._value_at(values, prereq_index)
            )
            for course_code in course_codes:
                entry = entries.setdefault(
                    course_code,
                    CourseCatalogEntry(
                        course_code=course_code,
                        course_name=course_name,
                        english_name=english_name,
                        prerequisites=prerequisites,
                    ),
                )
                entry.add_offering(offering)

        return {code: entry.to_dict() for code, entry in entries.items()}

    @staticmethod
    def _value_at(values: list[str], index: int | None) -> str:
        if index is None or index >= len(values):
            return ""
        return values[index]

    def _number_or_text(self, value: Any) -> int | float | str | None:
        text = self._normalize_text(value)
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return text
        return int(number) if number.is_integer() else number

    def _extract_program_type(self, program_name: str, xlsx_path: Path) -> str:
        text = f"{program_name} {xlsx_path.stem}"
        for value in ("辅修微专业", "辅修专业", "主修专业类", "主修专业"):
            if value in text:
                return value
        return "培养方案"

    @staticmethod
    def _merge_catalog(
        target: dict[str, dict[str, Any]],
        incoming: dict[str, dict[str, Any]],
    ) -> None:
        for code, course in incoming.items():
            if code not in target:
                target[code] = course
                continue
            existing = target[code]
            known_offerings = {
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                for item in existing.get("offerings", [])
            }
            for offering in course.get("offerings", []):
                key = json.dumps(offering, ensure_ascii=False, sort_keys=True)
                if key not in known_offerings:
                    existing.setdefault("offerings", []).append(offering)
                    known_offerings.add(key)
            existing["program_names"] = list(
                dict.fromkeys(
                    [
                        *existing.get("program_names", []),
                        *course.get("program_names", []),
                    ]
                )
            )

    def _split_prerequisites(self, value: Any) -> list[str]:
        """Split prerequisite course codes or names into a list."""
        text = self._normalize_text(value)
        if not text:
            return []
        parts = re.split(r"[、,，;/；\n]+", text)
        empty_values = {
            "无",
            "无要求",
            "无先修课程",
            "无先修课",
            "暂无",
            "不要求",
            "没有",
        }
        return [
            part.strip()
            for part in parts
            if part.strip() and part.strip("。 ") not in empty_values
        ]

    @staticmethod
    def _normalize_course_alias(value: Any) -> str:
        text = str(value or "").strip().casefold()
        text = text.strip("《》〈〉“”\"'()（）[]【】。 ")
        return re.sub(r"[\s\u3000·・_\-—:：/]+", "", text)

    def _course_alias_index(
        self,
        course_catalog: dict[str, dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build an exact alias index; fuzzy matches never create hard edges."""
        aliases: dict[str, list[dict[str, Any]]] = {}
        for code, course in course_catalog.items():
            values = [
                code,
                course.get("course_code"),
                course.get("course_name"),
                course.get("english_name"),
                *(course.get("aliases") or []),
            ]
            for value in values:
                normalized = self._normalize_course_alias(value)
                if not normalized:
                    continue
                bucket = aliases.setdefault(normalized, [])
                if all(item.get("course_code") != code for item in bucket):
                    bucket.append(course)
        return aliases

    def resolve_prerequisites(
        self,
        raw_names: list[str],
        course_catalog: dict[str, dict[str, Any]],
        *,
        dependent_course_code: str,
        source_file: str,
        section: str,
        source_year: str,
        source_type: str = "syllabus",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Resolve explicit names by exact aliases and retain unresolved evidence."""
        alias_index = self._course_alias_index(course_catalog)
        resolved: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        seen_resolved: set[str] = set()
        seen_unresolved: set[tuple[str, str]] = set()

        for raw_name in raw_names:
            raw_name = self._normalize_text(raw_name).strip("。 ")
            if not raw_name:
                continue
            matches = alias_index.get(self._normalize_course_alias(raw_name), [])
            matches = [
                match
                for match in matches
                if str(match.get("course_code", "")).casefold()
                != dependent_course_code.casefold()
            ]
            if len(matches) == 1:
                match = matches[0]
                code = str(match.get("course_code", ""))
                if code in seen_resolved:
                    continue
                resolved.append(
                    {
                        "raw_name": raw_name,
                        "course_code": code,
                        "course_name": self._normalize_text(
                            match.get("course_name", "")
                        ),
                        "relation": "PREREQUISITE_OF",
                        "confidence": 1.0,
                        "source_type": source_type,
                        "source_file": source_file,
                        "section": section,
                        "source_year": source_year,
                        "verified": True,
                    }
                )
                seen_resolved.add(code)
                continue

            reason = "ambiguous" if len(matches) > 1 else "not_found"
            key = (raw_name, reason)
            if key in seen_unresolved:
                continue
            unresolved.append(
                {
                    "raw_name": raw_name,
                    "dependent_course_code": dependent_course_code,
                    "source_file": source_file,
                    "section": section,
                    "source_year": source_year,
                    "reason": reason,
                }
            )
            seen_unresolved.add(key)
        return resolved, unresolved

    def _explicit_prerequisite_names(
        self,
        parsed_chunks: list[dict[str, str]],
    ) -> list[str]:
        values: list[str] = []
        pattern = re.compile(
            r"(?:先修课程|前置课程|预修课程)\s*[：:]\s*([^\n。；;]+)"
        )
        for chunk in parsed_chunks:
            if chunk.get("section_type") != "basic_info":
                continue
            for match in pattern.finditer(str(chunk.get("content", ""))):
                values.extend(self._split_prerequisites(match.group(1)))
        return list(dict.fromkeys(values))

    def _extract_course_code_from_text(self, text: str) -> str:
        """Extract a likely course code from a syllabus title or filename."""
        match = self.course_code_pattern.search(text)
        return match.group(0) if match else ""

    def _extract_course_codes_from_text(self, text: str) -> list[str]:
        """Extract and deduplicate valid course codes from a cell."""
        return list(dict.fromkeys(self.course_code_pattern.findall(text)))

    def _extract_course_name_from_path(self, path: Path) -> str:
        """Guess course name from the DOCX filename."""
        stem = path.stem
        stem = re.sub(r"[_\-]+", " ", stem)
        stem = re.sub(self.course_code_pattern, "", stem)
        return self._normalize_text(stem)

    def _is_heading(self, paragraph: Paragraph) -> bool:
        """Detect whether a paragraph should start a new logical section."""
        text = self._normalize_text(paragraph.text)
        if not text:
            return False

        style_name = paragraph.style.name if paragraph.style is not None else ""
        if style_name and "heading" in style_name.lower():
            return True

        if self.week_title_pattern.match(text):
            return True

        heading_text = re.sub(r"[（(].*?[）)]", "", text).strip(" ：:")
        if any(keyword in heading_text for keyword in self.section_keywords):
            return len(heading_text) <= 24

        if re.match(r"^\s*(第[一二三四五六七八九十百0-9]+[章节部分单元].*|\d+(\.\d+){0,2}\s*.+)$", text):
            return True

        return False

    def _paragraph_is_effectively_empty(self, paragraph: Paragraph) -> bool:
        """Skip blank paragraphs."""
        return not self._normalize_text(paragraph.text)

    def _is_template_noise(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        return any(pattern.search(normalized) for pattern in self.template_noise_patterns)

    def _unique_row_values(self, row: Any) -> list[str]:
        """Read a row without repeating values from horizontally merged cells."""
        values: list[str] = []
        seen_cells: set[int] = set()
        for cell in row.cells:
            cell_id = id(cell._tc)
            if cell_id in seen_cells:
                continue
            seen_cells.add(cell_id)
            values.append(self._normalize_text(cell.text))
        return values

    def _table_to_markdown(self, table: Table) -> str:
        """Convert a DOCX table into Markdown text."""
        rows: list[list[str]] = []
        for row in table.rows:
            row_values = self._unique_row_values(row)
            if any(row_values):
                rows.append(row_values)

        if not rows:
            return ""

        column_count = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]

        header = normalized_rows[0]
        separator = ["---"] * column_count
        markdown_lines = [
            f"| {' | '.join(header)} |",
            f"| {' | '.join(separator)} |",
        ]
        for row in normalized_rows[1:]:
            markdown_lines.append(f"| {' | '.join(row)} |")

        return "\n".join(markdown_lines)

    @staticmethod
    def _section_type(title: str) -> str:
        normalized = re.sub(r"\s+", "", title)
        if "课程目标" in normalized:
            return "course_objectives"
        if re.search(r"教学进度|章节|第.+章", normalized):
            return "teaching_schedule"
        if "教学内容" in normalized or "课程基本内容" in normalized:
            return "teaching_content"
        if re.search(r"考核|成绩评定", normalized):
            return "assessment"
        if "参考书" in normalized:
            return "references"
        if "教材" in normalized:
            return "textbook"
        if "先修" in normalized:
            return "prerequisites"
        if re.search(r"基本信息|课程简介|课程性质|课程任务", normalized):
            return "basic_info"
        return "overview"

    def _split_content(self, content: str) -> list[str]:
        """Bound chunk size while retaining a small, deterministic overlap."""
        content = self._normalize_text(content)
        if len(content) <= self.max_chunk_chars:
            return [content] if content else []

        units = [
            item.strip()
            for item in re.split(r"(?<=[。！？；])|\n+", content)
            if item.strip()
        ]
        parts: list[str] = []
        current = ""
        for unit in units:
            remaining = unit
            while len(remaining) > self.max_chunk_chars:
                if current:
                    parts.append(current)
                    current = current[-self.chunk_overlap_chars :]
                take = self.max_chunk_chars - len(current)
                current += remaining[:take]
                remaining = remaining[take:]
            candidate = f"{current}\n{remaining}".strip() if current else remaining
            if len(candidate) <= self.max_chunk_chars:
                current = candidate
                continue
            parts.append(current)
            overlap = current[-self.chunk_overlap_chars :]
            current = f"{overlap}\n{remaining}".strip()
        if current:
            parts.append(current)
        return parts

    def _table_sections(self, table: Table) -> list[dict[str, str]]:
        """Turn common syllabus tables into semantic, row-level sections."""
        rows = [self._unique_row_values(row) for row in table.rows]
        rows = [row for row in rows if any(row)]
        if not rows:
            return []

        combined = " ".join(" ".join(row) for row in rows)
        if "章节次序" in combined and "主要教学内容" in combined:
            header = rows[0]
            sections: list[dict[str, str]] = []
            for row in rows[1:]:
                if not any(row):
                    continue
                values = row + [""] * max(0, len(header) - len(row))
                chapter = (values[0] or "教学进度").splitlines()[0]
                facts = [
                    f"{label}：{value}"
                    for label, value in zip(header, values)
                    if label and value
                ]
                sections.append(
                    {
                        "section_title": f"教学进度 - {chapter}",
                        "content": "\n".join(facts),
                        "section_type": "teaching_schedule",
                    }
                )
            return sections

        if (
            ("课程代码" in combined or "课程编码" in combined)
            and ("学分" in combined or "课程类别" in combined)
        ):
            basic_facts: list[str] = []
            objective_parts: list[str] = []
            for row in rows:
                for index in range(0, len(row) - 1, 2):
                    key, value = row[index].strip(" ：:"), row[index + 1]
                    if not key or not value or key == value:
                        continue
                    if "课程目标" in key:
                        objective_parts.append(value)
                    else:
                        basic_facts.append(f"{key}：{value}")
                if row and "课程目标" in row[0] and len(row) > 1:
                    objective_parts.extend(row[1:])
            sections = []
            if basic_facts:
                sections.append(
                    {
                        "section_title": "课程基本信息",
                        "content": "\n".join(dict.fromkeys(basic_facts)),
                        "section_type": "basic_info",
                    }
                )
            objectives = "\n".join(dict.fromkeys(filter(None, objective_parts)))
            if objectives:
                sections.append(
                    {
                        "section_title": "课程目标",
                        "content": objectives,
                        "section_type": "course_objectives",
                    }
                )
            return sections

        markdown = self._table_to_markdown(table)
        return (
            [
                {
                    "section_title": "教学内容表",
                    "content": markdown,
                    "section_type": "teaching_content",
                }
            ]
            if markdown
            else []
        )

    def _iter_block_items(self, document: DocxDocumentType) -> Iterator[Paragraph | Table]:
        """Yield paragraphs and tables in their original document order."""
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                yield Paragraph(child, document)
            elif child.tag.endswith("}tbl"):
                yield Table(child, document)

    def _match_course_metadata(
        self,
        syllabus_path: Path,
        parsed_chunks: list[dict[str, str]],
        course_catalog: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Match a syllabus file against the loaded training plan metadata."""
        filename_code = self._extract_course_code_from_text(syllabus_path.stem)
        if filename_code and filename_code in course_catalog:
            return dict(course_catalog[filename_code])

        filename_name = self._extract_course_name_from_path(syllabus_path)
        if filename_code:
            return {
                "course_code": filename_code,
                "course_name": filename_name,
                "credits": None,
                "prerequisites": [],
            }

        for metadata in course_catalog.values():
            course_name = self._normalize_text(metadata.get("course_name", ""))
            if filename_name and course_name and filename_name == course_name:
                return dict(metadata)

        combined_text = "\n".join(
            [chunk.get("section_title", "") + "\n" + chunk.get("content", "") for chunk in parsed_chunks]
        )
        content_code = self._extract_course_code_from_text(combined_text)
        if content_code and content_code in course_catalog:
            return dict(course_catalog[content_code])

        for metadata in course_catalog.values():
            course_name = self._normalize_text(metadata.get("course_name", ""))
            if course_name and course_name in combined_text:
                return dict(metadata)

        return {
            "program_name": "",
            "course_code": filename_code,
            "course_name": filename_name,
            "credits": None,
            "prerequisites": [],
        }

    def parse_training_plan(self, xlsx_path: str | Path) -> dict[str, dict[str, Any]]:
        """Load structured course metadata from an Excel training plan."""
        xlsx_path = Path(xlsx_path)
        workbook = pd.ExcelFile(xlsx_path)
        course_catalog: dict[str, dict[str, Any]] = {}

        for sheet_name in workbook.sheet_names:
            dataframe = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
            self._merge_catalog(
                course_catalog,
                self._extract_sheet_course_catalog(
                    dataframe,
                    xlsx_path,
                    sheet_name,
                ),
            )

        if course_catalog:
            return course_catalog

        raise ValueError(
            f"Could not find a course table header in training plan: {xlsx_path.name}"
        )

    def parse_training_plans(
        self,
        plan_dir: str | Path,
    ) -> dict[str, dict[str, Any]]:
        """Load and merge every training plan without overwriting memberships."""
        plan_dir = Path(plan_dir)
        catalog: dict[str, dict[str, Any]] = {}
        for xlsx_file in sorted(plan_dir.glob("*.xlsx")):
            self._merge_catalog(catalog, self.parse_training_plan(xlsx_file))
        if not catalog:
            raise FileNotFoundError("未找到可解析的培养方案 Excel 文件（.xlsx）。")
        return catalog

    def build_training_plan_chunks(
        self,
        course_catalog: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create independently retrievable chunks for every plan offering."""
        chunks: list[dict[str, Any]] = []
        for code, course in sorted(course_catalog.items()):
            for offering in course.get("offerings", []):
                category = " / ".join(
                    part
                    for part in (
                        offering.get("course_category", ""),
                        offering.get("course_subcategory", ""),
                    )
                    if part
                )
                facts = [
                    f"培养方案：{offering.get('program_name', '')}",
                    f"方案类型：{offering.get('program_type', '')}",
                    f"课程类别：{category}",
                    f"课程：{course.get('course_name', '')}（{code}）",
                    f"英文名称：{course.get('english_name', '')}",
                    f"学分：{offering.get('credits')}",
                    f"理论学分：{offering.get('theory_credits')}",
                    f"实验实践学分：{offering.get('practice_credits')}",
                    f"总学时：{offering.get('hours')}",
                    f"理论学时：{offering.get('theory_hours')}",
                    f"实验实践学时：{offering.get('practice_hours')}",
                    f"开课学期：{offering.get('semester')}",
                    f"课程负责人：{offering.get('instructor', '')}",
                ]
                text = "\n".join(
                    fact
                    for fact in facts
                    if not fact.endswith(("：", "：None"))
                )
                source_file = (
                    f"training_plans/{offering.get('source_file', '')}"
                    f"#{offering.get('sheet_name', '')}"
                )
                chunk_id = hashlib.sha256(
                    f"training-plan|{source_file}|{code}".encode("utf-8")
                ).hexdigest()
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "source_file": source_file,
                        "section": "培养方案课程信息",
                        "metadata": {
                            "source_type": "training_plan",
                            "course_code": code,
                            "course_name": course.get("course_name", ""),
                            "program_name": offering.get("program_name", ""),
                            "program_type": offering.get("program_type", ""),
                            "course_category": offering.get(
                                "course_category",
                                "",
                            ),
                            "course_subcategory": offering.get(
                                "course_subcategory",
                                "",
                            ),
                            "credits": offering.get("credits"),
                            "hours": offering.get("hours"),
                            "semester": offering.get("semester"),
                            "instructor": offering.get("instructor", ""),
                            "prerequisites": course.get("prerequisites", []),
                        },
                    }
                )
        return chunks

    def parse_syllabus(self, docx_path: str | Path) -> list[dict[str, str]]:
        """Parse a DOCX syllabus into bounded, semantically typed chunks."""
        docx_path = Path(docx_path)
        document = docx.Document(docx_path)

        section_chunks: list[dict[str, str]] = []
        current_title = "文档概览"
        current_lines: list[str] = []

        def flush_current_section() -> None:
            content = self._normalize_text("\n\n".join(current_lines))
            if content:
                for part in self._split_content(content):
                    section_chunks.append(
                        {
                            "section_title": current_title,
                            "content": part,
                            "section_type": self._section_type(current_title),
                        }
                    )

        for block in self._iter_block_items(document):
            if isinstance(block, Paragraph):
                if self._paragraph_is_effectively_empty(block):
                    continue

                paragraph_text = self._normalize_text(block.text)
                if self._is_template_noise(paragraph_text):
                    continue
                if self._is_heading(block):
                    flush_current_section()
                    current_title = re.sub(
                        r"[（(].*?[）)]",
                        "",
                        paragraph_text,
                    ).strip(" ：:")
                    current_lines = []
                else:
                    current_lines.append(paragraph_text)
            else:
                flush_current_section()
                current_lines = []
                for table_section in self._table_sections(block):
                    for part in self._split_content(table_section["content"]):
                        section_chunks.append(
                            {
                                **table_section,
                                "content": part,
                            }
                        )

        flush_current_section()
        return section_chunks

    @staticmethod
    def _source_year(source_file: str) -> str:
        match = re.search(r"(20\d{2})", source_file)
        return match.group(1) if match else ""

    def build_syllabus_chunks(
        self,
        docx_path: str | Path,
        syllabus_dir: str | Path,
        course_catalog: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build final index records for one syllabus document."""
        docx_path = Path(docx_path)
        syllabus_dir = Path(syllabus_dir)
        parsed_chunks = self.parse_syllabus(docx_path)
        if not parsed_chunks:
            return []

        course_metadata = self._match_course_metadata(
            docx_path,
            parsed_chunks,
            course_catalog,
        )
        source_file = str(docx_path.relative_to(syllabus_dir).as_posix())
        source_year = self._source_year(source_file)
        catalog_prerequisites = course_metadata.get("prerequisites") or []
        if not isinstance(catalog_prerequisites, list):
            catalog_prerequisites = [catalog_prerequisites]
        raw_prerequisites = [
            *(
                name
                for value in catalog_prerequisites
                for name in self._split_prerequisites(value)
            ),
            *self._explicit_prerequisite_names(parsed_chunks),
        ]
        raw_prerequisites = list(dict.fromkeys(raw_prerequisites))
        prerequisite_evidence, unresolved_prerequisites = (
            self.resolve_prerequisites(
                raw_prerequisites,
                course_catalog,
                dependent_course_code=self._normalize_text(
                    course_metadata.get("course_code", "")
                ),
                source_file=source_file,
                section="课程基本信息",
                source_year=source_year,
            )
        )
        prerequisite_codes = [
            item["course_code"] for item in prerequisite_evidence
        ]
        document_hash = hashlib.sha256(docx_path.read_bytes()).hexdigest()
        final_chunks: list[dict[str, Any]] = []
        title_counts: dict[str, int] = {}
        for chunk in parsed_chunks:
            section = self._normalize_text(chunk.get("section_title", ""))
            content = self._normalize_text(chunk.get("content", ""))
            if not content:
                continue
            title_counts[section] = title_counts.get(section, 0) + 1
            part_number = title_counts[section]
            text = self._normalize_text(f"{section}\n\n{content}")
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            final_chunks.append(
                {
                    "chunk_id": hashlib.sha256(
                        f"{source_file}|{section}|{part_number}|{text_hash}".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "text": text,
                    "source_file": source_file,
                    "document_hash": document_hash,
                    "section": section,
                    "metadata": {
                        "source_type": "syllabus",
                        "course_code": self._normalize_text(
                            course_metadata.get("course_code", "")
                        ),
                        "course_name": self._normalize_text(
                            course_metadata.get("course_name", "")
                        ),
                        "syllabus_section": section,
                        "section_type": chunk.get("section_type")
                        or self._section_type(section),
                        "parent_document": source_file,
                        "parent_section": section,
                        "chunk_part": part_number,
                        "source_year": source_year,
                        "prerequisites": prerequisite_codes,
                        "prerequisite_evidence": prerequisite_evidence,
                        "unresolved_prerequisites": unresolved_prerequisites,
                        "credits": course_metadata.get("credits"),
                    },
                }
            )
        return final_chunks

    def run(
        self,
        plan_dir: str | Path,
        syllabus_dir: str | Path,
        output_path: str | Path,
    ) -> list[dict[str, Any]]:
        """Load all plans, process all syllabi recursively, and save merged chunks."""
        plan_dir = Path(plan_dir)
        syllabus_dir = Path(syllabus_dir)
        output_path = Path(output_path)

        if not plan_dir.exists():
            raise FileNotFoundError(f"培养方案目录不存在: {plan_dir}")
        if not syllabus_dir.exists():
            raise FileNotFoundError(f"课程大纲目录不存在: {syllabus_dir}")

        all_course_metadata = self.parse_training_plans(plan_dir)
        final_chunks = self.build_training_plan_chunks(all_course_metadata)
        docx_files = sorted(
            path for path in syllabus_dir.rglob("*.docx") if not path.name.startswith("~$")
        )

        for docx_file in docx_files:
            final_chunks.extend(
                self.build_syllabus_chunks(
                    docx_file,
                    syllabus_dir,
                    all_course_metadata,
                )
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(final_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return final_chunks


def main() -> None:
    """Provide a small CLI for parsing syllabi and training plans."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse Excel training plans and DOCX syllabi into chunks."
    )
    parser.add_argument(
        "--plan-dir",
        default="data/training_plans",
        help="Directory containing one or more training-plan .xlsx files.",
    )
    parser.add_argument(
        "--syllabus-dir",
        default="data/syllabi",
        help="Directory containing syllabus .docx files, scanned recursively.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/chunks.json",
        help="Destination JSON file for the processed chunks.",
    )
    args = parser.parse_args()

    chunker = SyllabusChunker()
    results = chunker.run(
        plan_dir=args.plan_dir,
        syllabus_dir=args.syllabus_dir,
        output_path=args.output_path,
    )
    print(f"Generated {len(results)} chunks.")


if __name__ == "__main__":
    main()
