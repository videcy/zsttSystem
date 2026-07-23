"""Parse training plans and course syllabi into structure-aware chunks."""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Any, Iterator

import docx
import pandas as pd
from docx.document import Document as DocxDocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph


class SyllabusChunker:
    """Parse XLSX training plans and DOCX syllabi into teaching-oriented chunks."""

    def __init__(self) -> None:
        self.section_keywords = (
            "课程目标",
            "教学内容",
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

        body = dataframe.iloc[data_start_index:].copy()
        body.columns = columns
        body = body.fillna("")

        code_column = self._find_column(columns, ("课程代码", "课程编码", "课程编号"))
        name_column = self._find_column(columns, ("课程名称", "课程中文名称", "课程中文名称/英文名称"))
        credits_column = self._find_column(
            columns,
            ("总学分", "学分情况-总学分", "学分", "课程学分"),
        )
        prereq_column = self._find_column(columns, ("先修课程代码", "先修课程", "前置课程", "先修"))

        if code_column is None or name_column is None:
            return {}

        program_name = self._extract_program_name(dataframe, xlsx_path)
        course_catalog: dict[str, dict[str, Any]] = {}
        for _, row in body.iterrows():
            raw_code = self._normalize_text(row.get(code_column, ""))
            raw_name = self._normalize_text(row.get(name_column, ""))
            if not raw_code or not raw_name:
                continue

            course_code = self._extract_course_code_from_text(raw_code) or raw_code.splitlines()[0].strip()
            course_name = raw_name.splitlines()[0].strip()
            if not course_code or not course_name:
                continue

            credits_value = row.get(credits_column, "") if credits_column else ""
            credits: int | float | None
            if credits_value == "":
                credits = None
            else:
                try:
                    numeric_value = float(str(credits_value).strip())
                    credits = int(numeric_value) if numeric_value.is_integer() else numeric_value
                except (TypeError, ValueError):
                    credits = None

            course_catalog[course_code] = {
                "program_name": program_name,
                "course_code": course_code,
                "course_name": course_name,
                "credits": credits,
                "prerequisites": self._split_prerequisites(row.get(prereq_column, "")) if prereq_column else [],
            }

        return course_catalog

    def _split_prerequisites(self, value: Any) -> list[str]:
        """Split prerequisite course codes or names into a list."""
        text = self._normalize_text(value)
        if not text:
            return []
        parts = re.split(r"[、,，;/；\n]+", text)
        return [part.strip() for part in parts if part.strip()]

    def _extract_course_code_from_text(self, text: str) -> str:
        """Extract a likely course code from a syllabus title or filename."""
        match = self.course_code_pattern.search(text)
        return match.group(0) if match else ""

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

        if any(keyword in text for keyword in self.section_keywords):
            return len(text) <= 40

        if re.match(r"^\s*(第[一二三四五六七八九十百0-9]+[章节部分单元].*|\d+(\.\d+){0,2}\s*.+)$", text):
            return True

        return False

    def _paragraph_is_effectively_empty(self, paragraph: Paragraph) -> bool:
        """Skip blank paragraphs."""
        return not self._normalize_text(paragraph.text)

    def _table_to_markdown(self, table: Table) -> str:
        """Convert a DOCX table into Markdown text."""
        rows: list[list[str]] = []
        for row in table.rows:
            row_values = [self._normalize_text(cell.text) for cell in row.cells]
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
        for metadata in course_catalog.values():
            course_name = self._normalize_text(metadata.get("course_name", ""))
            if filename_name and course_name and (
                filename_name in course_name or course_name in filename_name
            ):
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
            course_catalog.update(self._extract_sheet_course_catalog(dataframe, xlsx_path))

        if course_catalog:
            return course_catalog

        raise ValueError(
            f"Could not find a course table header in training plan: {xlsx_path.name}"
        )

    def parse_syllabus(self, docx_path: str | Path) -> list[dict[str, str]]:
        """Parse a DOCX syllabus and split it into section-level chunks."""
        docx_path = Path(docx_path)
        document = docx.Document(docx_path)

        section_chunks: list[dict[str, str]] = []
        current_title = "文档概览"
        current_lines: list[str] = []

        def flush_current_section() -> None:
            content = self._normalize_text("\n\n".join(current_lines))
            if content:
                section_chunks.append(
                    {
                        "section_title": current_title,
                        "content": content,
                    }
                )

        for block in self._iter_block_items(document):
            if isinstance(block, Paragraph):
                if self._paragraph_is_effectively_empty(block):
                    continue

                paragraph_text = self._normalize_text(block.text)
                if self._is_heading(block):
                    flush_current_section()
                    current_title = paragraph_text
                    current_lines = []
                else:
                    current_lines.append(paragraph_text)
            else:
                table_markdown = self._table_to_markdown(block)
                if table_markdown:
                    current_lines.append(table_markdown)

        flush_current_section()
        return section_chunks

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

        all_course_metadata: dict[str, dict[str, Any]] = {}
        for xlsx_file in sorted(plan_dir.glob("*.xlsx")):
            course_metadata = self.parse_training_plan(xlsx_file)
            all_course_metadata.update(course_metadata)

        if not all_course_metadata:
            raise FileNotFoundError("未找到可解析的培养方案 Excel 文件（.xlsx）。")

        final_chunks: list[dict[str, Any]] = []
        docx_files = sorted(
            path for path in syllabus_dir.rglob("*.docx") if not path.name.startswith("~$")
        )

        for docx_file in docx_files:
            syllabus_chunks = self.parse_syllabus(docx_file)
            if not syllabus_chunks:
                continue

            course_metadata = self._match_course_metadata(
                docx_file,
                syllabus_chunks,
                all_course_metadata,
            )
            for chunk in syllabus_chunks:
                text_parts = [chunk.get("section_title", ""), chunk.get("content", "")]
                text = self._normalize_text("\n\n".join(part for part in text_parts if part))
                if not text:
                    continue

                source_file = str(docx_file.relative_to(syllabus_dir).as_posix())
                section = self._normalize_text(chunk.get("section_title", ""))
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                chunk_id = hashlib.sha256(
                    f"{source_file}|{section}|{text_hash}".encode("utf-8")
                ).hexdigest()
                final_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "source_file": source_file,
                        "document_hash": hashlib.sha256(docx_file.read_bytes()).hexdigest(),
                        "section": section,
                        "metadata": {
                            "course_code": self._normalize_text(course_metadata.get("course_code", "")),
                            "course_name": self._normalize_text(course_metadata.get("course_name", "")),
                            "syllabus_section": self._normalize_text(chunk.get("section_title", "")),
                            "prerequisites": list(course_metadata.get("prerequisites", [])),
                            "credits": course_metadata.get("credits"),
                        },
                    }
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
        default="outputs/chunked_data.json",
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
