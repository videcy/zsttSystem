"""Accept training plans and syllabi at runtime instead of committing them.

The corpus used to live in the repository, which made replacing it a git
operation and published one school's teaching materials to anyone who cloned
it.  Uploaded files land in the same directories the pipeline already reads
(``data/training_plans`` for .xlsx, ``data/syllabi`` for .docx), so nothing
downstream changes.

Everything here is deliberately paranoid about filenames: an import endpoint
writes attacker-controlled names to disk, and this service is meant to run on
a public host.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Extension -> the directory the pipeline scans for it.
SYLLABUS_SUFFIXES = frozenset({".docx"})
TRAINING_PLAN_SUFFIXES = frozenset({".xlsx"})
ALLOWED_SUFFIXES = SYLLABUS_SUFFIXES | TRAINING_PLAN_SUFFIXES

_UNSAFE = re.compile(r"[^\w一-鿿.\- ]", re.UNICODE)


class ImportRejected(Exception):
    """A file was refused before anything touched the filesystem."""


@dataclass(frozen=True)
class StoredFile:
    filename: str
    kind: str
    size: int
    replaced: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "kind": self.kind,
            "size": self.size,
            "replaced": self.replaced,
        }


def safe_filename(raw: str) -> str:
    """Reduce an uploaded name to a plain filename inside the target directory.

    Directory components, traversal segments and control characters are
    dropped rather than escaped: none of them can be part of a legitimate
    syllabus name, so rejecting the intent is safer than sanitising it.
    """
    name = unicodedata.normalize("NFC", str(raw or "")).strip()
    # Windows and POSIX separators both, whatever the client sent.
    name = name.replace("\\", "/").split("/")[-1]
    name = _UNSAFE.sub("_", name).strip(" .")
    if not name or set(name) <= {"_"}:
        raise ImportRejected("文件名无效")
    if len(name) > 150:
        stem, _, suffix = name.rpartition(".")
        name = f"{stem[:140]}.{suffix}" if suffix else stem[:150]
    return name


def classify(filename: str) -> str:
    """Return ``syllabus`` or ``training_plan`` for an accepted extension."""
    suffix = Path(filename).suffix.lower()
    if suffix in SYLLABUS_SUFFIXES:
        return "syllabus"
    if suffix in TRAINING_PLAN_SUFFIXES:
        return "training_plan"
    raise ImportRejected(
        f"只接受 {'、'.join(sorted(ALLOWED_SUFFIXES))} 文件，收到 {suffix or '无扩展名'}"
    )


def target_directory(kind: str, *, syllabus_dir: Path, training_plan_dir: Path) -> Path:
    return syllabus_dir if kind == "syllabus" else training_plan_dir


def store_upload(
    filename: str,
    payload: bytes,
    *,
    syllabus_dir: Path,
    training_plan_dir: Path,
    max_bytes: int,
) -> StoredFile:
    """Validate one upload and write it into the pipeline's input tree."""
    if not payload:
        raise ImportRejected("文件为空")
    if len(payload) > max_bytes:
        raise ImportRejected(
            f"文件过大：{len(payload)} 字节，上限 {max_bytes} 字节"
        )

    # Checked on the raw basename: safe_filename replaces "~$" with
    # underscores, so testing the sanitised name would never see it.
    basename = str(filename or "").replace("\\", "/").split("/")[-1].strip()
    if basename.startswith("~$"):
        raise ImportRejected("Office 临时文件不导入")

    name = safe_filename(filename)
    kind = classify(name)

    # Both formats are ZIP containers; a mislabelled .docx would only fail
    # later inside the parser, with a far less obvious message.
    if not payload.startswith(b"PK\x03\x04"):
        raise ImportRejected("不是有效的 docx/xlsx（缺少 ZIP 文件头）")

    directory = target_directory(
        kind,
        syllabus_dir=syllabus_dir,
        training_plan_dir=training_plan_dir,
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    replaced = destination.exists()
    destination.write_bytes(payload)
    return StoredFile(
        filename=name,
        kind=kind,
        size=len(payload),
        replaced=replaced,
    )


def inventory(*, syllabus_dir: Path, training_plan_dir: Path) -> dict[str, object]:
    """Summarise what the pipeline would currently parse."""

    def _describe(directory: Path, suffixes: frozenset[str]) -> list[dict[str, object]]:
        if not directory.exists():
            return []
        return sorted(
            (
                {
                    "filename": path.name,
                    "size": path.stat().st_size,
                }
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in suffixes
                and not path.name.startswith("~$")
            ),
            key=lambda item: str(item["filename"]),
        )

    syllabi = _describe(syllabus_dir, frozenset(SYLLABUS_SUFFIXES))
    plans = _describe(training_plan_dir, frozenset(TRAINING_PLAN_SUFFIXES))
    return {
        "syllabi": syllabi,
        "training_plans": plans,
        "counts": {"syllabi": len(syllabi), "training_plans": len(plans)},
    }
