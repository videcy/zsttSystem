"""Resolve which training plan a question is about, using the plans' own names.

The router used to carry one school's four program names as literals, so
pointing the system at another school's 培养方案 silently broke every
catalogue question.  Everything here is derived from the parsed plans
instead: the program names, the plan types, and the "core" name a student
actually types (``档案学`` rather than
``信息管理学院2025级档案学专业培养方案``).

Only two things stay in code, and neither names a school:

* the structural affixes of a plan title (``...学院``, ``2025级``,
  ``培养方案``), which are a naming convention, and
* nothing else -- colloquial abbreviations such as ``信管`` are deployment
  configuration (``PROGRAM_ALIASES``), because they differ per campus.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

_TITLE_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^.*学院"),
    re.compile(r"^.*学校"),
    re.compile(r"^\d{4}\s*级"),
)

# Applied repeatedly, longest first, so that
# "档案学专业辅修微专业培养方案" unwinds to "档案学".
_TITLE_SUFFIXES: tuple[str, ...] = ("培养方案", "教学计划", "专业")


def core_program_name(
    program_name: str,
    program_types: Sequence[str] = (),
) -> str:
    """Strip the structural affixes of a plan title down to its core name."""
    text = str(program_name or "").strip()
    for pattern in _TITLE_PREFIXES:
        text = pattern.sub("", text, count=1)
    suffixes = (
        *sorted((value for value in program_types if value), key=len, reverse=True),
        *_TITLE_SUFFIXES,
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
                # Restart from the longest suffix: stripping "培养方案" off
                # "...辅修微专业培养方案" must not let the shorter "专业" bite
                # into "辅修微专业" on the same pass.
                break
    return text.strip()


class ProgramIndex:
    """Program names and types observed in the parsed course catalogue."""

    def __init__(
        self,
        courses: Iterable[Mapping[str, Any]],
        *,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        names: set[str] = set()
        type_counts: Counter[str] = Counter()
        for course in courses or ():
            for offering in course.get("offerings") or ():
                name = str(offering.get("program_name") or "").strip()
                program_type = str(offering.get("program_type") or "").strip()
                if name:
                    names.add(name)
                if program_type:
                    type_counts[program_type] += 1

        # Longest first everywhere: "辅修微专业" must win over "微专业", and
        # "信息管理与信息系统" over a shorter program that is a prefix of it.
        self.program_names: tuple[str, ...] = tuple(
            sorted(names, key=len, reverse=True)
        )
        self.program_types: tuple[str, ...] = tuple(
            sorted(type_counts, key=len, reverse=True)
        )
        self.aliases: dict[str, str] = {
            str(key).strip(): str(value).strip()
            for key, value in (aliases or {}).items()
            if str(key).strip() and str(value).strip()
        }
        cores = {
            core_program_name(name, self.program_types) for name in self.program_names
        }
        self.core_names: tuple[str, ...] = tuple(
            sorted((core for core in cores if core), key=len, reverse=True)
        )

    def match_keyword(self, query: str) -> str:
        """Return the program-name fragment the query refers to, or ``""``.

        Tried in order: configured aliases, derived core names, then the raw
        plan titles -- so typing the full official name always works even when
        the affix stripping does not fit a school's naming style.
        """
        text = str(query or "")
        for alias in sorted(self.aliases, key=len, reverse=True):
            if alias in text:
                return self.aliases[alias]
        for core in self.core_names:
            if core in text:
                return core
        for name in self.program_names:
            if name in text:
                return name
        return ""

    def match_type(
        self,
        query: str,
        offerings: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        """Return the plan type to filter by, or ``""`` to accept any.

        A query naming a type wins ("辅修微专业").  Otherwise the main plan is
        meant, and the main plan is simply the type carrying the most
        offerings among the candidates -- which requires no vocabulary
        knowledge of what "主修" means.
        """
        text = str(query or "")
        for program_type in self.program_types:
            if program_type in text:
                return program_type
        counts = Counter(
            str(offering.get("program_type") or "").strip()
            for offering in offerings
        )
        counts.pop("", None)
        return counts.most_common(1)[0][0] if counts else ""
