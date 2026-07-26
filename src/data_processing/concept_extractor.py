"""Course-level concept extraction with persistent, hash-keyed cache."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROMPT_VERSION = "course-concepts-v1"

_NOISE = {"教学进度表", "成绩评定", "参考书目", "参考书", "文档概览", "课程基本信息", "考核方式", "教材"}

def _rule_concepts(items: list[dict]) -> list[str]:
    text = "\n".join(c.get("text", "") for c in items)
    candidates = re.findall(r"[\u4e00-\u9fffA-Za-z]{2,16}(?:、[\u4e00-\u9fffA-Za-z]{2,16})?", text)
    result = []
    for value in candidates:
        for part in value.split("、"):
            part = part.strip()
            if part not in _NOISE and len(part) >= 2 and part not in result:
                result.append(part)
    return result[:40]

def extract_course_concepts(chunks: list[dict], cache_path: str | Path, model: str = "local", llm_client=None) -> list[dict]:
    path = Path(cache_path)
    cache = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    grouped = defaultdict(list)
    for c in chunks:
        code = c.get("metadata", {}).get("course_code") or c.get("course_code") or c.get("source_file", "unknown")
        grouped[code].append(c)
    concepts = []
    for code, items in grouped.items():
        doc_hash = hashlib.sha256("".join(c.get("document_hash", "") for c in items).encode()).hexdigest()
        old = cache.get(code, {})
        if old.get("document_hash") == doc_hash and old.get("model") == model and old.get("prompt_version") == PROMPT_VERSION:
            concepts.extend(old.get("concepts", []))
            continue
        values = _rule_concepts(items)
        rows = [{"concept_id": hashlib.sha256(f"{code}|{v}".encode()).hexdigest()[:20], "name": v, "course_code": code, "aliases": []} for v in values]
        cache[code] = {"document_hash": doc_hash, "model": model, "prompt_version": PROMPT_VERSION, "concepts": rows, "created_at": datetime.now(timezone.utc).isoformat()}
        concepts.extend(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return concepts
