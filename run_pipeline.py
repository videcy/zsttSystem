"""Command-line entry point for the offline zsttSystem data pipeline.

Use ``--incremental`` to skip unchanged syllabus files based on a SHA256
manifest. The ``all`` stage builds local parsing, concept, graph, and vector
artifacts in ChromaDB and optional Neo4j.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError

from src.config import config
from src.data_processing.parser_chunker import SyllabusChunker
from src.data_processing.chroma_index import build_index
from src.data_processing.lexical_stats import write_lexical_stats
from src.data_processing.concept_normalizer import ConceptNormalizer
from src.data_processing.graph_builder import (
    CONCEPT_DEPENDENCY_TYPES,
    build_graph_records,
    write_neo4j,
)
from src.utils.file_manifest import FileManifest

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
TRAINING_PLAN_DIR = config.training_plan_dir
SYLLABUS_DIR = config.syllabus_dir
CHUNKS_OUTPUT_PATH = config.chunks_output_path
OUTPUT_DIR = CHUNKS_OUTPUT_PATH.parent
NEO4J_URI = config.neo4j_uri
NEO4J_USER = config.neo4j_user
NEO4J_PASSWORD = config.neo4j_password

PipelineStep = Callable[[], None]

CONCEPT_SECTION_TYPES = {
    "course_objectives",
    "teaching_content",
    "teaching_schedule",
}

# 培养方案与大纲同义字段的对应关系（规范字段 -> courses.json 字段），
# 用于 M3.3 的冲突比对；不在表内的字段不参与冲突判定。
_PROFILE_PLAN_FIELDS = {
    "credits": "credits",
    "hours": "hours",
    "course_owner": "instructor",
}

# 视为「培养方案没写」的占位值，不参与冲突判定。
_MISSING_VALUES = {"", "/", "-", "—", "－", "无", "暂无", "nan", "none", "null", "n/a"}


# ---------------------------------------------------------------------------
# Derived artifacts: doc_tree.json / syllabus_profiles.json（大创改造 P1-3、P1-4）
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Any) -> None:
    """先写同目录临时文件再 ``replace``，避免在线服务读到半成品产物。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_or(path: Path, default: Any) -> Any:
    """容错读取 JSON：文件缺失或损坏时返回 *default*，不打断构建阶段。"""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _is_missing_value(value: Any) -> bool:
    """判断培养方案 / 大纲字段是否为空（``/`` 这类占位符也算空）。"""
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return str(value).strip().casefold() in _MISSING_VALUES


def _values_differ(plan_value: Any, syllabus_value: Any) -> bool:
    """比较两个字段值：能按数字比较时忽略 ``3`` 与 ``3.0`` 的写法差异。"""
    try:
        return float(plan_value) != float(syllabus_value)
    except (TypeError, ValueError):
        return str(plan_value).strip() != str(syllabus_value).strip()


def _profile_field_value(profile: dict[str, Any], field: str) -> Any:
    """取画像里的规范字段值，兼容平铺、``fields`` 子字典与 ``{"value": ...}`` 包装。"""
    for container in (profile, profile.get("fields")):
        if not isinstance(container, dict) or field not in container:
            continue
        value = container[field]
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value
    return None


def _profile_conflicts(
    course: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """列出培养方案与大纲不一致的字段；培养方案值不覆盖，只记录（M3.3）。"""
    conflicts: list[dict[str, Any]] = []
    for profile_field, plan_field in _PROFILE_PLAN_FIELDS.items():
        plan_value = course.get(plan_field)
        syllabus_value = _profile_field_value(profile, profile_field)
        if _is_missing_value(plan_value) or _is_missing_value(syllabus_value):
            continue
        if _values_differ(plan_value, syllabus_value):
            conflicts.append(
                {
                    "field": profile_field,
                    "plan_value": plan_value,
                    "syllabus_value": syllabus_value,
                }
            )
    return conflicts


def _write_doc_tree(chunks: list[dict[str, Any]]) -> None:
    """构建并原子写回 ``doc_tree.json``（M2.1 / M2.2）。

    构建模块（``src.data_processing.doc_tree``）并行开发中，缺失时只告警、
    保留磁盘上的旧产物，让解析阶段照常完成。
    """
    try:
        from src.data_processing.doc_tree import build_doc_tree
    except ImportError as exc:  # pragma: no cover - 依赖并行交付
        print(f"[parsing] 跳过 doc_tree：构建模块不可用（{exc}）")
        return
    try:
        tree = build_doc_tree(chunks)
    except Exception as exc:  # noqa: BLE001 - 树构建失败不应中断解析
        print(f"[parsing] doc_tree 构建失败：{type(exc).__name__}: {exc}")
        return
    if not isinstance(tree, dict):
        print("[parsing] doc_tree 构建返回了非 dict，已跳过写盘")
        return
    _atomic_write_json(config.doc_tree_path, tree)
    documents = tree.get("documents")
    count = len(documents) if isinstance(documents, dict) else 0
    print(f"[parsing] doc_tree: {count} documents → {config.doc_tree_path}")


def _syllabus_groups(
    chunks: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """按 ``course_code|source_year`` 归组大纲 chunk，再按来源文件归组。"""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        if metadata.get("source_type") != "syllabus":
            continue
        course_code = str(
            metadata.get("course_code") or chunk.get("course_code") or ""
        ).strip()
        source_file = str(chunk.get("source_file") or "").strip()
        if not course_code or not source_file:
            continue
        source_year = str(metadata.get("source_year") or "").strip()
        key = f"{course_code}|{source_year}"
        grouped.setdefault(key, {}).setdefault(source_file, []).append(chunk)
    return grouped


def _write_syllabus_profiles(
    chunks: list[dict[str, Any]],
    *,
    refresh_keys: set[str] | None = None,
) -> None:
    """汇总大纲结构化字段并原子写回 ``syllabus_profiles.json``（M3.1–M3.2）。

    *refresh_keys* 为 ``None`` 时全量重算；否则只重算这些 ``course_code|source_year``
    键，其余沿用磁盘上的旧记录（增量解析用）。
    """
    try:
        from src.data_processing.parser_chunker import collect_syllabus_profile
    except ImportError as exc:  # pragma: no cover - 依赖并行交付
        print(f"[parsing] 跳过 syllabus_profiles：抽取函数不可用（{exc}）")
        return

    grouped = _syllabus_groups(chunks)
    if refresh_keys is None:
        profiles: dict[str, Any] = {}
        targets = set(grouped)
    else:
        existing = _read_json_or(config.syllabus_profiles_path, {})
        profiles = (
            {key: value for key, value in existing.items() if key not in refresh_keys}
            if isinstance(existing, dict)
            else {}
        )
        targets = set(grouped) & set(refresh_keys)

    for key in sorted(targets):
        for source_file in sorted(grouped[key]):
            members = grouped[key][source_file]
            metadata = members[0].get("metadata") or {}
            course_code, source_year = key.split("|", 1)
            try:
                profile = collect_syllabus_profile(
                    members,
                    source_file=source_file,
                    course_code=course_code,
                    course_name=str(metadata.get("course_name") or ""),
                    source_year=source_year,
                )
            except Exception as exc:  # noqa: BLE001 - 单份大纲失败不影响其余
                print(
                    f"[parsing] syllabus_profile 抽取失败 {source_file}："
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if not isinstance(profile, dict):
                continue
            record = dict(profile)
            if not record.get("source_file"):
                record["source_file"] = source_file
            profiles[key] = record

    _atomic_write_json(config.syllabus_profiles_path, profiles)
    print(
        f"[parsing] syllabus profiles: {len(profiles)} 条 → "
        f"{config.syllabus_profiles_path}"
    )


def _write_parse_artifacts(
    chunks: list[dict[str, Any]],
    *,
    refresh_profile_keys: set[str] | None = None,
    write_chunks: bool = False,
) -> None:
    """在同一次回调里原子替换 chunks.json（可选）、doc_tree.json、syllabus_profiles.json。

    三份产物必须一起刷新：只换其中一份会让在线服务读到「新向量库 + 旧文档树」
    的半成品状态（§8 风险表）。
    """
    if write_chunks:
        _atomic_write_json(CHUNKS_OUTPUT_PATH, chunks)
    _write_doc_tree(chunks)
    _write_syllabus_profiles(chunks, refresh_keys=refresh_profile_keys)


def _merge_syllabus_profiles_into_courses(courses_path: Path) -> None:
    """把大纲画像合并进 ``courses.json``（M3.3），保持其余字段原样。

    每门课新增 ``syllabus_profile``；与培养方案冲突时**不覆盖**培养方案值，
    把 ``[{field, plan_value, syllabus_value}]`` 同时写进 ``syllabus_profile``
    与课程级 ``conflicts``，回答时可以两者都给并注明来源。同一门课有多份大纲
    时取 ``course_code|source_year`` 键序最大的那份（即最新学年）。
    """
    profiles = _read_json_or(config.syllabus_profiles_path, None)
    if not isinstance(profiles, dict) or not profiles:
        return
    courses = _read_json_or(courses_path, None)
    if not isinstance(courses, list):
        return

    by_code: dict[str, dict[str, Any]] = {}
    for key in sorted(profiles):
        record = profiles[key]
        if not isinstance(record, dict):
            continue
        course_code = str(record.get("course_code") or key.split("|", 1)[0]).strip()
        if course_code:
            by_code[course_code] = record

    merged: list[Any] = []
    changed = False
    for course in courses:
        if not isinstance(course, dict):
            merged.append(course)
            continue
        profile = by_code.get(str(course.get("course_code") or "").strip())
        if profile is None:
            merged.append(course)
            continue
        conflicts = _profile_conflicts(course, profile)
        record = dict(profile)
        record["conflicts"] = conflicts
        updated = {key: value for key, value in course.items() if key != "conflicts"}
        updated["syllabus_profile"] = record
        if conflicts:
            updated["conflicts"] = conflicts
        merged.append(updated)
        changed = True

    if changed:
        _atomic_write_json(courses_path, merged)
        print(
            f"[parsing] syllabus_profile 已合并进 {courses_path.name} "
            f"（{len(by_code)} 门课）"
        )



# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_parsing(*, incremental: bool = False, force: bool = False) -> bool:
    """Parse XLSX training plans + DOCX syllabi into structured chunks.

    When *incremental* is True and *force* is False, only re-parses syllabus
    files whose content has changed (detected via SHA256 manifest).  Chunks
    from unchanged files are preserved from the previous run.

    Returns True when the incremental patch path ran, so the caller knows the
    derived artifacts were already written inside that callback.
    """
    chunker = SyllabusChunker()

    if incremental and not force and CHUNKS_OUTPUT_PATH.exists():
        _incremental_parse(chunker)
        return True

    results = chunker.run(
        plan_dir=TRAINING_PLAN_DIR,
        syllabus_dir=SYLLABUS_DIR,
        output_path=CHUNKS_OUTPUT_PATH,
    )
    tag = "force" if force else "full"
    print(f"[parsing] ({tag}) Generated {len(results)} chunks → {CHUNKS_OUTPUT_PATH}")
    return False


def _pop_chunks_for_file(
    existing_by_file: dict[str, list[dict]],
    relative_path: str,
) -> list[dict]:
    """取走某个被删除 / 被替换文件对应的旧 chunk。

    ``FileManifest`` 记录的是构造时传入的路径（绝对路径），而 ``chunks.json``
    里的 ``source_file`` 相对大纲目录，因此除精确匹配外还要兜住后缀匹配，
    否则被删除文件的 chunk 会一直留在产物里。
    """
    if relative_path in existing_by_file:
        return existing_by_file.pop(relative_path)
    matched = [
        key
        for key in existing_by_file
        if key.endswith(f"/{relative_path}") or relative_path.endswith(f"/{key}")
    ]
    dropped: list[dict] = []
    for key in matched:
        dropped.extend(existing_by_file.pop(key))
    return dropped


def _profile_keys_of(chunks: list[dict]) -> set[str]:
    """这些 chunk 涉及的 ``course_code|source_year`` 画像键（增量重算用）。"""
    return set(_syllabus_groups(chunks))


def _incremental_parse(chunker: SyllabusChunker) -> None:
    """Detect changed syllabus files and patch the parse artifacts in-place."""
    manifest = FileManifest()
    course_catalog = chunker.parse_training_plans(TRAINING_PLAN_DIR)
    training_plan_chunks = chunker.build_training_plan_chunks(course_catalog)
    syllabus_files = sorted(
        p for p in SYLLABUS_DIR.rglob("*.docx") if not p.name.startswith("~$")
    )

    added, changed, removed = manifest.check(syllabus_files)

    # Load existing chunks
    existing = json.loads(CHUNKS_OUTPUT_PATH.read_text(encoding="utf-8"))
    existing_by_file: dict[str, list[dict]] = {}
    other_chunks: list[dict] = []
    for ch in existing:
        if (ch.get("metadata") or {}).get("source_type") == "training_plan":
            continue
        src = ch.get("source_file", "")
        if src:
            existing_by_file.setdefault(src, []).append(ch)
        else:
            other_chunks.append(ch)

    # 受影响的 (course_code, source_year) 画像键：这些键的 syllabus_profile 需要重算。
    affected_profile_keys: set[str] = set()

    # Remove chunks from deleted files
    for rel in removed:
        dropped = _pop_chunks_for_file(existing_by_file, rel)
        affected_profile_keys |= _profile_keys_of(dropped)
        print(f"[parsing] (incremental) Removed: {rel} ({len(dropped)} chunks)")

    # Re-parse changed / added files
    files_to_parse = added + changed
    if files_to_parse:
        for f in files_to_parse:
            rel = str(f.relative_to(SYLLABUS_DIR).as_posix())
            old_chunks = existing_by_file.get(rel, [])
            new_chunks = chunker.build_syllabus_chunks(
                f,
                SYLLABUS_DIR,
                course_catalog,
            )

            existing_by_file[rel] = new_chunks
            affected_profile_keys |= _profile_keys_of(old_chunks)
            affected_profile_keys |= _profile_keys_of(new_chunks)
            status = "Added" if f in added else "Changed"
            print(
                f"[parsing] (incremental) {status}: {rel} "
                f"({len(old_chunks)}→{len(new_chunks)} chunks)"
            )

    # Merge and write
    merged = [*training_plan_chunks, *other_chunks]
    for chunks in existing_by_file.values():
        merged.extend(chunks)

    # 增量同步 doc_tree / syllabus_profiles：同样在同一次回调里原子写回。
    # doc_tree 用「按合并后的 chunks 全量重算」而不是逐个 document_hash 打补丁：
    # document_hash 由解析层派生、不在冻结接口内，而全量重算只有 2677 条，
    # 既便宜又天然覆盖「文件被删除」与「内容变化」两种情形（M2.2 允许的退化）。
    _write_parse_artifacts(
        merged,
        refresh_profile_keys=affected_profile_keys,
        write_chunks=True,
    )
    print(f"[parsing] (incremental) Merged {len(merged)} chunks → {CHUNKS_OUTPUT_PATH}")

    # Update manifest
    manifest.update(syllabus_files)
    manifest.save()


def run_concept_stage() -> bool:
    chunks = json.loads(CHUNKS_OUTPUT_PATH.read_text(encoding="utf-8"))
    concept_chunks = [
        chunk
        for chunk in chunks
        if (chunk.get("metadata") or {}).get("source_type") == "syllabus"
        and (chunk.get("metadata") or {}).get("section_type")
        in CONCEPT_SECTION_TYPES
    ]
    if not concept_chunks:
        registry: list[dict] = []
        enriched_chunks: list[dict] = []
        for path in (
            config.concept_registry_path,
            config.concept_alias_path,
            config.concept_candidate_edge_path,
            config.concept_verified_edge_path,
        ):
            empty: list | dict = {} if path == config.concept_alias_path else []
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(empty, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        normalizer = ConceptNormalizer()
        if getattr(normalizer, "api_client", object()) is None:
            print(
                "[concept] skipped: DEEPSEEK_API_KEY is required for verified "
                "concept extraction; existing concept artifacts were preserved"
            )
            return False
        normalizer.allow_rule_fallback = False
        normalizer.require_complete_llm_validation = True
        normalizer.allow_embedding_fallback = False
        try:
            enriched_chunks, registry, _aliases = normalizer.preprocess_chunks(
                concept_chunks,
                registry_output_path=config.concept_registry_path,
                alias_output_path=config.concept_alias_path,
                candidate_output_path=config.concept_candidate_edge_path,
                verified_output_path=config.concept_verified_edge_path,
                extraction_cache_path=config.concept_extraction_cache_path,
                validation_cache_path=config.concept_validation_cache_path,
            )
        except RuntimeError as exc:
            cause = f" (cause: {exc.__cause__!r})" if exc.__cause__ else ""
            print(f"[concept] skipped: {exc}{cause}")
            return False

    enriched_by_id = {
        str(chunk.get("chunk_id")): chunk
        for chunk in enriched_chunks
        if chunk.get("chunk_id")
    }
    merged_chunks = [
        enriched_by_id.get(str(chunk.get("chunk_id")), chunk)
        for chunk in chunks
    ]
    CHUNKS_OUTPUT_PATH.write_text(
        json.dumps(merged_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Keep a compatibility projection for the course graph endpoint and older
    # local artifacts while the canonical registry remains the source of truth.
    concepts = [
        {
            **concept,
            "concept_id": concept["id"],
            "name": concept["canonical_name"],
            "course_code": course_code,
        }
        for concept in registry
        for course_code in (concept.get("source_course_codes") or [""])
    ]
    (OUTPUT_DIR / "concepts.json").write_text(
        json.dumps(concepts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[concept] {len(registry)} canonical concepts from "
        f"{len(concept_chunks)} teaching chunks"
    )
    return True

def run_embed_stage() -> None:
    chunks = json.loads(CHUNKS_OUTPUT_PATH.read_text(encoding="utf-8"))
    summary = build_index(
        chunks,
        OUTPUT_DIR,
        model_name=config.retrieval_model_name,
        dimensions=config.simple_embedding_dimensions,
    )
    print(
        f"[embed] ChromaDB collection {summary['collection']} "
        f"contains {summary['count']} chunks"
    )
    stats = write_lexical_stats(chunks, config.lexical_stats_path)
    print(
        f"[embed] lexical stats: {stats['document_count']} documents, "
        f"{stats['vocabulary_size']} terms "
        f"(kept df>={stats['min_document_frequency']})"
    )

def run_parse_stage(incremental: bool = False, force: bool = False) -> None:
    used_incremental = run_parsing(incremental=incremental, force=force)
    chunks = json.loads(CHUNKS_OUTPUT_PATH.read_text(encoding="utf-8"))
    if not used_incremental:
        # 增量路径已在 _incremental_parse 内写回这三份产物。
        # chunks.json 也重写一次：解析器自己的写盘不是原子的，这里用同一次
        # 回调把四份产物统一成「临时文件 + replace」。
        _write_parse_artifacts(chunks, write_chunks=True)
    chunker = SyllabusChunker()
    courses = chunker.parse_training_plans(TRAINING_PLAN_DIR)
    unresolved: list[dict] = []
    unresolved_keys: set[str] = set()
    for code, course in courses.items():
        raw_values = course.get("prerequisites") or []
        if not isinstance(raw_values, list):
            raw_values = [raw_values]
        offerings = course.get("offerings") or []
        source_file = offerings[0].get("source_file", "") if offerings else ""
        resolved, pending = chunker.resolve_prerequisites(
            [
                name
                for value in raw_values
                for name in chunker._split_prerequisites(value)
            ],
            courses,
            dependent_course_code=code,
            source_file=source_file,
            section="培养方案课程信息",
            source_year=chunker._source_year(source_file),
            source_type="training_plan",
        )
        course["prerequisites"] = [item["course_code"] for item in resolved]
        course["prerequisite_evidence"] = resolved
        for item in pending:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in unresolved_keys:
                unresolved.append(item)
                unresolved_keys.add(key)
    for c in chunks:
        if (c.get("metadata") or {}).get("source_type") == "training_plan":
            continue
        m = c.get("metadata", {})
        code = m.get("course_code") or c.get("course_code")
        if code:
            course = courses.setdefault(
                code,
                {
                    "course_code": code,
                    "course_name": m.get("course_name", ""),
                    "credits": m.get("credits"),
                    "hours": None,
                    "semester": None,
                    "instructor": "",
                    "course_category": "",
                    "course_subcategory": "",
                    "prerequisites": [],
                    "program_names": [],
                    "offerings": [],
                },
            )
            course["prerequisites"] = list(
                dict.fromkeys(
                    [
                        *(course.get("prerequisites") or []),
                        *(m.get("prerequisites") or []),
                    ]
                )
            )
            known_evidence = {
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                for item in course.get("prerequisite_evidence", []) or []
            }
            for item in m.get("prerequisite_evidence", []) or []:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if key not in known_evidence:
                    course.setdefault("prerequisite_evidence", []).append(item)
                    known_evidence.add(key)
            for item in m.get("unresolved_prerequisites", []) or []:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if key not in unresolved_keys:
                    unresolved.append(item)
                    unresolved_keys.add(key)

    course_rows = [courses[code] for code in sorted(courses)]
    courses_path = OUTPUT_DIR / "courses.json"
    _atomic_write_json(courses_path, course_rows)
    _atomic_write_json(OUTPUT_DIR / "unresolved_prerequisites.json", unresolved)
    # M3.3：把大纲画像并进 courses.json（不覆盖培养方案值，冲突只记录）。
    _merge_syllabus_profiles_into_courses(courses_path)

def run_graph_stage(*, concept_stage_succeeded: bool | None = None) -> None:
    """Write a deterministic graph manifest; Neo4j writing remains optional."""
    if concept_stage_succeeded is False:
        print(
            "[graph] skipped: the concept stage did not produce a current, "
            "verified snapshot"
        )
        return
    required_concept_artifacts = (
        config.concept_registry_path,
        config.concept_verified_edge_path,
    )
    missing_artifacts = [
        path.name for path in required_concept_artifacts if not path.exists()
    ]
    if missing_artifacts:
        print(
            "[graph] skipped: authoritative concept artifacts are missing: "
            + ", ".join(missing_artifacts)
        )
        return
    chunks = json.loads(CHUNKS_OUTPUT_PATH.read_text(encoding="utf-8"))
    concept_path = config.concept_registry_path
    concepts = json.loads(concept_path.read_text(encoding="utf-8")) if concept_path.exists() else []
    concept_dependencies = (
        json.loads(config.concept_verified_edge_path.read_text(encoding="utf-8"))
        if config.concept_verified_edge_path.exists()
        else []
    )
    courses = json.loads((OUTPUT_DIR / "courses.json").read_text(encoding="utf-8")) if (OUTPUT_DIR / "courses.json").exists() else []
    graph = build_graph_records(
        courses,
        concepts,
        chunks,
        concept_dependencies=concept_dependencies,
        verified_min_confidence=config.concept_verified_min_confidence,
        include_chunk_nodes=config.graph_include_chunk_nodes,
    )
    neo4j_status = "unavailable"
    build_id = None
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=2,
            connection_acquisition_timeout=2,
            max_transaction_retry_time=1,
        )
        driver.verify_connectivity()
        try:
            write_summary = write_neo4j(
                driver,
                graph,
                courses,
                concepts,
                # Chunk text is served from Chroma; mirroring it into Neo4j
                # duplicates the whole corpus for nodes no query reads.
                chunks if config.graph_include_chunk_nodes else [],
                reset_existing=config.reset_concept_subgraph,
            )
            build_id = write_summary["build_id"]
            neo4j_status = "written"
        finally:
            driver.close()
    # AuthError derives from Neo4jError, so the reachability cases have to be
    # caught first or they would be reported as failed writes.
    except (ServiceUnavailable, AuthError, OSError) as exc:
        print(f"[graph] Neo4j unavailable: {exc}")
    except Neo4jError as exc:
        # The server answered and rejected the write. Calling that "unavailable"
        # sends anyone debugging it after the connection instead of the data.
        neo4j_status = "write_failed"
        code = getattr(exc, "code", None) or "unknown"
        print(f"[graph] Neo4j rejected the graph write ({code}): {exc}")
        if "ConstraintValidationFailed" in code:
            print(
                "[graph] a node this pipeline does not recognise as its own "
                "already holds that key; check for nodes carrying the shared "
                "Course/Concept/Chunk labels without a managed_by stamp"
            )
    manifest = {
        "build_id": build_id,
        "courses": sorted(
            {
                str(c.get("course_code"))
                for c in courses
                if c.get("course_code")
            }
        ),
        "concepts": [
            concept_id
            for concept in concepts
            if (
                concept_id := concept.get("id")
                or concept.get("concept_id")
            )
        ],
        "concept_dependencies": sum(
            1
            for edge in graph["edges"]
            if edge.get("type") in CONCEPT_DEPENDENCY_TYPES
        ),
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "neo4j": neo4j_status,
        "concept_extraction_model": config.text_model,
        "concept_normalization_model": (
            "hash"
            if config.embedding_provider == "hash"
            else config.concept_normalization_model
        ),
        "concept_retrieval_model": (
            "hash"
            if config.embedding_provider == "hash"
            else config.concept_retrieval_model
        ),
        "embedding_provider": config.embedding_provider,
        "concept_llm_vote_count": config.concept_llm_vote_count,
        "concept_max_verification_candidates": (
            config.concept_max_verification_candidates
        ),
        "concept_min_extraction_coverage": config.concept_min_extraction_coverage,
        "verified_min_confidence": config.concept_verified_min_confidence,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    config.graph_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.graph_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[graph] manifest written ({len(manifest['courses'])} courses)")

def run_baseline_stage() -> None:
    """Freeze a reproducible baseline without requiring external services."""
    baseline = OUTPUT_DIR / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in OUTPUT_DIR.glob("*"):
        if src.is_file() and src.name != "baseline.json":
            dst = baseline / src.name
            shutil.copy2(src, dst)
            copied.append(src.name)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "outputs": copied, "chunk_count": 0, "api_calls": 0}
    if CHUNKS_OUTPUT_PATH.exists():
        try:
            summary["chunk_count"] = len(
                json.loads(CHUNKS_OUTPUT_PATH.read_text(encoding="utf-8"))
            )
        except json.JSONDecodeError:
            pass
    (baseline / "baseline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[baseline] saved to {baseline}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_stage_map(incremental: bool = False, force: bool = False) -> dict[str, list[PipelineStep]]:
    """Build the stage-to-functions mapping.

    When *incremental* is True, the parsing stage acts as a patch rather
    than a full re-parse.
    """
    all_stage_state: dict[str, bool | None] = {"concept_succeeded": None}

    def run_all_concept_stage() -> None:
        all_stage_state["concept_succeeded"] = run_concept_stage()

    def run_all_graph_stage() -> None:
        run_graph_stage(
            concept_stage_succeeded=all_stage_state["concept_succeeded"]
        )

    def run_all_embed_stage() -> None:
        if all_stage_state["concept_succeeded"] is False:
            print(
                "[embed] concept enrichment unavailable; building a base index "
                "from the current raw chunks"
            )
        run_embed_stage()

    return {
        "parse": [lambda: run_parse_stage(incremental=incremental, force=force)],
        "baseline": [run_baseline_stage],
        "concept": [run_concept_stage],
        "all": [
            lambda: run_parse_stage(incremental=incremental, force=force),
            run_all_concept_stage,
            run_all_graph_stage,
            run_all_embed_stage,
        ],
        "graph": [run_graph_stage],
        "embed": [run_embed_stage],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="zsttSystem offline data pipeline (v2.0)")
    parser.add_argument(
        "--stage",
        choices=["baseline", "parse", "concept", "graph", "embed", "all"],
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: only process changed syllabus files (parsing stage).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full reprocess even in incremental mode.",
    )
    return parser.parse_args()


def main() -> None:
    CHUNKS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    stage_map = build_stage_map(
        incremental=args.incremental,
        force=args.force,
    )

    for step in stage_map[args.stage]:
        step()


if __name__ == "__main__":
    main()
