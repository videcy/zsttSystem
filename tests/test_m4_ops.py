from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import run_eval
from scripts import export_concept_mermaid as mermaid_export

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _concept(concept_id: str, course_code: str) -> dict[str, object]:
    return {
        "id": concept_id,
        "canonical_name": concept_id,
        "source_course_codes": [course_code],
    }


def _course(code: str, name: str, major: str) -> dict[str, object]:
    return {
        "course_code": code,
        "course_name": name,
        "offerings": [
            {
                "program_name": f"信息管理学院2025级{major}专业培养方案",
                "program_type": "主修专业",
            }
        ],
    }


def test_aggregate_course_flows_dedupes_and_limits_top30() -> None:
    registry = []
    edges = []
    for index in range(35):
        source_course = f"S{index:02d}"
        target_course = f"T{index:02d}"
        for relation_index in range(index + 1):
            source_id = f"s{index}_{relation_index}"
            target_id = f"t{index}_{relation_index}"
            registry.extend(
                [
                    _concept(source_id, source_course),
                    _concept(target_id, target_course),
                ]
            )
            edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "requires": True,
                }
            )
    edges.append(dict(edges[0]))
    edges.append(
        {
            "source_id": "s00_0",
            "target_id": "t01_0",
            "requires": False,
        }
    )

    counts = mermaid_export.aggregate_course_flows(registry, edges)
    flows = mermaid_export.top_course_flows(
        counts,
        {f"S{index:02d}" for index in range(35)}
        | {f"T{index:02d}" for index in range(35)},
        limit=30,
    )

    assert counts[("S00", "T00")] == 1
    assert len(flows) == 30
    assert [flow.count for flow in flows] == list(range(35, 5, -1))
    assert all(counts[(flow.source, flow.target)] == flow.count for flow in flows)


def test_mermaid_export_uses_temp_json_and_selects_a_major(tmp_path: Path) -> None:
    registry = [
        _concept("lib-1", "IM101"),
        _concept("lib-2", "IM101"),
        _concept("lib-3", "IM201"),
        _concept("arc-1", "AR101"),
        _concept("arc-2", "AR201"),
    ]
    edges = [
        {"source_id": "lib-1", "target_id": "lib-3", "requires": True},
        {"source_id": "lib-2", "target_id": "lib-3", "requires": True},
        # Duplicate verdict rows must not inflate the course-pair count.
        {"source_id": "lib-1", "target_id": "lib-3", "requires": True},
        {"source_id": "lib-3", "target_id": "lib-1", "requires": False},
        {"source_id": "arc-1", "target_id": "arc-2", "requires": True},
    ]
    courses = [
        _course("IM101", "信息检索", "图书馆学"),
        _course("IM201", "信息组织", "图书馆学"),
        _course("AR101", "档案学基础", "档案学"),
        _course("AR201", "档案管理", "档案学"),
    ]
    registry_path = _write_json(tmp_path / "registry.json", registry)
    edges_path = _write_json(tmp_path / "edges.json", edges)
    courses_path = _write_json(tmp_path / "courses.json", courses)

    output_dir = tmp_path / "one-major"
    exit_code = mermaid_export.main(
        [
            "--registry",
            str(registry_path),
            "--edges",
            str(edges_path),
            "--courses",
            str(courses_path),
            "--program",
            "图书馆学",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    report = (output_dir / "01-图书馆学.md").read_text(encoding="utf-8")
    assert "```mermaid" in report
    assert "IM101" in report and "信息检索" in report
    assert "IM201" in report and "信息组织" in report
    assert "-->|2|" in report
    assert "AR101" not in report

    all_dir = tmp_path / "all-majors"
    assert (
        mermaid_export.main(
            [
                "--registry",
                str(registry_path),
                "--edges",
                str(edges_path),
                "--courses",
                str(courses_path),
                "--all",
                "--output-dir",
                str(all_dir),
            ]
        )
        == 0
    )
    assert len(list(all_dir.glob("*.md"))) == 2


def test_compose_persists_eval_data_and_uses_stage_flag() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "./eval/datasets:/app/eval/datasets" in compose
    assert 'command: ["python", "run_pipeline.py", "--stage", "all"]' in compose
    assert "GIT_COMMIT: ${GIT_COMMIT:-unknown}" in compose


def test_compose_writes_eval_reports_to_mounted_directory() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "EVAL_REPORT_DIR: outputs/eval-reports" in compose


def test_dockerfile_exports_build_commit() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG GIT_COMMIT=unknown" in dockerfile
    assert "GIT_COMMIT=${GIT_COMMIT}" in dockerfile


def test_git_commit_prefers_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_eval.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("git should not run when GIT_COMMIT is set"),
    )

    assert (
        run_eval.resolve_git_commit(
            env={"GIT_COMMIT": "from-build"}, repo_root=tmp_path
        )
        == "from-build"
    )


def test_git_commit_is_unknown_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(run_eval.subprocess, "run", missing_git)

    assert run_eval.resolve_git_commit(env={}, repo_root=tmp_path) == "unknown"


def test_git_commit_reads_repository_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_git(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs["cwd"]))
        return SimpleNamespace(stdout="0123456789abcdef\n")

    monkeypatch.setattr(run_eval.subprocess, "run", fake_git)

    assert (
        run_eval.resolve_git_commit(env={}, repo_root=tmp_path)
        == "0123456789abcdef"
    )
    assert calls == [(["git", "rev-parse", "HEAD"], tmp_path)]


def test_eval_report_environment_records_git_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_eval, "resolve_git_commit", lambda: "deadbeef")

    report = run_eval.evaluate([], stages=())

    assert report["environment"]["git_commit"] == "deadbeef"
