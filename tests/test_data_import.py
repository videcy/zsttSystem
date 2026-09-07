from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src.online_service.data_import import (
    ImportRejected,
    classify,
    inventory,
    safe_filename,
    store_upload,
)

# Minimal valid ZIP header: docx and xlsx are both ZIP containers.
DOCX_BYTES = b"PK\x03\x04" + b"0" * 64


# ---------------------------------------------------------------------------
# Filename handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("IM104档案学概论.docx", "IM104档案学概论.docx"),
        ("../../etc/passwd.docx", "passwd.docx"),
        (r"C:\Windows\System32\evil.docx", "evil.docx"),
        ("plans/2026/计划.xlsx", "计划.xlsx"),
        ("with spaces 与中文.docx", "with spaces 与中文.docx"),
    ],
)
def test_safe_filename_keeps_only_a_plain_name(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "..", "/", "///"])
def test_safe_filename_rejects_empty_and_traversal_only_names(raw: str) -> None:
    with pytest.raises(ImportRejected):
        safe_filename(raw)


def test_classify_maps_extensions_to_pipeline_inputs() -> None:
    assert classify("a.docx") == "syllabus"
    assert classify("b.XLSX") == "training_plan"
    with pytest.raises(ImportRejected, match="只接受"):
        classify("payload.exe")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "syllabus_dir": tmp_path / "syllabi",
        "training_plan_dir": tmp_path / "plans",
    }


def test_store_upload_routes_by_extension(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)

    syllabus = store_upload("课程大纲.docx", DOCX_BYTES, **dirs, max_bytes=1000)
    plan = store_upload("培养方案.xlsx", DOCX_BYTES, **dirs, max_bytes=1000)

    assert (dirs["syllabus_dir"] / "课程大纲.docx").read_bytes() == DOCX_BYTES
    assert (dirs["training_plan_dir"] / "培养方案.xlsx").exists()
    assert syllabus.kind == "syllabus" and plan.kind == "training_plan"
    assert syllabus.replaced is False


def test_store_upload_reports_replacement(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    store_upload("a.docx", DOCX_BYTES, **dirs, max_bytes=1000)

    second = store_upload("a.docx", DOCX_BYTES, **dirs, max_bytes=1000)

    assert second.replaced is True


def test_traversal_lands_inside_the_target_directory(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)

    store_upload("../../escaped.docx", DOCX_BYTES, **dirs, max_bytes=1000)

    assert (dirs["syllabus_dir"] / "escaped.docx").exists()
    assert not (tmp_path.parent / "escaped.docx").exists()


@pytest.mark.parametrize(
    ("filename", "payload", "match"),
    [
        ("a.docx", b"", "空"),
        ("a.docx", b"0" * 5000, "过大"),
        ("a.exe", DOCX_BYTES, "只接受"),
        ("a.docx", b"not a zip file", "ZIP"),
        ("~$draft.docx", DOCX_BYTES, "临时文件"),
    ],
)
def test_store_upload_rejections(
    tmp_path: Path,
    filename: str,
    payload: bytes,
    match: str,
) -> None:
    with pytest.raises(ImportRejected, match=match):
        store_upload(filename, payload, **_dirs(tmp_path), max_bytes=1000)


def test_inventory_lists_what_the_pipeline_would_parse(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    store_upload("b.docx", DOCX_BYTES, **dirs, max_bytes=1000)
    store_upload("a.docx", DOCX_BYTES, **dirs, max_bytes=1000)
    store_upload("plan.xlsx", DOCX_BYTES, **dirs, max_bytes=1000)
    (dirs["syllabus_dir"] / "~$tmp.docx").write_bytes(DOCX_BYTES)

    listing = inventory(**dirs)

    assert [item["filename"] for item in listing["syllabi"]] == ["a.docx", "b.docx"]
    assert listing["counts"] == {"syllabi": 2, "training_plans": 1}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def admin_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    from src.config import config

    monkeypatch.setattr(main_module, "QUERY_LOG_PATH", tmp_path / "query.jsonl")
    monkeypatch.setattr(main_module, "FEEDBACK_LOG_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(main_module, "create_deepseek_client", lambda: None)
    monkeypatch.setattr(main_module, "ChromaRetriever", lambda *_a, **_k: None)
    monkeypatch.setattr(main_module, "QueryRouter", lambda *_a, **_k: object())
    monkeypatch.setattr(
        main_module.GraphDatabase,
        "driver",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("no neo4j")),
    )
    monkeypatch.setattr(
        type(config), "syllabus_dir", property(lambda _s: tmp_path / "syllabi")
    )
    monkeypatch.setattr(
        type(config), "training_plan_dir", property(lambda _s: tmp_path / "plans")
    )
    main_module._REQUEST_TIMES.clear()
    main_module._REINDEX_STATE.clear()
    main_module._REINDEX_STATE["status"] = "idle"
    with TestClient(main_module.app) as client:
        yield client
    main_module._REQUEST_TIMES.clear()


def _upload(name: str = "课程.docx", payload: bytes = DOCX_BYTES):
    return {"files": (name, io.BytesIO(payload), "application/octet-stream")}


def test_admin_endpoints_are_disabled_without_a_token(
    admin_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import config

    monkeypatch.setattr(type(config), "admin_token", property(lambda _s: ""))

    for response in (
        admin_client.get("/admin/data"),
        admin_client.post("/admin/data/import", files=_upload()),
        admin_client.post("/admin/data/reindex"),
    ):
        assert response.status_code == 503
        assert "ADMIN_TOKEN" in response.json()["detail"]


def test_admin_rejects_a_wrong_token(
    admin_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import config

    monkeypatch.setattr(type(config), "admin_token", property(lambda _s: "right"))

    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "right"}):
        assert admin_client.get("/admin/data", headers=headers).status_code == 401


def test_import_stores_accepted_files_and_reports_rejections(
    admin_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.config import config

    monkeypatch.setattr(type(config), "admin_token", property(lambda _s: "secret"))

    response = admin_client.post(
        "/admin/data/import",
        headers={"Authorization": "Bearer secret"},
        files=[
            ("files", ("IM104档案学概论.docx", io.BytesIO(DOCX_BYTES), "application/octet-stream")),
            ("files", ("payload.exe", io.BytesIO(DOCX_BYTES), "application/octet-stream")),
        ],
    )

    body = response.json()
    assert response.status_code == 200
    assert [item["filename"] for item in body["stored"]] == ["IM104档案学概论.docx"]
    assert body["rejected"][0]["filename"] == "payload.exe"
    assert (tmp_path / "syllabi" / "IM104档案学概论.docx").exists()

    listing = admin_client.get(
        "/admin/data", headers={"Authorization": "Bearer secret"}
    ).json()
    assert listing["counts"]["syllabi"] == 1


def test_reindex_refuses_to_start_twice(
    admin_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import config

    monkeypatch.setattr(type(config), "admin_token", property(lambda _s: "secret"))
    main_module._REINDEX_STATE["status"] = "running"

    response = admin_client.post(
        "/admin/data/reindex", headers={"Authorization": "Bearer secret"}
    )

    assert response.status_code == 409


def test_reindex_reports_failure_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken pipeline surfaces through the status endpoint, not a 500."""
    import sys
    import types

    stub = types.ModuleType("run_pipeline")
    stub.run_parse_stage = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("boom")
    )
    stub.run_embed_stage = lambda: None
    monkeypatch.setitem(sys.modules, "run_pipeline", stub)

    main_module._run_reindex(main_module.app)

    assert main_module._REINDEX_STATE["status"] == "failed"
    assert "boom" in main_module._REINDEX_STATE["detail"]
    assert main_module._REINDEX_STATE["stages"] == []
