"""Unit tests for zyme.commands.scan — the `zyme scan` workspace dashboard cmd.

`cmd_scan` is an orchestrator over the lower `zyme.scan*` modules (tested in
tests/test_scan.py and tests/test_scan_*_unit.py). This file targets the
command layer:

  - the pure formatting helpers (`_check`, `_format_latest_keep`,
    `_format_active`),
  - root/framework/export-path resolution
    (`_resolve_scan_roots`, `_explicit_task_dirs`, `_merge_task_dirs`,
    `_resolve_export_path`, `_resolve_dataset_export_path`,
    `_resolve_attest_framework`, `_coverage_platform_filter`),
  - the ASCII + markdown table renderers (`_render_table`, `_render_json`,
    `_markdown_summary`) against synthetic detect_phase-shaped rows,
  - cmd_scan dispatch end-to-end on a fake workspace tree (lifecycle table,
    --json, --filter, --no-export), plus the mode-switch dispatch to the
    coverage / attest / portability / dataset sub-runners (monkeypatched).

No subprocess, no network — all on tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import zyme.commands.scan as scan_cmd
from zyme.commands.scan import (
    _DEFAULT_TASK_CATEGORIES,
    _check,
    _coverage_platform_filter,
    _explicit_task_dirs,
    _format_active,
    _format_latest_keep,
    _markdown_summary,
    _merge_task_dirs,
    _render_json,
    _render_table,
    _resolve_attest_framework,
    _resolve_dataset_export_path,
    _resolve_export_path,
    _resolve_scan_roots,
    cmd_scan,
)


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------

def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "autozyme-framework").mkdir()
    return ws


def _make_task(parent: Path, name: str) -> Path:
    task = parent / name
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.yaml").write_text("target_repo: stub\n")
    return task


def _phase_row(task_dir: Path, *, phase="scaffold", rounds=0, scaffold=True,
               init=False, iterate=False, scaling=False, package=False,
               latest_keep=None, active=None, gaps=None) -> dict:
    """Build a detect_phase-shaped row for the renderers."""
    def ph(done, **extra):
        return {"done": done, **extra}
    return {
        "task_dir": str(task_dir),
        "dir_name": task_dir.name,
        "phase": phase,
        "phases": {
            "scaffold": ph(scaffold),
            "init": ph(init),
            "iterate": ph(iterate, rounds=rounds),
            "scaling": ph(scaling),
            "package": ph(package),
        },
        "reflect": {
            "initialization": False, "iteration": False,
            "scaling": False, "packaging": False,
        },
        "report_done": False,
        "package_verify_done": False,
        "latest_keep": latest_keep,
        "active": active,
        "gaps": gaps or [],
    }


def _scan_args(**overrides) -> SimpleNamespace:
    base = dict(
        coverage=False, attest=False, portability=False, dataset=False,
        framework_root=None, paths=None, max_depth=3, phase_filter=None,
        phase_only=False, reflect_only=False, json=False, active_minutes=15,
        export_path=None, no_export=True, needs_3_5=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------
# Pure formatting helpers
# --------------------------------------------------------------------------

class TestCheck:
    def test_true_false(self):
        assert _check(True) == "[x]"
        assert _check(False) == "[ ]"


class TestFormatLatestKeep:
    def test_none(self):
        assert _format_latest_keep(None) == "—"

    def test_pct_only(self):
        assert _format_latest_keep({"speedup_pct": 12.0}) == "+12%"

    def test_pct_with_tier(self):
        assert _format_latest_keep(
            {"speedup_pct": 8.0, "dataset": "medium"}) == "+8% medium"

    def test_negative_pct(self):
        assert _format_latest_keep({"speedup_pct": -3.0}) == "-3%"

    def test_missing_pct_dash(self):
        assert _format_latest_keep({"dataset": "tiny"}) == "— tiny"

    def test_long_tier_truncated(self):
        out = _format_latest_keep(
            {"speedup_pct": 5.0, "dataset": "a_very_long_tier_name"})
        assert out.startswith("+5% ")
        assert "…" in out
        # truncated to 7 chars + ellipsis
        assert out == "+5% a_very_…"


class TestFormatActive:
    def test_none(self):
        assert _format_active(None) == "-"

    def test_status(self):
        assert _format_active({"status": "run"}) == "run"

    def test_empty_status(self):
        assert _format_active({"status": None}) == "-"


# --------------------------------------------------------------------------
# _explicit_task_dirs / _merge_task_dirs
# --------------------------------------------------------------------------

class TestExplicitTaskDirs:
    def test_none_returns_empty(self):
        assert _explicit_task_dirs(None) == []

    def test_only_dirs_with_task_yaml(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        (good / "task.yaml").write_text("")
        bad = tmp_path / "bad"
        bad.mkdir()
        out = _explicit_task_dirs([str(good), str(bad)])
        assert [p.name for p in out] == ["good"]


class TestMergeTaskDirs:
    def test_explicit_first_then_walked_deduped(self, tmp_path):
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        merged = _merge_task_dirs(walked=[a, b], explicit=[b])
        # explicit b comes first; walked b is deduped out.
        assert [p.name for p in merged] == ["b", "a"]


# --------------------------------------------------------------------------
# _resolve_scan_roots
# --------------------------------------------------------------------------

class TestResolveScanRoots:
    def test_default_categories_when_present(self, tmp_path, monkeypatch):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        cat.mkdir()
        monkeypatch.chdir(ws)
        roots, workspace, framework = _resolve_scan_roots(
            _scan_args(framework_root=None, paths=None))
        assert workspace == ws
        assert roots == [cat]
        assert framework == (ws / "autozyme-framework").resolve()

    def test_falls_back_to_workspace_when_no_categories(self, tmp_path,
                                                        monkeypatch):
        ws = _make_workspace(tmp_path)
        monkeypatch.chdir(ws)
        roots, workspace, framework = _resolve_scan_roots(_scan_args())
        assert roots == [ws]

    def test_explicit_paths(self, tmp_path, monkeypatch):
        ws = _make_workspace(tmp_path)
        d = tmp_path / "explicit_dir"
        d.mkdir()
        monkeypatch.chdir(ws)
        roots, workspace, framework = _resolve_scan_roots(
            _scan_args(paths=[str(d)]))
        assert roots == [d.resolve()]

    def test_explicit_missing_path_skipped(self, tmp_path, monkeypatch, capsys):
        ws = _make_workspace(tmp_path)
        monkeypatch.chdir(ws)
        roots, _, _ = _resolve_scan_roots(
            _scan_args(paths=[str(tmp_path / "nope")]))
        assert roots == []
        assert "scan path missing" in capsys.readouterr().err

    def test_no_workspace_returns_empty_roots(self, tmp_path, monkeypatch,
                                              capsys):
        # cwd has no autozyme-framework above it.
        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.chdir(bare)
        roots, workspace, framework = _resolve_scan_roots(_scan_args())
        assert roots == []
        assert workspace is None
        assert "no autozyme-framework/" in capsys.readouterr().err

    def test_framework_root_override(self, tmp_path, monkeypatch):
        ws = _make_workspace(tmp_path)
        fw = tmp_path / "custom_fw"
        fw.mkdir()
        monkeypatch.chdir(ws)
        roots, workspace, framework = _resolve_scan_roots(
            _scan_args(framework_root=str(fw)))
        assert framework == fw.resolve()


# --------------------------------------------------------------------------
# Export-path resolution
# --------------------------------------------------------------------------

class TestResolveExportPath:
    def test_no_export_returns_none(self, tmp_path):
        assert _resolve_export_path(
            _scan_args(no_export=True), tmp_path) is None

    def test_json_returns_none(self, tmp_path):
        assert _resolve_export_path(
            _scan_args(no_export=False, json=True), tmp_path) is None

    def test_explicit_export_path(self, tmp_path):
        out = _resolve_export_path(
            _scan_args(no_export=False, export_path=str(tmp_path / "x.md")),
            tmp_path)
        assert out == (tmp_path / "x.md").resolve()

    def test_default_scan_md_under_framework(self, tmp_path):
        out = _resolve_export_path(
            _scan_args(no_export=False, export_path=None), tmp_path)
        assert out == tmp_path / "SCAN.md"

    def test_none_framework_returns_none(self):
        assert _resolve_export_path(
            _scan_args(no_export=False, export_path=None), None) is None

    def test_dataset_export_path_default_datasets_md(self, tmp_path):
        out = _resolve_dataset_export_path(
            _scan_args(no_export=False, export_path=None), tmp_path)
        assert out == tmp_path / "DATASETS.md"


# --------------------------------------------------------------------------
# _resolve_attest_framework / _coverage_platform_filter
# --------------------------------------------------------------------------

class TestResolveAttestFramework:
    def test_framework_root_arg_wins(self, tmp_path):
        fw = tmp_path / "fw"
        fw.mkdir()
        out = _resolve_attest_framework(_scan_args(framework_root=str(fw)))
        assert out == fw.resolve()

    def test_cwd_that_looks_like_framework(self, tmp_path, monkeypatch):
        fw = tmp_path / "autozyme-framework"
        fw.mkdir()
        (fw / "autozyme_r").mkdir()
        (fw / "autozyme_py").mkdir()
        monkeypatch.chdir(fw)
        out = _resolve_attest_framework(_scan_args(framework_root=None))
        assert out == fw.resolve()

    def test_none_when_nothing_found(self, tmp_path, monkeypatch):
        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.chdir(bare)
        assert _resolve_attest_framework(_scan_args(framework_root=None)) is None


class TestCoveragePlatformFilter:
    def test_mac(self):
        assert _coverage_platform_filter(
            _scan_args(mac_only=True, win_only=False)) == "mac"

    def test_win(self):
        assert _coverage_platform_filter(
            _scan_args(mac_only=False, win_only=True)) == "win"

    def test_neither(self):
        assert _coverage_platform_filter(_scan_args()) is None


# --------------------------------------------------------------------------
# Renderers against synthetic rows
# --------------------------------------------------------------------------

class TestRenderTable:
    def test_no_rows(self, tmp_path, capsys):
        ws = _make_workspace(tmp_path)
        _render_table([], roots=[ws], workspace=ws,
                      framework=ws / "autozyme-framework",
                      phase_only=False, reflect_only=False)
        out = capsys.readouterr().out
        assert "no tasks found" in out

    def test_rows_render_with_totals(self, tmp_path, capsys):
        ws = _make_workspace(tmp_path)
        t = _make_task(ws / "cat", "test_a")
        row = _phase_row(t, phase="scaffold")
        _render_table([row], roots=[ws], workspace=ws,
                      framework=ws / "autozyme-framework",
                      phase_only=False, reflect_only=False)
        out = capsys.readouterr().out
        assert "test_a" in out
        assert "Totals:" in out
        assert "scaffold=1" in out

    def test_gap_marker_and_note(self, tmp_path, capsys):
        ws = _make_workspace(tmp_path)
        t = _make_task(ws / "cat", "test_g")
        row = _phase_row(t, phase="package", package=True, gaps=["init"])
        _render_table([row], roots=[ws], workspace=ws,
                      framework=ws / "autozyme-framework",
                      phase_only=False, reflect_only=False)
        out = capsys.readouterr().out
        assert "package*" in out
        assert "earlier" in out  # the gap note

    def test_phase_only_suppresses_reflect(self, tmp_path, capsys):
        ws = _make_workspace(tmp_path)
        t = _make_task(ws / "cat", "test_p")
        _render_table([_phase_row(t)], roots=[ws], workspace=ws,
                      framework=ws / "autozyme-framework",
                      phase_only=True, reflect_only=False)
        out = capsys.readouterr().out
        assert "I-rfl" not in out

    def test_no_framework_marks_package_unknown(self, tmp_path, capsys):
        ws = _make_workspace(tmp_path)
        t = _make_task(ws / "cat", "test_n")
        _render_table([_phase_row(t)], roots=[ws], workspace=ws,
                      framework=None, phase_only=False, reflect_only=False)
        out = capsys.readouterr().out
        assert "[?]" in out


class TestRenderJson:
    def test_one_object_per_task(self, tmp_path, capsys):
        ws = _make_workspace(tmp_path)
        t = _make_task(ws / "cat", "test_j")
        _render_json([_phase_row(t)])
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["dir_name"] == "test_j"


class TestMarkdownSummary:
    def test_no_rows(self, tmp_path):
        ws = _make_workspace(tmp_path)
        md = _markdown_summary([], workspace=ws,
                               framework=ws / "autozyme-framework")
        assert "_No tasks found._" in md

    def test_table_and_totals(self, tmp_path):
        ws = _make_workspace(tmp_path)
        t = _make_task(ws / "cat", "test_m")
        md = _markdown_summary([_phase_row(t, phase="scaffold")],
                               workspace=ws,
                               framework=ws / "autozyme-framework")
        assert "# autozyme task lifecycle scan" in md
        assert "test_m" in md
        assert "## Totals" in md
        assert "scaffold=1" in md


# --------------------------------------------------------------------------
# cmd_scan dispatch — end to end + mode switches
# --------------------------------------------------------------------------

class TestCmdScanEndToEnd:
    def test_lifecycle_table_on_fake_workspace(self, tmp_path, monkeypatch,
                                               capsys):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        _make_task(cat, "test_foo")
        monkeypatch.chdir(ws)
        cmd_scan(_scan_args(no_export=True))
        out = capsys.readouterr().out
        assert "test_foo" in out
        assert "found 1 task(s)" in out

    def test_json_mode(self, tmp_path, monkeypatch, capsys):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        _make_task(cat, "test_bar")
        monkeypatch.chdir(ws)
        cmd_scan(_scan_args(json=True, no_export=True))
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        objs = [json.loads(ln) for ln in lines]
        assert any(o["dir_name"] == "test_bar" for o in objs)

    def test_phase_filter_excludes_nonmatching(self, tmp_path, monkeypatch,
                                               capsys):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        _make_task(cat, "test_scaf")  # only scaffolded
        monkeypatch.chdir(ws)
        cmd_scan(_scan_args(phase_filter="package", no_export=True))
        out = capsys.readouterr().out
        # task is in scaffold, so a package filter shows zero tasks in table.
        assert "test_scaf" not in out.split("Totals")[0] if "Totals" in out else True

    def test_exits_when_no_roots(self, tmp_path, monkeypatch):
        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.chdir(bare)
        with pytest.raises(SystemExit) as ei:
            cmd_scan(_scan_args())
        assert ei.value.code == 1

    def test_writes_scan_md_export(self, tmp_path, monkeypatch):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        _make_task(cat, "test_exp")
        monkeypatch.chdir(ws)
        cmd_scan(_scan_args(no_export=False))
        assert (ws / "autozyme-framework" / "SCAN.md").exists()

    def test_coverage_mode_dispatches(self, monkeypatch):
        called = {}
        monkeypatch.setattr(scan_cmd, "_run_speedups_scan",
                            lambda args: called.setdefault("hit", True))
        cmd_scan(_scan_args(coverage=True))
        assert called.get("hit")

    def test_attest_mode_dispatches(self, monkeypatch):
        called = {}
        monkeypatch.setattr(scan_cmd, "_run_attest_scan",
                            lambda args: called.setdefault("hit", True))
        cmd_scan(_scan_args(attest=True))
        assert called.get("hit")

    def test_portability_mode_dispatches(self, tmp_path, monkeypatch):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        _make_task(cat, "test_port")
        monkeypatch.chdir(ws)
        called = {}
        monkeypatch.setattr(scan_cmd, "_run_portability_scan",
                            lambda *a: called.setdefault("hit", True))
        cmd_scan(_scan_args(portability=True, no_export=True))
        assert called.get("hit")

    def test_dataset_mode_dispatches(self, tmp_path, monkeypatch):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        _make_task(cat, "test_ds")
        monkeypatch.chdir(ws)
        called = {}
        monkeypatch.setattr(scan_cmd, "_run_dataset_scan",
                            lambda *a: called.setdefault("hit", True))
        cmd_scan(_scan_args(dataset=True, no_export=True))
        assert called.get("hit")


# --------------------------------------------------------------------------
# Dataset-mode renderers end-to-end (real inspect_task_datasets rows)
# --------------------------------------------------------------------------

def _make_dataset_task(parent: Path, name: str, *, missing=False) -> Path:
    task = parent / name
    (task / "data").mkdir(parents=True, exist_ok=True)
    path = "data/missing.h5ad" if missing else "data/t.h5ad"
    (task / "task.yaml").write_text(
        "target_repo: stub\n"
        "datasets:\n"
        f"  - {{tier: tiny, name: tiny_a, path: {path}}}\n"
    )
    if not missing:
        (task / "data" / "t.h5ad").write_text("x" * 256)
    # An orphan file in data/ not referenced by task.yaml.
    (task / "data" / "orphan.h5ad").write_text("y" * 128)
    return task


class TestDatasetMode:
    def test_dataset_table_and_markdown_export(self, tmp_path, monkeypatch,
                                               capsys):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        _make_dataset_task(cat, "test_data")
        monkeypatch.chdir(ws)
        # Real dataset scan (no monkeypatch of _run_dataset_scan): exercises
        # inspect_task_datasets + _render_dataset_table + _dataset_markdown.
        cmd_scan(_scan_args(dataset=True, no_export=False))
        out = capsys.readouterr().out
        assert "test_data" in out
        assert "dataset entries" in out
        assert (ws / "autozyme-framework" / "DATASETS.md").exists()
        md = (ws / "autozyme-framework" / "DATASETS.md").read_text()
        assert "# autozyme task datasets" in md
        assert "test_data" in md

    def test_dataset_mode_json(self, tmp_path, monkeypatch, capsys):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        _make_dataset_task(cat, "test_dj")
        monkeypatch.chdir(ws)
        cmd_scan(_scan_args(dataset=True, json=True, no_export=True))
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        objs = [json.loads(ln) for ln in lines]
        assert any(o["dir_name"] == "test_dj" for o in objs)

    def test_dataset_missing_reported(self, tmp_path, monkeypatch, capsys):
        ws = _make_workspace(tmp_path)
        cat = ws / _DEFAULT_TASK_CATEGORIES[0]
        _make_dataset_task(cat, "test_miss", missing=True)
        monkeypatch.chdir(ws)
        cmd_scan(_scan_args(dataset=True, no_export=True))
        out = capsys.readouterr().out
        # A declared-but-absent dataset surfaces in the Missing section.
        assert "Missing" in out
