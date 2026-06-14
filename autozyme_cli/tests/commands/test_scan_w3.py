"""Wave-3 coverage for zyme.commands.scan — the reachable render/runner
branches that tests/commands/test_scan_cmd.py leaves uncovered.

test_scan_cmd.py covers the lifecycle table, json/filter/export dispatch,
the format helpers, root/export resolution, and dataset mode end-to-end on a
single task with one orphan. It does NOT cover:

  - the shared / external dataset rosters, the orphans (venv/dir kind tails),
    and the missing section in _render_dataset_table and _dataset_markdown
    (we feed synthetic aggregate rows via monkeypatch),
  - the framework-display ValueError fallback when framework is not under
    workspace (in _render_dataset_table and _dataset_markdown),
  - the _run_attest_scan / _run_speedups_scan / _run_portability_scan
    sub-runners (audit functions monkeypatched to synthetic records — no
    real Rscript / patch tree),
  - _resolve_dataset_export_path no-export / no-framework branches.

All monkeypatched at the scan_cmd module boundary; no subprocess, no network.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import zyme.commands.scan as scan_cmd
from zyme.commands.scan import (
    _dataset_markdown,
    _render_dataset_table,
    _resolve_dataset_export_path,
    _run_attest_scan,
    _run_dataset_scan,
    _run_portability_scan,
    _run_speedups_scan,
)


def _scan_args(**overrides) -> SimpleNamespace:
    base = dict(
        coverage=False, attest=False, portability=False, dataset=False,
        framework_root=None, paths=None, max_depth=3, phase_filter=None,
        phase_only=False, reflect_only=False, json=False, active_minutes=15,
        export_path=None, no_export=True, needs_3_5=False, detail=False,
        strict=False, mac_only=False, win_only=False, skip_experimental=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _dataset_row(name="test_x", category="cat", *, missing=0, orphan=0) -> dict:
    """A synthetic inspect_task_datasets-shaped row with a full totals block."""
    return {
        "dir_name": name,
        "category": category,
        "task_dir": f"/ws/{category}/{name}",
        "totals": {
            "n_entries": 2,
            "local_count": 1, "local_bytes": 1024,
            "shared_count": 1, "shared_bytes": 2048,
            "external_count": 1, "external_bytes": 4096,
            "missing_count": missing, "orphan_count": orphan,
            "orphan_bytes": 512 if orphan else 0,
            "task_dir_bytes": 8192,
            "subdir": {"data": 4096, "reference_outputs": 1024,
                       "upstream_repo": 2048, "other": 1024},
        },
        "datasets": [],
    }


# --------------------------------------------------------------------------
# _render_dataset_table — rosters + orphans + missing + framework display
# --------------------------------------------------------------------------

def _stub_aggregates(monkeypatch, *, shared=None, external=None,
                     orphans=None, missing=None):
    monkeypatch.setattr(scan_cmd, "aggregate_shared", lambda rows: shared or [])
    monkeypatch.setattr(scan_cmd, "aggregate_external", lambda rows: external or [])
    monkeypatch.setattr(scan_cmd, "collect_orphans", lambda rows: orphans or [])
    monkeypatch.setattr(scan_cmd, "collect_missing", lambda rows: missing or [])


class TestRenderDatasetTable:
    def test_full_rosters_and_sections(self, tmp_path, monkeypatch, capsys):
        rows = [_dataset_row("test_a", "core", orphan=2)]
        _stub_aggregates(
            monkeypatch,
            shared=[{"path": "/datasets/single_cell/s.h5ad", "size_bytes": 9999,
                     "references": [{"task": "test_a", "tier": "tiny"}]}],
            external=[{"path": "/home/u/cache.h5", "size_bytes": 8888,
                       "references": [{"task": "test_a", "tier": "medium"}]}],
            orphans=[
                {"task": "test_a", "path": "venv", "size_bytes": 7777,
                 "kind": "venv"},
                {"task": "test_a", "path": "extracted", "size_bytes": 6666,
                 "kind": "dir", "n_files": 19000},
            ],
            missing=[{"task": "test_a", "tier": "tiny", "path": "data/gone.h5ad"}],
        )
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        _render_dataset_table(rows, workspace=ws, framework=fw)
        out = capsys.readouterr().out
        assert "Shared datasets" in out
        assert "/datasets/single_cell/s.h5ad" in out
        assert "External datasets" in out
        assert "/home/u/cache.h5" in out
        assert "data/ orphans" in out
        assert "[Python venv]" in out
        assert "19000 files" in out
        assert "Missing (1)" in out
        assert "data/gone.h5ad" in out

    def test_no_rows_prints_marker(self, monkeypatch, capsys):
        _stub_aggregates(monkeypatch)
        _render_dataset_table([], workspace=None, framework=None)
        assert "(no tasks found)" in capsys.readouterr().out

    def test_framework_not_under_workspace_value_error(self, tmp_path,
                                                       monkeypatch, capsys):
        # framework on a different branch than workspace -> relative_to raises
        # ValueError -> the str(framework) fallback path runs.
        _stub_aggregates(monkeypatch)
        ws = tmp_path / "a" / "ws"
        ws.mkdir(parents=True)
        fw = tmp_path / "b" / "fw"
        fw.mkdir(parents=True)
        _render_dataset_table([_dataset_row()], workspace=ws, framework=fw)
        out = capsys.readouterr().out
        assert str(fw) in out

    def test_framework_only_no_workspace(self, tmp_path, monkeypatch, capsys):
        _stub_aggregates(monkeypatch)
        fw = tmp_path / "fw"
        _render_dataset_table([_dataset_row()], workspace=None, framework=fw)
        out = capsys.readouterr().out
        assert f"Framework: {fw}" in out

    def test_two_categories_print_group_headers(self, monkeypatch, capsys):
        # >1 distinct category -> group headers printed (line 404).
        _stub_aggregates(monkeypatch)
        rows = [_dataset_row("test_a", "core"), _dataset_row("test_b", "general")]
        _render_dataset_table(rows, workspace=None, framework=None)
        out = capsys.readouterr().out
        assert "core/" in out
        assert "general/" in out

    def test_long_task_name_truncated(self, monkeypatch, capsys):
        # name longer than name_w (max 32) -> truncated with "..." (line 414).
        _stub_aggregates(monkeypatch)
        long_name = "test_" + "x" * 40
        _render_dataset_table([_dataset_row(long_name, "cat")],
                              workspace=None, framework=None)
        out = capsys.readouterr().out
        assert "..." in out

    def test_orphans_truncated_past_cap(self, monkeypatch, capsys):
        _stub_aggregates(
            monkeypatch,
            orphans=[{"task": "t", "path": f"o{i}", "size_bytes": 10,
                      "kind": "dir", "n_files": 1} for i in range(25)],
        )
        _render_dataset_table([_dataset_row(orphan=25)], workspace=None,
                              framework=None)
        out = capsys.readouterr().out
        assert "and 5 more" in out  # 25 - cap(20)


# --------------------------------------------------------------------------
# _dataset_markdown — rosters + orphans + missing + framework display
# --------------------------------------------------------------------------

class TestDatasetMarkdown:
    def test_full_markdown_sections(self, monkeypatch):
        rows = [_dataset_row("test_b", "general", missing=1, orphan=1)]
        _stub_aggregates(
            monkeypatch,
            shared=[{"path": "/datasets/single_cell/x.h5ad", "size_bytes": 100,
                     "references": [{"category": "general", "task": "test_b",
                                     "tier": "tiny", "name": "n"}]}],
            external=[{"path": "/home/cache.h5", "size_bytes": 200,
                       "references": [{"category": "general", "task": "test_b",
                                       "tier": "med", "name": "n2"}]}],
            orphans=[{"category": "general", "task": "test_b", "path": "venv",
                      "size_bytes": 300, "kind": "venv"},
                     {"category": "general", "task": "test_b", "path": "dir",
                      "size_bytes": 400, "kind": "dir", "n_files": 7}],
            missing=[{"category": "general", "task": "test_b", "tier": "tiny",
                      "name": "n", "path": "data/gone.h5ad"}],
        )
        md = _dataset_markdown(rows, workspace=Path("/ws"),
                               framework=Path("/ws/autozyme-framework"))
        assert "# autozyme task datasets" in md
        assert "## Shared datasets" in md
        assert "## External datasets" in md
        assert "## `data/` orphans" in md
        assert "_(Python venv)_" in md
        assert "_(7 files)_" in md
        assert "## Missing datasets" in md
        assert "data/gone.h5ad" in md
        assert "## Totals" in md

    def test_markdown_no_rows(self, monkeypatch):
        _stub_aggregates(monkeypatch)
        md = _dataset_markdown([], workspace=None, framework=None)
        assert "_No tasks found._" in md

    def test_markdown_framework_value_error_fallback(self, monkeypatch):
        _stub_aggregates(monkeypatch)
        md = _dataset_markdown(
            [_dataset_row()],
            workspace=Path("/a/ws"), framework=Path("/b/fw"))
        # relative_to fails -> raw str(framework).
        assert "/b/fw" in md


# --------------------------------------------------------------------------
# _resolve_dataset_export_path branches
# --------------------------------------------------------------------------

class TestResolveDatasetExportPath:
    def test_no_export_returns_none(self, tmp_path):
        assert _resolve_dataset_export_path(
            _scan_args(no_export=True), tmp_path) is None

    def test_json_returns_none(self, tmp_path):
        assert _resolve_dataset_export_path(
            _scan_args(no_export=False, json=True), tmp_path) is None

    def test_explicit_path(self, tmp_path):
        out = _resolve_dataset_export_path(
            _scan_args(no_export=False, export_path=str(tmp_path / "D.md")),
            tmp_path)
        assert out == (tmp_path / "D.md").resolve()

    def test_none_framework_returns_none(self):
        assert _resolve_dataset_export_path(
            _scan_args(no_export=False, export_path=None), None) is None


# --------------------------------------------------------------------------
# _run_dataset_scan write-failure branch (OSError on export)
# --------------------------------------------------------------------------

def test_run_dataset_scan_export_oserror(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scan_cmd, "inspect_task_datasets",
                        lambda td: _dataset_row("test_e"))
    _stub_aggregates(monkeypatch)
    fw = tmp_path / "fw"
    fw.mkdir()

    real_write = Path.write_text

    def boom(self, *a, **k):
        if self.name == "DATASETS.md":
            raise OSError("disk full")
        return real_write(self, *a, **k)
    monkeypatch.setattr(Path, "write_text", boom)
    args = _scan_args(dataset=True, no_export=False)
    _run_dataset_scan(args, roots=[fw], workspace=None, framework=fw,
                      task_dirs=[tmp_path / "task"])
    assert "failed to write" in capsys.readouterr().err


# --------------------------------------------------------------------------
# _run_attest_scan
# --------------------------------------------------------------------------

class TestRunAttestScan:
    def test_no_framework_exits(self, monkeypatch):
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: None)
        with pytest.raises(SystemExit):
            _run_attest_scan(_scan_args(attest=True))

    def test_no_patches_exits(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        fw.mkdir()
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: fw)
        monkeypatch.setattr(scan_cmd, "audit_packages", lambda f: [])
        with pytest.raises(SystemExit):
            _run_attest_scan(_scan_args(attest=True))

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        fw = tmp_path / "fw"
        fw.mkdir()
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: fw)
        monkeypatch.setattr(scan_cmd, "audit_packages", lambda f: ["p1"])
        monkeypatch.setattr(scan_cmd, "render_attest_json",
                            lambda patches: [{"patch": "p1"}])
        _run_attest_scan(_scan_args(attest=True, json=True))
        out = capsys.readouterr().out
        assert json.loads(out.strip()) == {"patch": "p1"}

    def test_table_and_markdown_export(self, tmp_path, monkeypatch, capsys):
        fw = tmp_path / "fw"
        fw.mkdir()
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: fw)
        monkeypatch.setattr(scan_cmd, "audit_packages", lambda f: ["p1"])
        monkeypatch.setattr(scan_cmd, "render_attest_table",
                            lambda patches, f: "ATTEST TABLE")
        monkeypatch.setattr(scan_cmd, "render_attest_markdown",
                            lambda patches, f: "# attest md")
        _run_attest_scan(_scan_args(attest=True, no_export=False))
        out = capsys.readouterr().out
        assert "ATTEST TABLE" in out
        assert (fw / "ATTEST.md").read_text() == "# attest md"
        assert "Exported attest coverage" in out

    def test_export_oserror_reported(self, tmp_path, monkeypatch, capsys):
        fw = tmp_path / "fw"
        fw.mkdir()
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: fw)
        monkeypatch.setattr(scan_cmd, "audit_packages", lambda f: ["p1"])
        monkeypatch.setattr(scan_cmd, "render_attest_table",
                            lambda patches, f: "T")
        monkeypatch.setattr(scan_cmd, "render_attest_markdown",
                            lambda patches, f: "M")

        def boom(self, *a, **k):
            raise OSError("ro fs")
        monkeypatch.setattr(Path, "write_text", boom)
        _run_attest_scan(_scan_args(attest=True, no_export=False))
        assert "failed to write" in capsys.readouterr().err


# --------------------------------------------------------------------------
# _run_speedups_scan (coverage mode)
# --------------------------------------------------------------------------

class TestRunSpeedupsScan:
    def test_no_framework_exits(self, monkeypatch):
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: None)
        with pytest.raises(SystemExit):
            _run_speedups_scan(_scan_args(coverage=True))

    def test_no_audits_exits(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        fw.mkdir()
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: fw)
        monkeypatch.setattr(scan_cmd, "audit_speedups_all",
                            lambda f, **k: [])
        with pytest.raises(SystemExit):
            _run_speedups_scan(_scan_args(coverage=True))

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        fw = tmp_path / "fw"
        fw.mkdir()
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: fw)
        monkeypatch.setattr(scan_cmd, "audit_speedups_all",
                            lambda f, **k: ["a1"])
        monkeypatch.setattr(scan_cmd, "render_speedups_json",
                            lambda audits: [{"a": 1}])
        _run_speedups_scan(_scan_args(coverage=True, json=True))
        assert json.loads(capsys.readouterr().out.strip()) == {"a": 1}

    def test_table_export_and_strict_clean(self, tmp_path, monkeypatch, capsys):
        fw = tmp_path / "fw"
        fw.mkdir()
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: fw)
        monkeypatch.setattr(scan_cmd, "audit_speedups_all",
                            lambda f, **k: ["a1"])
        monkeypatch.setattr(scan_cmd, "render_speedups_table",
                            lambda audits, **k: "COV TABLE")
        monkeypatch.setattr(scan_cmd, "render_speedups_markdown",
                            lambda audits, **k: "# cov md")
        monkeypatch.setattr(scan_cmd, "speedups_has_warn_or_fail",
                            lambda audits: False)
        # strict=True but no warn/fail -> exit 0 (no SystemExit).
        _run_speedups_scan(_scan_args(coverage=True, no_export=False, strict=True))
        out = capsys.readouterr().out
        assert "COV TABLE" in out
        assert (fw / "COVERAGE.md").read_text() == "# cov md"

    def test_strict_with_warn_exits(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        fw.mkdir()
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: fw)
        monkeypatch.setattr(scan_cmd, "audit_speedups_all",
                            lambda f, **k: ["a1"])
        monkeypatch.setattr(scan_cmd, "render_speedups_table",
                            lambda audits, **k: "T")
        monkeypatch.setattr(scan_cmd, "render_speedups_markdown",
                            lambda audits, **k: "M")
        monkeypatch.setattr(scan_cmd, "speedups_has_warn_or_fail",
                            lambda audits: True)
        with pytest.raises(SystemExit):
            _run_speedups_scan(
                _scan_args(coverage=True, no_export=True, strict=True))

    def _stub_speedups(self, monkeypatch, fw):
        monkeypatch.setattr(scan_cmd, "_resolve_attest_framework",
                            lambda args: fw)
        monkeypatch.setattr(scan_cmd, "audit_speedups_all",
                            lambda f, **k: ["a1"])
        monkeypatch.setattr(scan_cmd, "render_speedups_table",
                            lambda audits, **k: "T")
        monkeypatch.setattr(scan_cmd, "render_speedups_markdown",
                            lambda audits, **k: "M")
        monkeypatch.setattr(scan_cmd, "speedups_has_warn_or_fail",
                            lambda audits: False)

    def test_mac_platform_filter_export_name(self, tmp_path, monkeypatch, capsys):
        fw = tmp_path / "fw"
        fw.mkdir()
        self._stub_speedups(monkeypatch, fw)
        _run_speedups_scan(
            _scan_args(coverage=True, no_export=False, mac_only=True))
        # platform filter mac -> COVERAGE.mac.md.
        assert (fw / "COVERAGE.mac.md").exists()

    def test_win_platform_filter_export_name(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        fw.mkdir()
        self._stub_speedups(monkeypatch, fw)
        _run_speedups_scan(
            _scan_args(coverage=True, no_export=False, win_only=True))
        assert (fw / "COVERAGE.win.md").exists()

    def test_explicit_export_path(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        fw.mkdir()
        self._stub_speedups(monkeypatch, fw)
        target = tmp_path / "custom" / "MY_COV.md"
        _run_speedups_scan(
            _scan_args(coverage=True, no_export=False,
                       export_path=str(target)))
        assert target.exists()

    def test_export_oserror_reported(self, tmp_path, monkeypatch, capsys):
        fw = tmp_path / "fw"
        fw.mkdir()
        self._stub_speedups(monkeypatch, fw)

        def boom(self, *a, **k):
            raise OSError("ro fs")
        monkeypatch.setattr(Path, "write_text", boom)
        _run_speedups_scan(_scan_args(coverage=True, no_export=False))
        assert "failed to write" in capsys.readouterr().err


# --------------------------------------------------------------------------
# _run_portability_scan
# --------------------------------------------------------------------------

class TestRunPortabilityScan:
    def test_no_task_dirs(self, capsys):
        _run_portability_scan(_scan_args(portability=True), task_dirs=[],
                              framework=None)
        assert "(no tasks found)" in capsys.readouterr().out

    def test_table_output(self, tmp_path, monkeypatch, capsys):
        res = SimpleNamespace(
            task_dir=str(tmp_path), run_3_5=False,
            to_json_dict=lambda: {"task": "t"})
        monkeypatch.setattr(scan_cmd, "scan_task",
                            lambda td, framework_root: res)
        monkeypatch.setattr(scan_cmd, "save_portability_scan",
                            lambda p, r: None)
        monkeypatch.setattr(scan_cmd, "render_portability_table",
                            lambda results: "PORT TABLE")
        _run_portability_scan(_scan_args(portability=True),
                              task_dirs=[tmp_path], framework=None)
        out = capsys.readouterr().out
        assert "PORT TABLE" in out
        assert "Results saved" in out

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        res = SimpleNamespace(
            task_dir=str(tmp_path), run_3_5=True,
            to_json_dict=lambda: {"task": "t"})
        monkeypatch.setattr(scan_cmd, "scan_task",
                            lambda td, framework_root: res)
        monkeypatch.setattr(scan_cmd, "save_portability_scan",
                            lambda p, r: None)
        _run_portability_scan(_scan_args(portability=True, json=True),
                              task_dirs=[tmp_path], framework=None)
        assert json.loads(capsys.readouterr().out.strip()) == {"task": "t"}

    def test_needs_3_5_filter(self, tmp_path, monkeypatch, capsys):
        keep = SimpleNamespace(task_dir=str(tmp_path / "k"), run_3_5=True,
                               to_json_dict=lambda: {"task": "keep"})
        drop = SimpleNamespace(task_dir=str(tmp_path / "d"), run_3_5=False,
                               to_json_dict=lambda: {"task": "drop"})
        results = iter([keep, drop])
        monkeypatch.setattr(scan_cmd, "scan_task",
                            lambda td, framework_root: next(results))
        monkeypatch.setattr(scan_cmd, "save_portability_scan",
                            lambda p, r: None)
        _run_portability_scan(
            _scan_args(portability=True, json=True, needs_3_5=True),
            task_dirs=[tmp_path / "k", tmp_path / "d"], framework=None)
        out = capsys.readouterr().out
        assert "keep" in out
        assert "drop" not in out
