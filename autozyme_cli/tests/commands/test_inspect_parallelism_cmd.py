"""Unit tests for zyme.commands.inspect_parallelism.

The heavy scanning lives in zyme.scan_parallelism; this command is a thin
workspace-side wrapper. We cover _auto_target (task.yaml target_function
discovery + the ::/. bare-name peeling) and cmd_inspect_parallelism over a
stubbed scan boundary (scan / format_report / filter_knobs_by_target), plus the
not-found / not-a-dir error gates.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import inspect_parallelism as ip


# --------------------------------------------------------------------------
# _auto_target
# --------------------------------------------------------------------------

class TestAutoTarget:
    def test_reads_target_function_from_sibling_yaml(self, tmp_path):
        repo = tmp_path / "task" / "upstream_repo"
        repo.mkdir(parents=True)
        (tmp_path / "task" / "task.yaml").write_text(
            "target_function: mgcv::gam\n")
        assert ip._auto_target(repo) == "gam"

    def test_peels_dotted_name(self, tmp_path):
        repo = tmp_path / "task" / "repo"
        repo.mkdir(parents=True)
        (tmp_path / "task" / "task.yaml").write_text(
            "target_function: lifelines.CoxPHFitter.fit\n")
        assert ip._auto_target(repo) == "fit"

    def test_grandparent_yaml(self, tmp_path):
        repo = tmp_path / "a" / "b" / "repo"
        repo.mkdir(parents=True)
        # task.yaml two levels up (repo.parent.parent)
        (tmp_path / "a" / "task.yaml").write_text("target_function: foo::bar\n")
        assert ip._auto_target(repo) == "bar"

    def test_no_yaml_returns_none(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        assert ip._auto_target(repo) is None

    def test_empty_target_returns_none(self, tmp_path):
        repo = tmp_path / "task" / "repo"
        repo.mkdir(parents=True)
        (tmp_path / "task" / "task.yaml").write_text("target_function: \n")
        assert ip._auto_target(repo) is None

    def test_malformed_yaml_tolerated(self, tmp_path):
        repo = tmp_path / "task" / "repo"
        repo.mkdir(parents=True)
        (tmp_path / "task" / "task.yaml").write_text("::: not yaml :::\n[")
        # broad except -> None, no raise
        assert ip._auto_target(repo) is None


# --------------------------------------------------------------------------
# cmd_inspect_parallelism
# --------------------------------------------------------------------------

class TestCmdInspectParallelism:
    def _stub_scan(self, monkeypatch, *, expect_filter=False):
        import zyme.scan_parallelism as sp
        calls = {"filtered": False}
        monkeypatch.setattr(sp, "scan", lambda repo: {"inventory": "x"})
        monkeypatch.setattr(sp, "format_report",
                            lambda inv, max_hits_per_backend: "REPORT BODY")

        def fake_filter(inv, repo, target):
            calls["filtered"] = True
            calls["target"] = target
        monkeypatch.setattr(sp, "filter_knobs_by_target", fake_filter)
        return calls

    def test_missing_repo_dies(self, tmp_path):
        args = SimpleNamespace(repo=str(tmp_path / "nope"), target=None,
                               max_hits=5)
        with pytest.raises(SystemExit):
            ip.cmd_inspect_parallelism(args)

    def test_repo_not_dir_dies(self, tmp_path):
        f = tmp_path / "afile"
        f.write_text("x")
        args = SimpleNamespace(repo=str(f), target=None, max_hits=5)
        with pytest.raises(SystemExit):
            ip.cmd_inspect_parallelism(args)

    def test_prints_report_no_target(self, tmp_path, monkeypatch, capsys):
        repo = tmp_path / "repo"
        repo.mkdir()
        calls = self._stub_scan(monkeypatch)
        args = SimpleNamespace(repo=str(repo), target=None, max_hits=5)
        ip.cmd_inspect_parallelism(args)
        out = capsys.readouterr().out
        assert "REPORT BODY" in out
        # no task.yaml -> no auto target -> filter not called
        assert calls["filtered"] is False

    def test_explicit_target_triggers_filter(self, tmp_path, monkeypatch, capsys):
        repo = tmp_path / "repo"
        repo.mkdir()
        calls = self._stub_scan(monkeypatch)
        args = SimpleNamespace(repo=str(repo), target="mgcv::gam", max_hits=10)
        ip.cmd_inspect_parallelism(args)
        assert calls["filtered"] is True
        assert calls["target"] == "gam"  # bare name peeled

    def test_auto_target_from_task_yaml(self, tmp_path, monkeypatch, capsys):
        task = tmp_path / "task"
        repo = task / "upstream_repo"
        repo.mkdir(parents=True)
        (task / "task.yaml").write_text("target_function: foo::bar\n")
        calls = self._stub_scan(monkeypatch)
        args = SimpleNamespace(repo=str(repo), target=None, max_hits=5)
        ip.cmd_inspect_parallelism(args)
        assert calls["filtered"] is True
        assert calls["target"] == "bar"
