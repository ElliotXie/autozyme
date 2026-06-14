"""Wave-4 coverage tests for zyme.utils.

Targets REACHABLE lines left uncovered by test_utils.py / test_utils_deep.py:
  - resolve_reference_script bottom fallback chain (no task.yaml so the modes
    short-circuit is skipped): reference.R -> reference.py -> placeholder
  - read_baseline_stash empty-file return + V1 thread_mode->mode normalization
    + empty-mode backfill
  - upsert_baseline_stash read_active_mode fallback (no mode/thread_mode given)
  - read_upstream_repo_sha git-subprocess exception path (mocked, no real git)
  - check_upstream_version_drift corrupt-cache-JSON fallback + cache-write
    failure (read-only .zyme) both swallowed
  - read_installed_version exception swallow (mocked subprocess raises)

Subprocess-only branches that genuinely spawn an interpreter (the success
bodies of read_installed_version 532-544 / 555-576) are exercised end-to-end by
test_utils_deep.py::TestReadInstalledVersion and are NOT duplicated here; the
exception-swallow wrappers around them ARE covered (via mock).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import zyme.utils as U
from zyme.utils import (
    LEGACY_MODE,
    LEGACY_THREAD,
    baseline_stash_path,
    check_upstream_version_drift,
    read_baseline_stash,
    read_installed_version,
    read_upstream_repo_sha,
    resolve_reference_script,
    upsert_baseline_stash,
)


# --------------------------------------------------------------------------
# resolve_reference_script bottom fallback chain (mode=None, no task.yaml)
# --------------------------------------------------------------------------
class TestResolveReferenceScriptFallback:
    def test_reference_r_when_present(self, tmp_path: Path):
        # No task.yaml -> the active-mode return is skipped -> reference.R wins.
        (tmp_path / "reference.R").write_text("x")
        got = resolve_reference_script(tmp_path)
        assert got == (tmp_path / "reference.R").resolve()

    def test_reference_py_when_only_py(self, tmp_path: Path):
        (tmp_path / "reference.py").write_text("x")
        got = resolve_reference_script(tmp_path)
        assert got == (tmp_path / "reference.py").resolve()

    def test_placeholder_when_neither(self, tmp_path: Path):
        # Nothing exists -> placeholder reference.R path (caller errors later).
        got = resolve_reference_script(tmp_path)
        assert got == (tmp_path / "reference.R").resolve()


# --------------------------------------------------------------------------
# read_baseline_stash edge cases
# --------------------------------------------------------------------------
class TestReadBaselineStashEdges:
    def test_empty_file_returns_empty_list(self, tmp_path: Path):
        p = baseline_stash_path(tmp_path)
        p.parent.mkdir(parents=True)
        p.write_text("")  # truly empty -> lines == [] -> returns []
        assert read_baseline_stash(tmp_path) == []

    def test_v1_thread_mode_normalized_to_mode(self, tmp_path: Path):
        p = baseline_stash_path(tmp_path)
        p.parent.mkdir(parents=True)
        # header uses the V1 `thread_mode` column, no `mode` column.
        p.write_text(
            "tier\tname\tspeed_sec\tpeak_mb\tmetrics_json\tstatus\tthread_mode\tthread\n"
            "tiny\tt\t1.0\t100\t{}\tok\tfast\t4\n"
        )
        out = read_baseline_stash(tmp_path)
        assert out[0]["mode"] == "fast"
        assert out[0]["thread"] == "4"

    def test_blank_line_in_stash_skipped(self, tmp_path: Path):
        p = baseline_stash_path(tmp_path)
        p.parent.mkdir(parents=True)
        p.write_text(
            "tier\tname\tspeed_sec\tpeak_mb\tmetrics_json\tstatus\tmode\tthread\n"
            "\n"
            "tiny\tt\t1.0\t100\t{}\tok\tdefault\t1\n"
        )
        out = read_baseline_stash(tmp_path)
        assert len(out) == 1
        assert out[0]["tier"] == "tiny"

    def test_missing_mode_and_thread_backfilled(self, tmp_path: Path):
        p = baseline_stash_path(tmp_path)
        p.parent.mkdir(parents=True)
        # rows with blank mode/thread -> backfilled to LEGACY_MODE/LEGACY_THREAD
        p.write_text(
            "tier\tname\tspeed_sec\tpeak_mb\tmetrics_json\tstatus\tmode\tthread\n"
            "tiny\tt\t1.0\t100\t{}\tok\t\t\n"
        )
        out = read_baseline_stash(tmp_path)
        assert out[0]["mode"] == LEGACY_MODE
        assert out[0]["thread"] == str(LEGACY_THREAD)


# --------------------------------------------------------------------------
# upsert_baseline_stash read_active_mode fallback
# --------------------------------------------------------------------------
class TestUpsertActiveModeFallback:
    def test_no_mode_given_uses_active_mode(self, tmp_path: Path):
        # task.yaml present so read_active_mode resolves; entry has no mode or
        # thread_mode -> falls back to the task's active mode.
        (tmp_path / "task.yaml").write_text("target_repo: x\n")
        upsert_baseline_stash(tmp_path, {
            "tier": "tiny", "name": "t", "speed_sec": "1.0",
            "peak_mb": "100", "metrics_json": "{}", "status": "ok",
        })
        out = read_baseline_stash(tmp_path)
        assert out[0]["tier"] == "tiny"
        # active mode for a modes-less task is LEGACY_MODE
        assert out[0]["mode"] == LEGACY_MODE


# --------------------------------------------------------------------------
# read_upstream_repo_sha git-subprocess exception path
# --------------------------------------------------------------------------
class TestReadUpstreamRepoShaException:
    def test_subprocess_raises_returns_none(self, tmp_path: Path, monkeypatch):
        up = tmp_path / "upstream_repo"
        (up / ".git").mkdir(parents=True)

        def boom(*a, **k):
            raise OSError("git missing")

        monkeypatch.setattr(subprocess, "run", boom)
        assert read_upstream_repo_sha(tmp_path) is None

    def test_nonzero_returncode_returns_none(self, tmp_path: Path, monkeypatch):
        up = tmp_path / "upstream_repo"
        (up / ".git").mkdir(parents=True)

        class R:
            returncode = 128
            stdout = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
        assert read_upstream_repo_sha(tmp_path) is None


# --------------------------------------------------------------------------
# read_installed_version exception swallow (mocked subprocess raises)
# --------------------------------------------------------------------------
class TestReadInstalledVersionException:
    def test_subprocess_raises_returns_none(self, tmp_path: Path, monkeypatch):
        (tmp_path / "task.yaml").write_text("target_repo: x\n")
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.py").write_text("x = 1\n")

        def boom(*a, **k):
            raise OSError("no interpreter")

        monkeypatch.setattr(subprocess, "run", boom)
        assert read_installed_version(tmp_path, "numpy") is None


# --------------------------------------------------------------------------
# check_upstream_version_drift cache JSON corrupt + cache-write failure
# --------------------------------------------------------------------------
def _py_upstream_task(tmp_path: Path) -> Path:
    (tmp_path / "task.yaml").write_text("target_repo: x\n")
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "run.py").write_text("x = 1\n")
    up = tmp_path / "upstream_repo"
    up.mkdir()
    (up / "pyproject.toml").write_text(
        "[project]\nname = \"mypkg\"\nversion = \"1.2.3\"\n"
    )
    return tmp_path


class TestCheckUpstreamVersionDriftEdges:
    def test_corrupt_cache_falls_through(self, tmp_path: Path, monkeypatch):
        task = _py_upstream_task(tmp_path)
        # plant a git SHA so the cache key is non-None
        monkeypatch.setattr(U, "read_upstream_repo_sha", lambda td: "deadbeef")
        cache = task / ".zyme" / "version_check.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{corrupt json")
        # installed version resolves via a mocked reader so the function
        # proceeds past the cache to a real comparison result.
        monkeypatch.setattr(U, "read_installed_version", lambda td, pkg: "1.2.3")
        out = check_upstream_version_drift(task)
        assert out["package"] == "mypkg"
        assert out["agrees"] is True

    def test_cache_write_failure_swallowed(self, tmp_path: Path, monkeypatch):
        task = _py_upstream_task(tmp_path)
        monkeypatch.setattr(U, "read_upstream_repo_sha", lambda td: "cafef00d")
        monkeypatch.setattr(U, "read_installed_version", lambda td, pkg: "9.9.9")

        # Make the cache write raise; the function must still return the result.
        import json as _json
        real_dumps = _json.dumps

        def boom_dumps(obj, *a, **k):
            if isinstance(obj, dict) and "upstream_sha" in obj:
                raise OSError("disk full")
            return real_dumps(obj, *a, **k)

        monkeypatch.setattr(U.json, "dumps", boom_dumps)
        out = check_upstream_version_drift(task)
        assert out["installed_version"] == "9.9.9"
        assert out["agrees"] is False

    def test_no_upstream_metadata_returns_none(self, tmp_path: Path, monkeypatch):
        (tmp_path / "task.yaml").write_text("target_repo: x\n")
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.py").write_text("x = 1\n")
        # upstream_repo dir with no parseable metadata file
        (tmp_path / "upstream_repo").mkdir()
        monkeypatch.setattr(U, "read_upstream_repo_sha", lambda td: None)
        assert check_upstream_version_drift(tmp_path) is None

    def test_installed_none_returns_none(self, tmp_path: Path, monkeypatch):
        task = _py_upstream_task(tmp_path)
        monkeypatch.setattr(U, "read_upstream_repo_sha", lambda td: None)
        monkeypatch.setattr(U, "read_installed_version", lambda td, pkg: None)
        assert check_upstream_version_drift(task) is None
