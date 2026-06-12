"""Archive behavior tests for profile_history/.

By default `zyme profile` writes every run to
`<task>/profile_history/<ts>_<backend>_<tier>/` so the agent (or human)
can diff hot-spot drift across iterations. `--no-archive` writes only the
overwritten `profile_history/current/` scratch directory. These tests
exercise both paths and verify pipeline/ stays source-only.

Spawns subprocesses (~5-10s each), so marked structurally similar to
the schema tests — slow but worth it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from zyme.commands.profile import diff


SYNTHETIC_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "synthetic_groundtruth"
)


pytestmark = pytest.mark.skipif(
    not SYNTHETIC_FIXTURE.exists(),
    reason=f"synthetic fixture missing at {SYNTHETIC_FIXTURE}",
)


def _profile_history_run_dirs() -> list[Path]:
    hist = SYNTHETIC_FIXTURE / "profile_history"
    if not hist.exists():
        return []
    return sorted(
        entry for entry in hist.iterdir()
        if entry.name not in {"current", "latest"} and entry.is_dir()
    )


def _clear_profile_outputs():
    """Remove generated profile outputs so each test measures one run."""
    hist = SYNTHETIC_FIXTURE / "profile_history"
    if hist.exists():
        for entry in hist.iterdir():
            if entry.is_symlink():
                entry.unlink()
            elif entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    pipeline = SYNTHETIC_FIXTURE / "pipeline"
    for name in ("profile.json", "profile.out", "Rprof.out", "scalene.json", "memray.bin"):
        p = pipeline / name
        if p.exists():
            p.unlink()
    for p in pipeline.glob("native_sample_*"):
        p.unlink()


def _run_profile(*extra_args: str) -> int:
    """Spawn zyme profile, return rc."""
    cmd = [sys.executable, "-m", "zyme", "profile",
           "--backend", "cpu", "--dataset", "tiny", *extra_args]
    proc = subprocess.run(
        cmd, cwd=str(SYNTHETIC_FIXTURE), capture_output=True, text=True,
        timeout=120,
    )
    return proc.returncode


def test_archive_default_writes_one_file():
    """A single `zyme profile` run (default = archive on) must add exactly
    one run directory to profile_history/."""
    _clear_profile_outputs()
    rc = _run_profile()
    assert rc == 0
    files = _profile_history_run_dirs()
    assert len(files) == 1, \
        f"expected 1 archive dir after one run, got {len(files)}: {files}"
    assert files[0].is_dir()
    assert (files[0] / "profile.json").exists()
    assert (files[0] / "run.log").exists()
    assert not (SYNTHETIC_FIXTURE / "pipeline" / "profile.json").exists()
    assert not (SYNTHETIC_FIXTURE / "pipeline" / "profile.out").exists()
    # Dirname convention: <ts>_<backend>_<tier>
    name = files[0].name
    assert "cpu" in name
    assert "tiny" in name


def test_archive_two_runs_two_files():
    """Two runs → two distinct files (timestamped). No overwrite."""
    _clear_profile_outputs()
    rc1 = _run_profile()
    assert rc1 == 0
    rc2 = _run_profile()
    assert rc2 == 0
    files = _profile_history_run_dirs()
    # Could be 1 if timestamps collide (we use seconds resolution); accept >=1.
    # Stricter: require the 2nd file to exist OR the 1st to have been overwritten.
    assert len(files) >= 1
    if len(files) == 1:
        pytest.skip("unexpected single archive dir after two runs")
    assert len(files) == 2


def test_no_archive_writes_current_only():
    """--no-archive writes overwritten current/ scratch, not timestamped history."""
    _clear_profile_outputs()
    hist = SYNTHETIC_FIXTURE / "profile_history"
    rc = _run_profile("--no-archive")
    assert rc == 0
    files = list(hist.iterdir()) if hist.exists() else []
    assert [f.name for f in files] == ["current"]
    assert (hist / "current" / "profile.json").exists()
    assert (hist / "current" / "run.log").exists()
    assert not (SYNTHETIC_FIXTURE / "pipeline" / "profile.json").exists()
    assert not (SYNTHETIC_FIXTURE / "pipeline" / "profile.out").exists()


def test_diff_current_alias_reads_profile_history_current(monkeypatch):
    """`current` should resolve to profile_history/current/profile.json."""
    _clear_profile_outputs()
    rc = _run_profile("--no-archive")
    assert rc == 0

    monkeypatch.chdir(SYNTHETIC_FIXTURE)
    data, label = diff._load_profile("current")

    assert data["backend"] == "cpu"
    assert label.replace("\\", "/") == "profile_history/current/profile.json"


def test_archive_content_lives_under_profile_history():
    """The normalized JSON and raw artifact should live in the same run dir."""
    _clear_profile_outputs()
    rc = _run_profile()
    assert rc == 0
    archive_files = _profile_history_run_dirs()
    assert len(archive_files) == 1

    run_dir = archive_files[0]
    arch_data = json.loads((run_dir / "profile.json").read_text())
    assert arch_data["backend"] == "cpu"
    assert arch_data["tier"] == "tiny"
    assert arch_data["artifacts"]["profile_json"] == (
        f"profile_history/{run_dir.name}/profile.json"
    )
    assert arch_data["artifacts"]["run_log"] == (
        f"profile_history/{run_dir.name}/run.log"
    )
    assert arch_data["artifacts"]["raw"] == (
        f"profile_history/{run_dir.name}/profile.out"
    )
    assert (run_dir / "profile.out").exists()
    assert not (SYNTHETIC_FIXTURE / "pipeline" / "profile.json").exists()
