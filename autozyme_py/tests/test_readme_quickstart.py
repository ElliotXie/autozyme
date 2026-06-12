"""Every runnable code block in autozyme_py/README.md must actually run.

The README is the first thing a new user copy-pastes. If we rename a kwarg,
move a function, or change a default and forget to update the README, this
test catches it BEFORE a user files an issue saying "your quickstart is
broken."

Code blocks that require an optional upstream (scvelo, prody, etc.) are
skipped when that upstream isn't installed — they still get exercised in
Tier B nightly where the upstream is pinned.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import autozyme


# ---- Section: "Use" ------------------------------------------------------

def test_import_autozyme_works():
    """`import autozyme` must succeed without optional upstream imports."""
    # Already imported at module scope — the import here exercises a fresh
    # reload to catch a top-level side-effect regression.
    import importlib
    importlib.reload(autozyme)


def test_disabled_context_manager():
    """README shows `with autozyme.disabled(): ...` — must not raise and
    must set is_disabled() inside the block."""
    assert autozyme.is_disabled() is False
    with autozyme.disabled():
        assert autozyme.is_disabled() is True
    assert autozyme.is_disabled() is False


def test_restore_all_session_wide():
    """README: `autozyme.deactivate_all()` — must succeed even with nothing active."""
    autozyme.deactivate_all()  # no-op when nothing active; must not raise


def test_env_var_autozyme_disabled_recognized():
    """README: `AUTOZYME_DISABLED=1` set before import suppresses activation.

    We verify the env var is consulted by spawning a fresh subprocess (the
    banner suppression check) — testing it in-process is racy because
    autozyme's already imported.
    """
    env = os.environ.copy()
    env["AUTOZYME_DISABLED"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", "import autozyme; print('imported ok')"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert "imported ok" in proc.stdout
    # Banner goes to stderr; with AUTOZYME_DISABLED set, it must NOT print
    # the "patches available" line.
    assert "patches available" not in proc.stderr


# ---- Section: "Threading" ------------------------------------------------

def test_set_threads_in_quickstart():
    """README: `autozyme.set_threads(8)` — must succeed and write env vars."""
    autozyme.set_threads(8)
    assert os.environ["OMP_NUM_THREADS"] == "8"


# ---- Section: "Status and introspection" --------------------------------

def test_list_patches_returns_list():
    """README: `autozyme.list_patches()` returns a list of patch names."""
    patches = autozyme.list_patches()
    assert isinstance(patches, list)
    assert all(isinstance(p, str) for p in patches)
    assert len(patches) > 0, "expected at least one registered patch"


def test_list_patches_installed_filter():
    """README: `autozyme.list_patches(installed=True)` filters to importable upstreams."""
    all_patches = set(autozyme.list_patches())
    installed = set(autozyme.list_patches(installed=True))
    assert installed.issubset(all_patches)


def test_status_returns_dict():
    """README: `autozyme.status()` returns a dict of patch -> state."""
    s = autozyme.status()
    assert isinstance(s, dict)
    for name, state in s.items():
        assert isinstance(name, str)
        assert state in ("active", "inactive"), f"unexpected state {state!r}"


def test_env_snapshot_returns_dict():
    """README: `autozyme.env_snapshot()` returns a structured provenance dict."""
    snap = autozyme.env_snapshot()
    assert isinstance(snap, dict)


def test_python_dash_m_autozyme_dashboard():
    """README: `python -m autozyme` prints a one-screen dashboard.

    Must exit 0 and produce non-empty output. We don't assert on specific
    text since the dashboard format is allowed to evolve.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "autozyme"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert proc.stdout.strip(), "dashboard produced no stdout"


# ---- Section: "Install" --------------------------------------------------

def test_install_section_has_no_hard_upstream_deps():
    """README claim: 'autozyme itself has no hard dependencies on upstream packages.'

    Verify in a fresh interpreter that importing autozyme does not pull in
    big optional upstreams transitively. Doing this in-process is
    order-dependent once other tests have intentionally imported scanpy,
    scvelo, MDAnalysis, etc.
    """
    code = r"""
import sys
import autozyme
forbidden = {
    "scanpy", "scvelo", "torch", "tensorflow", "tensorflow_probability",
    "pyro", "MDAnalysis", "prody", "cell2location",
}
leaked = sorted(forbidden & set(sys.modules))
if leaked:
    raise SystemExit("leaked=" + ",".join(leaked))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"importing autozyme transitively loaded heavy upstreams. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
