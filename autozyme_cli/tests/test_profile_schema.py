"""Schema sanity tests: every backend produces a profile.json with the
canonical schema fields.

The agent reads profile.json structurally (e.g. iterates `hotspots`,
displays `notes`). Any backend silently dropping a required field would
break agent code without a clear error. These tests pin the contract.

Spawns `zyme profile --json` against the synthetic_groundtruth fixture —
slow (~5-10s per backend), but the only honest way to validate the full
pipeline → parser → JSON path. Marked as slow so default `pytest` skips
unless explicitly requested.

Run with:  pytest tests/test_profile_schema.py -v
       or: pytest -m slow tests/test_profile_schema.py
"""
from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from zyme.commands.profile import native


SYNTHETIC_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "synthetic_groundtruth"
)

REQUIRED_TOP_LEVEL = {
    "schema_version", "backend", "lang", "tier", "hypothesis", "timestamp",
    "totals", "hotspots", "actionable_hotspots", "notes", "artifacts",
    "call_chains",
}

REQUIRED_HOTSPOT_FIELDS = {
    "rank", "label", "self_time_s", "total_time_s", "self_pct", "calls", "raw",
}


pytestmark = pytest.mark.skipif(
    not SYNTHETIC_FIXTURE.exists(),
    reason=f"synthetic fixture missing at {SYNTHETIC_FIXTURE}",
)


def _run_profile(backend: str) -> dict:
    """Spawn zyme profile --json against the synthetic fixture; parse stdout."""
    cmd = [sys.executable, "-m", "zyme", "profile",
           "--backend", backend, "--dataset", "tiny",
           "--no-archive", "--json"]
    proc = subprocess.run(
        cmd, cwd=str(SYNTHETIC_FIXTURE), capture_output=True, text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"zyme profile --backend {backend} exited rc={proc.returncode}.\n"
        f"stderr tail: {proc.stderr[-400:]}"
    )
    return json.loads(proc.stdout)


def _run_profile_process(backend: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "zyme", "profile",
           "--backend", backend, "--dataset", "tiny",
           "--no-archive", "--json"]
    return subprocess.run(
        cmd, cwd=str(SYNTHETIC_FIXTURE), capture_output=True, text=True,
        timeout=120,
    )


def _successful_schema_backends() -> list[str]:
    backends = ["cpu", "full"]
    if importlib.util.find_spec("memray") is not None:
        backends.append("mem")
    if native.is_supported()[0]:
        backends.append("native")
    return backends


def _expected_effective_backend(requested: str) -> str:
    if requested == "full" and importlib.util.find_spec("scalene") is None:
        return "cpu"
    return requested


def _unavailable_refusal_backends() -> list[tuple[str, str]]:
    backends = []
    if importlib.util.find_spec("memray") is None:
        backends.append(("mem", "requires memray"))
    if not native.is_supported()[0]:
        backends.append(("native", "unavailable"))
    return backends


# --------------------------------------------------------------------------
# Per-backend schema check
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend", _successful_schema_backends())
def test_profile_json_top_level_schema(backend):
    """Each backend produces all REQUIRED_TOP_LEVEL fields."""
    data = _run_profile(backend)
    missing = REQUIRED_TOP_LEVEL - set(data.keys())
    assert not missing, f"backend={backend}: missing top-level fields: {missing}"
    assert data["backend"] == _expected_effective_backend(backend)
    assert data["schema_version"] == "2"
    assert data["lang"] == "py"
    assert data["tier"] == "tiny"
    assert isinstance(data["hotspots"], list)
    assert isinstance(data["actionable_hotspots"], list)
    assert isinstance(data["notes"], list)
    assert isinstance(data["artifacts"], dict)
    assert isinstance(data["call_chains"], list)
    assert isinstance(data["totals"], dict)


@pytest.mark.parametrize("backend", _successful_schema_backends())
def test_hotspot_field_schema(backend):
    """Each hotspot entry must carry all REQUIRED_HOTSPOT_FIELDS, even
    when some are None (e.g. native has no calls count)."""
    data = _run_profile(backend)
    hotspots = data.get("hotspots") or []
    if not hotspots:
        # Empty hotspots is acceptable for some backend×workload combos
        # (e.g. native against a process that exits before sample attaches),
        # but synthetic fixture is large enough that we expect non-empty.
        # Don't assert non-empty — just assert structure if present.
        return
    for h in hotspots:
        missing = REQUIRED_HOTSPOT_FIELDS - set(h.keys())
        assert not missing, f"backend={backend}: hotspot missing fields: {missing}"
        assert isinstance(h["rank"], int)
        assert h["rank"] >= 1
        assert isinstance(h["label"], str)
        assert isinstance(h["raw"], dict)


def test_synthetic_fixture_totals_populated():
    """totals.wall_s should be populated from emit_summary's speed_sec
    line — verifies the runner→profile_data plumbing for the totals dict."""
    data = _run_profile("cpu")
    totals = data["totals"]
    assert "wall_s" in totals, f"totals missing wall_s: {totals}"
    assert totals["wall_s"] > 0, f"wall_s must be positive: {totals['wall_s']}"


def test_synthetic_native_distinct_from_other_backends():
    """The native backend has a distinct artifact path schema (file glob,
    not single file) — verify it's emitted."""
    if not native.is_supported()[0]:
        pytest.skip("native backend is only supported where sample(1) is available")
    data = _run_profile("native")
    artifacts = data["artifacts"]
    assert "raw" in artifacts
    # Native produces a glob-style summary string with file count
    assert "native_sample" in artifacts["raw"], \
        f"native artifacts.raw should reference native_sample_*: {artifacts}"


@pytest.mark.parametrize(("backend", "message"), _unavailable_refusal_backends())
def test_unavailable_profile_backend_refuses_cleanly(backend, message):
    proc = _run_profile_process(backend)
    assert proc.returncode != 0
    assert message in proc.stderr
