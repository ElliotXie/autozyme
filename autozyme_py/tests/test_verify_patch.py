"""Integration tests for verify_patch() — the SDK's validation function.

verify_patch() is the function external patch authors use to validate their
patch end-to-end. It spawns worker subprocesses, reads task.yaml, runs
evaluate.{py,R}, parses metrics, and writes package_verify.tsv. None of
that pipeline was previously exercised in Tier A CI.

These tests use the `_test_json` synthetic submodule (stdlib-only, no
optional deps) so they complete in < 5 s and require no HF_TOKEN. The
synthetic patch targets `json.dumps` but its smoke recipe does not call
json.dumps in the critical path, so baseline and patched produce identical
outputs and the metric passes.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import autozyme
from autozyme._core import _REGISTRY, _import_submodule
from autozyme import verify_patch


# ---- Shared task fixtures ------------------------------------------------

_TASK_YAML_PASS = """\
task: _test_json
datasets:
  - {tier: tiny, name: unittest, path: /dev/null}
metrics:
  - {name: output_match, threshold: 1.0, comparator: gte}
"""

_TASK_YAML_FAIL = """\
task: _test_json_fail
datasets:
  - {tier: tiny, name: unittest, path: /dev/null}
metrics:
  - {name: output_match, threshold: 2.0, comparator: gte}
"""

_EVALUATE_PY = """\
import os
ref_dir = os.environ.get("ZYME_REFERENCE_DIR", "reference_output")
test_dir = os.environ.get("ZYME_TEST_DIR", "pipeline")
ref = open(os.path.join(ref_dir, "output.txt")).read().strip()
test = open(os.path.join(test_dir, "output.txt")).read().strip()
print(f"output_match: {1.0 if test == ref else 0.0}")
"""


@pytest.fixture(autouse=True)
def _ensure_test_patch_registered():
    """Import and register the synthetic patch; deactivate (but keep registered)
    after each test.

    We keep the _REGISTRY entry across tests because the module is cached in
    sys.modules after first import — re-importing it would be a no-op and
    register_patch() would never be called again. Deactivation via deactivate()
    is sufficient for test isolation.
    """
    _import_submodule("_test_json")
    yield
    if _REGISTRY.get("_test_json") and _REGISTRY["_test_json"].injected:
        autozyme.deactivate("_test_json")


@pytest.fixture()
def passing_task(tmp_path: Path) -> Path:
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML_PASS)
    (td / "evaluate.py").write_text(_EVALUATE_PY)
    return td


@pytest.fixture()
def failing_task(tmp_path: Path) -> Path:
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML_FAIL)
    (td / "evaluate.py").write_text(_EVALUATE_PY)
    return td


# ---- Tests ---------------------------------------------------------------

def test_verify_patch_returns_pass(passing_task: Path):
    """verify_patch() must spawn workers, run evaluate.py, and return
    all_pass=True when patched output matches baseline within threshold.
    """
    results = verify_patch(
        "_test_json",
        str(passing_task),
        tiers=("tiny",),
        reps=1,
        verbose=False,
        use_baseline_cache=False,
    )

    assert len(results) == 1
    r = results[0]
    assert r["tier"] == "tiny"
    assert r["all_pass"] is True
    assert len(r["baseline_secs"]) == 1
    assert len(r["patched_secs"]) == 1
    assert r["baseline_secs"][0] > 0
    assert r["patched_secs"][0] > 0


def test_verify_patch_writes_package_verify_tsv(passing_task: Path):
    """verify_patch() must create package_verify.tsv in the task dir."""
    verify_patch(
        "_test_json",
        str(passing_task),
        tiers=("tiny",),
        reps=1,
        verbose=False,
        use_baseline_cache=False,
    )

    tsv = passing_task / "package_verify.tsv"
    assert tsv.exists()
    lines = tsv.read_text(encoding="utf-8").splitlines()
    # Header + at least 2 data rows (1 baseline + 1 patched)
    assert len(lines) >= 3
    header = lines[0].split("\t")
    assert "patch_name" in header
    assert "tier" in header
    assert "variant" in header


def test_verify_patch_fails_on_impossible_threshold(failing_task: Path):
    """verify_patch() must return all_pass=False when metric can't meet threshold."""
    results = verify_patch(
        "_test_json",
        str(failing_task),
        tiers=("tiny",),
        reps=1,
        verbose=False,
        use_baseline_cache=False,
    )

    assert len(results) == 1
    assert results[0]["all_pass"] is False


def test_verify_patch_raises_on_missing_task_dir():
    """verify_patch() must raise FileNotFoundError for a non-existent task_dir."""
    with pytest.raises(FileNotFoundError):
        verify_patch("_test_json", "/nonexistent/path/task", tiers=("tiny",))


def test_verify_patch_raises_on_missing_metrics(tmp_path: Path):
    """verify_patch() must raise ValueError when task.yaml has no metrics."""
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text("task: empty\ndatasets:\n  - {tier: tiny, name: u, path: /dev/null}\n")
    (td / "evaluate.py").write_text(_EVALUATE_PY)

    with pytest.raises(ValueError, match="metrics"):
        verify_patch("_test_json", str(td), tiers=("tiny",))


def test_verify_patch_raises_on_unknown_patch(tmp_path: Path):
    """verify_patch() must raise KeyError for an unregistered patch name."""
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML_PASS)
    (td / "evaluate.py").write_text(_EVALUATE_PY)

    with pytest.raises((KeyError, ImportError)):
        verify_patch("definitely_not_a_real_patch", str(td), tiers=("tiny",))
