"""Wave-4 wrapper/smoke-line tests for autozyme.astropy_boxleastsquares.

Wave-1 (`_unit`) covered `_thread_count` + `_bls_chunk_edges`; wave-2 (`_e2e`)
covered the multi-worker dispatch concat path in `fast_bls_fast`. The only
COVERAGE-VISIBLE lines left are the smoke recipe (`_smoke_load` / `_smoke_call`
/ `_smoke_save`, lines 110-134), which read a tier npz via a task.yaml. We drive
them directly against a tiny synthetic npz built in a tmpdir — no external task
dir or dataset registry required.

(The BLS native kernel `bls_fast` itself is upstream Cython, invisible to
coverage and not ours; we only exercise the python smoke wrapper here.)
"""
from __future__ import annotations

import json
import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("astropy")
yaml = pytest.importorskip("yaml")
from astropy.timeseries import BoxLeastSquares

from autozyme import astropy_boxleastsquares as azbls


def _make_task_dir(tmp_path):
    """A minimal task dir: task.yaml pointing at one tier npz with the keys
    `_smoke_load` reads (t, y, dy, duration)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    n = 400
    t = np.sort(rng.uniform(0.0, 20.0, n))
    P_true = 1.7
    phase = (t % P_true) / P_true
    transit = (phase > 0.4) & (phase < 0.45)
    y = 1.0 - 0.04 * transit.astype(float) + rng.normal(scale=0.004, size=n)
    dy = np.full(n, 0.004)
    duration = np.array([0.05, 0.1])
    np.savez(
        data_dir / "small.npz",
        t=t.astype(np.float64),
        y=y.astype(np.float64),
        dy=dy.astype(np.float64),
        duration=duration.astype(np.float64),
    )
    task = {
        "datasets": [
            {"tier": "small", "name": "synth", "path": "./data/small.npz"},
        ]
    }
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    return str(tmp_path)


def test_smoke_load_builds_model_and_duration(tmp_path):
    """_smoke_load resolves the tier npz, builds a BoxLeastSquares model, and
    returns the duration array (covers lines 110-125)."""
    task_dir = _make_task_dir(tmp_path)
    out = azbls._smoke_load(task_dir, "small")
    assert isinstance(out["model"], BoxLeastSquares)
    assert out["duration"].shape == (2,)
    assert out["duration"].dtype == np.float64


def test_smoke_call_runs_autopower(tmp_path):
    """_smoke_call invokes model.autopower(duration) -> a BLSResults-like object
    with the canonical result keys (covers line 129)."""
    task_dir = _make_task_dir(tmp_path)
    inputs = azbls._smoke_load(task_dir, "small")
    result = azbls._smoke_call(inputs)
    for key in azbls._RESULT_KEYS:
        assert key in result, f"missing BLS result field {key}"
    assert len(result["period"]) == len(result["power"])


def test_smoke_save_writes_result_npz(tmp_path):
    """_smoke_save serializes every _RESULT_KEYS field into result.npz
    (covers lines 133-134)."""
    task_dir = _make_task_dir(tmp_path)
    inputs = azbls._smoke_load(task_dir, "small")
    result = azbls._smoke_call(inputs)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    azbls._smoke_save(result, str(out_dir))
    saved = np.load(out_dir / "result.npz")
    for key in azbls._RESULT_KEYS:
        assert key in saved.files
    assert len(saved["power"]) == len(result["power"])


def test_smoke_roundtrip_matches_direct_autopower(tmp_path):
    """The smoke path's saved power equals a direct autopower call on the same
    model — confirms the wrapper doesn't reshape/reorder the result."""
    task_dir = _make_task_dir(tmp_path)
    inputs = azbls._smoke_load(task_dir, "small")
    direct = inputs["model"].autopower(inputs["duration"])
    result = azbls._smoke_call(inputs)
    np.testing.assert_array_equal(
        np.asarray(result["power"]), np.asarray(direct["power"])
    )
