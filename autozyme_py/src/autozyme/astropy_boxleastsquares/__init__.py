"""Patch for astropy BoxLeastSquares.autopower.

Lifted from autozyme task `test_astropy_boxleastsquares`. The user-facing
target is ``astropy.timeseries.BoxLeastSquares.autopower``; the hot native
kernel underneath it is ``astropy.timeseries.periodograms.bls.methods.bls_fast``.

The patch parallelizes the exact upstream ``bls_fast`` implementation across
period chunks. Each chunk calls the captured upstream function on a disjoint
slice of the period grid, then concatenates Astropy's result tuple in the
original period order. Chunk boundaries are balanced by the same approximate
per-period work model used by Astropy's Cython kernel, so long-period tails do
not dominate a single worker.

This is an embarrassingly parallel case over candidate periods. Single-thread
or small-grid calls fall through to upstream unchanged; multi-thread speedup is
controlled by ``ZYME_THREADS`` / ``AUTOZYME_THREADS`` / ``AUTOZYMER_THREADS`` /
``OMP_NUM_THREADS`` or, with no override, the task's validated default of
``os.cpu_count() + 6`` workers.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import astropy
import numpy as np
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries.periodograms.bls import methods as _bls_methods

from autozyme._core import register_patch
from autozyme._utils import resolve_dataset_path


_orig_bls_fast = _bls_methods.bls_fast

_MIN_PERIODS_PER_WORKER = 16384
_RESULT_KEYS = (
    "objective",
    "period",
    "power",
    "depth",
    "depth_err",
    "duration",
    "transit_time",
    "depth_snr",
    "log_likelihood",
)


def _thread_count() -> int:
    """Resolve the period-worker count at call time."""
    for var in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
                "OMP_NUM_THREADS"):
        value = os.environ.get(var)
        if not value:
            continue
        try:
            n = int(value)
        except ValueError:
            continue
        if n >= 1:
            return n
    return max(1, (os.cpu_count() or 1) + 6)


def _bls_chunk_edges(t, period, duration, oversample, n_workers):
    bin_duration = float(np.min(duration)) / float(oversample)
    if not np.isfinite(bin_duration) or bin_duration <= 0.0:
        return np.linspace(0, len(period), n_workers + 1, dtype=np.int64)

    weights = (
        len(t)
        + len(duration) * np.asarray(period, dtype=np.float64) / bin_duration
    )
    cumulative = np.empty(len(period) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(weights, out=cumulative[1:])
    targets = np.linspace(0.0, cumulative[-1], n_workers + 1)
    edges = np.searchsorted(cumulative, targets, side="left").astype(np.int64)
    edges[0] = 0
    edges[-1] = len(period)
    return edges


def fast_bls_fast(t, y, ivar, period, duration, oversample, use_likelihood):
    n_periods = len(period)
    n_workers = min(_thread_count(), n_periods // _MIN_PERIODS_PER_WORKER)
    if n_workers <= 1:
        return _orig_bls_fast(
            t, y, ivar, period, duration, oversample, use_likelihood
        )

    edges = _bls_chunk_edges(t, period, duration, oversample, n_workers)

    def run_chunk(i):
        chunk = slice(int(edges[i]), int(edges[i + 1]))
        return _orig_bls_fast(
            t, y, ivar, period[chunk], duration, oversample, use_likelihood
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        chunks = list(executor.map(run_chunk, range(n_workers)))

    return tuple(
        np.concatenate([chunk[field] for chunk in chunks])
        for field in range(len(chunks[0]))
    )


def _smoke_load(task_dir, tier):
    import yaml

    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        task = yaml.safe_load(f)
    ds = next(d for d in task["datasets"] if d["tier"] == tier)
    data_path = resolve_dataset_path(task_dir, ds["path"])

    with np.load(data_path, allow_pickle=False) as data:
        t = np.ascontiguousarray(data["t"], dtype=np.float64)
        y = np.ascontiguousarray(data["y"], dtype=np.float64)
        dy = np.ascontiguousarray(data["dy"], dtype=np.float64)
        duration = np.ascontiguousarray(data["duration"], dtype=np.float64)

    model = BoxLeastSquares(t, y, dy)
    return {"model": model, "duration": duration}


def _smoke_call(inputs):
    return inputs["model"].autopower(inputs["duration"])


def _smoke_save(result, dir, **kwargs):
    payload = {key: np.asarray(result[key]) for key in _RESULT_KEYS}
    np.savez(os.path.join(dir, "result.npz"), **payload)


register_patch(
    name="astropy_boxleastsquares",
    targets=[
        (
            "astropy.timeseries.periodograms.bls.methods",
            "bls_fast",
            fast_bls_fast,
        ),
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="astropy 6.1.0",
    tested_upstream_versions={"astropy": ["6.1.0"]},
)
