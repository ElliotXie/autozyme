"""Multi-rep timing of a registered patch — complements verify_patch.

verify_patch answers "is the lift correct?" (1 rep + metric gate).
benchmark answers "is the speedup stable?" (N reps + summary stats).

Reuses the patch's smoke recipe; does not run evaluate.{R,py} (no metric
verification), so it's purely a timing tool. Run verify_patch FIRST to
confirm correctness, THEN benchmark for production-grade speedup numbers.
"""
from __future__ import annotations

import gc
import statistics
import sys
import time
from typing import Any

from autozyme._core import _REGISTRY, _import_submodule, activate, deactivate


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "min":    min(samples),
        "median": statistics.median(samples),
        "max":    max(samples),
        "mean":   statistics.fmean(samples),
        "stdev":  statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "all":    list(samples),
    }


def benchmark(
    name: str,
    task_dir: str,
    tier: str = "tiny",
    reps: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Time `reps` baseline + patched runs of the smoke recipe.

    Args:
        name: registered patch name.
        task_dir: path to the autozyme task directory.
        tier: dataset tier (default 'tiny').
        reps: number of timed runs per arm (default 3).
        verbose: print summary table.

    Returns:
        dict with keys baseline, patched, speedup_x, speedup_pct — each is a
        dict of {min, median, max, mean, stdev, all}.
    """
    if name not in _REGISTRY:
        _import_submodule(name)
    p = _REGISTRY.get(name)
    if p is None:
        raise KeyError(f"no patch registered for {name!r}")
    if p.smoke is None:
        raise ValueError(
            f"patch {name!r} has no smoke recipe — pass smoke=dict(load, call, save) "
            f"to register_patch()"
        )
    try:
        reps = int(reps)
    except (TypeError, ValueError):
        raise ValueError(
            f"reps must be a positive integer, got {reps!r} "
            f"({type(reps).__name__})"
        )
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")

    inputs = p.smoke["load"](task_dir, tier)

    if verbose:
        print(f"--- benchmark {name!r} (reps={reps}, tier={tier!r}) ---",
              file=sys.stderr)

    baseline_times: list[float] = []
    patched_times:  list[float] = []
    for i in range(reps):
        deactivate(name)
        gc.collect()
        t0 = time.perf_counter()
        p.smoke["call"](inputs)
        baseline_times.append(time.perf_counter() - t0)
        if verbose:
            print(f"  baseline rep {i + 1}/{reps}: {baseline_times[-1]:.3f} sec",
                  file=sys.stderr)

        activate(name)
        gc.collect()
        t0 = time.perf_counter()
        p.smoke["call"](inputs)
        patched_times.append(time.perf_counter() - t0)
        if verbose:
            print(f"  patched  rep {i + 1}/{reps}: {patched_times[-1]:.3f} sec",
                  file=sys.stderr)

    speedup_x_samples   = [b / p_ for b, p_ in zip(baseline_times, patched_times)]
    speedup_pct_samples = [(b - p_) / b * 100 for b, p_ in zip(baseline_times, patched_times)]

    result = {
        "baseline":    _summarize(baseline_times),
        "patched":     _summarize(patched_times),
        "speedup_x":   _summarize(speedup_x_samples),
        "speedup_pct": _summarize(speedup_pct_samples),
    }

    if verbose:
        print(f"\n--- summary (median over {reps} reps) ---", file=sys.stderr)
        print(f"  baseline: {result['baseline']['median']:.3f} sec  "
              f"(±{result['baseline']['stdev']:.3f})", file=sys.stderr)
        print(f"  patched:  {result['patched']['median']:.3f} sec  "
              f"(±{result['patched']['stdev']:.3f})", file=sys.stderr)
        print(f"  speedup:  {result['speedup_pct']['median']:.1f}%  "
              f"({result['speedup_x']['median']:.1f}x)", file=sys.stderr)

    return result
