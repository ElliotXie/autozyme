"""pipeline/run.py — alloc_heavy_synthetic.

Three distinct allocation patterns with controlled relative magnitudes:
  alloc_big:     single 800MB float64 array            (1 large alloc)
  alloc_medium:  100 × 3MB int64 arrays                (many medium allocs, churn)
  alloc_small:   dict of 100 × 1MB string-keyed arrays (small allocs in dict)

Magnitudes chosen so memray's allocation-by-size ranking is unambiguous:
  alloc_big > alloc_medium (cumulative) > alloc_small (cumulative)
even though alloc_medium has more individual allocations.
"""
import os
import sys
import time

# Locate framework helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_HELPERS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "zyme"))
if _HELPERS_DIR not in sys.path:
    sys.path.insert(0, _HELPERS_DIR)
from helpers import with_profile, with_subprofile, peak_memory_mb, emit_summary  # noqa: E402

import numpy as np  # noqa: E402


def alloc_big():
    """One large 2D float64 array + matmul-ish compute on it. ~800MB peak.
    Allocation dominates memray's ranking; compute time gives all
    backends (cpu/full/native) something to capture in the same call site.

    The matmul iterations are tuned so this function runs ~1.5-2s wall —
    long enough that native sample (1ms interval) captures hundreds of
    in-NumPy-internal samples after its 100ms settle delay."""
    # 10000 × 10000 float64 = 8e8 bytes = 800MB
    arr = np.zeros((10_000, 10_000), dtype=np.float64)
    arr += np.arange(10_000, dtype=np.float64)  # broadcast fill
    # 30 matmuls on a 1.5k×1.5k slice — keeps wall ~2-3s, not minutes.
    # BLAS parallelizes hard so wall << cpu_time; native sample captures
    # both Python-side and OpenBLAS internals during this stretch.
    sub = arr[:1500, :1500]
    s = 0.0
    for _ in range(30):
        s += np.dot(sub, sub.T).sum()
    del arr
    return s


def alloc_medium():
    """100 medium int64 arrays in a list, with summing compute.
    ~300MB cumulative, ~1s wall. Tests memray's churn handling
    (many distinct allocations from the same call site)."""
    arrays = []
    for i in range(100):
        # 400_000 int64 = 3.2MB each × 100 = 320MB
        a = np.full(400_000, i, dtype=np.int64)
        arrays.append(a)
    # Pairwise sums to give cpu/native non-trivial work proportional
    # to the allocations
    total = 0
    for a in arrays:
        total += int(a.sum())
    return total


def alloc_small():
    """Dict of 100 small float arrays + cumulative compute. ~100MB.
    Tests memray on dict-mediated allocations."""
    d = {}
    for i in range(100):
        # 125_000 float64 = 1MB × 100 = 100MB
        d[f"key_{i}"] = np.linspace(0, 1, 125_000)
    # Cumulative sum + std on each — gives ~0.5s of compute
    total = 0.0
    for v in d.values():
        total += float(np.std(v) + np.cumsum(v)[-1])
    return total


def main():
    t0 = time.perf_counter()
    with with_profile():
        with with_subprofile("alloc_big"):
            r1 = alloc_big()
        with with_subprofile("alloc_medium"):
            r2 = alloc_medium()
        with with_subprofile("alloc_small"):
            r3 = alloc_small()
    elapsed = time.perf_counter() - t0
    print(f"[alloc_heavy] big={r1:.2f} medium={r2} small={r3:.2f}", flush=True)
    emit_summary(speed_sec=elapsed, peak_mb=peak_memory_mb())


if __name__ == "__main__":
    main()
