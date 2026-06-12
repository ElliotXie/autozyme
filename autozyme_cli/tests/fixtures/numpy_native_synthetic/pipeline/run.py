"""Lightweight native-kernel fixture built from NumPy operations."""
from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_HELPERS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "zyme"))
if _HELPERS_DIR not in sys.path:
    sys.path.insert(0, _HELPERS_DIR)

from helpers import (  # noqa: E402
    emit_summary,
    peak_memory_mb,
    with_profile,
    with_subprofile,
)
import numpy as np  # noqa: E402


def _load_config() -> dict:
    data_path = os.environ.get("ZYME_DATA_PATH") or os.path.join(
        os.path.dirname(_HERE), "data", "tiny.json",
    )
    with open(data_path) as fh:
        return json.load(fh)


def matmul_kernel(n: int, min_reps: int, min_wall_s: float) -> float:
    rng = np.random.default_rng(123)
    a = rng.standard_normal((n, n), dtype=np.float64)
    b = rng.standard_normal((n, n), dtype=np.float64)
    total = 0.0
    deadline = time.perf_counter() + min_wall_s
    reps = 0
    while reps < min_reps or time.perf_counter() < deadline:
        c = a @ b
        total += float(c[0, 0])
        a, b = b, c * 1e-6
        reps += 1
    return total


def fft_kernel(n: int) -> float:
    x = np.linspace(0.0, 500.0, n, dtype=np.float64)
    y = np.sin(x) + np.cos(x * 0.25)
    spectrum = np.fft.rfft(y)
    return float(np.abs(spectrum).sum())


def sort_kernel(n: int) -> float:
    rng = np.random.default_rng(456)
    x = rng.random(n, dtype=np.float64)
    x.sort()
    return float(x[n // 2])


def main() -> None:
    cfg = _load_config()
    t0 = time.perf_counter()
    with with_profile():
        with with_subprofile("matmul_kernel"):
            a = matmul_kernel(
                int(cfg.get("matmul_n", 1200)),
                int(cfg.get("matmul_reps", 20)),
                float(cfg.get("matmul_min_s", 4.0)),
            )
        with with_subprofile("fft_kernel"):
            b = fft_kernel(int(cfg.get("fft_n", 2_097_152)))
        with with_subprofile("sort_kernel"):
            c = sort_kernel(int(cfg.get("sort_n", 1_800_000)))
    elapsed = time.perf_counter() - t0

    print(f"[numpy_native] matmul={a:.3f} fft={b:.3f} sort={c:.6f}")
    emit_summary(speed_sec=elapsed, peak_mb=peak_memory_mb())


if __name__ == "__main__":
    main()
