"""pipeline/run.py — synthetic_groundtruth fixture entry point.

Three pure-Python CPU-bound functions with controlled relative work shares.
The point of this fixture is to test profile backends' ranking — NOT to do
useful science. Iteration counts were tuned empirically (May 2026) to
roughly hit 50/30/20 wall-time shares on an Apple Silicon Mac. They will
drift on different hardware; the bench's hotspot-recall check is forgiving
enough (top-3 must contain {cpu_a, cpu_b, cpu_c} with cpu_a ranked first).

Why pure Python (no NumPy): we want the entire timing to live in
deterministic Python frames so cProfile and Scalene rank by the same
mechanism. Mixing native ops introduces backend-specific attribution
differences that complicate the ranking comparison.
"""
import os
import sys
import time

# Locate framework helpers. This file lives at:
#   autozyme_cli/tests/fixtures/synthetic_groundtruth/pipeline/run.py
# walk up: pipeline -> synthetic_groundtruth -> fixtures -> tests
#       -> autozyme_cli, then into zyme/ for helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_HELPERS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "zyme"))
if _HELPERS_DIR not in sys.path:
    sys.path.insert(0, _HELPERS_DIR)
from helpers import with_profile, with_subprofile, peak_memory_mb, emit_summary  # noqa: E402


# Iteration counts tuned empirically (May 2026, Apple Silicon) so that
# wall-time(cpu_a) > wall-time(cpu_b) > wall-time(cpu_c) by clear margins.
# Initial naive counts (50M / 30M / 20M) put cpu_c first because list.append
# is far more expensive per-iter than int multiply. Retuned to ratios
# ~2.3s / 1.5s / 1.0s — cpu_a 1.5x cpu_b, cpu_a 2.2x cpu_c — robust to noise.
def cpu_a():
    """Target: largest. Integer multiply accumulation (cheapest per-iter)."""
    s = 0
    for i in range(80_000_000):
        s += i * i
    return s


def cpu_b():
    """Target: middle. Float sqrt (different op kind from a)."""
    s = 0.0
    for i in range(20_000_000):
        s += float(i) ** 0.5
    return s


def cpu_c():
    """Target: smallest. List append + modulo (allocates, unlike a/b)."""
    a = []
    for i in range(9_000_000):
        a.append(i % 7)
    return len(a)


def main():
    t0 = time.perf_counter()
    with with_profile():
        with with_subprofile("cpu_a"):
            r_a = cpu_a()
        with with_subprofile("cpu_b"):
            r_b = cpu_b()
        with with_subprofile("cpu_c"):
            r_c = cpu_c()
    elapsed = time.perf_counter() - t0
    print(f"[synthetic] cpu_a={r_a} cpu_b={r_b:.3f} cpu_c={r_c}", flush=True)
    emit_summary(speed_sec=elapsed, peak_mb=peak_memory_mb())


if __name__ == "__main__":
    main()
