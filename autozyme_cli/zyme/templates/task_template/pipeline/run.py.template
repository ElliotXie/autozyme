"""pipeline/run.py — agent's optimization patch.

This is the ONLY file the main agent edits. Pattern:
  1. Import `install_override` from the framework helpers.
  2. Define optimized replacement(s) at module level (or thin wrappers).
  3. Call `install_override(func_name, module_path, replacement)` — handles
     attribute patching + verification + emits `[override active]` runtime marker.
  4. Run the same call as reference.py with same args.
  5. Save output in IDENTICAL format/keys as reference.py — evaluate.py
     compares symmetrically.

Print `speed_sec: <float>` to stdout — the zyme runner parses this for the
timing column. (Or use `emit_summary()` from helpers.)

The first round (= round 0) must be unmodified — pipeline = reference, no
overrides — to confirm baseline parity.
"""
import os
import sys
import time

# Framework helpers: install_override (with verification + runtime marker),
# time_it, peak_memory_mb, emit_summary, get_tier_params, get_threads.
# Locate autozyme-framework by walking up — no symlink dependency, works
# from any depth inside the workspace.
_cur = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_cur, "autozyme-framework", "autozyme_cli")):
    _parent = os.path.dirname(_cur)
    if _parent == _cur:
        raise RuntimeError(f"autozyme-framework not found above {__file__}")
    _cur = _parent
sys.path.insert(0, os.path.join(_cur, "autozyme-framework", "autozyme_cli", "zyme"))
from helpers import (  # noqa: E402
    install_override, peak_memory_mb, emit_summary,
    get_tier_params, get_threads, with_profile, with_subprofile,
)

# === Imports — same as reference.py ===
# import scanpy as sc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(SCRIPT_DIR)

# Dataset path: zyme runner sets ZYME_DATA_PATH per tier. Falls back to a
# task-default path if invoked outside the runner. ZYME_TIER tells you which
# tier you're on (tiny / medium / large) — useful for tier-conditional code.
DATA_PATH = os.environ.get("ZYME_DATA_PATH") or os.path.join(TASK_DIR, "<DEFAULT_DATA_FILE>")
TIER = os.environ.get("ZYME_TIER", "tiny")
REFERENCE_DIR = os.environ.get("ZYME_REFERENCE_DIR") or os.path.join(TASK_DIR, "reference_output")

# Threading: route every hardcoded thread/core count through `get_threads()`
# so `zyme verify` actually exercises the matrix axis. Without this, the
# verify probe will abort.
#   N_THREADS = get_threads(default=4)
#   os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
#   os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))
#   # numba: numba.set_num_threads(N_THREADS)
#   # joblib: Parallel(n_jobs=N_THREADS, ...)

# Optional: tier-specific subset sizes from task.yaml `datasets[].params`.
# Use this in place of a hardcoded TIER_PARAMS dict — adding a new tier
# later becomes yaml-only, no code edit.
#   P = get_tier_params()
#   N_CELLS = P.get("n_cells", 10000)
#   N_GENES = P.get("n_genes", 2000)


# ============================================================
# Optimized implementation (define here)
# ============================================================
# Full body replacement:
# def fast_<target_function>(data, ...):
#     """Faster replacement, semantically equivalent within concordance threshold."""
#     ...
#
# Or thin wrapper (preferred for one-parameter changes):
# orig = sys.modules["<module_path>"].<target_function>
# def fast_<target_function>(*args, **kwargs):
#     kwargs["<param>"] = "<new_value>"   # the only change
#     return orig(*args, **kwargs)


# ============================================================
# Install override (uncomment when you have an optimized impl)
# ============================================================
# install_override("<target_function>", "<module_path>", fast_<target_function>)


# ============================================================
# Execute (mirror reference.py exactly — same input, same call, same save format)
# ============================================================
print("[pipeline] Loading data...")
# data = sc.read_h5ad(DATA_PATH)

print("[pipeline] Running (possibly patched) function...")
start = time.perf_counter()
with with_profile():
    # Use with_subprofile("name") inside this block if you need named
    # timing for opaque native/library sub-calls during profiling.
    # result = sc.tl.<target_function>(data, **params)
    pass
elapsed = time.perf_counter() - start

# Save output to SCRIPT_DIR in EXACTLY the same format reference.py used,
# with identical keys / filenames. evaluate.py reads from both sides symmetrically.
# Example:
#   np.savez_compressed(os.path.join(SCRIPT_DIR, "X.npz"), arr=result.X)

emit_summary(speed_sec=elapsed, peak_mb=peak_memory_mb())
print(f"[pipeline] Done in {elapsed:.1f}s")
