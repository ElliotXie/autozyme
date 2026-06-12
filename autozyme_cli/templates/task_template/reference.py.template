"""reference.py — Generate ground truth for <TASK_NAME>.

Run once: `python reference.py`
Saves output to `reference_output/` for evaluate.py to compare against.

This file is READ-ONLY during experiments — the main agent does not edit it.
Init agent fills in the placeholders below from the upstream source.
"""
import os
import sys
import time

# === Imports for the upstream toolkit ===
# import scanpy as sc
# import anndata as ad
# from scipy import sparse

# === Helpers from autozyme_cli/zyme/helpers.py ===
# Provides emit_summary() (prints `speed_sec:` / `peak_mb:` / `cpu_sec:` in
# the exact format the zyme runner + `baseline record --from-log` parse) and
# peak_memory_mb() (cross-platform, ps-aware). Walks up to find
# autozyme-framework/ — works from any depth in the workspace.
_cur = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_cur, "autozyme-framework", "autozyme_cli")):
    _parent = os.path.dirname(_cur)
    if _parent == _cur:
        raise RuntimeError(f"autozyme-framework not found above {__file__}")
    _cur = _parent
sys.path.insert(0, os.path.join(_cur, "autozyme-framework", "autozyme_cli", "zyme"))
from helpers import emit_summary, peak_memory_mb  # noqa: E402
# Uncomment if you want tier params from task.yaml instead of hardcoding:
# from helpers import get_tier_params
# P = get_tier_params()  # reads ZYME_TIER, returns dict from task.yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "<DATA_FILE>")     # filled by init agent
OUTPUT_DIR = os.environ.get("ZYME_REFERENCE_DIR") or os.path.join(SCRIPT_DIR, "reference_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Stochastic algorithm? Read seed from env var ===
# If your target uses an RNG (MCMC / EM / random init / approximate methods),
# read the seed from `ZYME_RANDOM_SEED` (default 42). `zyme baseline noise` sets
# it to calibration values (default 43,44,45) to measure intrinsic noise — without
# this hook, calibration runs the same chain twice and reports zero noise.
# Skip this block for deterministic algorithms.
# SEED = int(os.environ.get("ZYME_RANDOM_SEED", "42"))
# random.seed(SEED); np.random.seed(SEED)
# # framework-specific seed setters (uncomment as needed):
# # tf.random.set_seed(SEED); torch.manual_seed(SEED)

# === Upstream threading (uncomment if `task.yaml::upstream_parallelism` is non-empty) ===
# Mirror the threading wiring from pipeline/run.py so `zyme baseline reference
# --thread N` actually engages the upstream knob — without this, baseline stays
# single-threaded even when --thread > 1 is requested.
#   from helpers import get_threads
#   N_THREADS = get_threads(default=1)
#   os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
#   os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))
#   # Then pass N_THREADS into the upstream knob — `n_jobs=N_THREADS`,
#   # `BPPARAM=MulticoreParam(N_THREADS)`, etc.


# === 1. Load input data ===
if not os.path.exists(DATA_PATH):
    sys.exit(f"Input not found at {DATA_PATH}. Did `zyme init` finish?")

print("[reference] Loading data...")
# data = sc.read_h5ad(DATA_PATH)
# print(f"[reference] Dataset: {data.n_obs} cells, {data.n_vars} genes")


# === 2. Run upstream baseline ===
print("[reference] Running upstream baseline...")
start = time.perf_counter()
# result = upstream_function(data, **params)
elapsed = time.perf_counter() - start
print(f"[reference] Baseline took {elapsed:.1f}s")


# === 3. Serialize output for evaluate.py ===
# Use whatever format captures everything evaluate.py needs to compare.
# Common patterns:
#   - dense matrix:   np.savez_compressed(os.path.join(OUTPUT_DIR, "X.npz"), arr=X)
#   - sparse matrix:  scipy.sparse.save_npz(os.path.join(OUTPUT_DIR, "X.npz"), X)
#   - dataframe:      df.to_csv(os.path.join(OUTPUT_DIR, "df.csv"), index=False)
#   - cluster labels: np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
#
# `pipeline/run.py` MUST save in IDENTICAL format and key names — evaluate.py
# loads from both sides symmetrically.

# === 4. Print baseline summary ===
# emit_summary() prints `speed_sec: <float>` / `peak_mb: <float>` / `cpu_sec:`
# in the exact format `zyme baseline record --from-log` and the runner parse.
# Do NOT replace with ad-hoc `[reference] elapsed:` lines — they bypass the parser.
print(f"\n[reference] Saved to {OUTPUT_DIR}")
emit_summary(speed_sec=elapsed)
print(f"[reference] Baseline speed: {elapsed:.1f} seconds — this is the time to beat.")
