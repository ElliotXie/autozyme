"""evaluate.py — Compare optimized output against reference.

Called by `the zyme runner` after each experiment.
Prints `metric_name: value` lines for the zyme runner to parse into results.tsv (`metrics_json` column).

This file is READ-ONLY during experiments — the main agent does not edit it.
Init agent writes the metrics inline based on the function's mathematical nature
(deterministic numerical → pearson + max-abs-diff; clustering → ARI/NMI;
embedding → rotation-invariant; ranked lists → top-K Jaccard).
"""
import os
import sys
import json
import numpy as np
# === Add metric-specific imports as needed ===
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# from scipy import sparse


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.environ.get("ZYME_REFERENCE_DIR") or os.path.join(SCRIPT_DIR, "reference_output")
# ZYME_TEST_DIR lets `zyme baseline noise` point evaluate at a non-pipeline
# directory so it can compare two reference runs (primary vs calibration seed).
# Default = <task>/pipeline/ as before.
TEST_DIR = os.environ.get("ZYME_TEST_DIR") or os.path.join(SCRIPT_DIR, "pipeline")

if not os.path.exists(REF_DIR):
    sys.exit("Reference output not found. Run reference.py first.")

# Framework helpers for F3 auto-compare (default-on, opt-out).
_cur = SCRIPT_DIR
while not os.path.isdir(os.path.join(_cur, "autozyme-framework", "autozyme_cli", "zyme")):
    _parent = os.path.dirname(_cur)
    if _parent == _cur:
        raise RuntimeError(f"autozyme-framework not found above {__file__}")
    _cur = _parent
sys.path.insert(0, os.path.join(_cur, "autozyme-framework", "autozyme_cli", "zyme"))
from helpers import auto_structure_check_all_slots, emit_auto_structure_summary  # noqa: E402


# === 1. Load reference + test outputs ===
# Mirror exactly what reference.py saved. Symmetric load on both sides.
# Example:
#   ref_X = np.load(os.path.join(REF_DIR, "X.npz"))["arr"]
#   test_X = np.load(os.path.join(TEST_DIR, "X.npz"))["arr"]


# === 2. Compute metrics declared in task.yaml ===
# Each metric is (ref, opt) -> float. Write them inline using whatever you need
# from numpy/scipy/sklearn.
#
# Convention: print "score" form (higher = better) for `gte` metrics, raw value
# for `lte` metrics (max_abs_diff, RMSE).
metrics = {}

# metrics["pearson_X"] = float(np.corrcoef(ref_X.ravel(), test_X.ravel())[0, 1])
# metrics["max_abs_diff_X"] = float(np.max(np.abs(ref_X - test_X)))


# === 3. F3 default-on structural check ===
# Auto-emits THREE boolean metrics per top-level slot in the reference output
# (1.0 = pass, 0.0 = fail):
#   <slot>_present         — test has the slot (catches dropped output)
#   <slot>_shape_match     — same shape / length as ref
#   <slot>_non_degenerate  — if ref has variance, test does too
#                             (catches "stub to constant" — e.g. an override
#                              returning 0 / NA / single value while ref varies)
#
# This is structure-level only, NOT value comparison. Pearson / max_abs_diff /
# Jaccard remain explicit metrics in `metrics` above — declare them for slots
# where you want strict value agreement.
#
# Waiver: append slot names to WAIVED_SLOTS only when a slot legitimately
# differs across runs (random init log, iteration history of variable length).
# Each waiver REQUIRES a one-line `# diagnostic: <reason>` comment — audit
# flags any waiver without one.
WAIVED_SLOTS = []  # e.g. ["loglikelihood_history"]  # diagnostic: ...

# `ref` and `test` should be dicts loaded above for this helper to work.
# If your reference output isn't a dict (e.g. raw ndarray), keep your explicit
# metrics above and leave WAIVED_SLOTS empty; the helper returns {} for
# non-dict inputs.
try:
    struct_metrics = auto_structure_check_all_slots(ref, test, waived=WAIVED_SLOTS)  # noqa: F821
except NameError:
    struct_metrics = {}


# === 4. Print metrics in `key: value` format ===
# the zyme runner greps these lines from stdout and packs into results.tsv.metrics_json.
for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.6f}")
    else:
        print(f"{name}: {value}")
# Auto-structure: one summary line when all perfect; individual lines for failures.
emit_auto_structure_summary(struct_metrics)

# Optional: extra diagnostic info (printed but not parsed as a metric).
# print(f"\n[evaluate] N samples compared: {len(ref_X)}")
