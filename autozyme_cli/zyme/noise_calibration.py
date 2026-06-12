"""Wall-time noise calibration + borderline detection.

Stores per-(tier, thread) wall-time CV for the upstream baseline, captured
during `zyme baseline reference --reps N`. `zyme run` consults this to decide
whether a fresh measurement is decisive or borderline; borderline measurements
auto-trigger a few extra reps before the decision row is committed, so the agent
stops burning rounds on judgement-call reruns.

File: <task_dir>/.zyme/baseline_noise.json

Schema (v2 — back-compat with v1; v1 readers ignore the new fields):
{
  "schema_version": 2,
  "tiers": {
    "tiny": {
      "8": {                          # thread count as string key
        "dataset_name": "pbmc68k",
        "n_reps": 5,
        "speed_mean": 12.345,
        "speed_stdev": 0.234,
        "speed_cv": 0.019,            # stdev / mean (fraction, not %)
        "peak_mean": 850.2,
        "peak_stdev": 5.1,
        "commit": "abc1234",
        "calibrated_at": "2026-05-20T12:34:56",

        # v2 baseline-cache fields (all optional on read):
        "upstream_versions": {"statsmodels": "0.14.6", "numpy": "1.26.4"},
        "output_artifact_relpath": "reference_output_tiny",
        "output_artifact_sha256": "8e9c0f...",
        "output_artifact_size_bytes": 192414,
        "produced_by": "iteration"    # or "attest" / "scaling"
      }
    }
  }
}
"""
from __future__ import annotations
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 2
NOISE_FILE_REL = ".zyme/baseline_noise.json"

# Borderline thresholds. Tuned to err on the side of "rerun when in doubt" at
# low cost: a 3 × CV speed window catches most fluke wins/losses while leaving
# decisive results (5% CV → 15% window) untouched. metric_zone is the relative
# distance to the effective threshold below which we trigger.
DEFAULT_SPEED_CV_MULT = 3.0
DEFAULT_METRIC_ZONE = 0.05
# Legacy: was used by auto-rerun gate (removed). Kept for any out-of-tree
# tooling that still imports it; no in-tree consumer.
DEFAULT_AUTO_RERUN_EXTRA = 2


def _noise_path(task_dir: Path) -> Path:
    return task_dir / NOISE_FILE_REL


def load_noise(task_dir: Path) -> dict:
    """Load the noise calibration file. Returns empty dict if missing."""
    p = _noise_path(task_dir)
    if not p.exists():
        return {"schema_version": SCHEMA_VERSION, "tiers": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"schema_version": SCHEMA_VERSION, "tiers": {}}
    if "tiers" not in data:
        data["tiers"] = {}
    return data


def save_noise(task_dir: Path, data: dict) -> None:
    p = _noise_path(task_dir)
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def record_tier_noise(
    task_dir: Path,
    tier: str,
    thread: int,
    dataset_name: str,
    speeds: list[float],
    peaks: list[float],
    commit: str,
    *,
    upstream_versions: Optional[dict] = None,
    output_artifact_relpath: Optional[str] = None,
    output_artifact_sha256: Optional[str] = None,
    output_artifact_size_bytes: Optional[int] = None,
    produced_by: str = "iteration",
) -> dict:
    """Compute + persist (mean, stdev, cv) for one (tier, thread). Returns the entry.

    v2 fields (all optional): `upstream_versions`, `output_artifact_relpath`,
    `output_artifact_sha256`, `output_artifact_size_bytes`, `produced_by`
    enable the baseline-cache lookup used by `autozyme.verify_patch`. When
    omitted, the entry behaves identically to v1 for borderline-gate readers
    but cache-readers will treat it as "no version stamp" (permissive HIT).
    """
    n = len(speeds)
    if n < 1:
        raise ValueError("record_tier_noise requires at least one speed sample")
    sp_mean = statistics.mean(speeds)
    sp_std = statistics.stdev(speeds) if n > 1 else 0.0
    sp_cv = (sp_std / sp_mean) if sp_mean > 0 else 0.0
    pk_mean = statistics.mean(peaks) if peaks else 0.0
    pk_std = statistics.stdev(peaks) if len(peaks) > 1 else 0.0

    entry = {
        "dataset_name": dataset_name,
        "n_reps": n,
        "speed_mean": round(sp_mean, 6),
        "speed_stdev": round(sp_std, 6),
        "speed_cv": round(sp_cv, 6),
        "peak_mean": round(pk_mean, 3),
        "peak_stdev": round(pk_std, 3),
        "commit": commit[:7] if commit else "",
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if upstream_versions:
        entry["upstream_versions"] = dict(upstream_versions)
    if output_artifact_relpath:
        entry["output_artifact_relpath"] = output_artifact_relpath
    if output_artifact_sha256:
        entry["output_artifact_sha256"] = output_artifact_sha256
    if output_artifact_size_bytes is not None:
        entry["output_artifact_size_bytes"] = int(output_artifact_size_bytes)
    entry["produced_by"] = produced_by

    data = load_noise(task_dir)
    data["schema_version"] = SCHEMA_VERSION
    data["tiers"].setdefault(tier, {})[str(int(thread))] = entry
    save_noise(task_dir, data)
    return entry


def get_tier_noise(task_dir: Path, tier: str, thread: int) -> Optional[dict]:
    """Return calibrated noise entry for (tier, thread), or None if missing."""
    data = load_noise(task_dir)
    return (data.get("tiers", {}).get(tier) or {}).get(str(int(thread)))


def is_speed_borderline(
    delta_pct_vs_reference: float,
    speed_cv: float,
    multiplier: float = DEFAULT_SPEED_CV_MULT,
) -> tuple[bool, str]:
    """True iff |delta| (in percent) falls within `multiplier × cv × 100`.

    `delta_pct_vs_reference` is the percent change vs whatever reference the
    caller cares about (best-median, baseline, prior rep — caller's choice).
    `speed_cv` is the fractional CV (e.g. 0.019 for 1.9%).
    """
    if speed_cv is None or speed_cv <= 0:
        return False, "no calibrated speed CV"
    window_pct = multiplier * speed_cv * 100.0
    inside = abs(delta_pct_vs_reference) < window_pct
    reason = (
        f"|Δ|={abs(delta_pct_vs_reference):.2f}% < {window_pct:.2f}% "
        f"({multiplier:g}× baseline CV {speed_cv*100:.2f}%)"
        if inside else
        f"|Δ|={abs(delta_pct_vs_reference):.2f}% ≥ {window_pct:.2f}% "
        f"({multiplier:g}× baseline CV {speed_cv*100:.2f}%)"
    )
    return inside, reason


def is_metric_borderline(
    metric_value: float,
    effective_threshold: float,
    comparator: str,
    zone: float = DEFAULT_METRIC_ZONE,
) -> tuple[bool, str]:
    """True iff `metric_value` is within `zone` (relative) of `effective_threshold`.

    For gte (larger=better): borderline when value is just barely above the
    threshold. Distance scale is `1 - threshold` (or threshold itself if that
    is larger), since gte metrics commonly live near 1.0 where pure relative
    distance becomes degenerate.

    For lte (smaller=better): borderline when value is just barely below the
    threshold. Distance scale is `threshold` (since lte thresholds live near 0).
    """
    if metric_value is None or effective_threshold is None:
        return False, "missing value or threshold"
    if comparator == "gte":
        # gte thresholds typically live in (0,1] (correlations, jaccard, etc.).
        # The natural distance scale is room above the threshold: `1 - thr`.
        # For thr=0.95, scale=0.05; passing 0.96 → margin 0.01 → rel 20% (NOT
        # borderline at 5% zone). Passing 0.951 → margin 0.001 → rel 2% → IS
        # borderline. Floor at 1e-6 avoids div-by-zero when threshold ≈ 1.0.
        scale = max(1.0 - effective_threshold, 1e-6)
        margin = metric_value - effective_threshold
    elif comparator == "lte":
        # lte thresholds typically live near 0 (RMSE, abs_diff). Scale is the
        # threshold itself (room below it). For thr=0.05, passing 0.049 →
        # margin 0.001 → rel 2% → borderline.
        scale = max(abs(effective_threshold), 1e-6)
        margin = effective_threshold - metric_value
    else:
        return False, f"unknown comparator '{comparator}'"
    if margin < 0:
        # Already failing — not borderline, just failed.
        return False, f"value already fails threshold (margin {margin:+.4g})"
    if scale <= 1e-5:
        # Threshold at the natural upper/lower bound (e.g. gte 1.0). Any pass
        # is decisive — there's no "room" above/below the threshold to be near.
        return False, f"threshold at natural limit (no rerun zone)"
    rel = margin / scale
    inside = rel < zone
    reason = (
        f"margin {margin:+.4g} / scale {scale:.4g} = {rel*100:.1f}% "
        f"{'<' if inside else '≥'} {zone*100:.0f}% zone"
    )
    return inside, reason


def aggregate_samples(samples: list[float]) -> tuple[float, float, float]:
    """(mean, stdev, cv) — cv as a fraction. Returns (mean, 0, 0) for n=1."""
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = statistics.mean(samples)
    s = statistics.stdev(samples) if n > 1 else 0.0
    cv = (s / m) if m > 0 else 0.0
    return m, s, cv
