"""Baseline-measurement cache for `verify_patch`.

`zyme attest` re-runs the unpatched upstream baseline from scratch every
invocation — ~93% of wall time on a 5-tier statsmodels attest is baseline.
The iteration phase already produces the canonical baseline outputs at
``task_dir/reference_output_<tier>/`` (or, in older tasks,
``task_dir/reference_outputs/<tier>/``) and (when run with ``--reps > 1``)
records per-(tier, thread) timing stats in
``task_dir/.zyme/baseline_noise.json``. This module lets `verify_patch`
reuse both.

The cache lives on existing artifacts (no new locations):

  * Output artifact dir: ``<task_dir>/reference_output_<tier>/`` (already
    written by iteration's ``reference.py``). Older tasks may instead use
    ``<task_dir>/reference_outputs/<tier>/``; lookup accepts both layouts.
  * Timing + metadata: ``<task_dir>/.zyme/baseline_noise.json`` v2 entries
    (extends the existing v1 schema with optional ``upstream_versions``,
    ``output_artifact_sha256``, ``output_artifact_size_bytes``,
    ``produced_by``).
  * Fallback timing source: ``<task_dir>/.zyme/baselines_history.tsv`` and
    ``<task_dir>/results.tsv`` for tasks that never ran noise calibration.

The validity check is permissive by design (see ``load_cached_baseline``):
when ``upstream_versions`` is absent on a candidate entry but the output
artifact dir exists, we HIT and emit a "no version stamp — first run will
backfill" log line; the writeback path stamps the entry on success. This
lets the first attest after deployment immediately reap the savings on
tasks that have a populated reference artifact dir from iteration
but no v2 noise-calibration entry.

This module is import-safe from `autozyme_py` (no `zyme` CLI dependency);
the JSON schema is duplicated here intentionally so the package stays
self-contained.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from autozyme._core import _installed_version, _top_level_pkg


# Same paths/schema as zyme.noise_calibration; duplicated to avoid a CLI dep.
NOISE_FILE_REL = ".zyme/baseline_noise.json"
HISTORY_FILE_REL = ".zyme/baselines_history.tsv"
RESULTS_FILE_REL = "results.tsv"
SCHEMA_VERSION = 2

# Size threshold above which we skip the artifact sha (fall back to size+exists).
# Override with ZYME_BASELINE_CACHE_MAX_HASH_MB env var; default 512 MB.
_DEFAULT_MAX_HASH_MB = 512


# ============================================================
# CachedBaseline result type
# ============================================================
@dataclass(frozen=True)
class CachedBaseline:
    """Successful cache-lookup result. All numeric fields are seconds / MB."""

    timing_mean: float
    timing_stdev: float            # 0.0 when only 1 sample available
    peak_mb: Optional[float]
    ref_dir_path: str              # absolute path to the persistent ref dir
    n_reps_observed: int
    has_version_stamp: bool        # False → permissive HIT, writeback should backfill
    produced_by: str               # "iteration" | "attest" | "scaling" | "history-fallback"
    timing_source: str             # "noise_json" | "baselines_history" | "results_tsv"


# ============================================================
# Threads / versions / hashing helpers
# ============================================================
def _resolve_threads(default: int = 1) -> int:
    """Mirror _verify._collect_system_info's thread resolution."""
    for var in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        raw = os.environ.get(var)
        if raw:
            try:
                return max(1, int(raw))
            except (ValueError, TypeError):
                continue
    return default


def _collect_versions(p) -> dict[str, str]:
    """Map top-level package name → installed version, for every upstream target.

    Sorted-key dict for stable JSON output. Packages whose version can't be
    resolved (rare) get omitted, not None-stamped — pretending we know nothing
    is safer than recording a wrong stamp.
    """
    pkgs = sorted({_top_level_pkg(u) for u, _, _ in p.targets})
    out: dict[str, str] = {}
    for pkg in pkgs:
        ver = _installed_version(pkg)
        if ver:
            out[pkg] = ver
    return out


def _hash_artifact_dir(path: Path,
                       max_mb: int = _DEFAULT_MAX_HASH_MB,
                      ) -> tuple[str, int]:
    """Return (sha256_hex, total_size_bytes). sha is "" if total > max_mb.

    Files are hashed in sorted relpath order (forward-slash normalized for
    cross-OS determinism). The relpath itself is mixed into the hash so a
    file rename inside the dir invalidates the cache.
    """
    if not path.is_dir():
        return "", 0
    try:
        max_bytes = int(os.environ.get(
            "ZYME_BASELINE_CACHE_MAX_HASH_MB", max_mb)) * 1024 * 1024
    except (ValueError, TypeError):
        max_bytes = max_mb * 1024 * 1024

    files = sorted(
        (p for p in path.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(path)).replace("\\", "/"),
    )
    total = sum(f.stat().st_size for f in files)
    if total > max_bytes:
        return "", total
    h = hashlib.sha256()
    for f in files:
        rel = str(f.relative_to(path)).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest(), total


# ============================================================
# JSON / TSV readers (no zyme CLI dependency)
# ============================================================
def _noise_path(task_dir: str | Path) -> Path:
    return Path(task_dir) / NOISE_FILE_REL


def _load_noise(task_dir: str | Path) -> dict:
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


def _save_noise(task_dir: str | Path, data: dict) -> None:
    p = _noise_path(task_dir)
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n",
                 encoding="utf-8")


def _read_baselines_history(task_dir: str | Path, tier: str,
                            threads: int, limit: int = 5,
                           ) -> list[tuple[float, Optional[float]]]:
    """Recent (speed_sec, peak_mb) rows for (tier, thread) from history TSV.

    Returns up to `limit` rows, most recent last. Empty list if file missing
    or no matching rows.
    """
    path = Path(task_dir) / HISTORY_FILE_REL
    if not path.is_file():
        return []
    rows: list[tuple[float, Optional[float]]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for r in reader:
                if (r.get("tier") or "").strip() != tier:
                    continue
                row_thread = (r.get("thread") or "").strip()
                # Thread column is sometimes empty (legacy rows). Treat empty
                # as matching only when we're querying thread=1 (the legacy
                # default).
                if row_thread:
                    try:
                        if int(row_thread) != int(threads):
                            continue
                    except (ValueError, TypeError):
                        continue
                elif int(threads) != 1:
                    continue
                try:
                    speed = float(r.get("speed_sec") or "")
                except (ValueError, TypeError):
                    continue
                try:
                    peak = float(r.get("peak_mb") or "")
                except (ValueError, TypeError):
                    peak = None
                rows.append((speed, peak))
    except (OSError, csv.Error):
        return []
    return rows[-limit:]


def _dataset_name_for_tier(task_dir: str | Path, tier: str) -> str:
    """Best-effort task.yaml dataset name for ``tier``."""
    try:
        import yaml
        with open(Path(task_dir) / "task.yaml", encoding="utf-8") as fh:
            task = yaml.safe_load(fh) or {}
        for ds in task.get("datasets") or []:
            if ds.get("tier") == tier:
                return str(ds.get("name") or "").strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _baseline_thread_invariant(task_dir: str | Path) -> bool:
    """True when task.yaml declares the baseline does NOT parallelize
    (``baseline_threading: not_applicable``).

    Such a baseline is measured ONCE per tier and reused across every thread
    count — only the patch is re-timed per thread (see ``load_cached_baseline``).
    Absent/blank marker → False → unchanged exact-thread behavior.
    """
    try:
        import yaml
        with open(Path(task_dir) / "task.yaml", encoding="utf-8") as fh:
            task = yaml.safe_load(fh) or {}
        val = str(task.get("baseline_threading") or "").strip().lower()
        return val in ("not_applicable", "none", "non_parallel", "serial")
    except Exception:  # noqa: BLE001
        return False


def _thread_matches(row_thread: str, threads: int) -> bool:
    raw = (row_thread or "").strip()
    if raw:
        try:
            return int(float(raw)) == int(threads)
        except (ValueError, TypeError):
            return False
    return int(threads) == 1


def _read_results_baseline(task_dir: str | Path, tier: str, threads: int,
                          ) -> Optional[tuple[float, Optional[float]]]:
    """Last `status="baseline"` row's (speed_sec, peak_mb) for `tier` from results.tsv.

    Iteration writes one such row per (dataset/tier) at setup time via
    `cmd_record_baseline`. Returns None when missing or malformed.
    """
    path = Path(task_dir) / RESULTS_FILE_REL
    if not path.is_file():
        return None
    dataset_name = _dataset_name_for_tier(task_dir, tier)
    try:
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            best: Optional[tuple[float, Optional[float]]] = None
            for r in reader:
                if (r.get("status") or "").strip() != "baseline":
                    continue
                if not _thread_matches(r.get("thread") or "", threads):
                    continue
                # The dataset column carries either a name or a tier-coded
                # identifier; tasks vary. Match on substring with the tier.
                ds = (r.get("dataset") or "").strip()
                if ds not in {tier, dataset_name} and tier not in ds:
                    continue
                try:
                    speed = float(r.get("speed_sec") or "")
                except (ValueError, TypeError):
                    continue
                try:
                    peak = float(r.get("peak_mb") or "")
                except (ValueError, TypeError):
                    peak = None
                best = (speed, peak)   # later rows overwrite (most recent wins)
            return best
    except (OSError, csv.Error):
        return None


def _candidate_ref_dirs(task_dir: str | Path, tier: str) -> list[Path]:
    base = Path(task_dir)
    return [
        base / f"reference_output_{tier}",
        base / "reference_outputs" / tier,
    ]


def _first_populated_ref_dir(task_dir: str | Path, tier: str) -> Path | None:
    for ref_dir in _candidate_ref_dirs(task_dir, tier):
        if ref_dir.is_dir() and any(ref_dir.rglob("*")):
            return ref_dir
    return None


def _ref_relpath(task_dir: str | Path, ref_dir: str | Path) -> str:
    try:
        return str(Path(ref_dir).resolve().relative_to(Path(task_dir).resolve()))
    except ValueError:
        return str(ref_dir)


# ============================================================
# Cache lookup
# ============================================================
def load_cached_baseline(name: str, task_dir: str, tier: str,
                         threads: Optional[int], p,
                        ) -> Optional[CachedBaseline]:
    """Permissive cache lookup. Returns None if NO usable cache entry exists.

    Validity steps (any failure → None):
      1. ``reference_output_<tier>/`` or ``reference_outputs/<tier>/`` exists
         and is non-empty.
      2. Timing source available: prefer v2 noise.json entry; fall back to
         baselines_history.tsv aggregation; last resort results.tsv row.
      3. (When ``upstream_versions`` is stamped) every package in
         ``{top_level(u) for u in p.targets}`` matches the installed version.
         Missing stamp → permissive HIT (caller backfills).
      4. (When ``output_artifact_sha256`` is stamped) directory contents
         hash to the recorded value. Stored sha "" (oversize) skips this.

    All version/sha checks are LENIENT when the stamp is absent — first-run
    artifacts on tasks that never wrote a v2 entry still HIT.
    """
    task_dir = os.path.abspath(task_dir)
    if threads is None:
        threads = _resolve_threads()

    ref_dir = _first_populated_ref_dir(task_dir, tier)
    if ref_dir is None:
        return None

    noise = _load_noise(task_dir)
    schema_v = int(noise.get("schema_version") or 1)

    # Baseline thread-invariance: a task whose baseline does NOT parallelize
    # (`baseline_threading: not_applicable` in task.yaml) measures the baseline
    # ONCE per tier and reuses it across all thread counts — only the patch is
    # re-timed per thread. Remap the cache lookup to whichever thread already
    # holds a baseline for this tier (prefer 1t); the patch still runs at the
    # requested `threads`. No marker → exact-thread lookup, unchanged.
    lookup_threads = int(threads)
    if _baseline_thread_invariant(task_dir):
        tier_entries = (noise.get("tiers", {}).get(tier) or {})
        if str(int(threads)) not in tier_entries and tier_entries:
            digit_keys = sorted(int(k) for k in tier_entries if str(k).isdigit())
            if digit_keys:
                lookup_threads = 1 if 1 in digit_keys else digit_keys[0]

    entry = (noise.get("tiers", {}).get(tier) or {}).get(str(lookup_threads))

    timing_mean: Optional[float] = None
    timing_stdev: float = 0.0
    peak_mb: Optional[float] = None
    n_reps_observed: int = 0
    timing_source: str = ""
    has_version_stamp: bool = False
    produced_by: str = "unknown"

    # --- Step 2/3/4: primary source = v2 noise.json entry --------------
    if entry and "speed_mean" in entry:
        timing_mean = float(entry["speed_mean"])
        timing_stdev = float(entry.get("speed_stdev") or 0.0)
        peak_mb = entry.get("peak_mean")
        peak_mb = float(peak_mb) if peak_mb is not None else None
        n_reps_observed = int(entry.get("n_reps") or 1)
        timing_source = "noise_json"
        produced_by = str(entry.get("produced_by") or "iteration")

        stamped_versions = entry.get("upstream_versions") or {}
        if stamped_versions:
            installed_versions = _collect_versions(p)
            required = set(installed_versions.keys())
            mismatch = False
            for pkg in required:
                stamped = stamped_versions.get(pkg)
                if stamped is None or stamped != installed_versions[pkg]:
                    mismatch = True
                    break
            if mismatch:
                return None
            has_version_stamp = True

        stamped_size = entry.get("output_artifact_size_bytes")
        stamped_sha = entry.get("output_artifact_sha256")
        if stamped_size is not None or stamped_sha:
            actual_sha, actual_size = _hash_artifact_dir(ref_dir)
            if stamped_size is not None and int(stamped_size) != actual_size:
                return None
            if stamped_sha and actual_sha and stamped_sha != actual_sha:
                return None

    # --- Step 2 fallback: baselines_history.tsv aggregation ------------
    if timing_mean is None:
        history = _read_baselines_history(task_dir, tier, int(lookup_threads))
        if len(history) >= 1:
            speeds = [s for s, _ in history]
            timing_mean = statistics.mean(speeds)
            timing_stdev = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
            peaks = [pk for _, pk in history if pk is not None]
            peak_mb = statistics.mean(peaks) if peaks else None
            n_reps_observed = len(speeds)
            timing_source = "baselines_history"
            produced_by = "history-fallback"

    # --- Step 2 last-resort: results.tsv baseline row ------------------
    if timing_mean is None:
        rr = _read_results_baseline(task_dir, tier, int(lookup_threads))
        if rr is not None:
            timing_mean, peak_mb = rr
            timing_stdev = 0.0
            n_reps_observed = 1
            timing_source = "results_tsv"
            produced_by = "history-fallback"

    if timing_mean is None or timing_mean <= 0:
        return None

    return CachedBaseline(
        timing_mean=timing_mean,
        timing_stdev=timing_stdev,
        peak_mb=peak_mb,
        ref_dir_path=str(ref_dir),
        n_reps_observed=n_reps_observed,
        has_version_stamp=has_version_stamp,
        produced_by=produced_by,
        timing_source=timing_source,
    )


# ============================================================
# Writeback
# ============================================================
def populate_persistent_ref_dir(task_dir: str, tier: str,
                                src_ref_dir: str,
                               ) -> str:
    """Promote a temp ref_dir to ``task_dir/reference_output_<tier>/``.

    Defensive: if the destination exists and is non-empty, this is a no-op
    (iteration's authoritative outputs always win — we never clobber them
    with attest-side outputs that might differ in RNG state). Returns the
    absolute path of the persistent dir.
    """
    dest = Path(task_dir) / f"reference_output_{tier}"
    if dest.is_dir() and any(dest.rglob("*")):
        return str(dest)
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(src_ref_dir)
    if not src.is_dir():
        return str(dest)
    for item in src.iterdir():
        target = dest / item.name
        # symlinks=True preserves symlinks as symlinks instead of dereferencing
        # them and copying the target's content. Important when src_ref_dir is
        # an attest temp dir whose contents might (intentionally or not)
        # contain a link pointing outside the task tree — we don't want
        # arbitrary host files leaking into the persistent reference dir.
        if item.is_symlink():
            os.symlink(os.readlink(item), target)
        elif item.is_dir():
            shutil.copytree(item, target, symlinks=True)
        else:
            shutil.copy2(item, target, follow_symlinks=False)
    return str(dest)


def write_cached_baseline(task_dir: str, tier: str, threads: int,
                          dataset_name: str,
                          speeds: Sequence[float],
                          peaks: Sequence[Optional[float]],
                          versions: dict[str, str],
                          ref_dir: str,
                          produced_by: str,
                         ) -> dict:
    """Persist v2 noise.json entry + artifact sha/size for a fresh measurement.

    Mirrors zyme.noise_calibration.record_tier_noise's body — duplicated to
    avoid importing the CLI package from autozyme_py. Both writers produce
    byte-compatible JSON.
    """
    speeds = [float(s) for s in speeds if s is not None]
    n = len(speeds)
    if n < 1:
        raise ValueError("write_cached_baseline requires at least one speed sample")
    sp_mean = statistics.mean(speeds)
    sp_std = statistics.stdev(speeds) if n > 1 else 0.0
    sp_cv = (sp_std / sp_mean) if sp_mean > 0 else 0.0
    peaks_clean = [float(pk) for pk in peaks if pk is not None]
    pk_mean = statistics.mean(peaks_clean) if peaks_clean else 0.0
    pk_std = statistics.stdev(peaks_clean) if len(peaks_clean) > 1 else 0.0

    ref_dir_abs = Path(ref_dir)
    rel = _ref_relpath(task_dir, ref_dir_abs)
    sha, size = _hash_artifact_dir(ref_dir_abs)

    entry: dict = {
        "dataset_name": dataset_name or "",
        "n_reps": n,
        "speed_mean": round(sp_mean, 6),
        "speed_stdev": round(sp_std, 6),
        "speed_cv": round(sp_cv, 6),
        "peak_mean": round(pk_mean, 3),
        "peak_stdev": round(pk_std, 3),
        "commit": "",
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if versions:
        entry["upstream_versions"] = dict(versions)
    entry["output_artifact_relpath"] = rel
    entry["output_artifact_size_bytes"] = int(size)
    if sha:
        entry["output_artifact_sha256"] = sha
    entry["produced_by"] = produced_by

    data = _load_noise(task_dir)
    data["schema_version"] = SCHEMA_VERSION
    data["tiers"].setdefault(tier, {})[str(int(threads))] = entry
    _save_noise(task_dir, data)
    return entry


# ============================================================
# Pretty-print helpers
# ============================================================
def cache_hit_msg(tier: str, cached: CachedBaseline) -> str:
    stamp_note = "" if cached.has_version_stamp else " (no version stamp — first run will backfill)"
    src_note = f" source={cached.timing_source}/{cached.produced_by}"
    sigma = f" ± {cached.timing_stdev:.3f}s" if cached.timing_stdev > 0 else ""
    return (
        f"[baseline-cache] HIT {tier}: "
        f"{cached.timing_mean:.3f}s{sigma} "
        f"(n={cached.n_reps_observed},{src_note}){stamp_note}"
    )


def cache_miss_msg(tier: str) -> str:
    return f"[baseline-cache] MISS {tier}: measuring fresh"
