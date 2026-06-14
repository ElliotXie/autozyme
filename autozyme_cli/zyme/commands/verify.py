"""`zyme verify` — runs the (tier, threads, cells) verify cube and writes verify.tsv.

Drives the loop that re-times pipeline/run.{py,R} across thread counts and
tiers, applies the scaling-tax verdict logic, and persists rows to
verify.tsv. The matplotlib panel rendering is in
`zyme.commands.verify_render` — this module imports `_render_verify_matrix`
+ `_format_scaling_tax` from there.
"""

import json
import os
import platform
import sys
import statistics
import time
from datetime import datetime
from pathlib import Path

from zyme.utils import (
    LEGACY_THREAD,
    die, die_resource, info, git, task_dir_from_args,
)
from zyme.parsers.task_yaml import (
    parse_datasets, parse_metrics, parse_threading_mode,
    parse_algorithm_class, parse_intrinsic_noise, parse_scaling_tax_thresholds,
    resolve_tiers,
    effective_threshold,
)
from zyme.parsers.results_tsv import (
    parse_log, get_baseline_speed,
    get_baseline_peak_mb, get_baseline_status,
    dataset_in_results, results_header_for_task, append_prompt_id_to_row,
    ensure_results_schema, ensure_k2_schema,
)
from zyme.runner import run_task
from zyme.commands.verify_render import (
    _format_scaling_tax,
    _render_verify_matrix,
)


# OS overhead added to per-cell ram_floor when computed from baseline peak_mb.
# Two reasons: (1) macOS holds inactive pages from prior cells that take a few
# seconds to release; (2) leaves a sliver for the kernel + Finder + your
# editor. Tuned for 16 GB Macs — bigger than necessary on workstations, but
# ram_floor is a soft pre-check, not a hard cap.
_VERIFY_RAM_FLOOR_HEADROOM_GB = 2.0

_VERIFY_RAM_FLOOR_PEAK_MULTIPLIER = 1.5

_VERIFY_RAM_FLOOR_FALLBACK_GB = 6.0

_VERIFY_MEM_CAP_TOTAL_FRACTION = 0.7

# Clamp on the file-size ratio used to extrapolate peak_mb from a known tier
# to an unknown one. Prevents absurd projections like "tiny 4MB → xlarge 80GB
# → 20000× ratio". 10× covers the realistic xlarge/large step; bigger means
# the tiers are too far apart for linear scaling to be trustworthy and we
# fall back to the file-size-only heuristic.
_VERIFY_PEAK_EXTRAPOLATION_MAX_RATIO = 10.0

# Multiplier applied to file size when no peak_mb exists for ANY tier. Bio
# matrices typically expand 2-4× from on-disk to in-RAM (sparse → dense, cell
# × gene factorization, etc.); 3× is a pragmatic middle ground.
_VERIFY_PEAK_FROM_FILE_SIZE_MULTIPLIER = 3.0






def _read_existing_verify_reps(output_path: Path, current_commit: str,
                               current_phase: str = None) -> dict:
    """Parse existing verify.tsv and return reps matching current_commit only.

    Returns {(thread, tier): [rep_dict, ...]} where rep_dict mirrors the
    in-memory shape used by cell_runs (speed_sec / peak_mb / speedup_pct /
    metrics / status / rep_idx). Old-schema files (no `commit` column) yield
    {}; callers must guard against that case before allowing --top-up.

    Crash rows (status=="crash") are filtered out so --top-up re-runs reps
    that crashed mid-matrix (host pressure, SIGTERM) instead of treating them
    as completed measurements. The crash rows themselves are scrubbed from
    verify.tsv by `_strip_crash_rows_for_topup` before new rows append, so
    the on-disk file stays consistent with rep-count semantics.

    `rep_idx` is preserved on each rep dict so callers can pick the next free
    number when stripped-crash gaps shift the count below the highest existing
    label.

    If `current_phase` is provided, also filter rows whose `phase` column
    differs (old-schema rows without a phase column are treated as
    phase="optimize" for backward compat).
    """
    if not output_path.exists():
        return {}
    lines = output_path.read_text().splitlines()
    if not lines:
        return {}
    header = lines[0].split("\t")
    if "commit" not in header:
        return {}
    col = {name: i for i, name in enumerate(header)}
    has_phase_col = "phase" in col
    by_cell: dict = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        if parts[col["commit"]] != current_commit:
            continue
        if current_phase is not None:
            row_phase = parts[col["phase"]] if has_phase_col else "optimize"
            if row_phase != current_phase:
                continue
        try:
            thread = int(parts[col["thread"]])
            tier = parts[col["tier"]]
            rep_idx = int(parts[col["rep"]] or 0)
            speed_sec = float(parts[col["speed_sec"]] or 0)
            peak_mb = float(parts[col["peak_mb"]] or 0)
            speedup_pct = float(parts[col["speedup_pct"]] or 0)
            status = parts[col["status"]]
            metrics = json.loads(parts[col["metrics_json"]] or "{}")
        except (ValueError, KeyError):
            continue
        if status == "crash":
            # Crash rows are not "completed reps" — the measurement never
            # landed. They get scrubbed by `_strip_crash_rows_for_topup`.
            continue
        # status="oom" rows ARE completed measurements — agent recorded the
        # tier as unmeasurable on this host. Keep them in-grid so --top-up
        # doesn't redundantly retry an OOM that is fundamentally unmeasurable.
        # If the agent does want to retry (different machine), they delete
        # the row by hand.
        by_cell.setdefault((thread, tier), []).append({
            "rep_idx": rep_idx,
            "speed_sec": speed_sec,
            "peak_mb": peak_mb,
            "speedup_pct": speedup_pct,
            "metrics": metrics,
            "status": status,
        })
    return by_cell




def _load_context_cells_from_verify_tsv(output_path: Path, task_dir: Path,
                                        current_commit: str, current_phase: str,
                                        metrics_spec: list, in_run_keys: set) -> list:
    """Aggregate (thread, tier) cells in verify.tsv that the current run skipped.

    Phase B's ship-gate matrix typically passes `--tiers medium,large` and so
    builds in-memory cells only for those. But verify.tsv at the same commit
    + phase may already hold ood_large / ood_xlarge rows from Phase A.3 — the
    user wants those visible on the plot, side-by-side with dev tiers and
    against their baselines. This function rehydrates those rows into the same
    cell dict shape so `_render_verify_matrix` can plot them. Cells the current
    run already produced (`in_run_keys`) are skipped — those are richer and win.

    Caller decides what to do with the returned list. Today: appended to
    `cells_for_plot` only — markdown summary and scaling-tax verdict stay
    scoped to the current run's matrix.
    """
    if not output_path.exists():
        return []
    lines = output_path.read_text().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    if "commit" not in header or "tier" not in header or "thread" not in header:
        return []
    col = {n: i for i, n in enumerate(header)}
    has_phase = "phase" in col
    by_cell: dict = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        if parts[col["commit"]] != current_commit:
            continue
        row_phase = parts[col["phase"]] if has_phase else "optimize"
        if row_phase != current_phase:
            continue
        try:
            thread = int(parts[col["thread"]])
            tier = parts[col["tier"]]
        except (ValueError, KeyError):
            continue
        key = (thread, tier)
        if key in in_run_keys:
            continue
        try:
            status = parts[col["status"]]
            dataset = parts[col["dataset"]] if "dataset" in col else ""
        except KeyError:
            continue
        speed_sec = _tolerant_float(parts[col["speed_sec"]])
        peak_mb = _tolerant_float(parts[col["peak_mb"]])
        speedup_pct = _tolerant_float(parts[col["speedup_pct"]])
        baseline = (_tolerant_float(parts[col["baseline_speed"]])
                    if "baseline_speed" in col else 0.0)
        # metrics_json may be legacy CSV-escaped — parse tolerantly.
        metrics = _parse_metrics_json_field(parts[col["metrics_json"]])
        slot = by_cell.setdefault(key, {
            "thread": thread, "tier": tier, "dataset": dataset,
            "baseline_speed": baseline,
            "speeds": [], "peaks": [], "pcts": [],
            "metrics_per_rep": [], "statuses": [],
        })
        slot["speeds"].append(speed_sec)
        slot["peaks"].append(peak_mb)
        slot["pcts"].append(speedup_pct)
        slot["metrics_per_rep"].append(metrics)
        slot["statuses"].append(status)
        if dataset and not slot["dataset"]:
            slot["dataset"] = dataset
        if baseline and not slot["baseline_speed"]:
            slot["baseline_speed"] = baseline

    out = []
    for (thread, tier), s in by_cell.items():
        speeds = [v for v in s["speeds"] if v > 0]
        peaks = [v for v in s["peaks"] if v > 0]
        pcts = s["pcts"]
        # Worst per metric across reps (mirrors aggregation logic above).
        all_names = set()
        for m in s["metrics_per_rep"]:
            all_names.update(m.keys())
        worst = {}
        for name in all_names:
            vals = []
            for m in s["metrics_per_rep"]:
                v = m.get(name)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            if not vals:
                worst[name] = None
                continue
            spec = next((m for m in metrics_spec if m["name"] == name), None)
            if spec and spec["comparator"] == "lte":
                worst[name] = max(vals)
            else:
                worst[name] = min(vals)
        # Baseline lookups: prefer the baseline_speed from verify.tsv (recorded
        # at run time), fall back to results.tsv. baseline_peak_mb is always
        # results.tsv (verify.tsv doesn't carry it).
        base_speed = s["baseline_speed"] or get_baseline_speed(task_dir, s["dataset"]) or 0.0
        base_peak = get_baseline_peak_mb(task_dir, s["dataset"]) or 0.0
        any_crash = any(st == "crash" for st in s["statuses"])
        any_oom = any(st == "oom" for st in s["statuses"])
        if any_oom:
            verdict = "OOM"
        elif any_crash:
            verdict = "CRASH"
        else:
            verdict = "—"
        # Filter zero-valued reps (OOM placeholders) from the medians so they
        # don't drag the central tendency toward 0.
        real_speeds = [v for v in speeds if v > 0]
        real_peaks  = [v for v in peaks if v > 0]
        # speedup_pct from OOM rows is always 0 — exclude.
        real_pcts = [p for p, st in zip(pcts, s["statuses"]) if st != "oom"]
        out.append({
            "thread": thread, "tier": tier, "dataset": s["dataset"],
            "baseline_speed": base_speed,
            "baseline_peak_mb": base_peak,
            "speed_sec_median": statistics.median(real_speeds) if real_speeds else 0.0,
            "speed_sec_min": min(real_speeds) if real_speeds else 0.0,
            "speed_sec_max": max(real_speeds) if real_speeds else 0.0,
            "peak_mb_median": statistics.median(real_peaks) if real_peaks else 0.0,
            "speedup_pct_median": statistics.median(real_pcts) if real_pcts else 0.0,
            "metrics_worst": worst,
            "any_crash": any_crash,
            "oom": any_oom,
            "n_reps": len(s["statuses"]),
            "in_current_run": False,
            "verdict": verdict,
            "reasons": [],
            "cv_pct": None,
        })
    return out




def _compute_scaling_tax(cells: list, thresholds: dict) -> dict:
    """Grade each OOD cell's speedup factor against thread-matched dev cells.

    Tier names starting with `ood_` are OOD; the rest are dev. Speedup
    factor = baseline / turbo. Each OOD cell is compared against the
    geometric mean of dev cells at the SAME thread count — this isolates
    size-generalization (what tax should measure) from parallelism-leverage
    (a confound when dev cells at higher threads inflate the overall mean
    and make every OOD thread=1 cell look taxed). Tax =
    dev_geom_at_thread / cell_factor (>1 means the OOD cell scaled worse
    than dev cells at the same thread). When no dev cell exists at the
    OOD cell's thread, falls back to the global dev geomean and tags the
    result with reference_kind="fallback_global".

    Returns dev_geom_mean (overall, kept for the plot reference line),
    dev_geom_by_thread, per-OOD-cell verdicts (PASS / SOFT / HARD / ZERO)
    with the dev_reference each was compared against, and rollup counts.
    Returns applicable=False when there's no OOD tier or no dev cell to
    compare against.

    OOM cells (`oom=True`) are excluded entirely — they're "this tier doesn't
    fit on this host", not "this tier scaled poorly". Counting them would
    poison both the dev geomeans (a 0× factor pulls geom toward 0) and the
    OOD verdict (a 0× factor would always look like a HARD fail). They show
    up separately in the figure's verdict panel.
    """
    import math
    dev_factors = []
    dev_factors_by_thread: dict = {}
    dev_cells = []
    ood_cells = []
    for c in cells:
        if c.get("oom"):
            continue
        base = c.get("baseline_speed", 0.0) or 0.0
        turbo = c.get("speed_sec_median", 0.0) or 0.0
        factor = (base / turbo) if (base > 0 and turbo > 0) else None
        if c["tier"].startswith("ood_"):
            ood_cells.append((c, factor))
        else:
            dev_cells.append((c, factor))
            if factor and factor > 0:
                dev_factors.append(factor)
                dev_factors_by_thread.setdefault(c["thread"], []).append(factor)
    if not dev_factors or not ood_cells:
        if not ood_cells:
            reason = "no OOD-tier cells in matrix (need a tier name starting with 'ood_')"
        elif not dev_cells:
            reason = "no dev-tier cells in matrix"
        else:
            reason = "dev-tier cells have no valid speedup factors (baseline_speed=0 or all crashes)"
        return {"applicable": False, "reason": reason, "dev_geom_mean": None,
                "dev_geom_by_thread": {},
                "dev_cells": [], "ood_results": [],
                "hard_fails": 0, "soft_flags": 0}

    def _geom(xs):
        return math.exp(sum(math.log(f) for f in xs) / len(xs))

    dev_geom = _geom(dev_factors)
    dev_geom_by_thread = {t: _geom(fs) for t, fs in dev_factors_by_thread.items()}
    hard = thresholds.get("hard_fail", 5.0)

    ood_results = []
    hard_fails = 0
    soft_flags = 0
    for c, factor in ood_cells:
        tier = c["tier"]
        thread = c["thread"]
        if tier == "ood_xlarge":
            soft, soft_label = thresholds.get("ood_xlarge_soft", 2.0), "ood_xlarge_soft"
        else:
            soft, soft_label = thresholds.get("ood_large_soft", 1.5), "ood_large_soft"
        if thread in dev_geom_by_thread:
            dev_ref = dev_geom_by_thread[thread]
            reference_kind = "thread_matched"
        else:
            dev_ref = dev_geom
            reference_kind = "fallback_global"
        if factor is None or factor <= 0:
            verdict, tax = "ZERO", float("inf")
            thr_used, thr_label = hard, "hard_fail"
            hard_fails += 1
        else:
            tax = dev_ref / factor
            if tax >= hard:
                verdict, thr_used, thr_label = "HARD", hard, "hard_fail"
                hard_fails += 1
            elif tax >= soft:
                verdict, thr_used, thr_label = "SOFT", soft, soft_label
                soft_flags += 1
            else:
                verdict, thr_used, thr_label = "PASS", soft, soft_label
        ood_results.append({
            "thread": thread, "tier": tier,
            "factor": factor, "tax": tax,
            "verdict": verdict,
            "threshold_used": thr_used,
            "threshold_label": thr_label,
            "dev_reference": dev_ref,
            "reference_kind": reference_kind,
        })
    return {"applicable": True, "dev_geom_mean": dev_geom,
            "dev_geom_by_thread": dev_geom_by_thread,
            "dev_cells": [(c["thread"], c["tier"], f) for c, f in dev_cells],
            "ood_results": ood_results,
            "hard_fails": hard_fails, "soft_flags": soft_flags}




def _per_rep_verify_status(thread: int, speedup_pct: float, metrics: dict,
                           metrics_spec: list, raw_status: str,
                           tier: str = "", intrinsic_noise: dict = None) -> str:
    """Compute pass/fail status for a single verify rep.

    Mirrors the cell-level pass criteria but applied per rep so verify.tsv's
    `status` column reflects what THAT rep showed — `pending` was misleading
    (the rep is done) and made downstream readers ambiguous.
    Crash always wins. Otherwise: pass if speedup is non-regressing for the
    thread count and every declared metric meets its threshold; fail otherwise.

    For stochastic metrics (carrying `noise_multiplier` + `absolute_floor`
    instead of fixed `threshold`), the effective gate is computed from
    `intrinsic_noise[tier][metric_name]` via `effective_threshold`. If
    intrinsic_noise is missing for this tier (calibration not run), the
    gate falls back to `absolute_floor`.
    """
    if raw_status == "crash":
        return "crash"
    # Speedup rule: thread=1 must not regress; thread>1 must improve.
    if thread == 1 and speedup_pct < 0:
        return "fail"
    if thread > 1 and speedup_pct <= 0:
        return "fail"
    for m in metrics_spec:
        actual = metrics.get(m["name"])
        if actual is None:
            return "fail"
        try:
            actual_f = float(actual)
        except (TypeError, ValueError):
            return "fail"
        thr, _label = effective_threshold(m, tier, intrinsic_noise or {})
        if m["comparator"] == "gte" and actual_f < thr:
            return "fail"
        if m["comparator"] == "lte" and actual_f > thr:
            return "fail"
    return "pass"




def _acquire_verify_lock(output_path: Path) -> Path:
    """Create an exclusive lockfile alongside verify.tsv. Returns its path.

    Uses O_CREAT|O_EXCL for atomicity. If a stale lockfile exists from a
    crashed run, detects via os.kill(pid, 0) — a dead pid means stale, ok
    to override. Concurrent verify processes would otherwise race the file
    (interleaved appends, lost rows, contaminated matrices).
    """
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    pid = os.getpid()
    stamp = datetime.now().isoformat(timespec="seconds")
    host = os.uname().nodename if hasattr(os, "uname") else platform.node()
    payload = f"pid={pid}\nstarted={stamp}\nhost={host}\n"
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, payload.encode())
            os.close(fd)
            return lock_path
        except FileExistsError:
            # Inspect existing lock to see whether it's stale.
            try:
                contents = lock_path.read_text()
            except Exception:
                contents = ""
            held_pid = None
            for line in contents.splitlines():
                if line.startswith("pid="):
                    try:
                        held_pid = int(line.split("=", 1)[1])
                    except ValueError:
                        held_pid = None
                    break
            stale = False
            if held_pid is not None:
                try:
                    os.kill(held_pid, 0)
                except ProcessLookupError:
                    stale = True
                except PermissionError:
                    # Process exists but owned by another user — treat as
                    # active rather than stale.
                    pass
            if stale:
                info(
                    f"verify lock stale (pid {held_pid} no longer running); "
                    f"removing {lock_path} and continuing."
                )
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            die(
                f"verify lock held: another `zyme verify` is running for this "
                f"task. Lock contents:\n{contents.rstrip()}\n"
                f"If you're sure no other process holds it, remove "
                f"{lock_path} manually and retry."
            )




def _release_verify_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass




def _dataset_size_mb(task_dir: Path, dataset_entry: dict) -> float:
    """Return on-disk size of a tier's dataset, in MB. 0.0 on probe failure.

    Used to compute file-size ratios when extrapolating peak_mb across tiers
    — the path the agent hits running verify on a brand-new tier whose
    baseline hasn't been recorded yet (e.g. xlarge).
    """
    raw = dataset_entry.get("path") or ""
    if not raw:
        return 0.0
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates.append(task_dir / raw)
    for cand in candidates:
        try:
            if cand.is_file():
                return cand.stat().st_size / (1024 ** 2)
            if cand.is_dir():
                total = 0
                for f in cand.rglob("*"):
                    if f.is_file():
                        try:
                            total += f.stat().st_size
                        except OSError:
                            pass
                return total / (1024 ** 2)
        except OSError:
            continue
    return 0.0




def _estimate_tier_peak_mb(task_dir: Path, target_entry: dict,
                           all_entries: list) -> tuple[float, str]:
    """Estimate peak_mb for `target_entry`, returning (peak_mb, source).

    `source` is a human-readable explanation for the guard banner:
      - "measured (8200 MB)"
      - "extrapolated from large (peak 8200 MB × 2.5 size ratio)"
      - "from file size (4500 MB × 3.0 RAM multiplier)"
      - ""  (no signal at all — caller falls back to flat default)

    Always extrapolates from the LARGEST known tier, not the closest. Fixed
    overhead (interpreter startup, library imports) is a smaller fraction of
    a large tier's peak_mb than of a tiny tier's, so linear scaling is
    closer to the truth when anchored on the larger end.
    """
    direct = get_baseline_peak_mb(task_dir, target_entry["name"]) or 0.0
    if direct > 0:
        return direct, f"measured ({direct:.0f} MB)"

    best_ref = None
    best_ref_peak = 0.0
    for entry in all_entries:
        if entry.get("name") == target_entry.get("name"):
            continue
        peak = get_baseline_peak_mb(task_dir, entry["name"]) or 0.0
        if peak > best_ref_peak:
            best_ref_peak = peak
            best_ref = entry

    target_size = _dataset_size_mb(task_dir, target_entry)
    if best_ref is not None and best_ref_peak > 0:
        ref_size = _dataset_size_mb(task_dir, best_ref)
        if ref_size > 0 and target_size > 0:
            ratio = target_size / ref_size
            if ratio <= _VERIFY_PEAK_EXTRAPOLATION_MAX_RATIO:
                est = best_ref_peak * ratio
                return est, (
                    f"extrapolated from {best_ref['tier']} "
                    f"(peak {best_ref_peak:.0f} MB × {ratio:.1f} size ratio)"
                )
            # Ratio too aggressive → drop through to file-size heuristic.

    if target_size > 0:
        est = target_size * _VERIFY_PEAK_FROM_FILE_SIZE_MULTIPLIER
        return est, (
            f"from file size ({target_size:.0f} MB × "
            f"{_VERIFY_PEAK_FROM_FILE_SIZE_MULTIPLIER:.1f} RAM multiplier)"
        )

    return 0.0, ""




def _resolve_verify_ram_floor(spec: str, cell_pairs, task_dir: Path,
                              task_yaml: Path) -> tuple[float, list]:
    """Resolve `--ram-floor` (string or 'auto') to (floor_gb, notes).

    `notes` is a list of one-line per-tier explanations of how peak_mb was
    sourced (measured / extrapolated / from-file-size / fallback). Caller
    prints them in the guard banner so the agent sees what's behind the
    auto floor and can override if the inference looks off.

    'auto' = max over tiers of (estimated_peak_mb × 1.5 + 2 GB). When the
    target tier has no measured peak_mb (e.g. running verify to RECORD an
    xlarge baseline), extrapolates from the largest known tier using
    file-size ratio. Falls back to 6 GB only when neither measurement nor
    file-size signal exists for any tier in the matrix.
    """
    if not (isinstance(spec, str) and spec.strip().lower() == "auto"):
        try:
            return float(spec), []
        except (TypeError, ValueError):
            die(f"--ram-floor: expected 'auto' or a number (GB), got {spec!r}")

    all_entries = parse_datasets(task_yaml)
    by_name = {e["name"]: e for e in all_entries}
    seen_tiers: set = set()
    notes: list = []
    max_floor_gb = 0.0
    for _, tier_entry in cell_pairs:
        if tier_entry["tier"] in seen_tiers:
            continue
        seen_tiers.add(tier_entry["tier"])
        canonical = by_name.get(tier_entry["name"], tier_entry)
        peak_mb, source = _estimate_tier_peak_mb(task_dir, canonical, all_entries)
        if peak_mb > 0:
            cell_floor = (peak_mb * _VERIFY_RAM_FLOOR_PEAK_MULTIPLIER) / 1024.0 \
                + _VERIFY_RAM_FLOOR_HEADROOM_GB
            if cell_floor > max_floor_gb:
                max_floor_gb = cell_floor
            notes.append(f"  tier={tier_entry['tier']}: peak ≈ {peak_mb:.0f} MB ({source})")
        else:
            notes.append(
                f"  tier={tier_entry['tier']}: no peak_mb signal "
                f"(no baseline, no file size) — using flat fallback"
            )
    if max_floor_gb <= 0:
        return _VERIFY_RAM_FLOOR_FALLBACK_GB, notes
    return max_floor_gb, notes




def _resolve_verify_mem_cap(spec: str) -> float:
    """Resolve `--mem-cap-gb` value (string or 'auto') to a float (GB).

    'auto' = total system RAM × 0.7. Returns 0.0 to mean "disabled" so the
    runner skips the watchdog thread entirely. Probe failure (total_ram_gb==0)
    also returns 0.0 with a warning — better to disable than to set a wrong
    cap that kills cells the user could actually run.
    """
    if isinstance(spec, str) and spec.strip().lower() == "auto":
        from zyme.dispatch.resources import total_ram_gb
        total = total_ram_gb()
        if total <= 0:
            info(
                "[verify] could not probe total system RAM; --mem-cap-gb "
                "disabled. Pass an explicit value to enable the watchdog."
            )
            return 0.0
        return total * _VERIFY_MEM_CAP_TOTAL_FRACTION
    try:
        return float(spec)
    except (TypeError, ValueError):
        die(f"--mem-cap-gb: expected 'auto' or a number (GB), got {spec!r}")




def _verify_ram_preflight(ram_floor_gb: float, n_threads: int,
                          tier_name: str, cell_pos: int, n_total: int) -> None:
    """Abort the matrix if free RAM < ram_floor_gb.

    Called before each cell launches. Aborting (vs waiting like dispatch
    does) is the right default for verify: it's an interactive command, the
    user is at the keyboard, and 'system can't fit this cell' isn't a
    transient resource-pressure issue — it's a fundamental sizing mismatch.
    For long unattended verify runs, the user can wrap with --ram-floor 0
    and rely on --mem-cap-gb to catch in-cell blowups instead.
    """
    if ram_floor_gb <= 0:
        return
    from zyme.dispatch.resources import free_ram_gb
    free_gb = free_ram_gb()
    if free_gb >= ram_floor_gb:
        return
    # When the guard fires mid-matrix, cells 1..cell_pos-1 already wrote to
    # verify.tsv; mention --top-up so the user resumes from cell_pos instead
    # of restarting from scratch. Audit log shows this is a common
    # retry-chain footgun (5×--top-up --ram-floor after 3× plain --ram-floor).
    resume_hint = ""
    if cell_pos > 1:
        resume_hint = (
            f"  Note: cells 1..{cell_pos - 1} already completed for this matrix. "
            f"After freeing memory / lowering the floor, add --top-up to "
            f"resume from cell {cell_pos} instead of re-running the whole grid.\n"
        )
    die_resource(
        f"[verify guard] cell {cell_pos}/{n_total} thread={n_threads} "
        f"tier={tier_name}: free RAM {free_gb:.1f} GB < floor {ram_floor_gb:.1f} GB.\n"
        f"  Free up memory (close browsers / Slack), or:\n"
        f"    - drop the matrix (use --cells to skip this combination)\n"
        f"    - lower the floor (--ram-floor {max(2.0, free_gb - 1.0):.1f}) "
        f"if you accept the thrash risk\n"
        f"    - move to a larger host and re-run.\n"
        f"{resume_hint}"
        f"  This guard exists because the prior `zyme verify` runs were "
        f"crashing the host on under-provisioned cells — the watchdog "
        f"(--mem-cap-gb) kills runaway cells but can't help if you're "
        f"already at the floor before the cell starts."
    )




def _write_verify_summary_txt(cells: list, metrics_spec: list, n_reps: int,
                              out_path: Path) -> None:
    """Plain-text fallback for `verify.png` when matplotlib isn't installed.

    Writes a fixed-width table of cells/verdicts so packaging always has a
    summary artifact to consume. Not a replacement for the heatmap, but
    enough that downstream prompts don't break on the missing PNG.
    """
    metric_names = [m["name"] for m in metrics_spec]
    headers = ["thread", "tier", "speed_sec", "speedup_pct", "verdict"] + metric_names
    rows = [headers]
    for c in sorted(cells, key=lambda x: (x["thread"], x["tier"])):
        row = [
            str(c["thread"]),
            c["tier"],
            f"{c['speed_sec_median']:.3f}",
            f"{c['speedup_pct_median']:+.1f}%",
            c["verdict"],
        ]
        for name in metric_names:
            v = c["metrics_worst"].get(name)
            row.append(f"{v:.4f}" if v is not None else "—")
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [
        f"# verify summary (matplotlib not available — plain-text fallback)",
        f"# reps_per_cell: {n_reps}",
        "",
    ]
    for i, r in enumerate(rows):
        lines.append(fmt.format(*r))
        if i == 0:
            lines.append(fmt.format(*("-" * w for w in widths)))
    out_path.write_text("\n".join(lines) + "\n")




def _verify_probe_cache_path(task_dir: Path) -> Path:
    return task_dir / ".zyme" / "verify_probe.cache"




def _verify_probe_cache_has(task_dir: Path, commit: str, t_lo: int, t_hi: int) -> bool:
    """True if a prior probe at this commit + thread set already passed.

    Probe failures are a property of the commit (whether ZYME_THREADS is wired
    through pipeline/run.{py,R}), not of which tier you happen to be verifying.
    Caching by (commit, t_lo, t_hi) lets back-to-back verify calls at the same
    commit (e.g. Phase A.2 dev tiers + A.3 ood_xlarge) skip the probe redundancy.
    """
    p = _verify_probe_cache_path(task_dir)
    if not p.exists():
        return False
    needle = f"{commit}\t{t_lo}\t{t_hi}\t"
    for line in p.read_text().splitlines():
        if line.startswith(needle):
            return True
    return False




def _verify_probe_cache_write(task_dir: Path, commit: str, t_lo: int, t_hi: int,
                              probe_tier_name: str) -> None:
    p = _verify_probe_cache_path(task_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = f"{commit}\t{t_lo}\t{t_hi}\t{probe_tier_name}\t{datetime.now().isoformat(timespec='seconds')}\n"
    if p.exists():
        with open(p, "a") as f:
            f.write(line)
    else:
        p.write_text(line)




def _migrate_verify_tsv_add_phase(output_path: Path) -> bool:
    """Add a `phase` column (default 'optimize') to a pre-phase verify.tsv.

    Returns True if the file was migrated, False if already up-to-date or
    not present. Idempotent. Required before `--top-up` / `--append` on a
    file written by an older zyme that lacks the column — without this,
    appending new rows would mismatch the header.
    """
    if not output_path.exists():
        return False
    lines = output_path.read_text().splitlines()
    if not lines:
        return False
    header = lines[0].split("\t")
    if "phase" in header:
        return False
    if "commit" not in header:
        return False
    new_lines = ["\t".join(header + ["phase"])]
    for line in lines[1:]:
        if not line.strip():
            new_lines.append(line)
            continue
        new_lines.append(line + "\toptimize")
    output_path.write_text("\n".join(new_lines) + "\n")
    return True




def _strip_crash_rows_for_topup(output_path: Path, current_commit: str,
                                current_phase: str = None) -> int:
    """Remove status=crash rows for the current commit (and phase) from verify.tsv.

    Called from --top-up so that crashed reps are re-run instead of counted
    as completed. Rows for OTHER commits/phases are preserved verbatim.
    Returns the number of rows stripped (for user-facing logging).
    """
    if not output_path.exists():
        return 0
    lines = output_path.read_text().splitlines()
    if len(lines) < 2:
        return 0
    header = lines[0].split("\t")
    if "commit" not in header or "status" not in header:
        return 0
    col = {name: i for i, name in enumerate(header)}
    has_phase_col = "phase" in col
    kept = [lines[0]]
    stripped = 0
    for line in lines[1:]:
        if not line.strip():
            kept.append(line)
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            kept.append(line)
            continue
        is_current = parts[col["commit"]] == current_commit
        if is_current and current_phase is not None:
            row_phase = parts[col["phase"]] if has_phase_col else "optimize"
            is_current = row_phase == current_phase
        if is_current and parts[col["status"]] == "crash":
            stripped += 1
            continue
        kept.append(line)
    if stripped:
        output_path.write_text("\n".join(kept) + "\n")
    return stripped




def _tolerant_float(raw, default: float = 0.0) -> float:
    """float(raw) but `NA` / `nan` / empty string → default.

    verify.tsv rows occasionally carry sentinel values like `NA` when the
    measurement was logically absent (e.g. baseline_speed=NA when the agent
    couldn't run upstream at the same thread count because upstream OOMs).
    Strict float() throws on those and silently drops the whole row in
    list-comprehension parsers; this helper turns the sentinel into a 0
    (or another caller-chosen default) so the surrounding row still renders.
    """
    if raw is None:
        return default
    s = str(raw).strip()
    if not s or s.lower() in ("na", "n/a", "nan", "none", "null"):
        return default
    try:
        v = float(s)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return v




def _parse_metrics_json_field(raw: str) -> dict:
    """Parse the verify.tsv metrics_json field — tolerant of legacy CSV-escape.

    Plain TSV (current writer): `{"jaccard":0.95,"...":...}` — passes straight
    through json.loads. But some pre-existing verify.tsv files were written
    (or re-saved through a CSV-aware tool) as `"{""jaccard"":0.95,...}"` —
    surrounding quotes plus doubled inner quotes. Detect that pattern and
    unescape before parsing. Returns {} on any parse failure (silent — caller
    won't usually have a way to fix bad rows mid-render).
    """
    if not raw:
        return {}
    s = raw.strip()
    if not s:
        return {}
    # CSV-escape pattern: starts with `"` and ends with `"`, with `""` inside.
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"' and '""' in s:
        s = s[1:-1].replace('""', '"')
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        return {}
    # The metrics_json column always holds an object; a valid-but-non-object
    # JSON value (bare scalar / array) is not usable by callers that do
    # `.get(...)`, so normalize it to {}.
    return parsed if isinstance(parsed, dict) else {}




def _cmd_verify_render_only(args, task_dir: Path) -> None:
    """`zyme verify --render-only`: rebuild verify.png/.pdf/.svg from verify.tsv.

    Reads verify.tsv at the current HEAD commit + --phase, aggregates / grades
    cells with the same rules as the live path, and writes the figure. No
    subprocess runs, no probe, no watchdog. Honors --tiers / --threads /
    --cells as a filter on which rows are included; defaults to all rows at
    the matching commit + phase.

    Used to refresh the figure after editing task.yaml thresholds, after a
    framework plot-rendering update, or to re-render a verify.tsv produced on
    another machine.
    """
    task_yaml = task_dir / "task.yaml"
    output_path = task_dir / args.output
    if not output_path.exists():
        die(
            f"--render-only: {output_path} does not exist. "
            f"Run `zyme verify` (without --render-only) first to produce the matrix."
        )

    metrics_spec = parse_metrics(task_yaml)
    intrinsic_noise = parse_intrinsic_noise(task_yaml)
    tax_thresholds = parse_scaling_tax_thresholds(task_yaml)
    super_linear_max = tax_thresholds.get("super_linear_max", 1.5)
    threading_mode = parse_threading_mode(task_yaml)
    threading_grades = (threading_mode != "not_applicable")

    current_commit = git("rev-parse", "HEAD", cwd=task_dir, check=False)
    current_commit = (current_commit or "")[:7] or "—"
    current_phase = getattr(args, "phase", None) or "optimize"

    # Optional filter: --cells / --threads / --tiers narrow which (thread, tier)
    # rows we include in the figure. Same parsing as the live path so the user
    # gets the same UX. None = include everything at the matching commit+phase.
    filter_keys: set | None = None
    if getattr(args, "cells", None):
        all_tiers = {e["tier"]: e for e in parse_datasets(task_yaml)}
        filter_keys = set()
        for raw in args.cells.split(","):
            raw = raw.strip()
            if not raw:
                continue
            if ":" not in raw:
                die(f"--cells: '{raw}' missing ':' (expected `thread:tier`)")
            t_str, tier_name = raw.split(":", 1)
            try:
                n = int(t_str.strip())
            except ValueError:
                die(f"--cells: '{raw}' has non-integer thread '{t_str}'")
            tier_name = tier_name.strip()
            if tier_name not in all_tiers:
                die(f"--cells: tier '{tier_name}' not in task.yaml. "
                    f"Available: {sorted(all_tiers.keys())}")
            filter_keys.add((n, tier_name))
    elif getattr(args, "tiers", None) or args.threads != "1,4,8":
        # Only filter when user explicitly narrowed the matrix; default
        # `--threads 1,4,8` with no `--tiers` means "show everything in TSV".
        threads = [int(t.strip()) for t in args.threads.split(",") if t.strip()]
        tier_request = (
            [t.strip() for t in args.tiers.split(",") if t.strip()]
            if getattr(args, "tiers", None) else None
        )
        if tier_request is not None:
            try:
                tier_entries = resolve_tiers(task_yaml, tier_request)
            except ValueError as e:
                die(str(e))
            tier_names = {te["tier"] for te in tier_entries}
            filter_keys = {(n, t) for n in threads for t in tier_names}

    # Read verify.tsv → group reps by (thread, tier) at this commit + phase.
    lines = output_path.read_text().splitlines()
    if not lines:
        die(f"--render-only: {output_path} is empty.")
    header = lines[0].split("\t")
    col = {name: i for i, name in enumerate(header)}
    if "commit" not in col or "status" not in col:
        die(f"--render-only: {output_path} schema is too old (missing commit/status column). "
            f"Re-run `zyme verify` to upgrade the schema.")
    has_phase_col = "phase" in col

    by_cell: dict = {}
    n_skipped_other_commit = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        if parts[col["commit"]] != current_commit:
            n_skipped_other_commit += 1
            continue
        row_phase = parts[col["phase"]] if has_phase_col else "optimize"
        if row_phase != current_phase:
            continue
        try:
            thread = int(parts[col["thread"]])
            tier = parts[col["tier"]]
            status = parts[col["status"]]
            dataset = parts[col["dataset"]]
        except (ValueError, KeyError):
            continue
        # NA / blank → 0 so a row with one missing measurement (e.g.
        # baseline_speed=NA when upstream itself OOMs) still renders.
        speed = _tolerant_float(parts[col["speed_sec"]])
        peak = _tolerant_float(parts[col["peak_mb"]])
        speedup = _tolerant_float(parts[col["speedup_pct"]])
        baseline_speed = _tolerant_float(parts[col["baseline_speed"]])
        # metrics_json may be legacy CSV-escaped — parse tolerantly.
        metrics = _parse_metrics_json_field(parts[col["metrics_json"]])
        if filter_keys is not None and (thread, tier) not in filter_keys:
            continue
        slot = by_cell.setdefault((thread, tier), {
            "thread": thread, "tier": tier, "dataset": dataset,
            "baseline_speed": baseline_speed,
            "reps": [],
        })
        slot["reps"].append({
            "speed_sec": speed, "peak_mb": peak,
            "speedup_pct": speedup, "metrics": metrics,
            "status": status,
        })

    if not by_cell:
        die(
            f"--render-only: no rows in {output_path} match commit {current_commit} "
            f"phase={current_phase}. {n_skipped_other_commit} row(s) at other commits.\n"
            f"  - Wrong commit? Checkout the verify-time HEAD and retry.\n"
            f"  - Wrong phase? Pass --phase optimize or --phase validate.\n"
            f"  - Filter too tight? Drop --tiers/--threads/--cells."
        )

    # Aggregate per cell — mirror live path's logic for medians, worst metrics,
    # CV%, oom, any_crash. Uses real_speeds (excluding 0 / OOM placeholders) so
    # OOM rows don't drag the median toward 0.
    cells = []
    for (thread, tier), agg in by_cell.items():
        reps = [r for r in agg["reps"] if r["status"] != "crash"]
        speeds = [r["speed_sec"] for r in reps]
        peaks = [r["peak_mb"] for r in reps]
        pcts = [r["speedup_pct"] for r in reps]
        real_speeds = [s for s in speeds if s > 0]
        real_peaks = [p for p in peaks if p > 0]
        real_pcts = [p for r, p in zip(reps, pcts) if r["status"] != "oom"]

        all_names = set()
        for r in reps:
            all_names.update(r["metrics"].keys())
        worst = {}
        for name in all_names:
            vals = []
            for r in reps:
                v = r["metrics"].get(name)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            if not vals:
                worst[name] = None
                continue
            spec = next((m for m in metrics_spec if m["name"] == name), None)
            worst[name] = max(vals) if (spec and spec["comparator"] == "lte") else min(vals)

        base_speed = agg["baseline_speed"] or get_baseline_speed(task_dir, agg["dataset"]) or 0.0
        base_peak = get_baseline_peak_mb(task_dir, agg["dataset"]) or 0.0
        any_oom = any(r["status"] == "oom" for r in agg["reps"])
        any_crash = any(r["status"] == "crash" for r in agg["reps"])
        cv = None
        if len(real_speeds) >= 2:
            mean = sum(real_speeds) / len(real_speeds)
            cv = statistics.stdev(real_speeds) / mean * 100.0 if mean > 0 else None

        speed_median = statistics.median(real_speeds) if real_speeds else 0.0
        # Recompute speedup_pct from base_speed + speed_median rather than
        # trusting the stored column. Otherwise verify.tsv rows with
        # speedup_pct=NA (e.g. when the verify-time baseline was unrecorded
        # but results.tsv has a usable one) would auto-FAIL on the speedup
        # rule despite Panel B showing a real factor from the fallback.
        if base_speed > 0 and speed_median > 0:
            speedup_median = (1 - speed_median / base_speed) * 100.0
        else:
            speedup_median = (statistics.median(real_pcts) if real_pcts else 0.0)

        # Promote cell to OOM when the tier's baseline is OOM-marked in
        # results.tsv — overrides the rep-level status. (Agent ran
        # `record-baseline --oom` after a successful prior verify; the
        # historical pass rows in verify.tsv no longer have a meaningful
        # baseline to compare against, so they render as OOM cells.)
        baseline_marked_oom = (get_baseline_status(task_dir, agg["dataset"]) == "oom")
        cell_oom = any_oom or baseline_marked_oom

        cells.append({
            "thread": thread, "tier": tier, "dataset": agg["dataset"],
            "baseline_speed": base_speed,
            "baseline_peak_mb": base_peak,
            "speed_sec_median": speed_median,
            "speed_sec_min": min(real_speeds) if real_speeds else 0.0,
            "speed_sec_max": max(real_speeds) if real_speeds else 0.0,
            "peak_mb_median": statistics.median(real_peaks) if real_peaks else 0.0,
            "speedup_pct_median": speedup_median,
            "metrics_worst": worst,
            "any_crash": any_crash,
            "oom": cell_oom,
            "n_reps": len(reps),
            "in_current_run": True,
            "cv_pct": cv,
        })

    # Grade — same rules as the live path. OOM cells get verdict=OOM and skip
    # all speedup/concordance checks.
    thread1_factor_by_tier = {}
    for c in cells:
        if c["thread"] == 1 and not c.get("oom"):
            base = c["baseline_speed"] or 0.0
            turbo = c["speed_sec_median"] or 0.0
            if base > 0 and turbo > 0:
                thread1_factor_by_tier[c["tier"]] = base / turbo

    for c in cells:
        if c.get("oom"):
            c["verdict"] = "OOM"
            c["reasons"] = ["OOM (tier did not fit on host)"]
            continue
        reasons = []
        if c["any_crash"]:
            reasons.append("CRASH (one or more reps)")
        else:
            if c["thread"] == 1:
                if c["speedup_pct_median"] < 0:
                    reasons.append(f"speedup_pct(median)={c['speedup_pct_median']:.1f} < 0")
            else:
                if threading_grades:
                    if c["speedup_pct_median"] <= 0:
                        reasons.append(f"speedup_pct(median)={c['speedup_pct_median']:.1f} <= 0")
                    ref = thread1_factor_by_tier.get(c["tier"])
                    base = c["baseline_speed"] or 0.0
                    turbo = c["speed_sec_median"] or 0.0
                    if ref and base > 0 and turbo > 0:
                        cf = base / turbo
                        if cf < ref:
                            reasons.append(
                                f"factor={cf:.1f}× < thread=1 factor={ref:.1f}× "
                                f"(multi-thread regression)"
                            )
                        elif cf / ref > super_linear_max * c["thread"]:
                            reasons.append(
                                f"factor/thread=1 ratio={cf/ref:.1f}× > "
                                f"{super_linear_max * c['thread']:.1f}× cap (super-linear)"
                            )
            for m in metrics_spec:
                actual = c["metrics_worst"].get(m["name"])
                if actual is None:
                    reasons.append(f"missing metric {m['name']}")
                    continue
                thr, thr_label = effective_threshold(m, c["tier"], intrinsic_noise)
                if m["comparator"] == "gte" and actual < thr:
                    reasons.append(f"{m['name']}(worst)={actual:.4f} < {thr:.4f} [{thr_label}]")
                elif m["comparator"] == "lte" and actual > thr:
                    reasons.append(f"{m['name']}(worst)={actual:.4f} > {thr:.4f} [{thr_label}]")
        c["verdict"] = "PASS" if not reasons else "FAIL"
        c["reasons"] = reasons

    n_reps_max = max((c["n_reps"] for c in cells), default=1)
    tax = _compute_scaling_tax(cells, tax_thresholds)

    fails = [c for c in cells if c["verdict"] == "FAIL"]

    print(
        f"[verify --render-only] commit={current_commit} phase={current_phase}: "
        f"{len(cells)} cell(s); "
        f"{sum(1 for c in cells if c['verdict']=='PASS')} PASS, "
        f"{len(fails)} FAIL, "
        f"{sum(1 for c in cells if c['verdict']=='CRASH' or c.get('any_crash'))} CRASH, "
        f"{sum(1 for c in cells if c.get('oom'))} OOM."
    )
    if fails:
        print("Failures:")
        for c in fails:
            print(f"  thread={c['thread']} tier={c['tier']}: {'; '.join(c['reasons'])}")

    plot_base = task_dir / Path(args.output).stem
    if getattr(args, "no_plot", False):
        info("--no-plot set; skipping figure render.")
    else:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            die(
                "--render-only requires matplotlib. Install with `pip install matplotlib` "
                "and retry, or pass --no-plot (which makes --render-only a no-op)."
            )
        _render_verify_matrix(
            cells, task_dir.name, plot_base, plt,
            metrics_spec=metrics_spec, n_reps=n_reps_max,
            scaling_tax=tax,
        )
        print(f"[verify --render-only] wrote {plot_base}.png / .pdf / .svg")

    # Scaling-tax verdict + exit code — mirrors the live verify path so
    # `--render-only` is the cheap way to get the combined dev+OOD verdict
    # (3_validate_scaling Phase A.4 relies on this).
    if tax["applicable"]:
        print(_format_scaling_tax(tax, tax_thresholds))
    else:
        print(
            f"\nScaling tax: N/A — {tax.get('reason', 'matrix does not have both dev and OOD tiers')}. "
            f"Include both dev tiers and at least one ood_* tier in the same render-only call "
            f"(e.g. --tiers tiny,medium,ood_large,ood_xlarge) to get the generalization verdict."
        )
    if tax["hard_fails"] > 0:
        sys.exit(2)
    if fails or tax["soft_flags"] > 0:
        sys.exit(1)




def cmd_verify(args):
    """Run pipeline at thread × tier matrix; verify concordance + speedup hold.

    Verification, NOT iteration: results land in `verify.tsv` (not `results.tsv`)
    and don't consume any round budget. Intended for `4_package.md` step 2's
    threading robustness check before lifting the patch into a package.

    Defaults: threads = {1, 4, 8} × tiers = all from task.yaml = 9 cells.
    Agent can override either axis via --threads / --tiers. Tasks declaring
    `threading: not_applicable` are forced to thread=1 before execution unless
    --allow-not-applicable-threads is explicitly set.

    Pass criteria per cell (hardcoded — these are the user-facing stability claim):
      - All concordance metrics in `task.yaml` honored (gte/lte against threshold)
      - speedup_pct > 0  at thread > 1  (turbo wins at multi-thread)
      - speedup_pct >= 0 at thread = 1  (turbo doesn't regress at serial)

    A cell crash doesn't abort the matrix; remaining cells still run so the agent
    sees the full diagnostic. Exit code is 1 if any cell failed, 0 otherwise.

    Pre-condition: `pipeline/run.{py,R}` must read thread count from env vars
    (OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / ZYME_THREADS). Hardcoded thread
    counts make the matrix silently produce identical results across cells —
    fix `pipeline/run` first if you see speed_sec uniform across thread axis.
    """
    task_dir = task_dir_from_args(args)
    task_yaml = task_dir / "task.yaml"

    # --render-only: skip matrix execution entirely; rebuild verify.png from
    # verify.tsv at current commit + phase. Useful for refreshing the figure
    # after framework plot changes or task.yaml threshold edits.
    if getattr(args, "render_only", False):
        return _cmd_verify_render_only(args, task_dir)

    # Probe matplotlib up-front so the matplotlib-missing fallback (verify.txt
    # instead of verify.png) is announced BEFORE the matrix runs, not silently
    # at the end. Downstream packaging expects verify.png; surfacing the
    # fallback early lets the agent decide whether to install matplotlib and
    # rerun, or proceed knowing only the TSV will be produced. Skipped when
    # --no-plot is set (user explicitly opted out).
    if not getattr(args, "no_plot", False):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            info(
                "[matplotlib] not available — verify will fall back to a "
                "plain-text summary (verify.txt). Downstream packaging that "
                "expects verify.png will need to read the TSV instead. "
                "Install with `pip install matplotlib` and re-run if you want "
                "the plot."
            )

    # Build the (thread, tier_entry) cell list. Two modes:
    #   - --cells "T:tier,T:tier,..."   → arbitrary sparse grid
    #   - --threads + --tiers (default) → cartesian product
    if args.cells:
        all_tiers = {e["tier"]: e for e in parse_datasets(task_yaml)}
        cell_pairs = []
        for raw in args.cells.split(","):
            raw = raw.strip()
            if not raw:
                continue
            if ":" not in raw:
                die(f"--cells: '{raw}' missing ':' (expected `thread:tier`, e.g. `4:medium`)")
            t_str, tier_name = raw.split(":", 1)
            try:
                n_threads = int(t_str.strip())
            except ValueError:
                die(f"--cells: '{raw}' has non-integer thread '{t_str}'")
            tier_name = tier_name.strip()
            if tier_name not in all_tiers:
                die(f"--cells: tier '{tier_name}' not in task.yaml. "
                    f"Available: {sorted(all_tiers.keys())}")
            cell_pairs.append((n_threads, all_tiers[tier_name]))
        if not cell_pairs:
            die("--cells: empty list")
    else:
        threads = [int(t.strip()) for t in args.threads.split(",") if t.strip()]
        if not threads:
            die("--threads: empty list")
        tier_request = None
        if args.tiers:
            tier_request = [t.strip() for t in args.tiers.split(",") if t.strip()]
        try:
            tiers = resolve_tiers(task_yaml, tier_request)
        except ValueError as e:
            die(str(e))
        cell_pairs = [(n, tier) for n in threads for tier in tiers]

    threading_mode = parse_threading_mode(task_yaml)
    allow_not_applicable_threads = bool(
        getattr(args, "allow_not_applicable_threads", False)
    )
    if threading_mode == "not_applicable" and not allow_not_applicable_threads:
        requested_threads = sorted({n for n, _ in cell_pairs})
        skipped_threads = [n for n in requested_threads if n != 1]
        if skipped_threads:
            info(
                "[verify] threading: not_applicable in task.yaml — forcing "
                "the verify matrix to thread=1 and skipping requested "
                f"multi-thread cells: {skipped_threads}"
            )
        deduped = []
        seen_tiers = set()
        for _n, tier_entry in cell_pairs:
            tier_name = tier_entry["tier"]
            if tier_name in seen_tiers:
                continue
            deduped.append((1, tier_entry))
            seen_tiers.add(tier_name)
        cell_pairs = deduped
    elif threading_mode == "not_applicable" and allow_not_applicable_threads:
        requested_threads = sorted({n for n, _ in cell_pairs})
        requested_multi = [n for n in requested_threads if n != 1]
        if requested_multi:
            info(
                "[verify] WARNING: --allow-not-applicable-threads set; "
                "task.yaml declares threading: not_applicable, but requested "
                f"multi-thread verify cells will run: {requested_multi}"
            )

    metrics_spec = parse_metrics(task_yaml)
    if not metrics_spec:
        info("WARN: task.yaml has no metrics declared — concordance pass criteria will be skipped")
    intrinsic_noise = parse_intrinsic_noise(task_yaml)
    algo_class = parse_algorithm_class(task_yaml)
    if algo_class == "stochastic":
        # Surface noise-calibration gaps that would silently fall back to
        # absolute_floor for affected tiers — agents need to know.
        for tier in {te["tier"] for _, te in cell_pairs}:
            tier_noise = intrinsic_noise.get(tier, {})
            stochastic_metrics = [m for m in metrics_spec
                                  if "noise_multiplier" in m and "absolute_floor" in m]
            missing = [m["name"] for m in stochastic_metrics
                       if m["name"] not in tier_noise]
            if missing:
                info(f"WARN: tier='{tier}' missing intrinsic_noise for "
                     f"{missing} — those metrics will gate at absolute_floor "
                     f"only. Run `zyme baseline noise --tier {tier}` to calibrate.")

    n_reps = max(1, args.reps)

    # Resolve HEAD commit + phase BEFORE the probe so the probe cache can use
    # them. (Probe pass is a property of the commit — same commit + same
    # thread set means the same wiring outcome, regardless of which tier the
    # user is verifying right now.)
    current_commit = git("rev-parse", "HEAD", cwd=task_dir, check=False)
    current_commit = (current_commit or "")[:7] or "—"
    current_phase = getattr(args, "phase", None) or "optimize"

    # K2: no mode axis. Each verify cell at (thread=K, tier=T) divides by
    # the (T, K) baseline. Baseline lookups, OOM detection, and verify.tsv
    # rows are all keyed on (tier, thread) only.

    # Threading-wired probe. The verify matrix is meaningless if pipeline/run
    # doesn't read thread count from ZYME_THREADS / OMP_NUM_THREADS — every
    # cell will produce ~identical speed_sec, and the agent reports "all PASS"
    # having tested nothing. Probe runs the chosen probe tier at thread=1 and
    # thread=max(matrix_threads); if they agree to within 1.5×, abort.
    # Skipped when task.yaml declares `threading: not_applicable` (sequential
    # algorithm, or wiring threading would require risky thread=1 rewrites).
    if threading_mode == "not_applicable":
        info(
            "[verify] threading: not_applicable in task.yaml — skipping the "
            "threading-wired probe and threading cell-level pass criteria."
        )
    if threading_mode != "not_applicable" and not getattr(args, "skip_probe", False):
        matrix_threads = sorted({n for n, _ in cell_pairs})
        if len(matrix_threads) >= 2:
            t_lo, t_hi = matrix_threads[0], matrix_threads[-1]

            # Probe tier resolution. Default = smallest tier in the matrix.
            # `--probe-tier` lets the user point at a SMALL dev tier when the
            # matrix only contains a single large tier (so wiring validation
            # doesn't take ~as long as the matrix itself).
            probe_tier_override = getattr(args, "probe_tier", None)
            if probe_tier_override:
                all_tiers = {e["tier"]: e for e in parse_datasets(task_yaml)}
                if probe_tier_override not in all_tiers:
                    die(f"--probe-tier: tier '{probe_tier_override}' not in task.yaml. "
                        f"Available: {sorted(all_tiers.keys())}")
                probe_tier = all_tiers[probe_tier_override]
            else:
                tiers_in_matrix = []
                seen = set()
                for _, te in cell_pairs:
                    if te["tier"] not in seen:
                        seen.add(te["tier"])
                        tiers_in_matrix.append(te)
                yaml_order = {e["tier"]: i for i, e in enumerate(parse_datasets(task_yaml))}
                tiers_in_matrix.sort(key=lambda te: yaml_order.get(te["tier"], 999))
                probe_tier = tiers_in_matrix[0]

            # Probe-pass cache (per-commit-per-thread-set). Skip the redundant
            # probe when a prior verify call already proved wiring at this
            # commit + thread set. --force-probe overrides; touching
            # framework/helpers.{py,R} without changing HEAD is the case where
            # you'd want that.
            cache_hit = (
                not getattr(args, "force_probe", False)
                and _verify_probe_cache_has(task_dir, current_commit, t_lo, t_hi)
            )
            if cache_hit:
                print(
                    f"\n[verify probe] cached PASS at commit {current_commit} "
                    f"for thread set {{{t_lo},{t_hi}}}; skipping. "
                    f"Pass --force-probe to re-run.",
                    flush=True,
                )
            else:
                # Wall-time estimate so the agent can see the probe cost
                # before it starts (useful when probe_tier is a large tier).
                baseline_for_probe = get_baseline_speed(task_dir, probe_tier["name"]) or 0.0
                est_msg = ""
                if baseline_for_probe > 0:
                    est_total = baseline_for_probe * 2  # one run at t_lo, one at t_hi
                    est_msg = f" (estimated wall-time ~{est_total / 60:.1f} min based on baseline)"
                print(
                    f"\n[verify probe] checking ZYME_THREADS is wired through "
                    f"pipeline/run.{{py,R}}: thread={t_lo} vs thread={t_hi} on "
                    f"tier={probe_tier['tier']} ({probe_tier['name']}){est_msg}. "
                    f"Skip with --skip-probe; override probe tier with --probe-tier.",
                    flush=True,
                )
                probe_speeds = {}
                probe_artifact_dir = (
                    task_dir / "artifacts"
                    / f"verify_probe_{current_commit}_{probe_tier['tier']}"
                )
                probe_artifact_dir.mkdir(parents=True, exist_ok=True)
                for n_threads in (t_lo, t_hi):
                    env = {
                        "OMP_NUM_THREADS": str(n_threads),
                        "OPENBLAS_NUM_THREADS": str(n_threads),
                        "MKL_NUM_THREADS": str(n_threads),
                        "ZYME_THREADS": str(n_threads),
                    }
                    log_content = run_task(task_dir, dataset_entry=probe_tier, extra_env=env)
                    # Persist every probe attempt's log so a failure can be
                    # inspected without rerunning. Without this, the probe
                    # crashed-out with a "look at artifacts/" message and
                    # nothing was actually written there.
                    probe_log_path = probe_artifact_dir / f"thread_{n_threads}.log"
                    probe_log_path.write_text(log_content, encoding="utf-8")
                    speed_sec, _, _, status = parse_log(log_content)
                    if status == "crash" or not speed_sec or speed_sec <= 0:
                        die(
                            f"[verify probe] thread={n_threads} crashed or produced no "
                            f"speed_sec. Cannot validate threading wiring. "
                            f"Probe log: {probe_log_path}. "
                            f"Fix the crash and rerun (or `--skip-probe` to bypass)."
                        )
                    probe_speeds[n_threads] = speed_sec
                    print(
                        f"[verify probe] thread={n_threads}: speed_sec={speed_sec:.3f} "
                        f"(log: {probe_log_path.relative_to(task_dir)})",
                        flush=True,
                    )
                ratio = max(probe_speeds.values()) / min(probe_speeds.values())
                if ratio < 1.5:
                    die(
                        f"[verify probe] FAIL — speed at thread={t_lo} ({probe_speeds[t_lo]:.3f}s) "
                        f"vs thread={t_hi} ({probe_speeds[t_hi]:.3f}s) differs by only "
                        f"{ratio:.2f}× (< 1.5×). Two possibilities, distinguish manually:\n"
                        f"  (a) ZYME_THREADS is NOT wired through pipeline/run.{{py,R}} → "
                        f"the full matrix would silently produce near-identical cells across "
                        f"the thread axis. Audit for hardcoded thread counts: "
                        f"`mc.cores`, `OMP_NUM_THREADS`, `RcppParallel::setThreadOptions`, "
                        f"`numba.set_num_threads`, `joblib.Parallel(n_jobs=...)`, "
                        f"`parallel::detectCores()`. Replace each with a read from the "
                        f"`ZYME_THREADS` env var (use `get_threads()` from "
                        f"`framework/helpers.{{py,R}}` for a one-liner).\n"
                        f"  (b) ZYME_THREADS IS wired but the parallel layer is intrinsically "
                        f"small (e.g. only ~10–30% of wall is parallelizable; Amdahl's law "
                        f"caps speedup well below 1.5×). The matrix would still produce real "
                        f"data — just modest.\n"
                        f"To distinguish, run pipeline/run.{{py,R}} manually under both "
                        f"`ZYME_THREADS=1` and `ZYME_THREADS={t_hi}` and compare the "
                        f"`N_THREADS=` print line your pipeline emits: identical wall "
                        f"AND identical N_THREADS = wiring missing (case a); identical wall "
                        f"AND different N_THREADS = wiring fine, parallel layer just small "
                        f"(case b — pass `--skip-probe` to continue).\n"
                        f"Then rerun `zyme verify` (with `--skip-probe` if case b)."
                    )
                _verify_probe_cache_write(
                    task_dir, current_commit, t_lo, t_hi, probe_tier["tier"]
                )
                print(
                    f"[verify probe] PASS — speed differs {ratio:.2f}× across the "
                    f"thread axis; threading is wired. Continuing to full matrix.",
                    flush=True,
                )

    output_path = task_dir / args.output
    write_mode = getattr(args, "write_mode", "overwrite")
    top_up = write_mode == "topup"

    # Acquire an exclusive lockfile alongside verify.tsv. Concurrent verify
    # processes (e.g. an agent forgetting it left one running, or a cron-
    # scheduled rerun firing while the first hasn't finished) would otherwise
    # interleave appends and silently contaminate the matrix.
    lock_path = _acquire_verify_lock(output_path)
    try:
        _cmd_verify_body(
            args, task_dir, task_yaml, cell_pairs, metrics_spec, n_reps,
            output_path, write_mode, top_up, current_commit, current_phase,
            intrinsic_noise,
        )
    finally:
        _release_verify_lock(lock_path)




def _cmd_verify_body(args, task_dir, task_yaml, cell_pairs, metrics_spec,
                     n_reps, output_path, write_mode, top_up,
                     current_commit, current_phase, intrinsic_noise):
    keeps_existing = write_mode in ("append", "topup")

    # If we're appending or topping up to an existing file, the file's schema
    # must already include the `commit` column written by this version. Old
    # files lack it and would corrupt on append. The `phase` column was added
    # later — auto-migrate (default 'optimize') so users with existing
    # commit-tagged files don't have to delete them after upgrading.
    if keeps_existing and output_path.exists():
        first_line = output_path.read_text().splitlines()[:1]
        if first_line and "commit" not in first_line[0].split("\t"):
            die(
                f"verify.tsv at {output_path} uses an older schema (no `commit` column). "
                f"Either delete it (lose old rows) or rerun with --write-mode overwrite "
                f"to rewrite with the new schema."
            )
        if _migrate_verify_tsv_add_phase(output_path):
            info(f"verify.tsv: backfilled `phase` column (existing rows tagged 'optimize').")

    # On --top-up, scrub any crash rows for the current commit + phase from
    # verify.tsv so they get RE-RUN instead of counted as completed reps.
    # Without this, a SIGTERM mid-matrix would leave the "no rep crashed"
    # gate failed forever no matter how many top-ups the agent attempts.
    if top_up:
        stripped = _strip_crash_rows_for_topup(output_path, current_commit, current_phase)
        if stripped:
            info(
                f"verify.tsv: stripped {stripped} crash row(s) at commit "
                f"{current_commit} (phase={current_phase}); they will be re-run."
            )

    # Pre-populate cell_runs with existing reps for the current commit when
    # --top-up. The loop below skips reps already counted here.
    existing_reps_per_cell: dict = {}
    if top_up:
        existing_reps_per_cell = _read_existing_verify_reps(
            output_path, current_commit, current_phase
        )

    if not keeps_existing or not output_path.exists():
        output_path.write_text(
            "timestamp\tthread\ttier\tdataset\trep\tspeed_sec\tpeak_mb\t"
            "baseline_speed\tspeedup_pct\tstatus\tmetrics_json\tcommit\tphase\n"
        )

    # K2: ensure results.tsv schema is up to date (adds `thread` column with
    # backfill 1 if absent). No more lazy stash promotion — baselines are
    # written directly by `cmd_record_baseline`. Pre-flight here just warns
    # for any (tier, thread) cell whose baseline is missing.
    results_tsv = task_dir / "results.tsv"
    ensure_k2_schema(task_dir)
    _missing_baseline = []
    for n_threads, _te in cell_pairs:
        if dataset_in_results(results_tsv, _te["name"], thread=int(n_threads)):
            continue
        _missing_baseline.append((_te["tier"], int(n_threads)))
    if _missing_baseline:
        info(
            f"[verify pre-flight] WARNING: no baseline for {len(_missing_baseline)} "
            f"(tier, thread) cell(s): "
            f"{', '.join(f'{t}@{th}' for t, th in _missing_baseline)}. "
            f"speedup_pct will be 0 → those cells will FAIL. Run "
            f"`zyme baseline-rebench --threads <list>` (or "
            f"`zyme record-baseline --tier T --thread N --speed-sec X`) to fix, "
            f"then rerun verify."
        )

    _tiers_in_matrix = {te["tier"]: te for _, te in cell_pairs}

    # OOM detection: if `record-baseline --oom` marked any (tier, thread) cell
    # as unmeasurable, skip subprocess execution and append status=oom rows to
    # verify.tsv directly. We check per (tier, thread) — a tier might fit at
    # thread=8 but OOM at thread=1.
    oom_tiers_in_matrix = {
        _te["tier"] for _te in _tiers_in_matrix.values()
        if get_baseline_status(task_dir, _te["name"]) == "oom"
    }
    if oom_tiers_in_matrix:
        # Collect cells of OOM tiers from the requested matrix.
        _oom_pairs = [(n, te) for n, te in cell_pairs if te["tier"] in oom_tiers_in_matrix]

        # Find which (thread, tier) cells already have an OOM row at this
        # commit+phase so we don't duplicate rows on repeat invocations.
        _existing_oom_keys: set = set()
        if output_path.exists():
            _vlines = output_path.read_text().splitlines()
            if _vlines:
                _vh = _vlines[0].split("\t")
                _vc = {n: i for i, n in enumerate(_vh)}
                if {"thread", "tier", "status", "commit"}.issubset(_vc):
                    _has_phase = "phase" in _vc
                    for _vl in _vlines[1:]:
                        _vp = _vl.split("\t")
                        if len(_vp) <= _vc["status"]:
                            continue
                        if _vp[_vc["commit"]] != current_commit:
                            continue
                        if _has_phase and _vp[_vc["phase"]] != current_phase:
                            continue
                        if _vp[_vc["status"]] != "oom":
                            continue
                        try:
                            _existing_oom_keys.add(
                                (int(_vp[_vc["thread"]]), _vp[_vc["tier"]])
                            )
                        except ValueError:
                            continue

        # Append OOM rows for any cell not already recorded. Mirrors the
        # main-loop schema (13 columns) so render-only and live aggregation
        # both see them as completed-with-status=oom reps.
        _appended = 0
        for n_threads, tier_entry in _oom_pairs:
            if (n_threads, tier_entry["tier"]) in _existing_oom_keys:
                continue
            _oom_row = "\t".join([
                datetime.now().isoformat(timespec="seconds"),
                str(n_threads), tier_entry["tier"], tier_entry["name"],
                "1",                               # rep
                "0", "0", "0", "0",                # speed_sec, peak_mb, baseline_speed, speedup_pct
                "oom", "{}",                       # status, metrics_json
                current_commit, current_phase,
            ]) + "\n"
            with open(output_path, "a") as _f:
                _f.write(_oom_row)
            # Pre-seed cell_runs as if --top-up had read this back, so the
            # downstream aggregation + plot picks the cell up regardless of
            # the --top-up flag value.
            existing_reps_per_cell.setdefault(
                (n_threads, tier_entry["tier"]), []
            ).append({
                "rep_idx": 1, "speed_sec": 0.0, "peak_mb": 0.0,
                "speedup_pct": 0.0, "metrics": {}, "status": "oom",
            })
            _appended += 1

        # Also seed already-recorded OOM cells into existing_reps_per_cell
        # so they show up in the figure on every invocation (not only --top-up).
        for n_threads, tier_entry in _oom_pairs:
            key = (n_threads, tier_entry["tier"])
            if key in _existing_oom_keys and not existing_reps_per_cell.get(key):
                existing_reps_per_cell[key] = [{
                    "rep_idx": 1, "speed_sec": 0.0, "peak_mb": 0.0,
                    "speedup_pct": 0.0, "metrics": {}, "status": "oom",
                }]

        info(
            f"[verify pre-flight] OOM tier(s): {sorted(oom_tiers_in_matrix)}. "
            f"Skipping subprocess runs for {len(_oom_pairs)} cell(s) "
            f"({_appended} new OOM row(s) appended, "
            f"{len(_oom_pairs) - _appended} already recorded). "
            f"Figure will render them as hatched OOM blocks."
        )
        # Remove OOM tiers from execution; their rows are already in verify.tsv.
        cell_pairs = [(n, te) for n, te in cell_pairs if te["tier"] not in oom_tiers_in_matrix]

    # Per-cell accumulator: one entry per (thread, tier), reps appended as we go.
    # Pre-seed with existing reps when --top-up so aggregation includes them.
    cell_runs = {}
    runs_remaining = 0
    reused_reps_total = 0
    for n_threads, tier_entry in cell_pairs:
        key = (n_threads, tier_entry["tier"])
        existing = existing_reps_per_cell.get(key, [])
        if existing:
            cell_runs[key] = {
                "thread": n_threads, "tier": tier_entry["tier"],
                "dataset": tier_entry["name"],
                "baseline_speed": get_baseline_speed(
                    task_dir, tier_entry["name"], thread=int(n_threads)) or 0.0,
                "baseline_peak_mb": get_baseline_peak_mb(
                    task_dir, tier_entry["name"], thread=int(n_threads)) or 0.0,
                "reps": list(existing),
            }
            reused_reps_total += min(len(existing), n_reps)
        runs_remaining += max(0, n_reps - len(existing))

    if top_up:
        # Surface rows that already exist at commit+phase but fall OUTSIDE
        # the requested cell grid (e.g., verify.tsv has 1:large rows from a
        # prior matrix; this top-up requested only --tiers medium). Without
        # this, the agent has no way to tell that verify.tsv contains
        # measurements not represented in the current run's report.
        requested_keys = {(n, te["tier"]) for n, te in cell_pairs}
        ignored_outside_grid = sum(
            len(reps)
            for key, reps in existing_reps_per_cell.items()
            if key not in requested_keys
        )
        ignored_keys = sorted(
            k for k in existing_reps_per_cell if k not in requested_keys
        )
        if reused_reps_total or ignored_outside_grid:
            parts = [f"\n[verify --top-up] commit={current_commit}, phase={current_phase}:"]
            if reused_reps_total:
                parts.append(
                    f"  reusing {reused_reps_total} rep(s) in-grid; "
                    f"running {runs_remaining} new rep(s)."
                )
            if ignored_outside_grid:
                cells_str = ", ".join(f"{t}:{tier}" for (t, tier) in ignored_keys)
                parts.append(
                    f"  ignoring {ignored_outside_grid} rep(s) at out-of-grid cells "
                    f"({cells_str}) — they stay in verify.tsv but are not in this matrix."
                )
            print("\n".join(parts), flush=True)

    # Resource guards. Pre-flight free-RAM gate (--ram-floor) before each
    # cell, plus an in-cell process-group RSS watchdog (--mem-cap-gb) that
    # SIGKILLs the cell if it crosses the cap. Together they prevent the
    # host-thrash crashes that motivated this guard — and let an OOM cell
    # show up as a clean `status=crash, crash_msg=killed-by-mem-watchdog`
    # row in verify.tsv, so the matrix continues and the agent (or a
    # reviewer) sees the OOM as a measurement, not a missing data point.
    ram_floor_gb, ram_floor_notes = _resolve_verify_ram_floor(
        getattr(args, "ram_floor", "auto"), cell_pairs, task_dir, task_yaml,
    )
    mem_cap_gb = _resolve_verify_mem_cap(getattr(args, "mem_cap_gb", "auto"))
    cell_settle_s = max(0.0, float(getattr(args, "cell_settle_s", 2.0)))
    floor_str = f"{ram_floor_gb:.1f} GB" if ram_floor_gb > 0 else "disabled"
    cap_str = f"{mem_cap_gb:.1f} GB" if mem_cap_gb > 0 else "disabled"
    banner_lines = [
        f"\n[verify guard] ram_floor={floor_str}  mem_cap={cap_str}  "
        f"cell_settle={cell_settle_s:.1f}s. "
        f"Override with --ram-floor / --mem-cap-gb / --cell-settle-s; "
        f"set 0 to disable."
    ]
    if ram_floor_notes:
        banner_lines.append("[verify guard] auto-floor sources:")
        banner_lines.extend(ram_floor_notes)
    print("\n".join(banner_lines), flush=True)

    n_total = runs_remaining
    n_done = 0
    cells_seen = 0
    for n_threads, tier_entry in cell_pairs:
        key = (n_threads, tier_entry["tier"])
        existing = existing_reps_per_cell.get(key, [])
        existing_count = len(existing)
        # Start new reps after the highest existing rep_idx, not after
        # existing_count — surviving reps from a stripped-crash file may
        # have been labelled (e.g.) rep=2 originally, so naïve count+1
        # numbering would collide with that label.
        next_rep_idx = (max((r["rep_idx"] for r in existing), default=0)) + 1
        n_new = max(0, n_reps - existing_count)
        if n_new <= 0:
            continue
        # Settle pause between cells (skip before the first one). Lets the
        # kernel finalize page reclaim from the prior cell so the next cell
        # doesn't race the swap-file writer.
        if cells_seen > 0 and cell_settle_s > 0:
            time.sleep(cell_settle_s)
        cells_seen += 1
        # Pre-flight RAM check fires once per cell, not per rep — the cell's
        # peak is dominated by the first rep's allocation, and re-checking
        # between reps would just be noise.
        _verify_ram_preflight(
            ram_floor_gb, n_threads, tier_entry["tier"], n_done + 1, n_total,
        )
        for offset in range(n_new):
            rep_idx = next_rep_idx + offset
            n_done += 1
            cell_rep_pos = existing_count + offset + 1  # 1-indexed within this cell
            print(
                f"\n=== verify {n_done}/{n_total}: thread={n_threads} "
                f"tier={tier_entry['tier']} ({tier_entry['name']}) "
                f"rep={cell_rep_pos}/{n_reps} (label={rep_idx}) ===",
                flush=True,
            )
            extra_env = {
                "OMP_NUM_THREADS": str(n_threads),
                "OPENBLAS_NUM_THREADS": str(n_threads),
                "MKL_NUM_THREADS": str(n_threads),
                "ZYME_THREADS": str(n_threads),
            }
            log_content = run_task(
                task_dir, dataset_entry=tier_entry, extra_env=extra_env,
                mem_cap_gb=(mem_cap_gb if mem_cap_gb > 0 else None),
                thread=int(n_threads),
            )
            print(log_content, flush=True)

            speed_sec, peak_mb, metrics, raw_status = parse_log(log_content)
            # A crash rep must not carry a payload: speed_sec=0 + a real
            # baseline otherwise yields speedup_pct=100, polluting the matrix.
            if raw_status == "crash":
                speed_sec = None
                peak_mb = None
                metrics = {}
            baseline_speed = get_baseline_speed(
                task_dir, tier_entry["name"], thread=int(n_threads))
            if (raw_status != "crash" and baseline_speed and baseline_speed > 0
                    and speed_sec is not None and speed_sec > 0):
                speedup_pct = (1 - speed_sec / baseline_speed) * 100.0
            else:
                speedup_pct = 0.0

            # Per-rep verdict: pass / fail / crash. Replaces the misleading
            # `pending` that parse_log returned for completed reps — verify
            # reps are never pending once they finish, and the cell-level
            # PASS/FAIL printed in the markdown table was the only place
            # where rep status was actually evaluated.
            status = _per_rep_verify_status(
                n_threads, speedup_pct, metrics, metrics_spec, raw_status,
                tier=tier_entry["tier"], intrinsic_noise=intrinsic_noise,
            )

            # One-line per-rep summary so progress is legible across the
            # verbose per-cell logs above. Without this the matrix is silent
            # for minutes between cells; the agent loses orientation.
            elapsed_str = f"{speed_sec:.2f}s" if speed_sec else "—"
            print(
                f"[verify {n_done}/{n_total} done] thread={n_threads} "
                f"tier={tier_entry['tier']} rep={cell_rep_pos}/{n_reps} "
                f"→ {elapsed_str} {status.upper()}",
                flush=True,
            )

            row = "\t".join([
                datetime.now().isoformat(timespec="seconds"),
                str(n_threads), tier_entry["tier"], tier_entry["name"],
                str(rep_idx),
                f"{speed_sec or 0:.3f}", f"{peak_mb or 0:.1f}",
                f"{baseline_speed or 0:.3f}", f"{speedup_pct:.1f}",
                status, json.dumps(metrics, separators=(",", ":")),
                current_commit, current_phase,
            ]) + "\n"
            with open(output_path, "a") as f:
                f.write(row)

            cell_runs.setdefault(key, {
                "thread": n_threads, "tier": tier_entry["tier"],
                "dataset": tier_entry["name"],
                "baseline_speed": baseline_speed or 0.0,
                "baseline_peak_mb": get_baseline_peak_mb(
                    task_dir, tier_entry["name"], thread=int(n_threads)) or 0.0,
                "reps": [],
            })
            cell_runs[key]["reps"].append({
                "rep_idx": rep_idx,
                "speed_sec": speed_sec or 0.0,
                "peak_mb": peak_mb or 0.0,
                "speedup_pct": speedup_pct,
                "metrics": metrics,
                "status": status,
            })

    # Aggregate per cell.
    cells = []
    for key, agg in cell_runs.items():
        reps = agg["reps"]
        speeds = [r["speed_sec"] for r in reps]
        peaks = [r["peak_mb"] for r in reps]
        pcts = [r["speedup_pct"] for r in reps]

        # Per-metric: take the WORST value across reps (most pessimistic) so a
        # one-rep concordance break can't be hidden by averaging.
        all_names = set()
        for r in reps:
            all_names.update(r["metrics"].keys())
        worst_metrics = {}
        for name in all_names:
            vals = []
            for r in reps:
                v = r["metrics"].get(name)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            if not vals:
                worst_metrics[name] = None
                continue
            spec = next((m for m in metrics_spec if m["name"] == name), None)
            if spec and spec["comparator"] == "lte":
                worst_metrics[name] = max(vals)  # higher is worse for "lower-is-better" metrics
            else:
                worst_metrics[name] = min(vals)  # lower is worse for "higher-is-better" metrics

        # Filter zero-valued speed/peak from medians when an OOM rep was
        # logged with placeholder zeros — otherwise the median gets dragged
        # toward 0 by an unmeasurable rep.
        real_speeds = [s for s in speeds if s > 0]
        real_peaks  = [p for p in peaks if p > 0]
        real_pcts   = [p for r, p in zip(reps, pcts) if r["status"] != "oom"]
        # Promote cell to OOM if either: (a) any rep status is "oom" (agent
        # explicitly marked it, or live pre-flight injected it), or (b) the
        # tier's baseline in results.tsv is OOM-marked. Either way, the cell
        # has no meaningful turbo-vs-baseline comparison.
        baseline_marked_oom = (get_baseline_status(task_dir, agg["dataset"]) == "oom")
        cells.append({
            "thread": agg["thread"], "tier": agg["tier"], "dataset": agg["dataset"],
            "baseline_speed":   agg.get("baseline_speed", 0.0),
            "baseline_peak_mb": agg.get("baseline_peak_mb", 0.0),
            "speed_sec_median": statistics.median(real_speeds) if real_speeds else 0.0,
            "speed_sec_min":    min(real_speeds) if real_speeds else 0.0,
            "speed_sec_max":    max(real_speeds) if real_speeds else 0.0,
            "peak_mb_median":   statistics.median(real_peaks) if real_peaks else 0.0,
            "speedup_pct_median": statistics.median(real_pcts) if real_pcts else 0.0,
            "metrics_worst":    worst_metrics,
            "any_crash":        any(r["status"] == "crash" for r in reps),
            "oom":              any(r["status"] == "oom" for r in reps) or baseline_marked_oom,
            "n_reps":           len(reps),
            "in_current_run":   True,
        })

    # Pass-criteria evaluation per cell. Speed is judged on median (run-to-run
    # variance is system noise); concordance is judged on the worst rep (a
    # single-rep break = non-determinism = ship-blocking). Threading rules
    # (no regression, no super-linear) compare same-tier thread > 1 cells
    # against thread=1 — only enforced when both ends are in the matrix.
    # Skipped entirely when task.yaml declares `threading: not_applicable`.
    tax_thresholds = parse_scaling_tax_thresholds(task_yaml)
    super_linear_max = tax_thresholds.get("super_linear_max", 1.5)
    threading_mode_body = parse_threading_mode(task_yaml)
    threading_grades = (threading_mode_body != "not_applicable")
    thread1_factor_by_tier = {}
    for c in cells:
        if c["thread"] != 1:
            continue
        base = c.get("baseline_speed", 0.0) or 0.0
        turbo = c.get("speed_sec_median", 0.0) or 0.0
        if base > 0 and turbo > 0:
            thread1_factor_by_tier[c["tier"]] = base / turbo

    fails = []
    for c in cells:
        reasons = []
        if c.get("oom"):
            # OOM is its own verdict — not a fail. Tier was tried, didn't
            # fit on this host, agent recorded it. Skip all speedup/concordance
            # rules; the cell is excluded from scaling tax (see _compute_scaling_tax).
            c["verdict"] = "OOM"
            c["reasons"] = ["OOM (tier did not fit on host)"]
            continue
        if c["any_crash"]:
            reasons.append("CRASH (one or more reps)")
        else:
            if c["thread"] == 1:
                if c["speedup_pct_median"] < 0:
                    reasons.append(f"speedup_pct(median)={c['speedup_pct_median']:.1f} < 0")
            else:
                # Multi-thread cells: only graded on speedup/threading when
                # threading is applicable. Otherwise the cell is judged on
                # concordance only (metrics still must pass below).
                if threading_grades:
                    if c["speedup_pct_median"] <= 0:
                        reasons.append(f"speedup_pct(median)={c['speedup_pct_median']:.1f} <= 0")
                    ref = thread1_factor_by_tier.get(c["tier"])
                    base = c.get("baseline_speed", 0.0) or 0.0
                    turbo = c.get("speed_sec_median", 0.0) or 0.0
                    if ref and base > 0 and turbo > 0:
                        cell_factor = base / turbo
                        if cell_factor < ref:
                            reasons.append(
                                f"factor={cell_factor:.1f}× < thread=1 factor={ref:.1f}× "
                                f"(multi-thread regression: oversubscription / NUMA / lock contention)"
                            )
                        else:
                            cap = super_linear_max * c["thread"]
                            ratio = cell_factor / ref
                            if ratio > cap:
                                reasons.append(
                                    f"factor/thread=1 ratio={ratio:.1f}× > {cap:.1f}× cap "
                                    f"({super_linear_max:.1f}×{c['thread']}t super-linear: "
                                    f"thread=1 path likely broken — fast path only firing when threaded)"
                                )
            for m in metrics_spec:
                actual = c["metrics_worst"].get(m["name"])
                if actual is None:
                    reasons.append(f"missing metric {m['name']}")
                    continue
                thr, thr_label = effective_threshold(m, c["tier"], intrinsic_noise)
                if m["comparator"] == "gte" and actual < thr:
                    reasons.append(f"{m['name']}(worst)={actual:.4f} < {thr:.4f} [{thr_label}]")
                elif m["comparator"] == "lte" and actual > thr:
                    reasons.append(f"{m['name']}(worst)={actual:.4f} > {thr:.4f} [{thr_label}]")
        c["verdict"] = "PASS" if not reasons else "FAIL"
        c["reasons"] = reasons
        if reasons:
            fails.append(c)

    # Per-cell CV (stdev/mean) so the markdown table flags outlier-driven
    # spreads at a glance — a 2× span across reps that the median hides is
    # exactly the host-pressure noise the 3-rep gate is meant to catch.
    for c in cells:
        # speed values for this cell: pulled from cell_runs (keyed by
        # (thread, tier)) since the aggregation already discarded reps[].
        agg = cell_runs.get((c["thread"], c["tier"]), {})
        speeds = [r["speed_sec"] for r in agg.get("reps", []) if r.get("speed_sec")]
        if len(speeds) >= 2:
            mean = sum(speeds) / len(speeds)
            sd = statistics.stdev(speeds)
            c["cv_pct"] = (sd / mean) * 100.0 if mean > 0 else None
        else:
            c["cv_pct"] = None

    # Print markdown summary table for direct paste into package README.
    metric_names = [m["name"] for m in metrics_spec]
    show_cv = n_reps > 1
    headers = ["thread", "tier", "dataset", "speed_sec"]
    if show_cv:
        headers.append("CV")
    headers += ["speedup_pct"] + metric_names + ["verdict"]
    print(f"\n=== Verification matrix (markdown — paste into README.md Performance section; reps={n_reps}) ===\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for c in cells:
        if n_reps > 1:
            speed_str = f"{c['speed_sec_median']:.2f} ({c['speed_sec_min']:.2f}–{c['speed_sec_max']:.2f})"
        else:
            speed_str = f"{c['speed_sec_median']:.2f}"
        row_vals = [
            str(c["thread"]), c["tier"], c["dataset"],
            speed_str,
        ]
        if show_cv:
            cv = c.get("cv_pct")
            if cv is None:
                cv_str = "—"
            elif cv >= 30.0:
                # Annotate large spreads inline so the agent doesn't have to
                # re-derive CV from min/max — likely host-pressure noise.
                cv_str = f"**{cv:.1f}%**"
            else:
                cv_str = f"{cv:.1f}%"
            row_vals.append(cv_str)
        row_vals.append(f"{c['speedup_pct_median']:.1f}")
        for name in metric_names:
            v = c["metrics_worst"].get(name)
            row_vals.append(f"{float(v):.4f}" if v is not None else "—")
        row_vals.append(c["verdict"])
        print("| " + " | ".join(row_vals) + " |")

    if show_cv and any((c.get("cv_pct") or 0) >= 30.0 for c in cells):
        print(
            "\n_Bolded CV ≥ 30% — likely host-pressure noise (wall/cpu drift, "
            "thermal throttling); consider rerunning under quieter load before "
            "treating speedup_pct as ground truth._"
        )

    print(f"\nMatrix written to {output_path}")

    # OOD scaling-tax verdict (tax_thresholds parsed earlier with the cell loop).
    # Computed from the in-run cells so the verdict is deterministic per call.
    tax = _compute_scaling_tax(cells, tax_thresholds)

    # Augment cells_for_plot with rows from verify.tsv at this commit + phase
    # that the current matrix didn't cover (e.g., Phase B's ship-gate at
    # medium+large still wants to display ood_large / ood_xlarge from Phase A).
    # Plot-only — the markdown summary + tax verdict above stay scoped to
    # cells the current run actually re-measured.
    in_run_keys = {(c["thread"], c["tier"]) for c in cells}
    context_cells = _load_context_cells_from_verify_tsv(
        output_path, task_dir, current_commit, current_phase, metrics_spec, in_run_keys
    )
    cells_for_plot = list(cells) + context_cells
    if context_cells:
        ctx_summary = ", ".join(
            sorted({f"{c['tier']}/{c['thread']}t" for c in context_cells})
        )
        info(
            f"[verify plot] including {len(context_cells)} additional cell(s) "
            f"from verify.tsv at commit {current_commit} phase={current_phase} "
            f"({ctx_summary}); these were not part of the current --tiers/--threads "
            f"matrix but stayed in the figure for full Phase 3 context."
        )

    # Optional PNG/PDF/SVG render of the speedup matrix for direct embed in README.
    plot_base = task_dir / Path(args.output).stem  # verify.tsv → verify.png/.pdf/.svg
    plot_rendered = False
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            info(
                "matplotlib not available; skipping PNG/PDF/SVG. "
                "Falling back to plain-text summary at "
                f"{plot_base.with_suffix('.txt')} — packaging reads the TSV "
                "regardless. (`pip install matplotlib` to enable plots.)"
            )
        else:
            try:
                _render_verify_matrix(
                    cells_for_plot, task_dir.name, plot_base, plt,
                    metrics_spec=metrics_spec, n_reps=n_reps,
                    scaling_tax=tax,
                )
                print(f"Plot written to {plot_base}.png / .pdf / .svg")
                plot_rendered = True
            except Exception as e:
                import traceback
                info(f"plot render failed: {e}; writing plain-text fallback instead.")
                info(traceback.format_exc())

    if not plot_rendered:
        txt_path = plot_base.with_suffix(".txt")
        try:
            _write_verify_summary_txt(cells, metrics_spec, n_reps, txt_path)
            print(f"Plain-text fallback written to {txt_path}")
        except Exception as e:
            info(f"plain-text fallback render failed: {e}")

    print(f"Result: {len(cells) - len(fails)}/{len(cells)} cells PASS")
    if fails:
        print("Failures:")
        for c in fails:
            print(f"  thread={c['thread']} tier={c['tier']}: {'; '.join(c['reasons'])}")

    # Scaling-tax verdict + exit code (3-tier: 0 clean, 1 cell fail or soft
    # flag, 2 hard scaling fail). Stops agents silently passing a 70× cliff
    # via the old "speedup_pct > 0" criterion. Always prints something so the
    # agent knows why the block is absent (e.g., OOD-only or dev-only matrix).
    if tax["applicable"]:
        print(_format_scaling_tax(tax, tax_thresholds))
    else:
        print(
            f"\nScaling tax: N/A — {tax.get('reason', 'matrix does not have both dev and OOD tiers')}. "
            f"Include both dev tiers and at least one ood_* tier in the same verify call "
            f"(e.g. --tiers tiny,medium,ood_large,ood_xlarge) to get the generalization verdict."
        )
    if tax["hard_fails"] > 0:
        sys.exit(2)
    if fails or tax["soft_flags"] > 0:
        sys.exit(1)
