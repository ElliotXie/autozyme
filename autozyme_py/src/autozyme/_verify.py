"""End-to-end verification of a registered patch against an autozyme task.

Mirrors `autozyme::verify_patch` in R: runs baseline (untouched upstream) and
patched (autozyme-activated upstream) across one or more tiers, stages
outputs in a temp directory mirroring the task layout, shells out to the
task's own evaluate.{R,py} for metric computation, parses 'metric: value'
lines, compares to task.yaml thresholds.

Each baseline / patched measurement runs in its OWN subprocess
(``python -m autozyme._verify_worker``). The subprocess does
``load → optional activate → time(call) → save → exit``; only the ``call``
region is timed. This isolates per-measurement state — no in-process
deepcopy / param-store reset is needed (those reset tasks contaminated the
timing region under the old in-process design and diluted reported
speedups for stateful patches like cell2location and sccoda).
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from autozyme._core import _REGISTRY, _import_submodule
from autozyme._baseline_cache import (
    CachedBaseline,
    cache_hit_msg,
    cache_miss_msg,
    load_cached_baseline,
    populate_persistent_ref_dir,
    write_cached_baseline,
    _collect_versions,
    _resolve_threads,
)
from autozyme._thread_env import ensure_process_thread_env


_METRIC_LINE_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*([-+0-9.eE]+)\s*$")


def _read_task_yaml(task_dir: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "verify_patch requires PyYAML; pip install pyyaml"
        ) from e
    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_evaluate(task_dir: str, temp_dir: str, ref_dir: str, tier: str) -> list[str]:
    """Copy + run task's evaluate.{py,R}; return stdout lines."""
    for cand in ("evaluate.py", "evaluate.R"):
        src = os.path.join(task_dir, cand)
        if os.path.exists(src):
            evaluate_path = src
            break
    else:
        raise FileNotFoundError(
            f"no evaluate.py / evaluate.R in {task_dir}"
        )

    dest = os.path.join(temp_dir, os.path.basename(evaluate_path))
    shutil.copy(evaluate_path, dest)

    env = os.environ.copy()
    env["ZYME_TIER"] = tier
    env["ZYME_REFERENCE_DIR"] = ref_dir
    env["ZYME_TEST_DIR"] = os.path.join(temp_dir, "pipeline")
    # Absolute task dir so evaluate.{py,R} can resolve bundled helpers (e.g.
    # Scanpy per-step attest imports under optimized_task/.../_shared/).
    env["ZYME_TASK_DIR"] = os.path.abspath(task_dir)

    if dest.endswith(".py"):
        cmd = [sys.executable, dest]
    else:
        cmd = ["Rscript", dest]
    cwd = temp_dir

    proc = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"evaluate exited {proc.returncode}:\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout.splitlines()


def _run_worker(name: str, task_dir: str, tier: str, output_dir: str,
                activate: bool, verbose: bool) -> tuple[float, float | None]:
    """Spawn a fresh-Python subprocess to do one (baseline OR patched) measurement.

    Returns ``(elapsed_sec, peak_mb)``:
      - ``elapsed_sec`` — timed call region only (`smoke['call']`).
      - ``peak_mb``     — OS peak RSS over the subprocess's whole lifetime
        (load + call + save), or ``None`` if the platform doesn't support
        ``getrusage``/psutil. Includes shared/mmap pages, so absolute values
        diverge from tracemalloc; use as baseline-vs-patched delta on the
        same workload.
    """
    cmd = [
        sys.executable, "-m", "autozyme._verify_worker",
        "--patch", name,
        "--task-dir", task_dir,
        "--tier", tier,
        "--output-dir", output_dir,
    ]
    if activate:
        cmd.append("--activate")

    # Stream worker stderr live to verbose users (banners, progress bars) AND
    # capture it for error reporting. Without the capture, a failing worker
    # leaves only "(streamed)" in the exception/skip-note and the actual
    # traceback scrolls past in the terminal — debugging across machines is
    # painful. The tee thread reads stderr line-by-line; verbose mode mirrors
    # to sys.stderr, the buffer is kept either way.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    captured_err: list[str] = []

    def _tee_stderr() -> None:
        try:
            for line in iter(proc.stderr.readline, ""):
                if not line:
                    break
                captured_err.append(line)
                if verbose:
                    sys.stderr.write(line)
                    sys.stderr.flush()
        finally:
            try:
                proc.stderr.close()
            except Exception:  # noqa: BLE001
                pass

    tee_thread = threading.Thread(target=_tee_stderr, daemon=True)
    tee_thread.start()
    stdout_data = proc.stdout.read()
    try:
        proc.stdout.close()
    except Exception:  # noqa: BLE001
        pass
    proc.wait()
    tee_thread.join(timeout=5)
    stderr_blob = "".join(captured_err)

    # Shim back into the rest of the function which expects `proc.stdout` /
    # `proc.stderr` strings and `proc.returncode`. Popen's `proc.returncode`
    # is already set by wait(); just attach the captured payloads.
    proc.stdout = stdout_data  # type: ignore[assignment]
    proc.stderr = stderr_blob  # type: ignore[assignment]

    if proc.returncode != 0:
        raise RuntimeError(
            f"verify-worker subprocess exited {proc.returncode} "
            f"(activate={activate}):\n"
            f"--- stdout ---\n{stdout_data}\n"
            f"--- stderr ---\n{stderr_blob or '(empty)'}"
        )

    # Parse the last non-empty stdout line as JSON; tolerates stray prints
    # from upstream packages (e.g. cell2location's training progress in print()
    # mode if the user's smoke recipe doesn't fully silence it).
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("verify-worker printed no stdout (expected JSON)")
    for ln in reversed(lines):
        try:
            payload = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "elapsed_sec" in payload:
            peak = payload.get("peak_mb")
            return (
                float(payload["elapsed_sec"]),
                None if peak is None else float(peak),
            )
    raise RuntimeError(
        "verify-worker stdout had no JSON line with 'elapsed_sec' key:\n"
        + proc.stdout
    )


def _effective_threshold(metric: dict, tier: str, intrinsic_noise: dict) -> tuple:
    """Return (threshold_value, label) for one (metric, tier).

    Mirrors `autozyme_cli/zyme/parsers/task_yaml.py::effective_threshold` so
    verify_patch handles both schemas the iteration phase already supports:

      - Deterministic: metric has `threshold` -> use as-is.
      - Stochastic: metric has `absolute_floor` (+ optional `noise_multiplier`,
        default 2.0). With calibrated `intrinsic_noise[tier][name]`:
          lte: effective = max(floor, multiplier * noise)
          gte: effective = max(floor, 1 - multiplier * (1 - noise))
        Without calibration, falls back to `absolute_floor`.
    """
    if "threshold" in metric:
        return float(metric["threshold"]), "absolute"
    if "absolute_floor" not in metric:
        raise RuntimeError(
            f"metric {metric.get('name')!r} in task.yaml has neither "
            f"`threshold` (deterministic) nor `absolute_floor` (stochastic) field"
        )
    floor = float(metric["absolute_floor"])
    multiplier = float(metric.get("noise_multiplier", 2.0))
    raw = (intrinsic_noise.get(tier) or {}).get(metric["name"])
    if raw is None:
        return floor, "absolute_floor (no intrinsic_noise calibrated)"
    raw = float(raw)
    if metric["comparator"] == "lte":
        relaxed = multiplier * raw
        return max(floor, relaxed), f"max(floor={floor}, {multiplier}×noise={relaxed:.4g})"
    relaxed = 1.0 - multiplier * (1.0 - raw)
    return max(floor, relaxed), f"max(floor={floor}, 1−{multiplier}×(1−noise)={relaxed:.4g})"


def _verify_one_tier(p, name: str, task_dir: str, tier: str,
                     thresholds: list, intrinsic_noise: dict,
                     reps: int, verbose: bool,
                     *,
                     use_baseline_cache: bool = True,
                     baseline_confirm_sigma: float = 3.0,
                     no_baseline_confirm: bool = False,
                     patched_only: bool = False) -> dict:
    """Run a single tier with `reps` measurement loops. Throws on failure.

    Each rep spawns two fresh-Python subprocesses (baseline + patched). The
    parent does NO in-process activate/deactivate — only orchestrates subprocesses
    and runs evaluate on their saved outputs.

    Baseline cache (``use_baseline_cache=True``, the default): if a valid
    cache entry exists for (tier, threads) — i.e. ``reference_output_<tier>/``
    is populated and timing is available from ``baseline_noise.json``,
    ``baselines_history.tsv``, or the ``results.tsv`` baseline row — the
    baseline subprocess is replaced by copying the cached outputs into
    ``ref_dir`` and reusing the cached timing. One confirmation rep is run
    by default (``no_baseline_confirm=False``): the measured baseline is
    compared to the cached mean; if within ``K * cached_stdev`` (with σ
    floored at 1% of mean), the cache is accepted for all remaining reps;
    otherwise the cache is invalidated and a full fresh baseline is taken.

    Auto-escalation: when ``reps == 2``, after the second rep, the per-rep
    patched spread (max/min) is checked. If > 1.20, one extra rep is run.
    Explicit ``reps >= 3`` disables this. Under the legacy semantics this
    looked at speedup_x spread (baseline/patched ratio); for cache hits
    baseline is constant so the patched-spread check is the equivalent
    signal — and it behaves identically on cache miss (since both noise
    sources are independent there).
    """
    # ---- Probe cache once per tier (before the rep loop) -----------
    threads_now = _resolve_threads()

    # ---- patched-only mode: measure ONLY the patch (no baseline, no
    # concordance). For cells whose baseline OOMs on THIS machine but whose
    # patch fits — proves the optimized version runs here and records its
    # wall-time + peak RSS. speedup is NA (no local baseline); correctness is
    # verified separately on a box that can run the baseline. The package_verify
    # writer leaves speedup_x/pass blank automatically when baseline_secs == [].
    if patched_only:
        po_patched_secs: list[float] = []
        po_patched_peaks: list[float | None] = []
        for rep in range(1, int(reps) + 1):
            if verbose:
                print(f"=== rep {rep}/{reps} (patched-only) ===", file=sys.stderr)
            temp_dir = tempfile.mkdtemp(prefix=".autozyme_verify_", dir=task_dir)
            try:
                pipeline_dir = os.path.join(temp_dir, "pipeline")
                os.makedirs(pipeline_dir, exist_ok=True)
                if verbose:
                    print(f"--- patched (autozyme::{name}) "
                          f"[patched-only, no baseline] ---", file=sys.stderr)
                elapsed_p, peak_p = _run_worker(name, task_dir, tier, pipeline_dir,
                                                activate=True, verbose=verbose)
                po_patched_secs.append(elapsed_p)
                po_patched_peaks.append(peak_p)
                if verbose:
                    peak_str = (f"  peak_rss: {peak_p:.1f} MB"
                                if peak_p is not None else "")
                    print(f"  time: {elapsed_p:.3f} sec{peak_str}", file=sys.stderr)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        pt_peaks = [pk for pk in po_patched_peaks if pk is not None]
        patched_sec = statistics.median(po_patched_secs) if po_patched_secs else 0.0
        patched_peak_mb = statistics.median(pt_peaks) if pt_peaks else None
        if verbose:
            p_str = ", ".join(f"{x:.2f}" for x in po_patched_secs)
            n = len(po_patched_secs)
            print(f"\n--- timing (patched-only, median of {n} rep"
                  f"{'s' if n != 1 else ''}) ---", file=sys.stderr)
            print(f"  patched:  {patched_sec:.3f} sec  (reps: {p_str})",
                  file=sys.stderr)
            if patched_peak_mb is not None:
                print(f"  peak_rss patched: {patched_peak_mb:.1f} MB", file=sys.stderr)
            print(f"  baseline: SKIPPED (patched-only) — speedup NA, "
                  f"concordance NA", file=sys.stderr)
        return {
            "baseline_sec":      None,
            "patched_sec":       patched_sec,
            "baseline_peak_mb":  None,
            "patched_peak_mb":   patched_peak_mb,
            "metrics":           {},
            "all_pass":          None,
            "reps":              len(po_patched_secs),
            "baseline_secs":     [],
            "patched_secs":      po_patched_secs,
            "baseline_peaks_mb": [],
            "patched_peaks_mb":  po_patched_peaks,
            "per_rep_pass":      [None] * len(po_patched_secs),
        }

    cached: Optional[CachedBaseline] = None
    if use_baseline_cache:
        cached = load_cached_baseline(name, task_dir, tier, threads_now, p)
        if verbose:
            if cached is not None:
                print(cache_hit_msg(tier, cached), file=sys.stderr)
            else:
                print(cache_miss_msg(tier), file=sys.stderr)

    cache_accepted = bool(cached is not None and no_baseline_confirm)
    # When cache hits and we accept without confirm, baseline doesn't run at
    # all; we still want `reps` patched samples. Decouple the two targets.
    target_baseline = (
        0 if (cached is not None and no_baseline_confirm)
        else (1 if cached is not None else reps)
    )
    target_patched = reps
    # Confirm-rep observation captured separately so the backfill writeback
    # can merge it with the cached mean as a second sample (n becomes 2 with
    # real stdev, instead of n=1 inherited from the history fallback).
    confirm_rep_sample: tuple[float, Optional[float]] | None = None

    baseline_secs: list[float] = []
    patched_secs: list[float] = []
    baseline_peaks_mb: list[float | None] = []
    patched_peaks_mb: list[float | None] = []
    per_rep_results: list[dict] = []
    per_rep_all_pass: list[bool] = []

    # `target_patched` may grow by 1 via auto-escalation; `rep` is 1-indexed.
    rep = 0
    while rep < target_patched:
        rep += 1
        if verbose and target_patched > 1:
            print(f"=== rep {rep}/{target_patched} ===", file=sys.stderr)

        # Anchor temp_dir under task_dir so an evaluate.py that resolves
        # framework helpers via an upward walk (looking for the
        # `autozyme-framework/` symlink at task root) finds them when
        # copied + run from this temp dir. The dir is removed in `finally`
        # below; this only changes where it lives, not its lifetime.
        temp_dir = tempfile.mkdtemp(prefix=".autozyme_verify_", dir=task_dir)
        try:
            pipeline_dir = os.path.join(temp_dir, "pipeline")
            ref_dir = os.path.join(temp_dir, f"reference_output_{tier}")
            os.makedirs(pipeline_dir, exist_ok=True)
            os.makedirs(ref_dir, exist_ok=True)

            # ---- BASELINE: decide per-rep what runs --------------------
            # 4 cases:
            #   (A) cache HIT, accepted     → copy cached outputs, no subprocess
            #   (B) cache HIT, --no-confirm → as (A), mark accepted
            #   (C) cache HIT, confirm rep  → subprocess + tolerance check
            #   (D) cache MISS / invalidated → subprocess every rep (legacy)
            ran_baseline_this_rep = False
            confirmed_this_rep = False
            if cached is not None and (cache_accepted or no_baseline_confirm):
                # (A) / (B): copy cached outputs into temp ref_dir
                shutil.rmtree(ref_dir)
                shutil.copytree(
                    cached.ref_dir_path,
                    ref_dir,
                    ignore_dangling_symlinks=True,
                )
                baseline_secs.append(cached.timing_mean)
                baseline_peaks_mb.append(cached.peak_mb)
                if (not cache_accepted) and no_baseline_confirm:
                    cache_accepted = True
            elif cached is not None and len(baseline_secs) == 0:
                # (C) Confirmation rep — run baseline subprocess to populate
                # ref_dir (subprocess writes its outputs; we may also accept
                # the cache mean as the reported timing).
                if verbose:
                    print(
                        f"--- baseline confirmation rep "
                        f"(cached {cached.timing_mean:.3f}s) ---",
                        file=sys.stderr,
                    )
                elapsed_b, peak_b = _run_worker(
                    name, task_dir, tier, ref_dir,
                    activate=False, verbose=verbose,
                )
                ran_baseline_this_rep = True
                # σ floor: 5% of cached mean. Wall-clock noise on shared
                # boxes routinely hits ±5% from CPU governor / page-cache /
                # background load — anything tighter rejects legit cache
                # hits as drift. Real stdev (when n>1) wins via max().
                sigma = max(cached.timing_stdev,
                            0.05 * cached.timing_mean)
                delta = abs(elapsed_b - cached.timing_mean)
                tol = baseline_confirm_sigma * sigma
                if delta <= tol:
                    cache_accepted = True
                    confirmed_this_rep = True
                    confirm_rep_sample = (elapsed_b, peak_b)
                    if verbose:
                        print(
                            f"  confirm OK: measured {elapsed_b:.3f}s "
                            f"vs cached {cached.timing_mean:.3f}s, "
                            f"|Δ|={delta:.3f} ≤ "
                            f"{baseline_confirm_sigma:g}σ "
                            f"({tol:.3f}s) — trusting cache",
                            file=sys.stderr,
                        )
                    # Use cached mean (median-of-many beats one fresh)
                    # for the reported stats.
                    baseline_secs.append(cached.timing_mean)
                    baseline_peaks_mb.append(cached.peak_mb)
                else:
                    if verbose:
                        print(
                            f"  confirm FAIL: measured {elapsed_b:.3f}s "
                            f"vs cached {cached.timing_mean:.3f}s, "
                            f"|Δ|={delta:.3f} > "
                            f"{baseline_confirm_sigma:g}σ "
                            f"({tol:.3f}s) — invalidating cache, "
                            f"falling back to fresh measurement",
                            file=sys.stderr,
                        )
                    cached = None
                    baseline_secs.append(elapsed_b)
                    baseline_peaks_mb.append(peak_b)
            else:
                # (D) Pure cache miss — run baseline subprocess every rep
                # (current/legacy semantics).
                if verbose:
                    upstreams = sorted({u for u, _, _ in p.targets})
                    print(
                        f"--- baseline (upstream "
                        f"{', '.join(upstreams)}) ---",
                        file=sys.stderr,
                    )
                elapsed_b, peak_b = _run_worker(
                    name, task_dir, tier, ref_dir,
                    activate=False, verbose=verbose,
                )
                ran_baseline_this_rep = True
                baseline_secs.append(elapsed_b)
                baseline_peaks_mb.append(peak_b)
                if verbose:
                    peak_str = (
                        f"  peak_rss: {peak_b:.1f} MB"
                        if peak_b is not None else ""
                    )
                    print(
                        f"  time: {elapsed_b:.3f} sec{peak_str}",
                        file=sys.stderr,
                    )

            # ---- PATCHED: unchanged ------------------------------------
            if verbose:
                print(f"--- patched (autozyme::{name}) ---", file=sys.stderr)
            elapsed_p, peak_p = _run_worker(name, task_dir, tier, pipeline_dir,
                                            activate=True, verbose=verbose)
            patched_secs.append(elapsed_p)
            patched_peaks_mb.append(peak_p)
            if verbose:
                peak_str = f"  peak_rss: {peak_p:.1f} MB" if peak_p is not None else ""
                print(f"  time: {elapsed_p:.3f} sec{peak_str}", file=sys.stderr)

            out_lines = _run_evaluate(task_dir, temp_dir, ref_dir, tier)

            # ---- OOD persistence: on first fresh baseline rep, promote ----
            # the temp ref_dir contents into task_dir/reference_output_<tier>/
            # so the next attest hits cache for this tier too. Defensive:
            # populate_persistent_ref_dir() never clobbers a non-empty dest.
            if cached is None and rep == 1:
                try:
                    populate_persistent_ref_dir(task_dir, tier, ref_dir)
                except (OSError, shutil.Error) as e:
                    if verbose:
                        print(
                            f"[baseline-cache] WARN: failed to promote "
                            f"ref_dir for {tier}: {e}",
                            file=sys.stderr,
                        )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        metrics = {}
        for line in out_lines:
            m = _METRIC_LINE_RE.match(line)
            if m:
                metrics[m.group(1)] = float(m.group(2))

        rep_results = {}
        rep_all_pass = True
        for t in thresholds:
            nm = t["name"]
            if nm not in metrics:
                raise RuntimeError(
                    f"metric '{nm}' declared in task.yaml not printed by evaluate"
                )
            val = metrics[nm]
            th, th_label = _effective_threshold(t, tier, intrinsic_noise)
            op = t["comparator"]
            if op == "gte":
                ok = val >= th
            elif op == "lte":
                ok = val <= th
            else:
                raise RuntimeError(f"unknown comparator '{op}' for metric '{nm}'")
            if not ok:
                rep_all_pass = False
            rep_results[nm] = {"value": val, "threshold": th,
                               "comparator": op, "pass": ok,
                               "threshold_label": th_label}
        per_rep_results.append(rep_results)
        per_rep_all_pass.append(rep_all_pass)

        # Auto-escalation: after the second rep on a target_patched=2 run,
        # check the spread of per-rep patched times (max/min). If it
        # disagrees by >20%, run one more. Only triggers when
        # target_patched started at 2 — explicit reps>=3 means the caller
        # wants a fixed sample size and we respect that.
        if rep == 2 and target_patched == 2:
            pp = [x for x in patched_secs if x > 0]
            if len(pp) >= 2:
                spread = max(pp) / min(pp)
                if spread > 1.20:
                    target_patched = 3
                    if verbose:
                        print(f"  -> auto-escalate: patched spread "
                              f"{spread:.2f}x > 1.20; running rep 3",
                              file=sys.stderr)

    reps_actual  = len(baseline_secs)
    baseline_sec = statistics.median(baseline_secs)
    patched_sec  = statistics.median(patched_secs)
    bl_peaks = [p for p in baseline_peaks_mb if p is not None]
    pt_peaks = [p for p in patched_peaks_mb  if p is not None]
    baseline_peak_mb = statistics.median(bl_peaks) if bl_peaks else None
    patched_peak_mb  = statistics.median(pt_peaks) if pt_peaks else None
    results      = per_rep_results[0]
    all_pass     = all(per_rep_all_pass)

    if verbose:
        print("\n--- metrics (rep 1) ---", file=sys.stderr)
        for nm, r in results.items():
            op_s = ">=" if r["comparator"] == "gte" else "<="
            verdict = "PASS" if r["pass"] else "FAIL"
            label = r.get("threshold_label", "absolute")
            suffix = f"  [{label}]" if label != "absolute" else ""
            print(f"  {nm:<32} {r['value']:.6f}   {op_s} {r['threshold']:.4g}   {verdict}{suffix}",
                  file=sys.stderr)
        suffix = "s" if reps_actual > 1 else ""
        print(f"\n--- timing (median of {reps_actual} rep{suffix}) ---", file=sys.stderr)
        b_str = ", ".join(f"{x:.2f}" for x in baseline_secs)
        p_str = ", ".join(f"{x:.2f}" for x in patched_secs)
        print(f"  baseline: {baseline_sec:.3f} sec  (reps: {b_str})", file=sys.stderr)
        print(f"  patched:  {patched_sec:.3f} sec  (reps: {p_str})", file=sys.stderr)
        speedup_pct = (baseline_sec - patched_sec) / baseline_sec * 100 if baseline_sec else 0.0
        speedup_x = baseline_sec / patched_sec if patched_sec else float("inf")
        print(f"  speedup:  {speedup_pct:.1f}% ({speedup_x:.1f}x)", file=sys.stderr)
        if baseline_peak_mb is not None or patched_peak_mb is not None:
            bp_s = ", ".join(f"{x:.1f}" for x in bl_peaks) if bl_peaks else "—"
            pp_s = ", ".join(f"{x:.1f}" for x in pt_peaks) if pt_peaks else "—"
            print(f"  peak_rss baseline: "
                  f"{baseline_peak_mb if baseline_peak_mb is not None else float('nan'):.1f} MB  "
                  f"(reps: {bp_s})", file=sys.stderr)
            print(f"  peak_rss patched:  "
                  f"{patched_peak_mb if patched_peak_mb is not None else float('nan'):.1f} MB  "
                  f"(reps: {pp_s})", file=sys.stderr)
            if (baseline_peak_mb is not None and patched_peak_mb is not None
                    and baseline_peak_mb > 0):
                delta_pct = (baseline_peak_mb - patched_peak_mb) / baseline_peak_mb * 100
                print(f"  peak_rss saving:   {delta_pct:+.1f}%", file=sys.stderr)
        if reps_actual > 1:
            n_pass = sum(per_rep_all_pass)
            print(f"  all reps pass: {'yes' if all_pass else 'NO'} ({n_pass}/{reps_actual})",
                  file=sys.stderr)
        print(f"\n--- verdict ({tier}): {'ALL PASS' if all_pass else 'SOME FAIL'} ---",
              file=sys.stderr)

    # ---- Cache writeback ----------------------------------------------
    # Writeback fires when we measured fresh baseline samples (cache miss
    # path, or confirmation-fail mid-loop). On confirmation-accepted cache
    # hits, `cached` is still set and the entry is already authoritative —
    # do not overwrite (would replace a multi-rep cached mean with a
    # single-rep observation). The persistent `reference_output_<tier>/`
    # was promoted inside the rep loop (rep == 1 of the fresh path).
    # Writeback policy: fresh observations always update the cache (even
    # under --rerun-baseline — otherwise the next attest would hit the
    # same stale entry). To suppress this, callers can pre-delete
    # .zyme/baseline_noise.json.
    fresh_observations = [s for s in baseline_secs if s > 0]
    if (cached is None and fresh_observations):
        try:
            persistent_ref = os.path.join(task_dir,
                                          f"reference_output_{tier}")
            if os.path.isdir(persistent_ref) and any(
                Path(persistent_ref).rglob("*")
            ):
                # task.yaml may give us the dataset name for this tier.
                dataset_name = ""
                try:
                    import yaml
                    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as fh:
                        tk = yaml.safe_load(fh) or {}
                    for d in (tk.get("datasets") or []):
                        if d.get("tier") == tier:
                            dataset_name = d.get("name") or ""
                            break
                except Exception:  # noqa: BLE001
                    pass
                write_cached_baseline(
                    task_dir=task_dir,
                    tier=tier,
                    threads=threads_now,
                    dataset_name=dataset_name,
                    speeds=fresh_observations,
                    peaks=[pk for pk in baseline_peaks_mb if pk is not None],
                    versions=_collect_versions(p),
                    ref_dir=persistent_ref,
                    produced_by="attest",
                )
                if verbose:
                    print(
                        f"[baseline-cache] WROTE {tier}: "
                        f"{statistics.mean(fresh_observations):.3f}s "
                        f"(n={len(fresh_observations)}) → "
                        f".zyme/baseline_noise.json + "
                        f"reference_output_{tier}/",
                        file=sys.stderr,
                    )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(
                    f"[baseline-cache] WARN: writeback failed for {tier}: {e}",
                    file=sys.stderr,
                )
    elif (cached is not None and cache_accepted
          and not cached.has_version_stamp):
        # Backfill version stamp + sha onto an existing artifact that we
        # accepted via the permissive HIT path (no measurement to record;
        # use the cached timing as the persisted speed_mean).
        try:
            persistent_ref = cached.ref_dir_path
            dataset_name = ""
            try:
                import yaml
                with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as fh:
                    tk = yaml.safe_load(fh) or {}
                for d in (tk.get("datasets") or []):
                    if d.get("tier") == tier:
                        dataset_name = d.get("name") or ""
                        break
            except Exception:  # noqa: BLE001
                pass
            # Merge confirm-rep observation (if any) with cached mean so the
            # backfill entry gets n>=2 with real stdev — subsequent attests
            # then see a properly-calibrated cache.
            backfill_speeds = [cached.timing_mean]
            backfill_peaks: list[float] = []
            if cached.peak_mb is not None:
                backfill_peaks.append(cached.peak_mb)
            if confirm_rep_sample is not None:
                backfill_speeds.append(float(confirm_rep_sample[0]))
                if confirm_rep_sample[1] is not None:
                    backfill_peaks.append(float(confirm_rep_sample[1]))
            write_cached_baseline(
                task_dir=task_dir,
                tier=tier,
                threads=threads_now,
                dataset_name=dataset_name,
                speeds=backfill_speeds,
                peaks=backfill_peaks,
                versions=_collect_versions(p),
                ref_dir=persistent_ref,
                produced_by="attest-backfill",
            )
            if verbose:
                print(
                    f"[baseline-cache] BACKFILL {tier}: stamped "
                    f"upstream_versions + sha256 onto existing artifact "
                    f"(cached {cached.timing_mean:.3f}s preserved)",
                    file=sys.stderr,
                )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(
                    f"[baseline-cache] WARN: backfill failed for {tier}: {e}",
                    file=sys.stderr,
                )

    return {
        "baseline_sec":     baseline_sec,
        "patched_sec":      patched_sec,
        "baseline_peak_mb": baseline_peak_mb,
        "patched_peak_mb":  patched_peak_mb,
        "metrics":          results,
        "all_pass":         all_pass,
        "reps":             reps_actual,
        "baseline_secs":    baseline_secs,
        "patched_secs":     patched_secs,
        "baseline_peaks_mb": baseline_peaks_mb,
        "patched_peaks_mb":  patched_peaks_mb,
        "per_rep_pass":     per_rep_all_pass,
    }


_PACKAGE_VERIFY_HEADER = (
    "timestamp\tpatch_name\ttier\tdataset\trep_idx\tvariant\t"
    "sec\tspeedup_pct\tspeedup_x\t"
    "peak_mb\tpeak_mb_change_pct\tpeak_mb_fold\t"
    "pass\tmetrics_json\tframework_version\tpackage_version\tnote\t"
    "system_os\tsystem_cpu\tsystem_ram_gb\tsystem_threads"
)

# Pre-2026-05-28 long format (had dataset, no package_version). Upgraded on append.
_PACKAGE_VERIFY_HEADER_PRE_PKGVER = (
    "timestamp\tpatch_name\ttier\tdataset\trep_idx\tvariant\t"
    "sec\tspeedup_pct\tspeedup_x\t"
    "peak_mb\tpeak_mb_change_pct\tpeak_mb_fold\t"
    "pass\tmetrics_json\tframework_version\tnote\t"
    "system_os\tsystem_cpu\tsystem_ram_gb\tsystem_threads"
)

# Pre-2026-05-26 long format (no dataset column). Upgraded on append.
_PACKAGE_VERIFY_HEADER_LEGACY = (
    "timestamp\tpatch_name\ttier\trep_idx\tvariant\t"
    "sec\tspeedup_pct\tspeedup_x\t"
    "peak_mb\tpeak_mb_change_pct\tpeak_mb_fold\t"
    "pass\tmetrics_json\tframework_version\tnote\t"
    "system_os\tsystem_cpu\tsystem_ram_gb\tsystem_threads"
)

def _detect_cpu_model() -> str:
    """Best-effort CPU model name string. Empty on failure."""
    sysname = platform.system()
    try:
        if sysname == "Windows":
            out = subprocess.run(
                ["wmic", "cpu", "get", "name", "/value"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Name="):
                    return line[len("Name="):].strip()
        elif sysname == "Linux":
            try:
                with open("/proc/cpuinfo") as fh:
                    for line in fh:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
        elif sysname == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
            if out:
                return out
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or ""


def _detect_ram_gb() -> float | None:
    """Total physical RAM in GB (1 decimal). None on failure."""
    try:
        import psutil  # type: ignore[import-not-found]
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:  # noqa: BLE001
        pass
    sysname = platform.system()
    try:
        if sysname == "Windows":
            import ctypes

            class _MEMSTATEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("sullAvailExtendedVirtual", ctypes.c_uint64),
                ]
            stat = _MEMSTATEX()
            stat.dwLength = ctypes.sizeof(_MEMSTATEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 1)
        if sysname == "Linux":
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        if sysname == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
            if out:
                return round(int(out) / (1024 ** 3), 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _collect_system_info() -> dict[str, str]:
    """Capture a compact system fingerprint for the attest row.

    All four fields are best-effort: missing values become empty strings.

      system_os       e.g. "Windows 11" / "Linux 5.15.0-92-generic" / "macOS 14.5"
      system_cpu      e.g. "AMD Ryzen 9 7950X 16-Core Processor"
      system_ram_gb   total physical RAM in GB, 1 decimal place
      system_threads  value of ZYME_THREADS at run time (falls back to
                      AUTOZYME_THREADS / OMP / MKL / OPENBLAS / "")
    """
    sysname = platform.system()
    if sysname == "Darwin":
        ver = platform.mac_ver()[0]
        os_str = f"macOS {ver}".strip()
    elif sysname == "Windows":
        release = platform.release()
        # On Windows 11 the kernel still reports release "10"; disambiguate
        # via build number (Win11 = build >= 22000).
        try:
            build = int(platform.version().split(".")[2])
            if build >= 22000 and release == "10":
                release = "11"
        except (ValueError, IndexError):
            pass
        os_str = f"Windows {release}".strip()
    else:
        os_str = f"{sysname} {platform.release()}".strip()
    cpu = _detect_cpu_model()
    ram = _detect_ram_gb()
    threads = (
        os.environ.get("ZYME_THREADS")
        or os.environ.get("AUTOZYME_THREADS")
        or os.environ.get("AUTOZYMER_THREADS")
        or os.environ.get("OMP_NUM_THREADS")
        or os.environ.get("MKL_NUM_THREADS")
        or os.environ.get("OPENBLAS_NUM_THREADS")
        or ""
    )
    return {
        "system_os": os_str,
        "system_cpu": cpu,
        "system_ram_gb": "" if ram is None else f"{ram:.1f}",
        "system_threads": threads,
    }


def _framework_version() -> str:
    try:
        from autozyme import __version__ as v
        return str(v)
    except Exception:  # noqa: BLE001
        return "unknown"


def _tsv_sanitize(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    for ch in ("\t", "\n", "\r"):
        s = s.replace(ch, " ")
    return s


def _normalize_tsv_header_line(line: str) -> str:
    """Strip BOM / CRLF / outer whitespace from a TSV header line."""
    if not line:
        return ""
    if line.startswith("\ufeff"):
        line = line[1:]
    if line.endswith("\r"):
        line = line[:-1]
    return line.strip()


def _header_cols_of_line(line: str) -> list[str]:
    return _normalize_tsv_header_line(line).split("\t")


def _is_known_package_verify_header_cols(cols: list[str]) -> bool:
    if not cols:
        return False
    current = _PACKAGE_VERIFY_HEADER.split("\t")
    pre_pkgver = _PACKAGE_VERIFY_HEADER_PRE_PKGVER.split("\t")
    legacy = _PACKAGE_VERIFY_HEADER_LEGACY.split("\t")
    if cols in (current, pre_pkgver, legacy):
        return True
    # Forward compat: on-disk header is a strict superset of a known schema.
    for known in (current, pre_pkgver, legacy):
        if all(c in cols for c in known):
            return True
    return False


def _assert_long_format_or_empty(tsv_path: str) -> None:
    """Refuse to append to a TSV whose header doesn't match the current schema."""
    if not os.path.exists(tsv_path):
        return
    with open(tsv_path, encoding="utf-8") as f:
        line1 = f.readline().rstrip("\n")
    norm = _normalize_tsv_header_line(line1)
    if not norm:
        return
    if norm in (_PACKAGE_VERIFY_HEADER, _PACKAGE_VERIFY_HEADER_PRE_PKGVER, _PACKAGE_VERIFY_HEADER_LEGACY):
        return
    if _is_known_package_verify_header_cols(_header_cols_of_line(line1)):
        return
    raise RuntimeError(
        f"{tsv_path} has an unrecognized header: {norm!r}"
    )


def _find_attest_manifest(task_dir: str, patch_name: str) -> Optional[str]:
    """Walk up from task_dir looking for scripts/<patch>_attest_manifest.yaml."""
    d = os.path.abspath(task_dir)
    for _ in range(6):
        candidate = os.path.join(d, "scripts", f"{patch_name}_attest_manifest.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _resolve_inst_speedup_path(task_dir: str, patch_name: str) -> Optional[str]:
    """Resolve the inst/speedups TSV that mirrors this task's package_verify.

    Reads ``<framework>/scripts/<patch>_attest_manifest.yaml``, matches the
    task by its ``path`` field, and returns the absolute path to
    ``<autozyme_pkg>/<patch>/speedups/<patch>_<legacy_key>.tsv``.

    Returns None when the manifest isn't reachable from task_dir (e.g. the
    task lives outside the framework tree) or the task isn't registered.
    """
    manifest_path = _find_attest_manifest(task_dir, patch_name)
    if manifest_path is None:
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
    except OSError:
        return None
    framework_root = os.path.dirname(os.path.dirname(manifest_path))
    task_abs = os.path.abspath(task_dir)
    legacy_key = None
    for task in manifest.get("tasks") or []:
        rel = task.get("path") or ""
        if os.path.abspath(os.path.join(framework_root, rel)) == task_abs:
            package_speedups = task.get("package_speedups", True)
            if (package_speedups is False or
                    str(package_speedups).strip().lower() in {"false", "0", "no"}):
                return None
            legacy_key = task.get("legacy_key") or task.get("id")
            break
    if not legacy_key:
        return None
    pkg_dir = os.path.dirname(__file__)
    return os.path.join(pkg_dir, patch_name, "speedups",
                        f"{patch_name}_{legacy_key}.tsv")


def _write_inst_speedup_tsv(path: str,
                            combined_rows: list[dict[str, str]]) -> None:
    """Mirror publishable rows to inst/speedups TSV.

    Atomic write (tmp + rename). Same header as package_verify.tsv. Skips
    sentinels, non-measured rows, and patched rows that did not pass; migrated
    benchmark rows are retained because they are still valid published results.
    """
    def _group_key(row: dict[str, str]) -> tuple[str, ...]:
        return (
            (row.get("timestamp") or "").strip(),
            (row.get("patch_name") or "").strip(),
            (row.get("tier") or "").strip(),
            (row.get("framework_version") or "").strip(),
            (row.get("note") or "").strip(),
            (row.get("system_os") or "").strip(),
            (row.get("system_cpu") or "").strip(),
            (row.get("system_ram_gb") or "").strip(),
            (row.get("system_threads") or "").strip(),
        )

    def _measured_variant(row: dict[str, str]) -> str:
        if not (row.get("sec") or "").strip():
            return ""
        variant = (row.get("variant") or "").strip()
        if variant not in {"baseline", "patched"}:
            return ""
        return variant

    passing_groups = {
        _group_key(r)
        for r in combined_rows
        if _measured_variant(r) == "patched"
        and (r.get("pass") or "").strip().lower() in {"true", "1", "yes"}
    }
    filtered = []
    for row in combined_rows:
        variant = _measured_variant(row)
        if not variant or _group_key(row) not in passing_groups:
            continue
        if variant == "patched":
            pass_cell = (row.get("pass") or "").strip().lower()
            if pass_cell not in {"true", "1", "yes"}:
                continue
        filtered.append(row)
    if not filtered:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header_cols = _PACKAGE_VERIFY_HEADER.split("\t")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_PACKAGE_VERIFY_HEADER + "\n")
        for row in filtered:
            cells = [row.get(c, "") for c in header_cols]
            f.write("\t".join(cells) + "\n")
    os.replace(tmp_path, path)


def _tier_dataset_map(task_dir: str) -> dict[str, str]:
    """Map tier -> task.yaml datasets[].name."""
    yaml_path = os.path.join(task_dir, "task.yaml")
    if not os.path.isfile(yaml_path):
        return {}
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as fh:
            task = yaml.safe_load(fh) or {}
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for ds in task.get("datasets") or []:
        tier = str(ds.get("tier") or "").strip()
        dname = str(ds.get("name") or "").strip()
        if tier and dname:
            out[tier] = dname
    return out


def _append_package_verify_tsv(task_dir: str, name: str,
                                rows: list[dict[str, Any]]) -> None:
    """Write per-rep × variant rows to <task_dir>/package_verify.tsv.

    Long format (since 2026-05-22): each batch in ``rows`` is expanded into
    2N rows — N baseline rows (rep_idx=1..N) and N patched rows. The file is
    kept globally sorted (variant → tier → platform → threads → timestamp →
    rep_idx) — read-modify-write keeps all baseline rows above all patched
    rows even across many attest runs.
    For batches that crashed (no measurements collected), a single sentinel
    row with empty rep_idx/variant/sec is appended so the failure stays in
    the file's history.
    """
    tsv_path = os.path.join(task_dir, "package_verify.tsv")
    _assert_long_format_or_empty(tsv_path)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    fw_version = _framework_version()
    sysinfo = _collect_system_info()
    tier_map = _tier_dataset_map(task_dir)
    # Read the active patch's declared tested_against (e.g. "scanpy 1.11.5")
    pkg_version = ""
    try:
        from ._core import _REGISTRY
        patch = _REGISTRY.get(name)
        if patch is not None and getattr(patch, "tested_against", None):
            pkg_version = str(patch.tested_against)
    except Exception:
        pass

    header_cols = _PACKAGE_VERIFY_HEADER.split("\t")
    pre_pkgver_cols = _PACKAGE_VERIFY_HEADER_PRE_PKGVER.split("\t")
    legacy_cols = _PACKAGE_VERIFY_HEADER_LEGACY.split("\t")

    # Read existing rows so we can sort the union with new ones.
    existing: list[dict[str, str]] = []
    if os.path.exists(tsv_path):
        import csv as _csv
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f, delimiter="\t")
            fieldnames = list(reader.fieldnames or [])
            upgraded_cols = legacy_cols[:3] + ["dataset"] + legacy_cols[3:]
            if fieldnames == header_cols:
                existing = [dict(r) for r in reader]
            elif fieldnames in (pre_pkgver_cols, legacy_cols, upgraded_cols):
                for r in reader:
                    row = dict(r)
                    tier = (row.get("tier") or "").strip()
                    if "dataset" not in row or not (row.get("dataset") or "").strip():
                        row["dataset"] = tier_map.get(tier, "")
                    # package_version missing in old files: leave blank
                    existing.append({c: row.get(c, "") for c in header_cols})

    def _fmt_num(x: Any) -> str:
        return f"{x:.6f}" if isinstance(x, (int, float)) and x is not None else ""

    def _pass_cell(value: Any) -> str:
        if value is None:
            return ""
        return "true" if value else "false"

    def _make_row(rep_idx: str, variant: str,
                  sec: str, speedup_pct: str, speedup_x: str,
                  peak_mb: str, peak_pct: str, peak_fold: str,
                  pass_val: str, metrics_json: str,
                  tier: str, note: str) -> dict[str, str]:
        ds = tier_map.get((tier or "").strip(), "")
        return {
            "timestamp": timestamp,
            "patch_name": _tsv_sanitize(name),
            "tier": _tsv_sanitize(tier),
            "dataset": _tsv_sanitize(ds),
            "rep_idx": rep_idx,
            "variant": variant,
            "sec": sec,
            "speedup_pct": speedup_pct,
            "speedup_x": speedup_x,
            "peak_mb": peak_mb,
            "peak_mb_change_pct": peak_pct,
            "peak_mb_fold": peak_fold,
            "pass": pass_val,
            "metrics_json": _tsv_sanitize(metrics_json),
            "framework_version": _tsv_sanitize(fw_version),
            "package_version": _tsv_sanitize(pkg_version),
            "note": _tsv_sanitize(note),
            "system_os": _tsv_sanitize(sysinfo["system_os"]),
            "system_cpu": _tsv_sanitize(sysinfo["system_cpu"]),
            "system_ram_gb": _tsv_sanitize(sysinfo["system_ram_gb"]),
            "system_threads": _tsv_sanitize(sysinfo["system_threads"]),
        }

    def _median_positive(xs: list[Any]) -> float | None:
        vals = [float(x) for x in xs
                if isinstance(x, (int, float)) and x is not None and x > 0]
        if not vals:
            return None
        from statistics import median
        return median(vals)

    new_rows: list[dict[str, str]] = []
    for r in rows:
        tier = r.get("tier") or ""
        note = r.get("note") or ""
        baseline_secs = r.get("baseline_secs") or []
        patched_secs = r.get("patched_secs") or []
        baseline_peaks = r.get("baseline_peaks_mb") or []
        patched_peaks = r.get("patched_peaks_mb") or []
        per_rep_pass = r.get("per_rep_pass") or []
        metrics_json = r.get("metrics_json") or ""

        if not baseline_secs and not patched_secs:
            new_rows.append(_make_row("", "", "", "", "", "", "", "",
                                      "", "", tier, note))
            continue

        b_sec = _median_positive(baseline_secs)
        b_mb = _median_positive(baseline_peaks)

        for i, sec in enumerate(baseline_secs, start=1):
            peak = baseline_peaks[i - 1] if i - 1 < len(baseline_peaks) else None
            new_rows.append(_make_row(str(i), "baseline",
                                      _fmt_num(sec), "", "",
                                      _fmt_num(peak), "", "",
                                      "", "", tier, note))
        for i, sec in enumerate(patched_secs, start=1):
            peak = patched_peaks[i - 1] if i - 1 < len(patched_peaks) else None
            rep_pass = per_rep_pass[i - 1] if i - 1 < len(per_rep_pass) else None
            speedup_x_cell = speedup_pct_cell = ""
            peak_fold_cell = peak_pct_cell = ""
            if (b_sec is not None and isinstance(sec, (int, float))
                    and sec is not None and sec > 0):
                speedup_x_cell = f"{b_sec / sec:.6f}"
                speedup_pct_cell = f"{(b_sec - sec) / b_sec * 100:.6f}"
            if (b_mb is not None and isinstance(peak, (int, float))
                    and peak is not None and peak > 0):
                peak_fold_cell = f"{b_mb / peak:.6f}"
                peak_pct_cell = f"{(b_mb - peak) / b_mb * 100:.6f}"
            new_rows.append(_make_row(str(i), "patched",
                                      _fmt_num(sec), speedup_pct_cell, speedup_x_cell,
                                      _fmt_num(peak), peak_pct_cell, peak_fold_cell,
                                      _pass_cell(rep_pass), metrics_json,
                                      tier, note))

    combined = existing + new_rows
    combined.sort(key=_sort_key_for_long_row)

    # Atomic write: temp file + rename.
    tmp_path = tsv_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_PACKAGE_VERIFY_HEADER + "\n")
        for row in combined:
            cells = [row.get(c, "") for c in header_cols]
            f.write("\t".join(cells) + "\n")
    os.replace(tmp_path, tsv_path)

    # Mirror to <pkg>/<patch>/speedups/<patch>_<legacy_key>.tsv — the
    # shipped per-step table that activate() readers consume. Folds in
    # what `scripts/sync_*_attest_speedups.py` used to do as a separate
    # step; running attest is now sufficient to update the published TSV.
    inst_path = _resolve_inst_speedup_path(task_dir, name)
    if inst_path:
        try:
            _write_inst_speedup_tsv(inst_path, combined)
        except OSError as e:
            print(f"warning: could not mirror to {inst_path}: {e}",
                  file=sys.stderr)


_TIERS_ORDER_LONG = ("tiny", "small", "medium", "large", "ood_large",
                     "ood_xlarge")
_VARIANT_ORDER_LONG = {"baseline": 0, "patched": 1}


def _sort_key_for_long_row(row: dict[str, str]) -> tuple:
    """Mirror parsers.package_verify_tsv.sort_rows_for_output."""
    variant = (row.get("variant") or "").strip()
    variant_idx = _VARIANT_ORDER_LONG.get(variant, 2)
    tier = (row.get("tier") or "").strip()
    tier_idx = (_TIERS_ORDER_LONG.index(tier) if tier in _TIERS_ORDER_LONG
                else len(_TIERS_ORDER_LONG))
    os_raw = (row.get("system_os") or "").strip().lower()
    if not os_raw:
        plat = "mac"
    elif os_raw.startswith("windows"):
        plat = "win"
    elif os_raw.startswith("macos") or os_raw.startswith("darwin"):
        plat = "mac"
    else:
        plat = "unknown"
    try:
        threads = int((row.get("system_threads") or "0").strip() or 0)
    except ValueError:
        threads = 0
    ts = (row.get("timestamp") or "").strip()
    try:
        rep_idx = int((row.get("rep_idx") or "0").strip() or 0)
    except ValueError:
        rep_idx = 0
    return (variant_idx, tier_idx, plat, threads, ts, rep_idx)


def verify_patch(
    name: str,
    task_dir: str,
    tiers: tuple[str, ...] = ("tiny", "medium", "large", "ood_large", "ood_xlarge"),
    reps: int = 2,
    verbose: bool = True,
    *,
    use_baseline_cache: bool = True,
    baseline_confirm_sigma: float = 3.0,
    no_baseline_confirm: bool = False,
    patched_only: bool = False,
) -> list[dict[str, Any]]:
    """Verify a registered patch end-to-end against an autozyme task.

    ``patched_only=True`` measures ONLY the patched variant (no baseline run,
    no concordance) — for cells whose baseline OOMs on this machine but whose
    patch fits. Rows carry the patched sec/peak with blank speedup/pass.

    Runs the patch's smoke recipe across one or more dataset tiers and
    produces a per-tier summary. Each tier's run does its own baseline +
    patched measurement (each in its own subprocess), saves outputs, then
    invokes the task's evaluate.{py,R} once on the saved pair, with `reps`
    repeats to mitigate timing noise. `all_pass` requires every rep at every
    tier to pass thresholds.

    Default `tiers` are designed for both CI and external reporting:
        tiny       — best-case algorithmic speedup (smallest dev tier)
        medium     — primary iterate-loop tier (in-distribution)
        large      — dev-set max scale (in-distribution scaling check)
        ood_large  — held-out OOD at dev-tier scale (generalization signal)
        ood_xlarge — production-scale OOD (full generalization + scale)

    For CI smoke tests, pass `tiers=("tiny",)` explicitly. A tier whose
    dataset is missing is reported with NA values and a skip note (common
    for naturally-bounded functions without an `ood_xlarge` candidate).

    Side effect: appends per-tier rows to `<task_dir>/package_verify.tsv`
    (header written on first call). This is the persisted form of the
    paper-headline "package" number — fresh-subprocess timings matched in
    thread config, distinct from `results.tsv` (iter, in-process) and
    `verify.tsv` (Phase 3 validate, threading × OOD sweep). Each invocation
    appends; the file accumulates history like `results.tsv`.

    Args:
        name: registered patch name.
        task_dir: path to the autozyme task directory.
        tiers: tier names to run.
        reps: per-tier measurement reps. Median timing reported; `all_pass`
            requires every rep to pass thresholds. Default 2 — fast-path
            for stable patches. When `reps == 2`, an auto-escalation kicks
            in: if the two reps' speedup_x disagree by >20% (max/min ratio
            > 1.20), one extra rep is added to stabilize the median.
            Explicit `reps >= 3` disables this — the caller asked for a
            fixed sample size and gets exactly that.
        verbose: if True, prints per-tier verbose progress and a final
            summary table.

    Returns:
        List of dicts (one per tier) with keys: tier, baseline_sec,
        patched_sec, speedup_x, all_pass, reps, note, baseline_secs,
        patched_secs, baseline_peak_mb, patched_peak_mb, baseline_peaks_mb,
        patched_peaks_mb, metrics_json. Peak fields are OS RSS (median of
        reps for the scalar, per-rep list for the *_peaks_mb variant);
        ``None`` when the platform lacks ``resource.getrusage`` and psutil.
    """
    if name not in _REGISTRY:
        _import_submodule(name)
    p = _REGISTRY.get(name)
    if p is None:
        raise KeyError(f"no patch registered for {name!r}")
    from autozyme._smoke import resolve_smoke
    if resolve_smoke(task_dir, p) is None:
        raise ValueError(
            f"patch {name!r} has no smoke recipe — add attest/smoke.py under the task "
            f"or pass smoke=dict(load, call, save) to register_patch()"
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
    if isinstance(tiers, str):
        tiers = (tiers,)
    tiers = tuple(tiers)
    if not tiers:
        raise ValueError("tiers must be non-empty")

    task_dir = os.path.abspath(task_dir)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"task_dir does not exist: {task_dir}")

    ensure_process_thread_env(task_dir, patch_name=name)

    task = _read_task_yaml(task_dir)
    thresholds = task.get("metrics") or []
    if not thresholds:
        raise ValueError(f"{task_dir}/task.yaml has no `metrics:` section")
    intrinsic_noise = task.get("intrinsic_noise") or {}

    rows: list[dict[str, Any]] = []
    for tier in tiers:
        if verbose:
            print(f"\n############ tier = {tier} ############", file=sys.stderr)
        try:
            res = _verify_one_tier(p, name, task_dir, tier, thresholds,
                                    intrinsic_noise, reps, verbose,
                                    use_baseline_cache=use_baseline_cache,
                                    baseline_confirm_sigma=baseline_confirm_sigma,
                                    no_baseline_confirm=no_baseline_confirm,
                                    patched_only=patched_only)
            metrics_payload = {nm: r["value"] for nm, r in res["metrics"].items()}
            rows.append({
                "tier":             tier,
                "all_pass":         res["all_pass"],
                "reps":             res["reps"],
                "baseline_secs":    list(res["baseline_secs"]),
                "patched_secs":     list(res["patched_secs"]),
                "baseline_peaks_mb": list(res["baseline_peaks_mb"]),
                "patched_peaks_mb":  list(res["patched_peaks_mb"]),
                "per_rep_pass":     list(res.get("per_rep_pass") or []),
                "metrics_json":     json.dumps(metrics_payload, sort_keys=True),
                "note":             "patched-only" if patched_only else "",
            })
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if verbose:
                print(f"  [skip] {tier}: {msg}", file=sys.stderr)
            rows.append({
                "tier":             tier,
                "all_pass":         None,
                "reps":             None,
                "baseline_secs":    [],
                "patched_secs":     [],
                "baseline_peaks_mb": [],
                "patched_peaks_mb":  [],
                "per_rep_pass":     [],
                "metrics_json":     "",
                "note":             msg,
            })

    try:
        _append_package_verify_tsv(task_dir, name, rows)
    except OSError as e:
        print(f"warning: could not write package_verify.tsv: {e}",
              file=sys.stderr)

    if verbose:
        print(f"\n############ verify_patch: {name} ############", file=sys.stderr)
        header = (f"{'tier':<12}  {'baseline_sec':>12}  {'patched_sec':>12}  "
                  f"{'speedup':>10}  {'bl_mb':>8}  {'pt_mb':>8}  "
                  f"{'all_pass':<8}  note")
        print(header, file=sys.stderr)
        for r in rows:
            # Long-format row: per-rep lists are the source of truth; derive
            # medians/peaks/speedup for the summary line on the fly.
            b_secs = [x for x in (r.get("baseline_secs") or []) if x is not None]
            p_secs = [x for x in (r.get("patched_secs") or []) if x is not None]
            if not b_secs or not p_secs:
                print(f"{r['tier']:<12}  {'—':>12}  {'—':>12}  {'—':>10}  "
                      f"{'—':>8}  {'—':>8}  "
                      f"{'—':<8}  {r['note']}", file=sys.stderr)
                continue
            bl_sec = statistics.median(b_secs)
            pt_sec = statistics.median(p_secs)
            speedup_x = (bl_sec / pt_sec) if pt_sec > 0 else float("inf")
            b_pk = [x for x in (r.get("baseline_peaks_mb") or []) if x is not None]
            p_pk = [x for x in (r.get("patched_peaks_mb") or []) if x is not None]
            bl_mb = statistics.median(b_pk) if b_pk else None
            pt_mb = statistics.median(p_pk) if p_pk else None
            bl_s = f"{bl_mb:>8.1f}" if bl_mb is not None else f"{'—':>8}"
            pt_s = f"{pt_mb:>8.1f}" if pt_mb is not None else f"{'—':>8}"
            pass_str = "yes" if r["all_pass"] else "NO"
            print(
                f"{r['tier']:<12}  {bl_sec:>12.3f}  "
                f"{pt_sec:>12.3f}  {speedup_x:>9.1f}x  "
                f"{bl_s}  {pt_s}  "
                f"{pass_str:<8}  {r['note']}",
                file=sys.stderr,
            )

    return rows
