"""baseline-rebench command — reference-only re-bench at multiple thread counts.

K2 audit / fairness retrofit workflow:

  1. Edit `reference.{R,py}` so it reads ZYME_THREADS and branches into a
     parallel path when N>1 (e.g., wraps DESeq calls in BPPARAM/MulticoreParam).
  2. Run `zyme baseline-rebench --threads 1,4,8` (optionally `--tiers all`).
  3. The CLI re-times the reference at each (tier, thread) combination and
     records each as a (tier, thread) baseline row in results.tsv.
  4. If a verify.tsv exists, the CLI updates each cell's `baseline_speed` and
     `speedup_pct` columns to point at the matching (tier, thread) baseline,
     and re-renders verify.png/pdf/svg.

Crucially, this command does NOT re-run pipeline. The optimized pipeline's
speed_sec measurements already exist in verify.tsv from prior `zyme verify`
runs and are correct (commit-tagged + thread-tagged); only the divisor was
wrong. baseline-rebench fixes only the divisor.

If the user later wants pipeline timings re-measured too, they re-run
`zyme verify` (the existing path that re-runs everything).
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from zyme.utils import (
    LEGACY_THREAD,
    die, info, task_dir_from_args, zyme_state,
    resolve_reference_script, resolve_reference_output_dir,
)
from zyme.parsers.task_yaml import (
    parse_datasets, parse_baseline_threads, parse_metrics,
)
from zyme.parsers.results_tsv import (
    parse_log, dataset_in_results, ensure_k2_schema,
    get_baseline_speed,
)
from zyme.commands.baseline import (
    _append_baseline_history,
    _write_baseline_row_to_results,
    _update_baseline_row_in_results,
)
from zyme.runner import build_reference_cmd


def _run_reference_once(task_dir: Path, entry: dict, thread: int,
                         ref_path: Path, ref_out_dir: Path) -> tuple[float, float]:
    """Spawn the reference subprocess once with `ZYME_THREADS=<thread>`,
    capture stdout, parse `speed_sec:` / `peak_mb:`, return them.

    Streams stdout to the user's terminal as it runs.
    """
    try:
        cmd = build_reference_cmd(task_dir / "task.yaml", ref_path)
    except RuntimeError as e:
        die(str(e))

    env = os.environ.copy()
    env["ZYME_DATA_PATH"] = str(entry["path"])
    env["ZYME_REFERENCE_DIR"] = str(ref_out_dir)
    env["ZYME_TIER"] = entry["tier"]
    env["ZYME_THREADS"] = str(thread)
    env["OMP_NUM_THREADS"] = str(thread)
    env["OPENBLAS_NUM_THREADS"] = str(thread)
    env["MKL_NUM_THREADS"] = str(thread)

    info(f"[rebench] {ref_path.name} tier={entry['tier']} thread={thread}")
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(task_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    captured = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    if proc.returncode != 0:
        die(
            f"reference subprocess exited {proc.returncode} for "
            f"tier={entry['tier']} thread={thread}; baseline NOT recorded."
        )
    log = "".join(captured)
    speed, peak, _, _ = parse_log(log)
    if speed is None:
        die(f"reference completed but no speed_sec for tier={entry['tier']} "
            f"thread={thread}. Is emit_summary() called?")
    if peak is None:
        peak = 0.0
    return float(speed), float(peak)


def _migrate_verify_tsv_for_k2(verify_tsv: Path) -> None:
    """Add `thread` column to old verify.tsv if missing (and drop V1
    `thread_mode` column reading by ignoring it). Backfills `thread=1`
    on existing rows, leaves a `.prek2.bak` snapshot.

    Note: K2's verify.tsv schema has `thread` as the second column (already
    present in old V1 files at the same position!), so the only structural
    change is appending V1's `thread_mode` removal — which we don't do
    destructively; we just leave thread_mode in the header on old files
    and ignore it on read. This function is a stub for future cleanup.
    """
    if not verify_tsv.exists():
        return
    text = verify_tsv.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return
    header = lines[0].split("\t")
    if "thread" in header:
        return  # already has thread column (it's the V1/K2 verify.tsv 2nd column)
    # No thread column at all — extremely old verify.tsv. Backfill 1.
    bak = verify_tsv.with_suffix(verify_tsv.suffix + ".prek2.bak")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")
    # Insert thread column at position 1 (right after timestamp).
    new_header = [header[0], "thread"] + header[1:]
    new_lines = ["\t".join(new_header)]
    for line in lines[1:]:
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split("\t")
        new_parts = [parts[0], str(LEGACY_THREAD)] + parts[1:]
        new_lines.append("\t".join(new_parts))
    verify_tsv.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    info(f"[rebench] verify.tsv: backfilled `thread=1` on existing rows; "
         f"`.prek2.bak` snapshot saved.")


def _update_verify_tsv_baselines(verify_tsv: Path,
                                   new_baselines: dict[tuple[str, int], float]):
    """Walk verify.tsv. For each row whose (tier, thread) is in new_baselines,
    update its `baseline_speed` to the new value and recompute speedup_pct =
    (1 - speed_sec / new_baseline) * 100. Leaves speed_sec, metrics_json,
    commit, phase, status untouched.

    Returns a list of (tier, thread, n_rows_updated) tuples for reporting.
    """
    if not verify_tsv.exists():
        return []
    lines = verify_tsv.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    needed = {"tier", "thread", "speed_sec", "baseline_speed", "speedup_pct"}
    if not needed.issubset(col):
        return []
    counts: dict[tuple[str, int], int] = {}
    out = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            out.append(line)
            continue
        parts = line.split("\t")
        if len(parts) <= col["speedup_pct"]:
            out.append(line)
            continue
        tier = parts[col["tier"]]
        try:
            thread = int(parts[col["thread"]])
        except ValueError:
            out.append(line)
            continue
        key = (tier, thread)
        if key not in new_baselines:
            out.append(line)
            continue
        new_b = new_baselines[key]
        try:
            speed = float(parts[col["speed_sec"]])
        except ValueError:
            out.append(line)
            continue
        parts[col["baseline_speed"]] = f"{new_b:.3f}"
        if new_b > 0 and speed > 0:
            new_pct = (1 - speed / new_b) * 100.0
        else:
            new_pct = 0.0
        parts[col["speedup_pct"]] = f"{new_pct:.1f}"
        out.append("\t".join(parts))
        counts[key] = counts.get(key, 0) + 1
    verify_tsv.write_text("\n".join(out) + "\n", encoding="utf-8")
    return [(t, th, n) for (t, th), n in counts.items()]


def cmd_baseline_rebench(args):
    """Re-time `reference.{R,py}` at each (tier, thread) combination and
    update `verify.tsv` baselines + speedup_pct in place.

    Does NOT re-run pipeline. Pipeline timings already in verify.tsv stay
    untouched; only the baseline divisor and the derived speedup_pct change.
    Re-renders the verify figure at the end so the smoking-gun gap is
    immediately visible.
    """
    task_dir = task_dir_from_args(args)
    task_yaml = task_dir / "task.yaml"
    zyme_state(task_dir)
    ensure_k2_schema(task_dir)

    # Resolve threads: --threads explicit, else task.yaml::baseline_threads.
    if args.threads:
        threads = []
        for tok in args.threads.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                threads.append(int(tok))
            except ValueError:
                die(f"--threads: '{tok}' is not an integer")
        if not threads:
            die("--threads: empty list")
    else:
        threads = parse_baseline_threads(task_yaml)

    # Resolve tiers: --tiers explicit (or "all"), else all in task.yaml.
    available = parse_datasets(task_yaml)
    if args.tiers and args.tiers.strip().lower() not in ("all", "*"):
        wanted = {t.strip() for t in args.tiers.split(",") if t.strip()}
        tiers = [e for e in available if e["tier"] in wanted or e["name"] in wanted]
        if not tiers:
            die(f"--tiers: no match in task.yaml. Available: "
                f"{sorted(e['tier'] for e in available)}")
    else:
        tiers = list(available)

    ref_path = resolve_reference_script(task_dir)
    if not ref_path.exists():
        die(f"reference script not found: {ref_path}.")

    info(f"[rebench] tiers={[e['tier'] for e in tiers]}, threads={threads}")
    info(f"[rebench] reference script: {ref_path.name}")
    info(f"[rebench] not re-running pipeline; only re-timing reference.")

    # For each (tier, thread): run the reference once (or N reps), record
    # the median into results.tsv as a (tier, thread) baseline row, then
    # later patch verify.tsv's baseline_speed / speedup_pct columns.
    #
    # `--replicated` short-circuit (outcome-B fairness retrofit): the
    # reference is known-serial because upstream exposes no parallelism
    # knob. Re-running it at thread=4 / thread=8 just re-measures the same
    # serial code at higher CPU caps — cold-cache noise apart, the wall is
    # constant. So we run the reference exactly ONCE per tier (at thread=1,
    # the natural upper-bound on system noise) and replicate that
    # speed_sec/peak_mb to every requested (tier, thread>1) cell. Each
    # replicated row is description-tagged so downstream readers can tell
    # it apart from a real measurement.
    new_baselines: dict[tuple[str, int], float] = {}  # (tier, thread) -> speed
    n_reps = max(1, int(getattr(args, "reps", None) or 1))
    replicated = bool(getattr(args, "replicated", False))
    if replicated:
        info("[rebench] --replicated: running reference at thread=1 once per "
             "tier; thread>1 baselines copied from thread=1 (outcome-B fairness "
             "retrofit; upstream is serial regardless of ZYME_THREADS).")
    task_metrics = parse_metrics(task_yaml)
    metrics = {
        m["name"]: 1.0 if m["comparator"] == "gte" else 0.0
        for m in task_metrics
    }
    metrics_json = json.dumps(metrics) if metrics else "{}"

    for entry in tiers:
        tier = entry["tier"]
        ref_out_dir = resolve_reference_output_dir(task_dir, tier=tier)
        ref_out_dir.mkdir(parents=True, exist_ok=True)

        if replicated:
            # Run reference once at thread=1; replicate the value to every
            # requested thread. Forces 1 rep regardless of --reps because
            # additional reps would just re-time the same serial code.
            speed_t1, peak_t1 = _run_reference_once(
                task_dir, entry, 1, ref_path, ref_out_dir
            )
            for thread in threads:
                desc = (
                    f"upstream reference baseline (replicated from thread=1; "
                    f"outcome B)" if thread != 1
                    else f"upstream reference baseline (rebench, replicated mode)"
                )
                results_tsv = task_dir / "results.tsv"
                prior = get_baseline_speed(task_dir, entry["name"], thread=thread)
                if dataset_in_results(results_tsv, entry["name"], thread=thread):
                    _update_baseline_row_in_results(
                        task_dir, entry, thread=thread,
                        speed_sec=speed_t1, peak_mb=peak_t1,
                        metrics_json=metrics_json,
                        description=desc,
                    )
                else:
                    _write_baseline_row_to_results(
                        task_dir, entry, thread=thread,
                        speed_sec=speed_t1, peak_mb=peak_t1,
                        metrics_json=metrics_json, status="baseline",
                        description=desc,
                    )
                _append_baseline_history(
                    task_dir,
                    tier=tier, name=entry["name"],
                    speed_sec=speed_t1, peak_mb=peak_t1,
                    source="baseline-rebench (replicated)",
                    prior_speed_sec=prior,
                    thread=thread,
                )
                new_baselines[(tier, thread)] = speed_t1
                marker = "= thread=1" if thread != 1 else ""
                info(f"  → baseline at (tier={tier}, thread={thread}) = "
                     f"{speed_t1:.3f}s {marker}")
            continue

        for thread in threads:
            speeds = []
            peaks = []
            for rep in range(n_reps):
                if n_reps > 1:
                    info(f"  rep {rep+1}/{n_reps}")
                speed, peak = _run_reference_once(
                    task_dir, entry, thread, ref_path, ref_out_dir
                )
                speeds.append(speed)
                peaks.append(peak)
            speeds.sort()
            n = len(speeds)
            median_speed = (
                speeds[n // 2] if n % 2 else 0.5 * (speeds[n // 2 - 1] + speeds[n // 2])
            )
            median_peak = sum(peaks) / len(peaks)

            results_tsv = task_dir / "results.tsv"
            prior = get_baseline_speed(task_dir, entry["name"], thread=thread)
            if dataset_in_results(results_tsv, entry["name"], thread=thread):
                _update_baseline_row_in_results(
                    task_dir, entry, thread=thread,
                    speed_sec=median_speed, peak_mb=median_peak,
                    metrics_json=metrics_json,
                    description=f"upstream reference baseline (rebench, n={n_reps})",
                )
            else:
                _write_baseline_row_to_results(
                    task_dir, entry, thread=thread,
                    speed_sec=median_speed, peak_mb=median_peak,
                    metrics_json=metrics_json, status="baseline",
                    description=f"upstream reference baseline (rebench, n={n_reps})",
                )
            _append_baseline_history(
                task_dir,
                tier=tier, name=entry["name"],
                speed_sec=median_speed, peak_mb=median_peak,
                source="baseline-rebench",
                prior_speed_sec=prior,
                thread=thread,
            )
            new_baselines[(tier, thread)] = median_speed
            info(f"  → baseline at (tier={tier}, thread={thread}) = "
                 f"{median_speed:.3f}s (peak={median_peak:.1f}MB)")

    # Update verify.tsv baselines + speedup_pct in place.
    verify_tsv_path = Path(args.verify_tsv) if args.verify_tsv else task_dir / "verify.tsv"
    if not verify_tsv_path.is_absolute():
        verify_tsv_path = task_dir / verify_tsv_path
    if verify_tsv_path.exists():
        _migrate_verify_tsv_for_k2(verify_tsv_path)
        updates = _update_verify_tsv_baselines(verify_tsv_path, new_baselines)
        if updates:
            info(f"[rebench] updated {sum(n for *_, n in updates)} verify.tsv "
                 f"row(s) across {len(updates)} (tier, thread) cell(s).")
            for tier, thread, n in updates:
                info(f"  - (tier={tier}, thread={thread}): {n} row(s) repointed")
        else:
            info(f"[rebench] verify.tsv exists but no rows matched the "
                 f"re-benched (tier, thread) cells. Was this a different commit "
                 f"or phase?")
    else:
        info(f"[rebench] verify.tsv not found at {verify_tsv_path} — "
             f"baselines recorded in results.tsv only.")

    # Re-render verify figure if matplotlib is available + verify.tsv exists.
    if verify_tsv_path.exists() and not getattr(args, "no_plot", False):
        try:
            from zyme.commands.verify_render import _render_verify_matrix
            _render_verify_matrix(task_dir, verify_tsv_path)
            info(f"[rebench] re-rendered verify figure(s).")
        except Exception as e:
            info(f"[rebench] could not re-render verify figure: {e}")

    info(f"[rebench] done.")
