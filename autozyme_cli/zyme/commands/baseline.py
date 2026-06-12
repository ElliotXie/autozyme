"""Baseline-management commands: record-baseline, reference, baseline noise, promote-baseline."""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from zyme.utils import (
    LEGACY_THREAD,
    die, info, task_dir_from_args, zyme_state,
    resolve_reference_script,
    resolve_reference_output_dir,
)
from zyme.parsers.task_yaml import (
    parse_algorithm_class,
    parse_datasets, parse_metrics, parse_random_seeds,
    parse_baseline_threads, write_baseline_threads,
    parse_synthesis,
    write_intrinsic_noise,
)
from zyme.fingerprint import check_reference_fingerprint, FingerprintViolation
from zyme.parsers.results_tsv import (
    parse_log, get_baseline_speed, dataset_in_results, results_header_for_task,
    append_results_row,
    ensure_results_schema, ensure_k2_schema,
)
from zyme.runner import build_reference_cmd
from zyme.noise_calibration import record_tier_noise


def _fingerprint_or_die(task_dir: Path, accept_synthesis: bool) -> None:
    """Run the reference.{py,R} fingerprint check; die on violation.

    Bypassed when `task.yaml::synthesis: <reason>` is declared AND
    `--accept-synthesis` was passed. The synthesis declaration alone is
    not enough — the explicit flag forces the agent to acknowledge that
    they are intentionally measuring against synthesized inputs.
    """
    synthesis_reason = parse_synthesis(task_dir / "task.yaml")
    if synthesis_reason and accept_synthesis:
        info(f"[fingerprint] task.yaml::synthesis declared "
             f"({synthesis_reason!r}); check bypassed via --accept-synthesis")
        return
    ref_path = resolve_reference_script(task_dir)
    if not ref_path.exists():
        return  # nothing to scan yet
    try:
        check_reference_fingerprint(ref_path)
    except FingerprintViolation as e:
        die(str(e))


_BASELINE_HISTORY_FIELDS = (
    "timestamp", "tier", "name", "speed_sec", "peak_mb",
    "source", "prior_speed_sec", "prior_peak_mb", "thread",
)


def _baseline_history_path(task_dir):
    return task_dir / ".zyme" / "baselines_history.tsv"


def _append_baseline_history(task_dir, tier, name, speed_sec, peak_mb, source,
                              prior_speed_sec=None, prior_peak_mb=None,
                              thread=None):
    """Append a record to the baseline audit log.

    Append-only — every record-baseline call (manual, --from-log, or via
    `zyme reference`) lands a row. Prior values overwritten on the same
    (tier, thread) are preserved here so you can audit how a baseline
    changed over time.
    """
    p = _baseline_history_path(task_dir)
    p.parent.mkdir(exist_ok=True)
    is_new = not p.exists()
    thread_val = LEGACY_THREAD if thread is None else int(thread)
    row = "\t".join([
        datetime.now().isoformat(timespec="seconds"),
        str(tier), str(name),
        f"{speed_sec:.3f}", f"{peak_mb:.1f}",
        source,
        f"{prior_speed_sec:.3f}" if prior_speed_sec is not None else "",
        f"{prior_peak_mb:.1f}" if prior_peak_mb is not None else "",
        str(thread_val),
    ]) + "\n"
    with open(p, "a") as f:
        if is_new:
            f.write("\t".join(_BASELINE_HISTORY_FIELDS) + "\n")
        f.write(row)


def _existing_baseline(task_dir, dataset_name, thread=None):
    """Look up existing baseline (speed_sec, peak_mb) for (dataset, thread).

    K2: reads only from results.tsv (no stash). Returns (None, None) when
    no row matches. `thread=None` → LEGACY_THREAD (=1).
    """
    target_thread = LEGACY_THREAD if thread is None else int(thread)
    speed = get_baseline_speed(task_dir, dataset_name, thread=target_thread)
    if not speed:
        return None, None
    # Walk results.tsv once more to grab peak_mb at the same row.
    from zyme.parsers.results_tsv import get_baseline_peak_mb
    peak = get_baseline_peak_mb(task_dir, dataset_name, thread=target_thread)
    return speed, peak


def _update_baseline_row_to_oom(results_tsv: Path, dataset_name: str,
                                  thread: int | None = None) -> bool:
    """Rewrite the baseline row of (dataset, thread) in results.tsv to status=oom.

    Used by `record-baseline --oom --force` when results.tsv already has a
    successful baseline for the (dataset, thread) pair. Zeroes speed_sec /
    peak_mb, sets status=oom, appends '(OOM override)' to the description.
    Returns True if a row was updated, False if not found. `thread=None` →
    match any thread (legacy semantics for files without the column).
    """
    if not results_tsv.exists():
        return False
    lines = results_tsv.read_text().splitlines()
    if len(lines) < 2:
        return False
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    if "dataset" not in col or "status" not in col or "speed_sec" not in col:
        return False
    thread_idx = col.get("thread")
    out = [lines[0]]
    updated = False
    target_thread = None if thread is None else int(thread)
    for line in lines[1:]:
        if not line.strip():
            out.append(line)
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            out.append(line)
            continue
        # Match (dataset, thread). Files without the thread column are
        # treated as thread=LEGACY_THREAD on every row.
        row_thread = LEGACY_THREAD
        if thread_idx is not None and thread_idx < len(parts):
            try:
                row_thread = int((parts[thread_idx].strip() or str(LEGACY_THREAD)))
            except ValueError:
                row_thread = LEGACY_THREAD
        thread_match = target_thread is None or row_thread == target_thread
        if (not updated
                and parts[col["dataset"]] == dataset_name
                and parts[col["status"]] in ("baseline", "oom")
                and thread_match):
            parts[col["status"]] = "oom"
            parts[col["speed_sec"]] = "0"
            if "peak_mb" in col and len(parts) > col["peak_mb"]:
                parts[col["peak_mb"]] = "0"
            if "description" in col and len(parts) > col["description"]:
                desc = parts[col["description"]]
                if "OOM" not in desc.upper():
                    parts[col["description"]] = f"{desc} (OOM override)".strip()
            out.append("\t".join(parts))
            updated = True
        else:
            out.append(line)
    if updated:
        results_tsv.write_text("\n".join(out) + "\n")
    return updated




def _baseline_show_rows(task_dir, show_tier):
    """Return [{tier, name, thread, speed_sec, peak_mb, status}] for a tier."""
    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        return []
    lines = results_tsv.read_text().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    from zyme.parsers.task_yaml import parse_datasets as _pd
    tasks = _pd(task_dir / "task.yaml")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= col.get("status", 0):
            continue
        if parts[col["status"]] not in ("baseline", "oom"):
            continue
        ds = parts[col.get("dataset", 2)]
        tier_for = next((e for e in tasks if e["name"] == ds), None)
        tier_name = tier_for["tier"] if tier_for else ds
        if tier_name != show_tier and ds != show_tier:
            continue
        t_val = parts[col["thread"]] if "thread" in col and col["thread"] < len(parts) else "1"
        rows.append({
            "tier": tier_name, "name": ds, "thread": t_val,
            "speed_sec": parts[col["speed_sec"]], "peak_mb": parts[col["peak_mb"]],
            "status": parts[col["status"]],
        })
    return rows


def cmd_baseline_list(args):
    """Show baseline rows.

    Default: current state from results.tsv (the truth `zyme verify` compares
    against — what the init agent or anyone asking 'did my baselines land?'
    actually wants). With --history: append-only audit log from
    .zyme/baselines_history.tsv (every record/rebench, useful for tracing how
    a value changed over time).
    """
    task_dir = task_dir_from_args(args)
    if getattr(args, "history", False):
        hp = _baseline_history_path(task_dir)
        if hp.exists():
            print(f"=== audit history at {hp} ===")
            sys.stdout.write(hp.read_text())
        else:
            print("(No audit history yet — first record creates it.)")
        return

    # Default: current baseline rows from results.tsv.
    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        print("(No results.tsv yet — record a baseline with "
              "`zyme baseline reference --tier <TIER>`.)")
        return

    lines = results_tsv.read_text().splitlines()
    if len(lines) < 2:
        print("(results.tsv has no rows yet.)")
        return
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= col.get("status", 0):
            continue
        if parts[col["status"]] not in ("baseline", "oom"):
            continue
        rows.append(parts)

    if not rows:
        print("(results.tsv has no baseline rows yet — record one with "
              "`zyme baseline reference --tier <TIER>`. "
              "Use `--history` to see the audit log instead.)")
        return

    print(f"=== current baselines in {results_tsv} ===")
    print(f"{'tier':<14} {'dataset':<22} {'thread':>6} {'speed_sec':>10} {'peak_mb':>10} {'status':<10}")
    # Map dataset name → tier label so we display tier alongside dataset name.
    from zyme.parsers.task_yaml import parse_datasets as _pd
    tier_for = {e["name"]: e["tier"] for e in _pd(task_dir / "task.yaml")}
    for parts in rows:
        ds = parts[col.get("dataset", 2)]
        tier = tier_for.get(ds, ds)
        thread = parts[col["thread"]] if "thread" in col and col["thread"] < len(parts) else "1"
        speed = parts[col["speed_sec"]] if "speed_sec" in col else "?"
        peak = parts[col["peak_mb"]] if "peak_mb" in col else "?"
        status = parts[col["status"]]
        print(f"{tier:<14} {ds:<22} {thread:>6} {speed:>10} {peak:>10} {status:<10}")
    print()
    print("(Use `zyme baseline list --history` to see the audit log of every record.)")


def cmd_baseline_show(args):
    """Print recorded baseline rows for one tier across all thread points."""
    task_dir = task_dir_from_args(args)
    show_tier = args.tier
    if not (task_dir / "results.tsv").exists():
        info(f"no results.tsv yet — nothing to show for '{show_tier}'.")
        return
    rows = _baseline_show_rows(task_dir, show_tier)
    if not rows:
        info(f"no baseline row in results.tsv for '{show_tier}'.")
        return
    for r in rows:
        print(f"tier={r['tier']} name={r['name']} thread={r['thread']} "
              f"speed_sec={r['speed_sec']} peak_mb={r['peak_mb']} "
              f"status={r['status']}")


def cmd_record_baseline(args):
    """Record an upstream-reference timing for one (tier, thread).

    Writes directly to results.tsv as a round=0/status=baseline row tagged
    with that (tier, thread). Idempotent: re-recording overwrites the prior
    row at the same (dataset, thread); every call also lands an immutable
    audit row in .zyme/baselines_history.tsv.

    Modes (mutually exclusive):
      - `--from-log PATH`: parse `speed_sec:` / `peak_mb:` from a captured
        stdout log file; fills in --speed-sec / --peak-mb automatically.
      - `--oom`: record this (tier, thread) as unmeasurable.
      - default: explicit `--tier` + `--speed-sec` (and optional --peak-mb / --thread).

    Read-only sibling commands: `baseline list` (audit history),
    `baseline show TIER` (rows for one tier).
    """
    task_dir = task_dir_from_args(args)
    _fingerprint_or_die(task_dir, getattr(args, "accept_synthesis", False))
    target_thread = int(
        getattr(args, "thread", None)
        or parse_baseline_threads(task_dir / "task.yaml")[0]
    )

    # Mode: --oom — record this (tier, thread) as unmeasurable.
    if getattr(args, "oom", False):
        if not args.tier:
            die("--oom: --tier is required to identify which tier is OOM.")
        available = parse_datasets(task_dir / "task.yaml")
        by_tier = {e["tier"]: e for e in available}
        by_name = {e["name"]: e for e in available}
        entry_yaml = by_tier.get(args.tier) or by_name.get(args.tier)
        if entry_yaml is None:
            die(
                f"--oom: tier '{args.tier}' not in task.yaml. "
                f"Available: {sorted(by_tier.keys())}"
            )
        zyme_state(task_dir)
        ensure_k2_schema(task_dir)

        results_tsv = task_dir / "results.tsv"
        prior_speed, prior_peak = _existing_baseline(
            task_dir, entry_yaml["name"], thread=target_thread)
        already_in_results = dataset_in_results(
            results_tsv, entry_yaml["name"], thread=target_thread)

        if already_in_results and prior_speed and prior_speed > 0 and not getattr(args, "force", False):
            die(
                f"--oom: results.tsv already has a successful baseline for "
                f"dataset '{entry_yaml['name']}' at thread={target_thread} "
                f"(speed_sec={prior_speed:.3f}s, "
                f"peak_mb={prior_peak or 0:.1f}). Refusing to mark OOM without --force.\n"
                f"  Common cause: re-recording on a different host or a tier typo.\n"
                f"  Audit log: {_baseline_history_path(task_dir)}"
            )

        if already_in_results:
            _update_baseline_row_to_oom(
                results_tsv, entry_yaml["name"], thread=target_thread)
        else:
            _write_baseline_row_to_results(
                task_dir, entry_yaml, thread=target_thread,
                speed_sec=0.0, peak_mb=0.0,
                metrics_json="{}", status="oom",
                description="(OOM)",
            )

        _append_baseline_history(
            task_dir,
            tier=entry_yaml["tier"],
            name=entry_yaml["name"],
            speed_sec=0.0,
            peak_mb=0.0,
            source=getattr(args, "_record_source", "manual --oom"),
            prior_speed_sec=prior_speed,
            prior_peak_mb=prior_peak,
            thread=target_thread,
        )
        info(
            f"baseline OOM recorded: tier={entry_yaml['tier']} "
            f"name={entry_yaml['name']} thread={target_thread}. `zyme verify` "
            f"will treat this (tier, thread) cell as OOM."
        )
        return

    # Mode: --from-log PATH (parse speed_sec / peak_mb from captured log)
    from_log = getattr(args, "from_log", None)
    if from_log:
        log_path = Path(from_log)
        if not log_path.exists():
            die(f"--from-log: file not found: {log_path}")
        log_content = log_path.read_text(encoding="utf-8", errors="replace")
        parsed_speed, parsed_peak, _, _ = parse_log(log_content)
        if parsed_speed is None:
            die(f"--from-log: no `speed_sec:` line in {log_path}. "
                f"Is the reference script calling emit_summary()?")
        args.speed_sec = parsed_speed
        if parsed_peak is not None:
            args.peak_mb = parsed_peak
        info(f"[record-baseline] from-log: parsed speed_sec={parsed_speed:.3f}, "
             f"peak_mb={(parsed_peak or 0.0):.1f}")

    # From here on, --tier and --speed-sec are required.
    if not args.tier:
        die("--tier is required (or use --list / --show <TIER>).")
    if args.speed_sec is None:
        die("--speed-sec is required (or pass --from-log <PATH> to extract it from a captured log).")

    # Nudge: manual `record --speed-sec` is the deprecated-in-practice path.
    # If --from-log wasn't used (i.e. the user is hand-typing values), point
    # them at `baseline reference` for next time. Don't block — `record` is
    # still the right tool when reference.{py,R} ran on a different host.
    # Print once per task (touch a sentinel under .zyme/) so an init agent
    # recording three tiers back-to-back doesn't see the same tip 3×.
    if not from_log:
        tip_seen = task_dir / ".zyme" / "record_tip_seen"
        if not tip_seen.exists():
            info(
                "[record-baseline] Tip: `zyme baseline reference --tier <TIER>` "
                "runs reference.{py,R} and records the baseline in one shot, "
                "with no manual --speed-sec / --peak-mb entry. Use `record` "
                "directly only when reference ran outside this CLI. "
                "(Tip suppressed on subsequent record calls for this task.)"
            )
            tip_seen.parent.mkdir(parents=True, exist_ok=True)
            tip_seen.touch()

    available = parse_datasets(task_dir / "task.yaml")
    by_tier = {e["tier"]: e for e in available}
    by_name = {e["name"]: e for e in available}

    entry_yaml = by_tier.get(args.tier) or by_name.get(args.tier)
    if entry_yaml is None:
        die(
            f"tier '{args.tier}' not found in task.yaml. "
            f"Available tiers: {sorted(by_tier.keys())}; names: {sorted(by_name.keys())}"
        )
    if args.name and args.name != entry_yaml["name"]:
        die(
            f"--name '{args.name}' does not match task.yaml dataset for tier "
            f"'{entry_yaml['tier']}' (expected '{entry_yaml['name']}')"
        )

    try:
        parsed_metrics = json.loads(args.metrics)
    except json.JSONDecodeError as e:
        die(f"--metrics is not valid JSON: {e}")

    if not parsed_metrics:
        task_metrics = parse_metrics(task_dir / "task.yaml")
        if task_metrics:
            parsed_metrics = {
                m["name"]: 1.0 if m["comparator"] == "gte" else 0.0
                for m in task_metrics
            }
            args_metrics = json.dumps(parsed_metrics)
            info(
                f"[record-baseline] auto-filled identity metrics from task.yaml "
                f"(gte->1.0, lte->0.0): {parsed_metrics}"
            )
        else:
            args_metrics = "{}"
            info(
                "[record-baseline] task.yaml has no metrics: block; "
                "recording empty metrics."
            )
    else:
        args_metrics = args.metrics

    # Sanity gate: if a prior baseline exists for THIS (dataset, thread)
    # and the new speed_sec differs by > 2× in either direction, refuse
    # unless --force. Different speeds at different thread counts are
    # expected — the gate keys on (dataset, thread) only.
    prior_speed, prior_peak = _existing_baseline(
        task_dir, entry_yaml["name"], thread=target_thread)
    if (prior_speed is not None and prior_speed > 0 and args.speed_sec > 0
            and not getattr(args, "force", False)):
        ratio = max(args.speed_sec / prior_speed, prior_speed / args.speed_sec)
        if ratio > 2.0:
            die(
                f"new baseline speed_sec={args.speed_sec:.3f}s differs from "
                f"prior {prior_speed:.3f}s by {ratio:.2f}× for "
                f"(dataset={entry_yaml['name']}, thread={target_thread}) (>2× sanity gate).\n"
                f"  Common cause: wrong tier / wrong dataset measured.\n"
                f"  If the change is real (host upgrade etc.), pass `--force` to override.\n"
                f"  Audit log: {_baseline_history_path(task_dir)}"
            )

    zyme_state(task_dir)
    ensure_k2_schema(task_dir)

    results_tsv = task_dir / "results.tsv"
    if dataset_in_results(results_tsv, entry_yaml["name"], thread=target_thread):
        # Existing (dataset, thread) baseline — overwrite in place.
        _update_baseline_row_in_results(
            task_dir, entry_yaml, thread=target_thread,
            speed_sec=args.speed_sec, peak_mb=args.peak_mb,
            metrics_json=args_metrics,
            description="upstream reference baseline (re-recorded)",
        )
        action = "updated"
    else:
        _write_baseline_row_to_results(
            task_dir, entry_yaml, thread=target_thread,
            speed_sec=args.speed_sec, peak_mb=args.peak_mb,
            metrics_json=args_metrics, status="baseline",
            description="upstream reference baseline",
        )
        action = "recorded"

    _append_baseline_history(
        task_dir,
        tier=entry_yaml["tier"],
        name=entry_yaml["name"],
        speed_sec=args.speed_sec,
        peak_mb=args.peak_mb,
        source=getattr(args, "_record_source", "manual"),
        prior_speed_sec=prior_speed,
        prior_peak_mb=prior_peak,
        thread=target_thread,
    )

    info(
        f"baseline {action}: tier={entry_yaml['tier']} name={entry_yaml['name']} "
        f"thread={target_thread} speed_sec={args.speed_sec:.3f} "
        f"peak_mb={args.peak_mb:.1f}."
    )


def _write_baseline_row_to_results(task_dir: Path, entry_yaml: dict,
                                    thread: int, speed_sec: float,
                                    peak_mb: float, metrics_json: str,
                                    status: str = "baseline",
                                    description: str = "upstream reference baseline"):
    """Write a fresh round=0/status=baseline row to results.tsv tagged
    (tier, thread). Creates the file with the right header if absent.

    K2 direct-write replacement for V1's stash → lazy-promote pipeline.
    """
    results_tsv = task_dir / "results.tsv"
    if results_tsv.exists():
        ensure_results_schema(task_dir, results_tsv)
        ensure_k2_schema(task_dir)

    desc_suffix = " (OOM)" if status == "oom" else ""
    append_results_row(results_tsv, task_dir, {
        "round": "0",
        "commit": "upstream",
        "dataset": entry_yaml["name"],
        "speed_sec": f"{speed_sec:.3f}",
        "speedup_pct": "0.0",
        "peak_mb": f"{peak_mb:.1f}",
        "status": status,
        "metrics_json": metrics_json,
        "hypothesis": "",
        "description": f"{description}{desc_suffix}",
        "phase": "optimize",
        "thread": str(int(thread)),
    })


def _update_baseline_row_in_results(task_dir: Path, entry_yaml: dict,
                                      thread: int, speed_sec: float,
                                      peak_mb: float, metrics_json: str,
                                      description: str):
    """Update the existing (dataset, thread) baseline row's speed/peak/metrics
    in place. Assumes one row per (dataset, thread) — call only when
    `dataset_in_results(..., thread=...)` returned True.
    """
    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        return
    lines = results_tsv.read_text().splitlines()
    if len(lines) < 2:
        return
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    if "thread" not in col or "dataset" not in col:
        return
    out = [lines[0]]
    updated = False
    for line in lines[1:]:
        if not line.strip():
            out.append(line)
            continue
        parts = line.split("\t")
        while len(parts) < len(header):
            parts.append("")
        if (not updated
                and parts[col["dataset"]] == entry_yaml["name"]
                and parts[col["status"]] in ("baseline", "oom")):
            try:
                row_thread = int(parts[col["thread"]] or LEGACY_THREAD)
            except ValueError:
                row_thread = LEGACY_THREAD
            if row_thread == int(thread):
                parts[col["speed_sec"]] = f"{speed_sec:.3f}"
                parts[col["peak_mb"]] = f"{peak_mb:.1f}"
                parts[col["status"]] = "baseline"
                if "metrics_json" in col:
                    parts[col["metrics_json"]] = metrics_json
                if "description" in col:
                    parts[col["description"]] = description
                out.append("\t".join(parts))
                updated = True
                continue
        out.append(line)
    if updated:
        results_tsv.write_text("\n".join(out) + "\n")




# Patterns indicating the reference process died from OOM rather than a
# normal script error. Used to print a more actionable failure message and
# point the agent at `zyme baseline record --oom`.
_OOM_PATTERNS = [
    re.compile(r"vector memory (?:limit|exhausted)", re.IGNORECASE),
    re.compile(r"cannot allocate vector of size", re.IGNORECASE),
    re.compile(r"\bstd::bad_alloc\b"),
    re.compile(r"\bMemoryError\b"),
    re.compile(r"\bout of memory\b", re.IGNORECASE),
    re.compile(r"\bnumpy.*could not allocate\b", re.IGNORECASE),
    re.compile(r"Killed\b"),  # SIGKILL from the OOM killer, typically Linux
]


def _looks_like_oom(log: str) -> bool:
    """Heuristic: scan the tail of captured output for OOM markers."""
    tail = log[-4096:] if len(log) > 4096 else log
    return any(p.search(tail) for p in _OOM_PATTERNS)


def cmd_reference(args):
    """Run reference.{py,R} for one tier at a given thread count and auto-record
    the baseline.

    K2: replaces V1's mode-aware lookup with a single reference script per task
    that reads `ZYME_THREADS` and branches as needed (e.g., serial when N=1,
    BPPARAM/MulticoreParam when N>1). Records the timing as a (tier, thread)
    baseline row directly into results.tsv.

    With:
        zyme reference --tier <T> --thread <N>

    The CLI sets `ZYME_THREADS=<N>` (alongside `ZYME_DATA_PATH`,
    `ZYME_REFERENCE_DIR`, `ZYME_TIER`), captures stdout, parses
    `speed_sec:` / `peak_mb:`, and hands off to `cmd_record_baseline` (which
    applies the >2× sanity gate per (dataset, thread) and writes the audit
    history).
    """
    task_dir = task_dir_from_args(args)
    _fingerprint_or_die(task_dir, getattr(args, "accept_synthesis", False))
    task_yaml = task_dir / "task.yaml"

    available = parse_datasets(task_yaml)
    by_tier = {e["tier"]: e for e in available}
    by_name = {e["name"]: e for e in available}
    entry = by_tier.get(args.tier) or by_name.get(args.tier)
    if entry is None:
        die(f"tier '{args.tier}' not in task.yaml. "
            f"Available tiers: {sorted(by_tier.keys())}; names: {sorted(by_name.keys())}")

    target_thread = int(
        getattr(args, "thread", None)
        or parse_baseline_threads(task_dir / "task.yaml")[0]
    )

    ref_path = resolve_reference_script(task_dir)
    if not ref_path.exists():
        die(
            f"reference script not found: {ref_path}. "
            f"`zyme init` should have created reference.{{R,py}} from the template; "
            f"if it didn't (language sniff failed), rename the .template file by hand."
        )
    try:
        cmd = build_reference_cmd(task_dir / "task.yaml", ref_path)
    except RuntimeError as e:
        die(str(e))

    ref_out_dir = resolve_reference_output_dir(task_dir, tier=entry["tier"])
    ref_out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ZYME_DATA_PATH"] = str(entry["path"])
    env["ZYME_REFERENCE_DIR"] = str(ref_out_dir)
    env["ZYME_TIER"] = entry["tier"]
    env["ZYME_THREADS"] = str(target_thread)
    # Set the cross-library thread caps so reference scripts that don't
    # explicitly read ZYME_THREADS still get the right parallelism via
    # OMP/MKL/BLAS auto-detection.
    env["OMP_NUM_THREADS"] = str(target_thread)
    env["OPENBLAS_NUM_THREADS"] = str(target_thread)
    env["MKL_NUM_THREADS"] = str(target_thread)

    reps = max(1, int(getattr(args, "reps", None) or 1))
    rep_msg = f" × {reps} reps" if reps > 1 else ""
    info(f"running {ref_path.name} for tier='{entry['tier']}' "
         f"(dataset='{entry['name']}', thread={target_thread}){rep_msg}")
    info(f"  ZYME_DATA_PATH={entry['path']}")
    info(f"  ZYME_REFERENCE_DIR={ref_out_dir}")
    info(f"  ZYME_TIER={entry['tier']}")
    info(f"  ZYME_THREADS={target_thread}")
    if reps > 1:
        info(f"  noise calibration: {reps} reps -> mean used as baseline, "
             f"CV stored in .zyme/baseline_noise.json")

    # Stream stdout to terminal AND capture for parse_log. The agent watching
    # the terminal sees progress as the reference runs; the captured copy is
    # what we extract speed_sec / peak_mb from.
    speeds: list[float] = []
    peaks: list[float] = []
    for rep_idx in range(reps):
        if reps > 1:
            print(f"\n=== reference rep {rep_idx + 1}/{reps} ===", flush=True)
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
            log_text = "".join(captured)
            if _looks_like_oom(log_text):
                thread_arg = (
                    f" --thread {target_thread}" if target_thread != 1 else ""
                )
                die(
                    f"reference exit code {proc.returncode}; OOM detected "
                    f"(host RAM insufficient for tier={entry['tier']}, "
                    f"thread={target_thread}). baseline NOT recorded.\n"
                    f"  - Pick a smaller input for this tier and re-run, OR\n"
                    f"  - Mark this (tier, thread) unmeasurable: "
                    f"`zyme baseline record --tier {entry['tier']} --oom"
                    f"{thread_arg}`"
                )
            die(f"reference exit code {proc.returncode}; baseline NOT recorded.")

        log = "".join(captured)
        speed, peak, _, _ = parse_log(log)
        if speed is None:
            die("reference completed but no `speed_sec:` line in stdout. "
                "Is emit_summary() called at the end of reference.{py,R}?")
        if peak is None:
            info("[reference] no peak_mb in stdout; treating as 0.0")
            peak = 0.0
        speeds.append(float(speed))
        peaks.append(float(peak))

    # Aggregate across reps. With reps=1 the mean is just the single measurement,
    # so we always use mean (keeps the baseline path single-codepath).
    import statistics as _stats
    speed_mean = _stats.mean(speeds)
    peak_mean = _stats.mean(peaks)
    if reps > 1:
        sp_std = _stats.stdev(speeds)
        sp_cv = (sp_std / speed_mean * 100) if speed_mean > 0 else 0.0
        pk_std = _stats.stdev(peaks) if len(peaks) > 1 else 0.0
        print(
            f"\n[reference] noise calibration ({reps} reps): "
            f"speed = {speed_mean:.3f} ± {sp_std:.3f}s ({sp_cv:.2f}% CV); "
            f"peak = {peak_mean:.1f} ± {pk_std:.1f}MB",
            flush=True,
        )

    # Hand off to cmd_record_baseline. Populate the args fields it expects;
    # tag _record_source so the audit log shows this came from `zyme reference`.
    args.tier = entry["tier"]
    args.name = entry.get("name")
    args.speed_sec = speed_mean
    args.peak_mb = peak_mean
    args.metrics = "{}"
    args.from_log = None
    args.oom = False
    args.thread = target_thread
    args._record_source = (
        f"zyme reference (mean of {reps} reps)" if reps > 1 else "zyme reference"
    )
    cmd_record_baseline(args)

    # Persist per-(tier, thread) noise calibration when reps > 1. The agent
    # reads `.zyme/baseline_noise.json` directly to judge whether a fresh
    # `zyme run` delta is decisive or noise.
    if reps > 1:
        try:
            commit = ""
            try:
                from zyme.utils import git
                commit = git("rev-parse", "HEAD", cwd=task_dir).strip() or ""
            except Exception:
                pass
            entry_recorded = record_tier_noise(
                task_dir=task_dir,
                tier=entry["tier"],
                thread=target_thread,
                dataset_name=entry["name"],
                speeds=speeds,
                peaks=peaks,
                commit=commit,
            )
            info(
                f"[noise] calibration saved -> .zyme/baseline_noise.json "
                f"({entry['tier']}/thread={target_thread}: "
                f"speed_cv={entry_recorded['speed_cv']*100:.2f}%, "
                f"n_reps={entry_recorded['n_reps']})"
            )
        except Exception as e:
            info(f"[noise] WARN: failed to persist calibration: {e}")




def cmd_record_noise(args):
    """Calibrate intrinsic noise for one tier (stochastic algorithms).

    Runs `reference.{py,R}` with one or more calibration seeds from
    `--seeds` or `random_seeds.noise_calibration`, computes per-metric diffs
    vs the existing primary reference output, aggregates the worst observed
    seed drift, and writes it to `task.yaml::intrinsic_noise[tier]`.

    Concordance gates downstream (`zyme verify`, `zyme run`'s post-evaluate
    threshold check) read `intrinsic_noise[tier][metric]` to compute
    effective per-tier thresholds:
        lte: max(absolute_floor, noise_multiplier × intrinsic_noise)
        gte: max(absolute_floor, 1 - noise_multiplier × (1 - intrinsic_noise))

    Without this calibration, OOD-tier and dev-tier gates fall back to
    `absolute_floor` and may false-positive — chains of stochastic
    algorithms (MCMC, EM, random-init) drift naturally between seeds, and
    that drift compounds with chain length. See `## DISCOVERY:
    stochastic-algorithm concordance` entries from sccoda for the empirical
    case.

    Requires `algorithm_class: stochastic` in task.yaml. The reference
    script must read its random seed from `ZYME_RANDOM_SEED` env var
    (fallback to a hardcoded default — typically 42 — when the var is
    unset, so deterministic behavior is preserved at init time).
    Likewise `evaluate.{py,R}` must accept `ZYME_TEST_DIR` env var to
    point at the calibration output instead of the default `pipeline/`
    directory.
    """
    task_dir = task_dir_from_args(args)
    task_yaml = task_dir / "task.yaml"

    algo_class = parse_algorithm_class(task_yaml)
    if algo_class != "stochastic":
        die(f"`zyme baseline noise` requires `algorithm_class: stochastic` in "
            f"task.yaml. Current value: {algo_class!r}. For deterministic "
            f"algorithms, the existing fixed-threshold metrics schema is "
            f"correct — no noise calibration needed.")

    seeds = parse_random_seeds(task_yaml)
    primary_seed = int(seeds.get("primary", 42))
    if getattr(args, "seeds", None):
        try:
            cal_seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
        except ValueError:
            die(f"--seeds must be a comma-separated integer list, got: {args.seeds!r}")
    else:
        raw_cal = seeds.get("noise_calibration", [43, 44, 45])
        if isinstance(raw_cal, list):
            cal_seeds = [int(s) for s in raw_cal]
        else:
            cal_seeds = [int(raw_cal)]
    cal_seeds = list(dict.fromkeys(cal_seeds))
    if not cal_seeds:
        die("no calibration seeds supplied. Use --seeds 43,44,45 or "
            "`random_seeds: {primary: 42, noise_calibration: [43, 44, 45]}`.")
    # If the primary seed is in cal_seeds, we'll reuse the existing primary
    # reference output for that seed instead of re-running (saves one full
    # reference run — ~4 min at tier=large). Drift for that seed is 0 by
    # construction (identical comparison). Aggregation uses worst-per-metric
    # (min for gte, max for lte), so a trivial 0 doesn't underestimate noise
    # — it just sits at the optimistic end and gets ignored by max/min.

    available = parse_datasets(task_yaml)
    by_tier = {e["tier"]: e for e in available}
    by_name = {e["name"]: e for e in available}
    entry = by_tier.get(args.tier) or by_name.get(args.tier)
    if entry is None:
        die(f"tier '{args.tier}' not in task.yaml. "
            f"Available tiers: {sorted(by_tier.keys())}; "
            f"names: {sorted(by_name.keys())}")
    tier = entry["tier"]

    # K2: single reference script + per-tier output dir. Layered fallback
    # for legacy task layouts (`reference_output_<tier>/` flat,
    # `reference_outputs/<tier>/` nested are both still readable).
    primary_ref_dir = resolve_reference_output_dir(task_dir, tier=tier)
    primary_legacy = task_dir / f"reference_output_{tier}"
    if primary_ref_dir.exists() and any(primary_ref_dir.iterdir()):
        pass
    elif primary_legacy.exists() and any(primary_legacy.iterdir()):
        primary_ref_dir = primary_legacy
    else:
        die(f"primary reference output missing for tier='{tier}'. "
            f"Tried: {primary_ref_dir}, {primary_legacy}. "
            f"Run `zyme reference --tier {tier}` first to generate the "
            f"primary baseline before calibrating noise against it.")

    ref_path = resolve_reference_script(task_dir)
    if not ref_path.exists():
        die(f"reference script not found: {ref_path}.")
    try:
        cmd = build_reference_cmd(task_dir / "task.yaml", ref_path)
    except RuntimeError as e:
        die(str(e))

    target_thread = int(
        getattr(args, "thread", None)
        or parse_baseline_threads(task_dir / "task.yaml")[0]
    )
    info(f"calibrating intrinsic noise for tier='{tier}' (dataset='{entry['name']}', thread={target_thread})")
    info(f"  primary seed={primary_seed}, calibration seeds={cal_seeds}")
    info(f"  primary ref dir: {primary_ref_dir}")

    eval_candidates = [
        ("python", task_dir / "evaluate.py"),
        ("Rscript", task_dir / "evaluate.R"),
    ]
    eval_cmd = None
    eval_path = None
    for interp, p in eval_candidates:
        if p.exists():
            eval_cmd = [interp, str(p)]
            eval_path = p
            break
    if eval_cmd is None:
        die(f"no evaluate.py or evaluate.R in {task_dir}; "
            f"cannot compute noise diffs.")

    metrics_spec = parse_metrics(task_yaml)
    declared = {m["name"] for m in metrics_spec}
    comparator_by_name = {m["name"]: m["comparator"] for m in metrics_spec}
    per_seed_metrics = []

    for cal_seed in cal_seeds:
        if cal_seed == primary_seed:
            # Reuse cached primary reference output instead of re-running.
            cal_ref_dir = primary_ref_dir
            info(f"  seed={cal_seed} (= primary): reusing {cal_ref_dir} "
                 f"(no reference re-run; drift will be 0)")
        else:
            cal_ref_dir = task_dir / "reference_outputs" / f"{tier}_noise_seed{cal_seed}"
            if cal_ref_dir.exists():
                shutil.rmtree(cal_ref_dir)
            cal_ref_dir.mkdir(parents=True, exist_ok=True)
            info(f"  calibration ref dir for seed={cal_seed}: {cal_ref_dir}")

            # ---- Run reference.{py,R} with this calibration seed ----
            env = os.environ.copy()
            env["ZYME_DATA_PATH"] = str(entry["path"])
            env["ZYME_REFERENCE_DIR"] = str(cal_ref_dir)
            env["ZYME_TIER"] = tier
            env["ZYME_RANDOM_SEED"] = str(cal_seed)
            env["ZYME_THREADS"] = str(target_thread)
            env["OMP_NUM_THREADS"] = str(target_thread)
            env["OPENBLAS_NUM_THREADS"] = str(target_thread)
            env["MKL_NUM_THREADS"] = str(target_thread)

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
                die(f"calibration reference run failed for seed={cal_seed} "
                    f"(exit {proc.returncode}); intrinsic_noise NOT recorded "
                    f"for {tier}.")

        # ---- Run evaluate.py against primary ref (test_dir = calibration) ----
        eval_env = os.environ.copy()
        eval_env["ZYME_REFERENCE_DIR"] = str(primary_ref_dir)
        eval_env["ZYME_TEST_DIR"] = str(cal_ref_dir)
        eval_env["ZYME_TIER"] = tier

        info(f"running {eval_path.name} to compute noise diffs "
             f"(ref={primary_ref_dir.name}, test={cal_ref_dir.name}, seed={cal_seed})")
        eval_proc = subprocess.run(
            eval_cmd, env=eval_env, cwd=str(task_dir),
            capture_output=True, text=True,
        )
        if eval_proc.returncode != 0:
            sys.stderr.write(eval_proc.stdout or "")
            sys.stderr.write(eval_proc.stderr or "")
            die(f"evaluate exit code {eval_proc.returncode} for seed={cal_seed}; "
                f"intrinsic_noise NOT recorded for {tier}. Hint: ensure "
                f"`evaluate.{{py,R}}` reads `ZYME_TEST_DIR` env var (fallback "
                f"`<task>/pipeline/`) so it can compare two reference dirs.")

        # Parse `metric: value` lines from evaluate output.
        metrics = {}
        for line in eval_proc.stdout.splitlines():
            m = __import__("re").match(
                r"^\s*(\w+)\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$", line)
            if m:
                try:
                    metrics[m.group(1)] = float(m.group(2))
                except ValueError:
                    continue
        if not metrics:
            die(f"evaluate completed for seed={cal_seed} but produced no "
                f"parseable `metric: value` lines. Stdout was:\n{eval_proc.stdout}")

        # Filter to the metric names declared in task.yaml — drop any extras
        # evaluate prints but task.yaml doesn't track.
        filtered = {k: v for k, v in metrics.items() if k in declared}
        if not filtered:
            die(f"none of the {len(metrics)} evaluate-output metrics for "
                f"seed={cal_seed} ({sorted(metrics)}) match any declared "
                f"metric in task.yaml ({sorted(declared)}). Check metric-name "
                f"consistency.")
        per_seed_metrics.append((cal_seed, filtered))

    aggregate = {}
    for name in sorted(set().union(*(m.keys() for _, m in per_seed_metrics))):
        missing = [seed for seed, metrics in per_seed_metrics if name not in metrics]
        if missing:
            die(f"metric {name!r} missing from calibration seed(s) {missing}; "
                f"intrinsic_noise NOT recorded for {tier}.")
        vals = [metrics[name] for _, metrics in per_seed_metrics]
        if comparator_by_name.get(name) == "gte":
            aggregate[name] = min(vals)  # lower similarity is worse
        else:
            aggregate[name] = max(vals)  # higher drift is worse

    write_intrinsic_noise(task_yaml, tier, aggregate)
    info(f"intrinsic_noise recorded for tier={tier} from {len(cal_seeds)} seed(s):")
    for seed, metrics in per_seed_metrics:
        inline = ", ".join(f"{name}={val:.6f}" for name, val in sorted(metrics.items()))
        info(f"  seed={seed}: {inline}")
    info("  aggregate (worst seed per metric):")
    for name, val in sorted(aggregate.items()):
        info(f"    {name}: {val:.6f}")




def cmd_promote_baseline(args):
    """K2: drain leftover V1 stash entries into results.tsv.

    K2 dropped V1's lazy stash → results.tsv promotion entirely:
    `cmd_record_baseline` now writes directly to results.tsv. This command
    is preserved as an escape hatch for tasks that still have entries in
    the legacy `.zyme/baselines_stash.tsv` file from before K2.

    Behavior: reads any stash entries, writes each as a (tier, thread)
    baseline row in results.tsv (default thread=1 if not present), then
    deletes the stash file. Idempotent — safe to run with no stash.
    """
    task_dir = task_dir_from_args(args)
    ensure_k2_schema(task_dir)

    from zyme.utils import read_baseline_stash, write_baseline_stash
    available = parse_datasets(task_dir / "task.yaml")
    by_name = {e["name"]: e for e in available}

    stash = read_baseline_stash(task_dir)
    if not stash:
        info("no stash entries to drain.")
        return

    drained = 0
    for entry in stash:
        name = entry.get("name")
        if name not in by_name:
            info(f"  skip: stash entry for unknown dataset '{name}'")
            continue
        tier_entry = by_name[name]
        try:
            thread = int(entry.get("thread") or LEGACY_THREAD)
        except ValueError:
            thread = LEGACY_THREAD
        try:
            speed = float(entry.get("speed_sec") or 0)
            peak = float(entry.get("peak_mb") or 0)
        except ValueError:
            info(f"  skip: stash entry for {name} has unparseable speed/peak")
            continue
        status = entry.get("status") or "baseline"
        results_tsv = task_dir / "results.tsv"
        if dataset_in_results(results_tsv, name, thread=thread):
            info(f"  skip: results.tsv already has (tier={tier_entry['tier']}, "
                 f"thread={thread}) row")
            continue
        _write_baseline_row_to_results(
            task_dir, tier_entry, thread=thread,
            speed_sec=speed, peak_mb=peak,
            metrics_json=entry.get("metrics_json", "{}"),
            status=status,
            description="upstream reference baseline (drained from stash)",
        )
        _append_baseline_history(
            task_dir,
            tier=tier_entry["tier"], name=name,
            speed_sec=speed, peak_mb=peak,
            source="promote-baseline (drain)",
            thread=thread,
        )
        drained += 1

    # Remove the stash file once drained (write_baseline_stash with [] deletes).
    write_baseline_stash(task_dir, [])
    info(f"drained {drained} stash entry/entries into results.tsv "
         f"(stash file removed).")
