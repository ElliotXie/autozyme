"""`zyme init-check` — end-of-init self-check across all tiers.

Reports a per-tier checklist of scaffold / baseline / noise / metrics state
so the agent (or human) can verify init landed coherently without grepping
results.tsv, task.yaml, and reference_outputs/ individually.

Default is read-only. `--parity` additionally runs pipeline + evaluate
against each tier's reference_outputs/<tier>/ to confirm metrics-perfect
parity — catches save/load asymmetry bugs before they leak into iterate.
"""

import sys
from pathlib import Path

from zyme.utils import task_dir_from_args, resolve_reference_output_dir
from zyme.parsers.task_yaml import (
    parse_datasets, parse_metrics, parse_intrinsic_noise,
    parse_baseline_threads, parse_algorithm_class,
)
from zyme.parsers.results_tsv import parse_log
from zyme.commands._shared import _read_results_rows


_OK = "\033[32m✓\033[0m"
_NO = "\033[31m✗\033[0m"
_NA = "·"


def _tick(passed: bool) -> str:
    return _OK if passed else _NO


def _baseline_present(rows: list[dict], dataset_name: str, thread: int) -> bool:
    for r in rows:
        if (r.get("status") == "baseline"
                and r.get("dataset") == dataset_name
                and str(r.get("thread") or "1") == str(thread)):
            try:
                if float(r.get("speed_sec") or 0) > 0:
                    return True
            except ValueError:
                continue
    return False


def _ref_output_nonempty(task_dir: Path, tier: str) -> bool:
    primary = resolve_reference_output_dir(task_dir, tier=tier)
    if primary.exists() and any(primary.iterdir()):
        return True
    legacy = task_dir / f"reference_output_{tier}"
    return legacy.exists() and any(legacy.iterdir())


def cmd_init_check(args):
    """Print a per-tier checklist of init readiness.

    Per tier columns: baseline (results.tsv row exists with speed_sec > 0),
    ref_out (reference_outputs/<tier>/ non-empty), noise (intrinsic_noise[tier]
    populated — only when algorithm_class=stochastic).

    Plus scaffold-level checks: reference.{py,R}, pipeline/run.{py,R},
    evaluate.{py,R}, metrics declared, baseline_threads declared.

    Exit status: 0 if all required checks pass, 1 if any required check
    fails (so the init prompt can gate handoff on `zyme init-check && ...`).
    """
    task_dir = task_dir_from_args(args)
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.exists():
        sys.stderr.write(f"[init-check] no task.yaml at {task_yaml}\n")
        sys.exit(2)

    datasets = parse_datasets(task_yaml)
    metrics = parse_metrics(task_yaml)
    intrinsic = parse_intrinsic_noise(task_yaml)
    threads = parse_baseline_threads(task_yaml)
    algo_class = parse_algorithm_class(task_yaml)

    results_tsv = task_dir / "results.tsv"
    rows = _read_results_rows(results_tsv) if results_tsv.exists() else []

    # ---- scaffold-level checks ----
    has_ref_py = (task_dir / "reference.py").exists()
    has_ref_r = (task_dir / "reference.R").exists()
    has_pipe_py = (task_dir / "pipeline" / "run.py").exists()
    has_pipe_r = (task_dir / "pipeline" / "run.R").exists()
    has_eval_py = (task_dir / "evaluate.py").exists()
    has_eval_r = (task_dir / "evaluate.R").exists()

    print(f"Task: {task_dir.name}")
    print(f"Scaffold:")
    print(f"  {_tick(has_ref_py or has_ref_r)} reference.{{py,R}}")
    print(f"  {_tick(has_pipe_py or has_pipe_r)} pipeline/run.{{py,R}}")
    print(f"  {_tick(has_eval_py or has_eval_r)} evaluate.{{py,R}}")
    print(f"  {_tick(bool(metrics))} metrics declared ({len(metrics)} metric{'s' if len(metrics)!=1 else ''})")
    print(f"  {_tick(bool(threads))} baseline_threads: {threads}")
    print(f"  algorithm_class: {algo_class}")
    print()

    if not datasets:
        print("(no datasets in task.yaml — add at least one tier before iterating)")
        sys.exit(1)

    needs_noise = (algo_class == "stochastic")
    header = f"  {'tier':<14} {'baseline':>10} {'ref_out':>10}"
    if needs_noise:
        header += f" {'noise':>8}"
    print("Per tier:")
    print(header)

    all_ok = True
    for ds in datasets:
        tier = ds["tier"]
        name = ds["name"]
        # Baseline ticks per thread point (single-threaded baseline is the
        # minimum; multi-thread is optional but tracked when declared).
        thread_ticks = []
        for t in (threads or [1]):
            ok = _baseline_present(rows, name, thread=t)
            thread_ticks.append(_tick(ok))
            if not ok:
                all_ok = False
        baseline_cell = "/".join(thread_ticks) + (f" (t={','.join(str(t) for t in (threads or [1]))})")

        ref_ok = _ref_output_nonempty(task_dir, tier)
        if not ref_ok:
            all_ok = False
        ref_cell = _tick(ref_ok)

        line = f"  {tier:<14} {baseline_cell:>10} {ref_cell:>10}"
        if needs_noise:
            noise_ok = bool(intrinsic.get(tier))
            if not noise_ok:
                all_ok = False
            line += f" {_tick(noise_ok):>8}"
        print(line)

    print()
    if not all_ok:
        print(f"{_NO} init-check found gaps. Next steps:")
        for ds in datasets:
            tier = ds["tier"]
            for t in (threads or [1]):
                if not _baseline_present(rows, ds["name"], thread=t):
                    print(f"  - record baseline: zyme baseline reference --tier {tier}"
                          + (f" --thread {t}" if t != 1 else ""))
                    break
            if not _ref_output_nonempty(task_dir, tier):
                print(f"  - generate reference output: zyme baseline reference --tier {tier}")
            if needs_noise and not intrinsic.get(tier):
                print(f"  - calibrate noise: zyme baseline noise --tier {tier}")
        sys.exit(1)

    if getattr(args, "parity", False):
        parity_ok = _run_parity_check(task_dir, datasets, metrics)
        if not parity_ok:
            sys.exit(1)

    print(f"{_OK} init-check passed.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# --parity: run pipeline + evaluate per tier, assert metrics-perfect
# ---------------------------------------------------------------------------

# Tolerances for "metrics-perfect": gte→1.0 within GTE_TOL, lte→0.0 within
# LTE_TOL. The pipeline+reference scripts use the same numerical code path,
# so any non-trivial drift here points at a save/load asymmetry — not at
# legitimate algorithmic noise.
_GTE_TOL = 1e-3
_LTE_TOL = 1e-3


def _run_parity_check(task_dir: Path, datasets: list, metrics: list) -> bool:
    """Run pipeline + evaluate against each tier's reference_outputs/.

    Returns True if every tier's metrics are within tolerance of identity
    (gte→1.0, lte→0.0). Prints a per-tier ✓/✗ table.
    """
    from zyme.runner import run_task

    if not metrics:
        print(f"{_NO} parity: task.yaml declares no metrics — nothing to check.")
        return False

    print("Parity check (pipeline vs reference per tier):")
    all_ok = True
    for ds in datasets:
        tier = ds["tier"]
        if not _ref_output_nonempty(task_dir, tier):
            print(f"  {_NO} {tier:<12} (no reference_outputs/{tier}/ — run "
                  f"`zyme baseline reference --tier {tier}` first)")
            all_ok = False
            continue
        log = run_task(task_dir, dataset_entry=ds)
        speed, peak, observed, status = parse_log(log)
        if status == "crash":
            print(f"  {_NO} {tier:<12} pipeline crashed — see log for traceback")
            print(log)
            all_ok = False
            continue
        tier_failures = []
        for m in metrics:
            name, comparator = m["name"], m["comparator"]
            val = observed.get(name)
            if val is None:
                tier_failures.append(f"{name}=<missing>")
                continue
            if comparator == "gte" and val < 1.0 - _GTE_TOL:
                tier_failures.append(f"{name}={val:.4f} (expected ~1.0)")
            elif comparator == "lte" and val > _LTE_TOL:
                tier_failures.append(f"{name}={val:.4f} (expected ~0.0)")
        if tier_failures:
            print(f"  {_NO} {tier:<12} " + "; ".join(tier_failures))
            all_ok = False
        else:
            print(f"  {_OK} {tier:<12} all {len(metrics)} metrics identity-perfect")
    if not all_ok:
        print(f"\n{_NO} parity failed. Candidate causes:")
        print("  - save asymmetry: format / key names / dropped output slot "
              "between reference.{R,py} and pipeline/run.{R,py}")
        print("  - BLAS-thread nondeterminism (value metrics drift ~1e-4, "
              "structural metrics green): pin OMP/OPENBLAS/MKL_NUM_THREADS "
              "at the top of both scripts before `import numpy`")
        print("  - RNG/seed drift if target is stochastic")
    return all_ok
