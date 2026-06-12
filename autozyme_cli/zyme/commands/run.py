"""Optimization-loop commands: run, dryrun, accept, reject, rollback."""

import collections
import json
import sys
import shutil
import statistics
import time
from datetime import datetime
from pathlib import Path

from zyme.utils import (
    LEGACY_THREAD,
    die, info, git, task_dir_from_args, zyme_state,
    pipeline_paths,
    check_upstream_version_drift,
)
from zyme.parsers.task_yaml import (
    resolve_tiers,
    parse_baseline_threads,
    parse_datasets,
    parse_target_function,
    parse_upstream_parallelism,
    parse_metrics,
    parse_intrinsic_noise,
    effective_threshold,
)
from zyme.argdiff import (
    diff_target_call_kwargs, format_divergences, check_target_prewarm,
    diff_side_channel_parallelism, format_side_channel_divergences,
)
from zyme.hoist_audit import (
    scan_pipeline_hoist, format_violations_message, append_hoist_log,
    read_hoist_exempt,
)
from zyme.parsers.results_tsv import (
    parse_log, update_last_status,
    count_decision_rounds, next_rerun_seq,
    get_baseline_speed,
    find_best_speed_at_dataset, best_cv_at_dataset,
    best_wall_cpu_ratio_at_dataset, migrate_results_add_phase,
    dataset_in_results, has_baseline_at_thread,
    results_header_for_task, append_results_row,
    ensure_results_schema,
)
from zyme.runner import run_task, dryrun_task
from zyme.commands._shared import _read_results_rows

DESCRIPTION_HINT = (
    "Tip: include speedup, mechanism, and (for discards) the failure mode.\n"
    "  ✓ \"vectorized wilcoxon → top20=0.81 < 0.95, broadcast drops zero-variance genes\"\n"
    "  ✓ \"RANN::nn2 k=20 → +12% speed, knn_overlap=0.998 ≥ 0.95\"\n"
    "  ✗ \"too slow\", \"didn't work\", \"sparse approach\""
)


_GENERATED_PIPELINE_FILE_SUFFIXES = (
    ".csv",
    ".tsv",
    ".parquet",
    ".h5ad",
    ".npz",
    ".rds",
)
_GENERATED_PIPELINE_FILENAMES = {
    "metrics.json",
    "profile.json",
    "profile.out",
    "Rprof.out",
    "scalene.json",
    "memray.bin",
    "profvis.html",
}
_TASK_DEFINITION_PATHS = (
    "task.yaml",
    "reference.py",
    "reference.R",
    "evaluate.py",
    "evaluate.R",
    "setup/",
)
_SETUP_EVALUATOR_FILES = {"evaluate.py", "evaluate.R"}


def _is_generated_pipeline_artifact(rel_path: str) -> bool:
    """True for normal pipeline outputs that should not affect edit audit.

    Pipeline code is intentionally staged from `pipeline/`, but dryruns/reruns
    often leave result files under `pipeline/output_<tier>/` or as direct files
    in `pipeline/`. Those are generated measurements, not source edits.
    """
    parts = Path(rel_path).parts
    if not parts or parts[0] != "pipeline":
        return False
    if (
        len(parts) >= 3
        and (parts[1] == "output"
             or parts[1].startswith("output_")
             or parts[1].startswith("output-"))
    ):
        return True
    if len(parts) != 2:
        return False
    name = parts[1]
    if name in _GENERATED_PIPELINE_FILENAMES:
        return True
    if name.startswith("native_sample_"):
        return True
    return name.endswith(_GENERATED_PIPELINE_FILE_SUFFIXES)


def _unstage_generated_pipeline_artifacts(task_dir: Path) -> list[str]:
    """Remove known generated pipeline outputs from the pending commit."""
    cached = git("diff", "--cached", "--name-only", cwd=task_dir, check=False)
    generated = [
        f.strip()
        for f in (cached or "").splitlines()
        if f.strip() and _is_generated_pipeline_artifact(f.strip())
    ]
    if generated:
        git("reset", "-q", "--", *generated, cwd=task_dir, check=False)
    return generated


def _row_thread_for_estimate(parts: list[str], col: dict[str, int]) -> int:
    if "thread" not in col or col["thread"] >= len(parts):
        return LEGACY_THREAD
    try:
        return int(parts[col["thread"]] or LEGACY_THREAD)
    except ValueError:
        return LEGACY_THREAD


def _current_head_speed_estimate(task_dir: Path, dataset_name: str,
                                 thread: int) -> tuple[float | None, str]:
    """Return a rerun estimate, preferring recent measurements of HEAD."""
    results_tsv = task_dir / "results.tsv"
    if results_tsv.exists():
        head = git("rev-parse", "--short=7", "HEAD", cwd=task_dir, check=False)
        lines = results_tsv.read_text().splitlines()
        if head and len(lines) >= 2:
            header = lines[0].split("\t")
            col = {n: i for i, n in enumerate(header)}
            required = {"commit", "dataset", "speed_sec", "status"}
            if required.issubset(col):
                speeds = []
                for line in lines[1:]:
                    parts = line.split("\t")
                    if len(parts) < len(header):
                        continue
                    if parts[col["commit"]] != head[:7]:
                        continue
                    if parts[col["dataset"]] != dataset_name:
                        continue
                    if parts[col["status"]] not in ("pending", "keep", "rerun"):
                        continue
                    if _row_thread_for_estimate(parts, col) != thread:
                        continue
                    try:
                        speed = float(parts[col["speed_sec"]])
                    except ValueError:
                        continue
                    if speed > 0:
                        speeds.append(speed)
                if speeds:
                    return statistics.median(speeds), f"current HEAD median, n={len(speeds)}"

    best = find_best_speed_at_dataset(task_dir, dataset_name, thread=thread)
    if best and best > 0:
        return best, "best median"
    baseline = get_baseline_speed(task_dir, dataset_name, thread=thread)
    if baseline and baseline > 0:
        # Phase-3 OOD-first-measurement: this dataset has only a baseline
        # row. Other dev-tier datasets have been optimized; their median
        # speedup ratio is a fair projection of what this tier will see
        # under the same patch. Avoids the "22 min projection, 31 s actual"
        # footgun where the gate fires on a stale baseline-only estimate.
        ratio = _median_speedup_ratio(task_dir, thread=thread,
                                       exclude_dataset=dataset_name)
        if ratio and ratio > 1.0:
            return baseline / ratio, (
                f"baseline ÷ median speedup {ratio:.1f}× from other tiers"
            )
        return baseline, "baseline"
    return None, "unknown"


def _median_speedup_ratio(task_dir: Path, thread: int,
                           exclude_dataset: str | None = None) -> float | None:
    """Median (baseline / best_speed) across datasets that have both at thread.

    Returns None when no datasets qualify.
    """
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.exists():
        return None
    try:
        entries = parse_datasets(yaml_path)
    except Exception:
        return None
    ratios: list[float] = []
    for entry in entries:
        name = entry.get("name")
        if not name or name == exclude_dataset:
            continue
        baseline = get_baseline_speed(task_dir, name, thread=thread)
        best = find_best_speed_at_dataset(task_dir, name, thread=thread)
        if baseline and baseline > 0 and best and best > 0 and baseline > best:
            ratios.append(baseline / best)
    if not ratios:
        return None
    return statistics.median(ratios)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _status_paths(task_dir: Path, pathspecs: tuple[str, ...]) -> list[str]:
    """Return changed paths under pathspecs, covering staged/untracked edits."""
    status = git(
        "status", "--porcelain", "--", *pathspecs,
        cwd=task_dir, check=False,
    )
    paths = []
    for line in (status or "").splitlines():
        if not line.strip():
            continue
        # git() strips leading whitespace, so an unstaged " M file" line can
        # arrive here as "M file". Handle both normal and stripped porcelain.
        if len(line) > 2 and line[2] == " ":
            path = line[3:].strip()
        elif len(line) > 1 and line[1] == " ":
            path = line[2:].strip()
        else:
            path = line[3:].strip() if len(line) > 3 else line.strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if path:
            paths.append(path)
    return sorted(set(paths))


def _task_definition_changes(task_dir: Path) -> list[str]:
    return _status_paths(task_dir, _TASK_DEFINITION_PATHS)


def _pending_rounds(results_tsv: Path) -> list[str]:
    if not results_tsv.exists():
        return []
    return [
        r.get("round", "?")
        for r in _read_results_rows(results_tsv)
        if r.get("status") == "pending"
    ]


def _commit_files(task_dir: Path, commit_sha: str) -> list[str]:
    """Paths in a commit, normalized to task-relative for in-task files.

    `git diff-tree` always returns repo-relative paths. When the task is a
    subdirectory of a shared repo, files like `test_fix/sc_leiden/pipeline/run.py`
    would never match the task-relative allowlist (`pipeline/run.py`) used by
    `_audit_commit_files`, producing false-positive scope warnings. Strip the
    task prefix here so in-task paths compare correctly; out-of-task files
    keep their repo-relative form, which is what audit *should* flag.
    """
    files_blob = git(
        "diff-tree", "--no-commit-id", "--name-only", "-r",
        commit_sha, cwd=task_dir, check=False,
    )
    files = [f.strip() for f in (files_blob or "").splitlines() if f.strip()]
    git_root_str = git("rev-parse", "--show-toplevel", cwd=task_dir, check=False).strip()
    if not git_root_str:
        return files
    try:
        task_rel = str(task_dir.resolve().relative_to(Path(git_root_str).resolve())).replace("\\", "/")
    except ValueError:
        return files
    if not task_rel or task_rel == ".":
        return files
    prefix = task_rel + "/"
    return [f[len(prefix):] if f.startswith(prefix) else f for f in files]


def _write_setup_audit(task_dir: Path, commit_sha: str, message: str,
                       files: list[str]) -> None:
    log_path = task_dir / ".zyme" / "setup_audit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit_sha[:7],
        "message": message[:200],
        "files": files,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")




def _extract_crash_msg_from_log(log: str) -> str:
    """Pull `crash_msg:` line emitted by runner._extract_crash_msg.

    Single line, already TSV-sanitized at emit time.
    """
    for line in log.splitlines():
        if line.startswith("crash_msg:"):
            return line[len("crash_msg:"):].strip()
    return ""




def _audit_commit_files(task_dir: Path, commit_sha: str, kind: str,
                        phase: str, hypothesis: str) -> None:
    """Log files outside the per-phase whitelist that landed in the commit.

    Non-blocking. Iterate (Phase 2) and validate fix-loop (Phase 3) prompts
    both restrict edits to `pipeline/run.{py,R}`; setup commits are also
    allowed to touch task-definition files and setup scripts. Anything else
    indicates the agent stepped outside the prompt's scope (e.g. relaxing
    thresholds in `task.yaml` mid fix-loop, monkey-patching `evaluate.{py,R}`).

    We don't block — there is a legitimate framework-fix carve-out (iterate
    prompt §220-222) — but we record every violation in `.zyme/edit_audit.log`
    (JSONL, gitignored) and print a visible warning so a human reviewer can
    sweep them after the fact.
    """
    # Native source extensions that run.R/run.py compile via sourceCpp /
    # cython_inline — the iterate prompt explicitly allows splitting into
    # sibling pipeline/*.{cpp,pyx} for non-trivial native kernels.
    _NATIVE_SRC_EXTS = (".cpp", ".c", ".h", ".hpp", ".pyx", ".pxd")

    if kind == "fix-loop":
        allowed_exact = {"pipeline/run.py", "pipeline/run.R"}
        allowed_prefixes: tuple[str, ...] = ()
        allowed_native_src = True
        scope_hint = "pipeline/run.{py,R}, pipeline/*.{cpp,c,h,pyx,pxd}"
    elif kind == "setup":
        allowed_exact = {
            "task.yaml",
            "reference.py",
            "reference.R",
            "evaluate.py",
            "evaluate.R",
        }
        allowed_prefixes = ("pipeline/", "setup/")
        allowed_native_src = False
        scope_hint = "pipeline/, setup/, task.yaml, reference/evaluate.{py,R}"
    else:
        return

    def _is_allowed_native(f: str) -> bool:
        if not allowed_native_src:
            return False
        return (f.startswith("pipeline/") and f.count("/") == 1
                and any(f.endswith(ext) for ext in _NATIVE_SRC_EXTS))

    files = _commit_files(task_dir, commit_sha)
    violations = [
        f for f in files
        if not _is_generated_pipeline_artifact(f)
        if f not in allowed_exact
        and not any(f.startswith(p) for p in allowed_prefixes)
        and not _is_allowed_native(f)
    ]
    if not violations:
        return

    log_path = task_dir / ".zyme" / "edit_audit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit_sha[:7],
        "kind": kind,
        "phase": phase,
        "hypothesis": hypothesis[:200],
        "violations": violations,
    }
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(
        f"\n[audit] {kind} commit {commit_sha[:7]} touched files outside "
        f"the {kind} scope ({scope_hint}):",
        flush=True,
    )
    for v in violations:
        print(f"  - {v}", flush=True)
    print(
        f"  Recorded to .zyme/edit_audit.log for human audit. "
        f"Non-blocking — the commit stands. If this was a deliberate "
        f"framework-fix carve-out, document the reason in memory/discoveries.md.\n",
        flush=True,
    )




def cmd_run(args):
    """Commit pipeline change with hypothesis, then run pipeline + evaluate.

    Default: requires a positional `hypothesis` arg AND a pending pipeline change.
    Stages pipeline/ + task.yaml, commits with hypothesis as message, then runs.

    With `--rerun`: skips the commit step and re-executes the current HEAD.
    Useful for measuring metric stability under flake (e.g. BLAS threading
    non-determinism). The hypothesis arg must be omitted in this mode.
    """
    task_dir = task_dir_from_args(args)
    _, _, round_counter_file = zyme_state(task_dir)

    # Git-presence guard. zyme run requires a git repo: results.tsv references
    # commit hashes, best.ref stores a SHA, --rerun re-measures HEAD. zyme
    # status degrades gracefully (read-only), so the inconsistency between
    # `zyme run` hard-fail and `zyme status` graceful confused users — surface
    # an actionable error here rather than the generic "git X failed" from the
    # subprocess wrapper.
    inside_repo = git("rev-parse", "--is-inside-work-tree",
                      cwd=task_dir, check=False)
    if not inside_repo or inside_repo.strip() != "true":
        die(
            f"not a git repository: {task_dir} has no .git/. zyme run requires "
            f"git for commit tracking (results.tsv rows reference commit hashes, "
            f"best.ref stores a SHA). Bootstrap with:\n"
            f"  cd {task_dir}\n"
            f"  git init && git add . && git commit -m 'initial'\n"
            f"or copy a working .git from a sibling task. (zyme status works "
            f"without git for read-only inspection; zyme run cannot.)"
        )

    # --setup short-circuit: commit canonical Setup-time files, no pipeline
    # run, no results.tsv row. Lets agents land threading wiring, tier
    # additions, etc. on HEAD before a verify/validate matrix without
    # consuming a round or polluting results.tsv with a phantom measurement.
    if getattr(args, "setup", None) is not None:
        if args.rerun:
            die("--setup and --rerun are mutually exclusive")
        if args.hypothesis:
            die("--setup takes its message via --setup MSG; "
                "do not also pass a positional hypothesis")
        pending = _pending_rounds(task_dir / "results.tsv")
        if pending:
            die(
                "--setup refused while a pending decision round exists "
                f"({', '.join(pending[:5])}). Accept or reject the pending "
                "round before changing task-definition files."
            )
        msg = args.setup.strip()
        if not msg:
            die("--setup requires a non-empty message")
        for path in (
            "pipeline/",
            "setup/",
            "task.yaml",
            "reference.py",
            "reference.R",
            "evaluate.py",
            "evaluate.R",
        ):
            target = task_dir / path
            if path.endswith("/") or target.exists():
                git("add", path, cwd=task_dir, check=False)
        _unstage_generated_pipeline_artifacts(task_dir)
        cached = git("diff", "--cached", "--name-only", cwd=task_dir)
        if not cached:
            die("--setup: nothing to commit. No staged changes in pipeline/, "
                "setup/, task.yaml, reference.{py,R}, or evaluate.{py,R}. "
                "Edit one of those first, then re-run --setup.")
        commit_msg = msg if msg.lower().startswith("setup:") else f"setup: {msg}"
        git("commit", "-m", commit_msg, cwd=task_dir)
        sha_full = git("rev-parse", "HEAD", cwd=task_dir)
        sha = sha_full[:7]
        setup_files = _commit_files(task_dir, sha_full)
        _audit_commit_files(
            task_dir, sha_full, "setup",
            getattr(args, "phase", None) or "optimize", commit_msg,
        )
        _write_setup_audit(task_dir, sha_full, commit_msg, setup_files)
        evaluator_files = sorted(set(setup_files) & _SETUP_EVALUATOR_FILES)
        if evaluator_files and (task_dir / "results.tsv").exists():
            info(
                "WARN: task truth surface changed "
                f"({', '.join(evaluator_files)}). Prior measurements may "
                "not be comparable; this setup commit was recorded in "
                ".zyme/setup_audit.log."
            )

        # Advance best.ref to the setup commit. best.ref is the "rejection
        # floor" that `zyme reject` resets to (commands.py: cmd_reject does
        # `git reset --hard best.ref`). Without this, a fix-loop reject
        # immediately after --setup would silently wipe the setup commit
        # along with the rejected attempt. Setup commits are by definition
        # additive plumbing on top of the canonical baseline, so advancing
        # the floor preserves them. Downstream queries (find_best_speed_at_
        # dataset etc.) gracefully return None when there's no results.tsv
        # row at the new floor — display becomes "—" until the next keep.
        _, best_ref, _ = zyme_state(task_dir)
        prior_best = best_ref.read_text().strip()[:7] if best_ref.exists() else None
        best_ref.write_text(sha_full)

        info(f"setup commit: {sha}  ({commit_msg})")
        info("No pipeline run; no results.tsv row; no round consumed.")
        if prior_best:
            info(
                f"best.ref advanced: {prior_best} → {sha}. "
                f"Future `zyme reject` resets here, preserving the setup commit."
            )
        else:
            info(f"best.ref initialized to {sha} (rejection floor for future attempts).")
        return

    task_def_changes = _task_definition_changes(task_dir)
    if task_def_changes:
        die(
            "task-definition files changed outside --setup: "
            f"{', '.join(task_def_changes)}\n"
            "Use `zyme run --setup \"<task-definition repair>\"` if this "
            "is a deliberate repair. Ordinary run/rerun measurements may "
            "only use the committed task definition."
        )

    # Version-drift check: warn loudly when upstream_repo/'s declared version
    # disagrees with the installed (runtime) version. Cached by upstream_repo
    # HEAD SHA — runs once per clone state, not once per round. Non-fatal:
    # if metadata is missing or interpreter spawn fails, returns None.
    drift = check_upstream_version_drift(task_dir)
    if drift and not drift.get("agrees", True):
        info(
            f"WARN: version drift detected. upstream_repo/{drift['source']} declares "
            f"{drift['package']} {drift['upstream_version']}, but the installed "
            f"(runtime) version is {drift['installed_version']}. Your override "
            f"wraps the INSTALLED code paths — read installed function bodies via "
            f"`getFromNamespace(fn, '{drift['package']}')` (R) or "
            f"`inspect.getsource({drift['package']}.fn)` (Python) BEFORE designing "
            f"the override. Treat upstream_repo/ as orientation only."
        )

    # Resolve which tiers we're running against. The primary tier (--dataset,
    # default = first listed) is the decision row in fresh mode. Each entry in
    # --extra-tiers is an additional free measurement (status=rerun) at the
    # same commit — semantics independent of --rerun.
    primary_request = [args.dataset] if args.dataset else None
    extras_raw = (args.extra_tiers or "").split(",") if args.extra_tiers else []
    extra_request = [t.strip() for t in extras_raw if t.strip()]
    try:
        primary_tiers = resolve_tiers(task_dir / "task.yaml", primary_request)
        extra_tiers_resolved = (
            resolve_tiers(task_dir / "task.yaml", extra_request)
            if extra_request else []
        )
    except ValueError as e:
        die(str(e))
    # Reject overlap: same tier in both --dataset and --extra-tiers would
    # double-count, write two rows for the same (tier, thread, commit), and
    # confuse downstream reads. Cheap to check up front.
    primary_names = {e["name"] for e in primary_tiers}
    overlap = [e for e in extra_tiers_resolved if e["name"] in primary_names]
    if overlap:
        die(
            f"--extra-tiers includes the primary tier ('{overlap[0]['tier']}'); "
            f"drop it from --extra-tiers."
        )
    tiers = [*primary_tiers, *extra_tiers_resolved]

    # --n defaults to 1 in --rerun mode (cheap quick re-check). Pass --n 3
    # explicitly when a borderline call needs mean ± stdev. Outside --rerun,
    # --n is meaningless.
    raw_n = getattr(args, "n_reps", None)
    if args.rerun:
        n_reps = max(1, raw_n if raw_n is not None else 1)
    else:
        if raw_n is not None and raw_n != 1:
            die("--n N requires --rerun (only meaningful for noise-floor measurement)")
        n_reps = 1

    if args.rerun:
        if args.hypothesis:
            die("--rerun is for re-measuring current HEAD; do not pass a hypothesis")
        rep_msg = f" × {n_reps}" if n_reps > 1 else ""
        info(f"re-run mode{rep_msg}: no commit; current HEAD will be re-measured (does not consume round budget)")
        # Up-front cost banner. The default --n=3 surprises agents who think
        # of `--rerun` as "one quick re-measurement"; at scale-validation
        # tiers (xlarge OOD), 3 reps × 2 tiers can mean hours of wall time.
        # Estimate from current-HEAD measurements when available. After large
        # speedups the upstream baseline can be minutes while HEAD is
        # sub-second; using baseline as the first choice makes the cost banner
        # stale and can trip the safety gate for cheap reruns.
        if n_reps * len(tiers) > 1:
            est_lines = []
            est_total_sec = 0.0
            est_unknown = False
            est_thread = int(
                getattr(args, "thread", None)
                or parse_baseline_threads(task_dir / "task.yaml")[0]
            )
            for t in tiers:
                est_speed, est_source = _current_head_speed_estimate(
                    task_dir, t["name"], thread=est_thread)
                if not est_speed or est_speed <= 0:
                    est_unknown = True
                    est_lines.append(
                        f"  - tier={t['tier']} ({t['name']}): "
                        f"n={n_reps}, estimate unknown"
                    )
                else:
                    sub = est_speed * n_reps
                    est_total_sec += sub
                    est_lines.append(
                        f"  - tier={t['tier']} ({t['name']}): n={n_reps} × "
                        f"~{_format_duration(est_speed)} ({est_source}) "
                        f"≈ {_format_duration(sub)}"
                    )
            total_str = (
                _format_duration(est_total_sec)
                + (" (excluding tiers with unknown estimates)" if est_unknown else "")
                if est_total_sec > 0 else "unknown — no prior timing recorded"
            )
            print(
                f"\n[zyme run --rerun] Total: {n_reps} rep(s) × {len(tiers)} tier(s) "
                f"= {n_reps * len(tiers)} measurement(s); est wall-time ≈ {total_str}.",
                flush=True,
            )
            for ln in est_lines:
                print(ln, flush=True)
            print(
                f"  Pass `--n 1` for a single-shot rerun "
                f"(cheap; skip the variance summary). Ctrl-C now if this is "
                f"too expensive at this scale.\n",
                flush=True,
            )
            # Hard safety gate: block when the projection exceeds 10 min and
            # --yes was not passed. Catches the recurring "agent missed --n 1
            # on OOD tier" footgun before paying ~60 min wall time. Skipped
            # for unknown-baseline runs (can't gate what we can't estimate).
            _RERUN_GATE_SEC = 10 * 60
            if est_total_sec > _RERUN_GATE_SEC and not getattr(args, "yes", False):
                die(
                    f"projection ({est_total_sec / 60:.1f} min) exceeds the "
                    f"{_RERUN_GATE_SEC // 60}-min safety gate.\n"
                    f"  - Pass `--n 1` for a single-rep rerun (often what you wanted).\n"
                    f"  - Or trim tiers via `--extra-tiers ...` to reduce coverage.\n"
                    f"  - Or pass `--yes` to acknowledge the cost and proceed.\n"
                )
    else:
        if not args.hypothesis:
            die("hypothesis is required (or use --rerun to re-measure current HEAD)")
        # Auto-prepend `[scale-fix]` for `--phase validate` rounds. Previously
        # this was a prompt-only convention agents had to remember; missing
        # it broke `grep [scale-fix] results.tsv` triage. Idempotent — skip
        # if the agent already prefixed (e.g. `[scale-fix] [conservative] ...`).
        hypothesis = args.hypothesis
        phase_resolved = getattr(args, "phase", None) or "optimize"
        if phase_resolved == "validate" and not hypothesis.lstrip().startswith("[scale-fix]"):
            hypothesis = f"[scale-fix] {hypothesis}"
            info(f"auto-prefixed `[scale-fix]` for --phase validate round.")

        # Hoist audit (runs BEFORE the commit so violations don't leave ghost
        # commits behind). Catches upstream-internal calls before t0 — the
        # decontx round-58 / fgsea pre-warm pattern. Brace-aware R scan skips
        # override-wrapper function bodies. Three outcomes:
        #   - task.yaml::hoist_exempt set       → skip + log "exempted"
        #   - --bypass-hoist <reason> passed    → warn + log "bypassed", continue
        #   - violation + no bypass             → print warning, log "blocked", die
        _bypass_reason = getattr(args, "bypass_hoist", None)
        _hoist_exempt_reason = read_hoist_exempt(task_dir)
        for _pipe_name in ("pipeline/run.R", "pipeline/run.py"):
            _pipe_path = task_dir / _pipe_name
            if not _pipe_path.exists():
                continue
            _violations = scan_pipeline_hoist(_pipe_path)
            if not _violations:
                continue
            if _hoist_exempt_reason is not None:
                print(
                    f"[hoist-audit] {_pipe_name}: {len(_violations)} violation(s) "
                    f"— skipped (task.yaml::hoist_exempt: {_hoist_exempt_reason})",
                    flush=True,
                )
                append_hoist_log(
                    task_dir, round_num=None, commit=None,
                    pipeline_rel=_pipe_name, violations=_violations,
                    outcome="exempted", bypass_reason=_hoist_exempt_reason,
                    hypothesis=hypothesis,
                )
                continue
            _msg = format_violations_message(
                _pipe_path, _violations, hypothesis=hypothesis,
            )
            print(_msg, flush=True)
            if _bypass_reason:
                print(
                    f"\n[hoist-audit] BYPASSED for this round "
                    f"(reason: {_bypass_reason}). Logged to .zyme/hoist_log.jsonl.",
                    flush=True,
                )
                append_hoist_log(
                    task_dir, round_num=None, commit=None,
                    pipeline_rel=_pipe_name, violations=_violations,
                    outcome="bypassed", bypass_reason=_bypass_reason,
                    hypothesis=hypothesis,
                )
            else:
                append_hoist_log(
                    task_dir, round_num=None, commit=None,
                    pipeline_rel=_pipe_name, violations=_violations,
                    outcome="blocked", bypass_reason=None,
                    hypothesis=hypothesis,
                )
                sys.exit(2)

        # Stage + commit pipeline changes (the "attempt commit")
        git("add", "pipeline/", cwd=task_dir)
        _unstage_generated_pipeline_artifacts(task_dir)
        cached = git("diff", "--cached", "--name-only", cwd=task_dir)
        if not cached:
            die("nothing to commit; pipeline unchanged from current HEAD — edit pipeline/run.py first, or pass --rerun to re-measure stability at the same commit")
        git("commit", "-m", hypothesis, cwd=task_dir)
        fix_loop_sha = git("rev-parse", "HEAD", cwd=task_dir)
        _audit_commit_files(
            task_dir, fix_loop_sha, "fix-loop", phase_resolved, hypothesis,
        )
        args.hypothesis = hypothesis  # downstream code reads args.hypothesis for results.tsv row

    commit_sha_full = git("rev-parse", "HEAD", cwd=task_dir)
    commit_sha = commit_sha_full[:7]

    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        results_tsv.write_text(results_header_for_task(task_dir))
    else:
        # Backfill phase column on a pre-phase results.tsv (idempotent).
        migrate_results_add_phase(results_tsv)
        # Backfill prompt_id column when the task is a bench task but
        # results.tsv pre-dates that scaffolding (idempotent).
        ensure_results_schema(task_dir, results_tsv)

    # Resolve phase for this run (default optimize). Stamped on every row
    # written below — baseline promotions, decision rows, secondary tiers.
    phase = getattr(args, "phase", None) or "optimize"

    # K2: resolve thread count for this run. Defaults to
    # `task.yaml::baseline_threads[0]` — the canonical thread regime the task
    # was initialized to baseline at, so candidate rows align with baseline
    # rows by construction. `--thread <N>` overrides. Falls back to
    # LEGACY_THREAD=1 only when task.yaml has no baseline_threads field
    # (legacy K1 tasks). Every row written this run is tagged with this
    # thread count and divides by the same-thread baseline.
    thread_explicit = getattr(args, "thread", None) is not None
    default_thread = parse_baseline_threads(task_dir / "task.yaml")[0]
    current_thread = int(getattr(args, "thread", None) or default_thread)
    if thread_explicit:
        print(f"[zyme] thread={current_thread} (explicit)", flush=True)
    else:
        print(
            f"[zyme] thread={current_thread} "
            f"(default from task.yaml::baseline_threads). "
            f"Override with --thread <N>.",
            flush=True,
        )

    # API-level kwarg divergence check: reference.{py,R} vs pipeline/run.{py,R}.
    # Warn (not fail) when pipeline passes a kwarg to the target function that
    # differs from reference's call — see 2_iterate.md "API-level kwarg" red
    # line. Threading + framework-output kwargs are excluded; path-like values
    # are normalized to their string-literal set so cosmetic __file__.parent
    # differences don't false-positive. Python-only for now.
    _target_function = parse_target_function(task_dir / "task.yaml")
    if _target_function and not _target_function.startswith("<"):
        _upstream_parallel = parse_upstream_parallelism(task_dir / "task.yaml")
        _argdiffs = diff_target_call_kwargs(
            task_dir, _target_function, upstream_parallelism=_upstream_parallel,
        )
        if _argdiffs:
            print(format_divergences(_argdiffs, _target_function), flush=True)
        # Pre-warm hack check: target call before the timer in pipeline/run.py
        # with args identical to the in-timer call (xclim-style). Synthetic
        # JIT warmups with dummy args don't trigger because args differ.
        _prewarm_msg = check_target_prewarm(task_dir, _target_function)
        if _prewarm_msg:
            print(_prewarm_msg, flush=True)
        # Side-channel parallelism: pipeline engages threads via numba.set_num_threads /
        # os.environ['OMP_NUM_THREADS'] / peer setters that reference.py doesn't.
        # Warning, not gate — outcome-B rounds legitimately add parallelism upstream lacks.
        _side_channel = diff_side_channel_parallelism(task_dir)
        if _side_channel:
            print(format_side_channel_divergences(_side_channel), flush=True)

    # Iterate tiers. First tier is the decision row (status=pending unless --rerun);
    # subsequent tiers are secondary measurements (status=rerun, terminal).
    decision_round_for_this_run = None  # set when idx=0 hits, used by idx>0
    measurements = collections.defaultdict(list)  # tier_name -> [(speed, peak, cpu)]
    for rep_idx in range(n_reps):
      if n_reps > 1:
        print(f"\n=== rerun rep {rep_idx + 1}/{n_reps} ===", flush=True)
      for idx, tier_entry in enumerate(tiers):
        # K2: no lazy stash promotion. Baselines are written directly to
        # results.tsv by `cmd_record_baseline`. If there's no thread-N
        # baseline, `get_baseline_speed` falls back to the dataset's
        # existing baseline (typically thread=1 upstream) so speedup_pct
        # stays meaningful. Surface that fallback so the agent can decide
        # whether re-baselining at the new thread budget is appropriate.
        if not has_baseline_at_thread(task_dir, tier_entry["name"], current_thread):
            fallback_speed = get_baseline_speed(task_dir, tier_entry["name"],
                                                thread=current_thread)
            if fallback_speed is not None and fallback_speed > 0:
                info(
                    f"NOTE: tier='{tier_entry['tier']}' ({tier_entry['name']}) "
                    f"has no thread={current_thread} baseline; "
                    f"`speedup_pct` will be computed vs the existing baseline "
                    f"({fallback_speed:.3f}s). If thread={current_thread} changes "
                    f"upstream's runtime materially, re-baseline via "
                    f"`zyme reference --tier {tier_entry['tier']} "
                    f"--thread {current_thread}`."
                )
            else:
                info(
                    f"WARN: tier='{tier_entry['tier']}' ({tier_entry['name']}) "
                    f"has no baseline in results.tsv. "
                    f"`speedup_pct` for this row will be 0. "
                    f"Record one via `zyme record-baseline --tier "
                    f"{tier_entry['tier']} --thread {current_thread} --speed-sec <X>` "
                    f"or `zyme reference --tier {tier_entry['tier']} "
                    f"--thread {current_thread}`."
                )

        # Compute round_label for this row.  Cross-check round_counter_file
        # against results.tsv to self-heal if the counter file was lost or
        # reverted (e.g. legacy task where .zyme/ was tracked by git).
        current_decision = int(round_counter_file.read_text()) if round_counter_file.exists() else 0
        tsv_decision = count_decision_rounds(results_tsv, phase=phase)
        if tsv_decision > current_decision:
            current_decision = tsv_decision
            round_counter_file.write_text(str(current_decision))
        if args.rerun:
            if current_decision < 1:
                die("--rerun requires a prior decision round; results.tsv has no committed attempt yet")
            seq = next_rerun_seq(results_tsv, current_decision)
            round_label = f"{current_decision}.{seq}"
        elif idx == 0:
            current_decision += 1
            round_counter_file.write_text(str(current_decision))
            round_label = str(current_decision)
            decision_round_for_this_run = current_decision
        else:
            # Multi-tier secondary at the same decision round (idx > 0, not --rerun).
            seq = next_rerun_seq(results_tsv, decision_round_for_this_run)
            round_label = f"{decision_round_for_this_run}.{seq}"

        label = args.hypothesis if args.hypothesis else "(re-run)"
        tier_label = f"{tier_entry['tier']} / {tier_entry['name']}"
        print(f"=== round {round_label} ({commit_sha}) [{tier_label}]: {label} ===", flush=True)
        # Start/done bracket — gives a visible signpost around the silent
        # run_task block (subprocess output is buffered, not streamed). For
        # --extra-tiers reruns this is the only way to tell which tier is
        # currently running between consecutive tier headers.
        _start_ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{_start_ts}] [tier={tier_label}] pipeline running ...", flush=True)
        _t0 = time.monotonic()

        log_content = run_task(task_dir, dataset_entry=tier_entry, thread=current_thread)
        _wall = time.monotonic() - _t0
        print(log_content, flush=True)
        print(f"[tier={tier_label}] pipeline done in {_wall:.1f}s", flush=True)

        speed_sec, peak_mb, metrics, status = parse_log(log_content)

        # Crash rows must never carry a speed/speedup/peak/metric payload.
        # Otherwise speed_sec=0.0 + a real baseline yields speedup_pct=100.0
        # (a falsely positive row that downstream summaries cannot distinguish
        # from a legitimate keep). Zero out everything; status=crash is the
        # only signal a crash row should carry.
        if status == "crash":
            speed_sec = None
            peak_mb = None
            metrics = {}

        if status != "crash":
            if args.rerun:
                status = "rerun"
            elif idx > 0:
                status = "rerun"

        baseline_speed = get_baseline_speed(
            task_dir, tier_entry["name"], thread=current_thread)
        if (status != "crash" and baseline_speed and baseline_speed > 0
                and speed_sec is not None):
            speedup_pct = (1 - speed_sec / baseline_speed) * 100.0
        else:
            speedup_pct = 0.0
            if speed_sec is not None and status != "crash":
                info(
                    f"WARN: no baseline for (dataset={tier_entry['name']}, "
                    f"thread={current_thread}); speedup_pct=0 for this row."
                )

        # Capture median speed of the current best commit at (tier, thread)
        # BEFORE writing the row, so the just-written measurement isn't
        # included in its own comparison. Thread-aware to keep apples-to-
        # apples within the same threading regime.
        prior_best_speed = find_best_speed_at_dataset(
            task_dir, tier_entry["name"], thread=current_thread)
        _, prior_best_cv, prior_best_n = best_cv_at_dataset(
            task_dir, tier_entry["name"], thread=current_thread)
        prior_wall_cpu, prior_wall_cpu_n = best_wall_cpu_ratio_at_dataset(
            task_dir, tier_entry["name"], thread=current_thread)

        metrics_json = json.dumps(metrics, separators=(",", ":"))
        hypothesis = args.hypothesis if args.hypothesis else ""
        description = ""
        if status == "crash":
            description = _extract_crash_msg_from_log(log_content)
        append_results_row(results_tsv, task_dir, {
            "round": round_label,
            "commit": commit_sha,
            "dataset": tier_entry["name"],
            "speed_sec": f"{speed_sec or 0:.3f}",
            "speedup_pct": f"{speedup_pct:.1f}",
            "peak_mb": f"{peak_mb or 0:.1f}",
            "status": status,
            "metrics_json": metrics_json,
            "hypothesis": hypothesis,
            "description": description,
            "phase": phase,
            "thread": str(current_thread),
        })

        # `vs best` line — answers the question agents actually have most
        # rounds ("did this beat best?"), without making them grep results.tsv.
        # Skipped silently when there's no prior measurement of best at this
        # tier (first attempt, or first time at a new tier).
        # When best has ≥3 prior measurements at this tier, also report the
        # rolling CV and flag deltas that fall inside that noise band — single-
        # rep delta vs best-median is otherwise an apples/oranges trap (a +18%
        # single-shot regression can be a -16% mean win once you rerun).
        if prior_best_speed and prior_best_speed > 0 and speed_sec is not None and speed_sec > 0 and status != "crash":
            delta_s = speed_sec - prior_best_speed
            delta_pct = (speed_sec / prior_best_speed - 1) * 100.0
            sign_s = "+" if delta_s >= 0 else ""
            sign_pct = "+" if delta_pct >= 0 else ""
            cv_suffix = ""
            if prior_best_cv is not None and prior_best_n >= 3:
                cv_suffix = f" ± {prior_best_cv:.1f}% CV (n={prior_best_n})"
            # CV suffix prints baseline calibration so the agent can self-judge
            # whether a borderline delta is real or noise without forced reruns.
            print(
                f"[zyme] vs best [{tier_entry['name']}]: "
                f"{sign_s}{delta_s:.3f}s ({sign_pct}{delta_pct:.1f}%) "
                f"— current {speed_sec:.3f}s, best-median {prior_best_speed:.3f}s"
                f"{cv_suffix}",
                flush=True,
            )

        # Auto-rerun removed (was: framework forced 2 extra reps on borderline
        # decision rows). The agent now reads baseline CV from the printed `vs
        # best` line + .zyme/baseline_noise.json and decides for itself whether
        # to invoke `zyme run --rerun` (e.g. on host-load suspicion).
        # Wall/cpu drift. A higher wall/cpu ratio usually means host pressure
        # or scheduling noise. A lower wall/cpu ratio usually means the current
        # code used more CPU cores per wall second, i.e. intentional
        # parallelism or a changed CPU-accounting profile, not host load.
        cur_cpu = metrics.get("cpu_sec")
        if (cur_cpu is not None and speed_sec is not None and status != "crash"
                and prior_wall_cpu is not None and prior_wall_cpu > 0):
            try:
                cur_cpu_f = float(cur_cpu)
            except (TypeError, ValueError):
                cur_cpu_f = 0.0
            if cur_cpu_f > 0 and speed_sec > 0:
                cur_ratio = speed_sec / cur_cpu_f
                if cur_ratio >= prior_wall_cpu * 2.0:
                    drift = cur_ratio / prior_wall_cpu
                    print(
                        f"[zyme] WARN wall/cpu = {cur_ratio:.2f} vs historical "
                        f"{prior_wall_cpu:.2f} ({drift:.1f}× drift, n={prior_wall_cpu_n}) "
                        f"— host load suspect; rerun before reject/rollback",
                        flush=True,
                    )
                elif cur_ratio <= prior_wall_cpu / 2.0:
                    drift = prior_wall_cpu / cur_ratio
                    print(
                        f"[zyme] WARN wall/cpu = {cur_ratio:.2f} vs historical "
                        f"{prior_wall_cpu:.2f} ({drift:.1f}× lower, n={prior_wall_cpu_n}) "
                        f"— parallelism detected; rerun for variance before deciding",
                        flush=True,
                    )

        # Track measurements for --rerun --n aggregate at end. cpu_sec lives
        # in metrics dict (parse_log puts it there since it's not in skip set).
        if n_reps > 1 and status != "crash" and speed_sec is not None:
            measurements[tier_entry["name"]].append({
                "speed": speed_sec,
                "peak": peak_mb or 0.0,
                "cpu": metrics.get("cpu_sec"),
            })

        # Zero-pad the integer part for readable directory listings.
        if "." in round_label:
            head, tail = round_label.split(".", 1)
            padded_round = f"{int(head):03d}.{tail}"
        else:
            padded_round = f"{int(round_label):03d}"
        artifacts_dir = task_dir / "artifacts" / f"{padded_round}_{commit_sha}_{tier_entry['tier']}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for p in pipeline_paths(task_dir):
            shutil.copy(p, artifacts_dir / p.name)
        metrics_file = task_dir / "pipeline" / "metrics.json"
        if metrics_file.exists():
            shutil.copy(metrics_file, artifacts_dir / "metrics.json")
        (artifacts_dir / "run.log").write_text(log_content, encoding="utf-8")

    # Aggregate stats across reps (only meaningful for --rerun --n>1).
    if n_reps > 1 and measurements:
        print(f"\n=== --rerun --n={n_reps} aggregate (mean ± stdev across reps) ===", flush=True)
        for tier_name, samples in measurements.items():
            speeds = [s["speed"] for s in samples]
            peaks = [s["peak"] for s in samples if s["peak"]]
            cpus = [s["cpu"] for s in samples if s["cpu"] is not None]
            n = len(speeds)
            sp_mean = statistics.mean(speeds) if speeds else 0.0
            sp_std = statistics.stdev(speeds) if n > 1 else 0.0
            sp_pct = (sp_std / sp_mean * 100) if sp_mean > 0 else 0.0
            line = f"  {tier_name}: speed={sp_mean:.3f} ± {sp_std:.3f}s ({sp_pct:.1f}% CV, n={n})"
            if peaks:
                pk_mean = statistics.mean(peaks)
                pk_std = statistics.stdev(peaks) if len(peaks) > 1 else 0.0
                line += f"  peak={pk_mean:.1f} ± {pk_std:.1f}MB"
            if cpus:
                cpu_mean = statistics.mean(cpus)
                line += f"  cpu={cpu_mean:.3f}s  wall/cpu={sp_mean/cpu_mean:.2f}" if cpu_mean > 0 else ""
            print(line, flush=True)

    decision_rounds = count_decision_rounds(results_tsv, phase=phase)
    rows_written = n_reps * len(tiers)
    cap = 30 if phase == "validate" else 50
    label = "scale-fix rounds" if phase == "validate" else "decision rounds"
    # Suppress the budget banner when this invocation contributed nothing to it
    # (pure --rerun, no new decision row). Otherwise it spams identical numbers.
    counter_advanced = (not args.rerun)
    if counter_advanced:
        print(
            f"\n[zyme] {rows_written} row(s) logged to results.tsv; "
            f"{label}: {decision_rounds}/{cap} (--rerun + secondary-tier rows excluded from cap)",
            flush=True,
        )
    else:
        print(
            f"\n[zyme] {rows_written} rerun row(s) logged to results.tsv ({label} counter unchanged: {decision_rounds}/{cap})",
            flush=True,
        )

    if counter_advanced:
        try:
            from zyme import progress_guard
            progress_guard.maybe_print_plateau_reminder(task_dir)
        except Exception as e:
            info(f"[progress] plateau scan skipped: {e}")




def cmd_dryrun(args):
    """Run pipeline/run only, without committing or writing results.tsv.

    Use this for syntactic / compilation / sanity checks where you want
    to see if pipeline even executes — without burning a decision round
    on a likely-broken attempt. Skips evaluate too. Doesn't touch git,
    .zyme/, results.tsv, or artifacts/.
    """
    task_dir = task_dir_from_args(args)
    requested = [args.dataset] if args.dataset else None
    try:
        tiers = resolve_tiers(task_dir / "task.yaml", requested)
    except ValueError as e:
        die(str(e))
    # dryrun is single-tier; default to first listed.
    tier_entry = tiers[0]
    info(
        f"dryrun: tier='{tier_entry['tier']}' ({tier_entry['name']}) — "
        f"no commit, no results.tsv, no .zyme update"
    )
    rc, stdout, stderr, wall = dryrun_task(task_dir, dataset_entry=tier_entry)
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    info(f"dryrun: returncode={rc}, wall={wall:.3f}s")




def cmd_accept(args):
    """Mark current attempt as keep; advance best.ref.

    Verifies HEAD matches the pending row's commit before advancing best.ref.
    In a shared repo where sibling tasks may interleave commits between this
    task's `zyme run` and `zyme accept`, a blind `git rev-parse HEAD` could
    capture a sibling's commit; without this check best.ref silently points
    at the wrong SHA and a later `zyme reject` resets to the wrong place.
    """
    task_dir = task_dir_from_args(args)
    _, best_ref, _ = zyme_state(task_dir)
    results_tsv = task_dir / "results.tsv"

    pending = _find_pending_row(results_tsv)
    if pending is None:
        die(
            "no pending row to accept. Run `zyme run` first, or this round was "
            "already accepted/rejected — check `zyme status`."
        )
    expected = (pending.get("commit") or "").strip()
    head_full = git("rev-parse", "HEAD", cwd=task_dir).strip()
    head_short = git("rev-parse", "--short", "HEAD", cwd=task_dir).strip()
    if expected and not (head_short.startswith(expected) or expected.startswith(head_short)):
        die(
            f"HEAD ({head_short}) does not match the pending round's commit ({expected}).\n"
            "Something advanced git HEAD between `zyme run` and `zyme accept` — in a "
            "shared repo this is usually a sibling task's commit slipping in.\n"
            "Inspect:\n"
            f"    git log --oneline {expected}..HEAD\n"
            "Then either:\n"
            f"    → cherry-pick {expected} back onto HEAD, then `zyme accept` again\n"
            "    → `zyme reject` this round and rerun"
        )

    if not args.description:
        info("note: -m description omitted — future you grep these. " + DESCRIPTION_HINT)

    best_ref.write_text(head_full)
    update_last_status(results_tsv, "keep", args.description)
    state_path = _regenerate_best_state(task_dir)
    if state_path is not None:
        info(f"accepted {head_short}; best advanced; best_state.md regenerated")
    else:
        info(f"accepted {head_short}; best advanced")

    # Reprofile hint: when cumulative speedup crosses a 20% boundary since
    # the last hint, nudge the agent to reprofile so stale assumptions about
    # where the bottleneck lives don't waste rounds.
    try:
        _maybe_reprofile_hint(task_dir, results_tsv)
    except Exception:
        pass

    # Memory budget warning: speed is the gate, but peak_mb can regress
    # silently — 11 such regressions accumulated unnoticed before this hook
    # was added (2026-05-31). Warns (does not block) when the accepted round's
    # peak_mb is >30% over the matching (dataset, thread) baseline so the
    # agent investigates before stacking another memory-allocating round.
    # Failures here MUST NOT block accept — accept has already committed.
    try:
        _maybe_memory_regression_warning(task_dir, results_tsv)
    except Exception:
        pass

    # Housekeeping reminder (mechanical scan + cooldown). Failures here MUST
    # NOT block accept — accept is the user-facing primitive.
    try:
        from zyme import housekeeping
        rows = _read_results_rows(task_dir / "results.tsv")
        latest = rows[-1] if rows else {}
        try:
            current_round = int(float(latest.get("round") or 0))
        except (ValueError, TypeError):
            current_round = 0
        hypothesis = latest.get("hypothesis", "")
        if getattr(args, "dismiss_housekeeping", False):
            housekeeping.mark_dismissed(task_dir, current_round)
        else:
            housekeeping.maybe_print_reminder(task_dir, current_round, hypothesis)
    except Exception as e:
        info(f"[housekeeping] scan skipped: {e}")




def _maybe_reprofile_hint(task_dir: Path, results_tsv: Path) -> None:
    """Print a reprofile suggestion when cumulative speedup crosses a 20% step."""
    rows = _read_results_rows(results_tsv)
    keeps = [r for r in rows if r.get("status") == "keep"]
    if len(keeps) < 2:
        return
    latest = keeps[-1]
    tier = latest.get("dataset", "")
    baseline_speed = None
    for r in rows:
        if r.get("status") == "baseline" and r.get("dataset") == tier:
            try:
                baseline_speed = float(r.get("speed_sec") or 0)
            except ValueError:
                pass
            break
    if not baseline_speed or baseline_speed <= 0:
        return
    try:
        current_speed = float(latest.get("speed_sec") or 0)
    except ValueError:
        return
    speedup_pct = (1 - current_speed / baseline_speed) * 100

    hint_file = task_dir / ".zyme" / "last_reprofile_hint_pct"
    last_hint_pct = 0.0
    if hint_file.exists():
        try:
            last_hint_pct = float(hint_file.read_text().strip())
        except (ValueError, OSError):
            pass

    step = 20.0
    if speedup_pct >= last_hint_pct + step:
        new_threshold = (speedup_pct // step) * step
        hint_file.parent.mkdir(parents=True, exist_ok=True)
        hint_file.write_text(str(new_threshold))
        info(
            f"[hint] cumulative speedup is now {speedup_pct:.0f}% — the bottleneck "
            f"has likely shifted. Consider `zyme profile` to check before the next round."
        )


# Memory regression threshold: warn when patched peak_mb exceeds this fraction
# of the matching baseline. 30% chosen as a balance between "loud enough to
# catch genuinely runaway allocators" and "quiet enough that random rep noise
# doesn't fire it" — typical rep-to-rep peak_mb noise on the established tasks
# is in single-digit percent.
_MEMORY_REGRESSION_PCT_THRESHOLD = 30.0


def _maybe_memory_regression_warning(task_dir: Path, results_tsv: Path) -> None:
    """Warn when the just-accepted round's peak_mb regresses >30% vs baseline.

    Speed is the headline number the accept gate enforces, but memory can
    regress silently — historically 11 such regressions stacked unnoticed
    before this hook was added (2026-05-31). This check reads the latest
    keep row and the same-(dataset, thread) baseline, computes peak_mb
    regression, and emits a one-line `[memory]` warning when it exceeds
    `_MEMORY_REGRESSION_PCT_THRESHOLD`. Non-blocking: accept has already
    committed; the warning tells the agent to investigate next round.

    Silently skips when peak_mb is missing on either side or no matching
    baseline is recorded yet. Keeps the same (dataset, thread) join logic
    as `_regenerate_best_state` so multi-thread keeps don't get falsely
    compared against a serial baseline.
    """
    if not results_tsv.exists():
        return
    rows = _read_results_rows(results_tsv)
    keeps = [r for r in rows if r.get("status") == "keep"]
    if not keeps:
        return
    latest = keeps[-1]
    tier = latest.get("dataset", "")
    try:
        latest_thread = int(latest.get("thread") or LEGACY_THREAD)
    except (ValueError, TypeError):
        latest_thread = LEGACY_THREAD
    try:
        latest_peak = float(latest.get("peak_mb") or 0)
    except (ValueError, TypeError):
        return
    if latest_peak <= 0:
        return

    baseline_peak = None
    for r in rows:
        if r.get("status") != "baseline":
            continue
        if r.get("dataset") != tier:
            continue
        try:
            row_thread = int(r.get("thread") or LEGACY_THREAD)
        except (ValueError, TypeError):
            row_thread = LEGACY_THREAD
        if row_thread != latest_thread:
            continue
        try:
            baseline_peak = float(r.get("peak_mb") or 0)
        except (ValueError, TypeError):
            pass
        break
    if not baseline_peak or baseline_peak <= 0:
        return

    regression_pct = (latest_peak - baseline_peak) / baseline_peak * 100.0
    if regression_pct <= _MEMORY_REGRESSION_PCT_THRESHOLD:
        return

    info(
        f"[memory] WARNING: peak_mb regressed {regression_pct:.0f}% "
        f"({latest_peak:.0f} MB vs {baseline_peak:.0f} MB baseline at "
        f"tier={tier!r}, thread={latest_thread}). The accepted round trades "
        f"memory for speed — investigate before stacking another "
        f"memory-allocating round. If intentional (precomputed cache, "
        f"materialized intermediate), record the rationale in -m on the "
        f"next `zyme accept`."
    )


def _regenerate_best_state(task_dir):
    """Write memory/best_state.md — auto-generated patch-stack snapshot.

    Called after every successful `zyme accept`. Reads results.tsv, lists
    every status=keep row in order, and shows headline numbers for the
    latest keep (current best). Intended for human-glance review and
    next-round agent context — agent doesn't need to grep the TSV to know
    what's stacked.

    Returns the written Path on success, or None if no file was written
    (results.tsv missing or no keep rows yet). Callers should not claim
    "best_state.md regenerated" unless the return is non-None.
    """
    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        return None
    rows = _read_results_rows(results_tsv)

    keeps = [r for r in rows if r.get("status") == "keep"]
    if not keeps:
        return None
    latest = keeps[-1]
    latest_tier = latest.get("dataset", "?")
    try:
        latest_thread = int(latest.get("thread") or LEGACY_THREAD)
    except ValueError:
        latest_thread = LEGACY_THREAD

    # Baseline for the latest keep's (tier, thread) — same-thread comparison
    # so multi-thread keeps don't show against a serial baseline.
    baseline_speed = None
    for r in rows:
        if r.get("status") != "baseline":
            continue
        if r.get("dataset") != latest_tier:
            continue
        try:
            row_thread = int(r.get("thread") or LEGACY_THREAD)
        except ValueError:
            row_thread = LEGACY_THREAD
        if row_thread != latest_thread:
            continue
        try:
            baseline_speed = float(r.get("speed_sec") or 0)
        except ValueError:
            pass
        break

    try:
        latest_speed = float(latest.get("speed_sec") or 0)
    except ValueError:
        latest_speed = 0.0
    try:
        latest_peak = float(latest.get("peak_mb") or 0)
    except ValueError:
        latest_peak = 0.0
    speedup_pct = (1 - latest_speed / baseline_speed) * 100 if baseline_speed else 0.0

    metrics = {}
    try:
        metrics = json.loads(latest.get("metrics_json") or "{}")
    except json.JSONDecodeError:
        pass

    lines = [
        "# Current best state",
        "",
        f"_Auto-generated by `zyme accept`. Don't edit — overwritten on every keep._",
        "",
        f"## Latest keep: round {latest.get('round', '?')} ({latest.get('commit', '?')})",
        "",
        f"- **Tier:** {latest_tier}",
    ]
    if baseline_speed:
        lines.append(f"- **Speed:** {latest_speed:.3f}s vs {baseline_speed:.3f}s baseline → **{speedup_pct:.1f}% faster**")
    else:
        lines.append(f"- **Speed:** {latest_speed:.3f}s (no baseline recorded for this tier)")
    if latest_peak:
        lines.append(f"- **Peak memory:** {latest_peak:.1f} MB")
    if metrics:
        cpu = metrics.pop("cpu_sec", None)
        if cpu is not None:
            wall_cpu = (latest_speed / cpu) if cpu > 0 else 0.0
            lines.append(f"- **CPU time:** {cpu:.3f}s (wall/cpu = {wall_cpu:.2f})")
        if metrics:
            metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            lines.append(f"- **Concordance:** {metric_str}")
    lines.append("")
    lines.append(f"## Patch stack ({len(keeps)} keep{'s' if len(keeps) != 1 else ''}, oldest first)")
    lines.append("")
    for k in keeps:
        rnd = k.get("round", "?")
        sha = k.get("commit", "?")
        sp = k.get("speedup_pct", "?")
        desc = (k.get("description") or k.get("hypothesis") or "").strip()
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.append(f"- round {rnd} ({sha}) — {sp}% — {desc}")

    out = task_dir / "memory" / "best_state.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out




def cmd_reject(args):
    """Mark current attempt as discard and reset HEAD to best.

    Hard reset (default) means git log shows only kept progress, not failed
    experiments. The rejected commit's code is preserved in
    `artifacts/<round>_<commit>_<tier>/` and its description in results.tsv,
    so nothing is truly lost.

    Dirty-tree safety: if the working tree has uncommitted changes, the
    default --hard reset would silently wipe them. Common cases: a fix the
    agent typed after diagnosing a crash, or filled-in scaffold content
    that init never committed (best.ref then points at the TBD template).
    We refuse to proceed and require an explicit --keep-tree (preserve the
    changes) or --force (drop them anyway).
    """
    task_dir = task_dir_from_args(args)
    _, best_ref, _ = zyme_state(task_dir)
    results_tsv = task_dir / "results.tsv"

    last_status = _last_data_row_status(results_tsv)
    porcelain = git("status", "--porcelain", cwd=task_dir)
    has_uncommitted = bool(porcelain.strip())

    if has_uncommitted and not args.keep_tree and not args.force:
        if last_status == "crash":
            args.keep_tree = True
            info(
                "last run was a crash and working tree has uncommitted edits — "
                "defaulting to --keep-tree (preserving your fix-in-progress). "
                "Use --force to drop changes instead."
            )
        else:
            die(
                "your working tree has uncommitted changes. The default "
                "`zyme reject` does `git reset --hard`, which would wipe them. Pick one:\n"
                "  → keep working-tree changes:\n"
                "      zyme reject --keep-tree -m '<desc>'\n"
                "  → drop them and reset to best:\n"
                "      zyme reject --force -m '<desc>'"
            )

    # Shared-repo safety: hard-reset to best.ref is repo-wide. If best..HEAD
    # contains commits from sibling tasks, the reset silently removes them
    # from HEAD's history. Reflog still has them, but any sibling whose
    # best.ref points at one of those SHAs now references a commit not on
    # HEAD's ancestry. Detect and refuse unless --force.
    if best_ref.exists() and not args.force:
        best_sha = best_ref.read_text().strip()
        cross_task = _cross_task_commits_between(task_dir, best_sha)
        if cross_task:
            details = "\n".join(
                f"    {sha[:7]}  touches: {', '.join(files[:3])}"
                + (f" (+{len(files) - 3} more)" if len(files) > 3 else "")
                for sha, files in cross_task[:5]
            )
            more = f"\n    ... and {len(cross_task) - 5} more" if len(cross_task) > 5 else ""
            die(
                f"`zyme reject` would `git reset --hard` past {len(cross_task)} commit(s) "
                f"in this repo that touch files OUTSIDE {task_dir.name}/ — likely sibling "
                "tasks sharing this git repo:\n"
                f"{details}{more}\n"
                "These commits stay in the reflog but leave HEAD's ancestry; any sibling "
                "task whose best.ref pins one of them would break.\n"
                "Options:\n"
                "  → if siblings won't be affected, `zyme reject --force -m '<desc>'`\n"
                "  → resolve manually with `git revert <sha>` after consulting siblings"
            )

    if not args.description:
        info("note: -m description omitted — future you grep these. " + DESCRIPTION_HINT)

    update_last_status(results_tsv, "discard", args.description)

    if best_ref.exists():
        best_sha = best_ref.read_text().strip()
        reset_mode = "--mixed" if args.keep_tree else "--hard"
        git("reset", reset_mode, best_sha, cwd=task_dir)
        if args.keep_tree:
            info(f"rejected; HEAD reset to best ({best_sha[:7]}) with --mixed; working tree preserved")
        else:
            info(f"rejected; HEAD reset to best ({best_sha[:7]})")
    else:
        info(
            "rejected; no `best.ref` yet — this was the very first attempt, so there's no "
            "prior 'best' to restore to. HEAD is left at the rejected commit; **re-edit "
            "`pipeline/run.{py,R}` from a clean state before the next `zyme run`**, or run "
            "`git reset --hard HEAD~1` to drop the commit entirely."
        )

    try:
        from zyme import progress_guard
        progress_guard.maybe_print_plateau_reminder(task_dir)
    except Exception as e:
        info(f"[progress] plateau scan skipped: {e}")




def _last_data_row_status(results_tsv: Path):
    """Return the status field of the most recent data row, or None."""
    if not results_tsv.exists():
        return None
    lines = results_tsv.read_text().splitlines()
    for i in range(len(lines) - 1, 0, -1):
        parts = lines[i].split("\t")
        if len(parts) > 6 and parts[6]:
            return parts[6]
    return None


def _find_pending_row(results_tsv: Path):
    """Return the most recent row with status=pending, or None.

    `zyme run` writes status=pending; accept/reject reference this row when
    deciding the round. Used by both to detect "nothing to decide" and (for
    accept) to verify HEAD matches the commit being judged.
    """
    if not results_tsv.exists():
        return None
    for row in reversed(_read_results_rows(results_tsv)):
        if row.get("status") == "pending":
            return row
    return None


def _cross_task_commits_between(task_dir: Path, best_sha: str):
    """List commits in best_sha..HEAD that touch files outside task_dir.

    Used by `zyme reject` to detect when a `git reset --hard best` would
    silently drop sibling-task commits in a shared repo. Returns a list of
    (sha, external_files) tuples; empty list means the reset is safe (every
    intervening commit touched only this task's files, or no intervening
    commits exist).
    """
    rev_list = git("rev-list", f"{best_sha}..HEAD", cwd=task_dir, check=False)
    commits = [c.strip() for c in (rev_list or "").splitlines() if c.strip()]
    if not commits:
        return []
    git_root_str = git("rev-parse", "--show-toplevel", cwd=task_dir, check=False).strip()
    if not git_root_str:
        return []
    try:
        task_rel = str(task_dir.resolve().relative_to(Path(git_root_str).resolve())).replace("\\", "/")
    except ValueError:
        return []
    if not task_rel or task_rel == ".":
        # Task IS the repo root — no sibling tasks possible.
        return []
    prefix = task_rel + "/"
    cross_task = []
    for sha in commits:
        files_blob = git(
            "diff-tree", "--no-commit-id", "--name-only", "-r",
            sha, cwd=task_dir, check=False,
        )
        files = [f.strip() for f in (files_blob or "").splitlines() if f.strip()]
        external = [f for f in files if not f.startswith(prefix) and f != task_rel]
        if external:
            cross_task.append((sha, external))
    return cross_task




def cmd_rollback(args):
    """Demote the most recent accepted (keep) commit; re-point best.ref to the prior keep.

    Used when an accepted round is later found broken — e.g., a tiny-only win that
    silently fails at medium 4 rounds later, or a measurement bug uncovered after a
    few subsequent rounds built on it. Without rollback, the agent has to issue a
    "revert" round (consuming budget + polluting leaderboard) to back out the bad
    accept; with rollback, history stays honest: round N stays at status=rollback,
    best.ref re-points to the prior keep, no new round is logged.

    Walks results.tsv backward to find the most recent status=keep row, flips it
    to status=rollback (appending the optional -m description), locates the prior
    keep, re-points .zyme/best.ref to that commit's SHA, and `git reset --hard`s
    HEAD to match. If no prior keep exists (you're rolling back the very first
    accept), best.ref is removed and HEAD is left as-is.
    """
    task_dir = task_dir_from_args(args)
    _, best_ref, _ = zyme_state(task_dir)
    results_tsv = task_dir / "results.tsv"

    if not results_tsv.exists():
        die("no results.tsv to roll back")

    lines = results_tsv.read_text().splitlines()
    if len(lines) < 2:
        die("no rows in results.tsv to roll back")

    # Walk back; collect all keep-row indices (most recent first).
    keep_indices = []
    for i in range(len(lines) - 1, 0, -1):
        parts = lines[i].split("\t")
        if len(parts) > 6 and parts[6] == "keep":
            keep_indices.append(i)

    if not keep_indices:
        die("no `keep` row found in results.tsv — nothing to roll back")

    last_idx = keep_indices[0]
    parts = lines[last_idx].split("\t")
    while len(parts) < 10:
        parts.append("")
    rolled_round = parts[0]
    rolled_commit = parts[1]

    # --dry-run: report what would change, don't touch anything.
    if getattr(args, "dry_run", False):
        if len(keep_indices) > 1:
            prior_parts = lines[keep_indices[1]].split("\t")
            prior_round = prior_parts[0]
            prior_commit_short = prior_parts[1]
            info(
                f"DRY RUN — would demote round {rolled_round} ({rolled_commit}) "
                f"to status=rollback, repoint best.ref to round {prior_round} "
                f"({prior_commit_short[:7]}), and `git reset --hard` HEAD there. "
                f"Re-run without --dry-run to apply."
            )
        else:
            info(
                f"DRY RUN — would demote round {rolled_round} ({rolled_commit}) "
                f"to status=rollback and remove best.ref (no prior keep). HEAD "
                f"would be left as-is. Re-run without --dry-run to apply."
            )
        return

    # Flip status + annotate description.
    parts[6] = "rollback"
    annotation = f"[rolled back: {args.description}]" if args.description else "[rolled back]"
    parts[9] = f"{parts[9]} {annotation}".strip() if parts[9] else annotation
    lines[last_idx] = "\t".join(parts)
    results_tsv.write_text("\n".join(lines) + "\n")

    if len(keep_indices) > 1:
        prior_parts = lines[keep_indices[1]].split("\t")
        prior_round = prior_parts[0]
        prior_commit_short = prior_parts[1]
        full_sha = git("rev-parse", prior_commit_short, cwd=task_dir).strip()
        best_ref.write_text(full_sha)
        git("reset", "--hard", full_sha, cwd=task_dir)
        info(
            f"rolled back round {rolled_round} ({rolled_commit}); "
            f"best now points at round {prior_round} ({prior_commit_short[:7]})"
        )
    else:
        best_ref.unlink(missing_ok=True)
        info(
            f"rolled back round {rolled_round} ({rolled_commit}); no prior keep exists — "
            f"best.ref removed, HEAD left as-is. Manually re-edit `pipeline/run.{{py,R}}` "
            f"from a clean state, or `git reset --hard <baseline_sha>`."
        )
