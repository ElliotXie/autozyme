"""runner.py — Execute a task's pipeline + evaluate, capture timing/memory/metrics.

Replaces framework/run_task.sh with pure Python so cmd_run doesn't need to
spawn an external shell. Cross-platform (Windows / macOS / Linux).
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from zyme.utils import detect_lang
from zyme.parsers.task_yaml import parse_executor, parse_metrics


def _looks_like_path(spec: str) -> bool:
    return ("/" in spec) or ("\\" in spec) or spec.lower().endswith(".exe")


def _resolve_python(spec: str) -> str:
    """Resolve `executor.python` to an absolute python interpreter path.

    Accepts either a conda env name (e.g. `myenv`) or an absolute path.
    For env names: probes `CONDA_EXE`/known install dirs, then falls back to
    `conda run -n <env> python -c "import sys; print(sys.executable)"`. Raises
    RuntimeError on unresolvable spec — `zyme run` must fail loud rather than
    silently fall through to system python.
    """
    if _looks_like_path(spec):
        if not Path(spec).exists():
            raise RuntimeError(f"executor.python path not found: {spec}")
        return spec

    # Treat as conda env name.
    env_name = spec
    candidates = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).parent.parent
        candidates.append(conda_root / "envs" / env_name)
    conda_root = os.environ.get("AUTOZYME_CONDA_ROOT")
    if conda_root:
        candidates.append(Path(conda_root) / "envs" / env_name)
    home = Path.home()
    for base in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        candidates.append(home / base / "envs" / env_name)
    # Common Windows install locations (conda often on D: while home is C:).
    for base in (Path("D:/anaconda3"), Path("D:/miniconda3"), Path("D:/miniforge3")):
        candidates.append(base / "envs" / env_name)
    for envs_dir in candidates:
        for rel in ("python.exe", "bin/python", "bin/python3"):
            cand = envs_dir / rel
            if cand.exists():
                return str(cand)

    # Fallback: ask conda directly. Slow (~1s) but accurate.
    conda_bin = conda_exe or shutil.which("conda")
    if conda_bin:
        try:
            proc = subprocess.run(
                [conda_bin, "run", "-n", env_name, "python", "-c",
                 "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                resolved = proc.stdout.strip().splitlines()[-1].strip()
                if resolved and Path(resolved).exists():
                    return resolved
        except Exception:
            pass

    raise RuntimeError(
        f"could not resolve conda env '{env_name}' to a python interpreter. "
        f"Tried CONDA_EXE, ~/miniconda3, ~/anaconda3, ~/miniforge3, ~/mambaforge, "
        f"and `conda run`. Set executor.python to an absolute path in task.yaml, "
        f"or create the env."
    )


def build_reference_cmd(task_yaml: Path, ref_path: Path) -> list[str]:
    """Return the argv list for spawning `reference.{py,R}` honoring task.yaml::executor.

    `.py` references resolve `executor.python` (conda env name or absolute
    path) the same way `zyme run` does, falling back to `sys.executable` so
    baseline subprocesses inherit the active interpreter rather than whatever
    `python` resolves to on PATH (which is system Python on macOS — almost
    never the env that actually has the task's deps).

    `.R` references honor `executor.rscript` if set, else fall back to
    `Rscript` on PATH.

    Raises RuntimeError on unresolvable `executor.python` (delegated to
    `_resolve_python` so the user sees the same error message as on `zyme run`).
    """
    executor = parse_executor(task_yaml)
    suffix = ref_path.suffix
    if suffix == ".py":
        py_bin = _resolve_python(executor["python"]) if executor.get("python") else sys.executable
        return [py_bin, str(ref_path)]
    if suffix == ".R":
        rscript_bin = executor.get("rscript", "Rscript")
        return [rscript_bin, str(ref_path)]
    raise RuntimeError(f"unsupported reference script extension: {suffix}")


_SKIP_KEYS = {
    "task", "speed_sec", "baseline_sec", "speedup_pct",
    "peak_memory_mb", "peak_mb", "internal_time", "status", "metrics_json",
}

# Suffixes emitted by `auto_structure_check_all_slots()` (helpers.{py,R}).
# These are framework-generated structural checks that bypass the task.yaml
# allowlist — agents don't declare them per-slot, so the allowlist would
# silently drop failing rows. Pass-only summaries print as `# auto_structure:
# N/N perfect` (comment line, ignored by the regex); failures print as
# `<slot>_<suffix>: 0.0` and need to survive into metrics_json.
_AUTO_STRUCT_SUFFIXES = ("_present", "_shape_match", "_non_degenerate")


def _extract_crash_msg(stderr: str, max_chars: int = 200) -> str:
    """Pull a one-line crash summary from the tail of a subprocess stderr.

    Prefers the last line containing "error" (case-insensitive) — usually the
    informative line in both R and Python tracebacks (e.g. `Error in ...`,
    `TypeError: ...`). Falls back to the last non-empty line. Truncates to
    `max_chars` and sanitizes for TSV (no tabs / newlines).
    """
    if not stderr:
        return ""
    lines = [ln.rstrip() for ln in stderr.splitlines() if ln.strip()]
    if not lines:
        return ""
    err_line = next((ln for ln in reversed(lines) if "error" in ln.lower()), lines[-1])
    msg = err_line.strip()
    if len(msg) > max_chars:
        msg = msg[: max_chars - 3] + "..."
    return msg.replace("\t", " ").replace("\n", " ").replace("\r", " ")


_SIGNAL_EXPLANATIONS = {
    # POSIX signal-died returncodes are 128 + signum. Listing the ones agents
    # actually hit when verify matrices run for hours and the host wobbles.
    134: "SIGABRT — assertion failure or abort() (e.g. memory corruption, glibc detected double-free)",
    137: "SIGKILL — likely OOM-killed by the OS or container (RLIMIT_AS, cgroup memory cap, or `kill -9`)",
    139: "SIGSEGV — segfault (bad memory access; often a C/Rcpp pointer bug)",
    143: "SIGTERM — external process sent a graceful kill (watchdog, init, container shutdown)",
    144: "SIGTERM-equivalent — typical of macOS background-task watchdogs / power-management under host pressure; not a code bug",
}


def _explain_returncode(rc: int) -> str:
    """Return a one-line human-readable hint for a non-zero subprocess returncode.

    Empty string if no special meaning. Used to annotate CRASH messages so
    agents don't have to look up exit code meanings (especially the macOS-
    specific 144 from background-task watchdogs, which is otherwise opaque).
    """
    if rc in _SIGNAL_EXPLANATIONS:
        return _SIGNAL_EXPLANATIONS[rc]
    if rc < 0:
        return f"died from signal {-rc}"
    if 128 <= rc <= 192:
        return f"likely died from signal {rc - 128} (POSIX 128+signum convention)"
    return ""


_TIER_WALL_CAP_S: dict[str, int] = {
    # Hard per-tier wall-clock caps in seconds. SIGKILL the pipeline /
    # evaluate subprocess (and its process group on POSIX) when exceeded —
    # catches runaway O(n²) regressions before they burn hours. Caps are
    # ~2× the init-time baseline bands (tiny ≤10min, medium ≤15min, large
    # ≤30min from 1_init.md). Tier names not listed here get no cap, which
    # preserves behaviour for custom tier configs. `zyme profile`
    # invocations also skip the cap because samplers add 2-5× overhead.
    "tiny":   20 * 60,
    "medium": 30 * 60,
    "large":  60 * 60,
}


def _wall_cap_for_tier(tier: str | None) -> float | None:
    if not tier:
        return None
    return _TIER_WALL_CAP_S.get(tier)


def _run_with_watchdog(cmd, *, cwd, env, mem_cap_gb: float | None,
                       poll_s: float = 2.0, side_observer=None,
                       wall_cap_s: float | None = None,
                       audit_threads: bool = False,
                       thread_budget: int | None = None):
    """Run `cmd` like subprocess.run(capture_output=True, text=True), but
    SIGKILL the entire process group if its summed RSS exceeds `mem_cap_gb`
    or if its wall time exceeds `wall_cap_s`.

    Returns a tuple (returncode, stdout, stderr, killed_peak_gb,
    timed_out_at_s, peak_procs, group_cpu_sec). The last two are populated
    whenever the Popen path's stats poller runs (mem watchdog, audit_threads,
    or any other reason to take the Popen path); the fast subprocess.run
    path returns (0, 0.0) for them since we have no PGID to poll. Caller
    uses peak_procs / group_cpu_sec for the thread-budget audit (catches
    pipelines that fork past ZYME_THREADS).

    `mem_cap_gb=None` or `<= 0` disables the memory watchdog entirely.
    `wall_cap_s=None` or `<= 0` disables the wall-clock cap entirely.
    `audit_threads=True` forces the Popen path even with no mem cap, so
    the stats poller collects peak_procs / group_cpu_sec for the caller's
    post-run audit.

    `side_observer`, when provided, is an object with attach(pid)/detach()
    methods. Called after Popen with the subprocess pid; detach() runs after
    communicate(). Used by `zyme profile --backend native` to spawn macOS
    `sample(1)` against the pipeline process tree without modifying the
    pipeline command.

    When watchdog, side_observer, OR (wall cap on POSIX) is active we use
    the Popen path so we can kill the whole process group; otherwise we
    fall through to plain subprocess.run for speed.

    Process-group SIGKILL (vs killing only the parent) is what catches
    forked workers — joblib/numba/mclapply children would otherwise survive
    a parent kill and keep eating RAM. Requires POSIX (`os.setsid` +
    `os.killpg`); Windows path falls back to a parent-only kill which is
    weaker but still better than no cap.
    """
    use_watchdog = (mem_cap_gb is not None and mem_cap_gb > 0
                    and sys.platform != "win32")
    use_observer = side_observer is not None
    use_wall_cap = wall_cap_s is not None and wall_cap_s > 0
    use_audit = audit_threads and sys.platform != "win32"
    # POSIX wall-cap goes through Popen so we can killpg the worker tree;
    # Windows wall-cap can ride the fast path (subprocess.run kills the
    # parent on timeout, no group concept anyway).
    force_popen = (use_wall_cap or use_audit) and sys.platform != "win32"

    if not use_watchdog and not use_observer and not force_popen:
        # Fast path. subprocess.run kills the child on timeout before raising.
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, env=env,
                timeout=wall_cap_s if use_wall_cap else None,
            )
            return proc.returncode, proc.stdout, proc.stderr, None, None, 0, 0.0
        except subprocess.TimeoutExpired as e:
            out = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode(errors="replace") if e.stdout else "")
            err = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode(errors="replace") if e.stderr else "")
            return -9, out, err, None, float(wall_cap_s), 0, 0.0

    # Popen path — used when either mem watchdog or side_observer is active.
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,  # makes the child its own pgid
    )

    # Side observer (e.g. native sample wrapper) attaches to the pipeline
    # PID right after spawn so it captures from t=0. attach() is best-
    # effort — failures must not prevent the pipeline from running.
    if use_observer:
        try:
            side_observer.attach(proc.pid)
        except Exception as e:
            print(f"[runner] side_observer.attach failed: {e}",
                  flush=True, file=sys.stderr)

    killed_peak_gb: float | None = None
    peak_procs_seen: int = 0
    cpu_sec_seen: float = 0.0
    watcher = None
    stop = threading.Event()

    if use_watchdog or use_audit:
        # Lazy-imported so `runner.py` doesn't pull dispatch_resources unless
        # a caller actually opts into the watchdog.
        from zyme.dispatch.resources import process_group_stats

        cap_mb = (mem_cap_gb * 1024.0) if use_watchdog else float("inf")
        pgid = proc.pid  # start_new_session ⇒ pgid == pid

        # Minimum consecutive over-budget polls before we record a
        # peak_procs violation.  Filters transient child processes such as
        # Rcpp::sourceCpp → clang++ → ld that live <5 s and are unrelated
        # to the pipeline's computational threading.  With the default 2 s
        # poll interval, 3 consecutive polls ≈ 6 s sustained — long enough
        # to catch real worker pools, short enough to not miss them.
        _PROC_SUSTAIN_POLLS = 3

        def _watch():
            nonlocal killed_peak_gb, peak_procs_seen, cpu_sec_seen
            peak_mb_seen = 0.0
            consec_over = 0          # consecutive polls with n_procs over budget
            consec_peak = 0          # n_procs value during the sustained run
            budget_thresh = 0
            if use_audit and thread_budget is not None:
                budget_thresh = max(int(thread_budget), 1) + 1  # +1 for parent
            while not stop.is_set():
                rss_mb, n_procs, cpu_sec_total = process_group_stats(pgid)
                if n_procs == 0:
                    return
                if rss_mb > peak_mb_seen:
                    peak_mb_seen = rss_mb
                # Only count sustained process-count spikes toward the
                # thread budget.  Transient compiler children (sourceCpp,
                # cython) spike n_procs for 1-2 polls then vanish.
                if budget_thresh > 0 and n_procs > budget_thresh:
                    consec_over += 1
                    consec_peak = max(consec_peak, n_procs)
                    if consec_over >= _PROC_SUSTAIN_POLLS:
                        peak_procs_seen = max(peak_procs_seen, consec_peak)
                else:
                    consec_over = 0
                    consec_peak = 0
                if cpu_sec_total > cpu_sec_seen:
                    cpu_sec_seen = cpu_sec_total
                if rss_mb > cap_mb:
                    killed_peak_gb = peak_mb_seen / 1024.0
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    return
                stop.wait(poll_s)

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()

    timed_out_at_s: float | None = None
    try:
        try:
            stdout, stderr = proc.communicate(
                timeout=wall_cap_s if use_wall_cap else None
            )
        except subprocess.TimeoutExpired:
            timed_out_at_s = float(wall_cap_s)
            # Kill the whole group so forked workers don't outlive the parent.
            killed = False
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                killed = True
            except (AttributeError, ProcessLookupError, PermissionError, OSError):
                pass
            if not killed:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            # Drain whatever buffered output we got before the kill.
            try:
                stdout, stderr = proc.communicate()
            except Exception:
                stdout, stderr = "", ""
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=poll_s + 1.0)
        if use_observer:
            try:
                side_observer.detach()
            except Exception as e:
                print(f"[runner] side_observer.detach failed: {e}",
                      flush=True, file=sys.stderr)
    return (proc.returncode, stdout, stderr, killed_peak_gb,
            timed_out_at_s, peak_procs_seen, cpu_sec_seen)


def run_task(task_dir: Path, dataset_entry=None, extra_env=None,
             mem_cap_gb: float | None = None,
             thread: int | None = None,
             mode=None,
             skip_evaluate: bool = False,
             pipeline_python_args: list | None = None,
             side_observer=None) -> str:
    """Run pipeline/run.{py,R} then evaluate.{py,R}. Return aggregated log content.

    Args:
        task_dir: Path to the task directory.
        dataset_entry: Optional dict {tier, name, path} from parse_datasets().
            If provided, sets ZYME_TIER + ZYME_DATA_PATH env vars so
            pipeline/run + evaluate can read tier-specific paths.
        extra_env: Optional dict of additional env vars to inject into the
            subprocess. Used by `zyme verify` to drive the thread axis.
        mem_cap_gb: Optional per-subprocess memory cap. When set (>0), spawns
            a watchdog thread that polls the child's process-group RSS and
            SIGKILLs the group if it exceeds the cap. Used by `zyme verify`
            to turn a host-killing OOM thrash into a clean per-cell crash.
            None / 0 disables. POSIX-only (Windows ignores).
        thread: Optional thread count. When set, forwarded to the subprocess
            as ZYME_THREADS (and OMP / OPENBLAS / MKL caps for libraries
            that pick those up).
        mode: Accepted-but-ignored for V1/K1 callsite compatibility. K2
            dropped the mode axis.
        skip_evaluate: When True, skip the evaluate phase entirely. Used by
            `zyme profile` — profiling is a diagnostic; correctness check
            against reference_output isn't needed for hotspot identification.
        pipeline_python_args: Optional list of args inserted between the
            python interpreter and "run.py" for Python tasks. Used by
            `zyme profile --backend full` to wrap the pipeline in scalene
            (e.g. ["-m", "scalene", "--outfile=...", "--json"]). Ignored
            for R tasks.
        side_observer: Optional object with attach(pid)/detach() methods.
            Called after the pipeline subprocess is spawned (with its PID)
            and after it completes. Used by `zyme profile --backend native`
            to spawn macOS sample(1) against the running process tree.
            Only the pipeline phase is observed (not evaluate).

    The returned string contains all stdout/stderr from both steps + a final
    summary block in `key: value` format that zyme.utils.parse_log can consume.
    """
    lang = detect_lang(task_dir)
    task_yaml = task_dir / "task.yaml"
    executor = parse_executor(task_yaml)
    declared_metrics = {m["name"] for m in parse_metrics(task_yaml)}
    log_parts = []
    log_parts.append(f"=== run_task: {task_dir} ===")
    log_parts.append(f"Lang:  {lang}")
    if dataset_entry:
        log_parts.append(
            f"Tier:  {dataset_entry.get('tier', '?')}  "
            f"({dataset_entry.get('name', '?')})"
        )
    if executor:
        log_parts.append(f"Exec:  {executor}")
    if extra_env:
        log_parts.append(f"Env:   {' '.join(f'{k}={v}' for k, v in sorted(extra_env.items()))}")

    # Reference output dir resolution. K2: per-tier nested layout
    # (`reference_outputs/<tier>/`) by default, with backward-compat
    # fallback to the historical flat layout (`reference_output_<tier>/`).
    tier = dataset_entry["tier"] if dataset_entry else None
    ref_dir = task_dir / "reference_output"
    if tier and (task_dir / "reference_outputs" / tier).exists():
        ref_dir = task_dir / "reference_outputs" / tier
    elif tier and (task_dir / f"reference_output_{tier}").exists():
        ref_dir = task_dir / f"reference_output_{tier}"
    if not ref_dir.exists() or not any(ref_dir.iterdir()):
        log_parts.append(f"\nERROR: reference_output empty in {task_dir}")
        log_parts.append(f"Run reference.{lang} once to populate it.")
        log_parts.append(_summary(speed=0.0, peak=0.0, status="crash"))
        return "\n".join(log_parts)

    # Build subprocess env with tier info passed through.
    env = os.environ.copy()
    if dataset_entry:
        env["ZYME_TIER"] = dataset_entry.get("tier", "")
        # Resolve dataset path: pipeline runs with cwd=pipeline_dir, but
        # task.yaml paths are relative to task_dir (e.g. ./data/tier_tiny.h5ad).
        # Without this, relative paths break inside the subprocess.
        raw_path = dataset_entry.get("path", "")
        if raw_path and not os.path.isabs(raw_path):
            raw_path = str((task_dir / raw_path).resolve())
        env["ZYME_DATA_PATH"] = raw_path
        env["ZYME_DATASET_NAME"] = dataset_entry.get("name", "")
        env["ZYME_REFERENCE_DIR"] = str(ref_dir)
    if extra_env:
        env.update(extra_env)
    if thread is not None:
        env["ZYME_THREADS"] = str(thread)
        # Pin BLAS/OMP/MKL caps the same way `baseline reference` does. Without
        # this, `init-check --parity` / `zyme run` / `verify` run with BLAS at
        # auto-detected core count while the baseline was recorded under a
        # pinned cap — multithreaded BLAS reductions aren't bit-reproducible,
        # so any numpy/scipy deterministic target sees parity drift ~1e-4 and
        # a "correct" round-0 pipeline fails parity. See baseline.py:680.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS"):
            env[var] = str(thread)

    # === Pipeline ===
    pipeline_dir = task_dir / "pipeline"
    if lang == "py":
        py_bin = _resolve_python(executor["python"]) if executor.get("python") else sys.executable
        if pipeline_python_args:
            run_cmd = [py_bin, *pipeline_python_args, "run.py"]
        else:
            run_cmd = [py_bin, "run.py"]
    else:
        rscript_bin = executor.get("rscript", "Rscript")
        run_cmd = [rscript_bin, "run.R"]

    # Per-tier wall cap: catches runaway O(n²) regressions. Skipped under
    # `zyme profile` (skip_evaluate=True) since samplers add 2-5× overhead.
    wall_cap_s = None if skip_evaluate else _wall_cap_for_tier(tier)

    log_parts.append(f"\n--- pipeline/{' '.join(run_cmd)} ---")
    t0 = time.perf_counter()
    # audit_threads=True whenever zyme injected a thread budget — the
    # Popen-path stats poller then collects peak_procs / group_cpu_sec so
    # we can fail-this-run if the pipeline ran past ZYME_THREADS (the
    # unfair-baseline pattern: candidate uses N cores while baseline used 1).
    audit_threads = thread is not None
    rc, stdout, stderr, killed_peak_gb, timed_out_at_s, peak_procs, group_cpu_sec = _run_with_watchdog(
        run_cmd, cwd=pipeline_dir, env=env, mem_cap_gb=mem_cap_gb,
        side_observer=side_observer, wall_cap_s=wall_cap_s,
        audit_threads=audit_threads,
        thread_budget=thread,
    )
    wall_elapsed = time.perf_counter() - t0
    log_parts.append(stdout)
    if stderr:
        log_parts.append(stderr)

    if rc != 0:
        if timed_out_at_s is not None:
            crash_msg = (
                f"TIMEOUT — exceeded wall cap {timed_out_at_s:.0f}s on tier={tier} "
                f"(pipeline phase). Often an O(n²) regression in a hot loop."
            )
            signal_hint = (
                f"SIGKILL sent by zyme's per-tier wall cap (tier={tier} cap="
                f"{timed_out_at_s:.0f}s); investigate algorithmic complexity."
            )
        elif killed_peak_gb is not None:
            crash_msg = (
                f"killed-by-mem-watchdog (peak {killed_peak_gb:.1f} GB > "
                f"cap {mem_cap_gb:.1f} GB; pipeline phase)"
            )
            signal_hint = (
                "SIGKILL sent by zyme's --mem-cap-gb watchdog to prevent host thrash; "
                "raise --mem-cap-gb on a larger machine, or shrink the tier"
            )
        else:
            crash_msg = _extract_crash_msg(stderr)
            signal_hint = _explain_returncode(rc)
        log_parts.append(f"\n--- CRASH: pipeline exited {rc} ---")
        if signal_hint:
            log_parts.append(f"exit_hint:        {signal_hint}")
        if crash_msg:
            log_parts.append(f"crash_msg:        {crash_msg}")
        log_parts.append(_summary(speed=0.0, peak=0.0, status="crash"))
        return "\n".join(log_parts)

    pipeline_stdout = stdout
    # Prefer internal timing printed by pipeline; fall back to wall clock.
    # Warn loudly on the fallback path — wall_elapsed includes interpreter
    # startup + library import (typically 5-10s), which masquerades as a
    # regression if pipeline simply forgot to call emit_summary.
    speed_raw = _try_extract_float(pipeline_stdout, "speed_sec")
    if speed_raw is None:
        speed_sec = wall_elapsed
        log_parts.append(
            f"\n[warn] pipeline did not print 'speed_sec:' — using wall_elapsed="
            f"{wall_elapsed:.3f}s as the round's speed (this includes interpreter "
            f"startup + library import, typically 5-10s; not comparable to other "
            f"rounds). Add `emit_summary(speed_sec=elapsed, peak_mb=...)` at the "
            f"end of pipeline/run to fix."
        )
    else:
        speed_sec = speed_raw
        if speed_sec < 0.1:
            log_parts.append(
                f"\n[warn] speed_sec={speed_sec*1000:.1f}ms is suspiciously fast "
                f"(< 100ms). Common causes: the override patched the function to "
                f"a no-op; the timed region is wrong (timer wraps too little); "
                f"or the dataset loaded was a stub. Verify before recording as a win."
            )

    peak_mb = _extract_float(pipeline_stdout, "peak_mb", default=0.0)
    cpu_sec = _try_extract_float(pipeline_stdout, "cpu_sec")  # optional; None if missing

    # Thread budget audit. Two distinct signals from the process-group poller,
    # mapping onto iterate.md §Parallelism's case A / case B split:
    #
    #   - peak_procs > budget + 1 → fork-based parallelism (joblib loky,
    #     multiprocessing.Pool, mclapply). Upstream's worker-pool knobs
    #     (typically declared in task.yaml::upstream_parallelism) drive this
    #     path. Pipeline burning more workers than the declared budget means
    #     case A without re-baseline: speedup_pct vs the thread=N baseline
    #     is apples-to-oranges. REJECT and point the agent at re-baseline.
    #
    #   - cpu/wall > budget * 1.3, with peak_procs ≤ budget + 1 → in-process
    #     parallelism (numba prange, OpenMP, BLAS thread pool). Upstream
    #     rarely exposes these as knobs, so this is almost always case B:
    #     a parallel layer the agent ADDED that upstream lacks. Per the
    #     iterate prompt, case B is a legitimate contribution — keep against
    #     upstream's actual-default baseline and document the parallel layer
    #     in memory/discoveries.md. WARN, don't reject.
    #
    # The asymmetry is deliberate (iterate.md:128): upstream users get what
    # upstream gives them; a parallel layer we add IS part of our contribution.
    if thread is not None and audit_threads:
        budget = max(int(thread), 1)
        cpu_ratio = (group_cpu_sec / wall_elapsed) if wall_elapsed > 0 else 0.0
        proc_violation = peak_procs > budget + 1   # +1 for the parent itself
        cpu_violation = cpu_ratio > budget * 1.3   # 1.3 slack for GC / brief BLAS bursts
        tier_hint = tier or "<tier>"

        if proc_violation:
            crash_msg = (
                f"thread budget violation (fork workers) — ZYME_THREADS={budget} "
                f"but pipeline spawned peak_procs={peak_procs}. This is the "
                f"case A pattern in iterate.md §Parallelism: an upstream "
                f"worker-pool knob (likely in task.yaml::upstream_parallelism) "
                f"driven past the declared budget without re-baseline, so "
                f"speedup_pct is no longer apples-to-apples vs the thread="
                f"{budget} baseline. Re-record at the new threading regime: "
                f"`zyme baseline reference --tier {tier_hint} --reps 3 --force`, "
                f"or revert the threading change."
            )
            log_parts.append("\n--- REJECT: thread budget violation ---")
            log_parts.append(f"crash_msg:        {crash_msg}")
            log_parts.append(_summary(speed=0.0, peak=peak_mb,
                                      cpu_sec=cpu_sec, status="crash"))
            return "\n".join(log_parts)

        if cpu_violation:
            log_parts.append(
                f"\n[warn] in-process threads exceeded ZYME_THREADS={budget} "
                f"(cpu/wall={cpu_ratio:.2f}, peak_procs={peak_procs}). Treated "
                f"as case B in iterate.md §Parallelism — a parallel layer not "
                f"exposed by upstream (numba prange / OpenMP / BLAS) — and "
                f"kept as a legitimate contribution; baseline stays at "
                f"upstream's actual default. Document the added parallel "
                f"layer in memory/discoveries.md so the headline number "
                f"isn't read as pure-algorithm. If the engaged knob is "
                f"actually in task.yaml::upstream_parallelism (case A), "
                f"re-baseline before claiming the speedup: "
                f"`zyme baseline reference --tier {tier_hint} --reps 3 --force`."
            )

    # `zyme profile` short-circuit: profiling is diagnostic; correctness vs
    # reference_output isn't needed to identify hotspots, and skipping
    # evaluate keeps a profile run a clean diagnostic primitive (no side
    # effects on results.tsv / git, no extra wall time).
    if skip_evaluate:
        log_parts.append("\n[skip_evaluate] profile mode — evaluate phase skipped")
        log_parts.append(_summary(speed=speed_sec, peak=peak_mb,
                                  cpu_sec=cpu_sec, status="ok"))
        return "\n".join(log_parts)

    # === Evaluate ===
    # cwd=task_dir (not pipeline_dir): some evaluate.R files fall back to
    # SCRIPT_DIR="." when sys.frame(1)$ofile errors under Rscript. Running
    # from task_dir makes that fallback resolve correctly; correct .R/.py
    # evaluates that derive their own script dir are unaffected.
    if lang == "py":
        eval_cmd = [py_bin, "evaluate.py"]
    else:
        eval_cmd = [rscript_bin, "evaluate.R"]

    log_parts.append(f"\n--- {' '.join(eval_cmd)} ---")
    rc, eval_stdout, eval_stderr, killed_peak_gb, timed_out_at_s, _, _ = _run_with_watchdog(
        eval_cmd, cwd=task_dir, env=env, mem_cap_gb=mem_cap_gb,
        wall_cap_s=wall_cap_s,
    )
    log_parts.append(eval_stdout)
    if eval_stderr:
        log_parts.append(eval_stderr)

    if rc != 0:
        if timed_out_at_s is not None:
            crash_msg = (
                f"TIMEOUT — exceeded wall cap {timed_out_at_s:.0f}s on tier={tier} "
                f"(evaluate phase). Likely a hang or O(n²) in the metric code."
            )
            signal_hint = (
                f"SIGKILL sent by zyme's per-tier wall cap (tier={tier} cap="
                f"{timed_out_at_s:.0f}s)"
            )
        elif killed_peak_gb is not None:
            crash_msg = (
                f"killed-by-mem-watchdog (peak {killed_peak_gb:.1f} GB > "
                f"cap {mem_cap_gb:.1f} GB; evaluate phase)"
            )
            signal_hint = (
                "SIGKILL sent by zyme's --mem-cap-gb watchdog to prevent host thrash; "
                "raise --mem-cap-gb on a larger machine, or shrink the tier"
            )
        else:
            crash_msg = _extract_crash_msg(eval_stderr)
            signal_hint = _explain_returncode(rc)
        log_parts.append(f"\n--- CRASH: evaluate exited {rc} ---")
        if signal_hint:
            log_parts.append(f"exit_hint:        {signal_hint}")
        if crash_msg:
            log_parts.append(f"crash_msg:        {crash_msg}")
        log_parts.append(_summary(speed=speed_sec, peak=peak_mb,
                                  cpu_sec=cpu_sec, status="crash"))
        return "\n".join(log_parts)

    # Extract metric: value lines from evaluate stdout. Strict allowlist: only
    # accept keys declared in task.yaml's `metrics:` block — otherwise debug
    # prints like `print("DEBUG_PATH: /foo")` from evaluate.{py,R} get scraped
    # as metrics and pollute results.tsv. Falls back to the legacy SKIP_KEYS
    # blocklist only when task.yaml declares no metrics (init-time scaffolds).
    # Exception: framework-emitted auto_structure metrics (suffix-matched) are
    # always allowed through — agents don't declare them per-slot, so the
    # allowlist would otherwise swallow real structural failures.
    metric_lines = []
    seen = set()
    for line in eval_stdout.splitlines():
        m = re.match(r"^([A-Za-z0-9_.]+):\s+(.+?)\s*$", line)
        if not m:
            continue
        key = m.group(1)
        is_auto_struct = key.endswith(_AUTO_STRUCT_SUFFIXES)
        if declared_metrics and not is_auto_struct:
            if key not in declared_metrics or key in seen:
                continue
        else:
            if key in _SKIP_KEYS or key in seen:
                continue
        seen.add(key)
        metric_lines.append(f"{key}: {m.group(2)}")

    log_parts.append(_summary(speed=speed_sec, peak=peak_mb, cpu_sec=cpu_sec,
                              status="ok", metric_lines=metric_lines))
    return "\n".join(log_parts)


def dryrun_task(task_dir: Path, dataset_entry=None, extra_env=None):
    """Run pipeline/run only — skip evaluate, no results.tsv, no commit.

    Used by `zyme dryrun` for syntactic / compilation / sanity checks
    where the agent wants to see if pipeline executes at all without
    burning a decision round.

    Returns (returncode, stdout, stderr, wall_elapsed). Caller prints.
    """
    lang = detect_lang(task_dir)
    task_yaml = task_dir / "task.yaml"
    executor = parse_executor(task_yaml)

    tier = dataset_entry["tier"] if dataset_entry else None
    ref_dir = task_dir / "reference_output"
    if tier and (task_dir / "reference_outputs" / tier).exists():
        ref_dir = task_dir / "reference_outputs" / tier
    elif tier and (task_dir / f"reference_output_{tier}").exists():
        ref_dir = task_dir / f"reference_output_{tier}"

    env = os.environ.copy()
    if dataset_entry:
        env["ZYME_TIER"] = dataset_entry.get("tier", "")
        env["ZYME_DATA_PATH"] = dataset_entry.get("path", "")
        env["ZYME_DATASET_NAME"] = dataset_entry.get("name", "")
        env["ZYME_REFERENCE_DIR"] = str(ref_dir)
    if extra_env:
        env.update(extra_env)

    pipeline_dir = task_dir / "pipeline"
    if lang == "py":
        py_bin = _resolve_python(executor["python"]) if executor.get("python") else sys.executable
        run_cmd = [py_bin, "run.py"]
    else:
        rscript_bin = executor.get("rscript", "Rscript")
        run_cmd = [rscript_bin, "run.R"]

    wall_cap_s = _wall_cap_for_tier(tier)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            run_cmd, cwd=pipeline_dir, capture_output=True, text=True, env=env,
            timeout=wall_cap_s,
        )
        wall_elapsed = time.perf_counter() - t0
        return proc.returncode, proc.stdout, proc.stderr, wall_elapsed
    except subprocess.TimeoutExpired as e:
        wall_elapsed = time.perf_counter() - t0
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode(errors="replace") if e.stdout else "")
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode(errors="replace") if e.stderr else "")
        err = (err or "") + (
            f"\n[zyme] dryrun killed: exceeded wall cap {wall_cap_s:.0f}s on tier={tier}.\n"
        )
        return -9, out, err, wall_elapsed


def _try_extract_float(stdout: str, key: str):
    """Returns the parsed float, or None if `key:` is missing / unparseable."""
    pattern = rf"^{re.escape(key)}:\s+(.+?)\s*$"
    for line in stdout.splitlines():
        m = re.match(pattern, line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _extract_float(stdout: str, key: str, default=0.0) -> float:
    """Backward-compatible wrapper: returns default when key is missing.

    Most callers want a float either way; the wall-fallback warning path uses
    _try_extract_float directly so it can distinguish "missing" from "0.0".
    """
    v = _try_extract_float(stdout, key)
    return default if v is None else v


def _summary(speed: float, peak: float, status: str,
             cpu_sec=None, metric_lines=None) -> str:
    lines = ["", "---",
             f"speed_sec:        {speed:.3f}",
             f"peak_mb:          {peak:.1f}"]
    if cpu_sec is not None:
        lines.append(f"cpu_sec:          {cpu_sec:.3f}")
    lines.append(f"status:           {status}")
    if metric_lines:
        lines.extend(metric_lines)
    lines.append("---")
    return "\n".join(lines)
