"""native.py — backend=native: macOS sample wrapper.

The motivation: in-process profilers (cProfile / Rprof / Scalene's CPU axis /
profvis) all stop at native code boundaries. cProfile sees `np.linalg.svd`
as one frame; Rprof sees `.Call()` and goes blind. For autozyme tasks where
the hot path lives inside Rcpp / OpenBLAS / Cython / Numba / FORTRAN, those
profilers are structurally limited.

This backend uses macOS's built-in `sample` command — a kernel-supported
sampling profiler that:
  - sees C/C++/Rust frames with full symbolication (atos automatic)
  - sees BLAS / OpenBLAS / Accelerate / vendor library internals
  - works without code changes to the pipeline (external observer)

What it does NOT do:
  - sample(1) only attaches to one PID; it does not natively follow forks.
    For multi-process workloads (mclapply / multiprocessing.Pool), we work
    around this with a child-watchdog thread (see NativeSampler._watch_children).
  - Linux is not supported — see the module-level `is_supported()` check.
    Linux equivalent would wrap `perf record`; left for a future iteration.
  - sample reports SAMPLE COUNTS, not time. We expose counts as `samples`
    and percentages in `raw`; absolute time is computed from sampling
    interval × count.

Parsing approach: we read the "Sort by top of stack, same collapsed (when >= 5)"
section from each sample output file (parent + each forked child), aggregate
counts per (function, library) across files, and surface the top-N. We
intentionally do NOT parse the call graph — for autozyme's "where's the
hot work" question, the flat self-time top-N is what the agent needs.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

POLL_INTERVAL_S = 0.05  # child-PID polling cadence — short to catch short-lived workers
INITIAL_ATTACH_DELAY_S = 0.03  # let exec() settle without missing 100ms workers
SAMPLE_BIN = "/usr/bin/sample"
PGREP_BIN = "/usr/bin/pgrep"
SAMPLE_INTERVAL_MS = 1  # 1ms = sample's default
TOP_N = 15


# Frame names that represent the process being IDLE (waiting on a lock,
# sleeping in a syscall, waiting for a worker, reading the wall clock).
# These aren't actionable hotspots — they tell you the program is blocked,
# not where the actual work lives. Sample(1) gives wall-clock samples, so
# without this filter every multi-threaded / multi-process task's top-N
# would be dominated by `__psynch_cvwait` / `__semwait_signal` / `__workq_kernreturn`.
#
# We don't drop these — they're surfaced in profile_data.idle_frames so the
# agent can see total wait time. They just don't crowd out the active top-N.
_IDLE_FRAME_PATTERNS = [
    re.compile(r"^__psynch_"),                # pthread cond/mutex
    re.compile(r"^_?pthread_cond_wait"),      # pthread condition wait
    re.compile(r"^_?pthread_mutex_"),         # pthread mutex waits/locks
    re.compile(r"^__ulock_wait"),             # Darwin userspace locks
    re.compile(r"^__semwait_signal"),         # semaphore
    re.compile(r"^_?sem_wait$"),              # POSIX semaphore wait
    re.compile(r"^semaphore_wait"),           # Mach semaphore wait
    re.compile(r"^__workq_kernreturn"),       # libdispatch worker idle
    re.compile(r"^__select$"),                # select(2)
    re.compile(r"^mach_msg2?_trap"),          # mach IPC
    re.compile(r"^mach_absolute_time$"),      # timing query
    re.compile(r"^__gettimeofday$"),
    re.compile(r".*__commpage_gettimeofday"),
    re.compile(r"^gettimeofday$"),
    re.compile(r"^_?wait4(_nocancel)?$"),     # parent waiting on child
    re.compile(r"^__wait4(_nocancel)?$"),
    re.compile(r"^__read(_nocancel)?$"),      # blocking read
    re.compile(r"^read$"),
    re.compile(r"^__write(_nocancel)?$"),     # blocking write
    re.compile(r"^write$"),
    re.compile(r"^__open(_nocancel)?$"),      # filesystem metadata/open
    re.compile(r"^open$"),
    re.compile(r"^_?stat(64)?$"),             # filesystem metadata
    re.compile(r"^_?lstat(64)?$"),
    re.compile(r"^_?fstat(64)?$"),
    re.compile(r"^_?access$"),
    re.compile(r"^__fcntl$"),                 # descriptor control, often dyld/IO
    re.compile(r"^fcntl$"),
    re.compile(r"^madvise$"),                 # VM paging hint, not user hot path
    re.compile(r"^kevent$"),                  # kqueue wait
    re.compile(r"^poll$"),                    # poll(2)
    re.compile(r"^read_child_ci$"),           # parallel.so worker IPC read
    re.compile(r"^selectChildren$"),          # parallel.so worker dispatch
    # Mach kernel scheduling: thread giving up the CPU. Surfaces when
    # OpenMP / threading-heavy code has more workers than active cores.
    re.compile(r"^swtch_pri"),
    re.compile(r"^thread_switch"),
    # OpenMP idle/wait — `libomp.dylib:kmp_flag_*::wait` etc. Workers in
    # an OpenMP team spin on these between work units. Common in NumPy/
    # SciPy-via-OpenBLAS and explicit OpenMP code (pocketfft, etc).
    re.compile(r"^kmp_flag_"),                # __kmp_flag_<>::wait, ::notdone_check
    re.compile(r"^__kmp_(wait|join|fork|barrier|yield|hyper|flag)"),
    re.compile(r"^kmp_yield"),
]


def _is_idle_frame(func: str) -> bool:
    return any(p.match(func) for p in _IDLE_FRAME_PATTERNS)


# ---------------------------------------------------------------------------
# Capability check
# ---------------------------------------------------------------------------

def is_supported() -> tuple[bool, str]:
    """Return (ok, reason). Used by backends.resolve()."""
    if sys.platform != "darwin":
        return False, ("backend=native is currently macOS-only "
                       "(wraps /usr/bin/sample). Linux equivalent (perf) is "
                       "not implemented yet.")
    if not os.path.exists(SAMPLE_BIN):
        return False, f"{SAMPLE_BIN} not found — macOS native sampler missing"
    if not os.path.exists(PGREP_BIN):
        return False, f"{PGREP_BIN} not found — required for child-process tracking"
    return True, ""


# ---------------------------------------------------------------------------
# Side-observer: NativeSampler
# ---------------------------------------------------------------------------

class NativeSampler:
    """Spawn sample(1) against a process tree; aggregate output on detach.

    Used by runner via the `side_observer` hook:
        sampler = NativeSampler(out_dir=profile_dir)
        sampler.attach(pipeline_proc.pid)   # spawn parent sample + watchdog
        ... pipeline runs ...
        sampler.detach()                    # stop watchdog, wait for samples
        data = native.parse_and_normalize(..., sampler=sampler)

    Files written to out_dir:
        native_sample_<pid>.txt    one per sampled process (parent + forked children)
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.parent_pid: int | None = None
        self._sample_procs: dict[int, subprocess.Popen] = {}
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._lock = threading.Lock()
        # Wipe stale outputs from a prior run so collect() returns only this run's data.
        for pattern in ("native_sample_*.txt", "native_sample_*.stderr"):
            for stale in out_dir.glob(pattern):
                try:
                    stale.unlink()
                except OSError:
                    pass

    def attach(self, pid: int) -> None:
        self.parent_pid = pid
        # Spawn the parent sample inside the watchdog thread (non-blocking
        # to runner) with a small initial settle delay. Attaching before
        # the child has finished exec()ing into its real image (e.g.
        # Rscript → R) produces an empty sample output silently — caught by
        # debugging test_fgsea where the parent PID's sample file came out
        # empty while children's worked.
        self._watchdog = threading.Thread(target=self._watch_loop,
                                          daemon=True, name="zyme-native-watchdog")
        self._watchdog.start()

    def detach(self) -> None:
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=2.0)
        # Two cases at this point:
        #   (a) sample's target has already exited → sample wrote output and
        #       exited cleanly. proc.wait() returns immediately.
        #   (b) sample is still attached to a live target (the parent
        #       pipeline process if it's long-lived, or a child that hasn't
        #       exited yet). We need sample to FLUSH its accumulated samples
        #       and exit. SIGTERM does NOT make sample flush — SIGINT does.
        #       (sample treats SIGINT like Ctrl+C: writes output, exits.)
        deadline = time.monotonic() + 6.0
        for pid, proc in list(self._sample_procs.items()):
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGINT)
                except (ProcessLookupError, OSError):
                    pass
        # Now wait. SIGINT'd samples should exit within a second or two.
        for pid, proc in list(self._sample_procs.items()):
            remaining = max(0.5, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                # Last resort: SIGKILL (we lose this proc's output but
                # don't hang).
                proc.kill()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

    def collect(self) -> list[Path]:
        """Return the list of native_sample_*.txt files for this run."""
        return sorted(self.out_dir.glob("native_sample_*.txt"))

    def _spawn_sample(self, pid: int) -> None:
        out_path = self.out_dir / f"native_sample_{pid}.txt"
        # Capture stderr to a per-pid log so attach failures (target dead,
        # permission, etc.) are recoverable post-hoc. stderr is intentionally
        # NOT routed to DEVNULL — silent sample failures were the recurring
        # debug-pain pattern when the parent process couldn't be attached.
        err_path = self.out_dir / f"native_sample_{pid}.stderr"
        try:
            proc = subprocess.Popen(
                # Long duration; sample exits when target exits. -mayDie
                # tolerates the target disappearing mid-sample.
                [SAMPLE_BIN, str(pid), "86400", "-mayDie",
                 "-file", str(out_path)],
                stdout=subprocess.DEVNULL,
                stderr=open(err_path, "w"),
            )
        except OSError as e:
            print(f"[native] failed to spawn sample for pid={pid}: {e}",
                  file=sys.stderr, flush=True)
            return
        with self._lock:
            self._sample_procs[pid] = proc

    def _watch_loop(self) -> None:
        """Poll the pipeline process group and attach sample(1) to new PIDs.

        We discover both direct children (`pgrep -P`) and any process in the
        pipeline process group (`pgrep -g`). The runner starts the pipeline in
        a new session, so pgid==parent pid; group polling catches grandchildren
        and late fork workers without task-specific hooks.
        """
        # Initial settle: let the child finish exec()ing into its real
        # interpreter image. Keep this short; fast-fork R tasks can create
        # workers within the first 100ms.
        if self._stop.wait(INITIAL_ATTACH_DELAY_S):
            return

        seen: set[int] = set()
        while not self._stop.is_set():
            for pid in self._discover_candidate_pids():
                if pid in seen:
                    continue
                seen.add(pid)
                self._spawn_sample(pid)
            self._stop.wait(POLL_INTERVAL_S)

    def _discover_candidate_pids(self) -> list[int]:
        if self.parent_pid is None:
            return []
        pids = {self.parent_pid}
        pids.update(_pgrep_pids(["-P", str(self.parent_pid)]))
        # start_new_session=True in runner makes the pipeline pgid equal to
        # the parent pid. This catches grandchildren as well as direct forks.
        pids.update(_pgrep_pids(["-g", str(self.parent_pid)]))
        return sorted(pid for pid in pids if pid > 0)


def _pgrep_pids(args: list[str]) -> set[int]:
    try:
        result = subprocess.run(
            [PGREP_BIN, *args],
            capture_output=True, text=True, timeout=2.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return set()
    if result.returncode != 0:
        return set()
    pids: set[int] = set()
    for line in result.stdout.split():
        try:
            pids.add(int(line.strip()))
        except ValueError:
            continue
    return pids


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Top-of-stack lines, e.g.:
#         inner_thread  (in libopenblas64_.0.dylib)        9980
# The regex has to tolerate function names containing spaces and special chars.
_TOP_LINE_RE = re.compile(
    r"^\s+(?P<func>.+?)\s+\(in\s+(?P<lib>[^)]+)\)\s+(?P<count>\d+)\s*$"
)


def parse_sample_file(path: Path) -> list[tuple[str, str, int]]:
    """Parse one sample(1) output file's "Sort by top of stack" section.

    Returns: list of (function, library, count) tuples. Empty list on any
    failure (file missing, malformed, no section found) — caller treats
    that as "no samples for this PID".
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    marker = "Sort by top of stack, same collapsed"
    idx = text.find(marker)
    if idx < 0:
        return []
    end = text.find("Binary Images:", idx)
    if end < 0:
        end = len(text)
    section = text[idx:end]

    rows = []
    for line in section.splitlines()[1:]:  # skip the marker line itself
        m = _TOP_LINE_RE.match(line)
        if m:
            rows.append((m.group("func").strip(),
                         m.group("lib").strip(),
                         int(m.group("count"))))
    return rows


def parse_and_normalize(out_dir: Path, sampler: NativeSampler,
                        lang: str, tier: str, hypothesis: str,
                        totals: dict | None = None,
                        artifact_prefix: str = "profile_history/current") -> dict:
    """Aggregate frames across all sample output files; build the
    canonical profile.json dict (matches parsers.normalize schema)."""
    files = sampler.collect()
    aggregated: dict[tuple[str, str], int] = defaultdict(int)
    per_file_counts: list[tuple[str, int]] = []
    for f in files:
        rows = parse_sample_file(f)
        per_file_counts.append((f.name, sum(c for _, _, c in rows)))
        for func, lib, count in rows:
            aggregated[(func, lib)] += count

    total_samples = sum(aggregated.values())

    # Partition into active (real work) vs idle (waits / locks / timing).
    # Without this split, the top-N for any threaded or multi-process task
    # is dominated by `__psynch_cvwait` and friends — wall-clock truth, but
    # not actionable for optimization. Idle counts are reported in notes.
    active = {k: v for k, v in aggregated.items() if not _is_idle_frame(k[0])}
    idle = {k: v for k, v in aggregated.items() if _is_idle_frame(k[0])}
    idle_total = sum(idle.values())

    sorted_frames = sorted(active.items(), key=lambda kv: -kv[1])[:TOP_N]

    # Locate the upstream source tree once so per-hotspot lookups are cheap.
    # Convention: `zyme init` clones target_repo into <task_dir>/upstream_repo/.
    # out_dir is typically <task_dir>/profile_history/<run>/, so go up two.
    upstream_dir = out_dir.parent.parent / "upstream_repo"

    interval_s = SAMPLE_INTERVAL_MS / 1000.0
    active_total = sum(active.values())
    hotspots: list[dict] = []
    for rank, ((func, lib), count) in enumerate(sorted_frames, 1):
        # Two percentages: of total samples (incl idle), and of ACTIVE samples
        # (excl idle). Active-pct is the more useful number for ranking real
        # work — it answers "of the time NOT spent waiting, where did it go?"
        pct_total = (count / total_samples * 100.0) if total_samples > 0 else None
        pct_active = (count / active_total * 100.0) if active_total > 0 else None
        est_self_s = count * interval_s
        source_loc = _locate_func_in_upstream(func, upstream_dir)
        raw_dict = {
            "func": func,
            "lib": lib,
            "samples": count,
            "samples_pct_active": round(pct_active, 4) if pct_active is not None else None,
            "samples_pct_total": round(pct_total, 4) if pct_total is not None else None,
        }
        if source_loc:
            raw_dict["source_location"] = source_loc
        hotspots.append({
            "rank": rank,
            "label": _short_label(lib, func),
            "self_time_s": round(est_self_s, 4),
            "total_time_s": None,
            "self_pct": round(pct_active, 2) if pct_active is not None else None,
            "calls": None,
            "raw": raw_dict,
        })

    # Top idle frames as a separate list (not crowding hotspots) — agent
    # can read this to know HOW the process spent its waiting time.
    sorted_idle = sorted(idle.items(), key=lambda kv: -kv[1])[:5]
    idle_summary = [
        {
            "func": func, "lib": lib, "samples": count,
            "samples_pct_total": round(count / total_samples * 100.0, 2)
                if total_samples > 0 else None,
        }
        for ((func, lib), count) in sorted_idle
    ]

    notes = [
        f"backend=native (macOS sample(1)) interval={SAMPLE_INTERVAL_MS}ms unit=samples",
    ]
    if files:
        n_children = max(0, len(files) - 1)
        notes.append(
            f"aggregated {total_samples:,} samples across {len(files)} processes "
            f"(parent + {n_children} forked child{'ren' if n_children != 1 else ''})"
        )
    else:
        notes.append("no sample output captured — process may have exited too "
                     "quickly for sample to attach (< ~50ms wall time)")
    if total_samples > 0:
        idle_pct = idle_total / total_samples * 100.0
        notes.append(
            f"idle/wait frames filtered out: {idle_total:,} samples "
            f"({idle_pct:.1f}%) — pthread/semaphore/mclapply waits, not actionable. "
            f"hotspots above are ranked by ACTIVE-time share, not wall-clock."
        )
        if idle_summary:
            top_idle_str = ", ".join(
                f"{i['func']}({i['samples_pct_total']}%)" for i in idle_summary[:3]
            )
            notes.append(f"top idle frames: {top_idle_str}")
    notes.append(
        "native frames (BLAS, Cython, Rcpp, FORTRAN) ARE visible in this "
        "backend — that is its raison d'être vs cProfile/Rprof/Scalene-CPU"
    )
    notes.append(
        "self_time_s here is approximate (samples × interval) and aggregates "
        "across all processes; for parallel workloads divide by n_procs to "
        "estimate per-worker wall time"
    )

    return {
        "schema_version": "1",
        "backend": "native",
        "lang": lang,
        "tier": tier,
        "hypothesis": hypothesis,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "totals": totals or {},
        "hotspots": hotspots,
        "actionable_hotspots": [],
        "override_summary": [],
        "override_markers": [],
        "call_chains": [],
        "notes": notes,
        "artifacts": {
            "raw": f"{artifact_prefix}/native_sample_*.txt ({len(files)} files)"
                if files else f"{artifact_prefix}/native_sample_*.txt (0 files — sampler captured nothing)",
        },
    }


def _short_label(lib: str, func: str) -> str:
    """Compact: '<lib_basename>:<func>'. Trim long lib paths and C++ argument
    lists. Mangled Rcpp/Armadillo symbols (e.g.
    `fast_iterate_t_impl(arma::Mat<double> const&, Rcpp::Vector<19, ...> ...)`)
    blow past 400 chars and unwrap the render. Full unmangled signature stays
    in `raw.func`; the label here is for human scanning only.
    """
    lib_short = os.path.basename(lib) if "/" in lib else lib
    paren = func.find("(")
    short_func = (func[:paren] + "(...)") if paren > 0 else func
    return f"{lib_short}:{short_func}"


# Source file extensions worth grepping for a native symbol's definition.
# Covers Rcpp/C++ (.cpp/.cc/.cxx/.h/.hpp), C (.c/.h), Cython (.pyx/.pxd),
# and Fortran (.f/.f90/.F90). Keep tight — broad globs (e.g. *.txt) just
# add noise and slow the search.
_NATIVE_SOURCE_EXTS = (
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh",
    ".pyx", ".pxd", ".f", ".f90", ".F90",
)


_DEFN_LINE_RE = re.compile(
    r"^[A-Za-z_][\w:&<>\*\s]*\s+(?P<name>\w+)\s*\("
)


def _locate_func_in_upstream(func: str, upstream_dir: Path) -> str | None:
    """Best-effort: find `func`'s source location in upstream_dir.

    Returns "<relpath>:<lineno>" relative to upstream_dir if a plausible
    definition is found, else None. Used to make native profile hotspots
    actionable — without this, the agent only sees a symbol name like
    'dmvnrm_arma_fast' and has to manually grep the upstream tree.

    Heuristic: grep for `funcname(` across native source extensions, then
    prefer lines whose content matches a C/C++ definition shape
    (`ReturnType funcname(...`). Rcpp packages typically auto-generate
    `RcppExports.cpp` wrapper calls + one real definition elsewhere — we
    want the latter. Returns None when no definition-shaped match is found.
    """
    if not func or not upstream_dir.exists():
        return None
    bare = func.split("(")[0].strip().rsplit("::", 1)[-1]
    if not bare or len(bare) < 3 or not bare[0].isalpha():
        return None
    try:
        include_flags: list[str] = []
        for ext in _NATIVE_SOURCE_EXTS:
            include_flags.append(f"--include=*{ext}")
        result = subprocess.run(
            ["grep", "-rn", "-E", *include_flags,
             rf"\b{re.escape(bare)}\s*\(", str(upstream_dir)],
            capture_output=True, text=True, timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode not in (0, 1):
        return None

    # Two pools: definition-shaped hits (preferred), and other hits (fallback).
    defn_hits: list[tuple[str, int]] = []
    other_hits: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, lineno_s, content = parts
        try:
            lineno = int(lineno_s)
        except ValueError:
            continue
        stripped = content.lstrip()
        if stripped.startswith(("//", "*", "/*", "#")):
            continue
        m = _DEFN_LINE_RE.match(content)
        if m and m.group("name") == bare and not content.rstrip().endswith(";"):
            # Looks like a definition: `Type funcname(` at column 0 AND
            # not a forward declaration (no trailing semicolon).
            defn_hits.append((path, lineno))
        else:
            other_hits.append((path, lineno))

    chosen = defn_hits[0] if defn_hits else (other_hits[0] if len(other_hits) == 1 else None)
    if not chosen:
        return None
    path, lineno = chosen
    try:
        rel = str(Path(path).resolve().relative_to(upstream_dir.resolve()))
    except ValueError:
        rel = path
    return f"{rel}:{lineno}"
