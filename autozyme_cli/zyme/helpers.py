"""helpers.py — Shared utilities for autozyme tasks (Python).

This file is READ-ONLY during experiments. Tasks import from it in their
pipeline/run.py for install_override / time_it / peak memory tracking.

Convention: pipeline/run.py prints `speed_sec: <float>` and `peak_mb: <float>`
to stdout; the zyme runner greps these for the summary.
"""
import contextlib
import cProfile
import importlib
import io
import os
import platform
import pstats
import sys
import time
import types


# The autozyme-framework directory always has this name. Code that needs
# to locate the framework programmatically (find_framework_root, scan.py,
# etc.) reads it from here. Task-side bootstraps inline the literal string
# because they run before helpers.py is on sys.path.
FRAMEWORK_DIR_NAME = "autozyme-framework"


def find_framework_root(start=None, max_depth=8):
    """Walk up from ``start`` to locate the autozyme-framework root.

    Identifies the framework by a directory named FRAMEWORK_DIR_NAME whose
    ``autozyme_cli/`` child exists — the second check disambiguates against
    a stray directory of the same name and rejects empty trees.

    Args:
        start: starting directory; defaults to this file's directory.
        max_depth: maximum levels to walk up before giving up.

    Returns:
        Absolute realpath (str) to the framework root.

    Raises:
        RuntimeError if no framework root is found within ``max_depth``.
    """
    if start is None:
        start = os.path.dirname(os.path.abspath(__file__))
    cur = os.path.abspath(start)
    for _ in range(max_depth + 1):
        cand = os.path.join(cur, FRAMEWORK_DIR_NAME)
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "autozyme_cli")):
            return os.path.realpath(cand)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise RuntimeError(
        f"{FRAMEWORK_DIR_NAME} not found above {start} (searched {max_depth} levels)"
    )


# Track which overrides have emitted [override active] (one-shot marker
# for `with_override_timing` deprecated path). The full per-call timing
# pipeline that previously fed `[override summary]` lines was removed —
# use `zyme profile` for hit counts and call timing.
_zyme_overrides_seen: set = set()


def record_override_timing(full_name: str, elapsed_s: float) -> None:
    """Deprecated no-op. The override_summary pipeline that consumed this has
    been removed; use `zyme profile` for timing data. Kept so existing task
    code that calls it doesn't error."""
    return None


@contextlib.contextmanager
def with_override_timing(full_name: str):
    """Deprecated: previously fed per-call timing into `[override summary]`.
    Now just emits a one-shot `[override active]` marker (agent sanity check
    that the lexically-shadowing wrapper actually fired) and yields. For
    timing, use `zyme profile` or `time_it()`.

        with with_override_timing("pkg.fast_fn"):
            return fast_fn(...)
    """
    if full_name not in _zyme_overrides_seen:
        print(f"[override active] {full_name}", flush=True)
        _zyme_overrides_seen.add(full_name)
    yield


def install_override(func_name, module_path, new_func, strict_aliases=False):
    """DEPRECATED alias for `patch_namespace`.

    Previously this wrapped `new_func` in a per-call `marker_wrapper` that
    emitted `[override active]` on first call and `[override summary]` on
    every call for the framework's profile pipeline to parse. That pipeline
    has been removed — use `zyme profile` for hit counts and timing.

    New code should call `patch_namespace(func_name, module_path, new_func,
    strict_aliases=...)` directly. This alias preserves the call signature
    so existing pipeline/run.py call sites keep working unmodified; it will
    be removed in a future release.
    """
    return patch_namespace(func_name, module_path, new_func,
                           strict_aliases=strict_aliases)


# Backwards-compat alias — old code still calls inject_override.
# Note the argument order shift: install_override is (name, mod, fn);
# inject_override was (name, fn, mod).
def inject_override(func_name, new_func, module_name="scanpy"):
    return install_override(func_name, module_name, new_func)


def _resolve_module(module_path):
    """Resolve a dotted path that may end in class attribute(s) rather than module."""
    module = sys.modules.get(module_path)
    if module is None:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            module = None
    if module is None:
        parts = module_path.split(".")
        for split in range(len(parts) - 1, 0, -1):
            head, tail = ".".join(parts[:split]), parts[split:]
            try:
                head_mod = sys.modules.get(head) or importlib.import_module(head)
            except ImportError:
                continue
            obj = head_mod
            try:
                for attr in tail:
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            module = obj
            break
    return module


def patch_namespace(func_name, module_path, new_func, strict_aliases=False):
    """Replace `<module_path>.<func_name>` with `new_func`. No instrumentation.

    Same patching semantics as `install_override` (setattr + sys.modules alias
    scan + verify, with strict/lenient alias handling), but does NOT wrap
    `new_func` in `marker_wrapper`. Use when:
      - the override fires inside a hot loop and per-call instrumentation tax
        would dominate;
      - you want per-call timing via `zyme profile` (cProfile) rather than the
        framework's parsed log lines.

    Args:
        func_name: name of the attribute inside the module.
        module_path: dotted module path (or `<importable>.<class>[.<class>...]`).
        new_func: replacement callable.
        strict_aliases: if True, raise (before patching) when the original
            function is bound under other names in already-imported modules
            (those imports bypass the override). Default False patches anyway
            and prints a WARN line.

    Returns: original function (for thin-wrapper / delegate patterns).
    """
    full_name = f"{module_path}.{func_name}"

    module = _resolve_module(module_path)
    if module is None:
        swap_hint = ""
        swapped = _resolve_module(func_name)
        if swapped is not None and hasattr(swapped, module_path.split(".")[-1]):
            swap_hint = (
                f" Did you swap the arguments? Try: "
                f"patch_namespace(\"{module_path}\", \"{func_name}\", ...)"
            )
        raise RuntimeError(
            f"patch_namespace: '{module_path}' is neither an importable "
            f"module nor a class reachable via attribute walk.{swap_hint}"
        )
    if not hasattr(module, func_name):
        swap_hint = ""
        if hasattr(module, module_path.split(".")[-1]):
            swap_hint = (
                f" Did you swap the arguments? Try: "
                f"patch_namespace(\"{module_path}\", \"{func_name}\", ...)"
            )
        raise RuntimeError(
            f"patch_namespace: '{func_name}' not found in '{module_path}'.{swap_hint}"
        )

    original = getattr(module, func_name)

    # Pre-patch alias scan (same logic as install_override).
    stale_aliases = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is module or mod_name.startswith("_"):
            continue
        try:
            v = getattr(mod, func_name, None)
        except Exception:
            continue
        if v is original:
            stale_aliases.append(f"{mod_name}.{func_name}")

    if stale_aliases and strict_aliases:
        shown = stale_aliases[:5]
        more = f" ... and {len(stale_aliases) - 5} more" if len(stale_aliases) > 5 else ""
        raise RuntimeError(
            f"patch_namespace({full_name}): strict_aliases=True but the original "
            f"function is also bound at: {shown}{more}. Those callers will "
            f"bypass the override. Either patch each alias's module too, or import "
            f"`{module_path}` inside your override chain (not at file top) so the "
            f"lookup happens after patch_namespace runs. Pass strict_aliases=False "
            f"to proceed anyway with a WARN."
        )

    setattr(module, func_name, new_func)

    bound = getattr(module, func_name)
    if bound is not new_func:
        raise RuntimeError(
            f"patch_namespace({full_name}) FAILED: attribute did not update"
        )

    if stale_aliases and not strict_aliases:
        patched_aliases = []
        failed_aliases = []
        for alias_path in stale_aliases:
            alias_mod_path, alias_attr = alias_path.rsplit(".", 1)
            alias_mod = sys.modules.get(alias_mod_path)
            if alias_mod is None:
                failed_aliases.append(alias_path)
                continue
            try:
                setattr(alias_mod, alias_attr, new_func)
                patched_aliases.append(alias_path)
            except (AttributeError, TypeError):
                failed_aliases.append(alias_path)
        if patched_aliases:
            print(
                f"[patch_namespace] auto-patched {len(patched_aliases)} alias(es): "
                f"{patched_aliases[:3]}{'...' if len(patched_aliases) > 3 else ''}",
                flush=True,
            )
        if failed_aliases:
            print(
                f"[patch_namespace] WARN: could not auto-patch alias(es): "
                f"{failed_aliases[:3]}{'...' if len(failed_aliases) > 3 else ''} — "
                f"patch them manually or move the override upstream of the fan-out.",
                flush=True,
            )

    print(f"[patch_namespace] {full_name} patched + verified", flush=True)
    return original


class _OverlayDict(dict):
    """dict subclass usable as function __globals__; supports shim overrides
    layered on top of an upstream globals snapshot. `__globals__` must be a
    real dict (not ChainMap), but a subclass works — live updates propagate to
    subsequent function calls."""
    pass


def inline_upstream(qualified_name):
    """Clone an upstream function with an editable shim-globals overlay.

    Args:
        qualified_name: dotted path including the function name
            (e.g. "squidpy.gr.co_occurrence", "MAST.zlm" if MAST were Python).

    Returns:
        A new function object that:
          - shares __code__ with the upstream original;
          - has __globals__ pointing at an _OverlayDict snapshot of upstream
            __globals__ (so subsequent overrides are visible to the clone);
          - exposes a `.shim` attribute pointing at that overlay dict.

    Use:
        my_co = inline_upstream("squidpy.gr.co_occurrence")
        my_co.shim["_co_occurrence_helper"] = fast_helper
        result = my_co(adata, ...)  # uses fast_helper; squidpy untouched

    Coverage: catches LOAD_GLOBAL in the CLONED FUNCTION'S OWN body. Inner
    helpers defined in the upstream module are looked up via THEIR __globals__
    (the unchanged module dict) — for those, use patch_namespace.

    Edge notes:
      - @functools.wraps decorators expose `__wrapped__`; pass that explicitly
        if you want to clone the underlying function (decorator is then lost
        unless you re-apply it).
      - numba @njit exposes `.py_func`; use that to clone the Python source.
      - Closure-captured names go through __closure__ cells, not __globals__,
        so the shim doesn't intercept them. Workaround:
        `fn.__closure__[i].cell_contents = new_fn`.
    """
    mod_path, dot, fn_name = qualified_name.rpartition(".")
    if not dot or not mod_path or not fn_name:
        raise ValueError(
            f"inline_upstream: expected 'module.path.func', got {qualified_name!r}"
        )

    module = _resolve_module(mod_path)
    if module is None:
        raise RuntimeError(f"inline_upstream: cannot resolve module {mod_path!r}")

    if not hasattr(module, fn_name):
        raise RuntimeError(f"inline_upstream: {fn_name!r} not found in {mod_path!r}")

    original = getattr(module, fn_name)
    if not callable(original):
        raise RuntimeError(f"inline_upstream: {qualified_name} is not callable")
    if not (hasattr(original, "__code__") and hasattr(original, "__globals__")):
        raise RuntimeError(
            f"inline_upstream: {qualified_name} is not a regular Python "
            f"function (got {type(original).__name__}). For decorator-wrapped "
            f"targets, pass `<obj>.__wrapped__`; for numba @njit, use "
            f"`<obj>.py_func`; then inline_upstream that name instead."
        )

    overlay = _OverlayDict(original.__globals__)
    cloned = types.FunctionType(
        original.__code__,
        overlay,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    if getattr(original, "__kwdefaults__", None) is not None:
        cloned.__kwdefaults__ = original.__kwdefaults__
    cloned.shim = overlay
    return cloned


def time_it(func, *args, **kwargs):
    """Time a single call. Returns (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


@contextlib.contextmanager
def with_profile():
    """Wrap a block in a profiler when ZYME_PROFILE=1 is set.

    Backend selected by ZYME_PROFILE_BACKEND (default cpu):
      cpu  → cProfile, deterministic, function-level (writes profile.out)
      full → Scalene, line-level CPU+mem+native split. Captured by an
             external `scalene` wrapper around the whole script (engaged
             by `zyme profile --backend full`); helper is a no-op in
             this mode because Scalene's in-process toggle is unreliable
             when the script wasn't launched via the scalene CLI.
      mem  → memray, allocation trace with native attribution. When the
             script is externally wrapped by `memray run`, helper is a
             no-op. Otherwise, helper starts an in-process memray Tracker
             scoped to this `with_profile()` block.

    No-op when ZYME_PROFILE is unset — zero overhead, zero behavior change.

    For sub-block timing inside opaque native code, use with_subprofile().
    For tracemalloc-based memory bisection (independent of backend), use
    with_memprof().
    """
    if not os.environ.get("ZYME_PROFILE", ""):
        yield
        return
    backend = os.environ.get("ZYME_PROFILE_BACKEND", "cpu").lower()
    if backend == "cpu":
        with _with_profile_cpu():
            yield
    elif backend == "full":
        # External scalene wrapper captures the whole script. Scalene's
        # in-process API only works when the script was launched via the
        # scalene CLI, so we don't try to scope it here — `zyme profile
        # --backend full` engages the wrapper and captures everything.
        if not os.environ.get("ZYME_SCALENE_ACTIVE"):
            print("[profile] backend=full requested but scalene wrapper "
                  "not active. Use `zyme profile --backend full` (not "
                  "`ZYME_PROFILE=1 zyme run`).",
                  flush=True, file=sys.stderr)
        else:
            print("[profile active] backend=scalene kind=sampling+alloc "
                  "unit=mixed (external wrapper captures full script)",
                  flush=True, file=sys.stderr)
        yield
    elif backend == "mem":
        if not os.environ.get("ZYME_MEMRAY_ACTIVE"):
            with _with_profile_memray():
                yield
        else:
            print("[profile active] backend=memray kind=allocation_trace "
                  "unit=bytes scope=whole_process "
                  "(external wrapper captures full script)",
                  flush=True, file=sys.stderr)
            yield
    else:
        print(f"[profile] unknown ZYME_PROFILE_BACKEND={backend!r}; "
              f"falling back to cpu", flush=True, file=sys.stderr)
        with _with_profile_cpu():
            yield


@contextlib.contextmanager
def _with_profile_cpu():
    prof_path = _profile_output_path("profile.out")
    pr = cProfile.Profile()
    pr.enable()
    print(f"[profile active] backend=cprofile kind=deterministic "
          f"unit=cpu_seconds writing to {prof_path}",
          flush=True, file=sys.stderr)
    try:
        yield
    finally:
        pr.disable()
        pr.dump_stats(prof_path)
        try:
            buf = io.StringIO()
            stats = pstats.Stats(prof_path, stream=buf)
            stats.sort_stats("tottime").print_stats(20)
            print("[profile] top 20 by tottime (self):", file=sys.stderr)
            print(buf.getvalue(), file=sys.stderr)
        except Exception as e:
            print(f"[profile] summary failed: {e}", file=sys.stderr)


@contextlib.contextmanager
def _with_profile_memray():
    """Scoped memray Tracker for Python `ZYME_PROFILE_BACKEND=mem`.

    This keeps allocation attribution focused on the target method when
    pipeline/run.py wraps that call in `with_profile()`. `zyme profile`
    still falls back to the external memray wrapper for scripts without a
    scoped region.
    """
    prof_path = _profile_output_path("memray.bin")
    try:
        import memray
    except Exception as e:
        print("[profile] backend=mem requested but memray is not importable "
              f"in this Python env ({e}); running without allocation trace.",
              flush=True, file=sys.stderr)
        yield
        return

    try:
        if os.path.exists(prof_path):
            os.unlink(prof_path)
    except OSError as e:
        print(f"[profile] could not remove stale memray output {prof_path}: {e}",
              flush=True, file=sys.stderr)

    tracker = None
    try:
        tracker = memray.Tracker(
            prof_path,
            native_traces=True,
            follow_fork=True,
        )
        tracker.__enter__()
    except Exception as e:
        print("[profile] scoped memray Tracker failed to start "
              f"({e}); running without allocation trace.",
              flush=True, file=sys.stderr)
        yield
        return

    print("[profile active] backend=memray kind=allocation_trace unit=bytes "
          f"scope=with_profile native=true writing to {prof_path}",
          flush=True, file=sys.stderr)
    try:
        yield
    finally:
        try:
            tracker.__exit__(None, None, None)
        except Exception as e:
            print(f"[profile] scoped memray Tracker failed to stop: {e}",
                  flush=True, file=sys.stderr)


def _profile_output_path(filename):
    out_dir = os.environ.get("ZYME_PROFILE_DIR") or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


@contextlib.contextmanager
def with_subprofile(name):
    """Named sub-block timer when ZYME_PROFILE=1 is set.

    When the main with_profile sampler points at an opaque native call
    (numba / Cython / BLAS) and you need to bisect time inside it,
    wrap segments with with_subprofile() to get clean named timings:

        with with_subprofile("normalize"):
            sc.pp.normalize_total(adata)
        with with_subprofile("hvg"):
            sc.pp.highly_variable_genes(adata)

    On exit, prints `[subprofile] <name>: <elapsed>s` to stderr. No-op
    when ZYME_PROFILE is unset — safe to leave in committed pipeline code.
    """
    if os.environ.get("ZYME_PROFILE", ""):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            print(f"[subprofile] {name}: {elapsed:.3f}s", flush=True, file=sys.stderr)
    else:
        yield


@contextlib.contextmanager
def with_memprof():
    """Memory profiling via tracemalloc, separate from with_profile.

    Tracemalloc has 10-30% overhead — too high to enable by default
    alongside cProfile. Use this helper when CPU profile suggests
    memory pressure (heavy gc activity, working set growth) or when
    investigating allocator churn.

    Captures top-15 allocation sites by total bytes. Stdlib only —
    no extra dependency. No-op when ZYME_PROFILE is unset.

    Usage:
        with with_memprof():
            with with_profile():     # both can nest
                run_pipeline(...)
    """
    if os.environ.get("ZYME_PROFILE", ""):
        import tracemalloc
        tracemalloc.start()
        try:
            yield
        finally:
            snapshot = tracemalloc.take_snapshot()
            top = snapshot.statistics("lineno")[:15]
            print("[memprof] top 15 allocators by size:", file=sys.stderr)
            for stat in top:
                print(f"  {stat}", file=sys.stderr)
            tracemalloc.stop()
    else:
        yield


_zyme_peak_mb_warned = False


def _warn_peak_mb_once(msg):
    """Emit a once-per-process warning about peak_mb measurement to stderr."""
    global _zyme_peak_mb_warned
    if _zyme_peak_mb_warned:
        return
    _zyme_peak_mb_warned = True
    print(f"[peak_mb] warning: {msg}", file=sys.stderr)


def peak_memory_mb():
    """Peak resident-set size in MB. Cross-platform.

    Windows: GetProcessMemoryInfo via ctypes — gives PeakWorkingSetSize
        (real RSS peak, includes native allocations). Requires explicit
        argtypes / restype declaration; without them, ctypes mismarshalls
        HANDLE on 64-bit Python and the call silently fails (returns 0).

    Linux: resource.getrusage(RUSAGE_SELF).ru_maxrss in KB.
    macOS:  resource.getrusage(RUSAGE_SELF).ru_maxrss in bytes.
    """
    if platform.system() == "Windows":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                             ctypes.POINTER(PMC),
                                             wintypes.DWORD]
            GetProcessMemoryInfo.restype = wintypes.BOOL

            pmc = PMC()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = GetProcessMemoryInfo(handle, ctypes.byref(pmc),
                                      ctypes.sizeof(pmc))
            if not ok:
                err = ctypes.get_last_error()
                _warn_peak_mb_once(
                    f"GetProcessMemoryInfo failed (Win32 error {err}); "
                    f"reporting 0.0. peak_mb in this round is unreliable."
                )
                return 0.0
            return pmc.PeakWorkingSetSize / (1024 * 1024)
        except Exception as e:
            _warn_peak_mb_once(
                f"ctypes path raised {type(e).__name__}: {e}; "
                f"reporting 0.0. peak_mb in this round is unreliable."
            )
            return 0.0
    else:
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            divisor = 1024 if platform.system() == "Linux" else 1024 * 1024
            return ru.ru_maxrss / divisor
        except Exception as e:
            _warn_peak_mb_once(
                f"resource.getrusage raised {type(e).__name__}: {e}; "
                f"reporting 0.0. peak_mb in this round is unreliable."
            )
            return 0.0


def emit_summary(speed_sec, peak_mb=None, cpu_sec=None, **metrics):
    """Print the standard summary lines that the zyme runner parses.

    Use at end of pipeline/run.py:
        emit_summary(speed_sec=elapsed, peak_mb=peak_memory_mb())

    cpu_sec defaults to time.process_time() (user + sys CPU time of this
    process). Together with speed_sec (wall time) it lets the agent tell a
    real algorithm win from a "throw threads at it" win: parallel-only wins
    show wall_time / cpu_time approaching the thread count, while algorithmic
    wins show cpu_time dropping in lockstep with wall_time.
    """
    print(f"speed_sec: {speed_sec:.6f}")
    if peak_mb is None:
        peak_mb = peak_memory_mb()
    print(f"peak_mb: {peak_mb:.1f}")
    if cpu_sec is None:
        cpu_sec = time.process_time()
    print(f"cpu_sec: {cpu_sec:.6f}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


# ---------------------------------------------------------------------------
# Tier-config + threading helpers
# ---------------------------------------------------------------------------
# These let task code read per-tier subset sizes from `task.yaml` and respect
# the thread count zyme injects (`ZYME_THREADS`). Both are opt-in: tasks that
# pre-bake per-tier data files or wire threading themselves can ignore them.

def _find_task_yaml():
    """Walk up from cwd to find task.yaml. Returns absolute path or None."""
    cur = os.path.abspath(os.getcwd())
    while True:
        candidate = os.path.join(cur, "task.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _parse_datasets_minimal(task_yaml_path):
    """Mirror of zyme/utils.py:parse_datasets, inlined here so task subprocesses
    don't need the zyme package on sys.path. Keep behavior in sync."""
    import re

    if not os.path.isfile(task_yaml_path):
        return []
    with open(task_yaml_path) as f:
        text = f.read()

    def split_top(s):
        parts, depth, buf = [], 0, []
        for ch in s:
            if ch == "{":
                depth += 1; buf.append(ch)
            elif ch == "}":
                depth -= 1; buf.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(buf)); buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append("".join(buf))
        return parts

    def coerce(v):
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            return v[1:-1]
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    def parse_val(v):
        v = v.strip()
        if v.startswith("{") and v.endswith("}"):
            inner = v[1:-1]
            out = {}
            for kv in split_top(inner):
                kv = kv.strip()
                if not kv:
                    continue
                # Accept dotted keys (burn.in, na.rm, BPPARAM) — R packages
                # routinely expose them. Bare `\w+` silently dropped these.
                m = re.match(r"\s*([A-Za-z0-9_.]+)\s*:\s*(.+)$", kv)
                if not m:
                    sys.stderr.write(
                        f"[parse_params] WARN: dropping malformed entry "
                        f"{kv!r} (no `key: value` shape)\n"
                    )
                    continue
                out[m.group(1)] = parse_val(m.group(2))
            return out
        return coerce(v)

    in_datasets = False
    out = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if re.match(r"^datasets:\s*$", line):
            in_datasets = True
            continue
        if in_datasets:
            if re.match(r"^\S", line) and not line.startswith("- "):
                in_datasets = False
                continue
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            inner = stripped[2:].strip()
            if inner.startswith("{") and inner.endswith("}"):
                inner = inner[1:-1]
            entry = {}
            for kv in split_top(inner):
                kv = kv.strip()
                if not kv:
                    continue
                m = re.match(r"\s*([A-Za-z0-9_.]+)\s*:\s*(.+)$", kv)
                if not m:
                    sys.stderr.write(
                        f"[parse_datasets] WARN: dropping malformed entry "
                        f"{kv!r} in datasets list (no `key: value` shape)\n"
                    )
                    continue
                key = m.group(1)
                val = m.group(2).strip()
                if val.startswith("{") and val.endswith("}"):
                    entry[key] = parse_val(val)
                else:
                    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                        val = val[1:-1]
                    entry[key] = val
            if "name" in entry and "path" in entry:
                entry.setdefault("tier", "tiny")
                entry.setdefault("params", {})
                out.append(entry)
    return out


def get_tier_params(tier=None):
    """Read tier-specific parameters from task.yaml's datasets[].params field.

    Args:
        tier: tier name to look up. Defaults to os.environ['ZYME_TIER'].

    Returns:
        dict of params declared in `task.yaml` for this tier (e.g.
        `{"n_cells": 50000, "n_genes": 2200}`). Empty dict if the tier
        exists but declares no params (e.g. tasks that pre-bake per-tier
        data files don't need params at all).

    Raises:
        RuntimeError: ZYME_TIER not set and tier= not passed.
        FileNotFoundError: task.yaml not located by walking up from cwd.
        KeyError: tier not present in task.yaml datasets.

    Use this from reference.py / pipeline/run.py instead of hardcoding a
    TIER_PARAMS dict literal — declaring tier params in task.yaml means
    adding a new tier (e.g. ood_large) needs only a yaml edit.
    """
    if tier is None:
        tier = os.environ.get("ZYME_TIER")
        if not tier:
            raise RuntimeError(
                "get_tier_params(): tier= not given and ZYME_TIER env var "
                "not set. Either pass tier explicitly or run via zyme."
            )
    yaml_path = _find_task_yaml()
    if yaml_path is None:
        raise FileNotFoundError(
            "get_tier_params(): no task.yaml found walking up from cwd."
        )
    entries = _parse_datasets_minimal(yaml_path)
    by_tier = {e["tier"]: e for e in entries}
    if tier not in by_tier:
        raise KeyError(
            f"get_tier_params(): tier '{tier}' not found in {yaml_path}. "
            f"Available tiers: {sorted(by_tier.keys())}"
        )
    return by_tier[tier].get("params", {})


def get_threads(default=None):
    """Read thread count from ZYME_THREADS env var, with fallback.

    Args:
        default: int returned when ZYME_THREADS is not set. Falls back
            further to os.cpu_count() if default is None.

    Returns:
        int — the effective thread count for this pipeline run.

    Use this anywhere the task currently hardcodes mc.cores / OMP_NUM_THREADS /
    n_jobs / numba.set_num_threads / etc. `zyme verify` sets ZYME_THREADS
    per matrix cell; routing through this helper makes the matrix actually
    exercise different thread counts.
    """
    raw = os.environ.get("ZYME_THREADS")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    if default is not None:
        return int(default)
    return os.cpu_count() or 1


# ============================================================
# auto_structure_check_all_slots — F3 defense (Output Truncation / Stub)
# ============================================================
# Default-on, opt-out structural check for every top-level slot in the
# reference output. Three boolean metrics per slot (1.0 = pass, 0.0 = fail):
#
#   <slot>_present         test has the slot (catches dropped output)
#   <slot>_shape_match     test slot has same shape/length as ref
#   <slot>_non_degenerate  if ref has variance, test does too
#                          (catches the "stub to constant" hack —
#                           override returns 0 / NA / single value while
#                           ref legitimately varies)
#
# This is NOT a value comparison. Pearson / max_abs_diff / Jaccard / ARI
# remain init agent's responsibility, declared explicitly in task.yaml::metrics
# for the slots that need them.
#
# What this DOES catch:
#   ✓ Output truncation (dipy_dti model_params 12 → 3): shape_match=0
#   ✓ Stub-to-constant (decontXLogLik=0, lifelines stale=0): non_degenerate=0
#   ✓ Dropped slot (override returns NULL / missing field): present=0
#
# What this does NOT catch (stays init agent's responsibility):
#   ✗ Stale-but-plausible values — needs explicit pearson_<slot>
#   ✗ Algorithmic drift within reasonable tolerance — needs explicit metric
#
# Waive specific slots via `waived = ["slot_a", ...]` only when the slot
# legitimately differs across runs (random init log, iteration history of
# different length). Pair every waiver with a `# diagnostic: <reason>`
# comment in evaluate.py.
#
# Returns dict {metric_name: float | None}; None when a check is undefined
# (unrecognized type, no numpy); treat None as a soft fail in audit.
def auto_structure_check_all_slots(ref, test, waived=None):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None

    if waived is None:
        waived = ()
    waived = set(waived)
    out = {}
    if ref is None:
        return out
    if not isinstance(ref, dict):
        return out

    for slot in list(ref.keys()):
        if slot in waived or not isinstance(slot, str) or not slot:
            continue
        r = ref.get(slot)
        if r is None:
            continue

        t = test.get(slot) if isinstance(test, dict) else None

        # 1. presence
        if t is None:
            out[f"{slot}_present"] = 0.0
            continue
        out[f"{slot}_present"] = 1.0

        # Recurse into nested dicts and pandas DataFrames one level. For
        # DataFrames, columns are checked individually — catches in-df hacks
        # (column deletion, column-to-constant, type swap) that the whole-df
        # shape check would miss.
        try:
            import pandas as _pd  # local import to keep numpy as the only hard dep
        except ImportError:  # pragma: no cover
            _pd = None
        if _pd is not None and isinstance(r, _pd.DataFrame) and isinstance(t, _pd.DataFrame):
            r_cols = {c: r[c].to_numpy() for c in r.columns}
            t_cols = {c: t[c].to_numpy() for c in t.columns if c in r.columns}
            nested = auto_structure_check_all_slots(r_cols, t_cols, waived=waived)
            for nm, val in nested.items():
                out[f"{slot}.{nm}"] = val
            continue
        if isinstance(r, dict) and isinstance(t, dict):
            nested = auto_structure_check_all_slots(r, t, waived=waived)
            for nm, val in nested.items():
                out[f"{slot}.{nm}"] = val
            continue

        # 2. shape match (numpy-aware; falls back to len() for plain lists)
        r_shape = _shape_of(r, np)
        t_shape = _shape_of(t, np)
        if r_shape is None or t_shape is None:
            out[f"{slot}_shape_match"] = None
            continue
        if r_shape != t_shape:
            out[f"{slot}_shape_match"] = 0.0
            continue
        out[f"{slot}_shape_match"] = 1.0

        # 3. non-degenerate (ref-has-variance → test-has-variance)
        out[f"{slot}_non_degenerate"] = _non_degenerate_check(r, t, np)

    return out


def _shape_of(x, np):
    if np is not None:
        try:
            return tuple(np.asarray(x).shape)
        except (ValueError, TypeError):
            pass
    try:
        return (len(x),)
    except TypeError:
        return None


def _non_degenerate_check(r, t, np):
    """Returns 1.0 if test has appropriate variance given ref, 0.0 if test
    is degenerate while ref isn't. None for unrecognized types."""
    if np is None:
        return None
    try:
        r_arr = np.asarray(r)
        t_arr = np.asarray(t)
    except (ValueError, TypeError):
        return None

    # Numeric path
    if np.issubdtype(r_arr.dtype, np.number) or np.issubdtype(r_arr.dtype, np.bool_):
        r_flat = r_arr.ravel().astype(float, copy=False)
        t_flat = t_arr.ravel().astype(float, copy=False)
        r_finite = r_flat[np.isfinite(r_flat)]
        t_finite = t_flat[np.isfinite(t_flat)]
        if r_finite.size == 0:
            return 1.0  # ref empty / all-NaN → no claim
        r_unique = np.unique(r_finite).size
        if r_unique <= 1:
            # Ref degenerate → test must match exactly
            try:
                return 1.0 if np.array_equal(r_flat, t_flat, equal_nan=True) else 0.0
            except TypeError:
                return 1.0 if np.array_equal(r_flat, t_flat) else 0.0
        # Ref non-degenerate → test must also be non-degenerate
        if t_finite.size == 0:
            return 0.0
        t_unique = np.unique(t_finite).size
        return 0.0 if t_unique <= 1 else 1.0

    # Categorical / string path
    if r_arr.dtype.kind in ("U", "S", "O"):
        r_flat = r_arr.ravel()
        t_flat = t_arr.ravel()
        r_unique = np.unique(r_flat).size
        if r_unique <= 1:
            return 1.0 if np.array_equal(r_flat, t_flat) else 0.0
        t_unique = np.unique(t_flat).size
        return 0.0 if t_unique <= 1 else 1.0

    return None


# Emit auto_structure metrics compactly: one summary line when all perfect,
# individual `name: value` lines only for failures (runner ignores `#` lines).
def emit_auto_structure_summary(struct_metrics):
    if not struct_metrics:
        return
    fail = {k: v for k, v in struct_metrics.items()
            if isinstance(v, (int, float)) and v is not None and v < 1.0}
    n_total = len(struct_metrics)
    n_fail  = len(fail)
    if n_fail == 0:
        print(f"# auto_structure: {n_total}/{n_total} perfect")
    else:
        print(f"# auto_structure: {n_total - n_fail}/{n_total} pass, {n_fail} FAIL:")
        for k, v in fail.items():
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
