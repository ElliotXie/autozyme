"""Single-knob thread configuration."""
from __future__ import annotations

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Module-level option override. Equivalent to R's getOption("autozyme.threads").
_AUTOZYME_THREADS_OPTION: int | None = None


def set_threads(n: int) -> int:
    """Propagate `n` threads to all parallelism sources patches use.

    Sets BLAS/OpenMP env vars. Patches that read per-package thread
    options should consult these vars or expose their own knob.
    """
    global _AUTOZYME_THREADS_OPTION
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
    _AUTOZYME_THREADS_OPTION = n
    return n


def auto_threads(cap: int | None = None) -> int:
    """Pick a sensible thread count for a patch.

    Resolves a worker count using this priority order:
      1. ``AUTOZYMER_THREADS`` environment variable (explicit user override;
         wins over everything, including ``cap``).
      2. The module-level option set by ``set_threads()`` (also wins over
         ``cap``).
      3. Hardware default: ``os.cpu_count() - 1``, bounded above by ``cap``
         (if given) and a hard ceiling of 16 to prevent runaway
         oversubscription on big machines.

    Designed for use inside lifted patches as a drop-in replacement for
    hardcoded thread counts (``mc.cores=12`` ->
    ``mc.cores=auto_threads(cap=12)``). ``cap`` should be the patch's max
    sensible worker count -- typically what the lift-time
    ``pipeline/run.py`` used. Tier-aware patches pass ``cap`` from a
    per-tier dict.

    Args:
        cap: Optional integer upper bound. Caps the hardware default; does
            NOT cap the env-var or option override (those represent
            explicit user intent and win even when above ``cap``, e.g. for
            CI thread sweeps).

    Returns:
        Positive integer thread count, always >= 1.

    Examples:
        >>> auto_threads()                        # hardware default
        >>> auto_threads(cap=8)                   # cap at 8
        >>> os.environ["AUTOZYMER_THREADS"] = "4"
        >>> auto_threads(cap=8)                   # 4 (env wins)
    """
    # 1. Thread budget from env -- wins over cap. Honor any of the env
    # vars the attest harness propagates (ZYME_THREADS is the primary;
    # AUTOZYME_THREADS / OMP_NUM_THREADS are mirrors). The legacy spelling
    # AUTOZYMER_THREADS (with a stray R) is kept as a last resort so
    # anyone who hardcoded it doesn't silently break.
    for var in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
                "AUTOZYMER_THREADS"):
        env = os.environ.get(var, "")
        if env:
            try:
                n = int(env)
                if n >= 1:
                    return n
            except ValueError:
                pass

    # 2. Module-level option (set by set_threads) -- also wins over cap
    if _AUTOZYME_THREADS_OPTION is not None:
        try:
            n = int(_AUTOZYME_THREADS_OPTION)
            if n >= 1:
                return n
        except (TypeError, ValueError):
            pass

    # 3. Hardware default, bounded by cap and 16-thread ceiling
    cores = os.cpu_count() or 1
    default = max(1, cores - 1)

    if cap is not None:
        # Match `set_threads`: an explicit cap that can't be made a positive
        # int is a programming error, not a soft hint. Silently coercing to
        # "no cap" used to mask typos like `auto_threads(cap="8 ")` that
        # would burst out of the requested ceiling on big machines.
        try:
            cap_int = int(cap)
        except (TypeError, ValueError):
            raise ValueError(
                f"auto_threads cap must be int or None, got {cap!r} "
                f"({type(cap).__name__})"
            )
        if cap_int < 1:
            raise ValueError(
                f"auto_threads cap must be >= 1, got {cap_int}"
            )
        default = min(default, cap_int)

    default = min(default, 16)
    return max(1, default)


def safe_set_num_threads(n: int) -> int:
    """Set numba's parallel-kernel thread count, tolerating a locked pool.

    Numba locks its thread pool the first time any ``@njit(parallel=True)``
    kernel runs in the process; subsequent ``nb.set_num_threads(x)`` calls
    with a different value raise ``RuntimeError``. When multiple autozyme
    patches activate in the same process (e.g. scanpy + xclim), one of
    them locks the pool and the other crashes on its own set call.

    This wrapper: returns the value actually in effect after the call.
    No-op when target equals current; falls back to current count on
    RuntimeError (the kernel still runs correctly at the locked count,
    just with mild perf degradation vs the requested count).
    """
    try:
        import numba as nb
    except ImportError:
        return int(n)
    current = nb.get_num_threads()
    target = int(n)
    if current == target:
        return current
    try:
        nb.set_num_threads(target)
        return target
    except RuntimeError:
        return current
