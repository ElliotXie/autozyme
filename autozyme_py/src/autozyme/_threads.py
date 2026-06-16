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


def auto_threads(cap: int | None = None, default: int | None = 4) -> int:
    """Pick a sensible thread count for a patch.

    Resolves a worker count using this priority order:
      1. The first set of these environment variables, in order:
         ``ZYME_THREADS`` (primary) > ``AUTOZYME_THREADS`` >
         ``OMP_NUM_THREADS`` > ``AUTOZYMER_THREADS`` (legacy last-resort
         spelling). An explicit env override wins over everything,
         including ``cap`` and ``default``.
      2. The module-level option set by ``set_threads()`` (also wins over
         ``cap`` and ``default``).
      3. ``default`` threads, bounded above by the machine's core count, by
         ``cap`` (if given), and by a hard ceiling of 16.

    ``default`` is **4** -- the count the finalized speedup sweeps showed is
    the best single conservative default: it captures the large 1->4 jump
    (~1.9x median wall-clock) while staying clear of the oversubscription
    cliff that makes >4 threads *slower* on small inputs for many patches.
    Pass ``default=None`` to opt a patch into hardware scaling
    (``os.cpu_count() - 1``, still capped at 16) -- reserve that for the few
    patches whose finalized data keeps improving past 4 threads (e.g. scvelo,
    nichenetr).

    Designed for use inside lifted patches as a drop-in replacement for
    hardcoded thread counts (``mc.cores=12`` ->
    ``mc.cores=auto_threads(cap=12)``). ``cap`` is the patch's max sensible
    worker count -- a ceiling layered on top of ``default``. With the default
    base of 4, ``cap`` only bites for caps < 4 (or together with
    ``default=None``).

    Args:
        cap: Optional integer upper bound on the resolved count. Does NOT cap
            the env-var or option override (those represent explicit user
            intent and win even when above ``cap``, e.g. for CI thread
            sweeps).
        default: Base thread count when no env/option override is present.
            Defaults to 4. ``None`` means "scale to hardware"
            (``os.cpu_count() - 1``).

    Returns:
        Positive integer thread count, always >= 1.

    Examples:
        >>> auto_threads()                        # 4 (bounded by core count)
        >>> auto_threads(cap=2)                   # 2
        >>> auto_threads(default=None)            # os.cpu_count() - 1, max 16
        >>> os.environ["ZYME_THREADS"] = "8"
        >>> auto_threads()                        # 8 (env wins)
    """
    # 1. Thread budget from env -- wins over cap/default. Honor any of the env
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

    # 2. Module-level option (set by set_threads) -- also wins over cap/default
    if _AUTOZYME_THREADS_OPTION is not None:
        try:
            n = int(_AUTOZYME_THREADS_OPTION)
            if n >= 1:
                return n
        except (TypeError, ValueError):
            pass

    # 3. Base target, bounded by hardware, cap, and the 16-thread ceiling.
    cores = os.cpu_count() or 1
    if default is None:
        # "Scale to hardware" -- for patches whose finalized sweeps keep
        # speeding up past 4 threads. Leave one core for the OS.
        base = max(1, cores - 1)
    else:
        # An explicit default that can't be made a positive int is a
        # programming error, not a soft hint (mirrors the cap contract).
        try:
            base = int(default)
        except (TypeError, ValueError):
            raise ValueError(
                f"auto_threads default must be int or None, got {default!r} "
                f"({type(default).__name__})"
            )
        if base < 1:
            raise ValueError(
                f"auto_threads default must be >= 1, got {base}"
            )
        # Never hand out more workers than the machine has cores.
        base = min(base, cores)

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
        base = min(base, cap_int)

    base = min(base, 16)
    return max(1, base)


def safe_set_num_threads(n: int) -> int:
    """Set numba's parallel-kernel thread count, tolerating a locked or
    capped pool.

    Numba locks its thread pool the first time any ``@njit(parallel=True)``
    kernel runs in the process; subsequent ``nb.set_num_threads(x)`` calls
    with a different value raise ``RuntimeError``. When multiple autozyme
    patches activate in the same process (e.g. scanpy + xclim), one of
    them locks the pool and the other crashes on its own set call.

    Numba also rejects any value outside ``[1, NUMBA_NUM_THREADS]`` with a
    ``ValueError``. That fires when the requested budget exceeds the
    launchable pool -- e.g. a user sets ``NUMBA_NUM_THREADS=8`` on a 14-core
    box but leaves OMP/ZYME/AUTOZYME_THREADS unset, so the budget resolves
    to the hardware default (~13). We clamp the target to the pool ceiling
    first, and treat both errors as "keep whatever is currently in effect".

    This wrapper: returns the value actually in effect after the call.
    """
    try:
        import numba as nb
    except ImportError:
        return int(n)
    current = nb.get_num_threads()
    # set_num_threads() only accepts [1, NUMBA_NUM_THREADS]; clamp so an
    # over-budget request degrades to the pool ceiling instead of crashing.
    pool_max = getattr(nb.config, "NUMBA_NUM_THREADS", None) or current
    target = max(1, min(int(n), int(pool_max)))
    if current == target:
        return current
    try:
        nb.set_num_threads(target)
        return target
    except (RuntimeError, ValueError):
        return current
