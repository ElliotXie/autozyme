"""Subprocess worker for verify_patch — one measurement per invocation.

Spawned by `autozyme._verify._verify_one_tier` once for baseline and once for
patched, per rep. The parent runs:

    python -m autozyme._verify_worker
        --patch <name>
        --task-dir <abs/path>
        --tier <name>
        --output-dir <abs/path>
        [--activate]

Inside the subprocess we:
  1. Import the patch submodule (registers it; needed even for baseline since
     baseline still calls `patch.smoke[...]`).
  2. If --activate: monkey-patch upstream via `autozyme.activate(name)`.
  3. Build inputs via `patch.smoke["load"](task_dir, tier)`        — untimed.
  4. Time only `result = patch.smoke["call"](inputs)`              — the timed
     region. Use time.perf_counter().
  5. Save outputs via `patch.smoke["save"](result, output_dir, tier=tier)`.
  6. Print a single JSON line on stdout: {"elapsed_sec": <float>}.

Rationale: in-process deactivate() → baseline_call() → activate() → patched_call()
required deepcopy + Pyro param-store reset / TF seed-reset code inside
_smoke_call, which were timed and diluted the reported speedup (cell2location
5x → ~2.1x). Spawning a fresh process per measurement removes the contamination
entirely; each subprocess starts from a clean Python interpreter so smoke
recipes don't need any reset boilerplate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def _peak_rss_mb() -> float | None:
    """Peak RSS of this subprocess, in MiB. Returns None if unavailable.

    Linux ``ru_maxrss`` is KiB, macOS is bytes. Windows lacks the ``resource``
    module — falls back to psutil if importable, else None.
    """
    try:
        import resource  # POSIX only
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return maxrss / (1024 * 1024)
        return maxrss / 1024
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / (1024 * 1024)
        except Exception:
            return None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m autozyme._verify_worker")
    p.add_argument("--patch", required=True,
                   help="registered patch name (e.g. 'cell2location')")
    p.add_argument("--task-dir", required=True,
                   help="absolute path to the autozyme task directory")
    p.add_argument("--tier", required=True,
                   help="tier name (e.g. 'tiny', 'medium', 'large')")
    p.add_argument("--output-dir", required=True,
                   help="absolute path where smoke.save() writes its outputs")
    p.add_argument("--activate", action="store_true",
                   help="activate the patch before timing (patched run)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    import autozyme
    from autozyme._core import _REGISTRY, _import_submodule

    # Always import the submodule — even baseline needs the patch's smoke
    # recipe (smoke.load/call/save live on the registered patch object).
    if args.patch not in _REGISTRY:
        _import_submodule(args.patch)
    patch = _REGISTRY.get(args.patch)
    if patch is None:
        raise RuntimeError(
            f"patch {args.patch!r} did not register on import"
        )
    if patch.smoke is None:
        raise RuntimeError(
            f"patch {args.patch!r} has no smoke recipe"
        )

    from autozyme._smoke import resolve_smoke
    smoke = resolve_smoke(args.task_dir, patch)
    if smoke is None:
        raise RuntimeError(
            f"no smoke recipe for patch {args.patch!r} "
            f"(add attest/smoke.py under task dir)"
        )

    # `zyme package check-intercept` opts in via ZYME_INSTRUMENT_INTERCEPTS=1.
    # The probe wraps each registered fast fn with a counter and writes JSON
    # to ZYME_INTERCEPT_OUT at exit. Must run AFTER the patch is registered
    # (so targets are present) and BEFORE activate() resolves dispatchers.
    from autozyme._intercept_probe import install_from_env
    install_from_env()

    if args.activate:
        autozyme.activate(args.patch)

    inputs = smoke["load"](args.task_dir, args.tier)
    t0 = time.perf_counter()
    result = smoke["call"](inputs)
    elapsed = time.perf_counter() - t0
    smoke["save"](result, args.output_dir, tier=args.tier)

    # Peak RSS over this subprocess's lifetime (load + call + save). Reported
    # alongside elapsed so the parent can store baseline/patched memory next
    # to baseline/patched timing in package_verify.tsv. NOTE: ru_maxrss is OS
    # RSS — it includes shared/mmap pages, so absolute values can disagree
    # with tracemalloc-based `peak_mb` in results.tsv. Use for *delta* between
    # baseline and patched on the same workload.
    payload = {"elapsed_sec": elapsed, "peak_mb": _peak_rss_mb()}
    # Single JSON line on stdout. Anything else (warnings, autozyme banners)
    # goes to stderr by convention. Parent parses the LAST non-empty line as
    # JSON to be robust to a stray third-party stdout print.
    print(json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
