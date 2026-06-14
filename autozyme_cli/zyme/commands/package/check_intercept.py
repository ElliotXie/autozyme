"""`zyme package check-intercept` — verify a patch's fast functions actually fire.

Silently-broken patches (monkey-patch bind site changed, entry-point cache holds
the original ref, `pkg::fn(...)` reflection bypassed the dispatcher) are one of
the most expensive package-time failure modes — attest reports speedup ≈ 1.0
and the agent has to guess why. This command makes that diagnosis a single
subprocess: instrument the dispatcher, run smoke once, dump the per-target
intercept count.

Mechanics:
  1. Create a tmp file for the count payload.
  2. Spawn ``_verify_worker`` (Python) or ``inst/verify_worker.R`` (R) with
     ``ZYME_INSTRUMENT_INTERCEPTS=1`` and ``ZYME_INTERCEPT_OUT=<tmpfile>``,
     plus ``--activate`` so the fast path is selected.
  3. Read the JSON payload after the subprocess exits.
  4. Print a per-target table; FAIL if any target's count is 0.

This deliberately reuses the existing worker entry point so the path that
runs under attest is exactly the path we measure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from zyme.commands.attest import _infer_patch_name
from zyme.runner import _resolve_python
from zyme.utils import detect_lang, die, task_dir_from_args


def _spawn_python_worker(
    patch: str, task_dir: Path, tier: str, output_dir: Path, env: dict
) -> int:
    py = _resolve_python_for(task_dir)
    cmd = [
        py, "-m", "autozyme._verify_worker",
        "--patch", patch,
        "--task-dir", str(task_dir),
        "--tier", tier,
        "--output-dir", str(output_dir),
        "--activate",
    ]
    print(f"[check-intercept] spawning: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd, env=env)


def _resolve_python_for(task_dir: Path) -> str:
    """Resolve the interpreter via task.yaml::executor.python; fall back to sys.executable."""
    try:
        from zyme.parsers.task_yaml import parse_executor
        # parse_executor expects the task.yaml FILE path (it .read_text()s it),
        # not the task directory; passing the dir raised IsADirectoryError that
        # the bare except swallowed, silently ignoring executor.python.
        execu = parse_executor(task_dir / "task.yaml")
        spec = execu.get("python") if isinstance(execu, dict) else None
        if spec:
            return _resolve_python(spec)
    except Exception:
        pass
    return sys.executable


def _spawn_r_worker(
    patch: str, task_dir: Path, tier: str, output_dir: Path, env: dict
) -> int:
    # autozyme R package installs verify_worker.R into inst/; the path is
    # `system.file("verify_worker.R", package = "autozyme")` — we let R
    # resolve that.
    script = (
        'worker <- system.file("verify_worker.R", package = "autozyme"); '
        'if (!nzchar(worker)) stop("autozyme R package not installed"); '
        'source(worker)'
    )
    # Set commandArgs by spawning Rscript with --args.
    cmd = [
        "Rscript", "-e", script,
        "--args",
        "--patch", patch,
        "--task-dir", str(task_dir),
        "--tier", tier,
        "--output-dir", str(output_dir),
        "--activate",
    ]
    print(f"[check-intercept] spawning: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd, env=env)


def cmd_package_check_intercept(args) -> int:
    task_dir = task_dir_from_args(args)
    patch = getattr(args, "patch", None) or _infer_patch_name(task_dir)
    if not patch:
        die("could not infer patch name; pass --patch <name>")
    tier = getattr(args, "tier", None) or "tiny"
    lang = getattr(args, "lang", None) or detect_lang(task_dir)

    with tempfile.TemporaryDirectory(prefix="zyme_intercept_") as tmp:
        out_path = Path(tmp) / "counts.json"
        output_dir = Path(tmp) / "smoke_out"
        output_dir.mkdir()

        env = os.environ.copy()
        env["ZYME_INSTRUMENT_INTERCEPTS"] = "1"
        env["ZYME_INTERCEPT_OUT"] = str(out_path)
        env["ZYME_INTERCEPT_PATCH"] = patch

        if lang == "py":
            rc = _spawn_python_worker(patch, task_dir, tier, output_dir, env)
        else:
            rc = _spawn_r_worker(patch, task_dir, tier, output_dir, env)

        if rc != 0 and not out_path.exists():
            die(f"worker exited {rc} without writing intercept counts; "
                f"smoke may have crashed — re-run with `zyme attest --dry-run` "
                f"to inspect the command.")

        if not out_path.exists():
            die(f"intercept probe did not write {out_path}; "
                f"ensure the installed autozyme package includes _intercept_probe.py "
                f"(Python) or R/intercept_probe.R (R).")

        counts = json.loads(out_path.read_text(encoding="utf-8") or "{}")

    if not counts:
        print(f"[FAIL] patch {patch!r}: no targets were dispatched. "
              f"Either the patch did not register any targets, or autozyme.activate() "
              f"did not bind them — check register_patch() and the activation log.")
        return 1

    n_zero = 0
    print(f"\n{patch}  tier={tier}")
    for key in sorted(counts):
        n = int(counts[key])
        status = "OK" if n > 0 else "FAIL"
        if n == 0:
            n_zero += 1
        print(f"  {status:<4}  {n:>6}  {key}")
    if n_zero:
        print(f"\n[FAIL] {n_zero} target(s) never fired — smoke `call` did not "
              f"reach the patched bind site.")
        return 1
    print(f"\n[OK] all {len(counts)} target(s) fired")
    return 0
