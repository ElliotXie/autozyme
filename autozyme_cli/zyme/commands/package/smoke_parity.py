"""`zyme package smoke-parity` — confirm smoke recipe times the same region as pipeline/run.

A patch can attest at speedup ≈ 1.0 not because the fast path is slow, but
because the smoke recipe times a different region than the canonical
`pipeline/run.{R,py}`. The classic failure: smoke `call` invokes the upstream
public API but skips a preprocessing step that pipeline runs inline, so the
output is structurally similar but numerically wrong — and attest finishes,
publishes, before anyone notices.

This command catches that early:

  1. Spawn ``_verify_worker --activate`` for one tier; smoke writes outputs.
  2. Run the task's ``evaluate.{R,py}`` with ``ZYME_REFERENCE_DIR`` pointing
     to ``reference_outputs/<tier>/`` and ``ZYME_TEST_DIR`` pointing to the
     smoke output dir.
  3. Parse evaluate stdout for the per-metric values, compare against
     ``task.yaml::metrics`` thresholds.
  4. FAIL if any threshold is missed.

Cost: one smoke iteration on the tiny tier (~seconds), no baseline rep.
Use before ``zyme attest`` when developing a new patch.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from zyme.commands.attest import _infer_patch_name
from zyme.commands.package.check_intercept import (
    _resolve_python_for,
    _spawn_python_worker,
    _spawn_r_worker,
)
from zyme.parsers.task_yaml import parse_metrics, resolve_smoke_tier
from zyme.utils import detect_lang, die, task_dir_from_args


# Matches lines like "metric_name: 0.987" emitted by evaluate.{R,py}. The
# parsing convention matches what verify_patch already does in _verify.py.
_METRIC_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")


def _parse_metric_lines(lines: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for ln in lines:
        m = _METRIC_LINE.match(ln)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                continue
    return out


def _run_evaluate(
    task_dir: Path, smoke_output_dir: Path, tier: str, lang: str
) -> tuple[int, list[str], list[str]]:
    """Run task's evaluate.{R,py} with smoke output as the test dir.

    Returns (returncode, stdout_lines, stderr_lines).
    """
    ref_dir = task_dir / "reference_outputs" / tier
    if not ref_dir.exists():
        # Backward-compat with the flat layout.
        alt = task_dir / f"reference_output_{tier}"
        if alt.exists():
            ref_dir = alt
        else:
            die(f"no reference output for tier {tier!r} in {task_dir}; "
                f"run `zyme baseline reference --tier {tier}` first")

    for cand in ("evaluate.py", "evaluate.R"):
        src = task_dir / cand
        if src.exists():
            evaluate_path = src
            break
    else:
        die(f"no evaluate.py / evaluate.R in {task_dir}")

    # evaluate runs in a temp dir mirroring verify_patch's contract: the
    # script is copied so its relative-path assumptions hold, smoke output
    # is staged as ``<tmp>/pipeline/``. The tmp dir is anchored UNDER
    # task_dir (matching R verify.R's .verify_one_tier) so evaluate.R's
    # upward walk for `autozyme-framework` finds the category-level
    # symlink alongside the task — /tmp would have no such ancestor.
    with tempfile.TemporaryDirectory(prefix=".zyme_smoke_parity_", dir=task_dir) as tmp:
        tmp_dir = Path(tmp)
        dest = tmp_dir / evaluate_path.name
        shutil.copy(evaluate_path, dest)
        # Symlink the category-level autozyme-framework into tmp_dir so
        # evaluate.R's `while .fw_path ... helpers.R` walk resolves on the
        # first iteration. Tasks that use file.path(TASK_DIR, "..",
        # "autozyme-framework") get the same hit. No-op when the symlink
        # target is missing or the link can't be created.
        fw_link_target = task_dir.parent / "autozyme-framework"
        if fw_link_target.exists():
            try:
                os.symlink(fw_link_target, tmp_dir / "autozyme-framework")
            except (OSError, NotImplementedError):
                pass
        staged_test = tmp_dir / "pipeline"
        if smoke_output_dir.is_dir():
            shutil.copytree(smoke_output_dir, staged_test)
        else:
            die(f"smoke worker did not produce output dir {smoke_output_dir}")
        env = os.environ.copy()
        env["ZYME_TIER"] = tier
        env["ZYME_REFERENCE_DIR"] = str(ref_dir)
        env["ZYME_TEST_DIR"] = str(staged_test)
        env["ZYME_TASK_DIR"] = str(task_dir)
        if dest.suffix == ".py":
            py = _resolve_python_for(task_dir)
            cmd = [py, str(dest)]
        else:
            cmd = ["Rscript", str(dest)]
        proc = subprocess.run(cmd, env=env, cwd=tmp_dir, capture_output=True, text=True)
        return proc.returncode, proc.stdout.splitlines(), proc.stderr.splitlines()


def _check_thresholds(
    task_dir: Path, metrics: dict[str, float]
) -> tuple[bool, list[str]]:
    """Compare evaluate metrics to task.yaml::metrics thresholds.

    Returns (all_pass, breakdown_lines).
    """
    declared = parse_metrics(task_dir / "task.yaml")
    lines: list[str] = []
    all_pass = True
    for spec in declared:
        name = spec.get("name")
        if not name or name not in metrics:
            lines.append(f"  MISS  {name!r}  not emitted by evaluate")
            all_pass = False
            continue
        v = metrics[name]
        # parse_metrics normalizes the field to `comparator`; legacy
        # `direction` / `op` kept for backward compat.
        direction = (
            spec.get("comparator")
            or spec.get("direction")
            or spec.get("op")
            or "gte"
        )
        threshold = spec.get("threshold")
        ok = True
        if threshold is None:
            ok = True
        elif direction in ("gte", "ge", ">="):
            ok = v >= float(threshold)
        elif direction in ("lte", "le", "<="):
            ok = v <= float(threshold)
        elif direction in ("gt", ">"):
            ok = v > float(threshold)
        elif direction in ("lt", "<"):
            ok = v < float(threshold)
        else:
            ok = True  # unknown direction — don't fail on metadata bugs
        flag = "OK" if ok else "FAIL"
        lines.append(f"  {flag:<4}  {name}={v:.4f}  ({direction} {threshold})")
        all_pass = all_pass and ok
    return all_pass, lines


def cmd_package_smoke_parity(args) -> int:
    task_dir = task_dir_from_args(args)
    patch = getattr(args, "patch", None) or _infer_patch_name(task_dir)
    if not patch:
        die("could not infer patch name; pass --patch <name>")
    tier = getattr(args, "tier", None) or resolve_smoke_tier(task_dir / "task.yaml")
    lang = getattr(args, "lang", None) or detect_lang(task_dir)

    with tempfile.TemporaryDirectory(prefix="zyme_smoke_") as tmp:
        smoke_out = Path(tmp) / "smoke_out"
        smoke_out.mkdir()
        env = os.environ.copy()
        print(f"[smoke-parity] running smoke for {patch} at {tier}", file=sys.stderr)
        if lang == "py":
            rc = _spawn_python_worker(patch, task_dir, tier, smoke_out, env)
        else:
            rc = _spawn_r_worker(patch, task_dir, tier, smoke_out, env)
        if rc != 0:
            die(f"smoke worker exited {rc}; cannot evaluate parity (see stderr above)")
        if not any(smoke_out.iterdir()):
            die(f"smoke worker produced no output in {smoke_out}; "
                f"check smoke `save` writes to the directory passed in")

        eval_rc, stdout_lines, stderr_lines = _run_evaluate(task_dir, smoke_out, tier, lang)

    print(f"\n[smoke-parity] evaluate exit={eval_rc}")
    for ln in stdout_lines:
        print(f"  | {ln}")
    if eval_rc != 0:
        for ln in stderr_lines[-20:]:
            print(f"  ! {ln}", file=sys.stderr)
        print(f"\n[FAIL] evaluate crashed under smoke output — smoke `save` "
              f"likely emitted unexpected file shape.")
        return 1

    metrics = _parse_metric_lines(stdout_lines)
    if not metrics:
        print(f"\n[WARN] evaluate emitted no parseable `<name>: <value>` "
              f"lines — cannot check thresholds. Inspect output above.")
        return 0

    all_pass, breakdown = _check_thresholds(task_dir, metrics)
    print("")
    for ln in breakdown:
        print(ln)
    if not all_pass:
        print(f"\n[FAIL] smoke output fails task.yaml metric thresholds. "
              f"Candidate causes:")
        print("  - smoke `call` times a different region than `pipeline/run` "
              "(different args / preprocessing / defaults)")
        print("  - BLAS-thread mismatch with reference (drift ~1e-4): pin "
              "OMP/OPENBLAS/MKL_NUM_THREADS in smoke `load`")
        return 1
    print(f"\n[OK] smoke output matches reference within thresholds for tier={tier}")
    return 0
