"""zyme attest-sweep — fill missing (patch, tier, threads) attest coverage.

Walks every packaged patch (autozyme_py + autozyme_r), reads each task's
package_verify.tsv to see which (tier, thread) cells already have a Windows
patched row, and either prints or runs the missing ones via `zyme attest`.

The runner is resumable: each combo re-checks coverage immediately before
running, so restarting a crashed sweep skips completed work without redoing.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FRAMEWORK_ROOT = Path(__file__).resolve().parents[3]
PY_PLUGINS = FRAMEWORK_ROOT / "autozyme_py" / "src" / "autozyme"
R_PLUGINS = FRAMEWORK_ROOT / "autozyme_r" / "inst" / "patches"
OPT_TASK = FRAMEWORK_ROOT / "optimized_task"
LOG_DIR = FRAMEWORK_ROOT / "attest_logs" / "sweep"

DEFAULT_TIERS = ("tiny", "medium", "large", "ood_large", "ood_xlarge")
DEFAULT_THREADS = (1, 4, 8)
DEFAULT_REPS = 2


@dataclass(frozen=True)
class Combo:
    patch: str
    lang: str
    task_dir: Path
    tier: str
    thread: int

    def spec(self) -> str:
        return f"{self.patch}:{self.tier}:{self.thread}"


def _discover_patches() -> list[tuple[str, str]]:
    """[(patch_name, lang)] for every patch with a finalized speedup snapshot."""
    out: list[tuple[str, str]] = []
    for d in sorted(PY_PLUGINS.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and (d / "speedups_finalized.tsv").exists():
            out.append((d.name, "py"))
    for d in sorted(R_PLUGINS.iterdir()):
        if d.is_dir() and (d / "speedups_finalized.tsv").exists():
            out.append((d.name, "R"))
    return out


def _find_task_dir(patch: str) -> Path | None:
    """Locate the canonical task dir by matching patch_name in any pv.tsv."""
    for cat in OPT_TASK.iterdir():
        if not cat.is_dir():
            continue
        for d in cat.iterdir():
            pv = d / "package_verify.tsv"
            if not pv.exists():
                continue
            try:
                with pv.open(encoding="utf-8") as f:
                    header = f.readline().rstrip("\n").split("\t")
                    if "patch_name" not in header:
                        continue
                    idx = header.index("patch_name")
                    for line in f:
                        cols = line.rstrip("\n").split("\t")
                        if len(cols) > idx and cols[idx] == patch:
                            return d
            except OSError:
                continue
    return None


def _threading_mode(task_dir: Path) -> str:
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.exists():
        return "default"
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("threading:"):
            return line.split(":", 1)[1].strip()
    return "default"


def _task_supports_tier(task_dir: Path, tier: str) -> bool:
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.exists():
        return True
    text = yaml_path.read_text(encoding="utf-8")
    return f"tier: {tier}" in text or f'tier: "{tier}"' in text


def _existing_win_coverage(pv_path: Path) -> dict[str, set[str]]:
    """tier -> set of thread counts already attested on Windows (patched rows)."""
    if not pv_path.exists():
        return {}
    out: dict[str, set[str]] = defaultdict(set)
    with pv_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if not (r.get("system_os") or "").strip().startswith("Windows"):
                continue
            if (r.get("variant") or "").strip() != "patched":
                continue
            tier = (r.get("tier") or "").strip()
            th = (r.get("system_threads") or "").strip() or "NA"
            if tier:
                out[tier].add(th)
    return out


def _parse_int_list(raw: str, default: tuple[int, ...]) -> list[int]:
    if not raw:
        return list(default)
    return [int(x) for x in raw.replace(",", " ").split() if x]


def _parse_str_list(raw: str, default: tuple[str, ...]) -> list[str]:
    if not raw:
        return list(default)
    return [x.strip() for x in raw.replace(",", " ").split() if x.strip()]


def _plan_combos(
    tiers: list[str],
    threads: list[int],
    skip_patches: set[str],
    only_patches: set[str],
) -> tuple[list[Combo], dict[str, list[str]]]:
    skips: dict[str, list[str]] = {
        "no_task_dir": [],
        "not_applicable": [],
        "user_skip": [],
    }
    combos: list[Combo] = []
    for patch, lang in _discover_patches():
        if only_patches and patch not in only_patches:
            continue
        if patch in skip_patches:
            skips["user_skip"].append(patch)
            continue
        td = _find_task_dir(patch)
        if td is None:
            skips["no_task_dir"].append(patch)
            continue
        if _threading_mode(td) == "not_applicable":
            skips["not_applicable"].append(patch)
            continue
        cov = _existing_win_coverage(td / "package_verify.tsv")
        for tier in tiers:
            if not _task_supports_tier(td, tier):
                continue
            present = cov.get(tier, set())
            for th in threads:
                if str(th) in present:
                    continue
                combos.append(Combo(patch, lang, td, tier, th))
    return combos, skips


def _run_one(combo: Combo, reps: int, timeout: int) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = LOG_DIR / f"{stamp}_{combo.patch}_{combo.tier}_{combo.thread}t.log"
    cmd = [
        sys.executable, "-m", "zyme", "attest", str(combo.task_dir),
        "--name", combo.patch,
        "--tiers", combo.tier,
        "--threads", str(combo.thread),
        "--reps", str(reps),
    ]
    flags = {}
    if platform.system() == "Windows":
        flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        flags["start_new_session"] = True
    with log.open("w", encoding="utf-8") as f:
        f.write(f"[start] {combo.spec()} reps={reps}\n[cmd] {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd, cwd=FRAMEWORK_ROOT, stdout=f, stderr=subprocess.STDOUT,
            text=True, **flags,
        )
        try:
            proc.wait(timeout=None if timeout <= 0 else timeout)
            rc = int(proc.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"\n[timeout] exceeded {timeout}s; killing pid={proc.pid}\n")
            f.flush()
            _kill(proc)
            rc = 124
        f.write(f"\n[done] rc={rc}\n")
    print(f"[done] rc={rc} {combo.spec()} log={log.relative_to(FRAMEWORK_ROOT)}",
          flush=True)
    return rc


def _kill(proc: subprocess.Popen) -> None:
    if platform.system() == "Windows":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=15)
            return
        except Exception:
            pass
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _print_plan(combos: list[Combo], skips: dict[str, list[str]],
                tiers: list[str], threads: list[int]) -> None:
    print(f"=== attest-sweep plan ===")
    print(f"  tiers   = {tiers}")
    print(f"  threads = {threads}")
    for label, names in skips.items():
        if names:
            print(f"  skipped[{label}] ({len(names):2d}): {' '.join(sorted(names))}")
    print(f"  scheduled: {len(combos)} combos across "
          f"{len({c.patch for c in combos})} patches")
    for c in combos:
        print(f"    {c.patch:<24} {c.lang:<2} {c.tier:<11} threads={c.thread}")


def cmd_attest_sweep(args: argparse.Namespace) -> int:
    tiers = _parse_str_list(args.tiers, DEFAULT_TIERS)
    threads = _parse_int_list(args.threads, DEFAULT_THREADS)
    skip_patches = set(_parse_str_list(args.skip or "", ()))
    only_patches = set(_parse_str_list(args.only or "", ()))

    combos, skips = _plan_combos(tiers, threads, skip_patches, only_patches)
    if args.limit > 0:
        combos = combos[: args.limit]

    _print_plan(combos, skips, tiers, threads)
    if args.plan:
        return 0
    if not combos:
        print("[complete] nothing to do")
        return 0

    failures = 0
    for c in combos:
        cov = _existing_win_coverage(c.task_dir / "package_verify.tsv")
        if str(c.thread) in cov.get(c.tier, set()):
            print(f"[skip] now present {c.spec()}", flush=True)
            continue
        print(f"[start] {c.spec()}", flush=True)
        if _run_one(c, args.reps, args.timeout) != 0:
            failures += 1
    print(f"[complete] failures={failures}", flush=True)
    return 1 if failures else 0
