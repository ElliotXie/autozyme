"""`zyme backfill` — bring under-replicated speedup cells up to a target n_reps.

Reads each patch's `speedups_finalized.tsv` to identify cells with `n_reps`
below the target, then drives `zyme attest` to add the missing reps and
re-aggregates with `scripts/finalize_speedups.{py,R}`.

Differs from `zyme attest-sweep`:
  - drives off `speedups_finalized.tsv` (n_reps), not package_verify presence
  - tier-aware concurrency: small tiers run 2 at a time, large/OOD serial
  - optionally targets unstable n=2 cells (high_rep_variance) for a 3rd rep
  - live progress dashboard
  - publishes + re-finalizes per completed cell, so progress is durable
"""
from __future__ import annotations

import argparse
import csv
import os
import platform as _platform
import re
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median as _stat_median

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    _RICH_OK = True
except ImportError:
    _RICH_OK = False


# ---- defaults ----------------------------------------------------------

_SMALL_TIERS = {"tiny", "small", "medium"}
_LARGE_TIERS_PREFIX = ("large", "ood_")

# Patches the user has decided to leave alone for now (known broken,
# documented exclusion, or out-of-scope). Surface as a CLI default; can
# be overridden via --skip.
#
# Three buckets:
#  1) User-requested exclusions (rctd, scvelo, fgsea, mast, vegan, sarsen,
#     seurat_sctransform).
#  2) Tasks whose task.yaml hardcodes a Mac executor.python path that
#     doesn't exist on Windows (lifelines, astropy, fipy, mdanalysis_rmsd,
#     xclim). attest fails before even launching the verify worker.
#     Discovered empirically 2026-06-02; fixing each task.yaml is a
#     separate cleanup.
_DEFAULT_SKIP = {
    # User-explicit defer list (2026-06-03): not measured this round.
    # NOTE: seurat_sctransform / seurat_integrate_cca / seurat_markers
    # temporarily un-skipped 2026-06-05 for the all-seurat/scanpy final
    # round (lift everything to n_reps=3). Restore these entries after
    # the sweep:
    #   "seurat_integrate_cca",
    #   "seurat_markers",
    #   "seurat_sctransform",
    "rctd", "scvelo", "fgsea", "mast",
}

_VARIANCE_TARGET_REPS = 3  # when --also-stabilize: lift unstable n=2 to n=3


# ---- data model --------------------------------------------------------


@dataclass
class CellState:
    """Current state of one (patch, tier, threads, platform, variant) cell."""
    n_reps: int
    sec_reps: list[float]
    status: str
    dataset: str
    sec_mean: float = 0.0  # historical mean per rep — cost estimator
    mem_mean: float = 0.0  # historical peak MB per rep — secondary sort

    @property
    def diff_pct_n2(self) -> float | None:
        if self.n_reps != 2 or len(self.sec_reps) != 2:
            return None
        mean = sum(self.sec_reps) / 2.0
        if mean <= 0:
            return None
        return abs(self.sec_reps[0] - self.sec_reps[1]) / mean * 100.0


@dataclass
class WorkItem:
    """One backfill job: run attest at (patch, tier, threads) to lift the
    weakest variant up to `target_reps`. attest produces baseline+patched
    in the same call; we set --reps to the larger of the two gaps."""
    patch: str
    lang: str        # "py" | "R"
    task_dir: Path
    tier: str
    threads: int
    target_reps: int
    baseline_n: int
    patched_n: int
    reason: str      # "low_reps" | "high_variance"
    dataset_hint: str = ""
    cost_sec: float = 0.0  # estimated seconds for this attest call
    cost_mem_mb: float = 0.0  # estimated peak MB for this cell

    @property
    def reps_to_run(self) -> int:
        return max(0, self.target_reps - min(self.baseline_n, self.patched_n))

    @property
    def is_small_tier(self) -> bool:
        return self.tier in _SMALL_TIERS

    def label(self) -> str:
        return (f"{self.patch}/{self.tier}/T{self.threads} "
                f"b={self.baseline_n} p={self.patched_n}→{self.target_reps}")


@dataclass
class ExecResult:
    item: WorkItem
    rc: int
    duration_sec: float
    note: str = ""


# ---- TSV reading -------------------------------------------------------


def _normalize_platform(value: str) -> str:
    s = (value or "").strip().lower()
    if not s:
        return "unknown"
    if s.startswith("windows") or s == "win":
        return "win"
    if s.startswith("macos") or s.startswith("darwin") or s == "mac":
        return "mac"
    return s


def _parse_int(value: str) -> int:
    try:
        return int((value or "").strip())
    except (ValueError, TypeError):
        return 0


def _parse_reps_float(value: str) -> list[float]:
    if not value:
        return []
    out: list[float] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


_TIER_ALIASES = {"tiny": "small"}


def _read_finalized(tsv_path: Path) -> dict[tuple[str, str, str, str], CellState]:
    """(tier, threads, platform, variant) -> CellState. Empty if file missing.

    Tier aliasing: the framework renamed `tiny`->`small` in task.yaml on
    2026-06-03, but historical finalized rows still carry tier="tiny".
    Map them to "small" at read time so the planner counts the combined
    n_reps under one key and attest is invoked with the live tier name.
    """
    out: dict[tuple[str, str, str, str], CellState] = {}
    if not tsv_path.is_file():
        return out
    try:
        with open(tsv_path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                raw_tier = r.get("tier", "")
                tier = _TIER_ALIASES.get(raw_tier, raw_tier)
                thr = str(r.get("threads", "")).strip()
                plat = _normalize_platform(r.get("platform", ""))
                variant = r.get("variant", "")
                key = (tier, thr, plat, variant)
                sec_mean = 0.0
                mem_mean = 0.0
                try:
                    sec_mean = float((r.get("sec_mean") or "0").strip() or 0)
                except ValueError:
                    pass
                try:
                    mem_mean = float((r.get("mem_mean") or "0").strip() or 0)
                except ValueError:
                    pass
                state = CellState(
                    n_reps=_parse_int(r.get("n_reps", "")),
                    sec_reps=_parse_reps_float(r.get("sec_reps", "")),
                    status=(r.get("status") or "").strip(),
                    dataset=(r.get("dataset") or "").strip(),
                    sec_mean=sec_mean,
                    mem_mean=mem_mean,
                )
                # When tier alias collapses legacy + current rows into one
                # key, keep the higher n_reps so we don't double-attest a
                # cell that already has enough total coverage.
                prev = out.get(key)
                if prev is None or state.n_reps > prev.n_reps:
                    out[key] = state
    except OSError:
        pass
    return out


# ---- patch → task_dir resolution ---------------------------------------


_LIFTED_FROM_RE = re.compile(r"Lifted from autozyme task `+([^`]+)`+")


def _extract_lifted_from(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:8192]
    except OSError:
        return None
    m = _LIFTED_FROM_RE.search(text)
    return m.group(1).strip() if m else None


def _discover_patches(framework: Path) -> list[tuple[str, str, Path]]:
    """[(patch_name, lang, patch_dir)] for every patch shipping a
    `speedups_finalized.tsv`."""
    out: list[tuple[str, str, Path]] = []
    py_root = framework / "autozyme_py" / "src" / "autozyme"
    if py_root.is_dir():
        for d in sorted(py_root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if (d / "speedups_finalized.tsv").exists():
                out.append((d.name, "py", d))
    r_root = framework / "autozyme_r" / "inst" / "patches"
    if r_root.is_dir():
        for d in sorted(r_root.iterdir()):
            if not d.is_dir():
                continue
            if (d / "speedups_finalized.tsv").exists():
                out.append((d.name, "R", d))
    return out


def _load_seurat_scanpy_manifests(framework: Path) -> dict[str, Path]:
    """Resolve `scanpy_<id>` / `seurat_<id>` patches via their per-step
    manifests under `scripts/`. The manifest lists `(id, path)` pairs;
    we prefix `id` with `scanpy_` or `seurat_` to match patch dir names."""
    out: dict[str, Path] = {}
    try:
        import yaml  # noqa: F401
    except ImportError:
        return out
    for filename, prefix in (
        ("scanpy_attest_manifest.yaml", "scanpy_"),
        ("seurat_attest_manifest.yaml", "seurat_"),
    ):
        mpath = framework / "scripts" / filename
        if not mpath.is_file():
            continue
        import yaml
        try:
            data = yaml.safe_load(mpath.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            continue
        for task_entry in data.get("tasks", []):
            tid = (task_entry.get("id") or "").strip()
            rel = (task_entry.get("path") or "").strip()
            if not tid or not rel:
                continue
            candidate = (framework / rel).resolve()
            if candidate.is_dir() and (candidate / "task.yaml").is_file():
                out[prefix + tid] = candidate
    return out


def _build_task_index(framework: Path) -> dict[str, Path]:
    """patch_name → task_dir. Combines two resolution paths:
      1. Per-patch `Lifted from autozyme task ``<name>``` marker in the
         patch source file (covers most patches).
      2. Per-step manifests at `scripts/scanpy_attest_manifest.yaml` and
         `scripts/seurat_attest_manifest.yaml` (covers scanpy_*/seurat_*
         which share one upstream package across many tasks)."""
    out: dict[str, Path] = {}
    out.update(_load_seurat_scanpy_manifests(framework))
    py_root = framework / "autozyme_py" / "src" / "autozyme"
    if py_root.is_dir():
        for d in py_root.iterdir():
            if d.name in out:
                continue
            init = d / "__init__.py"
            if init.is_file():
                task = _extract_lifted_from(init)
                if task:
                    found = _find_task_dir(framework, task)
                    if found:
                        out[d.name] = found
    r_root = framework / "autozyme_r" / "inst" / "patches"
    if r_root.is_dir():
        for d in r_root.iterdir():
            if d.name in out:
                continue
            patch_r = d / "patch.R"
            if patch_r.is_file():
                task = _extract_lifted_from(patch_r)
                if task:
                    found = _find_task_dir(framework, task)
                    if found:
                        out[d.name] = found
    return out


def _find_task_dir(framework: Path, task_name: str) -> Path | None:
    """Search optimized_task/* for a dir matching task_name."""
    root = framework / "optimized_task"
    if not root.is_dir():
        return None
    for cat in root.iterdir():
        if not cat.is_dir():
            continue
        cand = cat / task_name
        if cand.is_dir() and (cand / "task.yaml").is_file():
            return cand
    # Fallback recursive (slow but rare)
    for path in root.rglob(task_name):
        if path.is_dir() and (path / "task.yaml").is_file():
            return path
    return None


# ---- planning ----------------------------------------------------------


def _classify_tier(tier: str) -> str:
    t = (tier or "").lower()
    if t in _SMALL_TIERS:
        return "small"
    if t.startswith(_LARGE_TIERS_PREFIX):
        return "large"
    return "small"  # unknown: be conservative, allow parallelism


def _plan(framework: Path, *, target_reps: int, also_stabilize: bool,
          variance_pct: float, platform_keep: str,
          only_patches: set[str], skip_patches: set[str],
          tier_filter: set[str] | None,
          threads_filter: set[int] | None) -> tuple[list[WorkItem], dict]:
    """Build the work queue. Returns (items, stats)."""
    task_index = _build_task_index(framework)
    items: list[WorkItem] = []
    stats: dict = {"discovered": 0, "skipped_user": 0, "skipped_no_task": 0,
                   "skipped_excluded_platform": 0, "skipped_oom": 0,
                   "cells_under_target": 0, "cells_high_var": 0,
                   "cells_already_ok": 0}

    for patch_name, lang, patch_dir in _discover_patches(framework):
        stats["discovered"] += 1
        if only_patches and patch_name not in only_patches:
            continue
        if patch_name in skip_patches:
            stats["skipped_user"] += 1
            continue
        task_dir = task_index.get(patch_name)
        if task_dir is None:
            stats["skipped_no_task"] += 1
            continue

        cells = _read_finalized(patch_dir / "speedups_finalized.tsv")

        # Group cells by (tier, threads, platform); decide per group whether
        # the combined need warrants running attest there.
        groups: dict[tuple[str, str, str], dict[str, CellState]] = defaultdict(dict)
        for (tier, thr, plat, variant), st in cells.items():
            if plat != platform_keep:
                continue
            if tier_filter and tier not in tier_filter:
                continue
            if threads_filter is not None:
                try:
                    if int(thr) not in threads_filter:
                        continue
                except ValueError:
                    continue
            groups[(tier, thr, plat)].setdefault(variant, st)

        for (tier, thr, plat), variants in groups.items():
            b = variants.get("baseline")
            p = variants.get("patched")
            # Skip cells where the existing variant is OOM — that's a
            # documented boundary and adding reps won't change it.
            if (b and b.status == "OOM") or (p and p.status == "OOM"):
                stats["skipped_oom"] += 1
                continue
            b_n = b.n_reps if b else 0
            p_n = p.n_reps if p else 0
            target_for_cell = target_reps
            reason = "low_reps"
            if min(b_n, p_n) < target_reps:
                stats["cells_under_target"] += 1
            elif also_stabilize:
                # Either side could be the unstable one. Pick the worst.
                worst = max(
                    (b.diff_pct_n2 or 0.0) if b else 0.0,
                    (p.diff_pct_n2 or 0.0) if p else 0.0,
                )
                if worst > variance_pct:
                    target_for_cell = _VARIANCE_TARGET_REPS
                    reason = "high_variance"
                    stats["cells_high_var"] += 1
                else:
                    stats["cells_already_ok"] += 1
                    continue
            else:
                stats["cells_already_ok"] += 1
                continue

            try:
                thr_int = int(thr)
            except ValueError:
                # THREAD_INAPPLICABLE_PACKAGES collapse the thread label to
                # "any" in finalize, signaling the patch is single-threaded
                # and the per-thread distinction is meaningless. Dispatch
                # attest at threads=1 (the canonical single-thread value) so
                # backfill can still extend reps for these cells.
                if thr == "any":
                    thr_int = 1
                else:
                    continue

            # Cost estimate per attest call = baseline + patched + small
            # overhead. We rerun-baseline always, so both variants count
            # once per missing rep. The sec_mean is per-rep; multiply by
            # reps_to_run to estimate total wall.
            reps_run = max(1, target_for_cell - min(b_n, p_n))
            b_sec = (b.sec_mean if b else 0.0)
            p_sec = (p.sec_mean if p else 0.0)
            cost_sec = (b_sec + p_sec) * reps_run + 30.0  # 30s overhead
            cost_mem = max(b.mem_mean if b else 0.0,
                           p.mem_mean if p else 0.0)

            items.append(WorkItem(
                patch=patch_name, lang=lang, task_dir=task_dir,
                tier=tier, threads=thr_int, target_reps=target_for_cell,
                baseline_n=b_n, patched_n=p_n, reason=reason,
                dataset_hint=(b.dataset if b else (p.dataset if p else "")),
                cost_sec=cost_sec, cost_mem_mb=cost_mem,
            ))

    # Schedule: memory-first global sort. Lowest peak-RSS cells across ALL
    # patches run first so two concurrent workers can co-exist without OOM
    # pressure when the user has parallel jobs eating GBs elsewhere. Wall
    # time (cost_sec) breaks the tie when memory is comparable. Tier order
    # (small<medium<large<ood_*) keeps small things ahead of equally-cheap
    # large ones. Threads + patch are stable-sort tail keys.
    _TIER_ORDER = {"small": 0, "medium": 1, "large": 2,
                   "ood_large": 3, "ood_xlarge": 4}
    ordered = sorted(
        items,
        key=lambda x: (
            x.cost_mem_mb,
            x.cost_sec,
            _TIER_ORDER.get(x.tier, 5),
            x.threads,
            x.patch,
        ),
    )
    return ordered, stats


# ---- execution ---------------------------------------------------------


def _attest_patch_name(patch: str) -> str:
    """Map a per-cell patch directory name to the actual registered patch
    name attest passes to verify_patch.

    Most patches register under the same name as their directory. The
    scanpy_*/seurat_* family is the exception: every sub-step shares one
    registered patch (`scanpy` or `seurat`), differentiated by task_dir
    rather than patch name. Passing `seurat_integrate_cca` to R's
    verify_patch fails with `no patch named ...`."""
    if patch.startswith("scanpy_"):
        return "scanpy"
    if patch.startswith("seurat_"):
        return "seurat"
    return patch


def _build_attest_cmd(item: WorkItem, framework: Path) -> list[str]:
    """Construct the `zyme attest` invocation for one cell. We pass
    `--rerun-baseline` so attest doesn't short-circuit baseline via its
    cache — that cache hit would skip writing a new baseline rep, and
    n_reps for the baseline variant would stay at 1 forever."""
    return [
        sys.executable, "-m", "zyme", "attest", str(item.task_dir),
        "--name", _attest_patch_name(item.patch),
        "--tiers", item.tier,
        "--threads", str(item.threads),
        "--reps", str(max(1, item.reps_to_run)),
        "--lang", item.lang,
        "--no-preflight",
        "--rerun-baseline",
    ]


def _patch_dir(framework: Path, patch: str, lang: str) -> Path:
    if lang == "py":
        return framework / "autozyme_py" / "src" / "autozyme" / patch
    return framework / "autozyme_r" / "inst" / "patches" / patch


def _copy_new_pv_rows_to_speedups(item: WorkItem, framework: Path,
                                  attest_start_iso: str,
                                  log_handle) -> tuple[int, int]:
    """Append every package_verify.tsv row with timestamp >= attest_start_iso
    into the patch's bundled speedups.tsv.

    Bypasses attest's auto-publish filter, which groups by exact timestamp
    and drops baseline-only batches — fine in the steady state, but
    catastrophic here because R's verify writes baseline and patched at
    different `Sys.time()` instants. Without this copy, baseline reps would
    never accumulate in speedups.tsv even when attest measured them.

    Returns (new_rows_added, rows_skipped_as_dup)."""
    pv_path = item.task_dir / "package_verify.tsv"
    patch_dir = _patch_dir(framework, item.patch, item.lang)
    if not pv_path.is_file():
        log_handle.write(f"[pv-copy] no package_verify.tsv at {pv_path}\n")
        return 0, 0

    # For scanpy_*/seurat_* sub-step patches, the registered patch_name in
    # package_verify is `scanpy` or `seurat` — task_dir disambiguates the
    # sub-step. The bundled speedups shard we publish into is the sub-step's
    # own dir, so just filter by timestamp (the task_dir already scopes us).
    expected_patch = _attest_patch_name(item.patch)
    with pv_path.open(encoding="utf-8", newline="") as f:
        pv_reader = csv.DictReader(f, delimiter="\t")
        pv_header = pv_reader.fieldnames or []
        new_rows = [r for r in pv_reader
                    if (r.get("patch_name") or "").strip() == expected_patch
                    and (r.get("timestamp") or "") >= attest_start_iso]

    if not new_rows:
        log_handle.write(f"[pv-copy] no rows >= {attest_start_iso}\n")
        return 0, 0

    # Per-platform split: each row goes to its own speedups.<plat>.tsv shard so
    # cross-machine git merges stay disjoint. Dedup against every existing shard
    # (+ legacy combined) with the same signature finalize uses (idempotent).
    def _row_plat(r: dict) -> str:
        s = (r.get("system_os") or "").lower()
        if "mac" in s or "apple" in s or "darwin" in s:
            return "mac"
        return "win" if "win" in s else "other"

    sig_cols = ("timestamp", "patch_name", "tier", "rep_idx", "variant",
                "sec", "peak_mb", "system_os", "system_cpu", "system_threads")
    shard_paths = {p: patch_dir / f"speedups.{p}.tsv" for p in ("mac", "win", "other")}
    seen: set = set()
    sp_header = pv_header
    for sp in (*shard_paths.values(), patch_dir / "speedups.tsv"):
        if not sp.is_file():
            continue
        with sp.open(encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f, delimiter="\t")
            if rd.fieldnames:
                sp_header = rd.fieldnames
            for r in rd:
                seen.add(tuple((r.get(c) or "").strip() for c in sig_cols))

    added = 0
    skipped = 0
    by_plat: dict[str, list[dict[str, str]]] = {}
    for r in new_rows:
        key = tuple((r.get(c) or "").strip() for c in sig_cols)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        by_plat.setdefault(_row_plat(r), []).append(r)
        added += 1

    for plat, rows in by_plat.items():
        sp = shard_paths[plat]
        sp.parent.mkdir(parents=True, exist_ok=True)
        write_header = not sp.is_file()
        with sp.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sp_header, delimiter="\t",
                                    extrasaction="ignore", lineterminator="\n")
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow({c: (r.get(c) or "") for c in sp_header})

    log_handle.write(f"[pv-copy] added={added} dedup_skipped={skipped} "
                     f"new_pv_rows={len(new_rows)}\n")
    return added, skipped


# Seurat patches that load the PCA/CCA fast path which needs Python via
# reticulate. Other seurat patches (markers, ScaleData, FindNeighbors) are
# native-R only and don't need Python — running their cells doesn't risk
# the silent-fallback bug.
_SEURAT_PATCHES_NEEDING_PYTHON = {
    "seurat_pca", "seurat_integrate_cca",
}


def _seurat_patch_needs_python(patch_name: str) -> bool:
    return patch_name in _SEURAT_PATCHES_NEEDING_PYTHON


def _preflight_seurat_python(env: dict) -> tuple[bool, str]:
    """Verify that an Rscript subprocess can find Python+numpy+scipy the way
    the seurat PCA/CCA fast path will at run time. Rather than re-implement
    (and drift from) the discovery, the probe calls autozyme's OWN binder,
    `autozyme:::.az_py_bind()` -- the exact function patch.R's
    `.seurat_init_python` uses. It honors AUTOZYME_PYTHON / RETICULATE_PYTHON /
    CONDA_PREFIX and otherwise auto-discovers a numpy+scipy Python on PATH, so
    this preflight can never give a false result by mirroring the wrong logic.
    Runs in ~3 seconds."""
    rscript = _platform_rscript()
    if not rscript:
        return False, "Rscript not found on PATH"
    probe = (
        "suppressMessages(library(autozyme));"
        "ok <- tryCatch({"
        "  if (!isTRUE(autozyme:::.az_py_bind()))"
        "    stop('no Python with numpy+scipy located');"
        "  reticulate::py_config()$python"
        "}, error = function(e) paste('FAIL:', conditionMessage(e)));"
        "cat(ok, '\\n')"
    )
    try:
        result = subprocess.run(
            [rscript, "-e", probe], env=env, capture_output=True,
            text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"reticulate probe failed: {type(e).__name__}: {e}"
    out = (result.stdout or "").strip().splitlines()
    last = out[-1] if out else ""
    if last.startswith("FAIL:") or "FAIL:" in last:
        return False, last
    if not last or "python" not in last.lower() and "/" not in last and "\\" not in last:
        return False, f"reticulate probe returned no python path; stdout={result.stdout!r}"
    return True, last


def _platform_rscript() -> str | None:
    """Find Rscript on this platform."""
    candidates = ["Rscript"] if os.name != "nt" else [
        "Rscript.exe", "Rscript",
        r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe",
    ]
    for c in candidates:
        try:
            r = subprocess.run([c, "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return c
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


# Patterns that indicate the patched code silently fell back to upstream —
# usually a Python init failure on the seurat patch's PCA/CCA fast path.
# If the cell log contains any of these, the "patched" measurement is
# actually baseline code timed under a patched label, and publishing it
# would corrupt the speedup numbers (incident 2026-06-05: integrate_cca/
# medium ran 4.5h before this was noticed; rep 1 patched took 5762s vs
# 579s historical because it silently routed to upstream Seurat).
_FALLBACK_WARNING_PATTERNS = (
    "falling back to upstream",
    "no Python with numpy+scipy",
    "fast path needs Python",
)


def _check_no_fallback_warnings(log_path: Path) -> tuple[bool, str]:
    """Scan the just-finished cell's attest log for known silent-fallback
    warnings. Returns (ok, summary). When a warning is found, the patched
    timings in this run measure upstream code — refuse to publish."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True, ""  # if log unreadable, don't block (other checks catch)
    for pat in _FALLBACK_WARNING_PATTERNS:
        if pat in text:
            return False, f"patched-fallback: {pat!r} in log — patched ran upstream"
    return True, ""


def _check_pv_new_rows_passed(item: WorkItem,
                              attest_start_iso: str,
                              log_handle) -> tuple[bool, str]:
    """Inspect package_verify.tsv rows written by the just-finished attest.

    attest returns rc=0 even when its verify-worker subprocess crashed
    inside (e.g. FileNotFoundError on a missing dataset) — it catches
    the exception and writes a row with empty `pass` and an error in
    `note`. Without this check, backfill reports those silent failures
    as `OK ... (+0 rows)` and we waste a full pass thinking n_reps was
    bumped when nothing usable landed.

    Returns (all_passed, summary). When any new row has pass != "1",
    `summary` carries the first error note so the dashboard surfaces it.
    """
    pv_path = item.task_dir / "package_verify.tsv"
    if not pv_path.is_file():
        return False, "pv.tsv missing after attest"
    expected_patch = _attest_patch_name(item.patch)
    failures: list[str] = []
    n_new = 0
    n_ok = 0
    with pv_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if (r.get("patch_name") or "").strip() != expected_patch:
                continue
            if (r.get("timestamp") or "") < attest_start_iso:
                continue
            # Concurrent attests of the same patch (different tiers)
            # share the same pv.tsv and write rows in interleaved order.
            # Restrict the pass-check to rows whose tier matches the cell
            # we're attesting — otherwise a sibling cell's failure
            # poisons our verdict. Mac's task.yaml rename means historical
            # "tiny" rows now map to "small"; treat them as equivalent.
            row_tier = (r.get("tier") or "").strip()
            row_tier = _TIER_ALIASES.get(row_tier, row_tier)
            if row_tier and row_tier != item.tier:
                continue
            n_new += 1
            variant = (r.get("variant") or "").strip() or "?"
            tier = (r.get("tier") or "").strip() or "?"
            sec = (r.get("sec") or "").strip()
            note = (r.get("note") or "").strip()
            pass_val = (r.get("pass") or "").strip()
            # baseline rows have no `pass` concept (it's the reference,
            # no metric comparison). Treat them as OK iff sec was filled
            # (real measurement landed) and note has no Traceback signal.
            if variant == "baseline":
                if sec and "Traceback" not in note and "exited" not in note:
                    n_ok += 1
                    continue
            else:
                # Pass column: R writes "1"/"NA", Python writes "true"/"".
                if pass_val.lower() in ("1", "true"):
                    n_ok += 1
                    continue
            short = note.split(":")[0:3]
            short_note = "/".join(short)[:160] if note else "empty-row"
            failures.append(f"{variant}@{tier}={short_note}")
    if n_new == 0:
        return False, "no new pv.tsv rows from attest"
    if failures:
        log_handle.write(
            f"[pv-check] {n_ok}/{n_new} passed; failures: {failures}\n"
        )
        return False, f"silent-fail: {failures[0]}"
    log_handle.write(f"[pv-check] {n_ok}/{n_new} passed\n")
    return True, ""


def _publish_and_finalize(item: WorkItem, framework: Path,
                          attest_start_iso: str,
                          log_handle) -> tuple[int, str]:
    """Bring new attest rows from package_verify.tsv into the patch's
    speedups.tsv (bypassing attest's batch-grouping filter so baseline
    rows aren't dropped), then re-aggregate into speedups_finalized.tsv."""
    env = _subprocess_env(framework)
    added, _ = _copy_new_pv_rows_to_speedups(
        item, framework, attest_start_iso, log_handle,
    )
    if added == 0:
        log_handle.write(
            "[pv-copy] WARN no new rows copied — attest may have run but "
            "filed nothing under this patch_name in package_verify.tsv\n"
        )
    if item.lang == "py":
        script = framework / "autozyme_py" / "scripts" / "finalize_speedups.py"
        cwd = framework / "autozyme_py"
        finalize_cmd = [sys.executable, str(script), item.patch]
    else:
        script = framework / "autozyme_r" / "scripts" / "finalize_speedups.R"
        cwd = framework / "autozyme_r"
        finalize_cmd = ["Rscript", str(script), item.patch]
    log_handle.write(f"\n[finalize] cwd={cwd} {' '.join(finalize_cmd)}\n")
    log_handle.flush()
    p2 = subprocess.run(finalize_cmd, cwd=cwd, stdout=log_handle,
                        stderr=subprocess.STDOUT, text=True, env=env)
    if p2.returncode != 0:
        return p2.returncode, "finalize_speedups failed"
    return 0, f"ok (+{added} rows)"


def _kill_zombies():
    """Disabled. Earlier sweeps used a PowerShell pass that killed every
    R/Python proc older than 1 minute to clear Windows orphans, but with
    parallel workers that kills the SIBLING worker's still-running R/Py
    subprocesses (medium runs commonly exceed 60s). attest already does
    its own teardown via R-side `system2('taskkill', .../F, ...)`; we
    rely on that. If true zombies pile up across sessions, sweep them
    out-of-band rather than from inside a worker."""
    return


def _subprocess_env(framework: Path) -> dict[str, str]:
    """Env for subprocess invocations of `python -m zyme`. The cli package
    is `autozyme_cli/zyme` and isn't pip-installed, so we have to prepend
    it to PYTHONPATH or the subprocess hits `ImportError: __version__`.

    Also force attest's auto-publish into append mode so reps accumulate
    in the bundled `speedups.tsv` instead of being merge-replaced. (Merge
    keys by hardware signature, so the newest timestamp wins per machine —
    which is the opposite of what backfill needs.)"""
    env = os.environ.copy()
    cli_dir = str(framework / "autozyme_cli")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (cli_dir + os.pathsep + existing) if existing else cli_dir
    # Skip attest's built-in publish step entirely. Backfill runs its own
    # shard-aware `_copy_new_pv_rows_to_speedups` + finalize per cell. If
    # attest also publishes, it (a) re-creates the legacy combined
    # speedups.tsv that the framework retired, and (b) flushes the entire
    # pv.tsv history to disk every call — which would resurrect tier="tiny"
    # rows already cleaned out upstream by Mac's tier rename.
    env["AUTOZYME_ATTEST_PUBLISH_MODE"] = "skip"
    # Paper-headline mode for seurat::FindAllMarkers on Windows: the
    # `for_paper` variant uses parallel::mclapply (fork) which Windows
    # doesn't support — so it crashes on Win. `for_paper_omp` routes to
    # the OpenMP-threaded kernel (src/for_paper_markers_omp.cpp) instead,
    # which is the Windows-side paper headline path. Default `fusion` is
    # faster for end users but isn't what the paper measured. Affects
    # only the markers dispatcher; other patches ignore this env.
    env["AUTOZYME_MODE"] = "for_paper_omp"
    # autozyme R's seurat PCA/CCA fast path needs a numpy+scipy Python.
    # autozyme's binder (.az_py_bind) honors AUTOZYME_PYTHON /
    # RETICULATE_PYTHON / CONDA_PREFIX and otherwise auto-discovers one on
    # PATH, so normally nothing is needed here -- the inherited environment
    # (e.g. an already-activated conda env) is enough.
    #
    # On a box where the intended interpreter isn't already active (a common
    # Windows case: the conda env isn't on PATH and CONDA_PREFIX is unset),
    # point the campaign at it WITHOUT hardcoding a machine path: set
    # AUTOZYME_BACKFILL_CONDA_PREFIX (e.g. to ...\envs\<name> on Windows or
    # .../envs/<name> elsewhere) before launching. Skipping this is what once
    # silently measured baseline code at baseline speed (integrate_cca falls
    # back to upstream Seurat when no Python resolves: incident
    # 2026-06-05 / 2026-06-06, two 4-5h Wave 1 runs wasted before diagnosis).
    pfx = env.get("AUTOZYME_BACKFILL_CONDA_PREFIX", "")
    if pfx:
        py = Path(pfx) / ("python.exe" if os.name == "nt" else "bin/python")
        if py.exists():
            env.setdefault("CONDA_PREFIX", pfx)
            env.setdefault("RETICULATE_PYTHON", str(py))
    return env


def _execute_one(item: WorkItem, framework: Path, log_dir: Path,
                 timeout_sec: int) -> ExecResult:
    """Run attest + publish + finalize for one cell. Per-cell log under
    log_dir; stdout/stderr both go there so we can grep failures later."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = log_dir / f"{stamp}_{item.patch}_{item.tier}_T{item.threads}.log"
    cmd = _build_attest_cmd(item, framework)
    env = _subprocess_env(framework)

    start = time.monotonic()
    # ISO timestamp at attest launch — anything in package_verify.tsv with
    # timestamp >= this is "new from this run" and worth copying. Use a
    # ~1-second back-step so a tight clock skew doesn't drop our own row.
    attest_start_iso = time.strftime("%Y-%m-%dT%H:%M:%S",
                                     time.localtime(time.time() - 1))
    flags = {}
    if _platform.system() == "Windows":
        flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        flags["start_new_session"] = True

    with log.open("w", encoding="utf-8") as f:
        f.write(f"[start] {item.label()} reason={item.reason} "
                f"attest_start_iso={attest_start_iso}\n")
        f.write(f"[cmd] {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd, cwd=framework, stdout=f, stderr=subprocess.STDOUT,
            text=True, env=env, **flags,
        )
        try:
            proc.wait(timeout=None if timeout_sec <= 0 else timeout_sec)
            rc = int(proc.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"\n[timeout] exceeded {timeout_sec}s; killing pid={proc.pid}\n")
            f.flush()
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait()
            rc = 124

        note = "attest ok"
        if rc == 0:
            # Layer 1: detect silent patched→upstream fallback. If patched
            # code couldn't init (e.g. Python missing for seurat
            # PCA/CCA), the timing measures upstream — publishing would
            # corrupt speedup numbers.
            ok, fb_note = _check_no_fallback_warnings(log)
            if not ok:
                f.write(f"[fallback-check] FAIL: {fb_note}\n")
                rc, note = 3, fb_note
            else:
                # Layer 2: attest's verify-worker catches FileNotFoundError
                # and writes a row with empty `pass` + rc=0. Gate publish
                # on every new pv row passing — otherwise we report OK
                # while finalized.tsv silently sits at n_reps=1.
                ok, fail_note = _check_pv_new_rows_passed(
                    item, attest_start_iso, f
                )
                if not ok:
                    rc, note = 2, fail_note
                else:
                    rc, note = _publish_and_finalize(item, framework,
                                                     attest_start_iso, f)
        f.write(f"\n[done] rc={rc} duration={time.monotonic()-start:.1f}s note={note}\n")

    _kill_zombies()
    return ExecResult(item=item, rc=rc, duration_sec=time.monotonic()-start, note=note)


# ---- orchestrator with progress display --------------------------------


class _ProgressDashboard:
    """Live rich-table progress. Tracks per-tier counts, in-flight items,
    and recent completions. Falls back to plain stdout if rich missing."""

    def __init__(self, total: int, use_rich: bool):
        self.total = total
        self.done = 0
        self.failed = 0
        self.start_time = time.monotonic()
        self.in_flight: dict[str, float] = {}  # label → start_time
        self.recent: list[tuple[str, int, float, str]] = []  # (label, rc, dur, note)
        self.lock = threading.Lock()
        self.use_rich = use_rich and _RICH_OK
        if self.use_rich:
            self.console = Console()
            self.live = Live(self._render(), console=self.console,
                             refresh_per_second=2, transient=False)
        else:
            self.console = None
            self.live = None

    def __enter__(self):
        if self.use_rich:
            self.live.__enter__()
        return self

    def __exit__(self, *args):
        if self.use_rich:
            self.live.__exit__(*args)

    def _render(self):
        if not self.use_rich:
            return None
        elapsed = time.monotonic() - self.start_time
        rate = self.done / elapsed if elapsed > 0 else 0
        remaining = self.total - self.done
        eta_s = remaining / rate if rate > 0 else 0
        header = (f"Backfill: {self.done}/{self.total} done "
                  f"({self.failed} failed) | "
                  f"elapsed {elapsed/60:.1f}m | "
                  f"rate {rate*60:.1f}/min | "
                  f"ETA {eta_s/60:.1f}m")
        t = Table(title=header, show_lines=False, expand=False)
        t.add_column("State", width=8)
        t.add_column("Cell", width=48)
        t.add_column("Elapsed/Note", width=24)
        for label, start in self.in_flight.items():
            t.add_row("RUN", label, f"{time.monotonic()-start:.0f}s")
        for label, rc, dur, note in self.recent[-10:]:
            state = "OK" if rc == 0 else f"FAIL {rc}"
            t.add_row(state, label, f"{dur:.0f}s {note}")
        return t

    def start_item(self, label: str):
        with self.lock:
            self.in_flight[label] = time.monotonic()
            if self.live:
                self.live.update(self._render())

    def finish_item(self, label: str, rc: int, duration: float, note: str):
        with self.lock:
            self.in_flight.pop(label, None)
            self.recent.append((label, rc, duration, note))
            self.done += 1
            if rc != 0:
                self.failed += 1
            if self.live:
                self.live.update(self._render())
            if not self.use_rich:
                print(f"[{self.done}/{self.total}] "
                      f"{'OK' if rc==0 else f'FAIL rc={rc}'} "
                      f"{label} ({duration:.0f}s) {note}", flush=True)


_MAX_DYNAMIC_WORKERS = 8         # hard ceiling on concurrent cells
_MEM_BUDGET_FRAC = 0.80          # keep at least 20% RAM headroom
_CPU_BUDGET_FRAC = 0.80          # keep at least 20% CPU thread headroom
_DISPATCH_POLL_SEC = 5.0         # how often to re-probe when blocked
_STARVATION_OVERRIDE_SEC = 900.0  # head-of-queue waited this long -> force run
_PER_PATCH_CONCURRENCY = 1       # one cell per patch at a time (avoids dataset file-lock races)


def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except ImportError:
        return None


def _resource_fits(item: 'WorkItem', in_flight: list['WorkItem'],
                    psutil_mod, cpu_total: int,
                    total_mem_mb: float) -> tuple[bool, str]:
    """Check that adding `item` to the live set won't overshoot the
    memory/CPU budget. Returns (ok, why-not-string)."""
    # Per-patch concurrency cap: prevent N cells from the same patch
    # piling on at once. Most patches read a few big dataset files (xclim
    # 16GB .nc, statsmodels 5-7GB .npz) and HDF5 / mmap on Windows hits
    # lock contention when >2 readers open the same handle simultaneously.
    same_patch = sum(1 for c in in_flight if c.patch == item.patch)
    if same_patch >= _PER_PATCH_CONCURRENCY:
        return False, (f"patch={item.patch} already has {same_patch} "
                       f"in flight (cap={_PER_PATCH_CONCURRENCY})")
    # CPU: in-flight threads + new item's threads must stay under 80%.
    inflight_threads = sum(max(1, int(c.threads or 1)) for c in in_flight)
    new_threads = max(1, int(item.threads or 1))
    cpu_cap = max(1, int(cpu_total * _CPU_BUDGET_FRAC))
    if inflight_threads + new_threads > cpu_cap:
        return False, (f"cpu_used={inflight_threads}+need={new_threads} > "
                       f"cap={cpu_cap}")
    if not psutil_mod:
        return True, ""
    # Memory: probe live "available" and reserve a fixed buffer. The
    # cell's historical peak RSS (cost_mem_mb) is our forecast — if
    # available - item_peak would dip below the headroom, decline.
    try:
        avail_mb = psutil_mod.virtual_memory().available / (1024 * 1024)
    except Exception:
        return True, ""
    buffer_mb = total_mem_mb * (1.0 - _MEM_BUDGET_FRAC)
    needed_mb = max(0.0, float(item.cost_mem_mb or 0.0))
    if avail_mb - needed_mb < buffer_mb:
        return False, (f"mem_avail={avail_mb:.0f}MB - need={needed_mb:.0f}MB "
                       f"< buf={buffer_mb:.0f}MB")
    return True, ""


def _orchestrate(items: list[WorkItem], framework: Path, log_dir: Path,
                 *, small_workers: int, large_workers: int,
                 timeout_sec: int, use_rich: bool) -> list[ExecResult]:
    """Resource-aware dispatcher: one shared pool, dynamic admission.

    Each prospective spawn is gated by a memory + CPU-thread probe against
    the in-flight set and live OS counters. The pool capacity is the
    historical (`small_workers + large_workers`) sum, raised to a higher
    ceiling so a system with idle resources can run more small cells in
    parallel; the per-spawn probe enforces 80% RAM / 80% CPU caps so the
    user's other workloads (vegan, scvelo, etc.) keep room. When the next
    item in queue can't fit but has been head-of-queue for >15 min, we
    force-admit it so a single oversize cell never starves the queue.

    Falls back to the old "submit-everything-immediately" mode if `psutil`
    is missing (Linux/Mac CI without it), so behavior is non-blocking by
    default. Sort order is preserved — memory-first global sort means the
    cheapest cells run first."""
    requested_workers = max(1, int(small_workers) + int(large_workers))
    max_workers = max(requested_workers, _MAX_DYNAMIC_WORKERS)
    total = len(items)
    results: list[ExecResult] = []
    psutil_mod = _try_import_psutil()
    cpu_total = os.cpu_count() or 1
    if psutil_mod:
        try:
            total_mem_mb = psutil_mod.virtual_memory().total / (1024 * 1024)
        except Exception:
            total_mem_mb = 0.0
            psutil_mod = None
    else:
        total_mem_mb = 0.0

    inflight: list[WorkItem] = []
    inflight_lock = threading.Lock()

    with _ProgressDashboard(total, use_rich=use_rich) as dash:
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="cell") as pool:
            futures: list[Future] = []

            def submit_now(item: WorkItem) -> Future:
                # `inflight.append(item)` happens in the dispatcher under
                # the lock BEFORE this submit, so the per-patch cap and
                # resource probe see the item immediately. The worker
                # only removes on finish.
                def task():
                    try:
                        dash.start_item(item.label())
                        res = _execute_one(item, framework, log_dir,
                                           timeout_sec)
                        dash.finish_item(item.label(), res.rc,
                                         res.duration_sec, res.note)
                        return res
                    finally:
                        with inflight_lock:
                            try:
                                inflight.remove(item)
                            except ValueError:
                                pass
                return pool.submit(task)

            queue_idx = 0
            queue_head_wait_start: float | None = None
            while queue_idx < total:
                item = items[queue_idx]
                with inflight_lock:
                    in_flight_snapshot = list(inflight)
                ok, why = _resource_fits(
                    item, in_flight_snapshot, psutil_mod, cpu_total,
                    total_mem_mb,
                )
                # Starvation guard: if a single item has blocked dispatch
                # for >15 min on a soft resource cap (CPU/memory), force
                # admit it so the queue keeps flowing. NEVER override the
                # per-patch concurrency cap — that's a hard correctness
                # guarantee (one cell per patch at a time, to avoid
                # dataset file-lock races + concurrent same-patch R
                # sessions OOMing on huge baselines). For per-patch
                # blocks, wait indefinitely for the prior same-patch
                # cell to finish.
                now = time.monotonic()
                if not ok:
                    is_patch_block = why.startswith("patch=")
                    if is_patch_block:
                        # Hard wait: do not arm the starvation timer.
                        queue_head_wait_start = None
                    elif queue_head_wait_start is None:
                        queue_head_wait_start = now
                    elif now - queue_head_wait_start > _STARVATION_OVERRIDE_SEC:
                        ok = True
                        why = (f"starvation override after "
                               f"{(now-queue_head_wait_start):.0f}s")
                if ok:
                    # Atomically register the item as in-flight under the
                    # same lock the next probe will read — otherwise the
                    # ThreadPool's submit-to-worker latency lets the
                    # dispatcher admit a same-patch / over-budget sibling
                    # before the prior submission's task() has a chance
                    # to take the lock.
                    with inflight_lock:
                        inflight.append(item)
                    futures.append(submit_now(item))
                    queue_idx += 1
                    queue_head_wait_start = None
                    # Loop back immediately to admit more cells if budget
                    # still allows; only sleep when blocked.
                    continue
                # Blocked: wait for either a worker to free or the budget
                # to ease (user closing a side job). Re-probe periodically.
                time.sleep(_DISPATCH_POLL_SEC)

            for fut in futures:
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append(ExecResult(
                        item=items[0], rc=255, duration_sec=0.0,
                        note=f"exception {type(e).__name__}: {e}",
                    ))
    return results


# ---- top-level command -------------------------------------------------


def _print_plan_summary(items: list[WorkItem], stats: dict) -> None:
    print(f"=== backfill plan ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  work items: {len(items)} "
          f"({sum(1 for x in items if x.is_small_tier)} small/medium, "
          f"{sum(1 for x in items if not x.is_small_tier)} large/OOD)")
    if not items:
        return
    by_patch: dict[str, list[WorkItem]] = defaultdict(list)
    for it in items:
        by_patch[it.patch].append(it)
    for patch in sorted(by_patch):
        rows = by_patch[patch]
        print(f"  {patch:<32} {len(rows):>3} cells")
        for it in rows[:6]:
            print(f"    {it.tier:<10} T{it.threads:<3} "
                  f"b={it.baseline_n} p={it.patched_n} → "
                  f"target={it.target_reps} ({it.reason})")
        if len(rows) > 6:
            print(f"    ... and {len(rows)-6} more")


def cmd_backfill(args: argparse.Namespace) -> int:
    from zyme.scan import find_framework_root
    cwd = Path.cwd()
    framework = (Path(args.framework_root).resolve() if args.framework_root
                 else find_framework_root(cwd))
    if framework is None:
        print("zyme: backfill requires autozyme-framework/ — pass "
              "--framework-root or run from inside the monorepo.",
              file=sys.stderr)
        return 1

    platform_keep = args.platform
    skip_patches = set(_DEFAULT_SKIP) | set(_parse_csv(args.skip))
    only_patches = set(_parse_csv(args.only))
    tier_filter = set(_parse_csv(args.tiers)) or None
    threads_filter = None
    if args.threads:
        try:
            threads_filter = {int(x) for x in _parse_csv(args.threads)}
        except ValueError:
            print(f"zyme: bad --threads value {args.threads!r}", file=sys.stderr)
            return 1

    items, stats = _plan(
        framework,
        target_reps=args.target_reps,
        also_stabilize=args.also_stabilize,
        variance_pct=args.variance_pct,
        platform_keep=platform_keep,
        only_patches=only_patches,
        skip_patches=skip_patches,
        tier_filter=tier_filter,
        threads_filter=threads_filter,
    )

    if args.limit and args.limit > 0:
        items = items[: args.limit]

    _print_plan_summary(items, stats)
    if args.plan:
        return 0
    if not items:
        print("[complete] nothing to do")
        return 0

    log_dir = framework / "attest_logs" / "backfill"
    print(f"\nLogs: {log_dir}")
    print(f"Concurrency: {args.small_workers} small + {args.large_workers} large")
    if not _RICH_OK and not args.plain:
        print("(rich not installed — falling back to plain stdout progress)")

    # Pre-flight: if any seurat patch is in the plan, the patched
    # PCA/CCA path needs Python+numpy+scipy via reticulate. Probe it
    # before launching multi-hour cells; refuse to run if it's not
    # reachable. (Without this guard, integrate_cca/medium will silently
    # fall back to upstream and measure baseline-as-patched — incident
    # 2026-06-05 wasted 4.5h before noticing.)
    needs_py = any(_seurat_patch_needs_python(x.patch) for x in items)
    if needs_py:
        env = _subprocess_env(framework)
        ok, py_note = _preflight_seurat_python(env)
        if not ok:
            print(f"\n[preflight FAIL] {py_note}")
            print("Aborting — seurat patches need Python; would silently "
                  "fall back to upstream and corrupt speedups.")
            return 2
        print(f"[preflight OK] seurat Python: {py_note}")

    results = _orchestrate(
        items, framework, log_dir,
        small_workers=args.small_workers,
        large_workers=args.large_workers,
        timeout_sec=args.timeout,
        use_rich=(not args.plain),
    )
    failed = sum(1 for r in results if r.rc != 0)
    print(f"\n[complete] {len(results)} items, {failed} failed")
    return 1 if failed else 0


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]
