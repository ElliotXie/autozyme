"""`zyme attest` — produce the paper-headline speedup for a packaged patch.

Thin CLI shell over `autozyme.verify_patch()` (Python) / `autozyme::verify_patch()`
(R). The implementation lives in the installable autozyme_{py,r} package
because verify_patch is part of each patch's self-test contract — see
`autozyme_py/src/autozyme/_verify.py` and `autozyme_r/R/verify.R`. This
wrapper exists for CLI ergonomics: discoverable via `zyme --help`, fits the
rest of the lifecycle (init → run → ... → verify → attest → report), and
closes the loop with `zyme scan`'s `pv` column.

What gets produced:
  - Fresh-subprocess timings at 5 tiers × 2 reps (auto-escalates to 3 reps
    if the 2 reps' speedup_x disagree by >20%).
  - Per-tier row appended to `<task_dir>/package_verify.tsv`. Subsequent
    `zyme attest` calls append more rows — the file accumulates history
    like `results.tsv`.
  - On success, publishable passing rows and crash/OOM sentinel rows are
    merged into the package-bundled speedups snapshot
    (`autozyme_r/inst/speedups/...` or the Python package's `speedups.tsv`)
    so task-local `package_verify.tsv` stays the task-side source of truth.
  - Tasks declaring `threading: not_applicable` are forced to thread=1 for
    final attest; stale multi-thread rows are not published.
  - stdout/stderr from the underlying verify_patch streams through live;
    `zyme attest` exits with the worker's return code.

Distinction from `zyme verify`:
  - `zyme verify`  → Phase 3 validate_scaling. Threading × tier sweep with
                     OOD datasets, writes `verify.tsv`. Asks "does the patch
                     hold under threading and on held-out data?"
  - `zyme attest`  → Phase 4 packaging-time seal. Paper-headline number from
                     fresh-subprocess `verify_patch()`, writes
                     `package_verify.tsv`. Asks "what's the final speedup
                     the patch ships with?"
"""
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

from zyme.commands.publish_speedups import (
    _combine_for_write,
    _dest_for_patch,
    _partition_by_platform,
)
from zyme.parsers.package_verify_tsv import (
    PublishFilter,
    PublishFilterError,
    prepare_publish_content,
    prune_published_tsv_text,
    read_package_verify,
    row_all_pass,
    row_is_oom_sentinel,
    row_is_publishable,
    row_is_valid,
    summarize_batches,
)
from zyme.parsers.task_yaml import parse_executor, parse_threading_mode
from zyme.runner import _resolve_python
from zyme.scan import find_framework_root, _build_lifted_from_index
from zyme.scan_attest import _normalize_platform
from zyme.utils import task_dir_from_args, detect_lang, die


def _infer_patch_name(task_dir: Path) -> str | None:
    """Resolve registered patch name for this task.

    Two-stage resolution:

    1. **Authoritative (preferred):** scan packaged patches under the
       framework for the `Lifted from autozyme task <dir_name>` marker
       that every patch's header carries (same index `zyme scan` uses).
       This is exact: the patch file itself remembers which task it
       was lifted from.

    2. **Fallback:** parse `task.yaml::target_function` and take the
       upstream-package segment (e.g. `mgcv::gam` -> `mgcv`,
       `lifelines.CoxPHFitter.fit` -> `lifelines`). Skipped when the
       field still holds the scaffold placeholder `<PKG::FUNC>` or is
       empty.

    Returns None when neither stage produces a name; caller must then
    require `--name` explicitly.
    """
    # Stage 1: authoritative reverse index from patch headers.
    try:
        from zyme.scan import find_framework_root, _build_lifted_from_index
        framework = find_framework_root(task_dir)
        if framework:
            index = _build_lifted_from_index(str(framework))
            patch_path = index.get(task_dir.name)
            if patch_path:
                p = Path(patch_path)
                # R folder: .../inst/patches/<name>/patch.R   -> parent dir name
                # R legacy: .../inst/patches/<name>.R         -> file stem
                # Py:       .../autozyme/<name>/__init__.py   -> parent dir name
                if p.suffix == ".R" and p.name != "patch.R":
                    return p.stem
                return p.parent.name
    except Exception:
        pass

    # Stage 2: derive from task.yaml target_function (skips placeholders).
    try:
        import yaml
        with open(task_dir / "task.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        target = (cfg.get("target_function") or "").strip()
        if not target or "<" in target:
            return None
        # Registered patch dirs are lowercase (scanpy, seurat, rctd, mast...),
        # but R target_functions are CamelCase (Seurat::, RCTD::, MAST::), so
        # lowercase the upstream-package segment to match the registry.
        if "::" in target:
            return (target.split("::", 1)[0].strip() or "").lower() or None
        if "." in target:
            return (target.split(".", 1)[0].strip() or "").lower() or None
        return target.lower() or None
    except Exception:
        return None


def _python_interpreter(task_dir: Path) -> str:
    """Resolve the python executable consistent with `zyme run`.

    Honors `task.yaml::executor.python` (conda env name or absolute path);
    falls back to `sys.executable` when unset. Mirrors `build_reference_cmd`.
    """
    try:
        executor = parse_executor(task_dir / "task.yaml")
    except Exception:
        executor = {}
    spec = executor.get("python")
    if not spec:
        return sys.executable
    return _resolve_python(spec)


def _rscript_interpreter(task_dir: Path) -> str:
    """Resolve Rscript honoring `executor.rscript`."""
    try:
        executor = parse_executor(task_dir / "task.yaml")
    except Exception:
        executor = {}
    return executor.get("rscript", "Rscript")


def _resolve_one_task_dir(raw: str) -> Path:
    """Resolve a positional task-dir argument; die clearly on bad input."""
    p = Path(raw).resolve()
    if not p.is_dir():
        die(f"not a directory: {raw}")
    if not (p / "task.yaml").exists():
        die(f"not a zyme task directory (no task.yaml): {p}")
    return p


# Keep in sync with autozyme.verify_patch's default tiers (Python: _verify.py,
# R: verify.R). Used for --skip-tiers and for cached-OOM tier filtering.
_DEFAULT_TIERS = ("tiny", "medium", "large", "ood_large", "ood_xlarge")


def _parse_threads_arg(raw: str | None) -> list[int | None]:
    """Parse --threads. None means defer to task.yaml/env default."""
    if raw is None:
        return [None]
    vals: list[int] = []
    seen: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"full", "max", "all"}:
            threads = os.cpu_count() or 1
        else:
            try:
                threads = int(token)
            except ValueError:
                die(
                    f"--threads value {part!r} is not an integer or one of "
                    "'full', 'max', 'all'"
                )
            if threads < 1:
                die(f"--threads values must be >= 1, got {threads}")
        if threads not in seen:
            seen.add(threads)
            vals.append(threads)
    if not vals:
        die("--threads parsed to empty after splitting")
    return vals


def _thread_label(threads: int | None) -> str:
    return "task-default" if threads is None else f"{threads}t"


def _thread_values_for_task(
    task_dir: Path, requested: list[int | None], *, allow_not_applicable: bool,
) -> list[int | None]:
    """Enforce task-level thread policy before launching verify_patch."""
    if parse_threading_mode(task_dir / "task.yaml") != "not_applicable":
        return requested

    requested_multi = [
        str(t) for t in requested
        if t is not None and int(t) > 1
    ]
    if allow_not_applicable:
        if requested_multi:
            print(
                "[attest] WARNING: --allow-not-applicable-threads set; "
                f"{task_dir.name} declares threading: not_applicable, but "
                f"will run requested multi-thread attest rows: "
                f"{', '.join(requested_multi)}",
                file=sys.stderr,
            )
        return requested

    dropped = [
        str(t) for t in requested
        if t is not None and int(t) > 1
    ]
    if dropped or requested != [1]:
        detail = (
            f"; skipped requested threads {', '.join(dropped)}"
            if dropped else ""
        )
        print(
            "[attest] threading: not_applicable in task.yaml — forcing "
            f"{task_dir.name} to 1t{detail}",
            file=sys.stderr,
        )
    return [1]


def _parse_tiers_selection(
    tiers_arg: str | None,
    skip_tiers_arg: str | None,
) -> tuple[str, ...] | None:
    """Return explicit tiers, or None when verify_patch should use defaults."""
    if tiers_arg and skip_tiers_arg:
        die("--tiers and --skip-tiers are mutually exclusive")

    if tiers_arg:
        tiers = tuple(t.strip() for t in tiers_arg.split(",") if t.strip())
        if not tiers:
            die("--tiers parsed to empty after splitting; use comma-separated names")
        return tiers

    if skip_tiers_arg:
        skip = {s.strip() for s in skip_tiers_arg.split(",") if s.strip()}
        if not skip:
            die("--skip-tiers parsed to empty after splitting; use comma-separated names")
        unknown = skip - set(_DEFAULT_TIERS)
        if unknown:
            die(
                f"--skip-tiers contains unknown tier(s): {sorted(unknown)}. "
                f"Known defaults: {list(_DEFAULT_TIERS)}"
            )
        tiers = tuple(t for t in _DEFAULT_TIERS if t not in skip)
        if not tiers:
            die("--skip-tiers excluded every default tier; nothing left to run")
        return tiers

    return None


def _publish_filter_for_task(
    task_dir: Path, *, all_pass_only: bool, allow_not_applicable: bool,
) -> PublishFilter:
    max_threads = (
        1 if (
            not allow_not_applicable
            and parse_threading_mode(task_dir / "task.yaml") == "not_applicable"
        )
        else None
    )
    return PublishFilter(
        select="full",
        all_pass_only=all_pass_only,
        max_threads=max_threads,
    )


_stale_rlibs_warned: set[str] = set()


def _warn_if_stale_rlibs(framework: Path, local_rlibs: Path) -> None:
    """Warn once per process if .R_libs/autozyme is older than autozyme_r source.

    Silent staleness here is dangerous: the subprocess loads the old autozyme
    while foreground / direct invocations load the user-lib version, so bugs
    fixed in source disappear from foreground tests but resurface only via
    `zyme attest` (real incident 2026-05-31: WGCNA Win@4t attest silently
    dropped pv.tsv writes for a week because .R_libs/autozyme/verify.R was
    pre-CR-stripping). Respect AUTOZYME_SKIP_RLIBS_CHECK=1 to silence.

    Compares mtime of <local_rlibs>/autozyme/DESCRIPTION against the newest
    mtime in autozyme_r/{R,src,inst,DESCRIPTION}. Warns to stderr if stale.
    """
    if os.environ.get("AUTOZYME_SKIP_RLIBS_CHECK", "").lower() in {"1", "true", "yes"}:
        return
    desc = local_rlibs / "autozyme" / "DESCRIPTION"
    src_root = framework / "autozyme_r"
    if not desc.is_file() or not src_root.is_dir():
        return
    key = str(desc.resolve())
    if key in _stale_rlibs_warned:
        return
    # Only check user-editable source files. Build artifacts (.o, .so, .dll,
    # .rdb, .rds) under src/ get their mtime bumped by R CMD INSTALL itself,
    # which would make every freshly-installed lib look "stale" relative to
    # its own build outputs.
    source_suffixes = {".R", ".r", ".Rd", ".cpp", ".cc", ".c", ".h", ".hpp"}
    try:
        install_mtime = desc.stat().st_mtime
        newest_src = install_mtime
        for sub in ("R", "src", "inst"):
            d = src_root / sub
            if not d.is_dir():
                continue
            for p in d.rglob("*"):
                if p.is_file() and p.suffix in source_suffixes:
                    m = p.stat().st_mtime
                    if m > newest_src:
                        newest_src = m
        desc_src = src_root / "DESCRIPTION"
        if desc_src.is_file() and desc_src.stat().st_mtime > newest_src:
            newest_src = desc_src.stat().st_mtime
    except OSError:
        return
    if newest_src > install_mtime + 1:  # 1s slack to avoid filesystem jitter
        from datetime import datetime
        _stale_rlibs_warned.add(key)
        install_iso = datetime.fromtimestamp(install_mtime).isoformat(timespec="seconds")
        newest_iso = datetime.fromtimestamp(newest_src).isoformat(timespec="seconds")
        print(
            f"[attest] WARNING: framework-local autozyme install at "
            f"{local_rlibs / 'autozyme'} is older than autozyme_r/ source:\n"
            f"  install DESCRIPTION mtime: {install_iso}\n"
            f"  newest source mtime:       {newest_iso}\n"
            f"  This can cause silent worker bugs (e.g., dropped pv.tsv writes).\n"
            f"  Fix: R CMD INSTALL --library={local_rlibs} {src_root}\n"
            f"  Silence: export AUTOZYME_SKIP_RLIBS_CHECK=1",
            file=sys.stderr,
        )


def _attest_env(
    task_dir: Path,
    lang: str,
    patch_name: str | None,
    threads: int | None = None,
) -> dict[str, str]:
    """Framework-local R_LIBS (R) + thread caps from task.yaml (all langs)."""
    env = os.environ.copy()
    if lang == "R":
        try:
            from zyme.scan import find_framework_root
            framework = find_framework_root(task_dir)
            if framework:
                local = Path(framework) / ".R_libs"
                if local.is_dir():
                    prev = env.get("R_LIBS", "")
                    env["R_LIBS"] = str(local) + (os.pathsep + prev if prev else "")
                    _warn_if_stale_rlibs(Path(framework), local)
        except Exception:
            pass
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "autozyme_py" / "src"))
        if threads is None:
            from autozyme._thread_env import apply_task_thread_env
            apply_task_thread_env(env, task_dir, patch_name=patch_name)
        else:
            from autozyme._thread_env import apply_thread_env
            apply_thread_env(env, threads, scanpy_turbo=(patch_name == "scanpy"))
    except Exception:
        fallback = str(threads or 1)
        env.setdefault("ZYME_THREADS", fallback)
        env.setdefault("OMP_NUM_THREADS", fallback)
        env.setdefault("OPENBLAS_NUM_THREADS", fallback)
        env.setdefault("MKL_NUM_THREADS", fallback)
    return env


def _detect_cpu_model() -> str:
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
            if out:
                return out
        if sysname == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sysname == "Windows":
            out = subprocess.run(
                ["wmic", "cpu", "get", "name", "/value"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            for line in out.splitlines():
                if line.startswith("Name="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or ""


def _detect_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass

    sysname = platform.system()
    try:
        if sysname == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
            return round(int(out) / (1024 ** 3), 1) if out else None
        if sysname == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / (1024 ** 2), 1)
    except Exception:
        return None
    return None


def _parse_int_cell(raw: str | None) -> int | None:
    s = (raw or "").strip()
    if not s or s.upper() in {"NA", "NAN", "NONE"}:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _parse_float_cell(raw: str | None) -> float | None:
    s = (raw or "").strip()
    if not s or s.upper() in {"NA", "NAN", "NONE"}:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _current_attempt_fingerprint(env: dict[str, str]) -> dict[str, object]:
    threads = (
        env.get("ZYME_THREADS")
        or env.get("AUTOZYME_THREADS")
        or env.get("AUTOZYMER_THREADS")
        or env.get("OMP_NUM_THREADS")
        or env.get("MKL_NUM_THREADS")
        or env.get("OPENBLAS_NUM_THREADS")
        or ""
    )
    return {
        "platform": _normalize_platform(platform.system()),
        "cpu": _detect_cpu_model(),
        "ram_gb": _detect_ram_gb(),
        "threads": _parse_int_cell(threads),
    }


def _same_host_thread(row: dict[str, str], current: dict[str, object]) -> bool:
    if _normalize_platform(row.get("system_os")) != current.get("platform"):
        return False

    row_cpu = (row.get("system_cpu") or "").strip()
    cur_cpu = str(current.get("cpu") or "").strip()
    if row_cpu and cur_cpu and row_cpu != cur_cpu:
        return False

    row_ram = _parse_float_cell(row.get("system_ram_gb"))
    cur_ram = current.get("ram_gb")
    if row_ram is not None and isinstance(cur_ram, (int, float)):
        if abs(row_ram - float(cur_ram)) > 0.5:
            return False

    row_threads = _parse_int_cell(row.get("system_threads"))
    cur_threads = current.get("threads")
    if row_threads is not None and isinstance(cur_threads, int):
        if row_threads != cur_threads:
            return False

    return True


def _cached_oom_tiers(
    task_dir: Path,
    patch_name: str,
    tiers: tuple[str, ...],
    current: dict[str, object],
) -> list[str]:
    rows = _read_verify_rows(task_dir / "package_verify.tsv")
    if not rows:
        return []

    wanted = set(tiers)
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        tier = (row.get("tier") or "").strip()
        if tier not in wanted:
            continue
        row_patch = (row.get("patch_name") or "").strip()
        if row_patch and row_patch != patch_name:
            continue
        if not _same_host_thread(row, current):
            continue
        if not row_is_publishable(row):
            continue
        ts = (row.get("timestamp") or "").strip()
        if not ts:
            continue
        prev = latest.get(tier)
        if prev is None or (prev.get("timestamp") or "").strip() <= ts:
            latest[tier] = row

    return [tier for tier in tiers if row_is_oom_sentinel(latest.get(tier, {}))]


def _build_attest_cmd(task_dir: Path, name: str | None, tiers_arg: str | None,
                       skip_tiers_arg: str | None,
                       reps: int, lang_arg: str | None,
                       *,
                       rerun_baseline: bool = False,
                       baseline_confirm_sigma: float = 3.0,
                       no_baseline_confirm: bool = False,
                       patched_only: bool = False,
                       tiers_override: tuple[str, ...] | None = None,
                       ) -> tuple[list[str], str]:
    """Build the underlying interpreter command for a single task.

    Returns (cmd_argv, resolved_patch_name). Resolves language, infers
    patch name, parses --tiers / --skip-tiers, picks the right interpreter.
    Propagates --rerun-baseline / --baseline-confirm-sigma /
    --no-baseline-confirm into the verify_patch invocation kwargs.
    """
    lang = lang_arg or detect_lang(task_dir)

    patch_name = name or _infer_patch_name(task_dir)
    if not patch_name:
        die(
            f"could not infer patch name for {task_dir.name}; "
            f"pass --name explicitly (e.g. `zyme attest --name mgcv`)"
        )

    tiers = (
        tiers_override
        if tiers_override is not None
        else _parse_tiers_selection(tiers_arg, skip_tiers_arg)
    )

    if lang == "py":
        kwargs = [f"reps={reps}"]
        if tiers is not None:
            kwargs.append(f"tiers={tiers!r}")
        # Baseline-cache kwargs: only emit when non-default to keep the
        # dry-run output uncluttered for the common case.
        if rerun_baseline:
            kwargs.append("use_baseline_cache=False")
        if no_baseline_confirm:
            kwargs.append("no_baseline_confirm=True")
        if baseline_confirm_sigma != 3.0:
            kwargs.append(f"baseline_confirm_sigma={float(baseline_confirm_sigma)}")
        if patched_only:
            kwargs.append("patched_only=True")
        code = (
            f"import autozyme; autozyme.verify_patch("
            f"{patch_name!r}, {str(task_dir)!r}, {', '.join(kwargs)})"
        )
        cmd = [_python_interpreter(task_dir), "-c", code]
    elif lang == "R":
        kwargs = [f"reps = {reps}L"]
        if tiers is not None:
            r_tiers = "c(" + ", ".join(repr(t) for t in tiers) + ")"
            kwargs.append(f"tiers = {r_tiers}")
        if no_baseline_confirm:
            kwargs.append("no_baseline_confirm = TRUE")
        if patched_only:
            kwargs.append("patched_only = TRUE")
        # R verify_patch supports skipping baseline confirmation, but not the
        # Python baseline drift-confirmation knobs.
        if rerun_baseline or baseline_confirm_sigma != 3.0:
            import sys as _sys
            print(
                "zyme: warning: --rerun-baseline / --baseline-confirm-sigma "
                "are Python-only; ignored for R task",
                file=_sys.stderr,
            )
        verify_call = (
            f"autozyme::verify_patch({patch_name!r}, {str(task_dir)!r}, "
            f"{', '.join(kwargs)})"
        )
        code = (
            ".autozyme_status <- 0L; "
            f"tryCatch({{ {verify_call} }}, error = function(e) {{ "
            "message(conditionMessage(e)); .autozyme_status <<- 1L }); "
            "if (.Platform$OS.type == 'windows') { "
            "flush.console(); "
            "system2('taskkill', c('/PID', as.character(Sys.getpid()), '/F'), "
            "stdout = FALSE, stderr = FALSE, wait = FALSE); "
            "Sys.sleep(60) }; "
            "quit(save = 'no', status = .autozyme_status, runLast = FALSE)"
        )
        cmd = [_rscript_interpreter(task_dir), "-e", code]
    else:
        die(f"unsupported task language {lang!r} (expected 'py' or 'R')")

    return cmd, patch_name


def _resolve_manifest_publish_target(task_dir: Path) -> tuple[str, Path] | None:
    """Resolve per-method Seurat/Scanpy package speedup destination.

    Seurat and Scanpy are one logical patch with many per-step attest tasks, so
    the generic lifted-from patch index is not expressive enough. Their
    manifest maps each task directory to the package-bundled method TSV.
    """
    framework = find_framework_root(task_dir)
    if framework is None:
        return None
    framework = Path(framework).resolve()
    # Per-task patch dir layout (post-2026-05-28):
    #   Seurat tasks: autozyme_r/inst/patches/seurat_<task>/speedups.tsv
    #   Scanpy tasks: autozyme_py/src/autozyme/scanpy_<task>/speedups.tsv
    manifests = (
        (
            framework / "scripts" / "seurat_attest_manifest.yaml",
            framework / "autozyme_r" / "inst" / "patches",
            "seurat",
        ),
        (
            framework / "scripts" / "scanpy_attest_manifest.yaml",
            framework / "autozyme_py" / "src" / "autozyme",
            "scanpy",
        ),
    )
    task_resolved = task_dir.resolve()
    try:
        import yaml
    except ImportError:
        return None
    for manifest_path, base_dir, prefix in manifests:
        if not manifest_path.is_file():
            continue
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            continue
        for task in manifest.get("tasks", []):
            rel = task.get("path")
            if not rel:
                continue
            candidate = (framework / rel).resolve()
            if candidate != task_resolved:
                continue
            if task.get("package_speedups") is False:
                return None
            legacy = task.get("legacy_key") or task.get("id")
            if not legacy:
                return None
            dest = base_dir / f"{prefix}_{legacy}" / "speedups.tsv"
            return str(task.get("id") or legacy), dest
    return None


def _resolve_generic_publish_target(task_dir: Path) -> tuple[str, Path] | None:
    """Resolve standard one-task/one-patch bundled speedup destination."""
    framework = find_framework_root(task_dir)
    if framework is None:
        return None
    framework = Path(framework).resolve()
    index = _build_lifted_from_index(str(framework))
    patch_path_str = index.get(task_dir.name)
    if not patch_path_str:
        return None
    patch_path = Path(patch_path_str)
    return task_dir.name, _dest_for_patch(framework, patch_path)


def _publish_after_attest(
    task_dir: Path, *, allow_not_applicable_threads: bool = False,
) -> bool:
    """Publish task package_verify.tsv into the package-bundled speedups slot."""
    # `AUTOZYME_ATTEST_PUBLISH_MODE=skip` lets callers (e.g. `zyme backfill`)
    # bypass attest's auto-publish entirely. Backfill needs this because it
    # runs its own shard-aware pv-copy + finalize per cell; letting attest
    # also publish would (a) re-write the legacy combined speedups.tsv that
    # the framework just retired, and (b) dump the entire pv.tsv history
    # (including stale tier="tiny" rows already cleaned out upstream) back
    # into the shards.
    if os.environ.get("AUTOZYME_ATTEST_PUBLISH_MODE", "").lower() == "skip":
        return False
    src = task_dir / "package_verify.tsv"
    if not src.is_file():
        print(
            f"[publish-speedups] skipped; missing {src.name}",
            file=sys.stderr,
        )
        return False

    resolved = (
        _resolve_manifest_publish_target(task_dir)
        or _resolve_generic_publish_target(task_dir)
    )
    if resolved is None:
        print(
            "[publish-speedups] skipped; no package speedups destination "
            f"for task {task_dir}",
            file=sys.stderr,
        )
        return False
    task_id, dst = resolved

    try:
        publish_filter = _publish_filter_for_task(
            task_dir,
            all_pass_only=True,
            allow_not_applicable=allow_not_applicable_threads,
        )
        if (
            allow_not_applicable_threads
            and parse_threading_mode(task_dir / "task.yaml") == "not_applicable"
        ):
            print(
                "[publish-speedups] WARNING: "
                "--allow-not-applicable-threads set; publishing multi-thread "
                f"rows for {task_dir.name} if present",
                file=sys.stderr,
            )
        new_text, n_rows, summary = prepare_publish_content(
            src,
            publish_filter,
        )
        if n_rows == 0:
            print(
                f"[publish-speedups] skipped; no publishable rows in {src}",
                file=sys.stderr,
            )
            return False
        # Per-platform split: partition the publishable rows by system_os and
        # merge each platform into its OWN speedups.<plat>.tsv shard (mac/win/
        # other) so a machine only ever writes its own shard and cross-machine
        # git merges stay disjoint. Each shard path is the legacy dst with the
        # platform infixed (speedups.tsv -> speedups.<plat>.tsv).
        #
        # Default merge: replace rows sharing the (tier,OS,CPU,threads,fw)
        # signature so multiple attests don't bloat the bundled TSV with
        # near-duplicate rows. Backfill needs the opposite — every new rep
        # must be retained so finalize can aggregate n_reps. We honor
        # `AUTOZYME_ATTEST_PUBLISH_MODE=append` (or overwrite) as an opt-in
        # override; anything else stays on merge.
        publish_mode = os.environ.get("AUTOZYME_ATTEST_PUBLISH_MODE", "merge")
        if publish_mode not in ("merge", "append", "overwrite"):
            publish_mode = "merge"
        for plat, plat_text in _partition_by_platform(new_text).items():
            # Each platform shard is written independently: a stale/legacy
            # shard on one platform (e.g. a pre-`package_version` mac/win shard
            # whose header no longer row-merges) must not abort the platforms
            # we *can* write. The current run's own platform (e.g. linux) still
            # lands even when historical mac/win rows in package_verify.tsv hit
            # a header mismatch on their shards.
            try:
                n_plat = max(0, len(plat_text.splitlines()) - 1)
                plat_dst = dst.with_name(f"{dst.stem}.{plat}{dst.suffix}")
                existing_disk_text = (
                    plat_dst.read_text(encoding="utf-8") if plat_dst.is_file() else ""
                )
                existing_text, pruned_rows = prune_published_tsv_text(
                    existing_disk_text, max_threads=publish_filter.max_threads,
                )
                final_text, action_summary, total_rows = _combine_for_write(
                    existing_text, plat_text, n_plat, publish_mode,
                )
                if plat_dst.is_file() and final_text == existing_disk_text:
                    print(
                        f"[publish-speedups] already up to date: {plat_dst}",
                        file=sys.stderr,
                    )
                    continue
                plat_dst.parent.mkdir(parents=True, exist_ok=True)
                plat_dst.write_text(final_text, encoding="utf-8")
                prune_summary = (
                    f"pruned stale rows={pruned_rows} | " if pruned_rows else ""
                )
                print(
                    f"[publish-speedups] {task_id} [{plat}]: {summary} | "
                    f"{prune_summary}{action_summary} -> {plat_dst} "
                    f"({total_rows} rows)",
                    file=sys.stderr,
                )
            except (OSError, PublishFilterError, ValueError) as e:
                print(
                    f"[publish-speedups] {task_id} [{plat}]: skipped ({e})",
                    file=sys.stderr,
                )
                continue
    except (OSError, PublishFilterError, ValueError) as e:
        print(f"[publish-speedups] skipped; {e}", file=sys.stderr)
        return False
    return True


def _preflight_check_tsv_header(tsv_path: Path, lang: str) -> None:
    """Warn early if package_verify.tsv has a header the installed package
    may not recognize, instead of discovering the mismatch after hours of
    benchmarking."""
    if not tsv_path.is_file():
        return
    try:
        first_line = tsv_path.read_text(encoding="utf-8").split("\n", 1)[0].strip()
    except OSError:
        return
    if not first_line:
        return
    on_disk_cols = set(first_line.replace("﻿", "").split("\t"))
    if lang == "R":
        # The R package's current header; keep in sync with autozyme_r/R/verify.R
        known_cols = {
            "timestamp", "patch_name", "tier", "dataset",
            "rep_idx", "variant", "sec", "speedup_pct", "speedup_x",
            "peak_mb", "peak_mb_change_pct", "peak_mb_fold",
            "pass", "metrics_json", "framework_version", "package_version", "note",
            "system_os", "system_cpu", "system_ram_gb", "system_threads",
        }
    else:
        known_cols = on_disk_cols  # Python package validates its own header
    extra = on_disk_cols - known_cols
    if extra:
        print(
            f"[attest] warning: {tsv_path.name} has columns not in "
            f"this build's schema: {', '.join(sorted(extra))}. "
            "If write fails, reinstall the autozyme package from source.",
            file=sys.stderr,
        )


def _file_stamp(path: Path) -> tuple[int, int]:
    """Return a cheap change token for an optional file."""
    try:
        st = path.stat()
    except OSError:
        return (0, -1)
    return (st.st_mtime_ns, st.st_size)


def _read_verify_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        _header, rows = read_package_verify(path)
    except (OSError, PublishFilterError, ValueError):
        return []
    return rows


def _new_attest_rows_all_pass(
    before_rows: list[dict[str, str]],
    after_rows: list[dict[str, str]],
) -> bool:
    """True when this invocation added complete, passing measurement rows.

    This is intentionally stricter than "the file changed": crash sentinels,
    incomplete batches, and any failing patched rep all return False. It lets
    the CLI tolerate Windows/R teardown crashes after verify_patch has already
    written a fully passing attest batch, without masking real attest failures.
    """
    before_timestamps = {
        (r.get("timestamp") or "").strip()
        for r in before_rows
        if (r.get("timestamp") or "").strip()
    }
    new_timestamps = {
        (r.get("timestamp") or "").strip()
        for r in after_rows
        if (r.get("timestamp") or "").strip()
        and (r.get("timestamp") or "").strip() not in before_timestamps
    }
    if not new_timestamps:
        return False

    new_rows = [
        r for r in after_rows
        if (r.get("timestamp") or "").strip() in new_timestamps
    ]
    if not new_rows or any(not row_is_valid(r) for r in new_rows):
        return False

    new_patched = [
        r for r in new_rows
        if (r.get("variant") or "").strip() == "patched"
    ]
    if new_patched and all(row_all_pass(r) for r in new_patched):
        return True

    summaries = [
        b for b in summarize_batches(after_rows)
        if b.timestamp in new_timestamps
    ]
    if not summaries:
        return False
    return all(b.all_pass for b in summaries)


def _fmt_duration(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    return f"{secs / 3600:.2f}h"


def cmd_attest(args):
    # Resolve target task dirs: positional args win; else --task-dir / cwd.
    raw_dirs: list[str] = list(getattr(args, "task_dirs", None) or [])
    if raw_dirs:
        task_dirs = [_resolve_one_task_dir(p) for p in raw_dirs]
    else:
        task_dirs = [task_dir_from_args(args)]

    # Preflight gate (default ON). Runs lint + portability scan + smoke-parity
    # for each task; bail before measurement if any FAIL. Cheap relative to
    # attest itself; catches the bulk of "patch ships but never intercepted /
    # smoke times wrong region" failures before they consume tier-matrix time.
    if not getattr(args, "no_preflight", False) and not getattr(args, "dry_run", False):
        from types import SimpleNamespace
        from zyme.commands.package.preflight import cmd_package_preflight
        for td in task_dirs:
            pre_args = SimpleNamespace(
                task_dir=str(td),
                continue_on_fail=False,
                skip_parity=False,
                lang=getattr(args, "lang", None),
            )
            rc = cmd_package_preflight(pre_args)
            if rc != 0:
                die(
                    f"preflight failed for {td.name}; refusing to measure. "
                    f"Fix the findings or re-run with --no-preflight to bypass."
                )

    reps = int(args.reps)
    if reps < 1:
        die(f"--reps must be >= 1, got {reps}")
    thread_values = _parse_threads_arg(getattr(args, "threads", None))
    allow_not_applicable_threads = bool(
        getattr(args, "allow_not_applicable_threads", False)
    )

    # --name only makes sense with a single task (different tasks register
    # different patches). Enforce that to avoid silent footguns.
    if args.name and len(task_dirs) > 1:
        die("--name is only valid when targeting a single task")

    batch = len(task_dirs) > 1 or len(thread_values) > 1
    failures: list[tuple[str, int]] = []
    overall_start = time.time()
    planned_runs = 0

    for i, td in enumerate(task_dirs, 1):
        task_thread_values = _thread_values_for_task(
            td,
            thread_values,
            allow_not_applicable=allow_not_applicable_threads,
        )
        planned_runs += len(task_thread_values)
        patch_name = args.name or _infer_patch_name(td)
        if not patch_name:
            die(
                f"could not infer patch name for {td.name}; "
                f"pass --name explicitly (e.g. `zyme attest --name mgcv`)"
            )
        selected_tiers = _parse_tiers_selection(
            args.tiers, getattr(args, "skip_tiers", None)
        )
        lang = args.lang or detect_lang(td)
        retry_oom = bool(getattr(args, "retry_oom", False))
        # patched-only never runs the baseline, so the cached-OOM gate (which
        # tracks baseline OOMs) does not apply — bypass it like --retry-oom.
        patched_only = bool(getattr(args, "patched_only", False))

        for thread_idx, threads in enumerate(task_thread_values, 1):
            label = _thread_label(threads)
            run_name = f"{td.name}@{label}"
            env = _attest_env(td, lang, patch_name, threads)
            tiers_for_cache = selected_tiers or _DEFAULT_TIERS
            oom_tiers = [] if (retry_oom or patched_only) else _cached_oom_tiers(
                td,
                patch_name,
                tiers_for_cache,
                _current_attempt_fingerprint(env),
            )
            tiers_override = selected_tiers
            if oom_tiers:
                remaining = tuple(t for t in tiers_for_cache if t not in oom_tiers)
                print(
                    "[attest] cached OOM on this machine; skipping "
                    f"{td.name} tier(s) {', '.join(oom_tiers)} at {label}. "
                    "Use --retry-oom to force a rerun.",
                    file=sys.stderr,
                )
                if not remaining:
                    if args.dry_run:
                        print(
                            "# skipped; cached OOM for all requested tiers: "
                            + ",".join(oom_tiers)
                        )
                    else:
                        _publish_after_attest(
                            td,
                            allow_not_applicable_threads=(
                                allow_not_applicable_threads
                            ),
                        )
                    continue
                tiers_override = remaining

            cmd, _patch_name = _build_attest_cmd(
                td, patch_name, args.tiers, getattr(args, "skip_tiers", None),
                reps, args.lang,
                rerun_baseline=bool(getattr(args, "rerun_baseline", False)),
                baseline_confirm_sigma=float(
                    getattr(args, "baseline_confirm_sigma", 3.0)
                ),
                no_baseline_confirm=bool(
                    getattr(args, "no_baseline_confirm", False)
                ),
                patched_only=bool(getattr(args, "patched_only", False)),
                tiers_override=tiers_override,
            )

            if args.dry_run:
                prefix = "" if threads is None else f"ZYME_THREADS={threads} "
                print(prefix + shlex.join(cmd))
                continue

            if batch:
                print(
                    f"\n==================== [{i}/{len(task_dirs)} task, "
                    f"{thread_idx}/{len(task_thread_values)} thread] "
                    f"{td.name}  (patch={patch_name}, threads={label}) "
                    "====================",
                    file=sys.stderr,
                )
            print(f"+ {shlex.join(cmd)}", file=sys.stderr)

            package_verify = td / "package_verify.tsv"
            _preflight_check_tsv_header(package_verify, lang)
            before_stamp = _file_stamp(package_verify)
            before_rows = _read_verify_rows(package_verify)
            t0 = time.time()
            rc = subprocess.call(cmd, env=env)
            dur = time.time() - t0
            after_stamp = _file_stamp(package_verify)
            after_rows = _read_verify_rows(package_verify)
            wrote_passing_rows = (
                after_stamp != before_stamp
                and _new_attest_rows_all_pass(before_rows, after_rows)
            )

            status = "OK" if (rc == 0 or wrote_passing_rows) else f"FAIL rc={rc}"
            if batch:
                print(
                    f"[{run_name}] {status} in {_fmt_duration(dur)}",
                    file=sys.stderr,
                )
            if rc != 0:
                if wrote_passing_rows:
                    print(
                        f"zyme: warning: worker exited rc={rc} after writing "
                        "complete passing package_verify rows; treating attest "
                        "as complete",
                        file=sys.stderr,
                    )
                else:
                    failures.append((run_name, rc))
                    continue
            if patched_only:
                # patched-only rows carry no speedup/pass, so they're filtered
                # out by the all-pass-only publish anyway — and re-running the
                # publish re-reads the full package_verify history, which can
                # re-introduce stale rows and churn the shards. The measurement
                # lives in package_verify.tsv; skip the canonical publish.
                print(
                    "[publish-speedups] skipped (patched-only); measurement "
                    f"recorded in {package_verify}",
                    file=sys.stderr,
                )
            elif after_stamp != before_stamp:
                _publish_after_attest(
                    td,
                    allow_not_applicable_threads=allow_not_applicable_threads,
                )
            else:
                print(
                    f"[publish-speedups] skipped; {package_verify} was not updated",
                    file=sys.stderr,
                )

    if args.dry_run:
        return

    if batch:
        elapsed = _fmt_duration(time.time() - overall_start)
        total_runs = planned_runs
        ok = total_runs - len(failures)
        print(
            f"\n==================== SUMMARY: {ok}/{total_runs} OK"
            f"  ({elapsed} total) ====================",
            file=sys.stderr,
        )
        if failures:
            for name, rc in failures:
                print(f"  FAIL  {name}  rc={rc}", file=sys.stderr)
            sys.exit(1)
        return

    # Single-task semantics: propagate the worker's exit code.
    if failures:
        sys.exit(failures[0][1])
