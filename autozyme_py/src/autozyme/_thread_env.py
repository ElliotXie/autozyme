"""Thread-cap helpers for verify_patch / zyme attest / legacy-benchmark parity.

Legacy Scanpy/Seurat turbo drivers pin BLAS/numba/FAISS threads via env vars
before each fresh subprocess. Attest must set the same caps (from
task.yaml::baseline_threads[0], default 1) so speedups are comparable.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

STANDARD_THREAD_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

SCANPY_TURBO_THREAD_VARS: tuple[str, ...] = (
    "NUMBA_NUM_THREADS",
    "SCANPY_TURBO_THREADS",
)


def apply_thread_env(
    env: dict[str, str],
    threads: int,
    *,
    scanpy_turbo: bool = False,
) -> dict[str, str]:
    """Return *env* with ZYME + BLAS (+ optional Scanpy turbo) caps set."""
    t = str(max(1, int(threads)))
    env["ZYME_THREADS"] = t
    env["AUTOZYME_THREADS"] = t
    for var in STANDARD_THREAD_VARS:
        env[var] = t
    if scanpy_turbo:
        for var in SCANPY_TURBO_THREAD_VARS:
            env[var] = t
    return env


def _parse_baseline_threads_from_yaml(task_yaml: Path) -> int | None:
    if not task_yaml.is_file():
        return None
    text = task_yaml.read_text(encoding="utf-8")
    m = re.search(r"^baseline_threads\s*:\s*\[(.*?)\]\s*$", text, re.MULTILINE)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return None
    first = inner.split(",")[0].strip()
    try:
        return max(1, int(first))
    except (ValueError, TypeError):
        return None


def resolve_baseline_threads(task_dir: str | os.PathLike, default: int = 1) -> int:
    """Resolve thread count: explicit ZYME_THREADS env, else task.yaml, else default."""
    raw = os.environ.get("ZYME_THREADS") or os.environ.get("AUTOZYME_THREADS")
    if raw:
        try:
            return max(1, int(raw))
        except (ValueError, TypeError):
            pass
    parsed = _parse_baseline_threads_from_yaml(Path(task_dir) / "task.yaml")
    if parsed is not None:
        return parsed
    return max(1, int(default))


def apply_task_thread_env(
    env: dict[str, str],
    task_dir: str | os.PathLike,
    *,
    patch_name: str | None = None,
    default: int = 1,
) -> int:
    """Apply thread caps from task.yaml (unless ZYME_THREADS already set). Returns threads."""
    threads = resolve_baseline_threads(task_dir, default=default)
    apply_thread_env(env, threads, scanpy_turbo=(patch_name == "scanpy"))
    return threads


def ensure_process_thread_env(
    task_dir: str | os.PathLike,
    *,
    patch_name: str | None = None,
    default: int = 1,
) -> int:
    """Apply thread caps to ``os.environ`` for this process + child subprocesses."""
    env = os.environ
    threads = resolve_baseline_threads(task_dir, default=default)
    apply_thread_env(env, threads, scanpy_turbo=(patch_name == "scanpy"))
    return threads
