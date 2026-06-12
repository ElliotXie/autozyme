"""Resolve per-task attest smoke recipes (attest/smoke.py) vs patch defaults."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable


def resolve_smoke(task_dir: str | Path, patch: Any) -> dict[str, Callable] | None:
    """Priority: ``<task_dir>/attest/smoke.py`` defines ``smoke`` dict, else patch.smoke."""
    task_dir = Path(task_dir)
    smoke_file = task_dir / "attest" / "smoke.py"
    if smoke_file.is_file():
        spec = importlib.util.spec_from_file_location("_autozyme_task_smoke", smoke_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load attest smoke module: {smoke_file}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "smoke"):
            raise RuntimeError(
                f"attest/smoke.py must define smoke = dict(load=, call=, save=): {smoke_file}"
            )
        smoke = mod.smoke
        if not isinstance(smoke, dict):
            raise RuntimeError(f"attest/smoke.py `smoke` must be a dict: {smoke_file}")
        missing = {"load", "call", "save"} - set(smoke.keys())
        if missing:
            raise RuntimeError(
                f"attest/smoke.py smoke missing keys {sorted(missing)}: {smoke_file}"
            )
        return smoke
    return patch.smoke if patch is not None else None
