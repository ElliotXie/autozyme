"""Synthetic stdlib-only patch used exclusively by test_verify_patch.py.

Excluded from list_patches() and _AVAILABLE because the name starts with _
(see _core._populate_available). Present as a real submodule so the
verify_patch worker subprocess can find it via _import_submodule("_test_json")
without needing any optional upstream installed.
"""
from __future__ import annotations

import json
import os

import autozyme

_ORIG_DUMPS = json.dumps


def _fast_dumps(obj, **kwargs):
    kwargs.setdefault("separators", (",", ":"))
    return _ORIG_DUMPS(obj, **kwargs)


def _smoke_load(task_dir: str, tier: str) -> dict:
    return {"data": [1, 2, 3]}


def _smoke_call(inputs: dict) -> dict:
    return {"value": sum(inputs["data"])}


def _smoke_save(result: dict, output_dir: str, *, tier: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "output.txt"), "w", encoding="utf-8") as f:
        f.write(str(result["value"]) + "\n")


autozyme.register_patch(
    name="_test_json",
    targets=[("json", "dumps", _fast_dumps)],
    smoke=dict(load=_smoke_load, call=_smoke_call, save=_smoke_save),
)
