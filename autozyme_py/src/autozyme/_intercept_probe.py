"""Intercept-counting probe for `zyme package check-intercept`.

Activated by `_verify_worker` when the env var ``ZYME_INSTRUMENT_INTERCEPTS=1``
is set. Monkey-patches ``autozyme._core._make_dispatcher`` so that every fast
function dispatched through the patch's targets is wrapped in a counter. At
process exit, writes ``{target_key: count}`` JSON to the path in
``ZYME_INTERCEPT_OUT``.

This module must be imported *before* ``autozyme.activate(name)`` runs — once
the patch has been activated, ``_make_dispatcher`` has already produced
dispatchers and our wrapper would be too late.

Counter keys: ``"<upstream_path>:<attr>"`` (e.g. ``"scanpy.preprocessing:scale"``).
"""
from __future__ import annotations

import atexit
import functools
import json
import os
from pathlib import Path
from typing import Any, Callable

from autozyme import _core


_COUNTS: dict[str, int] = {}
# Maps id(fast_fn) -> target_key so the wrapped dispatcher can find which key
# to increment. Populated when register_patch passes fast_fn through
# _make_dispatcher AFTER we've recorded the target.
_KEYS: dict[int, str] = {}
_INSTALLED = False
_ORIG_MAKE_DISPATCHER: Callable | None = None


def _key_for_fast_fn(fast_fn: Callable) -> str | None:
    return _KEYS.get(id(fast_fn))


def _record_target_keys(patch: Any) -> None:
    """Walk a registered patch and remember target keys keyed by fast_fn id.

    Called lazily right before activation in the worker — by then the patch
    has been imported and ``patch.targets`` is fully populated.
    """
    for upstream_path, attr, fast_fn in getattr(patch, "targets", []):
        _KEYS[id(fast_fn)] = f"{upstream_path}:{attr}"


def _wrapped_make_dispatcher(fast_fn: Callable, original: Callable) -> Callable:
    """Stand-in for ``_core._make_dispatcher``: wrap fast_fn with a counter
    before delegating to the real dispatcher. The real dispatcher still
    applies its ``_disabled`` check, so behavior under ``autozyme.disabled()``
    is preserved.
    """
    key = _key_for_fast_fn(fast_fn)

    @functools.wraps(fast_fn)
    def counted(*args, **kwargs):
        if key is not None:
            _COUNTS[key] = _COUNTS.get(key, 0) + 1
        return fast_fn(*args, **kwargs)

    # Tag the counted wrapper so subsequent reflection still finds the fast fn.
    counted.__autozyme_fast__ = fast_fn  # type: ignore[attr-defined]
    assert _ORIG_MAKE_DISPATCHER is not None  # install() ran first
    return _ORIG_MAKE_DISPATCHER(counted, original)


def install(patch_name: str | None = None) -> None:
    """Monkey-patch ``_core._make_dispatcher``; register atexit writer.

    Idempotent — safe to call from multiple worker entry points. When
    ``patch_name`` is given and the patch is already in ``_core._REGISTRY``,
    we record its targets immediately so the first activation sees the
    counter.
    """
    global _INSTALLED, _ORIG_MAKE_DISPATCHER
    if not _INSTALLED:
        _ORIG_MAKE_DISPATCHER = _core._make_dispatcher
        _core._make_dispatcher = _wrapped_make_dispatcher  # type: ignore[assignment]
        atexit.register(_write_counts)
        _INSTALLED = True
    if patch_name and patch_name in _core._REGISTRY:
        _record_target_keys(_core._REGISTRY[patch_name])


def _write_counts() -> None:
    path = os.environ.get("ZYME_INTERCEPT_OUT")
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(_COUNTS, indent=2), encoding="utf-8")
    except OSError:
        pass


def install_from_env() -> bool:
    """Honor ``ZYME_INSTRUMENT_INTERCEPTS=1``. Returns True if installed."""
    if os.environ.get("ZYME_INSTRUMENT_INTERCEPTS") != "1":
        return False
    install(os.environ.get("ZYME_INTERCEPT_PATCH"))
    return True
