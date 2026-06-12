"""Patch registry and lifecycle.

Lazy model: submodules under `autozyme.<name>` are *enumerated* at package
import (via pkgutil) but *not imported* until `activate(name)` runs. This
keeps each patch's heavy deps (TF, torch, numba) out of the process unless
the user actually opts into that patch — preventing cross-patch interactions
(e.g. TF threading layer deadlocking numba parallel JIT compile).

Each registered target is `(upstream_path, attr, fast_fn)`. `upstream_path`
may resolve to either a module or a class reachable via attribute walk.
"""
from __future__ import annotations

import contextlib
import contextvars
import difflib
import functools
import importlib
import importlib.metadata
import importlib.util
import os
import pkgutil
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

from autozyme._subsets import CONFLICTS, SUBSETS, UPSTREAMS


# Context-local kill switch. When True, every patched fn falls through to the
# upstream original via the dispatcher wrapper installed in _activate_one.
# ContextVar (not threading.local) so it propagates correctly across asyncio
# tasks and is isolated per-task.
_disabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "autozyme_disabled", default=False
)


def is_disabled() -> bool:
    """True iff we're inside an `autozyme.disabled()` block."""
    return _disabled.get()


@contextlib.contextmanager
def disabled():
    """Temporarily disable all activated patches in this block.

    Within the with-block every patched fn forwards to its captured upstream
    original. This is the only opt-out mechanism that works for class-method
    patches (whose signatures can't grow a `zyme=` kwarg without breaking
    framework-side `inspect.signature` checks).

    `disabled()` is a hard kill: even an explicit `fn(zyme=True)` inside this
    block runs the original (the `zyme` kwarg is stripped before forwarding,
    so callers of class-method patches don't trip TypeError).
    """
    token = _disabled.set(True)
    try:
        yield
    finally:
        _disabled.reset(token)


def _make_dispatcher(fast_fn: Callable, original: Callable) -> Callable:
    """Wrap `fast_fn` with a dispatcher that short-circuits to `original`
    when we're inside an `autozyme.disabled()` block. The wrapper's signature
    mirrors `original` (via functools.wraps) so upstream `inspect.signature`
    introspection sees the un-patched contract — important for class-method
    patches consumed by frameworks (pyro/scvi-tools) that dispatch on it.
    """
    @functools.wraps(original)
    def dispatcher(*args, **kwargs):
        if _disabled.get():
            kwargs.pop("zyme", None)
            return original(*args, **kwargs)
        return fast_fn(*args, **kwargs)
    dispatcher.__autozyme_fast__ = fast_fn  # type: ignore[attr-defined]
    dispatcher.__autozyme_original__ = original  # type: ignore[attr-defined]
    return dispatcher


@dataclass
class _Patch:
    name: str
    targets: list[tuple[str, str, Callable]]
    smoke: dict[str, Callable] | None = None
    tested_against: str | None = None  # e.g. "cell2location 0.1.4"
    tested_upstream_versions: dict[str, list[str]] | None = None
    originals: dict[tuple[str, str], Any] = field(default_factory=dict)
    injected: bool = False


# Patches that have been imported and registered. Populated by submodule
# import (which calls register_patch). Empty until first activate(name).
_REGISTRY: dict[str, _Patch] = {}

# Submodule names enumerated by pkgutil at package import — these are the
# patches we *could* activate (assuming their upstream is installed).
_AVAILABLE: list[str] = []


def _populate_available(pkg_path: list[str]) -> None:
    """Called once by autozyme.__init__ to enumerate submodule names."""
    _AVAILABLE.clear()
    for _finder, name, _ispkg in pkgutil.iter_modules(pkg_path):
        if name.startswith("_"):
            continue
        _AVAILABLE.append(name)
    _AVAILABLE.sort()


def _resolve_target(upstream_path: str):
    """Resolve `upstream_path` to a module OR a class via attribute walk."""
    try:
        return importlib.import_module(upstream_path)
    except ImportError:
        pass
    parts = upstream_path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:split])
        tail = parts[split:]
        try:
            obj = importlib.import_module(head)
        except ImportError:
            continue
        try:
            for attr in tail:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise ImportError(
        f"{upstream_path!r} is neither an importable module nor a class "
        f"reachable via attribute walk from an importable module"
    )


def register_patch(
    name: str,
    targets: list[tuple[str, str, Callable]],
    smoke: dict[str, Callable] | None = None,
    tested_against: str | None = None,
    tested_upstream_versions: dict[str, list[str]] | None = None,
) -> None:
    """Register a patch. Called by submodule __init__ at import time
    (which only happens when the user explicitly activates the patch).

    `tested_against` is a free-form string like "cell2location 0.1.4"
    naming the upstream version the patch was lifted against. The activation
    marker compares it to the currently-installed version and warns on drift.

    `tested_upstream_versions` is structured metadata consumed by Tier C CI
    to build the drift-detection matrix. Format: ``{"pkg": ["X.Y.Z", ...]}``
    where each value lists upstream versions the patch is known to parity-pass
    on. CI installs one job per (plugin, version) cell and re-runs the tiny
    tier; any miss opens a drift issue.
    """
    # Validate each target tuple up front so misuses (None / non-callable fast
    # fn) fail at register_patch time, not at activate() time three commands
    # later when the traceback no longer points at the registration site.
    for idx, target in enumerate(targets):
        if (not isinstance(target, tuple)) or len(target) != 3:
            raise TypeError(
                f"patch {name!r} target[{idx}] must be a 3-tuple "
                f"(upstream_path, attr, fast_fn); got {target!r}"
            )
        upstream, attr, fast_fn = target
        if not isinstance(upstream, str) or not upstream:
            raise TypeError(
                f"patch {name!r} target[{idx}] upstream_path must be a "
                f"non-empty str; got {upstream!r}"
            )
        if not isinstance(attr, str) or not attr:
            raise TypeError(
                f"patch {name!r} target[{idx}] attr must be a non-empty "
                f"str; got {attr!r}"
            )
        if not callable(fast_fn):
            raise TypeError(
                f"patch {name!r} target[{idx}] fast_fn for "
                f"{upstream}.{attr} must be callable; got "
                f"{type(fast_fn).__name__}"
            )

    new_claims = {(u, a) for u, a, _ in targets}
    for other_name, other in _REGISTRY.items():
        if other_name == name:
            continue
        other_claims = {(u, a) for u, a, _ in other.targets}
        conflicts = new_claims & other_claims
        if conflicts:
            conflict_str = ", ".join(f"{u}::{a}" for u, a in sorted(conflicts))
            raise ValueError(
                f"patch {name!r} targets {{ {conflict_str} }}, "
                f"already claimed by patch {other_name!r}"
            )
    if smoke is not None:
        missing = {"load", "call", "save"} - set(smoke.keys())
        if missing:
            raise ValueError(
                f"smoke recipe missing keys: {missing} (need load, call, save)"
            )
    if tested_upstream_versions is not None:
        if not isinstance(tested_upstream_versions, dict):
            raise TypeError(
                f"tested_upstream_versions must be dict[str, list[str]], "
                f"got {type(tested_upstream_versions).__name__}"
            )
        for pkg, vers in tested_upstream_versions.items():
            if not isinstance(pkg, str) or not isinstance(vers, list):
                raise TypeError(
                    f"tested_upstream_versions[{pkg!r}] must be list[str]"
                )
            if not vers:
                raise ValueError(
                    f"tested_upstream_versions[{pkg!r}] is empty; "
                    f"declare at least one version or omit the key"
                )
            for v in vers:
                if not isinstance(v, str):
                    raise TypeError(
                        f"tested_upstream_versions[{pkg!r}] entries "
                        f"must be str, got {type(v).__name__}"
                    )
    _REGISTRY[name] = _Patch(
        name=name,
        targets=list(targets),
        smoke=smoke,
        tested_against=tested_against,
        tested_upstream_versions=tested_upstream_versions,
    )
    _PROBE_CACHE.pop(name, None)


def _import_submodule(name: str) -> None:
    """Lazy-import autozyme.<name> if not already imported."""
    if name in _REGISTRY:
        return
    try:
        importlib.import_module(f"autozyme.{name}")
    except ImportError as e:
        raise ImportError(
            f"could not load autozyme.{name}: {e}. "
            f"Is the upstream package installed in this environment?"
        ) from e
    if name not in _REGISTRY:
        raise RuntimeError(
            f"autozyme.{name} imported but did not register a patch — "
            f"the submodule must call register_patch(name={name!r}, ...)"
        )


# Cache for probe results. Key: patch name. Value: (installed: bool, err: str|None).
# Memoized per-process; env changes mid-session are rare.
_PROBE_CACHE: dict[str, tuple[bool, str | None]] = {}


def _probe_patch_installed(name: str) -> tuple[bool, str | None]:
    """Cheap check: are all of `name`'s declared upstreams importable?

    Uses `importlib.util.find_spec` against the UPSTREAMS manifest in
    _subsets.py. Does NOT execute the patch submodule or the upstream
    package bodies — critical for avoiding cross-patch import side effects
    (OpenMP duplicate-load, TF eager init, numba JIT contention).

    Registered user patches not listed in UPSTREAMS derive their upstream
    packages from their target paths. Lazy bundled patches not yet imported
    fall back to assuming the upstream package shares the patch's name.
    """
    if name in _PROBE_CACHE:
        return _PROBE_CACHE[name]
    if name in UPSTREAMS:
        upstreams = UPSTREAMS[name]
    elif name in _REGISTRY:
        upstreams = sorted({_top_level_pkg(u) for u, _, _ in _REGISTRY[name].targets})
    else:
        upstreams = [name]
    missing = []
    for pkg in upstreams:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(pkg)
    if missing:
        result: tuple[bool, str | None] = (
            False,
            f"upstream not installed: {', '.join(missing)}",
        )
    else:
        result = (True, None)
    _PROBE_CACHE[name] = result
    return result


def _top_level_pkg(dotted_path: str) -> str:
    """For 'cell2location.models._cell2location_module' → 'cell2location'."""
    return dotted_path.split(".", 1)[0]


def _installed_version(pkg: str) -> str | None:
    """Best-effort lookup of the distribution version for an importable pkg.
    Returns None when the package isn't installed via a metadata-recording
    installer (rare for upstreams we patch, but possible)."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _base_version(v: str) -> str:
    """PEP 440 *release* version (strip pre/post/dev/local segments) so an
    editable / git / dev install of the same release -- e.g.
    "0.9.6.dev5+g6c5e37d1d" or "4.0.2+4.gb847e2c" -- matches its declared
    release. Falls back to the pre-"+" string if `packaging` is unavailable
    or the version string is non-standard.
    """
    try:
        from packaging.version import Version
        return Version(v).base_version
    except Exception:
        return v.split("+", 1)[0]


def _emit_activation_marker(p: _Patch) -> None:
    """One-shot stderr line proving the binding happened. Greppable in run.log.

    Format:
        [autozyme] activated <name> -> N targets in <pkg> <ver> [WARN: drift]

    Silenced by AUTOZYME_QUIET=1.
    """
    if os.environ.get("AUTOZYME_QUIET"):
        return
    pkgs = {_top_level_pkg(u) for u, _, _ in p.targets}
    versions = {pkg: _installed_version(pkg) for pkg in sorted(pkgs)}
    ver_str = ", ".join(
        f"{pkg} {v}" if v else f"{pkg} (version unknown)"
        for pkg, v in versions.items()
    )
    drift = ""
    if p.tested_against:
        # tested_against is "<pkg> <ver>". Known-good = that literal plus any
        # versions in tested_upstream_versions for the named pkg. Compare on the
        # PEP 440 *release* version (drop pre/post/dev/local segments) so an
        # editable/git/dev install of the same release isn't flagged.
        try:
            tested_pkg, tested_ver = p.tested_against.rsplit(" ", 1)
        except ValueError:
            tested_pkg, tested_ver = None, None
        if tested_pkg and tested_pkg in versions:
            actual = versions[tested_pkg]
            if actual:
                known = {tested_ver}
                if p.tested_upstream_versions:
                    known.update(p.tested_upstream_versions.get(tested_pkg, []))
                if _base_version(actual) not in {_base_version(k) for k in known}:
                    drift = (
                        f" — WARN: lifted against {tested_pkg} {tested_ver}, "
                        f"installed {actual} (may be unstable)"
                    )
    print(
        f"[autozyme] activated {p.name} -> {len(p.targets)} target(s) "
        f"in {ver_str}{drift}",
        file=sys.stderr,
    )


def _emit_inactive_marker(name: str, err: "str | None") -> None:
    """One-line stderr note when a patch can't activate because its upstream
    isn't installed. The success path is loud (_emit_activation_marker), so
    without this a missing upstream is a silent no-op and the user wrongly
    believes they're accelerated. Silenced by AUTOZYME_QUIET=1.
    """
    if os.environ.get("AUTOZYME_QUIET"):
        return
    detail = f" ({err})" if err else ""
    print(
        f"[autozyme] {name} NOT activated -- upstream not installed{detail}",
        file=sys.stderr,
    )


def _activate_one(p: _Patch) -> bool:
    if p.injected:
        return True
    resolved = []
    missing = []
    for upstream, attr, fast_fn in p.targets:
        try:
            holder = _resolve_target(upstream)
        except ImportError as e:
            # Upstream module/class doesn't exist in this installed version.
            # Track as missing rather than fail the whole patch -- partial
            # activation is strictly better than total failure when only a
            # subset of targets has drifted out from under us.
            missing.append((upstream, attr, str(e)))
            continue
        if not hasattr(holder, attr):
            missing.append((upstream, attr, f"{upstream} has no attribute {attr!r}"))
            continue
        resolved.append((upstream, attr, holder, fast_fn))
    if not resolved:
        # Zero bindable targets: the upstream API drifted away entirely.
        # Caller gets the original "return False" contract.
        return False
    for upstream, attr, holder, fast_fn in resolved:
        original = getattr(holder, attr)
        p.originals[(upstream, attr)] = original
        setattr(holder, attr, _make_dispatcher(fast_fn, original))
    p.injected = True
    if missing:
        # Surface the drift via stderr so users see WHICH targets are out
        # of reach, plus a concrete upstream-version recommendation when
        # the patch declared one (so the user knows what to pin for full
        # coverage rather than guessing).
        import sys as _sys
        names = ", ".join(f"{u}.{a}" for u, a, _ in missing)
        lines = [
            f"[autozyme] {p.name}: partial activation -- "
            f"{len(resolved)}/{len(p.targets)} targets bound; "
            f"missing on installed upstream: {names}"
        ]
        if p.tested_upstream_versions:
            spec = "; ".join(
                f"{pkg}=={vers[0]}" if len(vers) == 1
                else f"{pkg} in {{{', '.join(vers)}}}"
                for pkg, vers in p.tested_upstream_versions.items()
            )
            lines.append(
                f"[autozyme] {p.name}: for full speedup install: {spec}"
            )
        elif p.tested_against:
            lines.append(
                f"[autozyme] {p.name}: lifted against {p.tested_against} "
                f"-- install that exact upstream for full coverage"
            )
        _sys.stderr.write("\n".join(lines) + "\n")
    _emit_activation_marker(p)
    return True


def _deactivate_one(p: _Patch) -> None:
    if not p.injected:
        return
    for (upstream, attr), original in p.originals.items():
        try:
            holder = _resolve_target(upstream)
        except ImportError:
            continue
        setattr(holder, attr, original)
    p.originals.clear()
    p.injected = False


def _did_you_mean(name: str, candidates: list[str], n: int = 2) -> str:
    """Build a 'Did you mean X?' tail string for KeyError messages, or '' if
    no close match. Uses difflib SequenceMatcher with cutoff=0.6 (default)."""
    matches = difflib.get_close_matches(name, candidates, n=n)
    if not matches:
        return ""
    if len(matches) == 1:
        return f" Did you mean {matches[0]!r}?"
    return f" Did you mean one of {matches!r}?"


def _resolve_activation_target(name) -> list[str]:
    """Turn `name` (str | list[str]) into a flat patch-name list, expanding
    subset names. Order is preserved; duplicates removed. Unknown names
    (neither a registered patch nor a subset) raise KeyError."""
    if isinstance(name, str):
        if name in SUBSETS:
            items = SUBSETS[name]
        elif name in _AVAILABLE or name in _REGISTRY:
            # Either a known submodule (lazy-import on activate) or already
            # registered directly (e.g. by tests via register_patch()).
            items = [name]
        else:
            candidates = sorted(set(_AVAILABLE) | set(_REGISTRY) | set(SUBSETS))
            hint = _did_you_mean(name, candidates)
            raise KeyError(
                f"{name!r} is neither a known patch nor a subset.{hint} "
                f"See autozyme.list_patches() / autozyme.list_subsets()."
            )
    elif isinstance(name, (list, tuple)):
        items = []
        for n in name:
            items.extend(_resolve_activation_target(n))
    else:
        raise TypeError(
            f"activate() expects str, list, or tuple, "
            f"got {type(name).__name__}"
        )
    seen = set()
    out = []
    for n in items:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _check_conflicts(newly_activating: list[str]) -> None:
    """Warn (do not raise) when the union of already-active patches and
    `newly_activating` matches any pair in CONFLICTS. Best-effort heuristic;
    user can ignore + tune knobs manually if they really want the combo.
    """
    live = {n for n, p in _REGISTRY.items() if p.injected}
    live.update(newly_activating)
    for pair, reason in CONFLICTS:
        if pair.issubset(live):
            warnings.warn(
                f"autozyme: activating {sorted(pair)} together is known to "
                f"interact badly. {reason}",
                RuntimeWarning,
                stacklevel=3,
            )


def activate(name) -> bool | dict[str, bool]:
    """Activate one patch, a subset, or a list of patches/subsets.

    Args:
        name: either a patch name, a subset name, or a list mixing them.

    Returns:
        For a single patch name: True if its targets were patched, False if
        upstream is missing or ``AUTOZYME_DISABLED`` is set in the environment.
        For a list/subset: dict mapping each resolved patch name to its
        activation result.

    Setting ``AUTOZYME_DISABLED=1`` (any non-empty value) before calling
    ``activate`` short-circuits every binding — useful for reproducibility
    runs that want to confirm a result still holds with upstream untouched
    without editing call sites.
    """
    targets = _resolve_activation_target(name)
    if os.environ.get("AUTOZYME_DISABLED") or os.environ.get("AUTOZYME_DISABLE"):
        if isinstance(name, str) and name not in SUBSETS and len(targets) == 1:
            return False
        return {n: False for n in targets}
    _check_conflicts(targets)
    if isinstance(name, str) and name not in SUBSETS and len(targets) == 1:
        # single patch path
        n = targets[0]
        installed, _err = _probe_patch_installed(n)
        if not installed:
            _emit_inactive_marker(n, _err)
            return False
        _import_submodule(n)
        return _activate_one(_REGISTRY[n])
    out = {}
    for n in targets:
        try:
            installed, _err = _probe_patch_installed(n)
            if not installed:
                _emit_inactive_marker(n, _err)
                out[n] = False
                continue
            _import_submodule(n)
            out[n] = _activate_one(_REGISTRY[n])
        except ImportError:
            out[n] = False
    return out


def deactivate(name) -> None:
    """Deactivate one patch, a subset, or a list of patches/subsets.

    Symmetric with `activate()`. Unknown names raise KeyError with a
    did-you-mean suggestion when available. Names that resolve but were
    never activated are skipped silently (idempotent deactivate).
    """
    targets = _resolve_activation_target(name)
    for n in targets:
        p = _REGISTRY.get(n)
        if p is None:
            # Patch was never activated in this process — nothing to undo.
            # (Resolve_target validated the name; skipping here is safe.)
            continue
        _deactivate_one(p)


def deactivate_all() -> None:
    for p in _REGISTRY.values():
        _deactivate_one(p)


def status() -> dict[str, str]:
    """Activation state of every available patch.

    Returns "active" for patches currently bound into their upstream and
    "inactive" otherwise. "inactive" covers both registered-but-not-yet-
    activated patches and patches still lazy in autozyme/<name>/. Mirrors
    R's autozyme::status() so the two surfaces agree.
    """
    out: dict[str, str] = {}
    for name in _AVAILABLE:
        p = _REGISTRY.get(name)
        out[name] = "active" if (p is not None and p.injected) else "inactive"
    # Include any registry-only entries (e.g. registered via test fixtures)
    # that weren't discovered by the pkgutil scan.
    for name, p in _REGISTRY.items():
        if name not in out:
            out[name] = "active" if p.injected else "inactive"
    return out


def inspect(name: str) -> dict[str, Any]:
    """Return a structured view of a patch's bindings.

    For debugging "did this actually patch what I expected?". The patch
    submodule is imported if it hasn't been yet, so this also forces
    registration of patches that were only `list_patches()`-discovered.

    Returns:
        dict with keys: name, status, tested_against, installed_version,
        targets (list of dicts: upstream, attr, fast_fn, original, currently_bound).
    """
    targets = _resolve_activation_target(name)
    if len(targets) != 1:
        raise ValueError(
            f"inspect() takes a single patch name; got {name!r} "
            f"(resolves to {targets})"
        )
    n = targets[0]
    installed, err = _probe_patch_installed(n)
    if not installed:
        return {"name": n, "status": "uninstalled", "error": err, "targets": []}
    try:
        _import_submodule(n)
    except ImportError as e:
        return {
            "name": n,
            "status": "uninstalled",
            "error": str(e),
            "targets": [],
        }
    p = _REGISTRY[n]
    upstream_pkgs = {_top_level_pkg(u) for u, _, _ in p.targets}
    installed = {pkg: _installed_version(pkg) for pkg in upstream_pkgs}
    target_views = []
    for upstream, attr, fast_fn in p.targets:
        try:
            holder = _resolve_target(upstream)
            current = getattr(holder, attr, None)
        except ImportError:
            current = None
        original = p.originals.get((upstream, attr))
        target_views.append({
            "upstream": upstream,
            "attr": attr,
            "fast_fn": f"{fast_fn.__module__}.{fast_fn.__qualname__}",
            "original": (
                f"{original.__module__}.{original.__qualname__}"
                if original is not None else None
            ),
            "currently_bound_to_fast": (
                getattr(current, "__autozyme_fast__", None) is fast_fn
            ),
        })
    return {
        "name": n,
        "status": "active" if p.injected else "inactive",
        "tested_against": p.tested_against,
        "installed_versions": installed,
        "targets": target_views,
    }


def list_patches(installed: bool = False) -> list[str]:
    """Patches discoverable in this autozyme install.

    Args:
        installed: when True, only return patches whose upstream package is
            importable in the current env. First call probes by importing
            each submodule (slow); subsequent calls hit a per-process cache.
            Default (False) returns every shipped patch regardless of upstream
            availability — same behavior as before.
    """
    if not installed:
        return list(_AVAILABLE)
    return [n for n in _AVAILABLE if _probe_patch_installed(n)[0]]


def env_snapshot() -> dict[str, Any]:
    """Structured snapshot for Methods-section / provenance capture.

    Includes autozyme version, platform, and per-patch state. Uses cheap
    `find_spec`-based probes — does NOT import any patch submodule.
    For patches that have been imported (i.e. activate or inspect was
    called), the snapshot includes their `tested_against` declaration;
    for ones not yet imported, only the discovered top-level upstreams.
    """
    import platform
    from autozyme import __version__ as _az_version

    patches = []
    for n in _AVAILABLE:
        installed, err = _probe_patch_installed(n)
        if not installed:
            patches.append({"name": n, "status": "uninstalled", "error": err})
            continue
        upstream_pkgs = UPSTREAMS.get(n, [n])
        entry: dict[str, Any] = {
            "name": n,
            "installed_versions": {
                pkg: _installed_version(pkg) for pkg in upstream_pkgs
            },
        }
        p = _REGISTRY.get(n)
        if p is not None:
            entry["status"] = "active" if p.injected else "inactive"
            entry["tested_against"] = p.tested_against
        else:
            entry["status"] = "inactive"
            entry["tested_against"] = None
        patches.append(entry)
    return {
        "autozyme_version": _az_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "patches": patches,
    }


def list_subsets() -> list[str]:
    return sorted(SUBSETS.keys())


def subset(name: str) -> list[str]:
    if name not in SUBSETS:
        raise KeyError(f"no subset named {name!r}; see autozyme.list_subsets()")
    return list(SUBSETS[name])
