"""Read bundled finalized speedup snapshots for each patch or method task.

Finalized TSV files live next to each patch's __init__.py:
    autozyme_py/src/autozyme/<name>/speedups_finalized.tsv

They are curated release snapshots derived from each patch's raw attest
history. The public package ships only these finalized files; raw
`speedups.tsv` histories stay in the framework repository.

"package speedup only": these numbers are *exclusively* the fresh-subprocess
verify_patch protocol output — the headline number that ships to the paper.
They are NOT the in-process iter speedups from `results.tsv` and NOT the
threading-sweep numbers from `verify.tsv`. See `autozyme_cli/prompts/Bio/
4_package.md` ("three speedup contexts") for the methodology distinction.

History is kept: re-running attest appends new rows. By default `speedups()`
returns the latest batch per tier; pass `history=True` to get every batch.
"""
from __future__ import annotations

import csv
import difflib
import warnings
from io import StringIO
from typing import Any

try:
    from importlib.resources import files as _ir_files
except ImportError:  # pragma: no cover — Python < 3.9 fallback
    from importlib_resources import files as _ir_files  # type: ignore


# Long-format column set. We don't enforce strict order on read, but use this
# to type-coerce.
_NUMERIC_COLS = {"rep_idx", "sec", "peak_mb", "system_ram_gb", "system_threads"}

# Facade patches that ship a consolidated implementation but store per-method
# finalized snapshots under sibling directories. `speedups(name)` aggregates
# `<name>_*/speedups_finalized.tsv` only for these.
_FACADES = frozenset({"scanpy"})

_BATCH_KEY_COLS = (
    "timestamp", "patch_name", "tier",
    "system_os", "system_cpu", "system_threads", "framework_version",
)
_FINALIZED_KEY_COLS = (
    "patch", "package_version", "tier", "threads", "platform", "dataset",
)


def _read_long_rows(text: str) -> list[dict[str, Any]]:
    if not text.strip():
        return []
    reader = csv.DictReader(StringIO(text), delimiter="\t")
    return [dict(r) for r in reader]


def _to_float(s: Any) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_int(s: Any) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _to_bool(s: Any) -> bool | None:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _split_float_list(s: Any) -> list[float]:
    if s is None:
        return []
    out: list[float] = []
    for part in str(s).split(","):
        val = _to_float(part.strip())
        if val is not None:
            out.append(val)
    return out


def _batch_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(c) or "").strip() for c in _BATCH_KEY_COLS)


def _summarize_batch(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    """Aggregate a single batch's baseline+patched rows into a summary dict.

    Returns None if the batch is incomplete (no measurements on one side).
    """
    baseline = [r for r in rows if (r.get("variant") or "") == "baseline"]
    patched = [r for r in rows if (r.get("variant") or "") == "patched"]

    def _sorted_by_rep(rs: list[dict[str, str]]) -> list[dict[str, str]]:
        return sorted(rs, key=lambda r: _to_int(r.get("rep_idx")) or 0)

    baseline = _sorted_by_rep(baseline)
    patched = _sorted_by_rep(patched)
    if not baseline or not patched:
        return None

    b_secs = [_to_float(r.get("sec")) for r in baseline]
    p_secs = [_to_float(r.get("sec")) for r in patched]
    b_peaks = [_to_float(r.get("peak_mb")) for r in baseline]
    p_peaks = [_to_float(r.get("peak_mb")) for r in patched]
    per_rep_pass = [_to_bool(r.get("pass")) for r in patched]

    valid_b = [x for x in b_secs if x is not None]
    valid_p = [x for x in p_secs if x is not None]
    if not valid_b or not valid_p:
        return None
    baseline_sec_mean = sum(valid_b) / len(valid_b)
    patched_sec_mean = sum(valid_p) / len(valid_p)
    speedup_x = (baseline_sec_mean / patched_sec_mean
                 if patched_sec_mean > 0 else None)

    valid_bp = [x for x in b_peaks if x is not None]
    valid_pp = [x for x in p_peaks if x is not None]
    baseline_peak_mb = max(valid_bp) if valid_bp else None
    patched_peak_mb = max(valid_pp) if valid_pp else None

    sample = patched[0]
    if all(p is True for p in per_rep_pass):
        all_pass: bool | None = True
    elif any(p is False for p in per_rep_pass):
        all_pass = False
    else:
        all_pass = None

    # Pull the batch-level metrics_json off the first patched row (we replicate
    # it on every patched row, so any non-empty value will do).
    metrics_json = ""
    for r in patched:
        mj = (r.get("metrics_json") or "").strip()
        if mj:
            metrics_json = mj
            break

    return {
        "timestamp":        sample.get("timestamp") or "",
        "patch_name":       sample.get("patch_name") or "",
        "tier":             sample.get("tier") or "",
        "dataset":          (sample.get("dataset") or "").strip(),
        "reps":             len(p_secs),
        "baseline_sec":     baseline_sec_mean,
        "patched_sec":      patched_sec_mean,
        "speedup_x":        speedup_x,
        "all_pass":         all_pass,
        "baseline_secs":    valid_b,
        "patched_secs":     valid_p,
        "baseline_peak_mb": baseline_peak_mb,
        "patched_peak_mb":  patched_peak_mb,
        "baseline_peaks_mb": [x for x in b_peaks if x is not None],
        "patched_peaks_mb":  [x for x in p_peaks if x is not None],
        "metrics_json":     metrics_json,
        "framework_version": sample.get("framework_version") or "",
        "note":             sample.get("note") or "",
        "system_os":        sample.get("system_os") or "",
        "system_cpu":       sample.get("system_cpu") or "",
        "system_ram_gb":    _to_float(sample.get("system_ram_gb")),
        "system_threads":   _to_int(sample.get("system_threads")),
    }


def _finalized_key(row: dict[str, str]) -> tuple[str, ...]:
    base = tuple((row.get(c) or "").strip() for c in _FINALIZED_KEY_COLS)
    # Disambiguate rows that share `patch`+tier+dataset but came from different
    # sub-task directories during facade aggregation (e.g. scanpy_normalize +
    # scanpy_pca both record patch=scanpy and may share the same tier).
    return base + ((row.get("_az_source") or "").strip(),)


def _summarize_finalized_group(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    baseline = next((r for r in rows if (r.get("variant") or "") == "baseline"), None)
    patched = next((r for r in rows if (r.get("variant") or "") == "patched"), None)
    if baseline is None or patched is None:
        return None

    baseline_secs = _split_float_list(baseline.get("sec_reps"))
    patched_secs = _split_float_list(patched.get("sec_reps"))
    baseline_peaks = _split_float_list(baseline.get("mem_reps"))
    patched_peaks = _split_float_list(patched.get("mem_reps"))
    pass_rate = _to_float(patched.get("pass_rate"))
    patched_status = (patched.get("status") or "").strip().lower()

    if patched_status and patched_status != "ok":
        all_pass: bool | None = False
    elif pass_rate is None:
        all_pass = None
    else:
        all_pass = pass_rate >= 1.0

    # When facade-aggregated, _az_source carries the sub-task directory name
    # (e.g. "scanpy_normalize"), which is more informative than the raw `patch`
    # column (all sub-task TSVs record patch=scanpy).
    patch_name = (patched.get("_az_source") or patched.get("patch") or "")
    return {
        "timestamp":        patched.get("ts_last") or patched.get("ts_first") or "",
        "patch_name":       patch_name,
        "tier":             patched.get("tier") or "",
        "dataset":          (patched.get("dataset") or "").strip(),
        "reps":             _to_int(patched.get("n_reps")) or len(patched_secs),
        "baseline_sec":     _to_float(baseline.get("sec_mean")),
        "patched_sec":      _to_float(patched.get("sec_mean")),
        "speedup_x":        _to_float(patched.get("speedup_x_mean")),
        "all_pass":         all_pass,
        "baseline_secs":    baseline_secs,
        "patched_secs":     patched_secs,
        "baseline_peak_mb": _to_float(baseline.get("mem_mean")),
        "patched_peak_mb":  _to_float(patched.get("mem_mean")),
        "baseline_peaks_mb": baseline_peaks,
        "patched_peaks_mb":  patched_peaks,
        "metrics_json":     patched.get("metrics_json_median") or "",
        "framework_version": patched.get("fw_versions") or "",
        "note":             patched.get("status") or "",
        "system_os":        patched.get("platform") or "",
        "system_cpu":       "",
        "system_ram_gb":    None,
        "system_threads":   _to_int(patched.get("threads")),
    }


def speedups(name: str, history: bool = False) -> list[dict[str, Any]]:
    """Return bundled finalized speedup summaries for patch ``name``.

    Each entry is one finalized platform/thread/tier cell, with raw per-rep
    arrays in ``baseline_secs`` / ``patched_secs``
    and aggregate scalars in ``baseline_sec`` / ``patched_sec`` /
    ``speedup_x``.

    Facade patches (currently ``scanpy``) ship their headline numbers under
    per-method sibling directories (``scanpy_normalize`` etc.); querying the
    facade name aggregates those siblings into one combined list.

    Args:
        name: registered patch name (same key as ``activate(name)``).
        history: accepted for backward compatibility. Finalized snapshots
            contain only release-curated rows, so both values return the same
            data.

    Returns:
        List of dicts. Empty list if no bundled file exists.
    """
    try:
        base = _ir_files("autozyme")
    except (ModuleNotFoundError, AttributeError):
        return []

    rows: list[dict[str, str]] = []
    rsrc = base / name / "speedups_finalized.tsv"
    if rsrc.is_file():
        rows.extend(_read_long_rows(rsrc.read_text(encoding="utf-8")))

    # Facade aggregation: scanpy ships one consolidated patch but stores its
    # per-method finalized snapshots under sibling directories
    # (scanpy_normalize/, scanpy_pca/, ...). `list_patches()` only surfaces
    # the facade name (sub-task dirs carry no __init__.py), so querying the
    # facade must return the union. Whitelisted to avoid false matches like
    # "mdanalysis" -> "mdanalysis_rmsd".
    if name in _FACADES:
        sibling_prefix = f"{name}_"
        try:
            sibling_dirs = sorted(
                p for p in base.iterdir()
                if p.name.startswith(sibling_prefix) and p.is_dir()
            )
        except (AttributeError, OSError):
            sibling_dirs = []
        for sib in sibling_dirs:
            sib_tsv = sib / "speedups_finalized.tsv"
            if sib_tsv.is_file():
                sib_rows = _read_long_rows(sib_tsv.read_text(encoding="utf-8"))
                for r in sib_rows:
                    r["_az_source"] = sib.name
                rows.extend(sib_rows)

    if not rows:
        # Common user error: passed "Scanpy" instead of "scanpy" — patch dir
        # names are lowercase. Retry once with lowercased name before giving
        # up (no infinite loop because the retry's name == name.lower()).
        if name != name.lower():
            return speedups(name.lower(), history=history)
        # Still empty: emit a hint when there's a close-name patch dir, so
        # typos like "snanpy"/"scvelos" surface a suggestion instead of a
        # silent empty list. Non-breaking: we still return [] for callers
        # that already handle the empty case.
        try:
            available = sorted(
                p.name for p in base.iterdir()
                if p.is_dir() and not p.name.startswith("_")
            )
        except (AttributeError, OSError):
            available = []
        matches = difflib.get_close_matches(name, available, n=2)
        if matches:
            warnings.warn(
                f"speedups({name!r}) returned no data. "
                f"Did you mean {matches!r}?",
                stacklevel=2,
            )
        return []

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(_finalized_key(r), []).append(r)

    summaries: list[dict[str, Any]] = []
    for batch_rows in grouped.values():
        s = _summarize_finalized_group(batch_rows)
        if s is not None:
            summaries.append(s)

    summaries.sort(key=lambda b: (
        b.get("tier") or "",
        b.get("system_os") or "",
        b.get("system_threads") or 0,
        b.get("dataset") or "",
        b.get("timestamp") or "",
    ))
    return summaries
