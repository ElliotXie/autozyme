"""Read and filter task-side `package_verify.tsv` rows for publishing.

Schema (long format, since 2026-05-22 refactor; ``dataset`` column since 2026-05-26):

  timestamp, patch_name, tier, dataset, rep_idx, variant,
  sec, speedup_pct, speedup_x,
  peak_mb, peak_mb_change_pct, peak_mb_fold,
  pass, metrics_json, framework_version, note,
  system_os, system_cpu, system_ram_gb, system_threads

Each `verify_patch()` invocation appends 2N rows per tier: N baseline rows
(rep_idx=1..N, variant=baseline) followed by N patched rows. All rows in a
batch share the same `timestamp` (batch start). The four derived columns
(speedup_pct, speedup_x, peak_mb_change_pct, peak_mb_fold) are populated on
patched rows only; baseline rows have those cells blank.

Per-rep derivations. With B_sec = median(this batch's baseline_secs) and
B_mb = median(this batch's baseline peak_mb):

  speedup_pct        = (B_sec - sec) / B_sec * 100
                       positive = patched faster (% wall-time saved)
  speedup_x          = B_sec / sec
                       >1     = patched faster (fold)
  peak_mb_change_pct = (B_mb - peak_mb) / B_mb * 100
                       positive = patched uses less memory (% saved)
                       negative = patched uses more memory

  Baseline rows with note prefix ``absorbed:`` or ``migrated:`` use legacy
  gc()-max-used peaks, not subprocess RSS. R verify_patch refuses to reuse
  those peaks when computing patched peak_mb_change_pct; only time (sec)
  may still be borrowed via no_baseline_confirm.
  peak_mb_fold       = B_mb / peak_mb
                       >1     = patched uses less memory
                       <1     = patched uses more memory

All four are robust to baseline noise (median, not mean). Variance across
patched rows shows per-rep stability directly. Batch-level headline numbers
(baseline_mean / patched_mean) are recomputed by readers when needed.

`zyme publish-speedups` copies all or a filtered subset into the bundled
`speedups.tsv` shipped with the installed package. Filter modes operate
batch-aware where it matters: `latest-per-tier` selects the latest batch
per (platform, tier); `--all-pass-only` drops a whole measured batch if any
of its patched rows failed metrics. Crash/OOM sentinel rows are publishable
attempt records, not measurements; they are retained so `scan --attest` can
show the failed tier and `attest` can avoid repeating known same-machine OOMs.
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from zyme.scan_attest import TIERS, _normalize_platform, normalize_tier


# Canonical long-format columns. Used by the renderer and the migration
# script — keep in sync with R/Py `verify_patch()` writers.
LONG_HEADER: list[str] = [
    "timestamp", "patch_name", "tier", "dataset",
    "rep_idx", "variant",
    "sec", "speedup_pct", "speedup_x",
    "peak_mb", "peak_mb_change_pct", "peak_mb_fold",
    "pass", "metrics_json",
    "framework_version", "package_version", "note",
    "system_os", "system_cpu", "system_ram_gb", "system_threads",
]

# Pre-2026-05-28 long format (had dataset, no package_version). Upgraded on append.
LEGACY_LONG_HEADER_PRE_PKGVER: list[str] = [
    c for c in LONG_HEADER if c != "package_version"
]

# Pre-2026-05-26 long format (no per-row dataset id). Still accepted on read
# and upgraded in place by verify_patch / backfill scripts.
LEGACY_LONG_HEADER: list[str] = [
    c for c in LONG_HEADER if c not in ("dataset", "package_version")
]


def tier_dataset_map_from_task_yaml(task_yaml: Path) -> dict[str, str]:
    """Map canonical tier -> task.yaml ``datasets[].name``."""
    if not task_yaml.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        task = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        # yaml.safe_load raises yaml.YAMLError (NOT a ValueError) on malformed
        # YAML; catch it so a bad task.yaml returns {} as the docstring promises.
        return {}
    if not isinstance(task, dict):
        return {}
    out: dict[str, str] = {}
    for ds in task.get("datasets") or []:
        tier = normalize_tier(str(ds.get("tier") or "").strip())
        name = str(ds.get("name") or "").strip()
        if tier and name:
            out[tier] = name
    return out


def dataset_for_tier(tier: str, tier_map: dict[str, str]) -> str:
    return tier_map.get(normalize_tier((tier or "").strip()), "")


def dataset_from_row(
    row: dict[str, str],
    tier_map: dict[str, str] | None = None,
) -> str:
    """Best-effort dataset id for one long-format row.

    Priority:
      1. ``metrics_json.dataset`` (migrated benchmark rows — ground truth)
      2. ``task.yaml`` tier_map only when note is not a legacy import
      3. Existing ``dataset`` cell
      4. ``task.yaml`` tier_map (fallback)
    """
    mj = (row.get("metrics_json") or "").strip()
    if mj.startswith("{"):
        try:
            import json
            parsed = json.loads(mj)
            ds = str(parsed.get("dataset") or "").strip()
            if ds:
                return ds
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    note = (row.get("note") or "").strip()
    if note.startswith("migrated:") or note.startswith("absorbed:"):
        return ""
    existing = (row.get("dataset") or "").strip()
    if existing:
        return existing
    return dataset_for_tier((row.get("tier") or "").strip(), tier_map or {})


def fill_batch_datasets(
    rows: list[dict[str, str]],
    tier_map: dict[str, str] | None = None,
    *,
    refresh_from_tier_map: bool = False,
) -> list[dict[str, str]]:
    """Assign ``dataset`` on every row; propagate within attest batches."""
    tier_map = tier_map or {}
    out: list[dict[str, str]] = []
    for r in rows:
        tier = normalize_tier((r.get("tier") or "").strip())
        if refresh_from_tier_map and tier in tier_map:
            out.append({**r, "dataset": tier_map[tier]})
        else:
            out.append({**r, "dataset": dataset_from_row(r, tier_map)})

    def batch_key(r: dict[str, str]) -> tuple[str, ...]:
        return (
            (r.get("timestamp") or "").strip(),
            normalize_tier((r.get("tier") or "").strip()),
            (r.get("system_threads") or "").strip(),
            (r.get("patch_name") or "").strip(),
        )

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for r in out:
        groups.setdefault(batch_key(r), []).append(r)

    for group in groups.values():
        known = {(g.get("dataset") or "").strip() for g in group if (g.get("dataset") or "").strip()}
        if len(known) == 1:
            ds = next(iter(known))
            for g in group:
                if not (g.get("dataset") or "").strip():
                    g["dataset"] = ds
        elif known:
            # Prefer non-_60k name when batch mixes migrated full + attest subsample
            full = sorted(known, key=lambda d: (d.endswith("_60k"), d))
            for g in group:
                note = (g.get("note") or "").strip()
                if note.startswith("migrated:") and not (g.get("dataset") or "").strip():
                    g["dataset"] = full[0]
    return out


def upgrade_rows_to_long_header(
    header: list[str],
    rows: list[dict[str, str]],
    tier_map: dict[str, str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Normalize rows to ``LONG_HEADER``, filling ``dataset`` from ``tier_map``."""
    tier_map = tier_map or {}
    if header == LONG_HEADER:
        out = []
        for r in rows:
            row = {c: (r.get(c) or "").strip() for c in LONG_HEADER}
            for c in LONG_HEADER:
                if c not in row and c in r:
                    row[c] = (r.get(c) or "").strip()
            out.append(row)
        return LONG_HEADER, fill_batch_datasets(out, tier_map)
    if header == LEGACY_LONG_HEADER_PRE_PKGVER:
        out = []
        for r in rows:
            row = {c: (r.get(c) or "").strip() for c in LONG_HEADER}
            for c in LEGACY_LONG_HEADER_PRE_PKGVER:
                if c in r:
                    row[c] = (r.get(c) or "").strip()
            out.append(row)
        return LONG_HEADER, fill_batch_datasets(out, tier_map)
    if header == LEGACY_LONG_HEADER:
        out = []
        for r in rows:
            row = {c: (r.get(c) or "").strip() for c in LONG_HEADER}
            for c in LEGACY_LONG_HEADER:
                if c in r:
                    row[c] = (r.get(c) or "").strip()
            out.append(row)
        return LONG_HEADER, fill_batch_datasets(out, tier_map)
    return header, rows


def format_tier_dataset_label(tier: str, dataset: str) -> str:
    """Human-readable tier line, e.g. ``medium: pbmc200k_glaucoma_60k``."""
    tier = normalize_tier((tier or "").strip())
    dataset = (dataset or "").strip()
    if tier and dataset:
        return f"{tier}: {dataset}"
    return tier or dataset


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def per_rep_speedup_x(baseline_secs: list[float], patched_sec: float) -> float | None:
    """``median(baseline_secs) / patched_sec`` — None when either side is empty."""
    if not baseline_secs or patched_sec is None or patched_sec <= 0:
        return None
    return statistics.median(baseline_secs) / patched_sec


def per_rep_speedup_pct(baseline_secs: list[float], patched_sec: float) -> float | None:
    """``(median(baseline_secs) - patched_sec) / median(baseline_secs) * 100``.

    Equivalent to ``(1 - 1/speedup_x) * 100``. Positive = patched is faster.
    """
    if not baseline_secs or patched_sec is None or patched_sec <= 0:
        return None
    b = statistics.median(baseline_secs)
    if b <= 0:
        return None
    return (b - patched_sec) / b * 100.0


def per_rep_peak_mb_fold(baseline_peaks: list[float],
                         patched_peak: float) -> float | None:
    """``median(baseline_peaks) / patched_peak``. >1 = patched uses less memory."""
    if not baseline_peaks or patched_peak is None or patched_peak <= 0:
        return None
    return statistics.median(baseline_peaks) / patched_peak


def per_rep_peak_mb_change_pct(baseline_peaks: list[float],
                               patched_peak: float) -> float | None:
    """``(median(baseline_peaks) - patched_peak) / median(baseline_peaks) * 100``.

    Positive = patched uses less memory. Negative = patched uses more.
    """
    if not baseline_peaks or patched_peak is None or patched_peak <= 0:
        return None
    b = statistics.median(baseline_peaks)
    if b <= 0:
        return None
    return (b - patched_peak) / b * 100.0

VARIANT_BASELINE = "baseline"
VARIANT_PATCHED = "patched"
_VALID_VARIANTS = {VARIANT_BASELINE, VARIANT_PATCHED}


def read_package_verify(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return (header columns, data rows as dicts)."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return [], []
        header = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    return header, rows


def row_is_valid(row: dict[str, str]) -> bool:
    """True for a real measurement row (non-empty `sec`, recognized variant)."""
    sec = (row.get("sec") or "").strip()
    variant = (row.get("variant") or "").strip()
    return bool(sec) and variant in _VALID_VARIANTS


def row_is_sentinel(row: dict[str, str]) -> bool:
    """True for a persisted failed/skipped attest attempt.

    Verify writers emit these rows when a tier produces no measurements. They
    intentionally have empty ``rep_idx``/``variant``/``sec`` cells plus a note.
    They are publishable as attest coverage metadata, but never count as
    speedup measurements.
    """
    sec = (row.get("sec") or "").strip()
    variant = (row.get("variant") or "").strip()
    tier = normalize_tier((row.get("tier") or "").strip())
    note = (row.get("note") or "").strip()
    return (not sec) and (not variant) and tier in TIERS and bool(note)


_OOM_NOTE_PATTERNS = (
    "oom",
    "out of memory",
    "memory limit",
    "cannot allocate",
    "bad_alloc",
    "bad allocation",
    "exited -9",
    "subprocess exited -9",
    "sigkill",
    "killed",
    "killed-by-mem-watchdog",
)


def row_is_oom_sentinel(row: dict[str, str]) -> bool:
    """True when a sentinel note looks like an out-of-memory attempt."""
    if not row_is_sentinel(row):
        return False
    note = (row.get("note") or "").strip().lower()
    return any(pat in note for pat in _OOM_NOTE_PATTERNS)


def row_is_publishable(row: dict[str, str]) -> bool:
    """True for rows that belong in package attest snapshots."""
    return row_is_valid(row) or row_is_sentinel(row)


def row_all_pass(row: dict[str, str]) -> bool:
    """True iff this patched row recorded pass=true.

    Baseline rows have no pass status of their own; callers should look at the
    matching patched rows in the same batch via ``batch_all_pass``.
    """
    ap = (row.get("pass") or "").strip().lower()
    return ap in ("true", "1", "yes")


@dataclass(frozen=True)
class PublishFilter:
    """Row-selection policy for publish-speedups."""

    select: str = "full"          # full | latest-per-tier | latest-run | tail
    tail: int | None = None       # used when select == tail
    tiers: tuple[str, ...] | None = None
    platform: str | None = None   # win | mac | unknown
    max_threads: int | None = None
    all_pass_only: bool = False
    require_all_tiers: bool = False
    require_all_pass: bool = False


# A "batch" is one verify_patch() invocation: same timestamp, host config,
# tier, threads, version. baseline rows + patched rows share this key.
_BATCH_KEY_COLS = (
    "timestamp", "patch_name", "tier",
    "system_os", "system_cpu", "system_threads", "framework_version",
)


def _batch_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(c) or "").strip() for c in _BATCH_KEY_COLS)


def _tier_ok(row: dict[str, str], tiers: tuple[str, ...] | None) -> bool:
    tier = normalize_tier((row.get("tier") or "").strip())
    if tier not in TIERS:
        return False
    if tiers is not None and tier not in tiers:
        return False
    return True


def _platform_ok(row: dict[str, str], platform: str | None) -> bool:
    if platform is None:
        return True
    return _normalize_platform(row.get("system_os")) == platform


def _row_threads(row: dict[str, str]) -> int | None:
    raw = (row.get("system_threads") or "").strip()
    if not raw or raw.upper() in {"NA", "NAN", "NONE"}:
        return None
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return None


def _threads_ok(row: dict[str, str], max_threads: int | None) -> bool:
    if max_threads is None:
        return True
    threads = _row_threads(row)
    # Legacy blank thread cells predate explicit system_threads recording and
    # are treated as task-default, which is thread=1 for not_applicable tasks.
    return threads is None or threads <= max_threads


def _row_passes_base_gates(row: dict[str, str], flt: PublishFilter) -> bool:
    """Per-row gates that are independent of other rows in the same batch."""
    if not _tier_ok(row, flt.tiers):
        return False
    if not _platform_ok(row, flt.platform):
        return False
    if not _threads_ok(row, flt.max_threads):
        return False
    if not row_is_publishable(row):
        return False
    return True


def _batch_all_pass(batch_rows: list[dict[str, str]]) -> bool:
    """A batch passes iff every patched row has pass=true."""
    patched = [r for r in batch_rows
               if (r.get("variant") or "").strip() == VARIANT_PATCHED]
    if not patched:
        return False
    return all(row_all_pass(r) for r in patched)


def _batch_is_sentinel(batch_rows: list[dict[str, str]]) -> bool:
    """A batch with no measurements, only failed/skipped attempt metadata."""
    return bool(batch_rows) and all(row_is_sentinel(r) for r in batch_rows)


def _group_by_batch(
    rows: list[dict[str, str]],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for r in rows:
        out.setdefault(_batch_key(r), []).append(r)
    return out


def _select_latest_per_tier(
    batches: dict[tuple[str, ...], list[dict[str, str]]],
) -> list[dict[str, str]]:
    """For each (platform, tier), keep all rows of the batch with max timestamp."""
    winning: dict[tuple[str, str], tuple[str, list[dict[str, str]]]] = {}
    for key, batch_rows in batches.items():
        sample = batch_rows[0]
        tier = (sample.get("tier") or "").strip()
        plat = _normalize_platform(sample.get("system_os"))
        ts = (sample.get("timestamp") or "").strip()
        pkey = (plat, tier)
        prev = winning.get(pkey)
        if prev is None or prev[0] <= ts:
            winning[pkey] = (ts, batch_rows)
    out: list[dict[str, str]] = []
    for _, rows in winning.values():
        out.extend(rows)
    return out


def _select_latest_run(
    batches: dict[tuple[str, ...], list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Return rows from the batch(es) sharing the global max timestamp."""
    if not batches:
        return []
    max_ts = max(
        (rows[0].get("timestamp") or "").strip() for rows in batches.values()
    )
    out: list[dict[str, str]] = []
    for rows in batches.values():
        if (rows[0].get("timestamp") or "").strip() == max_ts:
            out.extend(rows)
    return out


def _select_tail(rows: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    if n < 1:
        raise ValueError("--tail must be >= 1")
    return rows[-n:]


def filter_package_verify_rows(
    rows: list[dict[str, str]], flt: PublishFilter,
) -> list[dict[str, str]]:
    """Apply ``flt`` and return selected long-format rows (no header)."""
    base = [r for r in rows if _row_passes_base_gates(r, flt)]

    if flt.all_pass_only:
        # Batch-level filter: drop measured batches if any patched row failed.
        # Sentinel batches are retained as attempt records; `require_all_pass`
        # below still rejects them when the caller explicitly asks for that.
        grouped = _group_by_batch(base)
        base = []
        for batch_rows in grouped.values():
            if _batch_all_pass(batch_rows) or _batch_is_sentinel(batch_rows):
                base.extend(batch_rows)

    mode = flt.select
    if mode == "full":
        out = list(base)
    elif mode == "latest-per-tier":
        out = _select_latest_per_tier(_group_by_batch(base))
    elif mode == "latest-run":
        out = _select_latest_run(_group_by_batch(base))
    elif mode == "tail":
        n = flt.tail if flt.tail is not None else 5
        out = _select_tail(base, n)
    else:
        raise ValueError(
            f"unknown select mode {mode!r}; expected full, latest-per-tier, "
            f"latest-run, or tail"
        )

    out = _sort_rows_for_output(out)

    if flt.require_all_pass and out:
        grouped = _group_by_batch(out)
        bad = [k for k, batch_rows in grouped.items()
               if not _batch_all_pass(batch_rows)]
        if bad:
            raise PublishFilterError(
                "require-all-pass: selected rows include batch(es) with failed patched reps"
            )

    if flt.require_all_tiers:
        want = flt.tiers if flt.tiers is not None else tuple(TIERS)
        have = {(r.get("tier") or "").strip() for r in out}
        missing = [t for t in want if t not in have]
        if missing:
            raise PublishFilterError(
                f"require-all-tiers: missing tier(s): {', '.join(missing)}"
            )

    return out


class PublishFilterError(ValueError):
    """Selected rows fail a publish gate (--require-all-tiers, etc.)."""


def render_package_verify_tsv(
    header: list[str], rows: list[dict[str, str]],
) -> str:
    """Serialize header + rows to TSV text (trailing newline)."""
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in header})
    return buf.getvalue()


def prepare_publish_content(
    src: Path, flt: PublishFilter,
) -> tuple[str, int, str]:
    """Read ``src``, apply ``flt``, return (tsv_text, n_rows, select_summary)."""
    header, rows = read_package_verify(src)
    if not header:
        raise PublishFilterError(f"empty or unreadable TSV: {src}")
    _assert_long_header(header, src)
    selected = filter_package_verify_rows(rows, flt)
    if not selected and flt.select != "full":
        raise PublishFilterError(
            f"select={flt.select!r} matched 0 rows in {src.name}"
        )
    text = render_package_verify_tsv(header, selected)
    summary = _filter_summary(flt, len(rows), len(selected))
    return text, len(selected), summary


def prune_published_tsv_text(
    text: str, *, max_threads: int | None,
) -> tuple[str, int]:
    """Drop already-published rows outside a thread cap.

    Used for `threading: not_applicable` tasks so a normal merge publish cannot
    keep stale 4t/8t rows that were written before the rule was enforced.
    Returns (filtered_text, removed_rows).
    """
    if max_threads is None or not text.strip():
        return text, 0
    header, rows = _read_tsv_text(text)
    if not header:
        return text, 0
    kept = [r for r in rows if _threads_ok(r, max_threads)]
    removed = len(rows) - len(kept)
    if removed == 0:
        return text, 0
    return render_package_verify_tsv(header, kept), removed


def _assert_long_header(header: list[str], src: Path) -> None:
    if not {"variant", "rep_idx", "sec"}.issubset(set(header)):
        raise PublishFilterError(
            f"{src} has an unrecognized header (missing variant/rep_idx/sec). "
            f"Delete and re-attest."
        )


# --- merge / append helpers for `zyme publish-speedups --write-mode` ---


# Identity columns: rows sharing this tuple describe the same measurement
# event across machines/reruns. The row with the newer `timestamp` wins per
# key (so re-publishing on the same host with a fresher attest replaces the
# matching rep row in place).
_MERGE_KEY_COLS = (
    "patch_name", "tier",
    "system_os", "system_cpu", "system_threads",
    "framework_version",
    "rep_idx", "variant",
)


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(c) or "").strip() for c in _MERGE_KEY_COLS)


def _row_ts(row: dict[str, str]) -> str:
    return (row.get("timestamp") or "").strip()


def _tier_sort_idx(row: dict[str, str]) -> int:
    tier = normalize_tier((row.get("tier") or "").strip())
    return TIERS.index(tier) if tier in TIERS else len(TIERS)


def _variant_sort_idx(row: dict[str, str]) -> int:
    """baseline=0, patched=1, unknown=2 — baseline rows come first in output."""
    v = (row.get("variant") or "").strip()
    if v == VARIANT_BASELINE:
        return 0
    if v == VARIANT_PATCHED:
        return 1
    return 2


def _rep_sort_idx(row: dict[str, str]) -> int:
    try:
        return int((row.get("rep_idx") or "0").strip() or 0)
    except ValueError:
        return 0


def sort_rows_for_output(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Public alias of the canonical row sort — used by writers (R + Py
    verify_patch) to maintain global baseline-first ordering on append."""
    return _sort_rows_for_output(rows)


def _sort_rows_for_output(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stable, deterministic order with all baseline rows above all patched
    rows: variant → tier → platform → threads → timestamp → rep_idx.

    This is the file's canonical layout. ``verify_patch()`` enforces it on
    every append (read-modify-write); ``publish-speedups`` enforces it on
    every merge/overwrite. Crash sentinel rows (variant="") sort last.
    """
    def keyfn(r: dict[str, str]) -> tuple:
        try:
            threads = int((r.get("system_threads") or "0").strip() or 0)
        except ValueError:
            threads = 0
        return (
            _variant_sort_idx(r),
            _tier_sort_idx(r),
            _normalize_platform(r.get("system_os")),
            threads,
            _row_ts(r),
            _rep_sort_idx(r),
        )
    return sorted(rows, key=keyfn)


@dataclass(frozen=True)
class MergeStats:
    added: int       # new keys not in existing
    replaced: int    # keys whose existing row had older (or equal) timestamp
    kept: int        # existing rows untouched (newer-ts than incoming or no incoming match)


def merge_published_tsvs(
    existing_text: str, new_text: str,
) -> tuple[str, MergeStats]:
    """Merge ``new_text`` into ``existing_text`` by ``_MERGE_KEY_COLS``.

    Per key (host × tier × threads × version × rep_idx × variant): keep the
    row with the later ``timestamp``. Tie → incoming wins (so re-publishing
    on the same host overwrites stale rows from earlier runs of the same
    rep/variant).
    Invalid rows are dropped. Crash/OOM sentinel rows are retained because the
    bundled TSV is the package-side attest snapshot and should record failed
    attempts too.
    Headers must match; raises ``PublishFilterError`` otherwise so callers
    can fall back to overwrite explicitly.
    """
    ex_header, ex_rows = _read_tsv_text(existing_text)
    new_header, new_rows = _read_tsv_text(new_text)
    ex_rows = [r for r in ex_rows if row_is_publishable(r)]
    new_rows = [r for r in new_rows if row_is_publishable(r)]
    if not new_header:
        return existing_text, MergeStats(0, 0, len(ex_rows))
    if not ex_header:
        return new_text, MergeStats(len(new_rows), 0, 0)
    if ex_header != new_header:
        if ex_header == LEGACY_LONG_HEADER and new_header == LONG_HEADER:
            ex_header, ex_rows = upgrade_rows_to_long_header(ex_header, ex_rows)
        else:
            raise PublishFilterError(
                "header mismatch between existing and new TSV; "
                "cannot row-merge (use --write-mode overwrite to force)"
            )

    indexed: dict[tuple[str, ...], dict[str, str]] = {
        _row_key(r): r for r in ex_rows
    }
    new_keys: set[tuple[str, ...]] = set()
    added = replaced = kept = 0
    for row in new_rows:
        key = _row_key(row)
        new_keys.add(key)
        prev = indexed.get(key)
        if prev is None:
            indexed[key] = row
            added += 1
        elif _row_ts(prev) <= _row_ts(row):
            indexed[key] = row
            replaced += 1
        else:
            kept += 1
    untouched_existing = sum(
        1 for r in ex_rows if _row_key(r) not in new_keys
    )
    merged = _sort_rows_for_output(list(indexed.values()))
    text = render_package_verify_tsv(ex_header, merged)
    return text, MergeStats(added=added, replaced=replaced,
                            kept=kept + untouched_existing)


def append_published_tsvs(
    existing_text: str, new_text: str,
) -> tuple[str, int]:
    """Append ``new_text`` rows after existing ones; no dedup.

    Headers must match. Returns (merged_text, n_appended). Invalid rows are
    dropped; sentinel rows are retained as attest attempt metadata.
    """
    ex_header, ex_rows = _read_tsv_text(existing_text)
    new_header, new_rows = _read_tsv_text(new_text)
    ex_rows = [r for r in ex_rows if row_is_publishable(r)]
    new_rows = [r for r in new_rows if row_is_publishable(r)]
    if not new_header:
        return existing_text, 0
    if not ex_header:
        return new_text, len(new_rows)
    if ex_header != new_header:
        if ex_header == LEGACY_LONG_HEADER and new_header == LONG_HEADER:
            ex_header, ex_rows = upgrade_rows_to_long_header(ex_header, ex_rows)
        else:
            raise PublishFilterError(
                "header mismatch between existing and new TSV; "
                "cannot append (use --write-mode overwrite to force)"
            )
    combined = ex_rows + new_rows
    text = render_package_verify_tsv(ex_header, combined)
    return text, len(new_rows)


def _read_tsv_text(text: str) -> tuple[list[str], list[dict[str, str]]]:
    if not text.strip():
        return [], []
    reader = csv.DictReader(StringIO(text), delimiter="\t")
    if not reader.fieldnames:
        return [], []
    header = list(reader.fieldnames)
    rows = [dict(r) for r in reader]
    return header, rows


def _filter_summary(flt: PublishFilter, total: int, selected: int) -> str:
    parts = [f"select={flt.select}"]
    if flt.select == "tail":
        parts.append(f"tail={flt.tail or 5}")
    if flt.tiers:
        parts.append(f"tiers={','.join(flt.tiers)}")
    if flt.platform:
        parts.append(f"platform={flt.platform}")
    if flt.max_threads is not None:
        parts.append(f"max-threads={flt.max_threads}")
    if flt.all_pass_only:
        parts.append("all-pass-only")
    if flt.require_all_tiers:
        parts.append("require-all-tiers")
    if flt.require_all_pass:
        parts.append("require-all-pass")
    parts.append(f"{selected}/{total} rows")
    return " ".join(parts)


# --- summarization for downstream consumers (paper plots, autozyme.speedups) ---


@dataclass
class BatchSummary:
    """Aggregate stats for one (timestamp × host × tier × threads × version) batch.

    Produced by ``summarize_batches``; consumed by R/Py ``autozyme.speedups``
    wrappers and the paper plotting code that wants one row per batch with
    ready-to-use speedup numbers.
    """

    timestamp: str
    patch_name: str
    tier: str
    system_os: str
    system_cpu: str
    system_threads: str
    framework_version: str
    n_reps: int
    baseline_sec_secs: list[float]      # per-rep raw seconds
    patched_sec_secs: list[float]
    baseline_peak_mb_peaks: list[float] # per-rep raw peak MB (may be empty)
    patched_peak_mb_peaks: list[float]
    all_pass: bool                      # all patched reps pass=true
    note: str

    @property
    def baseline_sec_mean(self) -> float:
        return sum(self.baseline_sec_secs) / len(self.baseline_sec_secs)

    @property
    def patched_sec_mean(self) -> float:
        return sum(self.patched_sec_secs) / len(self.patched_sec_secs)

    @property
    def speedup_x(self) -> float:
        p = self.patched_sec_mean
        return self.baseline_sec_mean / p if p > 0 else float("nan")


def summarize_batches(rows: list[dict[str, str]]) -> list[BatchSummary]:
    """Group long-format rows by batch and compute summary stats per batch.

    Invalid rows are skipped silently. Sentinel-only batches are returned with
    empty measurement arrays so attest coverage scanners can mark them as
    crashed; speedup readers can continue to ignore them.
    """
    publishable = [r for r in rows if row_is_publishable(r)]
    grouped = _group_by_batch(publishable)
    out: list[BatchSummary] = []
    for key, batch_rows in grouped.items():
        baseline = sorted(
            (r for r in batch_rows
             if (r.get("variant") or "") == VARIANT_BASELINE),
            key=_rep_sort_idx,
        )
        patched = sorted(
            (r for r in batch_rows
             if (r.get("variant") or "") == VARIANT_PATCHED),
            key=_rep_sort_idx,
        )
        if not baseline or not patched:
            sentinels = [r for r in batch_rows if row_is_sentinel(r)]
            if sentinels:
                sample = sentinels[-1]
                out.append(BatchSummary(
                    timestamp=(sample.get("timestamp") or "").strip(),
                    patch_name=(sample.get("patch_name") or "").strip(),
                    tier=(sample.get("tier") or "").strip(),
                    system_os=(sample.get("system_os") or "").strip(),
                    system_cpu=(sample.get("system_cpu") or "").strip(),
                    system_threads=(sample.get("system_threads") or "").strip(),
                    framework_version=(
                        sample.get("framework_version") or ""
                    ).strip(),
                    n_reps=0,
                    baseline_sec_secs=[],
                    patched_sec_secs=[],
                    baseline_peak_mb_peaks=[],
                    patched_peak_mb_peaks=[],
                    all_pass=False,
                    note=(sample.get("note") or "").strip(),
                ))
            continue
        try:
            b_secs = [float((r.get("sec") or "0").strip()) for r in baseline]
            p_secs = [float((r.get("sec") or "0").strip()) for r in patched]
        except ValueError:
            continue
        b_peaks = _parse_float_list(r.get("peak_mb") for r in baseline)
        p_peaks = _parse_float_list(r.get("peak_mb") for r in patched)
        sample = patched[0]
        out.append(BatchSummary(
            timestamp=(sample.get("timestamp") or "").strip(),
            patch_name=(sample.get("patch_name") or "").strip(),
            tier=(sample.get("tier") or "").strip(),
            system_os=(sample.get("system_os") or "").strip(),
            system_cpu=(sample.get("system_cpu") or "").strip(),
            system_threads=(sample.get("system_threads") or "").strip(),
            framework_version=(sample.get("framework_version") or "").strip(),
            n_reps=len(p_secs),
            baseline_sec_secs=b_secs,
            patched_sec_secs=p_secs,
            baseline_peak_mb_peaks=b_peaks,
            patched_peak_mb_peaks=p_peaks,
            all_pass=_batch_all_pass(batch_rows),
            note=(sample.get("note") or "").strip(),
        ))
    return out


def _parse_float_list(cells) -> list[float]:
    out: list[float] = []
    for c in cells:
        s = (c or "").strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out
