"""results.tsv I/O + per-(dataset, thread) history queries.

This module owns both the schema-management of results.tsv (header
migration for prompt_id and thread columns) and the read-time queries
that the CLI uses to compute baseline/best/CV tables.

K2 schema:
  - `thread` (integer): parallel resource count axis. Verify sweeps it;
    baselines are recorded once per (tier, thread) so each verify cell
    has a matching same-thread baseline.

V1 had a `thread_mode` column conflating "upstream config" + "thread
count". K2 dropped the upstream-config axis entirely ("just open a
new task" is the natural workflow for non-thread differences) and
isolates thread as the only baseline axis. V1 columns are read as
junk by readers (ignored) and stripped on first K2 op.

Cross-module helpers that stay in zyme.utils (imported here):
  - `has_zyme_meta`, `get_prompt_id_for_task`, `baseline_stash_path`.
  - The constants `LEGACY_THREAD`, `ZYME_META_FILENAME`, and the
    `RESULTS_HEADER_*` strings.
"""
import json
import re
import statistics
from pathlib import Path

from zyme.utils import (
    LEGACY_MODE,
    LEGACY_THREAD,
    RESULTS_HEADER_BASE,
    RESULTS_HEADER_WITH_PROMPT_ID,
    baseline_stash_path,
    get_prompt_id_for_task,
    has_zyme_meta,
)


def _row_thread(parts: list, col: dict) -> int:
    """Extract the `thread` cell from a results.tsv / verify.tsv row.

    Falls back to LEGACY_THREAD (= 1) when the column is absent or
    the cell can't be parsed as an integer (e.g., empty backfill, or
    junk from a V1 file we haven't migrated yet).
    """
    idx = col.get("thread")
    if idx is None or len(parts) <= idx:
        return LEGACY_THREAD
    val = parts[idx].strip()
    if not val:
        return LEGACY_THREAD
    try:
        return int(val)
    except ValueError:
        return LEGACY_THREAD


def _row_mode(parts: list, col: dict) -> str:
    """Extract the deprecated `thread_mode`/`mode` cell from a row."""
    idx = col.get("thread_mode")
    if idx is None:
        idx = col.get("mode")
    if idx is None or len(parts) <= idx:
        return LEGACY_MODE
    val = parts[idx].strip()
    return val if val else LEGACY_MODE


def _legacy_mode_target(task_dir: Path, mode, col: dict) -> str | None:
    """Return a legacy mode filter only for old files carrying a mode column."""
    if "thread_mode" not in col and "mode" not in col:
        return None
    if mode is not None:
        return str(mode)
    try:
        from zyme.parsers.task_yaml import parse_active_mode
        return parse_active_mode(task_dir / "task.yaml")
    except Exception:
        return LEGACY_MODE


def _matches_legacy_mode(parts: list, col: dict, target: str | None) -> bool:
    return target is None or _row_mode(parts, col) == target


def results_header_for_task(task_dir: Path) -> str:
    """Header line (with trailing \\n) — 12-col when bench task, 11-col otherwise."""
    return RESULTS_HEADER_WITH_PROMPT_ID if has_zyme_meta(task_dir) else RESULTS_HEADER_BASE


def append_prompt_id_to_row(task_dir: Path, row: str) -> str:
    """If task has .zyme_meta.yaml, splice prompt_id at the end of the row.

    Caller is expected to pass a K2 base row (12 cols, 11 tabs, ending in
    `thread`). Idempotent: if the row already has prompt_id (>= 12 tabs),
    leave it alone.

    DEPRECATED: prefer `append_results_row` which builds rows from the
    on-disk header order, so it can't get the column position wrong even
    if the file was migrated by a different schema variant. Kept here only
    until the remaining callsite in `cmd_baseline_rebench` is migrated.
    """
    if not has_zyme_meta(task_dir):
        return row
    pid = get_prompt_id_for_task(task_dir)
    body = row[:-1] if row.endswith("\n") else row
    if body.count("\t") >= 12:
        return row
    return body + "\t" + pid + "\n"


def _format_results_row(header_line: str, values: dict) -> str:
    """Build a TSV row matching the on-disk header order.

    Defensive against schema drift — callers pass `{col_name: stringified_value}`
    and this helper maps each value to whatever position the actual file's
    header uses. Columns not present in `values` render as empty.

    This is the load-bearing fix for the "writer hardcodes column position"
    bug: pre-K2 tasks migrated by older variants of `_migrate_tsv_k2` ended
    up with `[..., description, thread, phase]` instead of the canonical
    `[..., description, phase, thread]`. The old writers built rows by
    sequence and silently put `phase` strings into the `thread` column on
    those tasks — `get_baseline_speed` would then read `int("optimize")`,
    fall back to LEGACY_THREAD, and the baseline lookup would silently miss.
    """
    cols = header_line.rstrip("\n").split("\t")
    return "\t".join(str(values.get(c, "")) for c in cols) + "\n"


def append_results_row(results_tsv: Path, task_dir: Path, values: dict) -> None:
    """Append one row to results.tsv. Creates the file with the canonical
    K2 header (12 cols base + prompt_id when bench task) if absent. Looks
    up actual on-disk column order to place each value in `values` at the
    right position.

    Auto-fills `prompt_id` from `.zyme_meta.yaml` when the header has that
    column and the caller didn't pass it explicitly.

    Use this in place of constructing rows by hand at CLI write sites.
    """
    if not results_tsv.exists():
        results_tsv.write_text(results_header_for_task(task_dir))
    header_line = results_tsv.read_text(encoding="utf-8").splitlines()[0]
    cols = header_line.split("\t")
    if "prompt_id" in cols and "prompt_id" not in values:
        values = dict(values)  # don't mutate caller's dict
        values["prompt_id"] = get_prompt_id_for_task(task_dir)
    with open(results_tsv, "a") as f:
        f.write(_format_results_row(header_line, values))


def ensure_results_schema(task_dir: Path, results_tsv: Path) -> None:
    """If task is a bench task and results.tsv exists without prompt_id,
    rewrite it appending the prompt_id column (backfill empty on existing rows).

    Idempotent. No-op when:
      - task has no .zyme_meta.yaml (don't migrate plain tasks)
      - results.tsv doesn't exist (will be written with the right header on first append)
      - results.tsv header already has prompt_id

    Note: this routine handles only the bench (prompt_id) migration. The
    thread migration is performed by `ensure_k2_schema` below, triggered
    lazily by K2 ops. The two are independent — pre-K2 bench tasks have
    prompt_id but no thread column, and the column order on disk is
    preserved exactly as-is until a K2 op fires.
    """
    if not has_zyme_meta(task_dir):
        return
    if not results_tsv.exists():
        return
    text = results_tsv.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return
    header_cols = lines[0].split("\t")
    if "prompt_id" in header_cols:
        return
    new_lines = [lines[0] + "\tprompt_id"]
    for line in lines[1:]:
        new_lines.append(line + "\t")  # blank prompt_id for pre-bench rows
    results_tsv.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _migrate_tsv_k2(path: Path,
                    insert_after: str | None = "phase",
                    backfill_thread: str = str(LEGACY_THREAD)) -> str:
    """In-place migrate a single TSV to K2 schema.

    K2 cares about exactly one new column: `thread` (integer). The V1
    `thread_mode` column, if present, is left untouched on disk — readers
    ignore it. Ensures `thread` exists; missing column is inserted after
    `insert_after` (or appended at end), with rows backfilled to
    `backfill_thread`.

    A `.prek2.bak` snapshot is left next to the file the FIRST time we
    rewrite it. Subsequent calls are no-ops.

    Returns one of: 'absent', 'already', 'migrated'.
    """
    if not path.exists():
        return "absent"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return "absent"
    header_cols = lines[0].split("\t")

    if "thread" in header_cols:
        return "already"

    bak = path.with_suffix(path.suffix + ".prek2.bak")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")

    if insert_after and insert_after in header_cols:
        anchor_idx = header_cols.index(insert_after)
    else:
        anchor_idx = len(header_cols) - 1

    new_header_cols = list(header_cols)
    new_header_cols.insert(anchor_idx + 1, "thread")

    new_lines = ["\t".join(new_header_cols)]
    insertion_point = anchor_idx + 1
    for line in lines[1:]:
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split("\t")
        while len(parts) < len(header_cols):
            parts.append("")
        parts.insert(insertion_point, backfill_thread)
        new_lines.append("\t".join(parts))

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return "migrated"


def _normalize_phase_thread_order(path: Path) -> str:
    """Detect [..., thread, phase] reverse-order schema and reorder to
    canonical [..., phase, thread]. Reorders header + every data row's two
    cells in-place.

    This catches pre-K2 tasks migrated by an OLDER variant of
    `_migrate_tsv_k2` that inserted thread BEFORE phase (current code
    inserts after). The canonical writers always wrote at positions
    matching `[phase, thread]`; on a reverse-schema file that meant `phase`
    strings went into the `thread` column and vice versa, silently breaking
    `get_baseline_speed`.

    Returns one of: 'absent', 'n/a' (no thread or no phase column), 'already'
    (canonical), 'normalized' (rewrote). Leaves `.precolfix.bak` next to
    the file the FIRST time we rewrite it; subsequent calls are no-ops.
    """
    if not path.exists():
        return "absent"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return "absent"
    header_cols = lines[0].split("\t")
    if "thread" not in header_cols or "phase" not in header_cols:
        return "n/a"
    t_idx = header_cols.index("thread")
    p_idx = header_cols.index("phase")
    if p_idx < t_idx:
        return "already"  # canonical [..., phase, thread]

    bak = path.with_suffix(path.suffix + ".precolfix.bak")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")

    new_header = list(header_cols)
    new_header[t_idx], new_header[p_idx] = new_header[p_idx], new_header[t_idx]
    new_lines = ["\t".join(new_header)]
    for line in lines[1:]:
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split("\t")
        while len(parts) < len(header_cols):
            parts.append("")
        parts[t_idx], parts[p_idx] = parts[p_idx], parts[t_idx]
        new_lines.append("\t".join(parts))
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return "normalized"


def ensure_k2_schema(task_dir: Path) -> dict:
    """K2 lazy schema migration. Brings results.tsv / baselines_history.tsv
    up to K2 by adding the `thread` column (backfill `1`) if absent, then
    normalizing any reverse-order [..., thread, phase] header back to the
    canonical [..., phase, thread] (older migration variants put thread
    before phase, breaking the position-hardcoded writers).

    Triggered by `cmd_record_baseline`, `cmd_baseline_rebench`, and
    `cmd_verify`'s pre-flight on append/top-up against an old verify.tsv.

    `.prek2.bak` snapshots are left so the migration is reversible by
    hand; `.precolfix.bak` for the order normalization. Idempotent.
    Returns {file_label: 'migrated' | 'already' | 'absent' | 'normalized'}
    for transparent reporting.

    V1 `thread_mode` columns are left untouched on disk — readers ignore
    them. Stash file is also ignored (K2 dropped the lazy-promotion
    workflow; record-baseline writes directly to results.tsv).
    """
    report = {}
    results_path = task_dir / "results.tsv"
    report["results.tsv"] = _migrate_tsv_k2(results_path, insert_after="phase")
    norm = _normalize_phase_thread_order(results_path)
    if norm == "normalized":
        report["results.tsv"] = "normalized"  # supersedes 'already'/'migrated'
    report["baselines_history.tsv"] = _migrate_tsv_k2(
        task_dir / ".zyme" / "baselines_history.tsv", insert_after=None
    )
    return report


def _migrate_tsv_mode(path: Path, backfill_mode: str = LEGACY_MODE) -> str:
    """Deprecated pre-K2 migration: add `thread_mode` if absent."""
    if not path.exists():
        return "absent"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return "absent"
    header_cols = lines[0].split("\t")
    if "thread_mode" in header_cols or "mode" in header_cols:
        return "already"

    bak = path.with_suffix(path.suffix + ".premode.bak")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")

    new_lines = [lines[0] + "\tthread_mode"]
    for line in lines[1:]:
        if not line.strip():
            new_lines.append(line)
        else:
            new_lines.append(line + "\t" + backfill_mode)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return "migrated"


def ensure_mode_schema(task_dir: Path) -> dict:
    """Deprecated pre-K2 schema migration retained for old task files/tests."""
    return {
        "results.tsv": _migrate_tsv_mode(task_dir / "results.tsv"),
        "baselines_history.tsv": _migrate_tsv_mode(
            task_dir / ".zyme" / "baselines_history.tsv"
        ),
    }


def _resolve_thread_arg(thread: int | None) -> int:
    """Resolve a None thread argument to LEGACY_THREAD (=1).

    Most lookups should pass `thread` explicitly. None is the safe fallback
    for callers that don't yet thread-parameterize (e.g. legacy commands
    that always meant single-threaded).
    """
    if thread is None:
        return LEGACY_THREAD
    return int(thread)


def get_baseline_speed(task_dir: Path, dataset_name: str = "",
                        thread: int | None = None,
                        mode=None):
    """Per-(dataset, thread) baseline: only rows with status ∈ {baseline, oom}.

    Lookup order:
      1. Exact match on (dataset, thread) with a baseline-class status.
      2. Fallback: any baseline-class row matching `dataset` regardless of
         thread (first encountered in file order — typically the thread=1
         upstream baseline).
      3. None if no baseline-class row exists for this dataset.

    The status filter is load-bearing: without it, the first kept candidate
    at a new thread budget would be silently promoted to baseline, and every
    subsequent row's speedup_pct would compare against that candidate
    instead of the original upstream baseline. The fallback preserves "vs
    original upstream" semantics when the agent introduces a new thread
    count without explicitly re-baselining via `zyme reference` /
    `zyme record-baseline`.

    `thread` defaults to LEGACY_THREAD (=1). For files lacking the `thread`
    column, every row is treated as thread=1.

    `mode` is accepted but ignored (K2 dropped the mode axis); kept in the
    signature so V1/K1 callsites compile during the rollout without raising
    TypeError on an unexpected kwarg.
    """
    tsv = task_dir / "results.tsv"
    if not tsv.exists():
        return None
    lines = tsv.read_text().splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].split("\t")
    if ("speed_sec" not in header or "dataset" not in header
            or "status" not in header):
        return None
    col = {n: i for i, n in enumerate(header)}
    target_thread = _resolve_thread_arg(thread)
    target_mode = _legacy_mode_target(task_dir, mode, col)
    fallback = None
    for line in lines[1:]:
        parts = line.split("\t")
        if (len(parts) <= col["speed_sec"]
                or len(parts) <= col["status"]):
            continue
        if not _matches_legacy_mode(parts, col, target_mode):
            continue
        if parts[col["status"]] not in ("baseline", "oom"):
            continue
        row_dataset = parts[col["dataset"]] if col["dataset"] < len(parts) else ""
        if dataset_name and row_dataset != dataset_name:
            continue
        try:
            val = float(parts[col["speed_sec"]])
        except (ValueError, IndexError):
            continue
        if _row_thread(parts, col) == target_thread:
            return val
        if fallback is None:
            fallback = val
    return fallback


def has_baseline_at_thread(task_dir: Path, dataset_name: str,
                            thread: int) -> bool:
    """True iff results.tsv has a baseline-class row (status ∈ {baseline,
    oom}) matching exactly (dataset, thread).

    Used by `zyme run` to distinguish "this thread budget has its own
    upstream baseline" from "we'll be falling back to the dataset's other
    baseline". The fallback case is fine (speedup_pct is still meaningful
    vs upstream-at-thread=1) but worth telling the user about.
    """
    tsv = task_dir / "results.tsv"
    if not tsv.exists():
        return False
    lines = tsv.read_text().splitlines()
    if len(lines) < 2:
        return False
    header = lines[0].split("\t")
    if ("status" not in header or "dataset" not in header):
        return False
    col = {n: i for i, n in enumerate(header)}
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= col["status"]:
            continue
        if parts[col["status"]] not in ("baseline", "oom"):
            continue
        if parts[col["dataset"]] != dataset_name:
            continue
        if _row_thread(parts, col) == int(thread):
            return True
    return False


def get_baseline_status(task_dir: Path, dataset_name: str = "",
                         thread: int | None = None,
                         mode=None) -> str:
    """Status of the baseline row for (dataset, thread).

    Returns the `status` column value ('baseline' or 'oom') of the matching
    baseline-class row. Same lookup rules as `get_baseline_speed`: exact
    (dataset, thread) match preferred; falls back to dataset's baseline at
    any thread; empty string if no baseline-class row exists.

    Used by `zyme verify` to detect OOM-marked (tier, thread) cells.
    """
    tsv = task_dir / "results.tsv"
    if not tsv.exists():
        return ""
    lines = tsv.read_text().splitlines()
    if len(lines) < 2:
        return ""
    header = lines[0].split("\t")
    if "status" not in header or "dataset" not in header:
        return ""
    col = {n: i for i, n in enumerate(header)}
    target_thread = _resolve_thread_arg(thread)
    target_mode = _legacy_mode_target(task_dir, mode, col)
    fallback = ""
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= col["status"]:
            continue
        if not _matches_legacy_mode(parts, col, target_mode):
            continue
        if parts[col["status"]] not in ("baseline", "oom"):
            continue
        if parts[col["dataset"]] != dataset_name:
            continue
        if _row_thread(parts, col) == target_thread:
            return parts[col["status"]]
        if not fallback:
            fallback = parts[col["status"]]
    return fallback


def get_baseline_peak_mb(task_dir: Path, dataset_name: str = "",
                          thread: int | None = None,
                          mode=None):
    """Per-(dataset, thread) baseline peak memory (MB).

    Mirrors `get_baseline_speed`: only rows with status ∈ {baseline, oom}
    qualify, exact (dataset, thread) match wins, falls back to dataset's
    baseline at any thread. Returns None if no baseline-class row exists or
    the column is missing.
    """
    tsv = task_dir / "results.tsv"
    if not tsv.exists():
        return None
    lines = tsv.read_text().splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].split("\t")
    if ("peak_mb" not in header or "dataset" not in header
            or "status" not in header):
        return None
    col = {n: i for i, n in enumerate(header)}
    target_thread = _resolve_thread_arg(thread)
    target_mode = _legacy_mode_target(task_dir, mode, col)
    fallback = None
    for line in lines[1:]:
        parts = line.split("\t")
        if (len(parts) <= col["peak_mb"]
                or len(parts) <= col["status"]):
            continue
        if not _matches_legacy_mode(parts, col, target_mode):
            continue
        if parts[col["status"]] not in ("baseline", "oom"):
            continue
        row_dataset = parts[col["dataset"]]
        if dataset_name and row_dataset != dataset_name:
            continue
        try:
            val = float(parts[col["peak_mb"]])
        except (ValueError, IndexError):
            continue
        if _row_thread(parts, col) == target_thread:
            return val
        if fallback is None:
            fallback = val
    return fallback


def best_speeds_at_dataset(task_dir: Path, dataset_name: str,
                            thread: int | None = None,
                            mode=None):
    """Return list of speed_sec values for current best commit at
    (dataset, thread).

    Reads `.zyme/best.ref` for the best commit SHA, then scans `results.tsv`
    for status ∈ {keep, rerun} rows matching (commit, dataset, thread).
    Returns the raw speeds (caller decides median / mean / CV / n). Empty
    list if best.ref is absent or no matching rows exist. `thread=None`
    resolves to LEGACY_THREAD — keeps "vs best" comparisons within the
    same threading regime.

    `mode` accepted-but-ignored for V1/K1 callsite compatibility.
    """
    z = task_dir / ".zyme"
    best_ref = z / "best.ref"
    if not best_ref.exists():
        return []
    best_full = best_ref.read_text().strip()
    if not best_full:
        return []
    best_short = best_full[:7]
    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        return []
    lines = results_tsv.read_text().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    target_thread = _resolve_thread_arg(thread)
    target_mode = _legacy_mode_target(task_dir, mode, col)
    speeds = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        if not _matches_legacy_mode(parts, col, target_mode):
            continue
        if parts[1] != best_short or parts[2] != dataset_name:
            continue
        if parts[6] not in ("keep", "rerun"):
            continue
        if _row_thread(parts, col) != target_thread:
            continue
        try:
            speeds.append(float(parts[3]))
        except ValueError:
            continue
    return speeds


def phase_speeds_at_dataset(task_dir: Path, dataset_name: str, phase: str = "validate"):
    """Return list of speed_sec values for keep+rerun rows in a given phase + dataset.

    Used by `zyme status --phase validate` to compute n / median / CV from
    measurements in the validate phase regardless of whether the commit
    matches `best.ref`. (`best_speeds_at_dataset` filters by best.ref's
    commit, which excludes rerun rows written at the same converged HEAD
    when best.ref hasn't been updated to point at this phase's measurements.)
    """
    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        return []
    lines = results_tsv.read_text().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    if "phase" not in col:
        return []  # pre-phase schema; validate phase didn't exist when written
    speeds = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        if parts[col["dataset"]] != dataset_name:
            continue
        if parts[col["phase"]] != phase:
            continue
        if parts[col["status"]] not in ("keep", "rerun"):
            continue
        try:
            speeds.append(float(parts[col["speed_sec"]]))
        except ValueError:
            continue
    return speeds


def phase_cv_at_dataset(task_dir: Path, dataset_name: str, phase: str = "validate"):
    """(median, cv_pct, n) for keep+rerun rows in a given phase + dataset."""
    speeds = phase_speeds_at_dataset(task_dir, dataset_name, phase=phase)
    if not speeds:
        return (None, None, 0)
    speeds_sorted = sorted(speeds)
    n = len(speeds_sorted)
    median = (
        speeds_sorted[n // 2] if n % 2
        else 0.5 * (speeds_sorted[n // 2 - 1] + speeds_sorted[n // 2])
    )
    if n < 3:
        return (median, None, n)
    mean = sum(speeds_sorted) / n
    cv_pct = (statistics.stdev(speeds_sorted) / mean) * 100.0 if mean > 0 else None
    return (median, cv_pct, n)


def find_best_speed_at_dataset(task_dir: Path, dataset_name: str,
                                 thread: int | None = None,
                                 mode=None):
    """Median speed_sec of the current best commit at (dataset, thread).

    Used by `zyme run` to print a per-row "vs best" delta within the same
    threading regime — comparing across thread counts would be an
    apples/oranges trap.

    `mode` accepted-but-ignored for V1/K1 callsite compatibility.
    """
    speeds = best_speeds_at_dataset(task_dir, dataset_name, thread=thread, mode=mode)
    if not speeds:
        return None
    speeds = sorted(speeds)
    n = len(speeds)
    if n % 2:
        return speeds[n // 2]
    return 0.5 * (speeds[n // 2 - 1] + speeds[n // 2])


def best_wall_cpu_ratio_at_dataset(task_dir: Path, dataset_name: str,
                                     thread: int | None = None,
                                     mode=None):
    """Return (median_ratio, n) of wall/cpu over best's keep+rerun rows
    in (dataset, thread).

    Used by `zyme run` to detect host-load events: when a same-commit
    measurement's wall/cpu ratio drifts ≥2× from the historical median,
    the wall blow-up is much more likely host contention than a code
    regression. Returns (None, n) when n < 2 — needs at least a couple
    of historical points to compare against.
    """
    z = task_dir / ".zyme"
    best_ref = z / "best.ref"
    if not best_ref.exists():
        return (None, 0)
    best_full = best_ref.read_text().strip()
    if not best_full:
        return (None, 0)
    best_short = best_full[:7]
    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        return (None, 0)
    lines = results_tsv.read_text().splitlines()
    if len(lines) < 2:
        return (None, 0)
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    target_thread = _resolve_thread_arg(thread)
    target_mode = _legacy_mode_target(task_dir, mode, col)
    ratios = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        if not _matches_legacy_mode(parts, col, target_mode):
            continue
        if parts[1] != best_short or parts[2] != dataset_name:
            continue
        if parts[6] not in ("keep", "rerun"):
            continue
        if _row_thread(parts, col) != target_thread:
            continue
        try:
            wall = float(parts[3])
        except ValueError:
            continue
        if wall <= 0:
            continue
        try:
            metrics = json.loads(parts[7] or "{}")
        except (ValueError, json.JSONDecodeError):
            continue
        cpu = metrics.get("cpu_sec")
        try:
            cpu = float(cpu) if cpu is not None else None
        except (TypeError, ValueError):
            cpu = None
        if cpu is None or cpu <= 0:
            continue
        ratios.append(wall / cpu)
    if len(ratios) < 2:
        return (None, len(ratios))
    return (statistics.median(ratios), len(ratios))


def best_cv_at_dataset(task_dir: Path, dataset_name: str,
                        thread: int | None = None,
                        mode=None):
    """Return (median, cv_pct, n) for current best commit at
    (dataset, thread).

    CV = stdev / mean × 100, computed across all keep+rerun rows of the best
    commit at this (tier, thread) within the same threading regime. Returns
    (None, None, n) when n < 3 (CV needs at least 3 samples to be meaningfully
    informative; below that the agent should just rerun more before trusting
    any CV-based heuristic).
    """
    speeds = best_speeds_at_dataset(task_dir, dataset_name, thread=thread, mode=mode)
    n = len(speeds)
    if n < 3:
        return (None, None, n)
    median = statistics.median(speeds)
    mean = statistics.mean(speeds)
    stdev = statistics.stdev(speeds)
    cv_pct = (stdev / mean * 100.0) if mean > 0 else 0.0
    return (median, cv_pct, n)


def parse_log(log: str):
    """Extract (speed_sec, peak_mb, metrics_dict, status) from zyme.runner output."""
    speed_sec = None
    peak_mb = None
    metrics = {}
    # Crash detection looks for the canonical _summary line emitted by the
    # runner when a subprocess returns non-zero (or an evaluate fails). Don't
    # match the substring "CRASH:" anywhere in the log — pipeline code might
    # legitimately print "CRASH: investigating gc artifact ..." while running
    # fine, and that should NOT be treated as a crash.
    crashed = "status:           crash" in log

    skip = {"task", "speed_sec", "baseline_sec", "speedup_pct",
            "peak_memory_mb", "internal_time", "status", "peak_mb",
            "Lang", "Start"}

    for line in log.splitlines():
        m = re.match(r"^([A-Za-z0-9_.]+):\s+(.+?)\s*$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key == "speed_sec":
            try:
                speed_sec = float(val)
            except ValueError:
                pass
        elif key in ("peak_memory_mb", "peak_mb"):
            try:
                peak_mb = float(val)
            except ValueError:
                pass
        elif key in skip:
            continue
        else:
            # Only keep numeric values as metrics; drop string-y diagnostic lines.
            try:
                metrics[key] = float(val)
            except ValueError:
                pass

    status = "crash" if crashed else "pending"
    return speed_sec, peak_mb, metrics, status


def results_header_width(results_tsv: Path) -> int:
    """Return the column count of `results.tsv`'s header row.

    Used by writers that want to pad rows to the file's actual width
    (instead of hardcoding a count that drifts as columns are added).
    Returns 0 when the file doesn't exist or is empty.
    """
    if not results_tsv.exists():
        return 0
    with open(results_tsv) as f:
        first = f.readline().rstrip("\n")
    if not first:
        return 0
    return len(first.split("\t"))


def results_phase_index(results_tsv: Path):
    """Return the column index of the `phase` field, or None if absent.

    Backward-compat helper: old results.tsv files have 10 columns and no
    phase column. Readers use this to locate phase if present and treat
    its absence as `phase=optimize` for every row.
    """
    if not results_tsv.exists():
        return None
    with open(results_tsv) as f:
        first = f.readline().rstrip("\n")
    if not first:
        return None
    cols = first.split("\t")
    try:
        return cols.index("phase")
    except ValueError:
        return None


def migrate_results_add_phase(results_tsv: Path):
    """Backfill an existing pre-phase results.tsv with a `phase` column.

    Idempotent: if the header already has `phase`, do nothing. Otherwise
    rewrite the header and append an empty 11th field to every data row
    (empty phase reads as `optimize` per `row_phase()`).
    """
    if not results_tsv.exists():
        return
    text = results_tsv.read_text()
    lines = text.splitlines()
    if not lines:
        return
    if "phase" in lines[0].split("\t"):
        return
    new_lines = [lines[0] + "\tphase"]
    for line in lines[1:]:
        new_lines.append(line + "\t")
    results_tsv.write_text("\n".join(new_lines) + "\n")


def row_phase(row: dict) -> str:
    """Read `phase` field from a parsed results.tsv row dict.

    Empty / missing phase -> "optimize" (legacy rows predate the column).
    """
    p = (row.get("phase") or "").strip()
    return p if p else "optimize"


def update_last_status(results_tsv: Path, status: str, description: str):
    """Flip the most recent pending row (the decision row for this commit).

    accept/reject decisions apply to the hypothesis commit being judged, not
    to subsequent --rerun measurement rows. --rerun writes rows at
    status=rerun (already terminal), which we leave untouched. The decision
    row is the most recent row at status=pending — that's what we update.

    Columns (0-indexed): 0 round, 1 commit, 2 dataset, 3 speed_sec, 4 speedup_pct,
    5 peak_mb, 6 status, 7 metrics_json, 8 hypothesis, 9 description, 10 phase.
    """
    if not results_tsv.exists():
        return
    lines = results_tsv.read_text().splitlines()
    if len(lines) < 2:
        return
    width = max(10, len(lines[0].split("\t")))
    target = None
    for i in range(len(lines) - 1, 0, -1):
        parts = lines[i].split("\t")
        if len(parts) > 6 and parts[6] == "pending":
            target = i
            break
    if target is None:
        return
    parts = lines[target].split("\t")
    while len(parts) < width:
        parts.append("")
    parts[6] = status
    if description:
        parts[9] = description
    lines[target] = "\t".join(parts)
    results_tsv.write_text("\n".join(lines) + "\n")


def count_decision_rounds(results_tsv: Path, phase: str = "optimize") -> int:
    """Count distinct decision rounds for the given phase.

    Round labels in `results.tsv`:
      - `0`           — upstream baseline (status=baseline). Doesn't count.
      - `N` (integer) — decision row at decision round N. **Counts.**
      - `N.k`         — rerun OR multi-tier secondary measurement attached to
                        decision round N (status=rerun). Doesn't count.

    Phase filter:
      - "optimize" (default): counts the 50-round optimize budget. Rows
        whose `phase` is empty are treated as optimize (legacy compat).
      - "validate": counts scale-fix rounds (the Phase 3 fix-loop budget).
      - "all": ignore phase column entirely.
    """
    if not results_tsv.exists():
        return 0
    lines = results_tsv.read_text().splitlines()
    if not lines:
        return 0
    header = lines[0].split("\t")
    try:
        phase_idx = header.index("phase")
    except ValueError:
        phase_idx = None
    n = 0
    for line in lines[1:]:
        parts = line.split("\t")
        if not parts:
            continue
        try:
            r = int(parts[0])
        except ValueError:
            continue  # has a `.k` suffix -> not a decision row
        if r < 1:
            continue
        if phase != "all":
            row_p = parts[phase_idx].strip() if phase_idx is not None and phase_idx < len(parts) else ""
            if not row_p:
                row_p = "optimize"
            if row_p != phase:
                continue
        n += 1
    return n


def next_rerun_seq(results_tsv: Path, parent_round: int) -> int:
    """Next available `.k` suffix under decision round `parent_round`.

    Scans `results.tsv` for rows whose round label starts with f"{parent_round}."
    (e.g. "100.1", "100.2") and returns max(k) + 1. Returns 1 if none exist.

    Used both for `--rerun` rows and for multi-tier secondary measurements
    (--extra-tiers entries), so a later rerun won't collide with an earlier
    secondary's `.k`.
    """
    if not results_tsv.exists():
        return 1
    prefix = f"{parent_round}."
    max_seq = 0
    for line in results_tsv.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if not parts:
            continue
        round_field = parts[0]
        if not round_field.startswith(prefix):
            continue
        try:
            seq = int(round_field[len(prefix):])
        except ValueError:
            continue
        if seq > max_seq:
            max_seq = seq
    return max_seq + 1


def dataset_in_results(results_tsv: Path, dataset_name: str,
                        thread: int | None = None,
                        mode=None) -> bool:
    """True iff results.tsv has at least one data row matching (dataset, thread).

    `thread=None` -> match any thread (legacy "any row for this dataset"
    behavior). When given, only rows matching that thread count. Files
    lacking the `thread` column are treated as all-rows = LEGACY_THREAD.

    `mode` accepted-but-ignored for V1/K1 callsite compatibility.
    """
    if not results_tsv.exists():
        return False
    lines = results_tsv.read_text().splitlines()
    if len(lines) < 2:
        return False
    header = lines[0].split("\t")
    if "dataset" not in header:
        return False
    col = {n: i for i, n in enumerate(header)}
    target_mode = None
    if "thread_mode" in col or "mode" in col:
        target_mode = str(mode) if mode is not None else None
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= col["dataset"]:
            continue
        if not _matches_legacy_mode(parts, col, target_mode):
            continue
        if parts[col["dataset"]] != dataset_name:
            continue
        if thread is not None and _row_thread(parts, col) != int(thread):
            continue
        return True
    return False
