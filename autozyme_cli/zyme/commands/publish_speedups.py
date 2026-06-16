"""`zyme publish-speedups` — copy each task's package_verify.tsv into its
matching patch's bundled snapshot slot in autozyme_{r,py}.

Task-side `package_verify.tsv` is the live source of truth (every
`zyme attest` appends to it); the bundled `speedups.tsv` is the frozen
snapshot the installed autozyme package ships to end users. `zyme attest`
auto-merges passing publishable rows and crash/OOM sentinel rows. This command
remains the explicit repair/backfill path for existing attest history,
filtering, dry-runs, or manual snapshot refreshes.

Mapping uses the same `Lifted from autozyme task <dir_name>` marker every
patch file carries — same index `zyme scan` consults. Patches without a
matching task dir, or task dirs without a `package_verify.tsv`, are
listed as skipped (with reason) but not treated as errors.

Task selection:
  - No positional args → every patch in the lifted-from index (legacy default).
  - One or more task/patch names → only those entries (task dir name or
    registered patch stem, e.g. `test_decontx` or `decontx`).

Row selection (--select):
  - `full` (default) — entire file (optionally gated by --tiers / --platform).
  - `latest-per-tier` — one row per (platform, tier), latest timestamp wins;
    matches what `autozyme.speedups()` returns to end users.
  - `latest-run` — all rows sharing the maximum timestamp (one attest batch
    when tiers were written in a single verify_patch call).
  - `tail` — last N data rows after other filters (--tail N, default 5).
  - Task policy is always applied: `threading: not_applicable` publishes only
    thread=1 rows and prunes stale multi-thread rows from existing snapshots.
    Pass `--allow-not-applicable-threads` only for diagnostics/reclassification
    work where those rows are intentionally being retained.

Gates (--require-all-tiers, --require-all-pass) skip a task instead of
erroring the whole command when the selected snapshot is incomplete.
"""
from __future__ import annotations

import sys
from pathlib import Path

from zyme.parsers.package_verify_tsv import (
    PublishFilter,
    PublishFilterError,
    append_published_tsvs,
    merge_published_tsvs,
    prepare_publish_content,
    prune_published_tsv_text,
)
from zyme.parsers.task_yaml import parse_threading_mode
from zyme.scan import (
    find_framework_root, find_workspace_root,
    _build_lifted_from_index,
)


_CATEGORY_DIRS = (
    "test_core_singlecell",
    "test_general_bio",
    "test_non_bio",
    "test_spatial",
    "test_autopilotmode",
)


def _find_task_dir(workspace: Path, task_name: str,
                   framework: Path | None = None) -> Path | None:
    """Locate the task directory across the layouts we support."""
    for cat in _CATEGORY_DIRS:
        candidate = workspace / cat / task_name
        if candidate.is_dir():
            return candidate
    if framework is not None:
        for cat in _CATEGORY_DIRS:
            candidate = framework / "optimized_task" / cat / task_name
            if candidate.is_dir():
                return candidate
    candidate = workspace / task_name
    if candidate.is_dir():
        return candidate
    return None


def _dest_for_patch(framework: Path, patch_path: Path, plat: str = "") -> Path:
    """Resolve where a patch's bundled speedups raw should live.

    Per-platform split: when ``plat`` (mac/win/other) is given, write to the
    platform shard ``speedups.<plat>.tsv`` so each machine only ever touches
    its own file and cross-machine git merges stay disjoint. When ``plat`` is
    empty the legacy combined ``speedups.tsv`` path is returned.

    R folder layout: ``inst/patches/<name>/patch.R``  → sibling speedups shard.
    R legacy layout: ``inst/patches/<name>.R`` → ``inst/speedups/<name>[.<plat>].tsv``.
    Py: next to the patch's ``__init__.py``.
    """
    fname = f"speedups.{plat}.tsv" if plat else "speedups.tsv"
    if patch_path.suffix == ".R":
        # Folder layout: patch lives at inst/patches/<name>/patch.R
        if patch_path.name == "patch.R":
            return patch_path.parent / fname
        # Legacy single-file layout
        stem = patch_path.stem
        legacy_name = f"{stem}.{plat}.tsv" if plat else f"{stem}.tsv"
        return framework / "autozyme_r" / "inst" / "speedups" / legacy_name
    return patch_path.parent / fname


def _partition_by_platform(content: str) -> dict[str, str]:
    """Split published-rows TSV text into per-platform shards.

    Classifies each data row by its ``system_os`` column into ``mac`` / ``win``
    / ``linux`` / ``other`` and returns ``{plat: tsv_text}`` where each value is
    a complete TSV (the shared header + that platform's rows). Empty input ->
    ``{}``.
    """
    lines = content.splitlines()
    if not lines:
        return {}
    header = lines[0]
    cols = header.split("\t")
    os_idx = cols.index("system_os") if "system_os" in cols else -1
    buckets: dict[str, list[str]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        os_val = parts[os_idx] if 0 <= os_idx < len(parts) else ""
        s = os_val.lower()
        if "mac" in s or "apple" in s or "darwin" in s:
            plat = "mac"
        elif "win" in s:
            plat = "win"
        elif "linux" in s:
            plat = "linux"
        else:
            plat = "other"
        buckets.setdefault(plat, []).append(line)
    return {p: "\n".join([header, *rows]) + "\n" for p, rows in buckets.items()}


def _build_patch_name_index(index: dict[str, str]) -> dict[str, str]:
    """Map registered patch stem -> task dir name."""
    by_patch: dict[str, str] = {}
    for task_name, patch_path_str in index.items():
        p = Path(patch_path_str)
        # Patch name resolution:
        #   R folder layout (.../patches/<name>/patch.R): use parent dir name
        #   R legacy single-file (.../patches/<name>.R):   use file stem
        #   Py (.../<name>/__init__.py):                    use parent dir name
        if p.suffix == ".R" and p.name != "patch.R":
            patch_name = p.stem
        else:
            patch_name = p.parent.name
        by_patch[patch_name] = task_name
    return by_patch


def _resolve_task_filter(
    names: list[str], index: dict[str, str],
) -> tuple[set[str], list[str]]:
    """Resolve positional names to task dir names; return (selected, unknown)."""
    if not names:
        return set(index.keys()), []
    by_patch = _build_patch_name_index(index)
    selected: set[str] = set()
    unknown: list[str] = []
    for raw in names:
        key = raw.strip()
        if not key:
            continue
        if key in index:
            selected.add(key)
        elif key in by_patch:
            selected.add(by_patch[key])
        else:
            unknown.append(key)
    return selected, unknown


def _publish_filter_from_args(args) -> PublishFilter:
    tiers = None
    if getattr(args, "tiers", None):
        tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
        if not tiers:
            from zyme.utils import die
            die("--tiers parsed to empty after splitting")
    tail = getattr(args, "tail", None)
    if args.select == "tail" and tail is None:
        tail = 5
    return PublishFilter(
        select=args.select,
        tail=tail,
        tiers=tiers,
        platform=getattr(args, "platform", None),
        all_pass_only=bool(getattr(args, "all_pass_only", False)),
        require_all_tiers=bool(getattr(args, "require_all_tiers", False)),
        require_all_pass=bool(getattr(args, "require_all_pass", False)),
    )


def _filter_for_task(
    task_dir: Path, flt: PublishFilter, *, allow_not_applicable: bool,
) -> PublishFilter:
    """Apply task-level publication policy."""
    if (
        allow_not_applicable
        or parse_threading_mode(task_dir / "task.yaml") != "not_applicable"
    ):
        return flt
    return PublishFilter(
        select=flt.select,
        tail=flt.tail,
        tiers=flt.tiers,
        platform=flt.platform,
        max_threads=1,
        all_pass_only=flt.all_pass_only,
        require_all_tiers=flt.require_all_tiers,
        require_all_pass=flt.require_all_pass,
    )


def cmd_publish_speedups(args):
    cwd = Path.cwd()
    workspace = find_workspace_root(cwd)
    if workspace is None:
        print("zyme: no autozyme-framework found above cwd; cannot resolve "
              "task dirs.", file=sys.stderr)
        sys.exit(1)
    framework = find_framework_root(cwd) or (workspace / "autozyme-framework")
    framework = framework.resolve()

    index = _build_lifted_from_index(str(framework))
    if not index:
        print("zyme: no patches with a `Lifted from autozyme task <dir>` "
              "marker found under autozyme_r/inst/patches/ or "
              "autozyme_py/src/autozyme/.", file=sys.stderr)
        sys.exit(1)

    raw_names = list(getattr(args, "tasks", None) or [])
    selected_tasks, unknown = _resolve_task_filter(raw_names, index)
    if unknown:
        print("zyme: unknown task/patch name(s): " + ", ".join(unknown),
              file=sys.stderr)
        sys.exit(1)
    if raw_names and not selected_tasks:
        print("zyme: no tasks matched the given name(s).", file=sys.stderr)
        sys.exit(1)

    flt = _publish_filter_from_args(args)
    allow_not_applicable_threads = bool(
        getattr(args, "allow_not_applicable_threads", False)
    )

    write_mode = getattr(args, "write_mode", "merge")

    copied: list[tuple[str, Path, Path, int, str, str]] = []
    skipped_no_task: list[str] = []
    skipped_no_tsv: list[tuple[str, Path]] = []
    skipped_unchanged: list[str] = []
    skipped_gate: list[tuple[str, str]] = []

    for task_name, patch_path_str in sorted(index.items()):
        if task_name not in selected_tasks:
            continue
        patch_path = Path(patch_path_str)
        task_dir = _find_task_dir(workspace, task_name, framework=framework)
        if task_dir is None:
            skipped_no_task.append(task_name)
            continue
        tsv_src = task_dir / "package_verify.tsv"
        if not tsv_src.is_file():
            skipped_no_tsv.append((task_name, task_dir))
            continue

        task_flt = _filter_for_task(
            task_dir,
            flt,
            allow_not_applicable=allow_not_applicable_threads,
        )
        if (
            allow_not_applicable_threads
            and parse_threading_mode(task_dir / "task.yaml") == "not_applicable"
        ):
            print(
                "[publish-speedups] WARNING: "
                "--allow-not-applicable-threads set; publishing multi-thread "
                f"rows for {task_name} if present",
                file=sys.stderr,
            )

        try:
            new_content, n_new_rows, filter_summary = prepare_publish_content(
                tsv_src, task_flt,
            )
        except PublishFilterError as e:
            skipped_gate.append((task_name, str(e)))
            continue

        # Per-platform split: partition the selected rows by their system_os
        # and merge each platform into its own speedups.<plat>.tsv shard, so a
        # mac machine only ever writes speedups.mac.tsv etc. (disjoint merges).
        for plat, plat_content in _partition_by_platform(new_content).items():
            label = f"{task_name} [{plat}]"
            n_plat_rows = max(0, len(plat_content.splitlines()) - 1)
            dst = _dest_for_patch(framework, patch_path, plat)
            dst.parent.mkdir(parents=True, exist_ok=True)

            existing_disk_text = ""
            if dst.is_file():
                try:
                    existing_disk_text = dst.read_text(encoding="utf-8")
                except OSError:
                    existing_disk_text = ""
            existing_text, pruned_rows = prune_published_tsv_text(
                existing_disk_text, max_threads=task_flt.max_threads,
            )

            try:
                final_text, action_summary, total_rows = _combine_for_write(
                    existing_text, plat_content, n_plat_rows, write_mode,
                )
            except PublishFilterError as e:
                skipped_gate.append((label, str(e)))
                continue

            if (
                dst.is_file()
                and final_text.encode("utf-8") == existing_disk_text.encode("utf-8")
            ):
                skipped_unchanged.append(label)
                continue

            prune_summary = (
                f"pruned stale rows={pruned_rows} | " if pruned_rows else ""
            )
            combined_summary = f"{filter_summary} | {prune_summary}{action_summary}"

            if args.dry_run:
                copied.append((label, tsv_src, dst, total_rows, combined_summary, write_mode))
                continue

            dst.write_text(final_text, encoding="utf-8")
            copied.append((label, tsv_src, dst, total_rows, combined_summary, write_mode))

    if args.dry_run:
        print(f"zyme publish-speedups [DRY-RUN, mode={write_mode}] — "
              f"no files written.\n")

    if copied:
        print(f"Wrote ({len(copied)}):")
        for task, src, dst, n, summary, _mode in copied:
            rel_dst = _shorten(dst, framework)
            print(f"  {task:<32}  rows={n:<3}  {summary}")
            print(f"{'':34}  ->  {rel_dst}")
        print()

    if skipped_unchanged:
        print(f"Already up to date ({len(skipped_unchanged)}):  "
              + ", ".join(skipped_unchanged))
    if skipped_gate:
        print(f"Skipped — filter/gate ({len(skipped_gate)}):")
        for task, reason in skipped_gate:
            print(f"  {task:<32}  {reason}")
    if skipped_no_tsv:
        print(f"No package_verify.tsv ({len(skipped_no_tsv)}):")
        for task, where in skipped_no_tsv:
            print(f"  {task:<32}  (looked in {where})")
    if skipped_no_task:
        print(f"No matching task directory ({len(skipped_no_task)}):  "
              + ", ".join(skipped_no_task))

    if skipped_gate and not copied and not skipped_unchanged:
        sys.exit(1)


def _shorten(p: Path, framework: Path) -> str:
    try:
        return str(p.relative_to(framework))
    except ValueError:
        return str(p)


def _combine_for_write(
    existing_text: str, new_content: str, n_new_rows: int, write_mode: str,
) -> tuple[str, str, int]:
    """Combine existing bundled TSV with newly-selected rows per write mode.

    Returns (final_text, action_summary, total_rows_in_output).
    Raises PublishFilterError on header mismatch (caller falls back to gate).
    """
    if write_mode == "overwrite":
        n_existing = _count_data_rows(existing_text)
        return (
            new_content,
            f"overwrite (was {n_existing} → now {n_new_rows} rows)",
            n_new_rows,
        )

    if write_mode == "append":
        merged, n_appended = append_published_tsvs(existing_text, new_content)
        total = _count_data_rows(merged)
        return (
            merged,
            f"append +{n_appended} (total {total} rows)",
            total,
        )

    if write_mode == "merge":
        merged, stats = merge_published_tsvs(existing_text, new_content)
        total = _count_data_rows(merged)
        return (
            merged,
            f"merge add={stats.added} replace={stats.replaced} "
            f"keep={stats.kept} (total {total} rows)",
            total,
        )

    raise PublishFilterError(f"unknown --write-mode: {write_mode!r}")


def _count_data_rows(text: str) -> int:
    if not text.strip():
        return 0
    lines = text.splitlines()
    if not lines:
        return 0
    # First line is the header; data rows are everything below.
    return max(0, len(lines) - 1)
