"""`zyme scan` — read-only workspace lifecycle scan; emits SCAN.md."""

import json
import sys
from datetime import datetime
from pathlib import Path

from zyme.scan import (
    PHASE_ORDER, find_workspace_root,
    find_framework_root, find_tasks, detect_phase,
)
from zyme.scan_datasets import (
    inspect_task_datasets, aggregate_external, aggregate_shared,
    collect_missing, collect_orphans, aggregate_totals,
    humanize_bytes, format_breakdown,
)
from zyme.scan_attest import (
    audit_packages, render_table as render_attest_table,
    render_markdown as render_attest_markdown,
    render_json_records as render_attest_json,
)
from zyme.scan_speedups import (
    audit_all_patches as audit_speedups_all,
    render_table as render_speedups_table,
    render_markdown as render_speedups_markdown,
    render_json_records as render_speedups_json,
    has_warn_or_fail as speedups_has_warn_or_fail,
)
from zyme.scan_portability import (
    scan_task, save_portability_scan, render_table as render_portability_table,
    PORTABILITY_SCAN_REL,
)


# ----------------------------------------------------------------------
# zyme scan — workspace-wide lifecycle dashboard
# ----------------------------------------------------------------------

_PHASE_HEADERS = ["scaf", "init", "iter", "scal", "pkg", "pv", "rpt"]

_REFLECT_HEADERS = ["I-rfl", "t-rfl", "s-rfl", "p-rfl"]

# Default scan roots: the three canonical task category dirs under the
# workspace. The bare-workspace fallback (whole tree) only kicks in if none
# of these dirs exist — useful for fresh setups before any category has
# tasks. Override by passing explicit `paths` to `zyme scan`.
_DEFAULT_TASK_CATEGORIES = (
    "test_core_singlecell",
    "test_general_bio",
    "test_non_bio",
    "test_spatial",
)


def _check(b: bool) -> str:
    return "[x]" if b else "[ ]"


def _format_latest_keep(latest_keep: dict | None) -> str:
    """Render the latest keep's speedup as e.g. '+12% medium' or '—'.

    Tier disambiguates which dataset the headline number was measured
    against — agents glancing at the scan should see both at once.
    """
    if not latest_keep:
        return "—"
    pct = latest_keep.get("speedup_pct")
    if pct is None:
        pct_str = "—"
    else:
        pct_str = f"{pct:+.0f}%"
    tier = latest_keep.get("dataset") or ""
    if tier:
        if len(tier) > 8:
            tier = tier[:7] + "…"
        return f"{pct_str} {tier}"
    return pct_str


def _format_active(active: dict | None) -> str:
    if not active:
        return "-"
    return str(active.get("status") or "-")




def _resolve_scan_roots(args) -> tuple[list[Path], Path | None, Path | None]:
    """Resolve scan roots, workspace root, framework root.

    Default (no `paths` given): scan only the canonical category dirs
    listed in `_DEFAULT_TASK_CATEGORIES` (test_core_singlecell,
    test_general_bio, test_non_bio). If none of them exist under the
    workspace, falls back to scanning the whole workspace tree — useful
    for fresh setups before categories have any tasks.

    Explicit `paths`: each becomes a root. Framework autodetect still
    runs (from cwd, then from the first valid root) to enable
    package/reflect resolution.
    """
    cwd = Path.cwd()
    workspace = find_workspace_root(cwd)
    if args.framework_root:
        framework = Path(args.framework_root).resolve()
    else:
        framework = (workspace / "autozyme-framework").resolve() if workspace else None

    if args.paths:
        roots = []
        for p in args.paths:
            pth = Path(p).resolve()
            if not pth.is_dir():
                print(f"zyme: scan path missing: {p}", file=sys.stderr)
                continue
            roots.append(pth)
        if framework is None and roots:
            framework = find_framework_root(roots[0])
        return roots, workspace, framework

    if workspace is None:
        print("zyme: no autozyme-framework/ found above cwd; "
              "pass paths explicitly or use --framework-root.", file=sys.stderr)
        return [], None, framework

    category_roots = [
        workspace / cat for cat in _DEFAULT_TASK_CATEGORIES
        if (workspace / cat).is_dir()
    ]
    if category_roots:
        return category_roots, workspace, framework

    return [workspace], workspace, framework




def _render_table(rows: list[dict], roots: list[Path], workspace: Path | None,
                  framework: Path | None, phase_only: bool, reflect_only: bool) -> None:
    """Render an ASCII dashboard. `rows` is a list of detect_phase results."""
    if workspace and framework:
        try:
            fw_disp = framework.relative_to(workspace).as_posix() + "/"
        except ValueError:
            fw_disp = str(framework)
        print(f"Workspace: {workspace}    Framework: {fw_disp}")
    elif framework:
        print(f"Framework: {framework}")
    print(f"Scanned {len(roots)} root(s); found {len(rows)} task(s).")
    if not rows:
        print("(no tasks found)")
        return
    print()

    # Group rows by immediate parent dir name. Show group headers only
    # when ≥2 distinct parents are present (otherwise headers are noise).
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = Path(row["task_dir"]).parent.name + "/"
        groups.setdefault(key, []).append(row)
    show_groups = len(groups) > 1
    if not show_groups:
        groups = {"": list(rows)}

    name_w = min(32, max(22, max(len(r["dir_name"]) for r in rows)))

    for group_key, group_rows in groups.items():
        if group_key:
            print(group_key)
        # Header
        head_parts = [f"  {'task':<{name_w}}"]
        if not reflect_only:
            head_parts.append(" ".join(f"{h:<4}" for h in _PHASE_HEADERS))
        if not reflect_only and not phase_only:
            head_parts.append("|")
        if not phase_only:
            head_parts.append(" ".join(f"{h:<5}" for h in _REFLECT_HEADERS))
        head_parts.append("|")
        head_parts.append(f"{'rounds':>6}  {'best_keep':<14}  {'active':<6}  phase")
        print(" ".join(head_parts))

        for row in group_rows:
            name = row["dir_name"]
            if len(name) > name_w:
                name = name[: name_w - 3] + "..."
            phases = row["phases"]
            reflect = row["reflect"]
            parts = [f"  {name:<{name_w}}"]
            if not reflect_only:
                phase_cells = [
                    _check(phases["scaffold"]["done"]),
                    _check(phases["init"]["done"]),
                    _check(phases["iterate"]["done"]),
                    _check(phases["scaling"]["done"]),
                    (_check(phases["package"]["done"]) if framework else "[?]"),
                    _check(row.get("package_verify_done", False)),
                    _check(row.get("report_done", False)),
                ]
                parts.append(" ".join(f"{c:<4}" for c in phase_cells))
            if not reflect_only and not phase_only:
                parts.append("|")
            if not phase_only:
                if framework:
                    refl_cells = [
                        _check(reflect["initialization"]),
                        _check(reflect["iteration"]),
                        _check(reflect["scaling"]),
                        _check(reflect["packaging"]),
                    ]
                else:
                    refl_cells = ["[?]"] * 4
                parts.append(" ".join(f"{c:<5}" for c in refl_cells))
            parts.append("|")
            phase_label = row["phase"] + ("*" if row.get("gaps") else "")
            keep_str = _format_latest_keep(row.get("latest_keep"))
            active_str = _format_active(row.get("active"))
            parts.append(
                f"{phases['iterate']['rounds']:>6}  {keep_str:<14}  "
                f"{active_str:<6}  {phase_label}"
            )
            print(" ".join(parts))
        print()

    # Totals
    totals = {p: 0 for p in PHASE_ORDER + ["unknown"]}
    gap_total = 0
    report_done_total = 0
    pv_done_total = 0
    for row in rows:
        totals[row["phase"]] += 1
        if row.get("gaps"):
            gap_total += 1
        if row.get("report_done"):
            report_done_total += 1
        if row.get("package_verify_done"):
            pv_done_total += 1
    parts = "  ".join(f"{p}={totals[p]}" for p in PHASE_ORDER if totals[p] > 0)
    if totals["unknown"]:
        parts = (parts + "  " if parts else "") + f"unknown={totals['unknown']}"
    print(f"Totals: {parts or '(none)'}")
    print(f"Package verify: {pv_done_total} / {len(rows)} task(s) have package_verify.tsv")
    print(f"Reports: {report_done_total} / {len(rows)} task(s) have report.html")
    if gap_total:
        print(f"Note: {gap_total} task(s) marked * have at least one earlier "
              f"phase incomplete (e.g. packaged without scaling).")




def _render_json(rows: list[dict]) -> None:
    """Emit one JSON object per task (line-delimited / jq-friendly)."""
    for row in rows:
        print(json.dumps(row, default=str))




def _markdown_summary(rows: list[dict], workspace: Path | None,
                      framework: Path | None) -> str:
    """Render a markdown report of the full (unfiltered) task list."""
    out: list[str] = []
    out.append("# autozyme task lifecycle scan")
    out.append("")
    out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if workspace:
        out.append(f"Workspace: `{workspace}`")
    if framework:
        try:
            fw_disp = framework.relative_to(workspace).as_posix() + "/" if workspace else str(framework)
        except ValueError:
            fw_disp = str(framework)
        out.append(f"Framework: `{fw_disp}`")
    out.append(f"Tasks found: {len(rows)}")
    out.append("")
    out.append("Phase columns: scaffold / init.md / iterate.md / scaling.md / package.md. `pv` = package_verify.tsv present (fresh-subprocess speedups from `verify_patch()`). `rpt` = report.html generated.")
    out.append("Reflect columns: per-phase reflection feedback (initialization / iteration / scaling / packaging).")
    out.append("Active column: `run` / `wait` / `pend` / `refl` from live dispatch; `recent` from recent task file activity; `stale` from dead dispatch state.")
    out.append("`phase*` = highest-completed phase has at least one earlier phase incomplete.")
    out.append("")

    if not rows:
        out.append("_No tasks found._")
        return "\n".join(out) + "\n"

    # Group by parent dir
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = Path(row["task_dir"]).parent.name + "/"
        groups.setdefault(key, []).append(row)

    header = (
        "| task | scaf | init | iter | scal | pkg | pv | rpt | I-rfl | t-rfl | s-rfl | p-rfl | rounds | best_keep | active | phase |"
    )
    sep = "|------|------|------|------|------|-----|----|-----|-------|-------|-------|-------|--------|-----------|--------|-------|"

    for group_key, group_rows in groups.items():
        out.append(f"## {group_key}")
        out.append("")
        out.append(header)
        out.append(sep)
        for row in group_rows:
            ph = row["phases"]
            rf = row["reflect"]
            phase_label = row["phase"] + ("\\*" if row.get("gaps") else "")
            cells = [
                row["dir_name"],
                _check(ph["scaffold"]["done"]),
                _check(ph["init"]["done"]),
                _check(ph["iterate"]["done"]),
                _check(ph["scaling"]["done"]),
                _check(ph["package"]["done"]),
                _check(row.get("package_verify_done", False)),
                _check(row.get("report_done", False)),
                _check(rf["initialization"]),
                _check(rf["iteration"]),
                _check(rf["scaling"]),
                _check(rf["packaging"]),
                str(ph["iterate"]["rounds"]),
                _format_latest_keep(row.get("latest_keep")),
                _format_active(row.get("active")),
                phase_label,
            ]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    # Totals
    totals = {p: 0 for p in PHASE_ORDER + ["unknown"]}
    gap_total = 0
    report_done_total = 0
    pv_done_total = 0
    for row in rows:
        totals[row["phase"]] += 1
        if row.get("gaps"):
            gap_total += 1
        if row.get("report_done"):
            report_done_total += 1
        if row.get("package_verify_done"):
            pv_done_total += 1
    parts = [f"{p}={totals[p]}" for p in PHASE_ORDER if totals[p] > 0]
    parts.append(f"package_verify={pv_done_total}/{len(rows)}")
    parts.append(f"reports={report_done_total}/{len(rows)}")
    if totals["unknown"]:
        parts.append(f"unknown={totals['unknown']}")
    out.append("## Totals")
    out.append("")
    out.append("  ".join(parts) if parts else "_(none)_")
    if gap_total:
        out.append("")
        out.append(f"> **Note:** {gap_total} task(s) marked `*` have at "
                   f"least one earlier phase incomplete (e.g. packaged without scaling).")
    out.append("")
    return "\n".join(out)




def _resolve_export_path(args, framework: Path | None) -> Path | None:
    """Resolve where SCAN.md should be written, or None to skip."""
    if args.no_export or args.json:
        return None
    if args.export_path:
        return Path(args.export_path).resolve()
    if framework is None:
        return None
    return framework / "SCAN.md"




# ----------------------------------------------------------------------
# Dataset-focused mode (`zyme scan --dataset`)
# ----------------------------------------------------------------------

def _render_dataset_table(rows: list[dict], workspace: Path | None,
                          framework: Path | None) -> None:
    """ASCII per-task dataset table + external-path roster + missing list."""
    if workspace and framework:
        try:
            fw_disp = framework.relative_to(workspace).as_posix() + "/"
        except ValueError:
            fw_disp = str(framework)
        print(f"Workspace: {workspace}    Framework: {fw_disp}")
    elif framework:
        print(f"Framework: {framework}")

    totals = aggregate_totals(rows)
    print(f"Scanned {len(rows)} task(s); {totals['n_entries']} dataset entries.")
    if not rows:
        print("(no tasks found)")
        return
    print()

    # Group by parent dir (same convention as lifecycle scan).
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["category"] + "/", []).append(row)
    show_groups = len(groups) > 1
    if not show_groups:
        groups = {"": list(rows)}

    name_w = min(32, max(22, max(len(r["dir_name"]) for r in rows)))

    for group_key, group_rows in groups.items():
        if group_key:
            print(group_key)
        head = (
            f"  {'task':<{name_w}}  "
            f"{'tiers':>5}  {'local':>14}  {'shared':>14}  "
            f"{'external':>14}  {'miss':>4}  {'task_dir':>8}  breakdown"
        )
        print(head)
        for row in group_rows:
            name = row["dir_name"]
            if len(name) > name_w:
                name = name[: name_w - 3] + "..."
            t = row["totals"]
            local = (f"{t['local_count']} ({humanize_bytes(t['local_bytes'])})"
                     if t["local_count"] else "0")
            shd = (f"{t['shared_count']} ({humanize_bytes(t['shared_bytes'])})"
                   if t["shared_count"] else "0")
            ext = (f"{t['external_count']} ({humanize_bytes(t['external_bytes'])})"
                   if t["external_count"] else "0")
            miss = str(t["missing_count"]) if t["missing_count"] else "0"
            task_dir_size = humanize_bytes(t["task_dir_bytes"])
            breakdown = format_breakdown(t["subdir"])
            # Append orphan tail so storage waste is visible on the main
            # row, not buried in the orphans section.
            if t["orphan_count"]:
                breakdown += (f"  ! orphan:{t['orphan_count']}/"
                              f"{humanize_bytes(t['orphan_bytes'])}")
            print(f"  {name:<{name_w}}  "
                  f"{t['n_entries']:>5}  {local:>14}  {shd:>14}  "
                  f"{ext:>14}  {miss:>4}  {task_dir_size:>8}  {breakdown}")
        print()

    # Shared roster — /datasets/single_cell/ entries referenced by tasks.
    # Recoverable from HF, but worth knowing the cross-task map.
    shareds = aggregate_shared(rows)
    if shareds:
        print(f"Shared datasets (in /datasets/single_cell/, HF-recoverable) — "
              f"{len(shareds)} unique path(s):")
        for item in shareds:
            refs = ", ".join(
                f"{r['task']}:{r['tier']}" for r in item["references"]
            )
            print(f"  {item['path']}  ({humanize_bytes(item['size_bytes'])})")
            print(f"    <- {refs}")
        print()

    # External path roster — what would orphan if the user moved tasks
    # without these files, sorted big-first so storage hogs are obvious.
    externals = aggregate_external(rows)
    if externals:
        print(f"External datasets (outside workspace, bring manually) — "
              f"{len(externals)} unique path(s):")
        for item in externals:
            refs = ", ".join(
                f"{r['task']}:{r['tier']}" for r in item["references"]
            )
            print(f"  {item['path']}  ({humanize_bytes(item['size_bytes'])})")
            print(f"    <- {refs}")
        print()

    # data/ orphans — top-level entries in a task's data/ that task.yaml
    # doesn't reference. Skips `_raw/` (v2 staging convention) and
    # symlinks (no real disk weight). Each top-level dir is summarised
    # as one entry with file count, so a 19k-leaf extracted archive
    # doesn't flood the report.
    orphans = collect_orphans(rows)
    if orphans:
        cap = 20
        print(f"data/ orphans — entries in data/ not referenced in task.yaml "
              f"({len(orphans)} total):")
        for o in orphans[:cap]:
            tail = ""
            if o.get("kind") == "venv":
                tail = "  [Python venv]"
            elif o.get("kind") == "dir":
                tail = f"  [{o.get('n_files', 0)} files]"
            print(f"  {o['task']}  {o['path']}  "
                  f"({humanize_bytes(o['size_bytes'])}){tail}")
        if len(orphans) > cap:
            print(f"  ... and {len(orphans) - cap} more (see DATASETS.md)")
        print()

    # Missing — surface loudly; this is exactly the case the user is
    # trying to avoid post-move.
    missing = collect_missing(rows)
    if missing:
        print(f"Missing ({len(missing)}):")
        for m in missing:
            print(f"  {m['task']}:{m['tier']}  ->  {m['path']}")
        print()

    sub = totals["subdir"]
    parts = [
        f"tasks={totals['n_tasks']}",
        f"entries={totals['n_entries']}",
        f"task_dir_total={humanize_bytes(totals['task_dir_bytes'])}",
        f"local={totals['local_count']}/{humanize_bytes(totals['local_bytes'])}",
        # shared + external both use unique-path bytes — a shared file
        # referenced by 3 tasks counts as 1 path / 1× its size in the
        # on-disk footprint number (not 3× as the raw ref count suggests).
        f"shared={totals['shared_count']} refs / "
        f"{totals['shared_unique_paths']} paths / "
        f"{humanize_bytes(totals['shared_unique_bytes'])}",
        f"external={totals['external_count']} refs / "
        f"{totals['external_unique_paths']} paths / "
        f"{humanize_bytes(totals['external_unique_bytes'])}",
        f"missing={totals['missing_count']}",
        f"orphans={totals['orphan_count']}/{humanize_bytes(totals['orphan_bytes'])}",
    ]
    print("Totals: " + "  ".join(parts))
    print(f"Storage by subdir: data={humanize_bytes(sub['data'])}  "
          f"reference_outputs={humanize_bytes(sub['reference_outputs'])}  "
          f"upstream_repo={humanize_bytes(sub['upstream_repo'])}  "
          f"other={humanize_bytes(sub['other'])}")


def _dataset_markdown(rows: list[dict], workspace: Path | None,
                      framework: Path | None) -> str:
    out: list[str] = []
    out.append("# autozyme task datasets")
    out.append("")
    out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if workspace:
        out.append(f"Workspace: `{workspace}`")
    if framework:
        try:
            fw_disp = framework.relative_to(workspace).as_posix() + "/" if workspace else str(framework)
        except ValueError:
            fw_disp = str(framework)
        out.append(f"Framework: `{fw_disp}`")
    totals = aggregate_totals(rows)
    out.append(f"Tasks: {totals['n_tasks']}  |  Dataset entries: {totals['n_entries']}")
    out.append("")
    out.append("Scope legend: **local** = travels with the task (inside `<task>/` "
               "or `/datasets/per_task/<task>/`). **shared** = in "
               "`/datasets/single_cell/`, hosted on HF, recoverable via "
               "`hf download`. **external** = outside the workspace "
               "entirely (user-home cache etc.), bring along manually. "
               "**missing** = declared in `task.yaml` but not on disk.")
    out.append("")

    if not rows:
        out.append("_No tasks found._")
        return "\n".join(out) + "\n"

    # Group by category
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["category"] + "/", []).append(row)

    out.append("## Per-task summary")
    out.append("")
    out.append("`breakdown` column: `d`=data/, `ro`=reference_outputs/, "
               "`u`=upstream_repo/, `o`=everything else in the task dir.")
    out.append("")
    out.append("| task | tiers | local | shared | external | missing | task_dir | breakdown | orphans |")
    out.append("|------|-------|-------|--------|----------|---------|----------|-----------|---------|")
    for group_key, group_rows in groups.items():
        for row in group_rows:
            t = row["totals"]
            local = (f"{t['local_count']} ({humanize_bytes(t['local_bytes'])})"
                     if t["local_count"] else "0")
            shd = (f"{t['shared_count']} ({humanize_bytes(t['shared_bytes'])})"
                   if t["shared_count"] else "0")
            ext = (f"{t['external_count']} ({humanize_bytes(t['external_bytes'])})"
                   if t["external_count"] else "0")
            miss = str(t["missing_count"])
            task_dir_size = humanize_bytes(t["task_dir_bytes"])
            breakdown = format_breakdown(t["subdir"])
            orphan_cell = (f"{t['orphan_count']} ({humanize_bytes(t['orphan_bytes'])})"
                           if t["orphan_count"] else "0")
            out.append(f"| `{group_key}{row['dir_name']}` | {t['n_entries']} | "
                       f"{local} | {shd} | {ext} | {miss} | {task_dir_size} | "
                       f"{breakdown} | {orphan_cell} |")
    out.append("")

    shareds = aggregate_shared(rows)
    if shareds:
        out.append(f"## Shared datasets ({len(shareds)} unique paths)")
        out.append("")
        out.append("Entries in `/datasets/single_cell/`. Hosted on HF "
                   "(`elliotxie/autozyme-datasets`), so a fresh checkout "
                   "can recover them via `hf download`. Sorted by size.")
        out.append("")
        for item in shareds:
            out.append(f"- `{item['path']}` — **{humanize_bytes(item['size_bytes'])}**")
            for r in item["references"]:
                out.append(f"  - `{r['category']}/{r['task']}` (tier `{r['tier']}`, name `{r['name']}`)")
        out.append("")

    externals = aggregate_external(rows)
    if externals:
        out.append(f"## External datasets ({len(externals)} unique paths)")
        out.append("")
        out.append("Paths outside the workspace entirely (user-home caches, "
                   "absolute paths on the local machine). Moving a task "
                   "without copying these will break `zyme run`. Sorted by size.")
        out.append("")
        for item in externals:
            out.append(f"- `{item['path']}` — **{humanize_bytes(item['size_bytes'])}**")
            for r in item["references"]:
                out.append(f"  - `{r['category']}/{r['task']}` (tier `{r['tier']}`, name `{r['name']}`)")
        out.append("")

    orphans = collect_orphans(rows)
    if orphans:
        out.append(f"## `data/` orphans ({len(orphans)})")
        out.append("")
        out.append("Top-level entries in `<task>/data/` not referenced by any "
                   "entry in `task.yaml`'s `datasets:` block. Often safe to "
                   "delete — review before purging. Skips `_raw/` (v2 layout "
                   "staging) and symlinks (no real disk weight). Each "
                   "top-level dir is summed as one row (file-count in "
                   "brackets). Sorted by size.")
        out.append("")
        for o in orphans:
            tail = ""
            if o.get("kind") == "venv":
                tail = " _(Python venv)_"
            elif o.get("kind") == "dir":
                tail = f" _({o.get('n_files', 0)} files)_"
            out.append(f"- `{o['category']}/{o['task']}`  `{o['path']}`  "
                       f"**{humanize_bytes(o['size_bytes'])}**{tail}")
        out.append("")

    missing = collect_missing(rows)
    if missing:
        out.append(f"## Missing datasets ({len(missing)})")
        out.append("")
        out.append("Declared in `task.yaml` but not present on disk. Either restore the "
                   "file or update the task's `datasets:` block.")
        out.append("")
        for m in missing:
            out.append(f"- `{m['category']}/{m['task']}` tier `{m['tier']}` "
                       f"(name `{m['name']}`) -> `{m['path']}`")
        out.append("")

    sub = totals["subdir"]
    out.append("## Totals")
    out.append("")
    out.append(f"- tasks: {totals['n_tasks']}")
    out.append(f"- dataset entries: {totals['n_entries']}")
    out.append(f"- task dir total: {humanize_bytes(totals['task_dir_bytes'])}")
    out.append(f"- local: {totals['local_count']} entries, "
               f"{humanize_bytes(totals['local_bytes'])}")
    out.append(f"- shared: {totals['shared_count']} references across "
               f"{totals['shared_unique_paths']} unique paths, "
               f"{humanize_bytes(totals['shared_unique_bytes'])}")
    out.append(f"- external: {totals['external_count']} references across "
               f"{totals['external_unique_paths']} unique paths, "
               f"{humanize_bytes(totals['external_unique_bytes'])}")
    out.append(f"- missing: {totals['missing_count']}")
    out.append(f"- orphans: {totals['orphan_count']} files, "
               f"{humanize_bytes(totals['orphan_bytes'])}")
    out.append("")
    out.append("### Storage by subdir bucket")
    out.append("")
    out.append(f"- `data/`: {humanize_bytes(sub['data'])}")
    out.append(f"- `reference_outputs*/`: {humanize_bytes(sub['reference_outputs'])}")
    out.append(f"- `upstream_repo/`: {humanize_bytes(sub['upstream_repo'])}")
    out.append(f"- everything else: {humanize_bytes(sub['other'])}")
    out.append("")
    return "\n".join(out)


def _resolve_dataset_export_path(args, framework: Path | None) -> Path | None:
    if args.no_export or args.json:
        return None
    if args.export_path:
        return Path(args.export_path).resolve()
    if framework is None:
        return None
    return framework / "DATASETS.md"


def _run_dataset_scan(args, roots, workspace, framework, task_dirs) -> None:
    rows = [inspect_task_datasets(td) for td in task_dirs]

    if args.json:
        for row in rows:
            print(json.dumps(row, default=str))
    else:
        _render_dataset_table(rows, workspace=workspace, framework=framework)

    export_path = _resolve_dataset_export_path(args, framework)
    if export_path is not None and rows:
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(_dataset_markdown(rows, workspace, framework))
            print(f"\nExported dataset summary to: {export_path}")
        except OSError as e:
            print(f"zyme: failed to write {export_path}: {e}", file=sys.stderr)




def _resolve_attest_framework(args) -> Path | None:
    """Resolve the framework root for attest mode without requiring a
    workspace. Honors `--framework-root` if given; otherwise walks up
    from cwd, and finally tries cwd itself in case the user is already
    inside autozyme-framework/."""
    if getattr(args, "framework_root", None):
        return Path(args.framework_root).resolve()
    cwd = Path.cwd()
    fr = find_framework_root(cwd)
    if fr is not None:
        return fr
    # Last resort: if cwd looks like the framework, use it.
    if (cwd / "autozyme_r").is_dir() and (cwd / "autozyme_py").is_dir():
        return cwd.resolve()
    return None


def _explicit_task_dirs(paths: list[str] | None) -> list[Path]:
    """Task dirs passed directly on the CLI (each contains task.yaml)."""
    out: list[Path] = []
    if not paths:
        return out
    for raw in paths:
        p = Path(raw).resolve()
        if (p / "task.yaml").is_file():
            out.append(p)
    return out


def _merge_task_dirs(walked: list[Path], explicit: list[Path]) -> list[Path]:
    seen: set[str] = set()
    merged: list[Path] = []
    for p in explicit + walked:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            merged.append(p)
    return merged


def _run_portability_scan(args, task_dirs, framework) -> None:
    if not task_dirs:
        print("(no tasks found)")
        return

    results = [scan_task(td, framework_root=framework) for td in task_dirs]

    if getattr(args, "needs_3_5", False):
        results = [r for r in results if r.run_3_5]

    for res in results:
        save_portability_scan(Path(res.task_dir), res)

    if args.json:
        for res in results:
            print(json.dumps(res.to_json_dict(), default=str))
    else:
        print(render_portability_table(results))
        if results:
            print(f"\nResults saved to {PORTABILITY_SCAN_REL} under each task dir.")


def _run_attest_scan(args) -> None:
    framework = _resolve_attest_framework(args)
    if framework is None:
        print("zyme: could not locate autozyme-framework/ (no autozyme_r + "
              "autozyme_py sibling found). Use --framework-root to point at "
              "it explicitly.", file=sys.stderr)
        sys.exit(1)

    patches = audit_packages(framework)
    if not patches:
        print(f"zyme: no patches found under {framework}/autozyme_r or "
              f"{framework}/autozyme_py.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        for rec in render_attest_json(patches):
            print(json.dumps(rec, default=str))
    else:
        print(render_attest_table(patches, framework))

    if args.no_export or args.json:
        return
    if args.export_path:
        export_path = Path(args.export_path).resolve()
    else:
        export_path = framework / "ATTEST.md"
    try:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(render_attest_markdown(patches, framework))
        print(f"\nExported attest coverage to: {export_path}")
    except OSError as e:
        print(f"zyme: failed to write {export_path}: {e}", file=sys.stderr)


def _coverage_platform_filter(args) -> str | None:
    if getattr(args, "mac_only", False):
        return "mac"
    if getattr(args, "win_only", False):
        return "win"
    return None


def _run_speedups_scan(args) -> None:
    """`zyme scan --coverage` runner: lint every patch's finalized speedup TSV.

    Pure package-side audit; no workspace walk. Writes COVERAGE.md alongside
    the framework root unless --no-export / --json is set. Exit code is 0 by
    default; --strict makes any WARN/FAIL exit 1 for release-gate use.
    """
    framework = _resolve_attest_framework(args)
    if framework is None:
        print("zyme: could not locate autozyme-framework/ (no autozyme_r + "
              "autozyme_py sibling found). Use --framework-root to point at "
              "it explicitly.", file=sys.stderr)
        sys.exit(1)

    platform_filter = _coverage_platform_filter(args)
    audits = audit_speedups_all(
        framework,
        patched_min_reps=getattr(args, "patched_min_reps", 2),
        baseline_min_reps=getattr(args, "baseline_min_reps", 2),
        rep_variance_pct=getattr(args, "rep_variance_pct", 10.0),
        rep_variance_abs_sec=getattr(args, "rep_variance_abs_sec", 2.0),
        mem_variance_pct=getattr(args, "mem_variance_pct", 20.0),
        mem_variance_abs_mb=getattr(args, "mem_variance_abs_mb", 1024.0),
        baseline_era_pct=getattr(args, "baseline_era_pct", 20.0),
        baseline_era_abs_mb=getattr(args, "baseline_era_abs_mb", 1024.0),
        platform_filter=platform_filter,
        skip_experimental=getattr(args, "skip_experimental", False),
    )
    if not audits:
        print(f"zyme: no patches with shipped speedups_finalized.tsv found "
              f"under {framework}.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        for rec in render_speedups_json(audits):
            print(json.dumps(rec, default=str))
    else:
        print(render_speedups_table(
            audits,
            detail=getattr(args, "detail", False),
            platform_filter=platform_filter,
        ))

    if not (args.no_export or args.json):
        if args.export_path:
            export_path = Path(args.export_path).resolve()
        elif platform_filter == "mac":
            export_path = framework / "COVERAGE.mac.md"
        elif platform_filter == "win":
            export_path = framework / "COVERAGE.win.md"
        else:
            export_path = framework / "COVERAGE.md"
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(render_speedups_markdown(
                audits, platform_filter=platform_filter,
            ))
            print(f"\nExported coverage report to: {export_path}")
        except OSError as e:
            print(f"zyme: failed to write {export_path}: {e}", file=sys.stderr)

    # --strict promotes any WARN or FAIL to a non-zero exit. INFO (e.g. Mac
    # OOM at large) stays exit 0 even under --strict — those are documented
    # hardware limits, not coverage failures.
    if getattr(args, "strict", False) and speedups_has_warn_or_fail(audits):
        sys.exit(1)


def cmd_scan(args):
    """Workspace-wide task lifecycle dashboard.

    Walks the given paths (or auto-detected workspace siblings), finds
    every dir containing task.yaml, and reports each task's furthest
    completed lifecycle phase plus per-phase reflection status.
    Read-only.
    """
    # Coverage mode is purely package-side; doesn't need workspace/task walk.
    if getattr(args, "coverage", False):
        _run_speedups_scan(args)
        return
    # Attest mode is purely package-side; doesn't need workspace/task walk.
    if getattr(args, "attest", False):
        _run_attest_scan(args)
        return

    roots, workspace, framework = _resolve_scan_roots(args)
    if not roots:
        sys.exit(1 if not args.paths else 0)

    if framework is None:
        print("zyme: warning — autozyme-framework/ not found; package and "
              "reflect columns will show [?].", file=sys.stderr)

    task_dirs = _merge_task_dirs(
        find_tasks(roots, max_depth=args.max_depth, framework_root=framework),
        _explicit_task_dirs(args.paths),
    )

    # Portability hazard scan — writes .zyme/portability_scan.json per task.
    if getattr(args, "portability", False):
        _run_portability_scan(args, task_dirs, framework)
        return

    # Dataset-focused mode is a complete switch — skip lifecycle detection
    # entirely. Cheaper, and avoids the SCAN.md/DATASETS.md split confusion.
    if getattr(args, "dataset", False):
        _run_dataset_scan(args, roots, workspace, framework, task_dirs)
        return

    active_minutes = getattr(args, "active_minutes", None)
    all_rows = [
        detect_phase(td, framework, active_recent_minutes=active_minutes)
        for td in task_dirs
    ]
    if args.phase_filter:
        display_rows = [r for r in all_rows if r["phase"] == args.phase_filter]
    else:
        display_rows = all_rows

    if args.json:
        _render_json(display_rows)
    else:
        _render_table(
            display_rows, roots=roots, workspace=workspace, framework=framework,
            phase_only=args.phase_only, reflect_only=args.reflect_only,
        )

    # Auto-export markdown summary (unfiltered, full state).
    export_path = _resolve_export_path(args, framework)
    if export_path is not None and all_rows:
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(_markdown_summary(all_rows, workspace, framework))
            print(f"\nExported summary to: {export_path}")
        except OSError as e:
            print(f"zyme: failed to write {export_path}: {e}", file=sys.stderr)
