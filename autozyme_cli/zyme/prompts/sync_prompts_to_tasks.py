#!/usr/bin/env python3
"""Sync framework prompts into every existing task's `prompts/` dir.

Recognized task layouts:

  * `<workspace>/test_*/`
  * `<workspace>/{test_,}{core_singlecell,general_bio,non_bio}/test_*/`

Each task has a `prompts/` subdir scaffolded by `zyme init`. When framework
prompts change (CLI flag updates, new sections, etc.), existing tasks keep
their old copies. This script propagates updates without depending on the
dev-only `autozyme_cli/prompts` symlink, which is not reliable on Windows
checkouts with `core.symlinks=false`.

Per task: detect flavor from `prompts/1_init.md` content first (old Bio tasks
may still have `3_expand_scaling.md`), then fall back to scaling prompt names.
Framework README files are not copied into tasks. For legacy tasks whose
`1_init.md` contains a task-specific `## Your inputs for this run` block, that
block is preserved while the rest of the prompt is refreshed.

Usage:
    python sync_prompts_to_tasks.py                # apply
    python sync_prompts_to_tasks.py --dry-run      # report only
    python sync_prompts_to_tasks.py --workspace /path/to/workspace_root
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Where the canonical prompts live: this script sits inside that directory.
FRAMEWORK_PROMPTS = Path(__file__).resolve().parent
REPO_DIR_NAMES = {"autozyme-framework", "autozyme", "autozyme_release"}


def _looks_like_autozyme_repo(path: Path) -> bool:
    return (
        path.name in REPO_DIR_NAMES
        or ((path / "autozyme_cli").is_dir() and (path / "autozyme_py").is_dir())
    )


def _find_workspace_root(start: Path) -> Path:
    """Walk up from `start` until we find the framework/release repo root,
    then return its parent (the workspace root). Robust to internal repackagings
    that change how deeply nested `prompts/` is — earlier this was hardcoded as
    `parent.parent.parent`, which broke when prompts moved from
    `autozyme_cli/prompts/` to `autozyme_cli/zyme/prompts/`.
    """
    cur = start
    while cur != cur.parent:
        if _looks_like_autozyme_repo(cur):
            return cur.parent
        cur = cur.parent
    # Installed wheels/sdists do not live under the development repo. Keep
    # module import safe and let callers pass --workspace explicitly when they
    # need to sync an external task tree.
    return Path.cwd()


# Default workspace root: parent of `autozyme-framework/`.
DEFAULT_WORKSPACE = _find_workspace_root(FRAMEWORK_PROMPTS)

# Workspace-level category dirs. The legacy short names (`core_singlecell`,
# `general_bio`, `non_bio`) are still recognized; current convention is the
# `test_`-prefixed form. Whichever exists at the workspace root wins per name.
TASK_ROOTS = (
    "test_core_singlecell", "test_general_bio", "test_non_bio", "test_spatial",
    "core_singlecell", "general_bio", "non_bio",
)

INIT_INPUTS_HEADING = "## Your inputs for this run"

# Situational prompts (transfer-init, thread-fairness audit) live in a
# `situational/` subdir, out of the task's top-level prompts/.
SITUATIONAL_SUBDIR = "situational"

# Prompts removed from the set entirely. A task's stale copy of any of these is
# deleted on sync, whether it sits at the top level or under `situational/`.
# (Distinct from `situational/` relocation, which moves the file rather than
# dropping it.)
REMOVED_PROMPTS = {"R_report.md", "6.3_cross_platform_port.md"}


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def detect_flavor(task_dir: Path) -> str | None:
    """Return 'Bio', 'OtherField', or None (no recognizable prompt set)."""
    init_text = _read_text_if_exists(task_dir / "prompts" / "1_init.md")
    init_lower = init_text.lower()
    if "init prompt (otherfield)" in init_lower or "expert performance engineer" in init_lower:
        return "OtherField"
    if "computational biologist" in init_lower or "single-cell" in init_lower:
        return "Bio"

    # Current prompt sets use distinct phase-3 names. Keep this fallback for
    # tasks that have lost 1_init.md or were scaffolded by an intermediate
    # framework version.
    prompts_dir = task_dir / "prompts"
    if (prompts_dir / "3_validate_scaling.md").exists():
        return "Bio"
    if (prompts_dir / "3_expand_scaling.md").exists():
        return "OtherField"
    return None


def maybe_task_dir(path: Path) -> tuple[Path, str] | None:
    """Return (path, flavor) when `path` looks like an autozyme task."""
    if not path.is_dir() or not path.name.startswith("test_"):
        return None
    if not (path / "prompts").is_dir():
        return None
    flavor = detect_flavor(path)
    if flavor is None:
        return None
    return path, flavor


def find_tasks(workspace: Path) -> list[tuple[Path, str]]:
    """Yield (task_dir, flavor) for every task with a recognizable prompts/ dir."""
    out = []
    seen = set()

    def add_task(candidate: Path) -> None:
        item = maybe_task_dir(candidate)
        if item is None:
            return
        task_dir, flavor = item
        key = task_dir.resolve()
        if key in seen:
            return
        seen.add(key)
        out.append((task_dir, flavor))

    # Windows migration layout: tasks live directly under the workspace root.
    if workspace.is_dir():
        for child in sorted(workspace.iterdir()):
            add_task(child)

    # Legacy/current categorized layouts.
    for root in TASK_ROOTS:
        root_path = workspace / root
        if not root_path.is_dir():
            continue
        for child in sorted(root_path.iterdir()):
            add_task(child)
    return out


def diff_files(a: Path, b: Path) -> bool:
    """True if a and b have different content (or one is missing)."""
    if not a.exists() or not b.exists():
        return True
    return a.read_bytes() != b.read_bytes()


def _extract_markdown_section(text: str, heading: str) -> str | None:
    """Return a top-level markdown section, including heading, if present."""
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.startswith("## ") and line.strip() != heading:
            end = i
            break
    return "".join(lines[start:end]).strip() + "\n"


def _merge_legacy_init_inputs(task_text: str, framework_text: str) -> str:
    """Preserve task-specific init inputs from old scaffolds when refreshing."""
    inputs = _extract_markdown_section(task_text, INIT_INPUTS_HEADING)
    if not inputs:
        return framework_text

    existing = _extract_markdown_section(framework_text, INIT_INPUTS_HEADING)
    if existing:
        return framework_text.replace(existing, inputs + "\n", 1)

    scope_heading = "\n## Scope\n"
    if scope_heading in framework_text:
        return framework_text.replace(scope_heading, "\n" + inputs + scope_heading, 1)

    return framework_text.rstrip() + "\n\n" + inputs


def copy_prompt(fw_file: Path, task_file: Path, dry_run: bool) -> bool:
    """Copy or merge one prompt. Returns True when the task file would change."""
    if fw_file.name == "1_init.md" and task_file.exists():
        merged = _merge_legacy_init_inputs(
            task_file.read_text(encoding="utf-8", errors="replace"),
            fw_file.read_text(encoding="utf-8"),
        )
        old = task_file.read_text(encoding="utf-8", errors="replace")
        if old == merged:
            return False
        if not dry_run:
            task_file.write_text(merged, encoding="utf-8")
        return True

    if not diff_files(task_file, fw_file):
        return False
    if not dry_run:
        shutil.copy2(fw_file, task_file)
    return True


def sync_task(task_dir: Path, flavor: str, dry_run: bool) -> dict:
    """Sync one task. Returns a per-task report dict.

    Top-level: updates files present in both framework and task when they
    differ; adds files present in framework but missing from the task (so a new
    framework prompt like `2.5_iterate_memory.md` reaches existing tasks).

    Situational subdir: mirrors `<field>/situational/*.md` into the task's
    `prompts/situational/`. A task's stale **top-level** copy of a now-
    situational prompt (e.g. an old flat `6.1_transfer_init.md`) is relocated:
    the canonical content is written under `situational/` and the top-level copy
    is deleted. A top-level copy of a fully-removed prompt (`REMOVED_PROMPTS`) is
    deleted outright.

    Other files present only in the task are reported as `skipped_no_framework`.
    """
    fw_dir = FRAMEWORK_PROMPTS / flavor
    sit_fw_dir = fw_dir / SITUATIONAL_SUBDIR
    task_prompts = task_dir / "prompts"
    task_sit_dir = task_prompts / SITUATIONAL_SUBDIR
    report = {
        "task": str(task_dir),
        "flavor": flavor,
        "updated": [],
        "added": [],
        "relocated": [],
        "removed": [],
        "identical": [],
        "skipped_no_framework": [],
        "errors": [],
    }
    fw_names = {p.name for p in fw_dir.glob("*.md") if p.name != "README.md"}
    sit_names = {p.name for p in sit_fw_dir.glob("*.md")} if sit_fw_dir.is_dir() else set()
    task_names = {p.name for p in task_prompts.glob("*.md")}
    # Captured before relocation writes into task_sit_dir, so a file relocated
    # from the top level isn't then re-reported as an identical situational file.
    task_sit_names = {p.name for p in task_sit_dir.glob("*.md")} if task_sit_dir.is_dir() else set()

    def _ensure_sit_dir():
        if not dry_run:
            task_sit_dir.mkdir(exist_ok=True)

    # Top-level prompts.
    for name in sorted(task_names):
        task_file = task_prompts / name
        if name in REMOVED_PROMPTS:
            if not dry_run:
                task_file.unlink()
            report["removed"].append(name)
            continue
        if name in sit_names:
            # Moved into situational/: write the canonical copy there, drop the
            # stale flat copy.
            _ensure_sit_dir()
            copy_prompt(sit_fw_dir / name, task_sit_dir / name, dry_run)
            if not dry_run:
                task_file.unlink()
            report["relocated"].append(name)
            continue
        fw_file = fw_dir / name
        if not fw_file.exists():
            report["skipped_no_framework"].append(name)
            continue
        if not copy_prompt(fw_file, task_file, dry_run):
            report["identical"].append(name)
            continue
        report["updated"].append(name)

    for name in sorted(fw_names - task_names):
        fw_file = fw_dir / name
        task_file = task_prompts / name
        copy_prompt(fw_file, task_file, dry_run)
        report["added"].append(name)

    # Situational subdir.
    if sit_names:
        for name in sorted(task_sit_names):
            if name in REMOVED_PROMPTS:
                if not dry_run:
                    (task_sit_dir / name).unlink()
                report["removed"].append(f"{SITUATIONAL_SUBDIR}/{name}")
                continue
            sit_fw_file = sit_fw_dir / name
            if not sit_fw_file.exists():
                report["skipped_no_framework"].append(f"{SITUATIONAL_SUBDIR}/{name}")
                continue
            if not copy_prompt(sit_fw_file, task_sit_dir / name, dry_run):
                report["identical"].append(f"{SITUATIONAL_SUBDIR}/{name}")
                continue
            report["updated"].append(f"{SITUATIONAL_SUBDIR}/{name}")
        missing = sorted(sit_names - task_sit_names - set(report["relocated"]))
        if missing:
            _ensure_sit_dir()
        for name in missing:
            copy_prompt(sit_fw_dir / name, task_sit_dir / name, dry_run)
            report["added"].append(f"{SITUATIONAL_SUBDIR}/{name}")

    return report


def display_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change; do not write.")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE,
                    help=f"Workspace root (default: {DEFAULT_WORKSPACE}).")
    ap.add_argument("--task", default=None,
                    help="Only sync the named task (e.g. test_slingshot).")
    args = ap.parse_args()

    if not FRAMEWORK_PROMPTS.is_dir():
        sys.exit(f"framework prompts not found at {FRAMEWORK_PROMPTS}")

    tasks = find_tasks(args.workspace)
    if args.task:
        task_arg = Path(args.task)
        if task_arg.exists():
            wanted = task_arg.resolve()
            tasks = [(t, f) for t, f in tasks if t.resolve() == wanted]
        else:
            tasks = [(t, f) for t, f in tasks if t.name == args.task]
        if not tasks:
            sys.exit(f"no task matching --task {args.task}")

    print(f"Workspace: {args.workspace}")
    print(f"Framework: {FRAMEWORK_PROMPTS}")
    print(f"Mode:      {'DRY RUN' if args.dry_run else 'APPLY'}")
    print(f"Tasks:     {len(tasks)} ({sum(1 for _, f in tasks if f == 'Bio')} Bio + "
          f"{sum(1 for _, f in tasks if f == 'OtherField')} OtherField)")
    print()

    totals = {"updated": 0, "added": 0, "relocated": 0, "removed": 0,
              "identical": 0, "skipped_no_framework": 0, "errors": 0}
    for task_dir, flavor in tasks:
        rep = sync_task(task_dir, flavor, args.dry_run)
        for key in totals:
            totals[key] += len(rep[key])
        rel = display_path(task_dir, args.workspace)
        if (rep["updated"] or rep["added"] or rep["relocated"] or rep["removed"]
                or rep["skipped_no_framework"] or rep["errors"]):
            parts = []
            if rep["updated"]:
                parts.append(f"updated={','.join(rep['updated'])}")
            if rep["added"]:
                parts.append(f"added={','.join(rep['added'])}")
            if rep["relocated"]:
                parts.append(f"relocated={','.join(rep['relocated'])}")
            if rep["removed"]:
                parts.append(f"removed={','.join(rep['removed'])}")
            if rep["skipped_no_framework"]:
                parts.append(f"skipped_no_framework={','.join(rep['skipped_no_framework'])}")
            if rep["errors"]:
                parts.append(f"ERRORS={rep['errors']}")
            print(f"  [{flavor:<10}] {rel}: {' | '.join(parts)}")
        else:
            print(f"  [{flavor:<10}] {rel}: clean")

    print()
    print("=" * 70)
    print(f"  Files updated (overwrite):   {totals['updated']}")
    print(f"  Files added (new):           {totals['added']}")
    print(f"  Files relocated to subdir:   {totals['relocated']}")
    print(f"  Files removed (dropped):     {totals['removed']}")
    print(f"  Files already identical:     {totals['identical']}")
    print(f"  Files not in framework:      {totals['skipped_no_framework']}")
    if totals["errors"]:
        print(f"  Errors:                      {totals['errors']}")
    print("=" * 70)
    if args.dry_run:
        print("DRY RUN — no files modified. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
