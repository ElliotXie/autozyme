"""`zyme datasets` — manage the workspace `/datasets/` layer.

Subcommands:
  migrate     Reorganize `/datasets/` into `shared/` + `per_task/<task>/` and
              turn each `<task>/data/` into a symlink to its per_task slot.
              Dry-run by default; pass `--execute` to actually move files.

Rationale: the workspace started with `<task>/data/` holding the per-task
tier files and `/datasets/` holding cross-task shared checkpoints. The
v2 layout makes `/datasets/` the *single* on-disk home for all data:

    /datasets/
    ├── shared/                ← cross-task (heart_adult, ifnb, ...)
    │   ├── heart_adult.h5ad
    │   └── ...
    └── per_task/
        └── <task>/            ← what used to live in <task>/data/
            └── tier_*.rds

Each task's old `data/` becomes a symlink → `/datasets/per_task/<task>/`,
so task.yaml's relative `./data/...` paths keep working untouched. The
whole `/datasets/` tree is what gets pushed to HF (mega-repo).

Compatibility symlinks: when we move `/datasets/X.rds` →
`/datasets/single_cell/X.rds`, we leave a relative symlink at the old
location so existing `_raw/X.rds → /datasets/X.rds` references in
task dirs keep working through one extra hop.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from zyme.scan import find_framework_root, find_tasks, find_workspace_root


# Top-level entries in /datasets/ that should stay at the root (not moved
# into single_cell/). README.md is documentation; setup/ is preprocessing
# code tracked alongside the data. Add to this list cautiously — anything
# not listed will be moved into single_cell/.
_DATASETS_ROOT_KEEP = {"README.md", "setup", "single_cell", "spatial", "per_task", ".DS_Store"}


# ----------------------------------------------------------------------
# Plan structures
# ----------------------------------------------------------------------

class MovePlan:
    """One move + compat-symlink operation.

    `src` is the current absolute path. `dst` is where it will live after
    the move. `compat_link` is an absolute path where a relative symlink
    pointing to `dst` should be placed (None if no compat link needed).
    `size_bytes` is for the plan summary.
    """

    __slots__ = ("src", "dst", "compat_link", "size_bytes", "kind", "note")

    def __init__(self, src: Path, dst: Path, *, compat_link: Path | None,
                 size_bytes: int, kind: str, note: str = ""):
        self.src = src
        self.dst = dst
        self.compat_link = compat_link
        self.size_bytes = size_bytes
        self.kind = kind  # "shared_file" | "shared_dir" | "task_data"
        self.note = note


def _path_size_bytes(path: Path) -> int:
    """File or recursive dir size, in bytes. 0 on error or symlink."""
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for dp, _, fns in os.walk(path, followlinks=False):
                for fn in fns:
                    fp = Path(dp) / fn
                    try:
                        if not fp.is_symlink():
                            total += fp.stat().st_size
                    except OSError:
                        pass
            return total
    except OSError:
        pass
    return 0


def _humanize(n: int) -> str:
    if n <= 0:
        return "0"
    for suf, scale in (("T", 1024 ** 4), ("G", 1024 ** 3),
                       ("M", 1024 ** 2), ("K", 1024)):
        if n >= scale:
            v = n / scale
            return f"{v:.1f}{suf}" if v < 10 else f"{v:.0f}{suf}"
    return f"{n}B"


# ----------------------------------------------------------------------
# Phase 1: /datasets/ top-level → /datasets/single_cell/
# ----------------------------------------------------------------------

def plan_shared_migration(datasets_root: Path) -> list[MovePlan]:
    """Plan moves for the /datasets/ top-level reorganization.

    Each top-level entry (file or dir) not in _DATASETS_ROOT_KEEP becomes
    a move into shared/<same-name> with a compat symlink left behind.
    Compat symlinks are RELATIVE (`shared/X` from inside /datasets/) so
    they survive renames of the workspace root.
    """
    plans: list[MovePlan] = []
    if not datasets_root.is_dir():
        return plans
    shared = datasets_root / "shared"
    for entry in sorted(datasets_root.iterdir()):
        if entry.name in _DATASETS_ROOT_KEEP:
            continue
        # Already a compat symlink (e.g. partial earlier run) — leave it.
        if entry.is_symlink():
            continue
        dst = shared / entry.name
        size = _path_size_bytes(entry)
        kind = "shared_dir" if entry.is_dir() else "shared_file"
        plans.append(MovePlan(
            src=entry, dst=dst,
            compat_link=entry,  # same path, recreated as symlink after move
            size_bytes=size, kind=kind,
        ))
    return plans


# ----------------------------------------------------------------------
# Phase 2: <task>/data/ → /datasets/per_task/<task>/
# ----------------------------------------------------------------------

def plan_task_migration(task_dirs: list[Path], datasets_root: Path) -> list[MovePlan]:
    """Plan moves for each task's real (non-symlink) data/ directory."""
    plans: list[MovePlan] = []
    per_task = datasets_root / "per_task"
    for td in task_dirs:
        data = td / "data"
        if not data.exists():
            continue
        if data.is_symlink():
            # Already migrated, or symlinked to a sibling task — leave alone.
            continue
        if not data.is_dir():
            continue
        dst = per_task / td.name
        size = _path_size_bytes(data)
        note = ""
        if dst.exists():
            note = f"DESTINATION EXISTS at {dst} — task migration will skip"
        plans.append(MovePlan(
            src=data, dst=dst,
            compat_link=data,  # data/ recreated as symlink after move
            size_bytes=size, kind="task_data", note=note,
        ))
    return plans


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------

def _execute_move(plan: MovePlan) -> tuple[bool, str]:
    """Perform one MovePlan. Returns (success, message)."""
    if plan.note and "DESTINATION EXISTS" in plan.note:
        return False, f"skip (dest exists): {plan.dst}"
    if not plan.src.exists() and not plan.src.is_symlink():
        return False, f"skip (src missing): {plan.src}"
    plan.dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # shutil.move uses os.rename when on same fs (atomic), else copytree+rmtree.
        shutil.move(str(plan.src), str(plan.dst))
    except OSError as e:
        return False, f"move FAILED ({plan.src} -> {plan.dst}): {e}"
    if plan.compat_link is not None:
        # Compute relative path: from compat_link's parent dir → dst.
        try:
            rel = os.path.relpath(plan.dst, plan.compat_link.parent)
        except ValueError:
            rel = str(plan.dst)
        # If something at compat_link still exists (rare race), back off.
        if plan.compat_link.exists() or plan.compat_link.is_symlink():
            return False, (f"move OK but compat symlink target exists: "
                           f"{plan.compat_link}")
        try:
            plan.compat_link.symlink_to(rel,
                                         target_is_directory=plan.dst.is_dir())
        except OSError as e:
            return False, f"move OK but symlink failed ({plan.compat_link}): {e}"
    return True, f"moved -> {plan.dst}"


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def _render_plans(shared_plans: list[MovePlan],
                  task_plans: list[MovePlan],
                  datasets_root: Path) -> None:
    print(f"Workspace datasets root: {datasets_root}")
    print()

    print(f"Phase 1 — /datasets/ top-level → /datasets/single_cell/  "
          f"({len(shared_plans)} entries)")
    if not shared_plans:
        print("  (nothing to move)")
    else:
        total_sz = 0
        for p in shared_plans:
            total_sz += p.size_bytes
            kind_marker = "[dir]" if p.kind == "shared_dir" else "[file]"
            print(f"  {kind_marker:>6}  {p.src.name:<40}  "
                  f"{_humanize(p.size_bytes):>8}  →  shared/{p.src.name}")
            print(f"          and leave compat symlink at {p.compat_link}")
        print(f"  total: {_humanize(total_sz)}")
    print()

    print(f"Phase 2 — <task>/data/ → /datasets/per_task/<task>/  "
          f"({len(task_plans)} tasks)")
    if not task_plans:
        print("  (nothing to move)")
    else:
        total_sz = 0
        for p in task_plans:
            total_sz += p.size_bytes
            task_name = p.src.parent.name
            tag = f"  {'!' if p.note else ' '}  {task_name:<40}"
            print(f"  {tag}  {_humanize(p.size_bytes):>8}  →  "
                  f"per_task/{task_name}/")
            if p.note:
                print(f"      note: {p.note}")
        print(f"  total: {_humanize(total_sz)}")
    print()

    grand = sum(p.size_bytes for p in shared_plans + task_plans)
    print(f"Grand total: {_humanize(grand)} across "
          f"{len(shared_plans) + len(task_plans)} ops")


# ----------------------------------------------------------------------
# Command entrypoints
# ----------------------------------------------------------------------

def cmd_datasets_migrate(args):
    """Reorganize /datasets/ + each task's data/. Dry-run by default."""
    workspace = find_workspace_root(Path.cwd())
    if workspace is None:
        print("zyme: no autozyme-framework/ found above cwd", file=sys.stderr)
        sys.exit(1)
    framework = find_framework_root(Path.cwd())
    datasets_root = workspace / "datasets"
    if not datasets_root.is_dir():
        print(f"zyme: {datasets_root} not found", file=sys.stderr)
        sys.exit(1)

    task_dirs = find_tasks([workspace], max_depth=3, framework_root=framework)

    # --task NAME [NAME ...] filters Phase 2 to just those task dirs.
    # When set, Phase 1 (shared/) is automatically skipped — re-running
    # shared migration would be a no-op anyway (already moved), so the
    # filter alone defines the intent: "I want to migrate just these tasks".
    if args.tasks:
        wanted = set(args.tasks)
        matched = [td for td in task_dirs if td.name in wanted]
        unmatched = wanted - {td.name for td in matched}
        if unmatched:
            print(f"zyme: --task target(s) not found: {sorted(unmatched)}",
                  file=sys.stderr)
            print(f"  known tasks: {sorted(td.name for td in task_dirs)}",
                  file=sys.stderr)
            sys.exit(1)
        task_dirs_to_plan = matched
        do_shared = False
    else:
        task_dirs_to_plan = task_dirs
        do_shared = args.phase in ("shared", "all")

    shared_plans = (plan_shared_migration(datasets_root)
                    if do_shared else [])
    task_plans = (plan_task_migration(task_dirs_to_plan, datasets_root)
                  if args.phase in ("tasks", "all") else [])

    _render_plans(shared_plans, task_plans, datasets_root)

    if not args.execute:
        print()
        print("DRY RUN — no files moved. Re-run with --execute to apply.")
        return

    # Real run. Phase 1 first (so per-task _raw/ symlinks pointing into
    # /datasets/X.rds get their compat layer in place before phase 2),
    # then phase 2.
    print()
    print("=" * 60)
    print("EXECUTING — moving files now.")
    print("=" * 60)
    failures = 0
    for label, plans in (("shared", shared_plans), ("task", task_plans)):
        for p in plans:
            ok, msg = _execute_move(p)
            tag = "OK" if ok else "FAIL"
            print(f"  [{label}/{tag}] {p.src.name}: {msg}")
            if not ok:
                failures += 1
    print()
    if failures:
        print(f"Done with {failures} failure(s) — review above.")
        sys.exit(2)
    else:
        print("Done. All migrations succeeded.")


def add_datasets_subparser(sub):
    """Register `zyme datasets ...` under the given top-level subparser."""
    p = sub.add_parser(
        "datasets",
        help="Manage the workspace /datasets/ layer (shared + per_task layout).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # See what migration WOULD do — does not touch any file:\n"
            "  zyme datasets migrate\n\n"
            "  # Actually move files into /datasets/single_cell/ + /datasets/per_task/<task>/\n"
            "  # and turn each <task>/data/ into a symlink. Run with a stable shell;\n"
            "  # do not have any zyme run or hf upload reading from /datasets/.\n"
            "  zyme datasets migrate --execute\n\n"
            "  # Only Phase 1 (top-level /datasets/ -> shared/):\n"
            "  zyme datasets migrate --phase shared --execute\n\n"
            "  # Only Phase 2 (each task's data/ -> per_task/<task>/):\n"
            "  zyme datasets migrate --phase tasks --execute\n\n"
            "  # Migrate just one finished task (typical post-task workflow):\n"
            "  zyme datasets migrate --task test_my_new_task --execute\n"
        ),
    )
    psub = p.add_subparsers(dest="datasets_cmd", required=True)

    pm = psub.add_parser("migrate",
        help="Reorganize /datasets/ into shared/ + per_task/<task>/ "
             "and convert each <task>/data/ to a symlink. Dry-run by default.")
    pm.add_argument("--phase", choices=["shared", "tasks", "all"], default="all",
                    help="Which phase(s) to run. Default: all.")
    pm.add_argument("--task", dest="tasks", nargs="+", default=None,
                    metavar="NAME",
                    help="Migrate only these task(s) by directory name. "
                         "Skips Phase 1 (shared/). Use after you've finished "
                         "a new task that was developed with data in "
                         "<task>/data/ — `zyme datasets migrate --task <name> "
                         "--execute` moves it into /datasets/per_task/<name>/ "
                         "and symlinks <task>/data/ back.")
    pm.add_argument("--execute", action="store_true",
                    help="Actually move files. Without this, only the plan is "
                         "printed (dry-run).")
    pm.set_defaults(func=cmd_datasets_migrate)
