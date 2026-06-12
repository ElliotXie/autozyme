"""Dataset-focused scan: for every task.yaml, report where its tier
datasets live and how big they are. Read-only.

Backs `zyme scan --dataset`. The motivation is two-fold:
  1) Tasks get moved between hosts/drives; datasets referenced by
     absolute path (e.g. /Users/.../datasets/foo.h5ad) silently get
     left behind and the next `zyme run` blows up at load time.
  2) Storage hygiene — knowing which shared datasets are still
     referenced (and by which tasks) makes it safe to delete the
     ones that aren't.

Each dataset entry is labelled by scope:
  - local:    travels with the task. Either physically inside task_dir
              (legacy <task>/data/...) OR inside /datasets/per_task/<task>/
              which the new layout symlinks back into <task>/data/.
  - shared:   inside /datasets/{single_cell,spatial,bulk}/ — referenced by
              potentially many tasks, hosted in the HF mega-repo. Not
              task-local but recoverable via `hf download`.
  - external: lives outside the workspace entirely (e.g. ~/.dipy/ cache,
              random absolute path on user's machine). Must be brought
              along manually when relocating a task.
  - missing:  path is declared in task.yaml but does not exist.
"""
import os
from pathlib import Path

from zyme.parsers.task_yaml import parse_datasets
from zyme.scan import find_workspace_root


def path_size_bytes(path: Path) -> int:
    """Return on-disk size of file or directory in bytes; 0 on error."""
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for dirpath, _, filenames in os.walk(path, followlinks=False):
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    try:
                        total += fp.stat().st_size
                    except OSError:
                        pass
            return total
    except OSError:
        pass
    return 0


def _is_inside(child: Path, parent: Path) -> bool:
    """True if `child` resolves to a path under `parent` (both already resolved)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _shared_roots(datasets_root: Path | None) -> list[Path]:
    """Category dirs under /datasets/ that hold workspace-shared corpora."""
    if not datasets_root:
        return []
    return [
        (datasets_root / name).resolve()
        for name in ("single_cell", "spatial", "bulk")
        if (datasets_root / name).exists()
    ]


def _classify_scope(resolved: Path | None, *, exists: bool, task_dir: Path,
                    per_task_self: Path | None, shared_dirs: list[Path]) -> str:
    """Bucket a dataset path into local / shared / external / missing.

    Order matters: per-task-self check beats shared check, because a
    well-formed task only references its own per_task slot. Legacy
    "data inside task_dir" still counts as local for pre-migration
    repos.
    """
    if not exists or resolved is None:
        return "missing"
    if _is_inside(resolved, task_dir):
        return "local"  # legacy: data physically under <task>/
    if per_task_self and _is_inside(resolved, per_task_self):
        return "local"  # new layout: <task>/data symlinks to here
    for shared_dir in shared_dirs:
        if _is_inside(resolved, shared_dir):
            return "shared"
    return "external"


# Top-level dirs in a task get bucketed into one of these labels. Order
# matters for first-match prefix: "reference_output" must beat "data"
# alphabetical accidents, etc. Anything that matches no bucket goes to
# "other" (still summed, just not broken out).
#
# Categories chosen by what a user moving / cleaning tasks actually
# wants to see distinctly:
#   - data/             — the dataset payload (matched against task.yaml)
#   - reference_outputs — golden reference outputs (init artifact)
#   - upstream_repo     — git clone of the library being optimized;
#                         re-cloneable, often the biggest "deletable" hog
_SUBDIR_BUCKETS = [
    ("data",              lambda n: n == "data" or n.startswith("data_")),
    ("reference_outputs", lambda n: n.startswith("reference_output")),
    ("upstream_repo",     lambda n: n == "upstream_repo"),
]


def _classify_subdir(name: str) -> str:
    for label, pred in _SUBDIR_BUCKETS:
        if pred(name):
            return label
    return "other"


def _task_subdir_breakdown(task_dir: Path) -> tuple[int, dict]:
    """Walk top-level entries of `task_dir`, return (total_bytes, {bucket: bytes}).

    Top-level files (e.g. task.yaml, results.tsv) go into "other". Dot
    files / dirs are skipped — `.git`, `.cache`, etc. don't belong in
    the storage-management view.

    Symlinks: framework symlinks (autozyme-framework → ../...) are
    skipped because they belong to the framework, not the task. The
    `data → /datasets/per_task/<task>/` symlink IS followed and its
    target's bytes are counted under the `data` bucket — the new
    layout puts the task's data behind that symlink, and from a
    storage-attribution perspective those bytes belong to this task.
    """
    breakdown = {label: 0 for label, _ in _SUBDIR_BUCKETS}
    breakdown["other"] = 0
    total = 0
    try:
        for child in task_dir.iterdir():
            if child.name.startswith("."):
                continue
            if child.is_symlink():
                # Special case: follow data/ symlink (new layout puts the
                # task's bytes there). Skip every other symlink — those
                # are framework / cross-task references.
                if child.name == "data":
                    try:
                        target = child.resolve()
                        if target.is_dir():
                            n = path_size_bytes(target)
                            breakdown["data"] += n
                            total += n
                    except OSError:
                        pass
                continue
            if child.is_file():
                try:
                    n = child.stat().st_size
                except OSError:
                    n = 0
                breakdown["other"] += n
                total += n
                continue
            if child.is_dir():
                n = path_size_bytes(child)
                breakdown[_classify_subdir(child.name)] += n
                total += n
    except OSError:
        pass
    return total, breakdown


def _is_python_venv(d: Path) -> bool:
    """Heuristic: dir is a Python virtualenv if it has a `pyvenv.cfg`."""
    try:
        return (d / "pyvenv.cfg").is_file()
    except OSError:
        return False


def _detect_data_orphans(task_dir: Path, declared_paths: set[str]) -> list[dict]:
    """List top-level entries under task_dir/data/ NOT declared in task.yaml.

    Walks `data/` one level deep only — each undeclared top-level entry
    (file OR directory) becomes one orphan record. This matters because:
      - Extracted archives (e.g. BBBC005_v1_images/) hold thousands of
        leaf files; reporting each individually buries the signal.
      - Python virtualenvs (data/pybamm_env/) have ~10k leaves likewise.
      - Symlinked _raw/ trees would otherwise dereference to /datasets/
        and falsely show as orphans pointing outside the task.

    Three classes are NOT flagged as orphans:
      - `_raw/` subdir — v2 layout convention for tracked raw upstream
        (often symlinks into /datasets/, intentional)
      - Symlinks at the top level — they don't occupy real disk weight
        in this task; the storage lives at the target.
      - Anything matching a declared dataset path (or being a parent of
        one — e.g. when task.yaml declares `data/foo/file.rds`, the
        `foo/` dir is covered).

    Returns list of {"path", "size_bytes", "kind", "n_files"} sorted by
    size desc. `kind` is "file", "dir", or "venv".
    """
    data_dir = task_dir / "data"
    if not data_dir.is_dir():
        return []

    # Normalize declared paths to absolute strings. `declared_paths` is
    # already-resolved (parse_datasets() does .resolve() at parse time),
    # so the entry side also needs to be resolved before comparison —
    # otherwise a post-migration entry like `<task>/data/foo.rds`
    # (a path *through* the data → /datasets/per_task/<task>/ symlink)
    # won't match its declared form `/datasets/per_task/<task>/foo.rds`.
    declared_set: set[str] = set()
    for p in declared_paths:
        if p:
            declared_set.add(str(Path(p)))

    def _covers_a_declared(entry_path: Path) -> bool:
        # Compare both raw and resolved forms — covers both legacy
        # (no symlink) and new-layout (symlinked data/) tasks.
        candidates = {str(entry_path)}
        try:
            candidates.add(str(entry_path.resolve()))
        except OSError:
            pass
        for ep in candidates:
            if ep in declared_set:
                return True
            prefix = ep.rstrip("/") + "/"
            if any(d.startswith(prefix) for d in declared_set):
                return True
        return False

    orphans = []
    try:
        children = list(data_dir.iterdir())
    except OSError:
        return []

    for entry in children:
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            # Symlinks don't take real bytes in this task dir; skip.
            continue
        if _covers_a_declared(entry):
            continue
        # `_raw/` is the v2 layout staging dir — tracked on purpose.
        if entry.is_dir() and entry.name == "_raw":
            continue
        # Report the canonical (resolved) path so post-migration orphans
        # show their real /datasets/per_task/<task>/... location, not
        # the symlink-traversed <task>/data/... path.
        try:
            display_path = str(entry.resolve())
        except OSError:
            display_path = str(entry)
        if entry.is_file():
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            orphans.append({
                "path": display_path,
                "size_bytes": size,
                "kind": "file",
                "n_files": 1,
            })
            continue
        if entry.is_dir():
            kind = "venv" if _is_python_venv(entry) else "dir"
            # Sum bytes + leaf count for the whole subtree (one orphan
            # record per top-level dir, regardless of how many leaves).
            n_files = 0
            size = 0
            for dirpath, _, filenames in os.walk(entry, followlinks=False):
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    try:
                        size += fp.stat().st_size
                        n_files += 1
                    except OSError:
                        pass
            orphans.append({
                "path": display_path,
                "size_bytes": size,
                "kind": kind,
                "n_files": n_files,
            })
    orphans.sort(key=lambda x: x["size_bytes"], reverse=True)
    return orphans


def inspect_task_datasets(task_dir: Path) -> dict:
    """Inspect one task's dataset declarations + on-disk storage footprint.

    Returns a dict shaped for the dataset table:
        {
          "task_dir": str, "dir_name": str, "category": str,
          "datasets": [ {tier, name, path, exists, is_dir,
                         size_bytes, scope}, ... ],
          "totals": {local_count, local_bytes,
                     external_count, external_bytes,
                     missing_count, n_entries,
                     task_dir_bytes,
                     subdir: {data, reference_outputs, upstream_repo, other}},
          "orphans": [ {path, size_bytes}, ... ],   # files under data/ not
                                                    # declared in task.yaml
        }

    The `subdir` + `task_dir_bytes` breakdown answers "how big is this
    task on disk, and where is the weight?" — declared-data may be a
    fraction of the total once reference_outputs/ and upstream_repo/
    pile up. Orphans flag forgotten data/ files (deletion candidates).
    """
    task_dir = Path(task_dir).resolve()
    task_yaml = task_dir / "task.yaml"
    entries = parse_datasets(task_yaml)

    # Resolve workspace layout roots once per task. Needed so the scope
    # classifier can recognize /datasets/per_task/<task>/ as "still
    # local to this task" (post-migration layout) and
    # /datasets/{single_cell,spatial,bulk}/ as shared buckets.
    workspace = find_workspace_root(task_dir)
    datasets_root = (workspace / "datasets").resolve() if workspace else None
    shared_dirs = _shared_roots(datasets_root)
    per_task_self = ((datasets_root / "per_task" / task_dir.name)
                     if datasets_root else None)

    out_entries = []
    declared_paths: set[str] = set()
    for e in entries:
        raw = e.get("path", "")
        p = Path(raw) if raw else None
        # parse_datasets() already resolves relative paths against task_dir,
        # so `p` should be absolute. Still guard the symlink case by
        # resolving once more.
        resolved = p.resolve() if p else None
        exists = bool(resolved and resolved.exists())
        is_dir = bool(resolved and resolved.is_dir())
        size = path_size_bytes(resolved) if exists else 0
        scope = _classify_scope(
            resolved, exists=exists, task_dir=task_dir,
            per_task_self=per_task_self, shared_dirs=shared_dirs,
        )
        out_entries.append({
            "tier": e.get("tier", ""),
            "name": e.get("name", ""),
            "path": str(resolved) if resolved else raw,
            "exists": exists,
            "is_dir": is_dir,
            "size_bytes": size,
            "scope": scope,
        })
        if resolved:
            declared_paths.add(str(resolved))

    task_dir_bytes, subdir_bytes = _task_subdir_breakdown(task_dir)
    orphans = _detect_data_orphans(task_dir, declared_paths)

    totals = {
        "n_entries": len(out_entries),
        "local_count": sum(1 for e in out_entries if e["scope"] == "local"),
        "local_bytes": sum(e["size_bytes"] for e in out_entries if e["scope"] == "local"),
        "shared_count": sum(1 for e in out_entries if e["scope"] == "shared"),
        "shared_bytes": sum(e["size_bytes"] for e in out_entries if e["scope"] == "shared"),
        "external_count": sum(1 for e in out_entries if e["scope"] == "external"),
        "external_bytes": sum(e["size_bytes"] for e in out_entries if e["scope"] == "external"),
        "missing_count": sum(1 for e in out_entries if e["scope"] == "missing"),
        "task_dir_bytes": task_dir_bytes,
        "subdir": subdir_bytes,
        "orphan_count": len(orphans),
        "orphan_bytes": sum(o["size_bytes"] for o in orphans),
    }

    return {
        "task_dir": str(task_dir),
        "dir_name": task_dir.name,
        "category": task_dir.parent.name,
        "datasets": out_entries,
        "totals": totals,
        "orphans": orphans,
    }


def _aggregate_by_scope(rows: list[dict], scope: str) -> list[dict]:
    """Collapse entries of `scope` across tasks by canonical path.

    Same path (e.g. `/datasets/single_cell/ifnb.rds`) often gets referenced
    by multiple tasks at different tiers. We report it once with the
    full reference list — that's the view that answers "what would I
    orphan if I deleted this file?".

    Returns list sorted by size desc:
        [{path, size_bytes, references: [{task, tier, name}, ...]}, ...]
    """
    by_path: dict[str, dict] = {}
    for row in rows:
        for e in row["datasets"]:
            if e["scope"] != scope:
                continue
            slot = by_path.setdefault(e["path"], {
                "path": e["path"],
                "size_bytes": e["size_bytes"],
                "references": [],
            })
            slot["references"].append({
                "task": row["dir_name"],
                "category": row["category"],
                "tier": e["tier"],
                "name": e["name"],
            })
    return sorted(by_path.values(), key=lambda x: x["size_bytes"], reverse=True)


def aggregate_external(rows: list[dict]) -> list[dict]:
    """External-path roster across all tasks (outside workspace)."""
    return _aggregate_by_scope(rows, "external")


def aggregate_shared(rows: list[dict]) -> list[dict]:
    """Shared-path roster across all tasks (/datasets/single_cell/ entries)."""
    return _aggregate_by_scope(rows, "shared")


def collect_missing(rows: list[dict]) -> list[dict]:
    """Flat list of every missing dataset reference across tasks."""
    out = []
    for row in rows:
        for e in row["datasets"]:
            if e["scope"] == "missing":
                out.append({
                    "task": row["dir_name"],
                    "category": row["category"],
                    "tier": e["tier"],
                    "name": e["name"],
                    "path": e["path"],
                })
    return out


def collect_orphans(rows: list[dict]) -> list[dict]:
    """Flat list of (task, orphan_path, bytes) for every data/ orphan."""
    out = []
    for row in rows:
        for o in row.get("orphans", []):
            out.append({
                "task": row["dir_name"],
                "category": row["category"],
                "path": o["path"],
                "size_bytes": o["size_bytes"],
            })
    out.sort(key=lambda x: x["size_bytes"], reverse=True)
    return out


def aggregate_totals(rows: list[dict]) -> dict:
    out = {
        "n_tasks": len(rows),
        "n_entries": 0,
        "local_count": 0, "local_bytes": 0,
        "shared_count": 0, "shared_bytes": 0,
        "external_count": 0, "external_bytes": 0,
        "missing_count": 0,
        "task_dir_bytes": 0,
        "orphan_count": 0, "orphan_bytes": 0,
        "subdir": {"data": 0, "reference_outputs": 0,
                   "upstream_repo": 0, "other": 0},
    }
    for r in rows:
        for k, v in r["totals"].items():
            if k == "subdir":
                for sk, sv in v.items():
                    out["subdir"][sk] = out["subdir"].get(sk, 0) + sv
            else:
                out[k] = out.get(k, 0) + v
    # Unique-path tallies for shared + external: a single file
    # referenced by N tasks counts as 1 path / 1× its size in the
    # on-disk footprint number (vs the raw count which sums refs).
    for scope_label in ("shared", "external"):
        seen: set[str] = set()
        unique_bytes = 0
        for r in rows:
            for e in r["datasets"]:
                if e["scope"] == scope_label and e["path"] not in seen:
                    seen.add(e["path"])
                    unique_bytes += e["size_bytes"]
        out[f"{scope_label}_unique_paths"] = len(seen)
        out[f"{scope_label}_unique_bytes"] = unique_bytes
    return out


def format_breakdown(subdir: dict) -> str:
    """Compact one-line breakdown: `d:54G ro:625M u:45M o:35M`. Skips zeros."""
    short = {"data": "d", "reference_outputs": "ro",
             "upstream_repo": "u", "other": "o"}
    parts = []
    for key in ("data", "reference_outputs", "upstream_repo", "other"):
        n = subdir.get(key, 0)
        if n > 0:
            parts.append(f"{short[key]}:{humanize_bytes(n)}")
    return " ".join(parts) if parts else "-"


def humanize_bytes(n: int) -> str:
    """Compact human-readable size: '12.4G', '450M', '120K', '0'."""
    if n <= 0:
        return "0"
    units = [("T", 1024 ** 4), ("G", 1024 ** 3), ("M", 1024 ** 2), ("K", 1024)]
    for suffix, scale in units:
        if n >= scale:
            v = n / scale
            return f"{v:.1f}{suffix}" if v < 10 else f"{v:.0f}{suffix}"
    return f"{n}B"
