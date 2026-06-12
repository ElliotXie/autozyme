"""Shared helpers for smoke recipes."""
from __future__ import annotations

import os


def _within(task_dir: str, candidate: str) -> bool:
    """True iff ``candidate`` is inside ``task_dir`` by literal path (after
    ``..`` collapse, but BEFORE following symlinks).

    Uses ``os.path.abspath``, not ``os.path.realpath``: abspath normalizes
    ``..`` so traversal like ``task_dir/../../etc/passwd`` is still rejected,
    but does NOT resolve symlinks. The canonical layout puts datasets on a
    symlinked / Windows-junctioned ``data/`` slot pointing at the shared
    dataset registry (`<DATASETS_ROOT>/per_task/...`); resolving that to its
    target would (correctly) land outside the task tree and bounce. The
    literal-path check is enough to block traversal attacks because the
    pre-symlink path is still controlled by the smoke recipe.

    Uses ``os.path.commonpath`` rather than string prefix so ``task_dir =
    /a/b`` does not accidentally accept ``/a/bc``.
    """
    try:
        common = os.path.commonpath([
            os.path.abspath(task_dir),
            os.path.abspath(candidate),
        ])
    except ValueError:
        return False
    return common == os.path.abspath(task_dir)


def resolve_dataset_path(task_dir: str, raw_path: str) -> str:
    """Resolve a task.yaml `datasets[i].path` field to an actual local file.

    task.yaml's `path` may be:
      - a clean relative path like `data/tiny.csv`
      - a `./...` relative path
      - an absolute path (sometimes stale from a previous repo layout)
      - a Windows path (`D:/...`) on tasks ported from a different OS

    Tries (in order): as-is (if absolute) or task_dir + path (if relative);
    task_dir + path stripped of leading `./`; task_dir/data/<basename>.

    Relative paths must resolve INSIDE ``task_dir`` — ``../../etc/passwd`` is
    rejected even if the file exists, since smoke recipes parse arbitrary
    task.yaml content and should not be able to read outside the task tree.
    Absolute paths are accepted as-is (legacy: tasks ported across machines
    may carry stale absolute paths that the user explicitly trusts).

    Raises FileNotFoundError if nothing matches.
    """
    if os.path.isabs(raw_path):
        candidates = [raw_path]
        bounded = [False]  # absolute path: trust user-provided value
    else:
        candidates = [os.path.join(task_dir, raw_path)]
        bounded = [True]
    candidates.append(os.path.join(task_dir, raw_path.lstrip("./")))
    bounded.append(True)
    candidates.append(os.path.join(task_dir, "data", os.path.basename(raw_path)))
    bounded.append(True)

    for c, must_bound in zip(candidates, bounded):
        if not os.path.exists(c):
            continue
        if must_bound and not _within(task_dir, c):
            continue
        return c
    raise FileNotFoundError(
        f"could not resolve dataset path {raw_path!r} from task_dir {task_dir!r}; "
        f"tried: {candidates!r}"
    )
