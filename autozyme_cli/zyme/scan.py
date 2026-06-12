"""Workspace-wide task lifecycle detection (read-only).

Backs the `zyme scan` subcommand: walk filesystem trees, find every task
(directory containing task.yaml), and infer which lifecycle phase each
task has reached based on filesystem markers.

Phases:
    scaffold  — `zyme init` ran, init.md has not.
    init      — init.md done (baselines + reference_outputs/, .template stripped).
    iterate   — iterate.md produced ≥1 decision round.
    scaling   — scaling.md done (ood_* tiers + verify.tsv).
    package   — package.md lifted a patch into autozyme_py or autozyme_r.

Reflect signal is orthogonal: feedback files in <framework>/reflections/
are tagged by category (initialization, iteration, scaling, packaging).
"""
import os
import re
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from zyme.parsers.task_yaml import parse_datasets
from zyme.parsers.results_tsv import count_decision_rounds

PHASE_ORDER = ["scaffold", "init", "iterate", "scaling", "package"]
REFLECT_CATEGORIES = ["initialization", "iteration", "scaling", "packaging"]
DEFAULT_ACTIVE_RECENT_MINUTES = 15

_ROUND_DIR_RE = re.compile(r"^\d+(\.\d+)?_[0-9a-f]{6,}_\w+$")
_TASK_FIELD_RE = re.compile(r"^task:\s*(\S+)\s*$")
_LIVE_DISPATCH_LABELS = {
    "pending": "pend",
    "resource_wait": "wait",
    "running": "run",
}


def find_workspace_root(start: Path) -> Path | None:
    """Walk up from `start` until finding a dir whose `autozyme-framework/`
    child is the REAL framework dir (not a symlink to elsewhere). Returns
    None if filesystem root is reached.

    The realpath check matters because every category dir
    (core_singlecell/, general_bio/, non_bio/) contains a symlink
    `autozyme-framework -> ../autozyme-framework` to keep relative path
    lookups working — those symlinks must not be mistaken for the
    workspace.
    """
    cur = Path(start).resolve()
    while True:
        cand = cur / "autozyme-framework"
        if cand.is_dir() and cand.resolve().parent == cur:
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def find_framework_root(start: Path) -> Path | None:
    ws = find_workspace_root(start)
    return (ws / "autozyme-framework").resolve() if ws else None


def find_tasks(roots: list[Path], max_depth: int, framework_root: Path | None) -> list[Path]:
    """Return task directories (those containing task.yaml) under `roots`.

    Uses os.walk with followlinks=False so the symlinked
    `<category>/autozyme-framework -> ../autozyme-framework` is not
    descended into. Also drops any visited dir whose RESOLVED realpath
    lies inside `framework_root`, in case a non-symlink path leads
    there. Stops descending once a task.yaml is found in a dir.
    """
    fw_real = framework_root.resolve() if framework_root else None
    seen: set[Path] = set()
    tasks: list[Path] = []

    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dp = Path(dirpath)
            try:
                depth = len(dp.relative_to(root).parts)
            except ValueError:
                continue
            if depth > max_depth:
                dirnames[:] = []
                continue
            real = dp.resolve()
            if fw_real and (real == fw_real or fw_real in real.parents):
                dirnames[:] = []
                continue
            if "task.yaml" in filenames and real not in seen:
                seen.add(real)
                tasks.append(dp)
                dirnames[:] = []

    tasks.sort(key=lambda p: (p.parent.as_posix(), p.name))
    return tasks


def _read_task_name(task_yaml: Path) -> str:
    """Extract `task:` field from task.yaml. Falls back to dir name."""
    try:
        for line in task_yaml.read_text().splitlines():
            m = _TASK_FIELD_RE.match(line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return task_yaml.parent.name


def _reference_output_dir(task_dir: Path, tier: str) -> Path:
    """Return the reference-output dir for a tier.

    Current tasks use `reference_outputs/<tier>/`; older tasks used
    `reference_output_<tier>/`. Scan accepts either layout.
    """
    current = task_dir / "reference_outputs" / tier
    if current.is_dir():
        return current
    return task_dir / f"reference_output_{tier}"


def _refs_present(task_dir: Path, datasets: list[dict]) -> bool:
    """All three reference-output tier dirs exist and are non-empty."""
    if len(datasets) < 3:
        return False
    for d in datasets[:3]:
        ref = _reference_output_dir(task_dir, str(d["tier"]))
        if not ref.is_dir():
            return False
        try:
            if not any(ref.iterdir()):
                return False
        except OSError:
            return False
    return True


def _has_round_dir(task_dir: Path) -> bool:
    artifacts = task_dir / "artifacts"
    if not artifacts.is_dir():
        return False
    try:
        return any(_ROUND_DIR_RE.match(p.name) for p in artifacts.iterdir() if p.is_dir())
    except OSError:
        return False


def _verify_data_rows(task_dir: Path) -> int:
    verify = task_dir / "verify.tsv"
    if not verify.is_file():
        return 0
    try:
        lines = verify.read_text().splitlines()
    except OSError:
        return 0
    return max(0, len(lines) - 1)


_LIFTED_FROM_RE = re.compile(r"Lifted from autozyme task\s+`+([^`]+)`+")


@lru_cache(maxsize=8)
def _build_lifted_from_index(framework_root_str: str) -> dict[str, str]:
    """Reverse index mapping task `dir_name` -> patch file path, built by
    parsing the ``Lifted from autozyme task `<dir_name>` `` marker in each
    patch's docstring/header. Convention every patch file follows.

    Lets scan detect task -> packaged-patch mappings even though the patch
    directory is named after the upstream package, not the task. Keyed on
    str for lru_cache hashability.
    """
    framework_root = Path(framework_root_str)
    out: dict[str, str] = {}
    r_dir = framework_root / "autozyme_r" / "inst" / "patches"
    if r_dir.is_dir():
        try:
            entries = list(r_dir.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            # Folder layout: inst/patches/<name>/patch.R
            if entry.is_dir():
                patch_r = entry / "patch.R"
                if patch_r.is_file():
                    task = _extract_lifted_from(patch_r)
                    if task:
                        out.setdefault(task, str(patch_r))
                continue
            # Legacy single-file: inst/patches/<name>.R
            if entry.suffix != ".R" or not entry.is_file():
                continue
            task = _extract_lifted_from(entry)
            if task:
                out.setdefault(task, str(entry))
    py_dir = framework_root / "autozyme_py" / "src" / "autozyme"
    if py_dir.is_dir():
        try:
            entries = list(py_dir.iterdir())
        except OSError:
            entries = []
        for d in entries:
            if not d.is_dir() or d.name.startswith("_"):
                continue
            init = d / "__init__.py"
            if not init.is_file():
                continue
            task = _extract_lifted_from(init)
            if task:
                out.setdefault(task, str(init))
    return out


def _extract_lifted_from(path: Path) -> str | None:
    try:
        text = path.read_text(errors="ignore")[:8192]
    except OSError:
        return None
    m = _LIFTED_FROM_RE.search(text)
    return m.group(1).strip() if m else None


def _find_patch(framework_root: Path, *stems: str) -> str | None:
    """Look for a packaged patch under autozyme_py or autozyme_r matching
    any of the given task-name candidates. Returns the first match.

    Primary lookup is by patch file/dir name (cheap). Fallback consults the
    `Lifted from autozyme task` reverse index — patches conventionally name
    themselves after the upstream package while task dirs are named after
    the task (`test_<upstream>`), so the names don't always overlap. The
    marker in each patch's docstring is the authoritative back-link.
    """
    for stem in stems:
        if not stem:
            continue
        py_patch = framework_root / "autozyme_py" / "src" / "autozyme" / stem / "__init__.py"
        if py_patch.is_file():
            return str(py_patch)
        r_patch = framework_root / "autozyme_r" / "inst" / "patches" / f"{stem}.R"
        if r_patch.is_file():
            return str(r_patch)
    idx = _build_lifted_from_index(str(framework_root))
    for stem in stems:
        if stem and stem in idx:
            return idx[stem]
    return None


def _detect_reflect(framework_root: Path, *stems: str) -> dict[str, bool]:
    """For each reflect category, check whether any feedback file matching
    `<category>_<stem>*.md` exists in either feedback dir.
    """
    out = {c: False for c in REFLECT_CATEGORIES}
    feedback_dirs = [
        framework_root / "reflections" / "prompt_reflect_feedback",
        framework_root / "reflections" / "zyme_cli_feedback",
    ]
    real_stems = [s for s in stems if s]
    for category in REFLECT_CATEGORIES:
        for d in feedback_dirs:
            if not d.is_dir():
                continue
            try:
                names = [p.name for p in d.iterdir()
                         if p.is_file() and p.suffix == ".md"]
            except OSError:
                continue
            for stem in real_stems:
                prefix = f"{category}_{stem}"
                if any(n.startswith(prefix) for n in names):
                    out[category] = True
                    break
            if out[category]:
                break
    return out


def _latest_keep(results_tsv: Path) -> dict | None:
    """Return the most recent status=keep row's headline numbers, or None.

    Surfaced by `zyme scan` so the dashboard shows progress at a glance —
    you can see "this task is at +12% on medium" without opening results.tsv.
    Walks the file backward to find the latest keep, returning a dict with
    the row's round, dataset (tier/name), thread, and speedup_pct (parsed
    as float). Returns None when results.tsv is missing, empty, or has no
    keep row yet.
    """
    if not results_tsv.exists():
        return None
    try:
        lines = results_tsv.read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    needed = ("status", "round", "dataset", "speedup_pct")
    if not all(n in col for n in needed):
        return None
    for line in reversed(lines[1:]):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) <= col["status"]:
            continue
        if parts[col["status"]] != "keep":
            continue
        try:
            pct = float(parts[col["speedup_pct"]])
        except (ValueError, IndexError):
            pct = None
        thread = None
        if "thread" in col and col["thread"] < len(parts):
            try:
                thread = int(parts[col["thread"]])
            except ValueError:
                pass
        return {
            "round": parts[col["round"]],
            "dataset": parts[col["dataset"]],
            "thread": thread,
            "speedup_pct": pct,
        }
    return None


def _utc_iso_from_mtime(mtime: float) -> str:
    return (
        datetime.fromtimestamp(mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _dispatch_state_paths_for_task(task_dir: Path) -> list[Path]:
    """Candidate ancestor dispatch states that may own this task.

    `zyme dispatch` writes `.zyme_dispatch/state.json` at its workspace root.
    For bench runs that root is the bench-run dir; for direct dispatch it is
    often a category dir or the workspace. Walking ancestors finds both.
    """
    out = []
    for cur in (task_dir, *task_dir.parents):
        sp = cur / ".zyme_dispatch" / "state.json"
        if sp.is_file():
            out.append(sp)
    return out


def _path_matches_task(raw: str, task_dir: Path) -> bool:
    if not raw:
        return False
    try:
        return Path(raw).resolve() == task_dir
    except OSError:
        return False


def _dispatch_activity(task_dir: Path) -> dict | None:
    """Return live/stale dispatch activity for a task, if a state owns it."""
    try:
        from zyme.dispatch.state import pid_alive
    except Exception:
        pid_alive = None

    for sp in _dispatch_state_paths_for_task(task_dir):
        state = _read_json(sp)
        if not state:
            continue
        master_pid = state.get("master_pid")
        alive = bool(
            isinstance(master_pid, int)
            and pid_alive is not None
            and pid_alive(master_pid)
        )
        for task in state.get("queue") or []:
            if not isinstance(task, dict):
                continue
            if not (
                _path_matches_task(str(task.get("task_dir") or ""), task_dir)
                or str(task.get("name") or "") == task_dir.name
            ):
                continue
            dispatch_status = str(task.get("status") or "")
            reflect_status = str(task.get("reflect_status") or "")
            label = None
            active = False
            if alive and reflect_status == "running":
                label = "refl"
                active = True
            elif alive and dispatch_status in _LIVE_DISPATCH_LABELS:
                label = _LIVE_DISPATCH_LABELS[dispatch_status]
                active = True
            elif (
                not alive
                and (dispatch_status in _LIVE_DISPATCH_LABELS or reflect_status == "running")
            ):
                label = "stale"

            if label is None:
                return None
            return {
                "active": active,
                "status": label,
                "source": "dispatch",
                "confidence": "live" if active else "stale",
                "dispatch_status": dispatch_status or None,
                "reflect_status": reflect_status or None,
                "dispatch_alive": alive,
                "master_pid": master_pid,
                "state_path": str(sp),
                "last_event_at": task.get("last_event_at"),
                "agent": state.get("agent"),
                "model": task.get("actual_model") or state.get("model"),
            }
    return None


def _activity_file_candidates(task_dir: Path):
    for rel in (
        "results.tsv",
        "verify.tsv",
        ".zyme/audit.jsonl",
        ".zyme/round.counter",
        ".zyme/best.ref",
    ):
        yield task_dir / rel
    for dirname in ("pipeline", "memory"):
        d = task_dir / dirname
        if d.is_dir():
            try:
                for p in d.iterdir():
                    if p.is_file():
                        yield p
            except OSError:
                pass
    artifacts = task_dir / "artifacts"
    if artifacts.is_dir():
        try:
            for round_dir in artifacts.iterdir():
                if not round_dir.is_dir():
                    continue
                for p in round_dir.iterdir():
                    if p.is_file():
                        yield p
        except OSError:
            pass


def _recent_activity(task_dir: Path, active_recent_minutes: int | float | None) -> dict:
    threshold_min = (
        DEFAULT_ACTIVE_RECENT_MINUTES
        if active_recent_minutes is None
        else float(active_recent_minutes)
    )
    newest = None
    newest_path = None
    for path in _activity_file_candidates(task_dir):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
            newest_path = path

    if newest is None:
        return {"active": False, "status": "-", "source": None}

    age_min = max(0.0, (time.time() - newest) / 60.0)
    rel_path = None
    if newest_path is not None:
        try:
            rel_path = newest_path.relative_to(task_dir).as_posix()
        except ValueError:
            rel_path = str(newest_path)

    if threshold_min > 0 and age_min <= threshold_min:
        return {
            "active": True,
            "status": "recent",
            "source": "mtime",
            "confidence": "recent",
            "age_min": round(age_min, 1),
            "last_activity_at": _utc_iso_from_mtime(newest),
            "path": rel_path,
            "threshold_min": threshold_min,
        }
    return {
        "active": False,
        "status": "-",
        "source": "mtime",
        "confidence": "old",
        "age_min": round(age_min, 1),
        "last_activity_at": _utc_iso_from_mtime(newest),
        "path": rel_path,
        "threshold_min": threshold_min,
    }


def _detect_activity(task_dir: Path, active_recent_minutes: int | float | None) -> dict:
    dispatch = _dispatch_activity(task_dir)
    if dispatch is not None:
        return dispatch
    return _recent_activity(task_dir, active_recent_minutes)


def detect_phase(
    task_dir: Path,
    framework_root: Path | None,
    active_recent_minutes: int | float | None = DEFAULT_ACTIVE_RECENT_MINUTES,
) -> dict:
    """Return the lifecycle state of a single task as a dict.

    See module docstring for the phase definitions. `framework_root` is
    needed to resolve package patches and reflection feedback files; if
    None, those signals are reported as None / False.
    """
    task_dir = Path(task_dir).resolve()
    yaml_path = task_dir / "task.yaml"
    task_name = _read_task_name(yaml_path)
    dir_name = task_dir.name

    # --- scaffold ---------------------------------------------------------
    scaffold_done = yaml_path.is_file()
    scaffold_markers = ["task.yaml"]

    # --- init.md done -----------------------------------------------------
    # Note: leftover *.template files are NOT a scaffold-only signal. The
    # template dir ships both .py.template and .R.template; init.md renames
    # only the language-active set, so a fully init'd R task still has
    # `reference.py.template` etc. as harmless cruft. Discriminate on the
    # positive markers (datasets, refs, real pipeline + evaluate files)
    # instead.
    datasets = parse_datasets(yaml_path) if scaffold_done else []
    has_three = len(datasets) >= 3
    refs_ok = _refs_present(task_dir, datasets)
    pipeline_real = (
        (task_dir / "pipeline" / "run.py").exists()
        or (task_dir / "pipeline" / "run.R").exists()
    )
    eval_real = (
        (task_dir / "evaluate.py").exists()
        or (task_dir / "evaluate.R").exists()
    )
    init_done = (
        scaffold_done
        and has_three
        and refs_ok
        and pipeline_real
        and eval_real
    )
    init_markers = [
        f"datasets={len(datasets)}",
        f"refs_present={refs_ok}",
        f"pipeline_real={pipeline_real}",
        f"evaluate_real={eval_real}",
    ]

    # --- iterate.md done --------------------------------------------------
    results_tsv = task_dir / "results.tsv"
    best_ref = task_dir / ".zyme" / "best.ref"
    rounds = count_decision_rounds(results_tsv, phase="optimize") if results_tsv.exists() else 0
    best_commit = best_ref.read_text().strip() if best_ref.is_file() else ""
    has_round = _has_round_dir(task_dir)
    iterate_done = rounds >= 1 and bool(best_commit) and has_round

    # --- scaling.md done --------------------------------------------------
    ood_tiers = [d["tier"] for d in datasets if d["tier"].startswith("ood_")]
    verify_rows = _verify_data_rows(task_dir)
    scaling_done = bool(ood_tiers) and verify_rows >= 1

    # --- package.md done --------------------------------------------------
    patch_path = None
    if framework_root is not None:
        # Try task.yaml's `task:` field first, then dir name, then both
        # with `test_` prefix stripped (legacy fallback).
        candidates = [task_name, dir_name]
        for s in (task_name, dir_name):
            if s.startswith("test_"):
                candidates.append(s[len("test_"):])
        # Dedupe preserving order
        seen: set[str] = set()
        unique = [c for c in candidates if not (c in seen or seen.add(c))]
        patch_path = _find_patch(framework_root, *unique)
    package_done = patch_path is not None

    # --- reflect ----------------------------------------------------------
    if framework_root is not None:
        reflect = _detect_reflect(framework_root, task_name, dir_name)
    else:
        reflect = {c: False for c in REFLECT_CATEGORIES}

    phases = {
        "scaffold": {"done": scaffold_done, "markers": scaffold_markers},
        "init":     {"done": init_done,     "markers": init_markers},
        "iterate":  {"done": iterate_done,  "rounds": rounds,
                     "best_commit": (best_commit[:7] or None)},
        "scaling":  {"done": scaling_done,  "ood_tiers": ood_tiers,
                     "verify_rows": verify_rows},
        "package":  {"done": package_done,  "patch_path": patch_path},
    }

    # `phase` = rightmost done phase in the canonical order. This is just
    # a display summary; phases are NOT linearly cascading. A task can be
    # packaged without scaling (or even without iterate, e.g. a quick
    # patch lift from a copy of a converged task), so package=True does
    # NOT imply scaling=True. `gaps` lists any earlier phases left
    # incomplete behind the highest-done phase, so callers can surface
    # the discrepancy.
    rightmost_idx = -1
    for i, p in enumerate(PHASE_ORDER):
        if phases[p]["done"]:
            rightmost_idx = i
    if rightmost_idx < 0:
        phase_summary = "unknown"
        gaps: list[str] = []
    else:
        phase_summary = PHASE_ORDER[rightmost_idx]
        gaps = [PHASE_ORDER[i] for i in range(rightmost_idx)
                if not phases[PHASE_ORDER[i]]["done"]]

    return {
        "task_name": task_name,
        "dir_name": dir_name,
        "task_dir": str(task_dir),
        "phase": phase_summary,
        "gaps": gaps,
        "phases": phases,
        "reflect": reflect,
        "report_done": (task_dir / "report.html").is_file(),
        "package_verify_done": (task_dir / "package_verify.tsv").is_file(),
        "latest_keep": _latest_keep(results_tsv),
        "active": _detect_activity(task_dir, active_recent_minutes),
    }
