"""profile_history/ storage for `zyme profile` runs.

Each persisted run gets a dedicated directory:

    profile_history/<UTC timestamp>_<backend>_<tier>/
      profile.json
      run.log
      <backend raw artifacts>

`--no-archive` uses profile_history/current/ as an overwritten scratch
directory. That preserves the "do not accumulate history" behavior while
still keeping profiler output out of pipeline/.
"""
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def prepare_run_dir(
    task_dir: Path,
    *,
    backend: str,
    tier: str,
    archive: bool = True,
) -> tuple[Path, str]:
    """Create the output directory for one profile run.

    Returns (absolute_path, task-relative path).
    """
    history_dir = task_dir / "profile_history"
    history_dir.mkdir(exist_ok=True)

    if archive:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"{ts}_{_safe_seg(backend)}_{_safe_seg(tier)}"
        run_dir = history_dir / base
        suffix = 2
        while run_dir.exists():
            run_dir = history_dir / f"{base}-{suffix}"
            suffix += 1
    else:
        run_dir = history_dir / "current"
        if run_dir.exists():
            shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_dir.relative_to(task_dir).as_posix()


def write_profile_json(run_dir: Path, profile_data: dict) -> Path:
    """Write the normalized profile JSON into a run directory."""
    out_path = run_dir / "profile.json"
    out_path.write_text(json.dumps(profile_data, indent=2, default=str))
    return out_path


def write_run_log(run_dir: Path, log: str) -> Path:
    """Persist the raw run_task log alongside profiler artifacts."""
    out_path = run_dir / "run.log"
    out_path.write_text(log)
    return out_path


def update_latest_symlink(history_dir: Path, run_dir: Path) -> None:
    """Point `profile_history/latest` at the just-completed run.

    Lets agents (and prompts) reference `profile_history/latest/profile.json`
    without knowing the timestamp. When a task runs both cpu and native
    backends back-to-back, this resolves the "which one is current?"
    ambiguity — the most recent run wins. Silent no-op if symlinking is
    unavailable (e.g. exotic filesystem); the timestamped dir is always
    authoritative.
    """
    link = history_dir / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        # Use a relative target so the link survives directory moves.
        link.symlink_to(run_dir.name)
    except OSError:
        pass


def _safe_seg(s: str) -> str:
    """Sanitize a string for use as a filename segment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s))[:40] or "x"
