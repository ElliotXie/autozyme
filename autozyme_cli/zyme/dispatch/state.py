"""State + event-log primitives for `zyme dispatch`.

Two artefacts per dispatch run, both inside `<workspace>/.zyme_dispatch/`:

  * `state.json` — current snapshot of the queue (one entry per task with
    status / timing / pid / event-count). Atomic write via tmp+rename.
    Read by `zyme dispatch status` and re-read by the master after every
    state mutation, so the file is always the truth.

  * `events.ndjson` — append-only log of every state transition + every
    parsed claude stream-json event (one JSON object per line). Written
    by the master and the per-task event parser. Used by
    `zyme dispatch logs`.

Both are best-effort: probe failures don't crash the master. Concurrent
writers aren't expected (master is the only writer in v1, sequential
execution).
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DISPATCH_DIRNAME = ".zyme_dispatch"


def workspace_dispatch_dir(workspace: Path) -> Path:
    return Path(workspace) / DISPATCH_DIRNAME


def state_path(workspace: Path) -> Path:
    return workspace_dispatch_dir(workspace) / "state.json"


def events_path(workspace: Path) -> Path:
    return workspace_dispatch_dir(workspace) / "events.ndjson"


def pid_path(workspace: Path) -> Path:
    return workspace_dispatch_dir(workspace) / "master.pid"


def master_log_path(workspace: Path) -> Path:
    return workspace_dispatch_dir(workspace) / "master.log"


def task_log_dir(workspace: Path) -> Path:
    return workspace_dispatch_dir(workspace) / "logs"


def task_out_path(workspace: Path, task_name: str) -> Path:
    return task_log_dir(workspace) / f"{task_name}.out"


def task_err_path(workspace: Path, task_name: str) -> Path:
    return task_log_dir(workspace) / f"{task_name}.err"


def task_events_path(workspace: Path, task_name: str) -> Path:
    return task_log_dir(workspace) / f"{task_name}.events.ndjson"


def ensure_dispatch_dirs(workspace: Path) -> None:
    workspace_dispatch_dir(workspace).mkdir(parents=True, exist_ok=True)
    task_log_dir(workspace).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp, second precision, with 'Z' suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Atomic JSON state
# ---------------------------------------------------------------------------

def write_state_atomic(path: Path, state: dict) -> None:
    """Write JSON state atomically: write tmp, fsync, rename.

    Renames are atomic on POSIX, so any reader either sees the old or
    new state — never a torn write. The state file is small (~few KB
    for a typical dispatch), so the cost is negligible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# State construction + mutation
# ---------------------------------------------------------------------------

def make_initial_state(
    *,
    workspace: Path,
    tasks: list[dict],
    prompt: str,
    model: str,
    effort: str,
    agent: str = "claude",
    ram_floor_gb: float,
    disk_floor_gb: float | None,
    detached: bool,
    max_rounds: int | None = None,
    force_mode: bool = False,
    reflect: bool = False,
    reflect_prompt: str | None = None,
    reflection_root: str | None = None,
    reflect_category: str | None = None,
) -> dict:
    """Build the initial state document for a fresh dispatch run.

    `tasks` is a list of `{name, task_dir}` dicts in dispatch order.
    Each grows a `status: "pending"` field plus runtime fields as it
    progresses (started_at, finished_at, rc, claude_pid, last_event_at,
    n_events, etc.).
    """
    return {
        "schema_version": 1,
        "started_at": utcnow_iso(),
        "finished_at": None,
        "workspace": str(Path(workspace).resolve()),
        "master_pid": os.getpid(),
        "detached": detached,
        "prompt": prompt,
        "agent": agent,
        "model": model,
        "effort": effort,
        "max_rounds": max_rounds,
        "force_mode": bool(force_mode),
        "reflect": bool(reflect),
        "reflect_prompt": reflect_prompt,
        "reflection_root": reflection_root,
        "reflect_category": reflect_category,
        "ram_floor_gb": ram_floor_gb,
        "disk_floor_gb": disk_floor_gb,
        "current_index": 0,
        "queue": [
            {
                "name": t["name"],
                "task_dir": str(Path(t["task_dir"]).resolve()),
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "duration_s": None,
                "agent_pid": None,
                "claude_pid": None,
                "rc": None,
                "n_events": 0,
                "last_event_at": None,
                "last_event_kind": None,
                "agent_session_id": None,
                "resume_attempts": 0,
                "round": None,
                "results_rounds": 0,
                "pending_results_rounds": 0,
                "last_results_round": None,
                "last_results_status": None,
                "accepts": 0,
                "rejects": 0,
                "stalled": False,
                "reflect_status": None,
                "reflection_dir": None,
                "reflection_prompt_feedback": None,
                "reflection_zyme_cli_feedback": None,
                "reflection_metadata": None,
            }
            for t in tasks
        ],
    }


def task_index(state: dict, name: str) -> int:
    for i, t in enumerate(state["queue"]):
        if t["name"] == name:
            return i
    raise KeyError(f"task not in queue: {name}")


# ---------------------------------------------------------------------------
# Event log (append-only ndjson)
# ---------------------------------------------------------------------------

def append_event(path: Path, event: dict) -> None:
    """Append one event as a single JSON line. Best-effort.

    `event` should always include `ts` (ISO timestamp) and `kind`.
    Callers are free to add arbitrary other fields.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, separators=(",", ":"))
    try:
        with open(path, "a") as f:
            f.write(line + "\n")
    except OSError:
        # event-log writes are best-effort; never crash the master
        pass


def make_event(kind: str, **fields: Any) -> dict:
    """Build an event dict with auto-stamped timestamp."""
    e = {"ts": utcnow_iso(), "kind": kind}
    e.update(fields)
    return e


def tail_events(path: Path, n: int = 50) -> list[dict]:
    """Read the last `n` ndjson events. Returns [] if file missing."""
    if not path.is_file():
        return []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def follow_events(path: Path, *, poll_s: float = 0.5):
    """Generator: yield ndjson events as they're appended. Blocks.

    Caller stops by breaking the loop (Ctrl-C from the CLI). Survives
    the file not existing yet — keeps polling until it appears. Does
    not handle file truncation/rotation (we never rotate).
    """
    pos = 0
    buf = ""
    while True:
        if not path.is_file():
            time.sleep(poll_s)
            continue
        try:
            with open(path) as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            time.sleep(poll_s)
            continue
        if chunk:
            buf += chunk
            while "\n" in buf:
                line, _, buf = buf.partition("\n")
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        else:
            time.sleep(poll_s)


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------

def write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid) + "\n")


def read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    """True if a process with `pid` exists. Uses kill(pid, 0) with no signal."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        # os.kill(pid, 0) is a POSIX existence probe. On Windows, os.kill can
        # terminate the target process for non-console-control signals, so use
        # the process query API instead.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process_query_limited_information = 0x1000
        still_active = 259

        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            wintypes.DWORD(pid),
        )
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # process exists but we don't own it; for our use case treat as alive
        return True
    return True
