"""Master daemon for `zyme dispatch` — sequential task execution with
resource gating, agent stream parsing, stall detection, and ETA.

The big-picture flow:

  1. Caller (cmd_dispatch) builds initial state with queue of tasks.
  2. If --detach, double-fork to daemonize. Parent waits for the master
     pid file then exits. Otherwise stay foreground.
  3. Master loop: for each pending task in queue,
     a. Wait for RAM + disk to clear floors (poll every 60s).
     b. Launch an agent CLI (`claude -p`, `codex exec`, or `cursor-agent -p`)
        with cwd =
        task_dir, stdout piped.
     c. Read stdout line-by-line, parse to normalized events, write to
        per-task events.ndjson, update state.json, optionally print
        one-line summaries (foreground mode).
     d. select-based read with 60s timeout doubles as stall watchdog —
        if no event in --stall-threshold seconds, flip task.stalled.
     e. On EOF + wait(): record rc, mark done/failed, recompute ETA.
  4. After all tasks done (or stop signal), write finished_at and exit.

Signal handling: SIGTERM cleanly stops the running claude subprocess
and writes status="stopped" to its task entry before exiting.
"""
import json
import os
import queue
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from zyme.dispatch.events import (
    apply_event_to_task_state, parse_agent_line,
)
from zyme.dispatch.resources import (
    estimate_disk_need_gb, free_disk_gb, free_ram_gb,
)
from zyme.dispatch.state import (
    append_event, ensure_dispatch_dirs, events_path, follow_events,
    make_event, make_initial_state, master_log_path, pid_alive,
    pid_path, read_pid, read_state, state_path, task_err_path,
    task_events_path, task_out_path, utcnow_iso, write_pid,
    write_state_atomic,
)


# ---------------------------------------------------------------------------
# Agent binary discovery
# ---------------------------------------------------------------------------

def find_agent_binary(agent: str) -> str:
    if agent == "auto":
        _name, path = detect_agent_binary()
        return path
    if agent == "claude":
        return find_claude_binary()
    if agent == "codex":
        return find_codex_binary()
    if agent == "cursor":
        return find_cursor_binary()
    raise RuntimeError(f"unsupported dispatch agent: {agent}")


def detect_agent_binary() -> tuple[str, str]:
    """Auto-detect the first available agent: claude → codex → cursor.

    Returns (agent_name, binary_path).
    """
    for name, finder in [("claude", find_claude_binary), ("codex", find_codex_binary), ("cursor", find_cursor_binary)]:
        try:
            return name, finder()
        except RuntimeError:
            continue
    raise RuntimeError(
        "no agent binary found. Install Claude Code, Cursor, or Codex, "
        "or pass --agent explicitly."
    )


def find_claude_binary() -> str:
    """Locate the `claude` CLI. Tries $ZYME_CLAUDE_BIN, PATH, then
    ~/.local/bin/claude (the install path Claude Code uses).

    The env var is the test hook — point it at a stub script to
    exercise the dispatch pipeline without burning real model calls.
    """
    override = os.environ.get("ZYME_CLAUDE_BIN")
    if override:
        if Path(override).is_file() and os.access(override, os.X_OK):
            return override
        raise RuntimeError(f"ZYME_CLAUDE_BIN={override!r} not executable")
    cand = shutil.which("claude")
    if cand:
        return cand
    fallback = Path.home() / ".local" / "bin" / "claude"
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    raise RuntimeError(
        "claude binary not found in PATH or ~/.local/bin/. "
        "Install Claude Code or add it to PATH."
    )


def find_codex_binary() -> str:
    """Locate the `codex` CLI. Tries $ZYME_CODEX_BIN, then PATH."""
    override = os.environ.get("ZYME_CODEX_BIN")
    if override:
        if Path(override).is_file() and os.access(override, os.X_OK):
            return override
        raise RuntimeError(f"ZYME_CODEX_BIN={override!r} not executable")
    cand = shutil.which("codex")
    if cand:
        return cand
    raise RuntimeError("codex binary not found in PATH. Install Codex CLI or set ZYME_CODEX_BIN.")


def find_cursor_binary() -> str:
    """Locate the Cursor agent CLI.

    Preferred command is `cursor-agent` (Cursor's headless agent). The env var
    lets local installs use a non-standard path without adding it to PATH.
    """
    override = os.environ.get("ZYME_CURSOR_AGENT_BIN")
    if override:
        if Path(override).is_file() and os.access(override, os.X_OK):
            return override
        raise RuntimeError(f"ZYME_CURSOR_AGENT_BIN={override!r} not executable")
    cand = shutil.which("cursor-agent")
    if cand:
        return cand
    fallback = Path.home() / ".local" / "bin" / "cursor-agent"
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    raise RuntimeError(
        "cursor-agent binary not found in PATH or ~/.local/bin. "
        "Install Cursor agent CLI or set ZYME_CURSOR_AGENT_BIN."
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RAM_FLOOR_GB = 10.0
DEFAULT_DISK_FLOOR_FALLBACK_GB = 5.0   # used when task.yaml dataset estimate is 0
DEFAULT_DISK_FLOOR_MIN_GB = 2.0        # never gate on less than 2GB free
DEFAULT_STALL_THRESHOLD_S = 15 * 60
DEFAULT_RESOURCE_POLL_S = 60
DEFAULT_MODEL = "claude-opus-4-7[1m]"
DEFAULT_CURSOR_MODEL = "composer-2.5"
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_EFFORT = "max"
DEFAULT_AGENT = "claude"
DEFAULT_NO_PROGRESS_RESUME_LIMIT = 3
DEFAULT_REFLECT_PROMPT = "prompts/5_reflect.md"


def _default_reflection_root() -> Path:
    """Framework-wide reflection corpus root used by scan and reflect prompts."""
    return Path(__file__).resolve().parents[3] / "reflections"


# ---------------------------------------------------------------------------
# Public entry: start a dispatch run
# ---------------------------------------------------------------------------

def start_dispatch(
    *,
    workspace: Path,
    tasks: list[dict],
    prompt: str,
    agent: str = DEFAULT_AGENT,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    ram_floor_gb: float = DEFAULT_RAM_FLOOR_GB,
    disk_floor_gb: float | None = None,   # None = auto from task.yaml
    detach: bool = False,
    stall_threshold_s: int = DEFAULT_STALL_THRESHOLD_S,
    resource_poll_s: int = DEFAULT_RESOURCE_POLL_S,
    max_rounds: int | None = None,
    force_mode: bool = False,
    reflect: bool = False,
    reflect_prompt: str | None = None,
    reflection_root: Path | None = None,
    reflect_category: str | None = None,
) -> int:
    """Launch the master.

    Returns the master PID. In detach mode, the original caller's
    process is the parent that returns immediately; the daemonized
    grandchild does the work. In foreground mode, this call blocks
    until the master loop exits.
    """
    workspace = Path(workspace).resolve()
    ensure_dispatch_dirs(workspace)

    # Refuse to start if a previous master is still alive.
    prior_pid = read_pid(pid_path(workspace))
    if prior_pid is not None and pid_alive(prior_pid):
        raise RuntimeError(
            f"another zyme dispatch is already running (PID {prior_pid}). "
            f"Run `zyme dispatch-stop` first or wait for it to finish."
        )

    state = make_initial_state(
        workspace=workspace,
        tasks=tasks,
        prompt=prompt,
        agent=agent,
        model=model,
        effort=effort,
        ram_floor_gb=ram_floor_gb,
        disk_floor_gb=disk_floor_gb,
        detached=detach,
        max_rounds=max_rounds,
        force_mode=force_mode,
        reflect=reflect,
        reflect_prompt=reflect_prompt,
        reflection_root=str(Path(reflection_root or _default_reflection_root()).resolve())
        if reflect else None,
        reflect_category=reflect_category,
    )
    state["stall_threshold_s"] = stall_threshold_s
    state["resource_poll_s"] = resource_poll_s
    write_state_atomic(state_path(workspace), state)

    if detach:
        return _spawn_daemon(workspace, state)
    return _run_master(workspace, state, foreground=True)


def resume_dispatch_task(
    *,
    workspace: Path,
    task_name: str | None = None,
    prompt: str | None = None,
    message: str | None = None,
    foreground: bool = True,
    stall_threshold_s: int | None = None,
) -> int | None:
    """Resume one completed/running dispatch task session with one message.

    This is intentionally single-shot: it sends a prompt/message to the agent's
    recorded session id, streams the response into the same dispatch logs, then
    exits. It does not re-enter the optimize max-round loop.
    """
    workspace = Path(workspace).resolve()
    ensure_dispatch_dirs(workspace)
    state = read_state(state_path(workspace))
    if state is None:
        raise RuntimeError(f"no dispatch state at {state_path(workspace)}")
    i = _state_task_index(state, task_name)
    if not message:
        if not prompt:
            raise RuntimeError("resume requires either prompt or message")
        message = f"read and follow {prompt}"
    if stall_threshold_s is not None:
        state["stall_threshold_s"] = stall_threshold_s
    rc = _run_resume_message(
        workspace,
        state,
        i,
        message=message,
        event_kind="task_manual_resume",
        foreground=foreground,
        event_fields={"prompt": prompt, "manual": True},
    )
    state["finished_at"] = utcnow_iso()
    write_state_atomic(state_path(workspace), state)
    append_event(events_path(workspace), make_event(
        "manual_resume_done",
        task=state["queue"][i]["name"],
        rc=rc,
    ))
    return rc


# ---------------------------------------------------------------------------
# Daemonization (double fork)
# ---------------------------------------------------------------------------

def _spawn_daemon(workspace: Path, state: dict) -> int:
    """Fork-fork-exec pattern. Parent returns the daemon PID.

    Standard POSIX double-fork: first fork detaches from caller's
    process group, setsid creates a new session, second fork prevents
    the daemon from re-acquiring a controlling terminal. Grandchild
    runs the master loop with stdio rerouted to master.log.
    """
    # Pre-create the log path so dup2 can succeed in the grandchild.
    log = master_log_path(workspace)
    log.parent.mkdir(parents=True, exist_ok=True)

    pid1 = os.fork()
    if pid1 > 0:
        # Original caller: wait briefly for the grandchild to write its
        # PID file, then return its PID to the user.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            grandchild_pid = read_pid(pid_path(workspace))
            if grandchild_pid is not None and pid_alive(grandchild_pid):
                return grandchild_pid
            time.sleep(0.1)
        # Daemon didn't come up; surface a clear error.
        raise RuntimeError(
            f"daemon failed to start within 10s; check {log} for details"
        )

    # First child
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # Grandchild — the daemon. Reset signal mask to defaults.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.chdir("/")
    fd = os.open(str(log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    if fd > 2:
        os.close(fd)

    state["master_pid"] = os.getpid()
    write_state_atomic(state_path(workspace), state)
    write_pid(pid_path(workspace), os.getpid())

    try:
        _run_master(workspace, state, foreground=False)
    finally:
        # Daemon exiting: clear pid file so a future dispatch can start.
        try:
            pid_path(workspace).unlink(missing_ok=True)
        except OSError:
            pass
    os._exit(0)


# ---------------------------------------------------------------------------
# Master loop
# ---------------------------------------------------------------------------

# Signal handler hook: set by _run_master; SIGTERM/SIGINT trip this so
# the read loop can stop the running claude subprocess cleanly.
_SHOULD_STOP = False


def _install_stop_handlers():
    def handler(signum, frame):
        global _SHOULD_STOP
        _SHOULD_STOP = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _run_master(workspace: Path, state: dict, *, foreground: bool) -> int:
    global _SHOULD_STOP
    _SHOULD_STOP = False
    _install_stop_handlers()

    write_pid(pid_path(workspace), os.getpid())

    sp = state_path(workspace)
    ep = events_path(workspace)
    append_event(ep, make_event(
        "dispatch_start",
        n_tasks=len(state["queue"]),
        tasks=[t["name"] for t in state["queue"]],
        prompt=state["prompt"],
        agent=state.get("agent", DEFAULT_AGENT),
        model=state["model"],
        effort=state["effort"],
        max_rounds=state.get("max_rounds"),
        force_mode=state.get("force_mode", False),
        reflect=state.get("reflect", False),
        reflect_prompt=state.get("reflect_prompt"),
        reflection_root=state.get("reflection_root"),
        ram_floor_gb=state["ram_floor_gb"],
        foreground=foreground,
    ))

    completed_durations: list[float] = []

    for i, task in enumerate(state["queue"]):
        if _SHOULD_STOP:
            break
        if task["status"] != "pending":
            continue
        state["current_index"] = i
        write_state_atomic(sp, state)

        # ---- resource gate -------------------------------------------------
        disk_floor = _resolve_disk_floor(state, task)
        _wait_for_resources(
            workspace, state, i,
            disk_floor_gb=disk_floor,
            poll_s=state.get("resource_poll_s", DEFAULT_RESOURCE_POLL_S),
        )
        if _SHOULD_STOP:
            task["status"] = "stopped"
            write_state_atomic(sp, state)
            break

        # ---- launch + monitor ---------------------------------------------
        rc = _run_one_task(workspace, state, i, foreground=foreground)
        task = state["queue"][i]
        if rc == 0 and state.get("reflect") and not _SHOULD_STOP:
            _run_reflect_task(workspace, state, i, foreground=foreground)
            task = state["queue"][i]
        if rc == 0:
            task["status"] = "done"
        elif rc is None:
            task["status"] = "stopped"
        else:
            task["status"] = "failed"
        task["rc"] = rc

        # ---- ETA recomputation --------------------------------------------
        if task.get("duration_s") is not None and rc == 0:
            completed_durations.append(task["duration_s"])
        state["eta_s"] = _project_eta(state, completed_durations)

        write_state_atomic(sp, state)
        append_event(ep, make_event(
            "task_finished",
            task=task["name"],
            status=task["status"],
            rc=rc,
            duration_s=task.get("duration_s"),
        ))

    state["finished_at"] = utcnow_iso()
    state["current_index"] = None
    write_state_atomic(sp, state)
    append_event(ep, make_event(
        "dispatch_done",
        completed=sum(1 for t in state["queue"] if t["status"] == "done"),
        failed=sum(1 for t in state["queue"] if t["status"] == "failed"),
        stopped=sum(1 for t in state["queue"] if t["status"] == "stopped"),
        pending=sum(1 for t in state["queue"] if t["status"] == "pending"),
    ))
    return 0


# ---------------------------------------------------------------------------
# Resource gate
# ---------------------------------------------------------------------------

def _resolve_disk_floor(state: dict, task: dict) -> float:
    """Per-task disk floor: explicit override → estimate ×2 → fallback."""
    if state.get("disk_floor_gb") is not None:
        return float(state["disk_floor_gb"])
    est = estimate_disk_need_gb(Path(task["task_dir"]))
    if est <= 0:
        return DEFAULT_DISK_FLOOR_FALLBACK_GB
    return max(est, DEFAULT_DISK_FLOOR_MIN_GB)


def _wait_for_resources(
    workspace: Path, state: dict, i: int, *,
    disk_floor_gb: float, poll_s: int,
) -> None:
    """Spin until both RAM and disk clear floors. Updates state to
    `resource_wait` between attempts so `zyme dispatch status` shows
    the reason for the pause.
    """
    sp = state_path(workspace)
    ep = events_path(workspace)
    task = state["queue"][i]

    notified_wait = False
    while not _SHOULD_STOP:
        ram = free_ram_gb()
        disk = free_disk_gb(Path(task["task_dir"]))
        ram_ok = ram >= state["ram_floor_gb"]
        disk_ok = disk >= disk_floor_gb
        snap = {
            "ram_gb": round(ram, 2),
            "disk_gb": round(disk, 2),
            "ram_floor_gb": state["ram_floor_gb"],
            "disk_floor_gb": disk_floor_gb,
        }
        append_event(ep, make_event(
            "resource_probe",
            task=task["name"],
            passed=(ram_ok and disk_ok),
            **snap,
        ))
        if ram_ok and disk_ok:
            return
        if not notified_wait:
            task["status"] = "resource_wait"
            task["resource_snapshot"] = snap
            write_state_atomic(sp, state)
            notified_wait = True
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# Per-task launch + stream parse
# ---------------------------------------------------------------------------

def _run_one_task(
    workspace: Path, state: dict, i: int, *, foreground: bool,
) -> int | None:
    """Launch an agent, parse stream-json, update state, and optionally resume.

    max_rounds always stops a live agent once enough completed decision rows
    appear in results.tsv. force_mode additionally treats clean early exits as
    incomplete and resumes the same agent session until the target is reached.
    """
    task = state["queue"][i]
    sp = state_path(workspace)
    ep = events_path(workspace)
    task_ev = task_events_path(workspace, task["name"])
    out_p = task_out_path(workspace, task["name"])
    err_p = task_err_path(workspace, task["name"])

    task["status"] = "running"
    task["started_at"] = utcnow_iso()
    _refresh_results_round_state(task)
    started_mono = time.monotonic()
    write_state_atomic(sp, state)

    agent = state.get("agent", DEFAULT_AGENT)
    stall_threshold = state.get("stall_threshold_s", DEFAULT_STALL_THRESHOLD_S)
    max_rounds = state.get("max_rounds")
    force_mode = bool(state.get("force_mode", False))
    if _max_rounds_reached(task, max_rounds):
        task["stop_reason"] = "max_rounds"
        task["finished_at"] = utcnow_iso()
        task["duration_s"] = 0
        write_state_atomic(sp, state)
        append_event(ep, make_event(
            "task_max_rounds_already_reached",
            task=task["name"],
            max_rounds=max_rounds,
            results_rounds=task.get("results_rounds"),
            last_results_round=task.get("last_results_round"),
            last_results_status=task.get("last_results_status"),
        ))
        return 0
    launch_index = 0
    no_progress_exits = 0
    previous_completed = task.get("results_rounds") or 0
    max_resume_attempts = max(int(max_rounds or 0), 1)

    err_f = open(err_p, "a", buffering=1)
    # Append so restarting a bench dispatch preserves prior token telemetry.
    out_f = open(out_p, "a", buffering=1)
    try:
        while not _SHOULD_STOP:
            is_resume = launch_index > 0
            if is_resume:
                if launch_index > max_resume_attempts:
                    task["stop_reason"] = "resume_attempt_limit"
                    append_event(ep, make_event(
                        "task_resume_attempt_limit",
                        task=task["name"],
                        max_resume_attempts=max_resume_attempts,
                        results_rounds=task.get("results_rounds"),
                    ))
                    return 1
                task["resume_attempts"] = launch_index

            resume_session_id = task.get("agent_session_id") if is_resume else None
            if is_resume and not resume_session_id:
                task["stop_reason"] = "resume_session_missing"
                append_event(ep, make_event(
                    "task_resume_session_missing",
                    task=task["name"],
                    results_rounds=task.get("results_rounds"),
                    max_rounds=max_rounds,
                ))
                return 1

            prompt_text = (
                _resume_prompt(state["prompt"], max_rounds, task)
                if is_resume else None
            )
            cmd = _build_agent_cmd(
                agent,
                state["prompt"],
                state.get("model"),
                state.get("effort"),
                resume_session_id=resume_session_id,
                message=prompt_text,
            )
            append_event(ep, make_event(
                "task_resume" if is_resume else "task_start",
                task=task["name"],
                cmd=cmd,
                cwd=task["task_dir"],
                resume_attempt=launch_index if is_resume else 0,
                session_id=resume_session_id,
                results_rounds=task.get("results_rounds"),
                max_rounds=max_rounds,
            ))
            if foreground and is_resume:
                print(
                    f"[resume] {task['name']}: session={resume_session_id} "
                    f"rounds={task.get('results_rounds')}/{max_rounds}",
                    file=sys.stderr,
                    flush=True,
                )

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=task["task_dir"],
                    stdout=subprocess.PIPE,
                    stderr=err_f,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except OSError as e:
                err_f.write(f"failed to launch {agent}: {e}\n")
                return -1

            task["agent_pid"] = proc.pid
            task["claude_pid"] = proc.pid if agent == "claude" else None
            write_state_atomic(sp, state)

            max_rounds_hit = False
            model_route_mismatch = False

            for kind, payload in _read_stream_with_stalls(proc, stall_threshold):
                if _SHOULD_STOP:
                    _terminate(proc)
                    break
                if kind == "line":
                    out_f.write(payload + "\n")
                    events = parse_agent_line(payload, agent=agent)
                    for ev in events:
                        append_event(task_ev, ev)
                        apply_event_to_task_state(task, ev)
                        if ev.get("kind") == "agent_model":
                            task["actual_model"] = ev.get("model")
                            mismatch = _agent_model_mismatch(agent, state.get("model"), ev.get("model"))
                            if mismatch:
                                task["model_route_warning"] = mismatch
                                task["agent_error"] = True
                                model_route_mismatch = True
                                append_event(ep, make_event(
                                    "task_model_route_warning",
                                    task=task["name"],
                                    agent=agent,
                                    requested_model=state.get("model"),
                                    actual_model=ev.get("model"),
                                    message=mismatch,
                                ))
                                _terminate(proc)
                                break
                        if ev.get("kind") == "claude_done" and ev.get("is_error"):
                            task["agent_error"] = True
                    if events:
                        write_state_atomic(sp, state)
                        if foreground:
                            for ev in events:
                                _print_summary_line(task["name"], ev)
                    if _max_rounds_reached(task, max_rounds):
                        max_rounds_hit = True
                        append_event(ep, make_event(
                            "task_max_rounds_reached",
                            task=task["name"],
                            max_rounds=max_rounds,
                            results_rounds=task.get("results_rounds"),
                            last_results_round=task.get("last_results_round"),
                            last_results_status=task.get("last_results_status"),
                        ))
                        if foreground:
                            print(
                                f"[max-rounds] {task['name']}: reached {task.get('results_rounds')} "
                                f"completed results.tsv round(s); stopping agent",
                                file=sys.stderr,
                                flush=True,
                            )
                        _terminate(proc)
                        break
                elif kind == "stall":
                    task["stalled"] = True
                    _refresh_results_round_state(task)
                    write_state_atomic(sp, state)
                    append_event(ep, make_event(
                        "task_stall",
                        task=task["name"],
                        seconds_since_last_event=int(payload),
                        last_event_at=task.get("last_event_at"),
                    ))
                    if _max_rounds_reached(task, max_rounds):
                        max_rounds_hit = True
                        append_event(ep, make_event(
                            "task_max_rounds_reached",
                            task=task["name"],
                            max_rounds=max_rounds,
                            results_rounds=task.get("results_rounds"),
                            last_results_round=task.get("last_results_round"),
                            last_results_status=task.get("last_results_status"),
                        ))
                        _terminate(proc)
                        break
                elif kind == "exit":
                    break

            rc = proc.wait()
            task["agent_pid"] = None
            task["claude_pid"] = None
            _refresh_results_round_state(task)
            write_state_atomic(sp, state)

            if _SHOULD_STOP:
                return None
            if model_route_mismatch:
                task["stop_reason"] = "model_route_mismatch"
                return 1
            if max_rounds_hit or _max_rounds_reached(task, max_rounds):
                task["stop_reason"] = "max_rounds"
                return 0
            if task.get("agent_error") and rc == 0:
                return 1
            if rc != 0:
                return rc
            if not max_rounds or not force_mode:
                return rc

            completed = task.get("results_rounds") or 0
            if completed <= previous_completed:
                no_progress_exits += 1
            else:
                no_progress_exits = 0
            previous_completed = completed
            if no_progress_exits >= DEFAULT_NO_PROGRESS_RESUME_LIMIT:
                task["stop_reason"] = "resume_no_progress"
                append_event(ep, make_event(
                    "task_resume_no_progress",
                    task=task["name"],
                    results_rounds=completed,
                    max_rounds=max_rounds,
                    no_progress_exits=no_progress_exits,
                ))
                return 1

            append_event(ep, make_event(
                "task_agent_exited_before_max_rounds",
                task=task["name"],
                rc=rc,
                results_rounds=completed,
                max_rounds=max_rounds,
                session_id=task.get("agent_session_id"),
                next_resume_attempt=launch_index + 1,
            ))
            launch_index += 1
    finally:
        out_f.flush(); out_f.close()
        err_f.flush(); err_f.close()
        task["finished_at"] = utcnow_iso()
        task["duration_s"] = int(time.monotonic() - started_mono)
        _refresh_results_round_state(task)
        write_state_atomic(sp, state)

    return None


def _state_task_index(state: dict, task_name: str | None) -> int:
    queue = state.get("queue") or []
    if not queue:
        raise RuntimeError("dispatch state has no tasks")
    if task_name:
        matches = [
            i for i, t in enumerate(queue)
            if t.get("name") == task_name or Path(str(t.get("task_dir", ""))).name == task_name
        ]
        if not matches:
            known = ", ".join(str(t.get("name")) for t in queue)
            raise RuntimeError(f"task {task_name!r} not found in dispatch state. Known: {known}")
        return matches[0]
    if len(queue) == 1:
        return 0
    current = state.get("current_index")
    if isinstance(current, int) and 0 <= current < len(queue):
        return current
    raise RuntimeError("task name is required when dispatch state has multiple tasks")


def _run_reflect_task(
    workspace: Path, state: dict, i: int, *, foreground: bool,
) -> int | None:
    task = state["queue"][i]
    ep = events_path(workspace)
    prompt = state.get("reflect_prompt") or DEFAULT_REFLECT_PROMPT
    root = Path(state.get("reflection_root") or _default_reflection_root())
    category = state.get("reflect_category") or "iteration"
    paths = _reflection_output_paths(root, category, task["name"])

    task["reflect_status"] = "running"
    task["reflection_dir"] = str(root)
    task["reflection_prompt_feedback"] = str(paths["prompt_feedback"])
    task["reflection_zyme_cli_feedback"] = str(paths["zyme_cli_feedback"])
    task["reflection_metadata"] = str(paths["metadata"])
    write_state_atomic(state_path(workspace), state)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_reflection_metadata(
        paths["metadata"], state, task, prompt, category,
        prompt_feedback_path=paths["prompt_feedback"],
        zyme_cli_feedback_path=paths["zyme_cli_feedback"],
        reflection_root=root,
    )

    if not task.get("agent_session_id"):
        task["reflect_status"] = "skipped"
        task["reflect_skip_reason"] = "agent_session_missing"
        write_state_atomic(state_path(workspace), state)
        append_event(ep, make_event(
            "task_reflect_skipped",
            task=task["name"],
            reason="agent_session_missing",
            reflection_root=str(root),
            prompt_feedback=str(paths["prompt_feedback"]),
            zyme_cli_feedback=str(paths["zyme_cli_feedback"]),
        ))
        return None

    message = _reflect_resume_message(
        prompt=prompt,
        state=state,
        task=task,
        prompt_feedback_path=paths["prompt_feedback"],
        zyme_cli_feedback_path=paths["zyme_cli_feedback"],
        metadata_path=paths["metadata"],
        category=category,
    )
    rc = _run_resume_message(
        workspace,
        state,
        i,
        message=message,
        event_kind="task_reflect_start",
        foreground=foreground,
        event_fields={
            "prompt": prompt,
            "reflection_root": str(root),
            "prompt_feedback": str(paths["prompt_feedback"]),
            "zyme_cli_feedback": str(paths["zyme_cli_feedback"]),
            "category": category,
        },
    )
    task = state["queue"][i]
    task["reflect_rc"] = rc
    task["reflect_status"] = "done" if rc == 0 else ("stopped" if rc is None else "failed")
    write_state_atomic(state_path(workspace), state)
    append_event(ep, make_event(
        "task_reflect_finished",
        task=task["name"],
        rc=rc,
        status=task["reflect_status"],
        reflection_root=str(root),
        prompt_feedback=str(paths["prompt_feedback"]),
        zyme_cli_feedback=str(paths["zyme_cli_feedback"]),
    ))
    return rc


def _reflection_output_paths(root: Path, category: str, task_name: str) -> dict[str, Path]:
    stem = _reflection_file_stem(category, task_name)
    return {
        "prompt_feedback": _available_reflection_path(
            root / "prompt_reflect_feedback", stem, ".md",
        ),
        "zyme_cli_feedback": _available_reflection_path(
            root / "zyme_cli_feedback", stem, ".md",
        ),
        "metadata": _available_reflection_path(root / "metadata", stem, ".yaml"),
    }


def _reflection_file_stem(category: str, task_name: str) -> str:
    raw = f"{category}_{task_name}"
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw)


def _available_reflection_path(directory: Path, stem: str, suffix: str) -> Path:
    base = directory / f"{stem}{suffix}"
    if not base.exists():
        return base
    date = time.strftime("%Y-%m-%d")
    candidate = directory / f"{stem}__{date}{suffix}"
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        candidate = directory / f"{stem}__{date}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _write_reflection_metadata(
    metadata_path: Path,
    state: dict,
    task: dict,
    prompt: str,
    category: str,
    *,
    prompt_feedback_path: Path,
    zyme_cli_feedback_path: Path,
    reflection_root: Path,
) -> None:
    metadata = {
        "task": task.get("name"),
        "task_dir": task.get("task_dir"),
        "category": category,
        "reflection_root": str(reflection_root),
        "prompt_feedback": str(prompt_feedback_path),
        "zyme_cli_feedback": str(zyme_cli_feedback_path),
        "agent": state.get("agent"),
        "model": state.get("model"),
        "actual_model": task.get("actual_model"),
        "effort": state.get("effort"),
        "optimize_prompt": state.get("prompt"),
        "reflect_prompt": prompt,
        "max_rounds": state.get("max_rounds"),
        "force_mode": state.get("force_mode"),
        "results_rounds": task.get("results_rounds"),
        "last_results_round": task.get("last_results_round"),
        "last_results_status": task.get("last_results_status"),
        "agent_session_id": task.get("agent_session_id"),
        "created_at": utcnow_iso(),
    }
    lines = []
    for key, value in metadata.items():
        lines.append(f"{key}: {_yaml_scalar(value)}")
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _reflect_resume_message(
    *,
    prompt: str,
    state: dict,
    task: dict,
    prompt_feedback_path: Path,
    zyme_cli_feedback_path: Path,
    metadata_path: Path,
    category: str,
) -> str:
    return (
        f"Read and follow {prompt} for a post-run reflection, with these overrides. "
        f"This is a PromptLab experiment reflection for task `{task['name']}` after "
        f"dispatch prompt `{state.get('prompt')}`. Category: `{category}`. "
        f"Write prompt feedback exactly to `{prompt_feedback_path}`. "
        f"Write zyme CLI feedback exactly to `{zyme_cli_feedback_path}`. "
        f"`{metadata_path}` already records run metadata. "
        "These exact paths override any generic path examples in the prompt. "
        "Do not commit or push. "
        "Skip a feedback file if there is genuinely nothing for that channel. "
        "Use this session context plus results.tsv, memory/, profile_history/, "
        "and .zyme_dispatch/ evidence when making concrete points."
    )


def _run_resume_message(
    workspace: Path,
    state: dict,
    i: int,
    *,
    message: str,
    event_kind: str,
    foreground: bool,
    event_fields: dict | None = None,
) -> int | None:
    task = state["queue"][i]
    session_id = task.get("agent_session_id")
    if not session_id:
        raise RuntimeError(f"task {task.get('name')} has no agent_session_id to resume")

    sp = state_path(workspace)
    ep = events_path(workspace)
    task_ev = task_events_path(workspace, task["name"])
    out_p = task_out_path(workspace, task["name"])
    err_p = task_err_path(workspace, task["name"])
    agent = state.get("agent", DEFAULT_AGENT)
    stall_threshold = state.get("stall_threshold_s", DEFAULT_STALL_THRESHOLD_S)

    cmd = _build_agent_cmd(
        agent,
        state.get("prompt") or "",
        state.get("model"),
        state.get("effort"),
        resume_session_id=session_id,
        message=message,
    )
    fields = dict(event_fields or {})
    append_event(ep, make_event(
        event_kind,
        task=task["name"],
        cmd=cmd,
        cwd=task["task_dir"],
        session_id=session_id,
        results_rounds=task.get("results_rounds"),
        **fields,
    ))
    if foreground:
        print(
            f"[resume] {task['name']}: {event_kind} session={session_id}",
            file=sys.stderr,
            flush=True,
        )

    with open(err_p, "a", buffering=1) as err_f, open(out_p, "a", buffering=1) as out_f:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=task["task_dir"],
                stdout=subprocess.PIPE,
                stderr=err_f,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            err_f.write(f"failed to launch {agent}: {e}\n")
            return -1

        task["agent_pid"] = proc.pid
        task["claude_pid"] = proc.pid if agent == "claude" else None
        write_state_atomic(sp, state)

        for kind, payload in _read_stream_with_stalls(proc, stall_threshold):
            if _SHOULD_STOP:
                _terminate(proc)
                break
            if kind == "line":
                out_f.write(payload + "\n")
                events = parse_agent_line(payload, agent=agent)
                for ev in events:
                    append_event(task_ev, ev)
                    apply_event_to_task_state(task, ev)
                    if ev.get("kind") == "claude_done" and ev.get("is_error"):
                        task["agent_error"] = True
                if events:
                    write_state_atomic(sp, state)
                    if foreground:
                        for ev in events:
                            _print_summary_line(task["name"], ev)
            elif kind == "stall":
                task["stalled"] = True
                write_state_atomic(sp, state)
                append_event(ep, make_event(
                    f"{event_kind}_stall",
                    task=task["name"],
                    seconds_since_last_event=int(payload),
                    last_event_at=task.get("last_event_at"),
                ))
            elif kind == "exit":
                break

        rc = proc.wait()
        task["agent_pid"] = None
        task["claude_pid"] = None
        _refresh_results_round_state(task)
        write_state_atomic(sp, state)
        if _SHOULD_STOP:
            return None
        return rc


def _build_agent_cmd(
    agent: str,
    prompt: str,
    model: str | None,
    effort: str | None,
    *,
    resume_session_id: str | None = None,
    message: str | None = None,
) -> list[str]:
    prompt_text = message or f"read and follow {prompt}"
    if agent == "claude":
        binary = find_claude_binary()
        cmd = [binary, "-p"]
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        cmd.append(prompt_text)
        if model:
            cmd.extend(["--model", model])
        if effort:
            cmd.extend(["--effort", effort])
        cmd.extend([
            "--dangerously-skip-permissions",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
        ])
        return cmd
    if agent == "codex":
        binary = find_codex_binary()
        cmd = [
            binary,
            "exec",
        ]
        if resume_session_id:
            cmd.append("resume")
        cmd.extend([
            "--dangerously-bypass-approvals-and-sandbox",
            "--json",
        ])
        codex_effort = _codex_reasoning_effort(effort)
        if codex_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{codex_effort}"'])
        if model:
            cmd.extend(["--model", model])
        if resume_session_id:
            cmd.append(resume_session_id)
        cmd.append(prompt_text)
        return cmd
    if agent == "cursor":
        binary = find_cursor_binary()
        cmd = [
            binary,
            "-p",
            "--output-format", "stream-json",
            "--force",
            "--trust",
            "--model", model or DEFAULT_CURSOR_MODEL,
        ]
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        cmd.append(prompt_text)
        return cmd
    raise RuntimeError(f"unsupported dispatch agent: {agent}")


def _resume_prompt(prompt: str, max_rounds: int | None, task: dict) -> str:
    completed = task.get("results_rounds") or 0
    last_round = task.get("last_results_round")
    last_status = task.get("last_results_status")
    encouragement = _resume_encouragement(completed)
    return (
        f"{encouragement} Read results.tsv, then continue following {prompt} "
        f"from the current state ({completed} completed round(s), last={last_round}:{last_status}). "
        "Stay in this main session: no subagents, task delegation, batch runners, or unattended loops. "
        "Keep running the measured optimize loop: make one thoughtful change, run `zyme run`, "
        "choose exactly one `zyme accept` or `zyme reject`, then immediately start the next round. "
        "Stop only for a real blocker or when the dispatcher stops you."
    )


def _resume_encouragement(completed: int) -> str:
    if completed < 2:
        return "Continue."
    if completed < 5:
        return "Continue, you can do this."
    if completed < 10:
        return "Continue with care; the small measured wins matter."
    if completed < 20:
        return "Continue; each careful benchmark round makes the experiment stronger."
    if completed < 35:
        return "Continue, you are advancing science one measured round at a time."
    return "Continue; this is deep work now, and the next careful insight still matters."


def _codex_reasoning_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    if effort == "max":
        return "high"
    if effort in ("low", "medium", "high", "xhigh"):
        return effort
    return None


def _agent_model_mismatch(agent: str, requested: str | None, actual: str | None) -> str | None:
    if not requested or not actual:
        return None
    if agent == "cursor":
        # Cursor currently reports Composer 2 as "Composer 2 Fast" for some
        # accounts/routes. Treat that as informational; the dispatcher should
        # not kill an otherwise valid benchmark run over this naming variance.
        return None
    return None


def _terminate(proc: subprocess.Popen) -> None:
    """Try graceful TERM, then KILL after 5s."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        proc.kill()
    except ProcessLookupError:
        return


# ---------------------------------------------------------------------------
# Stream reader with stall + exit detection
# ---------------------------------------------------------------------------

def _read_stream_with_stalls(proc: subprocess.Popen, stall_threshold_s: int):
    """Yield ("line", str) | ("stall", int_seconds) | ("exit", rc) tuples.

    Uses select() with a 60s timeout so we can interleave stall checks
    and process-alive checks with line reads. Buffers partial reads and
    splits on newlines to emit one event per JSON line.
    """
    if proc.stdout is None:
        yield ("exit", proc.wait())
        return
    if os.name == "nt":
        yield from _read_stream_with_stalls_threaded(proc, stall_threshold_s)
        return
    fd = proc.stdout.fileno()
    buf = ""
    last_data_mono = time.monotonic()
    stall_fired = False

    while True:
        rlist, _, _ = select.select([fd], [], [], 60.0)
        now = time.monotonic()
        if rlist:
            try:
                chunk = os.read(fd, 64 * 1024).decode("utf-8", errors="replace")
            except OSError:
                chunk = ""
            if not chunk:
                # EOF
                if buf.strip():
                    yield ("line", buf.strip())
                yield ("exit", proc.wait())
                return
            buf += chunk
            while "\n" in buf:
                line, _, buf = buf.partition("\n")
                if line.strip():
                    yield ("line", line)
                    last_data_mono = now
                    stall_fired = False
        else:
            elapsed = now - last_data_mono
            if elapsed >= stall_threshold_s and not stall_fired:
                yield ("stall", elapsed)
                stall_fired = True
            rc = proc.poll()
            if rc is not None:
                # Drain any remaining bytes
                try:
                    rest = proc.stdout.read()
                except OSError:
                    rest = ""
                if rest:
                    buf += rest
                    while "\n" in buf:
                        line, _, buf = buf.partition("\n")
                        if line.strip():
                            yield ("line", line)
                if buf.strip():
                    yield ("line", buf.strip())
                yield ("exit", rc)
                return


def _read_stream_with_stalls_threaded(
    proc: subprocess.Popen,
    stall_threshold_s: int,
):
    """Windows pipe reader: select() only works on sockets there."""
    if proc.stdout is None:
        yield ("exit", proc.wait())
        return
    events: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def _reader() -> None:
        try:
            for raw in proc.stdout:
                events.put(("line", raw.rstrip("\n")))
        finally:
            events.put(("eof", None))

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    last_data_mono = time.monotonic()
    stall_fired = False
    timeout = max(0.1, min(60.0, float(stall_threshold_s)))

    while True:
        try:
            kind, payload = events.get(timeout=timeout)
        except queue.Empty:
            now = time.monotonic()
            elapsed = now - last_data_mono
            if elapsed >= stall_threshold_s and not stall_fired:
                yield ("stall", elapsed)
                stall_fired = True
            continue

        if kind == "line":
            if payload and payload.strip():
                yield ("line", payload)
                last_data_mono = time.monotonic()
                stall_fired = False
            continue

        if kind == "eof":
            yield ("exit", proc.wait())
            return


# ---------------------------------------------------------------------------
# results.tsv round monitor
# ---------------------------------------------------------------------------

_TERMINAL_DECISION_STATUSES = {"keep", "discard", "crash", "rollback", "oom"}


def _refresh_results_round_state(task: dict) -> dict:
    snap = _results_round_snapshot(Path(task["task_dir"]))
    task["results_rounds"] = snap["completed_rounds"]
    if snap["completed_rounds"]:
        task["round"] = max(task.get("round") or 0, snap["completed_rounds"])
    task["last_results_round"] = snap["last_round"]
    task["last_results_status"] = snap["last_status"]
    task["pending_results_rounds"] = snap["pending_rounds"]
    return snap


def _max_rounds_reached(task: dict, max_rounds: int | None) -> bool:
    if not max_rounds or max_rounds <= 0:
        return False
    snap = _refresh_results_round_state(task)
    return snap["completed_rounds"] >= max_rounds


def _results_round_snapshot(task_dir: Path, *, phase: str = "optimize") -> dict:
    """Summarize decision rounds from task_dir/results.tsv.

    Counts terminal decision rows only. A raw integer round with
    status=pending means `zyme run` finished but `zyme accept/reject` has not
    resolved it yet; killing there leaves an awkward half-round, so the
    max-rounds guard waits for keep/discard/crash/rollback.
    """
    results_tsv = task_dir / "results.tsv"
    out = {
        "completed_rounds": 0,
        "pending_rounds": 0,
        "last_round": None,
        "last_status": None,
    }
    if not results_tsv.exists():
        return out
    try:
        lines = results_tsv.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    if len(lines) < 2:
        return out

    header = lines[0].split("\t")
    try:
        round_idx = header.index("round")
    except ValueError:
        round_idx = 0
    try:
        status_idx = header.index("status")
    except ValueError:
        status_idx = 6
    try:
        phase_idx = header.index("phase")
    except ValueError:
        phase_idx = None

    completed = set()
    pending = set()
    last_round = None
    last_status = None

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if round_idx >= len(parts):
            continue
        round_label = parts[round_idx].strip()
        try:
            round_num = int(round_label)
        except ValueError:
            continue
        if round_num < 1:
            continue
        if phase != "all":
            row_phase = ""
            if phase_idx is not None and phase_idx < len(parts):
                row_phase = parts[phase_idx].strip()
            if not row_phase:
                row_phase = "optimize"
            if row_phase != phase:
                continue
        status = parts[status_idx].strip() if status_idx < len(parts) else ""
        last_round = round_num
        last_status = status
        if status in _TERMINAL_DECISION_STATUSES:
            completed.add(round_num)
        elif status == "pending" or not status:
            pending.add(round_num)

    out["completed_rounds"] = len(completed)
    out["pending_rounds"] = len(pending - completed)
    out["last_round"] = last_round
    out["last_status"] = last_status
    return out


# ---------------------------------------------------------------------------
# ETA
# ---------------------------------------------------------------------------

def _project_eta(state: dict, completed_durations: list[float]) -> int | None:
    """Project remaining seconds = (pending count) × mean(completed).

    Returns None when no completed durations yet (we have no signal to
    project from). Mean is conservative — if the next task is harder
    than average it'll overshoot, but that's preferable to missing a
    spike.
    """
    if not completed_durations:
        return None
    pending = sum(1 for t in state["queue"] if t["status"] == "pending")
    if pending == 0:
        return 0
    avg = sum(completed_durations) / len(completed_durations)
    return int(pending * avg)


# ---------------------------------------------------------------------------
# Foreground pretty-printer
# ---------------------------------------------------------------------------

def _print_summary_line(task_name: str, event: dict) -> None:
    """One short stderr line per normalized event, foreground only."""
    ts = event.get("ts", "")[11:19]   # HH:MM:SS
    kind = event.get("kind")
    if kind == "claude_text":
        s = event.get("snippet") or ""
        line = f"[{ts}] {task_name}: ✎ {s}"
    elif kind == "claude_tool_use":
        s = event.get("summary") or ""
        line = f"[{ts}] {task_name}: ▸ {event.get('tool')}: {s}"
    elif kind == "claude_tool_result":
        ok = "ok" if event.get("ok") else "ERR"
        s = event.get("summary") or ""
        line = f"[{ts}] {task_name}: ◂ {ok}: {s}"
    elif kind == "claude_done":
        line = (
            f"[{ts}] {task_name}: ✔ done (turns={event.get('num_turns')}, "
            f"cost=${event.get('total_cost_usd')})"
        )
    elif kind in ("codex_event", "cursor_event"):
        agent = kind.split("_", 1)[0]
        line = f"[{ts}] {task_name}: · {agent} {event.get('event','?')}: {event.get('summary','')}"
    elif kind in ("codex_text", "cursor_text"):
        line = f"[{ts}] {task_name}: ✎ {event.get('snippet','')}"
    elif kind in ("zyme_run", "zyme_accept", "zyme_reject"):
        verb = kind[len("zyme_"):]
        text = event.get("hypothesis") or event.get("description") or ""
        line = f"[{ts}] {task_name}: ★ zyme {verb}: {text}"
    else:
        line = f"[{ts}] {task_name}: · {kind}"
    print(line[:300], file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Public: status / stop / logs
# ---------------------------------------------------------------------------

def render_status(workspace: Path) -> str:
    """One-screen status snapshot for the active or last dispatch."""
    state = read_state(state_path(workspace))
    if state is None:
        return f"no dispatch state at {state_path(workspace)}"

    pid = state.get("master_pid")
    alive = pid is not None and pid_alive(pid)
    started = state.get("started_at") or "?"
    finished = state.get("finished_at")
    eta_s = state.get("eta_s")

    lines: list[str] = []
    lines.append(f"workspace : {state.get('workspace')}")
    lines.append(f"prompt    : {state.get('prompt')}")
    lines.append(f"agent     : {state.get('agent', DEFAULT_AGENT)}")
    if state.get("max_rounds"):
        lines.append(f"max rounds: {state.get('max_rounds')}")
    if state.get("force_mode"):
        lines.append("force mode: on (resume early exits until max rounds)")
    if state.get("reflect"):
        lines.append(f"reflect   : on ({state.get('reflect_prompt') or DEFAULT_REFLECT_PROMPT})")
    lines.append(
        f"master    : pid={pid} {'alive' if alive else 'gone'} "
        f"(started {started}{', finished ' + finished if finished else ''})"
    )
    if eta_s is not None and not finished:
        lines.append(f"eta       : ~{_fmt_duration(eta_s)} remaining")

    counts = _status_counts(state)
    lines.append(
        f"progress  : {counts['done']}/{len(state['queue'])} done, "
        f"{counts['running']} running, {counts['resource_wait']} waiting, "
        f"{counts['failed']} failed, {counts['stopped']} stopped, "
        f"{counts['pending']} pending"
    )

    lines.append("")
    lines.append(f"{'task':32} {'status':14} {'duration':>9} {'round':>5} "
                 f"{'acc':>4} {'rej':>4}  last event")
    for t in state["queue"]:
        last = t.get("last_event_at") or ""
        last_kind = t.get("last_event_kind") or ""
        last_str = f"{last[11:19]} {last_kind}" if last else ""
        if t.get("reflect_status"):
            last_str = f"reflect={t.get('reflect_status')}  " + last_str
        if t.get("stalled"):
            last_str = "⚠ STALLED  " + last_str
        dur = _fmt_duration(t.get("duration_s")) if t.get("duration_s") is not None else (
            _fmt_duration(_running_duration_s(t)) if t["status"] == "running" else ""
        )
        lines.append(
            f"{_truncate(t['name'], 32):32} {t['status']:14} "
            f"{dur:>9} {str(t.get('round') or ''):>5} "
            f"{str(t.get('accepts') or 0):>4} {str(t.get('rejects') or 0):>4}  {last_str}"
        )
    return "\n".join(lines)


def _status_counts(state: dict) -> dict:
    out = {"pending": 0, "resource_wait": 0, "running": 0,
           "done": 0, "failed": 0, "stopped": 0}
    for t in state["queue"]:
        out[t["status"]] = out.get(t["status"], 0) + 1
    return out


def _running_duration_s(t: dict) -> int | None:
    started = t.get("started_at")
    if not started:
        return None
    try:
        from datetime import datetime
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        from datetime import timezone
        delta = datetime.now(timezone.utc) - s
        return int(delta.total_seconds())
    except (ValueError, TypeError):
        return None


def _fmt_duration(seconds) -> str:
    if seconds is None:
        return ""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def stop_dispatch(workspace: Path, *, timeout: float = 30.0) -> dict:
    """SIGTERM the master + running agent process, wait for clean exit.

    Returns a small report dict for the CLI to print.
    """
    pid = read_pid(pid_path(workspace))
    state = read_state(state_path(workspace))
    out = {"master_pid": pid, "master_stopped": False, "agent_pids": [], "claude_pids": []}

    if state and state.get("queue"):
        for t in state["queue"]:
            cp = t.get("agent_pid") or t.get("claude_pid")
            if cp and pid_alive(cp):
                out["agent_pids"].append(cp)
                out["claude_pids"].append(cp)

    if pid is None:
        out["error"] = "no master pid file"
        return out
    if not pid_alive(pid):
        out["master_stopped"] = True
        out["error"] = "master not running"
        try:
            pid_path(workspace).unlink(missing_ok=True)
        except OSError:
            pass
        return out

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        out["master_stopped"] = True
        return out

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            out["master_stopped"] = True
            break
        time.sleep(0.5)

    if not out["master_stopped"]:
        # Last resort: also kill child agent pids directly so they
        # don't keep producing tokens after we've given up.
        for cp in out["agent_pids"]:
            try:
                os.kill(cp, signal.SIGTERM)
            except ProcessLookupError:
                pass
        out["error"] = f"master did not exit within {timeout}s"

    return out


def stream_task_logs(workspace: Path, task_name: str, *, follow: bool, n: int | None = 100):
    """Yield human-readable lines from a task's parsed events ndjson.

    Caller iterates and prints to stdout. Re-uses the foreground
    summary formatter so live and historical output look the same.
    """
    p = task_events_path(workspace, task_name)
    if follow:
        for ev in follow_events(p):
            yield _format_log_line(task_name, ev)
        return
    if not p.is_file():
        return
    try:
        all_lines = p.read_text().splitlines()
        lines = all_lines if n is None else all_lines[-n:]
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            import json as _json
            ev = _json.loads(line)
        except Exception:
            continue
        yield _format_log_line(task_name, ev)


def _format_log_line(task_name: str, ev: dict) -> str:
    ts = (ev.get("ts") or "")[11:19]
    kind = ev.get("kind")
    if kind == "claude_text":
        return f"[{ts}] ✎ {ev.get('snippet','')}"
    if kind == "claude_tool_use":
        return f"[{ts}] ▸ {ev.get('tool')}: {ev.get('summary','')}"
    if kind == "claude_tool_result":
        ok = "ok" if ev.get("ok") else "ERR"
        return f"[{ts}] ◂ {ok}: {ev.get('summary','')}"
    if kind == "claude_done":
        return (f"[{ts}] ✔ done (turns={ev.get('num_turns')}, "
                f"cost=${ev.get('total_cost_usd')})")
    if kind in ("codex_event", "cursor_event"):
        agent = kind.split("_", 1)[0]
        return f"[{ts}] · {agent} {ev.get('event','?')}: {ev.get('summary','')}"
    if kind in ("codex_text", "cursor_text"):
        return f"[{ts}] ✎ {ev.get('snippet','')}"
    if kind in ("zyme_run", "zyme_accept", "zyme_reject"):
        verb = kind[len("zyme_"):]
        text = ev.get("hypothesis") or ev.get("description") or ""
        return f"[{ts}] ★ zyme {verb}: {text}"
    return f"[{ts}] · {kind}"
