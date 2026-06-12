"""zyme iterate — single-task auto-resume babysitter for the iterate phase.

Launches a Claude agent with the iterate prompt, monitors its output via
stream-json, and automatically resumes the session when the agent exits
before reaching --max-rounds. Uses the same resume logic as dispatch's
force_mode, without the multi-task orchestration overhead.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from zyme.dispatch.events import parse_agent_line
from zyme.dispatch.master import (
    find_claude_binary,
    _build_agent_cmd,
    _read_stream_with_stalls,
    _resume_prompt,
    _results_round_snapshot,
    _terminate,
)
from zyme.utils import task_dir_from_args

DEFAULT_MAX_ROUNDS = 50
DEFAULT_NO_PROGRESS_LIMIT = 3
DEFAULT_STALL_THRESHOLD_S = 900
DEFAULT_PROMPT = "prompts/2_iterate.md"

_SHOULD_STOP = False


def _install_signal_handlers():
    global _SHOULD_STOP

    def _handler(signum, frame):
        global _SHOULD_STOP
        _SHOULD_STOP = True
        print(
            f"\n[iterate] caught signal {signum}, stopping after current agent exits…",
            file=sys.stderr, flush=True,
        )

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _count_rounds(task_dir: Path) -> dict:
    return _results_round_snapshot(task_dir)


def _print_status(label: str, snap: dict, max_rounds: int):
    completed = snap["completed_rounds"]
    last_r = snap["last_round"]
    last_s = snap["last_status"]
    print(
        f"[iterate] {label}: {completed}/{max_rounds} rounds"
        f" (last={last_r}:{last_s})",
        file=sys.stderr, flush=True,
    )


def _strip_prompt_header(text: str) -> str:
    """Remove the '# <title>' and '## Role' header lines from the prompt."""
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") or stripped == "## Role" or not stripped:
            start = i + 1
        else:
            break
    return "\n".join(lines[start:])


def cmd_iterate(args):
    task_dir = task_dir_from_args(args)

    prompt = args.prompt
    prompt_path = task_dir / prompt
    if not prompt_path.exists():
        print(f"error: prompt not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)

    prompt_content = prompt_path.read_text(encoding="utf-8")
    # Strip the title + role header — agent doesn't need it
    prompt_content = _strip_prompt_header(prompt_content)

    max_rounds = args.max_rounds
    no_progress_limit = args.no_progress_limit
    stall_threshold = args.stall_threshold
    model = getattr(args, "model", None)
    effort = getattr(args, "effort", None)

    _install_signal_handlers()

    snap = _count_rounds(task_dir)
    if snap["completed_rounds"] >= max_rounds:
        print(
            f"[iterate] already at {snap['completed_rounds']}/{max_rounds} rounds, nothing to do.",
            file=sys.stderr, flush=True,
        )
        return

    _print_status("starting", snap, max_rounds)

    session_id = None
    launch_index = 0
    no_progress_exits = 0
    previous_completed = snap["completed_rounds"]

    while not _SHOULD_STOP:
        is_resume = launch_index > 0

        if is_resume:
            if not session_id:
                print(
                    "[iterate] no session_id from previous launch, cannot resume. "
                    "Starting fresh.",
                    file=sys.stderr, flush=True,
                )
                session_id = None
                is_resume = False

        # Build the resume task dict (mimics dispatch state shape)
        task_state = {
            "results_rounds": previous_completed,
            "last_results_round": snap.get("last_round"),
            "last_results_status": snap.get("last_status"),
        }

        if is_resume:
            resume_text = _resume_prompt(prompt, max_rounds, task_state)
            message = f"{resume_text}\n\n---\n\n{prompt_content}"
        else:
            message = prompt_content

        cmd = _build_agent_cmd(
            "claude",
            prompt,
            model,
            effort,
            resume_session_id=session_id if is_resume else None,
            message=message,
        )

        if is_resume:
            print(
                f"[iterate] resuming session {session_id[:12]}… "
                f"({previous_completed}/{max_rounds} rounds)",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                f"[iterate] launching claude with {prompt}",
                file=sys.stderr, flush=True,
            )

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(task_dir),
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            print(f"[iterate] failed to launch claude: {e}", file=sys.stderr)
            sys.exit(1)

        max_rounds_hit = False

        for kind, payload in _read_stream_with_stalls(proc, stall_threshold):
            if _SHOULD_STOP:
                _terminate(proc)
                break
            if kind == "line":
                events = parse_agent_line(payload, agent="claude")
                for ev in events:
                    if ev.get("kind") == "agent_session" and ev.get("session_id"):
                        session_id = ev["session_id"]
                    # Print zyme_run / zyme_accept / zyme_reject events
                    ek = ev.get("kind", "")
                    if ek.startswith("zyme_"):
                        hyp = ev.get("hypothesis") or ev.get("description") or ""
                        print(
                            f"  [{ek}] {hyp[:100]}",
                            file=sys.stderr, flush=True,
                        )

                # Check max rounds after processing events
                snap = _count_rounds(task_dir)
                if snap["completed_rounds"] >= max_rounds:
                    max_rounds_hit = True
                    _print_status("max rounds reached", snap, max_rounds)
                    _terminate(proc)
                    break
            elif kind == "stall":
                secs = int(payload)
                print(
                    f"[iterate] stall detected: {secs}s since last output",
                    file=sys.stderr, flush=True,
                )
                snap = _count_rounds(task_dir)
                if snap["completed_rounds"] >= max_rounds:
                    max_rounds_hit = True
                    _print_status("max rounds reached (during stall)", snap, max_rounds)
                    _terminate(proc)
                    break
            elif kind == "exit":
                break

        rc = proc.wait()

        if _SHOULD_STOP:
            snap = _count_rounds(task_dir)
            _print_status("interrupted", snap, max_rounds)
            return

        snap = _count_rounds(task_dir)

        if max_rounds_hit or snap["completed_rounds"] >= max_rounds:
            _print_status("done", snap, max_rounds)
            return

        if rc != 0:
            print(
                f"[iterate] claude exited with rc={rc}",
                file=sys.stderr, flush=True,
            )
            _print_status("agent error", snap, max_rounds)
            return

        # Agent exited cleanly but below max rounds — check progress
        completed = snap["completed_rounds"]
        if completed <= previous_completed:
            no_progress_exits += 1
            print(
                f"[iterate] agent exited with no new rounds "
                f"({no_progress_exits}/{no_progress_limit})",
                file=sys.stderr, flush=True,
            )
        else:
            no_progress_exits = 0

        previous_completed = completed

        if no_progress_exits >= no_progress_limit:
            print(
                f"[iterate] giving up: {no_progress_limit} consecutive exits "
                f"with no new rounds at {completed}/{max_rounds}",
                file=sys.stderr, flush=True,
            )
            return

        _print_status("agent exited early, will resume", snap, max_rounds)
        launch_index += 1

    # Loop ended via _SHOULD_STOP
    snap = _count_rounds(task_dir)
    _print_status("stopped", snap, max_rounds)
