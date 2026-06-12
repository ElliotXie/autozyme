"""`zyme dispatch[_status|_logs|_stop]` — CLI wrappers around the zyme.dispatch daemon."""

import os
import sys

from zyme.utils import die, info




# ===========================================================================
# zyme dispatch — multi-task agent driver
# ===========================================================================
# Architecture (3 layers):
#   manager (claude code session)  — picks tasks + prompt, monitors via status
#         ↓ invokes
#   CLI (zyme dispatch)            — resource gates, daemon, stream parsing
#         ↓ launches
#   workers (claude -p / codex exec) — one per task, runs the prompt
#
# All deterministic mechanics live here so the manager doesn't repeat them.

def _resolve_dispatch_workspace(args):
    """Workspace root: --workspace or cwd. Created if absent? No — we
    require an existing dir so we don't accidentally create one in $HOME.
    """
    from pathlib import Path as _P
    ws = _P(getattr(args, "workspace", None) or os.getcwd()).resolve()
    if not ws.is_dir():
        die(f"workspace not found: {ws}")
    return ws




def _resolve_task_paths(workspace, names):
    """Map task names/paths → list of {name, task_dir}.

    Three lookup modes per entry:
      1. Absolute or relative path that contains task.yaml → use directly.
      2. Bare name matching <workspace>/<name>/task.yaml → use that.
      3. Bare name matching <workspace>/test_<name>/task.yaml → legacy.
    Errors if any entry can't be resolved.
    """
    from pathlib import Path as _P
    out = []
    errors = []
    for raw in names:
        p = _P(raw)
        candidates = []
        if p.is_absolute() or "/" in raw:
            candidates.append(p)
        else:
            candidates.append(workspace / raw)
            if not raw.startswith("test_"):
                candidates.append(workspace / f"test_{raw}")
        resolved = None
        for c in candidates:
            if (c / "task.yaml").is_file():
                resolved = c.resolve()
                break
        if resolved is None:
            errors.append(raw)
            continue
        out.append({"name": resolved.name, "task_dir": str(resolved)})
    if errors:
        die("could not resolve task(s): " + ", ".join(errors)
            + f" (looked under {workspace})")
    return out




def _validate_prompt_files(prompt_rel, tasks):
    """Each task must have <task_dir>/<prompt_rel>. Otherwise error early."""
    from pathlib import Path as _P
    missing = []
    for t in tasks:
        p = _P(t["task_dir"]) / prompt_rel
        if not p.is_file():
            missing.append(f"{t['name']}: {p}")
    if missing:
        die("prompt file missing for:\n  " + "\n  ".join(missing))




def cmd_dispatch(args):
    """Start a multi-task claude -p run.

    See `zyme dispatch -h` for flags. Foreground by default; --detach
    daemonizes via POSIX double-fork.
    """
    from zyme.dispatch import (
        DEFAULT_CODEX_MODEL, DEFAULT_CURSOR_MODEL, DEFAULT_DISK_FLOOR_FALLBACK_GB,
        DEFAULT_RAM_FLOOR_GB, DEFAULT_REFLECT_PROMPT,
        find_agent_binary, start_dispatch,
    )
    from zyme.dispatch.resources import parse_size_gb
    from zyme.dispatch.state import (
        ensure_dispatch_dirs, master_log_path, pid_alive,
        pid_path, read_pid, state_path,
    )

    workspace = _resolve_dispatch_workspace(args)

    # Refuse early if a master is already running.
    prior = read_pid(pid_path(workspace))
    if prior is not None and pid_alive(prior):
        die(f"another zyme dispatch is already running in {workspace} "
            f"(PID {prior}). Run `zyme dispatch-stop` first.")

    # Resolve "auto" to a concrete agent name before anything else.
    if args.agent == "auto":
        from zyme.dispatch.master import detect_agent_binary
        args.agent, _ = detect_agent_binary()
    try:
        find_agent_binary(args.agent)
    except RuntimeError as e:
        die(str(e))
    if args.agent == "claude" and not args.model:
        args.model = "claude-opus-4-7[1m]"
    if args.agent == "codex" and not args.model:
        args.model = DEFAULT_CODEX_MODEL
    if args.agent == "cursor" and not args.model:
        args.model = DEFAULT_CURSOR_MODEL

    tasks = _resolve_task_paths(workspace, args.tasks)
    _validate_prompt_files(args.prompt, tasks)
    reflect_prompt = args.reflect_prompt or DEFAULT_REFLECT_PROMPT
    if args.reflect:
        _validate_prompt_files(reflect_prompt, tasks)

    try:
        parsed_ram_floor = parse_size_gb(args.ram_floor)
        ram_floor_gb = DEFAULT_RAM_FLOOR_GB if parsed_ram_floor is None else parsed_ram_floor
    except ValueError as e:
        die(str(e))
    if args.disk_floor == "auto":
        disk_floor_gb = None
    else:
        try:
            disk_floor_gb = parse_size_gb(args.disk_floor)
        except ValueError as e:
            die(str(e))

    if args.dry_run:
        print(f"workspace : {workspace}")
        print(f"prompt    : {args.prompt}")
        print(f"agent     : {args.agent}")
        print(f"model     : {args.model or '(agent default)'} (effort={args.effort})")
        print(f"max rounds: {args.max_rounds or '(none)'}")
        print(f"force mode: {bool(args.force_mode)}")
        print(f"reflect   : {bool(args.reflect)}"
              + (f" ({reflect_prompt})" if args.reflect else ""))
        print(f"ram floor : {ram_floor_gb:.1f} GB")
        print(f"disk floor: "
              + ("auto (per-task estimate ×2, fallback "
                 f"{DEFAULT_DISK_FLOOR_FALLBACK_GB} GB)"
                 if disk_floor_gb is None else f"{disk_floor_gb:.1f} GB"))
        print(f"detach    : {args.detach}")
        print(f"queue     ({len(tasks)} tasks):")
        for i, t in enumerate(tasks, 1):
            print(f"  {i:2d}. {t['name']}  ({t['task_dir']})")
        return

    ensure_dispatch_dirs(workspace)

    try:
        pid = start_dispatch(
            workspace=workspace,
            tasks=tasks,
            prompt=args.prompt,
            agent=args.agent,
            model=args.model,
            effort=args.effort,
            ram_floor_gb=ram_floor_gb,
            disk_floor_gb=disk_floor_gb,
            detach=args.detach,
            stall_threshold_s=args.stall_threshold,
            max_rounds=args.max_rounds,
            force_mode=args.force_mode,
            reflect=args.reflect,
            reflect_prompt=reflect_prompt,
            reflection_root=args.reflection_root,
            reflect_category=args.reflect_category,
        )
    except RuntimeError as e:
        die(str(e))

    if args.detach:
        info(f"dispatch started (PID {pid}, daemonized)")
        info(f"  state : {state_path(workspace)}")
        info(f"  log   : {master_log_path(workspace)}")
        info(f"  watch : zyme dispatch-status [--workspace {workspace}]")
        info(f"  stop  : zyme dispatch-stop [--workspace {workspace}]")




def cmd_dispatch_status(args):
    """Print current dispatch status (one screen)."""
    from zyme.dispatch import render_status
    workspace = _resolve_dispatch_workspace(args)
    print(render_status(workspace))


def cmd_dispatch_usage(args):
    """Print token/cost telemetry for a dispatch workspace."""
    import json
    from zyme.dispatch import collect_usage, render_usage
    workspace = _resolve_dispatch_workspace(args)
    summary = collect_usage(
        workspace,
        token_budget=args.token_budget,
        budget_basis=args.budget_basis,
        price_model=args.price_model,
    )
    if args.json_output:
        print(json.dumps(summary, indent=2, sort_keys=False))
    else:
        print(render_usage(summary))


def cmd_dispatch_prices(args):
    """Print the built-in model price registry."""
    import json
    from zyme.dispatch import list_prices, render_price_table
    if args.json_output:
        print(json.dumps(list_prices(), indent=2, sort_keys=False))
    else:
        print(render_price_table())


def cmd_dispatch_resume(args):
    """Send one prompt/message to a recorded dispatch agent session."""
    from pathlib import Path
    from zyme.dispatch import resume_dispatch_task
    from zyme.dispatch.state import pid_alive, pid_path, read_pid, read_state, state_path

    workspace = _resolve_dispatch_workspace(args)
    prior = read_pid(pid_path(workspace))
    if prior is not None and pid_alive(prior):
        die(f"dispatch is still running in {workspace} (PID {prior}); "
            "wait for it to finish or stop it before manual resume.")

    state = read_state(state_path(workspace))
    if state is None:
        die(f"no dispatch state at {state_path(workspace)}")
    task_name = args.task
    if task_name is None and len(state.get("queue") or []) == 1:
        task_name = state["queue"][0]["name"]
    if task_name is None:
        die("task is required when dispatch state has multiple tasks")
    task = next(
        (t for t in state.get("queue") or []
         if t.get("name") == task_name or Path(str(t.get("task_dir", ""))).name == task_name),
        None,
    )
    if task is None:
        die(f"task {task_name!r} not found in dispatch state")
    if not task.get("agent_session_id"):
        die(f"task {task_name!r} has no agent_session_id to resume")
    if args.prompt:
        prompt_path = Path(args.prompt)
        check_path = prompt_path if prompt_path.is_absolute() else Path(task["task_dir"]) / prompt_path
        if not check_path.is_file():
            die(f"prompt file not found for {task_name}: {check_path}")

    if args.dry_run:
        msg = args.message or f"read and follow {args.prompt}"
        print(f"workspace : {workspace}")
        print(f"task      : {task_name}")
        print(f"agent     : {state.get('agent')}")
        print(f"model     : {state.get('model')}")
        print(f"session   : {task.get('agent_session_id')}")
        print(f"message   : {msg}")
        return

    rc = resume_dispatch_task(
        workspace=workspace,
        task_name=task_name,
        prompt=args.prompt,
        message=args.message,
        foreground=True,
        stall_threshold_s=args.stall_threshold,
    )
    if rc not in (0, None):
        die(f"resume exited with rc={rc}")




def cmd_dispatch_logs(args):
    """Tail one task's parsed event stream."""
    from zyme.dispatch import stream_task_logs
    from zyme.dispatch.state import (
        read_state, state_path, task_events_path,
    )
    workspace = _resolve_dispatch_workspace(args)

    state = read_state(state_path(workspace))
    known_names = []
    if state and state.get("queue"):
        known_names = [t["name"] for t in state["queue"]]
        if args.task not in known_names:
            # Allow looking up logs by directory basename even if state
            # is missing — useful for inspecting old runs.
            ev_file = task_events_path(workspace, args.task)
            if not ev_file.is_file():
                die(f"task {args.task!r} not found in state and no events file at {ev_file}. "
                    f"Known: {', '.join(known_names) or '(none)'}")

    n = None if getattr(args, "full", False) else args.last_n
    try:
        for line in stream_task_logs(
            workspace, args.task, follow=args.follow, n=n,
        ):
            print(line, flush=True)
    except KeyboardInterrupt:
        pass




def cmd_dispatch_wait(args):
    """Block until the dispatch master exits, polling at --poll interval."""
    import time
    from zyme.dispatch.state import pid_alive, pid_path, read_pid, read_state, state_path

    workspace = _resolve_dispatch_workspace(args)
    poll = getattr(args, "poll", 60)
    timeout = getattr(args, "timeout", None)
    verbose = getattr(args, "verbose", False)
    start = time.monotonic()

    sp = state_path(workspace)
    pp = pid_path(workspace)

    state = read_state(sp)
    if state is None:
        die(f"no dispatch state at {sp}")

    while True:
        pid = read_pid(pp)
        if pid is None or not pid_alive(pid):
            state = read_state(sp)
            if state and state.get("finished_at"):
                if verbose:
                    elapsed = time.monotonic() - start
                    print(f"[wait] dispatch finished after {elapsed:.0f}s")
                sys.exit(0)
            else:
                print("[wait] master process gone but no finished_at in state — may have crashed")
                sys.exit(2)

        if timeout and (time.monotonic() - start) > timeout:
            print(f"[wait] timeout after {timeout}s")
            sys.exit(124)

        if verbose:
            state = read_state(sp)
            queue = state.get("queue", []) if state else []
            running = [t["name"] for t in queue if t.get("status") == "running"]
            done = sum(1 for t in queue if t.get("finished_at"))
            total = len(queue)
            elapsed = time.monotonic() - start
            print(f"[wait] {elapsed:.0f}s elapsed — {done}/{total} tasks done"
                  + (f", running: {', '.join(running)}" if running else ""))

        time.sleep(poll)


def cmd_dispatch_stop(args):
    """SIGTERM the master and any running claude; wait for clean exit."""
    from zyme.dispatch import stop_dispatch
    workspace = _resolve_dispatch_workspace(args)
    report = stop_dispatch(workspace)
    if report.get("error") and not report.get("master_stopped"):
        sys.stderr.write(f"zyme: {report['error']}\n")
    print(f"master pid    : {report.get('master_pid')}")
    print(f"master stopped: {report.get('master_stopped')}")
    if report.get("agent_pids"):
        print(f"agent pids    : {report['agent_pids']}")
    if report.get("error") and report.get("master_stopped"):
        # Soft note — e.g. master was already gone, we cleaned up the pid file
        info(report["error"])
