from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_real_cursor_dispatch_smoke_composer25(tmp_path: Path):
    """Opt-in smoke test for the real Cursor agent dispatch path.

    This intentionally does not run a real autozyme task. It creates a tiny
    task-shaped workspace, launches one real Cursor agent through `zyme
    dispatch`, and asks the agent to read the manager pipeline prompt and write
    a short probe report. The goal is to catch real CLI/model/stream/schema
    breaks that stub tests cannot see.
    """
    if os.environ.get("ZYME_RUN_REAL_AGENT_SMOKE") != "1":
        pytest.skip("set ZYME_RUN_REAL_AGENT_SMOKE=1 to launch a real Cursor agent")

    cursor_bin = os.environ.get("ZYME_CURSOR_AGENT_BIN") or shutil.which("cursor-agent")
    if not cursor_bin:
        pytest.skip("cursor-agent not found")

    cli_root = Path(__file__).resolve().parents[2]
    framework_root = cli_root.parent
    manager_prompt = framework_root / "autozyme_cli" / "zyme" / "prompts" / "manager" / "0_pipeline.md"
    assert manager_prompt.is_file()

    workspace = tmp_path / "workspace"
    task_dir = workspace / "probe_task"
    prompts_dir = task_dir / "prompts"
    prompts_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        textwrap.dedent(
            """\
            task: probe_task
            datasets:
              - {name: noop, path: /dev/null}
            metrics: []
            """
        ),
        encoding="utf-8",
    )
    (prompts_dir / "framework_probe.md").write_text(
        textwrap.dedent(
            f"""\
            You are running an autozyme framework dispatch smoke probe.
            This is not an optimization task.

            Hard constraints:
            - Do not run zyme init, zyme run, zyme baseline, zyme verify,
              zyme validate, or zyme dispatch.
            - Do not edit files outside this task directory.
            - Read this manager prompt: {manager_prompt}
            - Create `.zyme_probe/framework_audit.md` in this task directory.

            The report must start with this exact first line:
            probe_ok: true

            After that, add 2-5 bullets covering:
            - whether the manager prompt forces execution through zyme dispatch
            - one shallow risk or gap you noticed in the framework prompt, or
              `no major risk found in this shallow smoke`

            Stop immediately after writing the report.
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(cli_root)
        if not env.get("PYTHONPATH")
        else str(cli_root) + os.pathsep + env["PYTHONPATH"]
    )

    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "zyme",
            "dispatch",
            "run",
            "probe_task",
            "--workspace",
            str(workspace),
            "--prompt",
            "prompts/framework_probe.md",
            "--agent",
            "cursor",
            "--model",
            "composer-2.5",
            "--ram-floor",
            "0",
            "--disk-floor",
            "0",
            "--stall-threshold",
            "120",
        ],
        cwd=str(cli_root),
        capture_output=True,
        text=True,
        timeout=360,
        env=env,
    )
    assert cp.returncode == 0, (
        f"dispatch exited {cp.returncode}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
    )

    state = json.loads((workspace / ".zyme_dispatch" / "state.json").read_text(encoding="utf-8"))
    task_state = state["queue"][0]
    assert state["agent"] == "cursor"
    assert state["model"] == "composer-2.5"
    assert task_state["status"] == "done"
    assert task_state["rc"] == 0

    events = [
        json.loads(line)
        for line in (workspace / ".zyme_dispatch" / "events.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    start_events = [event for event in events if event.get("kind") == "task_start"]
    assert start_events, "dispatch did not record task_start"
    cmd = start_events[0]["cmd"]
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "composer-2.5"

    report = task_dir / ".zyme_probe" / "framework_audit.md"
    assert report.is_file(), "real agent did not write the probe report"
    assert report.read_text(encoding="utf-8").splitlines()[0] == "probe_ok: true"


def test_real_codex_dispatch_smoke_gpt55_effort_xhigh(tmp_path: Path):
    """Opt-in smoke test for Codex through the real dispatch path.

    Use Codex's explicit highest supported reasoning effort instead of zyme's
    legacy `max` compatibility label, which maps to high for Codex.
    """
    if os.environ.get("ZYME_RUN_REAL_AGENT_SMOKE") != "1":
        pytest.skip("set ZYME_RUN_REAL_AGENT_SMOKE=1 to launch a real Codex agent")

    codex_bin = os.environ.get("ZYME_CODEX_BIN") or shutil.which("codex")
    if not codex_bin:
        pytest.skip("codex not found")

    cli_root = Path(__file__).resolve().parents[2]
    framework_root = cli_root.parent
    manager_prompt = framework_root / "autozyme_cli" / "zyme" / "prompts" / "manager" / "0_pipeline.md"
    assert manager_prompt.is_file()

    workspace = tmp_path / "workspace"
    task_dir = workspace / "probe_task"
    prompts_dir = task_dir / "prompts"
    prompts_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        textwrap.dedent(
            """\
            task: probe_task
            datasets:
              - {name: noop, path: /dev/null}
            metrics: []
            """
        ),
        encoding="utf-8",
    )
    (prompts_dir / "framework_probe.md").write_text(
        textwrap.dedent(
            f"""\
            You are running an autozyme framework dispatch smoke probe.
            This is not an optimization task.

            Hard constraints:
            - Do not run zyme init, zyme run, zyme baseline, zyme verify,
              zyme validate, or zyme dispatch.
            - Do not edit files outside this task directory.
            - Read this manager prompt: {manager_prompt}
            - Create `.zyme_probe/codex_framework_audit.md` in this task directory.

            The report must start with this exact first line:
            probe_ok: true

            After that, add 1-3 bullets covering whether the manager prompt
            forces execution through zyme dispatch.

            Stop immediately after writing the report.
            """
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=str(task_dir), check=True)
    subprocess.run(["git", "config", "user.email", "smoke@test"], cwd=str(task_dir), check=True)
    subprocess.run(["git", "config", "user.name", "Smoke Test"], cwd=str(task_dir), check=True)
    subprocess.run(["git", "add", "."], cwd=str(task_dir), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial probe task"], cwd=str(task_dir), check=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(cli_root)
        if not env.get("PYTHONPATH")
        else str(cli_root) + os.pathsep + env["PYTHONPATH"]
    )

    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "zyme",
            "dispatch",
            "run",
            "probe_task",
            "--workspace",
            str(workspace),
            "--prompt",
            "prompts/framework_probe.md",
            "--agent",
            "codex",
            "--model",
            "gpt-5.5",
            "--effort",
            "xhigh",
            "--ram-floor",
            "0",
            "--disk-floor",
            "0",
            "--stall-threshold",
            "120",
        ],
        cwd=str(cli_root),
        capture_output=True,
        text=True,
        timeout=360,
        env=env,
    )
    assert cp.returncode == 0, (
        f"dispatch exited {cp.returncode}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
    )

    state = json.loads((workspace / ".zyme_dispatch" / "state.json").read_text(encoding="utf-8"))
    task_state = state["queue"][0]
    assert state["agent"] == "codex"
    assert state["model"] == "gpt-5.5"
    assert state["effort"] == "xhigh"
    assert task_state["status"] == "done"
    assert task_state["rc"] == 0

    events = [
        json.loads(line)
        for line in (workspace / ".zyme_dispatch" / "events.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    start_events = [event for event in events if event.get("kind") == "task_start"]
    assert start_events, "dispatch did not record task_start"
    cmd = start_events[0]["cmd"]
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "gpt-5.5"
    assert 'model_reasoning_effort="xhigh"' in cmd

    report = task_dir / ".zyme_probe" / "codex_framework_audit.md"
    assert report.is_file(), "real agent did not write the probe report"
    assert report.read_text(encoding="utf-8").splitlines()[0] == "probe_ok: true"
