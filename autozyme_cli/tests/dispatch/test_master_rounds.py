from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from zyme.dispatch.master import (
    _build_agent_cmd,
    _reflect_resume_message,
    _results_round_snapshot,
    _resume_prompt,
    resume_dispatch_task,
    start_dispatch,
)
from zyme.dispatch.state import events_path, read_state, state_path


def _script_entrypoint(script: Path) -> Path:
    if os.name != "nt":
        script.chmod(0o755)
        return script
    wrapper = script.with_suffix(".cmd")
    wrapper.write_text(
        f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
        encoding="utf-8",
    )
    return wrapper


HEADER = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\tthread\n"
)


def test_results_round_snapshot_counts_terminal_decisions(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "results.tsv").write_text(
        HEADER
        + "0\tbase\ttiny\t10\t0\t100\tbaseline\t{}\t\t\toptimize\t1\n"
        + "1\tc1\ttiny\t9\t10\t100\tkeep\t{}\ttry a\tok\toptimize\t1\n"
        + "1.1\tc1\tmedium\t20\t5\t200\trerun\t{}\ttry a\t\toptimize\t1\n"
        + "2\tc2\ttiny\t8\t20\t100\tpending\t{}\ttry b\t\toptimize\t1\n"
        + "3\tc3\ttiny\t7\t30\t100\tdiscard\t{}\ttry c\tbad\tvalidate\t1\n",
        encoding="utf-8",
    )

    snap = _results_round_snapshot(task_dir)
    assert snap["completed_rounds"] == 1
    assert snap["pending_rounds"] == 1
    assert snap["last_round"] == 2
    assert snap["last_status"] == "pending"


def test_results_round_snapshot_can_ignore_phase(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "results.tsv").write_text(
        HEADER
        + "1\tc1\ttiny\t9\t10\t100\tkeep\t{}\ttry a\tok\toptimize\t1\n"
        + "2\tc2\ttiny\t8\t20\t100\tdiscard\t{}\ttry b\tbad\tvalidate\t1\n",
        encoding="utf-8",
    )

    snap = _results_round_snapshot(task_dir, phase="all")
    assert snap["completed_rounds"] == 2
    assert snap["pending_rounds"] == 0


def test_cursor_resume_command_uses_same_session(monkeypatch):
    monkeypatch.setenv("ZYME_CURSOR_AGENT_BIN", sys.executable)
    cmd = _build_agent_cmd(
        "cursor",
        "prompts/2_iterate.md",
        "composer-2",
        "max",
        resume_session_id="cursor-session",
        message="continue",
    )
    assert cmd[:3] == [sys.executable, "-p", "--output-format"]
    assert "--resume" in cmd
    assert cmd[cmd.index("--resume") + 1] == "cursor-session"
    assert cmd[-1] == "continue"


def test_codex_resume_command_uses_exec_resume(monkeypatch):
    monkeypatch.setenv("ZYME_CODEX_BIN", sys.executable)
    cmd = _build_agent_cmd(
        "codex",
        "prompts/2_iterate.md",
        "gpt-5.5",
        "max",
        resume_session_id="codex-session",
        message="continue",
    )
    assert cmd[:3] == [sys.executable, "exec", "resume"]
    assert "codex-session" in cmd
    assert cmd[-1] == "continue"


def test_codex_resume_command_preserves_xhigh_effort(monkeypatch):
    monkeypatch.setenv("ZYME_CODEX_BIN", sys.executable)
    cmd = _build_agent_cmd(
        "codex",
        "prompts/2_iterate.md",
        "gpt-5.5",
        "xhigh",
        resume_session_id="codex-session",
        message="continue",
    )
    assert 'model_reasoning_effort="xhigh"' in cmd


def test_claude_resume_command_uses_resume_flag(monkeypatch):
    monkeypatch.setenv("ZYME_CLAUDE_BIN", sys.executable)
    cmd = _build_agent_cmd(
        "claude",
        "prompts/2_iterate.md",
        "claude-sonnet-4-6",
        "max",
        resume_session_id="claude-session",
        message="continue",
    )
    assert cmd[:4] == [sys.executable, "-p", "--resume", "claude-session"]
    assert "continue" in cmd


def test_resume_prompt_encourages_measured_loop_and_blocks_delegation():
    prompt = _resume_prompt("prompts/2_iterate.md", 30, {
        "results_rounds": 4,
        "last_results_round": 4,
        "last_results_status": "keep",
    })
    assert "Continue, you can do this." in prompt
    assert "4 completed round(s), last=4:keep" in prompt
    assert "zyme accept` or `zyme reject" in prompt
    assert "then immediately start the next round" in prompt
    assert "Stop only for a real blocker or when the dispatcher stops you" in prompt
    assert "no subagents" in prompt
    assert "batch runners" in prompt
    assert "pausing after" not in prompt
    assert "30 completed" not in prompt


def test_reflect_resume_message_redirects_to_experiment_reflections(tmp_path):
    reflection_dir = tmp_path / "reflections" / "task1"
    message = _reflect_resume_message(
        prompt="prompts/5_reflect.md",
        state={"prompt": "prompts/2_iterate.md"},
        task={"name": "task1"},
        prompt_feedback_path=reflection_dir / "prompt_feedback.md",
        zyme_cli_feedback_path=reflection_dir / "zyme_cli_feedback.md",
        metadata_path=reflection_dir / "metadata.yaml",
        category="iteration",
    )
    assert "Read and follow prompts/5_reflect.md" in message
    assert str(reflection_dir / "prompt_feedback.md") in message
    assert str(reflection_dir / "zyme_cli_feedback.md") in message
    assert str(reflection_dir / "metadata.yaml") in message
    assert "These exact paths override" in message
    assert "do not commit or push" in message.lower()


def _write_cursor_round_stub(tmp_path):
    stub = tmp_path / "cursor-agent-stub.py"
    stub.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

header = (
    "round\\tcommit\\tdataset\\tspeed_sec\\tspeedup_pct\\tpeak_mb\\tstatus\\t"
    "metrics_json\\thypothesis\\tdescription\\tphase\\tthread\\n"
)
is_resume = "--resume" in sys.argv
round_no = 2 if is_resume else 1
p = Path("results.tsv")
if not p.exists():
    p.write_text(header)
with p.open("a") as f:
    f.write(f"{round_no}\\tc{round_no}\\ttiny\\t1\\t1\\t1\\tkeep\\t{{}}\\th\\td\\toptimize\\t1\\n")
print(json.dumps({
    "type": "system",
    "subtype": "init",
    "session_id": "session-1",
    "model": "Composer 2",
}), flush=True)
print(json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "session_id": "session-1",
}), flush=True)
""",
        encoding="utf-8",
    )
    return _script_entrypoint(stub)


def _write_cursor_reflect_stub(tmp_path):
    stub = tmp_path / "cursor-agent-reflect-stub.py"
    stub.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

message = sys.argv[-1]
print(json.dumps({
    "type": "system",
    "subtype": "init",
    "session_id": "session-1",
    "model": "Composer 2",
}), flush=True)

if "post-run reflection" not in message:
    header = (
        "round\\tcommit\\tdataset\\tspeed_sec\\tspeedup_pct\\tpeak_mb\\tstatus\\t"
        "metrics_json\\thypothesis\\tdescription\\tphase\\tthread\\n"
    )
    p = Path("results.tsv")
    if not p.exists():
        p.write_text(header)
    with p.open("a") as f:
        f.write("1\\tc1\\ttiny\\t1\\t1\\t1\\tkeep\\t{}\\th\\td\\toptimize\\t1\\n")

print(json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "session_id": "session-1",
}), flush=True)
""",
        encoding="utf-8",
    )
    return _script_entrypoint(stub)


def test_dispatch_resumes_clean_exit_until_max_rounds_in_force_mode(tmp_path, monkeypatch):
    task_dir = tmp_path / "task1"
    task_dir.mkdir()
    stub = _write_cursor_round_stub(tmp_path)
    monkeypatch.setenv("ZYME_CURSOR_AGENT_BIN", str(stub))

    start_dispatch(
        workspace=tmp_path,
        tasks=[{"name": "task1", "task_dir": task_dir}],
        prompt="prompts/2_iterate.md",
        agent="cursor",
        model="composer-2",
        effort="max",
        ram_floor_gb=0,
        disk_floor_gb=0,
        detach=False,
        stall_threshold_s=1,
        resource_poll_s=1,
        max_rounds=2,
        force_mode=True,
    )

    state = read_state(state_path(tmp_path))
    task = state["queue"][0]
    assert task["status"] == "done"
    assert task["results_rounds"] == 2
    assert task["resume_attempts"] == 1
    assert task["agent_session_id"] == "session-1"
    assert task["stop_reason"] == "max_rounds"

    events = [json.loads(line) for line in events_path(tmp_path).read_text().splitlines()]
    assert "task_agent_exited_before_max_rounds" in [e["kind"] for e in events]
    assert "task_resume" in [e["kind"] for e in events]


def test_dispatch_does_not_resume_clean_exit_without_force_mode(tmp_path, monkeypatch):
    task_dir = tmp_path / "task1"
    task_dir.mkdir()
    stub = _write_cursor_round_stub(tmp_path)
    monkeypatch.setenv("ZYME_CURSOR_AGENT_BIN", str(stub))

    start_dispatch(
        workspace=tmp_path,
        tasks=[{"name": "task1", "task_dir": task_dir}],
        prompt="prompts/2_iterate.md",
        agent="cursor",
        model="composer-2",
        effort="max",
        ram_floor_gb=0,
        disk_floor_gb=0,
        detach=False,
        stall_threshold_s=1,
        resource_poll_s=1,
        max_rounds=2,
    )

    state = read_state(state_path(tmp_path))
    task = state["queue"][0]
    assert task["status"] == "done"
    assert task["results_rounds"] == 1
    assert task["resume_attempts"] == 0

    events = [json.loads(line) for line in events_path(tmp_path).read_text().splitlines()]
    assert "task_agent_exited_before_max_rounds" not in [e["kind"] for e in events]
    assert "task_resume" not in [e["kind"] for e in events]


def test_dispatch_auto_reflect_resumes_same_session_after_max_rounds(tmp_path, monkeypatch):
    task_dir = tmp_path / "task1"
    task_dir.mkdir()
    stub = _write_cursor_reflect_stub(tmp_path)
    monkeypatch.setenv("ZYME_CURSOR_AGENT_BIN", str(stub))

    start_dispatch(
        workspace=tmp_path,
        tasks=[{"name": "task1", "task_dir": task_dir}],
        prompt="prompts/2_iterate.md",
        agent="cursor",
        model="composer-2",
        effort="max",
        ram_floor_gb=0,
        disk_floor_gb=0,
        detach=False,
        stall_threshold_s=1,
        resource_poll_s=1,
        max_rounds=1,
        reflect=True,
        reflect_prompt="prompts/5_reflect.md",
        reflection_root=tmp_path / "reflections",
        reflect_category="iteration",
    )

    state = read_state(state_path(tmp_path))
    task = state["queue"][0]
    assert task["status"] == "done"
    assert task["results_rounds"] == 1
    assert task["agent_session_id"] == "session-1"
    assert task["reflect_status"] == "done"
    reflection_root = tmp_path / "reflections"
    assert task["reflection_dir"] == str(reflection_root)
    assert task["reflection_prompt_feedback"].endswith(
        "prompt_reflect_feedback\\iteration_task1.md"
    ) or task["reflection_prompt_feedback"].endswith(
        "prompt_reflect_feedback/iteration_task1.md"
    )
    assert task["reflection_zyme_cli_feedback"].endswith(
        "zyme_cli_feedback\\iteration_task1.md"
    ) or task["reflection_zyme_cli_feedback"].endswith(
        "zyme_cli_feedback/iteration_task1.md"
    )
    metadata = Path(task["reflection_metadata"]).read_text(encoding="utf-8")
    assert 'category: "iteration"' in metadata
    assert 'reflect_prompt: "prompts/5_reflect.md"' in metadata

    events = [json.loads(line) for line in events_path(tmp_path).read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert "task_reflect_start" in kinds
    assert "task_reflect_finished" in kinds


def test_manual_resume_sends_message_to_recorded_session(tmp_path, monkeypatch):
    task_dir = tmp_path / "task1"
    task_dir.mkdir()
    stub = _write_cursor_reflect_stub(tmp_path)
    monkeypatch.setenv("ZYME_CURSOR_AGENT_BIN", str(stub))

    start_dispatch(
        workspace=tmp_path,
        tasks=[{"name": "task1", "task_dir": task_dir}],
        prompt="prompts/2_iterate.md",
        agent="cursor",
        model="composer-2",
        effort="max",
        ram_floor_gb=0,
        disk_floor_gb=0,
        detach=False,
        stall_threshold_s=1,
        resource_poll_s=1,
        max_rounds=1,
    )

    rc = resume_dispatch_task(
        workspace=tmp_path,
        task_name="task1",
        message="post-run reflection follow-up",
        foreground=False,
        stall_threshold_s=1,
    )

    assert rc == 0
    events = [json.loads(line) for line in events_path(tmp_path).read_text().splitlines()]
    assert "task_manual_resume" in [e["kind"] for e in events]
