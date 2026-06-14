"""Unit tests for zyme.runner — pipeline/evaluate orchestration helpers.

Two layers:

  1. Pure helpers (no subprocess): path/python resolution, crash-message
     extraction, returncode explanation, tier wall caps, float extraction
     from stdout, summary block formatting, reference-command construction,
     metric allowlist filtering.

  2. run_task / dryrun_task orchestration, driven by monkeypatching the
     single subprocess boundary (`runner._run_with_watchdog` /
     `subprocess.run`) so the surrounding decision logic (env assembly,
     crash classification, thread-budget audit, metric scraping) runs without
     launching anything.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import zyme.runner as runner
from zyme.runner import (
    _explain_returncode,
    _extract_crash_msg,
    _extract_float,
    _looks_like_path,
    _resolve_python,
    _summary,
    _try_extract_float,
    _wall_cap_for_tier,
    build_reference_cmd,
    dryrun_task,
    run_task,
)


# --------------------------------------------------------------------------
# _looks_like_path
# --------------------------------------------------------------------------

class TestLooksLikePath:
    @pytest.mark.parametrize("spec,expected", [
        ("myenv", False),
        ("base", False),
        ("/opt/conda/envs/x/bin/python", True),
        ("D:\\anaconda3\\python.exe", True),
        ("python.exe", True),
        ("relative/path", True),
        ("C:\\Python\\python.EXE", True),
    ])
    def test_detection(self, spec, expected):
        assert _looks_like_path(spec) is expected


# --------------------------------------------------------------------------
# _resolve_python
# --------------------------------------------------------------------------

class TestResolvePython:
    def test_absolute_existing_path_returned(self, tmp_path):
        fake = tmp_path / "python"
        fake.write_text("#!/bin/sh\n")
        assert _resolve_python(str(fake)) == str(fake)

    def test_absolute_missing_path_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="path not found"):
            _resolve_python(str(tmp_path / "nope" / "python"))

    def test_env_name_resolved_via_conda_root(self, tmp_path, monkeypatch):
        # Build a fake conda root with envs/myenv/bin/python and point
        # AUTOZYME_CONDA_ROOT at it.
        root = tmp_path / "conda"
        env_bin = root / "envs" / "myenv" / "bin"
        env_bin.mkdir(parents=True)
        py = env_bin / "python"
        py.write_text("")
        monkeypatch.setenv("AUTOZYME_CONDA_ROOT", str(root))
        monkeypatch.delenv("CONDA_EXE", raising=False)
        assert _resolve_python("myenv") == str(py)

    def test_unresolvable_env_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CONDA_EXE", raising=False)
        monkeypatch.delenv("AUTOZYME_CONDA_ROOT", raising=False)
        # Point HOME at an empty dir so none of the ~/miniconda3 candidates exist.
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        # Make `conda` un-findable.
        monkeypatch.setattr(runner.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="could not resolve conda env"):
            _resolve_python("definitely_not_an_env")


# --------------------------------------------------------------------------
# build_reference_cmd
# --------------------------------------------------------------------------

class TestBuildReferenceCmd:
    def _task_yaml(self, tmp_path, body=""):
        ty = tmp_path / "task.yaml"
        ty.write_text(body)
        return ty

    def test_py_reference_falls_back_to_sys_executable(self, tmp_path):
        ty = self._task_yaml(tmp_path)  # no executor block
        ref = tmp_path / "reference.py"
        ref.write_text("")
        cmd = build_reference_cmd(ty, ref)
        assert cmd == [sys.executable, str(ref)]

    def test_py_reference_honors_executor_python_path(self, tmp_path):
        py = tmp_path / "py" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        ty = self._task_yaml(tmp_path, f"executor:\n  python: {py}\n")
        ref = tmp_path / "reference.py"
        ref.write_text("")
        cmd = build_reference_cmd(ty, ref)
        assert cmd == [str(py), str(ref)]

    def test_r_reference_default_rscript(self, tmp_path):
        ty = self._task_yaml(tmp_path)
        ref = tmp_path / "reference.R"
        ref.write_text("")
        cmd = build_reference_cmd(ty, ref)
        assert cmd == ["Rscript", str(ref)]

    def test_r_reference_honors_executor_rscript(self, tmp_path):
        ty = self._task_yaml(tmp_path, "executor:\n  rscript: /opt/R/bin/Rscript\n")
        ref = tmp_path / "reference.R"
        ref.write_text("")
        cmd = build_reference_cmd(ty, ref)
        assert cmd == ["/opt/R/bin/Rscript", str(ref)]

    def test_unsupported_extension_raises(self, tmp_path):
        ty = self._task_yaml(tmp_path)
        ref = tmp_path / "reference.jl"
        ref.write_text("")
        with pytest.raises(RuntimeError, match="unsupported reference"):
            build_reference_cmd(ty, ref)


# --------------------------------------------------------------------------
# _extract_crash_msg
# --------------------------------------------------------------------------

class TestExtractCrashMsg:
    def test_empty(self):
        assert _extract_crash_msg("") == ""
        assert _extract_crash_msg("\n\n  \n") == ""

    def test_prefers_last_error_line(self):
        stderr = "warning: foo\nError in bar(): boom\nsome trailing note\n"
        # picks last line containing 'error' (case-insensitive)
        assert _extract_crash_msg(stderr) == "Error in bar(): boom"

    def test_falls_back_to_last_nonempty(self):
        stderr = "line one\nline two\n"
        assert _extract_crash_msg(stderr) == "line two"

    def test_truncates_to_max_chars(self):
        msg = "Error: " + "z" * 500
        out = _extract_crash_msg(msg, max_chars=50)
        assert len(out) == 50
        assert out.endswith("...")

    def test_sanitizes_tabs_and_newlines(self):
        out = _extract_crash_msg("Error:\tboom\rmore")
        assert "\t" not in out
        assert "\r" not in out


# --------------------------------------------------------------------------
# _explain_returncode
# --------------------------------------------------------------------------

class TestExplainReturncode:
    def test_known_signals(self):
        assert "SIGKILL" in _explain_returncode(137)
        assert "SIGSEGV" in _explain_returncode(139)
        assert "SIGABRT" in _explain_returncode(134)
        assert "watchdog" in _explain_returncode(144).lower()

    def test_negative_rc(self):
        assert _explain_returncode(-9) == "died from signal 9"

    def test_generic_128_band(self):
        out = _explain_returncode(130)  # 128 + 2 (SIGINT)
        assert "signal 2" in out
        assert "128+signum" in out

    def test_no_special_meaning(self):
        assert _explain_returncode(1) == ""
        assert _explain_returncode(0) == ""


# --------------------------------------------------------------------------
# _wall_cap_for_tier
# --------------------------------------------------------------------------

class TestWallCapForTier:
    def test_known_tiers(self):
        assert _wall_cap_for_tier("tiny") == 20 * 60
        assert _wall_cap_for_tier("medium") == 30 * 60
        assert _wall_cap_for_tier("large") == 60 * 60

    def test_unknown_tier_no_cap(self):
        assert _wall_cap_for_tier("ood_large") is None
        assert _wall_cap_for_tier("custom") is None

    def test_none_tier(self):
        assert _wall_cap_for_tier(None) is None


# --------------------------------------------------------------------------
# _try_extract_float / _extract_float
# --------------------------------------------------------------------------

class TestExtractFloat:
    def test_extracts_value(self):
        assert _try_extract_float("speed_sec:  8.5\n", "speed_sec") == 8.5

    def test_missing_key_returns_none(self):
        assert _try_extract_float("nothing here\n", "speed_sec") is None

    def test_unparseable_returns_none(self):
        assert _try_extract_float("speed_sec: notanumber\n", "speed_sec") is None

    def test_first_match_wins(self):
        out = _try_extract_float("peak_mb: 100\npeak_mb: 200\n", "peak_mb")
        assert out == 100.0

    def test_wrapper_default_on_missing(self):
        assert _extract_float("nope\n", "peak_mb", default=42.0) == 42.0

    def test_wrapper_returns_value(self):
        assert _extract_float("peak_mb: 512.0\n", "peak_mb") == 512.0

    def test_key_with_dot_is_escaped(self):
        # _try_extract_float re.escapes the key, so a dotted key matches literally.
        assert _try_extract_float("a.b: 3.0\n", "a.b") == 3.0


# --------------------------------------------------------------------------
# _summary
# --------------------------------------------------------------------------

class TestSummary:
    def test_basic_block(self):
        out = _summary(speed=8.5, peak=512.0, status="ok")
        assert "speed_sec:        8.500" in out
        assert "peak_mb:          512.0" in out
        assert "status:           ok" in out
        assert out.startswith("\n---")
        assert out.endswith("---")

    def test_with_cpu_sec(self):
        out = _summary(speed=1.0, peak=1.0, status="ok", cpu_sec=3.25)
        assert "cpu_sec:          3.250" in out

    def test_with_metric_lines(self):
        out = _summary(speed=1.0, peak=1.0, status="ok",
                       metric_lines=["pearson: 0.99", "max_diff: 0.001"])
        assert "pearson: 0.99" in out
        assert "max_diff: 0.001" in out

    def test_crash_status(self):
        out = _summary(speed=0.0, peak=0.0, status="crash")
        assert "status:           crash" in out


# --------------------------------------------------------------------------
# run_task — orchestration with the subprocess boundary monkeypatched
# --------------------------------------------------------------------------

def _make_py_task(tmp_path: Path, *, with_ref=True, metrics="") -> Path:
    """Minimal Python task dir: pipeline/run.py, evaluate.py, reference_output."""
    task = tmp_path / "task"
    task.mkdir()
    (task / "pipeline").mkdir()
    (task / "pipeline" / "run.py").write_text("print('pipeline')\n")
    (task / "evaluate.py").write_text("print('eval')\n")
    yaml = "target_repo: stub\ntarget_function: foo\n"
    if metrics:
        yaml += metrics
    (task / "task.yaml").write_text(yaml)
    if with_ref:
        ref = task / "reference_output"
        ref.mkdir()
        (ref / "marker.txt").write_text("x")
    return task


class _FakeWatchdog:
    """Records calls and returns scripted (rc, stdout, stderr, ...) tuples."""
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, cmd, *, cwd, env, mem_cap_gb, **kw):
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": dict(env), **kw})
        if self.results:
            return self.results.pop(0)
        # default: clean run
        return (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0)


class TestRunTaskOrchestration:
    def test_missing_reference_output_returns_crash(self, tmp_path):
        task = _make_py_task(tmp_path, with_ref=False)
        out = run_task(task)
        assert "reference_output empty" in out
        assert "status:           crash" in out

    def test_happy_path_records_speed_and_metrics(self, tmp_path, monkeypatch):
        task = _make_py_task(
            tmp_path,
            metrics="metrics:\n  - {name: pearson, comparator: gte, threshold: 0.9}\n",
        )
        fake = _FakeWatchdog(
            # pipeline phase
            (0, "speed_sec: 2.5\npeak_mb: 100.0\ncpu_sec: 2.4\n", "", None, None, 0, 0.0),
            # evaluate phase — declared metric pearson + an undeclared debug print
            (0, "pearson: 0.995\nDEBUG_PATH: /tmp/x\n", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        assert "status:           ok" in out
        assert "speed_sec:        2.500" in out
        # The summary metric block (last `---`-delimited section) carries only
        # the declared metric, not the DEBUG_PATH debug print. The raw eval
        # stdout is still echoed earlier in the log, so scope the check to the
        # trailing summary block.
        summary_block = out.rsplit("\n---\n", 1)[-1] if "\n---\n" in out else out
        # find the final block after the last opening '---'
        final_block = out[out.rfind("\n---\n"):]
        assert "pearson: 0.995" in final_block
        assert "DEBUG_PATH" not in final_block

    def test_pipeline_crash_short_circuits(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (1, "", "Traceback...\nValueError: bad input\n", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        assert "CRASH: pipeline exited 1" in out
        assert "ValueError: bad input" in out
        assert "status:           crash" in out
        # evaluate must NOT have run
        assert len(fake.calls) == 1

    def test_pipeline_timeout_message(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        # rc=-9 + timed_out_at_s set → TIMEOUT crash message
        fake = _FakeWatchdog(
            (-9, "", "", None, 1200.0, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        dataset = {"tier": "tiny", "name": "t", "path": ""}
        out = run_task(task, dataset_entry=dataset)
        assert "TIMEOUT" in out
        assert "wall cap" in out
        assert "status:           crash" in out

    def test_mem_watchdog_kill_message(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (-9, "", "", 8.5, None, 0, 0.0),  # killed_peak_gb=8.5
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task, mem_cap_gb=4.0)
        assert "killed-by-mem-watchdog" in out
        assert "8.5 GB" in out

    def test_wall_fallback_warning_when_no_speed_sec(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "did not print speed\n", "", None, None, 0, 0.0),  # pipeline
            (0, "", "", None, None, 0, 0.0),                       # evaluate
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        assert "did not print 'speed_sec:'" in out
        assert "status:           ok" in out

    def test_suspiciously_fast_warning(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "speed_sec: 0.005\npeak_mb: 1.0\n", "", None, None, 0, 0.0),
            (0, "", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        assert "suspiciously fast" in out

    def test_skip_evaluate_short_circuits(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "speed_sec: 3.0\npeak_mb: 50.0\n", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task, skip_evaluate=True)
        assert "profile mode" in out
        assert "status:           ok" in out
        # only the pipeline phase ran
        assert len(fake.calls) == 1

    def test_evaluate_crash_keeps_pipeline_speed(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "speed_sec: 4.0\npeak_mb: 20.0\n", "", None, None, 0, 0.0),
            (1, "", "Error: metric blew up\n", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        assert "CRASH: evaluate exited 1" in out
        assert "Error: metric blew up" in out
        # pipeline speed retained in summary, not zeroed
        assert "speed_sec:        4.000" in out
        assert "status:           crash" in out

    def test_thread_budget_fork_violation_rejects(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        # peak_procs way over budget+1 → REJECT
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 9, 1.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task, thread=1)
        assert "REJECT: thread budget violation" in out
        assert "fork workers" in out
        assert "status:           crash" in out

    def test_thread_budget_cpu_violation_warns_only(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        # group_cpu_sec/wall huge but peak_procs within budget → WARN, keep
        fake = _FakeWatchdog(
            # wall ~ measured; cpu_sec=100 over budget*1.3 → cpu_violation
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 1, 100.0),
            (0, "", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task, thread=1)
        assert "in-process threads exceeded" in out
        # not rejected
        assert "REJECT" not in out
        assert "status:           ok" in out

    def test_thread_env_vars_injected(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (0, "", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        run_task(task, thread=4)
        env = fake.calls[0]["env"]
        assert env["ZYME_THREADS"] == "4"
        assert env["OMP_NUM_THREADS"] == "4"
        assert env["MKL_NUM_THREADS"] == "4"

    def test_dataset_env_injected_and_path_resolved(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        # relative path resolved against task dir
        (task / "data").mkdir()
        (task / "data" / "tiny.h5ad").write_text("x")
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (0, "", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        dataset = {"tier": "tiny", "name": "tiny_a", "path": "data/tiny.h5ad"}
        run_task(task, dataset_entry=dataset)
        env = fake.calls[0]["env"]
        assert env["ZYME_TIER"] == "tiny"
        assert env["ZYME_DATASET_NAME"] == "tiny_a"
        assert os.path.isabs(env["ZYME_DATA_PATH"])
        assert env["ZYME_DATA_PATH"].endswith("tiny.h5ad")

    def test_extra_env_merged(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (0, "", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        run_task(task, extra_env={"MY_FLAG": "yes"})
        assert fake.calls[0]["env"]["MY_FLAG"] == "yes"

    def test_legacy_flat_reference_output_layout(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path, with_ref=False)
        ref = task / "reference_output_tiny"
        ref.mkdir()
        (ref / "m.txt").write_text("x")
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (0, "", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task, dataset_entry={"tier": "tiny", "name": "t", "path": ""})
        assert str(ref) == fake.calls[0]["env"]["ZYME_REFERENCE_DIR"]
        assert "status:           ok" in out

    def test_executor_block_logged(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        # add an executor block so the `if executor:` log line fires
        (task / "task.yaml").write_text(
            "target_repo: stub\nexecutor:\n  rscript: /opt/R/bin/Rscript\n"
        )
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (0, "", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        assert "Exec:" in out

    def test_pipeline_python_args_inserted(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        run_task(task, skip_evaluate=True,
                 pipeline_python_args=["-m", "scalene", "--json"])
        cmd = fake.calls[0]["cmd"]
        assert cmd[1:4] == ["-m", "scalene", "--json"]
        assert cmd[-1] == "run.py"

    def test_evaluate_timeout_message(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (-9, "", "", None, 1200.0, 0, 0.0),  # evaluate timeout
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task, dataset_entry={"tier": "tiny", "name": "t", "path": ""})
        assert "TIMEOUT" in out
        assert "evaluate phase" in out

    def test_evaluate_mem_kill_message(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (-9, "", "", 9.0, None, 0, 0.0),  # evaluate killed by mem watchdog
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task, mem_cap_gb=4.0)
        assert "killed-by-mem-watchdog" in out
        assert "evaluate phase" in out

    def test_metric_allowlist_fallback_skip_keys(self, tmp_path, monkeypatch):
        # task.yaml declares NO metrics → falls back to the SKIP_KEYS blocklist.
        # Blocklisted keys (status, speed_sec) are dropped; others survive.
        task = _make_py_task(tmp_path)  # MINIMAL yaml has no metrics block
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (0, "pearson: 0.9\nstatus: bogus\nspeed_sec: 99\n", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        final_block = out[out.rfind("\n---\n"):]
        # pearson survives (not in SKIP_KEYS); status/speed_sec are blocklisted
        # from being scraped as a *metric* (real status/speed comes from summary)
        assert "pearson: 0.9" in final_block

    def test_duplicate_metric_key_only_recorded_once(self, tmp_path, monkeypatch):
        task = _make_py_task(
            tmp_path,
            metrics="metrics:\n  - {name: pearson, comparator: gte, threshold: 0.9}\n",
        )
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            # a free-text line that isn't `key: value` is skipped by the regex
            (0, "running evaluation now\npearson: 0.99\npearson: 0.88\n", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        final_block = out[out.rfind("\n---\n"):]
        # first occurrence wins; second is skipped (seen-set dedup)
        assert "pearson: 0.99" in final_block
        assert "pearson: 0.88" not in final_block

    def test_auto_structure_metrics_pass_through(self, tmp_path, monkeypatch):
        # Declared metrics present, but framework-emitted auto_structure
        # suffixes bypass the allowlist.
        task = _make_py_task(
            tmp_path,
            metrics="metrics:\n  - {name: pearson, comparator: gte, threshold: 0.9}\n",
        )
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (0, "pearson: 0.99\nslotA_present: 0.0\nundeclared: 1.0\n", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        final_block = out[out.rfind("\n---\n"):]
        assert "pearson: 0.99" in final_block
        assert "slotA_present: 0.0" in final_block  # auto_structure passes through
        assert "undeclared" not in final_block

    def test_per_tier_reference_outputs_layout(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path, with_ref=False)
        ref = task / "reference_outputs" / "medium"
        ref.mkdir(parents=True)
        (ref / "m.txt").write_text("x")
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (0, "", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task, dataset_entry={"tier": "medium", "name": "m", "path": ""})
        # resolved the per-tier ref dir; ZYME_REFERENCE_DIR points at it
        assert str(ref) == fake.calls[0]["env"]["ZYME_REFERENCE_DIR"]
        assert "status:           ok" in out


# --------------------------------------------------------------------------
# run_task — R-language branch
# --------------------------------------------------------------------------

def _make_r_task(tmp_path: Path) -> Path:
    task = tmp_path / "rtask"
    task.mkdir()
    (task / "pipeline").mkdir()
    (task / "pipeline" / "run.R").write_text("cat('pipeline\\n')\n")
    (task / "evaluate.R").write_text("cat('eval\\n')\n")
    (task / "task.yaml").write_text(
        "target_repo: stub\nexecutor:\n  rscript: /opt/R/bin/Rscript\n"
    )
    ref = task / "reference_output"
    ref.mkdir()
    (ref / "marker.txt").write_text("x")
    return task


class TestRunTaskRLanguage:
    def test_r_pipeline_and_evaluate_use_rscript(self, tmp_path, monkeypatch):
        task = _make_r_task(tmp_path)
        fake = _FakeWatchdog(
            (0, "speed_sec: 1.0\npeak_mb: 10.0\n", "", None, None, 0, 0.0),
            (0, "", "", None, None, 0, 0.0),
        )
        monkeypatch.setattr(runner, "_run_with_watchdog", fake)
        out = run_task(task)
        assert fake.calls[0]["cmd"] == ["/opt/R/bin/Rscript", "run.R"]
        assert fake.calls[1]["cmd"] == ["/opt/R/bin/Rscript", "evaluate.R"]
        assert "status:           ok" in out


# --------------------------------------------------------------------------
# dryrun_task — pipeline-only, returns the raw tuple
# --------------------------------------------------------------------------

class TestDryrunTask:
    def test_returns_tuple_and_runs_pipeline_only(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        calls = []

        def fake_run(cmd, *, cwd, capture_output, text, env, timeout):
            calls.append({"cmd": cmd, "cwd": cwd, "env": dict(env)})
            class P:
                returncode = 0
                stdout = "ran pipeline"
                stderr = ""
            return P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        rc, out, err, wall = dryrun_task(task)
        assert rc == 0
        assert out == "ran pipeline"
        assert wall >= 0.0
        # only one subprocess invocation (the pipeline)
        assert len(calls) == 1
        assert calls[0]["cmd"][-1] == "run.py"

    def test_timeout_returns_minus9_with_note(self, tmp_path, monkeypatch):
        import subprocess as _sp
        task = _make_py_task(tmp_path)

        def fake_run(cmd, *, cwd, capture_output, text, env, timeout):
            raise _sp.TimeoutExpired(cmd, timeout, output="partial", stderr="")

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        rc, out, err, wall = dryrun_task(
            task, dataset_entry={"tier": "tiny", "name": "t", "path": ""}
        )
        assert rc == -9
        assert out == "partial"
        assert "dryrun killed" in err

    def test_dataset_env_injected(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        seen = {}

        def fake_run(cmd, *, cwd, capture_output, text, env, timeout):
            seen.update(env)
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            return P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        dryrun_task(task, dataset_entry={"tier": "tiny", "name": "n", "path": "/abs/p"})
        assert seen["ZYME_TIER"] == "tiny"
        assert seen["ZYME_DATASET_NAME"] == "n"
        assert seen["ZYME_DATA_PATH"] == "/abs/p"

    def test_r_task_uses_rscript(self, tmp_path, monkeypatch):
        task = _make_r_task(tmp_path)
        seen = {}

        def fake_run(cmd, *, cwd, capture_output, text, env, timeout):
            seen["cmd"] = cmd
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            return P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        dryrun_task(task)
        assert seen["cmd"] == ["/opt/R/bin/Rscript", "run.R"]

    def test_per_tier_ref_dir_resolved(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path, with_ref=False)
        ref = task / "reference_outputs" / "medium"
        ref.mkdir(parents=True)
        (ref / "m.txt").write_text("x")
        seen = {}

        def fake_run(cmd, *, cwd, capture_output, text, env, timeout):
            seen.update(env)
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            return P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        dryrun_task(task, dataset_entry={"tier": "medium", "name": "m", "path": ""})
        assert seen["ZYME_REFERENCE_DIR"] == str(ref)

    def test_legacy_flat_ref_dir_resolved(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path, with_ref=False)
        ref = task / "reference_output_medium"
        ref.mkdir()
        seen = {}

        def fake_run(cmd, *, cwd, capture_output, text, env, timeout):
            seen.update(env)
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            return P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        dryrun_task(task, dataset_entry={"tier": "medium", "name": "m", "path": ""})
        assert seen["ZYME_REFERENCE_DIR"] == str(ref)

    def test_extra_env_merged(self, tmp_path, monkeypatch):
        task = _make_py_task(tmp_path)
        seen = {}

        def fake_run(cmd, *, cwd, capture_output, text, env, timeout):
            seen.update(env)
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            return P()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        dryrun_task(task, extra_env={"DRY_FLAG": "1"})
        assert seen["DRY_FLAG"] == "1"


# --------------------------------------------------------------------------
# _run_with_watchdog — fast path (no watchdog / observer) via real subprocess
# --------------------------------------------------------------------------

class TestRunWithWatchdogFastPath:
    def test_clean_exit(self):
        rc, out, err, killed, timed, procs, cpu = runner._run_with_watchdog(
            [sys.executable, "-c", "print('hello')"],
            cwd=None, env=os.environ.copy(), mem_cap_gb=None,
        )
        assert rc == 0
        assert "hello" in out
        assert killed is None
        assert timed is None
        assert procs == 0

    def test_nonzero_exit(self):
        rc, out, err, killed, timed, procs, cpu = runner._run_with_watchdog(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            cwd=None, env=os.environ.copy(), mem_cap_gb=None,
        )
        assert rc == 3

    def test_wall_cap_timeout_fast_path(self):
        # On POSIX, a wall cap forces the Popen path; this still kills on
        # timeout and returns timed_out_at_s set. We use a tiny cap.
        rc, out, err, killed, timed, procs, cpu = runner._run_with_watchdog(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=None, env=os.environ.copy(), mem_cap_gb=None,
            wall_cap_s=0.5,
        )
        assert timed == 0.5
        assert rc != 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group audit only")
class TestRunWithWatchdogPopenPath:
    def test_audit_threads_collects_stats(self):
        # audit_threads forces the Popen + poller path even with no mem cap.
        # A short-lived child should exit cleanly; we just confirm the poller
        # ran and returned the extended tuple without error.
        rc, out, err, killed, timed, procs, cpu = runner._run_with_watchdog(
            [sys.executable, "-c", "print('audited')"],
            cwd=None, env=os.environ.copy(), mem_cap_gb=None,
            audit_threads=True, thread_budget=2, poll_s=0.05,
        )
        assert rc == 0
        assert "audited" in out
        assert killed is None
        # peak_procs / cpu_sec are non-negative ints/floats from the poller
        assert procs >= 0
        assert cpu >= 0.0

    def test_mem_cap_does_not_kill_small_proc(self):
        # A trivial child stays well under a generous cap → not killed.
        rc, out, err, killed, timed, procs, cpu = runner._run_with_watchdog(
            [sys.executable, "-c", "print('ok')"],
            cwd=None, env=os.environ.copy(), mem_cap_gb=64.0,
            poll_s=0.05,
        )
        assert rc == 0
        assert killed is None

    def test_side_observer_attach_detach_called(self):
        class Obs:
            def __init__(self):
                self.attached = None
                self.detached = False
            def attach(self, pid):
                self.attached = pid
            def detach(self):
                self.detached = True
        obs = Obs()
        rc, *_ = runner._run_with_watchdog(
            [sys.executable, "-c", "print('hi')"],
            cwd=None, env=os.environ.copy(), mem_cap_gb=None,
            side_observer=obs, poll_s=0.05,
        )
        assert rc == 0
        assert obs.attached is not None
        assert obs.detached is True

    def test_side_observer_attach_failure_is_swallowed(self):
        # attach()/detach() failures are best-effort: they print a warning but
        # must not break the run.
        class BadObs:
            def attach(self, pid):
                raise RuntimeError("attach boom")
            def detach(self):
                raise RuntimeError("detach boom")
        rc, out, *_ = runner._run_with_watchdog(
            [sys.executable, "-c", "print('survived')"],
            cwd=None, env=os.environ.copy(), mem_cap_gb=None,
            side_observer=BadObs(), poll_s=0.05,
        )
        assert rc == 0
        assert "survived" in out

    def test_mem_watchdog_kills_overcap_child(self):
        # Allocate well past a 0.05 GB cap; the poller should SIGKILL the group
        # and report killed_peak_gb. Kept tiny + fast so it can't thrash the host.
        code = (
            "import time\n"
            "buf = bytearray(400 * 1024 * 1024)\n"  # ~400 MB, over 50 MB cap
            "time.sleep(10)\n"
        )
        rc, out, err, killed, timed, procs, cpu = runner._run_with_watchdog(
            [sys.executable, "-c", code],
            cwd=None, env=os.environ.copy(), mem_cap_gb=0.05,
            poll_s=0.05,
        )
        # Either the watchdog caught it (killed set) or the child finished
        # before a poll (rare); assert the watchdog path was taken and, if it
        # fired, returned a peak.
        assert rc != 0
        if killed is not None:
            assert killed > 0.0
