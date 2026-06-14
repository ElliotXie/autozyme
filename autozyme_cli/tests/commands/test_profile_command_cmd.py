"""Unit tests for zyme.commands.profile.command — orchestration helpers plus
an end-to-end cmd_profile drive with the runner boundary monkeypatched.

The real cmd_profile spawns a pipeline subprocess via runner.run_task; we
monkeypatch run_task (and the cProfile artifact it would normally produce) so
the orchestration glue — backend resolve -> archive dir -> normalize ->
enrich -> render/JSON -> latest symlink — is exercised without a subprocess.

Pure helpers (_extract_totals, _python_pipeline_uses_with_profile,
_top_hotspot_opaque_hint) are unit-tested directly. The with_profile detector
has partial coverage in test_profile_memray_scope.py; we add the AST edge
cases it leaves.
"""
from __future__ import annotations

import cProfile
import json
import pstats
from pathlib import Path

import pytest

from zyme.commands.profile import command


# ---------------------------------------------------------------------------
# _extract_totals — parse the runner summary block.
# ---------------------------------------------------------------------------

def test_extract_totals_all_fields():
    log = (
        "some preamble\n"
        "speed_sec: 12.5\n"
        "peak_mb: 256.0\n"
        "cpu_sec: 11.0\n"
        "trailing line\n"
    )
    totals = command._extract_totals(log)
    assert totals == {"wall_s": 12.5, "peak_mb": 256.0, "cpu_s": 11.0}


def test_extract_totals_partial():
    totals = command._extract_totals("speed_sec: 3.0\n")
    assert totals == {"wall_s": 3.0}


def test_extract_totals_ignores_non_numeric():
    totals = command._extract_totals("speed_sec: not_a_number\npeak_mb: 10\n")
    assert "wall_s" not in totals
    assert totals["peak_mb"] == 10.0


def test_extract_totals_regex_match_but_float_fails():
    # "1.2.3" / "+-" match the [0-9.eE+-]+ charset but float() rejects them,
    # exercising the except-ValueError continue path.
    totals = command._extract_totals("speed_sec: 1.2.3\npeak_mb: +-\ncpu_sec: 2.0\n")
    assert "wall_s" not in totals
    assert "peak_mb" not in totals
    assert totals["cpu_s"] == 2.0


def test_extract_totals_ignores_unknown_keys():
    totals = command._extract_totals("foobar: 9.9\nspeed_sec: 1.0\n")
    assert totals == {"wall_s": 1.0}


def test_extract_totals_empty_log():
    assert command._extract_totals("") == {}


def test_extract_totals_scientific_notation():
    totals = command._extract_totals("speed_sec: 1.5e1\n")
    assert totals["wall_s"] == 15.0


# ---------------------------------------------------------------------------
# _python_pipeline_uses_with_profile — AST detection edge cases.
# (test_profile_memray_scope covers the basic Name-call + comment cases;
#  here we add missing-file, syntax-error, and attribute-call variants.)
# ---------------------------------------------------------------------------

def test_with_profile_missing_file(tmp_path):
    assert command._python_pipeline_uses_with_profile(tmp_path / "absent.py") is False


def test_with_profile_syntax_error(tmp_path):
    p = tmp_path / "run.py"
    p.write_text("def broken(:\n")
    assert command._python_pipeline_uses_with_profile(p) is False


def test_with_profile_attribute_call(tmp_path):
    p = tmp_path / "run.py"
    p.write_text(
        "import helpers\n"
        "with helpers.with_profile():\n"
        "    pass\n"
    )
    assert command._python_pipeline_uses_with_profile(p) is True


def test_with_profile_no_call_present(tmp_path):
    p = tmp_path / "run.py"
    p.write_text("x = 1\nprint(x)\n")
    assert command._python_pipeline_uses_with_profile(p) is False


# ---------------------------------------------------------------------------
# _top_hotspot_opaque_hint
# ---------------------------------------------------------------------------

def test_opaque_hint_none_when_empty():
    assert command._top_hotspot_opaque_hint([], "cpu") is None


def test_opaque_hint_none_for_native_backend():
    actionable = [{"label": ".Call", "raw": {}}]
    assert command._top_hotspot_opaque_hint(actionable, "native") is None


def test_opaque_hint_native_frame_dot_call():
    actionable = [{"label": "x.Call y", "raw": {}}]
    hint = command._top_hotspot_opaque_hint(actionable, "cpu")
    assert hint is not None
    assert "opaque native frame" in hint
    assert "--backend native" in hint


def test_opaque_hint_psynch_cvwait():
    actionable = [{"label": "psynch_cvwait", "raw": {}}]
    hint = command._top_hotspot_opaque_hint(actionable, "cpu")
    assert hint is not None
    assert "native" in hint


def test_opaque_hint_worker_wait_frame():
    actionable = [{"label": "multiprocessing.pool.worker", "raw": {}}]
    hint = command._top_hotspot_opaque_hint(actionable, "cpu")
    assert hint is not None
    assert "multiprocessing" in hint
    assert "parent-side wait frame" in hint


def test_opaque_hint_normal_frame_no_hint():
    actionable = [{"label": "run.py:10:my_kernel", "raw": {"func": "my_kernel"}}]
    assert command._top_hotspot_opaque_hint(actionable, "cpu") is None


def test_opaque_hint_reads_raw_values():
    actionable = [{"label": "innocent", "raw": {"detail": "workq_kernreturn here"}}]
    hint = command._top_hotspot_opaque_hint(actionable, "cpu")
    assert hint is not None


# ---------------------------------------------------------------------------
# cmd_profile end-to-end with the runner monkeypatched.
# ---------------------------------------------------------------------------

_TASK_YAML = """\
target_repo: https://example.com/foo
target_function: myalgo.run

datasets:
  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}

metrics:
  - {name: speedup, comparator: gte, threshold: 1.0}
"""


class _Args:
    def __init__(self, **kw):
        defaults = dict(
            task_dir=None, backend="cpu", dataset=None, hypothesis="",
            no_archive=False, json=False, diff_a=None,
        )
        defaults.update(kw)
        self.__dict__.update(defaults)


def _make_task(tmp_path: Path) -> Path:
    (tmp_path / "task.yaml").write_text(_TASK_YAML)
    (tmp_path / ".zyme").mkdir()
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "run.py").write_text("print('hi')\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tiny.h5ad").write_text("placeholder")
    return tmp_path


def _fake_run_task_factory(profile_dir_holder):
    """Return a run_task stub that drops a real cProfile profile.out into the
    profile_history run dir (so parsers.normalize finds a parseable artifact)
    and returns a summary log."""
    def fake_run_task(task_dir, dataset_entry=None, extra_env=None,
                      pipeline_python_args=None, skip_evaluate=False,
                      side_observer=None):
        prof_dir = Path(extra_env["ZYME_PROFILE_DIR"])
        profile_dir_holder["dir"] = prof_dir
        prof = cProfile.Profile()
        prof.enable()
        s = 0
        for i in range(50_000):
            s += i * i
        prof.disable()
        pstats.Stats(prof).dump_stats(str(prof_dir / "profile.out"))
        return "speed_sec: 1.0\npeak_mb: 64.0\ncpu_sec: 0.9\n"
    return fake_run_task


def test_cmd_profile_cpu_end_to_end_json(tmp_path, monkeypatch, capsys):
    task = _make_task(tmp_path)
    holder = {}
    monkeypatch.setattr(command, "run_task", _fake_run_task_factory(holder))
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), json=True))

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["backend"] == "cpu"
    assert data["lang"] == "py"
    assert data["tier"] == "tiny"
    assert data["schema_version"] == "2"   # enrich() bumped it
    assert data["totals"]["wall_s"] == 1.0
    assert isinstance(data["hotspots"], list) and data["hotspots"]
    # enrichment fields present.
    assert "layer_breakdown" in data
    # artifacts include the normalized json + run log paths.
    assert "profile_json" in data["artifacts"]
    assert "run_log" in data["artifacts"]
    # archive dir was created with profile.json + run.log + raw artifact.
    run_dir = holder["dir"]
    assert (run_dir / "profile.json").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "profile.out").exists()
    # latest symlink updated (archived run).
    latest = task / "profile_history" / "latest"
    assert latest.exists()


def test_cmd_profile_evidence_card_to_stderr(tmp_path, monkeypatch, capsys):
    task = _make_task(tmp_path)
    holder = {}
    monkeypatch.setattr(command, "run_task", _fake_run_task_factory(holder))
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), json=False))

    captured = capsys.readouterr()
    assert "[profile]" in captured.err
    assert "backend=cpu" in captured.err


def test_cmd_profile_no_archive_uses_current(tmp_path, monkeypatch, capsys):
    task = _make_task(tmp_path)
    holder = {}
    monkeypatch.setattr(command, "run_task", _fake_run_task_factory(holder))
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), json=True, no_archive=True))

    capsys.readouterr()
    assert holder["dir"].name == "current"
    # no latest symlink for --no-archive runs.
    assert not (task / "profile_history" / "latest").exists()


def test_cmd_profile_dispatches_to_diff(tmp_path, monkeypatch, capsys):
    """When diff_a is set, cmd_profile bypasses capture entirely."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    base = {"backend": "cpu", "lang": "py", "tier": "tiny", "hypothesis": "",
            "timestamp": "t", "totals": {}, "hotspots": [], "override_summary": []}
    a.write_text(json.dumps(base))
    b.write_text(json.dumps(base))
    # run_task must NOT be called in diff mode.
    called = {"run": False}
    monkeypatch.setattr(command, "run_task",
                        lambda *a, **k: called.__setitem__("run", True) or "")

    command.cmd_profile(_Args(task_dir=str(tmp_path), diff_a=str(a),
                              diff_b=str(b), json=True))
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "common_hotspots" in parsed
    assert called["run"] is False


def test_cmd_profile_full_falls_back_to_cpu_with_banner(tmp_path, monkeypatch, capsys):
    task = _make_task(tmp_path)
    holder = {}
    monkeypatch.setattr(command, "run_task", _fake_run_task_factory(holder))
    # force scalene-missing so full -> cpu fallback fires.
    from zyme.commands.profile import backends
    monkeypatch.setattr(backends, "_probe_python_pkg", lambda pkg, ex: False)
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), backend="full", json=True))

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    # effective backend fell back to cpu.
    assert data["backend"] == "cpu"
    # loud fallback banner routed to stderr in json mode.
    assert "scalene" in captured.err.lower()


def test_cmd_profile_mem_refusal_calls_die(tmp_path, monkeypatch):
    task = _make_task(tmp_path)
    from zyme.commands.profile import backends
    monkeypatch.setattr(backends, "_probe_python_pkg", lambda pkg, ex: False)
    monkeypatch.chdir(task)
    # backend=mem with memray missing -> resolve raises -> die() -> SystemExit.
    with pytest.raises(SystemExit):
        command.cmd_profile(_Args(task_dir=str(task), backend="mem", json=True))


def _stub_normalize(monkeypatch):
    """Replace parsers.normalize with a stub returning a minimal valid dict,
    so cmd_profile's mem/full branches can be driven without producing the
    real backend artifact."""
    from zyme.commands.profile import parsers

    def fake_normalize(backend, lang, artifact_dir, hypothesis, tier,
                       artifact_prefix, executor=None, totals=None):
        return {
            "schema_version": "1", "backend": backend, "lang": lang,
            "tier": tier, "hypothesis": hypothesis, "timestamp": "t",
            "totals": totals or {}, "hotspots": [], "actionable_hotspots": [],
            "call_chains": [], "notes": [], "artifacts": {},
        }

    monkeypatch.setattr(parsers, "normalize", fake_normalize)


def test_cmd_profile_mem_scoped_branch(tmp_path, monkeypatch, capsys):
    """backend=mem with a with_profile() region -> scoped env + scope note."""
    task = _make_task(tmp_path)
    # give run.py a with_profile() call so the scoped path fires.
    (task / "pipeline" / "run.py").write_text(
        "from helpers import with_profile\n"
        "with with_profile():\n    pass\n"
    )
    from zyme.commands.profile import backends
    monkeypatch.setattr(backends, "_probe_python_pkg", lambda pkg, ex: True)
    _stub_normalize(monkeypatch)

    captured_env = {}

    def fake_run_task(task_dir, dataset_entry=None, extra_env=None,
                      pipeline_python_args=None, skip_evaluate=False,
                      side_observer=None):
        captured_env.update(extra_env or {})
        return "scope=with_profile\nspeed_sec: 1.0\n"

    monkeypatch.setattr(command, "run_task", fake_run_task)
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), backend="mem", json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "mem"
    assert captured_env.get("ZYME_MEMRAY_SCOPED") == "1"
    assert any("scope=with_profile" in n for n in out["notes"])


def test_cmd_profile_mem_whole_process_branch(tmp_path, monkeypatch, capsys):
    """backend=mem WITHOUT a with_profile() region -> whole-process wrapper."""
    task = _make_task(tmp_path)  # run.py has no with_profile()
    from zyme.commands.profile import backends
    monkeypatch.setattr(backends, "_probe_python_pkg", lambda pkg, ex: True)
    _stub_normalize(monkeypatch)

    captured = {}

    def fake_run_task(task_dir, dataset_entry=None, extra_env=None,
                      pipeline_python_args=None, skip_evaluate=False,
                      side_observer=None):
        captured["env"] = extra_env or {}
        captured["pyargs"] = pipeline_python_args
        return "speed_sec: 1.0\n"

    monkeypatch.setattr(command, "run_task", fake_run_task)
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), backend="mem", json=True))
    out = json.loads(capsys.readouterr().out)
    assert captured["env"].get("ZYME_MEMRAY_ACTIVE") == "1"
    # whole-process wrapper inserts the memray run args.
    assert "memray" in captured["pyargs"]
    assert any("scope=whole_process" in n for n in out["notes"])


def test_cmd_profile_full_scalene_branch(tmp_path, monkeypatch, capsys):
    """backend=full with scalene present -> scalene wrapper args + env."""
    task = _make_task(tmp_path)
    from zyme.commands.profile import backends
    monkeypatch.setattr(backends, "_probe_python_pkg", lambda pkg, ex: True)
    _stub_normalize(monkeypatch)

    captured = {}

    def fake_run_task(task_dir, dataset_entry=None, extra_env=None,
                      pipeline_python_args=None, skip_evaluate=False,
                      side_observer=None):
        captured["env"] = extra_env or {}
        captured["pyargs"] = pipeline_python_args
        return "speed_sec: 1.0\n"

    monkeypatch.setattr(command, "run_task", fake_run_task)
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), backend="full", json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "full"
    assert captured["env"].get("ZYME_SCALENE_ACTIVE") == "1"
    assert "scalene" in captured["pyargs"]


def test_cmd_profile_invalid_dataset_dies(tmp_path, monkeypatch):
    """An unknown --dataset tier -> resolve_tiers raises ValueError -> die()."""
    task = _make_task(tmp_path)
    monkeypatch.chdir(task)
    with pytest.raises(SystemExit):
        command.cmd_profile(_Args(task_dir=str(task), dataset="no_such_tier",
                                  json=True))


def test_cmd_profile_native_dispatch(tmp_path, monkeypatch, capsys):
    """backend=native -> NativeSampler side-observer + native.parse_and_normalize.
    Both are stubbed so no /usr/bin/sample subprocess runs."""
    task = _make_task(tmp_path)
    from zyme.commands.profile import native, backends

    # make native resolve succeed regardless of host.
    monkeypatch.setattr(native, "is_supported", lambda: (True, ""))

    class _StubSampler:
        def __init__(self, out_dir):
            self.out_dir = out_dir

    monkeypatch.setattr(native, "NativeSampler", _StubSampler)

    captured = {}

    def fake_parse_and_normalize(out_dir, sampler, lang, tier, hypothesis,
                                 totals=None, artifact_prefix="x"):
        captured["sampler"] = sampler
        return {
            "schema_version": "1", "backend": "native", "lang": lang,
            "tier": tier, "hypothesis": hypothesis, "timestamp": "t",
            "totals": totals or {}, "hotspots": [], "actionable_hotspots": [],
            "override_summary": [], "override_markers": [], "call_chains": [],
            "notes": [], "artifacts": {},
        }

    monkeypatch.setattr(native, "parse_and_normalize", fake_parse_and_normalize)

    def fake_run_task(task_dir, dataset_entry=None, extra_env=None,
                      pipeline_python_args=None, skip_evaluate=False,
                      side_observer=None):
        captured["side_observer"] = side_observer
        # native backend must NOT set ZYME_PROFILE env.
        captured["env"] = extra_env or {}
        return "speed_sec: 1.0\n"

    monkeypatch.setattr(command, "run_task", fake_run_task)
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), backend="native", json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "native"
    # native skips ZYME_PROFILE entirely (external observer).
    assert "ZYME_PROFILE" not in captured["env"]
    assert isinstance(captured["side_observer"], _StubSampler)
    assert captured["sampler"] is captured["side_observer"]


def test_cmd_profile_opaque_top_hotspot_appends_hint(tmp_path, monkeypatch, capsys):
    """When the top actionable hotspot is an opaque native frame, cmd_profile
    appends the 'rerun with --backend native' hint note."""
    task = _make_task(tmp_path)
    holder = {}
    monkeypatch.setattr(command, "run_task", _fake_run_task_factory(holder))

    from zyme.commands.profile import parsers

    def fake_actionable(profile_data, top_n=15):
        return ([{"rank": 1, "label": "psynch_cvwait", "raw": {}}], [])

    monkeypatch.setattr(parsers, "build_actionable_hotspots", fake_actionable)
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), json=True))
    out = json.loads(capsys.readouterr().out)
    assert any("opaque native frame" in n for n in out["notes"])


def test_cmd_profile_enrichment_failure_is_swallowed(tmp_path, monkeypatch, capsys):
    """An exception inside enrich() must not fail the run — it becomes a note."""
    task = _make_task(tmp_path)
    holder = {}
    monkeypatch.setattr(command, "run_task", _fake_run_task_factory(holder))

    from zyme.commands.profile import enrich as profile_enrich

    def boom(*a, **k):
        raise RuntimeError("enrich exploded")

    monkeypatch.setattr(profile_enrich, "enrich", boom)
    monkeypatch.chdir(task)

    command.cmd_profile(_Args(task_dir=str(task), json=True))
    out = json.loads(capsys.readouterr().out)
    assert any("enrichment skipped" in n for n in out["notes"])
    # the run still completes and writes profile.json.
    assert (holder["dir"] / "profile.json").exists()
