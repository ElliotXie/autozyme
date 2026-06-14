"""Unit tests for zyme.commands.init and zyme.commands.init_check.

init.py        — _is_repo_url, _detect_language sniffing, and cmd_init scaffolding
                 driven into tmp_path with git + bench-template boundaries stubbed.
init_check.py  — _tick, _baseline_present, _ref_output_nonempty, and the
                 cmd_init_check checklist over assembled task fixtures
                 (scaffold gaps, full-pass exit 0, parity-check stub).
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import init as initmod
from zyme.commands import init_check as ic


# ==========================================================================
# init.py — pure helpers
# ==========================================================================

class TestIsRepoUrl:
    @pytest.mark.parametrize("s", [
        "https://github.com/x/y", "http://x", "git@github.com:x/y.git",
        "git://x", "ssh://x",
    ])
    def test_urls(self, s):
        assert initmod._is_repo_url(s) is True

    @pytest.mark.parametrize("s", ["/local/path", "./rel", "mypkg", "C:\\win"])
    def test_local_paths(self, s):
        assert initmod._is_repo_url(s) is False


class TestDetectLanguage:
    def test_r_from_description(self, tmp_path):
        d = tmp_path / "repo"
        d.mkdir()
        (d / "DESCRIPTION").write_text("Package: foo\n")
        assert initmod._detect_language(d) == "R"

    def test_python_from_pyproject(self, tmp_path):
        d = tmp_path / "repo"
        d.mkdir()
        (d / "pyproject.toml").write_text("[project]\n")
        assert initmod._detect_language(d) == "python"

    def test_python_from_setup_py(self, tmp_path):
        d = tmp_path / "repo"
        d.mkdir()
        (d / "setup.py").write_text("from setuptools import setup\n")
        assert initmod._detect_language(d) == "python"

    def test_none_when_unknown(self, tmp_path):
        d = tmp_path / "repo"
        d.mkdir()
        assert initmod._detect_language(d) is None

    def test_first_candidate_wins(self, tmp_path):
        r = tmp_path / "r_repo"
        r.mkdir()
        (r / "DESCRIPTION").write_text("Package: foo\n")
        py = tmp_path / "py_repo"
        py.mkdir()
        (py / "pyproject.toml").write_text("[project]\n")
        assert initmod._detect_language(r, py) == "R"

    def test_skips_none_and_missing(self, tmp_path):
        py = tmp_path / "py_repo"
        py.mkdir()
        (py / "setup.py").write_text("x")
        assert initmod._detect_language(None, tmp_path / "absent", py) == "python"


# ==========================================================================
# init.py — cmd_init scaffolding
# ==========================================================================

def _make_template(tmp_path):
    """Build a minimal templates/task_template/ tree and point FRAMEWORK_ROOT at it."""
    fw = tmp_path / "fw"
    tpl = fw / "templates" / "task_template"
    (tpl / "pipeline").mkdir(parents=True)
    (tpl / "task.yaml").write_text(
        "task: <TASK_NAME>\n"
        "target_repo: <TARGET_REPO_URL_OR_LOCAL_PATH>\n"
        "target_function: <PKG::FUNC>\n"
        "signature: <CALL_SIGNATURE>\n"
    )
    (tpl / "evaluate.py.template").write_text("# eval py")
    (tpl / "evaluate.R.template").write_text("# eval R")
    (tpl / "reference.py.template").write_text("# ref py")
    (tpl / "reference.R.template").write_text("# ref R")
    (tpl / "pipeline" / "run.py.template").write_text("# run py")
    (tpl / "pipeline" / "run.R.template").write_text("# run R")
    # prompt set
    prompts = fw / "prompts" / "Bio"
    prompts.mkdir(parents=True)
    (prompts / "1_init.md").write_text("# init")
    (prompts / "2_iterate.md").write_text("# iterate")
    (prompts / "3_scaling.md").write_text("# scaling")
    (prompts / "README.md").write_text("# readme — skipped")
    return fw


class TestCmdInit:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        fw = _make_template(tmp_path)
        monkeypatch.setattr(initmod, "FRAMEWORK_ROOT", fw)
        # Stub git so no real repo / commits happen; report not-inside-repo.
        calls = []

        def fake_git(*args, cwd=None, check=True):
            calls.append(args)
            if args[:2] == ("rev-parse", "--is-inside-work-tree"):
                return "false"  # triggers git init path
            return ""
        monkeypatch.setattr(initmod, "git", fake_git)
        # Stub bench register so no real template snapshot is taken.
        import zyme.commands.bench as bench
        monkeypatch.setattr(bench, "_bench_template_path",
                            lambda stage, name: tmp_path / "no_such_template")
        monkeypatch.setattr(bench, "cmd_bench_register_template",
                            lambda ns: None)
        task_dir = tmp_path / "test_mytask"
        task_dir.mkdir()
        monkeypatch.chdir(task_dir)
        return SimpleNamespace(fw=fw, task_dir=task_dir, git_calls=calls)

    def _args(self, **kw):
        base = dict(target_repo="/local/path", target_function="mgcv::gam",
                    language="python", dataset=None, no_clone=True,
                    field="Bio", no_bench_snapshot=True)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_scaffolds_python_task(self, env, capsys):
        initmod.cmd_init(self._args())
        td = env.task_dir
        # task.yaml substituted
        yaml_text = (td / "task.yaml").read_text()
        assert "test_mytask" in yaml_text
        assert "/local/path" in yaml_text
        assert "mgcv::gam" in yaml_text
        # python templates renamed, R ones dropped
        assert (td / "evaluate.py").is_file()
        assert not (td / "evaluate.py.template").exists()
        assert not (td / "evaluate.R.template").exists()
        assert (td / "pipeline" / "run.py").is_file()
        # scaffold dirs
        for sub in ("data", "setup", "reference_outputs", "memory"):
            assert (td / sub).is_dir()
        # memory skeleton
        assert (td / "memory" / "discoveries.md").is_file()
        # family.md
        assert (td / "family.md").is_file()
        # prompts copied (README skipped)
        assert (td / "prompts" / "1_init.md").is_file()
        assert not (td / "prompts" / "README.md").exists()

    def test_refuses_existing_task(self, env):
        (env.task_dir / "task.yaml").write_text("already here")
        with pytest.raises(SystemExit):
            initmod.cmd_init(self._args())

    def test_r_language_keeps_r_drops_py(self, env):
        initmod.cmd_init(self._args(language="R"))
        td = env.task_dir
        assert (td / "evaluate.R").is_file()
        assert not (td / "evaluate.py.template").exists()
        assert (td / "pipeline" / "run.R").is_file()

    def test_dataset_hint_inserted(self, env):
        initmod.cmd_init(self._args(dataset="/data/x.h5ad"))
        yaml_text = (env.task_dir / "task.yaml").read_text()
        assert "dataset_hint: /data/x.h5ad" in yaml_text

    def test_unknown_field_dies(self, env):
        with pytest.raises(SystemExit):
            initmod.cmd_init(self._args(field="NoSuchField"))

    def test_language_none_leaves_templates(self, env, monkeypatch):
        # Local path that can't be sniffed -> language None -> .template kept.
        initmod.cmd_init(self._args(language=None))
        td = env.task_dir
        # templates remain because no language was detected
        assert (td / "evaluate.py.template").exists()

    def test_clone_url_invokes_git_clone(self, env, monkeypatch):
        # target_repo is a URL + no_clone False -> git clone is attempted, and
        # the cloned dir is sniffed for language.
        def fake_git(*args, cwd=None, check=True):
            env.git_calls.append(args)
            if args[:2] == ("rev-parse", "--is-inside-work-tree"):
                return "false"
            if args[0] == "clone":
                # simulate a successful clone of a python repo
                dest = Path(args[2])
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "pyproject.toml").write_text("[project]\n")
            return ""
        monkeypatch.setattr(initmod, "git", fake_git)
        initmod.cmd_init(self._args(
            target_repo="https://github.com/x/y", no_clone=False, language=None))
        assert any(c[0] == "clone" for c in env.git_calls)
        # language sniffed from the cloned repo -> python templates renamed
        assert (env.task_dir / "evaluate.py").is_file()

    def test_clone_failure_is_nonfatal(self, env, monkeypatch):
        def fake_git(*args, cwd=None, check=True):
            if args[:2] == ("rev-parse", "--is-inside-work-tree"):
                return "false"
            if args[0] == "clone":
                raise RuntimeError("clone refused")
            return ""
        monkeypatch.setattr(initmod, "git", fake_git)
        # Should not raise — clone failure only warns.
        initmod.cmd_init(self._args(
            target_repo="https://github.com/x/y", no_clone=False))

    def test_bench_snapshot_registered(self, env, monkeypatch):
        # no_bench_snapshot False + freshly-init'd repo -> auto-register a
        # bench template (when one doesn't already exist).
        import zyme.commands.bench as bench
        registered = {}
        monkeypatch.setattr(bench, "_bench_template_path",
                            lambda stage, name: env.task_dir / "ghost_template")
        monkeypatch.setattr(
            bench, "cmd_bench_register_template",
            lambda ns: registered.setdefault("name", ns.name))
        initmod.cmd_init(self._args(no_bench_snapshot=False))
        assert registered.get("name") == "test_mytask"


# ==========================================================================
# init_check.py — pure helpers
# ==========================================================================

class TestTick:
    def test_pass(self):
        assert ic._OK in ic._tick(True)

    def test_fail(self):
        assert ic._NO in ic._tick(False)


class TestBaselinePresent:
    def test_found(self):
        rows = [{"status": "baseline", "dataset": "ds", "thread": "1",
                 "speed_sec": "10.0"}]
        assert ic._baseline_present(rows, "ds", 1) is True

    def test_wrong_dataset(self):
        rows = [{"status": "baseline", "dataset": "other", "thread": "1",
                 "speed_sec": "10.0"}]
        assert ic._baseline_present(rows, "ds", 1) is False

    def test_zero_speed_not_counted(self):
        rows = [{"status": "baseline", "dataset": "ds", "thread": "1",
                 "speed_sec": "0"}]
        assert ic._baseline_present(rows, "ds", 1) is False

    def test_default_thread_one(self):
        rows = [{"status": "baseline", "dataset": "ds", "speed_sec": "5.0"}]
        # missing thread -> treated as "1"
        assert ic._baseline_present(rows, "ds", 1) is True

    def test_bad_speed_skipped(self):
        rows = [{"status": "baseline", "dataset": "ds", "thread": "1",
                 "speed_sec": "notnum"}]
        assert ic._baseline_present(rows, "ds", 1) is False


class TestRefOutputNonempty:
    def test_nested_layout(self, tmp_path, monkeypatch):
        ref = tmp_path / "reference_outputs" / "tiny"
        ref.mkdir(parents=True)
        (ref / "out.pkl").write_text("x")
        monkeypatch.setattr(ic, "resolve_reference_output_dir",
                            lambda td, tier: ref)
        assert ic._ref_output_nonempty(tmp_path, "tiny") is True

    def test_flat_legacy_layout(self, tmp_path, monkeypatch):
        # primary empty, legacy flat dir populated
        primary = tmp_path / "reference_outputs" / "tiny"
        primary.mkdir(parents=True)
        legacy = tmp_path / "reference_output_tiny"
        legacy.mkdir()
        (legacy / "o").write_text("x")
        monkeypatch.setattr(ic, "resolve_reference_output_dir",
                            lambda td, tier: primary)
        assert ic._ref_output_nonempty(tmp_path, "tiny") is True

    def test_empty_returns_false(self, tmp_path, monkeypatch):
        primary = tmp_path / "reference_outputs" / "tiny"
        primary.mkdir(parents=True)
        monkeypatch.setattr(ic, "resolve_reference_output_dir",
                            lambda td, tier: primary)
        assert ic._ref_output_nonempty(tmp_path, "tiny") is False


# ==========================================================================
# init_check.py — cmd_init_check
# ==========================================================================

class TestCmdInitCheck:
    def _task(self, tmp_path, *, datasets=True, baseline=True, ref=True,
              stochastic=False):
        td = tmp_path / "test_x"
        (td / "pipeline").mkdir(parents=True)
        ds_block = (
            "datasets:\n  - {tier: tiny, name: ds, path: data/x}\n"
            if datasets else "")
        algo = "algorithm_class: stochastic\n" if stochastic else ""
        noise = (
            "intrinsic_noise:\n  tiny: {max_diff: 0.001}\n"
            if stochastic else "")
        (td / "task.yaml").write_text(
            "target_function: foo\n"
            "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}\n"
            "baseline_threads: [1]\n"
            + algo + ds_block + noise)
        (td / "reference.py").write_text("x")
        (td / "pipeline" / "run.py").write_text("x")
        (td / "evaluate.py").write_text("x")
        if baseline:
            (td / "results.tsv").write_text(
                "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\t"
                "status\tmetrics_json\thypothesis\tdescription\tphase\tthread\n"
                "0\tabc\tds\t10.0\t0.0\t100\tbaseline\t{}\tup\t\toptimize\t1\n")
        if ref:
            # Modeless tasks resolve reference output to the FLAT legacy layout
            # (reference_output_<tier>/), so create that for the check to pass.
            (td / "reference_output_tiny").mkdir(parents=True)
            (td / "reference_output_tiny" / "o.pkl").write_text("x")
        return td

    def test_no_task_yaml_exits(self, tmp_path):
        # An empty dir is rejected by task_dir_from_args (exit 1) before
        # init-check's own task.yaml guard (which is therefore defensive).
        td = tmp_path / "empty"
        td.mkdir()
        args = SimpleNamespace(task_dir=str(td), parity=False)
        with pytest.raises(SystemExit):
            ic.cmd_init_check(args)

    def test_no_datasets_exits_1(self, tmp_path):
        td = self._task(tmp_path, datasets=False)
        args = SimpleNamespace(task_dir=str(td), parity=False)
        with pytest.raises(SystemExit) as ei:
            ic.cmd_init_check(args)
        assert ei.value.code == 1

    def test_all_pass_exits_0(self, tmp_path, capsys):
        td = self._task(tmp_path)
        args = SimpleNamespace(task_dir=str(td), parity=False)
        with pytest.raises(SystemExit) as ei:
            ic.cmd_init_check(args)
        assert ei.value.code == 0
        assert "init-check passed" in capsys.readouterr().out

    def test_missing_baseline_exits_1(self, tmp_path, capsys):
        td = self._task(tmp_path, baseline=False)
        args = SimpleNamespace(task_dir=str(td), parity=False)
        with pytest.raises(SystemExit) as ei:
            ic.cmd_init_check(args)
        assert ei.value.code == 1
        assert "found gaps" in capsys.readouterr().out

    def test_missing_ref_exits_1(self, tmp_path, capsys):
        td = self._task(tmp_path, ref=False)
        args = SimpleNamespace(task_dir=str(td), parity=False)
        with pytest.raises(SystemExit) as ei:
            ic.cmd_init_check(args)
        assert ei.value.code == 1

    def test_stochastic_requires_noise(self, tmp_path, capsys):
        # stochastic + noise populated -> passes
        td = self._task(tmp_path, stochastic=True)
        args = SimpleNamespace(task_dir=str(td), parity=False)
        with pytest.raises(SystemExit) as ei:
            ic.cmd_init_check(args)
        assert ei.value.code == 0
        out = capsys.readouterr().out
        assert "noise" in out

    def test_parity_runs_when_requested(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)
        # Stub the parity check so we don't run the real runner.
        monkeypatch.setattr(ic, "_run_parity_check",
                            lambda td, ds, m: True)
        args = SimpleNamespace(task_dir=str(td), parity=True)
        with pytest.raises(SystemExit) as ei:
            ic.cmd_init_check(args)
        assert ei.value.code == 0

    def test_parity_failure_exits_1(self, tmp_path, monkeypatch):
        td = self._task(tmp_path)
        monkeypatch.setattr(ic, "_run_parity_check",
                            lambda td, ds, m: False)
        args = SimpleNamespace(task_dir=str(td), parity=True)
        with pytest.raises(SystemExit) as ei:
            ic.cmd_init_check(args)
        assert ei.value.code == 1


class TestRunParityCheck:
    def test_no_metrics_returns_false(self, tmp_path, capsys):
        assert ic._run_parity_check(tmp_path, [], []) is False
        assert "no metrics" in capsys.readouterr().out

    def test_missing_ref_marks_fail(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ic, "_ref_output_nonempty", lambda td, t: False)
        datasets = [{"tier": "tiny", "name": "ds"}]
        metrics = [{"name": "speedup", "comparator": "gte"}]
        ok = ic._run_parity_check(tmp_path, datasets, metrics)
        assert ok is False
        assert "no reference_outputs" in capsys.readouterr().out

    def test_identity_pass(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ic, "_ref_output_nonempty", lambda td, t: True)
        import zyme.runner as runner
        monkeypatch.setattr(runner, "run_task", lambda td, dataset_entry: "log")
        # speed, peak, observed, status (parse_log is module-level in init_check)
        monkeypatch.setattr(ic, "parse_log",
                            lambda log: (1.0, 100, {"speedup": 1.0, "diff": 0.0}, "ok"))
        datasets = [{"tier": "tiny", "name": "ds"}]
        metrics = [{"name": "speedup", "comparator": "gte"},
                   {"name": "diff", "comparator": "lte"}]
        ok = ic._run_parity_check(tmp_path, datasets, metrics)
        assert ok is True
        assert "identity-perfect" in capsys.readouterr().out

    def test_crash_marks_fail(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ic, "_ref_output_nonempty", lambda td, t: True)
        import zyme.runner as runner
        monkeypatch.setattr(runner, "run_task", lambda td, dataset_entry: "crashlog")
        monkeypatch.setattr(ic, "parse_log",
                            lambda log: (None, None, {}, "crash"))
        datasets = [{"tier": "tiny", "name": "ds"}]
        metrics = [{"name": "speedup", "comparator": "gte"}]
        ok = ic._run_parity_check(tmp_path, datasets, metrics)
        assert ok is False
        assert "crashed" in capsys.readouterr().out

    def test_metric_drift_marks_fail(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ic, "_ref_output_nonempty", lambda td, t: True)
        import zyme.runner as runner
        monkeypatch.setattr(runner, "run_task", lambda td, dataset_entry: "log")
        # gte metric at 0.5 (< 1.0 - tol) -> fail
        monkeypatch.setattr(ic, "parse_log",
                            lambda log: (1.0, 100, {"speedup": 0.5}, "ok"))
        datasets = [{"tier": "tiny", "name": "ds"}]
        metrics = [{"name": "speedup", "comparator": "gte"}]
        ok = ic._run_parity_check(tmp_path, datasets, metrics)
        assert ok is False
        out = capsys.readouterr().out
        assert "parity failed" in out
