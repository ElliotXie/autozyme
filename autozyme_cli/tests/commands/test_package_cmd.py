"""Unit tests for the `zyme package <sub>` COMMAND modules.

These target the command wrappers in commands/package/{lint,preflight,
check_intercept,smoke_parity}.py — NOT the rule definitions in lint_rules.py
(those are covered by tests/test_package_lint.py). No overlap.

Covered:
  lint.py          — patch-file discovery (R folder + Py register_patch filter),
                     context builders, target resolution, run_lint aggregation,
                     cmd_package_lint output + exit code.
  preflight.py     — each _step_* over a stubbed boundary, the orchestrator's
                     stop-on-first-fail vs --continue, the summary.
  check_intercept.py — _resolve_python_for fallback, the count-table reporting
                     branches (all-fired / some-zero / no-targets) driven over a
                     stubbed worker that writes a counts.json.
  smoke_parity.py  — the metric-line regex parser + threshold checker across
                     directions, plus cmd output branches over stubbed workers.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands.package import lint as pkglint
from zyme.commands.package import preflight as pkgpre
from zyme.commands.package import check_intercept as pkgci
from zyme.commands.package import smoke_parity as pkgsp
from zyme.commands.package.lint_rules import LintFinding


# ==========================================================================
# lint.py
# ==========================================================================

def _make_framework(tmp_path):
    fw = tmp_path / "fw"
    (fw / "autozyme_r" / "inst" / "patches").mkdir(parents=True)
    (fw / "autozyme_py" / "src" / "autozyme").mkdir(parents=True)
    (fw / "autozyme_r" / "DESCRIPTION").write_text("Package: autozyme\n")
    (fw / "autozyme_r" / "NAMESPACE").write_text("export(foo)\n")
    return fw


class TestPatchFileDiscovery:
    def test_r_patch_files(self, tmp_path):
        fw = _make_framework(tmp_path)
        pdir = fw / "autozyme_r" / "inst" / "patches" / "decontx"
        pdir.mkdir()
        (pdir / "patch.R").write_text("# patch")
        # A non-patch dir (no patch.R) is skipped.
        (fw / "autozyme_r" / "inst" / "patches" / "notapatch").mkdir()
        out = pkglint._r_patch_files(fw)
        assert out == [("decontx", pdir / "patch.R")]

    def test_r_patch_files_missing_root(self, tmp_path):
        assert pkglint._r_patch_files(tmp_path / "nope") == []

    def test_py_patch_files_filters_register_patch(self, tmp_path):
        fw = _make_framework(tmp_path)
        base = fw / "autozyme_py" / "src" / "autozyme"
        # Real patch: calls register_patch.
        p1 = base / "scrublet"
        p1.mkdir()
        (p1 / "__init__.py").write_text("register_patch(...)\n")
        # Not a patch: no register_patch.
        p2 = base / "core_util"
        p2.mkdir()
        (p2 / "__init__.py").write_text("x = 1\n")
        # Dunder/private dir: skipped.
        p3 = base / "_private"
        p3.mkdir()
        (p3 / "__init__.py").write_text("register_patch(...)\n")
        out = pkglint._py_patch_files(fw)
        names = [n for n, _ in out]
        assert names == ["scrublet"]

    def test_py_patch_files_missing_root(self, tmp_path):
        assert pkglint._py_patch_files(tmp_path / "nope") == []


class TestLintContexts:
    def test_r_context_loads_desc_ns(self, tmp_path):
        fw = _make_framework(tmp_path)
        pdir = fw / "autozyme_r" / "inst" / "patches" / "x"
        pdir.mkdir()
        patch = pdir / "patch.R"
        patch.write_text("# r code")
        ctx = pkglint._r_context("x", patch, fw)
        assert ctx.language == "R"
        assert ctx.patch_text == "# r code"
        assert "Package: autozyme" in ctx.description_text
        assert "export(foo)" in ctx.namespace_text

    def test_py_context(self, tmp_path):
        patch = tmp_path / "__init__.py"
        patch.write_text("register_patch(...)")
        ctx = pkglint._py_context("scr", patch)
        assert ctx.language == "py"
        assert ctx.patch_name == "scr"


class TestResolveTargets:
    def test_named_patch_r(self, tmp_path):
        fw = _make_framework(tmp_path)
        pdir = fw / "autozyme_r" / "inst" / "patches" / "decontx"
        pdir.mkdir()
        (pdir / "patch.R").write_text("# patch")
        args = SimpleNamespace(patch="decontx", all=False)
        pairs = pkglint._resolve_targets(args, fw)
        assert len(pairs) == 1
        assert pairs[0][0].language == "R"

    def test_named_patch_not_found_dies(self, tmp_path):
        fw = _make_framework(tmp_path)
        args = SimpleNamespace(patch="ghost", all=False)
        with pytest.raises(SystemExit):
            pkglint._resolve_targets(args, fw)

    def test_all_lints_both_languages(self, tmp_path):
        fw = _make_framework(tmp_path)
        rdir = fw / "autozyme_r" / "inst" / "patches" / "rp"
        rdir.mkdir()
        (rdir / "patch.R").write_text("# r")
        pydir = fw / "autozyme_py" / "src" / "autozyme" / "pp"
        pydir.mkdir()
        (pydir / "__init__.py").write_text("register_patch(...)")
        args = SimpleNamespace(patch=None, all=True)
        pairs = pkglint._resolve_targets(args, fw)
        langs = sorted(ctx.language for ctx, _ in pairs)
        assert langs == ["R", "py"]


class TestRunLint:
    def test_no_framework_dies(self, monkeypatch):
        monkeypatch.setattr(pkglint, "find_framework_root", lambda p: None)
        with pytest.raises(SystemExit):
            pkglint.run_lint(SimpleNamespace(patch=None, all=True))

    def test_aggregates_findings_and_exit_code(self, tmp_path, monkeypatch):
        fw = _make_framework(tmp_path)
        pydir = fw / "autozyme_py" / "src" / "autozyme" / "pp"
        pydir.mkdir()
        (pydir / "__init__.py").write_text("register_patch(...)")

        fail = LintFinding(rule_id="r1", severity="FAIL",
                           file=pydir / "__init__.py", line=1, message="bad")

        def fake_rule(ctx):
            return [fail]
        monkeypatch.setattr(pkglint, "PY_RULES", (fake_rule,))
        monkeypatch.setattr(pkglint, "R_RULES", ())
        args = SimpleNamespace(patch=None, all=True)
        rc, findings = pkglint.run_lint(args, framework_root=fw)
        assert rc == 1
        assert findings == [fail]

    def test_rule_crash_becomes_warn(self, tmp_path, monkeypatch):
        fw = _make_framework(tmp_path)
        pydir = fw / "autozyme_py" / "src" / "autozyme" / "pp"
        pydir.mkdir()
        (pydir / "__init__.py").write_text("register_patch(...)")

        def boom(ctx):
            raise RuntimeError("rule broke")
        monkeypatch.setattr(pkglint, "PY_RULES", (boom,))
        monkeypatch.setattr(pkglint, "R_RULES", ())
        rc, findings = pkglint.run_lint(
            SimpleNamespace(patch=None, all=True), framework_root=fw)
        assert rc == 0  # WARN only, no FAIL
        assert findings[0].severity == "WARN"
        assert "lint rule crashed" in findings[0].message


class TestCmdPackageLint:
    def test_clean(self, monkeypatch, capsys):
        monkeypatch.setattr(pkglint, "run_lint", lambda args: (0, []))
        rc = pkglint.cmd_package_lint(SimpleNamespace())
        assert rc == 0
        assert "lint clean" in capsys.readouterr().out

    def test_groups_and_returns_exit_code(self, tmp_path, monkeypatch, capsys):
        f1 = LintFinding(rule_id="r1", severity="FAIL",
                         file=Path("a/patch.R"), line=3, message="m1")
        f2 = LintFinding(rule_id="r2", severity="WARN",
                         file=Path("a/patch.R"), line=5, message="m2")
        monkeypatch.setattr(pkglint, "run_lint", lambda args: (1, [f1, f2]))
        rc = pkglint.cmd_package_lint(SimpleNamespace())
        assert rc == 1
        out = capsys.readouterr().out
        assert "1 fail, 1 warn" in out
        assert "m1" in out and "m2" in out


# ==========================================================================
# preflight.py
# ==========================================================================

class TestPreflightSteps:
    def test_step_lint_infers_patch(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(pkgpre, "_infer_patch_name", lambda td: "decontx")
        monkeypatch.setattr(pkgpre, "cmd_package_lint", lambda args: 0)
        rc = pkgpre._step_lint(tmp_path, None)
        assert rc == 0
        assert "patch=decontx" in capsys.readouterr().out

    def test_step_lint_all_when_no_patch(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(pkgpre, "_infer_patch_name", lambda td: None)
        seen = {}

        def fake_lint(args):
            seen["all"] = args.all
            seen["patch"] = args.patch
            return 0
        monkeypatch.setattr(pkgpre, "cmd_package_lint", fake_lint)
        pkgpre._step_lint(tmp_path, None)
        assert seen["all"] is True
        assert seen["patch"] is None

    def test_step_portability_acceptable(self, tmp_path, monkeypatch, capsys):
        res = SimpleNamespace(verdict="CLEAN", hits=[])
        monkeypatch.setattr(pkgpre, "scan_task", lambda td, framework_root: res)
        monkeypatch.setattr(pkgpre, "save_portability_scan", lambda td, r: None)
        rc = pkgpre._step_portability(tmp_path, None)
        assert rc == 0
        assert "acceptable" in capsys.readouterr().out

    def test_step_portability_unsafe_fails(self, tmp_path, monkeypatch, capsys):
        hit = SimpleNamespace(ref="run.py:10", label="darwin-only")
        res = SimpleNamespace(verdict="UNSAFE", hits=[hit])
        monkeypatch.setattr(pkgpre, "scan_task", lambda td, framework_root: res)
        monkeypatch.setattr(pkgpre, "save_portability_scan", lambda td, r: None)
        rc = pkgpre._step_portability(tmp_path, None)
        assert rc == 1
        out = capsys.readouterr().out
        assert "not shippable" in out
        assert "run.py:10" in out

    def test_step_portability_many_hits_truncates(self, tmp_path, monkeypatch, capsys):
        hits = [SimpleNamespace(ref=f"r{i}", label="x") for i in range(15)]
        res = SimpleNamespace(verdict="NEEDS_REVIEW", hits=hits)
        monkeypatch.setattr(pkgpre, "scan_task", lambda td, framework_root: res)
        monkeypatch.setattr(pkgpre, "save_portability_scan", lambda td, r: None)
        pkgpre._step_portability(tmp_path, None)
        out = capsys.readouterr().out
        assert "and 5 more" in out

    def test_step_smoke_parity(self, tmp_path, monkeypatch):
        seen = {}

        def fake_sp(args):
            seen["tier"] = args.tier
            seen["patch"] = args.patch
            return 0
        monkeypatch.setattr(pkgpre, "cmd_package_smoke_parity", fake_sp)
        rc = pkgpre._step_smoke_parity(tmp_path, "decontx")
        assert rc == 0
        assert seen["tier"] == "tiny"
        assert seen["patch"] == "decontx"


class TestCmdPackagePreflight:
    @pytest.fixture
    def task(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        return tmp_path

    def _stub_all(self, monkeypatch, lint=0, port=0, smoke=0):
        monkeypatch.setattr(pkgpre, "find_framework_root", lambda td: None)
        monkeypatch.setattr(pkgpre, "_infer_patch_name", lambda td: "decontx")
        monkeypatch.setattr(pkgpre, "_step_lint", lambda td, fr: lint)
        monkeypatch.setattr(pkgpre, "_step_portability", lambda td, fr: port)
        monkeypatch.setattr(pkgpre, "_step_smoke_parity", lambda td, p: smoke)

    def test_all_pass(self, task, monkeypatch, capsys):
        self._stub_all(monkeypatch)
        args = SimpleNamespace(task_dir=str(task), continue_on_fail=False,
                               skip_parity=False)
        rc = pkgpre.cmd_package_preflight(args)
        assert rc == 0
        assert "preflight clean" in capsys.readouterr().out

    def test_stops_on_first_lint_fail(self, task, monkeypatch, capsys):
        self._stub_all(monkeypatch, lint=1)
        called = {"port": False}
        monkeypatch.setattr(pkgpre, "_step_portability",
                            lambda td, fr: called.__setitem__("port", True) or 0)
        args = SimpleNamespace(task_dir=str(task), continue_on_fail=False,
                               skip_parity=False)
        rc = pkgpre.cmd_package_preflight(args)
        assert rc == 1
        assert called["port"] is False  # short-circuited

    def test_continue_collects_all(self, task, monkeypatch, capsys):
        self._stub_all(monkeypatch, lint=1, port=1, smoke=1)
        args = SimpleNamespace(task_dir=str(task), continue_on_fail=True,
                               skip_parity=False)
        rc = pkgpre.cmd_package_preflight(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "lint" in out and "portability" in out and "smoke-parity" in out

    def test_skip_parity(self, task, monkeypatch):
        self._stub_all(monkeypatch)
        called = {"smoke": False}
        monkeypatch.setattr(pkgpre, "_step_smoke_parity",
                            lambda td, p: called.__setitem__("smoke", True) or 0)
        args = SimpleNamespace(task_dir=str(task), continue_on_fail=False,
                               skip_parity=True)
        rc = pkgpre.cmd_package_preflight(args)
        assert rc == 0
        assert called["smoke"] is False


class TestSummarize:
    def test_clean(self, capsys):
        assert pkgpre._summarize([]) == 0
        assert "clean" in capsys.readouterr().out

    def test_failures(self, capsys):
        assert pkgpre._summarize(["lint", "portability"]) == 1
        assert "2 step(s) failed" in capsys.readouterr().out


# ==========================================================================
# check_intercept.py
# ==========================================================================

class TestResolvePythonFor:
    def test_falls_back_to_sys_executable(self, tmp_path):
        (tmp_path / "task.yaml").write_text("x: 1\n")
        assert pkgci._resolve_python_for(tmp_path) == sys.executable

    def test_uses_executor_python_when_parser_called_correctly(self, tmp_path, monkeypatch):
        # _resolve_python_for resolves the spec returned by parse_executor.
        # We stub parse_executor (imported lazily inside the function) so the
        # happy path is exercised regardless of the parse_executor arg quirk
        # documented below.
        (tmp_path / "task.yaml").write_text("executor:\n  python: myenv\n")
        import zyme.parsers.task_yaml as tyaml
        monkeypatch.setattr(tyaml, "parse_executor", lambda p: {"python": "myenv"})
        monkeypatch.setattr(pkgci, "_resolve_python",
                            lambda spec: f"/envs/{spec}/bin/python")
        assert pkgci._resolve_python_for(tmp_path) == "/envs/myenv/bin/python"

    def test_honors_executor_python_via_real_parse_executor(self, tmp_path, monkeypatch):
        # B14 fix: _resolve_python_for now passes the task.yaml FILE path to
        # parse_executor (not the directory), so a declared executor.python is
        # honored end-to-end through the REAL parser. Only _resolve_python is
        # stubbed, for a deterministic return.
        (tmp_path / "task.yaml").write_text("executor:\n  python: myenv\n")
        monkeypatch.setattr(pkgci, "_resolve_python",
                            lambda spec: f"/envs/{spec}/bin/python")
        assert pkgci._resolve_python_for(tmp_path) == "/envs/myenv/bin/python"
        assert pkgci._resolve_python_for(tmp_path) != sys.executable


class TestSpawnWorkers:
    def test_spawn_python_worker_builds_cmd(self, tmp_path, monkeypatch):
        captured = {}

        def fake_call(cmd, env):
            captured["cmd"] = cmd
            captured["env"] = env
            return 0
        monkeypatch.setattr(pkgci.subprocess, "call", fake_call)
        monkeypatch.setattr(pkgci, "_resolve_python_for", lambda td: "/py")
        rc = pkgci._spawn_python_worker("decontx", tmp_path, "tiny",
                                        tmp_path / "out", {"X": "1"})
        assert rc == 0
        assert captured["cmd"][0] == "/py"
        assert "autozyme._verify_worker" in captured["cmd"]
        assert "--activate" in captured["cmd"]
        assert "--patch" in captured["cmd"] and "decontx" in captured["cmd"]

    def test_spawn_r_worker_builds_cmd(self, tmp_path, monkeypatch):
        captured = {}

        def fake_call(cmd, env):
            captured["cmd"] = cmd
            return 0
        monkeypatch.setattr(pkgci.subprocess, "call", fake_call)
        rc = pkgci._spawn_r_worker("decontx", tmp_path, "tiny",
                                   tmp_path / "out", {})
        assert rc == 0
        assert captured["cmd"][0] == "Rscript"
        assert "--activate" in captured["cmd"]
        assert "verify_worker.R" in captured["cmd"][2]


class TestRunEvaluate:
    def test_runs_evaluate_and_parses(self, tmp_path, monkeypatch):
        task = tmp_path / "task"
        (task / "reference_outputs" / "tiny").mkdir(parents=True)
        (task / "reference_outputs" / "tiny" / "ref.pkl").write_text("r")
        (task / "evaluate.py").write_text("print('speedup: 2.0')")
        smoke_out = tmp_path / "smoke"
        smoke_out.mkdir()
        (smoke_out / "out.pkl").write_text("o")

        class FakeProc:
            returncode = 0
            stdout = "speedup: 2.0\n"
            stderr = ""

        def fake_run(cmd, env, cwd, capture_output, text):
            return FakeProc()
        monkeypatch.setattr(pkgsp.subprocess, "run", fake_run)
        monkeypatch.setattr(pkgsp, "_resolve_python_for", lambda td: "/py")
        rc, out_lines, err_lines = pkgsp._run_evaluate(task, smoke_out, "tiny", "py")
        assert rc == 0
        assert "speedup: 2.0" in out_lines

    def test_no_reference_dies(self, tmp_path, monkeypatch):
        task = tmp_path / "task"
        task.mkdir()
        (task / "evaluate.py").write_text("p")
        smoke_out = tmp_path / "smoke"
        smoke_out.mkdir()
        with pytest.raises(SystemExit):
            pkgsp._run_evaluate(task, smoke_out, "tiny", "py")

    def test_flat_reference_layout(self, tmp_path, monkeypatch):
        task = tmp_path / "task"
        (task / "reference_output_tiny").mkdir(parents=True)
        (task / "reference_output_tiny" / "r").write_text("x")
        (task / "evaluate.R").write_text("cat('ari: 1.0\\n')")
        smoke_out = tmp_path / "smoke"
        smoke_out.mkdir()
        (smoke_out / "o").write_text("x")

        class FakeProc:
            returncode = 0
            stdout = "ari: 1.0\n"
            stderr = ""
        monkeypatch.setattr(pkgsp.subprocess, "run",
                            lambda *a, **k: FakeProc())
        rc, out_lines, _ = pkgsp._run_evaluate(task, smoke_out, "tiny", "R")
        assert rc == 0

    def test_no_evaluate_dies(self, tmp_path):
        task = tmp_path / "task"
        (task / "reference_outputs" / "tiny").mkdir(parents=True)
        (task / "reference_outputs" / "tiny" / "r").write_text("x")
        smoke_out = tmp_path / "smoke"
        smoke_out.mkdir()
        with pytest.raises(SystemExit):
            pkgsp._run_evaluate(task, smoke_out, "tiny", "py")


class TestCmdCheckIntercept:
    @pytest.fixture
    def task(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.py").write_text("x")
        return tmp_path

    def _stub_worker(self, monkeypatch, counts_json):
        """Stub the python worker to write counts_json into the out path."""
        def fake_spawn(patch, task_dir, tier, output_dir, env):
            Path(env["ZYME_INTERCEPT_OUT"]).write_text(counts_json)
            return 0
        monkeypatch.setattr(pkgci, "_spawn_python_worker", fake_spawn)
        monkeypatch.setattr(pkgci, "detect_lang", lambda td: "py")
        monkeypatch.setattr(pkgci, "_infer_patch_name", lambda td: "decontx")

    def test_all_fired(self, task, monkeypatch, capsys):
        self._stub_worker(monkeypatch, '{"foo": 3, "bar": 1}')
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang=None)
        rc = pkgci.cmd_package_check_intercept(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "all 2 target(s) fired" in out

    def test_some_zero_fails(self, task, monkeypatch, capsys):
        self._stub_worker(monkeypatch, '{"foo": 0, "bar": 2}')
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang=None)
        rc = pkgci.cmd_package_check_intercept(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "never fired" in out

    def test_no_targets_fails(self, task, monkeypatch, capsys):
        self._stub_worker(monkeypatch, '{}')
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang=None)
        rc = pkgci.cmd_package_check_intercept(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "no targets were dispatched" in out

    def test_no_patch_inferable_dies(self, task, monkeypatch):
        monkeypatch.setattr(pkgci, "_infer_patch_name", lambda td: None)
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang="py")
        with pytest.raises(SystemExit):
            pkgci.cmd_package_check_intercept(args)

    def test_worker_no_output_dies(self, task, monkeypatch):
        def fake_spawn(patch, task_dir, tier, output_dir, env):
            return 0  # writes nothing
        monkeypatch.setattr(pkgci, "_spawn_python_worker", fake_spawn)
        monkeypatch.setattr(pkgci, "detect_lang", lambda td: "py")
        monkeypatch.setattr(pkgci, "_infer_patch_name", lambda td: "decontx")
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang=None)
        with pytest.raises(SystemExit):
            pkgci.cmd_package_check_intercept(args)


# ==========================================================================
# smoke_parity.py
# ==========================================================================

class TestParseMetricLines:
    def test_parses_floats(self):
        out = pkgsp._parse_metric_lines([
            "speedup: 2.5", "max_diff: 0.001", "ari: 1.0",
        ])
        assert out == {"speedup": 2.5, "max_diff": 0.001, "ari": 1.0}

    def test_scientific_notation(self):
        out = pkgsp._parse_metric_lines(["drift: 1.2e-4"])
        assert out["drift"] == pytest.approx(1.2e-4)

    def test_ignores_non_metric_lines(self):
        out = pkgsp._parse_metric_lines([
            "Running evaluate...", "speedup: 3.0", "==== done ====",
        ])
        assert out == {"speedup": 3.0}

    def test_negative_value(self):
        out = pkgsp._parse_metric_lines(["delta: -0.5"])
        assert out["delta"] == -0.5


class TestCheckThresholds:
    def _task(self, tmp_path, metrics_yaml):
        (tmp_path / "task.yaml").write_text(metrics_yaml)
        return tmp_path

    def test_gte_pass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pkgsp, "parse_metrics", lambda p: [
            {"name": "speedup", "direction": "gte", "threshold": 1.0}])
        ok, lines = pkgsp._check_thresholds(tmp_path, {"speedup": 2.0})
        assert ok is True
        assert "OK" in lines[0]

    def test_gte_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pkgsp, "parse_metrics", lambda p: [
            {"name": "speedup", "direction": "gte", "threshold": 5.0}])
        ok, lines = pkgsp._check_thresholds(tmp_path, {"speedup": 2.0})
        assert ok is False
        assert "FAIL" in lines[0]

    def test_lte_directions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pkgsp, "parse_metrics", lambda p: [
            {"name": "diff", "op": "lte", "threshold": 0.01}])
        ok, _ = pkgsp._check_thresholds(tmp_path, {"diff": 0.005})
        assert ok is True

    def test_missing_metric_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pkgsp, "parse_metrics", lambda p: [
            {"name": "speedup", "direction": "gte", "threshold": 1.0}])
        ok, lines = pkgsp._check_thresholds(tmp_path, {"other": 1.0})
        assert ok is False
        assert "not emitted" in lines[0]

    def test_no_threshold_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pkgsp, "parse_metrics", lambda p: [
            {"name": "speedup", "direction": "gte", "threshold": None}])
        ok, _ = pkgsp._check_thresholds(tmp_path, {"speedup": 0.0})
        assert ok is True

    def test_gt_lt_directions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pkgsp, "parse_metrics", lambda p: [
            {"name": "a", "direction": "gt", "threshold": 1.0},
            {"name": "b", "direction": "lt", "threshold": 1.0}])
        ok, _ = pkgsp._check_thresholds(tmp_path, {"a": 2.0, "b": 0.5})
        assert ok is True

    def test_unknown_direction_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pkgsp, "parse_metrics", lambda p: [
            {"name": "a", "direction": "weird", "threshold": 1.0}])
        ok, _ = pkgsp._check_thresholds(tmp_path, {"a": 0.0})
        assert ok is True


class TestCmdSmokeParity:
    @pytest.fixture
    def task(self, tmp_path):
        (tmp_path / "task.yaml").write_text(
            "target_function: foo\n"
            "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}\n")
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.py").write_text("x")
        return tmp_path

    def _stub_smoke(self, monkeypatch, eval_rc, stdout_lines):
        """Stub worker (writes a file) + _run_evaluate."""
        def fake_spawn(patch, task_dir, tier, output_dir, env):
            (Path(output_dir) / "out.pkl").write_text("x")
            return 0
        monkeypatch.setattr(pkgsp, "_spawn_python_worker", fake_spawn)
        monkeypatch.setattr(pkgsp, "detect_lang", lambda td: "py")
        monkeypatch.setattr(pkgsp, "_infer_patch_name", lambda td: "decontx")
        monkeypatch.setattr(pkgsp, "_run_evaluate",
                            lambda td, so, t, l: (eval_rc, stdout_lines, []))

    def test_pass(self, task, monkeypatch, capsys):
        self._stub_smoke(monkeypatch, 0, ["speedup: 2.0"])
        monkeypatch.setattr(pkgsp, "parse_metrics", lambda p: [
            {"name": "speedup", "direction": "gte", "threshold": 1.0}])
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang=None)
        rc = pkgsp.cmd_package_smoke_parity(args)
        assert rc == 0
        assert "matches reference" in capsys.readouterr().out

    def test_evaluate_crash_fails(self, task, monkeypatch, capsys):
        self._stub_smoke(monkeypatch, 1, ["traceback line"])
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang=None)
        rc = pkgsp.cmd_package_smoke_parity(args)
        assert rc == 1
        assert "evaluate crashed" in capsys.readouterr().out

    def test_no_parseable_metrics_warns(self, task, monkeypatch, capsys):
        self._stub_smoke(monkeypatch, 0, ["no metrics here"])
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang=None)
        rc = pkgsp.cmd_package_smoke_parity(args)
        assert rc == 0  # WARN, not FAIL
        assert "no parseable" in capsys.readouterr().out

    def test_threshold_miss_fails(self, task, monkeypatch, capsys):
        self._stub_smoke(monkeypatch, 0, ["speedup: 0.5"])
        monkeypatch.setattr(pkgsp, "parse_metrics", lambda p: [
            {"name": "speedup", "direction": "gte", "threshold": 1.0}])
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang=None)
        rc = pkgsp.cmd_package_smoke_parity(args)
        assert rc == 1
        assert "fails task.yaml metric thresholds" in capsys.readouterr().out

    def test_worker_nonzero_dies(self, task, monkeypatch):
        def fake_spawn(patch, task_dir, tier, output_dir, env):
            return 3
        monkeypatch.setattr(pkgsp, "_spawn_python_worker", fake_spawn)
        monkeypatch.setattr(pkgsp, "detect_lang", lambda td: "py")
        monkeypatch.setattr(pkgsp, "_infer_patch_name", lambda td: "decontx")
        args = SimpleNamespace(task_dir=str(task), patch=None, tier=None,
                               lang=None)
        with pytest.raises(SystemExit):
            pkgsp.cmd_package_smoke_parity(args)
