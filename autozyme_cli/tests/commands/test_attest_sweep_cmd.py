"""Unit tests for zyme.commands.attest_sweep — pure planning/coverage layer.

attest_sweep walks packaged patches, reads each task's package_verify.tsv to
find which (tier, threads) Windows cells already have a patched row, and plans
the missing ones. The subprocess-launching `_run_one` / `_kill` are exercised
only via a monkeypatched boundary; everything else (discovery, task-dir
lookup, coverage parsing, combo planning, dedup, arg parsing, plan printing,
and the cmd_attest_sweep orchestration) is pure given on-disk fixtures.

The module hard-codes FRAMEWORK_ROOT / PY_PLUGINS / R_PLUGINS / OPT_TASK at
import time (derived from __file__), so the tests monkeypatch those module
globals to point at a tmp_path fake framework.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import zyme.commands.attest_sweep as asw
from zyme.commands.attest_sweep import (
    Combo,
    DEFAULT_REPS,
    DEFAULT_THREADS,
    DEFAULT_TIERS,
    _discover_patches,
    _existing_win_coverage,
    _find_task_dir,
    _parse_int_list,
    _parse_str_list,
    _plan_combos,
    _print_plan,
    _task_supports_tier,
    _threading_mode,
    cmd_attest_sweep,
)


# ==========================================================================
# Fake-framework + package_verify fixture builders
# ==========================================================================

PV_HEADER = [
    "timestamp", "patch_name", "tier", "dataset", "rep_idx", "variant",
    "sec", "speedup_pct", "speedup_x", "peak_mb", "peak_mb_change_pct",
    "peak_mb_fold", "pass", "metrics_json", "framework_version",
    "package_version", "note", "system_os", "system_cpu", "system_ram_gb",
    "system_threads",
]


def _pv_row(**ov) -> dict:
    base = {
        "timestamp": "2026-06-01T00:00:00", "patch_name": "p", "tier": "tiny",
        "dataset": "ds", "rep_idx": "1", "variant": "patched", "sec": "1.0",
        "speedup_pct": "50.0", "speedup_x": "2.0", "peak_mb": "100",
        "peak_mb_change_pct": "", "peak_mb_fold": "", "pass": "true",
        "metrics_json": "{}", "framework_version": "0.3.0",
        "package_version": "1.0", "note": "", "system_os": "Windows 11",
        "system_cpu": "CPU", "system_ram_gb": "127.2", "system_threads": "1",
    }
    base.update(ov)
    return base


def _write_pv(path: Path, *rows: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(PV_HEADER)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in PV_HEADER))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_framework(
    tmp_path: Path,
    *,
    py_patches: dict[str, dict] | None = None,
    r_patches: dict[str, dict] | None = None,
    pv_rows: dict[str, list[dict]] | None = None,
    task_yaml: dict[str, str] | None = None,
) -> Path:
    """Build a fake framework tree and point the module globals at it.

    py_patches/r_patches: {patch_name: {"finalized": bool}}.
    pv_rows: {patch_name: [rows]} written into a task dir under optimized_task/.
    task_yaml: {patch_name: yaml_text} for the task dir.
    """
    fw = tmp_path / "fw"
    py_root = fw / "autozyme_py" / "src" / "autozyme"
    r_root = fw / "autozyme_r" / "inst" / "patches"
    opt = fw / "optimized_task" / "cat"
    py_root.mkdir(parents=True)
    r_root.mkdir(parents=True)
    opt.mkdir(parents=True)

    for name, cfg in (py_patches or {}).items():
        d = py_root / name
        d.mkdir()
        (d / "__init__.py").write_text("# patch\n")
        if cfg.get("finalized", True):
            (d / "speedups_finalized.tsv").write_text("x\n")
    for name, cfg in (r_patches or {}).items():
        d = r_root / name
        d.mkdir()
        (d / "patch.R").write_text("foo <- function() {}\n")
        if cfg.get("finalized", True):
            (d / "speedups_finalized.tsv").write_text("x\n")

    all_patches = list((py_patches or {})) + list((r_patches or {}))
    for name in all_patches:
        td = opt / name
        td.mkdir(parents=True, exist_ok=True)
        rows = (pv_rows or {}).get(name)
        if rows is not None:
            _write_pv(td / "package_verify.tsv", *rows)
        ytxt = (task_yaml or {}).get(name)
        if ytxt is not None:
            (td / "task.yaml").write_text(ytxt, encoding="utf-8")
    return fw


@pytest.fixture
def patch_globals(monkeypatch):
    """Return a helper that repoints the module's path globals at `fw`."""
    def _apply(fw: Path):
        monkeypatch.setattr(asw, "FRAMEWORK_ROOT", fw)
        monkeypatch.setattr(asw, "PY_PLUGINS", fw / "autozyme_py" / "src" / "autozyme")
        monkeypatch.setattr(asw, "R_PLUGINS", fw / "autozyme_r" / "inst" / "patches")
        monkeypatch.setattr(asw, "OPT_TASK", fw / "optimized_task")
        monkeypatch.setattr(asw, "LOG_DIR", fw / "attest_logs" / "sweep")
    return _apply


# ==========================================================================
# Combo dataclass
# ==========================================================================

class TestCombo:
    def test_spec_format(self):
        c = Combo("mgcv", "py", Path("/x"), "tiny", 4)
        assert c.spec() == "mgcv:tiny:4"

    def test_frozen_hashable(self):
        c = Combo("p", "py", Path("/x"), "tiny", 1)
        assert c in {c}


# ==========================================================================
# arg list parsers
# ==========================================================================

class TestParsers:
    def test_int_list_default_when_empty(self):
        assert _parse_int_list("", (1, 4, 8)) == [1, 4, 8]
        assert _parse_int_list(None or "", DEFAULT_THREADS) == [1, 4, 8]

    def test_int_list_comma_and_space(self):
        assert _parse_int_list("1, 4 8", ()) == [1, 4, 8]

    def test_str_list_default_when_empty(self):
        assert _parse_str_list("", ("tiny", "medium")) == ["tiny", "medium"]

    def test_str_list_strips_and_splits(self):
        assert _parse_str_list("tiny , medium ood_large", ()) == [
            "tiny", "medium", "ood_large",
        ]

    def test_str_list_drops_blanks(self):
        assert _parse_str_list(",,tiny,,", ()) == ["tiny"]


# ==========================================================================
# _discover_patches
# ==========================================================================

class TestDiscoverPatches:
    def test_finds_py_and_r_with_finalized(self, tmp_path, patch_globals):
        fw = _make_framework(
            tmp_path,
            py_patches={"pyone": {}},
            r_patches={"rone": {}},
        )
        patch_globals(fw)
        out = dict(_discover_patches())
        assert out == {"pyone": "py", "rone": "R"}

    def test_skips_underscore_py_dirs(self, tmp_path, patch_globals):
        fw = _make_framework(tmp_path, py_patches={"_private": {}, "real": {}})
        patch_globals(fw)
        names = {n for n, _ in _discover_patches()}
        assert names == {"real"}

    def test_skips_patches_without_finalized(self, tmp_path, patch_globals):
        fw = _make_framework(
            tmp_path,
            py_patches={"has": {"finalized": True}, "nope": {"finalized": False}},
        )
        patch_globals(fw)
        names = {n for n, _ in _discover_patches()}
        assert names == {"has"}

    def test_sorted_order(self, tmp_path, patch_globals):
        fw = _make_framework(tmp_path, py_patches={"zeta": {}, "alpha": {}})
        patch_globals(fw)
        py = [n for n, lang in _discover_patches() if lang == "py"]
        assert py == ["alpha", "zeta"]


# ==========================================================================
# _find_task_dir
# ==========================================================================

class TestFindTaskDir:
    def test_locates_by_patch_name_in_pv(self, tmp_path, patch_globals):
        fw = _make_framework(
            tmp_path,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv")]},
        )
        patch_globals(fw)
        td = _find_task_dir("mgcv")
        assert td is not None and td.name == "mgcv"

    def test_returns_none_when_no_match(self, tmp_path, patch_globals):
        fw = _make_framework(
            tmp_path,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="other")]},
        )
        patch_globals(fw)
        assert _find_task_dir("mgcv") is None

    def test_skips_task_dirs_without_pv(self, tmp_path, patch_globals):
        # task dir exists but has no package_verify.tsv -> not matched
        fw = _make_framework(tmp_path, py_patches={"mgcv": {}})
        patch_globals(fw)
        assert _find_task_dir("mgcv") is None

    def test_skips_pv_without_patch_name_header(self, tmp_path, patch_globals):
        fw = _make_framework(tmp_path, py_patches={"mgcv": {}})
        patch_globals(fw)
        td = fw / "optimized_task" / "cat" / "mgcv"
        (td / "package_verify.tsv").write_text("foo\tbar\n1\t2\n", encoding="utf-8")
        assert _find_task_dir("mgcv") is None


# ==========================================================================
# _threading_mode / _task_supports_tier
# ==========================================================================

class TestTaskYamlReads:
    def test_threading_default_when_missing(self, tmp_path):
        assert _threading_mode(tmp_path) == "default"

    def test_threading_not_applicable(self, tmp_path):
        (tmp_path / "task.yaml").write_text("threading: not_applicable\n")
        assert _threading_mode(tmp_path) == "not_applicable"

    def test_threading_explicit_default(self, tmp_path):
        (tmp_path / "task.yaml").write_text("threading: default\n")
        assert _threading_mode(tmp_path) == "default"

    def test_supports_tier_true_when_no_yaml(self, tmp_path):
        # no task.yaml -> assume all tiers supported (permissive)
        assert _task_supports_tier(tmp_path, "tiny") is True

    def test_supports_tier_matches_unquoted(self, tmp_path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n  - tier: tiny\n  - tier: medium\n"
        )
        assert _task_supports_tier(tmp_path, "tiny") is True
        assert _task_supports_tier(tmp_path, "ood_xlarge") is False

    def test_supports_tier_matches_quoted(self, tmp_path):
        (tmp_path / "task.yaml").write_text('datasets:\n  - tier: "large"\n')
        assert _task_supports_tier(tmp_path, "large") is True


# ==========================================================================
# _existing_win_coverage
# ==========================================================================

class TestExistingWinCoverage:
    def test_empty_when_file_missing(self, tmp_path):
        assert _existing_win_coverage(tmp_path / "nope.tsv") == {}

    def test_collects_windows_patched_threads(self, tmp_path):
        path = _write_pv(
            tmp_path / "pv.tsv",
            _pv_row(tier="tiny", system_threads="1"),
            _pv_row(tier="tiny", system_threads="8"),
            _pv_row(tier="medium", system_threads="4"),
        )
        cov = _existing_win_coverage(path)
        assert cov["tiny"] == {"1", "8"}
        assert cov["medium"] == {"4"}

    def test_ignores_non_windows(self, tmp_path):
        path = _write_pv(
            tmp_path / "pv.tsv",
            _pv_row(tier="tiny", system_os="macOS 24", system_threads="1"),
        )
        assert _existing_win_coverage(path) == {}

    def test_ignores_baseline_variant(self, tmp_path):
        path = _write_pv(
            tmp_path / "pv.tsv",
            _pv_row(tier="tiny", variant="baseline", system_threads="1"),
        )
        assert _existing_win_coverage(path) == {}

    def test_blank_threads_become_NA(self, tmp_path):
        path = _write_pv(
            tmp_path / "pv.tsv",
            _pv_row(tier="tiny", system_threads=""),
        )
        assert _existing_win_coverage(path)["tiny"] == {"NA"}


# ==========================================================================
# _plan_combos
# ==========================================================================

class TestPlanCombos:
    def _fw(self, tmp_path, patch_globals, **kw):
        fw = _make_framework(tmp_path, **kw)
        patch_globals(fw)
        return fw

    def test_full_gap_when_no_coverage(self, tmp_path, patch_globals):
        self._fw(
            tmp_path, patch_globals,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv")]},
        )
        combos, skips = _plan_combos(["tiny"], [1, 8], set(), set())
        # the one existing row is tier=tiny threads=1 -> only (tiny,8) missing
        specs = {c.spec() for c in combos}
        assert specs == {"mgcv:tiny:8"}

    def test_no_task_dir_skip(self, tmp_path, patch_globals):
        self._fw(tmp_path, patch_globals, py_patches={"orphan": {}})
        combos, skips = _plan_combos(["tiny"], [1], set(), set())
        assert combos == []
        assert skips["no_task_dir"] == ["orphan"]

    def test_not_applicable_skip(self, tmp_path, patch_globals):
        self._fw(
            tmp_path, patch_globals,
            py_patches={"seq": {}},
            pv_rows={"seq": [_pv_row(patch_name="seq")]},
            task_yaml={"seq": "threading: not_applicable\n"},
        )
        combos, skips = _plan_combos(["tiny"], [1], set(), set())
        assert combos == []
        assert skips["not_applicable"] == ["seq"]

    def test_user_skip(self, tmp_path, patch_globals):
        self._fw(
            tmp_path, patch_globals,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv")]},
        )
        combos, skips = _plan_combos(["tiny"], [1], {"mgcv"}, set())
        assert combos == []
        assert skips["user_skip"] == ["mgcv"]

    def test_only_filter_restricts(self, tmp_path, patch_globals):
        self._fw(
            tmp_path, patch_globals,
            py_patches={"mgcv": {}, "other": {}},
            pv_rows={
                "mgcv": [_pv_row(patch_name="mgcv", system_threads="1")],
                "other": [_pv_row(patch_name="other", system_threads="1")],
            },
        )
        combos, skips = _plan_combos(["tiny"], [8], set(), {"mgcv"})
        assert {c.patch for c in combos} == {"mgcv"}

    def test_tier_unsupported_excluded(self, tmp_path, patch_globals):
        self._fw(
            tmp_path, patch_globals,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv")]},
            task_yaml={"mgcv": "datasets:\n  - tier: tiny\n"},
        )
        # request tiny (supported) + ood_xlarge (not in yaml) -> only tiny combos
        combos, _ = _plan_combos(["tiny", "ood_xlarge"], [8], set(), set())
        assert {c.tier for c in combos} == {"tiny"}

    def test_existing_thread_not_replanned(self, tmp_path, patch_globals):
        self._fw(
            tmp_path, patch_globals,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [
                _pv_row(patch_name="mgcv", tier="tiny", system_threads="1"),
                _pv_row(patch_name="mgcv", tier="tiny", system_threads="4"),
            ]},
        )
        combos, _ = _plan_combos(["tiny"], [1, 4, 8], set(), set())
        assert {c.thread for c in combos} == {8}


# ==========================================================================
# _print_plan (formatting smoke)
# ==========================================================================

class TestPrintPlan:
    def test_prints_combos_and_skips(self, capsys):
        combos = [Combo("mgcv", "py", Path("/x"), "tiny", 8)]
        skips = {"no_task_dir": ["orphan"], "not_applicable": [], "user_skip": ["sk"]}
        _print_plan(combos, skips, ["tiny"], [8])
        out = capsys.readouterr().out
        assert "attest-sweep plan" in out
        assert "mgcv" in out
        assert "skipped[no_task_dir]" in out
        assert "orphan" in out
        # empty skip categories are not printed
        assert "skipped[not_applicable]" not in out
        assert "scheduled: 1 combos" in out

    def test_empty_plan(self, capsys):
        _print_plan([], {"no_task_dir": [], "not_applicable": [], "user_skip": []},
                    ["tiny"], [1])
        out = capsys.readouterr().out
        assert "scheduled: 0 combos" in out


# ==========================================================================
# cmd_attest_sweep — end-to-end with monkeypatched _run_one
# ==========================================================================

def _sweep_args(**ov) -> argparse.Namespace:
    base = dict(
        plan=False, tiers=None, threads="1", reps=DEFAULT_REPS, timeout=0,
        limit=0, skip=None, only=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


class TestCmdAttestSweep:
    def test_plan_mode_returns_zero_without_running(
        self, tmp_path, patch_globals, monkeypatch, capsys
    ):
        fw = _make_framework(
            tmp_path,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv", system_threads="8")]},
        )
        patch_globals(fw)
        ran = []
        monkeypatch.setattr(asw, "_run_one", lambda *a, **k: ran.append(a) or 0)
        rc = cmd_attest_sweep(_sweep_args(plan=True, threads="1"))
        assert rc == 0
        assert ran == []
        assert "attest-sweep plan" in capsys.readouterr().out

    def test_nothing_to_do_when_fully_covered(
        self, tmp_path, patch_globals, monkeypatch, capsys
    ):
        fw = _make_framework(
            tmp_path,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv", tier="tiny", system_threads="1")]},
        )
        patch_globals(fw)
        monkeypatch.setattr(asw, "_run_one", lambda *a, **k: 0)
        rc = cmd_attest_sweep(_sweep_args(tiers="tiny", threads="1"))
        assert rc == 0
        assert "nothing to do" in capsys.readouterr().out

    def test_runs_missing_combos_success(
        self, tmp_path, patch_globals, monkeypatch, capsys
    ):
        fw = _make_framework(
            tmp_path,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv", tier="tiny", system_threads="1")]},
        )
        patch_globals(fw)
        calls = []
        monkeypatch.setattr(
            asw, "_run_one",
            lambda combo, reps, timeout: calls.append((combo.spec(), reps)) or 0,
        )
        rc = cmd_attest_sweep(_sweep_args(tiers="tiny", threads="1,8", reps=3))
        assert rc == 0
        # only (tiny,8) was missing
        assert calls == [("mgcv:tiny:8", 3)]
        assert "failures=0" in capsys.readouterr().out

    def test_failure_propagates_exit_code(
        self, tmp_path, patch_globals, monkeypatch, capsys
    ):
        fw = _make_framework(
            tmp_path,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv", tier="tiny", system_threads="1")]},
        )
        patch_globals(fw)
        monkeypatch.setattr(asw, "_run_one", lambda *a, **k: 17)
        rc = cmd_attest_sweep(_sweep_args(tiers="tiny", threads="8"))
        assert rc == 1
        assert "failures=1" in capsys.readouterr().out

    def test_limit_caps_combos(self, tmp_path, patch_globals, monkeypatch):
        fw = _make_framework(
            tmp_path,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv", tier="tiny", system_threads="1")]},
        )
        patch_globals(fw)
        calls = []
        monkeypatch.setattr(asw, "_run_one",
                            lambda combo, *a: calls.append(combo.spec()) or 0)
        rc = cmd_attest_sweep(_sweep_args(tiers="tiny", threads="4,8", limit=1))
        assert rc == 0
        assert len(calls) == 1

    def test_resumable_skip_when_run_one_fills_sibling(
        self, tmp_path, patch_globals, monkeypatch, capsys
    ):
        """When running the first combo writes the pv row a later combo needs,
        the runner's per-combo recheck skips the now-present combo instead of
        re-running it (exercises the `[skip] now present` resume path)."""
        fw = _make_framework(
            tmp_path,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv", tier="tiny", system_threads="1")]},
        )
        patch_globals(fw)
        pv_path = fw / "optimized_task" / "cat" / "mgcv" / "package_verify.tsv"

        def fake_run_one(combo, reps, timeout):
            # Running (tiny,4) also lands the (tiny,8) row (concurrent attest).
            existing = pv_path.read_text(encoding="utf-8")
            for th in (str(combo.thread), "8"):
                existing += "\t".join(
                    str(_pv_row(patch_name="mgcv", tier="tiny",
                                system_threads=th).get(c, ""))
                    for c in PV_HEADER
                ) + "\n"
            pv_path.write_text(existing, encoding="utf-8")
            return 0

        monkeypatch.setattr(asw, "_run_one", fake_run_one)
        rc = cmd_attest_sweep(_sweep_args(tiers="tiny", threads="4,8"))
        assert rc == 0
        out = capsys.readouterr().out
        assert "[skip] now present mgcv:tiny:8" in out


# ==========================================================================
# _run_one / _kill subprocess boundary (monkeypatched, no real subprocess)
# ==========================================================================

class _FakeProc:
    def __init__(self, rc=0, raise_timeout=False, pid=4242):
        self.returncode = rc
        self.pid = pid
        self._raise_timeout = raise_timeout
        self.killed = False

    def wait(self, timeout=None):
        if self._raise_timeout and timeout is not None:
            import subprocess as _sp
            self._raise_timeout = False  # only first wait() times out
            raise _sp.TimeoutExpired(cmd="x", timeout=timeout)
        return self.returncode

    def kill(self):
        self.killed = True

    def send_signal(self, sig):
        pass


class TestRunOneBoundary:
    def test_run_one_success(self, tmp_path, patch_globals, monkeypatch):
        fw = _make_framework(tmp_path, py_patches={"mgcv": {}})
        patch_globals(fw)
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            captured["cwd"] = kw.get("cwd")
            return _FakeProc(rc=0)

        monkeypatch.setattr(asw.subprocess, "Popen", fake_popen)
        combo = Combo("mgcv", "py", tmp_path, "tiny", 8)
        rc = asw._run_one(combo, reps=2, timeout=0)
        assert rc == 0
        # the attest invocation carries the combo's task dir, name, tier, threads
        assert "attest" in captured["cmd"]
        assert "--name" in captured["cmd"] and "mgcv" in captured["cmd"]
        assert "8" in captured["cmd"]
        assert captured["cwd"] == fw
        # a log file was written
        logs = list((fw / "attest_logs" / "sweep").glob("*.log"))
        assert len(logs) == 1
        assert "[done] rc=0" in logs[0].read_text(encoding="utf-8")

    def test_run_one_timeout_kills_and_returns_124(
        self, tmp_path, patch_globals, monkeypatch
    ):
        fw = _make_framework(tmp_path, py_patches={"mgcv": {}})
        patch_globals(fw)
        proc = _FakeProc(rc=0, raise_timeout=True)
        monkeypatch.setattr(asw.subprocess, "Popen", lambda *a, **k: proc)
        killed = []
        monkeypatch.setattr(asw, "_kill", lambda p: killed.append(p))
        combo = Combo("mgcv", "py", tmp_path, "tiny", 1)
        rc = asw._run_one(combo, reps=1, timeout=5)
        assert rc == 124
        assert killed == [proc]
        log = next((fw / "attest_logs" / "sweep").glob("*.log"))
        assert "[timeout]" in log.read_text(encoding="utf-8")

    def test_kill_posix_sigterm_path(self, monkeypatch):
        # On non-Windows, _kill goes through os.killpg(SIGTERM) then wait.
        monkeypatch.setattr(asw.platform, "system", lambda: "Darwin")
        sent = []
        monkeypatch.setattr(asw.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
        proc = _FakeProc(pid=999)
        asw._kill(proc)
        assert sent and sent[0][0] == 999

    def test_kill_posix_escalates_to_sigkill_on_timeout(self, monkeypatch):
        import subprocess as _sp
        monkeypatch.setattr(asw.platform, "system", lambda: "Linux")
        sent = []

        def fake_killpg(pid, sig):
            sent.append(sig)

        monkeypatch.setattr(asw.os, "killpg", fake_killpg)

        class _TOProc(_FakeProc):
            def __init__(self):
                super().__init__()
                self._waits = 0

            def wait(self, timeout=None):
                self._waits += 1
                if self._waits == 1:
                    raise _sp.TimeoutExpired(cmd="x", timeout=timeout)
                return 0

        asw._kill(_TOProc())
        # SIGTERM first, then SIGKILL after the timeout
        assert len(sent) == 2

    def test_kill_posix_process_already_gone(self, monkeypatch):
        # os.killpg raises ProcessLookupError when the group already exited;
        # _kill swallows it and still calls wait().
        monkeypatch.setattr(asw.platform, "system", lambda: "Darwin")

        def boom(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(asw.os, "killpg", boom)
        import subprocess as _sp

        class _TOProc(_FakeProc):
            def wait(self, timeout=None):
                # First wait (with timeout=30) times out -> escalation branch;
                # final wait (timeout=None) returns normally like a real exit.
                if timeout is not None:
                    raise _sp.TimeoutExpired(cmd="x", timeout=timeout)
                return 0

        # first killpg ProcessLookupError -> except -> second killpg also raises
        # ProcessLookupError (swallowed) -> final wait() returns.
        asw._kill(_TOProc())  # must not propagate

    def test_kill_windows_ctrl_break_then_kill(self, monkeypatch):
        monkeypatch.setattr(asw.platform, "system", lambda: "Windows")
        # CTRL_BREAK_EVENT may not exist on this OS's signal module; inject it.
        monkeypatch.setattr(asw.signal, "CTRL_BREAK_EVENT", 0, raising=False)
        proc = _FakeProc()
        # send_signal succeeds and wait returns -> early return, no kill
        asw._kill(proc)
        assert proc.killed is False

    def test_find_task_dir_skips_unreadable_pv(self, tmp_path, patch_globals, monkeypatch):
        # OSError while reading a pv.tsv is swallowed (continue), returns None.
        fw = _make_framework(
            tmp_path,
            py_patches={"mgcv": {}},
            pv_rows={"mgcv": [_pv_row(patch_name="mgcv")]},
        )
        patch_globals(fw)
        import builtins
        real_open = builtins.open

        def boom_open(path, *a, **k):
            if str(path).endswith("package_verify.tsv"):
                raise OSError("unreadable")
            return real_open(path, *a, **k)

        # _find_task_dir opens pv via Path.open; patch Path.open instead
        real_path_open = Path.open

        def boom_path_open(self, *a, **k):
            if self.name == "package_verify.tsv":
                raise OSError("unreadable")
            return real_path_open(self, *a, **k)

        monkeypatch.setattr(Path, "open", boom_path_open)
        assert _find_task_dir("mgcv") is None
