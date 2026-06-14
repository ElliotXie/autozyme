"""Unit tests for zyme.commands.attest — the `zyme attest` CLI shell.

attest is a thin shell over autozyme.verify_patch (Py) / autozyme::verify_patch
(R). The pure layer here is: patch-name inference, interpreter resolution,
--threads / --tiers / --skip-tiers parsing, task threading policy, the cached-OOM
gate (host fingerprint matching + sentinel detection), the verify_patch command
construction (Py and R code strings), the publish-target resolution + per-platform
shard write, the new-rows all-pass detector, and duration formatting. The
subprocess-launching `cmd_attest` body is driven via --dry-run (no subprocess)
and via a monkeypatched `subprocess.call` boundary.

`tests/test_attest_publish.py` (wave 1) already covers _publish_after_attest /
_cached_oom_tiers / _new_attest_rows_all_pass / _resolve_manifest_publish_target
end-to-end; these tests probe the surrounding helpers and the cmd_attest driver.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import zyme.commands.attest as at
from zyme.commands.attest import (
    _attest_env,
    _build_attest_cmd,
    _cached_oom_tiers,
    _current_attempt_fingerprint,
    _detect_cpu_model,
    _detect_ram_gb,
    _file_stamp,
    _fmt_duration,
    _infer_patch_name,
    _new_attest_rows_all_pass,
    _parse_float_cell,
    _parse_int_cell,
    _parse_threads_arg,
    _parse_tiers_selection,
    _preflight_check_tsv_header,
    _publish_after_attest,
    _publish_filter_for_task,
    _python_interpreter,
    _read_verify_rows,
    _resolve_generic_publish_target,
    _resolve_manifest_publish_target,
    _resolve_one_task_dir,
    _rscript_interpreter,
    _same_host_thread,
    _thread_label,
    _thread_values_for_task,
    _warn_if_stale_rlibs,
    cmd_attest,
    _DEFAULT_TIERS,
)


# ==========================================================================
# package_verify fixtures
# ==========================================================================

PV_PASS = """\
timestamp\tpatch_name\ttier\tdataset\trep_idx\tvariant\tsec\tspeedup_pct\tspeedup_x\tpeak_mb\tpeak_mb_change_pct\tpeak_mb_fold\tpass\tmetrics_json\tframework_version\tpackage_version\tnote\tsystem_os\tsystem_cpu\tsystem_ram_gb\tsystem_threads
2026-06-01T00:00:00\tmgcv\tsmall\tds\t1\tbaseline\t3.0\t\t\t1000\t\t\t\t{}\t0.3.0\t1.0\t\tmacOS 24\tarm64\t16.0\t1
2026-06-01T00:00:00\tmgcv\tsmall\tds\t1\tpatched\t0.1\t96.7\t30.0\t900\t10.0\t1.1\ttrue\t{"max_rel_err": 0}\t0.3.0\t1.0\t\tmacOS 24\tarm64\t16.0\t1
"""


def _write_pv(task_dir: Path, text: str = PV_PASS) -> Path:
    p = task_dir / "package_verify.tsv"
    p.write_text(text, encoding="utf-8")
    return p


# ==========================================================================
# _infer_patch_name (stage 2 fallback from task.yaml::target_function)
# ==========================================================================

class TestInferPatchName:
    def _td(self, tmp_path, target):
        (tmp_path / "task.yaml").write_text(
            f"target_function: {target}\n", encoding="utf-8"
        )
        return tmp_path

    def test_double_colon_takes_pkg(self, tmp_path):
        assert _infer_patch_name(self._td(tmp_path, "mgcv::gam")) == "mgcv"

    def test_dotted_takes_first_segment(self, tmp_path):
        assert _infer_patch_name(
            self._td(tmp_path, "lifelines.CoxPHFitter.fit")
        ) == "lifelines"

    def test_bare_name(self, tmp_path):
        assert _infer_patch_name(self._td(tmp_path, "foofunc")) == "foofunc"

    def test_placeholder_returns_none(self, tmp_path):
        assert _infer_patch_name(self._td(tmp_path, "<PKG::FUNC>")) is None

    def test_empty_returns_none(self, tmp_path):
        assert _infer_patch_name(self._td(tmp_path, "")) is None

    def test_no_task_yaml_returns_none(self, tmp_path):
        assert _infer_patch_name(tmp_path) is None


# ==========================================================================
# interpreter resolution
# ==========================================================================

class TestInterpreters:
    def test_python_falls_back_to_sys_executable(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        assert _python_interpreter(tmp_path) == sys.executable

    def test_python_no_yaml_falls_back(self, tmp_path):
        assert _python_interpreter(tmp_path) == sys.executable

    def test_rscript_default(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        assert _rscript_interpreter(tmp_path) == "Rscript"

    def test_rscript_honors_executor(self, tmp_path):
        (tmp_path / "task.yaml").write_text(
            "executor:\n  rscript: /opt/R/bin/Rscript\n"
        )
        assert _rscript_interpreter(tmp_path) == "/opt/R/bin/Rscript"


# ==========================================================================
# _resolve_one_task_dir
# ==========================================================================

class TestResolveOneTaskDir:
    def test_valid(self, tmp_path):
        (tmp_path / "task.yaml").write_text("x: 1\n")
        assert _resolve_one_task_dir(str(tmp_path)) == tmp_path.resolve()

    def test_not_a_dir(self, tmp_path):
        with pytest.raises(SystemExit):
            _resolve_one_task_dir(str(tmp_path / "missing"))

    def test_no_task_yaml(self, tmp_path):
        with pytest.raises(SystemExit):
            _resolve_one_task_dir(str(tmp_path))


# ==========================================================================
# _parse_threads_arg
# ==========================================================================

class TestParseThreadsArg:
    def test_none_means_task_default(self):
        assert _parse_threads_arg(None) == [None]

    def test_single(self):
        assert _parse_threads_arg("4") == [4]

    def test_comma_and_semicolon(self):
        assert _parse_threads_arg("1,4;8") == [1, 4, 8]

    def test_dedup_preserves_order(self):
        assert _parse_threads_arg("4,4,1") == [4, 1]

    def test_full_maps_to_cpu_count(self, monkeypatch):
        monkeypatch.setattr(at.os, "cpu_count", lambda: 12)
        assert _parse_threads_arg("full") == [12]
        assert _parse_threads_arg("max") == [12]
        assert _parse_threads_arg("all") == [12]

    def test_non_integer_dies(self):
        with pytest.raises(SystemExit):
            _parse_threads_arg("bogus")

    def test_below_one_dies(self):
        with pytest.raises(SystemExit):
            _parse_threads_arg("0")

    def test_empty_after_split_dies(self):
        with pytest.raises(SystemExit):
            _parse_threads_arg(",,")

    def test_thread_label(self):
        assert _thread_label(None) == "task-default"
        assert _thread_label(8) == "8t"


# ==========================================================================
# _thread_values_for_task (threading: not_applicable policy)
# ==========================================================================

class TestThreadValuesForTask:
    def _td(self, tmp_path, threading="default"):
        (tmp_path / "task.yaml").write_text(f"threading: {threading}\n")
        return tmp_path

    def test_default_passthrough(self, tmp_path):
        td = self._td(tmp_path, "default")
        assert _thread_values_for_task(td, [1, 4, 8], allow_not_applicable=False) == [1, 4, 8]

    def test_not_applicable_forces_1t(self, tmp_path, capsys):
        td = self._td(tmp_path, "not_applicable")
        out = _thread_values_for_task(td, [1, 4, 8], allow_not_applicable=False)
        assert out == [1]
        assert "not_applicable" in capsys.readouterr().err

    def test_allow_override_keeps_multi(self, tmp_path, capsys):
        td = self._td(tmp_path, "not_applicable")
        out = _thread_values_for_task(td, [1, 4], allow_not_applicable=True)
        assert out == [1, 4]
        assert "WARNING" in capsys.readouterr().err

    def test_not_applicable_already_1t_no_drop_message(self, tmp_path, capsys):
        td = self._td(tmp_path, "not_applicable")
        out = _thread_values_for_task(td, [1], allow_not_applicable=False)
        assert out == [1]


# ==========================================================================
# _parse_tiers_selection
# ==========================================================================

class TestParseTiersSelection:
    def test_none_when_both_unset(self):
        assert _parse_tiers_selection(None, None) is None

    def test_explicit_tiers(self):
        assert _parse_tiers_selection("tiny,large", None) == ("tiny", "large")

    def test_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            _parse_tiers_selection("tiny", "large")

    def test_tiers_empty_after_split(self):
        with pytest.raises(SystemExit):
            _parse_tiers_selection(",,", None)

    def test_skip_tiers_removes_from_default(self):
        out = _parse_tiers_selection(None, "ood_xlarge")
        assert out == tuple(t for t in _DEFAULT_TIERS if t != "ood_xlarge")

    def test_skip_unknown_tier_dies(self):
        with pytest.raises(SystemExit):
            _parse_tiers_selection(None, "nonexistent_tier")

    def test_skip_empty_dies(self):
        with pytest.raises(SystemExit):
            _parse_tiers_selection(None, ",,")

    def test_skip_all_dies(self):
        with pytest.raises(SystemExit):
            _parse_tiers_selection(None, ",".join(_DEFAULT_TIERS))


# ==========================================================================
# _publish_filter_for_task
# ==========================================================================

class TestPublishFilterForTask:
    def test_default_no_max_threads(self, tmp_path):
        (tmp_path / "task.yaml").write_text("threading: default\n")
        flt = _publish_filter_for_task(tmp_path, all_pass_only=True, allow_not_applicable=False)
        assert flt.max_threads is None
        assert flt.all_pass_only is True
        assert flt.select == "full"

    def test_not_applicable_caps_threads(self, tmp_path):
        (tmp_path / "task.yaml").write_text("threading: not_applicable\n")
        flt = _publish_filter_for_task(tmp_path, all_pass_only=True, allow_not_applicable=False)
        assert flt.max_threads == 1

    def test_allow_not_applicable_lifts_cap(self, tmp_path):
        (tmp_path / "task.yaml").write_text("threading: not_applicable\n")
        flt = _publish_filter_for_task(tmp_path, all_pass_only=False, allow_not_applicable=True)
        assert flt.max_threads is None


# ==========================================================================
# numeric cell parsers
# ==========================================================================

class TestCellParsers:
    @pytest.mark.parametrize("raw,expected", [
        ("5", 5), (" 7 ", 7), ("3.9", 3), ("", None), (None, None),
        ("NA", None), ("nan", None), ("none", None), ("xx", None),
    ])
    def test_parse_int_cell(self, raw, expected):
        assert _parse_int_cell(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("5.5", 5.5), (" 7 ", 7.0), ("", None), (None, None),
        ("NA", None), ("NaN", None), ("NONE", None), ("xx", None),
    ])
    def test_parse_float_cell(self, raw, expected):
        assert _parse_float_cell(raw) == expected


# ==========================================================================
# host fingerprint matching
# ==========================================================================

class TestSameHostThread:
    def _cur(self, **ov):
        base = {"platform": "mac", "cpu": "M1", "ram_gb": 16.0, "threads": 1}
        base.update(ov)
        return base

    def test_match(self):
        row = {"system_os": "macOS 24", "system_cpu": "M1",
               "system_ram_gb": "16.0", "system_threads": "1"}
        assert _same_host_thread(row, self._cur()) is True

    def test_platform_mismatch(self):
        row = {"system_os": "Windows", "system_cpu": "M1",
               "system_ram_gb": "16.0", "system_threads": "1"}
        assert _same_host_thread(row, self._cur()) is False

    def test_cpu_mismatch(self):
        row = {"system_os": "macOS", "system_cpu": "Intel",
               "system_ram_gb": "16.0", "system_threads": "1"}
        assert _same_host_thread(row, self._cur()) is False

    def test_ram_within_tolerance(self):
        row = {"system_os": "macOS", "system_cpu": "M1",
               "system_ram_gb": "16.3", "system_threads": "1"}
        assert _same_host_thread(row, self._cur()) is True

    def test_ram_outside_tolerance(self):
        row = {"system_os": "macOS", "system_cpu": "M1",
               "system_ram_gb": "32.0", "system_threads": "1"}
        assert _same_host_thread(row, self._cur()) is False

    def test_threads_mismatch(self):
        row = {"system_os": "macOS", "system_cpu": "M1",
               "system_ram_gb": "16.0", "system_threads": "8"}
        assert _same_host_thread(row, self._cur()) is False

    def test_blank_cpu_does_not_disqualify(self):
        row = {"system_os": "macOS", "system_cpu": "",
               "system_ram_gb": "", "system_threads": ""}
        assert _same_host_thread(row, self._cur()) is True


# ==========================================================================
# _current_attempt_fingerprint
# ==========================================================================

class TestCurrentFingerprint:
    def test_reads_thread_env(self, monkeypatch):
        monkeypatch.setattr(at, "_detect_cpu_model", lambda: "TestCPU")
        monkeypatch.setattr(at, "_detect_ram_gb", lambda: 16.0)
        fp = _current_attempt_fingerprint({"ZYME_THREADS": "4"})
        assert fp["threads"] == 4
        assert fp["cpu"] == "TestCPU"
        assert fp["ram_gb"] == 16.0
        assert fp["platform"] in {"mac", "win", "unknown"}

    def test_falls_through_thread_env_keys(self, monkeypatch):
        monkeypatch.setattr(at, "_detect_cpu_model", lambda: "")
        monkeypatch.setattr(at, "_detect_ram_gb", lambda: None)
        fp = _current_attempt_fingerprint({"OMP_NUM_THREADS": "2"})
        assert fp["threads"] == 2


# ==========================================================================
# system probes (smoke — must not raise, return correct types)
# ==========================================================================

class TestSystemProbes:
    def test_detect_cpu_model_returns_str(self):
        assert isinstance(_detect_cpu_model(), str)

    def test_detect_ram_gb_type(self):
        ram = _detect_ram_gb()
        assert ram is None or isinstance(ram, float)


# ==========================================================================
# _build_attest_cmd — Py + R code-string construction
# ==========================================================================

class TestBuildAttestCmd:
    def _td(self, tmp_path, lang="py"):
        (tmp_path / "task.yaml").write_text("target_function: mgcv::gam\n")
        return tmp_path

    def test_py_basic(self, tmp_path):
        td = self._td(tmp_path)
        cmd, name = _build_attest_cmd(td, "mgcv", "tiny", None, 2, "py")
        assert name == "mgcv"
        assert cmd[0] == sys.executable
        assert cmd[1] == "-c"
        code = cmd[2]
        assert "autozyme.verify_patch('mgcv'" in code
        assert "reps=2" in code
        assert "tiers=('tiny',)" in code

    def test_py_rerun_baseline_and_flags(self, tmp_path):
        td = self._td(tmp_path)
        cmd, _ = _build_attest_cmd(
            td, "mgcv", None, None, 3, "py",
            rerun_baseline=True, no_baseline_confirm=True,
            baseline_confirm_sigma=2.0, patched_only=True,
        )
        code = cmd[2]
        assert "use_baseline_cache=False" in code
        assert "no_baseline_confirm=True" in code
        assert "baseline_confirm_sigma=2.0" in code
        assert "patched_only=True" in code
        # no tiers selected -> no tiers kwarg
        assert "tiers=" not in code

    def test_r_basic(self, tmp_path):
        td = self._td(tmp_path)
        cmd, name = _build_attest_cmd(td, "mgcv", "tiny,large", None, 2, "R")
        assert cmd[0] == "Rscript"
        assert cmd[1] == "-e"
        code = cmd[2]
        assert "autozyme::verify_patch('mgcv'" in code
        assert "reps = 2L" in code
        assert "c('tiny', 'large')" in code
        assert "taskkill" in code  # Windows teardown clause always present

    def test_r_warns_on_py_only_flags(self, tmp_path, capsys):
        td = self._td(tmp_path)
        _build_attest_cmd(td, "mgcv", None, None, 2, "R", rerun_baseline=True)
        assert "Python-only" in capsys.readouterr().err

    def test_tiers_override_wins(self, tmp_path):
        td = self._td(tmp_path)
        cmd, _ = _build_attest_cmd(
            td, "mgcv", "tiny", None, 2, "py",
            tiers_override=("medium", "large"),
        )
        assert "('medium', 'large')" in cmd[2]

    def test_unknown_lang_dies(self, tmp_path):
        td = self._td(tmp_path)
        with pytest.raises(SystemExit):
            _build_attest_cmd(td, "mgcv", None, None, 2, "rust")

    def test_uninferable_patch_name_dies(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: <PKG::FUNC>\n")
        with pytest.raises(SystemExit):
            _build_attest_cmd(tmp_path, None, None, None, 2, "py")


# ==========================================================================
# _file_stamp / _read_verify_rows
# ==========================================================================

class TestFileStampAndRows:
    def test_file_stamp_missing(self, tmp_path):
        assert _file_stamp(tmp_path / "nope") == (0, -1)

    def test_file_stamp_changes_on_write(self, tmp_path):
        p = tmp_path / "f.tsv"
        p.write_text("a")
        s1 = _file_stamp(p)
        assert s1 != (0, -1)
        p.write_text("abcdef")
        assert _file_stamp(p)[1] != s1[1]  # size changed

    def test_read_verify_rows_missing(self, tmp_path):
        assert _read_verify_rows(tmp_path / "nope.tsv") == []

    def test_read_verify_rows_parses(self, tmp_path):
        p = _write_pv(tmp_path)
        rows = _read_verify_rows(p)
        assert len(rows) == 2
        assert rows[0]["variant"] == "baseline"


# ==========================================================================
# _new_attest_rows_all_pass
# ==========================================================================

class TestNewRowsAllPass:
    def test_new_passing_batch(self, tmp_path):
        after = _read_verify_rows(_write_pv(tmp_path))
        assert _new_attest_rows_all_pass([], after) is True

    def test_no_new_timestamps(self, tmp_path):
        after = _read_verify_rows(_write_pv(tmp_path))
        # before == after -> no new timestamps -> False
        assert _new_attest_rows_all_pass(after, after) is False

    def test_failed_patched_row(self, tmp_path):
        failed = PV_PASS.replace("\ttrue\t", "\tfalse\t")
        after = _read_verify_rows(_write_pv(tmp_path, failed))
        assert _new_attest_rows_all_pass([], after) is False


# ==========================================================================
# _preflight_check_tsv_header
# ==========================================================================

class TestPreflightCheckTsvHeader:
    def test_missing_file_no_error(self, tmp_path):
        _preflight_check_tsv_header(tmp_path / "nope.tsv", "R")  # no raise

    def test_r_extra_column_warns(self, tmp_path, capsys):
        p = tmp_path / "pv.tsv"
        p.write_text("timestamp\tpatch_name\tbogus_extra_col\n1\t2\t3\n")
        _preflight_check_tsv_header(p, "R")
        assert "bogus_extra_col" in capsys.readouterr().err

    def test_r_known_columns_no_warn(self, tmp_path, capsys):
        p = _write_pv(tmp_path)
        _preflight_check_tsv_header(p, "R")
        # package_version etc. are all known R cols -> no extra-column warning
        err = capsys.readouterr().err
        assert "columns not in" not in err

    def test_py_never_warns(self, tmp_path, capsys):
        p = tmp_path / "pv.tsv"
        p.write_text("timestamp\tpatch_name\tanything\n1\t2\t3\n")
        _preflight_check_tsv_header(p, "py")
        assert "columns not in" not in capsys.readouterr().err


# ==========================================================================
# _fmt_duration
# ==========================================================================

class TestFmtDuration:
    @pytest.mark.parametrize("secs,expected", [
        (5.0, "5.0s"),
        (59.9, "59.9s"),
        (90.0, "1.5m"),
        (3600.0, "1.00h"),
        (7200.0, "2.00h"),
    ])
    def test_fmt(self, secs, expected):
        assert _fmt_duration(secs) == expected


# ==========================================================================
# _warn_if_stale_rlibs
# ==========================================================================

class TestWarnIfStaleRlibs:
    def test_env_skip_silences(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("AUTOZYME_SKIP_RLIBS_CHECK", "1")
        _warn_if_stale_rlibs(tmp_path, tmp_path / "rl")
        assert capsys.readouterr().err == ""

    def test_missing_desc_no_warn(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("AUTOZYME_SKIP_RLIBS_CHECK", raising=False)
        # no DESCRIPTION on disk -> returns silently
        _warn_if_stale_rlibs(tmp_path, tmp_path / "rl")
        assert capsys.readouterr().err == ""

    def test_warns_when_source_newer(self, tmp_path, monkeypatch, capsys):
        import os
        import time
        monkeypatch.delenv("AUTOZYME_SKIP_RLIBS_CHECK", raising=False)
        at._stale_rlibs_warned.clear()
        framework = tmp_path
        rlibs = tmp_path / "rl"
        desc = rlibs / "autozyme" / "DESCRIPTION"
        desc.parent.mkdir(parents=True)
        desc.write_text("Package: autozyme\n")
        # make the install look old
        old = time.time() - 1000
        os.utime(desc, (old, old))
        src = framework / "autozyme_r" / "R"
        src.mkdir(parents=True)
        newer = src / "verify.R"
        newer.write_text("x <- 1\n")  # mtime ~ now, newer than install
        _warn_if_stale_rlibs(framework, rlibs)
        assert "older than autozyme_r/ source" in capsys.readouterr().err

    def test_warns_once_only(self, tmp_path, monkeypatch, capsys):
        import os
        import time
        monkeypatch.delenv("AUTOZYME_SKIP_RLIBS_CHECK", raising=False)
        at._stale_rlibs_warned.clear()
        framework = tmp_path
        rlibs = tmp_path / "rl"
        desc = rlibs / "autozyme" / "DESCRIPTION"
        desc.parent.mkdir(parents=True)
        desc.write_text("Package: autozyme\n")
        old = time.time() - 1000
        os.utime(desc, (old, old))
        src = framework / "autozyme_r" / "R"
        src.mkdir(parents=True)
        (src / "verify.R").write_text("x <- 1\n")
        _warn_if_stale_rlibs(framework, rlibs)
        capsys.readouterr()
        _warn_if_stale_rlibs(framework, rlibs)  # second call: deduped
        assert capsys.readouterr().err == ""


# ==========================================================================
# _attest_env
# ==========================================================================

class TestAttestEnv:
    def test_returns_dict_with_thread_caps(self, tmp_path):
        (tmp_path / "task.yaml").write_text("threading: default\n")
        env = _attest_env(tmp_path, "py", "mgcv", threads=4)
        assert isinstance(env, dict)
        # apply_thread_env should have set thread-count vars
        assert any(
            k in env for k in
            ("ZYME_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        )


# ==========================================================================
# _resolve_generic_publish_target
# ==========================================================================

def _make_generic_framework(tmp_path: Path) -> tuple[Path, Path]:
    """Framework with one generic lifted-from py patch + its task dir."""
    framework = tmp_path / "autozyme-framework"
    (framework / "autozyme_cli").mkdir(parents=True)
    patch_dir = framework / "autozyme_py" / "src" / "autozyme" / "mgcvpatch"
    patch_dir.mkdir(parents=True)
    (patch_dir / "__init__.py").write_text(
        "# Lifted from autozyme task `test_mgcv`\n", encoding="utf-8"
    )
    task_dir = framework / "optimized_task" / "cat" / "test_mgcv"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text("target_function: mgcv::gam\n")
    return framework, task_dir


class TestResolveGenericPublishTarget:
    def test_resolves_to_patch_speedups(self, tmp_path):
        framework, task_dir = _make_generic_framework(tmp_path)
        resolved = _resolve_generic_publish_target(task_dir)
        assert resolved is not None
        task_name, dst = resolved
        assert task_name == "test_mgcv"
        assert dst.name == "speedups.tsv"
        assert dst.parent.name == "mgcvpatch"

    def test_none_when_no_lifted_marker(self, tmp_path):
        framework = tmp_path / "autozyme-framework"
        (framework / "autozyme_cli").mkdir(parents=True)
        task_dir = framework / "optimized_task" / "cat" / "test_orphan"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("x: 1\n")
        assert _resolve_generic_publish_target(task_dir) is None


# ==========================================================================
# cmd_attest — dry-run (no subprocess) + monkeypatched subprocess.call
# ==========================================================================

def _attest_args(task_dir, **ov):
    base = dict(
        task_dirs=[str(task_dir)], task_dir=str(task_dir), name="mgcv",
        tiers="tiny", skip_tiers=None, reps=2, threads="1",
        allow_not_applicable_threads=False, retry_oom=False, lang="py",
        dry_run=True, rerun_baseline=False, baseline_confirm_sigma=3.0,
        no_baseline_confirm=False, no_preflight=True, patched_only=False,
    )
    base.update(ov)
    return argparse.Namespace(**base)


class TestCmdAttestDryRun:
    def test_dry_run_prints_command_no_subprocess(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "task.yaml").write_text(
            "target_function: mgcv::gam\nthreading: default\n"
        )
        called = []
        monkeypatch.setattr(at.subprocess, "call",
                            lambda *a, **k: called.append(a) or 0)
        cmd_attest(_attest_args(tmp_path))
        assert called == []
        out = capsys.readouterr().out
        assert "verify_patch" in out
        assert "ZYME_THREADS=1" in out

    def test_dry_run_name_with_multiple_tasks_dies(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: mgcv::gam\n")
        td2 = tmp_path.parent / "td2"
        td2.mkdir()
        (td2 / "task.yaml").write_text("target_function: mgcv::gam\n")
        args = _attest_args(tmp_path, task_dirs=[str(tmp_path), str(td2)])
        with pytest.raises(SystemExit):
            cmd_attest(args)

    def test_reps_below_one_dies(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: mgcv::gam\n")
        with pytest.raises(SystemExit):
            cmd_attest(_attest_args(tmp_path, reps=0))

    def test_uninferable_name_dies(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: <PKG::FUNC>\n")
        with pytest.raises(SystemExit):
            cmd_attest(_attest_args(tmp_path, name=None))


class TestCmdAttestExecution:
    """Drive the real cmd_attest body with subprocess.call monkeypatched."""

    def _setup(self, tmp_path):
        (tmp_path / "task.yaml").write_text(
            "target_function: mgcv::gam\nthreading: default\n"
        )
        return tmp_path

    def test_single_task_success_publishes(self, tmp_path, monkeypatch):
        td = self._setup(tmp_path)

        def fake_call(cmd, env=None):
            # Simulate the worker writing a fresh passing pv batch.
            _write_pv(td)
            return 0

        monkeypatch.setattr(at.subprocess, "call", fake_call)
        published = []
        monkeypatch.setattr(
            at, "_publish_after_attest",
            lambda task_dir, **k: published.append(task_dir) or True,
        )
        # Single task, rc=0: function returns None (no sys.exit).
        rc = cmd_attest(_attest_args(td, dry_run=False))
        assert rc is None
        assert published == [td]

    def test_single_task_failure_exits_nonzero(self, tmp_path, monkeypatch):
        td = self._setup(tmp_path)
        # worker fails and writes nothing -> failure propagates via sys.exit
        monkeypatch.setattr(at.subprocess, "call", lambda *a, **k: 5)
        monkeypatch.setattr(at, "_publish_after_attest", lambda *a, **k: True)
        with pytest.raises(SystemExit) as ei:
            cmd_attest(_attest_args(td, dry_run=False))
        assert ei.value.code == 5

    def test_batch_summary_all_ok(self, tmp_path, monkeypatch, capsys):
        td = self._setup(tmp_path)

        def fake_call(cmd, env=None):
            _write_pv(td)
            return 0

        monkeypatch.setattr(at.subprocess, "call", fake_call)
        monkeypatch.setattr(at, "_publish_after_attest", lambda *a, **k: True)
        # batch mode = multiple thread values
        args = _attest_args(td, dry_run=False, threads="1,4")
        rc = cmd_attest(args)
        assert rc is None
        err = capsys.readouterr().err
        assert "SUMMARY" in err
        assert "2/2 OK" in err

    def test_worker_nonzero_but_wrote_passing_rows_tolerated(self, tmp_path, monkeypatch, capsys):
        td = self._setup(tmp_path)

        def fake_call(cmd, env=None):
            _write_pv(td)  # complete passing batch written
            return 3  # but worker exits nonzero (e.g. R teardown crash)

        monkeypatch.setattr(at.subprocess, "call", fake_call)
        monkeypatch.setattr(at, "_publish_after_attest", lambda *a, **k: True)
        rc = cmd_attest(_attest_args(td, dry_run=False))
        assert rc is None  # tolerated
        assert "treating attest as complete" in capsys.readouterr().err

    def test_patched_only_skips_publish(self, tmp_path, monkeypatch, capsys):
        td = self._setup(tmp_path)

        def fake_call(cmd, env=None):
            _write_pv(td)
            return 0

        monkeypatch.setattr(at.subprocess, "call", fake_call)
        published = []
        monkeypatch.setattr(at, "_publish_after_attest",
                            lambda *a, **k: published.append(1))
        cmd_attest(_attest_args(td, dry_run=False, patched_only=True))
        assert published == []
        assert "skipped (patched-only)" in capsys.readouterr().err

    def test_preflight_gate_runs_when_enabled(self, tmp_path, monkeypatch):
        td = self._setup(tmp_path)
        # preflight passes (rc=0) -> proceeds to (dry-run) command print
        from zyme.commands.package import preflight as pf
        monkeypatch.setattr(pf, "cmd_package_preflight", lambda args: 0)
        monkeypatch.setattr(at.subprocess, "call", lambda *a, **k: 0)
        # dry_run still prints, but the preflight gate is what we exercise
        cmd_attest(_attest_args(td, dry_run=True, no_preflight=False))

    def test_preflight_failure_aborts(self, tmp_path, monkeypatch):
        td = self._setup(tmp_path)
        from zyme.commands.package import preflight as pf
        monkeypatch.setattr(pf, "cmd_package_preflight", lambda args: 1)
        with pytest.raises(SystemExit):
            cmd_attest(_attest_args(td, dry_run=False, no_preflight=False))


# ==========================================================================
# _infer_patch_name stage 1 — authoritative lifted-from reverse index
# ==========================================================================

class TestInferPatchNameStage1:
    def test_resolves_via_lifted_from_index(self, tmp_path, monkeypatch):
        framework, task_dir = _make_generic_framework(tmp_path)
        # bust the lru_cache so the freshly-built tmp framework is indexed
        from zyme import scan
        scan._build_lifted_from_index.cache_clear()
        # task.yaml target is mgcv::gam, but the lifted-from index maps the
        # task dir name (test_mgcv) -> patch dir (mgcvpatch); stage 1 wins.
        assert _infer_patch_name(task_dir) == "mgcvpatch"
        scan._build_lifted_from_index.cache_clear()

    def test_r_legacy_dot_r_uses_stem(self, tmp_path):
        from zyme import scan
        framework = tmp_path / "autozyme-framework"
        (framework / "autozyme_cli").mkdir(parents=True)
        rp = framework / "autozyme_r" / "inst" / "patches"
        rp.mkdir(parents=True)
        (rp / "legacypatch.R").write_text(
            "# Lifted from autozyme task `test_lp`\n", encoding="utf-8"
        )
        task_dir = framework / "optimized_task" / "cat" / "test_lp"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("target_function: foo\n")
        scan._build_lifted_from_index.cache_clear()
        assert _infer_patch_name(task_dir) == "legacypatch"
        scan._build_lifted_from_index.cache_clear()


# ==========================================================================
# _cached_oom_tiers — sentinel detection + no-match cases (own fixtures)
# ==========================================================================

OOM_PV = """\
timestamp\tpatch_name\ttier\tdataset\trep_idx\tvariant\tsec\tspeedup_pct\tspeedup_x\tpeak_mb\tpeak_mb_change_pct\tpeak_mb_fold\tpass\tmetrics_json\tframework_version\tpackage_version\tnote\tsystem_os\tsystem_cpu\tsystem_ram_gb\tsystem_threads
2026-06-01T00:00:00\tmgcv\tlarge\tds\t\t\t\t\t\t\t\t\t\t\t0.3.0\t1.0\tOOM: memory limit reached\tmacOS 24\tarm64\t16.0\t1
"""


class TestCachedOomTiers:
    def _cur(self, **ov):
        base = {"platform": "mac", "cpu": "arm64", "ram_gb": 16.0, "threads": 1}
        base.update(ov)
        return base

    def test_no_pv_file_empty(self, tmp_path):
        assert _cached_oom_tiers(tmp_path, "mgcv", ("large",), self._cur()) == []

    def test_matching_oom_returned(self, tmp_path):
        (tmp_path / "package_verify.tsv").write_text(OOM_PV, encoding="utf-8")
        assert _cached_oom_tiers(tmp_path, "mgcv", ("large",), self._cur()) == ["large"]

    def test_different_host_not_matched(self, tmp_path):
        (tmp_path / "package_verify.tsv").write_text(OOM_PV, encoding="utf-8")
        assert _cached_oom_tiers(
            tmp_path, "mgcv", ("large",), self._cur(threads=8)
        ) == []

    def test_tier_not_requested(self, tmp_path):
        (tmp_path / "package_verify.tsv").write_text(OOM_PV, encoding="utf-8")
        assert _cached_oom_tiers(tmp_path, "mgcv", ("tiny",), self._cur()) == []

    def test_non_oom_pass_row_not_sentinel(self, tmp_path):
        _write_pv(tmp_path)  # passing small batch
        assert _cached_oom_tiers(tmp_path, "mgcv", ("small",), self._cur()) == []


# ==========================================================================
# _resolve_manifest_publish_target — seurat/scanpy per-step mapping
# ==========================================================================

def _make_manifest_framework(tmp_path: Path) -> tuple[Path, Path]:
    framework = tmp_path / "autozyme-framework"
    (framework / "autozyme_cli").mkdir(parents=True)
    task_dir = (
        framework / "optimized_task" / "test_seurat" / "normalize_data" / "v1"
    )
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text("target_function: Seurat::NormalizeData\n")
    scripts = framework / "scripts"
    scripts.mkdir()
    (scripts / "seurat_attest_manifest.yaml").write_text(
        "patch: seurat\nlang: R\ntasks:\n"
        "  - id: normalize\n"
        "    path: optimized_task/test_seurat/normalize_data/v1\n"
        "    legacy_key: normalize\n",
        encoding="utf-8",
    )
    return framework, task_dir


class TestResolveManifestPublishTarget:
    def test_resolves_seurat_method_dest(self, tmp_path):
        framework, task_dir = _make_manifest_framework(tmp_path)
        resolved = _resolve_manifest_publish_target(task_dir)
        assert resolved is not None
        task_id, dst = resolved
        assert task_id == "normalize"
        assert dst == (
            framework / "autozyme_r" / "inst" / "patches"
            / "seurat_normalize" / "speedups.tsv"
        )

    def test_package_speedups_false_returns_none(self, tmp_path):
        framework, task_dir = _make_manifest_framework(tmp_path)
        # rewrite manifest with package_speedups: false
        (framework / "scripts" / "seurat_attest_manifest.yaml").write_text(
            "tasks:\n"
            "  - id: normalize\n"
            "    path: optimized_task/test_seurat/normalize_data/v1\n"
            "    legacy_key: normalize\n"
            "    package_speedups: false\n",
            encoding="utf-8",
        )
        assert _resolve_manifest_publish_target(task_dir) is None

    def test_unmatched_task_returns_none(self, tmp_path):
        framework, task_dir = _make_manifest_framework(tmp_path)
        other = framework / "optimized_task" / "test_other" / "v1"
        other.mkdir(parents=True)
        (other / "task.yaml").write_text("x: 1\n")
        assert _resolve_manifest_publish_target(other) is None


# ==========================================================================
# _publish_after_attest — skip-mode + missing-src + per-platform shard write
# ==========================================================================

class TestPublishAfterAttest:
    def test_skip_mode_bypasses(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOZYME_ATTEST_PUBLISH_MODE", "skip")
        framework, task_dir = _make_generic_framework(tmp_path)
        _write_pv(task_dir)
        assert _publish_after_attest(task_dir) is False

    def test_missing_src_skips(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("AUTOZYME_ATTEST_PUBLISH_MODE", raising=False)
        framework, task_dir = _make_generic_framework(tmp_path)
        # no package_verify.tsv present
        assert _publish_after_attest(task_dir) is False
        assert "missing package_verify.tsv" in capsys.readouterr().err

    def test_writes_platform_shard(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AUTOZYME_ATTEST_PUBLISH_MODE", raising=False)
        from zyme import scan
        scan._build_lifted_from_index.cache_clear()
        framework, task_dir = _make_generic_framework(tmp_path)
        _write_pv(task_dir)
        ok = _publish_after_attest(task_dir)
        assert ok is True
        # macOS rows -> speedups.mac.tsv shard next to the patch
        shard = (
            framework / "autozyme_py" / "src" / "autozyme"
            / "mgcvpatch" / "speedups.mac.tsv"
        )
        assert shard.is_file()
        assert "patched" in shard.read_text(encoding="utf-8")
        scan._build_lifted_from_index.cache_clear()


# ==========================================================================
# cmd_attest cached-OOM skip path
# ==========================================================================

class TestCmdAttestCachedOom:
    def test_cached_oom_skips_tier_and_publishes(self, tmp_path, monkeypatch, capsys):
        from zyme import scan
        scan._build_lifted_from_index.cache_clear()
        framework, task_dir = _make_generic_framework(tmp_path)
        (task_dir / "task.yaml").write_text(
            "target_function: mgcv::gam\nthreading: default\n"
        )
        # seed a same-host OOM sentinel for the only requested tier
        (task_dir / "package_verify.tsv").write_text(
            OOM_PV.replace("\t1\n", "\t1\n"), encoding="utf-8"
        )
        # match the running host's fingerprint so the cache hits
        monkeypatch.setattr(at, "_current_attempt_fingerprint",
                            lambda env: {"platform": "mac", "cpu": "arm64",
                                         "ram_gb": 16.0, "threads": 1})
        monkeypatch.setattr(at, "_normalize_platform",
                            lambda v: "mac" if "mac" in (v or "").lower() else "win")
        called = []
        monkeypatch.setattr(at.subprocess, "call",
                            lambda *a, **k: called.append(a) or 0)
        published = []
        monkeypatch.setattr(at, "_publish_after_attest",
                            lambda *a, **k: published.append(1) or True)
        args = _attest_args(task_dir, dry_run=False, name="mgcv", tiers="large")
        cmd_attest(args)
        # all requested tiers cached-OOM -> worker never launched, publish runs
        assert called == []
        assert "cached OOM" in capsys.readouterr().err
        scan._build_lifted_from_index.cache_clear()


# ==========================================================================
# _attest_env R branch + thread-env fallback
# ==========================================================================

class TestAttestEnvBranches:
    def test_r_branch_sets_r_libs(self, tmp_path, monkeypatch):
        # Build a workspace with a real framework + .R_libs so the R branch
        # prepends R_LIBS. find_framework_root needs the autozyme-framework
        # child to resolve to the cwd parent.
        ws = tmp_path / "ws"
        framework = ws / "autozyme-framework"
        framework.mkdir(parents=True)
        rlibs = framework / ".R_libs"
        rlibs.mkdir()
        task_dir = ws / "task"
        task_dir.mkdir()
        (task_dir / "task.yaml").write_text("threading: default\n")
        monkeypatch.delenv("R_LIBS", raising=False)
        monkeypatch.setenv("AUTOZYME_SKIP_RLIBS_CHECK", "1")
        env = _attest_env(task_dir, "R", "seurat", threads=1)
        assert str(rlibs) in env.get("R_LIBS", "")

    def test_thread_env_fallback_on_exception(self, tmp_path, monkeypatch):
        # Force apply_task_thread_env to raise so the except-fallback runs.
        (tmp_path / "task.yaml").write_text("threading: default\n")
        import autozyme._thread_env as te

        def boom(*a, **k):
            raise RuntimeError("nope")

        monkeypatch.setattr(te, "apply_thread_env", boom)
        # The fallback uses setdefault, so clear any inherited caps first to
        # observe it filling them from str(threads).
        for var in ("ZYME_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS"):
            monkeypatch.delenv(var, raising=False)
        env = _attest_env(tmp_path, "py", "mgcv", threads=4)
        # fallback sets the thread caps directly to str(threads)
        assert env.get("ZYME_THREADS") == "4"
        assert env.get("OMP_NUM_THREADS") == "4"


# ==========================================================================
# _detect_cpu_model / _detect_ram_gb subprocess branches
# ==========================================================================

class TestDetectProbesMocked:
    def test_cpu_darwin_sysctl(self, monkeypatch):
        monkeypatch.setattr(at.platform, "system", lambda: "Darwin")

        def fake_run(cmd, **kw):
            return SimpleNamespace(stdout="Apple M9 Max\n", returncode=0)

        monkeypatch.setattr(at.subprocess, "run", fake_run)
        assert _detect_cpu_model() == "Apple M9 Max"

    def test_cpu_fallback_to_processor(self, monkeypatch):
        monkeypatch.setattr(at.platform, "system", lambda: "Plan9")
        monkeypatch.setattr(at.platform, "processor", lambda: "fakeproc")
        assert _detect_cpu_model() == "fakeproc"

    def test_ram_darwin_sysctl_when_psutil_missing(self, monkeypatch):
        # Force the psutil import inside _detect_ram_gb to fail -> sysctl path.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "psutil":
                raise ImportError
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(at.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            at.subprocess, "run",
            lambda cmd, **kw: SimpleNamespace(
                stdout=str(16 * 1024 ** 3), returncode=0
            ),
        )
        assert _detect_ram_gb() == 16.0


# ==========================================================================
# _publish_after_attest deeper branches
# ==========================================================================

class TestPublishAfterAttestBranches:
    def test_no_publishable_rows_skips(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("AUTOZYME_ATTEST_PUBLISH_MODE", raising=False)
        from zyme import scan
        scan._build_lifted_from_index.cache_clear()
        framework, task_dir = _make_generic_framework(tmp_path)
        # only a baseline row (no patched) + all_pass_only filter -> 0 rows
        only_baseline = "\n".join(PV_PASS.splitlines()[:2]) + "\n"
        (task_dir / "package_verify.tsv").write_text(only_baseline, encoding="utf-8")
        assert _publish_after_attest(task_dir) is False
        assert "no publishable rows" in capsys.readouterr().err
        scan._build_lifted_from_index.cache_clear()

    def test_already_up_to_date_second_publish(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("AUTOZYME_ATTEST_PUBLISH_MODE", raising=False)
        from zyme import scan
        scan._build_lifted_from_index.cache_clear()
        framework, task_dir = _make_generic_framework(tmp_path)
        _write_pv(task_dir)
        assert _publish_after_attest(task_dir) is True
        capsys.readouterr()
        # second publish of identical content -> "already up to date"
        assert _publish_after_attest(task_dir) is True
        assert "already up to date" in capsys.readouterr().err
        scan._build_lifted_from_index.cache_clear()

    def test_no_destination_skips(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("AUTOZYME_ATTEST_PUBLISH_MODE", raising=False)
        from zyme import scan
        scan._build_lifted_from_index.cache_clear()
        # framework with a task dir but no patch carrying its lifted-from marker
        framework = tmp_path / "autozyme-framework"
        (framework / "autozyme_cli").mkdir(parents=True)
        (framework / "autozyme_py" / "src" / "autozyme").mkdir(parents=True)
        task_dir = framework / "optimized_task" / "cat" / "test_orphan"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("target_function: foo\n")
        _write_pv(task_dir)
        assert _publish_after_attest(task_dir) is False
        assert "no package speedups destination" in capsys.readouterr().err
        scan._build_lifted_from_index.cache_clear()
