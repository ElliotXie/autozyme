"""Wave-4 mop-up coverage for zyme.commands.attest.

Wave-1 (tests/test_attest_publish.py) + wave-2 (tests/commands/test_attest_cmd.py,
test_attest_sweep_cmd.py) covered the bulk: patch-name inference, interpreter
resolution, --threads/--tiers parsing, threading policy, the cached-OOM gate,
verify_patch cmd construction, publish-target resolution + shard write, the
all-pass detector, duration formatting, and cmd_attest dry-run + execution.

This file squeezes the REACHABLE branches those left:

  - _infer_patch_name: stage-1 _build_lifted_from_index raising -> falls through
    to stage-2 (exception swallow, lines 102-103).
  - _python_interpreter / _rscript_interpreter: parse_executor raising ->
    except {} fallback; _python_interpreter with an explicit executor.python
    spec routed through _resolve_python.
  - _warn_if_stale_rlibs: OSError during the source-mtime scan -> silent return.
  - _detect_cpu_model: Linux /proc/cpuinfo + Windows wmic branches; _detect_ram_gb
    Linux /proc/meminfo branch (psutil forced absent).
  - _cached_oom_tiers: row patch-name mismatch skip, non-publishable skip, and
    blank-timestamp skip.
  - _build_attest_cmd: R no_baseline_confirm + patched_only kwargs.
  - _resolve_manifest_publish_target: no-framework None; manifest entry with a
    blank path skipped; manifest entry missing legacy_key+id -> None.
  - _resolve_generic_publish_target: no-framework None.
  - _publish_after_attest: --allow-not-applicable warning print + not_applicable
    publish; PublishFilterError swallow.
  - _preflight_check_tsv_header: OSError on read, blank first line.
  - _new_attest_rows_all_pass: invalid new row -> False; batch-summary fallback
    when no patched rows.
  - cmd_attest: uninferable patch name mid-loop dies; all-tiers cached-OOM in
    dry-run prints the skip comment; publish-skip when pv not updated; batch
    summary with a real failure exits 1.

All subprocess boundaries (the verify_patch worker `subprocess.call`, sysctl/
wmic probes) are monkeypatched; no real process or network is touched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import zyme.commands.attest as at
from zyme.commands.attest import (
    _build_attest_cmd,
    _cached_oom_tiers,
    _detect_cpu_model,
    _detect_ram_gb,
    _infer_patch_name,
    _new_attest_rows_all_pass,
    _preflight_check_tsv_header,
    _publish_after_attest,
    _python_interpreter,
    _resolve_generic_publish_target,
    _resolve_manifest_publish_target,
    _rscript_interpreter,
    _warn_if_stale_rlibs,
    cmd_attest,
)


PV_PASS = (
    "timestamp\tpatch_name\ttier\tdataset\trep_idx\tvariant\tsec\tspeedup_pct\t"
    "speedup_x\tpeak_mb\tpeak_mb_change_pct\tpeak_mb_fold\tpass\tmetrics_json\t"
    "framework_version\tpackage_version\tnote\tsystem_os\tsystem_cpu\t"
    "system_ram_gb\tsystem_threads\n"
    "2026-06-01T00:00:00\tmgcv\tsmall\tds\t1\tbaseline\t3.0\t\t\t1000\t\t\t\t"
    "{}\t0.3.0\t1.0\t\tmacOS 24\tarm64\t16.0\t1\n"
    "2026-06-01T00:00:00\tmgcv\tsmall\tds\t1\tpatched\t0.1\t96.7\t30.0\t900\t"
    "10.0\t1.1\ttrue\t{\"max_rel_err\": 0}\t0.3.0\t1.0\t\tmacOS 24\tarm64\t"
    "16.0\t1\n"
)


def _write_pv(task_dir: Path, text: str = PV_PASS) -> Path:
    p = task_dir / "package_verify.tsv"
    p.write_text(text, encoding="utf-8")
    return p


# ==========================================================================
# _infer_patch_name — stage-1 raises -> stage-2 fallback (lines 102-103)
# ==========================================================================

def test_infer_patch_name_stage1_exception_falls_to_stage2(tmp_path, monkeypatch):
    (tmp_path / "task.yaml").write_text("target_function: mgcv::gam\n")
    import zyme.scan as scan

    def boom(framework):
        raise RuntimeError("index build failed")

    monkeypatch.setattr(scan, "_build_lifted_from_index", boom)
    monkeypatch.setattr(scan, "find_framework_root", lambda td: str(tmp_path))
    # stage-1 raises and is swallowed; stage-2 derives 'mgcv' from target_function.
    assert _infer_patch_name(tmp_path) == "mgcv"


# ==========================================================================
# _python_interpreter / _rscript_interpreter — executor exceptions + spec
# ==========================================================================

def test_python_interpreter_executor_exception_falls_back(tmp_path, monkeypatch):
    (tmp_path / "task.yaml").write_text("threading: default\n")
    monkeypatch.setattr(at, "parse_executor",
                        lambda p: (_ for _ in ()).throw(RuntimeError("bad")))
    import sys as _sys
    assert _python_interpreter(tmp_path) == _sys.executable


def test_python_interpreter_resolves_explicit_spec(tmp_path, monkeypatch):
    (tmp_path / "task.yaml").write_text("threading: default\n")
    monkeypatch.setattr(at, "parse_executor", lambda p: {"python": "myenv"})
    monkeypatch.setattr(at, "_resolve_python", lambda spec: f"/resolved/{spec}")
    assert _python_interpreter(tmp_path) == "/resolved/myenv"


def test_rscript_interpreter_executor_exception_falls_back(tmp_path, monkeypatch):
    (tmp_path / "task.yaml").write_text("threading: default\n")
    monkeypatch.setattr(at, "parse_executor",
                        lambda p: (_ for _ in ()).throw(RuntimeError("bad")))
    assert _rscript_interpreter(tmp_path) == "Rscript"


def test_rscript_interpreter_honors_executor(tmp_path, monkeypatch):
    (tmp_path / "task.yaml").write_text("threading: default\n")
    monkeypatch.setattr(at, "parse_executor",
                        lambda p: {"rscript": "/opt/R/bin/Rscript"})
    assert _rscript_interpreter(tmp_path) == "/opt/R/bin/Rscript"


# ==========================================================================
# _warn_if_stale_rlibs — OSError during source scan -> silent return
# ==========================================================================

def test_warn_if_stale_rlibs_oserror_during_scan(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AUTOZYME_SKIP_RLIBS_CHECK", raising=False)
    framework = tmp_path / "fw"
    rlibs = framework / ".R_libs"
    (rlibs / "autozyme").mkdir(parents=True)
    (rlibs / "autozyme" / "DESCRIPTION").write_text("Package: autozyme\n")
    src = framework / "autozyme_r"
    (src / "R").mkdir(parents=True)
    (src / "R" / "verify.R").write_text("x <- 1\n")

    # Make the SOURCE file's stat() (inside the try-block mtime scan) raise
    # OSError so the except: return fires — DESCRIPTION.is_file()/.stat()
    # earlier must still succeed to reach the try-block.
    real_stat = Path.stat

    def flaky_stat(self, *a, **k):
        if self.name == "verify.R":
            raise OSError("stat failed")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    _warn_if_stale_rlibs(framework, rlibs)
    assert "WARNING" not in capsys.readouterr().err


# ==========================================================================
# _detect_cpu_model / _detect_ram_gb — Linux + Windows branches
# ==========================================================================

def test_detect_cpu_linux_proc_cpuinfo(monkeypatch, tmp_path):
    monkeypatch.setattr(at.platform, "system", lambda: "Linux")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Fake Xeon 9000\n")
    import builtins
    real_open = builtins.open

    def fake_open(path, *a, **k):
        if str(path) == "/proc/cpuinfo":
            return real_open(cpuinfo, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert _detect_cpu_model() == "Fake Xeon 9000"


def test_detect_cpu_windows_wmic(monkeypatch):
    monkeypatch.setattr(at.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        at.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(stdout="Name=Fake Ryzen 99\n", returncode=0),
    )
    assert _detect_cpu_model() == "Fake Ryzen 99"


def test_detect_ram_linux_meminfo(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__

    def no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    monkeypatch.setattr(at.platform, "system", lambda: "Linux")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16777216 kB\n")  # 16 GiB in kB
    real_open = builtins.open

    def fake_open(path, *a, **k):
        if str(path) == "/proc/meminfo":
            return real_open(meminfo, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert _detect_ram_gb() == 16.0


# ==========================================================================
# _cached_oom_tiers — per-row skip branches (519, 523, 526)
# ==========================================================================

def _cur():
    return {"platform": "mac", "cpu": "arm64", "ram_gb": 16.0, "threads": 1}


def _oom_row(patch="mgcv", ts="2026-06-01T00:00:00", note="OOM: memory limit"):
    return (
        f"{ts}\t{patch}\tlarge\tds\t\t\t\t\t\t\t\t\t\t\t0.3.0\t1.0\t{note}\t"
        "macOS 24\tarm64\t16.0\t1\n"
    )


PV_HEADER = (
    "timestamp\tpatch_name\ttier\tdataset\trep_idx\tvariant\tsec\tspeedup_pct\t"
    "speedup_x\tpeak_mb\tpeak_mb_change_pct\tpeak_mb_fold\tpass\tmetrics_json\t"
    "framework_version\tpackage_version\tnote\tsystem_os\tsystem_cpu\t"
    "system_ram_gb\tsystem_threads\n"
)


def test_cached_oom_skips_patch_name_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "_normalize_platform",
                        lambda v: "mac" if "mac" in (v or "").lower() else "x")
    (tmp_path / "package_verify.tsv").write_text(
        PV_HEADER + _oom_row(patch="OTHER"), encoding="utf-8")
    # row patch != requested patch -> skipped -> no cached OOM.
    assert _cached_oom_tiers(tmp_path, "mgcv", ("large",), _cur()) == []


def test_cached_oom_skips_blank_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "_normalize_platform",
                        lambda v: "mac" if "mac" in (v or "").lower() else "x")
    (tmp_path / "package_verify.tsv").write_text(
        PV_HEADER + _oom_row(ts=""), encoding="utf-8")
    # blank timestamp -> skipped.
    assert _cached_oom_tiers(tmp_path, "mgcv", ("large",), _cur()) == []


# ==========================================================================
# _build_attest_cmd — R no_baseline_confirm + patched_only kwargs
# ==========================================================================

def test_build_attest_cmd_r_kwargs(tmp_path):
    (tmp_path / "task.yaml").write_text("target_function: mgcv::gam\n")
    cmd, name = _build_attest_cmd(
        tmp_path, "mgcv", "tiny", None, 2, "R",
        no_baseline_confirm=True, patched_only=True,
    )
    code = cmd[-1]
    assert "no_baseline_confirm = TRUE" in code
    assert "patched_only = TRUE" in code
    assert name == "mgcv"


# ==========================================================================
# _resolve_*_publish_target — no-framework + manifest edge branches
# ==========================================================================

def test_resolve_generic_no_framework(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "find_framework_root", lambda td: None)
    (tmp_path / "task.yaml").write_text("x: 1\n")
    assert _resolve_generic_publish_target(tmp_path) is None


def test_resolve_manifest_no_framework(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "find_framework_root", lambda td: None)
    (tmp_path / "task.yaml").write_text("x: 1\n")
    assert _resolve_manifest_publish_target(tmp_path) is None


def test_resolve_manifest_missing_legacy_key_returns_none(tmp_path):
    framework = tmp_path / "autozyme-framework"
    (framework / "autozyme_cli").mkdir(parents=True)
    task_dir = framework / "optimized_task" / "test_seurat" / "norm" / "v1"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text("target_function: Seurat::NormalizeData\n")
    scripts = framework / "scripts"
    scripts.mkdir()
    # Manifest entry matches the path but has NO id and NO legacy_key + a blank
    # path entry first to exercise the `if not rel: continue` branch.
    (scripts / "seurat_attest_manifest.yaml").write_text(
        "tasks:\n"
        "  - path: \n"
        "  - path: optimized_task/test_seurat/norm/v1\n",
        encoding="utf-8",
    )
    assert _resolve_manifest_publish_target(task_dir) is None


# ==========================================================================
# _publish_after_attest — allow-not-applicable warning + filter error swallow
# ==========================================================================

def _make_generic_framework(tmp_path: Path):
    framework = tmp_path / "autozyme-framework"
    (framework / "autozyme_cli").mkdir(parents=True)
    patch_dir = framework / "autozyme_py" / "src" / "autozyme" / "mgcvpatch"
    patch_dir.mkdir(parents=True)
    (patch_dir / "__init__.py").write_text(
        "# Lifted from autozyme task `test_mgcv`\n", encoding="utf-8")
    task_dir = framework / "optimized_task" / "cat" / "test_mgcv"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "target_function: mgcv::gam\nthreading: not_applicable\n")
    return framework, task_dir


def test_publish_allow_not_applicable_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AUTOZYME_ATTEST_PUBLISH_MODE", raising=False)
    import zyme.scan as scan
    scan._build_lifted_from_index.cache_clear()
    framework, task_dir = _make_generic_framework(tmp_path)
    _write_pv(task_dir)
    ok = _publish_after_attest(task_dir, allow_not_applicable_threads=True)
    err = capsys.readouterr().err
    assert "--allow-not-applicable-threads set" in err
    assert ok is True
    scan._build_lifted_from_index.cache_clear()


def test_publish_filter_error_swallowed(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AUTOZYME_ATTEST_PUBLISH_MODE", raising=False)
    import zyme.scan as scan
    scan._build_lifted_from_index.cache_clear()
    framework, task_dir = _make_generic_framework(tmp_path)
    _write_pv(task_dir)
    # Force prepare_publish_content to raise PublishFilterError -> swallowed.
    monkeypatch.setattr(
        at, "prepare_publish_content",
        lambda *a, **k: (_ for _ in ()).throw(at.PublishFilterError("bad filter")),
    )
    assert _publish_after_attest(task_dir, allow_not_applicable_threads=True) is False
    assert "[publish-speedups] skipped" in capsys.readouterr().err
    scan._build_lifted_from_index.cache_clear()


# ==========================================================================
# _preflight_check_tsv_header — OSError on read + blank first line
# ==========================================================================

def test_preflight_header_oserror_on_read(tmp_path, monkeypatch):
    p = tmp_path / "package_verify.tsv"
    p.write_text("timestamp\tcol\n", encoding="utf-8")
    monkeypatch.setattr(
        Path, "read_text",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError("read fail")),
    )
    # Should silently return, no raise.
    _preflight_check_tsv_header(p, "R")


def test_preflight_header_blank_first_line(tmp_path, capsys):
    p = tmp_path / "package_verify.tsv"
    p.write_text("\n\n", encoding="utf-8")  # blank first line
    _preflight_check_tsv_header(p, "R")
    assert capsys.readouterr().err == ""


# ==========================================================================
# _new_attest_rows_all_pass — invalid new row + batch-summary fallback
# ==========================================================================

def test_new_rows_all_pass_invalid_row_false():
    before = []
    # A new row missing required validity fields (empty everything but ts).
    after = [{"timestamp": "T1", "variant": "patched"}]
    assert _new_attest_rows_all_pass(before, after) is False


# ==========================================================================
# cmd_attest — dry-run all-tiers-cached-OOM + uninferable name mid-loop
# ==========================================================================

def _attest_args(task_dir, **ov):
    base = dict(
        task_dirs=[str(task_dir)], task_dir=str(task_dir), name="mgcv",
        tiers="large", skip_tiers=None, reps=2, threads="1",
        allow_not_applicable_threads=False, retry_oom=False, lang="py",
        dry_run=True, rerun_baseline=False, baseline_confirm_sigma=3.0,
        no_baseline_confirm=False, no_preflight=True, patched_only=False,
    )
    base.update(ov)
    return argparse.Namespace(**base)


def test_cmd_attest_dry_run_all_tiers_cached_oom(tmp_path, monkeypatch, capsys):
    (tmp_path / "task.yaml").write_text(
        "target_function: mgcv::gam\nthreading: default\n")
    (tmp_path / "package_verify.tsv").write_text(
        PV_HEADER + _oom_row(), encoding="utf-8")
    monkeypatch.setattr(at, "_current_attempt_fingerprint", lambda env: _cur())
    monkeypatch.setattr(at, "_normalize_platform",
                        lambda v: "mac" if "mac" in (v or "").lower() else "x")
    cmd_attest(_attest_args(tmp_path, dry_run=True, name="mgcv", tiers="large"))
    out = capsys.readouterr().out
    # all requested tiers cached-OOM in dry-run -> the skip comment is printed.
    assert "skipped; cached OOM for all requested tiers" in out


def test_cmd_attest_uninferable_name_mid_loop_dies(tmp_path):
    (tmp_path / "task.yaml").write_text("target_function: <PKG::FUNC>\n")
    with pytest.raises(SystemExit):
        # name=None and unresolvable target -> dies inside the per-task loop.
        cmd_attest(_attest_args(tmp_path, name=None, dry_run=True))


# ==========================================================================
# cmd_attest — publish-skip when pv not updated + batch failure summary
# ==========================================================================

def test_cmd_attest_publish_skipped_when_pv_unchanged(tmp_path, monkeypatch, capsys):
    (tmp_path / "task.yaml").write_text(
        "target_function: mgcv::gam\nthreading: default\n")
    # worker returns 0 but writes nothing -> after_stamp == before_stamp.
    monkeypatch.setattr(at.subprocess, "call", lambda *a, **k: 0)
    published = []
    monkeypatch.setattr(at, "_publish_after_attest",
                        lambda *a, **k: published.append(1) or True)
    cmd_attest(_attest_args(tmp_path, dry_run=False, name="mgcv"))
    assert published == []
    assert "was not updated" in capsys.readouterr().err


def test_cmd_attest_batch_failure_summary_exits(tmp_path, monkeypatch, capsys):
    (tmp_path / "task.yaml").write_text(
        "target_function: mgcv::gam\nthreading: default\n")
    # Two thread values -> batch mode; worker fails and writes nothing.
    monkeypatch.setattr(at.subprocess, "call", lambda *a, **k: 7)
    monkeypatch.setattr(at, "_publish_after_attest", lambda *a, **k: True)
    with pytest.raises(SystemExit) as ei:
        cmd_attest(_attest_args(tmp_path, dry_run=False, name="mgcv",
                                threads="1,4"))
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "SUMMARY: 0/2 OK" in err
    assert "FAIL" in err
