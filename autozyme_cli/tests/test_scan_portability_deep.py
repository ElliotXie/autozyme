"""Deep coverage tests for zyme.scan_portability.

Targets gaps not covered by tests/test_scan_portability.py:
  - load_portability_scan corrupt-JSON path
  - _is_comment for cpp/default + comment-line skipping in _scan_file
  - _mac_only_guess for the python/platform.system guards
  - _collect_sources cpp glob + dedup
  - _compute_verdict NEEDS-KERNEL/portable-only/zyme-mclapply branches
  - render_table output (with rows + totals)
"""
from __future__ import annotations

from pathlib import Path

from zyme.scan_portability import (
    Hit,
    ScanResult,
    _compute_verdict,
    _is_comment,
    _mac_only_guess,
    _scan_file,
    load_portability_scan,
    portability_scan_path,
    render_table,
    save_portability_scan,
    scan_task,
)


def _scaffold(tmp_path: Path, name: str = "test_foo") -> Path:
    task = tmp_path / name
    task.mkdir()
    (task / "task.yaml").write_text(f"task: {name}\n")
    return task


def _write_pipeline(task: Path, content: str, lang: str = "R") -> None:
    (task / "pipeline").mkdir(parents=True, exist_ok=True)
    name = "run.R" if lang == "R" else "run.py"
    (task / "pipeline" / name).write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------
# load_portability_scan corrupt JSON
# --------------------------------------------------------------------------

class TestLoadPortabilityScan:
    def test_missing_returns_none(self, tmp_path: Path):
        assert load_portability_scan(tmp_path) is None

    def test_corrupt_json_returns_none(self, tmp_path: Path):
        p = portability_scan_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not valid json")
        assert load_portability_scan(tmp_path) is None


# --------------------------------------------------------------------------
# _is_comment
# --------------------------------------------------------------------------

class TestIsComment:
    def test_r_hash(self):
        assert _is_comment("# foo", "pipeline/run.R") is True
        assert _is_comment("x <- 1", "pipeline/run.R") is False

    def test_py_hash(self):
        assert _is_comment("# foo", "pipeline/run.py") is True

    def test_cpp_slashes(self):
        assert _is_comment("// foo", "kernel.cpp") is True
        assert _is_comment("/* foo", "kernel.cpp") is True
        assert _is_comment("int x = 1;", "kernel.cpp") is False

    def test_unknown_ext_never_comment(self):
        assert _is_comment("# foo", "data.txt") is False


# --------------------------------------------------------------------------
# _mac_only_guess
# --------------------------------------------------------------------------

class TestMacOnlyGuess:
    def test_r_os_type_not_windows(self):
        lines = ['if (.Platform$OS.type != "windows") {', '  mclapply(...)']
        assert _mac_only_guess(lines, 1) is True

    def test_r_os_type_unix(self):
        lines = ['if (.Platform$OS.type == "unix") {', '  mclapply(...)']
        assert _mac_only_guess(lines, 1) is True

    def test_py_sys_platform(self):
        lines = ['if sys.platform != "win32":', '    Pool()']
        assert _mac_only_guess(lines, 1) is True

    def test_py_platform_system(self):
        lines = ['if platform.system() != "Windows":', '    Pool()']
        assert _mac_only_guess(lines, 1) is True

    def test_no_guard(self):
        lines = ['x = 1', 'Pool()']
        assert _mac_only_guess(lines, 1) is False


# --------------------------------------------------------------------------
# _scan_file edge cases
# --------------------------------------------------------------------------

class TestScanFile:
    def test_oserror_returns_empty(self, tmp_path: Path):
        # Path to a non-existent file -> read_text raises OSError -> [].
        missing = tmp_path / "nope.R"
        hits, portable, mit = _scan_file(missing, tmp_path)
        assert hits == [] and portable == [] and mit == []

    def test_huge_file_skipped(self, tmp_path: Path):
        big = tmp_path / "big.R"
        big.write_text("mclapply(x)\n" + ("# pad\n" * 1) + "a" * 2_000_001)
        hits, _, _ = _scan_file(big, tmp_path)
        assert hits == []

    def test_comment_line_skipped(self, tmp_path: Path):
        f = tmp_path / "run.R"
        f.write_text("# parallel::mclapply(1:10, identity)\nx <- 1\n")
        hits, _, _ = _scan_file(f, tmp_path)
        # The mclapply is inside a comment -> not a hit.
        assert hits == []

    def test_long_line_skipped(self, tmp_path: Path):
        f = tmp_path / "run.R"
        f.write_text("mclapply(" + "x" * 2001 + ")\n")
        hits, _, _ = _scan_file(f, tmp_path)
        assert hits == []

    def test_portable_and_mitigation_collected(self, tmp_path: Path):
        f = tmp_path / "run.R"
        f.write_text(
            "RcppParallel::parallelFor(0, n, w)\n"
            ".zyme_mclapply(1:10, identity)\n"
        )
        hits, portable, mit = _scan_file(f, tmp_path)
        assert any(p["pattern"] == "rcpp_parallel" for p in portable)
        assert any(m["pattern"] == "zyme_mclapply" for m in mit)

    def test_crash_hit_with_note_when_guarded(self, tmp_path: Path):
        f = tmp_path / "run.R"
        f.write_text(
            'if (.Platform$OS.type != "windows") {\n'
            '  parallel::mclapply(1:10, identity)\n'
            '}\n'
        )
        hits, _, _ = _scan_file(f, tmp_path)
        crash = [h for h in hits if h.kind == "crash_on_win"]
        assert crash
        assert crash[0].mac_only_guess is True
        assert "Unix-only branch" in crash[0].note


# --------------------------------------------------------------------------
# _compute_verdict branches
# --------------------------------------------------------------------------

class TestComputeVerdict:
    def test_harmful_default(self):
        hits = [Hit("harmful_default", "high", "psock", "f", 1, "x")]
        verdict, run, _ = _compute_verdict(hits, [], [])
        assert verdict == "HARMFUL-DEFAULT" and run is True

    def test_crash_on_win_unguarded(self):
        hits = [Hit("crash_on_win", "high", "mclapply", "f", 1, "x",
                    mac_only_guess=False)]
        verdict, run, _ = _compute_verdict(hits, [], [])
        assert verdict == "CRASH-ON-WIN" and run is True

    def test_needs_kernel_review(self):
        hits = [Hit("review", "medium", "bplapply", "f", 1, "x")]
        verdict, run, _ = _compute_verdict(hits, [], [])
        assert verdict == "NEEDS-KERNEL" and run is True

    def test_mac_bonus_when_all_guarded(self):
        hits = [Hit("crash_on_win", "high", "mclapply", "f", 1, "x",
                    mac_only_guess=True)]
        verdict, run, _ = _compute_verdict(hits, [], [])
        assert verdict == "MAC_BONUS_ONLY" and run is False

    def test_mac_bonus_zyme_mclapply_only(self):
        mitigations = [{"pattern": "zyme_mclapply"}]
        verdict, run, _ = _compute_verdict([], mitigations, [])
        assert verdict == "MAC_BONUS_ONLY" and run is False

    def test_clean_portable_only(self):
        portable = [{"pattern": "openmp"}]
        verdict, run, _ = _compute_verdict([], [], portable)
        assert verdict == "CLEAN" and run is False

    def test_clean_no_hazards(self):
        verdict, run, _ = _compute_verdict([], [], [])
        assert verdict == "CLEAN" and run is False


# --------------------------------------------------------------------------
# _collect_sources cpp glob via scan_task
# --------------------------------------------------------------------------

class TestCollectSourcesCpp:
    def test_scans_cpp_in_pipeline(self, tmp_path: Path):
        task = _scaffold(tmp_path)
        (task / "pipeline").mkdir()
        (task / "pipeline" / "run.R").write_text("x <- 1\n")
        (task / "pipeline" / "kernel.cpp").write_text(
            "#pragma omp parallel for\nfor (int i=0;i<n;i++){}\n"
        )
        res = scan_task(task)
        # The .cpp file is collected and its openmp pragma classified portable.
        assert any(p["pattern"] == "openmp" for p in res.portable_signals)
        assert any(s.endswith("kernel.cpp") for s in res.sources)


# --------------------------------------------------------------------------
# render_table
# --------------------------------------------------------------------------

class TestRenderTable:
    def test_empty(self):
        assert "no tasks scanned" in render_table([])

    def test_renders_rows_and_totals(self, tmp_path: Path):
        clean = _scaffold(tmp_path, "test_clean")
        _write_pipeline(clean, "x <- 1\n")
        crash = _scaffold(tmp_path, "test_crash")
        _write_pipeline(crash, "parallel::mclapply(1:10, identity, mc.cores=4)\n")
        rows = [scan_task(clean), scan_task(crash)]
        out = render_table(rows)
        assert "test_clean" in out
        assert "test_crash" in out
        assert "Totals:" in out
        assert "run_3_5=1" in out

    def test_long_name_truncated(self, tmp_path: Path):
        long_name = "test_" + "z" * 50
        t = _scaffold(tmp_path, long_name)
        _write_pipeline(t, "x <- 1\n")
        out = render_table([scan_task(t)])
        assert "..." in out


# --------------------------------------------------------------------------
# save/load roundtrip with hits
# --------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path):
        task = _scaffold(tmp_path)
        _write_pipeline(task, "parallel::mclapply(1:2, identity, mc.cores=2)\n")
        res = scan_task(task)
        save_portability_scan(task, res)
        loaded = load_portability_scan(task)
        assert loaded["verdict"] == res.verdict
        assert loaded["hits"]

    def test_scanresult_to_json_dict_shape(self):
        sr = ScanResult(verdict="CLEAN", summary="ok")
        d = sr.to_json_dict()
        assert d["verdict"] == "CLEAN"
        assert d["schema_version"] == 1
        assert isinstance(d["hits"], list)
