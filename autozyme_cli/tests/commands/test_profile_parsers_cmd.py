"""Unit tests for zyme.commands.profile.parsers — pure parse/normalize logic.

These avoid spawning real profiler subprocesses:
  - cProfile path: build a *real* profile.out via cProfile on a deterministic
    workload (no subprocess; pstats reads the file directly).
  - scalene path: hand-built scalene.json fixtures (the JSON schema parsers
    expects), driven straight through _parse_scalene.
  - Rprof / memray paths: monkeypatch the subprocess.run boundary so the
    JSON-shaped stdout / stats.json is supplied directly — exercising the
    Python-side normalization without R / memray binaries.

Existing tests (test_profile_call_chains, test_profile_schema) cover the
Rprof call-chain extraction and the end-to-end subprocess schema; this file
targets the pure helpers + the cProfile / scalene / memray normalizers and
their edge cases (missing artifact, empty profile, malformed lines, deep
chains, low-actionability demotion).
"""
from __future__ import annotations

import cProfile
import json
import pstats
from pathlib import Path

import pytest

from zyme.commands.profile import parsers


# ---------------------------------------------------------------------------
# helpers: _short_label / _maybe_float / _split_memray_location / _strip_*
# ---------------------------------------------------------------------------

def test_short_label_trims_to_basename():
    assert parsers._short_label("/abs/path/to/foo.py", 42, "bar") == "foo.py:42:bar"


def test_short_label_handles_falsy_file():
    assert parsers._short_label("", 0, "fn") == "?:0:fn"
    assert parsers._short_label(None, 1, "fn") == "?:1:fn"


def test_maybe_float_coerces_and_tolerates():
    assert parsers._maybe_float("3.5") == 3.5
    assert parsers._maybe_float(2) == 2.0
    assert parsers._maybe_float(None) is None
    assert parsers._maybe_float("notanumber") is None
    assert parsers._maybe_float([1, 2]) is None


def test_split_memray_location_three_parts():
    func, file, line = parsers._split_memray_location("my_func:/abs/path.py:42")
    assert (func, file, line) == ("my_func", "/abs/path.py", 42)


def test_split_memray_location_windows_drive_colon():
    # B19 fix: func splits off the LEFT, line off the RIGHT, so a Windows drive
    # letter's colon stays in the FILE part (as the docstring promises).
    func, file, line = parsers._split_memray_location("fn:C:\\proj\\a.py:7")
    assert func == "fn"
    assert file == "C:\\proj\\a.py"
    assert line == 7

    # The simple POSIX case still works cleanly.
    func2, file2, line2 = parsers._split_memray_location("fn:/proj/a.py:7")
    assert (func2, file2, line2) == ("fn", "/proj/a.py", 7)


def test_split_memray_location_non_numeric_line_defaults_zero():
    func, file, line = parsers._split_memray_location("fn:/p.py:notaline")
    assert line == 0


def test_split_memray_location_unsplittable():
    func, file, line = parsers._split_memray_location("just_a_blob")
    assert (func, file, line) == ("just_a_blob", "?", 0)


def test_strip_rprof_lineinfo_removes_suffix():
    assert parsers._strip_rprof_lineinfo("La.svd#/path/svd.R#42") == "La.svd"
    assert parsers._strip_rprof_lineinfo("plainfn") == "plainfn"
    # Leading '#' (idx==0) is not stripped.
    assert parsers._strip_rprof_lineinfo("#leading") == "#leading"


def test_clean_rprof_label_unquotes_and_strips():
    assert parsers._clean_rprof_label('"fitPixels"') == "fitPixels"
    assert parsers._clean_rprof_label('""nested""') == "nested"
    assert parsers._clean_rprof_label(None) == ""
    assert parsers._clean_rprof_label('"La.svd#file.R#3"') == "La.svd"


def test_scalene_pct_coercion():
    assert parsers._scalene_pct(None) == 0.0
    assert parsers._scalene_pct(12) == 12.0
    assert parsers._scalene_pct("5.5") == 5.5
    assert parsers._scalene_pct("bad") == 0.0
    assert parsers._scalene_pct([1]) == 0.0


# ---------------------------------------------------------------------------
# cProfile parsing — build a real profile.out, parse it directly.
# ---------------------------------------------------------------------------

def _outer(n):
    return _inner(n)


def _inner(n):
    s = 0
    for i in range(n):
        s += i * i
    return s


def _make_cprofile(tmp_path: Path, n: int = 80_000) -> Path:
    prof = cProfile.Profile()
    prof.enable()
    _outer(n)
    prof.disable()
    out = tmp_path / "profile.out"
    pstats.Stats(prof).dump_stats(str(out))
    return out


def test_parse_cprofile_missing_returns_note(tmp_path):
    hotspots, notes, chains = parsers._parse_cprofile(tmp_path / "nope.out")
    assert hotspots == []
    assert chains == []
    assert any("missing" in n for n in notes)


def test_parse_cprofile_malformed_file_returns_note(tmp_path):
    bad = tmp_path / "profile.out"
    bad.write_text("this is not a pstats dump")
    hotspots, notes, chains = parsers._parse_cprofile(bad)
    assert hotspots == []
    assert any("failed to load" in n for n in notes)


def test_parse_cprofile_real_workload_ranks_hotspots(tmp_path):
    out = _make_cprofile(tmp_path)
    hotspots, notes, chains = parsers._parse_cprofile(out)
    assert hotspots, "expected at least one hotspot from a real workload"
    # ranks are 1-based, contiguous, ascending.
    assert [h["rank"] for h in hotspots] == list(range(1, len(hotspots) + 1))
    for h in hotspots:
        assert set(h.keys()) >= {
            "rank", "label", "self_time_s", "total_time_s",
            "self_pct", "calls", "raw",
        }
        assert isinstance(h["raw"], dict)
        assert {"file", "line", "func", "ncalls"} <= set(h["raw"].keys())
    # our _inner should be the hottest (or near it) and appear by name.
    labels = " ".join(h["label"] for h in hotspots)
    assert "_inner" in labels
    # notes mention deterministic + native-opacity caveats.
    assert any("deterministic" in n for n in notes)
    assert any("native code" in n for n in notes)


def test_parse_cprofile_self_pct_sums_reasonably(tmp_path):
    out = _make_cprofile(tmp_path)
    hotspots, _notes, _chains = parsers._parse_cprofile(out)
    pcts = [h["self_pct"] for h in hotspots if h["self_pct"] is not None]
    assert pcts, "self_pct should be computed when total_self > 0"
    # each percentage is in [0, 100].
    assert all(0.0 <= p <= 100.0 for p in pcts)


def test_cprofile_call_chains_root_to_leaf(tmp_path):
    out = _make_cprofile(tmp_path)
    stats = pstats.Stats(str(out))
    stats.sort_stats("tottime")
    keys = stats.fcn_list[: parsers.TOP_N]
    hotspots, _notes, _chains = parsers._parse_cprofile(out)
    chains = parsers._cprofile_call_chains(stats, keys, hotspots)
    assert chains, "expected call chains"
    for c in chains:
        assert c["evidence"] == "cProfile callers"
        assert isinstance(c["chain"], list)
        # leaf is the last frame of chain when chain is non-empty.
        if c["chain"]:
            assert c["chain"][-1] == c["leaf"]
        assert c["hotspot_rank"] is not None


def test_best_cprofile_chain_terminates_on_cycle(tmp_path):
    """The seen-set guard must prevent infinite loops on recursive callers."""
    out = _make_cprofile(tmp_path)
    stats = pstats.Stats(str(out))
    stats.sort_stats("tottime")
    leaf = stats.fcn_list[0]
    chain = parsers._best_cprofile_chain(stats, leaf, max_depth=8)
    assert len(chain) <= 8
    # no duplicate frames (seen-set guarantees uniqueness).
    assert len(chain) == len(set(chain))
    # chain ends at the leaf (root->leaf order).
    assert chain[-1] == leaf


# ---------------------------------------------------------------------------
# scalene parsing — hand-built JSON fixtures (no subprocess).
# ---------------------------------------------------------------------------

def _scalene_fixture(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "scalene.json"
    p.write_text(json.dumps(data))
    return p


def test_parse_scalene_missing(tmp_path):
    hotspots, notes = parsers._parse_scalene(tmp_path / "nope.json")
    assert hotspots == []
    assert any("missing" in n for n in notes)


def test_parse_scalene_malformed_json(tmp_path):
    p = tmp_path / "scalene.json"
    p.write_text("{ not valid json")
    hotspots, notes = parsers._parse_scalene(p)
    assert hotspots == []
    assert any("failed to read" in n for n in notes)


def test_parse_scalene_functions_ranked_by_score(tmp_path):
    data = {
        "elapsed_time_sec": 10.0,
        "max_footprint_mb": 256.0,
        "files": {
            "/proj/run.py": {
                "functions": [
                    {"line": "hot_fn", "n_cpu_percent_python": 50.0,
                     "n_cpu_percent_c": 10.0, "n_sys_percent": 0.0,
                     "n_peak_mb": 20.0, "n_growth_mb": 5.0, "n_avg_mb": 10.0},
                    {"line": "warm_fn", "n_cpu_percent_python": 20.0,
                     "n_cpu_percent_c": 0.0, "n_sys_percent": 0.0,
                     "n_peak_mb": 0.0, "n_growth_mb": 0.0, "n_avg_mb": 0.0},
                    # zero-score function is dropped entirely.
                    {"line": "cold_fn", "n_cpu_percent_python": 0.0,
                     "n_cpu_percent_c": 0.0, "n_sys_percent": 0.0,
                     "n_peak_mb": 0.0, "n_growth_mb": 0.0, "n_avg_mb": 0.0},
                ],
                "lines": [],
            }
        },
    }
    p = _scalene_fixture(tmp_path, data)
    hotspots, notes = parsers._parse_scalene(p)
    assert [h["label"] for h in hotspots] == [
        "run.py:hot_fn()", "run.py:warm_fn()",
    ]
    hot = hotspots[0]
    assert hot["raw"]["kind"] == "func"
    assert hot["raw"]["cpu_native_pct"] == 10.0
    # self_time_s = cpu_total_pct/100 * elapsed = 0.60 * 10 = 6.0
    assert hot["self_time_s"] == pytest.approx(6.0)
    assert any("sampling+alloc" in n for n in notes)
    assert any("peak process memory" in n for n in notes)


def test_parse_scalene_promotes_lines_when_no_functions(tmp_path):
    """When functions[] is empty (workload in C ext / module-level), lines
    are promoted to hotspots — the astropy_lombscargle failure mode."""
    data = {
        "elapsed_time_sec": 4.0,
        "files": {
            "/proj/run.py": {
                "functions": [],
                "lines": [
                    {"lineno": 12, "n_cpu_percent_python": 0.0,
                     "n_cpu_percent_c": 80.0, "n_sys_percent": 0.0,
                     "n_peak_mb": 0.0, "n_growth_mb": 0.0, "n_avg_mb": 0.0},
                    {"lineno": 99, "n_cpu_percent_python": 5.0,
                     "n_cpu_percent_c": 0.0, "n_sys_percent": 0.0,
                     "n_peak_mb": 0.0, "n_growth_mb": 0.0, "n_avg_mb": 0.0},
                ],
            }
        },
    }
    p = _scalene_fixture(tmp_path, data)
    hotspots, notes = parsers._parse_scalene(p)
    assert hotspots[0]["label"] == "run.py:12"
    assert hotspots[0]["raw"]["kind"] == "line"
    assert any("no functions[] entries" in n for n in notes)


def test_parse_scalene_notable_lines_surfaced_in_notes(tmp_path):
    """When functions exist AND a >=5% line exists, it shows up as a note."""
    data = {
        "elapsed_time_sec": 2.0,
        "files": {
            "/proj/run.py": {
                "functions": [
                    {"line": "f", "n_cpu_percent_python": 30.0,
                     "n_cpu_percent_c": 0.0, "n_sys_percent": 0.0,
                     "n_peak_mb": 0.0, "n_growth_mb": 0.0, "n_avg_mb": 0.0},
                ],
                "lines": [
                    {"lineno": 7, "n_cpu_percent_python": 12.0,
                     "n_cpu_percent_c": 0.0, "n_sys_percent": 0.0,
                     "n_peak_mb": 0.0, "n_growth_mb": 0.0, "n_avg_mb": 0.0},
                    # below 5% threshold — not surfaced.
                    {"lineno": 8, "n_cpu_percent_python": 1.0,
                     "n_cpu_percent_c": 0.0, "n_sys_percent": 0.0,
                     "n_peak_mb": 0.0, "n_growth_mb": 0.0, "n_avg_mb": 0.0},
                ],
            }
        },
    }
    p = _scalene_fixture(tmp_path, data)
    hotspots, notes = parsers._parse_scalene(p)
    notable = [n for n in notes if "notable non-function lines" in n]
    assert notable
    assert "run.py:7" in notable[0]
    assert "run.py:8" not in notable[0]


def test_parse_scalene_empty_elapsed_self_time_none(tmp_path):
    data = {
        "files": {
            "/p/run.py": {
                "functions": [
                    {"name": "g", "n_cpu_percent_python": 40.0,
                     "n_cpu_percent_c": 0.0, "n_sys_percent": 0.0,
                     "n_peak_mb": 0.0, "n_growth_mb": 0.0, "n_avg_mb": 0.0},
                ],
                "lines": [],
            }
        },
    }
    p = _scalene_fixture(tmp_path, data)
    hotspots, notes = parsers._parse_scalene(p)
    assert hotspots[0]["self_time_s"] is None
    # name fallback used when "line" key absent.
    assert hotspots[0]["label"] == "run.py:g()"


# ---------------------------------------------------------------------------
# Rprof normalization — monkeypatch the Rscript subprocess.run boundary.
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_parse_rprof_missing_file(tmp_path):
    hotspots, notes, chains = parsers._parse_rprof(tmp_path / "nope.out", None)
    assert hotspots == [] and chains == []
    assert any("missing" in n for n in notes)


def test_parse_rprof_normalizes_subprocess_json(tmp_path, monkeypatch):
    prof = tmp_path / "Rprof.out"
    prof.write_text(
        "sample.interval=20000\n"
        '"hot_r" "caller" "run"\n'
        '"hot_r" "caller" "run"\n'
    )
    payload = {
        "hotspots": [
            {"rank": 1, "label": "hot_r", "self_time_s": 0.2,
             "total_time_s": 0.2, "self_pct": 80.0, "calls": None,
             "raw": {"total_pct": 80.0, "mem_total_mb": 12.5}},
        ],
        "sampling_interval_s": 0.02,
        "total_sampled_s": 0.25,
        "has_memory": False,
    }

    def fake_run(cmd, **kw):
        return _Result(0, json.dumps(payload))

    monkeypatch.setattr(parsers.subprocess, "run", fake_run)
    hotspots, notes, chains = parsers._parse_rprof(prof, {"rscript": "Rscript"})
    assert len(hotspots) == 1
    h = hotspots[0]
    assert h["label"] == "hot_r"
    assert h["self_time_s"] == 0.2
    assert h["calls"] is None
    assert h["raw"]["mem_total_mb"] == 12.5
    assert any("kind=sampling" in n for n in notes)
    # call chains reconstructed from the raw stack file.
    assert chains and chains[0]["leaf"] == "hot_r"


def test_parse_rprof_subprocess_nonzero_rc(tmp_path, monkeypatch):
    prof = tmp_path / "Rprof.out"
    prof.write_text("sample.interval=20000\n")
    monkeypatch.setattr(parsers.subprocess, "run",
                        lambda c, **k: _Result(1, "", "boom"))
    hotspots, notes, chains = parsers._parse_rprof(prof, None)
    assert hotspots == []
    assert any("Rscript returned 1" in n for n in notes)


def test_parse_rprof_subprocess_not_json(tmp_path, monkeypatch):
    prof = tmp_path / "Rprof.out"
    prof.write_text("sample.interval=20000\n")
    monkeypatch.setattr(parsers.subprocess, "run",
                        lambda c, **k: _Result(0, "not json at all"))
    hotspots, notes, chains = parsers._parse_rprof(prof, None)
    assert hotspots == []
    assert any("not JSON" in n for n in notes)


def test_parse_rprof_subprocess_empty_output(tmp_path, monkeypatch):
    prof = tmp_path / "Rprof.out"
    prof.write_text("sample.interval=20000\n")
    monkeypatch.setattr(parsers.subprocess, "run",
                        lambda c, **k: _Result(0, "   "))
    hotspots, notes, chains = parsers._parse_rprof(prof, None)
    assert hotspots == []
    assert any("no output" in n for n in notes)


def test_parse_rprof_r_parser_error_field(tmp_path, monkeypatch):
    prof = tmp_path / "Rprof.out"
    prof.write_text("sample.interval=20000\n")
    monkeypatch.setattr(
        parsers.subprocess, "run",
        lambda c, **k: _Result(0, json.dumps({"error": "jsonlite not installed"})))
    hotspots, notes, chains = parsers._parse_rprof(prof, None)
    assert hotspots == []
    assert any("R parser error" in n for n in notes)


def test_parse_rprof_invocation_failure(tmp_path, monkeypatch):
    import subprocess as sp
    prof = tmp_path / "Rprof.out"
    prof.write_text("sample.interval=20000\n")

    def boom(cmd, **kw):
        raise FileNotFoundError("Rscript")

    monkeypatch.setattr(parsers.subprocess, "run", boom)
    hotspots, notes, chains = parsers._parse_rprof(prof, None)
    assert hotspots == []
    assert any("Rscript invocation failed" in n for n in notes)


def test_parse_rprof_mem_focus_note(tmp_path, monkeypatch):
    prof = tmp_path / "Rprof.out"
    prof.write_text("sample.interval=20000\n")
    payload = {"hotspots": [], "sampling_interval_s": 0.02, "has_memory": True}
    monkeypatch.setattr(parsers.subprocess, "run",
                        lambda c, **k: _Result(0, json.dumps(payload)))
    _h, notes, _c = parsers._parse_rprof(prof, None, mem_focus=True)
    assert any("memory.profiling=TRUE" in n for n in notes)

    payload2 = {"hotspots": [], "sampling_interval_s": 0.02, "has_memory": False}
    monkeypatch.setattr(parsers.subprocess, "run",
                        lambda c, **k: _Result(0, json.dumps(payload2)))
    _h2, notes2, _c2 = parsers._parse_rprof(prof, None, mem_focus=True)
    assert any("memory column unavailable" in n for n in notes2)


# ---------------------------------------------------------------------------
# Rprof stacks + call chains (pure file parsing — no subprocess).
# ---------------------------------------------------------------------------

def test_read_rprof_stacks_strips_lineinfo(tmp_path):
    prof = tmp_path / "Rprof.out"
    prof.write_text(
        "sample.interval=5000\n"
        '"La.svd#/p/svd.R#3" "caller" "run"\n'
    )
    stacks = parsers._read_rprof_stacks(prof)
    assert stacks == [["La.svd", "caller", "run"]]


def test_read_rprof_stacks_missing_file_returns_empty(tmp_path):
    assert parsers._read_rprof_stacks(tmp_path / "absent.out") == []


def test_rprof_call_chains_no_match_skips(tmp_path):
    prof = tmp_path / "Rprof.out"
    prof.write_text('sample.interval=5000\n"a" "b"\n')
    chains = parsers._rprof_call_chains(prof, [{"rank": 1, "label": "zzz"}])
    assert chains == []


def test_rprof_call_chains_empty_stacks(tmp_path):
    prof = tmp_path / "Rprof.out"
    prof.write_text("sample.interval=5000\n")
    chains = parsers._rprof_call_chains(prof, [{"rank": 1, "label": "a"}])
    assert chains == []


# ---------------------------------------------------------------------------
# memray normalization — monkeypatch backends._resolve_python_bin + subprocess.
# ---------------------------------------------------------------------------

def test_parse_memray_missing_bin(tmp_path):
    hotspots, notes = parsers._parse_memray(tmp_path / "nope.bin", None)
    assert hotspots == []
    assert any("missing" in n for n in notes)


def test_parse_memray_no_python_interpreter(tmp_path, monkeypatch):
    bin_path = tmp_path / "memray.bin"
    bin_path.write_bytes(b"\x00")
    from zyme.commands.profile import backends as backends_mod
    monkeypatch.setattr(backends_mod, "_resolve_python_bin", lambda ex: None)
    hotspots, notes = parsers._parse_memray(bin_path, None)
    assert hotspots == []
    assert any("could not resolve task python" in n for n in notes)


def test_parse_memray_normalizes_stats_json(tmp_path, monkeypatch):
    bin_path = tmp_path / "memray.bin"
    bin_path.write_bytes(b"\x00")
    from zyme.commands.profile import backends as backends_mod
    monkeypatch.setattr(backends_mod, "_resolve_python_bin",
                        lambda ex: "/fake/python")

    stats = {
        "total_num_allocations": 1000,
        "total_bytes_allocated": 2 * 1024**3,
        "metadata": {"peak_memory": 3 * 1024**3},
        "top_allocations_by_size": [
            {"location": "big_alloc:/proj/run.py:10", "size": 1024**3},
            {"location": "small:/proj/run.py:20", "size": 1024},
        ],
        "top_allocations_by_count": [
            {"location": "big_alloc:/proj/run.py:10", "count": 42},
        ],
    }

    def fake_run(cmd, **kw):
        # memray writes JSON to the -o path; find it in argv.
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_text(json.dumps(stats))
        return _Result(0, "")

    monkeypatch.setattr(parsers.subprocess, "run", fake_run)
    hotspots, notes = parsers._parse_memray(bin_path, None)
    assert len(hotspots) == 2
    top = hotspots[0]
    assert top["label"] == "run.py:10:big_alloc"
    assert top["raw"]["alloc_bytes"] == 1024**3
    assert top["raw"]["alloc_count"] == 42
    assert top["calls"] == 42
    assert any("allocation_trace" in n for n in notes)
    assert any("totals:" in n for n in notes)
    assert any("peak resident" in n for n in notes)
    # the sidecar stats.json should have been cleaned up.
    assert not (bin_path.with_suffix(".stats.json")).exists()


def test_parse_memray_subprocess_nonzero_rc(tmp_path, monkeypatch):
    bin_path = tmp_path / "memray.bin"
    bin_path.write_bytes(b"\x00")
    from zyme.commands.profile import backends as backends_mod
    monkeypatch.setattr(backends_mod, "_resolve_python_bin", lambda ex: "/p")
    monkeypatch.setattr(parsers.subprocess, "run",
                        lambda c, **k: _Result(1, "", "memray exploded"))
    hotspots, notes = parsers._parse_memray(bin_path, None)
    assert hotspots == []
    assert any("memray stats returned 1" in n for n in notes)


def test_parse_memray_invocation_failure(tmp_path, monkeypatch):
    bin_path = tmp_path / "memray.bin"
    bin_path.write_bytes(b"\x00")
    from zyme.commands.profile import backends as backends_mod
    monkeypatch.setattr(backends_mod, "_resolve_python_bin", lambda ex: "/p")

    def boom(cmd, **kw):
        raise OSError("no memray")

    monkeypatch.setattr(parsers.subprocess, "run", boom)
    hotspots, notes = parsers._parse_memray(bin_path, None)
    assert hotspots == []
    assert any("memray stats invocation failed" in n for n in notes)


def test_parse_memray_produces_no_json(tmp_path, monkeypatch):
    bin_path = tmp_path / "memray.bin"
    bin_path.write_bytes(b"\x00")
    from zyme.commands.profile import backends as backends_mod
    monkeypatch.setattr(backends_mod, "_resolve_python_bin", lambda ex: "/p")
    # subprocess "succeeds" but writes nothing to -o.
    monkeypatch.setattr(parsers.subprocess, "run", lambda c, **k: _Result(0, ""))
    hotspots, notes = parsers._parse_memray(bin_path, None)
    assert hotspots == []
    assert any("produced no JSON" in n for n in notes)


# ---------------------------------------------------------------------------
# normalize() dispatch + actionable hotspots + annotate_call_chains.
# ---------------------------------------------------------------------------

def test_normalize_unknown_backend_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown backend"):
        parsers.normalize("weird", "py", tmp_path, "hyp", "tiny", "prefix")


def test_normalize_cpu_py_end_to_end(tmp_path):
    _make_cprofile(tmp_path)
    data = parsers.normalize(
        "cpu", "py", tmp_path, hypothesis="hyp", tier="tiny",
        artifact_prefix="profile_history/current",
    )
    assert data["schema_version"] == "1"
    assert data["backend"] == "cpu"
    assert data["lang"] == "py"
    assert data["tier"] == "tiny"
    assert data["hypothesis"] == "hyp"
    assert data["artifacts"]["raw"] == "profile_history/current/profile.out"
    assert isinstance(data["hotspots"], list) and data["hotspots"]
    assert data["actionable_hotspots"] == []
    assert "timestamp" in data


def test_normalize_full_py_scalene(tmp_path):
    (tmp_path / "scalene.json").write_text(json.dumps({
        "elapsed_time_sec": 1.0,
        "files": {"/p/run.py": {
            "functions": [{"line": "f", "n_cpu_percent_python": 50.0}],
            "lines": [],
        }},
    }))
    data = parsers.normalize("full", "py", tmp_path, "", "tiny",
                             "profile_history/current")
    assert data["backend"] == "full"
    assert data["artifacts"]["raw"].endswith("scalene.json")
    assert data["hotspots"][0]["label"] == "run.py:f()"


def test_normalize_passes_totals_through(tmp_path):
    _make_cprofile(tmp_path)
    data = parsers.normalize("cpu", "py", tmp_path, "", "tiny", "p",
                             totals={"wall_s": 3.0})
    assert data["totals"] == {"wall_s": 3.0}


def test_normalize_cpu_R_dispatches_to_rprof(tmp_path, monkeypatch):
    (tmp_path / "Rprof.out").write_text("sample.interval=5000\n")
    calls = {}

    def fake_parse_rprof(path, executor, mem_focus=False):
        calls["path"] = path
        calls["mem_focus"] = mem_focus
        return ([{"rank": 1, "label": "fn", "raw": {}}], ["rnote"], [])

    monkeypatch.setattr(parsers, "_parse_rprof", fake_parse_rprof)
    data = parsers.normalize("cpu", "R", tmp_path, "", "tiny",
                             "profile_history/current")
    assert data["backend"] == "cpu"
    assert data["lang"] == "R"
    assert data["artifacts"]["raw"].endswith("Rprof.out")
    assert data["hotspots"][0]["label"] == "fn"
    assert calls["mem_focus"] is False


def test_normalize_full_R_attaches_profvis_viewer(tmp_path, monkeypatch):
    (tmp_path / "Rprof.out").write_text("sample.interval=5000\n")
    (tmp_path / "profvis.html").write_text("<html></html>")
    monkeypatch.setattr(parsers, "_parse_rprof",
                        lambda p, e, mem_focus=False: ([], [], []))
    data = parsers.normalize("full", "R", tmp_path, "", "tiny",
                             "profile_history/current")
    assert data["artifacts"]["raw"].endswith("Rprof.out")
    assert data["artifacts"]["viewer"].endswith("profvis.html")


def test_normalize_mem_R_sets_mem_focus(tmp_path, monkeypatch):
    (tmp_path / "Rprof.out").write_text("sample.interval=5000\n")
    seen = {}

    def fake_parse_rprof(path, executor, mem_focus=False):
        seen["mem_focus"] = mem_focus
        return ([], [], [])

    monkeypatch.setattr(parsers, "_parse_rprof", fake_parse_rprof)
    data = parsers.normalize("mem", "R", tmp_path, "", "tiny", "p")
    assert seen["mem_focus"] is True
    assert data["backend"] == "mem"


def test_normalize_mem_py_dispatches_to_memray(tmp_path, monkeypatch):
    (tmp_path / "memray.bin").write_bytes(b"\x00")
    monkeypatch.setattr(parsers, "_parse_memray",
                        lambda p, e: ([{"rank": 1, "label": "alloc", "raw": {}}],
                                      ["mnote"]))
    data = parsers.normalize("mem", "py", tmp_path, "", "tiny",
                             "profile_history/current")
    assert data["backend"] == "mem"
    assert data["artifacts"]["raw"].endswith("memray.bin")
    assert data["hotspots"][0]["label"] == "alloc"


def test_build_actionable_hotspots_demotes_runtime_frames():
    profile = {
        "hotspots": [
            {"rank": 1, "label": "unserialize", "raw": {}},
            {"rank": 2, "label": "real_compute", "raw": {}},
            {"rank": 3, "label": "pthread_cond_wait", "raw": {}},
        ]
    }
    actionable, notes = parsers.build_actionable_hotspots(profile)
    assert [h["label"] for h in actionable] == ["real_compute"]
    assert actionable[0]["rank"] == 1
    assert actionable[0]["raw"]["source"] == "profiler_hotspot"
    assert any("demoted" in n for n in notes)


def test_build_actionable_hotspots_all_noise_keeps_raw():
    """If every frame is low-actionability, keep the raw list (don't return
    empty — the blocked-on-runtime evidence is still useful)."""
    profile = {
        "hotspots": [
            {"rank": 1, "label": "mcfork", "raw": {}},
            {"rank": 2, "label": "sem_wait", "raw": {}},
        ]
    }
    actionable, notes = parsers.build_actionable_hotspots(profile)
    assert len(actionable) == 2
    assert actionable[0]["raw"]["source"] == "profiler_hotspot"


def test_build_actionable_hotspots_respects_top_n():
    profile = {"hotspots": [{"rank": i, "label": f"fn{i}", "raw": {}}
                            for i in range(1, 30)]}
    actionable, _notes = parsers.build_actionable_hotspots(profile, top_n=5)
    assert len(actionable) == 5


def test_build_actionable_hotspots_empty():
    actionable, notes = parsers.build_actionable_hotspots({"hotspots": []})
    assert actionable == []
    assert notes == []


def test_is_low_actionability_checks_raw_values():
    # the pattern appears in a raw value, not the label.
    h = {"label": "innocent", "raw": {"detail": "calls sem_wait internally"}}
    assert parsers._is_low_actionability_hotspot(h)
    h2 = {"label": "innocent", "raw": {"n": 5}}
    assert not parsers._is_low_actionability_hotspot(h2)


def test_annotate_call_chains_passthrough_copy():
    profile = {"call_chains": [{"rank": 1, "chain": ["a", "b"]}]}
    out = parsers.annotate_call_chains(profile)
    assert out == [{"rank": 1, "chain": ["a", "b"]}]
    # returns copies, not the same dict objects.
    out[0]["rank"] = 99
    assert profile["call_chains"][0]["rank"] == 1


def test_annotate_call_chains_missing_key():
    assert parsers.annotate_call_chains({}) == []
