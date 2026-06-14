"""Unit tests for zyme.commands.profile.native — sample(1) output parsing,
label shortening, upstream source location, percentage math, and the
capability check.

Existing test_profile_native.py covers _is_idle_frame, the idle filtering in
parse_and_normalize, and process-group discovery. This file targets the
gaps: parse_sample_file (section extraction + malformed lines), _short_label
mangling, _locate_func_in_upstream (real grep over a tmp upstream tree),
is_supported, _pgrep_pids, and active/total percentage attribution.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.commands.profile import native


class _FakeSampler:
    def __init__(self, files):
        self._files = files

    def collect(self):
        return self._files


# ---------------------------------------------------------------------------
# is_supported
# ---------------------------------------------------------------------------

def test_is_supported_non_darwin(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "linux")
    ok, reason = native.is_supported()
    assert not ok
    assert "macOS-only" in reason


def test_is_supported_missing_sample_bin(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "darwin")
    monkeypatch.setattr(native.os.path, "exists",
                        lambda p: p != native.SAMPLE_BIN)
    ok, reason = native.is_supported()
    assert not ok
    assert "native sampler missing" in reason


def test_is_supported_missing_pgrep(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "darwin")
    monkeypatch.setattr(native.os.path, "exists",
                        lambda p: p != native.PGREP_BIN)
    ok, reason = native.is_supported()
    assert not ok
    assert "child-process tracking" in reason


def test_is_supported_all_present(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "darwin")
    monkeypatch.setattr(native.os.path, "exists", lambda p: True)
    ok, reason = native.is_supported()
    assert ok
    assert reason == ""


# ---------------------------------------------------------------------------
# _short_label
# ---------------------------------------------------------------------------

def test_short_label_basenames_lib_and_trims_args():
    label = native._short_label(
        "/usr/lib/libopenblas.dylib",
        "dgemm_kernel(double const*, double*)",
    )
    assert label == "libopenblas.dylib:dgemm_kernel(...)"


def test_short_label_no_args_kept_whole():
    assert native._short_label("libfoo.dylib", "plain_fn") == "libfoo.dylib:plain_fn"


def test_short_label_lib_without_slash():
    assert native._short_label("python3.12", "x_add") == "python3.12:x_add"


# ---------------------------------------------------------------------------
# parse_sample_file
# ---------------------------------------------------------------------------

def test_parse_sample_file_missing(tmp_path):
    assert native.parse_sample_file(tmp_path / "absent.txt") == []


def test_parse_sample_file_no_section(tmp_path):
    p = tmp_path / "native_sample_1.txt"
    p.write_text("just a header\nCall graph:\n   stuff\n")
    assert native.parse_sample_file(p) == []


def test_parse_sample_file_extracts_top_of_stack(tmp_path):
    p = tmp_path / "native_sample_1.txt"
    p.write_text(
        "Header\n"
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        compute_kernel  (in libtarget.dylib)        500\n"
        "        helper  (in libfoo.dylib)        120\n"
        "Binary Images:\n"
        "        ignored_after  (in libx.dylib)        99\n"
    )
    rows = native.parse_sample_file(p)
    assert rows == [
        ("compute_kernel", "libtarget.dylib", 500),
        ("helper", "libfoo.dylib", 120),
    ]


def test_parse_sample_file_no_binary_images_marker(tmp_path):
    # when "Binary Images:" is absent, parse to EOF.
    p = tmp_path / "native_sample_1.txt"
    p.write_text(
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        k  (in lib.dylib)        7\n"
    )
    rows = native.parse_sample_file(p)
    assert rows == [("k", "lib.dylib", 7)]


def test_parse_sample_file_ignores_malformed_lines(tmp_path):
    p = tmp_path / "native_sample_1.txt"
    p.write_text(
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        garbage line no count\n"
        "        good  (in lib.dylib)        3\n"
        "no leading whitespace  (in lib.dylib)        4\n"  # regex requires ^\s+
        "Binary Images:\n"
    )
    rows = native.parse_sample_file(p)
    assert rows == [("good", "lib.dylib", 3)]


# ---------------------------------------------------------------------------
# parse_and_normalize — percentage math + idle/active split + empties.
# ---------------------------------------------------------------------------

def test_parse_and_normalize_no_files_emits_note(tmp_path):
    data = native.parse_and_normalize(
        out_dir=tmp_path, sampler=_FakeSampler([]),
        lang="py", tier="tiny", hypothesis="",
    )
    assert data["hotspots"] == []
    assert any("no sample output captured" in n for n in data["notes"])
    assert "0 files" in data["artifacts"]["raw"]


def test_parse_and_normalize_active_pct_excludes_idle(tmp_path):
    sample = tmp_path / "native_sample_1.txt"
    sample.write_text(
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        _pthread_cond_wait  (in libsystem_pthread.dylib)        900\n"
        "        work_a  (in libtarget.dylib)        75\n"
        "        work_b  (in libtarget.dylib)        25\n"
        "Binary Images:\n"
    )
    data = native.parse_and_normalize(
        out_dir=tmp_path, sampler=_FakeSampler([sample]),
        lang="py", tier="tiny", hypothesis="",
    )
    labels = [h["label"] for h in data["hotspots"]]
    assert labels == ["libtarget.dylib:work_a", "libtarget.dylib:work_b"]
    # active total = 100; work_a self_pct = 75% of active, not of 1000 total.
    work_a = data["hotspots"][0]
    assert work_a["self_pct"] == pytest.approx(75.0)
    assert work_a["raw"]["samples_pct_active"] == pytest.approx(75.0)
    assert work_a["raw"]["samples_pct_total"] == pytest.approx(7.5)
    # est self_time = samples * interval (1ms).
    assert work_a["self_time_s"] == pytest.approx(75 * native.SAMPLE_INTERVAL_MS / 1000.0)
    # idle note present.
    assert any("idle/wait frames filtered out" in n for n in data["notes"])
    assert any("top idle frames" in n for n in data["notes"])


def test_parse_and_normalize_aggregates_across_files(tmp_path):
    s1 = tmp_path / "native_sample_1.txt"
    s2 = tmp_path / "native_sample_2.txt"
    body = (
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        shared  (in libt.dylib)        {n}\n"
        "Binary Images:\n"
    )
    s1.write_text(body.format(n=10))
    s2.write_text(body.format(n=30))
    data = native.parse_and_normalize(
        out_dir=tmp_path, sampler=_FakeSampler([s1, s2]),
        lang="py", tier="tiny", hypothesis="",
    )
    assert data["hotspots"][0]["raw"]["samples"] == 40
    assert any("2 processes" in n for n in data["notes"])
    assert any("parent + 1 forked child" in n for n in data["notes"])


def test_parse_and_normalize_schema_fields(tmp_path):
    sample = tmp_path / "native_sample_1.txt"
    sample.write_text(
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        k  (in lib.dylib)        5\n"
        "Binary Images:\n"
    )
    data = native.parse_and_normalize(
        out_dir=tmp_path, sampler=_FakeSampler([sample]),
        lang="R", tier="medium", hypothesis="my hyp",
        totals={"wall_s": 2.0}, artifact_prefix="profile_history/run9",
    )
    assert data["backend"] == "native"
    assert data["lang"] == "R"
    assert data["tier"] == "medium"
    assert data["hypothesis"] == "my hyp"
    assert data["totals"] == {"wall_s": 2.0}
    assert data["override_summary"] == []
    assert data["call_chains"] == []
    assert data["artifacts"]["raw"].startswith("profile_history/run9/native_sample")


# ---------------------------------------------------------------------------
# _pgrep_pids
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, returncode, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_pgrep_pids_parses_lines(monkeypatch):
    monkeypatch.setattr(native.subprocess, "run",
                        lambda c, **k: _Result(0, "101\n202\n303\n"))
    assert native._pgrep_pids(["-P", "1"]) == {101, 202, 303}


def test_pgrep_pids_nonzero_rc_returns_empty(monkeypatch):
    monkeypatch.setattr(native.subprocess, "run",
                        lambda c, **k: _Result(1, ""))
    assert native._pgrep_pids(["-P", "1"]) == set()


def test_pgrep_pids_subprocess_error(monkeypatch):
    def boom(cmd, **kw):
        raise OSError("nope")
    monkeypatch.setattr(native.subprocess, "run", boom)
    assert native._pgrep_pids(["-P", "1"]) == set()


def test_pgrep_pids_skips_garbage_tokens(monkeypatch):
    monkeypatch.setattr(native.subprocess, "run",
                        lambda c, **k: _Result(0, "101\nnotapid\n202\n"))
    assert native._pgrep_pids(["-P", "1"]) == {101, 202}


# ---------------------------------------------------------------------------
# NativeSampler.collect + stale wipe at construction.
# ---------------------------------------------------------------------------

def test_sampler_wipes_stale_outputs_on_init(tmp_path):
    (tmp_path / "native_sample_99.txt").write_text("stale")
    (tmp_path / "native_sample_99.stderr").write_text("stale err")
    native.NativeSampler(tmp_path)
    assert not (tmp_path / "native_sample_99.txt").exists()
    assert not (tmp_path / "native_sample_99.stderr").exists()


def test_sampler_collect_sorted(tmp_path):
    # NativeSampler() wipes stale native_sample_* at construction, so write
    # the run's files AFTER constructing the sampler.
    s = native.NativeSampler(tmp_path)
    (tmp_path / "native_sample_2.txt").write_text("x")
    (tmp_path / "native_sample_1.txt").write_text("x")
    names = [p.name for p in s.collect()]
    assert names == ["native_sample_1.txt", "native_sample_2.txt"]


def test_sampler_discover_no_parent_pid(tmp_path):
    s = native.NativeSampler(tmp_path)
    # parent_pid defaults to None.
    assert s._discover_candidate_pids() == []


# ---------------------------------------------------------------------------
# _locate_func_in_upstream — real grep over a tmp upstream tree.
# ---------------------------------------------------------------------------

def test_locate_func_finds_cpp_definition(tmp_path):
    upstream = tmp_path / "upstream_repo"
    src = upstream / "src"
    src.mkdir(parents=True)
    (src / "kernel.cpp").write_text(
        "// some header\n"
        "NumericVector fast_iterate(NumericVector x) {\n"
        "  return x;\n"
        "}\n"
    )
    loc = native._locate_func_in_upstream("fast_iterate", upstream)
    assert loc is not None
    assert loc.startswith("src/kernel.cpp:")
    # the definition is on line 2.
    assert loc.endswith(":2")


def test_locate_func_strips_cpp_namespace_and_args(tmp_path):
    upstream = tmp_path / "upstream_repo"
    upstream.mkdir()
    (upstream / "a.cpp").write_text(
        "double compute(double const& x) {\n  return x;\n}\n"
    )
    # mangled name with namespace + args still resolves the bare symbol.
    loc = native._locate_func_in_upstream(
        "ns::compute(double const&)", upstream)
    assert loc is not None
    assert "a.cpp" in loc


def test_locate_func_no_upstream_dir(tmp_path):
    assert native._locate_func_in_upstream("foo", tmp_path / "missing") is None


def test_locate_func_too_short_symbol(tmp_path):
    upstream = tmp_path / "upstream_repo"
    upstream.mkdir()
    # bare symbol < 3 chars -> skip.
    assert native._locate_func_in_upstream("ab", upstream) is None


def test_locate_func_not_found_returns_none(tmp_path):
    upstream = tmp_path / "upstream_repo"
    upstream.mkdir()
    (upstream / "a.cpp").write_text("int unrelated() { return 0; }\n")
    assert native._locate_func_in_upstream("nonexistent_symbol", upstream) is None


def test_locate_func_skips_forward_declaration(tmp_path):
    """A line ending in ';' is a declaration, not a definition; if it's the
    only hit, the helper still returns it (single other_hit fallback) but a
    real definition is preferred."""
    upstream = tmp_path / "upstream_repo"
    upstream.mkdir()
    (upstream / "decl.h").write_text("void mykernel(int x);\n")
    (upstream / "impl.cpp").write_text("void mykernel(int x) {\n  return;\n}\n")
    loc = native._locate_func_in_upstream("mykernel", upstream)
    # definition (impl.cpp) preferred over declaration (decl.h).
    assert loc is not None
    assert "impl.cpp" in loc
