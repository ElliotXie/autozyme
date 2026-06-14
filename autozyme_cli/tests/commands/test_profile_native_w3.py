"""Wave-3 coverage for zyme.commands.profile.native — the pure-parsing /
locate branches the existing tests (test_profile_native.py +
test_profile_native_cmd.py) leave uncovered.

Targets:
  - parse_sample_file: OSError on read() -> [].
  - parse_and_normalize: source_location attached when _locate finds a defn
    (line 382), and the n_children==0 single-process "parent + 0 forked
    child" wording.
  - _locate_func_in_upstream reachable branches:
      * grep OSError / TimeoutExpired -> None
      * grep returncode not in (0,1) -> None
      * malformed grep line (no lineno) skipped
      * comment-line hit skipped
      * single non-definition other_hit fallback returns it
      * relative_to ValueError fallback (symlinked / outside path) -> raw path
  - _discover_candidate_pids with a parent_pid set, _pgrep_pids monkeypatched.

The /usr/bin/sample subprocess machinery (NativeSampler.attach/detach/
_spawn_sample/_watch_loop) is NOT exercised: it spawns real `sample`/`pgrep`
binaries against live PIDs and signals them, which the brief marks as the
unreachable subprocess boundary. Those stay uncovered by design.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from zyme.commands.profile import native


class _FakeSampler:
    def __init__(self, files):
        self._files = files

    def collect(self):
        return self._files


class _GrepResult:
    def __init__(self, returncode, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


# --------------------------------------------------------------------------
# parse_sample_file — OSError on read
# --------------------------------------------------------------------------

def test_parse_sample_file_oserror_on_read(tmp_path, monkeypatch):
    p = tmp_path / "native_sample_1.txt"
    p.write_text("Sort by top of stack, same collapsed (when >= 5):\n")

    def boom(*a, **k):
        raise OSError("read failed")
    monkeypatch.setattr(Path, "read_text", boom)
    assert native.parse_sample_file(p) == []


# --------------------------------------------------------------------------
# parse_and_normalize — source_location attachment (line 382) + single proc
# --------------------------------------------------------------------------

def test_parse_and_normalize_attaches_source_location(tmp_path):
    # out_dir layout: <task>/profile_history/<run>/  ; upstream at <task>/upstream_repo
    task = tmp_path / "task"
    run_dir = task / "profile_history" / "run1"
    run_dir.mkdir(parents=True)
    upstream = task / "upstream_repo"
    src = upstream / "src"
    src.mkdir(parents=True)
    (src / "kernel.cpp").write_text(
        "// header\n"
        "NumericVector my_hot_kernel(NumericVector x) {\n"
        "  return x;\n"
        "}\n"
    )
    sample = run_dir / "native_sample_1.txt"
    sample.write_text(
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        my_hot_kernel  (in libtarget.dylib)        500\n"
        "Binary Images:\n"
    )
    data = native.parse_and_normalize(
        out_dir=run_dir, sampler=_FakeSampler([sample]),
        lang="R", tier="tiny", hypothesis="",
    )
    hot = data["hotspots"][0]
    assert "source_location" in hot["raw"]
    assert hot["raw"]["source_location"].startswith("src/kernel.cpp:")

    def _children_note():
        return next(n for n in data["notes"] if "forked child" in n)
    # single process -> "parent + 0 forked children".
    assert "parent + 0 forked children" in _children_note()


# --------------------------------------------------------------------------
# _locate_func_in_upstream — grep error / returncode / line-parse branches
# --------------------------------------------------------------------------

class TestLocateFuncBranches:
    def _upstream(self, tmp_path):
        up = tmp_path / "upstream_repo"
        up.mkdir()
        return up

    def test_grep_oserror_returns_none(self, tmp_path, monkeypatch):
        up = self._upstream(tmp_path)
        (up / "a.cpp").write_text("void foo() {}\n")

        def boom(*a, **k):
            raise OSError("grep missing")
        monkeypatch.setattr(native.subprocess, "run", boom)
        assert native._locate_func_in_upstream("foo_func", up) is None

    def test_grep_timeout_returns_none(self, tmp_path, monkeypatch):
        up = self._upstream(tmp_path)
        (up / "a.cpp").write_text("void foo() {}\n")

        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="grep", timeout=5.0)
        monkeypatch.setattr(native.subprocess, "run", boom)
        assert native._locate_func_in_upstream("foo_func", up) is None

    def test_grep_returncode_2_returns_none(self, tmp_path, monkeypatch):
        up = self._upstream(tmp_path)
        monkeypatch.setattr(native.subprocess, "run",
                            lambda *a, **k: _GrepResult(2, ""))
        assert native._locate_func_in_upstream("foo_func", up) is None

    def test_malformed_grep_line_skipped(self, tmp_path, monkeypatch):
        up = self._upstream(tmp_path)
        # first line has <3 colon parts; second has a non-int lineno; both skip.
        stdout = "noColonsAtAll\n/p/a.cpp:notanint:void foo_func() {\n"
        monkeypatch.setattr(native.subprocess, "run",
                            lambda *a, **k: _GrepResult(0, stdout))
        assert native._locate_func_in_upstream("foo_func", up) is None

    def test_comment_line_hit_skipped(self, tmp_path, monkeypatch):
        up = self._upstream(tmp_path)
        # the only hit is a comment line -> stripped -> skipped -> None.
        stdout = "/p/a.cpp:5:// foo_func( is mentioned in a comment\n"
        monkeypatch.setattr(native.subprocess, "run",
                            lambda *a, **k: _GrepResult(0, stdout))
        assert native._locate_func_in_upstream("foo_func", up) is None

    def test_single_other_hit_fallback(self, tmp_path, monkeypatch):
        up = self._upstream(tmp_path)
        # one non-definition-shaped hit (a call, not `Type name(`); single
        # other_hit -> returned as fallback. Use a path under upstream so the
        # relative_to succeeds.
        callsite = up / "use.cpp"
        callsite.write_text("  result = foo_func(x);\n")
        stdout = f"{callsite}:1:  result = foo_func(x);\n"
        monkeypatch.setattr(native.subprocess, "run",
                            lambda *a, **k: _GrepResult(0, stdout))
        loc = native._locate_func_in_upstream("foo_func", up)
        assert loc == "use.cpp:1"

    def test_multiple_other_hits_no_defn_returns_none(self, tmp_path, monkeypatch):
        up = self._upstream(tmp_path)
        stdout = (
            "/p/a.cpp:1:  foo_func(x);\n"
            "/p/b.cpp:2:  foo_func(y);\n"
        )
        monkeypatch.setattr(native.subprocess, "run",
                            lambda *a, **k: _GrepResult(0, stdout))
        # >1 other_hit and no defn_hit -> chosen is None.
        assert native._locate_func_in_upstream("foo_func", up) is None

    def test_relative_to_value_error_falls_back_to_raw_path(self, tmp_path, monkeypatch):
        up = self._upstream(tmp_path)
        # grep reports a path OUTSIDE upstream_dir -> relative_to raises
        # ValueError -> rel falls back to the raw path string.
        outside = tmp_path / "elsewhere.cpp"
        outside.write_text("void foo_func() {\n}\n")
        stdout = f"{outside}:1:void foo_func() {{\n"
        monkeypatch.setattr(native.subprocess, "run",
                            lambda *a, **k: _GrepResult(0, stdout))
        loc = native._locate_func_in_upstream("foo_func", up)
        assert loc is not None
        # raw path retained (could not be made relative).
        assert "elsewhere.cpp" in loc

    def test_definition_preferred_over_call(self, tmp_path, monkeypatch):
        up = self._upstream(tmp_path)
        defn = up / "impl.cpp"
        defn.write_text("void foo_func(int x) {\n}\n")
        stdout = (
            "/p/call.cpp:9:  foo_func(z);\n"
            f"{defn}:1:void foo_func(int x) {{\n"
        )
        monkeypatch.setattr(native.subprocess, "run",
                            lambda *a, **k: _GrepResult(0, stdout))
        loc = native._locate_func_in_upstream("foo_func", up)
        assert loc == "impl.cpp:1"

    def test_empty_func_returns_none(self, tmp_path):
        up = self._upstream(tmp_path)
        assert native._locate_func_in_upstream("", up) is None

    def test_non_alpha_first_char_returns_none(self, tmp_path):
        up = self._upstream(tmp_path)
        # bare symbol starts with non-alpha -> skip.
        assert native._locate_func_in_upstream("123abc", up) is None


# --------------------------------------------------------------------------
# _discover_candidate_pids — with parent set, _pgrep_pids stubbed
# --------------------------------------------------------------------------

def test_discover_candidate_pids_unions_pgrep(tmp_path, monkeypatch):
    s = native.NativeSampler(tmp_path)
    s.parent_pid = 1000

    def fake_pgrep(args):
        # -P (children) returns {1001}; -g (group) returns {1002}.
        if args[0] == "-P":
            return {1001}
        return {1002}
    monkeypatch.setattr(native, "_pgrep_pids", fake_pgrep)
    pids = s._discover_candidate_pids()
    # parent + child + group member, sorted, positives only.
    assert pids == [1000, 1001, 1002]


def test_discover_candidate_pids_filters_nonpositive(tmp_path, monkeypatch):
    s = native.NativeSampler(tmp_path)
    s.parent_pid = 500
    monkeypatch.setattr(native, "_pgrep_pids", lambda args: {0, -1, 501})
    pids = s._discover_candidate_pids()
    assert pids == [500, 501]
