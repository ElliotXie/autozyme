"""Wave-3 coverage for zyme.commands.package.check_versions.

tests/test_package_check_versions.py covers _parse_tested_against,
_scan_python/_scan_r happy paths, and _build_rows status classification
(with both probes monkeypatched). This file fills the reachable gaps:

  - _scan_python skip branches: non-dir child, missing __init__, no
    register_patch/tested_against marker, SyntaxError, register_patch with
    only one of name/tested, name-as-bare-Name() call form.
  - _scan_r skip branches: missing root, non-dir child, no patch.R,
    unbalanced parens, missing name/tested.
  - _installed_python real lookup (installed + PackageNotFound).
  - _installed_r_batch: empty list short-circuit, FileNotFoundError /
    nonzero-rc / bad-json all -> None, happy JSON path — all with
    subprocess.run monkeypatched (no real Rscript).
  - _print_table: empty filtered, filter_status, full render.
  - cmd_package_check_versions: no-framework die, no-rows, ok-only exit 0,
    drift exit 1, --only-drift filter.

No real subprocess; _installed_r_batch's subprocess.run is monkeypatched.
"""
from __future__ import annotations

import importlib.metadata
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands.package import check_versions as cv
from zyme.commands.package.check_versions import (
    _VersionRow,
    _installed_python,
    _installed_r_batch,
    _print_table,
    _scan_python,
    _scan_r,
    cmd_package_check_versions,
)


# --------------------------------------------------------------------------
# _scan_python — skip branches
# --------------------------------------------------------------------------

def _py_root(tmp_path: Path) -> Path:
    root = tmp_path / "autozyme_py" / "src" / "autozyme"
    root.mkdir(parents=True)
    return tmp_path


def _add_py_patch(fr: Path, name: str, body: str) -> Path:
    d = fr / "autozyme_py" / "src" / "autozyme" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "__init__.py").write_text(textwrap.dedent(body))
    return d


class TestScanPythonSkips:
    def test_non_dir_child_skipped(self, tmp_path):
        fr = _py_root(tmp_path)
        # a stray file (not a dir) sitting in the autozyme/ package root.
        (fr / "autozyme_py" / "src" / "autozyme" / "stray.txt").write_text("x")
        assert _scan_python(fr) == []

    def test_dir_without_init_skipped(self, tmp_path):
        fr = _py_root(tmp_path)
        (fr / "autozyme_py" / "src" / "autozyme" / "noinit").mkdir()
        assert _scan_python(fr) == []

    def test_init_without_markers_skipped(self, tmp_path):
        fr = _py_root(tmp_path)
        _add_py_patch(fr, "plain", "x = 1\n")
        assert _scan_python(fr) == []

    def test_init_with_marker_but_no_tested_against_text(self, tmp_path):
        fr = _py_root(tmp_path)
        # has register_patch but not the literal "tested_against" token.
        _add_py_patch(fr, "p", "register_patch(name='p')\n")
        assert _scan_python(fr) == []

    def test_syntax_error_skipped(self, tmp_path):
        fr = _py_root(tmp_path)
        _add_py_patch(fr, "broken",
                      "register_patch( tested_against = 'scanpy 1.0'  # unbalanced\n")
        assert _scan_python(fr) == []

    def test_register_patch_missing_name_kw_skipped(self, tmp_path):
        fr = _py_root(tmp_path)
        _add_py_patch(fr, "noname",
                      "register_patch(tested_against='scanpy 1.0')\n")
        assert _scan_python(fr) == []

    def test_register_patch_unparseable_tested_skipped(self, tmp_path):
        fr = _py_root(tmp_path)
        # name + tested present, but tested_against can't be parsed (no version).
        _add_py_patch(fr, "bad",
                      "register_patch(name='bad', tested_against='scanpy')\n")
        assert _scan_python(fr) == []

    def test_bare_name_call_form_resolves(self, tmp_path):
        fr = _py_root(tmp_path)
        # register_patch called as a bare Name (imported), not attribute.
        _add_py_patch(fr, "bare", textwrap.dedent('''
            from autozyme._core import register_patch
            register_patch(name="bare", tested_against="scanpy 1.2.3")
        '''))
        assert _scan_python(fr) == [("bare", "scanpy", "1.2.3")]

    def test_non_register_patch_call_ignored(self, tmp_path):
        fr = _py_root(tmp_path)
        # an unrelated call with name= and tested_against= kwargs must not
        # be picked up (only register_patch counts), but the file still has
        # the markers so it is parsed.
        _add_py_patch(fr, "x", textwrap.dedent('''
            register_patch  # token present so the file is parsed
            something_else(name="x", tested_against="scanpy 1.0")
        '''))
        assert _scan_python(fr) == []


# --------------------------------------------------------------------------
# _scan_r — skip branches
# --------------------------------------------------------------------------

def _r_root(tmp_path: Path) -> Path:
    (tmp_path / "autozyme_r" / "inst" / "patches").mkdir(parents=True)
    return tmp_path


def _add_r_patch(fr: Path, name: str, body: str) -> Path:
    d = fr / "autozyme_r" / "inst" / "patches" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "patch.R").write_text(textwrap.dedent(body))
    return d


class TestScanRSkips:
    def test_missing_root_returns_empty(self, tmp_path):
        # no autozyme_r/ at all.
        assert _scan_r(tmp_path) == []

    def test_non_dir_child_skipped(self, tmp_path):
        fr = _r_root(tmp_path)
        (fr / "autozyme_r" / "inst" / "patches" / "stray.txt").write_text("x")
        assert _scan_r(fr) == []

    def test_dir_without_patch_r_skipped(self, tmp_path):
        fr = _r_root(tmp_path)
        (fr / "autozyme_r" / "inst" / "patches" / "empty").mkdir()
        assert _scan_r(fr) == []

    def test_unbalanced_parens_skipped(self, tmp_path):
        fr = _r_root(tmp_path)
        # register_patch( without a closing ) -> depth never returns to 0.
        _add_r_patch(fr, "unbal",
                     'register_patch( name = "u", tested_against = "MAST 1.0"')
        assert _scan_r(fr) == []

    def test_missing_name_or_tested_breaks(self, tmp_path):
        fr = _r_root(tmp_path)
        _add_r_patch(fr, "nm",
                     'register_patch( upstream = "MAST" )\n')
        assert _scan_r(fr) == []

    def test_happy_path(self, tmp_path):
        fr = _r_root(tmp_path)
        _add_r_patch(fr, "mast", '''
            register_patch(
              name = "mast",
              tested_against = "MAST 1.36.0"
            )
        ''')
        assert _scan_r(fr) == [("mast", "MAST", "1.36.0")]


# --------------------------------------------------------------------------
# _installed_python — real importlib.metadata
# --------------------------------------------------------------------------

class TestInstalledPython:
    def test_installed_pkg_returns_version(self):
        # pytest is definitely installed in this env.
        v = _installed_python("pytest")
        assert v is not None
        assert v == importlib.metadata.version("pytest")

    def test_missing_pkg_returns_none(self):
        assert _installed_python("definitely-not-a-real-pkg-xyz999") is None


# --------------------------------------------------------------------------
# _installed_r_batch — subprocess monkeypatched
# --------------------------------------------------------------------------

class _Proc:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


class TestInstalledRBatch:
    def test_empty_pkgs_short_circuit(self):
        assert _installed_r_batch([]) == {}

    def test_rscript_not_found(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("Rscript")
        monkeypatch.setattr(cv.subprocess, "run", boom)
        assert _installed_r_batch(["MAST", "spacexr"]) == {"MAST": None, "spacexr": None}

    def test_timeout(self, monkeypatch):
        import subprocess as sp

        def boom(*a, **k):
            raise sp.TimeoutExpired(cmd="Rscript", timeout=60)
        monkeypatch.setattr(cv.subprocess, "run", boom)
        assert _installed_r_batch(["MAST"]) == {"MAST": None}

    def test_nonzero_returncode(self, monkeypatch):
        monkeypatch.setattr(cv.subprocess, "run",
                            lambda *a, **k: _Proc(returncode=1, stdout=""))
        assert _installed_r_batch(["MAST"]) == {"MAST": None}

    def test_bad_json(self, monkeypatch):
        monkeypatch.setattr(cv.subprocess, "run",
                            lambda *a, **k: _Proc(returncode=0, stdout="not json"))
        assert _installed_r_batch(["MAST"]) == {"MAST": None}

    def test_happy_json_path(self, monkeypatch):
        monkeypatch.setattr(
            cv.subprocess, "run",
            lambda *a, **k: _Proc(returncode=0, stdout='{"MAST": "1.36.0"}'),
        )
        # spacexr missing from JSON -> None via data.get.
        out = _installed_r_batch(["MAST", "spacexr"])
        assert out == {"MAST": "1.36.0", "spacexr": None}


# --------------------------------------------------------------------------
# _print_table
# --------------------------------------------------------------------------

def _row(patch, status, installed="1.0.0", tested="1.0.0"):
    return _VersionRow(patch=patch, language="py", upstream="scanpy",
                       tested=tested, installed=installed, status=status)


class TestPrintTable:
    def test_no_rows_prints_marker(self, capsys):
        _print_table([])
        assert "(no rows)" in capsys.readouterr().out

    def test_filter_removes_all_prints_marker(self, capsys):
        _print_table([_row("a", "ok")], filter_status={"drift"})
        assert "(no rows)" in capsys.readouterr().out

    def test_renders_header_and_rows(self, capsys):
        rows = [_row("scanpy_x", "ok"),
                _row("mast_y", "drift", installed="2.0.0", tested="1.0.0")]
        _print_table(rows)
        out = capsys.readouterr().out
        assert "patch" in out and "status" in out
        assert "scanpy_x" in out and "mast_y" in out
        assert "drift" in out

    def test_filter_keeps_only_matching(self, capsys):
        rows = [_row("ok_one", "ok"), _row("drift_one", "drift")]
        _print_table(rows, filter_status={"drift"})
        out = capsys.readouterr().out
        assert "drift_one" in out
        assert "ok_one" not in out

    def test_none_installed_rendered_as_dash(self, capsys):
        rows = [_VersionRow("m", "py", "missingpkg", "1.0", None, "missing")]
        _print_table(rows)
        out = capsys.readouterr().out
        assert "missing" in out
        # the installed column shows "-" for a None install.
        assert " - " in out or out.rstrip().endswith("-") or "-  missing" in out


# --------------------------------------------------------------------------
# cmd_package_check_versions
# --------------------------------------------------------------------------

def _make_full_framework(tmp_path: Path) -> Path:
    fr = tmp_path / "autozyme-framework"
    _add_py_patch(fr, "scanpy_test", '''
        from autozyme._core import register_patch
        register_patch(name="scanpy_test", tested_against="scanpy 1.11.5")
    ''')
    return fr


class TestCmdCheckVersions:
    def test_no_framework_dies(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cv, "find_framework_root", lambda p: None)
        args = SimpleNamespace(framework_root=None)
        with pytest.raises(SystemExit):
            cmd_package_check_versions(args)

    def test_no_rows_returns_zero(self, tmp_path, capsys):
        # empty framework -> no patches with tested_against.
        fr = tmp_path / "autozyme-framework"
        (fr / "autozyme_py" / "src" / "autozyme").mkdir(parents=True)
        args = SimpleNamespace(framework_root=str(fr))
        rc = cmd_package_check_versions(args)
        assert rc == 0
        assert "no patches with tested_against" in capsys.readouterr().out

    def test_all_ok_returns_zero(self, tmp_path, monkeypatch, capsys):
        fr = _make_full_framework(tmp_path)
        monkeypatch.setattr(cv, "_installed_python", lambda pkg: "1.11.5")
        args = SimpleNamespace(framework_root=str(fr), only_drift=False)
        rc = cmd_package_check_versions(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 ok" in out
        assert "scanpy_test" in out

    def test_drift_returns_one(self, tmp_path, monkeypatch, capsys):
        fr = _make_full_framework(tmp_path)
        monkeypatch.setattr(cv, "_installed_python", lambda pkg: "1.99.0")
        args = SimpleNamespace(framework_root=str(fr), only_drift=False)
        rc = cmd_package_check_versions(args)
        assert rc == 1
        assert "drift" in capsys.readouterr().out

    def test_missing_returns_one(self, tmp_path, monkeypatch, capsys):
        fr = _make_full_framework(tmp_path)
        monkeypatch.setattr(cv, "_installed_python", lambda pkg: None)
        args = SimpleNamespace(framework_root=str(fr), only_drift=False)
        rc = cmd_package_check_versions(args)
        assert rc == 1
        assert "missing" in capsys.readouterr().out

    def test_only_drift_filter(self, tmp_path, monkeypatch, capsys):
        # two patches: one ok, one drift; --only-drift hides the ok one.
        fr = tmp_path / "autozyme-framework"
        _add_py_patch(fr, "ok_patch", '''
            from autozyme._core import register_patch
            register_patch(name="ok_patch", tested_against="numpy 1.0.0")
        ''')
        _add_py_patch(fr, "drift_patch", '''
            from autozyme._core import register_patch
            register_patch(name="drift_patch", tested_against="pandas 1.0.0")
        ''')

        def fake_installed(pkg):
            return "1.0.0" if pkg == "numpy" else "9.9.9"
        monkeypatch.setattr(cv, "_installed_python", fake_installed)
        args = SimpleNamespace(framework_root=str(fr), only_drift=True)
        rc = cmd_package_check_versions(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "drift_patch" in out
        assert "ok_patch" not in out.split("ok,")[0]  # not in the table body
