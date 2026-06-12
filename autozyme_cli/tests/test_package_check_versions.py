"""check-versions: parse tested_against, classify drift status."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zyme.commands.package.check_versions import (
    _build_rows,
    _parse_tested_against,
    _scan_python,
    _scan_r,
)


# --------------------------------------------------------------------------
# _parse_tested_against
# --------------------------------------------------------------------------

@pytest.mark.parametrize("s,expected", [
    ("scanpy 1.11.5",        ("scanpy", "1.11.5")),
    ("MAST 1.36.0",          ("MAST", "1.36.0")),
    ("scvelo 0.3.3",         ("scvelo", "0.3.3")),
    ("cell2location 0.1.5",  ("cell2location", "0.1.5")),
    ("cellchat 2.2.0.9001",  ("cellchat", "2.2.0.9001")),
    ("xclim 0.60.0",         ("xclim", "0.60.0")),
    ("Seurat 5.2.1",         ("Seurat", "5.2.1")),
])
def test_parse_tested_against_canonical(s, expected):
    assert _parse_tested_against(s) == expected


@pytest.mark.parametrize("s", [
    "",                  # empty
    "scanpy",            # no version
    "1.0.0 scanpy",      # version first (reject — version starts with digit)
    "scanpy ",           # trailing space, no version
])
def test_parse_tested_against_rejects_bad_input(s):
    assert _parse_tested_against(s) is None


# --------------------------------------------------------------------------
# scan helpers
# --------------------------------------------------------------------------

def _make_framework(tmp_path: Path) -> Path:
    fr = tmp_path / "autozyme-framework"
    (fr / "autozyme_py" / "src" / "autozyme" / "scanpy_test").mkdir(parents=True)
    (fr / "autozyme_py" / "src" / "autozyme" / "scanpy_test" / "__init__.py").write_text(textwrap.dedent('''\
        from autozyme._core import register_patch
        def fast_fn(*a, **k): pass
        register_patch(
            name="scanpy_test",
            targets=[("scanpy.tools", "umap", fast_fn)],
            tested_against="scanpy 1.11.5",
        )
    '''))
    rpat = fr / "autozyme_r" / "inst" / "patches" / "mast_test"
    rpat.mkdir(parents=True)
    (rpat / "patch.R").write_text(textwrap.dedent('''\
        register_patch(
          name = "mast_test",
          upstream = "MAST",
          targets = list(lrTest = fast_lrTest),
          tested_against = "MAST 1.36.0"
        )
    '''))
    return fr


def test_scan_python_extracts_tested(tmp_path: Path):
    fr = _make_framework(tmp_path)
    rows = _scan_python(fr)
    assert rows == [("scanpy_test", "scanpy", "1.11.5")]


def test_scan_r_extracts_tested(tmp_path: Path):
    fr = _make_framework(tmp_path)
    rows = _scan_r(fr)
    assert rows == [("mast_test", "MAST", "1.36.0")]


def test_scan_python_skips_patches_without_tested(tmp_path: Path):
    fr = tmp_path / "autozyme-framework"
    (fr / "autozyme_py" / "src" / "autozyme" / "no_tested").mkdir(parents=True)
    (fr / "autozyme_py" / "src" / "autozyme" / "no_tested" / "__init__.py").write_text(
        'from autozyme._core import register_patch\n'
        'def fast_fn(*a, **k): pass\n'
        'register_patch(name="no_tested", targets=[("pkg", "fn", fast_fn)])\n'
    )
    assert _scan_python(fr) == []


# --------------------------------------------------------------------------
# _build_rows / classification (stub Rscript so the test is hermetic)
# --------------------------------------------------------------------------

def test_build_rows_classifies_status(tmp_path: Path, monkeypatch):
    fr = _make_framework(tmp_path)
    # Stub the Python installed-version probe to return a controlled value.
    monkeypatch.setattr(
        "zyme.commands.package.check_versions._installed_python",
        lambda pkg: "1.11.5" if pkg == "scanpy" else None,
    )
    # Stub the R batch probe so the test doesn't need Rscript on PATH.
    monkeypatch.setattr(
        "zyme.commands.package.check_versions._installed_r_batch",
        lambda pkgs: {"MAST": "1.36.0"} if "MAST" in pkgs else {},
    )
    rows = _build_rows(fr)
    by_patch = {r.patch: r for r in rows}
    assert by_patch["scanpy_test"].status == "ok"
    assert by_patch["mast_test"].status == "ok"


def test_build_rows_detects_drift(tmp_path: Path, monkeypatch):
    fr = _make_framework(tmp_path)
    monkeypatch.setattr(
        "zyme.commands.package.check_versions._installed_python",
        lambda pkg: "1.11.9",  # newer than tested 1.11.5
    )
    monkeypatch.setattr(
        "zyme.commands.package.check_versions._installed_r_batch",
        lambda pkgs: {"MAST": "1.40.0"},  # newer than tested 1.36.0
    )
    rows = _build_rows(fr)
    statuses = {r.patch: r.status for r in rows}
    assert statuses == {"scanpy_test": "drift", "mast_test": "drift"}


def test_build_rows_detects_missing(tmp_path: Path, monkeypatch):
    fr = _make_framework(tmp_path)
    monkeypatch.setattr(
        "zyme.commands.package.check_versions._installed_python",
        lambda pkg: None,
    )
    monkeypatch.setattr(
        "zyme.commands.package.check_versions._installed_r_batch",
        lambda pkgs: {p: None for p in pkgs},
    )
    rows = _build_rows(fr)
    statuses = {r.patch: r.status for r in rows}
    assert statuses == {"scanpy_test": "missing", "mast_test": "missing"}
