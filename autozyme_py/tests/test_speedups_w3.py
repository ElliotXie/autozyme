"""Wave-3 gap-fill for autozyme._speedups.

Wave-1's test_speedups_unit.py covered the coercion helpers, the two
summarizers, and the happy-path public reader. This file fills the remaining
defensive branches:
  - _ir_files() raising ModuleNotFoundError/AttributeError -> [] (262-263)
  - facade sibling iterdir() raising -> empty siblings (283-284)
  - mixed-case name retried lowercase from the empty-result path (298)
  - close-name suggestion path with base.iterdir() raising -> no matches (308-309)

We drive these by monkeypatching the module-level ``_ir_files`` symbol or by
replacing the ``base`` object's ``iterdir`` with a raising stub via a fake
package root.
"""
from __future__ import annotations

import warnings

import pytest

from autozyme import _speedups as S


# --------------------------------------------------------------------------
# _ir_files raising -> early [] (lines 262-263)
# --------------------------------------------------------------------------
def test_speedups_ir_files_module_not_found(monkeypatch):
    def _raise(_pkg):
        raise ModuleNotFoundError("no autozyme resources")

    monkeypatch.setattr(S, "_ir_files", _raise)
    assert S.speedups("lifelines") == []


def test_speedups_ir_files_attribute_error(monkeypatch):
    def _raise(_pkg):
        raise AttributeError("files() missing")

    monkeypatch.setattr(S, "_ir_files", _raise)
    assert S.speedups("scanpy") == []


# --------------------------------------------------------------------------
# A fake resources root so we can control is_file()/iterdir() behavior.
# --------------------------------------------------------------------------
class _FakeRsrc:
    """Minimal stand-in for a Traversable that is never a real file."""

    def __init__(self, name="x"):
        self.name = name

    def __truediv__(self, _other):
        return _FakeRsrc(_other)

    def is_file(self):
        return False

    def is_dir(self):
        return False


class _FakeBaseIterdirRaises(_FakeRsrc):
    def iterdir(self):
        raise OSError("cannot list package dir")


def test_speedups_facade_sibling_iterdir_error(monkeypatch):
    """scanpy is a facade; if base.iterdir() raises while gathering siblings,
    the except clause yields no sibling dirs (lines 283-284) and, with no rows,
    we fall through to the empty path (also exercising the close-name
    suggestion branch with another iterdir failure)."""
    base = _FakeBaseIterdirRaises()
    monkeypatch.setattr(S, "_ir_files", lambda _pkg: base)
    # scanpy is in _FACADES -> sibling-gather path runs and hits 283-284;
    # base/scanpy/speedups_finalized.tsv is_file() -> False so no direct rows.
    out = S.speedups("scanpy")
    assert out == []


def test_speedups_close_name_iterdir_error(monkeypatch):
    """When the final close-name suggestion also can't list the package dir,
    available stays [] (lines 308-309) and we return [] with no warning."""
    base = _FakeBaseIterdirRaises()
    monkeypatch.setattr(S, "_ir_files", lambda _pkg: base)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail the test
        # lowercase, non-facade name -> straight to empty path; iterdir raises
        out = S.speedups("notarealpatch")
    assert out == []


# --------------------------------------------------------------------------
# mixed-case retry from the empty-result path (line 298)
# --------------------------------------------------------------------------
def test_speedups_mixedcase_retry_empty(monkeypatch):
    """A mixed-case name that yields no rows recurses with name.lower().
    Use a fake root so neither the original nor the lowercased name finds a
    file, forcing the 298 -> recurse -> still-empty branch deterministically."""
    base = _FakeRsrc()  # is_file()/is_dir() always False; iterdir absent on dirs

    # iterdir on the empty-path needs to exist; provide one returning nothing.
    def _iterdir():
        return iter(())

    base.iterdir = _iterdir  # type: ignore[attr-defined]
    monkeypatch.setattr(S, "_ir_files", lambda _pkg: base)
    # "Xyz" != "xyz" -> line 298 recurse; lowercased still empty -> []
    out = S.speedups("Xyz")
    assert out == []
