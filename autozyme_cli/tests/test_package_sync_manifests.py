"""sync-manifests: scan synthetic patch trees, diff, --apply round-trip."""
from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands.package.sync_manifests import (
    _format_py_block,
    _format_r_block,
    _load_current_python,
    _load_current_r,
    _scan_python,
    _scan_r,
    cmd_package_sync_manifests,
)


def _make_framework(tmp_path: Path) -> Path:
    fr = tmp_path / "autozyme-framework"
    (fr / "autozyme_py" / "src" / "autozyme" / "myplug").mkdir(parents=True)
    (fr / "autozyme_py" / "src" / "autozyme" / "myplug" / "__init__.py").write_text(textwrap.dedent('''\
        from autozyme._core import register_patch

        def fast_fn(*a, **k): pass

        register_patch(
            name="myplug",
            targets=[("scanpy.tools", "umap", fast_fn)],
            tested_against="scanpy 1.10",
        )
    '''))
    (fr / "autozyme_py" / "src" / "autozyme" / "_subsets.py").write_text(textwrap.dedent('''\
        UPSTREAMS: dict[str, list[str]] = {
            "stale": ["foo"],
        }
    '''))
    # R patch
    rpat = fr / "autozyme_r" / "inst" / "patches" / "myr"
    rpat.mkdir(parents=True)
    (rpat / "patch.R").write_text(textwrap.dedent('''\
        # not a register_patch:
        helper <- function(object.name = "object1") object.name
        register_patch(
          name = "myr",
          upstream = "MyUpstream",
          targets = list(foo = fast_foo),
          smoke = list(load = NULL)
        )
    '''))
    (fr / "autozyme_r" / "R").mkdir(parents=True, exist_ok=True)
    (fr / "autozyme_r" / "R" / "subsets.R").write_text(textwrap.dedent('''\
        # Header comment.
        .zyme_upstreams <- list(
          old_entry = "OldUpstream"
        )
        # Trailing comment.
    '''))
    return fr


def test_scan_python_extracts_targets_and_tested(tmp_path: Path):
    fr = _make_framework(tmp_path)
    got = _scan_python(fr)
    assert "myplug" in got
    # scanpy from targets + nothing from tested_against (no dict here).
    assert got["myplug"] == ["scanpy"]


def test_scan_r_ignores_unrelated_name_arg(tmp_path: Path):
    fr = _make_framework(tmp_path)
    got = _scan_r(fr)
    # "object.name = 'object1'" must NOT be picked up; only register_patch's
    # name/upstream pair.
    assert got == {"myr": "MyUpstream"}


def test_load_current_python_parses_dict(tmp_path: Path):
    fr = _make_framework(tmp_path)
    py = fr / "autozyme_py" / "src" / "autozyme" / "_subsets.py"
    got = _load_current_python(py)
    assert got == {"stale": ["foo"]}


def test_load_current_r_parses_list(tmp_path: Path):
    fr = _make_framework(tmp_path)
    r = fr / "autozyme_r" / "R" / "subsets.R"
    got = _load_current_r(r)
    assert got == {"old_entry": "OldUpstream"}


def test_apply_round_trip_writes_then_in_sync(tmp_path: Path, capsys):
    fr = _make_framework(tmp_path)
    args = SimpleNamespace(framework_root=str(fr), apply=True)
    rc = cmd_package_sync_manifests(args)
    assert rc == 0
    # Second run with no apply should report in-sync now.
    args2 = SimpleNamespace(framework_root=str(fr), apply=False)
    rc2 = cmd_package_sync_manifests(args2)
    out = capsys.readouterr().out
    assert "(in sync)" in out
    assert rc2 == 0


def test_dry_run_returns_nonzero_when_diff_present(tmp_path: Path):
    fr = _make_framework(tmp_path)
    args = SimpleNamespace(framework_root=str(fr), apply=False)
    rc = cmd_package_sync_manifests(args)
    assert rc != 0


def test_format_py_block_is_double_quoted():
    block = _format_py_block({"a": ["x", "y"]})
    assert '"a"' in block
    # No single-quoted dict keys (consistent with the rest of _subsets.py).
    assert "'a'" not in block


def test_format_r_block_aligns_padding():
    block = _format_r_block({"a": "X", "longer_key": "Y"})
    # Both names visible; longest key dictates alignment.
    assert "a         " in block or "  a " in block
    assert "longer_key = \"Y\"" in block
