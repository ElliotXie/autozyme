"""Wave-4 coverage for zyme.commands.package.sync_manifests — the diff /
merge / scan / rewrite-fallback branches left by
tests/test_package_sync_manifests.py.

Targets:
  - _top_level_pkg
  - _scan_python: non-dir skip, no __init__ skip, no register_patch skip,
    SyntaxError skip, name-only-no-upstreams skip, tested_upstream_versions merge
  - _scan_r: no-patches-dir, non-dir skip, no patch.R skip, unbalanced parens,
    name-without-upstream skip
  - _load_current_python: no UPSTREAMS / non-dict / non-str key/value paths
  - _load_current_r: no block returns {}
  - _diff_py / _diff_r: added / removed / changed-common lines
  - _format_py_block / _format_r_block
  - _find_balanced_block: no header / no open char / unbalanced
  - _rewrite_with_sentinels: sentinel path + fallback-header path + die path
  - cmd_package_sync_manifests: missing framework root -> die; R-only diff;
    py-only diff
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands.package import sync_manifests as sm
from zyme.commands.package.sync_manifests import (
    _diff_py,
    _diff_r,
    _find_balanced_block,
    _format_py_block,
    _format_r_block,
    _load_current_python,
    _load_current_r,
    _rewrite_with_sentinels,
    _scan_python,
    _scan_r,
    _top_level_pkg,
    cmd_package_sync_manifests,
)


# --------------------------------------------------------------------------
# _top_level_pkg
# --------------------------------------------------------------------------

def test_top_level_pkg():
    assert _top_level_pkg("scanpy.tools.umap") == "scanpy"
    assert _top_level_pkg("scanpy") == "scanpy"


# --------------------------------------------------------------------------
# _scan_python edge branches
# --------------------------------------------------------------------------

def _py_root(tmp_path: Path) -> Path:
    root = tmp_path / "autozyme_py" / "src" / "autozyme"
    root.mkdir(parents=True)
    return tmp_path


class TestScanPython:
    def test_skips_non_dir_and_missing_init_and_no_register(self, tmp_path: Path):
        fr = _py_root(tmp_path)
        root = fr / "autozyme_py" / "src" / "autozyme"
        # A file (not a dir) at the top level -> skipped.
        (root / "loose_file.py").write_text("x = 1\n")
        # A dir with no __init__.py -> skipped.
        (root / "no_init").mkdir()
        # A dir whose __init__ has no register_patch -> skipped.
        (root / "plain").mkdir()
        (root / "plain" / "__init__.py").write_text("VALUE = 3\n")
        assert _scan_python(fr) == {}

    def test_syntax_error_skipped(self, tmp_path: Path):
        fr = _py_root(tmp_path)
        root = fr / "autozyme_py" / "src" / "autozyme"
        (root / "broken").mkdir()
        (root / "broken" / "__init__.py").write_text(
            "register_patch(\n  def bad syntax here\n"
        )
        # contains 'register_patch' substring so it's parsed, but SyntaxError.
        assert _scan_python(fr) == {}

    def test_name_without_upstreams_skipped(self, tmp_path: Path):
        fr = _py_root(tmp_path)
        root = fr / "autozyme_py" / "src" / "autozyme"
        (root / "nameonly").mkdir()
        (root / "nameonly" / "__init__.py").write_text(textwrap.dedent('''\
            def reg(**k): pass
            register_patch(name="nameonly")
        '''))
        # name present but no targets/tested -> not emitted.
        assert _scan_python(fr) == {}

    def test_non_register_call_and_non_tuple_target_skipped(self, tmp_path: Path):
        # Exercises the `fname != "register_patch"` continue (a plain call in
        # the module) AND a target element that is a bare string, not a tuple
        # (the `not isinstance(elt, (Tuple, List))` continue).
        fr = _py_root(tmp_path)
        root = fr / "autozyme_py" / "src" / "autozyme"
        (root / "weird").mkdir()
        (root / "weird" / "__init__.py").write_text(textwrap.dedent('''\
            def f(*a, **k): pass
            f(1, 2)  # a non-register_patch Call node
            register_patch(
                name="weird",
                targets=["not-a-tuple", ("scanpy.tools", "umap", f), ()],
            )
        '''))
        got = _scan_python(fr)
        # The bare-string target + empty tuple are skipped; scanpy survives.
        assert got["weird"] == ["scanpy"]

    def test_tested_upstream_versions_merged(self, tmp_path: Path):
        fr = _py_root(tmp_path)
        root = fr / "autozyme_py" / "src" / "autozyme"
        (root / "merged").mkdir()
        (root / "merged" / "__init__.py").write_text(textwrap.dedent('''\
            def f(*a, **k): pass
            register_patch(
                name="merged",
                targets=[("scanpy.tools", "umap", f)],
                tested_upstream_versions={"pyro.infer": "1.0", "scanpy": "1.10"},
            )
        '''))
        got = _scan_python(fr)
        # scanpy (from targets + tested) + pyro (from tested only).
        assert got["merged"] == ["pyro", "scanpy"]


# --------------------------------------------------------------------------
# _scan_r edge branches
# --------------------------------------------------------------------------

class TestScanR:
    def test_no_patches_dir(self, tmp_path: Path):
        # No autozyme_r/inst/patches -> {}.
        assert _scan_r(tmp_path) == {}

    def test_skips_non_dir_no_patchfile_and_unbalanced(self, tmp_path: Path):
        root = tmp_path / "autozyme_r" / "inst" / "patches"
        root.mkdir(parents=True)
        # loose file at patches level -> skipped.
        (root / "loose.txt").write_text("x")
        # dir without patch.R -> skipped.
        (root / "nofile").mkdir()
        # dir with unbalanced register_patch( -> depth never closes -> skipped.
        (root / "unbalanced").mkdir()
        (root / "unbalanced" / "patch.R").write_text(
            'register_patch(\n  name = "u",\n  upstream = "U"\n'  # no closing )
        )
        assert _scan_r(tmp_path) == {}

    def test_name_without_upstream_skipped(self, tmp_path: Path):
        root = tmp_path / "autozyme_r" / "inst" / "patches"
        root.mkdir(parents=True)
        (root / "p").mkdir()
        (root / "p" / "patch.R").write_text(
            'register_patch(\n  name = "p",\n  targets = list(x = f)\n)\n'
        )
        # name but no upstream -> not added.
        assert _scan_r(tmp_path) == {}

    def test_valid_pair(self, tmp_path: Path):
        root = tmp_path / "autozyme_r" / "inst" / "patches"
        root.mkdir(parents=True)
        (root / "p").mkdir()
        (root / "p" / "patch.R").write_text(
            'register_patch(\n  name = "p",\n  upstream = "Seurat"\n)\n'
        )
        assert _scan_r(tmp_path) == {"p": "Seurat"}


# --------------------------------------------------------------------------
# _load_current_python branches
# --------------------------------------------------------------------------

class TestLoadCurrentPython:
    def test_no_upstreams_symbol(self, tmp_path: Path):
        p = tmp_path / "_subsets.py"
        p.write_text("OTHER = 1\nUPSTREAMS_X = {}\n")
        assert _load_current_python(p) == {}

    def test_upstreams_not_a_dict(self, tmp_path: Path):
        p = tmp_path / "_subsets.py"
        p.write_text("UPSTREAMS = []\n")
        # Value isn't ast.Dict -> falls through to {}.
        assert _load_current_python(p) == {}

    def test_skips_non_str_keys_and_non_list_values(self, tmp_path: Path):
        p = tmp_path / "_subsets.py"
        p.write_text(textwrap.dedent('''\
            UPSTREAMS: dict = {
                "good": ["scanpy"],
                123: ["bad_key"],
                "scalar": "not-a-list",
            }
        '''))
        got = _load_current_python(p)
        assert got == {"good": ["scanpy"]}

    def test_annassign_form(self, tmp_path: Path):
        p = tmp_path / "_subsets.py"
        p.write_text('UPSTREAMS: dict[str, list[str]] = {"a": ["x", "y"]}\n')
        assert _load_current_python(p) == {"a": ["x", "y"]}

    def test_skips_non_assign_nodes(self, tmp_path: Path):
        # Import + function-def nodes precede UPSTREAMS; the loop must skip
        # any node that's neither Assign nor AnnAssign (line 147 continue).
        p = tmp_path / "_subsets.py"
        p.write_text(textwrap.dedent('''\
            import os

            def helper():
                return 1

            UPSTREAMS = {"a": ["x"]}
        '''))
        assert _load_current_python(p) == {"a": ["x"]}


# --------------------------------------------------------------------------
# _load_current_r no-block branch
# --------------------------------------------------------------------------

class TestLoadCurrentR:
    def test_no_block_returns_empty(self, tmp_path: Path):
        p = tmp_path / "subsets.R"
        p.write_text("# just a comment, no .zyme_upstreams list\n")
        assert _load_current_r(p) == {}

    def test_parses_pairs(self, tmp_path: Path):
        p = tmp_path / "subsets.R"
        p.write_text('.zyme_upstreams <- list(\n  a = "X",\n  b = "Y"\n)\n')
        assert _load_current_r(p) == {"a": "X", "b": "Y"}


# --------------------------------------------------------------------------
# _diff_py / _diff_r added/removed/changed
# --------------------------------------------------------------------------

class TestDiffs:
    def test_diff_py_added_removed_changed(self):
        current = {"keep": ["x"], "remove": ["r"], "change": ["old"]}
        target = {"keep": ["x"], "add": ["a"], "change": ["new"]}
        lines = _diff_py(current, target)
        joined = "\n".join(lines)
        assert "+ 'add'" in joined
        assert "- 'remove'" in joined
        assert "~ 'change'" in joined and "->" in joined
        # 'keep' unchanged -> no line.
        assert "keep" not in joined

    def test_diff_r_added_removed_changed(self):
        current = {"keep": "X", "remove": "R", "change": "Old"}
        target = {"keep": "X", "add": "A", "change": "New"}
        lines = _diff_r(current, target)
        joined = "\n".join(lines)
        assert "+ 'add'" in joined
        assert "- 'remove'" in joined
        assert "~ 'change'" in joined and "->" in joined


# --------------------------------------------------------------------------
# format blocks
# --------------------------------------------------------------------------

class TestFormatBlocks:
    def test_py_block_has_sentinels(self):
        block = _format_py_block({"a": ["x"]})
        assert sm._PY_BEGIN in block and sm._PY_END in block
        assert '"a": ["x"]' in block

    def test_r_block_alignment(self):
        block = _format_r_block({"short": "A", "longer_one": "B"})
        assert sm._R_BEGIN in block and sm._R_END in block
        assert 'longer_one = "B"' in block

    def test_r_block_empty(self):
        block = _format_r_block({})
        assert ".zyme_upstreams <- list(" in block


# --------------------------------------------------------------------------
# _find_balanced_block branches
# --------------------------------------------------------------------------

class TestFindBalancedBlock:
    def test_no_header_match(self):
        assert _find_balanced_block("nothing here", re.compile("HEADER"), "{", "}") is None

    def test_no_open_char(self):
        # Header present but no open bracket after it.
        text = "HEADER = no brackets at all"
        assert _find_balanced_block(text, re.compile("HEADER ="), "{", "}") is None

    def test_unbalanced(self):
        text = "HEADER = {a: 1"  # never closes
        assert _find_balanced_block(text, re.compile("HEADER ="), "{", "}") is None

    def test_balanced_span(self):
        text = "x = 1\nHEADER = {a: {b: 1}}\ny = 2\n"
        span = _find_balanced_block(text, re.compile("HEADER ="), "{", "}")
        assert span is not None
        start, end = span
        assert text[start:end].startswith("HEADER = {")
        assert text[start:end].endswith("}")


# --------------------------------------------------------------------------
# _rewrite_with_sentinels
# --------------------------------------------------------------------------

class TestRewriteWithSentinels:
    def test_sentinel_replace_path(self, tmp_path: Path):
        p = tmp_path / "_subsets.py"
        p.write_text(
            f"prefix\n{sm._PY_BEGIN}\nold stuff\n{sm._PY_END}\nsuffix\n"
        )
        new_block = f"{sm._PY_BEGIN}\nNEW\n{sm._PY_END}"
        _rewrite_with_sentinels(
            p, sm._PY_BEGIN, sm._PY_END, new_block,
            re.compile("UPSTREAMS ="), "{", "}",
        )
        text = p.read_text()
        assert "NEW" in text and "old stuff" not in text
        assert "prefix" in text and "suffix" in text

    def test_fallback_header_path_when_no_sentinels(self, tmp_path: Path):
        p = tmp_path / "_subsets.py"
        p.write_text("before\nUPSTREAMS = {\n  'a': ['x'],\n}\nafter\n")
        new_block = f"{sm._PY_BEGIN}\nUPSTREAMS = {{}}\n{sm._PY_END}"
        _rewrite_with_sentinels(
            p, sm._PY_BEGIN, sm._PY_END, new_block,
            re.compile(r"UPSTREAMS\s*=\s*"), "{", "}",
        )
        text = p.read_text()
        assert sm._PY_BEGIN in text
        assert "before" in text and "after" in text
        assert "'a': ['x']" not in text

    def test_die_when_no_sentinel_and_no_header(self, tmp_path: Path):
        p = tmp_path / "_subsets.py"
        p.write_text("nothing to anchor on here\n")
        with pytest.raises(SystemExit):
            _rewrite_with_sentinels(
                p, sm._PY_BEGIN, sm._PY_END, "BLOCK",
                re.compile("UPSTREAMS ="), "{", "}",
            )


# --------------------------------------------------------------------------
# cmd_package_sync_manifests top-level branches
# --------------------------------------------------------------------------

class TestCmdSyncManifests:
    def test_missing_framework_root_dies(self, tmp_path: Path, monkeypatch):
        # find_framework_root returns None -> die().
        monkeypatch.setattr(sm, "find_framework_root", lambda p: None)
        args = SimpleNamespace(framework_root=None, apply=False)
        with pytest.raises(SystemExit):
            cmd_package_sync_manifests(args)

    def test_in_sync_returns_zero(self, tmp_path: Path, capsys):
        # Build a framework where current manifests already match scanned.
        fr = self._framework(tmp_path)
        # First apply to sync, then dry-run reports in sync + rc 0.
        cmd_package_sync_manifests(SimpleNamespace(framework_root=str(fr), apply=True))
        capsys.readouterr()
        rc = cmd_package_sync_manifests(SimpleNamespace(framework_root=str(fr), apply=False))
        out = capsys.readouterr().out
        assert rc == 0
        assert out.count("(in sync)") == 2

    def test_py_only_diff_dry_run_returns_one(self, tmp_path: Path, capsys):
        fr = self._framework(tmp_path)
        # Stale python manifest, R manifest empty + no R patches -> only py diff.
        rc = cmd_package_sync_manifests(SimpleNamespace(framework_root=str(fr), apply=False))
        out = capsys.readouterr().out
        assert rc == 1
        assert "Run with --apply" in out

    @staticmethod
    def _framework(tmp_path: Path) -> Path:
        fr = tmp_path / "fw"
        py = fr / "autozyme_py" / "src" / "autozyme"
        (py / "plug").mkdir(parents=True)
        (py / "plug" / "__init__.py").write_text(textwrap.dedent('''\
            def f(*a, **k): pass
            register_patch(name="plug", targets=[("scanpy.tools", "umap", f)])
        '''))
        (py / "_subsets.py").write_text('UPSTREAMS: dict = {"stale": ["foo"]}\n')
        # R side: empty manifest + no patches dir -> r_target == r_current == {}.
        (fr / "autozyme_r" / "R").mkdir(parents=True)
        (fr / "autozyme_r" / "R" / "subsets.R").write_text(
            ".zyme_upstreams <- list(\n)\n"
        )
        return fr
