"""Tests for `zyme init-attest` — the post-publication attest scaffolder.

Clones the canonical find_markers task structure into a fresh task dir,
substitutes the name/target, and replicates the data symlink. These assert the
scaffold lands the right files and never drags along measurements/caches.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from zyme.commands.init_attest import cmd_init_attest, _FRAMEWORK


def _args(name, dest, **kw):
    return SimpleNamespace(
        name=name, like=kw.get("like", "find_markers"),
        target=kw.get("target"), patch=kw.get("patch", "seurat"),
        dest=str(dest), force=kw.get("force", False),
    )


@pytest.mark.skipif(
    not (_FRAMEWORK / "postpublication" / "find_markers" / "task.yaml").is_file(),
    reason="canonical find_markers task not present",
)
def test_scaffold_clones_structure(tmp_path, capsys):
    rc = cmd_init_attest(_args("my_new_task", tmp_path, target="Seurat::FindNeighbors"))
    assert rc == 0
    d = tmp_path / "my_new_task"
    # canonical files cloned
    for rel in ("task.yaml", "attest/smoke.R", "evaluate.R", ".gitignore"):
        assert (d / rel).is_file(), f"missing {rel}"
    # measurements / caches NOT cloned
    assert not (d / "package_verify.tsv").exists()
    assert not list(d.glob("reference_output_*"))
    # name + target substituted, TODO banner prepended
    ty = (d / "task.yaml").read_text()
    assert ty.startswith("# TODO(init-attest):")
    assert "task: my_new_task" in ty
    assert "target_function: Seurat::FindNeighbors" in ty
    # data symlink replicated AND actually resolves (a relative target copied
    # verbatim would dangle under --dest; assert the link points at a real dir,
    # not just that the string looks right).
    data = d / "data"
    assert data.is_symlink()
    assert "optimized_task" in os.readlink(data)
    assert os.path.isdir(os.path.realpath(data)), f"data symlink dangles: {os.readlink(data)}"
    # next-steps checklist printed
    out = capsys.readouterr().out
    assert "init-attest" in out and "publish auto-skips" in out


def test_unknown_like_dies(tmp_path):
    with pytest.raises(SystemExit):
        cmd_init_attest(_args("x", tmp_path, like="does_not_exist"))


def test_existing_dest_needs_force(tmp_path):
    cmd_init_attest(_args("dup", tmp_path))
    with pytest.raises(SystemExit):
        cmd_init_attest(_args("dup", tmp_path))  # exists, no --force
    # --force overwrites cleanly
    assert cmd_init_attest(_args("dup", tmp_path, force=True)) == 0
