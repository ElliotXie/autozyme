"""Wave-4 mop-up for zyme.commands.prompt.

Wave-2 (test_prompt_cmd.py) drove the happy lifecycle (save/list/show/use/
annotate/diff) over a real fake-framework registry tree. This file fills the
remaining REACHABLE error / decision branches that the happy path skipped:

  - cmd_prompt_save: registry raises (live prompt not found) -> die.
  - cmd_prompt_list: a label filter that excludes every snapshot (continue).
  - cmd_prompt_show: find_snapshot raises ValueError (ambiguous id) -> die.
  - cmd_prompt_diff: missing second snapshot -> die; git rc not in (0,1) ->
    sys.exit propagates the unusual rc.
  - cmd_prompt_use: install_snapshot RegistryError -> die.
  - cmd_prompt_annotate: annotate_snapshot RegistryError -> die.

Only the git subprocess boundary is stubbed; the registry runs for real.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from zyme.commands import prompt as promptmod
from zyme import registry as reg


@pytest.fixture
def fw(tmp_path, monkeypatch):
    """Fake framework tree the registry resolves against (mirrors wave-2)."""
    root = tmp_path / "ws" / "autozyme-framework" / "autozyme_cli" / "zyme"
    bio = root / "prompts" / "Bio"
    bio.mkdir(parents=True)
    (root / "prompts" / "OtherField").mkdir(parents=True)
    live = bio / "2_iterate.md"
    live.write_text("# Iterate\nbody v1\n")
    monkeypatch.setattr(promptmod, "FRAMEWORK_ROOT", root)
    return SimpleNamespace(root=root, live=live, bio=bio)


def _save(fw, **over):
    args = SimpleNamespace(live_path=str(fw.live), labels=None, name="snap",
                           message="", notes="")
    for k, v in over.items():
        setattr(args, k, v)
    promptmod.cmd_prompt_save(args)


def _pid_from(capsys):
    out = capsys.readouterr().out
    return next(l for l in out.splitlines()
                if "saved:" in l).split("saved:")[1].strip()


# --------------------------------------------------------------------------
# cmd_prompt_save — registry error path (lines 55-56)
# --------------------------------------------------------------------------

def test_save_missing_live_prompt_dies(fw):
    # Point save at a file that doesn't exist -> RegistryError -> die().
    args = SimpleNamespace(live_path=str(fw.bio / "nonexistent.md"),
                           labels=None, name="x", message="", notes="")
    with pytest.raises(SystemExit):
        promptmod.cmd_prompt_save(args)


def test_save_prints_full_card_block(fw, capsys):
    # Exercises the message/labels echo branches (62-64).
    args = SimpleNamespace(
        live_path=str(fw.live),
        labels=["aggressiveness=7"],
        name="full", message="a hypothesis line\nsecond line",
        notes="n")
    promptmod.cmd_prompt_save(args)
    out = capsys.readouterr().out
    assert "labels:" in out
    assert "hypothesis:" in out
    assert "aggressiveness=7" in out


# --------------------------------------------------------------------------
# cmd_prompt_list — filter excludes a snapshot (line 85)
# --------------------------------------------------------------------------

def test_list_filter_excludes_all(fw, capsys):
    _save(fw, labels=["aggressiveness=3"], name="lowsnap")
    capsys.readouterr()
    # Filter demands aggressiveness >= 9; the only snapshot is 3 -> excluded
    # via the `continue` branch -> "(no snapshots match)".
    args = SimpleNamespace(label_filters=["aggressiveness>=9"],
                           field=None, slot=None)
    promptmod.cmd_prompt_list(args)
    assert "no snapshots match" in capsys.readouterr().out


# --------------------------------------------------------------------------
# cmd_prompt_show — ambiguous identifier ValueError (lines 113-114)
# --------------------------------------------------------------------------

def test_show_ambiguous_id_dies(fw, monkeypatch):
    def boom(framework_root, ident):
        raise ValueError(f"identifier {ident!r} is ambiguous")
    monkeypatch.setattr(reg, "find_snapshot", boom)
    with pytest.raises(SystemExit):
        promptmod.cmd_prompt_show(SimpleNamespace(prompt_id="ab", card_only=False))


# --------------------------------------------------------------------------
# cmd_prompt_diff — error gates (lines 134-135, 139, 148)
# --------------------------------------------------------------------------

def test_diff_second_missing_dies(fw, capsys):
    _save(fw, name="d1")
    pid_a = _pid_from(capsys)
    with pytest.raises(SystemExit):
        promptmod.cmd_prompt_diff(SimpleNamespace(id_a=pid_a, id_b="ghost"))


def test_diff_ambiguous_id_dies(fw, monkeypatch):
    def boom(framework_root, ident):
        raise ValueError("ambiguous")
    monkeypatch.setattr(reg, "find_snapshot", boom)
    with pytest.raises(SystemExit):
        promptmod.cmd_prompt_diff(SimpleNamespace(id_a="x", id_b="y"))


def test_diff_unusual_git_rc_propagates(fw, monkeypatch, capsys):
    _save(fw, name="g1")
    pid_a = _pid_from(capsys)
    fw.live.write_text("# Iterate\nchanged\n")
    _save(fw, name="g2")
    pid_b = _pid_from(capsys)

    # git diff returns 128 (a real error, e.g. bad object) -> sys.exit(128).
    monkeypatch.setattr(promptmod.subprocess, "call", lambda cmd: 128)
    with pytest.raises(SystemExit) as ei:
        promptmod.cmd_prompt_diff(SimpleNamespace(id_a=pid_a, id_b=pid_b))
    assert ei.value.code == 128


def test_diff_rc_zero_is_clean(fw, monkeypatch, capsys):
    _save(fw, name="s1")
    pid_a = _pid_from(capsys)
    # The registry refuses a no-op save (identical content), so mutate the
    # live file before the second snapshot.
    fw.live.write_text("# Iterate\nbody v2 for s2\n")
    _save(fw, name="s2")
    pid_b = _pid_from(capsys)
    # rc 0 (identical) is not an error -> no SystemExit.
    monkeypatch.setattr(promptmod.subprocess, "call", lambda cmd: 0)
    promptmod.cmd_prompt_diff(SimpleNamespace(id_a=pid_a, id_b=pid_b))


# --------------------------------------------------------------------------
# cmd_prompt_use — install error path (lines 159-160)
# --------------------------------------------------------------------------

def test_use_unknown_snapshot_dies(fw):
    with pytest.raises(SystemExit):
        promptmod.cmd_prompt_use(SimpleNamespace(prompt_id="nope", force=False))


# --------------------------------------------------------------------------
# cmd_prompt_annotate — error path (lines 178-179) + note-cleared (184)
# --------------------------------------------------------------------------

def test_annotate_unknown_snapshot_dies(fw):
    with pytest.raises(SystemExit):
        promptmod.cmd_prompt_annotate(
            SimpleNamespace(prompt_id="nope", labels=["k=1"], note=None))


def test_annotate_note_only_clear(fw, capsys):
    _save(fw, name="annc")
    pid = _pid_from(capsys)
    # Empty-string note clears notes -> the "(cleared)" branch (line 184).
    promptmod.cmd_prompt_annotate(
        SimpleNamespace(prompt_id=pid, labels=None, note=""))
    out = capsys.readouterr().out
    assert "annotated:" in out
    assert "(cleared)" in out
