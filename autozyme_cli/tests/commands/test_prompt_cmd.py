"""Unit tests for zyme.commands.prompt — the `zyme prompt` snapshot CLI wrappers.

This is the COMMAND module (commands/prompt.py), which wraps zyme.registry (the
prompt-snapshot store). The registry internals are covered by
tests/test_registry_unit.py; here we cover the label-kv parser and drive each
cmd_prompt_* wrapper over a fake framework tree built under tmp_path so
real save/install/list/show/annotate I/O runs (no mocking of registry).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import prompt as promptmod
from zyme import registry as reg


# --------------------------------------------------------------------------
# _parse_label_kvs (pure)
# --------------------------------------------------------------------------

class TestParseLabelKvs:
    def test_numeric_coerced(self):
        out = promptmod._parse_label_kvs(["aggressiveness=10", "rerun=2.5"])
        assert out == {"aggressiveness": 10.0, "rerun": 2.5}

    def test_string_kept(self):
        out = promptmod._parse_label_kvs(["name=foo"])
        assert out == {"name": "foo"}

    def test_strips_whitespace(self):
        out = promptmod._parse_label_kvs([" k = v "])
        assert out == {"k": "v"}

    def test_none_returns_empty(self):
        assert promptmod._parse_label_kvs(None) == {}

    def test_missing_eq_dies(self):
        with pytest.raises(SystemExit):
            promptmod._parse_label_kvs(["noequals"])

    def test_empty_key_dies(self):
        with pytest.raises(SystemExit):
            promptmod._parse_label_kvs(["=v"])


# --------------------------------------------------------------------------
# Fake framework tree so registry resolves to the legacy framework-local root.
# Mirrors the fixture used by tests/test_registry_unit.py.
# --------------------------------------------------------------------------

@pytest.fixture
def fw(tmp_path, monkeypatch):
    root = tmp_path / "ws" / "autozyme-framework" / "autozyme_cli" / "zyme"
    bio = root / "prompts" / "Bio"
    bio.mkdir(parents=True)
    (root / "prompts" / "OtherField").mkdir(parents=True)
    # A live phase prompt the registry can snapshot.
    live = bio / "2_iterate.md"
    live.write_text("# Iterate\nbody v1\n")
    monkeypatch.setattr(promptmod, "FRAMEWORK_ROOT", root)
    return SimpleNamespace(root=root, live=live, bio=bio)


# --------------------------------------------------------------------------
# cmd_prompt_save + list + show + annotate + use
# --------------------------------------------------------------------------

class TestPromptLifecycle:
    def test_save_then_list_show(self, fw, capsys):
        save_args = SimpleNamespace(
            live_path=str(fw.live), labels=["aggressiveness=8"],
            name="v1snap", message="first snapshot hypothesis",
            notes="some notes")
        promptmod.cmd_prompt_save(save_args)
        out = capsys.readouterr().out
        assert "saved:" in out
        assert "active.lock now" in out

        # list shows the snapshot with active marker
        list_args = SimpleNamespace(label_filters=None, field=None, slot=None)
        promptmod.cmd_prompt_list(list_args)
        out = capsys.readouterr().out
        assert "v1snap" in out
        assert "snapshot(s)" in out

    def test_list_no_match(self, fw, capsys):
        list_args = SimpleNamespace(label_filters=None, field=None, slot=None)
        promptmod.cmd_prompt_list(list_args)
        out = capsys.readouterr().out
        assert "no snapshots match" in out

    def test_list_label_filter(self, fw, capsys):
        promptmod.cmd_prompt_save(SimpleNamespace(
            live_path=str(fw.live), labels=["aggressiveness=8"],
            name="snapA", message="", notes=""))
        capsys.readouterr()
        # filter on a label that matches
        list_args = SimpleNamespace(label_filters=["aggressiveness>=5"],
                                    field=None, slot=None)
        promptmod.cmd_prompt_list(list_args)
        out = capsys.readouterr().out
        assert "snapA" in out

    def test_list_bad_filter_dies(self, fw):
        list_args = SimpleNamespace(label_filters=["totally bogus filter !!"],
                                    field=None, slot=None)
        with pytest.raises(SystemExit):
            promptmod.cmd_prompt_list(list_args)

    def test_show(self, fw, capsys):
        promptmod.cmd_prompt_save(SimpleNamespace(
            live_path=str(fw.live), labels=None, name="snapShow",
            message="hyp", notes=""))
        out = capsys.readouterr().out
        # extract the prompt id from "saved: <id>"
        pid = next(l for l in out.splitlines() if "saved:" in l).split("saved:")[1].strip()
        show_args = SimpleNamespace(prompt_id=pid, card_only=False)
        promptmod.cmd_prompt_show(show_args)
        out = capsys.readouterr().out
        assert "card.yaml" in out
        assert "prompt.md" in out

    def test_show_card_only(self, fw, capsys):
        promptmod.cmd_prompt_save(SimpleNamespace(
            live_path=str(fw.live), labels=None, name="snapCard",
            message="", notes=""))
        out = capsys.readouterr().out
        pid = next(l for l in out.splitlines() if "saved:" in l).split("saved:")[1].strip()
        promptmod.cmd_prompt_show(SimpleNamespace(prompt_id=pid, card_only=True))
        out = capsys.readouterr().out
        assert "card.yaml" in out
        assert "prompt.md" not in out

    def test_show_not_found_dies(self, fw):
        with pytest.raises(SystemExit):
            promptmod.cmd_prompt_show(
                SimpleNamespace(prompt_id="does-not-exist", card_only=False))

    def test_annotate(self, fw, capsys):
        promptmod.cmd_prompt_save(SimpleNamespace(
            live_path=str(fw.live), labels=None, name="snapAnn",
            message="", notes=""))
        out = capsys.readouterr().out
        pid = next(l for l in out.splitlines() if "saved:" in l).split("saved:")[1].strip()
        promptmod.cmd_prompt_annotate(SimpleNamespace(
            prompt_id=pid, labels=["thread_breadth=4"], note="reviewed"))
        out = capsys.readouterr().out
        assert "annotated:" in out
        assert "thread_breadth" in out

    def test_annotate_nothing_to_do_dies(self, fw):
        with pytest.raises(SystemExit):
            promptmod.cmd_prompt_annotate(SimpleNamespace(
                prompt_id="x", labels=None, note=None))

    def test_use_reinstalls(self, fw, capsys):
        promptmod.cmd_prompt_save(SimpleNamespace(
            live_path=str(fw.live), labels=None, name="snapUse",
            message="", notes=""))
        out = capsys.readouterr().out
        pid = next(l for l in out.splitlines() if "saved:" in l).split("saved:")[1].strip()
        # Mutate the live file, then `use` to restore the snapshot.
        fw.live.write_text("# changed\nbody v2\n")
        promptmod.cmd_prompt_use(SimpleNamespace(prompt_id=pid, force=True))
        out = capsys.readouterr().out
        assert "installed:" in out
        assert "active.lock now" in out
        # live file restored to the snapshot content
        assert "body v1" in fw.live.read_text()


# --------------------------------------------------------------------------
# cmd_prompt_diff (git subprocess boundary stubbed)
# --------------------------------------------------------------------------

class TestPromptDiff:
    def test_diff_invokes_git(self, fw, monkeypatch, capsys):
        # Two snapshots to diff.
        promptmod.cmd_prompt_save(SimpleNamespace(
            live_path=str(fw.live), labels=None, name="d1", message="", notes=""))
        out = capsys.readouterr().out
        pid_a = next(l for l in out.splitlines() if "saved:" in l).split("saved:")[1].strip()
        fw.live.write_text("# Iterate\nbody v2 different\n")
        promptmod.cmd_prompt_save(SimpleNamespace(
            live_path=str(fw.live), labels=None, name="d2", message="", notes=""))
        out = capsys.readouterr().out
        pid_b = next(l for l in out.splitlines() if "saved:" in l).split("saved:")[1].strip()

        captured = {}

        def fake_call(cmd):
            captured["cmd"] = cmd
            return 1  # git diff returns 1 when files differ — not an error
        monkeypatch.setattr(promptmod.subprocess, "call", fake_call)
        promptmod.cmd_prompt_diff(SimpleNamespace(id_a=pid_a, id_b=pid_b))
        assert captured["cmd"][0] == "git"
        assert "diff" in captured["cmd"]

    def test_diff_missing_snapshot_dies(self, fw):
        with pytest.raises(SystemExit):
            promptmod.cmd_prompt_diff(SimpleNamespace(id_a="ghost1", id_b="ghost2"))
