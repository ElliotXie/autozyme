"""Unit tests for zyme.registry — versioned prompt snapshots (pure file I/O).

Covers: path helpers, slot/field derivation, slugify/make_prompt_id, the
mini-YAML round-trip (write_card_yaml / read_card_yaml), active.lock + index
I/O, listing/lookup (iter_snapshots / find_snapshot incl. ambiguity), the
save/install/annotate lifecycle, and the label-filter parser.

A fake framework tree is built under tmp_path so registry_root() resolves to
the legacy framework-local path (no PromptLab sibling exists in tmp).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme import registry as reg
from zyme.registry import (
    RegistryError,
    _legacy_registry_root,
    _parse_scalar,
    _yaml_scalar,
    annotate_snapshot,
    append_to_index,
    field_from_path,
    find_snapshot,
    install_snapshot,
    iter_snapshots,
    label_matches,
    make_prompt_id,
    parse_label_filter,
    read_active_lock,
    read_card_yaml,
    read_index,
    registry_root,
    save_snapshot,
    sha256_of,
    slot_from_filename,
    slugify,
    write_active_lock,
    write_card_yaml,
)


# --------------------------------------------------------------------------
# framework-tree fixture
# --------------------------------------------------------------------------
@pytest.fixture
def fw(tmp_path: Path) -> Path:
    """A fake framework_root = .../autozyme-framework/autozyme_cli/zyme/.

    registry_root() walks 3 parents up looking for a PromptLab sibling; under
    tmp_path none exists, so it falls back to the legacy framework-local root.
    """
    root = tmp_path / "ws" / "autozyme-framework" / "autozyme_cli" / "zyme"
    (root / "prompts" / "Bio").mkdir(parents=True)
    (root / "prompts" / "OtherField").mkdir(parents=True)
    return root


# --------------------------------------------------------------------------
# path helpers
# --------------------------------------------------------------------------
class TestPaths:
    def test_registry_root_falls_back_to_legacy(self, fw: Path):
        assert registry_root(fw) == _legacy_registry_root(fw)
        assert registry_root(fw) == fw / "prompts" / "_registry"

    def test_registry_root_prefers_promptlab_when_present(self, fw: Path):
        # framework_root.parent.parent.parent is .../ws ; create PromptLab there.
        promptlab = fw.parent.parent.parent / "PromptLab"
        promptlab.mkdir()
        assert registry_root(fw) == promptlab / "prompt_snapshots"

    def test_slot_dir_and_children(self, fw: Path):
        sd = reg.slot_dir(fw, "Bio", "iterate")
        assert sd == fw / "prompts" / "_registry" / "Bio" / "iterate"
        assert reg.snapshots_dir(fw, "Bio", "iterate") == sd / "snapshots"
        assert reg.index_path(fw, "Bio", "iterate") == sd / "index.yaml"
        assert reg.active_lock_path(fw, "Bio", "iterate") == sd / "active.lock"

    def test_live_prompts_dir(self, fw: Path):
        assert reg.live_prompts_dir(fw, "Bio") == fw / "prompts" / "Bio"


# --------------------------------------------------------------------------
# slot / field derivation
# --------------------------------------------------------------------------
class TestSlotFromFilename:
    @pytest.mark.parametrize("fn,slot", [
        ("2_iterate.md", "iterate"),
        ("1_init.md", "init"),
        ("6.1_transfer_init.md", "transfer_init"),
        ("M_thread_baseline_fairness.md", "thread_baseline_fairness"),
    ])
    def test_basic(self, fn, slot):
        assert slot_from_filename(fn) == slot

    def test_6_2_iterate_becomes_transfer_iterate(self):
        # special-cased so it doesn't collide with 2_iterate.md's "iterate" slot
        assert slot_from_filename("6.2_iterate.md") == "transfer_iterate"

    def test_6_2_non_iterate_unchanged(self):
        assert slot_from_filename("6.2_scale.md") == "scale"

    @pytest.mark.parametrize("bad", ["iterate.md", "noextension", "2-iterate.md", ""])
    def test_bad_filenames_raise(self, bad):
        with pytest.raises(ValueError):
            slot_from_filename(bad)


class TestFieldFromPath:
    def test_extracts_field(self, fw: Path):
        p = fw / "prompts" / "Bio" / "2_iterate.md"
        p.write_text("x")
        assert field_from_path(p, fw) == "Bio"

    def test_otherfield(self, fw: Path):
        p = fw / "prompts" / "OtherField" / "2_iterate.md"
        p.write_text("x")
        assert field_from_path(p, fw) == "OtherField"

    def test_outside_prompts_raises(self, fw: Path, tmp_path: Path):
        outside = tmp_path / "elsewhere" / "2_iterate.md"
        outside.parent.mkdir(parents=True)
        outside.write_text("x")
        with pytest.raises(ValueError):
            field_from_path(outside, fw)

    def test_registry_path_rejected(self, fw: Path):
        p = fw / "prompts" / "_registry" / "Bio" / "x.md"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        with pytest.raises(ValueError):
            field_from_path(p, fw)

    def test_directly_under_prompts_rejected(self, fw: Path):
        # prompts/2_iterate.md has no <field> level
        p = fw / "prompts" / "2_iterate.md"
        p.write_text("x")
        with pytest.raises(ValueError):
            field_from_path(p, fw)


# --------------------------------------------------------------------------
# IDs / slugify
# --------------------------------------------------------------------------
class TestIds:
    def test_sha256_stable_and_sensitive(self):
        a = sha256_of("hello")
        assert a == sha256_of("hello")
        assert a != sha256_of("hello ")
        assert len(a) == 64

    @pytest.mark.parametrize("name,want", [
        ("My Prompt!", "my_prompt"),
        ("a---b", "a_b"),
        ("  spaced  ", "spaced"),
        ("UPPER", "upper"),
        ("", "unnamed"),
        ("!!!", "unnamed"),
    ])
    def test_slugify(self, name, want):
        assert slugify(name) == want

    def test_slugify_truncates_to_40(self):
        s = slugify("a" * 100)
        assert len(s) == 40

    def test_make_prompt_id_shape(self):
        pid = make_prompt_id("iterate", "Test Run", "abcdef0123456789")
        parts = pid.split("_")
        assert parts[0] == "zyme"
        assert parts[1] == "iterate"
        assert len(parts[2]) == 8 and parts[2].isdigit()  # YYYYMMDD
        assert pid.endswith("abcdef01")  # first 8 of sha
        assert "test_run" in pid


# --------------------------------------------------------------------------
# mini-YAML scalar round-trip
# --------------------------------------------------------------------------
class TestYamlScalar:
    @pytest.mark.parametrize("val,emitted", [
        (None, "null"),
        (True, "true"),
        (False, "false"),
        (42, "42"),
        (3.5, "3.5"),
        ("plain", "plain"),
        ("", '""'),
    ])
    def test_emit(self, val, emitted):
        assert _yaml_scalar(val) == emitted

    def test_quotes_special_chars(self):
        assert _yaml_scalar("a: b") == '"a: b"'
        assert _yaml_scalar("has#hash") == '"has#hash"'

    def test_quotes_yaml_keywords(self):
        assert _yaml_scalar("true") == '"true"'
        assert _yaml_scalar("yes") == '"yes"'
        assert _yaml_scalar("null") == '"null"'

    def test_escapes_backslash_and_quote(self):
        # A bare double-quote alone does NOT trigger quoting (`"` isn't in
        # _NEEDS_QUOTE); it must co-occur with a needs-quote char like ':'.
        # When quoting fires, embedded quotes + backslashes are escaped.
        assert _yaml_scalar('a:"b') == '"a:\\"b"'
        # bare quote, no other special char -> passes through unquoted
        assert _yaml_scalar('hi"there') == 'hi"there'

    def test_roundtrip_string_with_quote_and_colon(self):
        v = 'a:"b'
        assert _parse_scalar(_yaml_scalar(v)) == v

    @pytest.mark.parametrize("s,want", [
        ("plain", "plain"),
        ("", ""),
        ('""', ""),
        ("null", None),
        ("true", True),
        ("false", False),
        ("42", 42),
        ("3.5", 3.5),
        ("1e3", 1000.0),
    ])
    def test_parse_scalar(self, s, want):
        assert _parse_scalar(s) == want

    def test_parse_scalar_unparseable_stays_string(self):
        assert _parse_scalar("not-a-number") == "not-a-number"

    def test_emit_then_parse_roundtrips_strings(self):
        for v in ["hello world", "a: b", 'with"quote', "trailing#"]:
            assert _parse_scalar(_yaml_scalar(v)) == v


# --------------------------------------------------------------------------
# card.yaml write + read round-trip
# --------------------------------------------------------------------------
class TestCardRoundTrip:
    def _card(self):
        return {
            "id": "zyme_iterate_20260101_x_deadbeef",
            "slot": "iterate",
            "field": "Bio",
            "source_path": "prompts/Bio/2_iterate.md",
            "parent": None,
            "content_sha256": "abc123",
            "name": "experiment one",
            "created_at": "2026-01-01T00:00:00",
            "hypothesis": "line1\nline2",
            "labels": {"aggressiveness": 8, "tag": "fast"},
            "notes": "some notes",
        }

    def test_roundtrip(self, tmp_path: Path):
        write_card_yaml(tmp_path, self._card())
        out = read_card_yaml(tmp_path / "card.yaml")
        c = self._card()
        for k in ("id", "slot", "field", "source_path", "content_sha256", "name"):
            assert out[k] == c[k]
        assert out["parent"] is None
        assert out["hypothesis"] == "line1\nline2"
        assert out["notes"] == "some notes"
        assert out["labels"] == {"aggressiveness": 8, "tag": "fast"}

    def test_empty_labels_and_blocks(self, tmp_path: Path):
        card = self._card()
        card["labels"] = {}
        card["hypothesis"] = ""
        card["notes"] = ""
        write_card_yaml(tmp_path, card)
        out = read_card_yaml(tmp_path / "card.yaml")
        assert out["labels"] == {}
        assert out["hypothesis"] == ""
        assert out["notes"] == ""

    def test_missing_keys_emit_null(self, tmp_path: Path):
        write_card_yaml(tmp_path, {"id": "x"})
        text = (tmp_path / "card.yaml").read_text()
        assert "slot: null" in text
        out = read_card_yaml(tmp_path / "card.yaml")
        assert out["id"] == "x"
        assert out["slot"] is None

    def test_read_ignores_comments_and_blanks(self, tmp_path: Path):
        (tmp_path / "card.yaml").write_text(
            "# header comment\n\nid: z\n  \nname: foo\n")
        out = read_card_yaml(tmp_path / "card.yaml")
        assert out == {"id": "z", "name": "foo"}

    def test_labels_with_numeric_and_bool(self, tmp_path: Path):
        card = self._card()
        card["labels"] = {"n": 5, "flag": True, "ratio": 2.5}
        write_card_yaml(tmp_path, card)
        out = read_card_yaml(tmp_path / "card.yaml")
        assert out["labels"] == {"n": 5, "flag": True, "ratio": 2.5}


# --------------------------------------------------------------------------
# active.lock + index I/O
# --------------------------------------------------------------------------
class TestLockAndIndex:
    def test_active_lock_roundtrip(self, fw: Path):
        assert read_active_lock(fw, "Bio", "iterate") is None
        write_active_lock(fw, "Bio", "iterate", "zyme_iterate_x")
        assert read_active_lock(fw, "Bio", "iterate") == "zyme_iterate_x"

    def test_empty_lock_reads_none(self, fw: Path):
        p = reg.active_lock_path(fw, "Bio", "iterate")
        p.parent.mkdir(parents=True)
        p.write_text("   \n")
        assert read_active_lock(fw, "Bio", "iterate") is None

    def test_index_append_and_read(self, fw: Path):
        assert read_index(fw, "Bio", "iterate") == []
        append_to_index(fw, "Bio", "iterate", "id_a")
        append_to_index(fw, "Bio", "iterate", "id_b")
        assert read_index(fw, "Bio", "iterate") == ["id_a", "id_b"]


# --------------------------------------------------------------------------
# save / list / lookup / install / annotate lifecycle
# --------------------------------------------------------------------------
class TestLifecycle:
    def _live(self, fw: Path, content="initial prompt body\n"):
        p = fw / "prompts" / "Bio" / "2_iterate.md"
        p.write_text(content)
        return p

    def test_save_creates_snapshot_and_artifacts(self, fw: Path):
        live = self._live(fw)
        card = save_snapshot(fw, live, name="exp1", hypothesis="h", labels={"a": 8})
        assert card["slot"] == "iterate"
        assert card["field"] == "Bio"
        assert card["name"] == "exp1"
        assert card["content_sha256"] == sha256_of(live.read_text())
        # active.lock + index updated
        assert read_active_lock(fw, "Bio", "iterate") == card["id"]
        assert card["id"] in read_index(fw, "Bio", "iterate")
        # prompt.md persisted
        snap = reg.snapshot_dir_for(fw, "Bio", "iterate", card["id"])
        assert (snap / "prompt.md").read_text() == live.read_text()

    def test_save_missing_live_raises(self, fw: Path):
        with pytest.raises(RegistryError):
            save_snapshot(fw, fw / "prompts" / "Bio" / "absent.md", "n", "", {})

    def test_save_identical_content_rejected(self, fw: Path):
        live = self._live(fw)
        save_snapshot(fw, live, name="exp1", hypothesis="", labels={})
        # second save with no edit -> identical sha to active -> rejected
        with pytest.raises(RegistryError, match="identical"):
            save_snapshot(fw, live, name="exp2", hypothesis="", labels={})

    def test_iter_and_find_snapshot(self, fw: Path):
        live = self._live(fw)
        card = save_snapshot(fw, live, name="findme", hypothesis="", labels={})
        rows = list(iter_snapshots(fw))
        assert len(rows) == 1
        f, s, pid, c = rows[0]
        assert (f, s, pid) == ("Bio", "iterate", card["id"])
        # find by exact id, exact name, and unique substring
        assert find_snapshot(fw, card["id"])[2] == card["id"]
        assert find_snapshot(fw, "findme")[2] == card["id"]
        assert find_snapshot(fw, "findm")[2] == card["id"]
        assert find_snapshot(fw, "does-not-exist") is None

    def test_find_snapshot_ambiguous_substring_raises(self, fw: Path):
        # two snapshots whose names share a substring -> ambiguous lookup
        live = self._live(fw, "body one\n")
        save_snapshot(fw, live, name="alpha_run", hypothesis="", labels={})
        live.write_text("body two\n")
        save_snapshot(fw, live, name="alpha_test", hypothesis="", labels={})
        with pytest.raises(ValueError, match="ambiguous"):
            find_snapshot(fw, "alpha")

    def test_install_restores_live_content(self, fw: Path):
        live = self._live(fw, "original\n")
        card = save_snapshot(fw, live, name="v1", hypothesis="", labels={})
        live.write_text("garbage local edit\n")
        # install with force to bypass unsaved-change guard
        install_snapshot(fw, card["id"], force=True)
        assert live.read_text() == "original\n"
        assert read_active_lock(fw, "Bio", "iterate") == card["id"]

    def test_install_unknown_raises(self, fw: Path):
        with pytest.raises(RegistryError):
            install_snapshot(fw, "nope")

    def test_install_unsaved_changes_guard(self, fw: Path):
        live = self._live(fw, "v1 body\n")
        c1 = save_snapshot(fw, live, name="v1", hypothesis="", labels={})
        live.write_text("v2 body\n")
        c2 = save_snapshot(fw, live, name="v2", hypothesis="", labels={})
        # now live == c2's content (active). Hand-edit live without saving:
        live.write_text("unsaved hand edit\n")
        # installing c1 without force must refuse (live has unsaved changes
        # vs the active snapshot c2)
        with pytest.raises(RegistryError, match="unsaved changes"):
            install_snapshot(fw, c1["id"])
        # with force it goes through
        install_snapshot(fw, c1["id"], force=True)
        assert live.read_text() == "v1 body\n"
        assert c2["id"] != c1["id"]

    def test_annotate_merges_labels_replaces_notes(self, fw: Path):
        live = self._live(fw)
        card = save_snapshot(fw, live, name="ann", hypothesis="",
                             labels={"a": 1, "b": 2})
        updated = annotate_snapshot(fw, card["id"],
                                    labels={"b": 9, "c": 3}, notes="new notes")
        assert updated["labels"] == {"a": 1, "b": 9, "c": 3}
        assert updated["notes"] == "new notes"
        # persisted on disk
        reread = read_card_yaml(
            reg.snapshot_dir_for(fw, "Bio", "iterate", card["id"]) / "card.yaml")
        assert reread["labels"] == {"a": 1, "b": 9, "c": 3}
        assert reread["notes"] == "new notes"

    def test_annotate_unknown_raises(self, fw: Path):
        with pytest.raises(RegistryError):
            annotate_snapshot(fw, "missing", labels={"x": 1})


# --------------------------------------------------------------------------
# label-filter parsing + matching
# --------------------------------------------------------------------------
class TestLabelFilters:
    @pytest.mark.parametrize("expr,key,op,val", [
        ("aggressiveness>=8", "aggressiveness", ">=", 8.0),
        ("a<=3", "a", "<=", 3.0),
        ("a>1", "a", ">", 1.0),
        ("a<2", "a", "<", 2.0),
        ("a=5", "a", "=", 5.0),
        ("a==5", "a", "=", 5.0),       # == normalized to =
        ("tag=fast", "tag", "=", "fast"),
    ])
    def test_parse(self, expr, key, op, val):
        assert parse_label_filter(expr) == (key, op, val)

    def test_parse_strips_whitespace(self):
        assert parse_label_filter("  a >= 8 ") == ("a", ">=", 8.0)

    def test_parse_unparseable_raises(self):
        with pytest.raises(ValueError):
            parse_label_filter("nooperator")

    def test_parse_nonnumeric_with_inequality_raises(self):
        with pytest.raises(ValueError, match="numeric RHS"):
            parse_label_filter("tag>=fast")

    @pytest.mark.parametrize("labels,key,op,want,expect", [
        ({"a": 8}, "a", ">=", 8, True),
        ({"a": 8}, "a", ">=", 9, False),
        ({"a": 8}, "a", ">", 7, True),
        ({"a": 8}, "a", "<=", 8, True),
        ({"a": 8}, "a", "<", 8, False),
        ({"a": 5}, "a", "=", 5, True),
        ({"tag": "fast"}, "tag", "=", "fast", True),
        ({"tag": "fast"}, "tag", "=", "slow", False),
    ])
    def test_match(self, labels, key, op, want, expect):
        assert label_matches(labels, key, op, want) is expect

    def test_match_missing_key_false(self):
        assert label_matches({"a": 1}, "b", "=", 1) is False

    def test_match_none_labels_false(self):
        assert label_matches(None, "a", "=", 1) is False

    def test_match_nonnumeric_value_inequality_false(self):
        # have="x" cannot be coerced to float for a > comparison
        assert label_matches({"a": "x"}, "a", ">", 1) is False

    def test_match_numeric_string_equality(self):
        # `have` is a str -> str-compared on both sides: str(5) == "5" -> True
        assert label_matches({"a": "5"}, "a", "=", 5) is True
        # mismatched string value
        assert label_matches({"a": "6"}, "a", "=", 5) is False
        # both numeric -> float-compared
        assert label_matches({"a": 5}, "a", "=", 5) is True
