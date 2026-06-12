"""Tests for zyme.utils — mode resolution, baseline_stash I/O, version drift,
and the small filesystem-resolution helpers (zyme_state, detect_lang,
pipeline_paths)."""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.utils import (
    DEFAULT_MODE,
    LEGACY_MODE,
    _versions_loose_equal,
    baseline_stash_path,
    detect_lang,
    has_zyme_meta,
    parse_zyme_meta,
    pipeline_paths,
    pop_baseline_stash,
    read_active_mode,
    read_baseline_stash,
    read_mode_definitions,
    read_upstream_repo_metadata,
    read_upstream_repo_sha,
    resolve_reference_output_dir,
    resolve_reference_script,
    synthesize_legacy_mode,
    upsert_baseline_stash,
    write_baseline_stash,
    zyme_state,
)


# --------------------------------------------------------------------------
# Filesystem-resolution helpers: zyme_state / detect_lang / pipeline_paths
# --------------------------------------------------------------------------

class TestZymeState:
    def test_creates_dir_if_missing(self, tmp_path: Path):
        z, best_ref, round_counter = zyme_state(tmp_path)
        assert z.is_dir()
        assert z == tmp_path / ".zyme"
        assert best_ref == tmp_path / ".zyme" / "best.ref"
        assert round_counter == tmp_path / ".zyme" / "round.counter"

    def test_idempotent(self, tmp_path: Path):
        zyme_state(tmp_path)
        z2, _, _ = zyme_state(tmp_path)
        assert z2.is_dir()


class TestDetectLang:
    def test_python(self, tmp_path: Path):
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.py").write_text("# stub\n")
        assert detect_lang(tmp_path) == "py"

    def test_r(self, tmp_path: Path):
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.R").write_text("# stub\n")
        assert detect_lang(tmp_path) == "R"

    def test_python_wins_when_both_exist(self, tmp_path: Path):
        # Document the precedence (python checked first).
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.py").write_text("")
        (tmp_path / "pipeline" / "run.R").write_text("")
        assert detect_lang(tmp_path) == "py"

    def test_neither_exits(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            detect_lang(tmp_path)


class TestPipelinePaths:
    def test_empty_when_no_files(self, tmp_path: Path):
        assert pipeline_paths(tmp_path) == []

    def test_lists_existing(self, tmp_path: Path):
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.py").write_text("")
        (tmp_path / "pipeline" / "run.R").write_text("")
        out = pipeline_paths(tmp_path)
        names = sorted(p.name for p in out)
        assert names == ["run.R", "run.py"]


# --------------------------------------------------------------------------
# .zyme_meta.yaml helpers
# --------------------------------------------------------------------------

class TestZymeMeta:
    def test_has_zyme_meta_false(self, tmp_path: Path):
        assert has_zyme_meta(tmp_path) is False

    def test_has_zyme_meta_true(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: foo\n")
        assert has_zyme_meta(tmp_path) is True

    def test_parse_returns_empty_when_absent(self, tmp_path: Path):
        assert parse_zyme_meta(tmp_path) == {}

    def test_parse_reads_flat_scalars(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text(
            "prompt_id: p_iterate_x\n"
            "framework_sha: abc1234\n"
            "suite: bench_v1\n"
            "replicate: 3\n"
        )
        out = parse_zyme_meta(tmp_path)
        assert out["prompt_id"] == "p_iterate_x"
        assert out["framework_sha"] == "abc1234"
        # registry's parser may coerce numeric-looking values; we don't
        # care here, just that the key is present.
        assert "replicate" in out


# --------------------------------------------------------------------------
# Mode resolution stack
# --------------------------------------------------------------------------

class TestSynthesizeLegacyMode:
    def test_picks_R_when_present(self, tmp_path: Path):
        (tmp_path / "reference.R").write_text("")
        out = synthesize_legacy_mode(tmp_path)
        assert out["reference_script"] == "reference.R"
        assert "pre-migration" in out["description"]

    def test_picks_py_when_R_absent(self, tmp_path: Path):
        (tmp_path / "reference.py").write_text("")
        out = synthesize_legacy_mode(tmp_path)
        assert out["reference_script"] == "reference.py"

    def test_falls_back_to_R_when_neither_exists(self, tmp_path: Path):
        # Documented "placeholder; caller will likely error" behavior.
        out = synthesize_legacy_mode(tmp_path)
        assert out["reference_script"] == "reference.R"


class TestReadModeDefinitions:
    def test_no_block_returns_synthesized_legacy(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        (tmp_path / "reference.R").write_text("")
        out = read_mode_definitions(tmp_path)
        assert list(out.keys()) == [LEGACY_MODE]
        assert out[LEGACY_MODE]["reference_script"] == "reference.R"

    def test_explicit_block(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n"
            "  default: {description: serial, reference_script: reference.R}\n"
            "  parallel_t8: {description: \"mc.cores=8\", reference_script: reference.parallel_t8.R, threads: 8}\n"
        )
        out = read_mode_definitions(tmp_path)
        assert set(out.keys()) == {"default", "parallel_t8"}
        assert out["parallel_t8"]["threads"] == "8"


class TestReadActiveMode:
    def test_default_legacy(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert read_active_mode(tmp_path) == LEGACY_MODE

    def test_explicit(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("active_mode: parallel_t8\n")
        assert read_active_mode(tmp_path) == "parallel_t8"


class TestResolveReferenceScript:
    def test_uses_active_when_mode_none(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "active_mode: default\n"
            "modes:\n  default: {description: x, reference_script: reference.R}\n"
        )
        (tmp_path / "reference.R").write_text("")
        p = resolve_reference_script(tmp_path)
        assert p == (tmp_path / "reference.R").resolve()

    def test_explicit_mode(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n"
            "  default: {description: d, reference_script: reference.R}\n"
            "  parallel_t8: {description: p, reference_script: reference.parallel_t8.R}\n"
        )
        p = resolve_reference_script(tmp_path, mode="parallel_t8")
        assert p.name == "reference.parallel_t8.R"

    def test_unknown_mode_dies(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n  default: {description: x, reference_script: reference.R}\n"
        )
        with pytest.raises(SystemExit):
            resolve_reference_script(tmp_path, mode="nonexistent")

    def test_legacy_fallback_when_no_modes_block(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        (tmp_path / "reference.R").write_text("")
        # Active mode resolves to LEGACY → synthesized legacy mode → reference.R.
        p = resolve_reference_script(tmp_path)
        assert p.name == "reference.R"


class TestResolveReferenceOutputDir:
    def test_default_layout_when_unset(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n  default: {description: x, reference_script: reference.R}\n"
        )
        out = resolve_reference_output_dir(tmp_path, "default", "tiny")
        assert out.name == "reference_output_tiny"

    def test_template_substitution(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n"
            "  parallel_t8: {description: x, reference_script: ref.R, "
            "reference_output_dir: \"reference_outputs/<mode>/<tier>\"}\n"
        )
        out = resolve_reference_output_dir(tmp_path, "parallel_t8", "medium")
        assert out.parts[-3:] == ("reference_outputs", "parallel_t8", "medium")

    def test_unknown_mode_dies(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n  default: {description: x, reference_script: reference.R}\n"
        )
        with pytest.raises(SystemExit):
            resolve_reference_output_dir(tmp_path, "no_such_mode", "tiny")


# --------------------------------------------------------------------------
# baseline_stash I/O
# --------------------------------------------------------------------------

class TestBaselineStash:
    def test_path(self, tmp_path: Path):
        assert baseline_stash_path(tmp_path) == tmp_path / ".zyme" / "baselines_stash.tsv"

    def test_read_empty_when_absent(self, tmp_path: Path):
        assert read_baseline_stash(tmp_path) == []

    def test_write_then_read_roundtrip(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        entry = {
            "tier": "tiny", "name": "tiny_a",
            "speed_sec": "10.0", "peak_mb": "512.0",
            "metrics_json": "{}", "status": "baseline",
            "thread_mode": "default",
        }
        write_baseline_stash(tmp_path, [entry])
        out = read_baseline_stash(tmp_path)
        assert len(out) == 1
        assert out[0]["tier"] == "tiny"
        assert out[0]["thread_mode"] == "default"

    def test_read_legacy_file_without_thread_mode(self, tmp_path: Path):
        # Old stash files (pre-mode) lack the thread_mode column. Reader must
        # default missing entries to LEGACY_MODE.
        (tmp_path / ".zyme").mkdir()
        stash = tmp_path / ".zyme" / "baselines_stash.tsv"
        stash.write_text(
            "tier\tname\tspeed_sec\tpeak_mb\tmetrics_json\tstatus\n"
            "tiny\ttiny_a\t10.0\t512.0\t{}\tbaseline\n"
        )
        out = read_baseline_stash(tmp_path)
        assert out[0]["thread_mode"] == LEGACY_MODE

    def test_write_empty_deletes_file(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        stash = tmp_path / ".zyme" / "baselines_stash.tsv"
        stash.write_text("tier\tname\nx\ty\n")
        write_baseline_stash(tmp_path, [])
        assert not stash.exists()

    def test_upsert_inserts_new(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        upsert_baseline_stash(tmp_path, {
            "tier": "tiny", "name": "tiny_a",
            "speed_sec": "10.0", "peak_mb": "512.0",
            "metrics_json": "{}", "status": "baseline",
            "thread_mode": "default",
        })
        assert len(read_baseline_stash(tmp_path)) == 1

    def test_upsert_replaces_same_tier_mode(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        upsert_baseline_stash(tmp_path, {
            "tier": "tiny", "name": "x", "speed_sec": "10",
            "peak_mb": "100", "metrics_json": "{}", "status": "baseline",
            "thread_mode": "default",
        })
        upsert_baseline_stash(tmp_path, {
            "tier": "tiny", "name": "x", "speed_sec": "9",
            "peak_mb": "120", "metrics_json": "{}", "status": "baseline",
            "thread_mode": "default",
        })
        out = read_baseline_stash(tmp_path)
        assert len(out) == 1
        assert out[0]["speed_sec"] == "9"

    def test_upsert_distinct_modes_coexist(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        for mode in ("default", "parallel_t8"):
            upsert_baseline_stash(tmp_path, {
                "tier": "tiny", "name": "x", "speed_sec": "10",
                "peak_mb": "100", "metrics_json": "{}", "status": "baseline",
                "thread_mode": mode,
            })
        out = read_baseline_stash(tmp_path)
        assert {e["thread_mode"] for e in out} == {"default", "parallel_t8"}

    def test_pop_returns_match_and_removes(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        (tmp_path / "task.yaml").write_text("active_mode: default\n")
        upsert_baseline_stash(tmp_path, {
            "tier": "tiny", "name": "x", "speed_sec": "10",
            "peak_mb": "100", "metrics_json": "{}", "status": "baseline",
            "thread_mode": "default",
        })
        popped = pop_baseline_stash(tmp_path, "tiny", mode="default")
        assert popped is not None
        assert popped["tier"] == "tiny"
        assert read_baseline_stash(tmp_path) == []

    def test_pop_returns_none_no_match(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert pop_baseline_stash(tmp_path, "tiny") is None

    def test_pop_uses_active_mode_when_none(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        (tmp_path / "task.yaml").write_text("active_mode: parallel_t8\n")
        upsert_baseline_stash(tmp_path, {
            "tier": "tiny", "name": "x", "speed_sec": "10",
            "peak_mb": "100", "metrics_json": "{}", "status": "baseline",
            "thread_mode": "default",
        })
        # Active mode is parallel_t8; the only entry is default → no match.
        assert pop_baseline_stash(tmp_path, "tiny") is None


# --------------------------------------------------------------------------
# upstream_repo metadata + version drift
# --------------------------------------------------------------------------

class TestReadUpstreamRepoMetadata:
    def test_no_upstream_returns_none(self, tmp_path: Path):
        assert read_upstream_repo_metadata(tmp_path) is None

    def test_r_description_file(self, tmp_path: Path):
        upstream = tmp_path / "upstream_repo"
        upstream.mkdir()
        (upstream / "DESCRIPTION").write_text(
            "Package: spacexr\nVersion: 2.2.1\nTitle: stub\n"
        )
        out = read_upstream_repo_metadata(tmp_path)
        assert out["package"] == "spacexr"
        assert out["version"] == "2.2.1"
        assert out["source"] == "DESCRIPTION"

    def test_python_pyproject(self, tmp_path: Path):
        upstream = tmp_path / "upstream_repo"
        upstream.mkdir()
        (upstream / "pyproject.toml").write_text(
            "[project]\nname = \"foo\"\nversion = \"1.2.3\"\n"
        )
        out = read_upstream_repo_metadata(tmp_path)
        assert out["package"] == "foo"
        assert out["version"] == "1.2.3"
        assert out["source"] == "pyproject.toml"

    def test_python_setup_py(self, tmp_path: Path):
        upstream = tmp_path / "upstream_repo"
        upstream.mkdir()
        (upstream / "setup.py").write_text(
            "from setuptools import setup\nsetup(name='bar', version='0.1.0')\n"
        )
        out = read_upstream_repo_metadata(tmp_path)
        assert out["package"] == "bar"
        assert out["version"] == "0.1.0"

    def test_dynamic_version_skipped(self, tmp_path: Path):
        # setup.cfg with version=attr:foo.__version__ should not be returned
        # as a literal — that path is a dynamic specifier, not a string.
        upstream = tmp_path / "upstream_repo"
        upstream.mkdir()
        (upstream / "setup.cfg").write_text(
            "[metadata]\nname = foo\nversion = attr:foo.__version__\n"
        )
        assert read_upstream_repo_metadata(tmp_path) is None


class TestReadUpstreamRepoSha:
    def test_no_upstream(self, tmp_path: Path):
        assert read_upstream_repo_sha(tmp_path) is None

    def test_not_a_git_repo(self, tmp_path: Path):
        (tmp_path / "upstream_repo").mkdir()
        # No .git/ inside.
        assert read_upstream_repo_sha(tmp_path) is None


class TestVersionsLooseEqual:
    def test_exact_match(self):
        assert _versions_loose_equal("1.4.0", "1.4.0") is True

    def test_dash_dot_aliasing(self):
        # R uses 1.4.0-3, Python sees 1.4.0.3 — they're the same release.
        assert _versions_loose_equal("1.4.0-3", "1.4.0.3") is True

    def test_pre_release_suffix_treated_as_extra_segment(self):
        # The function uses re.findall(r"\d+") which captures EVERY digit run,
        # including the "1" inside "a1" — so "1.0.0a1" → [1,0,0,1] vs
        # "1.0.0" → [1,0,0]. Different segment counts ⇒ not equal.
        # (The docstring's "ignores pre-release suffixes" claim is aspirational;
        # this test pins down the actual behavior.)
        assert _versions_loose_equal("1.0.0a1", "1.0.0") is False

    def test_matching_pre_release_extracts_match(self):
        # Two versions with the SAME pre-release suffix do match.
        assert _versions_loose_equal("1.0.0a1", "1.0.0a1") is True

    def test_genuine_inequality(self):
        assert _versions_loose_equal("1.4.0", "1.5.0") is False

    def test_empty_strings(self):
        assert _versions_loose_equal("", "") is True

    def test_one_empty(self):
        # Loose equality for "" vs "0" — both have no nonempty digit segments,
        # so segment comparison returns False (per `bool(seg_a)` guard).
        assert _versions_loose_equal("", "0") is False
