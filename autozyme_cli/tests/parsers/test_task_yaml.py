"""Tests for zyme.parsers.task_yaml."""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.parsers.task_yaml import (
    _coerce_scalar,
    _parse_inline_value,
    _split_top_level_commas,
    effective_threshold,
    parse_active_mode,
    parse_algorithm_class,
    parse_dataset_name,
    parse_datasets,
    parse_executor,
    parse_intrinsic_noise,
    parse_metrics,
    parse_modes_block,
    parse_random_seeds,
    parse_scaling_tax_thresholds,
    parse_threading_mode,
    remove_mode_entry,
    resolve_tiers,
    write_active_mode,
    write_intrinsic_noise,
    write_mode_entry,
)
from zyme.utils import LEGACY_MODE


# --------------------------------------------------------------------------
# Private inline-yaml helpers
# --------------------------------------------------------------------------

class TestSplitTopLevelCommas:
    def test_simple(self):
        assert _split_top_level_commas("a, b, c") == ["a", " b", " c"]

    def test_nested_brace_protected(self):
        # Comma inside {} stays attached to its parent kv.
        assert _split_top_level_commas("a: 1, b: {x: 1, y: 2}, c: 3") == [
            "a: 1", " b: {x: 1, y: 2}", " c: 3",
        ]

    def test_empty(self):
        assert _split_top_level_commas("") == []


class TestCoerceScalar:
    def test_int(self):
        assert _coerce_scalar("42") == 42

    def test_float(self):
        assert _coerce_scalar("1.5e-3") == 1.5e-3

    def test_string(self):
        assert _coerce_scalar("hello") == "hello"

    def test_quoted_strips(self):
        assert _coerce_scalar('"quoted"') == "quoted"
        assert _coerce_scalar("'sq'") == "sq"


class TestParseInlineValue:
    def test_scalar(self):
        assert _parse_inline_value("42") == 42

    def test_nested_dict(self):
        assert _parse_inline_value("{x: 1, y: 2}") == {"x": 1, "y": 2}

    def test_deeply_nested(self):
        assert _parse_inline_value("{a: {b: {c: 1}}}") == {"a": {"b": {"c": 1}}}


# --------------------------------------------------------------------------
# parse_datasets / parse_dataset_name
# --------------------------------------------------------------------------

class TestParseDatasets:
    def test_missing_file(self, tmp_path: Path):
        assert parse_datasets(tmp_path / "noexist.yaml") == []

    def test_no_datasets_block(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_datasets(tmp_path / "task.yaml") == []

    def test_single_inline_entry(self, tmp_path: Path):
        abs_path = tmp_path / "abs" / "foo.h5"
        (tmp_path / "task.yaml").write_text(
            f"datasets:\n  - {{tier: tiny, name: foo, path: {abs_path}}}\n"
        )
        out = parse_datasets(tmp_path / "task.yaml")
        assert len(out) == 1
        assert out[0]["tier"] == "tiny"
        assert out[0]["name"] == "foo"
        assert out[0]["path"] == str(abs_path)
        assert out[0]["params"] == {}

    def test_relative_path_resolved_against_task_dir(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: foo, path: data/foo.h5}\n"
        )
        out = parse_datasets(tmp_path / "task.yaml")
        # Resolved to absolute path relative to task_yaml.parent.
        resolved = Path(out[0]["path"])
        assert resolved.is_absolute()
        assert resolved.parts[-2:] == ("data", "foo.h5")

    def test_default_tier_tiny(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n  - {name: foo, path: /abs/foo.h5}\n"
        )
        out = parse_datasets(tmp_path / "task.yaml")
        assert out[0]["tier"] == "tiny"  # legacy default

    def test_with_nested_params(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: medium, name: m, path: /abs/m.h5, params: {n_cells: 1000, n_genes: 500}}\n"
        )
        out = parse_datasets(tmp_path / "task.yaml")
        assert out[0]["params"] == {"n_cells": 1000, "n_genes": 500}

    def test_multiple_entries(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
            "  - {tier: medium, name: m, path: /abs/m.h5}\n"
        )
        out = parse_datasets(tmp_path / "task.yaml")
        assert [e["tier"] for e in out] == ["tiny", "medium"]

    def test_skips_comments_and_blanks(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  # this is a comment\n"
            "\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
        )
        assert len(parse_datasets(tmp_path / "task.yaml")) == 1

    def test_block_ends_at_top_level_key(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
            "metrics:\n"
            "  - {name: speedup, comparator: gte, threshold: 1}\n"
        )
        out = parse_datasets(tmp_path / "task.yaml")
        assert len(out) == 1


class TestParseDatasetName:
    def test_first_entry(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: foo, path: /abs/foo.h5}\n"
        )
        assert parse_dataset_name(tmp_path / "task.yaml") == "foo"

    def test_empty_when_no_datasets(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_dataset_name(tmp_path / "task.yaml") == ""


# --------------------------------------------------------------------------
# parse_metrics
# --------------------------------------------------------------------------

class TestParseMetrics:
    def test_deterministic(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}\n"
        )
        out = parse_metrics(tmp_path / "task.yaml")
        assert out == [{"name": "speedup", "comparator": "gte", "threshold": 1.0}]

    def test_stochastic(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "metrics:\n"
            "  - {name: max_diff, comparator: lte, noise_multiplier: 2.0, absolute_floor: 0.05}\n"
        )
        out = parse_metrics(tmp_path / "task.yaml")
        assert out[0]["noise_multiplier"] == 2.0
        assert out[0]["absolute_floor"] == 0.05

    def test_invalid_comparator_skipped(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "metrics:\n  - {name: x, comparator: wat, threshold: 1.0}\n"
        )
        assert parse_metrics(tmp_path / "task.yaml") == []

    def test_missing_required_skipped(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "metrics:\n  - {name: x, threshold: 1.0}\n"  # no comparator
        )
        assert parse_metrics(tmp_path / "task.yaml") == []

    def test_neither_threshold_nor_stochastic_skipped(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "metrics:\n  - {name: x, comparator: gte}\n"
        )
        assert parse_metrics(tmp_path / "task.yaml") == []


# --------------------------------------------------------------------------
# Mode subsystem (parse_active_mode, parse_modes_block, write_*, remove_*)
# --------------------------------------------------------------------------

class TestParseActiveMode:
    def test_missing_returns_legacy(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_active_mode(tmp_path / "task.yaml") == LEGACY_MODE

    def test_missing_file(self, tmp_path: Path):
        assert parse_active_mode(tmp_path / "noexist.yaml") == LEGACY_MODE

    def test_explicit(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("active_mode: parallel_t8\n")
        assert parse_active_mode(tmp_path / "task.yaml") == "parallel_t8"

    def test_quoted(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text('active_mode: "default"\n')
        assert parse_active_mode(tmp_path / "task.yaml") == "default"


class TestParseModesBlock:
    def test_empty_when_absent(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_modes_block(tmp_path / "task.yaml") == {}

    def test_parses_inline_entries(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n"
            "  default: {description: serial, reference_script: reference.R}\n"
            "  parallel_t8: {description: \"mclapply mc.cores=8\", reference_script: reference.parallel_t8.R, threads: 8}\n"
        )
        out = parse_modes_block(tmp_path / "task.yaml")
        assert set(out.keys()) == {"default", "parallel_t8"}
        assert out["default"]["reference_script"] == "reference.R"
        assert out["parallel_t8"]["threads"] == "8"  # parsed as string per impl

    def test_block_ends_at_top_level_key(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n  default: {description: serial}\n"
            "metrics:\n  - {name: x, comparator: gte, threshold: 1}\n"
        )
        out = parse_modes_block(tmp_path / "task.yaml")
        assert list(out.keys()) == ["default"]


class TestWriteModeEntry:
    def test_inserts_block_when_absent(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("target_repo: foo\n")
        write_mode_entry(p, "parallel_t8", {"description": "x", "reference_script": "ref.R"})
        text = p.read_text()
        assert "modes:" in text
        assert "parallel_t8:" in text

    def test_replaces_existing_entry(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "modes:\n"
            "  parallel_t8: {description: old, reference_script: old.R}\n"
        )
        write_mode_entry(p, "parallel_t8", {"description": "new", "reference_script": "new.R"})
        out = parse_modes_block(p)
        assert out["parallel_t8"]["description"] == "new"
        assert "old" not in p.read_text()

    def test_preserves_other_entries(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "modes:\n"
            "  default: {description: d, reference_script: r.R}\n"
        )
        write_mode_entry(p, "parallel_t8", {"description": "p", "reference_script": "p.R"})
        out = parse_modes_block(p)
        assert set(out.keys()) == {"default", "parallel_t8"}


class TestRemoveModeEntry:
    def test_returns_false_when_missing(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("modes:\n  default: {description: d, reference_script: r.R}\n")
        assert remove_mode_entry(p, "nonexistent") is False

    def test_removes_entry_keeps_block(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "modes:\n"
            "  default: {description: d, reference_script: r.R}\n"
            "  parallel_t8: {description: p, reference_script: p.R}\n"
        )
        assert remove_mode_entry(p, "parallel_t8") is True
        assert list(parse_modes_block(p).keys()) == ["default"]

    def test_removes_block_when_empty_after(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("modes:\n  default: {description: d, reference_script: r.R}\n")
        assert remove_mode_entry(p, "default") is True
        assert "modes:" not in p.read_text()


class TestWriteActiveMode:
    def test_inserts_when_absent(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("target_repo: foo\n")
        write_active_mode(p, "parallel_t8")
        assert parse_active_mode(p) == "parallel_t8"

    def test_replaces_existing(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("active_mode: default\n")
        write_active_mode(p, "parallel_t8")
        assert parse_active_mode(p) == "parallel_t8"
        # Old line gone (only one active_mode line in file).
        assert p.read_text().count("active_mode:") == 1


# --------------------------------------------------------------------------
# parse_executor / parse_threading_mode / parse_algorithm_class /
# parse_random_seeds / parse_scaling_tax_thresholds
# --------------------------------------------------------------------------

class TestParseExecutor:
    def test_empty_when_absent(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_executor(tmp_path / "task.yaml") == {}

    def test_python_only(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "executor:\n  python: myenv\n"
        )
        assert parse_executor(tmp_path / "task.yaml") == {"python": "myenv"}

    def test_quotes_stripped(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            'executor:\n  python: "/opt/envs/x/bin/python"\n'
        )
        assert parse_executor(tmp_path / "task.yaml") == {"python": "/opt/envs/x/bin/python"}

    def test_block_ends_at_top_level_key(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "executor:\n  python: myenv\n"
            "datasets:\n  - {tier: tiny, name: t, path: /abs/t.h5}\n"
        )
        out = parse_executor(tmp_path / "task.yaml")
        assert "python" in out
        assert "tier" not in out  # didn't bleed in from datasets


class TestParseThreadingMode:
    def test_default_when_absent(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_threading_mode(tmp_path / "task.yaml") == "default"

    def test_not_applicable(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("threading: not_applicable\n")
        assert parse_threading_mode(tmp_path / "task.yaml") == "not_applicable"

    def test_unknown_value_treated_as_default(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("threading: weird_value\n")
        assert parse_threading_mode(tmp_path / "task.yaml") == "default"


class TestParseAlgorithmClass:
    def test_default_deterministic(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_algorithm_class(tmp_path / "task.yaml") == "deterministic"

    def test_stochastic(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("algorithm_class: stochastic\n")
        assert parse_algorithm_class(tmp_path / "task.yaml") == "stochastic"

    def test_unknown_falls_back_to_deterministic(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("algorithm_class: cheese\n")
        assert parse_algorithm_class(tmp_path / "task.yaml") == "deterministic"


class TestParseRandomSeeds:
    def test_defaults_when_absent(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        out = parse_random_seeds(tmp_path / "task.yaml")
        assert out == {"primary": 42, "noise_calibration": [43, 44, 45]}

    def test_legacy_scalar_noise_calibration_becomes_list(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("random_seeds: {primary: 7, noise_calibration: 11}\n")
        out = parse_random_seeds(tmp_path / "task.yaml")
        assert out == {"primary": 7, "noise_calibration": [11]}

    def test_explicit_noise_calibration_list(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "random_seeds: {primary: 7, noise_calibration: [11, 13, 17]}\n"
        )
        out = parse_random_seeds(tmp_path / "task.yaml")
        assert out == {"primary": 7, "noise_calibration": [11, 13, 17]}


class TestParseScalingTaxThresholds:
    def test_defaults_when_absent(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        out = parse_scaling_tax_thresholds(tmp_path / "task.yaml")
        assert out == {
            "ood_large_soft": 1.5,
            "ood_xlarge_soft": 2.0,
            "hard_fail": 5.0,
            "super_linear_max": 1.5,
        }

    def test_overrides(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "scaling_tax_thresholds:\n"
            "  ood_large_soft: 1.2\n"
            "  hard_fail: 10.0\n"
        )
        out = parse_scaling_tax_thresholds(tmp_path / "task.yaml")
        assert out["ood_large_soft"] == 1.2
        assert out["hard_fail"] == 10.0
        # Untouched defaults preserved.
        assert out["ood_xlarge_soft"] == 2.0


# --------------------------------------------------------------------------
# parse_intrinsic_noise / write_intrinsic_noise (round-trip)
# --------------------------------------------------------------------------

class TestIntrinsicNoise:
    def test_empty_when_absent(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_intrinsic_noise(tmp_path / "task.yaml") == {}

    def test_round_trip(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("target_repo: foo\n")
        write_intrinsic_noise(p, "tiny", {"max_diff": 0.0008, "pearson": 0.9994})
        out = parse_intrinsic_noise(p)
        assert out["tiny"]["max_diff"] == pytest.approx(0.0008)
        assert out["tiny"]["pearson"] == pytest.approx(0.9994)

    def test_update_existing_tier(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "intrinsic_noise:\n  tiny: {max_diff: 0.001}\n"
        )
        write_intrinsic_noise(p, "tiny", {"max_diff": 0.002})
        out = parse_intrinsic_noise(p)
        assert out["tiny"]["max_diff"] == pytest.approx(0.002)
        # Old value not duplicated.
        assert p.read_text().count("max_diff: 0.001") == 0

    def test_appends_new_tier_to_existing_block(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("intrinsic_noise:\n  tiny: {max_diff: 0.001}\n")
        write_intrinsic_noise(p, "medium", {"max_diff": 0.003})
        out = parse_intrinsic_noise(p)
        assert set(out.keys()) == {"tiny", "medium"}
        assert out["medium"]["max_diff"] == pytest.approx(0.003)


# --------------------------------------------------------------------------
# effective_threshold
# --------------------------------------------------------------------------

class TestEffectiveThreshold:
    def test_deterministic_returns_threshold(self):
        metric = {"name": "x", "comparator": "gte", "threshold": 0.95}
        eff, label = effective_threshold(metric, "tiny", {})
        assert eff == 0.95
        assert label == "absolute"

    def test_stochastic_lte_uses_max(self):
        metric = {
            "name": "max_diff", "comparator": "lte",
            "noise_multiplier": 2.0, "absolute_floor": 0.05,
        }
        # noise_at_tier = 0.01 → relaxed = 2 * 0.01 = 0.02 < floor 0.05 → use floor.
        eff, _ = effective_threshold(metric, "tiny", {"tiny": {"max_diff": 0.01}})
        assert eff == 0.05

    def test_stochastic_lte_relaxed_above_floor(self):
        metric = {
            "name": "max_diff", "comparator": "lte",
            "noise_multiplier": 2.0, "absolute_floor": 0.05,
        }
        # noise_at_tier = 0.10 → relaxed = 0.20, > floor 0.05 → use relaxed.
        eff, _ = effective_threshold(metric, "tiny", {"tiny": {"max_diff": 0.10}})
        assert eff == pytest.approx(0.20)

    def test_stochastic_gte_uses_distance_from_perfect(self):
        metric = {
            "name": "pearson", "comparator": "gte",
            "noise_multiplier": 2.0, "absolute_floor": 0.5,
        }
        # noise_at_tier = 0.9994 → distance = 0.0006 → relaxed = 1 - 2*0.0006 = 0.9988.
        eff, _ = effective_threshold(metric, "tiny", {"tiny": {"pearson": 0.9994}})
        assert eff == pytest.approx(0.9988)

    def test_stochastic_falls_back_to_floor_when_no_calibration(self):
        metric = {
            "name": "max_diff", "comparator": "lte",
            "noise_multiplier": 2.0, "absolute_floor": 0.05,
        }
        eff, label = effective_threshold(metric, "tiny", {})
        assert eff == 0.05
        assert "no intrinsic_noise" in label


# --------------------------------------------------------------------------
# resolve_tiers
# --------------------------------------------------------------------------

class TestResolveTiers:
    def test_no_request_picks_first(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
            "  - {tier: medium, name: m, path: /abs/m.h5}\n"
        )
        out = resolve_tiers(tmp_path / "task.yaml", None)
        assert len(out) == 1
        assert out[0]["tier"] == "tiny"

    def test_resolve_by_tier(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
            "  - {tier: medium, name: m, path: /abs/m.h5}\n"
        )
        out = resolve_tiers(tmp_path / "task.yaml", ["medium"])
        assert out[0]["tier"] == "medium"

    def test_resolve_by_name(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: foo, path: /abs/t.h5}\n"
        )
        out = resolve_tiers(tmp_path / "task.yaml", ["foo"])
        assert out[0]["name"] == "foo"

    def test_unknown_token_raises(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: t, path: /abs/t.h5}\n"
        )
        with pytest.raises(ValueError, match="not found"):
            resolve_tiers(tmp_path / "task.yaml", ["nonexistent"])

    def test_empty_datasets_raises(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        with pytest.raises(ValueError, match="no datasets"):
            resolve_tiers(tmp_path / "task.yaml", None)
