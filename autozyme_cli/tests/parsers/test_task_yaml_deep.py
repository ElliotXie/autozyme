"""Deep coverage tests for zyme.parsers.task_yaml.

Targets gaps not covered by tests/parsers/test_task_yaml.py:
  - malformed-entry warning paths in parse_datasets / parse_metrics /
    _parse_inline_value
  - _strip_trailing_yaml_comment
  - parse_baseline_threads / write_baseline_threads
  - parse_target_function / parse_upstream_parallelism / parse_synthesis
  - parse_intrinsic_noise + write_intrinsic_noise (append-new-block path)
  - effective_threshold gte stochastic + floor-clamp labels
  - write_active_mode (insert-before-modes path) + remove_mode_entry edges
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.parsers.task_yaml import (
    _render_mode_value,
    _strip_trailing_yaml_comment,
    effective_threshold,
    parse_baseline_threads,
    parse_datasets,
    parse_intrinsic_noise,
    parse_metrics,
    parse_synthesis,
    parse_target_function,
    parse_upstream_parallelism,
    remove_mode_entry,
    resolve_tiers,
    write_active_mode,
    write_baseline_threads,
    write_intrinsic_noise,
    write_mode_entry,
)
from zyme.utils import LEGACY_THREAD


# --------------------------------------------------------------------------
# _strip_trailing_yaml_comment
# --------------------------------------------------------------------------

class TestStripTrailingComment:
    def test_strips_comment_after_brace(self):
        s = "{name: foo}  # the reason"
        assert _strip_trailing_yaml_comment(s) == "{name: foo}"

    def test_keeps_hash_inside_value(self):
        # No `}` before `#` -> conservative, leave it alone.
        s = "color: #abc"
        assert _strip_trailing_yaml_comment(s) == s

    def test_no_hash_passthrough(self):
        assert _strip_trailing_yaml_comment("{a: 1}") == "{a: 1}"


# --------------------------------------------------------------------------
# parse_datasets: trailing comment + malformed entries
# --------------------------------------------------------------------------

class TestParseDatasetsEdge:
    def test_strips_trailing_comment(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: foo, path: /abs/foo.h5}  # primary dev tier\n"
        )
        out = parse_datasets(tmp_path / "task.yaml")
        assert len(out) == 1
        assert out[0]["name"] == "foo"

    def test_malformed_kv_dropped_with_warning(self, tmp_path: Path, capsys):
        # An entry with a token that has no `key: value` shape is dropped
        # (and a WARN is written to stderr).
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: foo, path: /abs/foo.h5, junkwithoutcolon}\n"
        )
        out = parse_datasets(tmp_path / "task.yaml")
        assert out[0]["name"] == "foo"  # other keys still parsed
        err = capsys.readouterr().err
        assert "parse_datasets" in err and "WARN" in err

    def test_entry_without_name_or_path_skipped(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: foo}\n"  # no path -> skipped
        )
        assert parse_datasets(tmp_path / "task.yaml") == []

    def test_quoted_scalar_unquoted(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            'datasets:\n  - {tier: tiny, name: "foo bar", path: /abs/x.h5}\n'
        )
        out = parse_datasets(tmp_path / "task.yaml")
        assert out[0]["name"] == "foo bar"


# --------------------------------------------------------------------------
# parse_metrics: rejection-warning paths
# --------------------------------------------------------------------------

class TestParseMetricsEdge:
    def test_non_flow_entry_rejected(self, tmp_path: Path, capsys):
        (tmp_path / "task.yaml").write_text(
            "metrics:\n  - name: x\n"  # not `{...}` -> rejected
        )
        out = parse_metrics(tmp_path / "task.yaml")
        assert out == []
        assert "rejected" in capsys.readouterr().err

    def test_non_numeric_threshold_rejected(self, tmp_path: Path, capsys):
        (tmp_path / "task.yaml").write_text(
            "metrics:\n  - {name: x, comparator: gte, threshold: notanumber}\n"
        )
        assert parse_metrics(tmp_path / "task.yaml") == []
        assert "non-numeric" in capsys.readouterr().err

    def test_many_rejections_truncates_to_three(self, tmp_path: Path, capsys):
        lines = ["metrics:"] + [f"  - name: m{i}\n" for i in range(5)]
        (tmp_path / "task.yaml").write_text("\n".join(lines))
        parse_metrics(tmp_path / "task.yaml")
        err = capsys.readouterr().err
        assert "and 2 more" in err

    def test_trailing_comment_on_metric(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}  # core\n"
        )
        out = parse_metrics(tmp_path / "task.yaml")
        assert out[0]["name"] == "speedup"


# --------------------------------------------------------------------------
# parse_baseline_threads / write_baseline_threads
# --------------------------------------------------------------------------

class TestParseBaselineThreads:
    def test_default_when_absent(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_baseline_threads(tmp_path / "task.yaml") == [LEGACY_THREAD]

    def test_missing_file(self, tmp_path: Path):
        assert parse_baseline_threads(tmp_path / "noexist.yaml") == [LEGACY_THREAD]

    def test_parses_list(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("baseline_threads: [1, 4, 8]\n")
        assert parse_baseline_threads(tmp_path / "task.yaml") == [1, 4, 8]

    def test_non_int_tokens_skipped(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("baseline_threads: [1, x, 8]\n")
        assert parse_baseline_threads(tmp_path / "task.yaml") == [1, 8]

    def test_empty_list_defaults(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("baseline_threads: []\n")
        assert parse_baseline_threads(tmp_path / "task.yaml") == [LEGACY_THREAD]


class TestWriteBaselineThreads:
    def test_insert_when_absent(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("target_repo: foo\n")
        write_baseline_threads(p, [1, 4])
        assert parse_baseline_threads(p) == [1, 4]

    def test_replace_existing(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("baseline_threads: [1]\n")
        write_baseline_threads(p, [1, 8])
        assert parse_baseline_threads(p) == [1, 8]
        assert p.read_text().count("baseline_threads:") == 1

    def test_empty_threads_writes_default(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("target_repo: foo\n")
        write_baseline_threads(p, [])
        assert parse_baseline_threads(p) == [LEGACY_THREAD]

    def test_insert_after_active_mode(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("active_mode: default\nmodes:\n  default: {description: d}\n")
        write_baseline_threads(p, [1, 4])
        lines = p.read_text().splitlines()
        am_idx = next(i for i, l in enumerate(lines) if l.startswith("active_mode"))
        bt_idx = next(i for i, l in enumerate(lines) if l.startswith("baseline_threads"))
        assert bt_idx == am_idx + 1


# --------------------------------------------------------------------------
# parse_target_function / parse_upstream_parallelism / parse_synthesis
# --------------------------------------------------------------------------

class TestParseTargetFunction:
    def test_absent(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_target_function(tmp_path / "task.yaml") == ""

    def test_missing_file(self, tmp_path: Path):
        assert parse_target_function(tmp_path / "noexist.yaml") == ""

    def test_parses_value(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_function: spacexr::run.RCTD\n")
        assert parse_target_function(tmp_path / "task.yaml") == "spacexr::run.RCTD"

    def test_quoted_stripped(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text('target_function: "mod.fn"\n')
        assert parse_target_function(tmp_path / "task.yaml") == "mod.fn"


class TestParseUpstreamParallelism:
    def test_absent(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_upstream_parallelism(tmp_path / "task.yaml") == []

    def test_parses_list(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            'upstream_parallelism: [BPPARAM, "n.cores"]\n'
        )
        assert parse_upstream_parallelism(tmp_path / "task.yaml") == ["BPPARAM", "n.cores"]

    def test_empty_list(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("upstream_parallelism: []\n")
        assert parse_upstream_parallelism(tmp_path / "task.yaml") == []


class TestParseSynthesis:
    def test_absent_returns_none(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert parse_synthesis(tmp_path / "task.yaml") is None

    def test_missing_file(self, tmp_path: Path):
        assert parse_synthesis(tmp_path / "noexist.yaml") is None

    def test_parses_reason(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("synthesis: PDE solver, no real data\n")
        assert parse_synthesis(tmp_path / "task.yaml") == "PDE solver, no real data"


# --------------------------------------------------------------------------
# parse_intrinsic_noise edge + write append-new-block
# --------------------------------------------------------------------------

class TestIntrinsicNoiseEdge:
    def test_missing_file_empty(self, tmp_path: Path):
        assert parse_intrinsic_noise(tmp_path / "noexist.yaml") == {}

    def test_write_appends_new_block_when_absent(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("target_repo: foo\n")
        write_intrinsic_noise(p, "tiny", {"max_diff": 0.001})
        out = parse_intrinsic_noise(p)
        assert out["tiny"]["max_diff"] == pytest.approx(0.001)

    def test_block_ends_at_top_level_key(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "intrinsic_noise:\n  tiny: {max_diff: 0.001}\n"
            "metrics:\n  - {name: x, comparator: gte, threshold: 1}\n"
        )
        out = parse_intrinsic_noise(p)
        assert set(out.keys()) == {"tiny"}


# --------------------------------------------------------------------------
# effective_threshold: gte stochastic + floor-clamp labels
# --------------------------------------------------------------------------

class TestEffectiveThresholdEdge:
    def test_gte_relaxed_below_floor_clamps(self):
        # gte with a high floor that beats the relaxed value -> clamp to floor.
        metric = {"name": "pearson", "comparator": "gte",
                  "noise_multiplier": 2.0, "absolute_floor": 0.999}
        # noise 0.99 -> relaxed = 1 - 2*0.01 = 0.98 < floor 0.999 -> floor.
        eff, label = effective_threshold(metric, "tiny", {"tiny": {"pearson": 0.99}})
        assert eff == pytest.approx(0.999)
        assert "absolute_floor" in label

    def test_lte_relaxed_above_floor_label(self):
        metric = {"name": "max_diff", "comparator": "lte",
                  "noise_multiplier": 2.0, "absolute_floor": 0.01}
        eff, label = effective_threshold(metric, "tiny", {"tiny": {"max_diff": 0.10}})
        assert eff == pytest.approx(0.20)
        assert "max(" in label

    def test_gte_relaxed_above_floor_label(self):
        metric = {"name": "pearson", "comparator": "gte",
                  "noise_multiplier": 2.0, "absolute_floor": 0.5}
        eff, label = effective_threshold(metric, "tiny", {"tiny": {"pearson": 0.9994}})
        assert eff == pytest.approx(0.9988)
        assert "max(" in label

    def test_default_multiplier_when_absent(self):
        # noise_multiplier defaults to 2.0 when not present.
        metric = {"name": "max_diff", "comparator": "lte", "absolute_floor": 0.01}
        eff, _ = effective_threshold(metric, "tiny", {"tiny": {"max_diff": 0.10}})
        assert eff == pytest.approx(0.20)


# --------------------------------------------------------------------------
# write_active_mode insert-before-modes + remove_mode_entry edges
# --------------------------------------------------------------------------

class TestWriteActiveModeEdge:
    def test_inserts_before_modes_block(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("modes:\n  default: {description: d}\n")
        write_active_mode(p, "default")
        lines = p.read_text().splitlines()
        am_idx = next(i for i, l in enumerate(lines) if l.startswith("active_mode"))
        modes_idx = next(i for i, l in enumerate(lines) if l.startswith("modes:"))
        assert am_idx < modes_idx


class TestRemoveModeEntryEdge:
    def test_missing_file(self, tmp_path: Path):
        assert remove_mode_entry(tmp_path / "noexist.yaml", "x") is False

    def test_no_modes_block(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("target_repo: foo\n")
        assert remove_mode_entry(p, "x") is False


# --------------------------------------------------------------------------
# _render_mode_value (quoting)
# --------------------------------------------------------------------------

class TestRenderModeValue:
    def test_empty_is_quoted(self):
        assert _render_mode_value("") == '""'

    def test_plain_unquoted(self):
        assert _render_mode_value("reference.R") == "reference.R"

    def test_comma_value_quoted(self):
        assert _render_mode_value("a, b") == '"a, b"'

    def test_comma_value_roundtrips_captured(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("target_repo: foo\n")
        write_mode_entry(p, "m", {"description": "mc.cores=8, fork"})
        # The renderer wraps comma-bearing values in quotes on disk, and
        # parse_modes_block's comma splitter is quote-aware, so the
        # write/read roundtrip preserves the comma-bearing value intact.
        assert '"mc.cores=8, fork"' in p.read_text()
        from zyme.parsers.task_yaml import parse_modes_block
        out = parse_modes_block(p)
        assert out["m"]["description"] == "mc.cores=8, fork"

    def test_plain_value_roundtrips(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text("target_repo: foo\n")
        write_mode_entry(p, "m", {"description": "serial",
                                  "reference_script": "reference.R"})
        from zyme.parsers.task_yaml import parse_modes_block
        out = parse_modes_block(p)
        assert out["m"]["description"] == "serial"
        assert out["m"]["reference_script"] == "reference.R"


# --------------------------------------------------------------------------
# resolve_tiers: by-name path already covered; verify ordering preserved
# --------------------------------------------------------------------------

class TestResolveTiersOrder:
    def test_order_follows_request(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
            "  - {tier: medium, name: m, path: /abs/m.h5}\n"
        )
        out = resolve_tiers(tmp_path / "task.yaml", ["medium", "tiny"])
        assert [e["tier"] for e in out] == ["medium", "tiny"]
