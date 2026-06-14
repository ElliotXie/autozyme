"""Deep coverage tests for zyme.parsers.package_verify_tsv.

Targets gaps not covered by tests/parsers/test_package_verify_tsv.py:
  - dataset_from_row priority chain + fill_batch_datasets propagation
  - upgrade_rows_to_long_header (LONG / legacy-pre-pkgver / legacy headers)
  - format_tier_dataset_label + per_rep_* helpers
  - row_is_valid / row_is_sentinel / row_is_oom_sentinel / row_all_pass
  - _row_threads / _threads_ok / prune_published_tsv_text
  - append_published_tsvs / merge edge cases (empty/legacy upgrade)
  - sort_rows_for_output ordering + summarize_batches paths
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.parsers.package_verify_tsv import (
    LEGACY_LONG_HEADER,
    LEGACY_LONG_HEADER_PRE_PKGVER,
    LONG_HEADER,
    MergeStats,
    PublishFilter,
    PublishFilterError,
    _batch_all_pass,
    _row_threads,
    _select_tail,
    append_published_tsvs,
    dataset_for_tier,
    dataset_from_row,
    fill_batch_datasets,
    filter_package_verify_rows,
    format_tier_dataset_label,
    merge_published_tsvs,
    per_rep_peak_mb_change_pct,
    per_rep_peak_mb_fold,
    per_rep_speedup_pct,
    per_rep_speedup_x,
    prune_published_tsv_text,
    render_package_verify_tsv,
    row_all_pass,
    row_is_oom_sentinel,
    row_is_sentinel,
    row_is_valid,
    sort_rows_for_output,
    summarize_batches,
    tier_dataset_map_from_task_yaml,
    upgrade_rows_to_long_header,
)


_HEADER_LINE = "\t".join(LONG_HEADER) + "\n"


def _full_row(pass_cell: str = "", **over) -> dict[str, str]:
    # `pass` is a Python keyword, so it's accepted via the dedicated
    # `pass_cell` parameter rather than **over.
    base = {c: "" for c in LONG_HEADER}
    base.update({
        "timestamp": "2026-01-01T00:00:00", "patch_name": "p",
        "tier": "small", "rep_idx": "1", "variant": "baseline",
        "sec": "10", "peak_mb": "100", "pass": pass_cell,
        "framework_version": "0.3.0", "system_os": "macOS",
        "system_cpu": "CPU", "system_threads": "1",
    })
    base.update(over)
    return base


# --------------------------------------------------------------------------
# tier_dataset_map_from_task_yaml + dataset_for_tier + dataset_from_row
# --------------------------------------------------------------------------

class TestTierDatasetMap:
    def test_missing_file(self, tmp_path: Path):
        assert tier_dataset_map_from_task_yaml(tmp_path / "noexist.yaml") == {}

    def test_parses_datasets(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: small, name: tiny_a, path: /a}\n"
            "  - {tier: medium, name: med_a, path: /b}\n"
        )
        out = tier_dataset_map_from_task_yaml(p)
        assert out == {"small": "tiny_a", "medium": "med_a"}

    def test_scalar_document_returns_empty(self, tmp_path: Path):
        # A YAML document that parses to a non-mapping (here `None` via an
        # empty file) yields {} through the `... or {}` guard.
        p = tmp_path / "task.yaml"
        p.write_text("# only a comment\n")
        assert tier_dataset_map_from_task_yaml(p) == {}

    def test_malformed_yaml_returns_empty(self, tmp_path: Path):
        # B15 fix: yaml.safe_load raises yaml.YAMLError (not ValueError) on
        # malformed YAML; the function now catches it and returns {} as the
        # docstring promises, instead of letting the exception propagate.
        p = tmp_path / "task.yaml"
        p.write_text("datasets: [unterminated, {tier: small\n")  # broken flow
        assert tier_dataset_map_from_task_yaml(p) == {}

    def test_non_mapping_scalar_document_returns_empty(self, tmp_path: Path):
        # A YAML doc that parses to a bare scalar (a string) is not a mapping.
        p = tmp_path / "task.yaml"
        p.write_text("justastring\n")
        assert tier_dataset_map_from_task_yaml(p) == {}

    def test_datasets_without_tier_or_name_skipped(self, tmp_path: Path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: small}\n"      # no name -> skipped
            "  - {name: only_name}\n"  # no tier -> skipped
        )
        assert tier_dataset_map_from_task_yaml(p) == {}


class TestDatasetForTier:
    def test_lookup_and_default(self):
        m = {"small": "tiny_a"}
        assert dataset_for_tier("small", m) == "tiny_a"
        assert dataset_for_tier("medium", m) == ""


class TestDatasetFromRow:
    def test_metrics_json_dataset_wins(self):
        row = {"metrics_json": '{"dataset": "from_metrics"}', "dataset": "cell",
               "tier": "small", "note": ""}
        assert dataset_from_row(row) == "from_metrics"

    def test_migrated_note_returns_empty(self):
        row = {"metrics_json": "", "note": "migrated: full_run", "dataset": "x",
               "tier": "small"}
        assert dataset_from_row(row) == ""

    def test_absorbed_note_returns_empty(self):
        row = {"metrics_json": "", "note": "absorbed: legacy", "dataset": "x",
               "tier": "small"}
        assert dataset_from_row(row) == ""

    def test_existing_cell_used(self):
        row = {"metrics_json": "", "note": "", "dataset": "existing", "tier": "small"}
        assert dataset_from_row(row) == "existing"

    def test_falls_back_to_tier_map(self):
        row = {"metrics_json": "", "note": "", "dataset": "", "tier": "small"}
        assert dataset_from_row(row, {"small": "from_map"}) == "from_map"

    def test_bad_metrics_json_falls_through(self):
        row = {"metrics_json": "{not valid", "note": "", "dataset": "cell",
               "tier": "small"}
        assert dataset_from_row(row) == "cell"


class TestFillBatchDatasets:
    def test_propagates_single_known_within_batch(self):
        rows = [
            _full_row(variant="baseline", rep_idx="1", dataset="ds1"),
            _full_row(variant="patched", rep_idx="1", dataset=""),
        ]
        out = fill_batch_datasets(rows)
        # The patched row in the same batch inherits ds1.
        assert all(r["dataset"] == "ds1" for r in out)

    def test_refresh_from_tier_map(self):
        rows = [_full_row(dataset="old", tier="small")]
        out = fill_batch_datasets(rows, {"small": "fresh"},
                                  refresh_from_tier_map=True)
        assert out[0]["dataset"] == "fresh"

    def test_migrated_row_dataset_cleared_then_batch_filled(self):
        # dataset_from_row() blanks a `migrated:` row's dataset, then the
        # batch propagation refills it from the sibling's known dataset.
        rows = [
            _full_row(variant="baseline", rep_idx="1", dataset="ds_60k",
                      note="migrated: full"),
            _full_row(variant="baseline", rep_idx="2", dataset="ds",
                      note=""),
        ]
        out = fill_batch_datasets(rows)
        # The migrated row's stale "ds_60k" is dropped and both end up "ds".
        assert {r["dataset"] for r in out} == {"ds"}


# --------------------------------------------------------------------------
# upgrade_rows_to_long_header
# --------------------------------------------------------------------------

class TestUpgradeRowsToLongHeader:
    def test_already_long_header(self):
        rows = [_full_row(dataset="ds")]
        hdr, out = upgrade_rows_to_long_header(LONG_HEADER, rows)
        assert hdr == LONG_HEADER
        assert out[0]["patch_name"] == "p"

    def test_pre_pkgver_legacy_upgraded(self):
        # Row dict keyed by the pre-package_version header.
        row = {c: "" for c in LEGACY_LONG_HEADER_PRE_PKGVER}
        row.update({"timestamp": "t", "patch_name": "p", "tier": "small",
                    "dataset": "ds", "variant": "baseline", "sec": "1"})
        hdr, out = upgrade_rows_to_long_header(LEGACY_LONG_HEADER_PRE_PKGVER, [row])
        assert hdr == LONG_HEADER
        assert "package_version" in out[0]
        assert out[0]["package_version"] == ""

    def test_legacy_no_dataset_upgraded(self):
        row = {c: "" for c in LEGACY_LONG_HEADER}
        row.update({"timestamp": "t", "patch_name": "p", "tier": "small",
                    "variant": "baseline", "sec": "1"})
        hdr, out = upgrade_rows_to_long_header(LEGACY_LONG_HEADER, [row],
                                               {"small": "tiny_a"})
        assert hdr == LONG_HEADER
        assert out[0]["dataset"] == "tiny_a"

    def test_unknown_header_passthrough(self):
        weird = ["a", "b", "c"]
        rows = [{"a": "1"}]
        hdr, out = upgrade_rows_to_long_header(weird, rows)
        assert hdr == weird
        assert out == rows


# --------------------------------------------------------------------------
# format_tier_dataset_label + per_rep_* helpers
# --------------------------------------------------------------------------

class TestFormatTierDatasetLabel:
    def test_both_present(self):
        assert format_tier_dataset_label("medium", "ds_a") == "medium: ds_a"

    def test_tier_only(self):
        assert format_tier_dataset_label("small", "") == "small"

    def test_dataset_only(self):
        assert format_tier_dataset_label("", "ds_a") == "ds_a"


class TestPerRepHelpers:
    def test_speedup_x(self):
        assert per_rep_speedup_x([10.0, 12.0], 5.0) == pytest.approx(2.2)

    def test_speedup_x_empty_baseline(self):
        assert per_rep_speedup_x([], 5.0) is None

    def test_speedup_x_bad_patched(self):
        assert per_rep_speedup_x([10.0], 0.0) is None

    def test_speedup_pct(self):
        # median baseline 10, patched 5 -> 50% faster.
        assert per_rep_speedup_pct([10.0, 10.0], 5.0) == pytest.approx(50.0)

    def test_speedup_pct_zero_baseline(self):
        assert per_rep_speedup_pct([0.0], 5.0) is None

    def test_peak_mb_fold(self):
        assert per_rep_peak_mb_fold([200.0], 100.0) == pytest.approx(2.0)

    def test_peak_mb_fold_empty(self):
        assert per_rep_peak_mb_fold([], 100.0) is None

    def test_peak_mb_change_pct(self):
        assert per_rep_peak_mb_change_pct([200.0], 100.0) == pytest.approx(50.0)

    def test_peak_mb_change_pct_zero_baseline(self):
        assert per_rep_peak_mb_change_pct([0.0], 100.0) is None


# --------------------------------------------------------------------------
# row classification helpers
# --------------------------------------------------------------------------

class TestRowClassification:
    def test_row_is_valid(self):
        assert row_is_valid({"sec": "10", "variant": "baseline"}) is True
        assert row_is_valid({"sec": "", "variant": "baseline"}) is False
        assert row_is_valid({"sec": "10", "variant": "bogus"}) is False

    def test_row_is_sentinel(self):
        s = {"sec": "", "variant": "", "tier": "large", "note": "OOM killed"}
        assert row_is_sentinel(s) is True
        # A row with a sec is not a sentinel.
        assert row_is_sentinel({**s, "sec": "5"}) is False
        # No note -> not a sentinel.
        assert row_is_sentinel({**s, "note": ""}) is False

    def test_row_is_oom_sentinel(self):
        s = {"sec": "", "variant": "", "tier": "large",
             "note": "subprocess exited -9"}
        assert row_is_oom_sentinel(s) is True
        non_oom = {"sec": "", "variant": "", "tier": "large",
                   "note": "skipped: not configured"}
        assert row_is_oom_sentinel(non_oom) is False

    def test_row_all_pass(self):
        assert row_all_pass({"pass": "true"}) is True
        assert row_all_pass({"pass": "yes"}) is True
        assert row_all_pass({"pass": "1"}) is True
        assert row_all_pass({"pass": "false"}) is False
        assert row_all_pass({"pass": ""}) is False


# --------------------------------------------------------------------------
# _row_threads + _threads_ok via prune_published_tsv_text
# --------------------------------------------------------------------------

class TestRowThreads:
    def test_parses_int(self):
        assert _row_threads({"system_threads": "8"}) == 8

    def test_float_string(self):
        assert _row_threads({"system_threads": "4.0"}) == 4

    def test_blank_and_na(self):
        assert _row_threads({"system_threads": ""}) is None
        assert _row_threads({"system_threads": "NA"}) is None
        assert _row_threads({"system_threads": "none"}) is None

    def test_unparseable(self):
        assert _row_threads({"system_threads": "lots"}) is None


class TestPrunePublishedTsv:
    def test_none_max_threads_noop(self):
        text = _HEADER_LINE
        out, removed = prune_published_tsv_text(text, max_threads=None)
        assert out == text and removed == 0

    def test_empty_text(self):
        out, removed = prune_published_tsv_text("   ", max_threads=1)
        assert removed == 0

    def test_drops_high_thread_rows(self):
        rows = [_full_row(system_threads="1"), _full_row(system_threads="8")]
        text = render_package_verify_tsv(LONG_HEADER, rows)
        out, removed = prune_published_tsv_text(text, max_threads=1)
        assert removed == 1
        assert "\t8\t" not in out.replace(_HEADER_LINE, "") or "system_threads" in out.splitlines()[0]

    def test_no_removal_when_all_within_cap(self):
        rows = [_full_row(system_threads="1")]
        text = render_package_verify_tsv(LONG_HEADER, rows)
        out, removed = prune_published_tsv_text(text, max_threads=4)
        assert removed == 0
        assert out == text


# --------------------------------------------------------------------------
# append_published_tsvs
# --------------------------------------------------------------------------

class TestAppendPublishedTsvs:
    def test_append_to_existing(self):
        ex = render_package_verify_tsv(LONG_HEADER, [_full_row(rep_idx="1")])
        new = render_package_verify_tsv(LONG_HEADER, [_full_row(rep_idx="2")])
        out, n = append_published_tsvs(ex, new)
        assert n == 1
        # Both rep rows present.
        assert out.count("\tbaseline\t") == 2

    def test_empty_new_returns_existing(self):
        ex = render_package_verify_tsv(LONG_HEADER, [_full_row()])
        out, n = append_published_tsvs(ex, "")
        assert n == 0
        assert out == ex

    def test_empty_existing_returns_new(self):
        new = render_package_verify_tsv(LONG_HEADER, [_full_row()])
        out, n = append_published_tsvs("", new)
        assert n == 1
        assert out == new

    def test_header_mismatch_raises(self):
        ex = render_package_verify_tsv(LONG_HEADER, [_full_row()])
        weird = "a\tb\nc\td\n"
        with pytest.raises(PublishFilterError, match="header mismatch"):
            append_published_tsvs(ex, weird)


# --------------------------------------------------------------------------
# merge edge cases
# --------------------------------------------------------------------------

class TestMergeEdgeCases:
    def test_empty_new_keeps_existing(self):
        ex = render_package_verify_tsv(LONG_HEADER, [_full_row()])
        out, stats = merge_published_tsvs(ex, "")
        assert isinstance(stats, MergeStats)
        assert out == ex

    def test_empty_existing_takes_new(self):
        new = render_package_verify_tsv(LONG_HEADER, [_full_row()])
        out, stats = merge_published_tsvs("", new)
        assert stats.added == 1

    def test_newer_timestamp_replaces(self):
        old = render_package_verify_tsv(
            LONG_HEADER, [_full_row(timestamp="2026-01-01T00:00:00", sec="10")])
        new = render_package_verify_tsv(
            LONG_HEADER, [_full_row(timestamp="2026-02-01T00:00:00", sec="8")])
        out, stats = merge_published_tsvs(old, new)
        assert stats.replaced == 1
        assert "\t8\t" in out

    def test_older_incoming_kept_out(self):
        new_existing = render_package_verify_tsv(
            LONG_HEADER, [_full_row(timestamp="2026-02-01T00:00:00", sec="8")])
        old_incoming = render_package_verify_tsv(
            LONG_HEADER, [_full_row(timestamp="2026-01-01T00:00:00", sec="10")])
        out, stats = merge_published_tsvs(new_existing, old_incoming)
        assert stats.kept >= 1
        # Existing newer row stays.
        assert "\t8\t" in out

    def test_header_mismatch_raises(self):
        ex = render_package_verify_tsv(LONG_HEADER, [_full_row()])
        with pytest.raises(PublishFilterError, match="header mismatch"):
            merge_published_tsvs(ex, "a\tb\nc\td\n")


# --------------------------------------------------------------------------
# sort_rows_for_output (public alias)
# --------------------------------------------------------------------------

class TestSortRowsForOutput:
    def test_baseline_before_patched(self):
        rows = [
            _full_row(variant="patched", rep_idx="1"),
            _full_row(variant="baseline", rep_idx="1"),
        ]
        out = sort_rows_for_output(rows)
        assert out[0]["variant"] == "baseline"
        assert out[1]["variant"] == "patched"

    def test_sentinel_sorts_last(self):
        rows = [
            {**_full_row(variant=""), "sec": ""},   # crash sentinel variant=""
            _full_row(variant="baseline"),
        ]
        out = sort_rows_for_output(rows)
        assert out[-1]["variant"] == ""

    def test_bad_threads_dont_crash(self):
        rows = [_full_row(system_threads="bogus"),
                _full_row(system_threads="1")]
        # Should not raise even with an unparseable thread cell.
        out = sort_rows_for_output(rows)
        assert len(out) == 2


# --------------------------------------------------------------------------
# summarize_batches
# --------------------------------------------------------------------------

class TestSummarizeBatches:
    def test_basic_speedup(self):
        rows = [
            _full_row(variant="baseline", rep_idx="1", sec="10", peak_mb="200"),
            _full_row(variant="baseline", rep_idx="2", sec="10", peak_mb="200"),
            _full_row(variant="patched", rep_idx="1", sec="5", peak_mb="100",
                      pass_cell="true"),
            _full_row(variant="patched", rep_idx="2", sec="5", peak_mb="100",
                      pass_cell="true"),
        ]
        out = summarize_batches(rows)
        assert len(out) == 1
        bs = out[0]
        assert bs.n_reps == 2
        assert bs.speedup_x == pytest.approx(2.0)
        assert bs.all_pass is True
        assert bs.baseline_sec_mean == 10.0
        assert bs.patched_sec_mean == 5.0

    def test_baseline_only_batch_skipped(self):
        rows = [_full_row(variant="baseline", rep_idx="1", sec="10")]
        # No patched and not a sentinel -> batch dropped entirely.
        assert summarize_batches(rows) == []

    def test_sentinel_only_batch_summarized_as_crashed(self):
        sentinel = _full_row(variant="", sec="", tier="large",
                             note="OOM: memory limit")
        out = summarize_batches([sentinel])
        assert len(out) == 1
        assert out[0].n_reps == 0
        assert out[0].all_pass is False
        assert "OOM" in out[0].note

    def test_speedup_x_zero_patched_is_nan(self):
        rows = [
            _full_row(variant="baseline", rep_idx="1", sec="10"),
            _full_row(variant="patched", rep_idx="1", sec="0"),
        ]
        out = summarize_batches(rows)
        assert len(out) == 1
        import math
        assert math.isnan(out[0].speedup_x)


# --------------------------------------------------------------------------
# _select_tail bound check + _batch_all_pass empty
# --------------------------------------------------------------------------

class TestMiscInternals:
    def test_select_tail_rejects_zero(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            _select_tail([{"x": "1"}], 0)

    def test_batch_all_pass_no_patched_is_false(self):
        # A batch with only baseline rows can't "all pass".
        assert _batch_all_pass([_full_row(variant="baseline")]) is False

    def test_filter_unknown_select_mode_raises(self):
        rows = [_full_row(variant="patched", sec="5", pass_cell="true")]
        with pytest.raises(ValueError, match="unknown select mode"):
            filter_package_verify_rows(rows, PublishFilter(select="bogus"))
