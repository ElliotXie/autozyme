"""Tests for zyme.parsers.results_tsv."""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.parsers.results_tsv import (
    _row_mode,
    append_prompt_id_to_row,
    best_cv_at_dataset,
    best_speeds_at_dataset,
    count_decision_rounds,
    dataset_in_results,
    ensure_mode_schema,
    ensure_results_schema,
    find_best_speed_at_dataset,
    get_baseline_peak_mb,
    get_baseline_speed,
    get_baseline_status,
    migrate_results_add_phase,
    next_rerun_seq,
    parse_log,
    results_header_for_task,
    results_header_width,
    results_phase_index,
    row_phase,
    update_last_status,
)
from zyme.utils import LEGACY_MODE


# --------------------------------------------------------------------------
# _row_mode
# --------------------------------------------------------------------------

class TestRowMode:
    def test_column_absent_returns_legacy(self):
        assert _row_mode(["a", "b"], {"dataset": 0}) == LEGACY_MODE

    def test_column_empty_returns_legacy(self):
        col = {"thread_mode": 1}
        assert _row_mode(["a", ""], col) == LEGACY_MODE

    def test_column_present(self):
        col = {"thread_mode": 1}
        assert _row_mode(["a", "parallel_t8"], col) == "parallel_t8"

    def test_row_too_short_returns_legacy(self):
        col = {"thread_mode": 5}
        assert _row_mode(["a", "b"], col) == LEGACY_MODE


# --------------------------------------------------------------------------
# parse_log
# --------------------------------------------------------------------------

class TestParseLog:
    def test_extracts_speed_and_peak(self):
        log = "speed_sec: 12.5\npeak_memory_mb: 480.0\n"
        speed, peak, metrics, status = parse_log(log)
        assert speed == 12.5
        assert peak == 480.0
        assert status == "pending"

    def test_crash_detection_canonical(self):
        log = "status:           crash\nspeed_sec: 0\n"
        _, _, _, status = parse_log(log)
        assert status == "crash"

    def test_does_not_match_diagnostic_message(self):
        # Pipeline code printing 'CRASH:' diagnostically should NOT be flagged.
        log = "speed_sec: 5.0\nCRASH: investigating gc artifact\n"
        _, _, _, status = parse_log(log)
        assert status == "pending"

    def test_extracts_arbitrary_numeric_metrics(self):
        log = "speed_sec: 1.0\ncpu_sec: 0.95\nknn_overlap: 0.998\n"
        _, _, metrics, _ = parse_log(log)
        assert metrics == {"cpu_sec": 0.95, "knn_overlap": 0.998}

    def test_skips_string_only_metrics(self):
        log = "speed_sec: 1.0\nLang: Python\nstatus: pending\n"
        _, _, metrics, _ = parse_log(log)
        assert "Lang" not in metrics

    def test_peak_mb_alias(self):
        # Both peak_memory_mb and peak_mb keys recognized.
        log = "peak_mb: 100.0\n"
        _, peak, _, _ = parse_log(log)
        assert peak == 100.0


# --------------------------------------------------------------------------
# results_header_width / results_phase_index / migrate_results_add_phase /
# row_phase
# --------------------------------------------------------------------------

class TestResultsHeaderWidth:
    def test_missing_file(self, tmp_path: Path):
        assert results_header_width(tmp_path / "noexist.tsv") == 0

    def test_empty_file(self, tmp_path: Path):
        (tmp_path / "results.tsv").write_text("")
        assert results_header_width(tmp_path / "results.tsv") == 0

    def test_counts_columns(self, task_dir_with_results_v0: Path):
        tsv = task_dir_with_results_v0 / "results.tsv"
        # v0 schema has 11 columns.
        assert results_header_width(tsv) == 11


class TestResultsPhaseIndex:
    def test_present_in_v0(self, task_dir_with_results_v0: Path):
        idx = results_phase_index(task_dir_with_results_v0 / "results.tsv")
        assert idx == 10

    def test_present_in_v2(self, task_dir_with_results_v2: Path):
        idx = results_phase_index(task_dir_with_results_v2 / "results.tsv")
        assert idx == 10

    def test_missing_returns_none(self, tmp_path: Path):
        (tmp_path / "results.tsv").write_text("round\tcommit\n0\tabc\n")
        assert results_phase_index(tmp_path / "results.tsv") is None


class TestMigrateResultsAddPhase:
    def test_idempotent_when_already_has_phase(self, task_dir_with_results_v0: Path):
        tsv = task_dir_with_results_v0 / "results.tsv"
        before = tsv.read_text()
        migrate_results_add_phase(tsv)
        assert tsv.read_text() == before

    def test_adds_phase_to_pre_phase_schema(self, tmp_path: Path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\n"
            "0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tup\t\n"
        )
        migrate_results_add_phase(tsv)
        text = tsv.read_text()
        assert text.startswith(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
        )


class TestRowPhase:
    def test_empty_returns_optimize(self):
        assert row_phase({}) == "optimize"
        assert row_phase({"phase": ""}) == "optimize"

    def test_explicit(self):
        assert row_phase({"phase": "validate"}) == "validate"


# --------------------------------------------------------------------------
# update_last_status
# --------------------------------------------------------------------------

class TestUpdateLastStatus:
    def test_flips_pending_to_keep(self, tmp_path: Path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
            "1\tabc\ttiny_a\t8.0\t20.0\t500\tpending\t{}\tH\t\toptimize\n"
        )
        update_last_status(tsv, "keep", "great")
        # Last row's status now keep, description great.
        last = tsv.read_text().strip().split("\n")[-1].split("\t")
        assert last[6] == "keep"
        assert last[9] == "great"

    def test_no_pending_row_is_noop(self, tmp_path: Path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
            "1\tabc\ttiny_a\t8.0\t20.0\t500\tkeep\t{}\tH\t\toptimize\n"
        )
        before = tsv.read_text()
        update_last_status(tsv, "discard", "n/a")
        assert tsv.read_text() == before

    def test_skips_rerun_rows(self, tmp_path: Path):
        # accept/reject must update the original DECISION row, not subsequent
        # rerun rows. Decision = pending; rerun = rerun (terminal).
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
            "1\tabc\ttiny_a\t8.0\t20.0\t500\tpending\t{}\tH\t\toptimize\n"
            "1.1\tabc\ttiny_a\t8.1\t19.0\t500\trerun\t{}\tH\t\toptimize\n"
        )
        update_last_status(tsv, "keep", "ok")
        rows = tsv.read_text().strip().split("\n")[1:]
        # Round-1 row updated to keep; round-1.1 still rerun.
        assert rows[0].split("\t")[6] == "keep"
        assert rows[1].split("\t")[6] == "rerun"


# --------------------------------------------------------------------------
# count_decision_rounds / next_rerun_seq
# --------------------------------------------------------------------------

class TestCountDecisionRounds:
    def test_excludes_baseline_and_reruns(self, task_dir_with_results_v0: Path):
        # v0 fixture has: round 0 (baseline), 1 (keep), 1.1 (rerun), 2 (discard).
        # Phase=optimize for all. So decision count = 2.
        n = count_decision_rounds(task_dir_with_results_v0 / "results.tsv", phase="optimize")
        assert n == 2

    def test_phase_filter(self, tmp_path: Path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
            "1\ta\ttiny_a\t1\t1\t1\tkeep\t{}\tH\t\toptimize\n"
            "2\tb\ttiny_a\t1\t1\t1\tkeep\t{}\tH\t\tvalidate\n"
        )
        assert count_decision_rounds(tsv, phase="optimize") == 1
        assert count_decision_rounds(tsv, phase="validate") == 1
        assert count_decision_rounds(tsv, phase="all") == 2

    def test_empty_phase_treated_as_optimize(self, tmp_path: Path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
            "1\ta\ttiny_a\t1\t1\t1\tkeep\t{}\tH\t\t\n"  # empty phase
        )
        assert count_decision_rounds(tsv, phase="optimize") == 1

    def test_missing_file_returns_zero(self, tmp_path: Path):
        assert count_decision_rounds(tmp_path / "noexist.tsv") == 0


class TestNextRerunSeq:
    def test_no_existing_returns_one(self, task_dir_with_results_v0: Path):
        # v0 fixture has round 1.1 as rerun under round 1.
        assert next_rerun_seq(task_dir_with_results_v0 / "results.tsv", parent_round=2) == 1

    def test_picks_max_plus_one(self, task_dir_with_results_v0: Path):
        # round 1 already has 1.1 → next is 1.2.
        assert next_rerun_seq(task_dir_with_results_v0 / "results.tsv", parent_round=1) == 2

    def test_missing_file(self, tmp_path: Path):
        assert next_rerun_seq(tmp_path / "noexist.tsv", parent_round=1) == 1


# --------------------------------------------------------------------------
# get_baseline_speed / get_baseline_peak_mb / get_baseline_status
# --------------------------------------------------------------------------

class TestGetBaselineSpeed:
    def test_v0_legacy_mode(self, task_dir_with_results_v0: Path):
        # No thread_mode column → every row reads as LEGACY. The fixture's
        # task.yaml has no modes block → active_mode resolves to LEGACY.
        speed = get_baseline_speed(task_dir_with_results_v0, "tiny_a")
        assert speed == 10.0

    def test_v2_specific_mode(self, task_dir_with_results_v2: Path):
        # Two baselines: default=10.0, parallel_t8=12.0.
        assert get_baseline_speed(task_dir_with_results_v2, "tiny_a", mode="default") == 10.0
        assert get_baseline_speed(task_dir_with_results_v2, "tiny_a", mode="parallel_t8") == 12.0

    def test_v2_default_resolves_to_active_mode(self, task_dir_with_results_v2: Path):
        # task.yaml has active_mode: default → mode=None should return 10.0.
        assert get_baseline_speed(task_dir_with_results_v2, "tiny_a") == 10.0

    def test_no_baseline_returns_none(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert get_baseline_speed(tmp_path, "missing_dataset") is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        assert get_baseline_speed(tmp_path, "tiny_a") is None


class TestGetBaselinePeakMb:
    def test_v0(self, task_dir_with_results_v0: Path):
        assert get_baseline_peak_mb(task_dir_with_results_v0, "tiny_a") == 512.0

    def test_v2_per_mode(self, task_dir_with_results_v2: Path):
        assert get_baseline_peak_mb(task_dir_with_results_v2, "tiny_a", mode="default") == 512.0
        assert get_baseline_peak_mb(task_dir_with_results_v2, "tiny_a", mode="parallel_t8") == 520.0


class TestGetBaselineStatus:
    def test_v0_returns_baseline(self, task_dir_with_results_v0: Path):
        assert get_baseline_status(task_dir_with_results_v0, "tiny_a") == "baseline"

    def test_no_match_returns_empty(self, task_dir_with_results_v0: Path):
        assert get_baseline_status(task_dir_with_results_v0, "nonexistent") == ""


# --------------------------------------------------------------------------
# best_speeds_at_dataset / find_best_speed_at_dataset / best_cv_at_dataset
# --------------------------------------------------------------------------

class TestBestSpeedsAtDataset:
    def test_no_best_ref(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        (tmp_path / ".zyme").mkdir()
        assert best_speeds_at_dataset(tmp_path, "tiny_a") == []

    def test_collects_keep_and_rerun(self, task_dir_with_results_v0: Path):
        # Best.ref points at 'def5678', which has keep + rerun rows.
        best_ref = task_dir_with_results_v0 / ".zyme" / "best.ref"
        best_ref.write_text("def5678901234567890\n")
        speeds = best_speeds_at_dataset(task_dir_with_results_v0, "tiny_a")
        assert sorted(speeds) == [8.0, 8.1]

    def test_excludes_discards(self, task_dir_with_results_v0: Path):
        # round-2 (ghi9012) was a discard → not included.
        best_ref = task_dir_with_results_v0 / ".zyme" / "best.ref"
        best_ref.write_text("ghi9012345678901234\n")
        speeds = best_speeds_at_dataset(task_dir_with_results_v0, "tiny_a")
        assert speeds == []


class TestFindBestSpeedAtDataset:
    def test_returns_median(self, task_dir_with_results_v0: Path):
        best_ref = task_dir_with_results_v0 / ".zyme" / "best.ref"
        best_ref.write_text("def5678901234567890\n")
        # Two speeds [8.0, 8.1] → median = 8.05.
        assert find_best_speed_at_dataset(task_dir_with_results_v0, "tiny_a") == pytest.approx(8.05)


class TestBestCvAtDataset:
    def test_n_below_3_returns_nones(self, task_dir_with_results_v0: Path):
        best_ref = task_dir_with_results_v0 / ".zyme" / "best.ref"
        best_ref.write_text("def5678901234567890\n")
        median, cv, n = best_cv_at_dataset(task_dir_with_results_v0, "tiny_a")
        assert median is None and cv is None and n == 2


# --------------------------------------------------------------------------
# dataset_in_results
# --------------------------------------------------------------------------

class TestDatasetInResults:
    def test_present_any_mode(self, task_dir_with_results_v0: Path):
        assert dataset_in_results(task_dir_with_results_v0 / "results.tsv", "tiny_a") is True

    def test_specific_mode_match(self, task_dir_with_results_v2: Path):
        assert dataset_in_results(
            task_dir_with_results_v2 / "results.tsv", "tiny_a", mode="parallel_t8"
        ) is True

    def test_specific_mode_no_match(self, task_dir_with_results_v2: Path):
        assert dataset_in_results(
            task_dir_with_results_v2 / "results.tsv", "tiny_a", mode="nonexistent_mode"
        ) is False

    def test_missing_file(self, tmp_path: Path):
        assert dataset_in_results(tmp_path / "noexist.tsv", "x") is False

    def test_legacy_treats_v0_as_legacy(self, task_dir_with_results_v0: Path):
        # mode=LEGACY_MODE should match all v0 rows (no thread_mode column).
        assert dataset_in_results(
            task_dir_with_results_v0 / "results.tsv", "tiny_a", mode=LEGACY_MODE
        ) is True


# --------------------------------------------------------------------------
# Schema migrations: ensure_results_schema, ensure_mode_schema
# --------------------------------------------------------------------------

class TestEnsureResultsSchema:
    def test_no_meta_yaml_skips(self, tmp_path: Path):
        # Without .zyme_meta.yaml, no migration even if results.tsv exists.
        tsv = tmp_path / "results.tsv"
        tsv.write_text("round\tcommit\n0\tabc\n")
        before = tsv.read_text()
        ensure_results_schema(tmp_path, tsv)
        assert tsv.read_text() == before

    def test_idempotent_when_prompt_id_present(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: p_iterate_x\n")
        tsv = tmp_path / "results.tsv"
        tsv.write_text("round\tcommit\tprompt_id\n0\tabc\tpid\n")
        before = tsv.read_text()
        ensure_results_schema(tmp_path, tsv)
        assert tsv.read_text() == before

    def test_appends_prompt_id_column(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: p_iterate_x\n")
        tsv = tmp_path / "results.tsv"
        tsv.write_text("round\tcommit\n0\tabc\n")
        ensure_results_schema(tmp_path, tsv)
        text = tsv.read_text()
        assert text.startswith("round\tcommit\tprompt_id\n")
        # Existing rows backfilled with empty prompt_id.
        assert "0\tabc\t\n" in text


class TestEnsureModeSchema:
    def test_adds_thread_mode_to_v0_results(self, task_dir_with_results_v0: Path):
        report = ensure_mode_schema(task_dir_with_results_v0)
        assert report["results.tsv"] == "migrated"
        text = (task_dir_with_results_v0 / "results.tsv").read_text()
        assert "thread_mode" in text.split("\n")[0]
        # Backfilled with LEGACY_MODE.
        assert LEGACY_MODE in text

    def test_idempotent_when_already_migrated(self, task_dir_with_results_v2: Path):
        report = ensure_mode_schema(task_dir_with_results_v2)
        assert report["results.tsv"] == "already"

    def test_creates_premode_bak(self, task_dir_with_results_v0: Path):
        ensure_mode_schema(task_dir_with_results_v0)
        bak = task_dir_with_results_v0 / "results.tsv.premode.bak"
        assert bak.exists()
        # bak should still be 11-col (pre-migration snapshot).
        first = bak.read_text().splitlines()[0]
        assert "thread_mode" not in first


# --------------------------------------------------------------------------
# results_header_for_task / append_prompt_id_to_row
# --------------------------------------------------------------------------

class TestResultsHeaderForTask:
    def test_no_meta_returns_base(self, tmp_path: Path):
        from zyme.utils import RESULTS_HEADER_BASE
        assert results_header_for_task(tmp_path) == RESULTS_HEADER_BASE

    def test_with_meta_returns_with_prompt_id(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: p_x\n")
        h = results_header_for_task(tmp_path)
        assert "prompt_id" in h


class TestAppendPromptIdToRow:
    def test_no_meta_passthrough(self, tmp_path: Path):
        row = "1\ta\tb\n"
        assert append_prompt_id_to_row(tmp_path, row) == row

    def test_appends_when_meta_present(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: p_x\n")
        # 12-tab row (11 cols + thread_mode + nothing). We pass an 11-tab row
        # (12 cols including thread_mode at end).
        row = "1\ta\tb\tc\td\te\tf\tg\th\ti\tj\tdefault\n"
        out = append_prompt_id_to_row(tmp_path, row)
        assert out.endswith("\tp_x\n")

    def test_idempotent_when_already_present(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: p_x\n")
        # Row already has 12+ tabs — no further append.
        row = "1\ta\tb\tc\td\te\tf\tg\th\ti\tj\tdefault\tpid\n"
        out = append_prompt_id_to_row(tmp_path, row)
        assert out == row
