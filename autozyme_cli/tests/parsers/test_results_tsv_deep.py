"""Deep coverage tests for zyme.parsers.results_tsv.

Targets the gaps not exercised by tests/parsers/test_results_tsv.py:
  - _format_results_row / append_results_row (header-order placement,
    prompt_id autofill)
  - _migrate_tsv_k2 / _normalize_phase_thread_order / ensure_k2_schema
    (thread-column insertion + reverse-order normalization + .bak snapshots)
  - thread-axis branches of the baseline/best query helpers
  - has_baseline_at_thread, best_wall_cpu_ratio_at_dataset,
    phase_speeds_at_dataset / phase_cv_at_dataset
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.parsers.results_tsv import (
    _format_results_row,
    _migrate_tsv_k2,
    _normalize_phase_thread_order,
    _resolve_thread_arg,
    append_results_row,
    best_cv_at_dataset,
    best_speeds_at_dataset,
    best_wall_cpu_ratio_at_dataset,
    dataset_in_results,
    ensure_k2_schema,
    find_best_speed_at_dataset,
    get_baseline_peak_mb,
    get_baseline_speed,
    get_baseline_status,
    has_baseline_at_thread,
    phase_cv_at_dataset,
    phase_speeds_at_dataset,
)
from zyme.utils import LEGACY_THREAD, RESULTS_HEADER_BASE


# K2 header (12 cols ending in thread), used by several tests below.
K2_HEADER = RESULTS_HEADER_BASE.rstrip("\n")


def _write(path: Path, header: str, *rows: str) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + ("\n" if rows else ""))


# --------------------------------------------------------------------------
# _format_results_row + append_results_row
# --------------------------------------------------------------------------

class TestFormatResultsRow:
    def test_places_values_by_header_order(self):
        header = "round\tcommit\tphase\tthread"
        out = _format_results_row(header, {"round": "1", "thread": "8",
                                           "commit": "abc", "phase": "optimize"})
        assert out == "1\tabc\toptimize\t8\n"

    def test_unknown_columns_render_empty(self):
        header = "round\tcommit\tthread"
        # 'commit' missing from values -> empty cell.
        out = _format_results_row(header, {"round": "2", "thread": "1"})
        assert out == "2\t\t1\n"

    def test_reverse_order_schema_places_correctly(self):
        # The load-bearing fix: header where thread precedes phase. The value
        # must still land under its own column name, not by sequence.
        header = "round\tcommit\tthread\tphase"
        out = _format_results_row(header, {"round": "3", "commit": "d",
                                           "thread": "4", "phase": "validate"})
        # thread=4 lands at position 2, phase=validate at position 3.
        assert out == "3\td\t4\tvalidate\n"


class TestAppendResultsRow:
    def test_creates_file_with_canonical_header(self, tmp_path: Path):
        tsv = tmp_path / "results.tsv"
        append_results_row(tsv, tmp_path, {"round": "1", "commit": "abc",
                                           "dataset": "tiny_a", "thread": "1"})
        text = tsv.read_text()
        assert text.startswith(RESULTS_HEADER_BASE)
        # Row appended under header order; thread is the last base column.
        assert "1\tabc\ttiny_a" in text
        assert text.rstrip().endswith("\t1")

    def test_autofills_prompt_id_for_bench_task(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: p_iterate_x\n")
        tsv = tmp_path / "results.tsv"
        append_results_row(tsv, tmp_path, {"round": "1", "commit": "abc",
                                           "dataset": "tiny_a", "thread": "1"})
        text = tsv.read_text()
        assert "prompt_id" in text.splitlines()[0]
        # The data row should carry the autofilled prompt_id in the last cell.
        assert text.rstrip().endswith("p_iterate_x")

    def test_caller_prompt_id_not_overwritten(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: from_meta\n")
        tsv = tmp_path / "results.tsv"
        append_results_row(tsv, tmp_path, {"round": "1", "commit": "abc",
                                           "dataset": "tiny_a", "thread": "1",
                                           "prompt_id": "explicit"})
        assert tsv.read_text().rstrip().endswith("explicit")

    def test_appends_using_existing_header_order(self, tmp_path: Path):
        # Pre-existing file with a non-canonical (reverse) column order.
        tsv = tmp_path / "results.tsv"
        _write(tsv, "round\tcommit\tdataset\tthread\tphase")
        append_results_row(tsv, tmp_path, {"round": "1", "commit": "c",
                                           "dataset": "d", "thread": "8",
                                           "phase": "optimize"})
        last = tsv.read_text().splitlines()[-1].split("\t")
        # thread(8) at idx 3, phase(optimize) at idx 4 — placed by name.
        assert last[3] == "8"
        assert last[4] == "optimize"


# --------------------------------------------------------------------------
# _migrate_tsv_k2
# --------------------------------------------------------------------------

class TestMigrateTsvK2:
    def test_absent_file(self, tmp_path: Path):
        assert _migrate_tsv_k2(tmp_path / "noexist.tsv") == "absent"

    def test_empty_file_is_absent(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("")
        assert _migrate_tsv_k2(p) == "absent"

    def test_already_has_thread(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        _write(p, K2_HEADER, "1\tc\td\t1\t0\t1\tkeep\t{}\th\tx\toptimize\t1")
        assert _migrate_tsv_k2(p) == "already"

    def test_inserts_thread_after_phase_and_backfills(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        header_no_thread = "\t".join([
            "round", "commit", "dataset", "speed_sec", "speedup_pct",
            "peak_mb", "status", "metrics_json", "hypothesis",
            "description", "phase",
        ])
        _write(p, header_no_thread,
               "0\tabc\ttiny_a\t10\t0\t512\tbaseline\t{}\tup\t\toptimize")
        assert _migrate_tsv_k2(p, insert_after="phase") == "migrated"
        lines = p.read_text().splitlines()
        cols = lines[0].split("\t")
        assert cols[-1] == "thread"  # inserted after phase (last col)
        # Backfilled to LEGACY_THREAD on existing rows.
        assert lines[1].split("\t")[-1] == str(LEGACY_THREAD)

    def test_leaves_prek2_bak_snapshot(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        _write(p, "round\tphase", "1\toptimize")
        _migrate_tsv_k2(p, insert_after="phase")
        bak = p.with_suffix(".tsv.prek2.bak")
        assert bak.exists()
        assert "thread" not in bak.read_text().splitlines()[0]

    def test_append_at_end_when_anchor_absent(self, tmp_path: Path):
        p = tmp_path / "history.tsv"
        _write(p, "tier\tname\tsec", "tiny\ta\t1.0")
        # insert_after=None -> appended at end of header.
        assert _migrate_tsv_k2(p, insert_after=None) == "migrated"
        assert p.read_text().splitlines()[0].split("\t")[-1] == "thread"

    def test_preserves_blank_lines(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("round\tphase\n1\toptimize\n\n2\toptimize\n")
        _migrate_tsv_k2(p, insert_after="phase")
        lines = p.read_text().splitlines()
        # The blank line is kept verbatim (no thread cell injected into it).
        assert "" in lines


# --------------------------------------------------------------------------
# _normalize_phase_thread_order
# --------------------------------------------------------------------------

class TestNormalizePhaseThreadOrder:
    def test_absent(self, tmp_path: Path):
        assert _normalize_phase_thread_order(tmp_path / "x.tsv") == "absent"

    def test_empty(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("")
        assert _normalize_phase_thread_order(p) == "absent"

    def test_na_when_no_thread_or_phase(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        _write(p, "round\tcommit", "1\tabc")
        assert _normalize_phase_thread_order(p) == "n/a"

    def test_already_canonical(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        _write(p, K2_HEADER, "1\tc\td\t1\t0\t1\tkeep\t{}\th\tx\toptimize\t1")
        assert _normalize_phase_thread_order(p) == "already"

    def test_normalizes_reverse_order(self, tmp_path: Path):
        # Header with thread BEFORE phase — older migration variant.
        p = tmp_path / "results.tsv"
        header = "round\tcommit\tthread\tphase"
        _write(p, header, "1\tabc\t8\toptimize")
        assert _normalize_phase_thread_order(p) == "normalized"
        cols = p.read_text().splitlines()[0].split("\t")
        assert cols.index("phase") < cols.index("thread")
        # Data row's two cells reordered too.
        row = p.read_text().splitlines()[1].split("\t")
        assert row[2] == "optimize"
        assert row[3] == "8"

    def test_leaves_precolfix_bak(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        _write(p, "round\tthread\tphase", "1\t4\toptimize")
        _normalize_phase_thread_order(p)
        assert p.with_suffix(".tsv.precolfix.bak").exists()


# --------------------------------------------------------------------------
# ensure_k2_schema
# --------------------------------------------------------------------------

class TestEnsureK2Schema:
    def test_migrates_results_and_reports(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        header_no_thread = "\t".join([
            "round", "commit", "dataset", "speed_sec", "speedup_pct",
            "peak_mb", "status", "metrics_json", "hypothesis",
            "description", "phase",
        ])
        _write(tmp_path / "results.tsv", header_no_thread,
               "0\tabc\ttiny_a\t10\t0\t512\tbaseline\t{}\tup\t\toptimize")
        report = ensure_k2_schema(tmp_path)
        assert report["results.tsv"] == "migrated"
        assert report["baselines_history.tsv"] == "absent"

    def test_normalized_supersedes(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        # results.tsv already has thread but in reverse order vs phase.
        _write(tmp_path / "results.tsv",
               "round\tcommit\tthread\tphase",
               "1\tabc\t8\toptimize")
        report = ensure_k2_schema(tmp_path)
        assert report["results.tsv"] == "normalized"


# --------------------------------------------------------------------------
# _resolve_thread_arg + thread-axis branches
# --------------------------------------------------------------------------

class TestResolveThreadArg:
    def test_none_is_legacy(self):
        assert _resolve_thread_arg(None) == LEGACY_THREAD

    def test_int_coerced(self):
        assert _resolve_thread_arg("8") == 8
        assert _resolve_thread_arg(4) == 4


class TestThreadAxisBaseline:
    def _setup(self, tmp_path: Path) -> Path:
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        _write(tmp_path / "results.tsv", K2_HEADER,
               # baseline at thread=1 and thread=8 (different speeds)
               "0\tabc\ttiny_a\t10\t0\t512\tbaseline\t{}\tup\t\toptimize\t1",
               "0\tabc\ttiny_a\t6\t0\t600\tbaseline\t{}\tup\t\toptimize\t8")
        return tmp_path

    def test_exact_thread_match(self, tmp_path: Path):
        d = self._setup(tmp_path)
        assert get_baseline_speed(d, "tiny_a", thread=8) == 6.0
        assert get_baseline_speed(d, "tiny_a", thread=1) == 10.0

    def test_falls_back_to_other_thread(self, tmp_path: Path):
        d = self._setup(tmp_path)
        # thread=99 has no exact baseline -> first baseline-class row wins.
        assert get_baseline_speed(d, "tiny_a", thread=99) == 10.0

    def test_peak_mb_thread_match(self, tmp_path: Path):
        d = self._setup(tmp_path)
        assert get_baseline_peak_mb(d, "tiny_a", thread=8) == 600.0

    def test_status_thread_match(self, tmp_path: Path):
        d = self._setup(tmp_path)
        assert get_baseline_status(d, "tiny_a", thread=8) == "baseline"

    def test_oom_status_baseline(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        _write(tmp_path / "results.tsv", K2_HEADER,
               "0\tabc\ttiny_a\t\t0\t\toom\t{}\tup\t\toptimize\t1")
        assert get_baseline_status(tmp_path, "tiny_a", thread=1) == "oom"


class TestHasBaselineAtThread:
    def test_true_for_exact(self, tmp_path: Path):
        _write(tmp_path / "results.tsv", K2_HEADER,
               "0\tabc\ttiny_a\t10\t0\t512\tbaseline\t{}\tup\t\toptimize\t4")
        assert has_baseline_at_thread(tmp_path, "tiny_a", 4) is True

    def test_false_for_other_thread(self, tmp_path: Path):
        _write(tmp_path / "results.tsv", K2_HEADER,
               "0\tabc\ttiny_a\t10\t0\t512\tbaseline\t{}\tup\t\toptimize\t1")
        assert has_baseline_at_thread(tmp_path, "tiny_a", 8) is False

    def test_missing_file(self, tmp_path: Path):
        assert has_baseline_at_thread(tmp_path, "tiny_a", 1) is False

    def test_no_status_column(self, tmp_path: Path):
        _write(tmp_path / "results.tsv", "round\tcommit", "0\tabc")
        assert has_baseline_at_thread(tmp_path, "tiny_a", 1) is False

    def test_only_header(self, tmp_path: Path):
        (tmp_path / "results.tsv").write_text(K2_HEADER + "\n")
        assert has_baseline_at_thread(tmp_path, "tiny_a", 1) is False


# --------------------------------------------------------------------------
# best_wall_cpu_ratio_at_dataset
# --------------------------------------------------------------------------

class TestBestWallCpuRatio:
    def test_no_best_ref(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        assert best_wall_cpu_ratio_at_dataset(tmp_path, "tiny_a") == (None, 0)

    def test_empty_best_ref(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        (tmp_path / ".zyme" / "best.ref").write_text("\n")
        assert best_wall_cpu_ratio_at_dataset(tmp_path, "tiny_a") == (None, 0)

    def test_n_below_2_returns_none(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        (tmp_path / ".zyme" / "best.ref").write_text("def5678\n")
        _write(tmp_path / "results.tsv", K2_HEADER,
               '1\tdef5678\ttiny_a\t8.0\t20\t500\tkeep\t{"cpu_sec": 4.0}\th\tx\toptimize\t1')
        median, n = best_wall_cpu_ratio_at_dataset(tmp_path, "tiny_a")
        assert median is None and n == 1

    def test_computes_median_ratio(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        (tmp_path / ".zyme" / "best.ref").write_text("def5678\n")
        _write(tmp_path / "results.tsv", K2_HEADER,
               '1\tdef5678\ttiny_a\t8.0\t20\t500\tkeep\t{"cpu_sec": 4.0}\th\tx\toptimize\t1',
               '1.1\tdef5678\ttiny_a\t9.0\t20\t500\trerun\t{"cpu_sec": 3.0}\th\tx\toptimize\t1')
        median, n = best_wall_cpu_ratio_at_dataset(tmp_path, "tiny_a")
        # ratios 8/4=2.0 and 9/3=3.0 -> median 2.5.
        assert median == pytest.approx(2.5)
        assert n == 2

    def test_skips_unparseable_metrics(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        (tmp_path / ".zyme" / "best.ref").write_text("def5678\n")
        _write(tmp_path / "results.tsv", K2_HEADER,
               '1\tdef5678\ttiny_a\t8.0\t20\t500\tkeep\tnotjson\th\tx\toptimize\t1',
               '1.1\tdef5678\ttiny_a\t9.0\t20\t500\trerun\t{"cpu_sec": 0}\th\tx\toptimize\t1')
        # First row metrics unparseable, second has cpu_sec=0 (<=0) -> both skipped.
        median, n = best_wall_cpu_ratio_at_dataset(tmp_path, "tiny_a")
        assert median is None and n == 0


# --------------------------------------------------------------------------
# phase_speeds_at_dataset / phase_cv_at_dataset
# --------------------------------------------------------------------------

class TestPhaseSpeeds:
    def test_missing_file(self, tmp_path: Path):
        assert phase_speeds_at_dataset(tmp_path, "tiny_a") == []

    def test_no_phase_column(self, tmp_path: Path):
        _write(tmp_path / "results.tsv",
               "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\t"
               "status\tmetrics_json\thypothesis\tdescription",
               "1\tc\ttiny_a\t8\t20\t500\tkeep\t{}\th\tx")
        assert phase_speeds_at_dataset(tmp_path, "tiny_a") == []

    def test_collects_validate_phase(self, tmp_path: Path):
        _write(tmp_path / "results.tsv", K2_HEADER,
               "1\tc\ttiny_a\t8.0\t20\t500\tkeep\t{}\th\tx\tvalidate\t1",
               "2\tc\ttiny_a\t8.5\t20\t500\trerun\t{}\th\tx\tvalidate\t1",
               "3\tc\ttiny_a\t9.0\t20\t500\tkeep\t{}\th\tx\toptimize\t1")
        speeds = phase_speeds_at_dataset(tmp_path, "tiny_a", phase="validate")
        assert sorted(speeds) == [8.0, 8.5]

    def test_skips_non_keep_rerun(self, tmp_path: Path):
        _write(tmp_path / "results.tsv", K2_HEADER,
               "1\tc\ttiny_a\t8.0\t20\t500\tdiscard\t{}\th\tx\tvalidate\t1")
        assert phase_speeds_at_dataset(tmp_path, "tiny_a", phase="validate") == []


class TestPhaseCv:
    def test_empty_returns_none_tuple(self, tmp_path: Path):
        assert phase_cv_at_dataset(tmp_path, "tiny_a") == (None, None, 0)

    def test_n_below_3_returns_median_only(self, tmp_path: Path):
        _write(tmp_path / "results.tsv", K2_HEADER,
               "1\tc\ttiny_a\t8.0\t20\t500\tkeep\t{}\th\tx\tvalidate\t1",
               "2\tc\ttiny_a\t10.0\t20\t500\trerun\t{}\th\tx\tvalidate\t1")
        median, cv, n = phase_cv_at_dataset(tmp_path, "tiny_a", phase="validate")
        assert median == 9.0  # mean of two sorted values
        assert cv is None
        assert n == 2

    def test_computes_cv_for_n_ge_3(self, tmp_path: Path):
        _write(tmp_path / "results.tsv", K2_HEADER,
               "1\tc\ttiny_a\t8.0\t20\t500\tkeep\t{}\th\tx\tvalidate\t1",
               "2\tc\ttiny_a\t10.0\t20\t500\trerun\t{}\th\tx\tvalidate\t1",
               "3\tc\ttiny_a\t12.0\t20\t500\trerun\t{}\th\tx\tvalidate\t1")
        median, cv, n = phase_cv_at_dataset(tmp_path, "tiny_a", phase="validate")
        assert median == 10.0
        assert cv is not None and cv > 0
        assert n == 3


# --------------------------------------------------------------------------
# thread-aware best + dataset_in_results
# --------------------------------------------------------------------------

class TestBestSpeedsThreadAxis:
    def test_filters_by_thread(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        (tmp_path / ".zyme" / "best.ref").write_text("def5678\n")
        _write(tmp_path / "results.tsv", K2_HEADER,
               "1\tdef5678\ttiny_a\t8.0\t20\t500\tkeep\t{}\th\tx\toptimize\t1",
               "1\tdef5678\ttiny_a\t6.0\t40\t500\tkeep\t{}\th\tx\toptimize\t8")
        # thread=8 should only collect the 6.0 row.
        assert best_speeds_at_dataset(tmp_path, "tiny_a", thread=8) == [6.0]
        assert find_best_speed_at_dataset(tmp_path, "tiny_a", thread=8) == 6.0

    def test_best_cv_thread(self, tmp_path: Path):
        (tmp_path / ".zyme").mkdir()
        (tmp_path / ".zyme" / "best.ref").write_text("def5678\n")
        _write(tmp_path / "results.tsv", K2_HEADER,
               "1\tdef5678\ttiny_a\t8.0\t20\t500\tkeep\t{}\th\tx\toptimize\t1",
               "1\tdef5678\ttiny_a\t9.0\t20\t500\trerun\t{}\th\tx\toptimize\t1",
               "1\tdef5678\ttiny_a\t10.0\t20\t500\trerun\t{}\th\tx\toptimize\t1")
        median, cv, n = best_cv_at_dataset(tmp_path, "tiny_a", thread=1)
        assert n == 3 and median == 9.0 and cv is not None


class TestDatasetInResultsThread:
    def test_thread_filter(self, tmp_path: Path):
        _write(tmp_path / "results.tsv", K2_HEADER,
               "1\tc\ttiny_a\t8.0\t20\t500\tkeep\t{}\th\tx\toptimize\t8")
        assert dataset_in_results(tmp_path / "results.tsv", "tiny_a", thread=8) is True
        assert dataset_in_results(tmp_path / "results.tsv", "tiny_a", thread=1) is False

    def test_no_dataset_column(self, tmp_path: Path):
        _write(tmp_path / "results.tsv", "round\tcommit", "1\tc")
        assert dataset_in_results(tmp_path / "results.tsv", "tiny_a") is False
