"""Wave-4 coverage for zyme.parsers.results_tsv — the scattered guard /
fallback / malformed-row branches left by test_results_tsv.py +
test_results_tsv_deep.py.

Targets:
  - _row_thread non-integer / short-row fallbacks
  - append_prompt_id_to_row idempotence + ensure_results_schema no-op paths
  - get_baseline_speed / _status / _peak_mb: missing-required-column guards,
    short-row skips, ValueError speed/peak skips, fallback-across-thread
  - best_speeds_at_dataset short-row / status / thread skips + empty best.ref
  - best_wall_cpu_ratio: short row, wall<=0, missing/zero cpu, json error
  - phase_speeds / phase_cv malformed rows
  - count_decision_rounds non-int + empty file
  - next_rerun_seq prefix-mismatch + non-int suffix
  - dataset_in_results short row / mode column / no-dataset-column
  - results_header_width / results_phase_index empty-file
  - update_last_status no-file / header-only / no-pending
"""
from __future__ import annotations

from pathlib import Path

from zyme.parsers.results_tsv import (
    _legacy_mode_target,
    _migrate_tsv_k2,
    _normalize_phase_thread_order,
    _row_thread,
    ensure_mode_schema,
    has_baseline_at_thread,
    append_prompt_id_to_row,
    best_speeds_at_dataset,
    best_wall_cpu_ratio_at_dataset,
    count_decision_rounds,
    dataset_in_results,
    ensure_results_schema,
    find_best_speed_at_dataset,
    get_baseline_peak_mb,
    get_baseline_speed,
    get_baseline_status,
    migrate_results_add_phase,
    next_rerun_seq,
    parse_log,
    phase_cv_at_dataset,
    phase_speeds_at_dataset,
    results_header_width,
    results_phase_index,
    update_last_status,
)
from zyme.utils import LEGACY_THREAD, RESULTS_HEADER_BASE

HDR = RESULTS_HEADER_BASE.rstrip("\n")
# cols: round commit dataset speed_sec speedup_pct peak_mb status
#       metrics_json hypothesis description phase thread


def _write(path: Path, *rows: str) -> None:
    body = "\n".join(rows)
    path.write_text(HDR + "\n" + body + ("\n" if rows else ""))


def _results(tmp_path: Path, *rows: str) -> Path:
    p = tmp_path / "results.tsv"
    _write(p, *rows)
    return p


# --------------------------------------------------------------------------
# _row_thread fallbacks
# --------------------------------------------------------------------------

class TestRowThread:
    def test_no_thread_column(self):
        # col dict without "thread" -> LEGACY_THREAD
        assert _row_thread(["1", "c"], {"round": 0}) == LEGACY_THREAD

    def test_short_row(self):
        col = {"thread": 11}
        assert _row_thread(["1", "c"], col) == LEGACY_THREAD

    def test_empty_cell(self):
        col = {"thread": 1}
        assert _row_thread(["1", ""], col) == LEGACY_THREAD

    def test_non_integer_cell(self):
        col = {"thread": 1}
        assert _row_thread(["1", "optimize"], col) == LEGACY_THREAD

    def test_valid_integer(self):
        col = {"thread": 1}
        assert _row_thread(["1", "8"], col) == 8


# --------------------------------------------------------------------------
# append_prompt_id_to_row + ensure_results_schema no-op paths
# --------------------------------------------------------------------------

class TestAppendPromptIdNoMeta:
    def test_no_meta_passthrough(self, tmp_path: Path):
        # No .zyme_meta.yaml -> row returned unchanged.
        out = append_prompt_id_to_row(tmp_path, "a\tb\tc\n")
        assert out == "a\tb\tc\n"


class TestEnsureResultsSchemaNoops:
    def test_no_meta_yaml_returns(self, tmp_path: Path):
        # No meta -> no migration even if file exists.
        rt = _results(tmp_path, "1\tc\td\t10\t0\t5\tkeep\t{}\t\t\toptimize\t1")
        before = rt.read_text()
        ensure_results_schema(tmp_path, rt)
        assert rt.read_text() == before

    def test_meta_but_missing_results_file(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: p1\n")
        rt = tmp_path / "results.tsv"
        # File absent -> early return, no crash, no file created.
        ensure_results_schema(tmp_path, rt)
        assert not rt.exists()

    def test_meta_empty_results_file(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: p1\n")
        rt = tmp_path / "results.tsv"
        rt.write_text("")  # exists but empty -> `if not lines: return`
        ensure_results_schema(tmp_path, rt)
        assert rt.read_text() == ""


# --------------------------------------------------------------------------
# get_baseline_speed / status / peak_mb guards
# --------------------------------------------------------------------------

class TestGetBaselineSpeedGuards:
    def test_missing_required_column_returns_none(self, tmp_path: Path):
        # Header without speed_sec -> None.
        p = tmp_path / "results.tsv"
        p.write_text("round\tcommit\tdataset\tstatus\n0\tc\td\tbaseline\n")
        assert get_baseline_speed(tmp_path, "d") is None

    def test_short_row_skipped_then_fallback(self, tmp_path: Path):
        # First data row is too short to read speed/status -> skipped; the
        # next valid baseline row at a different thread becomes the fallback.
        _results(
            tmp_path,
            "0\tc",  # short row (only 2 cells) -> skip
            "0\tc\td\t12.5\t0\t5\tbaseline\t{}\t\t\toptimize\t4",
        )
        # Asking for thread=1 -> no exact match, fallback to the thread=4 row.
        assert get_baseline_speed(tmp_path, "d", thread=1) == 12.5

    def test_unparseable_speed_skipped(self, tmp_path: Path):
        _results(
            tmp_path,
            "0\tc\td\tnot-a-float\t0\t5\tbaseline\t{}\t\t\toptimize\t1",
        )
        assert get_baseline_speed(tmp_path, "d", thread=1) is None

    def test_non_baseline_status_skipped(self, tmp_path: Path):
        _results(
            tmp_path,
            "1\tc\td\t9\t0\t5\tkeep\t{}\t\t\toptimize\t1",
        )
        assert get_baseline_speed(tmp_path, "d", thread=1) is None

    def test_dataset_mismatch_skipped(self, tmp_path: Path):
        _results(
            tmp_path,
            "0\tc\tother\t9\t0\t5\tbaseline\t{}\t\t\toptimize\t1",
        )
        assert get_baseline_speed(tmp_path, "d", thread=1) is None


class TestGetBaselineStatusGuards:
    def test_missing_file(self, tmp_path: Path):
        assert get_baseline_status(tmp_path, "d") == ""

    def test_header_only(self, tmp_path: Path):
        _results(tmp_path)
        assert get_baseline_status(tmp_path, "d") == ""

    def test_no_status_column(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("round\tcommit\tdataset\n0\tc\td\n")
        assert get_baseline_status(tmp_path, "d") == ""

    def test_short_row_skipped(self, tmp_path: Path):
        _results(tmp_path, "0\tc")  # too short -> skip; no baseline -> ""
        assert get_baseline_status(tmp_path, "d") == ""

    def test_fallback_across_thread(self, tmp_path: Path):
        _results(
            tmp_path,
            "0\tc\td\t9\t0\t5\toom\t{}\t\t\toptimize\t4",
        )
        # thread=1 not found -> falls back to the oom row at thread=4.
        assert get_baseline_status(tmp_path, "d", thread=1) == "oom"


class TestGetBaselinePeakMbGuards:
    def test_missing_peak_column(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("round\tcommit\tdataset\tstatus\n0\tc\td\tbaseline\n")
        assert get_baseline_peak_mb(tmp_path, "d") is None

    def test_unparseable_peak_skipped(self, tmp_path: Path):
        _results(
            tmp_path,
            "0\tc\td\t9\t0\tNaNish\tbaseline\t{}\t\t\toptimize\t1",
        )
        # peak_mb cell unparseable -> skipped -> None.
        assert get_baseline_peak_mb(tmp_path, "d", thread=1) is None

    def test_fallback_across_thread(self, tmp_path: Path):
        _results(
            tmp_path,
            "0\tc\td\t9\t0\t512\tbaseline\t{}\t\t\toptimize\t4",
        )
        assert get_baseline_peak_mb(tmp_path, "d", thread=1) == 512.0

    def test_missing_file(self, tmp_path: Path):
        assert get_baseline_peak_mb(tmp_path, "d") is None


# --------------------------------------------------------------------------
# best_speeds_at_dataset guards
# --------------------------------------------------------------------------

class TestBestSpeedsGuards:
    def _best_ref(self, tmp_path: Path, sha: str) -> None:
        z = tmp_path / ".zyme"
        z.mkdir(exist_ok=True)
        (z / "best.ref").write_text(sha)

    def test_empty_best_ref_returns_empty(self, tmp_path: Path):
        self._best_ref(tmp_path, "   ")
        _results(tmp_path, "1\tabcdef0\td\t8\t0\t5\tkeep\t{}\t\t\toptimize\t1")
        assert best_speeds_at_dataset(tmp_path, "d") == []

    def test_missing_results_returns_empty(self, tmp_path: Path):
        self._best_ref(tmp_path, "abcdef0123")
        # best.ref present but no results.tsv.
        assert best_speeds_at_dataset(tmp_path, "d") == []

    def test_short_rows_and_status_and_thread_skips(self, tmp_path: Path):
        self._best_ref(tmp_path, "abcdef0123")
        _results(
            tmp_path,
            "1\tabcdef0",                                    # <7 cells -> skip
            "1\tabcdef0\td\t8\t0\t5\tdiscard\t{}\t\t\to\t1",  # not keep/rerun
            "1\tabcdef0\td\t7\t0\t5\tkeep\t{}\t\t\to\t8",     # wrong thread
            "1\tabcdef0\td\tnotnum\t0\t5\tkeep\t{}\t\t\to\t1",  # unparseable speed
            "1\tabcdef0\td\t6.5\t0\t5\tkeep\t{}\t\t\to\t1",   # the one kept
        )
        assert best_speeds_at_dataset(tmp_path, "d", thread=1) == [6.5]

    def test_header_only_returns_empty(self, tmp_path: Path):
        self._best_ref(tmp_path, "abcdef0123")
        _results(tmp_path)  # header only
        assert best_speeds_at_dataset(tmp_path, "d") == []


# --------------------------------------------------------------------------
# best_wall_cpu_ratio_at_dataset guards
# --------------------------------------------------------------------------

class TestBestWallCpuRatioGuards:
    def _best(self, tmp_path: Path) -> None:
        z = tmp_path / ".zyme"
        z.mkdir(exist_ok=True)
        (z / "best.ref").write_text("abcdef0123")

    def test_all_skip_branches(self, tmp_path: Path):
        self._best(tmp_path)
        _results(
            tmp_path,
            "1\tabcdef0\td",                                       # <8 cells -> skip
            "1\twrongsha\td\t8\t0\t5\tkeep\t{\"cpu_sec\":4}\t\t\to\t1",   # commit mismatch
            "1\tabcdef0\tother\t8\t0\t5\tkeep\t{\"cpu_sec\":4}\t\t\to\t1",  # dataset mismatch
            "1\tabcdef0\td\t8\t0\t5\tdiscard\t{\"cpu_sec\":4}\t\t\to\t1",   # not keep/rerun
            "1\tabcdef0\td\t8\t0\t5\tkeep\t{\"cpu_sec\":4}\t\t\to\t8",       # wrong thread
            "1\tabcdef0\td\tnotnum\t0\t5\tkeep\t{\"cpu_sec\":4}\t\t\to\t1",  # wall ValueError
            "1\tabcdef0\td\t0\t0\t5\tkeep\t{\"cpu_sec\":4}\t\t\to\t1",       # wall<=0
            "1\tabcdef0\td\t8\t0\t5\tkeep\t{}\t\t\to\t1",          # no cpu_sec -> skip
            "1\tabcdef0\td\t8\t0\t5\tkeep\t{\"cpu_sec\":0}\t\t\to\t1",  # cpu<=0 -> skip
            "1\tabcdef0\td\t8\t0\t5\tkeep\tnot-json\t\t\to\t1",     # json error -> skip
        )
        # No usable ratios -> (None, 0).
        assert best_wall_cpu_ratio_at_dataset(tmp_path, "d", thread=1) == (None, 0)

    def test_n_below_2(self, tmp_path: Path):
        self._best(tmp_path)
        _results(
            tmp_path,
            "1\tabcdef0\td\t8\t0\t5\tkeep\t{\"cpu_sec\":4}\t\t\to\t1",
        )
        ratio, n = best_wall_cpu_ratio_at_dataset(tmp_path, "d", thread=1)
        assert ratio is None and n == 1

    def test_computes_median(self, tmp_path: Path):
        self._best(tmp_path)
        _results(
            tmp_path,
            "1\tabcdef0\td\t8\t0\t5\tkeep\t{\"cpu_sec\":4}\t\t\to\t1",
            "2\tabcdef0\td\t12\t0\t5\trerun\t{\"cpu_sec\":4}\t\t\to\t1",
        )
        ratio, n = best_wall_cpu_ratio_at_dataset(tmp_path, "d", thread=1)
        assert n == 2
        assert ratio == 2.5  # median of [2.0, 3.0]

    def test_cpu_non_numeric_skipped(self, tmp_path: Path):
        self._best(tmp_path)
        _results(
            tmp_path,
            "1\tabcdef0\td\t8\t0\t5\tkeep\t{\"cpu_sec\":\"abc\"}\t\t\to\t1",
        )
        assert best_wall_cpu_ratio_at_dataset(tmp_path, "d", thread=1) == (None, 0)


# --------------------------------------------------------------------------
# phase_speeds / phase_cv malformed rows
# --------------------------------------------------------------------------

class TestPhaseSpeedsGuards:
    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("")
        assert phase_speeds_at_dataset(tmp_path, "d") == []

    def test_short_row_and_status_and_unparseable(self, tmp_path: Path):
        _results(
            tmp_path,
            "1\tc\td\t8",                                   # < len(header) -> skip
            "1\tc\td\t8\t0\t5\tdiscard\t{}\t\t\tvalidate\t1",  # not keep/rerun
            "1\tc\td\tnotnum\t0\t5\tkeep\t{}\t\t\tvalidate\t1",  # unparseable speed
            "1\tc\td\t7\t0\t5\tkeep\t{}\t\t\tvalidate\t1",     # the kept one
        )
        assert phase_speeds_at_dataset(tmp_path, "d", phase="validate") == [7.0]

    def test_wrong_phase_skipped(self, tmp_path: Path):
        _results(
            tmp_path,
            "1\tc\td\t7\t0\t5\tkeep\t{}\t\t\toptimize\t1",
        )
        assert phase_speeds_at_dataset(tmp_path, "d", phase="validate") == []


class TestPhaseCv:
    def test_empty_returns_none_tuple(self, tmp_path: Path):
        _results(tmp_path)
        assert phase_cv_at_dataset(tmp_path, "d", phase="validate") == (None, None, 0)

    def test_n_below_3_median_only(self, tmp_path: Path):
        _results(
            tmp_path,
            "1\tc\td\t8\t0\t5\tkeep\t{}\t\t\tvalidate\t1",
            "2\tc\td\t10\t0\t5\tkeep\t{}\t\t\tvalidate\t1",
        )
        median, cv, n = phase_cv_at_dataset(tmp_path, "d", phase="validate")
        assert median == 9.0 and cv is None and n == 2

    def test_cv_for_n_ge_3(self, tmp_path: Path):
        _results(
            tmp_path,
            "1\tc\td\t8\t0\t5\tkeep\t{}\t\t\tvalidate\t1",
            "2\tc\td\t9\t0\t5\tkeep\t{}\t\t\tvalidate\t1",
            "3\tc\td\t10\t0\t5\tkeep\t{}\t\t\tvalidate\t1",
        )
        median, cv, n = phase_cv_at_dataset(tmp_path, "d", phase="validate")
        assert median == 9.0 and n == 3 and cv is not None and cv > 0


# --------------------------------------------------------------------------
# count_decision_rounds + next_rerun_seq
# --------------------------------------------------------------------------

class TestCountDecisionRounds:
    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("")
        assert count_decision_rounds(p) == 0

    def test_non_integer_round_skipped(self, tmp_path: Path):
        p = _results(
            tmp_path,
            "1.1\tc\td\t8\t0\t5\trerun\t{}\t\t\toptimize\t1",  # .k suffix skip
            "0\tc\td\t8\t0\t5\tbaseline\t{}\t\t\toptimize\t1",  # round 0 skip
            "2\tc\td\t8\t0\t5\tkeep\t{}\t\t\toptimize\t1",      # counts
        )
        assert count_decision_rounds(p, phase="optimize") == 1

    def test_phase_all_ignores_phase(self, tmp_path: Path):
        p = _results(
            tmp_path,
            "1\tc\td\t8\t0\t5\tkeep\t{}\t\t\tvalidate\t1",
            "2\tc\td\t8\t0\t5\tkeep\t{}\t\t\toptimize\t1",
        )
        assert count_decision_rounds(p, phase="all") == 2


class TestNextRerunSeq:
    def test_missing_file_returns_one(self, tmp_path: Path):
        assert next_rerun_seq(tmp_path / "nope.tsv", 5) == 1

    def test_prefix_mismatch_and_non_int_suffix(self, tmp_path: Path):
        p = _results(
            tmp_path,
            "5\tc\td\t8\t0\t5\tkeep\t{}\t\t\to\t1",       # exact "5" -> no prefix "5."
            "50.1\tc\td\t8\t0\t5\trerun\t{}\t\t\to\t1",   # different parent prefix
            "5.x\tc\td\t8\t0\t5\trerun\t{}\t\t\to\t1",    # non-int suffix
            "5.2\tc\td\t8\t0\t5\trerun\t{}\t\t\to\t1",    # the real max
        )
        assert next_rerun_seq(p, 5) == 3


# --------------------------------------------------------------------------
# dataset_in_results guards
# --------------------------------------------------------------------------

class TestDatasetInResults:
    def test_missing_file(self, tmp_path: Path):
        assert dataset_in_results(tmp_path / "nope.tsv", "d") is False

    def test_header_only(self, tmp_path: Path):
        p = _results(tmp_path)
        assert dataset_in_results(p, "d") is False

    def test_no_dataset_column(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("round\tcommit\n1\tc\n")
        assert dataset_in_results(p, "d") is False

    def test_short_row_skipped(self, tmp_path: Path):
        p = _results(tmp_path, "1\tc")  # too short to read dataset col
        assert dataset_in_results(p, "d") is False

    def test_thread_filter_no_match(self, tmp_path: Path):
        p = _results(
            tmp_path,
            "1\tc\td\t8\t0\t5\tkeep\t{}\t\t\to\t8",
        )
        # dataset present but thread=1 requested, row is thread=8 -> False.
        assert dataset_in_results(p, "d", thread=1) is False
        # thread=8 -> match.
        assert dataset_in_results(p, "d", thread=8) is True


# --------------------------------------------------------------------------
# results_header_width / results_phase_index empty-file
# --------------------------------------------------------------------------

class TestHeaderWidthAndPhaseIndex:
    def test_header_width_missing_and_empty(self, tmp_path: Path):
        assert results_header_width(tmp_path / "nope.tsv") == 0
        p = tmp_path / "results.tsv"
        p.write_text("")
        assert results_header_width(p) == 0

    def test_header_width_counts(self, tmp_path: Path):
        p = _results(tmp_path)
        assert results_header_width(p) == 12

    def test_phase_index_missing_and_empty(self, tmp_path: Path):
        assert results_phase_index(tmp_path / "nope.tsv") is None
        p = tmp_path / "results.tsv"
        p.write_text("")
        assert results_phase_index(p) is None

    def test_phase_index_absent_column(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("round\tcommit\tdataset\n1\tc\td\n")
        assert results_phase_index(p) is None

    def test_phase_index_present(self, tmp_path: Path):
        p = _results(tmp_path)
        assert results_phase_index(p) == 10


# --------------------------------------------------------------------------
# update_last_status edge branches
# --------------------------------------------------------------------------

class TestUpdateLastStatus:
    def test_missing_file_noop(self, tmp_path: Path):
        # Must not raise.
        update_last_status(tmp_path / "nope.tsv", "keep", "desc")

    def test_header_only_noop(self, tmp_path: Path):
        p = _results(tmp_path)
        update_last_status(p, "keep", "desc")
        assert p.read_text() == HDR + "\n"

    def test_no_pending_row_noop(self, tmp_path: Path):
        p = _results(
            tmp_path,
            "1\tc\td\t8\t0\t5\tkeep\t{}\t\t\to\t1",
        )
        before = p.read_text()
        update_last_status(p, "discard", "x")
        assert p.read_text() == before

    def test_flips_most_recent_pending(self, tmp_path: Path):
        p = _results(
            tmp_path,
            "1\tc\td\t8\t0\t5\tpending\t{}\t\thypo\toptimize\t1",
            "2\tc2\td\t7\t0\t5\trerun\t{}\t\t\toptimize\t1",
        )
        update_last_status(p, "keep", "my decision")
        lines = p.read_text().splitlines()
        parts = lines[1].split("\t")
        assert parts[6] == "keep"
        assert parts[9] == "my decision"
        # rerun row untouched.
        assert lines[2].split("\t")[6] == "rerun"

    def test_pads_short_pending_row(self, tmp_path: Path):
        # A pending row shorter than the header width is padded before its
        # status cell is set (line 910 `parts.append("")`).
        p = tmp_path / "results.tsv"
        p.write_text(HDR + "\n1\tc\td\t8\t0\t5\tpending\n")
        update_last_status(p, "keep", "")
        parts = p.read_text().splitlines()[1].split("\t")
        assert parts[6] == "keep"
        assert len(parts) >= 12


# --------------------------------------------------------------------------
# _legacy_mode_target — parse_active_mode exception -> LEGACY_MODE
# --------------------------------------------------------------------------

class TestLegacyModeTarget:
    def test_no_mode_column_returns_none(self, tmp_path: Path):
        # col has neither thread_mode nor mode -> None (no legacy filter).
        assert _legacy_mode_target(tmp_path, None, {"round": 0}) is None

    def test_explicit_mode_str(self, tmp_path: Path):
        col = {"thread_mode": 5}
        assert _legacy_mode_target(tmp_path, "fast", col) == "fast"

    def test_parse_active_mode_exception(self, tmp_path: Path, monkeypatch):
        import zyme.parsers.task_yaml as ty

        def boom(_path):
            raise RuntimeError("bad yaml")
        monkeypatch.setattr(ty, "parse_active_mode", boom)
        col = {"mode": 5}
        # mode is None -> tries parse_active_mode -> raises -> LEGACY_MODE.
        from zyme.utils import LEGACY_MODE
        assert _legacy_mode_target(tmp_path, None, col) == LEGACY_MODE


# --------------------------------------------------------------------------
# legacy thread_mode column path on the baseline readers
# --------------------------------------------------------------------------

class TestLegacyModeColumnReaders:
    """A pre-K2 file carrying a thread_mode column exercises the
    `_matches_legacy_mode` skip branches in the baseline readers."""

    def _v2(self, tmp_path: Path, *rows: str) -> Path:
        # header with thread_mode appended after thread.
        hdr = HDR + "\tthread_mode"
        p = tmp_path / "results.tsv"
        p.write_text(hdr + "\n" + "\n".join(rows) + "\n")
        return p

    def test_baseline_speed_mode_mismatch_skipped(self, tmp_path: Path):
        self._v2(
            tmp_path,
            "0\tc\td\t9\t0\t5\tbaseline\t{}\t\t\toptimize\t1\tslow",
        )
        # Looking for mode "fast" -> the only baseline row is "slow" -> None.
        assert get_baseline_speed(tmp_path, "d", thread=1, mode="fast") is None
        # Matching mode -> found.
        assert get_baseline_speed(tmp_path, "d", thread=1, mode="slow") == 9.0

    def test_baseline_status_mode_mismatch(self, tmp_path: Path):
        self._v2(
            tmp_path,
            "0\tc\td\t9\t0\t5\toom\t{}\t\t\toptimize\t1\tslow",
        )
        assert get_baseline_status(tmp_path, "d", thread=1, mode="fast") == ""
        assert get_baseline_status(tmp_path, "d", thread=1, mode="slow") == "oom"

    def test_baseline_peak_mode_mismatch(self, tmp_path: Path):
        self._v2(
            tmp_path,
            "0\tc\td\t9\t0\t512\tbaseline\t{}\t\t\toptimize\t1\tslow",
        )
        assert get_baseline_peak_mb(tmp_path, "d", thread=1, mode="fast") is None
        assert get_baseline_peak_mb(tmp_path, "d", thread=1, mode="slow") == 512.0

    def test_dataset_in_results_mode_mismatch(self, tmp_path: Path):
        p = self._v2(
            tmp_path,
            "1\tc\tother\t9\t0\t5\tkeep\t{}\t\t\toptimize\t1\tslow",  # mode-match, dataset miss
            "1\tc\td\t9\t0\t5\tkeep\t{}\t\t\toptimize\t1\tfast",      # mode miss
            "1\tc\td\t9\t0\t5\tkeep\t{}\t\t\toptimize\t1\tslow",      # the match
        )
        # mode='fast' excludes the 'slow' rows; the 'fast' row has dataset 'd'
        # so it matches when mode='fast' is requested.
        assert dataset_in_results(p, "d", mode="fast") is True
        # mode='slow': first row dataset mismatch (line 1024), last row matches.
        assert dataset_in_results(p, "d", mode="slow") is True
        # mode='nonexistent': all rows filtered out by mode -> False.
        assert dataset_in_results(p, "d", mode="nope") is False


# --------------------------------------------------------------------------
# header-only guards on the remaining readers
# --------------------------------------------------------------------------

class TestHeaderOnlyGuards:
    def test_baseline_speed_header_only(self, tmp_path: Path):
        _results(tmp_path)
        assert get_baseline_speed(tmp_path, "d") is None

    def test_baseline_peak_header_only(self, tmp_path: Path):
        _results(tmp_path)
        assert get_baseline_peak_mb(tmp_path, "d") is None

    def test_find_best_speed_no_data(self, tmp_path: Path):
        # No best.ref -> best_speeds empty -> find_best_speed returns None.
        assert find_best_speed_at_dataset(tmp_path, "d") is None


# --------------------------------------------------------------------------
# parse_log
# --------------------------------------------------------------------------

class TestParseLog:
    def test_extracts_and_skips(self):
        log = (
            "task:             mytask\n"          # in skip set
            "speed_sec:        12.5\n"
            "peak_memory_mb:   2048\n"
            "internal_time:    5.0\n"             # in skip set
            "cpu_sec:          11.0\n"            # numeric metric
            "note:             not a number\n"    # string -> dropped
            "random text without colon\n"        # no match -> skip
        )
        speed, peak, metrics, status = parse_log(log)
        assert speed == 12.5
        assert peak == 2048.0
        assert metrics == {"cpu_sec": 11.0}
        assert "note" not in metrics
        assert status == "pending"

    def test_crash_status(self):
        log = "status:           crash\nspeed_sec:        9.0\n"
        speed, peak, metrics, status = parse_log(log)
        assert status == "crash"

    def test_unparseable_speed_and_peak_left_none(self):
        log = "speed_sec:        notnum\npeak_mb:          alsonan\n"
        speed, peak, metrics, status = parse_log(log)
        assert speed is None
        assert peak is None

    def test_peak_mb_alias(self):
        log = "peak_mb:          777\n"
        _, peak, _, _ = parse_log(log)
        assert peak == 777.0


# --------------------------------------------------------------------------
# migrate_results_add_phase
# --------------------------------------------------------------------------

class TestMigrateResultsAddPhase:
    def test_missing_file_noop(self, tmp_path: Path):
        migrate_results_add_phase(tmp_path / "nope.tsv")  # no raise

    def test_empty_file_noop(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text("")
        migrate_results_add_phase(p)
        assert p.read_text() == ""

    def test_already_has_phase_noop(self, tmp_path: Path):
        p = _results(tmp_path, "1\tc\td\t8\t0\t5\tkeep\t{}\t\t\toptimize\t1")
        before = p.read_text()
        migrate_results_add_phase(p)
        assert p.read_text() == before

    def test_adds_phase_column(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\n"
            "1\tc\td\t8\t20\t5\tkeep\n"
        )
        migrate_results_add_phase(p)
        lines = p.read_text().splitlines()
        assert lines[0].endswith("\tphase")
        # data row gets an empty trailing phase cell.
        assert lines[1].endswith("\t")


# --------------------------------------------------------------------------
# count_decision_rounds: no phase column + empty parts
# --------------------------------------------------------------------------

class TestCountDecisionRoundsNoPhase:
    def test_no_phase_column_optimize_default(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\n"
            "1\tc\td\t8\t20\t5\tkeep\n"
            "2\tc\td\t7\t30\t5\tkeep\n"
        )
        # No phase column -> rows treated as optimize.
        assert count_decision_rounds(p, phase="optimize") == 2

    def test_blank_data_line_skipped(self, tmp_path: Path):
        # A blank line in the body must be skipped without crashing.
        p = tmp_path / "results.tsv"
        p.write_text(HDR + "\n\n1\tc\td\t8\t0\t5\tkeep\t{}\t\t\toptimize\t1\n")
        assert count_decision_rounds(p, phase="optimize") == 1


# --------------------------------------------------------------------------
# has_baseline_at_thread skip branches
# --------------------------------------------------------------------------

class TestHasBaselineAtThreadSkips:
    def test_short_row_status_and_dataset_skips(self, tmp_path: Path):
        p = _results(
            tmp_path,
            "0\tc",                                              # < status idx -> skip
            "1\tc\td\t8\t0\t5\tkeep\t{}\t\t\to\t1",              # not baseline/oom
            "0\tc\tother\t8\t0\t5\tbaseline\t{}\t\t\to\t1",      # dataset mismatch
            "0\tc\td\t8\t0\t5\tbaseline\t{}\t\t\to\t1",          # the match
        )
        assert has_baseline_at_thread(p.parent, "d", 1) is True
        # No baseline at thread=8 in this file.
        assert has_baseline_at_thread(p.parent, "d", 8) is False


# --------------------------------------------------------------------------
# get_baseline_peak_mb short-row + status skips
# --------------------------------------------------------------------------

class TestGetBaselinePeakSkips:
    def test_short_row_and_status_skips_then_match(self, tmp_path: Path):
        _results(
            tmp_path,
            "0\tc",                                              # short row -> skip
            "1\tc\td\t8\t0\t512\tkeep\t{}\t\t\to\t1",            # not baseline/oom
            "0\tc\tother\t8\t0\t99\tbaseline\t{}\t\t\to\t1",     # dataset mismatch
            "0\tc\td\t8\t0\t256\tbaseline\t{}\t\t\to\t1",        # the match
        )
        assert get_baseline_peak_mb(tmp_path, "d", thread=1) == 256.0


# --------------------------------------------------------------------------
# best_speeds_at_dataset legacy-mode skip (line 594)
# --------------------------------------------------------------------------

class TestBestSpeedsLegacyMode:
    def test_mode_mismatch_skipped(self, tmp_path: Path):
        z = tmp_path / ".zyme"
        z.mkdir()
        (z / "best.ref").write_text("abcdef0123")
        hdr = HDR + "\tthread_mode"
        p = tmp_path / "results.tsv"
        p.write_text(
            hdr + "\n"
            "1\tabcdef0\td\t8\t0\t5\tkeep\t{}\t\t\to\t1\tslow\n"
        )
        # mode 'fast' filter excludes the only 'slow' row.
        assert best_speeds_at_dataset(tmp_path, "d", thread=1, mode="fast") == []
        assert best_speeds_at_dataset(tmp_path, "d", thread=1, mode="slow") == [8.0]


# --------------------------------------------------------------------------
# phase_speeds_at_dataset short-row skip (line 633)
# --------------------------------------------------------------------------

class TestPhaseSpeedsShortRow:
    def test_short_row_and_dataset_mismatch_skipped(self, tmp_path: Path):
        # short row (len < header), dataset mismatch, then a full matching row.
        p = tmp_path / "results.tsv"
        p.write_text(
            HDR + "\n"
            "1\tc\td\t7\n"                                       # too short -> skip
            "1\tc\tother\t7\t0\t5\tkeep\t{}\t\t\tvalidate\t1\n"  # dataset mismatch
            "1\tc\td\t7\t0\t5\tkeep\t{}\t\t\tvalidate\t1\n"      # full -> kept
        )
        assert phase_speeds_at_dataset(tmp_path, "d", phase="validate") == [7.0]


# --------------------------------------------------------------------------
# best_wall_cpu_ratio: no/empty best.ref (lines 706, 709)
# --------------------------------------------------------------------------

class TestBestWallCpuRefGuards:
    def test_no_best_ref(self, tmp_path: Path):
        assert best_wall_cpu_ratio_at_dataset(tmp_path, "d") == (None, 0)

    def test_empty_best_ref(self, tmp_path: Path):
        z = tmp_path / ".zyme"
        z.mkdir()
        (z / "best.ref").write_text("   ")
        assert best_wall_cpu_ratio_at_dataset(tmp_path, "d") == (None, 0)

    def test_best_ref_but_no_results_file(self, tmp_path: Path):
        z = tmp_path / ".zyme"
        z.mkdir()
        (z / "best.ref").write_text("abcdef0123")
        # best.ref set but results.tsv absent -> (None, 0) (line 706).
        assert best_wall_cpu_ratio_at_dataset(tmp_path, "d") == (None, 0)

    def test_best_ref_header_only_results(self, tmp_path: Path):
        z = tmp_path / ".zyme"
        z.mkdir()
        (z / "best.ref").write_text("abcdef0123")
        _results(tmp_path)  # header only -> len(lines) < 2 (line 709).
        assert best_wall_cpu_ratio_at_dataset(tmp_path, "d") == (None, 0)

    def test_legacy_mode_skip(self, tmp_path: Path):
        # thread_mode column + mode filter mismatch -> the legacy-mode skip
        # (line 720) fires for the only candidate row.
        z = tmp_path / ".zyme"
        z.mkdir()
        (z / "best.ref").write_text("abcdef0123")
        hdr = HDR + "\tthread_mode"
        p = tmp_path / "results.tsv"
        p.write_text(
            hdr + "\n"
            "1\tabcdef0\td\t8\t0\t5\tkeep\t{\"cpu_sec\":4}\t\t\to\t1\tslow\n"
        )
        # mode='fast' -> the 'slow' row is filtered out -> no ratios.
        assert best_wall_cpu_ratio_at_dataset(tmp_path, "d", thread=1, mode="fast") == (None, 0)


# --------------------------------------------------------------------------
# next_rerun_seq blank-line skip (line 981)
# --------------------------------------------------------------------------

class TestNextRerunSeqBlankLine:
    def test_blank_line_skipped(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(HDR + "\n\n5.1\tc\td\t8\t0\t5\trerun\t{}\t\t\to\t1\n")
        assert next_rerun_seq(p, 5) == 2


# --------------------------------------------------------------------------
# dataset_in_results legacy-mode skip already covered; ensure short-circuit
# blank line behaviour is fine (line 1024 mode skip covered above).
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# migration helpers: blank-line + short-row padding + mode migration
# --------------------------------------------------------------------------

class TestMigrationEdges:
    def test_k2_pads_short_data_row(self, tmp_path: Path):
        # A pre-K2 results.tsv (no thread col) with a short data row: migration
        # pads the row to header width before inserting the thread cell.
        p = tmp_path / "results.tsv"
        p.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
            "1\tc\td\t8\n"  # short data row
        )
        res = _migrate_tsv_k2(p, insert_after="phase")
        assert res == "migrated"
        assert "thread" in p.read_text().splitlines()[0]

    def test_normalize_preserves_blank_and_pads(self, tmp_path: Path):
        # Reverse-order [..., thread, phase] header with a blank line + short
        # row -> normalized; blank preserved, short row padded.
        p = tmp_path / "results.tsv"
        p.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tthread\tphase\n"
            "\n"                                # blank line preserved
            "1\tc\td\t8\n"                      # short row padded
            "2\tc\td\t9\t0\t5\tkeep\t{}\t\t\t1\toptimize\n"
        )
        res = _normalize_phase_thread_order(p)
        assert res == "normalized"
        header = p.read_text().splitlines()[0].split("\t")
        # canonical: phase precedes thread now.
        assert header.index("phase") < header.index("thread")

    def test_ensure_mode_schema_absent_and_blank(self, tmp_path: Path):
        # Deprecated mode migration: empty file -> 'absent'; a file with a
        # blank data line preserves it.
        empty = tmp_path / "results.tsv"
        empty.write_text("")
        rep = ensure_mode_schema(tmp_path)
        assert rep["results.tsv"] == "absent"

        # Now a real pre-mode file with a blank line.
        empty.write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\n"
            "\n"
            "1\tc\td\t8\t0\t5\tkeep\n"
        )
        rep2 = ensure_mode_schema(tmp_path)
        assert rep2["results.tsv"] == "migrated"
        assert "thread_mode" in empty.read_text().splitlines()[0]
