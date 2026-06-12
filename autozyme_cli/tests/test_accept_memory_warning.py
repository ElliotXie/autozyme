"""Unit tests for the memory-regression warning attached to `zyme accept`.

`_maybe_memory_regression_warning` is the post-accept hook that surfaces
peak_mb regressions over the matching-(dataset, thread) baseline. It must:
  - fire when patched > 1.30 * baseline,
  - stay silent at the threshold and below,
  - stay silent on memory improvements,
  - stay silent when peak_mb is missing on either side,
  - stay silent when no matching baseline exists for the keep's (dataset, thread),
  - compare against the same-thread baseline (not cross-thread).

It is non-blocking by construction (the caller wraps it in try/except), so
these tests assert *behavior*, not *return value*.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.commands.run import (
    _MEMORY_REGRESSION_PCT_THRESHOLD,
    _maybe_memory_regression_warning,
)
from zyme.utils import RESULTS_HEADER_BASE


# results.tsv columns (must match RESULTS_HEADER_BASE).
_COLS = RESULTS_HEADER_BASE.strip().split("\t")


def _row(**fields) -> str:
    """Build one TSV row, filling unspecified cols with empty string."""
    return "\t".join(str(fields.get(c, "")) for c in _COLS)


def _write_results(task_dir: Path, *rows: str) -> Path:
    p = task_dir / "results.tsv"
    p.write_text(RESULTS_HEADER_BASE + "\n".join(rows) + ("\n" if rows else ""))
    return p


def test_threshold_constant_is_30():
    # Documented behavior — if someone changes the constant, the test forces
    # them to also update this assertion (and presumably the docs).
    assert _MEMORY_REGRESSION_PCT_THRESHOLD == 30.0


def test_fires_when_regression_exceeds_threshold(tmp_path, capsys):
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="100", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="150", status="keep", thread="1"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    out = capsys.readouterr().out
    assert "[memory] WARNING" in out
    assert "50%" in out  # (150 - 100) / 100
    assert "150 MB" in out
    assert "100 MB" in out


def test_silent_at_exactly_threshold(tmp_path, capsys):
    # 130 / 100 = +30.0% → not strictly greater than 30, no fire.
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="100", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="130", status="keep", thread="1"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    assert "[memory]" not in capsys.readouterr().out


def test_silent_below_threshold(tmp_path, capsys):
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="100", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="125", status="keep", thread="1"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    assert "[memory]" not in capsys.readouterr().out


def test_silent_on_memory_improvement(tmp_path, capsys):
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="200", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="100", status="keep", thread="1"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    assert "[memory]" not in capsys.readouterr().out


def test_silent_when_latest_peak_missing(tmp_path, capsys):
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="100", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="", status="keep", thread="1"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    assert "[memory]" not in capsys.readouterr().out


def test_silent_when_baseline_peak_missing(tmp_path, capsys):
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="0", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="500", status="keep", thread="1"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    assert "[memory]" not in capsys.readouterr().out


def test_silent_when_no_keep_row(tmp_path, capsys):
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="100", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="500", status="pending", thread="1"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    assert "[memory]" not in capsys.readouterr().out


def test_silent_when_no_matching_baseline_dataset(tmp_path, capsys):
    # Baseline is for `small`, but the keep is on `tiny` — no join, skip silently.
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="small", peak_mb="100", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="500", status="keep", thread="1"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    assert "[memory]" not in capsys.readouterr().out


def test_silent_when_thread_mismatch(tmp_path, capsys):
    # Baseline is 1-thread; the keep is 8-thread — different parallelism
    # regime, comparison would be meaningless.
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="100", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="500", status="keep", thread="8"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    assert "[memory]" not in capsys.readouterr().out


def test_uses_latest_keep_when_multiple_keeps_present(tmp_path, capsys):
    # When the agent has stacked several keeps, the warning fires (or not)
    # based on the *most recent* keep — i.e., the round that just got accepted.
    # Earlier non-regressive keeps must not suppress a later regressive one.
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="100", status="baseline", thread="1"),
        _row(round=1, dataset="tiny", peak_mb="105", status="keep", thread="1"),
        _row(round=2, dataset="tiny", peak_mb="200", status="keep", thread="1"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    out = capsys.readouterr().out
    assert "[memory] WARNING" in out
    assert "100%" in out  # (200 - 100) / 100 = 100%


def test_legacy_thread_column_missing(tmp_path, capsys):
    # Old tasks pre-thread-col write empty thread; both baseline + keep then
    # fall back to LEGACY_THREAD and match — regression is still detectable.
    results = _write_results(
        tmp_path,
        _row(round=0, dataset="tiny", peak_mb="100", status="baseline"),
        _row(round=1, dataset="tiny", peak_mb="200", status="keep"),
    )
    _maybe_memory_regression_warning(tmp_path, results)
    assert "[memory] WARNING" in capsys.readouterr().out


def test_does_not_raise_on_empty_results(tmp_path):
    p = tmp_path / "results.tsv"
    p.write_text(RESULTS_HEADER_BASE)
    # Must not raise on a header-only file.
    _maybe_memory_regression_warning(tmp_path, p)


def test_does_not_raise_on_missing_results(tmp_path):
    # cmd_accept's try/except is the safety net, but the helper itself should
    # also be robust to a vanished file (e.g. another tool moved it mid-flight).
    p = tmp_path / "results.tsv"
    # File does not exist — _read_results_rows returns [], we return early.
    # No assertion needed; the test passes if no exception is raised.
    try:
        _maybe_memory_regression_warning(tmp_path, p)
    except FileNotFoundError:
        pytest.fail("helper should tolerate missing results.tsv")
