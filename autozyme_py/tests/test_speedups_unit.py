"""Unit tests for autozyme._speedups — finalized-TSV loading / parsing / selection.

Pure-python helpers, no upstream needed. We test the type-coercion helpers,
the per-group finalized summarizer, the long-batch summarizer, and the public
``speedups()`` reader (against real bundled finalized TSVs + synthetic ones we
write to a temp package-like dir via monkeypatching ``_ir_files``).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from autozyme import _speedups as S


# --------------------------------------------------------------------------
# scalar coercion helpers
# --------------------------------------------------------------------------
def test_to_float_valid_and_invalid():
    assert S._to_float("1.5") == 1.5
    assert S._to_float("3") == 3.0
    assert S._to_float("-2.5e1") == -25.0
    assert S._to_float("") is None
    assert S._to_float(None) is None
    assert S._to_float("nan") != S._to_float("nan")  # nan != nan
    assert S._to_float("abc") is None
    assert S._to_float([1, 2]) is None


def test_to_int_valid_and_invalid():
    assert S._to_int("4") == 4
    assert S._to_int("-7") == -7
    assert S._to_int("") is None
    assert S._to_int(None) is None
    assert S._to_int("1.5") is None  # int() rejects float strings
    assert S._to_int("x") is None


def test_to_bool_truthy_falsy_unknown():
    for t in ("true", "1", "yes", "True", "  YES  "):
        assert S._to_bool(t) is True
    for f in ("false", "0", "no", "False", "NO"):
        assert S._to_bool(f) is False
    assert S._to_bool(None) is None
    assert S._to_bool("maybe") is None
    assert S._to_bool("") is None


def test_split_float_list():
    assert S._split_float_list("1.0, 2.5, 3") == [1.0, 2.5, 3.0]
    assert S._split_float_list("") == []
    assert S._split_float_list(None) == []
    # malformed entries are dropped, not fatal
    assert S._split_float_list("1.0, x, 3.0") == [1.0, 3.0]
    assert S._split_float_list("5") == [5.0]


def test_read_long_rows_empty_and_nonempty():
    assert S._read_long_rows("") == []
    assert S._read_long_rows("   \n  ") == []
    text = "a\tb\n1\t2\n3\t4\n"
    rows = S._read_long_rows(text)
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


# --------------------------------------------------------------------------
# batch key + long-batch summarizer (_summarize_batch)
# --------------------------------------------------------------------------
def test_batch_key_strips_and_orders():
    row = {"timestamp": " t ", "patch_name": "p", "tier": "tiny"}
    key = S._batch_key(row)
    assert key[0] == "t"  # stripped
    assert key[1] == "p"
    assert len(key) == len(S._BATCH_KEY_COLS)


def _long_row(variant, rep_idx, sec, peak=None, **extra):
    r = {
        "variant": variant,
        "rep_idx": str(rep_idx),
        "sec": "" if sec is None else str(sec),
        "peak_mb": "" if peak is None else str(peak),
        "pass": "true",
        "tier": "tiny",
        "patch_name": "demo",
        "timestamp": "2026-01-01T00:00:00",
    }
    r.update({k: str(v) for k, v in extra.items()})
    return r


def test_summarize_batch_basic_speedup_and_pass():
    rows = [
        _long_row("baseline", 1, 4.0, 100.0),
        _long_row("baseline", 2, 6.0, 120.0),
        _long_row("patched", 1, 2.0, 50.0),
        _long_row("patched", 2, 2.0, 60.0),
    ]
    s = S._summarize_batch(rows)
    assert s is not None
    assert s["baseline_sec"] == 5.0
    assert s["patched_sec"] == 2.0
    assert s["speedup_x"] == pytest.approx(2.5)
    assert s["all_pass"] is True
    assert s["baseline_peak_mb"] == 120.0  # max
    assert s["patched_peak_mb"] == 60.0
    assert s["reps"] == 2


def test_summarize_batch_incomplete_returns_none():
    # only baseline rows
    assert S._summarize_batch([_long_row("baseline", 1, 4.0)]) is None
    # only patched rows
    assert S._summarize_batch([_long_row("patched", 1, 2.0)]) is None
    # empty
    assert S._summarize_batch([]) is None


def test_summarize_batch_no_valid_secs_returns_none():
    rows = [
        _long_row("baseline", 1, None),
        _long_row("patched", 1, None),
    ]
    assert S._summarize_batch(rows) is None


def test_summarize_batch_pass_mixed_and_unknown():
    # one failing rep -> all_pass False
    rows = [
        _long_row("baseline", 1, 4.0),
        _long_row("patched", 1, 2.0, **{"pass": "false"}),
    ]
    assert S._summarize_batch(rows)["all_pass"] is False
    # unknown pass values -> None
    rows2 = [
        _long_row("baseline", 1, 4.0),
        _long_row("patched", 1, 2.0, **{"pass": "?"}),
    ]
    assert S._summarize_batch(rows2)["all_pass"] is None


def test_summarize_batch_zero_patched_sec_speedup_none():
    rows = [
        _long_row("baseline", 1, 4.0),
        _long_row("patched", 1, 0.0),
    ]
    s = S._summarize_batch(rows)
    assert s["speedup_x"] is None


def test_summarize_batch_pulls_metrics_json():
    rows = [
        _long_row("baseline", 1, 4.0),
        _long_row("patched", 1, 2.0, metrics_json=""),
        _long_row("patched", 2, 2.0, metrics_json='{"x":1}'),
    ]
    s = S._summarize_batch(rows)
    assert s["metrics_json"] == '{"x":1}'


# --------------------------------------------------------------------------
# finalized-group summarizer (_summarize_finalized_group)
# --------------------------------------------------------------------------
def _fin_row(variant, **extra):
    r = {"variant": variant, "patch": "demo", "tier": "tiny"}
    r.update({k: ("" if v is None else str(v)) for k, v in extra.items()})
    return r


def test_summarize_finalized_group_basic():
    rows = [
        _fin_row("baseline", sec_reps="4.0, 5.0", sec_mean="4.5",
                 mem_reps="100, 110", mem_mean="105", status="ok"),
        _fin_row("patched", sec_reps="2.0, 2.2", sec_mean="2.1",
                 mem_reps="50, 55", mem_mean="52.5", speedup_x_mean="2.14",
                 pass_rate="1.0", n_reps="2", status="ok",
                 threads="4", platform="macOS", dataset="ds1",
                 ts_last="2026-01-02", metrics_json_median='{"m":1}'),
    ]
    s = S._summarize_finalized_group(rows)
    assert s is not None
    assert s["baseline_secs"] == [4.0, 5.0]
    assert s["patched_secs"] == [2.0, 2.2]
    assert s["baseline_sec"] == 4.5
    assert s["patched_sec"] == 2.1
    assert s["speedup_x"] == 2.14
    assert s["all_pass"] is True
    assert s["reps"] == 2
    assert s["system_threads"] == 4
    assert s["dataset"] == "ds1"
    assert s["timestamp"] == "2026-01-02"
    assert s["metrics_json"] == '{"m":1}'


def test_summarize_finalized_group_missing_side_returns_none():
    assert S._summarize_finalized_group([_fin_row("baseline", sec_mean="4")]) is None
    assert S._summarize_finalized_group([_fin_row("patched", sec_mean="2")]) is None


def test_summarize_finalized_group_pass_rate_below_one_fails():
    rows = [
        _fin_row("baseline", sec_mean="4", status="ok"),
        _fin_row("patched", sec_mean="2", pass_rate="0.5", status="ok"),
    ]
    assert S._summarize_finalized_group(rows)["all_pass"] is False


def test_summarize_finalized_group_status_not_ok_fails():
    rows = [
        _fin_row("baseline", sec_mean="4", status="ok"),
        _fin_row("patched", sec_mean="2", pass_rate="1.0", status="error"),
    ]
    assert S._summarize_finalized_group(rows)["all_pass"] is False


def test_summarize_finalized_group_no_pass_rate_unknown():
    rows = [
        _fin_row("baseline", sec_mean="4", status="ok"),
        _fin_row("patched", sec_mean="2", status="ok"),  # no pass_rate
    ]
    assert S._summarize_finalized_group(rows)["all_pass"] is None


def test_summarize_finalized_group_az_source_overrides_patch_name():
    rows = [
        _fin_row("baseline", sec_mean="4", status="ok"),
        _fin_row("patched", sec_mean="2", pass_rate="1.0", status="ok",
                 patch="scanpy", _az_source="scanpy_normalize"),
    ]
    s = S._summarize_finalized_group(rows)
    assert s["patch_name"] == "scanpy_normalize"


def test_finalized_key_includes_az_source():
    base = S._finalized_key({"patch": "scanpy", "tier": "tiny",
                             "_az_source": "scanpy_pca"})
    other = S._finalized_key({"patch": "scanpy", "tier": "tiny",
                              "_az_source": "scanpy_normalize"})
    assert base != other  # disambiguated by source


# --------------------------------------------------------------------------
# public speedups() against real bundled TSVs
# --------------------------------------------------------------------------
def test_speedups_real_lifelines_returns_summaries():
    out = S.speedups("lifelines")
    assert isinstance(out, list)
    assert len(out) > 0
    one = out[0]
    for key in ("patch_name", "tier", "baseline_secs", "patched_secs",
                "speedup_x", "all_pass"):
        assert key in one
    # sorted by tier, os, threads, dataset, timestamp
    tiers = [r["tier"] for r in out]
    assert tiers == sorted(tiers, key=lambda t: t or "")


def test_speedups_unknown_name_returns_empty():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert S.speedups("definitely_not_a_patch_xyz") == []


def test_speedups_close_name_warns():
    with pytest.warns(UserWarning, match="Did you mean"):
        out = S.speedups("lifeline")  # close to "lifelines"
    assert out == []


def test_speedups_case_insensitive_retry():
    # "Lifelines" -> retried as lowercase "lifelines" and finds data
    out = S.speedups("Lifelines")
    assert len(out) > 0


def test_speedups_facade_scanpy_aggregates_siblings():
    out = S.speedups("scanpy")
    assert len(out) > 0
    # facade aggregation surfaces per-method sub-task names
    names = {r["patch_name"] for r in out}
    assert any(n.startswith("scanpy_") for n in names)


def test_speedups_history_flag_same_data():
    a = S.speedups("lifelines", history=False)
    b = S.speedups("lifelines", history=True)
    assert len(a) == len(b)
