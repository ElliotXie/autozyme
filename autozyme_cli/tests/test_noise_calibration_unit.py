"""Unit tests for zyme.noise_calibration — borderline detection math, the
(mean, stdev, cv) aggregator, and the per-(tier, thread) noise persistence
roundtrip.
"""
from __future__ import annotations

import json
import statistics

import pytest

from zyme import noise_calibration as NC


# ---------------------------------------------------------------------------
# aggregate_samples
# ---------------------------------------------------------------------------

def test_aggregate_samples_empty():
    assert NC.aggregate_samples([]) == (0.0, 0.0, 0.0)


def test_aggregate_samples_single():
    m, s, cv = NC.aggregate_samples([12.0])
    assert m == 12.0
    assert s == 0.0
    assert cv == 0.0


def test_aggregate_samples_many():
    samples = [10.0, 12.0, 11.0, 13.0]
    m, s, cv = NC.aggregate_samples(samples)
    assert m == pytest.approx(statistics.mean(samples))
    assert s == pytest.approx(statistics.stdev(samples))
    assert cv == pytest.approx(s / m)


def test_aggregate_samples_zero_mean_cv_is_zero():
    m, s, cv = NC.aggregate_samples([0.0, 0.0, 0.0])
    assert m == 0.0
    assert cv == 0.0


# ---------------------------------------------------------------------------
# is_speed_borderline
# ---------------------------------------------------------------------------

def test_is_speed_borderline_no_cv():
    inside, reason = NC.is_speed_borderline(1.0, 0.0)
    assert inside is False
    assert "no calibrated" in reason


def test_is_speed_borderline_none_cv():
    inside, reason = NC.is_speed_borderline(1.0, None)
    assert inside is False


def test_is_speed_borderline_negative_cv():
    inside, _ = NC.is_speed_borderline(1.0, -0.05)
    assert inside is False


def test_is_speed_borderline_inside_window():
    # cv 0.02, mult 3 -> window 6%. |delta|=4% < 6% -> borderline.
    inside, reason = NC.is_speed_borderline(4.0, 0.02)
    assert inside is True
    assert "<" in reason


def test_is_speed_borderline_outside_window():
    # cv 0.02, mult 3 -> window 6%. |delta|=10% -> decisive.
    inside, reason = NC.is_speed_borderline(10.0, 0.02)
    assert inside is False
    assert "≥" in reason


def test_is_speed_borderline_exact_boundary_excluded():
    # delta exactly equal to window -> strict < so NOT borderline.
    inside, _ = NC.is_speed_borderline(6.0, 0.02)  # window = 3*0.02*100 = 6
    assert inside is False


def test_is_speed_borderline_uses_abs_delta():
    pos, _ = NC.is_speed_borderline(4.0, 0.02)
    neg, _ = NC.is_speed_borderline(-4.0, 0.02)
    assert pos is True and neg is True


def test_is_speed_borderline_custom_multiplier():
    # mult 1 -> window 2%; |delta|=3% -> not borderline.
    inside, _ = NC.is_speed_borderline(3.0, 0.02, multiplier=1.0)
    assert inside is False
    # mult 5 -> window 10%; |delta|=3% -> borderline.
    inside, _ = NC.is_speed_borderline(3.0, 0.02, multiplier=5.0)
    assert inside is True


# ---------------------------------------------------------------------------
# is_metric_borderline
# ---------------------------------------------------------------------------

def test_is_metric_borderline_missing_inputs():
    assert NC.is_metric_borderline(None, 0.95, "gte")[0] is False
    assert NC.is_metric_borderline(0.96, None, "gte")[0] is False


def test_is_metric_borderline_unknown_comparator():
    inside, reason = NC.is_metric_borderline(0.96, 0.95, "eq")
    assert inside is False
    assert "unknown comparator" in reason


def test_is_metric_borderline_gte_just_above_is_borderline():
    # thr 0.95, scale = 1-0.95 = 0.05. value 0.951 -> margin 0.001 -> rel 2% < 5%.
    inside, reason = NC.is_metric_borderline(0.951, 0.95, "gte")
    assert inside is True
    assert "<" in reason


def test_is_metric_borderline_gte_comfortably_above_decisive():
    # value 0.96 -> margin 0.01 -> rel 20% -> not borderline.
    inside, _ = NC.is_metric_borderline(0.96, 0.95, "gte")
    assert inside is False


def test_is_metric_borderline_gte_failing():
    # value below threshold -> already failing, not borderline.
    inside, reason = NC.is_metric_borderline(0.94, 0.95, "gte")
    assert inside is False
    assert "already fails" in reason


def test_is_metric_borderline_gte_threshold_at_natural_limit():
    # thr 1.0 -> scale clamps to 1e-6 <= 1e-5 -> decisive (no rerun zone).
    inside, reason = NC.is_metric_borderline(1.0, 1.0, "gte")
    assert inside is False
    assert "natural limit" in reason


def test_is_metric_borderline_lte_just_below_is_borderline():
    # thr 0.05, scale 0.05. value 0.049 -> margin 0.001 -> rel 2% < 5%.
    inside, _ = NC.is_metric_borderline(0.049, 0.05, "lte")
    assert inside is True


def test_is_metric_borderline_lte_comfortably_below_decisive():
    # value 0.03 -> margin 0.02 -> rel 40% -> not borderline.
    inside, _ = NC.is_metric_borderline(0.03, 0.05, "lte")
    assert inside is False


def test_is_metric_borderline_lte_failing():
    inside, reason = NC.is_metric_borderline(0.06, 0.05, "lte")
    assert inside is False
    assert "already fails" in reason


def test_is_metric_borderline_custom_zone():
    # thr 0.95, value 0.96 -> rel 20%; zone 0.25 -> borderline.
    inside, _ = NC.is_metric_borderline(0.96, 0.95, "gte", zone=0.25)
    assert inside is True


def test_is_metric_borderline_lte_zero_threshold_natural_limit():
    # thr exactly 0 -> abs scale clamps to 1e-6 <= 1e-5 -> decisive.
    inside, reason = NC.is_metric_borderline(0.0, 0.0, "lte")
    assert inside is False
    assert "natural limit" in reason


# ---------------------------------------------------------------------------
# load_noise / save_noise / get_tier_noise
# ---------------------------------------------------------------------------

def test_load_noise_missing(tmp_path):
    data = NC.load_noise(tmp_path)
    assert data == {"schema_version": NC.SCHEMA_VERSION, "tiers": {}}


def test_load_noise_malformed_json(tmp_path):
    (tmp_path / ".zyme").mkdir()
    (tmp_path / ".zyme" / "baseline_noise.json").write_text("{not json")
    data = NC.load_noise(tmp_path)
    assert data["tiers"] == {}


def test_load_noise_injects_tiers_key(tmp_path):
    (tmp_path / ".zyme").mkdir()
    (tmp_path / ".zyme" / "baseline_noise.json").write_text('{"schema_version": 2}')
    data = NC.load_noise(tmp_path)
    assert data["tiers"] == {}


def test_save_then_load_roundtrip(tmp_path):
    payload = {"schema_version": 2, "tiers": {"tiny": {"8": {"n_reps": 5}}}}
    NC.save_noise(tmp_path, payload)
    loaded = NC.load_noise(tmp_path)
    assert loaded == payload
    # file ends with newline
    raw = (tmp_path / ".zyme" / "baseline_noise.json").read_text()
    assert raw.endswith("\n")


def test_get_tier_noise_present_and_absent(tmp_path):
    NC.save_noise(tmp_path, {"schema_version": 2,
                             "tiers": {"tiny": {"8": {"n_reps": 5, "speed_cv": 0.02}}}})
    entry = NC.get_tier_noise(tmp_path, "tiny", 8)
    assert entry is not None and entry["n_reps"] == 5
    assert NC.get_tier_noise(tmp_path, "tiny", 4) is None
    assert NC.get_tier_noise(tmp_path, "medium", 8) is None


# ---------------------------------------------------------------------------
# record_tier_noise
# ---------------------------------------------------------------------------

def test_record_tier_noise_requires_a_sample(tmp_path):
    with pytest.raises(ValueError):
        NC.record_tier_noise(tmp_path, "tiny", 8, "ds", [], [], "abc1234")


def test_record_tier_noise_single_sample(tmp_path):
    entry = NC.record_tier_noise(tmp_path, "tiny", 8, "pbmc", [10.0], [800.0], "abcdef1234")
    assert entry["n_reps"] == 1
    assert entry["speed_mean"] == 10.0
    assert entry["speed_stdev"] == 0.0
    assert entry["speed_cv"] == 0.0
    assert entry["peak_mean"] == 800.0
    assert entry["peak_stdev"] == 0.0
    assert entry["commit"] == "abcdef1"  # truncated to 7
    assert entry["produced_by"] == "iteration"


def test_record_tier_noise_multi_sample_stats(tmp_path):
    speeds = [10.0, 12.0, 11.0]
    peaks = [800.0, 810.0, 805.0]
    entry = NC.record_tier_noise(tmp_path, "medium", 4, "ds", speeds, peaks, "deadbeef")
    assert entry["speed_mean"] == round(statistics.mean(speeds), 6)
    assert entry["speed_stdev"] == round(statistics.stdev(speeds), 6)
    assert entry["speed_cv"] == round(statistics.stdev(speeds) / statistics.mean(speeds), 6)
    assert entry["peak_stdev"] == round(statistics.stdev(peaks), 3)


def test_record_tier_noise_persists_and_reads_back(tmp_path):
    NC.record_tier_noise(tmp_path, "tiny", 8, "ds", [10.0, 11.0], [1.0, 2.0], "c")
    got = NC.get_tier_noise(tmp_path, "tiny", 8)
    assert got is not None and got["n_reps"] == 2
    data = NC.load_noise(tmp_path)
    assert data["schema_version"] == NC.SCHEMA_VERSION


def test_record_tier_noise_empty_commit_string(tmp_path):
    entry = NC.record_tier_noise(tmp_path, "tiny", 8, "ds", [10.0], [1.0], "")
    assert entry["commit"] == ""


def test_record_tier_noise_no_peaks(tmp_path):
    entry = NC.record_tier_noise(tmp_path, "tiny", 8, "ds", [10.0, 11.0], [], "c")
    assert entry["peak_mean"] == 0.0
    assert entry["peak_stdev"] == 0.0


def test_record_tier_noise_v2_optional_fields(tmp_path):
    entry = NC.record_tier_noise(
        tmp_path, "tiny", 8, "ds", [10.0], [1.0], "c",
        upstream_versions={"numpy": "1.26.4"},
        output_artifact_relpath="ref_out",
        output_artifact_sha256="abc",
        output_artifact_size_bytes=1234,
        produced_by="attest",
    )
    assert entry["upstream_versions"] == {"numpy": "1.26.4"}
    assert entry["output_artifact_relpath"] == "ref_out"
    assert entry["output_artifact_sha256"] == "abc"
    assert entry["output_artifact_size_bytes"] == 1234
    assert entry["produced_by"] == "attest"


def test_record_tier_noise_thread_key_is_string(tmp_path):
    NC.record_tier_noise(tmp_path, "tiny", 8, "ds", [10.0], [1.0], "c")
    data = NC.load_noise(tmp_path)
    assert "8" in data["tiers"]["tiny"]
    assert isinstance(list(data["tiers"]["tiny"].keys())[0], str)


def test_record_tier_noise_multiple_tiers_coexist(tmp_path):
    NC.record_tier_noise(tmp_path, "tiny", 8, "a", [10.0], [1.0], "c")
    NC.record_tier_noise(tmp_path, "medium", 8, "b", [20.0], [2.0], "c")
    NC.record_tier_noise(tmp_path, "tiny", 4, "a", [10.0], [1.0], "c")
    data = NC.load_noise(tmp_path)
    assert set(data["tiers"].keys()) == {"tiny", "medium"}
    assert set(data["tiers"]["tiny"].keys()) == {"8", "4"}


def test_record_tier_noise_zero_mean_speed_cv_zero(tmp_path):
    entry = NC.record_tier_noise(tmp_path, "tiny", 8, "ds", [0.0, 0.0], [1.0], "c")
    assert entry["speed_cv"] == 0.0
