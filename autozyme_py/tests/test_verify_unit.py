"""Unit tests for autozyme._verify pure helpers.

verify_patch end-to-end (subprocess workers) is exercised in
tests/test_verify_patch.py. Here we hit the pure comparison / tolerance /
threshold / TSV formatting / sort / system-fingerprint / metric-parse helpers
directly with crafted inputs. No upstream needed (PyYAML is present).
"""
from __future__ import annotations

import csv
import json
import os
import platform
import re
from pathlib import Path

import pytest

from autozyme import _verify as V


# --------------------------------------------------------------------------
# _METRIC_LINE_RE + metric parsing semantics
# --------------------------------------------------------------------------
def test_metric_line_re_matches():
    m = V._METRIC_LINE_RE.match("output_match: 1.0")
    assert m and m.group(1) == "output_match" and m.group(2) == "1.0"
    m = V._METRIC_LINE_RE.match("ari_score:0.95")
    assert m and m.group(2) == "0.95"
    m = V._METRIC_LINE_RE.match("rel_err: -3.2e-05")
    assert m and m.group(2) == "-3.2e-05"


def test_metric_line_re_rejects():
    assert V._METRIC_LINE_RE.match("not a metric") is None
    assert V._METRIC_LINE_RE.match("123abc: 4") is None       # name must start alpha/_
    assert V._METRIC_LINE_RE.match("metric: not_number") is None
    assert V._METRIC_LINE_RE.match("two words: 1.0 extra") is None


# --------------------------------------------------------------------------
# _effective_threshold — deterministic vs stochastic + noise multiplier
# --------------------------------------------------------------------------
def test_effective_threshold_deterministic():
    th, label = V._effective_threshold(
        {"name": "m", "threshold": 0.9, "comparator": "gte"}, "tiny", {})
    assert th == 0.9
    assert label == "absolute"


def test_effective_threshold_missing_both_raises():
    with pytest.raises(RuntimeError, match="neither"):
        V._effective_threshold({"name": "m", "comparator": "gte"}, "tiny", {})


def test_effective_threshold_floor_no_calibration():
    th, label = V._effective_threshold(
        {"name": "m", "absolute_floor": 0.8, "comparator": "lte"}, "tiny", {})
    assert th == 0.8
    assert "no intrinsic_noise calibrated" in label


def test_effective_threshold_lte_with_noise():
    # lte: effective = max(floor, multiplier * noise)
    metric = {"name": "err", "absolute_floor": 0.01,
              "noise_multiplier": 2.0, "comparator": "lte"}
    noise = {"tiny": {"err": 0.05}}
    th, label = V._effective_threshold(metric, "tiny", noise)
    assert th == pytest.approx(0.10)  # 2 * 0.05 > floor 0.01
    assert "noise" in label


def test_effective_threshold_lte_floor_wins():
    metric = {"name": "err", "absolute_floor": 0.5,
              "noise_multiplier": 2.0, "comparator": "lte"}
    noise = {"tiny": {"err": 0.05}}
    th, _ = V._effective_threshold(metric, "tiny", noise)
    assert th == 0.5  # floor 0.5 > 2*0.05


def test_effective_threshold_gte_with_noise():
    # gte: effective = max(floor, 1 - multiplier*(1-noise))
    metric = {"name": "ari", "absolute_floor": 0.5,
              "noise_multiplier": 2.0, "comparator": "gte"}
    noise = {"tiny": {"ari": 0.95}}
    th, _ = V._effective_threshold(metric, "tiny", noise)
    # 1 - 2*(1-0.95) = 1 - 0.1 = 0.9 > floor 0.5
    assert th == pytest.approx(0.9)


def test_effective_threshold_default_multiplier_is_two():
    metric = {"name": "err", "absolute_floor": 0.0, "comparator": "lte"}
    noise = {"tiny": {"err": 0.03}}
    th, _ = V._effective_threshold(metric, "tiny", noise)
    assert th == pytest.approx(0.06)  # default multiplier 2.0


# --------------------------------------------------------------------------
# TSV header normalization + detection
# --------------------------------------------------------------------------
def test_normalize_tsv_header_line_strips_bom_crlf():
    assert V._normalize_tsv_header_line("﻿a\tb\r") == "a\tb"
    assert V._normalize_tsv_header_line("  a\tb  ") == "a\tb"
    assert V._normalize_tsv_header_line("") == ""


def test_header_cols_of_line():
    cols = V._header_cols_of_line("﻿a\tb\tc\r")
    assert cols == ["a", "b", "c"]


def test_is_known_package_verify_header_current():
    cols = V._PACKAGE_VERIFY_HEADER.split("\t")
    assert V._is_known_package_verify_header_cols(cols) is True


def test_is_known_package_verify_header_legacy_and_pre_pkgver():
    assert V._is_known_package_verify_header_cols(
        V._PACKAGE_VERIFY_HEADER_LEGACY.split("\t")) is True
    assert V._is_known_package_verify_header_cols(
        V._PACKAGE_VERIFY_HEADER_PRE_PKGVER.split("\t")) is True


def test_is_known_package_verify_header_superset_forward_compat():
    cols = V._PACKAGE_VERIFY_HEADER.split("\t") + ["extra_future_col"]
    assert V._is_known_package_verify_header_cols(cols) is True


def test_is_known_package_verify_header_unknown():
    assert V._is_known_package_verify_header_cols(["foo", "bar"]) is False
    assert V._is_known_package_verify_header_cols([]) is False


def test_assert_long_format_missing_file_ok(tmp_path):
    # nonexistent path -> no raise
    V._assert_long_format_or_empty(str(tmp_path / "nope.tsv"))


def test_assert_long_format_empty_header_ok(tmp_path):
    p = tmp_path / "v.tsv"
    p.write_text("\n")
    V._assert_long_format_or_empty(str(p))  # blank -> ok


def test_assert_long_format_current_header_ok(tmp_path):
    p = tmp_path / "v.tsv"
    p.write_text(V._PACKAGE_VERIFY_HEADER + "\n")
    V._assert_long_format_or_empty(str(p))


def test_assert_long_format_bad_header_raises(tmp_path):
    p = tmp_path / "v.tsv"
    p.write_text("foo\tbar\tbaz\n")
    with pytest.raises(RuntimeError, match="unrecognized header"):
        V._assert_long_format_or_empty(str(p))


# --------------------------------------------------------------------------
# _tsv_sanitize
# --------------------------------------------------------------------------
def test_tsv_sanitize():
    assert V._tsv_sanitize(None) == ""
    assert V._tsv_sanitize("a\tb\nc\rd") == "a b c d"
    assert V._tsv_sanitize(42) == "42"
    assert V._tsv_sanitize("clean") == "clean"


# --------------------------------------------------------------------------
# _sort_key_for_long_row
# --------------------------------------------------------------------------
def test_sort_key_variant_order():
    base = V._sort_key_for_long_row({"variant": "baseline", "tier": "tiny"})
    patched = V._sort_key_for_long_row({"variant": "patched", "tier": "tiny"})
    unknown = V._sort_key_for_long_row({"variant": "weird", "tier": "tiny"})
    assert base[0] == 0 and patched[0] == 1 and unknown[0] == 2


def test_sort_key_tier_order():
    tiny = V._sort_key_for_long_row({"variant": "baseline", "tier": "tiny"})
    large = V._sort_key_for_long_row({"variant": "baseline", "tier": "large"})
    unknown_tier = V._sort_key_for_long_row({"variant": "baseline", "tier": "zzz"})
    assert tiny[1] < large[1] < unknown_tier[1]


def test_sort_key_platform_detection():
    win = V._sort_key_for_long_row({"system_os": "Windows 11"})
    mac = V._sort_key_for_long_row({"system_os": "macOS 14.5"})
    darwin = V._sort_key_for_long_row({"system_os": "Darwin 24"})
    empty = V._sort_key_for_long_row({"system_os": ""})
    other = V._sort_key_for_long_row({"system_os": "Linux 5.15"})
    assert win[2] == "win"
    assert mac[2] == "mac"
    assert darwin[2] == "mac"
    assert empty[2] == "mac"   # blank defaults to mac
    assert other[2] == "unknown"


def test_sort_key_threads_and_rep_idx_parse():
    k = V._sort_key_for_long_row({
        "variant": "patched", "tier": "tiny",
        "system_threads": "8", "rep_idx": "3", "timestamp": "t",
    })
    assert k[3] == 8 and k[5] == 3
    # bad ints -> 0
    k2 = V._sort_key_for_long_row({"system_threads": "x", "rep_idx": "y"})
    assert k2[3] == 0 and k2[5] == 0


# --------------------------------------------------------------------------
# _detect_cpu_model / _detect_ram_gb / _collect_system_info — best effort
# --------------------------------------------------------------------------
def test_detect_cpu_model_returns_str():
    cpu = V._detect_cpu_model()
    assert isinstance(cpu, str)  # may be empty on exotic platforms


def test_detect_ram_gb_type():
    ram = V._detect_ram_gb()
    assert ram is None or (isinstance(ram, float) and ram > 0)


def test_collect_system_info_keys(monkeypatch):
    monkeypatch.setenv("ZYME_THREADS", "4")
    info = V._collect_system_info()
    assert set(info.keys()) == {"system_os", "system_cpu", "system_ram_gb",
                                "system_threads"}
    assert info["system_threads"] == "4"
    assert isinstance(info["system_os"], str) and info["system_os"]


def test_collect_system_info_threads_fallback(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    info = V._collect_system_info()
    assert info["system_threads"] == ""


def test_framework_version_str():
    v = V._framework_version()
    assert isinstance(v, str) and v


# --------------------------------------------------------------------------
# _tier_dataset_map
# --------------------------------------------------------------------------
def test_tier_dataset_map(tmp_path):
    (tmp_path / "task.yaml").write_text(
        "datasets:\n  - {tier: tiny, name: a}\n  - {tier: large, name: b}\n")
    m = V._tier_dataset_map(str(tmp_path))
    assert m == {"tiny": "a", "large": "b"}


def test_tier_dataset_map_no_file(tmp_path):
    assert V._tier_dataset_map(str(tmp_path)) == {}


def test_tier_dataset_map_skips_incomplete(tmp_path):
    (tmp_path / "task.yaml").write_text(
        "datasets:\n  - {tier: tiny}\n  - {name: orphan}\n  - {tier: ok, name: n}\n")
    assert V._tier_dataset_map(str(tmp_path)) == {"ok": "n"}


# --------------------------------------------------------------------------
# _append_package_verify_tsv — the long-format writer + speedup math
# --------------------------------------------------------------------------
def _read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_append_package_verify_writes_long_rows(tmp_path):
    (tmp_path / "task.yaml").write_text(
        "datasets:\n  - {tier: tiny, name: dsx}\n")
    rows = [{
        "tier": "tiny",
        "note": "",
        "baseline_secs": [4.0, 6.0],
        "patched_secs": [2.0, 2.0],
        "baseline_peaks_mb": [100.0, 120.0],
        "patched_peaks_mb": [50.0, 60.0],
        "per_rep_pass": [True, True],
        "metrics_json": '{"m":1}',
    }]
    V._append_package_verify_tsv(str(tmp_path), "demo", rows)
    out = _read_tsv(tmp_path / "package_verify.tsv")
    # 2 baseline + 2 patched rows
    assert len(out) == 4
    baselines = [r for r in out if r["variant"] == "baseline"]
    patched = [r for r in out if r["variant"] == "patched"]
    assert len(baselines) == 2 and len(patched) == 2
    # all rows carry dataset name + patch name
    assert all(r["dataset"] == "dsx" for r in out)
    assert all(r["patch_name"] == "demo" for r in out)
    # speedup_x computed from median baseline (5.0) / patched sec (2.0) = 2.5
    for r in patched:
        assert float(r["speedup_x"]) == pytest.approx(2.5)
        assert r["pass"] == "true"
        assert r["metrics_json"] == '{"m":1}'


def test_append_package_verify_crashed_sentinel(tmp_path):
    (tmp_path / "task.yaml").write_text("datasets: []\n")
    rows = [{
        "tier": "tiny", "note": "boom",
        "baseline_secs": [], "patched_secs": [],
        "baseline_peaks_mb": [], "patched_peaks_mb": [],
        "per_rep_pass": [], "metrics_json": "",
    }]
    V._append_package_verify_tsv(str(tmp_path), "demo", rows)
    out = _read_tsv(tmp_path / "package_verify.tsv")
    assert len(out) == 1
    assert out[0]["variant"] == ""
    assert out[0]["sec"] == ""
    assert out[0]["note"] == "boom"


def test_append_package_verify_appends_and_sorts(tmp_path):
    (tmp_path / "task.yaml").write_text("datasets: []\n")
    base_row = {
        "tier": "tiny", "note": "",
        "baseline_secs": [4.0], "patched_secs": [2.0],
        "baseline_peaks_mb": [None], "patched_peaks_mb": [None],
        "per_rep_pass": [True], "metrics_json": "",
    }
    V._append_package_verify_tsv(str(tmp_path), "demo", [dict(base_row)])
    V._append_package_verify_tsv(str(tmp_path), "demo", [dict(base_row)])
    out = _read_tsv(tmp_path / "package_verify.tsv")
    assert len(out) == 4  # accumulates history
    # globally sorted: all baseline variants precede patched variants
    variants = [r["variant"] for r in out]
    assert variants == sorted(variants, key=lambda v: V._VARIANT_ORDER_LONG.get(v, 2))


def test_append_package_verify_no_speedup_when_baseline_zero(tmp_path):
    (tmp_path / "task.yaml").write_text("datasets: []\n")
    rows = [{
        "tier": "tiny", "note": "",
        "baseline_secs": [0.0], "patched_secs": [2.0],
        "baseline_peaks_mb": [], "patched_peaks_mb": [],
        "per_rep_pass": [None], "metrics_json": "",
    }]
    V._append_package_verify_tsv(str(tmp_path), "demo", rows)
    out = _read_tsv(tmp_path / "package_verify.tsv")
    patched = [r for r in out if r["variant"] == "patched"]
    # baseline 0 -> median_positive returns None -> blank speedup
    assert patched[0]["speedup_x"] == ""


# --------------------------------------------------------------------------
# _resolve_inst_speedup_path / _find_attest_manifest
# --------------------------------------------------------------------------
def test_find_attest_manifest_none(tmp_path):
    assert V._find_attest_manifest(str(tmp_path), "demo") is None


def test_find_attest_manifest_walks_up(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    manifest = scripts / "demo_attest_manifest.yaml"
    manifest.write_text("tasks: []\n")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    found = V._find_attest_manifest(str(deep), "demo")
    assert found == str(manifest)


def test_resolve_inst_speedup_path_no_manifest(tmp_path):
    assert V._resolve_inst_speedup_path(str(tmp_path), "demo") is None


def test_resolve_inst_speedup_path_package_speedups_false(tmp_path):
    # framework_root = tmp_path; task lives at tmp_path/task
    task = tmp_path / "task"
    task.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "demo_attest_manifest.yaml").write_text(
        "tasks:\n  - {path: task, legacy_key: k1, package_speedups: false}\n")
    assert V._resolve_inst_speedup_path(str(task), "demo") is None


def test_resolve_inst_speedup_path_resolves(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "demo_attest_manifest.yaml").write_text(
        "tasks:\n  - {path: task, legacy_key: tiny_1t}\n")
    path = V._resolve_inst_speedup_path(str(task), "demo")
    assert path is not None
    assert path.endswith(os.path.join("demo", "speedups", "demo_tiny_1t.tsv"))


# --------------------------------------------------------------------------
# _write_inst_speedup_tsv — filtering of passing/measured rows
# --------------------------------------------------------------------------
def _full_row(**kw):
    cols = V._PACKAGE_VERIFY_HEADER.split("\t")
    r = {c: "" for c in cols}
    r.update(kw)
    return r


def test_write_inst_speedup_filters_to_passing(tmp_path):
    out_path = tmp_path / "demo" / "speedups" / "demo_k.tsv"
    common = dict(timestamp="t1", patch_name="demo", tier="tiny",
                  framework_version="0.3.0", note="", system_os="macOS 14",
                  system_cpu="cpu", system_ram_gb="16.0", system_threads="1")
    rows = [
        _full_row(variant="baseline", sec="4.0", **common),
        _full_row(variant="patched", sec="2.0", pass_field="true", **common),
    ]
    # 'pass' is a python keyword-safe column; set it explicitly
    rows[1]["pass"] = "true"
    V._write_inst_speedup_tsv(str(out_path), rows)
    assert out_path.is_file()
    written = _read_tsv(out_path)
    # baseline + patched both retained (same group, patched passed)
    assert len(written) == 2


def test_write_inst_speedup_skips_failing_group(tmp_path):
    out_path = tmp_path / "demo" / "speedups" / "demo_k.tsv"
    common = dict(timestamp="t1", patch_name="demo", tier="tiny",
                  framework_version="0.3.0", note="", system_os="macOS 14",
                  system_cpu="cpu", system_ram_gb="16.0", system_threads="1")
    rows = [
        _full_row(variant="baseline", sec="4.0", **common),
        _full_row(variant="patched", sec="2.0", **common),
    ]
    rows[1]["pass"] = "false"
    V._write_inst_speedup_tsv(str(out_path), rows)
    # nothing passing -> no file written
    assert not out_path.exists()


def test_write_inst_speedup_skips_non_measured(tmp_path):
    out_path = tmp_path / "demo" / "speedups" / "demo_k.tsv"
    rows = [_full_row(variant="", sec="", patch_name="demo", tier="tiny")]
    V._write_inst_speedup_tsv(str(out_path), rows)
    assert not out_path.exists()


# --------------------------------------------------------------------------
# _read_task_yaml
# --------------------------------------------------------------------------
def test_read_task_yaml(tmp_path):
    (tmp_path / "task.yaml").write_text(
        "task: demo\nmetrics:\n  - {name: m, threshold: 1.0, comparator: gte}\n")
    task = V._read_task_yaml(str(tmp_path))
    assert task["task"] == "demo"
    assert task["metrics"][0]["name"] == "m"


# --------------------------------------------------------------------------
# _run_evaluate — drives a tiny evaluate.py subprocess
# --------------------------------------------------------------------------
def test_run_evaluate_runs_and_returns_lines(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "evaluate.py").write_text(
        "import os\n"
        "print('ref_dir:', os.environ['ZYME_REFERENCE_DIR'])\n"
        "print('output_match: 1.0')\n"
    )
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    ref_dir = temp_dir / "reference_output_tiny"
    ref_dir.mkdir()
    lines = V._run_evaluate(str(task_dir), str(temp_dir), str(ref_dir), "tiny")
    assert any(ln.strip() == "output_match: 1.0" for ln in lines)


def test_run_evaluate_no_evaluate_file(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no evaluate"):
        V._run_evaluate(str(task_dir), str(tmp_path), str(tmp_path), "tiny")


def test_run_evaluate_nonzero_exit_raises(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "evaluate.py").write_text("import sys; sys.exit(3)\n")
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    with pytest.raises(RuntimeError, match="evaluate exited 3"):
        V._run_evaluate(str(task_dir), str(temp_dir), str(temp_dir), "tiny")


# --------------------------------------------------------------------------
# _append_package_verify_tsv — legacy-header upgrade path
# --------------------------------------------------------------------------
def test_append_package_verify_upgrades_legacy_header(tmp_path):
    (tmp_path / "task.yaml").write_text(
        "datasets:\n  - {tier: tiny, name: dsx}\n")
    tsv = tmp_path / "package_verify.tsv"
    # seed with a PRE-2026-05-26 legacy header (no dataset, no package_version)
    legacy_cols = V._PACKAGE_VERIFY_HEADER_LEGACY.split("\t")
    legacy_row = {c: "" for c in legacy_cols}
    legacy_row.update({"timestamp": "old", "patch_name": "demo", "tier": "tiny",
                       "rep_idx": "1", "variant": "baseline", "sec": "9.0"})
    tsv.write_text(
        V._PACKAGE_VERIFY_HEADER_LEGACY + "\n"
        + "\t".join(legacy_row[c] for c in legacy_cols) + "\n")
    rows = [{
        "tier": "tiny", "note": "",
        "baseline_secs": [4.0], "patched_secs": [2.0],
        "baseline_peaks_mb": [None], "patched_peaks_mb": [None],
        "per_rep_pass": [True], "metrics_json": "",
    }]
    V._append_package_verify_tsv(str(tmp_path), "demo", rows)
    out = _read_tsv(tsv)
    # header upgraded to current; old legacy row's dataset backfilled from tier map
    assert all("dataset" in r and "package_version" in r for r in out)
    old = [r for r in out if r["timestamp"] == "old"]
    assert old and old[0]["dataset"] == "dsx"
