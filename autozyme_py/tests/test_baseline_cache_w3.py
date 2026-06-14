"""Wave-3 gap-fill for autozyme._baseline_cache.

Targets the error/edge branches wave-1's test_baseline_cache_unit.py left:
  - malformed thread column in baselines_history.tsv (int() ValueError -> skip)
  - OSError/csv.Error guards in the two TSV readers
  - results.tsv thread-mismatch / dataset-mismatch / malformed-speed branches
  - load_cached_baseline(threads=None) -> _resolve_threads()
  - stamped sha256 mismatch -> cache miss
  - populate_persistent_ref_dir symlink-preservation branch

Pure-python, temp task dirs, no heavy upstream needed.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from autozyme import _baseline_cache as BC


class _FakePatch:
    def __init__(self, targets):
        self.targets = targets


def _write_task_yaml(tmp_path, text):
    (Path(tmp_path) / "task.yaml").write_text(text)


def _write_history(tmp_path, rows, header="tier\tthread\tspeed_sec\tpeak_mb"):
    p = Path(tmp_path) / BC.HISTORY_FILE_REL
    p.parent.mkdir(exist_ok=True, parents=True)
    lines = [header] + ["\t".join(str(c) for c in r) for r in rows]
    p.write_text("\n".join(lines) + "\n")


def _make_task_with_ref(tmp_path, tier="tiny", content="hello"):
    ref = Path(tmp_path) / f"reference_output_{tier}"
    ref.mkdir()
    (ref / "out.txt").write_text(content)
    return ref


# --------------------------------------------------------------------------
# _read_baselines_history — malformed thread (lines 201-202) + reader guard
# --------------------------------------------------------------------------
def test_read_baselines_history_malformed_thread_skipped(tmp_path):
    # a non-integer thread token (e.g. "1x") -> int() raises -> row skipped
    _write_history(tmp_path, [
        ("tiny", "1x", 4.0, 100.0),   # bad thread -> skip (lines 201-202)
        ("tiny", 1, 5.0, 110.0),      # good -> kept
    ])
    rows = BC._read_baselines_history(tmp_path, "tiny", 1)
    assert rows == [(5.0, 110.0)]


def test_read_baselines_history_reader_error_returns_empty(tmp_path, monkeypatch):
    # Make the file exist so we reach the open(); then force csv to blow up.
    _write_history(tmp_path, [("tiny", 1, 4.0, 100.0)])

    import csv as _csv

    def _boom(*a, **k):
        raise _csv.Error("synthetic parse failure")

    monkeypatch.setattr(BC.csv, "DictReader", _boom)
    # OSError/csv.Error -> returns [] (lines 214-215)
    assert BC._read_baselines_history(tmp_path, "tiny", 1) == []


def test_read_baselines_history_malformed_peak_to_none(tmp_path):
    # speed parses, peak doesn't -> peak None (covers the inner peak except)
    _write_history(tmp_path, [("tiny", 1, 3.0, "notanumber")])
    assert BC._read_baselines_history(tmp_path, "tiny", 1) == [(3.0, None)]


# --------------------------------------------------------------------------
# _read_results_baseline — mismatch + malformed branches (280, 285, 288-293)
# --------------------------------------------------------------------------
def test_read_results_baseline_thread_mismatch_skipped(tmp_path):
    _write_task_yaml(tmp_path, "datasets:\n  - {tier: tiny, name: dsx}\n")
    p = Path(tmp_path) / BC.RESULTS_FILE_REL
    p.write_text(
        "status\tthread\tdataset\tspeed_sec\tpeak_mb\n"
        "baseline\t4\tdsx\t9.0\t90\n"   # thread 4, we query 1 -> skip (280)
    )
    assert BC._read_results_baseline(tmp_path, "tiny", 1) is None


def test_read_results_baseline_dataset_mismatch_skipped(tmp_path):
    _write_task_yaml(tmp_path, "datasets:\n  - {tier: tiny, name: dsx}\n")
    p = Path(tmp_path) / BC.RESULTS_FILE_REL
    p.write_text(
        "status\tthread\tdataset\tspeed_sec\tpeak_mb\n"
        # dataset 'other' != tier/dsx and 'tiny' not in 'other' -> skip (285)
        "baseline\t1\tother\t9.0\t90\n"
    )
    assert BC._read_results_baseline(tmp_path, "tiny", 1) is None


def test_read_results_baseline_malformed_speed_skipped(tmp_path):
    _write_task_yaml(tmp_path, "datasets:\n  - {tier: tiny, name: dsx}\n")
    p = Path(tmp_path) / BC.RESULTS_FILE_REL
    p.write_text(
        "status\tthread\tdataset\tspeed_sec\tpeak_mb\n"
        "baseline\t1\tdsx\tbadspeed\t90\n"   # speed unparseable -> skip (288-289)
        "baseline\t1\tdsx\t8.0\tbadpeak\n"    # peak unparseable -> None (292-293)
    )
    assert BC._read_results_baseline(tmp_path, "tiny", 1) == (8.0, None)


def test_read_results_baseline_reader_error_returns_none(tmp_path, monkeypatch):
    _write_task_yaml(tmp_path, "datasets:\n  - {tier: tiny, name: dsx}\n")
    p = Path(tmp_path) / BC.RESULTS_FILE_REL
    p.write_text("status\tthread\tdataset\tspeed_sec\tpeak_mb\nbaseline\t1\tdsx\t8.0\t90\n")

    import csv as _csv

    def _boom(*a, **k):
        raise _csv.Error("synthetic")

    monkeypatch.setattr(BC.csv, "DictReader", _boom)
    # OSError/csv.Error -> None (lines 296-297)
    assert BC._read_results_baseline(tmp_path, "tiny", 1) is None


# --------------------------------------------------------------------------
# load_cached_baseline(threads=None) -> _resolve_threads()  (line 346)
# --------------------------------------------------------------------------
def test_load_cached_baseline_threads_none_resolves_env(tmp_path, monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "1")  # resolves to thread-1 cache key
    _make_task_with_ref(tmp_path)
    BC._save_noise(tmp_path, {
        "schema_version": 2,
        "tiers": {"tiny": {"1": {"speed_mean": 5.0}}},
    })
    p = _FakePatch([("json", "dumps", lambda: None)])
    cached = BC.load_cached_baseline("demo", str(tmp_path), "tiny", None, p)
    assert cached is not None
    assert cached.timing_mean == 5.0


# --------------------------------------------------------------------------
# load_cached_baseline — stamped sha256 mismatch -> miss (line 410)
# --------------------------------------------------------------------------
def test_load_cached_baseline_sha_mismatch_returns_none(tmp_path):
    ref = _make_task_with_ref(tmp_path, content="hello")
    # size matches "hello" (5 bytes) so the size check passes; the sha does not.
    BC._save_noise(tmp_path, {
        "schema_version": 2,
        "tiers": {"tiny": {"1": {
            "speed_mean": 5.0,
            "output_artifact_size_bytes": len("hello"),
            "output_artifact_sha256": "0" * 64,  # deliberately wrong sha
        }}},
    })
    p = _FakePatch([("json", "dumps", lambda: None)])
    assert BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, p) is None


def test_load_cached_baseline_sha_match_hits(tmp_path):
    # control: round-tripping a real write_cached_baseline stamps a matching sha
    ref = _make_task_with_ref(tmp_path, content="hello")
    BC.write_cached_baseline(
        str(tmp_path), "tiny", 1, "", speeds=[5.0], peaks=[None],
        versions={}, ref_dir=str(ref), produced_by="attest",
    )
    p = _FakePatch([("json", "dumps", lambda: None)])
    cached = BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, p)
    assert cached is not None and cached.timing_mean == 5.0


# --------------------------------------------------------------------------
# populate_persistent_ref_dir — symlink preservation (line 478)
# --------------------------------------------------------------------------
def test_populate_persistent_ref_dir_preserves_symlink(tmp_path):
    src = Path(tmp_path) / "src"
    src.mkdir()
    real = src / "real.txt"
    real.write_text("payload")
    link = src / "alias.txt"
    os.symlink("real.txt", link)   # relative symlink within src

    dest = BC.populate_persistent_ref_dir(str(tmp_path), "tiny", str(src))
    destp = Path(dest)
    alias = destp / "alias.txt"
    # symlink copied AS a symlink (not dereferenced) -> line 478
    assert alias.is_symlink()
    assert os.readlink(alias) == "real.txt"
    assert (destp / "real.txt").read_text() == "payload"
