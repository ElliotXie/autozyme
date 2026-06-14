"""Unit tests for autozyme._baseline_cache — key/store/load/invalidate logic.

Pure-python helpers operating on temp task dirs. No upstream needed except a
fake ``p`` object exposing ``.targets`` for ``_collect_versions`` and a
``yaml`` import (PyYAML ships in this env). Heavy scientific libs not required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from autozyme import _baseline_cache as BC


# --------------------------------------------------------------------------
# _resolve_threads — env-var precedence
# --------------------------------------------------------------------------
def test_resolve_threads_default(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    assert BC._resolve_threads(default=3) == 3


def test_resolve_threads_zyme_wins(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("ZYME_THREADS", "4")
    assert BC._resolve_threads() == 4  # ZYME_THREADS precedes OMP


def test_resolve_threads_invalid_falls_through(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "garbage")
    monkeypatch.setenv("AUTOZYME_THREADS", "6")
    assert BC._resolve_threads() == 6


def test_resolve_threads_clamps_to_one(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "0")
    assert BC._resolve_threads() == 1  # max(1, 0)


# --------------------------------------------------------------------------
# _thread_matches
# --------------------------------------------------------------------------
def test_thread_matches():
    assert BC._thread_matches("4", 4) is True
    assert BC._thread_matches("4.0", 4) is True
    assert BC._thread_matches("8", 4) is False
    assert BC._thread_matches("", 1) is True   # empty == legacy thread 1
    assert BC._thread_matches("", 4) is False
    assert BC._thread_matches("bad", 4) is False


# --------------------------------------------------------------------------
# _hash_artifact_dir
# --------------------------------------------------------------------------
def test_hash_artifact_dir_nonexistent(tmp_path):
    sha, size = BC._hash_artifact_dir(tmp_path / "nope")
    assert sha == "" and size == 0


def test_hash_artifact_dir_stable_and_size(tmp_path):
    d = tmp_path / "ref"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    (d / "b.txt").write_text("world!")
    sha1, size1 = BC._hash_artifact_dir(d)
    sha2, size2 = BC._hash_artifact_dir(d)
    assert sha1 == sha2 and sha1 != ""
    assert size1 == size2 == len("hello") + len("world!")


def test_hash_artifact_dir_changes_on_content(tmp_path):
    d = tmp_path / "ref"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    sha_a, _ = BC._hash_artifact_dir(d)
    (d / "a.txt").write_text("HELLO")
    sha_b, _ = BC._hash_artifact_dir(d)
    assert sha_a != sha_b


def test_hash_artifact_dir_oversize_skips_sha(tmp_path, monkeypatch):
    d = tmp_path / "ref"
    d.mkdir()
    (d / "a.txt").write_text("x" * 2048)
    # cap at "0 MB" via env override -> any non-empty dir is oversize
    monkeypatch.setenv("ZYME_BASELINE_CACHE_MAX_HASH_MB", "0")
    sha, size = BC._hash_artifact_dir(d)
    assert sha == ""
    assert size == 2048


def test_hash_artifact_dir_bad_env_uses_default(tmp_path, monkeypatch):
    d = tmp_path / "ref"
    d.mkdir()
    (d / "a.txt").write_text("hi")
    monkeypatch.setenv("ZYME_BASELINE_CACHE_MAX_HASH_MB", "not-a-number")
    sha, size = BC._hash_artifact_dir(d)  # falls back to default 512MB
    assert sha != "" and size == 2


# --------------------------------------------------------------------------
# noise.json read/write round-trip
# --------------------------------------------------------------------------
def test_load_noise_missing_returns_empty_schema(tmp_path):
    data = BC._load_noise(tmp_path)
    assert data["schema_version"] == BC.SCHEMA_VERSION
    assert data["tiers"] == {}


def test_load_noise_malformed_json(tmp_path):
    p = BC._noise_path(tmp_path)
    p.parent.mkdir(exist_ok=True)
    p.write_text("{not valid json")
    data = BC._load_noise(tmp_path)
    assert data["tiers"] == {}


def test_load_noise_adds_missing_tiers_key(tmp_path):
    p = BC._noise_path(tmp_path)
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"schema_version": 2}))
    data = BC._load_noise(tmp_path)
    assert data["tiers"] == {}


def test_save_and_reload_noise(tmp_path):
    data = {"schema_version": 2, "tiers": {"tiny": {"1": {"speed_mean": 3.0}}}}
    BC._save_noise(tmp_path, data)
    assert BC._noise_path(tmp_path).is_file()
    reloaded = BC._load_noise(tmp_path)
    assert reloaded["tiers"]["tiny"]["1"]["speed_mean"] == 3.0


# --------------------------------------------------------------------------
# baselines_history.tsv reader
# --------------------------------------------------------------------------
def test_read_baselines_history_missing(tmp_path):
    assert BC._read_baselines_history(tmp_path, "tiny", 1) == []


def _write_history(tmp_path, rows):
    p = Path(tmp_path) / BC.HISTORY_FILE_REL
    p.parent.mkdir(exist_ok=True)
    header = "tier\tthread\tspeed_sec\tpeak_mb"
    lines = [header] + ["\t".join(str(c) for c in r) for r in rows]
    p.write_text("\n".join(lines) + "\n")


def test_read_baselines_history_thread_filter(tmp_path):
    _write_history(tmp_path, [
        ("tiny", 1, 4.0, 100.0),
        ("tiny", 4, 2.0, 50.0),
        ("medium", 1, 8.0, 200.0),
    ])
    rows = BC._read_baselines_history(tmp_path, "tiny", 1)
    assert rows == [(4.0, 100.0)]
    rows4 = BC._read_baselines_history(tmp_path, "tiny", 4)
    assert rows4 == [(2.0, 50.0)]


def test_read_baselines_history_empty_thread_matches_one(tmp_path):
    _write_history(tmp_path, [("tiny", "", 4.0, 100.0)])
    assert BC._read_baselines_history(tmp_path, "tiny", 1) == [(4.0, 100.0)]
    # querying thread=4 with empty thread column -> no match
    assert BC._read_baselines_history(tmp_path, "tiny", 4) == []


def test_read_baselines_history_limit(tmp_path):
    _write_history(tmp_path, [("tiny", 1, float(i), None) for i in range(10)])
    rows = BC._read_baselines_history(tmp_path, "tiny", 1, limit=3)
    assert len(rows) == 3
    # most recent (last) kept
    assert [s for s, _ in rows] == [7.0, 8.0, 9.0]


def test_read_baselines_history_malformed_speed_skipped(tmp_path):
    _write_history(tmp_path, [
        ("tiny", 1, "oops", 10.0),
        ("tiny", 1, 5.0, "alsobad"),
    ])
    rows = BC._read_baselines_history(tmp_path, "tiny", 1)
    assert rows == [(5.0, None)]  # bad speed dropped, bad peak -> None


# --------------------------------------------------------------------------
# task.yaml-derived helpers
# --------------------------------------------------------------------------
def _write_task_yaml(tmp_path, text):
    (Path(tmp_path) / "task.yaml").write_text(text)


def test_dataset_name_for_tier(tmp_path):
    _write_task_yaml(tmp_path, "datasets:\n  - {tier: tiny, name: ds_tiny}\n")
    assert BC._dataset_name_for_tier(tmp_path, "tiny") == "ds_tiny"
    assert BC._dataset_name_for_tier(tmp_path, "huge") == ""


def test_dataset_name_for_tier_no_yaml(tmp_path):
    assert BC._dataset_name_for_tier(tmp_path, "tiny") == ""


def test_baseline_thread_invariant_true(tmp_path):
    for marker in ("not_applicable", "none", "non_parallel", "serial"):
        _write_task_yaml(tmp_path, f"baseline_threading: {marker}\n")
        assert BC._baseline_thread_invariant(tmp_path) is True


def test_baseline_thread_invariant_false(tmp_path):
    _write_task_yaml(tmp_path, "baseline_threading: parallel\n")
    assert BC._baseline_thread_invariant(tmp_path) is False
    # absent marker
    _write_task_yaml(tmp_path, "task: x\n")
    assert BC._baseline_thread_invariant(tmp_path) is False


# --------------------------------------------------------------------------
# results.tsv baseline reader
# --------------------------------------------------------------------------
def test_read_results_baseline(tmp_path):
    _write_task_yaml(tmp_path, "datasets:\n  - {tier: tiny, name: dsx}\n")
    p = Path(tmp_path) / BC.RESULTS_FILE_REL
    p.write_text(
        "status\tthread\tdataset\tspeed_sec\tpeak_mb\n"
        "baseline\t1\tdsx\t4.0\t100\n"
        "baseline\t1\tdsx\t5.0\t110\n"  # later row wins
        "iter\t1\tdsx\t1.0\t10\n"
    )
    res = BC._read_results_baseline(tmp_path, "tiny", 1)
    assert res == (5.0, 110.0)


def test_read_results_baseline_missing_file(tmp_path):
    assert BC._read_results_baseline(tmp_path, "tiny", 1) is None


def test_read_results_baseline_tier_substring_match(tmp_path):
    _write_task_yaml(tmp_path, "datasets: []\n")
    p = Path(tmp_path) / BC.RESULTS_FILE_REL
    p.write_text(
        "status\tthread\tdataset\tspeed_sec\tpeak_mb\n"
        "baseline\t1\tmydata_tiny_v2\t3.0\t30\n"
    )
    # dataset name contains "tiny" -> matches
    assert BC._read_results_baseline(tmp_path, "tiny", 1) == (3.0, 30.0)


# --------------------------------------------------------------------------
# ref-dir discovery
# --------------------------------------------------------------------------
def test_candidate_ref_dirs(tmp_path):
    cands = BC._candidate_ref_dirs(tmp_path, "tiny")
    assert cands[0].name == "reference_output_tiny"
    assert cands[1].parts[-2:] == ("reference_outputs", "tiny")


def test_first_populated_ref_dir_new_layout(tmp_path):
    d = Path(tmp_path) / "reference_output_tiny"
    d.mkdir()
    (d / "out.txt").write_text("x")
    assert BC._first_populated_ref_dir(tmp_path, "tiny") == d


def test_first_populated_ref_dir_old_layout(tmp_path):
    d = Path(tmp_path) / "reference_outputs" / "tiny"
    d.mkdir(parents=True)
    (d / "out.txt").write_text("x")
    assert BC._first_populated_ref_dir(tmp_path, "tiny") == d


def test_first_populated_ref_dir_empty_is_none(tmp_path):
    (Path(tmp_path) / "reference_output_tiny").mkdir()  # empty
    assert BC._first_populated_ref_dir(tmp_path, "tiny") is None
    assert BC._first_populated_ref_dir(tmp_path, "missing") is None


def test_ref_relpath_inside_task(tmp_path):
    ref = Path(tmp_path) / "reference_output_tiny"
    ref.mkdir()
    assert BC._ref_relpath(tmp_path, ref) == "reference_output_tiny"


def test_ref_relpath_outside_task_returns_str(tmp_path):
    outside = tmp_path.parent / "elsewhere"
    # not under task_dir -> ValueError path -> returns str(ref_dir)
    assert BC._ref_relpath(tmp_path, outside) == str(outside)


# --------------------------------------------------------------------------
# _collect_versions
# --------------------------------------------------------------------------
class _FakePatch:
    def __init__(self, targets):
        self.targets = targets


def test_collect_versions_known_pkg():
    # json is stdlib (no dist version) but pytest is installed -> use it
    p = _FakePatch([("pytest", "main", lambda: None)])
    out = BC._collect_versions(p)
    assert "pytest" in out
    assert isinstance(out["pytest"], str)


def test_collect_versions_unresolvable_omitted():
    p = _FakePatch([("json", "dumps", lambda: None)])
    out = BC._collect_versions(p)
    # json has no installed distribution -> omitted, not None-stamped
    assert "json" not in out


# --------------------------------------------------------------------------
# write_cached_baseline round-trip + validation
# --------------------------------------------------------------------------
def test_write_cached_baseline_requires_samples(tmp_path):
    with pytest.raises(ValueError, match="at least one speed sample"):
        BC.write_cached_baseline(str(tmp_path), "tiny", 1, "ds",
                                 speeds=[], peaks=[], versions={},
                                 ref_dir=str(tmp_path), produced_by="attest")


def test_write_cached_baseline_persists_entry(tmp_path):
    ref = Path(tmp_path) / "reference_output_tiny"
    ref.mkdir()
    (ref / "out.txt").write_text("hello")
    entry = BC.write_cached_baseline(
        str(tmp_path), "tiny", 4, "ds_tiny",
        speeds=[4.0, 6.0], peaks=[100.0, 120.0],
        versions={"pkg": "1.2.3"}, ref_dir=str(ref),
        produced_by="attest",
    )
    assert entry["n_reps"] == 2
    assert entry["speed_mean"] == 5.0
    assert entry["upstream_versions"] == {"pkg": "1.2.3"}
    assert entry["produced_by"] == "attest"
    assert "output_artifact_sha256" in entry
    # persisted under the right (tier, threads) key
    data = BC._load_noise(tmp_path)
    assert data["tiers"]["tiny"]["4"]["speed_mean"] == 5.0


def test_write_cached_baseline_single_sample_zero_stdev(tmp_path):
    ref = Path(tmp_path) / "reference_output_tiny"
    ref.mkdir()
    (ref / "out.txt").write_text("x")
    entry = BC.write_cached_baseline(
        str(tmp_path), "tiny", 1, "", speeds=[3.0], peaks=[None],
        versions={}, ref_dir=str(ref), produced_by="iteration",
    )
    assert entry["speed_stdev"] == 0.0
    assert "upstream_versions" not in entry  # empty versions omitted


# --------------------------------------------------------------------------
# populate_persistent_ref_dir
# --------------------------------------------------------------------------
def test_populate_persistent_ref_dir_copies(tmp_path):
    src = Path(tmp_path) / "src"
    src.mkdir()
    (src / "a.txt").write_text("data")
    sub = src / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("nested")
    dest = BC.populate_persistent_ref_dir(str(tmp_path), "tiny", str(src))
    destp = Path(dest)
    assert destp.name == "reference_output_tiny"
    assert (destp / "a.txt").read_text() == "data"
    assert (destp / "sub" / "b.txt").read_text() == "nested"


def test_populate_persistent_ref_dir_no_clobber(tmp_path):
    dest_pre = Path(tmp_path) / "reference_output_tiny"
    dest_pre.mkdir()
    (dest_pre / "authoritative.txt").write_text("keep me")
    src = Path(tmp_path) / "src"
    src.mkdir()
    (src / "attest.txt").write_text("should not appear")
    out = BC.populate_persistent_ref_dir(str(tmp_path), "tiny", str(src))
    assert Path(out) == dest_pre
    assert (dest_pre / "authoritative.txt").exists()
    assert not (dest_pre / "attest.txt").exists()  # no clobber


def test_populate_persistent_ref_dir_missing_src(tmp_path):
    out = BC.populate_persistent_ref_dir(str(tmp_path), "tiny",
                                         str(tmp_path / "does_not_exist"))
    # dest dir is created but empty
    assert Path(out).is_dir()


# --------------------------------------------------------------------------
# load_cached_baseline — the integration of the above
# --------------------------------------------------------------------------
def _make_task_with_ref(tmp_path, tier="tiny", content="hello"):
    ref = Path(tmp_path) / f"reference_output_{tier}"
    ref.mkdir()
    (ref / "out.txt").write_text(content)
    return ref


def test_load_cached_baseline_miss_no_ref_dir(tmp_path):
    p = _FakePatch([("json", "dumps", lambda: None)])
    assert BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, p) is None


def test_load_cached_baseline_hit_from_noise_json(tmp_path):
    _make_task_with_ref(tmp_path)
    BC._save_noise(tmp_path, {
        "schema_version": 2,
        "tiers": {"tiny": {"1": {"speed_mean": 5.0, "speed_stdev": 0.5,
                                 "peak_mean": 100.0, "n_reps": 3}}},
    })
    p = _FakePatch([("json", "dumps", lambda: None)])
    cached = BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, p)
    assert cached is not None
    assert cached.timing_mean == 5.0
    assert cached.timing_stdev == 0.5
    assert cached.peak_mb == 100.0
    assert cached.n_reps_observed == 3
    assert cached.timing_source == "noise_json"
    assert cached.has_version_stamp is False  # no upstream_versions stamped


def test_load_cached_baseline_version_mismatch_returns_none(tmp_path):
    _make_task_with_ref(tmp_path)
    BC._save_noise(tmp_path, {
        "schema_version": 2,
        "tiers": {"tiny": {"1": {
            "speed_mean": 5.0,
            "upstream_versions": {"pytest": "0.0.0-wrong"},
        }}},
    })
    p = _FakePatch([("pytest", "main", lambda: None)])
    # stamped version != installed pytest version -> miss
    assert BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, p) is None


def test_load_cached_baseline_version_match_has_stamp(tmp_path):
    import importlib.metadata as m
    pytest_ver = m.version("pytest")
    _make_task_with_ref(tmp_path)
    BC._save_noise(tmp_path, {
        "schema_version": 2,
        "tiers": {"tiny": {"1": {
            "speed_mean": 5.0,
            "upstream_versions": {"pytest": pytest_ver},
        }}},
    })
    p = _FakePatch([("pytest", "main", lambda: None)])
    cached = BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, p)
    assert cached is not None
    assert cached.has_version_stamp is True


def test_load_cached_baseline_size_mismatch_returns_none(tmp_path):
    ref = _make_task_with_ref(tmp_path, content="hello")
    BC._save_noise(tmp_path, {
        "schema_version": 2,
        "tiers": {"tiny": {"1": {
            "speed_mean": 5.0,
            "output_artifact_size_bytes": 999999,  # wrong size
        }}},
    })
    p = _FakePatch([("json", "dumps", lambda: None)])
    assert BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, p) is None


def test_load_cached_baseline_history_fallback(tmp_path):
    _make_task_with_ref(tmp_path)
    # no noise.json entry; provide baselines_history instead
    _write_history(tmp_path, [("tiny", 1, 4.0, 100.0), ("tiny", 1, 6.0, 120.0)])
    p = _FakePatch([("json", "dumps", lambda: None)])
    cached = BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, p)
    assert cached is not None
    assert cached.timing_source == "baselines_history"
    assert cached.timing_mean == 5.0
    assert cached.n_reps_observed == 2


def test_load_cached_baseline_results_last_resort(tmp_path):
    _make_task_with_ref(tmp_path)
    _write_task_yaml(tmp_path, "datasets:\n  - {tier: tiny, name: dsx}\n")
    p = Path(tmp_path) / BC.RESULTS_FILE_REL
    p.write_text(
        "status\tthread\tdataset\tspeed_sec\tpeak_mb\n"
        "baseline\t1\tdsx\t7.0\t70\n"
    )
    fp = _FakePatch([("json", "dumps", lambda: None)])
    cached = BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, fp)
    assert cached is not None
    assert cached.timing_source == "results_tsv"
    assert cached.timing_mean == 7.0


def test_load_cached_baseline_zero_timing_returns_none(tmp_path):
    _make_task_with_ref(tmp_path)
    BC._save_noise(tmp_path, {
        "schema_version": 2,
        "tiers": {"tiny": {"1": {"speed_mean": 0.0}}},
    })
    p = _FakePatch([("json", "dumps", lambda: None)])
    assert BC.load_cached_baseline("demo", str(tmp_path), "tiny", 1, p) is None


def test_load_cached_baseline_thread_invariant_remap(tmp_path):
    # baseline_threading: not_applicable -> a request for threads=8 reuses
    # the thread=1 cache entry.
    _make_task_with_ref(tmp_path)
    _write_task_yaml(tmp_path, "baseline_threading: not_applicable\n")
    BC._save_noise(tmp_path, {
        "schema_version": 2,
        "tiers": {"tiny": {"1": {"speed_mean": 9.0}}},
    })
    p = _FakePatch([("json", "dumps", lambda: None)])
    cached = BC.load_cached_baseline("demo", str(tmp_path), "tiny", 8, p)
    assert cached is not None
    assert cached.timing_mean == 9.0


# --------------------------------------------------------------------------
# pretty-print helpers
# --------------------------------------------------------------------------
def test_cache_hit_msg_with_stamp_and_sigma():
    c = BC.CachedBaseline(
        timing_mean=5.0, timing_stdev=0.3, peak_mb=100.0,
        ref_dir_path="/x", n_reps_observed=3, has_version_stamp=True,
        produced_by="iteration", timing_source="noise_json",
    )
    msg = BC.cache_hit_msg("tiny", c)
    assert "HIT tiny" in msg
    assert "5.000s" in msg
    assert "± 0.300s" in msg
    assert "first run will backfill" not in msg  # stamped


def test_cache_hit_msg_no_stamp_no_sigma():
    c = BC.CachedBaseline(
        timing_mean=5.0, timing_stdev=0.0, peak_mb=None,
        ref_dir_path="/x", n_reps_observed=1, has_version_stamp=False,
        produced_by="history-fallback", timing_source="results_tsv",
    )
    msg = BC.cache_hit_msg("medium", c)
    assert "first run will backfill" in msg
    assert "±" not in msg  # stdev == 0


def test_cache_miss_msg():
    assert BC.cache_miss_msg("large") == "[baseline-cache] MISS large: measuring fresh"
