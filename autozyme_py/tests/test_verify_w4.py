"""Wave-4 deep tests for autozyme._verify — push the reachable remainder.

Wave-1 ``test_verify_unit.py`` hit the pure helpers; wave-2 ``test_verify_deep.py``
drove the happy path + cache HIT/MISS/patched-only via the stdlib ``_test_json``
worker subprocess. This file closes the still-uncovered REACHABLE branches in
_verify.py that those two left behind:

  * deterministic baseline-cache confirm-OK ACCEPT path (verbose 420-435) by
    seeding noise.json with a mean+stdev that bracket the real ~1e-6 s call,
  * the confirm-rep merge into the version-stamp backfill (684-687) and the
    backfill dataset-name lookup from task.yaml,
  * the cache-MISS writeback dataset-name resolution (624-632) when task.yaml
    declares a dataset for the tier,
  * verbose multi-rep banner (362) + "all reps pass" summary line (596-597),
  * the per-tier verbose metrics-with-noise-label rendering,
  * pure-helper error/edge branches not yet hit: _assert_long_format superset
    header (939-940), _resolve_inst_speedup_path manifest-OSError (980-981) and
    missing-legacy-key (994-995), _write_inst_speedup_tsv non-baseline/patched
    variant filter (1026-1027) and failing-patched skip (1043-1044),
    _tier_dataset_map malformed-yaml swallow (1068-1069),
  * _append_package_verify_tsv pkg_version-from-tested_against (1103-1104),
  * verify_patch resolve_smoke-None ValueError (1350-1351).

Everything drives the synthetic stdlib-only ``_test_json`` patch (json.dumps
target, trivial smoke) so each spawned worker subprocess is < 0.1 s and no
optional upstream is needed. The worker subprocess IS spawned for real where
end-to-end coverage demands it. Disjoint from test_core_w4.py's patch claims.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import autozyme
from autozyme import _verify as V
from autozyme import _baseline_cache as BC
from autozyme._core import _REGISTRY, _import_submodule
from autozyme import verify_patch


# --------------------------------------------------------------------------
# Shared fixtures — a passing _test_json task with a NAMED dataset for the
# tier (so the writeback/backfill dataset-name lookups have something to find).
# --------------------------------------------------------------------------
_TASK_YAML_NAMED_DS = """\
task: _test_json
datasets:
  - {tier: tiny, name: ds_named_w4, path: /dev/null}
metrics:
  - {name: output_match, threshold: 1.0, comparator: gte}
"""

_EVALUATE_PY = """\
import os
ref_dir = os.environ.get("ZYME_REFERENCE_DIR", "reference_output")
test_dir = os.environ.get("ZYME_TEST_DIR", "pipeline")
ref = open(os.path.join(ref_dir, "output.txt")).read().strip()
test = open(os.path.join(test_dir, "output.txt")).read().strip()
print(f"output_match: {1.0 if test == ref else 0.0}")
"""


@pytest.fixture(autouse=True)
def _ensure_test_patch_registered():
    _import_submodule("_test_json")
    yield
    if _REGISTRY.get("_test_json") and _REGISTRY["_test_json"].injected:
        autozyme.deactivate("_test_json")


@pytest.fixture()
def named_ds_task(tmp_path: Path) -> Path:
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML_NAMED_DS)
    (td / "evaluate.py").write_text(_EVALUATE_PY)
    return td


def _read_tsv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _seed_cache_then_relax(task_dir: Path, *, mean: float, stdev: float) -> str:
    """Run once to populate the cache+ref_dir, then overwrite the persisted
    mean/stdev so the next run's confirmation rep is DETERMINISTICALLY within
    tolerance (mean ~ the real call time, generous stdev). Returns the thread
    key under which the entry was stored."""
    verify_patch("_test_json", str(task_dir), tiers=("tiny",),
                 reps=1, verbose=False, use_baseline_cache=True)
    noise = BC._load_noise(task_dir)
    thr_key = next(iter(noise["tiers"]["tiny"]))
    noise["tiers"]["tiny"][thr_key]["speed_mean"] = mean
    noise["tiers"]["tiny"][thr_key]["speed_stdev"] = stdev
    BC._save_noise(task_dir, noise)
    return thr_key


# ==========================================================================
# 1. Cache HIT confirm-OK ACCEPT path (verbose 420-435) — deterministic
# ==========================================================================
def test_cache_confirm_ok_accept_verbose(named_ds_task, capsys):
    """Seed a cache mean ~1ms with a 1s stdev so the real sub-microsecond
    confirmation measurement is comfortably within 3sigma. The confirm-OK
    branch prints 'confirm OK' and trusts the cached mean."""
    _seed_cache_then_relax(named_ds_task, mean=1e-3, stdev=1.0)
    capsys.readouterr()  # drain seeding output
    results = verify_patch(
        "_test_json", str(named_ds_task), tiers=("tiny",),
        reps=1, verbose=True, use_baseline_cache=True,
        baseline_confirm_sigma=3.0,
    )
    err = capsys.readouterr().err
    assert "[baseline-cache] HIT" in err
    assert "confirm OK" in err
    assert "trusting cache" in err
    # confirm-OK accepts the cached mean as the reported baseline timing
    assert results[0]["all_pass"] is True
    assert results[0]["baseline_secs"] == [pytest.approx(1e-3)]


def test_cache_confirm_ok_triggers_backfill_with_confirm_sample(named_ds_task,
                                                                capsys):
    """On confirm-OK the entry's version stamp is backfilled, MERGING the
    confirm-rep observation with the cached mean so n>=2 (684-687 + the
    dataset-name lookup from task.yaml in the backfill block 668-676)."""
    # First seed lacks a version stamp the first time around, so the backfill
    # block fires. Use a generous stdev for a deterministic confirm-OK.
    _seed_cache_then_relax(named_ds_task, mean=2e-3, stdev=1.0)
    capsys.readouterr()
    verify_patch("_test_json", str(named_ds_task), tiers=("tiny",),
                 reps=1, verbose=True, use_baseline_cache=True)
    err = capsys.readouterr().err
    # backfill stamps upstream_versions + sha onto the existing artifact
    assert "BACKFILL" in err
    # the cache entry should now carry the dataset name we declared
    noise = BC._load_noise(named_ds_task)
    entry = next(iter(noise["tiers"]["tiny"].values()))
    assert entry.get("dataset_name") == "ds_named_w4"


def test_cache_hit_no_confirm_fast_path_sets_accepted(named_ds_task, capsys):
    """no_baseline_confirm=True on a cache HIT copies the cached outputs and
    flips cache_accepted True inside the rep loop (the 394-395 branch), running
    NO baseline subprocess, then backfills the version stamp."""
    _seed_cache_then_relax(named_ds_task, mean=1e-3, stdev=1.0)
    capsys.readouterr()
    results = verify_patch(
        "_test_json", str(named_ds_task), tiers=("tiny",),
        reps=2, verbose=True, use_baseline_cache=True,
        no_baseline_confirm=True,
    )
    err = capsys.readouterr().err
    assert "[baseline-cache] HIT" in err
    # no confirmation rep was run; every rep copies the cached outputs and
    # reports the cached mean (the rep count may auto-escalate from 2->3 when
    # the sub-millisecond patched spread exceeds 1.20x, so don't pin the
    # length — only that all baseline samples equal the cached mean).
    assert results[0]["all_pass"] is True
    assert len(results[0]["baseline_secs"]) >= 2
    assert all(b == pytest.approx(1e-3) for b in results[0]["baseline_secs"])
    assert "BACKFILL" in err


def test_writeback_failure_is_warned_not_raised(named_ds_task, monkeypatch,
                                                 capsys):
    """A cache-MISS writeback whose write_cached_baseline raises is caught and
    only WARNed (verbose), and the run still returns a passing verdict
    (653-658)."""
    def boom(**kwargs):
        raise RuntimeError("simulated noise.json write failure")

    monkeypatch.setattr(V, "write_cached_baseline", boom)
    results = verify_patch(
        "_test_json", str(named_ds_task), tiers=("tiny",),
        reps=1, verbose=True, use_baseline_cache=True,
    )
    assert results[0]["all_pass"] is True
    err = capsys.readouterr().err
    assert "writeback failed" in err


def test_backfill_failure_is_warned_not_raised(named_ds_task, monkeypatch,
                                               capsys):
    """A cache-HIT confirm-OK backfill whose write_cached_baseline raises is
    caught + WARNed (706-711) without failing the run."""
    _seed_cache_then_relax(named_ds_task, mean=1e-3, stdev=1.0)
    capsys.readouterr()
    calls = {"n": 0}
    real = V.write_cached_baseline

    def boom_on_backfill(**kwargs):
        # the only write on a confirm-OK HIT run is the backfill; raise there
        if kwargs.get("produced_by") == "attest-backfill":
            raise RuntimeError("simulated backfill failure")
        return real(**kwargs)

    monkeypatch.setattr(V, "write_cached_baseline", boom_on_backfill)
    results = verify_patch(
        "_test_json", str(named_ds_task), tiers=("tiny",),
        reps=1, verbose=True, use_baseline_cache=True,
    )
    assert results[0]["all_pass"] is True
    err = capsys.readouterr().err
    assert "backfill failed" in err


def test_promote_refdir_failure_is_warned(named_ds_task, monkeypatch, capsys):
    """If populate_persistent_ref_dir raises OSError on the first fresh rep, the
    failure is caught + WARNed (497-503) and the run still completes."""
    def boom(task_dir, tier, ref_dir):
        raise OSError("simulated promote failure")

    monkeypatch.setattr(V, "populate_persistent_ref_dir", boom)
    results = verify_patch(
        "_test_json", str(named_ds_task), tiers=("tiny",),
        reps=1, verbose=True, use_baseline_cache=True,
    )
    assert results[0]["all_pass"] is True
    err = capsys.readouterr().err
    assert "failed to promote ref_dir" in err


# ==========================================================================
# 2. Cache MISS writeback — dataset name resolved from task.yaml (624-632)
# ==========================================================================
def test_cache_miss_writeback_records_dataset_name(named_ds_task):
    """A fresh cached run promotes the ref_dir + writes noise.json, and the
    writeback resolves the tier's dataset name from task.yaml (the 624-632
    yaml-walk branch)."""
    verify_patch("_test_json", str(named_ds_task), tiers=("tiny",),
                 reps=1, verbose=False, use_baseline_cache=True)
    noise = BC._load_noise(named_ds_task)
    entry = next(iter(noise["tiers"]["tiny"].values()))
    assert entry["produced_by"] == "attest"
    assert entry.get("dataset_name") == "ds_named_w4"


# ==========================================================================
# 3. Verbose multi-rep: rep banner (362) + "all reps pass" line (596-597)
# ==========================================================================
def test_verbose_multi_rep_summary(named_ds_task, capsys):
    """reps=2 verbose prints the per-rep '=== rep k/N ===' banner and the
    'all reps pass' aggregate line (only emitted when reps_actual > 1)."""
    verify_patch(
        "_test_json", str(named_ds_task), tiers=("tiny",),
        reps=2, verbose=True, use_baseline_cache=False,
    )
    err = capsys.readouterr().err
    assert "=== rep 1/2 ===" in err
    assert "=== rep 2/2 ===" in err
    assert "all reps pass" in err


# ==========================================================================
# 4. Verbose metrics with a noise-calibrated threshold LABEL (the
#    threshold_label != "absolute" suffix branch at 569-571)
# ==========================================================================
def test_verbose_metric_noise_label(tmp_path, capsys):
    """A stochastic metric (absolute_floor + calibrated intrinsic_noise)
    yields a non-'absolute' threshold label, exercising the suffix print."""
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(
        "task: _test_json\n"
        "datasets:\n  - {tier: tiny, name: dsn, path: /dev/null}\n"
        "metrics:\n  - {name: output_match, absolute_floor: 0.5,"
        " noise_multiplier: 2.0, comparator: gte}\n"
        "intrinsic_noise:\n  tiny:\n    output_match: 0.99\n"
    )
    (td / "evaluate.py").write_text(_EVALUATE_PY)
    res = verify_patch(
        "_test_json", str(td), tiers=("tiny",),
        reps=1, verbose=True, use_baseline_cache=False,
    )
    err = capsys.readouterr().err
    # the metric still passes (output_match == 1.0 >= effective 0.98)
    assert res[0]["all_pass"] is True
    # the verbose metric line carries the bracketed noise label, proving the
    # threshold_label != "absolute" suffix branch ran. effective gte threshold
    # = max(0.5, 1 - 2*(1-0.99)) = 0.98.
    assert "[max(floor=0.5" in err


# ==========================================================================
# 5. _assert_long_format_or_empty — superset (forward-compat) header (939-940)
# ==========================================================================
def test_assert_long_format_superset_header_ok(tmp_path):
    """A header that is a strict SUPERSET of the current schema passes the
    forward-compat check (not just the exact-match shortcut at 937)."""
    p = tmp_path / "v.tsv"
    p.write_text(V._PACKAGE_VERIFY_HEADER + "\textra_future_col\n")
    V._assert_long_format_or_empty(str(p))  # must not raise


# ==========================================================================
# 6. _resolve_inst_speedup_path — manifest read OSError + missing legacy key
# ==========================================================================
def test_resolve_inst_speedup_manifest_oserror(tmp_path, monkeypatch):
    """A manifest that exists but can't be opened (OSError) -> None (980-981)."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "demo_attest_manifest.yaml").write_text("tasks: []\n")
    real_open = open

    def boom(path, *a, **k):
        if str(path).endswith("demo_attest_manifest.yaml"):
            raise OSError("forced read error")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", boom)
    assert V._resolve_inst_speedup_path(str(tmp_path), "demo") is None


def test_resolve_inst_speedup_no_matching_task(tmp_path):
    """Manifest reachable but its tasks don't match task_dir -> no legacy_key
    -> None (994-995)."""
    task = tmp_path / "task"
    task.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    # task path points at a DIFFERENT subdir, so the abspath compare misses
    (scripts / "demo_attest_manifest.yaml").write_text(
        "tasks:\n  - {path: some_other_task, legacy_key: k}\n")
    assert V._resolve_inst_speedup_path(str(task), "demo") is None


# ==========================================================================
# 7. _write_inst_speedup_tsv — variant filter + failing-patched skip
# ==========================================================================
def _full_row(**kw):
    cols = V._PACKAGE_VERIFY_HEADER.split("\t")
    r = {c: "" for c in cols}
    r.update(kw)
    return r


def test_write_inst_speedup_ignores_unknown_variant(tmp_path):
    """A measured row with a non-baseline/patched variant is dropped by
    _measured_variant (1026-1027); only baseline+patched survive."""
    out_path = tmp_path / "demo" / "speedups" / "demo_k.tsv"
    common = dict(timestamp="t1", patch_name="demo", tier="tiny",
                  framework_version="0.3.0", note="", system_os="macOS 14",
                  system_cpu="cpu", system_ram_gb="16.0", system_threads="1")
    rows = [
        _full_row(variant="baseline", sec="4.0", **common),
        _full_row(variant="patched", sec="2.0", **common),
        # a stray 'iter' variant with a sec value -> _measured_variant returns ""
        _full_row(variant="iter", sec="3.0", **common),
    ]
    rows[1]["pass"] = "true"
    V._write_inst_speedup_tsv(str(out_path), rows)
    written = _read_tsv(out_path)
    assert {r["variant"] for r in written} == {"baseline", "patched"}
    assert len(written) == 2


def test_write_inst_speedup_skips_failing_patched_in_passing_group(tmp_path):
    """Two patched reps in the same group: one passes (keeps the group) and one
    fails -> the failing patched row is skipped (1043-1044) while baseline +
    passing patched remain."""
    out_path = tmp_path / "demo" / "speedups" / "demo_k.tsv"
    common = dict(timestamp="t1", patch_name="demo", tier="tiny",
                  framework_version="0.3.0", note="", system_os="macOS 14",
                  system_cpu="cpu", system_ram_gb="16.0", system_threads="1")
    rows = [
        _full_row(variant="baseline", sec="4.0", **common),
        _full_row(variant="patched", sec="2.0", **common),  # pass
        _full_row(variant="patched", sec="9.0", **common),  # fail
    ]
    rows[1]["pass"] = "true"
    rows[2]["pass"] = "false"
    V._write_inst_speedup_tsv(str(out_path), rows)
    written = _read_tsv(out_path)
    # group kept (a passing patched exists); failing patched dropped
    variants = sorted(r["variant"] for r in written)
    assert variants == ["baseline", "patched"]
    assert all(r["pass"] in ("", "true") for r in written)


# ==========================================================================
# 8. _tier_dataset_map — malformed YAML is swallowed -> {} (1068-1069)
# ==========================================================================
def test_tier_dataset_map_malformed_yaml(tmp_path):
    (tmp_path / "task.yaml").write_text("datasets: [unterminated\n  : : :\n")
    assert V._tier_dataset_map(str(tmp_path)) == {}


# ==========================================================================
# 9. _append_package_verify_tsv — pkg_version from a patch's tested_against
#    (1103-1104) flows into the package_version column.
# ==========================================================================
def test_append_package_verify_pkg_version_from_tested_against(tmp_path):
    """Register a patch carrying tested_against; the long-format writer reads
    it from the registry and stamps package_version on every row."""
    autozyme.register_patch(
        "w4_pkgver",
        [("json", "JSONEncoder", object)],   # disjoint claim, never activated
        tested_against="json 9.8.7",
    )
    try:
        (tmp_path / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: dsx}\n")
        rows = [{
            "tier": "tiny", "note": "",
            "baseline_secs": [4.0], "patched_secs": [2.0],
            "baseline_peaks_mb": [None], "patched_peaks_mb": [None],
            "per_rep_pass": [True], "metrics_json": "",
        }]
        V._append_package_verify_tsv(str(tmp_path), "w4_pkgver", rows)
        out = _read_tsv(tmp_path / "package_verify.tsv")
        assert out and all(r["package_version"] == "json 9.8.7" for r in out)
    finally:
        _REGISTRY.pop("w4_pkgver", None)


# ==========================================================================
# 10. verify_patch — patch registered but no smoke recipe -> ValueError (1350)
# ==========================================================================
def test_verify_patch_no_smoke_raises(tmp_path):
    """A patch with NO smoke recipe (and no attest/smoke.py under the task)
    raises ValueError before any subprocess spawns (1350-1351)."""
    autozyme.register_patch("w4_nosmoke", [("json", "JSONDecoder", object)])
    try:
        td = tmp_path / "task"
        td.mkdir()
        (td / "task.yaml").write_text(_TASK_YAML_NAMED_DS)
        (td / "evaluate.py").write_text(_EVALUATE_PY)
        with pytest.raises(ValueError, match="no smoke recipe"):
            verify_patch("w4_nosmoke", str(td), tiers=("tiny",))
    finally:
        _REGISTRY.pop("w4_nosmoke", None)


# ==========================================================================
# 11. verify_patch top-level OSError on TSV write is caught + warned (1425-1426)
# ==========================================================================
def test_verify_patch_tsv_write_oserror_warned(named_ds_task, monkeypatch,
                                                capsys):
    """If _append_package_verify_tsv raises OSError, verify_patch swallows it
    with a 'could not write package_verify.tsv' warning rather than failing."""
    def boom(task_dir, name, rows):
        raise OSError("disk full")

    monkeypatch.setattr(V, "_append_package_verify_tsv", boom)
    results = verify_patch(
        "_test_json", str(named_ds_task), tiers=("tiny",),
        reps=1, verbose=False, use_baseline_cache=False,
    )
    # the run itself still returns rows; only the persistence step warned
    assert results[0]["all_pass"] is True
    err = capsys.readouterr().err
    assert "could not write package_verify.tsv" in err
