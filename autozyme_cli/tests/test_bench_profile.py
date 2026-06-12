"""Fast tests for the on-demand profile benchmark harness.

`bench_profile.py` is intentionally not a pytest benchmark because the real
matrix can take tens of minutes. These tests cover the harness logic that
decides whether a captured profile is trustworthy.
"""
from __future__ import annotations

import pytest

from tests import bench_profile


def _profile(**overrides) -> dict:
    data = {
        "schema_version": "1",
        "backend": "cpu",
        "lang": "py",
        "tier": "tiny",
        "hypothesis": "",
        "timestamp": "2026-05-09T18:17:52+00:00",
        "totals": {"wall_s": 1.0, "cpu_s": 0.9, "peak_mb": 64.0},
        "hotspots": [
            {
                "rank": 1,
                "label": "run.py:10:cpu_a",
                "self_time_s": 0.5,
                "total_time_s": 0.6,
                "self_pct": 50.0,
                "calls": 1,
                "raw": {"file": "run.py", "line": 10, "func": "cpu_a"},
            },
            {
                "rank": 2,
                "label": "run.py:20:cpu_b",
                "self_time_s": 0.3,
                "total_time_s": 0.4,
                "self_pct": 30.0,
                "calls": 1,
                "raw": {"file": "run.py", "line": 20, "func": "cpu_b"},
            },
        ],
        "actionable_hotspots": [
            {
                "rank": 1,
                "label": "pkg.fast_fn",
                "self_time_s": 2.0,
                "total_time_s": 2.0,
                "self_pct": None,
                "calls": 4,
                "raw": {"source": "override_summary", "name": "pkg.fast_fn"},
            }
        ],
        "notes": ["kind=deterministic unit=cpu_seconds"],
        "artifacts": {"raw": "profile_history/current/profile.out"},
        "override_summary": [
            {
                "name": "pkg.fast_fn",
                "calls": 4,
                "total_s": 2.0,
                "mean_s": 0.5,
                "min_s": 0.1,
                "max_s": 0.9,
                "n_workers": 2,
            },
        ],
        "override_markers": [
            {
                "name": "pkg.marker_fn",
                "order": 1,
                "source": "override_active",
            }
        ],
        "call_chains": [],
    }
    data.update(overrides)
    return data


def test_validate_accepts_schema_and_declared_checks():
    ok, detail = bench_profile.validate(
        _profile(),
        {
            "min_hotspots": 2,
            "top3_all": ["cpu_a", "cpu_b"],
            "rank1_contains": "cpu_a",
            "all_of": ["run.py", "cpu_b"],
            "notes_any_of": ["cpu_seconds"],
            "artifact_contains": "profile.out",
            "override_present": ["fast_fn"],
            "override_min_workers": {"name": "pkg.fast_fn", "min": 2},
            "actionable_rank1_contains": "fast_fn",
        },
    )
    assert ok, detail


def test_validate_rejects_missing_top_level_schema_field():
    data = _profile()
    data.pop("override_summary")

    ok, detail = bench_profile.validate(data, {})

    assert not ok
    assert "missing top-level fields" in detail
    assert "override_summary" in detail


def test_validate_rejects_malformed_hotspot_schema():
    data = _profile()
    data["hotspots"][0] = {
        "rank": 1,
        "label": "run.py:10:cpu_a",
        # missing raw and timing fields
    }

    ok, detail = bench_profile.validate(data, {})

    assert not ok
    assert "hotspot[0] missing fields" in detail


def test_validate_uses_raw_fields_when_searching_hotspots():
    data = _profile(
        hotspots=[
            {
                "rank": 1,
                "label": "opaque label",
                "self_time_s": None,
                "total_time_s": None,
                "self_pct": None,
                "calls": None,
                "raw": {"func": "native_symbol_from_raw"},
            }
        ]
    )

    ok, detail = bench_profile.validate(
        data,
        {"rank1_contains": "native_symbol_from_raw"},
    )

    assert ok, detail


def test_validate_rejects_forbidden_hotspot_text():
    ok, detail = bench_profile.validate(
        _profile(),
        {"not_any_of": ["cpu_b"]},
    )

    assert not ok
    assert "forbidden" in detail
    assert "cpu_b" in detail


def test_validate_reports_too_many_hotspots():
    ok, detail = bench_profile.validate(
        _profile(),
        {"max_hotspots": 1},
    )

    assert not ok
    assert "max_hotspots: 2 > 1" in detail


def test_validate_reports_override_worker_shortfall():
    ok, detail = bench_profile.validate(
        _profile(),
        {"override_min_workers": {"name": "pkg.fast_fn", "min": 3}},
    )

    assert not ok
    assert "expected >= 3" in detail


def test_validate_accepts_override_marker_checks():
    ok, detail = bench_profile.validate(
        _profile(),
        {"override_marker_present": ["pkg.marker_fn"]},
    )

    assert ok, detail


def test_cell_result_variance_and_overhead_flags():
    cell = bench_profile.CellResult(fixture="fx", backend="cpu")
    cell.walls = [1.0, 1.0, 2.0]

    cell.compute_variance()
    cell.compute_overhead(baseline_s=1.0, limit_x=1.2)

    assert cell.wall_cv_pct is not None
    assert cell.unstable
    assert cell.overhead_x == pytest.approx(1.0)
    assert cell.overhead_pass


def test_result_counts_include_overhead_failures():
    good = bench_profile.CellResult(fixture="fx", backend="cpu", check_pass=True)
    slow = bench_profile.CellResult(fixture="fx", backend="mem", check_pass=True)
    slow.overhead_x = 4.0
    slow.overhead_limit_x = 2.0
    slow.overhead_pass = False
    failed = bench_profile.CellResult(fixture="fx", backend="native", check_pass=False)
    skipped = bench_profile.CellResult(fixture="fx", backend="full", skipped=True)

    counts = bench_profile.result_counts([({}, 1.0, [good, slow, failed, skipped])])

    assert counts == {
        "pass": 2,
        "fail": 1,
        "skip": 1,
        "unstable": 0,
        "overhead_fail": 1,
    }


def test_bench_fixture_uses_backend_specific_skip_reason(tmp_path, monkeypatch):
    (tmp_path / "task.yaml").write_text("name: skip-fixture\n")
    monkeypatch.setattr(bench_profile, "baseline_wall", lambda *_args: 1.0)

    def fail_profile_run(*_args):
        raise AssertionError("skipped backend should not run profile")

    monkeypatch.setattr(bench_profile, "profile_run", fail_profile_run)

    _base, cells = bench_profile.bench_fixture(
        {
            "name": "skip-fixture",
            "task": str(tmp_path),
            "tier": "tiny",
            "lang": "py",
            "skip_backends": ["mem"],
            "skip_reasons": {"mem": "executor env lacks memray"},
        },
        reps=1,
        only_backends=["mem"],
    )

    assert len(cells) == 1
    assert cells[0].skipped
    assert cells[0].skip_reason == "executor env lacks memray"


def test_select_fixtures_excludes_slow_by_default(monkeypatch):
    monkeypatch.setattr(
        bench_profile,
        "FIXTURES",
        [
            {"name": "fast", "slow": False},
            {"name": "implicit_fast"},
            {"name": "slow", "slow": True},
        ],
    )

    selected = bench_profile.select_fixtures(None)

    assert [fx["name"] for fx in selected] == ["fast", "implicit_fast"]


def test_select_fixtures_includes_slow_when_requested_or_explicit(monkeypatch):
    monkeypatch.setattr(
        bench_profile,
        "FIXTURES",
        [
            {"name": "fast"},
            {"name": "slow", "slow": True},
        ],
    )

    selected_all = bench_profile.select_fixtures(None, include_slow=True)
    selected_explicit = bench_profile.select_fixtures({"slow"})

    assert [fx["name"] for fx in selected_all] == ["fast", "slow"]
    assert [fx["name"] for fx in selected_explicit] == ["slow"]
