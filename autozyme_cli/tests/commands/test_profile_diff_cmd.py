"""Unit tests for zyme.commands.profile.diff — pure structured diff +
profile resolution + render. No subprocess.

diff_profiles is the heart: hotspot appeared/disappeared/common deltas,
override timing deltas, and cross-backend / tier-mismatch warnings.
test_profile_archive.py already covers the `current` alias end-to-end via a
real run; here we drive the resolution + diff + render helpers directly on
synthetic profile.json files in tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zyme.commands.profile import diff


# ---------------------------------------------------------------------------
# _fmt_delta
# ---------------------------------------------------------------------------

def test_fmt_delta_positive():
    s = diff._fmt_delta(1.0, 1.5)
    assert s.startswith("+0.500s")
    assert "+50.0%" in s


def test_fmt_delta_negative():
    s = diff._fmt_delta(2.0, 1.0)
    assert s.startswith("-1.000s")
    assert "-50.0%" in s


def test_fmt_delta_none():
    assert diff._fmt_delta(None, 1.0) == "—"
    assert diff._fmt_delta(1.0, None) == "—"


def test_fmt_delta_zero_base_no_pct():
    s = diff._fmt_delta(0.0, 1.0)
    assert "%" not in s   # pct undefined when a == 0
    assert "+1.000s" in s


def test_fmt_delta_custom_unit_precision():
    s = diff._fmt_delta(0.001, 0.002, unit="ms", precision=4)
    assert "ms" in s
    assert "0.0010" in s


# ---------------------------------------------------------------------------
# _hotspot_index / _override_index
# ---------------------------------------------------------------------------

def test_hotspot_index_skips_labelless():
    data = {"hotspots": [{"label": "a"}, {"no_label": 1}, {"label": "b"}]}
    idx = diff._hotspot_index(data)
    assert set(idx) == {"a", "b"}


def test_override_index_keyed_by_name():
    data = {"override_summary": [{"name": "ov1", "calls": 3}, {"calls": 1}]}
    idx = diff._override_index(data)
    assert set(idx) == {"ov1"}


# ---------------------------------------------------------------------------
# diff_profiles
# ---------------------------------------------------------------------------

def _profile(hotspots=None, overrides=None, backend="cpu", tier="tiny",
             totals=None, lang="py"):
    return {
        "backend": backend, "lang": lang, "tier": tier,
        "hypothesis": "h", "timestamp": "2026-01-01T00:00:00+00:00",
        "totals": totals or {},
        "hotspots": hotspots or [],
        "override_summary": overrides or [],
    }


def test_diff_appeared_disappeared_common():
    a = _profile(hotspots=[
        {"label": "f_gone", "self_pct": 30.0, "self_time_s": 0.3, "rank": 1},
        {"label": "f_both", "self_pct": 50.0, "self_time_s": 0.5, "rank": 2},
    ])
    b = _profile(hotspots=[
        {"label": "f_both", "self_pct": 20.0, "self_time_s": 0.2, "rank": 1},
        {"label": "f_new", "self_pct": 60.0, "self_time_s": 0.6, "rank": 2},
    ])
    d = diff.diff_profiles(a, b, "A", "B")
    assert [h["label"] for h in d["appeared_in_b"]] == ["f_new"]
    assert [h["label"] for h in d["disappeared_from_a"]] == ["f_gone"]
    common = {c["label"]: c for c in d["common_hotspots"]}
    assert "f_both" in common
    assert common["f_both"]["a_pct"] == 50.0
    assert common["f_both"]["b_pct"] == 20.0
    assert common["f_both"]["a_rank"] == 2
    assert common["f_both"]["b_rank"] == 1


def test_diff_common_sorted_by_b_pct():
    a = _profile(hotspots=[
        {"label": "x", "self_pct": 10.0, "rank": 1},
        {"label": "y", "self_pct": 10.0, "rank": 2},
    ])
    b = _profile(hotspots=[
        {"label": "x", "self_pct": 20.0, "rank": 2},
        {"label": "y", "self_pct": 80.0, "rank": 1},
    ])
    d = diff.diff_profiles(a, b)
    # y has the higher B pct -> first.
    assert [c["label"] for c in d["common_hotspots"]] == ["y", "x"]


def test_diff_override_deltas_union_of_names():
    a = _profile(overrides=[{"name": "ov", "calls": 4, "total_s": 1.0,
                             "mean_s": 0.25, "n_workers": 1}])
    b = _profile(overrides=[
        {"name": "ov", "calls": 4, "total_s": 0.5, "mean_s": 0.125, "n_workers": 1},
        {"name": "ov2", "calls": 2, "total_s": 0.2, "mean_s": 0.1, "n_workers": 8},
    ])
    d = diff.diff_profiles(a, b)
    by_name = {o["name"]: o for o in d["override_diffs"]}
    assert set(by_name) == {"ov", "ov2"}
    assert by_name["ov"]["a_total_s"] == 1.0
    assert by_name["ov"]["b_total_s"] == 0.5
    # ov2 only in B -> a side is None.
    assert by_name["ov2"]["a_calls"] is None
    assert by_name["ov2"]["b_workers"] == 8


def test_diff_cross_backend_warning():
    a = _profile(backend="cpu")
    b = _profile(backend="full")
    d = diff.diff_profiles(a, b)
    assert any("cross-backend" in n for n in d["notes"])


def test_diff_tier_mismatch_warning():
    a = _profile(tier="tiny")
    b = _profile(tier="medium")
    d = diff.diff_profiles(a, b)
    assert any("tier mismatch" in n for n in d["notes"])


def test_diff_same_backend_tier_no_warnings():
    d = diff.diff_profiles(_profile(), _profile())
    assert d["notes"] == []


def test_diff_meta_extracted():
    a = _profile(totals={"wall_s": 2.0})
    b = _profile(totals={"wall_s": 1.0})
    d = diff.diff_profiles(a, b)
    assert d["a_totals"] == {"wall_s": 2.0}
    assert d["b_totals"] == {"wall_s": 1.0}
    assert d["a_meta"]["backend"] == "cpu"
    assert d["b_meta"]["lang"] == "py"


# ---------------------------------------------------------------------------
# render_diff — text rendering of the structured diff.
# ---------------------------------------------------------------------------

def test_render_diff_includes_wall_and_hotspots():
    a = _profile(hotspots=[{"label": "f_both", "self_pct": 50.0, "rank": 1}],
                 totals={"wall_s": 2.0})
    b = _profile(hotspots=[
        {"label": "f_both", "self_pct": 20.0, "rank": 1},
        {"label": "f_new", "self_pct": 60.0, "rank": 2},
    ], totals={"wall_s": 1.0})
    d = diff.diff_profiles(a, b, "before", "after")
    text = diff.render_diff(d)
    assert "[diff] before  →  after" in text
    assert "wall:" in text
    assert "f_both" in text
    assert "Appeared in after" in text
    assert "f_new" in text


def test_render_diff_with_overrides_and_warnings():
    a = _profile(backend="cpu",
                 overrides=[{"name": "ov", "calls": 4, "total_s": 1.0,
                             "mean_s": 0.25, "n_workers": 1}])
    b = _profile(backend="full",
                 overrides=[{"name": "ov", "calls": 4, "total_s": 0.5,
                             "mean_s": 0.125, "n_workers": 1}])
    d = diff.diff_profiles(a, b)
    text = diff.render_diff(d)
    assert "Override timing changes" in text
    assert "cross-backend" in text


def test_render_diff_disappeared_section():
    a = _profile(hotspots=[{"label": "old_fn", "self_pct": 40.0, "rank": 1}])
    b = _profile(hotspots=[])
    d = diff.diff_profiles(a, b, "A", "B")
    text = diff.render_diff(d)
    assert "Disappeared from A" in text
    assert "old_fn" in text


def test_render_diff_empty_is_minimal():
    d = diff.diff_profiles(_profile(), _profile())
    text = diff.render_diff(d)
    # header lines always present; no crash on empty content.
    assert text.startswith("[diff]")


# ---------------------------------------------------------------------------
# _load_profile + resolution helpers (path/dir/history/aliases).
# ---------------------------------------------------------------------------

def _write_profile(path: Path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_profile(**kw)))


def test_load_profile_direct_file(tmp_path):
    p = tmp_path / "p.json"
    _write_profile(p, backend="cpu")
    data, label = diff._load_profile(str(p))
    assert data["backend"] == "cpu"
    assert label == str(p)


def test_load_profile_directory(tmp_path):
    run_dir = tmp_path / "run1"
    _write_profile(run_dir / "profile.json", backend="full")
    data, label = diff._load_profile(str(run_dir))
    assert data["backend"] == "full"
    assert label.endswith("profile.json")


def test_load_profile_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="profile not found"):
        diff._load_profile(str(tmp_path / "absent.json"))


def test_load_profile_history_alias(tmp_path, monkeypatch):
    hist = tmp_path / "profile_history"
    _write_profile(hist / "20260101_cpu_tiny" / "profile.json", backend="cpu")
    monkeypatch.chdir(tmp_path)
    data, label = diff._load_profile("history:cpu")
    assert data["backend"] == "cpu"


def test_load_profile_history_no_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no profile_history"):
        diff._load_profile("history:foo")


def test_load_profile_history_no_match(tmp_path, monkeypatch):
    (tmp_path / "profile_history").mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no profile_history file matched"):
        diff._load_profile("history:nomatch")


def test_load_profile_current_alias_resolves_latest(tmp_path, monkeypatch):
    hist = tmp_path / "profile_history"
    _write_profile(hist / "run_old" / "profile.json", backend="cpu", tier="tiny")
    _write_profile(hist / "run_new" / "profile.json", backend="full", tier="tiny")
    # bump mtime of run_new so it's "latest".
    import os, time
    newp = hist / "run_new" / "profile.json"
    os.utime(newp, (time.time() + 10, time.time() + 10))
    monkeypatch.chdir(tmp_path)
    data, label = diff._load_profile("current")
    assert data["backend"] == "full"


def test_load_profile_current_falls_back_to_legacy(tmp_path, monkeypatch):
    # no profile_history -> _latest_history_profile returns pipeline/profile.json.
    _write_profile(tmp_path / "pipeline" / "profile.json", backend="cpu")
    monkeypatch.chdir(tmp_path)
    data, label = diff._load_profile("current")
    assert data["backend"] == "cpu"
    assert "pipeline" in label.replace("\\", "/")


def test_load_profile_current_missing_everything(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        diff._load_profile("current")


def test_profile_path_from_history_entry_dir_vs_file(tmp_path):
    d = tmp_path / "rundir"
    d.mkdir()
    assert diff._profile_path_from_history_entry(d) == d / "profile.json"
    f = tmp_path / "legacy.json"
    f.write_text("{}")
    assert diff._profile_path_from_history_entry(f) == f


# ---------------------------------------------------------------------------
# cmd_profile_diff — the argparse entry point.
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_cmd_profile_diff_text(tmp_path, monkeypatch, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_profile(a, backend="cpu", totals={"wall_s": 2.0})
    _write_profile(b, backend="cpu", totals={"wall_s": 1.0})
    args = _Args(diff_a=str(a), diff_b=str(b), json=False)
    diff.cmd_profile_diff(args)
    err = capsys.readouterr().err
    assert "[diff]" in err
    assert "wall:" in err


def test_cmd_profile_diff_json(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_profile(a, backend="cpu")
    _write_profile(b, backend="full")
    args = _Args(diff_a=str(a), diff_b=str(b), json=True)
    diff.cmd_profile_diff(args)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "common_hotspots" in parsed
    assert any("cross-backend" in n for n in parsed["notes"])


def test_cmd_profile_diff_missing_calls_die(tmp_path, monkeypatch):
    # die() raises SystemExit in zyme.utils; assert it propagates on missing file.
    args = _Args(diff_a=str(tmp_path / "nope.json"),
                 diff_b=str(tmp_path / "alsono.json"), json=False)
    with pytest.raises(SystemExit):
        diff.cmd_profile_diff(args)
