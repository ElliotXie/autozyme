"""Unit tests for zyme.commands.profile.render — the evidence-card stderr
formatter and its per-hotspot / call-chain / byte-humanizing helpers.

render.py is pure formatting: profile dict -> grep-friendly stderr lines. We
capture stderr and assert the lines the agent greps for. No subprocess.
"""
from __future__ import annotations

import pytest

from zyme.commands.profile import render


def _capture(profile_data, profile_dir=None):
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        render.evidence_card(profile_data, profile_dir=profile_dir)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _humanize_bytes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("n", "expect"), [
    (512, "512.0B"),
    (1536, "1.5KB"),
    (5 * 1024**2, "5.0MB"),
    (3 * 1024**3, "3.0GB"),
    (2 * 1024**4, "2.0TB"),
    (5 * 1024**5, "5120.0TB"),   # caps at TB
])
def test_humanize_bytes(n, expect):
    assert render._humanize_bytes(n) == expect


# ---------------------------------------------------------------------------
# header line
# ---------------------------------------------------------------------------

def test_header_line_has_backend_lang_tier():
    out = _capture({"backend": "cpu", "lang": "py", "tier": "tiny",
                    "totals": {}, "hotspots": []})
    assert "[profile] backend=cpu lang=py tier=tiny" in out


def test_header_includes_totals_when_present():
    out = _capture({"backend": "cpu", "lang": "py", "tier": "tiny",
                    "totals": {"wall_s": 1.234, "cpu_s": 1.0, "peak_mb": 256.0},
                    "hotspots": []})
    assert "wall=1.234s" in out
    assert "cpu=1.000s" in out
    assert "peak=256.0MB" in out


def test_no_hotspots_message():
    out = _capture({"backend": "cpu", "lang": "py", "tier": "tiny",
                    "totals": {}, "hotspots": []})
    assert "No hotspots extracted" in out


# ---------------------------------------------------------------------------
# hotspot rendering
# ---------------------------------------------------------------------------

def _hotspot(**kw):
    base = {"rank": 1, "label": "run.py:10:fn", "self_time_s": 0.5,
            "self_pct": 50.0, "calls": 3, "raw": {}}
    base.update(kw)
    return base


def test_top_hotspots_rendered():
    out = _capture({"backend": "cpu", "lang": "py", "tier": "tiny",
                    "totals": {}, "hotspots": [_hotspot()]})
    assert "Top 1 hotspots" in out
    assert "run.py:10:fn" in out
    assert "self=" in out
    assert "50.0%" in out   # rendered as "( 50.0%)" via :5.1f
    assert "calls=3" in out


def test_format_hotspot_layer_and_ncalls():
    h = _hotspot(calls=None, layer="task", n_calls=7, raw={})
    line = render._format_hotspot(h, "cpu")
    assert "layer=task" in line
    assert "n_calls=7" in line


def test_format_hotspot_ncalls_est_fallback():
    h = _hotspot(calls=None, n_calls_est=42, raw={})
    line = render._format_hotspot(h, "cpu")
    assert "n_calls=42" in line


def test_format_hotspot_ncalls_suppressed_when_calls_present():
    # avoid double-counting: cProfile gives `calls`, so n_calls is omitted.
    h = _hotspot(calls=5, n_calls=5, raw={})
    line = render._format_hotspot(h, "cpu")
    assert "calls=5" in line
    assert "n_calls=" not in line


def test_format_hotspot_full_backend_py_native_split():
    h = _hotspot(raw={"cpu_python_pct": 30.0, "cpu_native_pct": 70.0,
                      "mem_peak_mb": 128.0})
    line = render._format_hotspot(h, "full")
    assert "py=30.0%" in line
    assert "native=70.0%" in line
    assert "peak_mb=128.0" in line


def test_format_hotspot_mem_backend_alloc():
    h = _hotspot(raw={"alloc_bytes": 5 * 1024**2, "alloc_count": 12})
    line = render._format_hotspot(h, "mem")
    assert "alloc=5.0MB" in line
    assert "n=12" in line


def test_format_hotspot_cpu_mem_total_mb():
    h = _hotspot(raw={"mem_total_mb": 33.5})
    line = render._format_hotspot(h, "cpu")
    assert "mem=33.5MB" in line


def test_format_hotspot_source_location_and_source():
    h = _hotspot(raw={"source_location": "src/x.cpp:42", "source": "profiler_hotspot"})
    line = render._format_hotspot(h, "cpu")
    assert "src=src/x.cpp:42" in line
    assert "source=profiler_hotspot" in line


def test_format_hotspot_override_summary_source_extras():
    h = _hotspot(raw={"source": "override_summary", "mean_s": 0.002, "n_workers": 4})
    line = render._format_hotspot(h, "cpu")
    assert "mean=2.000ms" in line
    assert "workers=4" in line


def test_format_hotspot_override_marker_source():
    h = _hotspot(raw={"source": "override_marker"})
    line = render._format_hotspot(h, "cpu")
    assert "timing=unavailable" in line


# ---------------------------------------------------------------------------
# actionable hotspots + raw hotspots blocks
# ---------------------------------------------------------------------------

def test_actionable_block_renders_with_raw_below():
    out = _capture({
        "backend": "cpu", "lang": "py", "tier": "tiny", "totals": {},
        "hotspots": [_hotspot(rank=1, label="raw_frame")],
        "actionable_hotspots": [_hotspot(rank=1, label="actionable_frame")],
    })
    assert "actionable targets" in out
    assert "actionable_frame" in out
    assert "Raw profiler hotspots" in out
    assert "raw_frame" in out


# ---------------------------------------------------------------------------
# layer breakdown / per-layer top / python-native split blocks
# ---------------------------------------------------------------------------

def test_layer_breakdown_rendered_first():
    out = _capture({
        "backend": "cpu", "lang": "py", "tier": "tiny", "totals": {},
        "hotspots": [_hotspot()],
        "layer_breakdown": [
            {"layer_group": "task", "self_time_s": 0.6, "pct": 60.0},
            {"layer_group": "library", "self_time_s": 0.4, "pct": 40.0},
        ],
    })
    assert "Layer breakdown" in out
    assert "task" in out
    # breakdown line appears before the hotspots header.
    assert out.index("Layer breakdown") < out.index("Top 1 hotspots")


def test_per_layer_top_renders_in_priority_order():
    out = _capture({
        "backend": "cpu", "lang": "py", "tier": "tiny", "totals": {},
        "hotspots": [_hotspot()],
        "per_layer_top": {
            "library": [{"label": "lib_fn", "self_pct": 30.0}],
            "task": [{"label": "task_fn", "self_pct": 60.0, "n_calls": 5}],
        },
    })
    assert "Per-layer top" in out
    assert "task_fn" in out
    assert "lib_fn" in out
    # task is rendered before library (priority order), even though library
    # came first in the dict.
    assert out.index("task_fn") < out.index("lib_fn")
    assert "n_calls=5" in out


def test_python_native_split_block():
    out = _capture({
        "backend": "full", "lang": "py", "tier": "tiny", "totals": {},
        "hotspots": [_hotspot()],
        "python_native_split": {
            "python_s": 0.25, "python_pct": 25.0,
            "native_s": 0.70, "native_pct": 70.0,
            "system_s": 0.05, "system_pct": 5.0,
        },
    })
    assert "Python/native split" in out
    assert "python" in out and "native" in out
    assert "system" in out  # system_pct > 1 -> shown


def test_python_native_split_hides_tiny_system():
    out = _capture({
        "backend": "full", "lang": "py", "tier": "tiny", "totals": {},
        "hotspots": [_hotspot()],
        "python_native_split": {
            "python_s": 0.5, "python_pct": 50.0,
            "native_s": 0.5, "native_pct": 50.0,
            "system_s": 0.0, "system_pct": 0.0,
        },
    })
    assert "system" not in out.split("Python/native split")[1].split("Top")[0]


# ---------------------------------------------------------------------------
# override summary / markers / call chains / notes / artifacts / profile dir
# ---------------------------------------------------------------------------

def test_override_summary_block():
    out = _capture({
        "backend": "cpu", "lang": "py", "tier": "tiny", "totals": {},
        "hotspots": [_hotspot()],
        "override_summary": [
            {"name": "ov", "calls": 4, "total_s": 1.0, "mean_s": 0.25, "n_workers": 1},
            {"name": "ov2", "calls": 8, "total_s": 2.0, "mean_s": 0.25, "n_workers": 4},
        ],
    })
    assert "Override timing (2 tracked)" in out
    assert "aggregated across 4 workers" in out


def test_override_markers_block():
    out = _capture({
        "backend": "cpu", "lang": "py", "tier": "tiny", "totals": {},
        "hotspots": [_hotspot()],
        "override_markers": [{"name": "marker_a"}],
    })
    assert "Active override markers without timing (1)" in out
    assert "marker_a" in out


def test_call_chains_block():
    out = _capture({
        "backend": "cpu", "lang": "py", "tier": "tiny", "totals": {},
        "hotspots": [_hotspot()],
        "call_chains": [
            {"rank": 1, "chain": ["root", "mid", "leaf"], "samples": 12},
        ],
    })
    assert "Representative call chains" in out
    assert "root -> mid -> leaf" in out
    assert "samples=12" in out


def test_notes_and_artifacts_and_dir():
    out = _capture({
        "backend": "cpu", "lang": "py", "tier": "tiny", "totals": {},
        "hotspots": [_hotspot()],
        "notes": ["caveat one", "caveat two"],
        "artifacts": {"raw": "profile_history/run/profile.out"},
    }, profile_dir="profile_history/run")
    assert "Notes:" in out
    assert "- caveat one" in out
    assert "Artifacts: raw=profile_history/run/profile.out" in out
    assert "Profile dir: profile_history/run" in out


# ---------------------------------------------------------------------------
# _format_call_chain helper directly
# ---------------------------------------------------------------------------

def test_format_call_chain_truncates_long_chains():
    chain = {"rank": 2, "chain": [f"f{i}" for i in range(8)], "samples": 5}
    s = render._format_call_chain(chain)
    # >5 frames -> first + "..." + last 3.
    assert "f0" in s
    assert "..." in s
    assert "f7" in s
    assert "samples=5" in s


def test_format_call_chain_with_owner():
    chain = {"rank": 1, "chain": ["a", "b"],
             "nearest_actionable_frame": "owner_fn", "samples": 3}
    s = render._format_call_chain(chain)
    assert "owner=owner_fn" in s


def test_format_call_chain_empty_uses_leaf():
    chain = {"rank": 1, "chain": [], "leaf": "leaf_fn"}
    s = render._format_call_chain(chain)
    assert "leaf_fn" in s
