"""Wave-4 mop-up for the profile sub-package remnants:
zyme.commands.profile.{parsers, enrich, render, diff}.

Wave-2 (tests/commands/test_profile_{parsers,enrich,render,diff}_cmd.py) plus
the wave-1 tests/test_profile_*.py drove these modules to 96-99%. This file
picks off the last cheap, REACHABLE branches without re-running the slow memray
subprocess parsers:

  - parsers.build_actionable_hotspots: the ">5 demoted noise frames" preview
    ellipsis (line 116, needs 6+ low-actionability hotspots).
  - parsers._short_label: the os.path.basename exception fallback (769-770).
  - enrich.count_r_calls: read_text raising -> {} (lines 102-103).
  - enrich.enrich (R): n_calls_est attached via the FULL label key when the
    quote-stripped `bare` key misses but the label key hits (lines 466-467).
  - render._humanize_bytes: the TB ceiling branch (line 272).
  - diff._latest_history_profile: profile_history dir exists but yields no
    candidate profile.json -> falls back to pipeline/profile.json (line 85).

No subprocesses, no network; everything runs in-process.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.commands.profile import parsers, enrich, render, diff


# ==========================================================================
# parsers.build_actionable_hotspots — >5 demoted-noise preview ellipsis
# ==========================================================================

def test_build_actionable_hotspots_many_noise_frames_ellipsis():
    # 6 low-actionability frames (fork/IPC/wait names) so the preview is
    # truncated to 5 with a trailing ", ...".
    noise_labels = [
        "R:unserialize", "R:readChild", "R:mcfork", "R:lazyLoadDBfetch",
        "R:selectChildren", "R:recvData",
    ]
    hotspots = [
        {"rank": i + 1, "label": lbl, "self_time_s": 1.0, "raw": {}}
        for i, lbl in enumerate(noise_labels)
    ]
    actionable, notes = parsers.build_actionable_hotspots(
        {"hotspots": hotspots})
    note = next(n for n in notes if "demoted" in n)
    assert "6 profiler" in note
    assert note.rstrip().endswith("...")


def test_build_actionable_hotspots_few_noise_no_ellipsis():
    hotspots = [
        {"rank": 1, "label": "R:unserialize", "self_time_s": 1.0, "raw": {}},
        {"rank": 2, "label": "lib:real_kernel", "self_time_s": 5.0, "raw": {}},
    ]
    actionable, notes = parsers.build_actionable_hotspots(
        {"hotspots": hotspots})
    note = next(n for n in notes if "demoted" in n)
    # only one demoted -> no ", ..." ellipsis.
    assert "1 profiler" in note
    assert not note.rstrip().endswith("...")
    # the real kernel survived as actionable.
    assert any(h["label"] == "lib:real_kernel" for h in actionable)


# ==========================================================================
# parsers._short_label — basename exception fallback (769-770)
# ==========================================================================

def test_short_label_basename_exception_fallback(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("basename exploded")

    monkeypatch.setattr(parsers.os.path, "basename", boom)
    out = parsers._short_label("/some/long/path/to/file.py", 42, "myfunc")
    # falls back to str(file)[:60]:line:name
    assert out.endswith(":42:myfunc")
    assert "/some/long/path" in out


def test_short_label_empty_file_question_mark():
    # file is falsy -> base="?" (the non-exception branch, for contrast).
    assert parsers._short_label("", 7, "fn") == "?:7:fn"


# ==========================================================================
# enrich.count_r_calls — read_text raising -> {} (102-103)
# ==========================================================================

def test_count_r_calls_read_exception_returns_empty(tmp_path, monkeypatch):
    p = tmp_path / "Rprof.out"
    p.write_text("sample.interval=20000\n", encoding="utf-8")

    def boom(self, *a, **k):
        raise OSError("read failed")

    monkeypatch.setattr(Path, "read_text", boom)
    assert enrich.count_r_calls(p) == {}


def test_count_r_calls_missing_file_returns_empty(tmp_path):
    assert enrich.count_r_calls(tmp_path / "absent.out") == {}


# ==========================================================================
# enrich.enrich (R) — n_calls_est via full-label key (466-467)
# ==========================================================================

def test_enrich_r_n_calls_est_via_label_key(tmp_path, monkeypatch):
    # r_call_counts keyed by the QUOTED label; the bare (unquoted) lookup
    # misses, so the elif `label in r_call_counts` branch fires.
    rprof = tmp_path / "Rprof.out"
    rprof.write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(enrich, "count_r_calls",
                        lambda path: {'"hot_fn"': 12})
    # avoid spawning R for the namespace index.
    monkeypatch.setattr(enrich, "build_r_namespace_index",
                        lambda executor, aux: {})
    profile_data = {
        "lang": "R",
        "hotspots": [
            {"label": '"hot_fn"', "self_time_s": 2.0, "raw": {}},
        ],
    }
    enrich.enrich(profile_data, target_pkg=None, rprof_path=rprof)
    h = profile_data["hotspots"][0]
    assert h.get("n_calls_est") == 12
    assert "layer" in h  # layer was assigned during enrichment
    assert profile_data["schema_version"] == "2"


# ==========================================================================
# render._humanize_bytes — TB ceiling (272)
# ==========================================================================

def test_humanize_bytes_tb():
    # 5 TB in bytes -> loop caps at the TB unit and returns "5.0TB".
    five_tb = 5 * 1024 ** 4
    out = render._humanize_bytes(five_tb)
    assert out.endswith("TB")
    assert out.startswith("5.0")


def test_humanize_bytes_small_units():
    assert render._humanize_bytes(512).endswith("B")
    assert render._humanize_bytes(2 * 1024).startswith("2.0")


# ==========================================================================
# diff._latest_history_profile — dir present, no candidates -> legacy fallback
# ==========================================================================

def test_latest_history_profile_empty_dir_falls_back(tmp_path, monkeypatch):
    # profile_history/ exists with entries that have NO profile.json inside ->
    # candidates empty -> returns pipeline/profile.json (line 84-85).
    hist = tmp_path / "profile_history"
    (hist / "run_empty").mkdir(parents=True)  # dir entry, no profile.json
    monkeypatch.chdir(tmp_path)
    result = diff._latest_history_profile()
    assert result == Path("pipeline") / "profile.json"


def test_latest_history_profile_no_dir_falls_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # no profile_history at all -> immediate legacy fallback (line 78).
    assert diff._latest_history_profile() == Path("pipeline") / "profile.json"
