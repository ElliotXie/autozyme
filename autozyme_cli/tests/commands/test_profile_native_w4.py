"""Wave-4 mop-up for zyme.commands.profile.native.

Wave-2/3 (test_profile_native.py, test_profile_native_cmd.py,
test_profile_native_w3.py) covered is_supported, _is_idle_frame, _short_label,
parse_sample_file, parse_and_normalize percentage/idle math + source_location,
_pgrep_pids, _discover_candidate_pids, and the full _locate_func_in_upstream
branch matrix.

This file squeezes the few REACHABLE lines those left:

  - NativeSampler.__init__ stale-output wipe: the `except OSError: pass`
    swallow when unlink() of a stale native_sample_* file fails (line 159-160).
  - _watch_loop early-exit when _stop is already set before the settle delay
    (line 245-246) — driven with the watchdog thread but a pre-set _stop so it
    never spawns a real `sample` subprocess.
  - _watch_loop single discovery pass: _discover_candidate_pids returns a pid,
    _spawn_sample is monkeypatched (no real subprocess), _stop flips after one
    iteration so the loop terminates deterministically. Covers the seen-set
    dedup branch (line 251-254) without /usr/bin/sample.
  - parse_and_normalize when EVERY sampled frame is idle: active set empty,
    hotspots empty, idle note still emitted, active_total==0 guard.

NOT exercised (subprocess/native boundary, per the brief): attach()'s real
thread start against a live PID, detach()'s SIGINT/SIGKILL flush of a live
`sample` proc, and _spawn_sample's real subprocess.Popen of /usr/bin/sample.
"""
from __future__ import annotations

import threading
from pathlib import Path

from zyme.commands.profile import native


class _FakeSampler:
    def __init__(self, files):
        self._files = files

    def collect(self):
        return self._files


# --------------------------------------------------------------------------
# __init__ stale wipe — OSError on unlink is swallowed
# --------------------------------------------------------------------------

def test_init_swallows_unlink_oserror(tmp_path, monkeypatch):
    # Seed a stale file so the wipe loop has something to unlink.
    (tmp_path / "native_sample_7.txt").write_text("stale")

    real_unlink = Path.unlink

    def flaky_unlink(self, *a, **k):
        if self.name == "native_sample_7.txt":
            raise OSError("permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    # Must not raise even though unlink failed.
    s = native.NativeSampler(tmp_path)
    assert s.parent_pid is None
    # The file remains because unlink failed — confirms we hit the swallow.
    assert (tmp_path / "native_sample_7.txt").exists()


# --------------------------------------------------------------------------
# _watch_loop — early return when _stop is set before the settle delay
# --------------------------------------------------------------------------

def test_watch_loop_returns_immediately_when_stopped(tmp_path, monkeypatch):
    s = native.NativeSampler(tmp_path)
    s.parent_pid = 1234
    s._stop.set()  # already stopped

    spawned = []
    monkeypatch.setattr(s, "_spawn_sample", lambda pid: spawned.append(pid))
    monkeypatch.setattr(s, "_discover_candidate_pids", lambda: [1234])

    # _stop.wait(delay) returns True immediately -> early return, no spawn.
    s._watch_loop()
    assert spawned == []


# --------------------------------------------------------------------------
# _watch_loop — one discovery pass then stop; _spawn_sample stubbed
# --------------------------------------------------------------------------

def test_watch_loop_single_pass_spawns_then_stops(tmp_path, monkeypatch):
    s = native.NativeSampler(tmp_path)
    s.parent_pid = 1000

    spawned: list[int] = []
    monkeypatch.setattr(s, "_spawn_sample", lambda pid: spawned.append(pid))
    monkeypatch.setattr(s, "_discover_candidate_pids", lambda: [1000, 1001])

    # Make the settle wait return False (proceed), then after the first
    # poll wait flip _stop so the while-loop exits.
    calls = {"n": 0}

    def fake_wait(timeout):
        calls["n"] += 1
        # First call is the INITIAL_ATTACH_DELAY settle -> proceed (False).
        if calls["n"] == 1:
            return False
        # After the first discovery pass, stop the loop.
        s._stop.set()
        return True

    monkeypatch.setattr(s._stop, "wait", fake_wait)
    s._watch_loop()
    # Both discovered pids spawned exactly once (seen-set dedup means no repeats).
    assert spawned == [1000, 1001]


def test_watch_loop_dedups_already_seen_pids(tmp_path, monkeypatch):
    s = native.NativeSampler(tmp_path)
    s.parent_pid = 1000

    spawned: list[int] = []
    monkeypatch.setattr(s, "_spawn_sample", lambda pid: spawned.append(pid))
    # Same pid returned on every poll; should only spawn once.
    monkeypatch.setattr(s, "_discover_candidate_pids", lambda: [2000])

    calls = {"n": 0}

    def fake_wait(timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return False  # settle -> proceed
        if calls["n"] >= 3:
            s._stop.set()
            return True
        return False  # let the loop run another iteration

    monkeypatch.setattr(s._stop, "wait", fake_wait)
    s._watch_loop()
    assert spawned == [2000]  # deduped across iterations


# --------------------------------------------------------------------------
# parse_and_normalize — all frames idle -> empty hotspots, active_total==0
# --------------------------------------------------------------------------

def test_parse_and_normalize_all_idle_no_hotspots(tmp_path):
    sample = tmp_path / "native_sample_1.txt"
    sample.write_text(
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        __psynch_cvwait  (in libsystem_pthread.dylib)        500\n"
        "        __workq_kernreturn  (in libsystem_kernel.dylib)        300\n"
        "Binary Images:\n"
    )
    data = native.parse_and_normalize(
        out_dir=tmp_path, sampler=_FakeSampler([sample]),
        lang="py", tier="tiny", hypothesis="",
    )
    # Everything is idle -> no actionable hotspots.
    assert data["hotspots"] == []
    # Idle frames still summarized in notes.
    assert any("idle/wait frames filtered out" in n for n in data["notes"])
    assert any("top idle frames" in n for n in data["notes"])


def test_parse_and_normalize_attach_when_active_total_only(tmp_path):
    # No idle frames at all -> idle note absent, but hotspot still ranked.
    sample = tmp_path / "native_sample_1.txt"
    sample.write_text(
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        only_work  (in libt.dylib)        42\n"
        "Binary Images:\n"
    )
    data = native.parse_and_normalize(
        out_dir=tmp_path, sampler=_FakeSampler([sample]),
        lang="py", tier="tiny", hypothesis="",
    )
    assert data["hotspots"][0]["self_pct"] == 100.0
    # No idle frames -> no "idle/wait frames filtered out" with a nonzero count?
    # idle_total==0 still emits the note (count 0), but no "top idle frames".
    assert not any("top idle frames" in n for n in data["notes"])
