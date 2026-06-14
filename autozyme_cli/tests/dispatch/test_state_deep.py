"""Deep coverage tests for zyme.dispatch.state.

Targets gaps not covered by tests/dispatch/test_state.py:
  - tail_events OSError swallowing
  - follow_events generator (file-appears-late + multi-line drain + skip-corrupt)
  - pid_alive POSIX ProcessLookupError branch (dead pid)

The Windows branch of pid_alive (os.name=='nt') is uncoverable on macOS.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from zyme.dispatch.state import (
    append_event,
    follow_events,
    pid_alive,
    tail_events,
)


# --------------------------------------------------------------------------
# tail_events OSError path
# --------------------------------------------------------------------------

class TestTailEventsOsError:
    def test_swallows_read_error(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "events.ndjson"
        path.write_text('{"ts":"t","kind":"k"}\n')

        original_read_text = Path.read_text

        def _raising_read_text(self, *a, **kw):
            if self.name == "events.ndjson":
                raise OSError("simulated read error")
            return original_read_text(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", _raising_read_text)
        # read fails inside tail_events -> returns [] (not raise).
        assert tail_events(path) == []


# --------------------------------------------------------------------------
# follow_events generator
# --------------------------------------------------------------------------

class TestFollowEvents:
    def test_yields_appended_events_and_skips_corrupt(self, tmp_path: Path):
        path = tmp_path / "events.ndjson"
        # Pre-seed two good lines, one corrupt line between.
        path.write_text(
            '{"ts":"a","kind":"k1"}\n'
            'not-json\n'
            '{"ts":"b","kind":"k2"}\n'
        )
        gen = follow_events(path, poll_s=0.01)
        # The generator blocks at EOF; pull exactly the two valid events.
        first = next(gen)
        second = next(gen)
        gen.close()
        assert first["kind"] == "k1"
        assert second["kind"] == "k2"

    def test_waits_for_file_to_appear(self, tmp_path: Path):
        path = tmp_path / "later.ndjson"
        results = []

        def _consume():
            gen = follow_events(path, poll_s=0.01)
            results.append(next(gen))
            gen.close()

        t = threading.Thread(target=_consume, daemon=True)
        t.start()
        # File doesn't exist yet; the generator should be polling.
        time.sleep(0.05)
        append_event(path, {"ts": "z", "kind": "delayed"})
        t.join(timeout=3.0)
        assert not t.is_alive()
        assert results and results[0]["kind"] == "delayed"

    def test_handles_partial_line_then_completion(self, tmp_path: Path):
        path = tmp_path / "events.ndjson"
        path.write_text("")  # exists but empty
        results = []

        def _consume():
            gen = follow_events(path, poll_s=0.01)
            results.append(next(gen))
            gen.close()

        t = threading.Thread(target=_consume, daemon=True)
        t.start()
        time.sleep(0.03)
        # Write a line in two chunks to exercise the buffer-partition path.
        with open(path, "a") as f:
            f.write('{"ts":"p","ki')
            f.flush()
            time.sleep(0.03)
            f.write('nd":"split"}\n')
            f.flush()
        t.join(timeout=3.0)
        assert not t.is_alive()
        assert results and results[0]["kind"] == "split"


# --------------------------------------------------------------------------
# pid_alive POSIX branches
# --------------------------------------------------------------------------

class TestPidAlivePosix:
    def test_dead_pid_returns_false(self):
        # A very high PID is almost certainly not a live process -> the
        # ProcessLookupError branch fires and returns False.
        assert pid_alive(2_000_000_000) is False

    def test_negative_pid_false(self):
        assert pid_alive(-5) is False
