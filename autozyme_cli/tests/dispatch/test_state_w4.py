"""Wave-4 coverage for zyme.dispatch.state — the reachable gaps left by
test_state.py + test_state_deep.py.

Targets:
  - follow_events OSError-on-open branch (lines 265-267)
  - follow_events skip-blank-line-in-buffer branch (line 274)
  - pid_alive PermissionError branch (lines 335-337)

Documented uncoverable: the Windows pid_alive ctypes branch (lines 306-330,
`os.name == 'nt'`) cannot run on macOS.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from zyme.dispatch.state import follow_events, pid_alive


# --------------------------------------------------------------------------
# follow_events — OSError on open is swallowed (keeps polling)
# --------------------------------------------------------------------------

class TestFollowEventsOsError:
    def test_open_oserror_keeps_polling_then_recovers(self, tmp_path: Path, monkeypatch):
        """First open() raises OSError (the `except OSError: sleep; continue`
        branch), then a later open succeeds and the appended event is yielded.
        """
        path = tmp_path / "events.ndjson"
        path.write_text('{"ts":"t","kind":"recovered"}\n')

        original_open = open
        state = {"calls": 0}

        def _flaky_open(*a, **kw):
            # Only intercept the follow_events read open (the events file).
            if a and str(a[0]).endswith("events.ndjson"):
                state["calls"] += 1
                if state["calls"] == 1:
                    raise OSError("transient open failure")
            return original_open(*a, **kw)

        monkeypatch.setattr("builtins.open", _flaky_open)

        results = []

        def _consume():
            gen = follow_events(path, poll_s=0.01)
            results.append(next(gen))
            gen.close()

        t = threading.Thread(target=_consume, daemon=True)
        t.start()
        t.join(timeout=3.0)
        assert not t.is_alive()
        assert results and results[0]["kind"] == "recovered"
        assert state["calls"] >= 2  # first failed, second succeeded


# --------------------------------------------------------------------------
# follow_events — blank lines inside the streamed chunk are skipped
# --------------------------------------------------------------------------

class TestFollowEventsBlankLines:
    def test_skips_blank_lines_between_events(self, tmp_path: Path):
        path = tmp_path / "events.ndjson"
        # Blank line + whitespace-only line interleaved with two valid events.
        path.write_text(
            '{"ts":"a","kind":"k1"}\n'
            '\n'
            '   \n'
            '{"ts":"b","kind":"k2"}\n'
        )
        gen = follow_events(path, poll_s=0.01)
        first = next(gen)
        second = next(gen)
        gen.close()
        assert first["kind"] == "k1"
        assert second["kind"] == "k2"


# --------------------------------------------------------------------------
# pid_alive — PermissionError POSIX branch (process exists, not ours)
# --------------------------------------------------------------------------

class TestPidAlivePermissionError:
    def test_permission_error_treated_as_alive(self, monkeypatch):
        """os.kill raising PermissionError means the process exists but we
        don't own it; the impl treats that as alive (True)."""
        import zyme.dispatch.state as state_mod

        def _raise_perm(pid, sig):
            raise PermissionError("not owner")

        # Force the POSIX path (os.name != 'nt') and a PermissionError.
        monkeypatch.setattr(state_mod.os, "name", "posix")
        monkeypatch.setattr(state_mod.os, "kill", _raise_perm)
        assert pid_alive(4242) is True

    def test_process_lookup_error_is_dead(self, monkeypatch):
        import zyme.dispatch.state as state_mod

        def _raise_lookup(pid, sig):
            raise ProcessLookupError("gone")

        monkeypatch.setattr(state_mod.os, "name", "posix")
        monkeypatch.setattr(state_mod.os, "kill", _raise_lookup)
        assert pid_alive(4242) is False

    def test_kill_succeeds_is_alive(self, monkeypatch):
        import zyme.dispatch.state as state_mod

        monkeypatch.setattr(state_mod.os, "name", "posix")
        monkeypatch.setattr(state_mod.os, "kill", lambda pid, sig: None)
        assert pid_alive(4242) is True
