"""Smoke test for the May 2026 Windows ``UnicodeEncodeError`` bug.

When Python's default stdout encoding is cp1252 (English Windows default),
the CLI used to crash whenever it printed a non-ASCII glyph such as the
right-arrow ``→`` that appeared in three baseline.py info() messages.
The reference output saved correctly, but the user saw a stack trace and
believed the run had failed.

The fix lives in ``zyme.cli.main()`` (utf-8 reconfigure at entry) plus
ASCII replacement of the three arrow glyphs as belt-and-braces.

These tests are self-contained: they spawn subprocesses with hostile
encoding rather than depending on CI host config. Adding Windows to the
CI matrix (Cat 1) doubles the catch rate by also stressing OS-specific
codecs, but these tests will reproduce the bug on any host.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


def _run_under_hostile_encoding(code: str):
    """Run ``code`` in a fresh subprocess with PYTHONIOENCODING=cp1252.

    Captures stdout/stderr as raw bytes so the test never imposes its own
    decoding on the subprocess output (which is the whole point).
    """
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
    )


def test_negative_control_cp1252_arrow_crashes_without_reconfigure():
    """Sanity: confirm the test env actually reproduces the bug without fix.

    If THIS test stops failing, the test environment changed (e.g. newer
    Python silently force-utf-8) and the positive tests below stop being
    meaningful. Treat this as a tripwire.
    """
    result = _run_under_hostile_encoding(
        "print('→')"
    )
    # Either the print raises UnicodeEncodeError -> nonzero exit, or some
    # Python builds map cp1252 onto a permissive codec. Accept either,
    # but flag clearly if the bug doesn't reproduce.
    if result.returncode == 0:
        pytest.skip(
            "this Python/host does not reproduce the cp1252 arrow crash; "
            "positive tests below are still meaningful but the negative "
            "control no longer verifies the test environment"
        )
    assert b"UnicodeEncodeError" in result.stderr


def test_cli_main_reconfigures_stdio_at_entry():
    """zyme.cli.main() must make subsequent arrow prints safe under cp1252.

    Strategy: invoke main() with ``--help`` (always SystemExit(0) after
    reconfigure has fired), then in the same process print an arrow and
    confirm it lands in stdout intact.
    """
    code = textwrap.dedent('''
        import sys
        from zyme.cli import main
        try:
            sys.argv = ["zyme", "--help"]
            main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
        # main() has run its stdio reconfigure by now. The arrow that
        # used to crash baseline.py:796 must now print without raising.
        print("post-main arrow: →", flush=True)
    ''')
    result = _run_under_hostile_encoding(code)
    assert result.returncode == 0, (
        f"main() failed under cp1252: stderr={result.stderr!r}"
    )
    # UTF-8 bytes for U+2192 are 0xE2 0x86 0x92.
    assert b"\xe2\x86\x92" in result.stdout, (
        "arrow did not survive cp1252 -> utf-8 reconfigure: "
        f"stdout={result.stdout!r}"
    )


def test_cli_info_helper_does_not_crash_after_main_entry():
    """zyme.utils.info() must be safe to call with non-ASCII after main()."""
    code = textwrap.dedent('''
        import sys
        from zyme.cli import main
        try:
            sys.argv = ["zyme", "--help"]
            main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
        from zyme.utils import info
        info("calibration -> baseline (→ arrow probe)")
    ''')
    result = _run_under_hostile_encoding(code)
    assert result.returncode == 0, (
        f"info() crashed after main() under cp1252: stderr={result.stderr!r}"
    )
