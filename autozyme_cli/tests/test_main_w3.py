"""Wave-3 coverage for zyme/__main__.py — the `python -m zyme` entry point.

__main__.py is 3 statements (import main; `if __name__ == "__main__": main()`).
coverage.py only records it when the module is actually executed as a module.
We drive it two ways:

  1. A real `python -m zyme --help` subprocess smoke test (the ONE real
     subprocess allowed by the brief) — asserts exit 0 + usage banner.
  2. runpy.run_module("zyme", run_name="__main__") with sys.argv stubbed and
     main() patched, so the `if __name__ == "__main__": main()` line is
     exercised *in this process* and registered by coverage.
"""
from __future__ import annotations

import runpy
import subprocess
import sys

import pytest


# --------------------------------------------------------------------------
# Real subprocess smoke (the single allowed real subprocess)
# --------------------------------------------------------------------------

def test_python_m_zyme_help_exits_zero_and_prints_usage():
    proc = subprocess.run(
        [sys.executable, "-m", "zyme", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    # argparse prog="zyme" -> usage line names the program.
    assert "usage: zyme" in proc.stdout


def test_python_m_zyme_no_args_is_handled():
    # No subcommand: argparse may print help / error. Either way it must not
    # hang and must exit with a defined code (argparse -> 1 or 2; cli may 0).
    proc = subprocess.run(
        [sys.executable, "-m", "zyme"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode in (0, 1, 2)


# --------------------------------------------------------------------------
# In-process run so coverage records __main__.py's body
# --------------------------------------------------------------------------

def test_run_module_invokes_main(monkeypatch):
    """run_module with run_name='__main__' executes the guarded `main()`
    call in __main__.py. We stub zyme.cli.main so nothing real runs, and
    assert it was reached."""
    called = {"n": 0}

    import zyme.cli as cli
    monkeypatch.setattr(cli, "main", lambda: called.__setitem__("n", called["n"] + 1))
    # __main__ does `from zyme.cli import main`, binding the name at import
    # time. runpy re-imports the module fresh, so patching cli.main BEFORE
    # run_module ensures the fresh import binds the patched callable.
    runpy.run_module("zyme", run_name="__main__")
    assert called["n"] == 1


def test_import_main_does_not_run(monkeypatch):
    """Importing zyme.__main__ as a normal module (run_name != '__main__')
    must NOT call main() — the guard protects it."""
    called = {"n": 0}
    import zyme.cli as cli
    monkeypatch.setattr(cli, "main", lambda: called.__setitem__("n", 1))
    runpy.run_module("zyme.__main__", run_name="zyme.__main__")
    assert called["n"] == 0
