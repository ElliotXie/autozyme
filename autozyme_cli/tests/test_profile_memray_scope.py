"""Tests for Python memray scoping through helpers.with_profile()."""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.commands.profile.command import _python_pipeline_uses_with_profile


def test_python_pipeline_uses_with_profile_detects_real_call(tmp_path):
    run_py = tmp_path / "run.py"
    run_py.write_text(
        "from helpers import with_profile\n"
        "with with_profile():\n"
        "    pass\n"
    )

    assert _python_pipeline_uses_with_profile(run_py)


def test_python_pipeline_uses_with_profile_ignores_imports_and_comments(tmp_path):
    run_py = tmp_path / "run.py"
    run_py.write_text(
        "from helpers import with_profile\n"
        "# with with_profile(): would be used here later\n"
        "print('no scoped call yet')\n"
    )

    assert not _python_pipeline_uses_with_profile(run_py)


def test_with_profile_mem_backend_starts_scoped_memray(
        monkeypatch, tmp_path, capsys):
    pytest.importorskip("memray")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZYME_PROFILE", "1")
    monkeypatch.setenv("ZYME_PROFILE_BACKEND", "mem")
    monkeypatch.delenv("ZYME_MEMRAY_ACTIVE", raising=False)

    from zyme.helpers import with_profile

    with with_profile():
        chunks = [bytearray(1024 * 1024) for _ in range(4)]
        assert len(chunks) == 4

    captured = capsys.readouterr()
    assert "scope=with_profile" in captured.err
    memray_bin = tmp_path / "memray.bin"
    assert memray_bin.exists()
    assert memray_bin.stat().st_size > 0
