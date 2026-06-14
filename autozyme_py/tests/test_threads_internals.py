"""Unit tests for autozyme._threads + autozyme._thread_env internals.

These cover the thread-count math (set_threads validation, safe_set_num_threads,
auto_threads precedence not already covered in test_auto_threads.py) and the
thread-env resolution helpers (yaml baseline_threads parse, env precedence,
apply_thread_env / apply_task_thread_env / ensure_process_thread_env).

Thread-count assertions that depend on the host's CPU count are made
RELATIONALLY (<=, >=) per the briefing — never on absolute numbers.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from autozyme import _threads as T
from autozyme import _thread_env as TE


@pytest.fixture(autouse=True)
def _clean_thread_env(monkeypatch):
    """Each test starts from a known env + reset module option."""
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
              "NUMBA_NUM_THREADS", "SCANPY_TURBO_THREADS"):
        monkeypatch.delenv(v, raising=False)
    saved = T._AUTOZYME_THREADS_OPTION
    T._AUTOZYME_THREADS_OPTION = None
    yield
    T._AUTOZYME_THREADS_OPTION = saved


# --------------------------------------------------------------------------
# set_threads
# --------------------------------------------------------------------------
def test_set_threads_writes_all_vars_and_option():
    assert T.set_threads(3) == 3
    for var in T._THREAD_VARS:
        assert os.environ[var] == "3"
    assert T._AUTOZYME_THREADS_OPTION == 3


def test_set_threads_rejects_below_one():
    with pytest.raises(ValueError, match="n must be >= 1"):
        T.set_threads(0)
    with pytest.raises(ValueError, match="n must be >= 1"):
        T.set_threads(-2)


def test_set_threads_coerces_float():
    assert T.set_threads(4.0) == 4
    assert os.environ["OMP_NUM_THREADS"] == "4"


# --------------------------------------------------------------------------
# auto_threads — branches not in test_auto_threads.py
# --------------------------------------------------------------------------
def test_auto_threads_zyme_threads_env_wins():
    os.environ["ZYME_THREADS"] = "9"
    assert T.auto_threads(cap=2) == 9


def test_auto_threads_default_relational_to_cpu():
    n = T.auto_threads()
    cores = os.cpu_count() or 1
    expected_default = min(max(1, cores - 1), 16)
    assert n == expected_default
    assert n >= 1


def test_auto_threads_cap_floats_accepted():
    # cap is int-coerced; a float-valued cap works
    n = T.auto_threads(cap=2.0)
    assert 1 <= n <= 2


def test_auto_threads_option_invalid_falls_to_hardware(monkeypatch):
    # corrupt the module option to a non-int -> falls through to hw default
    monkeypatch.setattr(T, "_AUTOZYME_THREADS_OPTION", "garbage")
    n = T.auto_threads()
    assert isinstance(n, int) and n >= 1


# --------------------------------------------------------------------------
# safe_set_num_threads
# --------------------------------------------------------------------------
def test_safe_set_num_threads_no_numba(monkeypatch):
    # Simulate numba not installed -> returns requested int unchanged.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "numba":
            raise ImportError("no numba")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert T.safe_set_num_threads(5) == 5


def test_safe_set_num_threads_noop_when_equal(monkeypatch):
    class _FakeNB:
        @staticmethod
        def get_num_threads():
            return 4

        @staticmethod
        def set_num_threads(x):
            raise AssertionError("should not be called when current == target")

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "numba":
            return _FakeNB
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert T.safe_set_num_threads(4) == 4


def test_safe_set_num_threads_sets_when_different(monkeypatch):
    state = {"current": 2, "set_to": None}

    class _FakeNB:
        @staticmethod
        def get_num_threads():
            return state["current"]

        @staticmethod
        def set_num_threads(x):
            state["set_to"] = x

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "numba":
            return _FakeNB
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert T.safe_set_num_threads(8) == 8
    assert state["set_to"] == 8


def test_safe_set_num_threads_locked_pool_falls_back(monkeypatch):
    class _FakeNB:
        @staticmethod
        def get_num_threads():
            return 3

        @staticmethod
        def set_num_threads(x):
            raise RuntimeError("pool locked")

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "numba":
            return _FakeNB
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # locked -> returns current count, not the requested one
    assert T.safe_set_num_threads(8) == 3


# ==========================================================================
# _thread_env
# ==========================================================================
def test_apply_thread_env_standard():
    env: dict[str, str] = {}
    out = TE.apply_thread_env(env, 4)
    assert out is env  # mutates + returns same dict
    assert env["ZYME_THREADS"] == "4"
    assert env["AUTOZYME_THREADS"] == "4"
    for var in TE.STANDARD_THREAD_VARS:
        assert env[var] == "4"
    # turbo vars NOT set without scanpy_turbo
    for var in TE.SCANPY_TURBO_THREAD_VARS:
        assert var not in env


def test_apply_thread_env_clamps_and_turbo():
    env = TE.apply_thread_env({}, 0, scanpy_turbo=True)
    assert env["ZYME_THREADS"] == "1"  # max(1, 0)
    for var in TE.SCANPY_TURBO_THREAD_VARS:
        assert env[var] == "1"


def test_parse_baseline_threads_from_yaml(tmp_path):
    y = tmp_path / "task.yaml"
    y.write_text("baseline_threads: [4, 8]\n")
    assert TE._parse_baseline_threads_from_yaml(y) == 4


def test_parse_baseline_threads_missing_file(tmp_path):
    assert TE._parse_baseline_threads_from_yaml(tmp_path / "nope.yaml") is None


def test_parse_baseline_threads_no_key(tmp_path):
    y = tmp_path / "task.yaml"
    y.write_text("task: x\nmetrics: []\n")
    assert TE._parse_baseline_threads_from_yaml(y) is None


def test_parse_baseline_threads_empty_list(tmp_path):
    y = tmp_path / "task.yaml"
    y.write_text("baseline_threads: []\n")
    assert TE._parse_baseline_threads_from_yaml(y) is None


def test_parse_baseline_threads_clamps_and_bad_value(tmp_path):
    y = tmp_path / "task.yaml"
    y.write_text("baseline_threads: [0]\n")
    assert TE._parse_baseline_threads_from_yaml(y) == 1  # max(1, 0)
    y.write_text("baseline_threads: [notint]\n")
    assert TE._parse_baseline_threads_from_yaml(y) is None


def test_resolve_baseline_threads_env_wins(monkeypatch, tmp_path):
    y = tmp_path / "task.yaml"
    y.write_text("baseline_threads: [8]\n")
    monkeypatch.setenv("ZYME_THREADS", "2")
    assert TE.resolve_baseline_threads(tmp_path) == 2  # env over yaml


def test_resolve_baseline_threads_autozyme_threads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOZYME_THREADS", "5")
    assert TE.resolve_baseline_threads(tmp_path) == 5


def test_resolve_baseline_threads_bad_env_uses_yaml(monkeypatch, tmp_path):
    y = tmp_path / "task.yaml"
    y.write_text("baseline_threads: [6]\n")
    monkeypatch.setenv("ZYME_THREADS", "bad")
    assert TE.resolve_baseline_threads(tmp_path) == 6


def test_resolve_baseline_threads_default_when_no_yaml(monkeypatch, tmp_path):
    assert TE.resolve_baseline_threads(tmp_path, default=7) == 7


def test_apply_task_thread_env_uses_yaml(monkeypatch, tmp_path):
    y = tmp_path / "task.yaml"
    y.write_text("baseline_threads: [3]\n")
    env: dict[str, str] = {}
    n = TE.apply_task_thread_env(env, tmp_path)
    assert n == 3
    assert env["ZYME_THREADS"] == "3"
    # not scanpy -> turbo vars absent
    assert "NUMBA_NUM_THREADS" not in env


def test_apply_task_thread_env_scanpy_turbo(monkeypatch, tmp_path):
    y = tmp_path / "task.yaml"
    y.write_text("baseline_threads: [2]\n")
    env: dict[str, str] = {}
    TE.apply_task_thread_env(env, tmp_path, patch_name="scanpy")
    assert env["NUMBA_NUM_THREADS"] == "2"
    assert env["SCANPY_TURBO_THREADS"] == "2"


def test_ensure_process_thread_env_mutates_os_environ(monkeypatch, tmp_path):
    y = tmp_path / "task.yaml"
    y.write_text("baseline_threads: [2]\n")
    n = TE.ensure_process_thread_env(tmp_path)
    assert n == 2
    assert os.environ["ZYME_THREADS"] == "2"
    assert os.environ["OMP_NUM_THREADS"] == "2"
