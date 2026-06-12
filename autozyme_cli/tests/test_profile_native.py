"""Unit tests for native sample parsing and idle-frame filtering."""
from __future__ import annotations

from pathlib import Path

from zyme.commands.profile import native


class _FakeSampler:
    def __init__(self, files: list[Path]):
        self._files = files

    def collect(self) -> list[Path]:
        return self._files


def test_pthread_cond_wait_is_idle_frame():
    assert native._is_idle_frame("_pthread_cond_wait")
    assert native._is_idle_frame("pthread_cond_wait")
    assert native._is_idle_frame("sem_wait")
    assert native._is_idle_frame("__wait4")
    assert native._is_idle_frame("__wait4_nocancel")
    assert native._is_idle_frame("__read_nocancel")
    assert native._is_idle_frame("__write_nocancel")
    assert native._is_idle_frame("__fcntl")
    assert native._is_idle_frame("stat")
    assert native._is_idle_frame("madvise")
    assert native._is_idle_frame("DYLD-STUB$$__commpage_gettimeofday")


def test_native_normalizer_filters_pthread_wait_from_hotspots(tmp_path):
    sample = tmp_path / "native_sample_123.txt"
    sample.write_text(
        "Header\n"
        "Sort by top of stack, same collapsed (when >= 5):\n"
        "        _pthread_cond_wait  (in libsystem_pthread.dylib)        100\n"
        "        sem_wait  (in libsystem_kernel.dylib)        90\n"
        "        __read_nocancel  (in libsystem_kernel.dylib)        80\n"
        "        __wait4  (in libsystem_kernel.dylib)        70\n"
        "        stat  (in libsystem_kernel.dylib)        60\n"
        "        madvise  (in libsystem_kernel.dylib)        55\n"
        "        useful_compute  (in libtarget.dylib)        50\n"
        "Binary Images:\n"
    )

    data = native.parse_and_normalize(
        out_dir=tmp_path,
        sampler=_FakeSampler([sample]),
        lang="py",
        tier="tiny",
        hypothesis="",
        totals={"wall_s": 1.0},
    )

    labels = [h["label"] for h in data["hotspots"]]
    assert labels == ["libtarget.dylib:useful_compute"]
    assert any("_pthread_cond_wait" in n for n in data["notes"])
    assert data["override_markers"] == []
    assert data["call_chains"] == []


def test_native_sampler_discovers_process_group_descendants(monkeypatch, tmp_path):
    calls = []

    class Result:
        def __init__(self, rc: int, out: str):
            self.returncode = rc
            self.stdout = out

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[:2] == [native.PGREP_BIN, "-P"]:
            return Result(0, "101\n")
        if cmd[:2] == [native.PGREP_BIN, "-g"]:
            return Result(0, "100\n101\n202\n")
        return Result(1, "")

    monkeypatch.setattr(native.subprocess, "run", fake_run)
    sampler = native.NativeSampler(tmp_path)
    sampler.parent_pid = 100

    assert sampler._discover_candidate_pids() == [100, 101, 202]
    assert [native.PGREP_BIN, "-P", "100"] in calls
    assert [native.PGREP_BIN, "-g", "100"] in calls
