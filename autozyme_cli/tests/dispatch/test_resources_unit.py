"""Unit tests for zyme.dispatch.resources — RAM/disk probes, size parsing,
disk-need estimation, and the combined resource gate.

The subprocess-backed probes (free_ram_gb, total_ram_gb, process_group_stats)
are exercised by monkeypatching subprocess.check_output / sys.platform so we
test the parsing+accounting logic, not the host's actual vm_stat/ps output.
"""
from __future__ import annotations

import subprocess

import pytest

from zyme.dispatch import resources as R


# ---------------------------------------------------------------------------
# parse_size_bytes / parse_size_gb
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s, expected", [
    ("1024", 1024),
    ("1k", 1024),
    ("1K", 1024),
    ("1kb", 1024),
    ("1KB", 1024),
    ("10g", 10 * 1024 ** 3),
    ("500m", 500 * 1024 ** 2),
    ("1.5t", int(1.5 * 1024 ** 4)),
    ("1.5T", int(1.5 * 1024 ** 4)),
    ("10gb", 10 * 1024 ** 3),
    ("0", 0),
    ("  2g  ", 2 * 1024 ** 3),
])
def test_parse_size_bytes_valid(s, expected):
    assert R.parse_size_bytes(s) == expected


def test_parse_size_bytes_none_and_empty():
    assert R.parse_size_bytes(None) is None
    assert R.parse_size_bytes("") is None
    assert R.parse_size_bytes("   ") is None


@pytest.mark.parametrize("bad", ["abc", "10x", "g", "1.2.3g", "ten"])
def test_parse_size_bytes_unparseable_raises(bad):
    with pytest.raises(ValueError):
        R.parse_size_bytes(bad)


def test_parse_size_bytes_accepts_non_string_coercible():
    # int gets str()'d internally
    assert R.parse_size_bytes(1024) == 1024


def test_parse_size_gb():
    assert R.parse_size_gb("1g") == pytest.approx(1.0)
    assert R.parse_size_gb("2048m") == pytest.approx(2.0)
    assert R.parse_size_gb(None) is None
    assert R.parse_size_gb("") is None


# ---------------------------------------------------------------------------
# _parse_ps_time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s, expected", [
    ("", 0.0),
    ("0:00.00", 0.0),
    ("01:30.50", 90.5),          # MM:SS.frac
    ("00:01:05", 65.0),          # HH:MM:SS
    ("02:00:00", 7200.0),        # HH:MM:SS
    ("1-00:00:00", 86400.0),     # DD-HH:MM:SS
    ("2-01:00:30", 2 * 86400 + 3600 + 30.0),
    ("45.5", 45.5),              # bare seconds
])
def test_parse_ps_time(s, expected):
    assert R._parse_ps_time(s) == pytest.approx(expected)


def test_parse_ps_time_garbage_returns_zero():
    assert R._parse_ps_time("not:a:time:value") == 0.0
    assert R._parse_ps_time("a:b") == 0.0


# ---------------------------------------------------------------------------
# free_ram_gb / total_ram_gb — platform dispatch + parsing
# ---------------------------------------------------------------------------

VM_STAT_SAMPLE = """\
Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                              1000.
Pages active:                            5000.
Pages inactive:                          2000.
Pages speculative:                        500.
Pages throttled:                            0.
Pages wired down:                        3000.
Pages purgeable:                          300.
"""


def test_free_ram_gb_macos(monkeypatch):
    def fake_check_output(cmd, text=True):
        if cmd[0] == "sysctl":
            return "4096\n"
        if cmd[0] == "vm_stat":
            return VM_STAT_SAMPLE
        raise AssertionError(cmd)

    monkeypatch.setattr(R.sys, "platform", "darwin")
    monkeypatch.setattr(R.subprocess, "check_output", fake_check_output)
    # free+inactive+speculative+purgeable = 1000+2000+500+300 = 3800 pages
    expected = 3800 * 4096 / (1024 ** 3)
    assert R.free_ram_gb() == pytest.approx(expected)


def test_free_ram_gb_macos_probe_failure(monkeypatch):
    def boom(cmd, text=True):
        raise FileNotFoundError("vm_stat missing")
    monkeypatch.setattr(R.sys, "platform", "darwin")
    monkeypatch.setattr(R.subprocess, "check_output", boom)
    assert R.free_ram_gb() == 0.0


def test_free_ram_gb_linux(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       16384000 kB\n"
        "MemFree:         1000000 kB\n"
        "MemAvailable:    8388608 kB\n"
    )
    monkeypatch.setattr(R.sys, "platform", "linux")
    import builtins
    real_open = builtins.open

    def fake_open(path, *a, **k):
        if path == "/proc/meminfo":
            return real_open(meminfo, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    # 8388608 kB / 1024^2 = 8.0 GB
    assert R.free_ram_gb() == pytest.approx(8.0)


def test_free_ram_gb_linux_no_memavailable(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 100 kB\nMemFree: 50 kB\n")
    monkeypatch.setattr(R.sys, "platform", "linux")
    import builtins
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open",
                        lambda p, *a, **k: real_open(meminfo if p == "/proc/meminfo" else p, *a, **k))
    assert R.free_ram_gb() == 0.0


def test_free_ram_gb_linux_non_numeric_value(monkeypatch, tmp_path):
    # MemAvailable present but non-integer -> ValueError -> 0.0 (lines 62-63).
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:    notanumber kB\n")
    monkeypatch.setattr(R.sys, "platform", "linux")
    import builtins
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open",
                        lambda p, *a, **k: real_open(meminfo if p == "/proc/meminfo" else p, *a, **k))
    assert R.free_ram_gb() == 0.0


def test_free_ram_gb_unknown_platform(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "sunos5")
    assert R.free_ram_gb() == 0.0


def test_total_ram_gb_macos(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "darwin")
    monkeypatch.setattr(R.subprocess, "check_output",
                        lambda cmd, text=True: str(8 * 1024 ** 3) + "\n")
    assert R.total_ram_gb() == pytest.approx(8.0)


def test_total_ram_gb_linux(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       8388608 kB\nMemFree: 100 kB\n")
    monkeypatch.setattr(R.sys, "platform", "linux")
    import builtins
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open",
                        lambda p, *a, **k: real_open(meminfo if p == "/proc/meminfo" else p, *a, **k))
    assert R.total_ram_gb() == pytest.approx(8.0)


def test_total_ram_gb_failure(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "darwin")
    monkeypatch.setattr(R.subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "x")))
    assert R.total_ram_gb() == 0.0


def test_total_ram_gb_unknown_platform(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "freebsd")
    assert R.total_ram_gb() == 0.0


# ---------------------------------------------------------------------------
# process_group_stats / process_group_rss_mb
# ---------------------------------------------------------------------------

PS_SAMPLE = """\
  1024  500 00:01:00
  2048  500 00:00:30.50
  4096  999 10:00:00
   512  500 garbage
"""


def test_process_group_stats_sums_matching_pgid(monkeypatch):
    monkeypatch.setattr(R.subprocess, "check_output", lambda *a, **k: PS_SAMPLE)
    rss_mb, n_procs, cpu = R.process_group_stats(500)
    # pgid 500 rows: rss 1024, 2048, 512 KB -> 3584 KB; the "garbage" time
    # parses to 0.0 but the row still counts (rss + n_procs).
    assert n_procs == 3
    assert rss_mb == pytest.approx(3584 / 1024.0)
    # 60s + 30.5s + 0 (garbage)
    assert cpu == pytest.approx(90.5)


def test_process_group_stats_no_match(monkeypatch):
    monkeypatch.setattr(R.subprocess, "check_output", lambda *a, **k: PS_SAMPLE)
    assert R.process_group_stats(7777) == (0.0, 0, 0.0)


def test_process_group_stats_ps_failure(monkeypatch):
    monkeypatch.setattr(R.subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no ps")))
    assert R.process_group_stats(1) == (0.0, 0, 0.0)


def test_process_group_stats_malformed_rows_skipped(monkeypatch):
    sample = "short\nnot_int pgid time\n  100  500 00:00:01\n"
    monkeypatch.setattr(R.subprocess, "check_output", lambda *a, **k: sample)
    rss_mb, n_procs, cpu = R.process_group_stats(500)
    assert n_procs == 1
    assert cpu == pytest.approx(1.0)


def test_process_group_rss_mb_wrapper(monkeypatch):
    monkeypatch.setattr(R.subprocess, "check_output", lambda *a, **k: PS_SAMPLE)
    assert R.process_group_rss_mb(500) == pytest.approx(3584 / 1024.0)


# ---------------------------------------------------------------------------
# free_disk_gb
# ---------------------------------------------------------------------------

def test_free_disk_gb_real_path(tmp_path):
    # Real disk_usage call on tmp_path — value is host-dependent but must be > 0.
    assert R.free_disk_gb(tmp_path) > 0.0


def test_free_disk_gb_accepts_str(tmp_path):
    assert R.free_disk_gb(str(tmp_path)) > 0.0


def test_free_disk_gb_failure(monkeypatch):
    monkeypatch.setattr(R.shutil, "disk_usage",
                        lambda p: (_ for _ in ()).throw(OSError("gone")))
    assert R.free_disk_gb("/nonexistent") == 0.0


# ---------------------------------------------------------------------------
# estimate_disk_need_gb
# ---------------------------------------------------------------------------

def _write_task_yaml(task_dir, datasets_block):
    (task_dir / "task.yaml").write_text(
        "target_repo: x\ntarget_function: f\n\ndatasets:\n" + datasets_block + "\n"
        "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}\n"
    )


def test_estimate_disk_need_no_yaml(tmp_path):
    assert R.estimate_disk_need_gb(tmp_path) == 0.0


def test_estimate_disk_need_file(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    payload = data / "tiny.bin"
    payload.write_bytes(b"\0" * (1024 ** 3))  # 1 GB
    _write_task_yaml(tmp_path, "  - {tier: tiny, name: a, path: data/tiny.bin}")
    # 1 GB file * 2 = 2 GB
    assert R.estimate_disk_need_gb(tmp_path) == pytest.approx(2.0, rel=1e-6)


def test_estimate_disk_need_dir_recursive(tmp_path):
    d = tmp_path / "data" / "tree"
    d.mkdir(parents=True)
    (d / "a.bin").write_bytes(b"\0" * (512 * 1024 ** 2))
    (d / "sub").mkdir()
    (d / "sub" / "b.bin").write_bytes(b"\0" * (512 * 1024 ** 2))
    _write_task_yaml(tmp_path, "  - {tier: tiny, name: a, path: data/tree}")
    # 1 GB total * 2 = 2 GB
    assert R.estimate_disk_need_gb(tmp_path) == pytest.approx(2.0, rel=1e-6)


def test_estimate_disk_need_missing_path_is_zero(tmp_path):
    _write_task_yaml(tmp_path, "  - {tier: tiny, name: a, path: data/missing.h5ad}")
    assert R.estimate_disk_need_gb(tmp_path) == 0.0


def test_estimate_disk_need_skips_empty_path_and_relative(tmp_path, monkeypatch):
    # Hit the empty-path `continue` and the relative-path candidate branch by
    # feeding parse_datasets a raw (unresolved) relative path + a no-path entry.
    (tmp_path / "task.yaml").write_text("datasets:\n")
    data = tmp_path / "data"
    data.mkdir()
    (data / "rel.bin").write_bytes(b"\0" * (1024 ** 3))
    import zyme.parsers.task_yaml as ty
    monkeypatch.setattr(ty, "parse_datasets", lambda p: [
        {"path": ""},                       # empty -> continue (line 237)
        {"path": "data/rel.bin"},           # relative -> task_dir/ candidate (line 241)
    ])
    assert R.estimate_disk_need_gb(tmp_path) == pytest.approx(2.0, rel=1e-6)


def test_estimate_disk_need_parse_failure(tmp_path, monkeypatch):
    (tmp_path / "task.yaml").write_text("datasets:\n  - {name: a, path: data/x}\n")
    # Force parse_datasets import path to raise.
    import zyme.parsers.task_yaml as ty
    monkeypatch.setattr(ty, "parse_datasets",
                        lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert R.estimate_disk_need_gb(tmp_path) == 0.0


# ---------------------------------------------------------------------------
# gate_check
# ---------------------------------------------------------------------------

def test_gate_check_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "free_ram_gb", lambda: 20.0)
    monkeypatch.setattr(R, "free_disk_gb", lambda p: 100.0)
    ok, snap = R.gate_check(ram_floor_gb=10.0, disk_path=tmp_path, disk_floor_gb=50.0)
    assert ok is True
    assert snap == {"ram_gb": 20.0, "disk_gb": 100.0,
                    "ram_floor_gb": 10.0, "disk_floor_gb": 50.0}


def test_gate_check_ram_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "free_ram_gb", lambda: 5.0)
    monkeypatch.setattr(R, "free_disk_gb", lambda p: 100.0)
    ok, snap = R.gate_check(ram_floor_gb=10.0, disk_path=tmp_path, disk_floor_gb=50.0)
    assert ok is False
    assert snap["ram_gb"] == 5.0


def test_gate_check_disk_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "free_ram_gb", lambda: 20.0)
    monkeypatch.setattr(R, "free_disk_gb", lambda p: 10.0)
    ok, _ = R.gate_check(ram_floor_gb=10.0, disk_path=tmp_path, disk_floor_gb=50.0)
    assert ok is False


def test_gate_check_exact_boundary_passes(monkeypatch, tmp_path):
    # floor uses >= so an exact match passes.
    monkeypatch.setattr(R, "free_ram_gb", lambda: 10.0)
    monkeypatch.setattr(R, "free_disk_gb", lambda p: 50.0)
    ok, _ = R.gate_check(ram_floor_gb=10.0, disk_path=tmp_path, disk_floor_gb=50.0)
    assert ok is True
