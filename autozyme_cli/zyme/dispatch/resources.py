"""Resource probes for `zyme dispatch` — RAM, disk, dataset-size estimation.

All probes are pure-Python with no third-party deps. macOS uses vm_stat;
Linux falls back to /proc/meminfo (so `zyme dispatch` works on the CI/server
side too, not just local Mac development).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def free_ram_gb() -> float:
    """Available RAM, in GB.

    macOS: free + inactive + speculative + purgeable pages from vm_stat.
    Inactive pages count because they're reclaimable on demand — matching
    what Activity Monitor calls "App Memory available". Returns 0.0 on
    probe failure (callers gate on the value, so a conservative zero
    blocks rather than mis-launches).
    """
    if sys.platform == "darwin":
        return _free_ram_gb_macos()
    if sys.platform.startswith("linux"):
        return _free_ram_gb_linux()
    return 0.0


def _free_ram_gb_macos() -> float:
    try:
        ps_out = subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True)
        pagesize = int(ps_out.strip())
        vm_out = subprocess.check_output(["vm_stat"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError):
        return 0.0

    wanted = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    total_pages = 0
    for line in vm_out.splitlines():
        for prefix in wanted:
            if line.startswith(prefix):
                m = re.search(r"(\d+)", line)
                if m:
                    total_pages += int(m.group(1))
                break
    return total_pages * pagesize / (1024 ** 3)


def _free_ram_gb_linux() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) / (1024 ** 2)  # kB → GB
    except (OSError, ValueError):
        pass
    return 0.0


def total_ram_gb() -> float:
    """Total physical RAM, in GB. 0.0 on probe failure.

    Used by `zyme verify` to derive a default --mem-cap-gb (auto = total × 0.7)
    so the watchdog has a sensible cap on hosts the user hasn't manually
    sized for.
    """
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / (1024 ** 3)
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) / (1024 ** 2)  # kB → GB
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError):
        pass
    return 0.0


def process_group_rss_mb(pgid: int) -> float:
    """Sum RSS (in MB) of every process in process group `pgid`. Thin
    wrapper around process_group_stats kept for backward compat with
    callers that only need RSS."""
    rss_mb, _, _ = process_group_stats(pgid)
    return rss_mb


def process_group_stats(pgid: int) -> tuple[float, int, float]:
    """Snapshot (rss_mb, n_procs, cpu_sec_total) for every process in `pgid`.

    Single `ps` call collects: RSS (KB on macOS+Linux), CPU time (column
    `time` formatted as [DD-]HH:MM:SS or MM:SS.frac), and process count.
    Returns (0.0, 0, 0.0) when the group is gone or `ps` fails.

    Used by the runner watchdog for two duties:
      - mem cap enforcement (existing `--mem-cap-gb`)
      - thread budget audit (catch unfair-baseline runs where pipeline
        forks more workers than ZYME_THREADS declares)

    Summing across the process group catches forked workers (joblib,
    numba, mclapply) that wouldn't show up if we only watched the parent.
    """
    try:
        out = subprocess.check_output(
            ["ps", "-A", "-o", "rss=,pgid=,time="], text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return (0.0, 0, 0.0)
    total_kb = 0
    n_procs = 0
    cpu_sec_total = 0.0
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rss_kb = int(parts[0])
            row_pgid = int(parts[1])
        except ValueError:
            continue
        if row_pgid != pgid:
            continue
        n_procs += 1
        total_kb += rss_kb
        cpu_sec_total += _parse_ps_time(parts[2].strip())
    return (total_kb / 1024.0, n_procs, cpu_sec_total)


def _parse_ps_time(s: str) -> float:
    """Parse ps -o time= output: [[DD-]HH:]MM:SS[.frac] → seconds.

    macOS BSD ps typically emits `MM:SS.cc` for short runs and `HH:MM:SS`
    for longer ones; Linux ps adds `DD-HH:MM:SS` past 24h. Anything we
    can't parse becomes 0.0 (better to under-report than crash the
    watchdog poll loop)."""
    if not s:
        return 0.0
    try:
        days = 0
        if "-" in s:
            day_part, s = s.split("-", 1)
            days = int(day_part)
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return days * 86400 + int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return days * 86400 + int(m) * 60 + float(sec)
        return float(s)
    except (ValueError, AttributeError):
        return 0.0


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def free_disk_gb(path: str | Path) -> float:
    """Free disk space at the filesystem hosting `path`, in GB."""
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# Size string parsing
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([kmgt]?)b?\s*$", re.IGNORECASE)
_UNIT_FACTORS = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}


def parse_size_bytes(s: str | None) -> int | None:
    """Parse '10g', '500m', '1.5T', '1024' → bytes (int). None / empty → None.

    Trailing 'b'/'B' is tolerated ('10gb' == '10g'). Raises ValueError on
    non-empty unparseable input — callers should validate user input early.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    m = _SIZE_RE.match(s)
    if not m:
        raise ValueError(f"unparseable size: {s!r}")
    return int(float(m.group(1)) * _UNIT_FACTORS[m.group(2).lower()])


def parse_size_gb(s: str | None) -> float | None:
    """Parse a size string to GB (float). None passthrough."""
    n = parse_size_bytes(s)
    return None if n is None else n / (1024 ** 3)


# ---------------------------------------------------------------------------
# Per-task disk-need estimate (for default --disk-floor)
# ---------------------------------------------------------------------------

def estimate_disk_need_gb(task_dir: Path) -> float:
    """Sum dataset sizes under task_dir × 2 (input + scratch buffer), in GB.

    Reads `datasets:` from task.yaml, walks each `path:`, sums file
    sizes. The ×2 multiplier is a rough guard for intermediate artifacts
    (cached transforms, reference outputs, etc.) — it's a floor, not a
    forecast.

    Returns 0.0 if task.yaml is missing, parse fails, or no dataset
    paths resolve. Callers should treat 0.0 as "unknown" and either
    skip the disk gate or apply a conservative default.
    """
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.is_file():
        return 0.0
    try:
        from zyme.parsers.task_yaml import parse_datasets
        datasets = parse_datasets(yaml_path)
    except Exception:
        return 0.0

    total = 0
    for entry in datasets:
        path_str = entry.get("path")
        if not path_str:
            continue
        # parse_datasets may already resolve relative paths; try both.
        candidates = [Path(path_str)]
        if not Path(path_str).is_absolute():
            candidates.append(task_dir / path_str)
        for cand in candidates:
            if cand.is_file():
                try:
                    total += cand.stat().st_size
                except OSError:
                    pass
                break
            if cand.is_dir():
                for f in cand.rglob("*"):
                    if f.is_file():
                        try:
                            total += f.stat().st_size
                        except OSError:
                            pass
                break

    return (total * 2) / (1024 ** 3)


# ---------------------------------------------------------------------------
# Combined gate (used by master loop)
# ---------------------------------------------------------------------------

def gate_check(
    *,
    ram_floor_gb: float,
    disk_path: Path,
    disk_floor_gb: float,
) -> tuple[bool, dict]:
    """Return (pass, snapshot). `snapshot` always populated for logging.

    Pass = ram >= ram_floor AND disk >= disk_floor. The snapshot includes
    the actual measurements so the master can log "RAM 8.2GB < 10GB,
    waiting" without re-probing.
    """
    ram = free_ram_gb()
    disk = free_disk_gb(disk_path)
    snapshot = {
        "ram_gb": round(ram, 2),
        "disk_gb": round(disk, 2),
        "ram_floor_gb": ram_floor_gb,
        "disk_floor_gb": disk_floor_gb,
    }
    ram_ok = ram >= ram_floor_gb
    disk_ok = disk >= disk_floor_gb
    return (ram_ok and disk_ok), snapshot
