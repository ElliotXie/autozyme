"""Shared CLI helpers: errors, git wrapper, task-dir resolution, mode resolution,
baseline stash I/O, upstream-version drift checker.

Pure parsing of task.yaml and results.tsv lives in zyme.parsers.task_yaml
and zyme.parsers.results_tsv respectively — import from those directly.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# Exit code conventions. Agents branch on these; keeping the buckets distinct
# lets a caller decide "retry me" vs "stop and ask the user" vs "the host is
# unhappy, back off and retry later." Default `die()` stays exit=1 so existing
# callsites keep their meaning; opt into 2/3 for the new buckets.
EXIT_USER_ERROR = 1      # bad args, missing file, wrong state — user fixes and retries
EXIT_INTERNAL = 2        # zyme bug or corrupted on-disk state — repro / file an issue
EXIT_RESOURCE = 3        # RAM/disk/timeout — environment problem, retry later may work


def die(msg, code=EXIT_USER_ERROR):
    sys.stderr.write(f"zyme: {msg}\n")
    sys.exit(code)


def die_internal(msg):
    """Use when the failure indicates a CLI bug or corrupted state, not user error."""
    die(msg, code=EXIT_INTERNAL)


def die_resource(msg):
    """Use when the host is the problem (out of RAM/disk, watchdog kill, timeout)."""
    die(msg, code=EXIT_RESOURCE)


def info(msg):
    print(f"[zyme] {msg}")


# ----------------------------------------------------------------------------
# .zyme_meta.yaml + results.tsv schema variant for bench tasks
# ----------------------------------------------------------------------------
# Tasks scaffolded by `zyme bench init` carry a .zyme_meta.yaml at task root
# recording prompt_id / framework_sha / suite / replicate. When present, every
# results.tsv row written by `zyme run` (and baseline-promotion paths) gains
# a 12th `prompt_id` column. Tasks WITHOUT .zyme_meta.yaml keep the original
# 11-col schema untouched — old tasks aren't migrated.

ZYME_META_FILENAME = ".zyme_meta.yaml"

# Legacy mode label: synthesized when reading a task whose results.tsv /
# baselines_stash.tsv predates the thread_mode column and whose task.yaml
# has no `modes:` block. New tasks scaffolded by `zyme init` start in
# `default` mode; legacy is reserved for "we don't know the threading
# regime under which the row was recorded."
LEGACY_MODE = "legacy"
DEFAULT_MODE = "default"
# K1: thread is the parallel-resource axis (integer column on results.tsv /
# verify.tsv / baselines_history.tsv). `1` is the natural single-threaded
# baseline that pre-K2 rows are backfilled to during ensure_k2_schema.
LEGACY_THREAD = 1

# Schema evolution:
#   v0 (pre-bench)  : 11 cols (round..phase)
#   v1 (bench)      : 11 cols + prompt_id (last)
#   V1-mode (RIP)   : v0/v1 + thread_mode (briefly shipped; conflated mode + thread)
#   K2 (current)    : v0/v1 + thread (after phase, before prompt_id)
#
# K2 dropped the mode axis entirely (every "different upstream config" use
# case maps to "open a new task"). The only baseline-axis is `thread` —
# verify already sweeps it; baselines need to vary along it for fair speedup.
#
# Old files lacking the `thread` column read fine — `_row_thread` falls
# back to LEGACY_THREAD (=1). The first K2 op (record-baseline,
# baseline-rebench, or verify on append) triggers `ensure_k2_schema` to
# rewrite the header + backfill values.
_RESULTS_HEADER_CORE_COLS = (
    "round", "commit", "dataset", "speed_sec", "speedup_pct", "peak_mb",
    "status", "metrics_json", "hypothesis", "description", "phase",
)
_RESULTS_HEADER_THREAD_COL = ("thread",)
_RESULTS_HEADER_PROMPT_ID_COL = ("prompt_id",)

RESULTS_HEADER_BASE = "\t".join(
    _RESULTS_HEADER_CORE_COLS + _RESULTS_HEADER_THREAD_COL
) + "\n"
RESULTS_HEADER_WITH_PROMPT_ID = "\t".join(
    _RESULTS_HEADER_CORE_COLS + _RESULTS_HEADER_THREAD_COL + _RESULTS_HEADER_PROMPT_ID_COL
) + "\n"


def has_zyme_meta(task_dir: Path) -> bool:
    return (task_dir / ZYME_META_FILENAME).exists()


def parse_zyme_meta(task_dir: Path) -> dict:
    """Read .zyme_meta.yaml as a flat dict. Returns {} when absent.

    Reuses registry's tiny YAML parser (handles flat key:value scalars).
    """
    p = task_dir / ZYME_META_FILENAME
    if not p.exists():
        return {}
    from zyme import registry  # local import to avoid circularity
    return registry.read_card_yaml(p)


def get_prompt_id_for_task(task_dir: Path) -> str:
    """Return prompt_id from .zyme_meta.yaml, or empty string if absent."""
    meta = parse_zyme_meta(task_dir)
    return str(meta.get("prompt_id") or "")


def git(*args, cwd=None, check=True, capture=True):
    """Subprocess wrapper for git. Returns stdout (stripped) on success."""
    res = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=capture, text=True
    )
    if check and res.returncode != 0:
        sys.stderr.write(res.stderr or "")
        die(f"git {' '.join(args)} failed (exit {res.returncode})")
    return res.stdout.strip() if capture else ""


def task_dir_from_args(args):
    """Resolve task directory from --task-dir flag or cwd; require task.yaml present."""
    p = Path(getattr(args, "task_dir", None) or os.getcwd()).resolve()
    if not (p / "task.yaml").exists():
        die(f"not a zyme task directory (no task.yaml): {p}")
    return p


def zyme_state(task_dir: Path):
    """Ensure .zyme/ exists; return (state_dir, best_ref_path, round_counter_path)."""
    z = task_dir / ".zyme"
    z.mkdir(exist_ok=True)
    return z, z / "best.ref", z / "round.counter"


def detect_lang(task_dir: Path):
    if (task_dir / "pipeline" / "run.py").exists():
        return "py"
    if (task_dir / "pipeline" / "run.R").exists():
        return "R"
    die(f"no pipeline/run.{{py,R}} in {task_dir}")


def pipeline_paths(task_dir: Path):
    """Return absolute paths to pipeline/run.{py,R} that exist."""
    paths = []
    for ext in ("py", "R"):
        p = task_dir / "pipeline" / f"run.{ext}"
        if p.exists():
            paths.append(p)
    return paths


# ---- reference resolution (K2: thread-axis only) -------------------------


def synthesize_legacy_mode(task_dir: Path) -> dict:
    """Build a pre-K2 mode definition for old task.yaml files.

    K2 no longer uses mode definitions for execution, but old task fixtures
    and on-disk task directories may still need to resolve their reference
    script during migration.
    """
    if (task_dir / "reference.R").exists():
        script = "reference.R"
    elif (task_dir / "reference.py").exists():
        script = "reference.py"
    else:
        script = "reference.R"
    return {
        "description": "pre-migration legacy reference mode",
        "reference_script": script,
    }


def read_mode_definitions(task_dir: Path) -> dict:
    """Return old `modes:` definitions, or a synthesized legacy mode."""
    from zyme.parsers.task_yaml import parse_modes_block

    task_yaml = task_dir / "task.yaml"
    modes = parse_modes_block(task_yaml)
    if modes:
        return modes
    return {LEGACY_MODE: synthesize_legacy_mode(task_dir)}


def read_active_mode(task_dir: Path) -> str:
    """Return old `active_mode:` value, defaulting to LEGACY_MODE."""
    from zyme.parsers.task_yaml import parse_active_mode

    return parse_active_mode(task_dir / "task.yaml")


def resolve_reference_script(task_dir: Path, mode: str | None = None) -> Path:
    """Return the single `reference.{R,py}` for the task.

    The script reads ZYME_THREADS internally and branches into a parallel
    path when N>1. Per-thread or per-mode variants are not a thing in K2.
    """
    if mode is not None:
        modes = read_mode_definitions(task_dir)
        if mode not in modes:
            die(f"unknown mode {mode!r}; available: {', '.join(sorted(modes))}")
        script = modes[mode].get("reference_script") or "reference.R"
        return (task_dir / script).resolve()

    # Compatibility path for pre-K2 task.yaml files that still declare an
    # active mode and a modes block.
    modes = read_mode_definitions(task_dir)
    active = read_active_mode(task_dir)
    if active in modes and (task_dir / "task.yaml").exists():
        script = modes[active].get("reference_script")
        if script:
            return (task_dir / script).resolve()

    ref_R = task_dir / "reference.R"
    ref_py = task_dir / "reference.py"
    if ref_R.exists():
        return ref_R.resolve()
    if ref_py.exists():
        return ref_py.resolve()
    return ref_R.resolve()  # placeholder; caller will likely error on the run


def resolve_reference_output_dir(
    task_dir: Path,
    mode_or_tier: str = "",
    tier: str | None = None,
) -> Path:
    """Return per-tier reference-output dir.

    Default layout: `reference_outputs/<tier>/`. Backward-compat reads in
    `runner.py` layer their own fallback chain against the legacy
    `reference_output_<tier>/` flat layout for pre-K1 task directories.
    """
    if tier is not None:
        # Callers pass `tier=` positionally with `mode_or_tier` left at its
        # default (""), so an empty mode means "use the task's active mode"
        # (LEGACY_MODE for K2 tasks that dropped the modes block).
        mode = mode_or_tier or read_active_mode(task_dir)
        modes = read_mode_definitions(task_dir)
        if mode not in modes:
            die(f"unknown mode {mode!r}; available: {', '.join(sorted(modes))}")
        template = modes[mode].get("reference_output_dir")
        if template:
            rendered = template.replace("<mode>", mode).replace("<tier>", tier)
            return (task_dir / rendered).resolve()
        return (task_dir / f"reference_output_{tier}").resolve()

    tier = mode_or_tier
    if not tier:
        die("resolve_reference_output_dir: tier required")
    return (task_dir / "reference_outputs" / tier).resolve()


# ---- baseline stash (DEPRECATED in K1; kept for backward read) ------------
#
# Pre-K1 design: `zyme record-baseline` wrote to a stash and `zyme run` /
# `zyme verify` lazily promoted the entry to results.tsv on first touch.
# K1 drops lazy promotion: record-baseline writes directly to results.tsv.
# The stash file `.zyme/baselines_stash.tsv` is no longer written by new
# code. Old files remain readable via read_baseline_stash; `cmd_promote_baseline`
# can drain any leftover entries into results.tsv as a one-shot migration.
#
# Schema retained for read backward compat (V1 had `thread_mode`; K1 reads
# either column name; readers normalize to `mode`).
BASELINE_STASH_FIELDS = (
    "tier", "name", "speed_sec", "peak_mb", "metrics_json", "status",
    "mode", "thread",
)


def baseline_stash_path(task_dir: Path) -> Path:
    return task_dir / ".zyme" / "baselines_stash.tsv"


def read_baseline_stash(task_dir: Path):
    """Return list of dicts (one per (tier, mode, thread) entry in stash).

    DEPRECATED in K1: kept for backward read against pre-K1 stash files.
    New code writes directly to results.tsv via cmd_record_baseline.

    Empty list if no stash. Normalizes V1 `thread_mode` → `mode` and
    backfills missing `mode`/`thread` to (LEGACY_MODE, LEGACY_THREAD).
    """
    p = baseline_stash_path(task_dir)
    if not p.exists():
        return []
    out = []
    lines = p.read_text().splitlines()
    if not lines:
        return out
    header = lines[0].split("\t")
    for raw in lines[1:]:
        if not raw.strip():
            continue
        parts = raw.split("\t")
        entry = {h: (parts[i] if i < len(parts) else "") for i, h in enumerate(header)}
        # V1 thread_mode → K1 mode normalization on read.
        if "mode" not in entry and "thread_mode" in entry:
            entry["mode"] = entry["thread_mode"]
        if not entry.get("mode"):
            entry["mode"] = LEGACY_MODE
        entry.setdefault("thread_mode", entry["mode"])
        # K1 thread column: backfill default for pre-K1 stashes.
        if not entry.get("thread"):
            entry["thread"] = str(LEGACY_THREAD)
        out.append(entry)
    return out


def write_baseline_stash(task_dir: Path, entries):
    """Overwrite the stash file with `entries` (list of dicts). Deletes file if empty.

    DEPRECATED in K2: kept so `cmd_promote_baseline` can drain a pre-K2 stash.
    New record-baseline calls write directly to results.tsv.
    """
    p = baseline_stash_path(task_dir)
    if not entries:
        if p.exists():
            p.unlink()
        return
    p.parent.mkdir(exist_ok=True)
    out_lines = ["\t".join(BASELINE_STASH_FIELDS)]
    for e in entries:
        row = dict(e)
        if not row.get("mode") and row.get("thread_mode"):
            row["mode"] = row["thread_mode"]
        if not row.get("thread"):
            row["thread"] = str(LEGACY_THREAD)
        out_lines.append("\t".join(str(row.get(f, "")) for f in BASELINE_STASH_FIELDS))
    p.write_text("\n".join(out_lines) + "\n")


def upsert_baseline_stash(task_dir: Path, entry: dict):
    """Insert/replace a deprecated baseline stash entry.

    Kept for old task migration and test coverage. New baseline commands write
    directly to results.tsv.
    """
    zyme_state(task_dir)
    row = dict(entry)
    if not row.get("mode") and row.get("thread_mode"):
        row["mode"] = row["thread_mode"]
    if not row.get("mode"):
        row["mode"] = read_active_mode(task_dir)
    row.setdefault("thread_mode", row["mode"])
    if not row.get("thread"):
        row["thread"] = str(LEGACY_THREAD)

    entries = read_baseline_stash(task_dir)
    replaced = False
    out = []
    for old in entries:
        same_key = (
            old.get("tier") == row.get("tier")
            and (old.get("mode") or old.get("thread_mode")) == row.get("mode")
            and str(old.get("thread") or LEGACY_THREAD) == str(row.get("thread") or LEGACY_THREAD)
        )
        if same_key:
            if not replaced:
                out.append(row)
                replaced = True
            continue
        out.append(old)
    if not replaced:
        out.append(row)
    write_baseline_stash(task_dir, out)


def pop_baseline_stash(
    task_dir: Path,
    tier: str,
    *,
    mode: str | None = None,
    thread: int | None = None,
):
    """Pop one deprecated baseline stash entry matching tier/mode/thread."""
    target_mode = mode or read_active_mode(task_dir)
    target_thread = str(thread or LEGACY_THREAD)
    entries = read_baseline_stash(task_dir)
    kept = []
    popped = None
    for entry in entries:
        entry_mode = entry.get("mode") or entry.get("thread_mode") or LEGACY_MODE
        entry_thread = str(entry.get("thread") or LEGACY_THREAD)
        if (
            popped is None
            and entry.get("tier") == tier
            and entry_mode == target_mode
            and entry_thread == target_thread
        ):
            popped = entry
            continue
        kept.append(entry)
    if popped is not None:
        write_baseline_stash(task_dir, kept)
    return popped


# ---- upstream_repo / installed-version drift detection -------------------
#
# Why: agents read source code at task_dir/upstream_repo/ to design overrides,
# but the override actually wraps the *installed* version that pipeline/run
# resolves at runtime. If the two diverge (e.g. upstream_repo is at v4.19.8
# but the conda env has v4.16.0), overrides target functions/paths that don't
# exist or behave differently at runtime — agent burns rounds discovering this.
# The check fires on `zyme run` start and warns when versions disagree.
#
# Cached in .zyme/version_check.json keyed on upstream_repo HEAD SHA: if the
# clone hasn't moved, we skip the interpreter spawn (~1-2s) on every run.


def _grep_dcf_field(text: str, field: str) -> str:
    """Extract a `Field: value` from R DESCRIPTION (Debian Control File) format."""
    for line in text.splitlines():
        m = re.match(rf"^{re.escape(field)}\s*:\s*(.+)\s*$", line)
        if m:
            return m.group(1).strip()
    return ""


def read_upstream_repo_metadata(task_dir: Path):
    """Best-effort: package name + version from `task_dir/upstream_repo/`.

    Returns dict {package, version, source} or None. Tries DESCRIPTION (R),
    pyproject.toml ([project] block), setup.cfg ([metadata] block), and
    setup.py (regex) in order. Failures return None — the check is non-fatal.
    """
    upstream = task_dir / "upstream_repo"
    if not upstream.exists() or not upstream.is_dir():
        return None

    desc = upstream / "DESCRIPTION"
    if desc.exists():
        text = desc.read_text(encoding="utf-8", errors="replace")
        pkg = _grep_dcf_field(text, "Package")
        ver = _grep_dcf_field(text, "Version")
        if pkg and ver:
            return {"package": pkg, "version": ver, "source": "DESCRIPTION"}

    pyproj = upstream / "pyproject.toml"
    if pyproj.exists():
        text = pyproj.read_text(encoding="utf-8", errors="replace")
        in_project = False
        pkg = ver = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_project = (stripped == "[project]")
                continue
            if not in_project:
                continue
            m = re.match(r'^\s*name\s*=\s*["\']([^"\']+)["\']', line)
            if m:
                pkg = m.group(1)
            m = re.match(r'^\s*version\s*=\s*["\']([^"\']+)["\']', line)
            if m:
                ver = m.group(1)
        if pkg and ver:
            return {"package": pkg, "version": ver, "source": "pyproject.toml"}

    setup_cfg = upstream / "setup.cfg"
    if setup_cfg.exists():
        text = setup_cfg.read_text(encoding="utf-8", errors="replace")
        in_metadata = False
        pkg = ver = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_metadata = (stripped == "[metadata]")
                continue
            if not in_metadata:
                continue
            m = re.match(r"^\s*name\s*=\s*(\S.*?)\s*$", line)
            if m:
                pkg = m.group(1).strip(' "\'')
            m = re.match(r"^\s*version\s*=\s*(\S.*?)\s*$", line)
            if m:
                ver = m.group(1).strip(' "\'')
        if pkg and ver and not ver.startswith("attr:") and not ver.startswith("file:"):
            return {"package": pkg, "version": ver, "source": "setup.cfg"}

    setup_py = upstream / "setup.py"
    if setup_py.exists():
        text = setup_py.read_text(encoding="utf-8", errors="replace")
        m_name = re.search(r"""name\s*=\s*["']([^"']+)["']""", text)
        m_ver = re.search(r"""version\s*=\s*["']([^"']+)["']""", text)
        if m_name and m_ver:
            return {"package": m_name.group(1), "version": m_ver.group(1), "source": "setup.py"}

    return None


def read_upstream_repo_sha(task_dir: Path):
    """HEAD SHA of upstream_repo/ if it's a git clone. Used as cache key."""
    upstream = task_dir / "upstream_repo"
    if not upstream.exists() or not (upstream / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=upstream, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def read_installed_version(task_dir: Path, package_name: str):
    """Spawn the task's interpreter and ask for the installed package version.

    R: `cat(as.character(packageVersion("PKG")))` via Rscript.
    Python: try `pkg.__version__`, fall back to `importlib.metadata.version`.
    Returns the version string, or None on any failure.
    """
    from zyme.parsers.task_yaml import parse_executor  # lazy: avoid cycle
    lang = detect_lang(task_dir)
    executor = parse_executor(task_dir / "task.yaml")

    try:
        if lang == "R":
            rscript = executor.get("rscript", "Rscript")
            code = (
                f'tryCatch('
                f'cat(as.character(packageVersion("{package_name}"))), '
                f'error = function(e) invisible(NULL))'
            )
            result = subprocess.run(
                [rscript, "-e", code],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                v = result.stdout.strip()
                return v or None
        elif lang == "py":
            from zyme.runner import _resolve_python  # lazy: avoid circular import
            try:
                py_bin = (
                    _resolve_python(executor["python"])
                    if executor.get("python") else sys.executable
                )
            except Exception:
                py_bin = sys.executable
            code = (
                f"import importlib\n"
                f"v = None\n"
                f"try:\n"
                f"    m = importlib.import_module('{package_name}')\n"
                f"    v = getattr(m, '__version__', None)\n"
                f"except Exception:\n"
                f"    pass\n"
                f"if v is None:\n"
                f"    try:\n"
                f"        from importlib.metadata import version\n"
                f"        v = version('{package_name}')\n"
                f"    except Exception:\n"
                f"        pass\n"
                f"print(v if v else '')\n"
            )
            result = subprocess.run(
                [py_bin, "-c", code],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                v = result.stdout.strip()
                return v or None
    except Exception:
        pass
    return None


def _versions_loose_equal(a: str, b: str) -> bool:
    """Loose version equality: tolerates R's `1.4.0-3` ↔ Python's `1.4.0.3`.

    Considered equal if (a) the strings match, (b) they match after replacing
    dashes with dots, or (c) the digit-run sequences (re.findall(r"\\d+", ...))
    are identical.

    Note (c) extracts EVERY digit run, including the trailing digit inside a
    pre-release suffix: `"1.0.0a1"` → [1,0,0,1] vs `"1.0.0"` → [1,0,0]. So
    pre-release tags do NOT collapse to the base version under this check —
    `_versions_loose_equal("1.0.0a1", "1.0.0")` returns False. The function
    targets the R-vs-Python release-format mismatch, not pre-release equating.
    """
    if a == b:
        return True
    if a.replace("-", ".") == b.replace("-", "."):
        return True
    seg_a = re.findall(r"\d+", a)
    seg_b = re.findall(r"\d+", b)
    return bool(seg_a) and seg_a == seg_b


def check_upstream_version_drift(task_dir: Path):
    """Compare upstream_repo's declared version vs installed version.

    Returns dict {package, upstream_version, installed_version, agrees, source},
    or None when the check can't run (no upstream_repo, no parseable
    metadata, interpreter spawn failed). Cached by upstream_repo HEAD SHA.
    """
    cache_file = task_dir / ".zyme" / "version_check.json"
    upstream_sha = read_upstream_repo_sha(task_dir)

    if cache_file.exists() and upstream_sha:
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("upstream_sha") == upstream_sha:
                return {k: cached[k] for k in
                        ("package", "upstream_version", "installed_version", "agrees", "source")
                        if k in cached}
        except Exception:
            pass

    meta = read_upstream_repo_metadata(task_dir)
    if not meta:
        return None
    installed_version = read_installed_version(task_dir, meta["package"])
    if installed_version is None:
        return None

    result = {
        "package": meta["package"],
        "upstream_version": meta["version"],
        "installed_version": installed_version,
        "agrees": _versions_loose_equal(meta["version"], installed_version),
        "source": meta["source"],
    }
    if upstream_sha:
        try:
            cache_file.parent.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps(dict(result, upstream_sha=upstream_sha), indent=2))
        except Exception:
            pass
    return result
