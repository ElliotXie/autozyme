"""`zyme scan --coverage` — speedups_finalized.tsv coverage lint.

Pure inside-out check: reads each patch's bundled finalized speedup TSV and
reports gaps without consulting any external metadata (no task.yaml, no
manifest.yml, no expected-tier list). Each patch is its own source of truth
about which (tier, threads, platform, variant) cells should exist.

Rules
-----
1. **bad_status** — every row's `status` must be one of {ok, OOM, N/A, partial}.
   Unknown values → FAIL.

2. **low_reps** — `status=ok` rows must have `n_reps >= patched_min_reps`
   (default **2**) for patched variants and `n_reps >= baseline_min_reps`
   (default 2) for baseline variants. Less → WARN. Exception: OOD tiers
   (`ood_*`) report as **INFO**, not WARN — single OOD runs commonly take
   10–30 min, so n=1 there is usually intentional, not a missing rep.

   Why 2 (not 3) for patched: empirical audit of 365 shipped patched n=2
   cells found median 2-rep diff of 1.88% and 86% of cells under 10%
   diff. Patched kernels are typically deterministic; n=2 is enough to
   surface obvious instability. Cells that are truly unstable are
   caught by rule 2b.

2b. **high_rep_variance** — `status=ok` rows whose `sec_reps` disagree
    too much → WARN. n=2: percent diff > `rep_variance_pct` (default
    10%). n>=3: max/min spread > `spread_ratio` (default 2x) → catches
    bimodal / slow-outlier / fake-fast reps that drag the median into a
    misleading aggregate (e.g. a real speedup that medians out to a
    regression). Both gated by an absolute floor so tiny-value jitter is
    ignored.

2c. **high_mem_rep_variance** — same as 2b on `mem_reps` / peak MB
    (baseline and patched). Memory RSS is noisier; optional absolute
    floor via `mem_variance_abs_mb` (default 0 = percent-only, matching
    speed).

2d. **baseline_era_drift** — reads per-platform raw shards
    ``speedups.mac.tsv`` / ``speedups.win.tsv`` (not legacy ``speedups.tsv``;
    finalize ignores legacy when shards exist). When baseline rows for one
    cell span multiple measurement dates and per-date peak medians differ by
    more than `baseline_era_pct` (default 20%) AND `baseline_era_abs_mb`
    (default 1024 MB), the pooled finalized median mixes eras → INFO (baseline
    PEAK memory is a secondary metric; speed/speedup are unaffected, so this is
    informational noise, not a coverage gap).

2e. **stale_baseline_pairing** — same raw shards: when the latest patched
    row predates the latest baseline row for a cell, speedup/memory
    comparisons use a refreshed baseline against an old patched run →
    INFO (the ratio is still valid for a deterministic patched on one
    machine; only re-run patched if the machine era actually changed).

3. **missing_variant** — every (tier, threads, platform) that appears in the
   TSV needs BOTH `baseline` and `patched` rows. A patched row with no
   matching baseline (or vice versa) → WARN. Exception: if the only
   present variant has `status=OOM`, the cell is intentionally one-sided
   (the patched run would have OOMed too) — silently skipped.

4. **platform_asymmetry** — if a patch's TSV contains both Windows and macOS
   rows somewhere, every (tier, threads) cell present on one platform should
   also be present on the other. Asymmetry → INFO (lower severity than WARN
   because some gaps are legitimate, e.g. Mac large CCA OOM).

5. **thread_gap** — per-platform expected thread set, checked per (tier,
   platform). Two sources for the expected set:
     a. **Headline patches (scanpy_*/seurat_*)** ship a known per-platform
        schedule: `{1, 4, fullt}` where fullt=14 on mac and 32 on win.
        Anything outside the expected set on that platform is ignored;
        anything missing inside it → WARN. Exception: the fullt thread is
        skipped on OOD tiers — fullt OOD generally OOMs, so missing it is
        by design.
     b. **All other patches** use the set of threads actually observed on
        THAT platform (not the cross-platform union — that produced false
        positives when Win and Mac were intentionally swept at different
        thread counts). Each (tier, platform) should cover its platform's
        observed thread set; missing → WARN.
   A patch whose entire TSV only has threads=1 is single-thread by
   construction, and no thread_gap can fire. Rows with an empty/unknown
   `threads` field are ignored for this rule (TSV hygiene problem, not a
   coverage gap).

Cross-platform symmetry (rule 4) is INFO; everything else above n_reps is
WARN. Bad statuses (rule 1) are FAIL. By design the default exit code is 0
even when issues are reported — gating decisions belong to the caller
(`--strict` in `zyme scan`).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


VALID_STATUSES = {"ok", "OOM", "N/A", "partial"}

# Statuses worth considering as "the row exists" for symmetry / thread-gap
# joins. Bad statuses are reported separately (rule 1) and excluded from
# downstream rules to avoid noise about cells the user already knows are bad.
JOIN_STATUSES = {"ok", "OOM", "partial"}

# Thread-sweep rubric — single source of truth in scripts/thread_rubric.yaml.
# 1 and 4 are universal; the third is the machine's "full" core tier (mac 14
# headline / 8 default; win 32). Single-threaded patches expect only {1}. The
# lint scores against this rubric (not the cross-platform thread union) so that
# legitimate per-platform sweeps AND rubric gaps are reported correctly. The
# hardcoded values below are the fallback when the YAML is unreadable.
_HEADLINE_THREAD_SCHEDULE: dict[str, set[str]] = {"win": {"1", "4", "32"}, "mac": {"1", "4", "14"}}
_DEFAULT_THREAD_SCHEDULE: dict[str, set[str]] = {"win": {"1", "4", "32"}, "mac": {"1", "4", "8"}}
# Platform -> patches single-threaded ON THAT PLATFORM (scan expects only t1).
# both-platform singles appear under both keys; fork-based (mast,
# seurat_sctransform) appear under "win" only (they are parallel on mac).
_SINGLE_THREADED_PATCHES: dict[str, set[str]] = {}
# {task: {platform: set(variants)}} -- the (platform, variant) cells the rubric
# reduces to t1 (schedule==single -> both on both platforms; reduce_t1 -> the
# listed variants). Used by missing_variant to accept a missing REDUCED variant
# at thr>1; _SINGLE_THREADED_PATCHES (above) is the both-variants-reduced subset.
# See paper/rubric/task_threading.yaml.
_REDUCED: dict[str, dict[str, set[str]]] = {}


def _load_thread_rubric() -> None:
    """Populate the thread schedules + the reduction map from
    paper/rubric/task_threading.yaml (the single source of truth). Each `tasks:`
    entry {schedule, reduce_t1} yields, per (platform, variant), whether it is
    reduced to t1 (schedule==single reduces both on both; reduce_t1 lists
    per-platform variants). A patch expects only {1} on a platform when BOTH
    variants are reduced there.

    PyYAML is used when importable; otherwise a regex fallback parses the same
    fields (the CLI may run under a python without PyYAML, e.g. the system
    python3 that `bin/zyme` shebangs to)."""
    global _HEADLINE_THREAD_SCHEDULE, _DEFAULT_THREAD_SCHEDULE
    global _SINGLE_THREADED_PATCHES, _REDUCED
    root = Path(__file__).resolve().parents[2]
    path = root / "paper" / "rubric" / "task_threading.yaml"

    def _derive(tasks):
        reduced: dict = {}
        for name, e in tasks.items():
            e = e or {}
            m = {"mac": set(), "win": set()}
            if e.get("schedule") == "single":
                m = {"mac": {"baseline", "patched"}, "win": {"baseline", "patched"}}
            for plat, vs in (e.get("reduce_t1") or {}).items():
                m.setdefault(plat, set()).update(str(x) for x in (vs or []))
            reduced[name] = m
        single = {"mac": set(), "win": set()}  # both variants reduced -> expect {1}
        for name, m in reduced.items():
            for plat in ("mac", "win"):
                if {"baseline", "patched"} <= m.get(plat, set()):
                    single[plat].add(name)
        return single, reduced

    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            r = yaml.safe_load(f) or {}
        sch = r.get("schedules") or {}
        if sch.get("headline"):
            _HEADLINE_THREAD_SCHEDULE = {p: {str(x) for x in v} for p, v in sch["headline"].items()}
        if sch.get("default"):
            _DEFAULT_THREAD_SCHEDULE = {p: {str(x) for x in v} for p, v in sch["default"].items()}
        _SINGLE_THREADED_PATCHES, _REDUCED = _derive(r.get("tasks") or {})
        return
    except ImportError:
        pass  # PyYAML unavailable -> yaml-free text fallback below
    except Exception:  # noqa: BLE001 — malformed file: keep hardcoded defaults
        return
    # yaml-free fallback: regex-parse the `tasks:` entries for schedule /
    # single_on / inapplicable_on (the fields linting needs).
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    import re
    tasks: dict = {}
    in_tasks = False
    for ln in lines:
        if re.match(r"^tasks:\s*$", ln):
            in_tasks = True
            continue
        if in_tasks and ln[:1] not in (" ", "\t") and ln.strip() and not ln.lstrip().startswith("#"):
            in_tasks = False
        m = re.match(r"^\s+([A-Za-z0-9_]+):\s*\{(.*)\}", ln)
        if not (in_tasks and m):
            continue
        body, ent = m.group(2), {}
        s = re.search(r"schedule:\s*(\w+)", body)
        if s:
            ent["schedule"] = s.group(1)
        rt = re.search(r"reduce_t1:\s*\{([^}]*)\}", body)
        if rt:
            ent["reduce_t1"] = {
                pm.group(1): [x.strip() for x in pm.group(2).split(",") if x.strip()]
                for pm in re.finditer(r"(\w+):\s*\[([^\]]*)\]", rt.group(1))
            }
        tasks[m.group(1)] = ent
    if tasks:
        _SINGLE_THREADED_PATCHES, _REDUCED = _derive(tasks)


_load_thread_rubric()


def _has_headline_schedule(patch_name: str) -> bool:
    return patch_name.startswith("scanpy_") or patch_name.startswith("seurat_")


# Patches that are NOT measured via plain `zyme attest` and so should be
# skippable from the coverage lint with --skip-experimental:
#   - fusion-experiment patches (live under optimized_task/test_fusion/, run by
#     the multi-agent codex/claude/cursor experiment, not the standard sweep)
#   - orchestrate headline patches (scanpy_*/seurat_*, measured via the
#     orchestrate driver, not attest) — detected by _has_headline_schedule
FUSION_PATCHES = frozenset({"tradeseq", "mast", "infercnv", "find_all_markers"})


def _is_experimental(patch_name: str) -> bool:
    return patch_name in FUSION_PATCHES or _has_headline_schedule(patch_name)


_OOM_SKIP: set | None = None


def _load_oom_skip() -> set:
    """Cells that OOM on the 36GB Mac (scripts/mac_oom_skip.tsv). The sweep
    already skips these; the mac lint reads the same list so an intentionally
    absent mac baseline doesn't fire low_reps / missing_variant. Keyed
    (patch, tier, variant)."""
    global _OOM_SKIP
    if _OOM_SKIP is not None:
        return _OOM_SKIP
    _OOM_SKIP = set()
    f = Path(__file__).resolve().parents[2] / "scripts" / "mac_oom_skip.tsv"
    if f.is_file():
        for ln in f.read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.split("\t")
            if len(parts) >= 3:
                _OOM_SKIP.add((parts[0].strip(), parts[1].strip(),
                               parts[2].strip()))
    return _OOM_SKIP


def _oom_skipped(patch: str, tier: str, variant: str,
                 platform_filter: str | None) -> bool:
    """True when this mac cell is a known OOM (only applied under --mac-only)."""
    return (platform_filter == "mac"
            and (patch, tier, variant) in _load_oom_skip())


def _is_ood_tier(tier: str) -> bool:
    """OOD tiers (ood_large, ood_xlarge, ood_large1/2/3, etc.) are the
    largest datasets we sweep; a single run commonly takes 10–30 minutes,
    so n=1 there is usually intentional rather than a missing replicate."""
    return (tier or "").lower().startswith("ood")


def _is_blank_threads(threads: str) -> bool:
    """A few legacy TSVs have `threads=` empty or `threads=unknown`. Skip
    those in thread_gap so we don't invent ghost cells from bad input."""
    t = (threads or "").strip().lower()
    return not t or t == "unknown"


@dataclass
class Issue:
    """One coverage finding. `severity` is the rendering priority; `kind` is
    the rule it came from. Empty fields are dropped when rendered."""
    severity: str   # "FAIL" | "WARN" | "INFO"
    kind: str       # "bad_status" | "low_reps" | "high_rep_variance" |
                    # "high_mem_rep_variance" | "baseline_era_drift" |
                    # "stale_baseline_pairing" | "missing_variant" |
                    # "platform_asymmetry" | "thread_gap" | "missing_tsv" |
                    # "empty_tsv"
    patch: str
    tier: str = ""
    threads: str = ""
    platform: str = ""
    variant: str = ""
    msg: str = ""


@dataclass
class PatchCoverage:
    patch: str
    tsv_path: Path
    exists: bool
    n_rows: int = 0
    issues: list[Issue] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        c = {"FAIL": 0, "WARN": 0, "INFO": 0}
        for i in self.issues:
            c[i.severity] = c.get(i.severity, 0) + 1
        return c

    @property
    def is_clean(self) -> bool:
        return not self.issues


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _normalize_platform(value: str) -> str:
    s = (value or "").strip().lower()
    if not s:
        return "unknown"
    if s.startswith("windows") or s == "win":
        return "win"
    if s.startswith("macos") or s.startswith("darwin") or s == "mac":
        return "mac"
    return s


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    """Read a TSV with DictReader. Returns [] on read failure (caller checks)."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    except OSError:
        return []


def _parse_int(value: str) -> int | None:
    try:
        return int((value or "").strip())
    except (ValueError, TypeError):
        return None


def _parse_reps(value: str) -> list[float]:
    """Parse a `sec_reps` cell ('1.23, 1.45, 1.30') into floats. Returns []
    on any parse failure — caller treats that as 'can't check variance'."""
    if not value:
        return []
    out: list[float] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def _rep_diff_pct(vals: list[float]) -> float | None:
    """For n=2: percent diff between the two reps relative to the mean.
    Returns None if input isn't usable (not 2 vals, zero mean, etc.)."""
    if len(vals) != 2:
        return None
    mean = (vals[0] + vals[1]) / 2.0
    if mean <= 0:
        return None
    return abs(vals[0] - vals[1]) / mean * 100.0


def _append_rep_variance_issues(
    cov: PatchCoverage,
    patch_name: str,
    rows: list[dict[str, str]],
    *,
    rep_col: str,
    kind: str,
    rep_variance_pct: float,
    abs_floor: float = 0.0,
    spread_ratio: float = 2.0,
    value_fmt: str = ".3f",
) -> None:
    """Flag cells whose reps disagree too much. n=2: percent diff vs the mean
    (> rep_variance_pct). n>=3: max/min spread (> spread_ratio) -- catches
    bimodal / slow-outlier / fake-fast reps that drag the median into a
    misleading aggregate (e.g. a real speedup that medians out to a regression).
    Both gated by abs_floor so tiny-value jitter is ignored."""
    for r in rows:
        if (r.get("status") or "").strip() != "ok":
            continue
        n = _parse_int(r.get("n_reps", ""))
        if n is None or n < 2:
            continue
        vals = _parse_reps(r.get(rep_col, ""))
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        if abs_floor > 0 and (hi - lo) < abs_floor:
            continue
        if n == 2:
            diff_pct = _rep_diff_pct(vals)
            if diff_pct is None or diff_pct <= rep_variance_pct:
                continue
            detail = (f"n=2 {rep_col} differ by {diff_pct:.1f}% > "
                      f"{rep_variance_pct:.0f}% "
                      f"({vals[0]:{value_fmt}}, {vals[1]:{value_fmt}})")
        else:
            if not (lo > 0) or hi / lo <= spread_ratio:
                continue
            detail = (f"n={n} {rep_col} spread {hi / lo:.1f}x > "
                      f"{spread_ratio:.1f}x "
                      f"(min {lo:{value_fmt}}, max {hi:{value_fmt}})")
        tier = r.get("tier", "")
        severity = "INFO" if _is_ood_tier(tier) else "WARN"
        cov.issues.append(Issue(
            severity, kind, patch_name,
            tier=tier, threads=r.get("threads", ""),
            platform=_normalize_platform(r.get("platform", "")),
            variant=r.get("variant", ""),
            msg=detail,
        ))


def _raw_note_skip(note: str) -> bool:
    n = (note or "").lower()
    return "migrated:" in n or "absorbed:" in n


def _raw_date(ts: str) -> str:
    return (ts or "")[:10]


def _raw_threads(row: dict[str, str]) -> str:
    for key in ("system_threads", "threads"):
        val = (row.get(key) or "").strip()
        if val:
            return val
    return ""


def _raw_peak_mb(row: dict[str, str]) -> float | None:
    try:
        v = float((row.get("peak_mb") or "").strip())
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _row_platform(row: dict[str, str]) -> str:
    """Platform label for a finalized or raw speedups row."""
    return _normalize_platform(
        row.get("platform") or row.get("system_os") or ""
    )


def _matches_platform_filter(row: dict[str, str], platform_filter: str | None) -> bool:
    if not platform_filter:
        return True
    plat = _row_platform(row)
    if platform_filter == "mac":
        return plat == "mac"
    if platform_filter == "win":
        return plat == "win"
    return True


def _filter_rows_by_platform(
    rows: list[dict[str, str]], platform_filter: str | None,
) -> list[dict[str, str]]:
    if not platform_filter:
        return rows
    return [r for r in rows if _matches_platform_filter(r, platform_filter)]


def _discover_raw_speedup_paths(
    patch_dir: Path, platform_filter: str | None = None,
) -> list[Path]:
    """Platform shards only — mirrors ``finalize_speedups._read_raw_shards``.

    Legacy ``speedups.tsv`` is intentionally excluded: every shipped patch
    has ``speedups.mac.tsv`` + ``speedups.win.tsv``, and finalize already
    ignores the combined file when shards exist. Reading legacy here would
    risk double-counting migrated rows.
    """
    if platform_filter == "mac":
        plat_names = ("mac",)
    elif platform_filter == "win":
        plat_names = ("win",)
    else:
        plat_names = ("mac", "win", "other", "linux")
    out: list[Path] = []
    for plat in plat_names:
        p = patch_dir / f"speedups.{plat}.tsv"
        if p.is_file():
            out.append(p)
    return out


def _read_raw_speedup_rows(
    patch_dir: Path, platform_filter: str | None = None,
) -> list[dict[str, str]]:
    """Load per-rep rows from raw speedups shards (deduped)."""
    sig_cols = (
        "timestamp", "patch_name", "tier", "dataset", "rep_idx", "variant",
        "sec", "peak_mb", "system_os", "system_threads",
    )
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, str]] = []
    for path in _discover_raw_speedup_paths(patch_dir, platform_filter):
        for r in _read_tsv_rows(path):
            if not _matches_platform_filter(r, platform_filter):
                continue
            if _raw_note_skip(r.get("note") or ""):
                continue
            if _raw_peak_mb(r) is None:
                continue
            key = tuple((r.get(c) or "").strip() for c in sig_cols)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _audit_raw_era_issues(
    cov: PatchCoverage,
    patch_name: str,
    patch_dir: Path,
    *,
    baseline_era_pct: float,
    baseline_era_abs_mb: float,
    platform_filter: str | None = None,
) -> None:
    """Rules 2d/2e on raw per-rep TSV rows beside speedups_finalized.tsv."""
    raw_rows = _read_raw_speedup_rows(patch_dir, platform_filter)
    if not raw_rows:
        return

    # cell key: tier, threads, platform, dataset
    by_cell: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in raw_rows:
        plat = _row_platform(r)
        ds = (r.get("dataset") or "").strip()
        key = (r.get("tier") or "", _raw_threads(r), plat, ds)
        by_cell[key].append(r)

    for (tier, thr, plat, ds), cell_rows in sorted(by_cell.items()):
        baseline_rows = [r for r in cell_rows if (r.get("variant") or "") == "baseline"]
        patched_rows = [r for r in cell_rows if (r.get("variant") or "") == "patched"]
        if not baseline_rows:
            continue

        # Rule 2d: baseline peak medians drift across measurement dates.
        era_peaks: dict[str, list[float]] = defaultdict(list)
        for r in baseline_rows:
            peak = _raw_peak_mb(r)
            if peak is None:
                continue
            era_peaks[_raw_date(r.get("timestamp") or "")].append(peak)
        era_peaks = {d: v for d, v in era_peaks.items() if d and v}
        if len(era_peaks) >= 2:
            era_medians = {d: _median(v) for d, v in era_peaks.items()}
            era_medians = {d: m for d, m in era_medians.items() if m is not None}
            if len(era_medians) >= 2:
                meds = list(era_medians.values())
                lo, hi = min(meds), max(meds)
                spread_pct = (hi - lo) / lo * 100.0 if lo > 0 else 0.0
                if (spread_pct > baseline_era_pct
                        and (hi - lo) >= baseline_era_abs_mb):
                    dates_sorted = sorted(era_medians.keys())
                    detail = ", ".join(
                        f"{d}={era_medians[d]:.0f}MB" for d in dates_sorted
                    )
                    # INFO, not WARN: this is baseline PEAK memory drifting
                    # across measurement dates — a secondary metric (speed and
                    # speedup are unaffected), usually a single outlier rep or a
                    # GC/load-regime difference, not a data error. Noise, not a
                    # gap.
                    cov.issues.append(Issue(
                        "INFO", "baseline_era_drift", patch_name,
                        tier=tier, threads=thr, platform=plat,
                        variant="baseline",
                        msg=f"baseline peaks span {len(era_medians)} eras "
                            f"({spread_pct:.0f}% spread, +{hi - lo:.0f}MB): "
                            f"{detail}",
                    ))

        # Rule 2e: patched measured before latest baseline refresh.
        if patched_rows:
            bl_dates = [_raw_date(r.get("timestamp") or "")
                        for r in baseline_rows if _raw_date(r.get("timestamp") or "")]
            pt_dates = [_raw_date(r.get("timestamp") or "")
                        for r in patched_rows if _raw_date(r.get("timestamp") or "")]
            if bl_dates and pt_dates:
                latest_bl = max(bl_dates)
                latest_pt = max(pt_dates)
                if latest_pt < latest_bl:
                    ds_note = f" dataset={ds}" if ds else ""
                    # INFO, not WARN: a refreshed baseline paired with an older
                    # patched is a same-era reminder, not a data error. The
                    # speedup ratio stays valid (deterministic patched, one
                    # machine); re-run patched only if the machine era changed.
                    cov.issues.append(Issue(
                        "INFO", "stale_baseline_pairing", patch_name,
                        tier=tier, threads=thr, platform=plat,
                        variant="patched",
                        msg=f"patched last measured {latest_pt} but baseline "
                            f"refreshed {latest_bl}{ds_note}; re-run patched "
                            f"on latest baseline era",
                    ))


# ----------------------------------------------------------------------------
# audit one patch
# ----------------------------------------------------------------------------


def audit_patch(patch_name: str, tsv_path: Path, *,
                patched_min_reps: int = 2,
                baseline_min_reps: int = 2,
                rep_variance_pct: float = 10.0,
                rep_variance_abs_sec: float = 2.0,
                mem_variance_pct: float = 20.0,
                mem_variance_abs_mb: float = 1024.0,
                baseline_era_pct: float = 20.0,
                baseline_era_abs_mb: float = 1024.0,
                platform_filter: str | None = None) -> PatchCoverage:
    """Run all coverage rules on one patch's finalized TSV. See module docstring.

    ``platform_filter``: ``None`` = both platforms; ``"mac"`` or ``"win"`` =
    lint only that platform's rows (finalized ``platform`` column and raw
    ``speedups.{mac,win}.tsv`` shards).
    """
    cov = PatchCoverage(patch=patch_name, tsv_path=tsv_path,
                        exists=tsv_path.exists())
    if not cov.exists:
        cov.issues.append(Issue(
            "FAIL", "missing_tsv", patch_name,
            msg=f"speedups_finalized.tsv not found at {tsv_path}",
        ))
        return cov

    all_rows = _read_tsv_rows(tsv_path)
    if not all_rows:
        cov.issues.append(Issue(
            "FAIL", "empty_tsv", patch_name,
            msg="TSV exists but has no data rows",
        ))
        return cov

    rows = _filter_rows_by_platform(all_rows, platform_filter)
    cov.n_rows = len(rows)
    if not rows:
        return cov

    # ------------------------------------------------------------------
    # Rule 1: bad_status
    # ------------------------------------------------------------------
    for r in rows:
        status = (r.get("status") or "").strip()
        if status not in VALID_STATUSES:
            cov.issues.append(Issue(
                "FAIL", "bad_status", patch_name,
                tier=r.get("tier", ""), threads=r.get("threads", ""),
                platform=_normalize_platform(r.get("platform", "")),
                variant=r.get("variant", ""),
                msg=f"status='{status}' not in {sorted(VALID_STATUSES)}",
            ))

    # ------------------------------------------------------------------
    # Rule 2: low_reps (only on status=ok rows; OOM/partial are knowingly
    # incomplete and shouldn't double-trigger). OOD tiers report as INFO
    # because a single OOD run is typically the largest data point we ship
    # and replicating it is often cost-prohibitive.
    # ------------------------------------------------------------------
    for r in rows:
        if (r.get("status") or "").strip() != "ok":
            continue
        n_reps = _parse_int(r.get("n_reps", ""))
        if n_reps is None:
            continue
        variant = r.get("variant", "")
        threshold = (baseline_min_reps if variant == "baseline"
                     else patched_min_reps)
        if n_reps < threshold:
            tier = r.get("tier", "")
            if _oom_skipped(patch_name, tier, variant, platform_filter):
                continue
            severity = "INFO" if _is_ood_tier(tier) else "WARN"
            cov.issues.append(Issue(
                severity, "low_reps", patch_name,
                tier=tier, threads=r.get("threads", ""),
                platform=_normalize_platform(r.get("platform", "")),
                variant=variant,
                msg=f"n_reps={n_reps} < {threshold}",
            ))

    # ------------------------------------------------------------------
    # Rule 2b: high_rep_variance (sec)
    # ------------------------------------------------------------------
    _append_rep_variance_issues(
        cov, patch_name, rows,
        rep_col="sec_reps", kind="high_rep_variance",
        rep_variance_pct=rep_variance_pct,
        abs_floor=rep_variance_abs_sec,
    )

    # ------------------------------------------------------------------
    # Rule 2c: high_mem_rep_variance (peak MB)
    # ------------------------------------------------------------------
    _append_rep_variance_issues(
        cov, patch_name, rows,
        rep_col="mem_reps", kind="high_mem_rep_variance",
        rep_variance_pct=mem_variance_pct,
        abs_floor=mem_variance_abs_mb,
        spread_ratio=3.0,
        value_fmt=".1f",
    )

    # ------------------------------------------------------------------
    # Rules 2d/2e: baseline era drift + stale baseline/patched pairing
    # (raw per-rep TSV beside speedups_finalized.tsv)
    # ------------------------------------------------------------------
    _audit_raw_era_issues(
        cov, patch_name, tsv_path.parent,
        baseline_era_pct=baseline_era_pct,
        baseline_era_abs_mb=baseline_era_abs_mb,
        platform_filter=platform_filter,
    )

    # ------------------------------------------------------------------
    # Build join cells (rules 3, 4, 5 — exclude bad-status rows so they
    # don't double-trigger; rule 1 already covered them).
    # ------------------------------------------------------------------
    variants_by_cell: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    statuses_by_cell_variant: dict[tuple[str, str, str, str], str] = {}
    platforms_by_tier_thread: dict[tuple[str, str], set[str]] = defaultdict(set)
    threads_by_tier_platform: dict[tuple[str, str], set[str]] = defaultdict(set)
    all_threads_in_patch: set[str] = set()

    for r in rows:
        status = (r.get("status") or "").strip()
        if status not in JOIN_STATUSES:
            continue
        tier = r.get("tier", "")
        thr = r.get("threads", "")
        plat = _normalize_platform(r.get("platform", ""))
        variant = r.get("variant", "")
        variants_by_cell[(tier, thr, plat)].add(variant)
        statuses_by_cell_variant[(tier, thr, plat, variant)] = status
        platforms_by_tier_thread[(tier, thr)].add(plat)
        threads_by_tier_platform[(tier, plat)].add(thr)
        all_threads_in_patch.add(thr)

    # ------------------------------------------------------------------
    # Rule 3: missing_variant (baseline/patched symmetry per cell). A cell
    # whose only present variant has status=OOM is intentionally one-sided
    # — the missing twin would have OOMed too — so we skip it instead of
    # warning.
    # ------------------------------------------------------------------
    for (tier, thr, plat), variants in sorted(variants_by_cell.items()):
        if "baseline" not in variants and "patched" in variants:
            patched_status = statuses_by_cell_variant.get(
                (tier, thr, plat, "patched"), "")
            if patched_status == "OOM" or _oom_skipped(
                    patch_name, tier, "baseline", platform_filter):
                continue
            if (thr != "1" and "baseline" in _REDUCED.get(patch_name, {}).get(
                    platform_filter or "", set())):
                continue  # baseline reduced to t1 on THIS platform; higher-thread
                          # baseline is optional (patched kept because it scales)
            cov.issues.append(Issue(
                "WARN", "missing_variant", patch_name,
                tier=tier, threads=thr, platform=plat, variant="baseline",
                msg=f"patched exists but baseline missing at {tier}/{thr}/{plat}",
            ))
        elif "patched" not in variants and "baseline" in variants:
            baseline_status = statuses_by_cell_variant.get(
                (tier, thr, plat, "baseline"), "")
            if baseline_status == "OOM" or _oom_skipped(
                    patch_name, tier, "patched", platform_filter):
                continue
            if (thr != "1" and "patched" in _REDUCED.get(patch_name, {}).get(
                    platform_filter or "", set())):
                continue  # patched reduced to t1 on THIS platform; higher-thread
                          # patched is optional (baseline kept for the record)
            cov.issues.append(Issue(
                "WARN", "missing_variant", patch_name,
                tier=tier, threads=thr, platform=plat, variant="patched",
                msg=f"baseline exists but patched missing at {tier}/{thr}/{plat}",
            ))

    # ------------------------------------------------------------------
    # Rule 4: platform_asymmetry — only fires if the patch claims dual-platform
    # coverage somewhere (Mac AND Win both appear in the TSV). For headline
    # scanpy_*/seurat_* patches the per-platform fullt is different (14 vs 32),
    # so a cell at threads=14 will never have a Win twin and vice versa — those
    # asymmetries are by design and we skip them.
    # ------------------------------------------------------------------
    all_platforms = set()
    for plats in platforms_by_tier_thread.values():
        all_platforms.update(plats)
    headline_fullt_threads: set[str] = set()
    if _has_headline_schedule(patch_name):
        win_sched = _HEADLINE_THREAD_SCHEDULE["win"]
        mac_sched = _HEADLINE_THREAD_SCHEDULE["mac"]
        headline_fullt_threads = (win_sched ^ mac_sched)
    if (not platform_filter
            and "win" in all_platforms and "mac" in all_platforms):
        for (tier, thr), plats in sorted(platforms_by_tier_thread.items()):
            if thr in headline_fullt_threads:
                continue
            if "win" in plats and "mac" not in plats:
                cov.issues.append(Issue(
                    "INFO", "platform_asymmetry", patch_name,
                    tier=tier, threads=thr, platform="mac",
                    msg=f"Win has data at {tier}/{thr} but Mac doesn't",
                ))
            elif "mac" in plats and "win" not in plats:
                cov.issues.append(Issue(
                    "INFO", "platform_asymmetry", patch_name,
                    tier=tier, threads=thr, platform="win",
                    msg=f"Mac has data at {tier}/{thr} but Win doesn't",
                ))

    # ------------------------------------------------------------------
    # Rule 5: thread_gap — per-platform expected thread set, checked per
    # (tier, platform).
    #
    # - Headline scanpy_*/seurat_* patches use the fixed schedule
    #   {1, 4, 14/32} (fullt depends on platform).
    # - All other patches: each platform's expected set is the union of
    #   threads ACTUALLY observed on that platform (not across platforms).
    #
    # Single-thread patches (only {1} on every platform) trivially pass.
    # ------------------------------------------------------------------
    threads_by_platform: dict[str, set[str]] = defaultdict(set)
    tiers_by_platform: dict[str, set[str]] = defaultdict(set)
    for (tier, plat), thrs in threads_by_tier_platform.items():
        clean = {t for t in thrs if not _is_blank_threads(t)}
        threads_by_platform[plat].update(clean)
        tiers_by_platform[plat].add(tier)

    # Expected thread set per platform follows the rubric (thread_rubric.yaml):
    # single-threaded patches -> {1}; headline (scanpy_*/seurat_*) -> headline
    # schedule; everything else -> default schedule. Applied regardless of the
    # observed threads, so a parallel patch that has only run 1t on a platform
    # is correctly flagged as missing 4/full there.
    headline = _has_headline_schedule(patch_name)
    expected_by_platform: dict[str, set[str]] = {}
    fullt_by_platform: dict[str, str | None] = {}
    for plat, obs_threads in threads_by_platform.items():
        if patch_name in _SINGLE_THREADED_PATCHES.get(plat, set()):
            sched = {"1"}
        elif headline and plat in _HEADLINE_THREAD_SCHEDULE:
            sched = set(_HEADLINE_THREAD_SCHEDULE[plat])
        elif plat in _DEFAULT_THREAD_SCHEDULE:
            sched = set(_DEFAULT_THREAD_SCHEDULE[plat])
        else:
            sched = set(obs_threads)
        expected_by_platform[plat] = sched
        fullt_by_platform[plat] = next(iter(sched - {"1", "4"}), None)

    for plat, expected in sorted(expected_by_platform.items()):
        if len(expected) <= 1:
            continue
        for tier in sorted(tiers_by_platform[plat]):
            present = {t for t in threads_by_tier_platform.get((tier, plat), set())
                       if not _is_blank_threads(t)}
            missing = sorted(expected - present)
            # Whole tier OOMs on this Mac (baseline can't run at any thread) —
            # the missing threads are expected, not a coverage gap.
            if _oom_skipped(patch_name, tier, "baseline", platform_filter):
                continue
            for m in missing:
                # fullt on OOD tiers commonly OOMs (both headline and default
                # schedules). Treat as INFO so the gap stays visible but isn't
                # presented as a coverage failure.
                if (_is_ood_tier(tier)
                        and m == fullt_by_platform.get(plat)):
                    cov.issues.append(Issue(
                        "INFO", "thread_gap", patch_name,
                        tier=tier, threads=m, platform=plat,
                        msg=f"{plat} fullt={m} on OOD tier {tier} "
                            f"(commonly OOMs)",
                    ))
                    continue
                cov.issues.append(Issue(
                    "WARN", "thread_gap", patch_name,
                    tier=tier, threads=m, platform=plat,
                    msg=f"{plat} expected threads={sorted(expected)}, "
                        f"missing {tier}/{plat}@threads={m}",
                ))

    return cov


# ----------------------------------------------------------------------------
# discover all patches
# ----------------------------------------------------------------------------


def discover_patch_tsvs(
    framework: Path, *, skip_experimental: bool = False,
) -> list[tuple[str, Path]]:
    """Walk both ecosystems and return (patch_name, tsv_path) for every
    `<patch>/speedups_finalized.tsv` under autozyme_r/inst/patches/ and
    autozyme_py/src/autozyme/. Sorted by patch name; missing TSVs are skipped
    (caller's discover step) — `audit_patch` handles the missing case at
    audit time, but here we only enumerate dirs that actually ship a TSV.

    `skip_experimental=True` drops fusion-experiment and orchestrate headline
    patches (see `_is_experimental`) so the lint focuses on directly-attestable
    standalone patches."""
    out: list[tuple[str, Path]] = []
    r_root = framework / "autozyme_r" / "inst" / "patches"
    if r_root.is_dir():
        for child in sorted(r_root.iterdir()):
            if not child.is_dir():
                continue
            if skip_experimental and _is_experimental(child.name):
                continue
            tsv = child / "speedups_finalized.tsv"
            if tsv.exists():
                out.append((child.name, tsv))
    py_root = framework / "autozyme_py" / "src" / "autozyme"
    if py_root.is_dir():
        for child in sorted(py_root.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            if skip_experimental and _is_experimental(child.name):
                continue
            tsv = child / "speedups_finalized.tsv"
            if tsv.exists():
                out.append((child.name, tsv))
    return out


def audit_all_patches(
    framework: Path, *, skip_experimental: bool = False, **kwargs,
) -> list[PatchCoverage]:
    """Audit every patch with a shipped finalized TSV. Stable order: R patches
    alphabetically, then Py patches alphabetically. Duplicates (rare — same
    patch name in both ecosystems) keep both entries; downstream rendering
    disambiguates by ecosystem path in the patch column when needed.

    `skip_experimental=True` excludes fusion + orchestrate headline patches."""
    return [audit_patch(name, tsv, **kwargs)
            for name, tsv in discover_patch_tsvs(
                framework, skip_experimental=skip_experimental)]


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------


def _platform_filter_label(platform_filter: str | None) -> str:
    if platform_filter == "mac":
        return "macOS only"
    if platform_filter == "win":
        return "Windows only"
    return "all platforms"


def render_table(audits: list[PatchCoverage], *, detail: bool = False,
                 platform_filter: str | None = None) -> str:
    """Plain-text table for terminal. `detail=False` shows just the
    PASS/FAIL/WARN/INFO counts per patch; `detail=True` adds one line per
    issue grouped by patch underneath."""
    if not audits:
        return "Speedup coverage scan: no patches with shipped TSVs found.\n"

    plat_note = _platform_filter_label(platform_filter)
    lines = ["",
             "Speedup coverage scan (inside-out lint of speedups_finalized.tsv)",
             f"Platform filter: {plat_note}",
             "=" * 63, ""]
    name_w = max(8, max(len(a.patch) for a in audits))
    header = f"  {'patch':<{name_w}}  rows  FAIL  WARN  INFO  status"
    lines.append(header)
    lines.append("  " + "-" * (name_w + 32))
    total = {"FAIL": 0, "WARN": 0, "INFO": 0}
    for a in audits:
        c = a.counts()
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
        status_marker = ("clean" if a.is_clean
                         else "FAIL" if c["FAIL"] else "WARN" if c["WARN"] else "INFO")
        lines.append(
            f"  {a.patch:<{name_w}}  {a.n_rows:>4}  "
            f"{c['FAIL']:>4}  {c['WARN']:>4}  {c['INFO']:>4}  {status_marker}"
        )
    lines.append("  " + "-" * (name_w + 32))
    lines.append(
        f"  {'TOTAL':<{name_w}}  {sum(a.n_rows for a in audits):>4}  "
        f"{total['FAIL']:>4}  {total['WARN']:>4}  {total['INFO']:>4}"
    )

    if detail:
        for a in audits:
            if a.is_clean:
                continue
            lines.append(f"\n  ## {a.patch}")
            for issue in a.issues:
                where = "/".join(p for p in (issue.tier, issue.threads,
                                              issue.platform, issue.variant) if p)
                lines.append(
                    f"    [{issue.severity:<4}] {issue.kind:<19} "
                    f"{where:<35} {issue.msg}"
                )
    lines.append("")
    return "\n".join(lines)


def render_markdown(audits: list[PatchCoverage],
                    platform_filter: str | None = None) -> str:
    """Markdown section for embedding into SCAN.md. Always emits the full
    detail listing (markdown is a written deliverable, not a glance view)."""
    plat_note = _platform_filter_label(platform_filter)
    lines = [
        "## Speedup coverage",
        "",
        f"Platform filter: **{plat_note}**. "
        "Inside-out lint of `inst/patches/<patch>/speedups_finalized.tsv` "
        "(no external metadata). Severity legend: **FAIL** = bad status / "
        "missing TSV; **WARN** = under-replicated, rep variance, baseline-era "
        "drift, stale baseline/patched pairing, missing variant, or thread "
        "gap; **INFO** = cross-platform asymmetry (often legitimate, e.g. Mac "
        "OOM at large scale).",
        "",
        "| patch | rows | FAIL | WARN | INFO | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for a in audits:
        c = a.counts()
        status_marker = ("clean" if a.is_clean
                         else ":x: FAIL" if c["FAIL"]
                         else ":warning: WARN" if c["WARN"]
                         else ":information_source: INFO")
        lines.append(
            f"| {a.patch} | {a.n_rows} | {c['FAIL']} | {c['WARN']} | "
            f"{c['INFO']} | {status_marker} |"
        )
    lines.append("")

    any_issues = any(not a.is_clean for a in audits)
    if any_issues:
        lines.append("### Details")
        lines.append("")
        for a in audits:
            if a.is_clean:
                continue
            lines.append(f"#### {a.patch}")
            lines.append("")
            for issue in a.issues:
                where = "/".join(p for p in (issue.tier, issue.threads,
                                              issue.platform, issue.variant) if p)
                lines.append(
                    f"- **{issue.severity}** `{issue.kind}` "
                    f"{('— ' + where) if where else ''}: {issue.msg}"
                )
            lines.append("")
    return "\n".join(lines)


def render_json_records(audits: list[PatchCoverage]) -> list[dict]:
    """NDJSON-friendly: one record per issue. Patches with no issues emit a
    single 'clean' record so the consumer can tell them apart from missing
    entries."""
    out: list[dict] = []
    for a in audits:
        if a.is_clean:
            out.append({
                "patch": a.patch, "n_rows": a.n_rows,
                "severity": "PASS", "kind": "clean",
            })
            continue
        for issue in a.issues:
            out.append({
                "patch": a.patch, "n_rows": a.n_rows,
                "severity": issue.severity, "kind": issue.kind,
                "tier": issue.tier, "threads": issue.threads,
                "platform": issue.platform, "variant": issue.variant,
                "msg": issue.msg,
            })
    return out


def has_fail(audits: list[PatchCoverage]) -> bool:
    return any(i.severity == "FAIL" for a in audits for i in a.issues)


def has_warn_or_fail(audits: list[PatchCoverage]) -> bool:
    return any(i.severity in ("FAIL", "WARN") for a in audits for i in a.issues)
