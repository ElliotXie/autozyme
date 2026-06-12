"""Patch-coverage audit for the autozyme_r and autozyme_py packages.

Backs `zyme scan --attest`. Walks both packages, enumerates every patch
shipped, and reports — for each — whether `speedups_finalized.tsv` is
present and which workload tiers have valid attest data. Pure package-side
inspection: does NOT walk the workspace.

Release packages ship finalized TSVs only. Raw `speedups.tsv` histories stay
in the framework repository and are accepted here only as a legacy fallback.

Platform handling: every batch is bucketed by `system_os` into one of
`win`, `mac`, `unknown`. **Batches with an empty `system_os` are
treated as mac** — the framework only began auto-populating that
column when Windows porting started, so all legacy rows were mac
runs. Latest-wins is done per (platform, tier), so a patch with both
Mac and Windows data for the same tier keeps both batches visible.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median


TIERS_5 = ["small", "medium", "large", "ood_large", "ood_xlarge"]
TIERS_6 = ["small", "medium", "large", "ood_large1", "ood_large2", "ood_large3"]
TIERS = sorted(set(TIERS_5 + TIERS_6), key=lambda t: (TIERS_5 + TIERS_6).index(t))
TIER_LETTERS = {"small": "S", "medium": "M", "large": "L",
                "ood_large": "o", "ood_xlarge": "O",
                "ood_large1": "1", "ood_large2": "2", "ood_large3": "3"}

_TIER_ALIASES = {"tiny": "small"}


def normalize_tier(tier: str) -> str:
    """Map legacy tier names to their current canonical form."""
    return _TIER_ALIASES.get(tier, tier)


PLATFORMS = ("win", "mac", "unknown")
PLATFORM_LABEL = {"win": "W", "mac": "M", "unknown": "?"}

_CORE_PANEL_PATCHES: set[str] = set()

_LIFTED_FROM_RE = re.compile(r"Lifted from autozyme task\s+`+([^`]+)`+")


def _normalize_platform(system_os: object) -> str:
    """Map a raw `system_os` cell to one of `win` / `mac` / `unknown`.

    Empty / blank → `mac` (legacy rows pre-platform tagging were all
    run on macOS; see module docstring)."""
    if system_os is None:
        return "mac"
    s = str(system_os).strip()
    if not s:
        return "mac"
    low = s.lower()
    if low.startswith("windows"):
        return "win"
    if low.startswith("macos") or low.startswith("darwin"):
        return "mac"
    return "unknown"


@dataclass
class TierStatus:
    tier: str
    platform: str       # "win" | "mac" | "unknown"
    state: str          # "ok" | "crashed" | "missing"
    speedup_x: float | None = None
    concordance: float | None = None
    note: str = ""
    prev_speedup_x: float | None = None
    prev_concordance: float | None = None
    prev_timestamp: str = ""
    latest_timestamp: str = ""


DRIFT_SPEED_THRESHOLD = 0.30   # 30% change in speedup_x
DRIFT_CONC_THRESHOLD = 0.005   # concordance drop > 0.005


@dataclass
class DriftAlert:
    patch: str
    platform: str
    tier: str
    kind: str          # "speedup_up" | "speedup_down" | "concordance_drop"
    prev_val: float
    curr_val: float
    prev_ts: str
    curr_ts: str

    @property
    def description(self) -> str:
        if self.kind == "speedup_up":
            return (f"speedup {_fmt_fold(self.prev_val)} -> {_fmt_fold(self.curr_val)} "
                    f"(+{(self.curr_val / self.prev_val - 1) * 100:.0f}%)")
        if self.kind == "speedup_down":
            return (f"speedup {_fmt_fold(self.prev_val)} -> {_fmt_fold(self.curr_val)} "
                    f"({(self.curr_val / self.prev_val - 1) * 100:.0f}%)")
        return (f"concordance {self.prev_val:.4f} -> {self.curr_val:.4f} "
                f"({(self.curr_val - self.prev_val):.4f})")


@dataclass
class PatchAudit:
    name: str
    language: str               # "R" | "Py"
    patch_path: Path
    speedups_path: Path
    speedups_exists: bool
    lifted_from_task: str | None
    # Keyed by (platform, tier). A patch with both Mac and Windows data
    # for the same tier appears as two entries.
    tier_status: dict[tuple[str, str], TierStatus] = field(default_factory=dict)
    median_fold_by_platform: dict[str, float] = field(default_factory=dict)
    median_conc_by_platform: dict[str, float] = field(default_factory=dict)
    is_core: bool = False       # seurat / scanpy
    expected_tiers_override: list[str] | None = None

    # ------------------------------------------------------------------
    # Per-platform views
    # ------------------------------------------------------------------
    def state_at(self, platform: str, tier: str) -> str:
        st = self.tier_status.get((platform, tier))
        return st.state if st else "missing"

    def tiers_ok_on(self, platform: str) -> list[str]:
        return [t for t in TIERS if self.state_at(platform, t) == "ok"]

    def tiers_crashed_on(self, platform: str) -> list[str]:
        return [t for t in TIERS if self.state_at(platform, t) == "crashed"]

    def tiers_missing_on(self, platform: str) -> list[str]:
        return [t for t in TIERS if self.state_at(platform, t) == "missing"]

    def platforms_with_data(self) -> list[str]:
        seen = {plat for (plat, _t) in self.tier_status.keys()}
        return [p for p in PLATFORMS if p in seen]

    # ------------------------------------------------------------------
    # Aggregate views (used by summary stats)
    # ------------------------------------------------------------------
    @property
    def expected_tiers(self) -> list[str]:
        """The tier scheme this patch uses — detected from data."""
        if self.expected_tiers_override:
            return [t for t in self.expected_tiers_override if t in TIERS]
        data_tiers = {t for (_, t) in self.tier_status}
        if data_tiers & {"ood_large1", "ood_large2", "ood_large3"}:
            return TIERS_6
        return TIERS_5

    @property
    def tiers_ok_anywhere(self) -> list[str]:
        """Expected tiers that ran ok on at least one platform."""
        return [
            t for t in self.expected_tiers
            if any(self.state_at(p, t) == "ok" for p in PLATFORMS)
        ]

    @property
    def status(self) -> str:
        """Patch-level status. `full` = every expected tier ran ok on at
        least one platform. Cross-platform coverage is reported separately."""
        if not self.speedups_exists:
            return "no_attest"
        if not self.tier_status:
            return "no_data"
        if len(self.tiers_ok_anywhere) == len(self.expected_tiers):
            return "full"
        if self.tiers_ok_anywhere:
            return "partial"
        return "no_data"

    def platform_status(self, platform: str) -> str:
        """Status for a single platform. `none` = no rows on this
        platform at all (don't report as a gap)."""
        plat_keys = [k for k in self.tier_status if k[0] == platform]
        if not plat_keys:
            return "none"
        ok = [t for t in self.expected_tiers if self.state_at(platform, t) == "ok"]
        if len(ok) == len(self.expected_tiers):
            return "full"
        return "partial" if ok else "no_data"

    def drift_alerts(self) -> list[DriftAlert]:
        """Detect significant changes between latest and previous batch."""
        alerts: list[DriftAlert] = []
        for (plat, tier), st in self.tier_status.items():
            if st.state != "ok" or st.speedup_x is None:
                continue
            # Speedup drift
            if st.prev_speedup_x is not None and st.prev_speedup_x > 0:
                ratio = st.speedup_x / st.prev_speedup_x
                if ratio > 1 + DRIFT_SPEED_THRESHOLD:
                    alerts.append(DriftAlert(
                        self.name, plat, tier, "speedup_up",
                        st.prev_speedup_x, st.speedup_x,
                        st.prev_timestamp, st.latest_timestamp,
                    ))
                elif ratio < 1 - DRIFT_SPEED_THRESHOLD:
                    alerts.append(DriftAlert(
                        self.name, plat, tier, "speedup_down",
                        st.prev_speedup_x, st.speedup_x,
                        st.prev_timestamp, st.latest_timestamp,
                    ))
            # Concordance drop
            if (st.concordance is not None and st.prev_concordance is not None
                    and st.prev_concordance - st.concordance > DRIFT_CONC_THRESHOLD):
                alerts.append(DriftAlert(
                    self.name, plat, tier, "concordance_drop",
                    st.prev_concordance, st.concordance,
                    st.prev_timestamp, st.latest_timestamp,
                ))
        return alerts


def _extract_lifted_from(path: Path) -> str | None:
    try:
        text = path.read_text(errors="ignore")[:8192]
    except OSError:
        return None
    m = _LIFTED_FROM_RE.search(text)
    return m.group(1).strip() if m else None


def _as_bool_false(value: object) -> bool:
    if value is False:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "0", "no"}:
        return True
    return False


def _load_manifest_expected_tiers(framework: Path) -> dict[Path, list[str]]:
    """Map core per-step speedup files to explicit manifest tiers.

    Seurat/Scanpy ship multiple speedup TSVs under one patch package. A few
    tasks intentionally have a narrower tier set than the default 5/6-tier
    families, so data-shape inference reports false gaps unless we honor the
    attest manifest.
    """
    try:
        import yaml
    except ImportError:
        return {}

    specs = [
        (
            framework / "scripts" / "seurat_attest_manifest.yaml",
            framework / "autozyme_r" / "inst" / "speedups",
            "seurat",
        ),
        (
            framework / "scripts" / "scanpy_attest_manifest.yaml",
            framework / "autozyme_py" / "src" / "autozyme" / "scanpy" / "speedups",
            "scanpy",
        ),
    ]
    out: dict[Path, list[str]] = {}
    for manifest_path, dest_dir, prefix in specs:
        if not manifest_path.is_file():
            continue
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            continue
        for task in manifest.get("tasks") or []:
            if _as_bool_false(task.get("package_speedups")):
                continue
            explicit_tiers = task.get("tiers")
            if not explicit_tiers:
                continue
            legacy = task.get("legacy_key") or task.get("id")
            if not legacy:
                continue
            tiers = [normalize_tier(str(t)) for t in explicit_tiers]
            out[(dest_dir / f"{prefix}_{legacy}.tsv").resolve()] = tiers
    return out


def _concordance_min(metrics: dict) -> float | None:
    incl = ("pearson", "spearman", "jaccard", "ari", "agreement",
            "match", "overlap", "coverage", "corr", "c_index",
            "structure_key")
    excl = ("diff", "max_abs", "rel_diff", "cpu_sec", "rmse", "rmsd", "error")
    vals: list[float] = []
    for key, value in metrics.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if not (0 <= v <= 1.000001):
            continue
        n = key.lower()
        if any(t in n for t in incl) and not any(t in n for t in excl):
            vals.append(min(v, 1.0))
    return min(vals) if vals else None


def _enumerate_patches(framework: Path) -> list[PatchAudit]:
    out: list[PatchAudit] = []
    expected_tiers_by_speedup = _load_manifest_expected_tiers(framework)

    r_patches = framework / "autozyme_r" / "inst" / "patches"
    r_speedups = framework / "autozyme_r" / "inst" / "speedups"
    if r_patches.is_dir():
        # Two coexisting layouts:
        #   folder: inst/patches/<name>/patch.R  + speedups_finalized.tsv
        #   legacy: inst/patches/<name>.R        + ../speedups/<name>.tsv
        # Per-method folder patches (currently Seurat) store method snapshots
        # in inst/speedups/<patch>_*.tsv.
        for d in sorted(r_patches.iterdir()):
            if not d.is_dir():
                continue
            patch_r = d / "patch.R"
            if not patch_r.is_file():
                continue
            sp = d / "speedups_finalized.tsv"
            # Sub-speedups: a single patch can ship many per-method TSVs
            # named "<prefix>_*.tsv". Canonical storage is inst/speedups/;
            # patch-dir lookup remains for legacy snapshots.
            sub_tsvs = sorted({
                *d.glob(f"{d.name}_*.tsv"),
                *r_speedups.glob(f"{d.name}_*.tsv"),
            })
            if sub_tsvs:
                lifted = _extract_lifted_from(patch_r)
                for sub in sub_tsvs:
                    out.append(PatchAudit(
                        name=sub.stem,
                        language="R",
                        patch_path=patch_r,
                        speedups_path=sub,
                        speedups_exists=True,
                        lifted_from_task=lifted,
                        is_core=(d.name in _CORE_PANEL_PATCHES),
                        expected_tiers_override=expected_tiers_by_speedup.get(sub.resolve()),
                    ))
            else:
                out.append(PatchAudit(
                    name=d.name,
                    language="R",
                    patch_path=patch_r,
                    speedups_path=sp,
                    speedups_exists=sp.is_file(),
                    lifted_from_task=_extract_lifted_from(patch_r),
                    is_core=(d.name in _CORE_PANEL_PATCHES),
                ))
        for f in sorted(r_patches.iterdir()):
            if f.suffix != ".R" or not f.is_file():
                continue
            sp = r_speedups / f"{f.stem}.tsv"
            if sp.is_file():
                out.append(PatchAudit(
                    name=f.stem,
                    language="R",
                    patch_path=f,
                    speedups_path=sp,
                    speedups_exists=True,
                    lifted_from_task=_extract_lifted_from(f),
                    is_core=(f.stem in _CORE_PANEL_PATCHES),
                    expected_tiers_override=expected_tiers_by_speedup.get(sp.resolve()),
                ))
            else:
                sub_tsvs = sorted(r_speedups.glob(f"{f.stem}_*.tsv"))
                if sub_tsvs:
                    lifted = _extract_lifted_from(f)
                    for sub in sub_tsvs:
                        out.append(PatchAudit(
                            name=sub.stem,
                            language="R",
                            patch_path=f,
                            speedups_path=sub,
                            speedups_exists=True,
                            lifted_from_task=lifted,
                            is_core=(f.stem in _CORE_PANEL_PATCHES),
                            expected_tiers_override=expected_tiers_by_speedup.get(sub.resolve()),
                        ))
                else:
                    out.append(PatchAudit(
                        name=f.stem,
                        language="R",
                        patch_path=f,
                        speedups_path=sp,
                        speedups_exists=False,
                        lifted_from_task=_extract_lifted_from(f),
                        is_core=(f.stem in _CORE_PANEL_PATCHES),
                    ))

    py_root = framework / "autozyme_py" / "src" / "autozyme"
    if py_root.is_dir():
        for d in sorted(py_root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            init = d / "__init__.py"
            if not init.is_file():
                continue
            sp = d / "speedups_finalized.tsv"
            sp_dir = d / "speedups"
            if sp.is_file():
                out.append(PatchAudit(
                    name=d.name,
                    language="Py",
                    patch_path=init,
                    speedups_path=sp,
                    speedups_exists=True,
                    lifted_from_task=_extract_lifted_from(init),
                    is_core=(d.name in _CORE_PANEL_PATCHES),
                    expected_tiers_override=expected_tiers_by_speedup.get(sp.resolve()),
                ))
            elif sp_dir.is_dir():
                lifted = _extract_lifted_from(init)
                for sub in sorted(sp_dir.glob("*.tsv")):
                    out.append(PatchAudit(
                        name=sub.stem,
                        language="Py",
                        patch_path=init,
                        speedups_path=sub,
                        speedups_exists=True,
                        lifted_from_task=lifted,
                        is_core=(d.name in _CORE_PANEL_PATCHES),
                        expected_tiers_override=expected_tiers_by_speedup.get(sub.resolve()),
                    ))
            else:
                out.append(PatchAudit(
                    name=d.name,
                    language="Py",
                    patch_path=init,
                    speedups_path=sp,
                    speedups_exists=False,
                    lifted_from_task=_extract_lifted_from(init),
                    is_core=(d.name in _CORE_PANEL_PATCHES),
                ))
    return out


def _parse_speedups(p: PatchAudit) -> None:
    if not p.speedups_exists:
        return
    try:
        rows = list(csv.DictReader(p.speedups_path.open(newline=""), delimiter="\t"))
    except OSError:
        return

    if rows and "speedup_x_mean" in rows[0]:
        _parse_finalized_speedups(p, rows)
        return

    # Long-format reader: group rows by batch, then keep the latest batch per
    # (platform, tier). Lazy import avoids circular dependency with the
    # parser module that imports TIERS/_normalize_platform from us.
    from zyme.parsers.package_verify_tsv import summarize_batches

    batches = summarize_batches(rows)

    # Keep latest + second-latest batch per (platform, tier) for drift
    # detection. Cross-platform data for the same tier is preserved.
    latest: dict[tuple[str, str], object] = {}   # BatchSummary
    previous: dict[tuple[str, str], object] = {}  # second-latest
    for b in batches:
        tier = normalize_tier(b.tier)
        if tier not in TIERS:
            continue
        plat = _normalize_platform(b.system_os)
        key = (plat, tier)
        cur = latest.get(key)
        if cur is None:
            latest[key] = b
        elif b.timestamp > cur.timestamp:
            previous[key] = cur
            latest[key] = b
        elif previous.get(key) is None or b.timestamp > previous[key].timestamp:
            previous[key] = b

    folds_by_plat: dict[str, list[float]] = {plat: [] for plat in PLATFORMS}
    concs_by_plat: dict[str, list[float]] = {plat: [] for plat in PLATFORMS}

    for (plat, tier), batch in latest.items():
        metrics = _pick_batch_metrics(rows, batch)
        conc = _concordance_min(metrics)
        note = (batch.note or "").strip()

        prev_fold: float | None = None
        prev_conc: float | None = None
        prev_ts = ""
        prev_batch = previous.get((plat, tier))
        if prev_batch is not None:
            if prev_batch.n_reps and prev_batch.baseline_sec_secs and prev_batch.patched_sec_secs:
                prev_fold = prev_batch.speedup_x
            prev_metrics = _pick_batch_metrics(rows, prev_batch)
            prev_conc = _concordance_min(prev_metrics)
            prev_ts = prev_batch.timestamp

        if batch.n_reps and batch.baseline_sec_secs and batch.patched_sec_secs:
            fold_f = batch.speedup_x
            p.tier_status[(plat, tier)] = TierStatus(
                tier=tier, platform=plat, state="ok",
                speedup_x=fold_f, concordance=conc, note=note,
                prev_speedup_x=prev_fold, prev_concordance=prev_conc,
                prev_timestamp=prev_ts, latest_timestamp=batch.timestamp,
            )
            if fold_f == fold_f:  # not NaN
                folds_by_plat[plat].append(fold_f)
            if conc is not None:
                concs_by_plat[plat].append(conc)
        else:
            p.tier_status[(plat, tier)] = TierStatus(
                tier=tier, platform=plat, state="crashed",
                note=note or "empty batch",
                prev_speedup_x=prev_fold, prev_concordance=prev_conc,
                prev_timestamp=prev_ts, latest_timestamp=batch.timestamp,
            )

    for plat in PLATFORMS:
        if folds_by_plat[plat]:
            p.median_fold_by_platform[plat] = median(folds_by_plat[plat])
        if concs_by_plat[plat]:
            p.median_conc_by_platform[plat] = median(concs_by_plat[plat])


def _float_cell(value: object) -> float | None:
    try:
        s = "" if value is None else str(value).strip()
        if not s or s.upper() in {"NA", "NAN", "NONE"}:
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _finalized_group_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        (row.get("patch") or "").strip(),
        normalize_tier((row.get("tier") or "").strip()),
        (row.get("platform") or "").strip(),
        (row.get("threads") or "").strip(),
        (row.get("dataset") or "").strip(),
        (row.get("package_version") or "").strip(),
    )


def _parse_finalized_speedups(p: PatchAudit, rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_finalized_group_key(row), []).append(row)

    folds_by_plat: dict[str, list[float]] = {plat: [] for plat in PLATFORMS}
    concs_by_plat: dict[str, list[float]] = {plat: [] for plat in PLATFORMS}

    for key, group_rows in grouped.items():
        _patch, tier, _platform, _threads, _dataset, _pkg = key
        if tier not in TIERS:
            continue
        patched = next((r for r in group_rows if (r.get("variant") or "") == "patched"), None)
        baseline = next((r for r in group_rows if (r.get("variant") or "") == "baseline"), None)
        if patched is None:
            continue

        plat = _normalize_platform(patched.get("platform"))
        timestamp = (patched.get("ts_last") or patched.get("ts_first") or "").strip()
        status = (patched.get("status") or "").strip().lower()
        speedup_x = _float_cell(patched.get("speedup_x_mean"))
        reps = _float_cell(patched.get("n_reps")) or 0
        metrics = _metrics_from_finalized_row(patched)
        conc = _concordance_min(metrics)
        state = "ok" if baseline is not None and status == "ok" and speedup_x is not None and reps > 0 else "crashed"

        p.tier_status[(plat, tier)] = TierStatus(
            tier=tier, platform=plat, state=state,
            speedup_x=speedup_x if state == "ok" else None,
            concordance=conc, note=patched.get("status") or "",
            latest_timestamp=timestamp,
        )
        if state == "ok" and speedup_x is not None and speedup_x == speedup_x:
            folds_by_plat[plat].append(speedup_x)
        if state == "ok" and conc is not None:
            concs_by_plat[plat].append(conc)

    for plat in PLATFORMS:
        if folds_by_plat[plat]:
            p.median_fold_by_platform[plat] = median(folds_by_plat[plat])
        if concs_by_plat[plat]:
            p.median_conc_by_platform[plat] = median(concs_by_plat[plat])


def _metrics_from_finalized_row(row: dict[str, str]) -> dict:
    raw = row.get("metrics_json_median") or ""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _pick_batch_metrics(rows: list[dict], batch) -> dict:
    """Return parsed metrics_json from the first patched row of ``batch``."""
    for r in rows:
        if (r.get("variant") or "") != "patched":
            continue
        if (r.get("timestamp") or "").strip() != batch.timestamp:
            continue
        if (r.get("tier") or "").strip() != batch.tier:
            continue
        if (r.get("system_os") or "").strip() != batch.system_os:
            continue
        if (r.get("system_cpu") or "").strip() != batch.system_cpu:
            continue
        if (r.get("system_threads") or "").strip() != batch.system_threads:
            continue
        try:
            return json.loads(r.get("metrics_json") or "{}")
        except json.JSONDecodeError:
            return {}
    return {}


def audit_packages(framework: Path) -> list[PatchAudit]:
    """Enumerate every patch in autozyme_r + autozyme_py and parse its
    bundled finalized speedup TSV. Returns one PatchAudit per patch."""
    patches = _enumerate_patches(framework)
    for p in patches:
        _parse_speedups(p)
    return patches


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

_TIER_GLYPH_PLAIN = {"ok": "o", "crashed": "x", "missing": "."}
_TIER_GLYPH_MD = {"ok": "●", "crashed": "✗", "missing": "·"}


def _dual_cell(p: PatchAudit, tier: str, glyph_map: dict[str, str]) -> str:
    """Two-char cell: Windows glyph then macOS glyph."""
    return (
        glyph_map[p.state_at("win", tier)] +
        glyph_map[p.state_at("mac", tier)]
    )


def _fmt_fold(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v / 1000:.1f}kx"
    if v >= 100:
        return f"{v:.0f}x"
    if v >= 10:
        return f"{v:.0f}x"
    if v >= 2:
        return f"{v:.1f}x"
    return f"{v:.2f}x"


def _fmt_conc(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _platform_tag(p: PatchAudit) -> str:
    plats = p.platforms_with_data()
    if not plats:
        return "—"
    # Show in a stable W+M+? order
    order = {"win": 0, "mac": 1, "unknown": 2}
    return "+".join(PLATFORM_LABEL[x] for x in sorted(plats, key=order.get))


def _short_note(p: PatchAudit) -> str:
    if not p.speedups_exists:
        return "patch present, no speedups_finalized.tsv"
    if not p.tier_status:
        return "speedups_finalized.tsv empty"
    plats = p.platforms_with_data()
    bits: list[str] = []
    for plat in plats:
        st = p.platform_status(plat)
        expected = p.expected_tiers
        crashed = [t for t in expected if p.state_at(plat, t) == "crashed"]
        # Missing-on-platform only counts as a gap if THIS platform has
        # any data — otherwise the patch is simply "the other platform
        # only" and shouldn't read as a per-tier gap.
        present = [t for t in expected if p.state_at(plat, t) != "missing"]
        if not present:
            continue
        missing = [t for t in expected if p.state_at(plat, t) == "missing"]
        pieces: list[str] = []
        if st == "full":
            pieces.append("full")
        else:
            if crashed:
                pieces.append("crashed " + ",".join(crashed))
            if missing:
                if len(missing) >= 3:
                    pieces.append(f"missing {len(missing)} tiers")
                else:
                    pieces.append("missing " + ",".join(missing))
        bits.append(f"{PLATFORM_LABEL[plat]}={'/'.join(pieces) or 'full'}")
    return "; ".join(bits) if bits else "speedups_finalized.tsv empty"


def render_table(patches: list[PatchAudit], framework: Path) -> str:
    rows = [p for p in patches if not p.is_core]
    core = [p for p in patches if p.is_core]

    out: list[str] = []
    out.append(f"Framework: {framework}")
    out.append(f"Scanned {len(patches)} patches "
               f"({sum(1 for p in patches if p.language == 'R')} R + "
               f"{sum(1 for p in patches if p.language == 'Py')} Py).")
    out.append("")

    if rows:
        name_w = max(8, max(len(p.name) for p in rows))
        task_w = max(8, max(len(p.lifted_from_task or "—") for p in rows))
        tier_header = " ".join(f"{TIER_LETTERS[t]:<2}" for t in TIERS)
        out.append(
            f"  {'patch':<{name_w}}  {'lang':<4}  {'task':<{task_w}}  "
            f"{'pkg':<3}  {tier_header}   {'fold(W/M)':<13}  "
            f"{'conc(W/M)':<13}  plat  note"
        )
        out.append("  " + "-" * (name_w + task_w + 6 + 4 + 4 + 16 + 16 + 16 + 8))
        for p in rows:
            pkg = "ok" if p.speedups_exists else "x"
            task = p.lifted_from_task or "—"
            tier_cells = " ".join(_dual_cell(p, t, _TIER_GLYPH_PLAIN) for t in TIERS)
            fw = p.median_fold_by_platform.get("win")
            fm = p.median_fold_by_platform.get("mac")
            cw = p.median_conc_by_platform.get("win")
            cm = p.median_conc_by_platform.get("mac")
            fold_cell = f"{_fmt_fold(fw)}/{_fmt_fold(fm)}"
            conc_cell = f"{_fmt_conc(cw)}/{_fmt_conc(cm)}"
            out.append(
                f"  {p.name:<{name_w}}  {p.language:<4}  {task:<{task_w}}  "
                f"{pkg:<3}  {tier_cells}   {fold_cell:<13}  "
                f"{conc_cell:<13}  {_platform_tag(p):<4}  {_short_note(p)}"
            )
        out.append("")

    if core:
        out.append("Core panels (panel A/B; tracked separately, not part of attest coverage):")
        for p in core:
            task = p.lifted_from_task or "—"
            out.append(f"  - {p.name} ({p.language})  task={task}")
        out.append("")

    out.append("Legend:  o = tier ran     x = verify-worker crashed     . = tier not run")
    out.append("Tier cells: 2 chars per tier — first = Windows, second = macOS.")
    out.append("Tiers:   S = small  M = medium  L = large  o = ood_large  O = ood_xlarge  1/2/3 = ood_large1/2/3")
    out.append("Platforms: W = Windows, M = macOS, ? = unknown. Pre-tagging legacy rows are macOS.")
    out.append("")

    out.extend(_summary_lines(rows))
    out.extend(_drift_lines(rows))
    return "\n".join(out)


def _summary_lines(rows: list[PatchAudit]) -> list[str]:
    out: list[str] = []
    n = len(rows)
    n_attest = sum(1 for p in rows if p.speedups_exists)
    n_full = sum(1 for p in rows if p.status == "full")
    n_partial = sum(1 for p in rows if p.status == "partial")
    n_no_attest = sum(1 for p in rows if p.status == "no_attest")
    n_no_data = sum(1 for p in rows if p.status == "no_data")

    n_win = sum(1 for p in rows if "win" in p.platforms_with_data())
    n_mac = sum(1 for p in rows if "mac" in p.platforms_with_data())
    n_both = sum(1 for p in rows if {"win", "mac"}.issubset(p.platforms_with_data()))
    n_win_only = sum(
        1 for p in rows
        if p.platforms_with_data() and "mac" not in p.platforms_with_data()
        and "win" in p.platforms_with_data()
    )
    n_mac_only = sum(
        1 for p in rows
        if p.platforms_with_data() and "win" not in p.platforms_with_data()
        and "mac" in p.platforms_with_data()
    )
    n_full_win = sum(1 for p in rows if p.platform_status("win") == "full")
    n_full_mac = sum(1 for p in rows if p.platform_status("mac") == "full")

    out.append("Summary")
    out.append(f"  patches:           {n}")
    out.append(f"  attest run:        {n_attest} ({100 * n_attest // n if n else 0}%)")
    out.append(f"  full coverage (any platform):  {n_full}")
    out.append(f"  partial:           {n_partial}")
    out.append(f"  finalized TSV only header: {n_no_data}")
    out.append(f"  no finalized TSV:  {n_no_attest}")
    out.append(f"  Windows data:      {n_win} ({n_win_only} win-only)")
    out.append(f"  macOS data:        {n_mac} ({n_mac_only} mac-only)")
    out.append(f"  both platforms:    {n_both}")
    out.append(f"  full coverage on Windows: {n_full_win}")
    out.append(f"  full coverage on macOS:   {n_full_mac}")

    # Per-platform tier gaps (only count tiers in each patch's expected set).
    for plat in ("win", "mac"):
        missing_counts: dict[str, int] = {}
        crashed_counts: dict[str, int] = {}
        for p in rows:
            if plat not in p.platforms_with_data():
                continue
            for t in p.expected_tiers:
                state = p.state_at(plat, t)
                if state == "missing":
                    missing_counts[t] = missing_counts.get(t, 0) + 1
                elif state == "crashed":
                    crashed_counts[t] = crashed_counts.get(t, 0) + 1
        miss_break = ", ".join(f"{t}={missing_counts[t]}" for t in TIERS if missing_counts.get(t))
        crash_break = ", ".join(f"{t}={crashed_counts[t]}" for t in TIERS if crashed_counts.get(t))
        if miss_break:
            out.append(f"  {PLATFORM_LABEL[plat]} tier gaps:    {miss_break}")
        if crash_break:
            out.append(f"  {PLATFORM_LABEL[plat]} tier crashes: {crash_break}")

    if n_no_attest:
        names = ", ".join(p.name for p in rows if p.status == "no_attest")
        out.append(f"  awaiting attest:   {names}")
    return out


def _drift_lines(rows: list[PatchAudit]) -> list[str]:
    """Collect and render drift alerts across all patches."""
    all_alerts: list[DriftAlert] = []
    for p in rows:
        all_alerts.extend(p.drift_alerts())
    if not all_alerts:
        return []

    out: list[str] = ["", "Drift detection  (latest vs previous batch, same platform+tier)"]
    speed_up = [a for a in all_alerts if a.kind == "speedup_up"]
    speed_down = [a for a in all_alerts if a.kind == "speedup_down"]
    conc_drop = [a for a in all_alerts if a.kind == "concordance_drop"]

    if speed_up:
        out.append(f"  speedup increased (>{DRIFT_SPEED_THRESHOLD:.0%}):")
        for a in speed_up:
            out.append(f"    {a.patch} [{PLATFORM_LABEL.get(a.platform, '?')}] {a.tier}: {a.description}")
    if speed_down:
        out.append(f"  speedup decreased (>{DRIFT_SPEED_THRESHOLD:.0%}):")
        for a in speed_down:
            out.append(f"    {a.patch} [{PLATFORM_LABEL.get(a.platform, '?')}] {a.tier}: {a.description}")
    if conc_drop:
        out.append(f"  concordance dropped (>{DRIFT_CONC_THRESHOLD}):")
        for a in conc_drop:
            out.append(f"    {a.patch} [{PLATFORM_LABEL.get(a.platform, '?')}] {a.tier}: {a.description}")

    if speed_down or conc_drop:
        out.append("")
        out.append("  ^ these patches may need re-investigation or re-attest")
    return out


def render_markdown(patches: list[PatchAudit], framework: Path) -> str:
    rows = [p for p in patches if not p.is_core]
    core = [p for p in patches if p.is_core]

    lines: list[str] = []
    lines.append("# autozyme attest coverage")
    lines.append("")
    lines.append(f"_Generated by `zyme scan --attest`._  Framework: `{framework}`")
    lines.append("")
    lines.append(
        f"Scanned **{len(patches)}** patches "
        f"({sum(1 for p in patches if p.language == 'R')} R + "
        f"{sum(1 for p in patches if p.language == 'Py')} Py)."
    )
    lines.append("")
    lines.append(
        "Each tier cell is **two glyphs**: first = Windows, second = macOS. "
        "`●` = tier ran, `✗` = verify-worker crashed, `·` = tier not run. "
        "Legacy rows (empty `system_os`) are bucketed as macOS — the column "
        "was only auto-populated once the Windows port began, so pre-tagging "
        "data is all from the original Mac runs."
    )
    lines.append("")

    if rows:
        lines.append("## Per-patch coverage")
        lines.append("")
        tier_hdr = " | ".join(TIER_LETTERS[t] for t in TIERS)
        lines.append(f"| Patch | Lang | Task | Pkg | {tier_hdr} | "
                     "Fold W | Fold M | Conc W | Conc M | Plat | Note |")
        sep_count = 4 + len(TIERS) + 6
        lines.append("|" + "|".join(["---"] * sep_count) + "|")
        for p in rows:
            cells = [
                p.name, p.language, p.lifted_from_task or "—",
                "ok" if p.speedups_exists else "✗",
            ]
            for t in TIERS:
                cells.append(_dual_cell(p, t, _TIER_GLYPH_MD))
            cells.append(_fmt_fold(p.median_fold_by_platform.get("win")))
            cells.append(_fmt_fold(p.median_fold_by_platform.get("mac")))
            cells.append(_fmt_conc(p.median_conc_by_platform.get("win")))
            cells.append(_fmt_conc(p.median_conc_by_platform.get("mac")))
            cells.append(_platform_tag(p))
            cells.append(_short_note(p))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Summary
    n = len(rows)
    n_attest = sum(1 for p in rows if p.speedups_exists)
    n_full = sum(1 for p in rows if p.status == "full")
    n_partial = sum(1 for p in rows if p.status == "partial")
    n_no_attest = sum(1 for p in rows if p.status == "no_attest")
    n_win = sum(1 for p in rows if "win" in p.platforms_with_data())
    n_mac = sum(1 for p in rows if "mac" in p.platforms_with_data())
    n_both = sum(1 for p in rows if {"win", "mac"}.issubset(p.platforms_with_data()))
    n_full_win = sum(1 for p in rows if p.platform_status("win") == "full")
    n_full_mac = sum(1 for p in rows if p.platform_status("mac") == "full")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- patches: **{n}**")
    lines.append(f"- attest run: **{n_attest}** ({100 * n_attest // n if n else 0}%)")
    lines.append(f"- full coverage (any platform): **{n_full}**")
    lines.append(f"- partial: **{n_partial}**")
    lines.append(f"- no finalized TSV: **{n_no_attest}**")
    lines.append("")
    lines.append("### Platform breakdown")
    lines.append("")
    lines.append(f"- Windows data: **{n_win}** patches")
    lines.append(f"- macOS data: **{n_mac}** patches")
    lines.append(f"- both platforms: **{n_both}** patches")
    lines.append(f"- full coverage on Windows: **{n_full_win}**")
    lines.append(f"- full coverage on macOS: **{n_full_mac}**")
    lines.append("")

    # Per-platform tier gaps (only expected tiers per patch).
    has_any_plat_gap = False
    plat_gap_lines: list[str] = []
    for plat in ("win", "mac"):
        plat_rows = [p for p in rows if plat in p.platforms_with_data()]
        if not plat_rows:
            continue
        missing_lists: dict[str, list[str]] = {}
        crashed_lists: dict[str, list[str]] = {}
        for p in plat_rows:
            for t in p.expected_tiers:
                state = p.state_at(plat, t)
                if state == "missing":
                    missing_lists.setdefault(t, []).append(p.name)
                elif state == "crashed":
                    crashed_lists.setdefault(t, []).append(p.name)
        plat_has_gap = any(missing_lists.values()) or any(crashed_lists.values())
        if not plat_has_gap:
            continue
        has_any_plat_gap = True
        plat_name = "Windows" if plat == "win" else "macOS"
        plat_gap_lines.append(f"### Tier gaps — {plat_name}")
        plat_gap_lines.append("")
        for t in TIERS:
            ms = missing_lists.get(t, [])
            cs = crashed_lists.get(t, [])
            if not ms and not cs:
                continue
            bits: list[str] = []
            if ms:
                bits.append(f"missing in: {', '.join(ms)}")
            if cs:
                bits.append(f"crashed in: {', '.join(cs)}")
            plat_gap_lines.append(f"- **{t}** — " + "; ".join(bits))
        plat_gap_lines.append("")
    if has_any_plat_gap:
        lines.extend(plat_gap_lines)

    # Cross-platform comparison: patches with data on both platforms.
    cross = [p for p in rows if {"win", "mac"}.issubset(p.platforms_with_data())]
    if cross:
        lines.append("## Cross-platform comparison")
        lines.append("")
        lines.append("Patches with data on both platforms — Win vs Mac median fold per tier.")
        lines.append("")
        lines.append("| Patch | Lang | Tier | Win fold | Mac fold | Win/Mac | Note |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in cross:
            for t in TIERS:
                wst = p.tier_status.get(("win", t))
                mst = p.tier_status.get(("mac", t))
                if not wst or not mst:
                    continue
                if wst.state != "ok" or mst.state != "ok":
                    continue
                if wst.speedup_x is None or mst.speedup_x is None:
                    continue
                ratio = wst.speedup_x / mst.speedup_x if mst.speedup_x else None
                ratio_str = f"{ratio:.2f}" if ratio is not None else "—"
                note_bits: list[str] = []
                if ratio is not None and (ratio >= 1.5 or ratio <= 0.67):
                    note_bits.append("⚠ platform divergence >1.5x")
                lines.append(
                    f"| {p.name} | {p.language} | {t} | "
                    f"{_fmt_fold(wst.speedup_x)} | {_fmt_fold(mst.speedup_x)} | "
                    f"{ratio_str} | {' '.join(note_bits)} |"
                )
        lines.append("")

    if n_no_attest:
        lines.append("## Awaiting first attest")
        lines.append("")
        for p in rows:
            if p.status == "no_attest":
                task = f" (task `{p.lifted_from_task}`)" if p.lifted_from_task else ""
                lines.append(f"- **{p.name}** ({p.language}){task}")
        lines.append("")

    if core:
        lines.append("## Core panels (tracked separately)")
        lines.append("")
        lines.append("These patches back panel A/B and are excluded from headline coverage.")
        lines.append("")
        for p in core:
            task = f" — task `{p.lifted_from_task}`" if p.lifted_from_task else ""
            lines.append(f"- **{p.name}** ({p.language}){task}")
        lines.append("")

    # Drift detection
    all_alerts: list[DriftAlert] = []
    for p in rows:
        all_alerts.extend(p.drift_alerts())
    if all_alerts:
        lines.append("## Drift detection")
        lines.append("")
        lines.append("Significant changes between latest and previous batch "
                     f"(speedup >{DRIFT_SPEED_THRESHOLD:.0%}, concordance "
                     f"drop >{DRIFT_CONC_THRESHOLD}).")
        lines.append("")
        lines.append("| Patch | Plat | Tier | Type | Previous | Current | Delta |")
        lines.append("|---|---|---|---|---|---|---|")
        for a in all_alerts:
            plat_lbl = PLATFORM_LABEL.get(a.platform, "?")
            if a.kind in ("speedup_up", "speedup_down"):
                pv, cv = _fmt_fold(a.prev_val), _fmt_fold(a.curr_val)
                delta = f"{(a.curr_val / a.prev_val - 1) * 100:+.0f}%"
                kind = "speedup " + ("up" if a.kind == "speedup_up" else "**DOWN**")
            else:
                pv, cv = f"{a.prev_val:.4f}", f"{a.curr_val:.4f}"
                delta = f"{a.curr_val - a.prev_val:+.4f}"
                kind = "concordance **DROP**"
            lines.append(f"| {a.patch} | {plat_lbl} | {a.tier} | {kind} | {pv} | {cv} | {delta} |")
        lines.append("")

    return "\n".join(lines)


def render_json_records(patches: list[PatchAudit]) -> list[dict]:
    out: list[dict] = []
    for p in patches:
        tier_records: dict[str, dict] = {}
        for t in TIERS:
            per_plat: dict[str, dict] = {}
            for plat in PLATFORMS:
                st = p.tier_status.get((plat, t))
                if st is None:
                    continue
                per_plat[plat] = {
                    "state": st.state,
                    "speedup_x": st.speedup_x,
                    "concordance": st.concordance,
                    "note": st.note,
                }
            tier_records[t] = per_plat
        out.append({
            "patch": p.name,
            "language": p.language,
            "patch_path": str(p.patch_path),
            "speedups_path": str(p.speedups_path),
            "speedups_exists": p.speedups_exists,
            "lifted_from_task": p.lifted_from_task,
            "is_core_panel": p.is_core,
            "status": p.status,
            "platforms_with_data": p.platforms_with_data(),
            "platform_status": {
                plat: p.platform_status(plat) for plat in PLATFORMS
            },
            "tiers": tier_records,
            "median_fold_by_platform": dict(p.median_fold_by_platform),
            "median_concordance_by_platform": dict(p.median_conc_by_platform),
        })
    return out
