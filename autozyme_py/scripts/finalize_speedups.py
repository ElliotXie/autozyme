#!/usr/bin/env python3
"""Aggregate raw speedups.tsv -> speedups_finalized.tsv per patch.

Rows: tier x threads x platform x variant
Cells: comma-separated per-rep values + mean/median

Usage:
  python scripts/finalize_speedups.py            # all patches
  python scripts/finalize_speedups.py mdanalysis # one patch
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import re
import sys
from pathlib import Path
from statistics import median as _stat_median

import pandas as pd

PATCHES_DIR = Path("src/autozyme")

# Packages where blank-platform rows were confirmed by timing comparison
INFERRED_MAC_PACKAGES: set[str] = {
    "cell2location", "lifelines", "mdanalysis", "obspy", "sarsen", "scvelo",
}
INFERRED_WIN_PACKAGES: set[str] = {"prody", "statsmodels"}

# Single-threaded patch sets come from the SHARED thread rubric
# (scripts/thread_rubric.yaml) so the scan and both finalizers can never drift
# apart. A single-threaded patch's per-thread reps collapse into one "any"
# summary row (else 3 per-thread speedups disagree by GIL/serial noise, e.g.
# cellphonedb win 8.4/6.5/7.2 with T8<T4). Platform-keyed: `both` collapses
# everywhere; `win`/`mac` collapse only that platform (fork-based patches are
# serial on Windows but parallel on macOS).
def _load_reduced() -> dict:
    """{task: {'mac': set(variants), 'win': set(variants)}} from the shared task
    rubric (paper/rubric/task_threading.yaml): the (platform, variant) cells to
    reduce to t1. schedule==single reduces BOTH variants on BOTH platforms;
    reduce_t1 lists per-platform variants. The finalizer then pools the per-thread
    reps within +/-10% of t1 onto t1 (more reps, balanced double-sided noise) and
    DROPS threads >10% off (fork/thread overhead). See task_threading.yaml."""
    out: dict = {}
    try:
        import yaml
        p = (Path(__file__).resolve().parents[2] / "paper" / "rubric"
             / "task_threading.yaml")
        tasks = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("tasks") or {}
        for name, e in tasks.items():
            e = e or {}
            m = {"mac": set(), "win": set()}
            if e.get("schedule") == "single":
                m = {"mac": {"baseline", "patched"}, "win": {"baseline", "patched"}}
            for plat, vs in (e.get("reduce_t1") or {}).items():
                m.setdefault(plat, set()).update(str(x) for x in (vs or []))
            out[name] = m
    except Exception as e:  # noqa: BLE001
        print(f"[finalize] WARN: task_threading.yaml unreadable ({e}); "
              "no reduction applied", file=sys.stderr)
    return out


_REDUCED = _load_reduced()


def _load_flat_label() -> dict:
    """{task: {'mac': set(variants), 'win': set(variants)}} from the rubric's
    thread_scaling_flat section: cells whose thread_scaling LABEL is forced to
    "flat" despite being kept per-thread (empirically thread-flat). Label-only:
    these rows are NOT reduced/dropped, only relabelled."""
    out: dict = {}
    try:
        import yaml
        p = (Path(__file__).resolve().parents[2] / "paper" / "rubric"
             / "task_threading.yaml")
        fl = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("thread_scaling_flat") or {}
        for name, e in fl.items():
            m = {"mac": set(), "win": set()}
            for plat, vs in (e or {}).items():
                m.setdefault(plat, set()).update(str(x) for x in (vs or []))
            out[name] = m
    except Exception as e:  # noqa: BLE001
        print(f"[finalize] WARN: thread_scaling_flat unreadable ({e})", file=sys.stderr)
    return out


_FLAT_LABEL = _load_flat_label()
_TOL = 0.10  # +/-10%: pool a t>1 thread onto t1 if its median is within this band


def fmt_num(series: pd.Series, digits: int = 3) -> str:
    vals = series.dropna().tolist()
    if not vals:
        return ""
    return ", ".join(f"{v:.{digits}f}" for v in vals)


def aggregate_metrics_json(values: pd.Series) -> str:
    """Collect per-rep metrics_json dicts and emit median per numeric key."""
    by_key: dict[str, list[float]] = {}
    for raw in values.dropna():
        s = str(raw).strip()
        if not s or s.upper() == "NA":
            continue
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        for k, v in parsed.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                by_key.setdefault(k, []).append(f)
    if not by_key:
        return ""
    medians = {k: round(_stat_median(vs), 6) for k, vs in sorted(by_key.items())}
    return json.dumps(medians, separators=(",", ":"))


def get_package_version(patch_dir: Path) -> str:
    """Resolve upstream package version (e.g. 'scanpy 1.11.5').
    Priority: manifest.yml in patch_dir > parent manifest.yml (scanpy subtasks) > tested_against in source."""
    def read_target(mf: Path) -> str | None:
        if not mf.exists():
            return None
        for line in mf.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"\s*compatibility_target:\s*(.+)", line)
            if m:
                return m.group(1).strip()
        return None

    v = read_target(patch_dir / "manifest.yml")
    if v:
        return v
    if patch_dir.name.startswith("scanpy_"):
        v = read_target(patch_dir.parent / "scanpy" / "manifest.yml")
        if v:
            return v
    for src in patch_dir.glob("*.py"):
        try:
            txt = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = re.search(r'tested_against\s*=\s*"([^"]+)"', txt)
        if m:
            return m.group(1)
        m = re.search(r"tested_against\s*=\s*f\"([^{]*)\{(\w+)\.__version__\}", txt)
        if m:
            prefix, mod = m.groups()
            try:
                imported = importlib.import_module(mod)
                v = getattr(imported, "__version__", None)
                if v is None:
                    v = importlib.metadata.version(mod)
                return f"{prefix.strip()} {v}"
            except Exception:
                return f"{prefix.strip()} <{mod}.__version__>"
    return ""


def classify_platform(os: str) -> str:
    if not isinstance(os, str) or not os:
        return "unknown"
    low = os.lower()
    if any(k in low for k in ("macos", "darwin", "apple")):
        return "macOS"
    if "win" in low:
        return "Windows"
    if "linux" in low:
        return "Linux"
    return "unknown"


def _read_raw_shards(patch_dir: Path):
    """Read the raw speedup history for a patch.

    Per-platform split: each machine writes its own ``speedups.<plat>.tsv``
    (mac/win/other) so cross-machine git merges stay disjoint. We read and
    concatenate all shards here. Backward-compatible: if no shard exists we
    fall back to the legacy combined ``speedups.tsv``.
    """
    shards = [patch_dir / f"speedups.{p}.tsv"
              for p in ("mac", "win", "other", "linux")]
    shards = [s for s in shards if s.exists()]
    if not shards:
        legacy = patch_dir / "speedups.tsv"
        shards = [legacy] if legacy.exists() else []
    if not shards:
        return None
    frames = [pd.read_csv(s, sep="\t", na_values=["", "NA"], keep_default_na=False)
              for s in shards]
    return pd.concat(frames, ignore_index=True) if frames else None


def finalize_one(patch_dir: Path) -> None:
    raw = _read_raw_shards(patch_dir)
    if raw is None:
        return

    raw = raw.where(raw.notna(), pd.NA)  # normalize all NA
    # convert empty strings to NA for relevant cols
    for col in raw.columns:
        if raw[col].dtype == object:
            raw[col] = raw[col].replace("", pd.NA)

    if raw.empty:
        return

    # drop rows with no variant; keep explicit markers (note ~ "OOM"/"NA_applicable") even with NA sec
    raw = raw.dropna(subset=["variant"]).copy()
    note_str = raw["note"].fillna("").astype(str)
    is_marker_note = note_str.str.contains("OOM|NA_applicable", case=False, na=False)
    raw = raw[raw["sec"].notna() | is_marker_note].copy()
    if raw.empty:
        print(f"[{patch_dir.name}] no usable rows (all error logs)")
        return

    # dedupe by full measurement signature (R side had this bug; Py side hasn't, but stay safe)
    n_before = len(raw)
    raw = raw.drop_duplicates(
        subset=[
            "timestamp", "patch_name", "tier", "rep_idx", "variant",
            "sec", "peak_mb", "system_os", "system_cpu", "system_threads",
        ],
        keep="first",
    )
    n_dupe = n_before - len(raw)

    pkg_name = raw["patch_name"].iloc[0]
    def _thread_lbl(x):
        if pd.isna(x):
            return "unknown"
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return str(x)
    raw["threads_lbl"] = raw["system_threads"].apply(_thread_lbl)
    # (Per-rubric thread reduction happens after platform_lbl / dataset_lbl are
    # finalized -- see the conditional-pool block before the baseline reference.)
    _note = raw["note"].fillna("").astype(str)
    raw["is_oom"] = _note.str.contains("OOM", case=False, na=False) & raw["sec"].isna()
    raw["is_na"] = _note.str.contains("NA_applicable", case=False, na=False) & raw["sec"].isna()
    if "dataset" in raw.columns:
        raw["dataset_lbl"] = raw["dataset"].fillna("—").replace("", "—").astype(str)
    else:
        raw["dataset_lbl"] = "—"
    raw["platform_raw"] = raw["system_os"].fillna("").map(classify_platform)
    # The blank-OS rows of these packages were timing-confirmed to be macOS /
    # Windows, so they are now formalized as the plain platform label (no
    # "(inferred)" qualifier) — they merge with any genuine same-platform rows.
    if pkg_name in INFERRED_MAC_PACKAGES:
        inferred_lbl = "macOS"
    elif pkg_name in INFERRED_WIN_PACKAGES:
        inferred_lbl = "Windows"
    else:
        inferred_lbl = None
    if inferred_lbl is not None:
        raw["platform_lbl"] = raw["platform_raw"].where(
            raw["platform_raw"] != "unknown", inferred_lbl
        )
    else:
        raw["platform_lbl"] = raw["platform_raw"]

    # Mac/inferred-Mac rows in scanpy_* tasks with blank dataset → pbmc68k inferred
    apply_mac_pbmc68k = pkg_name == "scanpy" and "dataset" in raw.columns
    if apply_mac_pbmc68k:
        is_mac = raw["platform_lbl"].isin(["macOS", "macOS (inferred)"])
        is_blank_ds = raw["dataset_lbl"] == "—"
        raw.loc[is_mac & is_blank_ds, "dataset_lbl"] = "pbmc68k (inferred)"
    raw["pass_bool"] = raw["pass"].astype(str).str.lower().isin(("true", "1"))

    # ------------------------------------------------------------------
    # Backfill dataset_lbl on legacy "—" rows (recorded before `dataset`
    # was populated into speedups.tsv). For each (tier, threads, platform)
    # cell where there's a real dataset name AND there are "—" rows, copy
    # the real name into the "—" rows so they merge into the same group_by
    # below — recovering the legacy reps.
    #
    # Consistency guard, all-or-nothing per (tier, threads, platform):
    # rescue only when BOTH baseline AND patched legacy medians are within
    # ±RESCUE_TOLERANCE_PCT of the established medians. A per-variant
    # decision would let baseline get pooled when patched diverges
    # (different code path between batches), producing a speedup_x ratio
    # that mixes new patched against old+new baseline — a misleading
    # headline number. Requiring both variants to pass keeps speedup_x
    # internally consistent. The post-grouping filter below then drops
    # the diverged "—" rows exactly as before.
    RESCUE_TOLERANCE_PCT = 0.25
    not_skip = ~(raw["is_oom"] | raw["is_na"])
    cell_check = (
        raw[not_skip]
        .groupby(["tier", "threads_lbl", "platform_lbl", "variant"], dropna=False)
        .apply(lambda g: pd.Series({
            "has_em": (g["dataset_lbl"] == "—").any(),
            "has_real": (g["dataset_lbl"] != "—").any(),
            "em_median": g.loc[g["dataset_lbl"] == "—", "sec"].median(),
            "real_median": g.loc[g["dataset_lbl"] != "—", "sec"].median(),
            "real_name": (g.loc[g["dataset_lbl"] != "—", "dataset_lbl"].iloc[0]
                          if (g["dataset_lbl"] != "—").any() else None),
            "n_em": int((g["dataset_lbl"] == "—").sum()),
        }), include_groups=False)
        .reset_index()
    )
    cell_check = cell_check[cell_check["has_em"] & cell_check["has_real"]].copy()
    if len(cell_check) > 0:
        cell_check["rel_diff"] = (
            (cell_check["em_median"] - cell_check["real_median"]).abs()
            / cell_check["real_median"]
        )
        cell_check["within"] = cell_check["rel_diff"] <= RESCUE_TOLERANCE_PCT
        # All-or-nothing verdict per (tier, threads, platform)
        verdict = (
            cell_check
            .groupby(["tier", "threads_lbl", "platform_lbl"], dropna=False)
            .agg(all_within=("within", "all"),
                 worst_diff=("rel_diff", "max"),
                 n_em_total=("n_em", "sum"),
                 real_name=("real_name", "first"))
            .reset_index()
        )
        rescued_log = []
        skipped_log = []
        for _, row in verdict.iterrows():
            cell_label = (
                f"{row['tier']}/{row['threads_lbl']}/{row['platform_lbl']}"
            )
            if bool(row["all_within"]):
                m = (
                    (raw["tier"] == row["tier"])
                    & (raw["threads_lbl"] == row["threads_lbl"])
                    & (raw["platform_lbl"] == row["platform_lbl"])
                    & (raw["dataset_lbl"] == "—")
                )
                raw.loc[m, "dataset_lbl"] = row["real_name"]
                rescued_log.append(
                    f"{cell_label}: {int(row['n_em_total'])} legacy reps "
                    f"(worst variant {row['worst_diff']*100:+.0f}% within "
                    f"{RESCUE_TOLERANCE_PCT*100:.0f}% tol)"
                )
            else:
                skipped_log.append(
                    f"{cell_label}: {int(row['n_em_total'])} legacy reps "
                    f"skipped (worst variant {row['worst_diff']*100:+.0f}% > "
                    f"{RESCUE_TOLERANCE_PCT*100:.0f}% tol)"
                )
        if rescued_log:
            print(f"[{patch_dir.name}] rescued {len(rescued_log)} legacy-dataset cells:")
            for e in rescued_log:
                print(f"  rescued: {e}")
        if skipped_log:
            print(f"[{patch_dir.name}] skipped {len(skipped_log)} legacy-dataset cells (out of tolerance):")
            for e in skipped_log:
                print(f"  skipped: {e}")

    # === Per-rubric thread reduction (conditional pool) ===
    # For each (platform, variant) the rubric marks thread-inapplicable, collapse
    # its per-thread reps onto t1: pool the reps of threads whose median is within
    # +/-_TOL of t1 (statistically flat -> more reps; the +/- noise is double-sided
    # and cancels), and DROP threads >_TOL off (fork / thread overhead). Reports
    # the operating point (t1) with maximum robustness.
    red_map = _REDUCED.get(patch_dir.name, {})
    flat_map = _FLAT_LABEL.get(patch_dir.name, {})
    # Per-row threading verdict for Supplementary Data 2: "flat" if this
    # (platform, variant) is either rubric-reduced (its t>1 reps pool onto t1)
    # OR in thread_scaling_flat (kept per-thread but empirically thread-flat),
    # else "scale". Set BEFORE the pool so pooled/dropped rows keep the flag.
    # The flat_map branch is LABEL-ONLY -- those rows are not reduced/dropped.
    raw["thread_scaling"] = "scale"
    for _pk, _pl in (("mac", "macOS"), ("win", "Windows")):
        for _v in set(red_map.get(_pk, ())) | set(flat_map.get(_pk, ())):
            raw.loc[(raw["platform_lbl"] == _pl) & (raw["variant"] == _v),
                    "thread_scaling"] = "flat"
    if red_map:
        drop_idx: list = []
        for plat_key, plat_lbl in (("mac", "macOS"), ("win", "Windows")):
            for var in red_map.get(plat_key, ()):
                sel = (raw["platform_lbl"] == plat_lbl) & (raw["variant"] == var)
                for _, g in raw[sel].groupby(["tier", "dataset_lbl"], dropna=False):
                    t1 = g.loc[g["threads_lbl"] == "1", "sec"].dropna()
                    if len(t1) == 0:
                        continue
                    t1m = t1.median()
                    if not (t1m > 0):
                        continue
                    for _thr, gt in g[g["threads_lbl"] != "1"].groupby("threads_lbl"):
                        tm = gt["sec"].dropna().median()
                        if pd.notna(tm) and abs(tm / t1m - 1) <= _TOL:
                            raw.loc[gt.index, "threads_lbl"] = "1"     # pool onto t1
                        else:
                            drop_idx.extend(gt.index.tolist())         # overhead -> drop
        if drop_idx:
            raw = raw.drop(index=drop_idx)

    # baseline reference: mean sec per (tier, threads, platform, dataset)
    base = (
        raw[raw["variant"] == "baseline"]
        .groupby(["tier", "threads_lbl", "platform_lbl", "dataset_lbl"], dropna=False)["sec"]
        .mean()
        .rename("baseline_sec_mean")
        .reset_index()
    )
    raw = raw.merge(base, on=["tier", "threads_lbl", "platform_lbl", "dataset_lbl"], how="left")
    # Wildcard pairing: a reduced (serial) baseline lives only at threads_lbl="1",
    # so a per-thread patched kept because it scales (e.g. vegan on Windows) has no
    # same-thread baseline. Fall back to the t1 baseline (a serial baseline is the
    # same at every thread). Per-thread baselines keep DIRECT pairing -- their
    # same-thread value is present, so this fallback never fires for them.
    base_t1 = (
        raw[(raw["variant"] == "baseline") & (raw["threads_lbl"] == "1")]
        .groupby(["tier", "platform_lbl", "dataset_lbl"], dropna=False)["sec"]
        .mean()
        .rename("baseline_t1_mean")
        .reset_index()
    )
    raw = raw.merge(base_t1, on=["tier", "platform_lbl", "dataset_lbl"], how="left")
    # FLAT baselines (thread_scaling=="flat") measure the same serial quantity at
    # every thread, so their reps are pooled ACROSS threads and the speedup uses the
    # robust pooled MEDIAN (why a flat baseline may show one rep per thread -- they
    # aggregate; also de-noises single-thread outliers). SCALE baselines keep the
    # same-thread mean pairing.
    bgrp = raw[raw["variant"] == "baseline"].groupby(
        ["tier", "platform_lbl", "dataset_lbl"], dropna=False)
    base_pooled = pd.DataFrame({
        "baseline_pooled_med": bgrp["sec"].median(),
        "baseline_is_flat": bgrp["thread_scaling"].apply(lambda s: (s == "flat").any()),
    }).reset_index()
    raw = raw.merge(base_pooled, on=["tier", "platform_lbl", "dataset_lbl"], how="left")
    same_thread = raw["baseline_sec_mean"].fillna(raw["baseline_t1_mean"])
    use_pooled = (raw["baseline_is_flat"] == True) & raw["baseline_pooled_med"].notna()  # noqa: E712
    raw["eff_baseline"] = raw["baseline_pooled_med"].where(use_pooled, same_thread)
    raw["speedup_x_calc"] = pd.NA
    pat = raw["variant"] == "patched"
    valid_base = raw["eff_baseline"].notna() & (raw["eff_baseline"] > 0)
    raw.loc[pat & valid_base, "speedup_x_calc"] = (
        raw.loc[pat & valid_base, "eff_baseline"] / raw.loc[pat & valid_base, "sec"]
    )

    def agg(g: pd.DataFrame) -> pd.Series:
        oom_mask = g["is_oom"]
        na_mask = g["is_na"]
        skip_mask = oom_mask | na_mask
        all_na = na_mask.all()
        all_oom = oom_mask.all()
        any_skip = skip_mask.any()
        ok = g[~skip_mask]
        if all_na:
            status = "N/A"
        elif all_oom:
            status = "OOM"
        elif any_skip:
            status = "partial"
        else:
            status = "ok"
        return pd.Series({
            "status": status,
            "n_reps": len(ok),
            "sec_reps": fmt_num(ok["sec"], 3),
            "sec_mean": round(ok["sec"].mean(), 3) if ok["sec"].notna().any() else pd.NA,
            "sec_median": round(ok["sec"].median(), 3) if ok["sec"].notna().any() else pd.NA,
            "mem_reps": fmt_num(ok["peak_mb"], 1),
            "mem_mean": round(ok["peak_mb"].mean(), 3) if ok["peak_mb"].notna().any() else pd.NA,
            "mem_median": round(ok["peak_mb"].median(), 3) if ok["peak_mb"].notna().any() else pd.NA,
            "speedup_x_reps": fmt_num(ok["speedup_x_calc"], 3),
            "speedup_x_mean": round(ok["speedup_x_calc"].mean(), 3) if ok["speedup_x_calc"].notna().any() else pd.NA,
            "speedup_x_median": round(ok["speedup_x_calc"].median(), 3) if ok["speedup_x_calc"].notna().any() else pd.NA,
            "pass_rate": round(ok["pass_bool"].mean(), 3) if len(ok) > 0 else pd.NA,
            "metrics_json_median": (
                aggregate_metrics_json(ok["metrics_json"]) if "metrics_json" in g.columns else ""
            ),
            "fw_versions": ", ".join(sorted(g["framework_version"].dropna().unique())),
            "ts_first": g["timestamp"].min(),
            "ts_last": g["timestamp"].max(),
            "thread_scaling": g["thread_scaling"].iloc[0],
        })

    fallback_pkg_version = get_package_version(patch_dir)
    placeholder_re = re.compile(r"<\w+\.__version__>")
    if "package_version" in raw.columns:
        pv = raw["package_version"].fillna("").astype(str)
        real_versions = [
            v for v in pv.unique() if v and not placeholder_re.search(v)
        ]
        canonical = real_versions[0] if real_versions else fallback_pkg_version
        if canonical and not placeholder_re.search(canonical):
            is_bad = (pv == "") | pv.str.contains(placeholder_re, regex=True)
            pv = pv.where(~is_bad, canonical)
        else:
            pv = pv.where(pv != "", fallback_pkg_version)
        raw["package_version"] = pv
    else:
        raw["package_version"] = fallback_pkg_version
    # Group without package_version: reps across nearby upstream patch
    # releases (e.g. statsmodels 0.14.5 vs 0.14.6) are merged into one
    # cell rather than producing two n=1 rows. The cell's reported
    # package_version becomes the latest (max-timestamp) row's value.
    group_keys = ["patch_name", "tier", "threads_lbl", "platform_lbl", "dataset_lbl", "variant"]
    pkg_per_cell = (
        raw.sort_values("timestamp")
        .groupby(group_keys, dropna=False)["package_version"]
        .last()
        .rename("package_version")
    )
    out = (
        raw.groupby(group_keys, dropna=False)
        .apply(agg, include_groups=False)
        .reset_index()
        .merge(pkg_per_cell.reset_index(), on=group_keys, how="left")
        .rename(columns={"patch_name": "patch", "threads_lbl": "threads",
                         "platform_lbl": "platform", "dataset_lbl": "dataset"})
        .sort_values(["patch", "platform", "threads", "dataset", "tier", "variant"])
    )

    # blank speedup/pass cols on baseline rows
    is_base = out["variant"] == "baseline"
    out.loc[is_base, ["speedup_x_reps", "speedup_x_mean", "speedup_x_median", "pass_rate"]] = pd.NA
    out.loc[is_base, "metrics_json_median"] = ""

    # Drop legacy dataset="—" rows when a real-dataset row exists for the same
    # (patch, tier, threads, platform). Keeps solo-"—" tools (e.g. clusterprofiler).
    has_real = out.groupby(["patch", "tier", "threads", "platform"], dropna=False)["dataset"].transform(
        lambda s: (s != "—").any()
    )
    n_before_drop = len(out)
    out = out[~((out["dataset"] == "—") & has_real)].copy()
    n_dropped_em = n_before_drop - len(out)

    # Preserve historical column order: patch, package_version, tier, ...
    leading = ["patch", "package_version", "tier", "threads", "platform",
               "dataset", "variant"]
    rest = [c for c in out.columns if c not in leading]
    out = out[[c for c in leading if c in out.columns] + rest]

    out_path = patch_dir / "speedups_finalized.tsv"
    out.to_csv(out_path, sep="\t", index=False, na_rep="")
    print(
        f"[{patch_dir.name}] {n_before} raw rows ({n_dupe} dupes dropped) -> "
        f"{len(out)} summary rows ({n_dropped_em} redundant '—' rows dropped)"
    )


def main() -> None:
    args = sys.argv[1:]
    if args:
        patches = args
    else:
        patches = sorted(p.name for p in PATCHES_DIR.iterdir() if p.is_dir() and (p / "speedups.tsv").exists())
    for p in patches:
        d = PATCHES_DIR / p
        if not d.exists():
            print(f"skip: {p} (no dir)")
            continue
        finalize_one(d)


if __name__ == "__main__":
    main()
