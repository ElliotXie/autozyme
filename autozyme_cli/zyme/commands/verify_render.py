"""Panel rendering for `zyme verify` — matplotlib painting layer.

Pure presentation: takes already-computed `cells` dicts (from
zyme.commands.verify) and draws the 5-panel matrix plus the textual
scaling-tax summary. No I/O on results.tsv / verify.tsv, no subprocess
calls — every input is in-memory data.

Imported by `zyme.commands.verify` only. Public surface:
  _render_verify_matrix  — entry point, draws + saves verify.{png,pdf,svg}
  _format_scaling_tax    — text block printed after the cell verdict
"""
from __future__ import annotations




# Paul Tol "muted" qualitative palette (colorblind-safe, Nature-Methods-friendly).
# Used to color-encode tiers across all panels so a tier reads consistently
# whether you're looking at Panel A (wall time) or Panel C (memory). Dev tiers
# get the cool colors; OOD tiers get the warm colors via _tier_color() below.
_TIER_PALETTE_DEV = ["#4477AA", "#228833", "#66CCEE", "#AA3377"]   # cool: tiny/medium/large/...

_TIER_PALETTE_OOD = ["#EE6677", "#CCBB44", "#882255", "#DDAA33"]   # warm: ood_*

_BASELINE_COLOR = "#B0B0B0"

_BASELINE_EDGE  = "#606060"

_PASS_COLOR  = "#117733"

_FAIL_COLOR  = "#CC3311"

_CRASH_COLOR = "#444444"

_OOM_COLOR   = "#1A1A1A"   # near-black; distinguishable from CRASH gray

_SOFT_COLOR  = "#DDAA33"






def _format_scaling_tax(tax: dict, thresholds: dict) -> str:
    """Render the scaling-tax block printed after the cell verdict.

    Sections: (1) overall + per-thread dev geomeans (the references each
    OOD cell is compared against), (2) per-OOD-cell verdict table showing
    the thread-matched dev reference used, (3) tail line explaining the
    workflow implication.
    """
    soft_xl = thresholds.get("ood_xlarge_soft", 2.0)
    soft_lg = thresholds.get("ood_large_soft", 1.5)
    hard = thresholds.get("hard_fail", 5.0)
    geom = tax["dev_geom_mean"]
    geom_by_thread = tax.get("dev_geom_by_thread", {}) or {}
    lines = [
        "",
        "=== Scaling tax analysis ===",
        f"Dev-tier speedup factor (overall geometric mean): {geom:.1f}×",
    ]
    if geom_by_thread:
        per_thread = ", ".join(
            f"{t}t={geom_by_thread[t]:.1f}×"
            for t in sorted(geom_by_thread)
        )
        lines.append(f"  per-thread (used for tax): {per_thread}")
    dev_str = ", ".join(
        f"{tier}/{thread}t={f:.0f}×"
        for thread, tier, f in tax["dev_cells"] if f
    )
    if dev_str:
        lines.append(f"  from: {dev_str}")
    lines.append("")
    lines.append("OOD cells (compared against thread-matched dev geomean):")
    SYM = {"PASS": "✓", "SOFT": "⚠", "HARD": "✗", "ZERO": "✗"}
    any_fallback = False
    for r in tax["ood_results"]:
        sym = SYM[r["verdict"]]
        if r["factor"] is None or r["factor"] <= 0:
            factor_str, tax_str = "0×", "∞"
        else:
            factor_str = f"{r['factor']:.1f}×" if r['factor'] < 100 else f"{r['factor']:.0f}×"
            tax_str = f"{r['tax']:.1f}×"
        verdict = r["verdict"]
        ref = r.get("dev_reference")
        ref_kind = r.get("reference_kind", "thread_matched")
        if ref:
            ref_str = f"vs {ref:.1f}×"
            if ref_kind == "fallback_global":
                ref_str += "(global*)"
                any_fallback = True
        else:
            ref_str = ""
        rule = f"(tax > {r['threshold_used']:.0f}×, {r['threshold_label']})" if verdict != "PASS" else ""
        lines.append(
            f"  {sym} {r['tier']:<10} {r['thread']}t   "
            f"speedup={factor_str:>8}   {ref_str:<16}  tax={tax_str:>6}   {verdict:<5} {rule}"
        )
    if any_fallback:
        lines.append("  (*) no dev cell at this thread count — fell back to overall dev geomean")
    lines.append("")
    lines.append(
        f"Scaling verdict: {tax['hard_fails']} HARD FAIL, {tax['soft_flags']} SOFT FLAG "
        f"(out of {len(tax['ood_results'])} OOD cells)."
    )
    if tax["hard_fails"]:
        lines.append(
            f"  → HARD FAIL means tax > {hard:.0f}× the thread-matched dev speedup. "
            f"Phase A FAIL: enter fix-loop. Profile at the failing OOD cell first."
        )
        lines.append(
            "  → If after dig-in the cliff is fundamental and unfixable, "
            "document with `## DISCOVERY:` in memory/discoveries.md before SUMMARY."
        )
    elif tax["soft_flags"]:
        lines.append(
            f"  → SOFT FLAG means tax > {soft_lg:.0f}×/{soft_xl:.0f}× "
            f"(ood_large/ood_xlarge) vs thread-matched dev. Investigate once "
            f"and explicitly note the cause in SUMMARY. Not auto-blocking."
        )
    else:
        lines.append("  → all OOD cells within healthy scaling tax.")
    return "\n".join(lines)




def _tier_color(tier, dev_tiers, ood_tiers):
    """Stable per-tier color: dev tiers use the cool palette, OOD the warm."""
    if tier in ood_tiers:
        return _TIER_PALETTE_OOD[ood_tiers.index(tier) % len(_TIER_PALETTE_OOD)]
    if tier in dev_tiers:
        return _TIER_PALETTE_DEV[dev_tiers.index(tier) % len(_TIER_PALETTE_DEV)]
    return "#888888"




def _order_cells_for_plot(cells):
    """Order cells along the plot's x-axis: dev tiers first, OOD last.

    Within each group, tiers come in YAML appearance order (tiny < medium <
    large < ood_large < ood_xlarge); within a tier, threads are ascending
    (1 < 4 < 8). Returns (sorted_cells, dev_tiers, ood_tiers, all_tiers, all_threads).
    """
    yaml_order = {}
    for c in cells:
        yaml_order.setdefault(c["tier"], len(yaml_order))
    all_tiers = sorted({c["tier"] for c in cells}, key=lambda t: yaml_order[t])
    dev_tiers = [t for t in all_tiers if not t.startswith("ood_")]
    ood_tiers = [t for t in all_tiers if t.startswith("ood_")]
    ordered_tiers = dev_tiers + ood_tiers
    tier_rank = {t: i for i, t in enumerate(ordered_tiers)}
    all_threads = sorted({c["thread"] for c in cells})
    sorted_cells = sorted(cells, key=lambda c: (tier_rank[c["tier"]], c["thread"]))
    return sorted_cells, dev_tiers, ood_tiers, ordered_tiers, all_threads




def _cell_xtick_label(c):
    """Two-line x-tick: tier on top, thread on bottom — readable at 0° rotation."""
    return f"{c['tier']}\n{c['thread']}t"




def _draw_tier_dividers(ax, cells_ordered):
    """Light vertical separators between tier groups so the eye groups by tier."""
    last_tier = None
    for i, c in enumerate(cells_ordered):
        if last_tier is not None and c["tier"] != last_tier:
            ax.axvline(i - 0.5, color="#D0D0D0", linewidth=0.7,
                       linestyle="-", zorder=0.5)
        last_tier = c["tier"]




def _render_verify_matrix(cells, task_name, out_base, plt, metrics_spec=None, n_reps=1,
                          scaling_tax=None):
    """Nature-Methods-style six-panel verify dashboard (3 rows × 2 cols).

    Layout:
      A. Wall time           B. Speedup factor
      C. Peak memory         D. Concordance margin
      E. Run-to-run CV%      F. Verdict + scaling-tax footer

    Each panel uses a consistent per-tier color so the reader's eye tracks
    a single tier across panels (e.g., `ood_xlarge` is the same warm-red
    shade in panels A, B, C). Cells are ordered dev → OOD with light
    vertical separators between tier groups so the comparison is one-glance.

    All panels written into a single `verify.png` / `.pdf` / `.svg` so
    downstream packaging prompts that reference `verify.png` keep working.
    """
    metrics_spec = metrics_spec or []
    plt.rcParams.update({
        # Arial first (Nature-Methods house style on macOS) but fall back to
        # DejaVu Sans (matplotlib's bundled default) — Arial is missing some
        # glyphs (⚠/✗/✓) we use in panel labels, and DejaVu has them all.
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "figure.dpi": 220,
        "axes.labelsize": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 6,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "savefig.facecolor": "white",
    })

    cells_ordered, dev_tiers, ood_tiers, all_tiers, all_threads = _order_cells_for_plot(cells)

    # 3 × 2 grid; widths slightly biased so D (heatmap with metric labels)
    # has room without squishing the ones to its left.
    fig = plt.figure(figsize=(13.0, 13.5))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 1.05], width_ratios=[1.0, 1.05],
        hspace=0.55, wspace=0.28,
    )
    ax_wall      = fig.add_subplot(gs[0, 0])
    ax_speedup   = fig.add_subplot(gs[0, 1])
    ax_memory    = fig.add_subplot(gs[1, 0])
    ax_concord   = fig.add_subplot(gs[1, 1])
    ax_cv        = fig.add_subplot(gs[2, 0])
    ax_verdict   = fig.add_subplot(gs[2, 1])

    _panel_walltime(ax_wall, cells_ordered, dev_tiers, ood_tiers, plt)
    _panel_speedup_factor(ax_speedup, cells_ordered, dev_tiers, ood_tiers, plt,
                          scaling_tax=scaling_tax)
    _panel_memory(ax_memory, cells_ordered, dev_tiers, ood_tiers, plt)
    _panel_concordance(ax_concord, cells_ordered, metrics_spec, plt)
    _panel_cv(ax_cv, cells_ordered, n_reps, plt)
    _panel_verdict(ax_verdict, cells_ordered, scaling_tax, plt)

    n_dev = sum(1 for c in cells_ordered if not c["tier"].startswith("ood_"))
    n_ood = sum(1 for c in cells_ordered if c["tier"].startswith("ood_"))
    rep_label = f"{n_reps} rep" + ("s" if n_reps != 1 else "") + "/cell"
    subtitle_bits = [rep_label, f"{n_dev} dev cell" + ("s" if n_dev != 1 else "")]
    if n_ood:
        subtitle_bits.append(f"{n_ood} OOD cell" + ("s" if n_ood != 1 else ""))
    if any(not c.get("in_current_run", True) for c in cells_ordered):
        subtitle_bits.append("includes prior-phase cells from verify.tsv")
    fig.suptitle(
        f"{task_name}  —  verify dashboard",
        fontsize=14, y=0.995, fontweight="bold", x=0.06, ha="left",
    )
    fig.text(0.06, 0.972, "  ·  ".join(subtitle_bits),
             fontsize=9.5, style="italic", color="#444444", ha="left")

    # Save before tight_layout so the manually-placed suptitle/subtitle survive.
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)




def _fmt_mb(v):
    """Memory label: GB at >=1024 MB, otherwise MB."""
    if v >= 1024:
        return f"{v / 1024:.1f}G"
    return f"{int(round(v))}M"




def _fmt_seconds(v):
    """Wall-time label: minutes when ≥120 s, otherwise seconds with sensible precision."""
    if v >= 120:
        return f"{v / 60:.1f}m"
    if v >= 10:
        return f"{int(round(v))}s"
    if v >= 1:
        return f"{v:.1f}s"
    return f"{v:.2g}s"




def _setup_cell_xaxis(ax, cells_ordered):
    """Standard x-axis used by panels A, B, C, D, E (one tick per cell)."""
    n = len(cells_ordered)
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [_cell_xtick_label(c) for c in cells_ordered],
        fontsize=7.2, linespacing=1.2,
    )
    ax.set_xlim(-0.6, n - 0.4)
    _draw_tier_dividers(ax, cells_ordered)




def _panel_walltime(ax, cells_ordered, dev_tiers, ood_tiers, plt):
    """Panel A: per-cell paired bars — upstream baseline (gray) vs turbo (color).

    One pair per cell along the x-axis (cells ordered dev → OOD). Speedup
    factor and turbo wall time are labeled above each turbo bar so the reader
    can read off `5.2×, 4.8s` directly. Log y-scale kicks in only when the
    dynamic range of speeds across cells exceeds 20× — otherwise linear.
    """
    bar_w = 0.36
    all_speeds = []
    for i, c in enumerate(cells_ordered):
        base = c.get("baseline_speed", 0.0) or 0.0
        turbo = c.get("speed_sec_median", 0.0) or 0.0
        is_oom = bool(c.get("oom"))
        all_speeds += [v for v in (base, turbo) if v > 0]
        color = _tier_color(c["tier"], dev_tiers, ood_tiers)
        if base > 0:
            ax.bar(i - bar_w / 2, base, width=bar_w,
                   color=_BASELINE_COLOR, edgecolor=_BASELINE_EDGE,
                   linewidth=0.6, zorder=2)
            # Baseline label directly above its own gray bar.
            ax.text(i - bar_w / 2, base, _fmt_seconds(base),
                    ha="center", va="bottom",
                    fontsize=6.4, color="#606060")
        if is_oom:
            # OOM cell: replace turbo bar with a hatched dark block at the
            # baseline's height (or a small fixed height when no baseline).
            # Visual: "we tried, didn't fit." White "OOM" label inside.
            oom_h = base if base > 0 else 1.0
            ax.bar(i + bar_w / 2, oom_h, width=bar_w,
                   color=_OOM_COLOR, edgecolor="#202020", linewidth=0.6,
                   hatch="xx", zorder=2)
            ax.text(i + bar_w / 2, oom_h * 0.5, "OOM",
                    ha="center", va="center",
                    fontsize=7.0, color="white", fontweight="bold",
                    rotation=90 if oom_h < 1 else 0)
        elif turbo > 0:
            ax.bar(i + bar_w / 2, turbo, width=bar_w,
                   color=color, edgecolor="#202020", linewidth=0.6,
                   alpha=0.95 if c.get("in_current_run", True) else 0.55,
                   hatch=None if c.get("in_current_run", True) else "//",
                   zorder=2)
            # Turbo label directly above its own bar — speedup factor first
            # (the headline number), wall time second.
            if base > 0:
                factor = base / turbo
                lbl = f"{factor:.1f}×\n{_fmt_seconds(turbo)}"
            else:
                lbl = _fmt_seconds(turbo)
            ax.text(i + bar_w / 2, turbo, lbl,
                    ha="center", va="bottom",
                    fontsize=6.8, color="#202020", linespacing=1.0,
                    fontweight="bold")

    if all_speeds and (max(all_speeds) / max(min(all_speeds), 1e-6)) > 20:
        ax.set_yscale("log")
        ax.set_ylim(top=max(all_speeds) * 1.8)
    elif all_speeds:
        ax.set_ylim(0, max(all_speeds) * 1.22)

    _setup_cell_xaxis(ax, cells_ordered)
    ax.set_ylabel("Wall time (s)")
    ax.set_title("A   Wall time: upstream vs turbo", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_BASELINE_COLOR, edgecolor=_BASELINE_EDGE, label="upstream baseline"),
        Patch(facecolor="#777777", edgecolor="#202020", label="turbo (this build)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.2, ncol=2)




def _panel_speedup_factor(ax, cells_ordered, dev_tiers, ood_tiers, plt, scaling_tax=None):
    """Panel B: per-cell speedup factor as bar chart with HARD/SOFT shading.

    Replaces the heatmap, which made it hard to read absolute factors when
    many cells clustered close together. Instead: a vertical bar per cell,
    color-coded by tier, with the factor labeled above each bar. The 1×
    reference line is dashed; the dev-tier geometric-mean factor is dashed
    in green; HARD/SOFT scaling-tax cells get a thick red/yellow border so
    cliffs are impossible to miss.
    """
    factors = []
    for c in cells_ordered:
        base = c.get("baseline_speed", 0.0) or 0.0
        turbo = c.get("speed_sec_median", 0.0) or 0.0
        factors.append((base / turbo) if (base > 0 and turbo > 0) else None)

    bar_w = 0.7
    for i, (c, f) in enumerate(zip(cells_ordered, factors)):
        if c.get("oom"):
            # OOM: small dark hatched block + white "OOM" label so the slot
            # isn't visually empty (an empty slot reads as "data missing,
            # not yet measured" — OOM is a positive measurement).
            ax.bar(i, 0.7, width=bar_w, color=_OOM_COLOR, edgecolor="#202020",
                   linewidth=0.6, hatch="xx", zorder=2)
            ax.text(i, 0.35, "OOM", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white")
            continue
        if f is None or f <= 0:
            ax.text(i, 0.05, "—", ha="center", va="bottom",
                    fontsize=8, color="#888")
            continue
        color = _tier_color(c["tier"], dev_tiers, ood_tiers)
        ax.bar(i, f, width=bar_w, color=color, edgecolor="#202020",
               linewidth=0.6,
               alpha=0.95 if c.get("in_current_run", True) else 0.55,
               hatch=None if c.get("in_current_run", True) else "//",
               zorder=2)
        txt = f"{f:.1f}×" if f < 100 else f"{f:.0f}×"
        ax.text(i, f, txt, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#202020")

    finite = [f for f in factors if f and f > 0]
    ymax = max([1.5] + [f * 1.18 for f in finite]) if finite else 2.0
    ax.set_ylim(0, ymax)

    # Reference lines: 1× (parity) and dev-tier geom mean (when available).
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.7, zorder=1)
    ax.text(len(cells_ordered) - 0.4, 1.0, " 1× (no speedup)",
            va="bottom", ha="right", fontsize=6.8, color="#666666")

    geom = scaling_tax.get("dev_geom_mean") if scaling_tax else None
    if geom and geom > 0:
        ax.axhline(geom, color="#117733", linestyle="--", linewidth=0.9, zorder=1)
        ax.text(0.4, geom, f" dev geom mean (overall) = {geom:.1f}×",
                va="bottom", ha="left", fontsize=6.8, color="#117733",
                fontweight="bold")

    # HARD/SOFT outline overlay on individual cells.
    if scaling_tax and scaling_tax.get("applicable"):
        from matplotlib.patches import Rectangle
        cell_idx = {(c["thread"], c["tier"]): i for i, c in enumerate(cells_ordered)}
        flagged = {"HARD": False, "SOFT": False, "ZERO": False}
        for r in scaling_tax.get("ood_results", []):
            verdict = r["verdict"]
            i = cell_idx.get((r["thread"], r["tier"]))
            if i is None:
                continue
            if verdict in ("HARD", "ZERO"):
                edge, lw = _FAIL_COLOR, 2.0
            elif verdict == "SOFT":
                edge, lw = _SOFT_COLOR, 1.6
            else:
                continue
            f = factors[i] or 0.05
            ax.add_patch(Rectangle(
                (i - bar_w / 2, 0), bar_w, max(f, 0.05),
                fill=False, edgecolor=edge, linewidth=lw, zorder=4,
            ))
            flagged[verdict] = True
        from matplotlib.patches import Patch
        legend_handles = []
        if flagged.get("HARD") or flagged.get("ZERO"):
            legend_handles.append(Patch(facecolor="none", edgecolor=_FAIL_COLOR,
                                        linewidth=2.0, label="scaling-tax HARD"))
        if flagged.get("SOFT"):
            legend_handles.append(Patch(facecolor="none", edgecolor=_SOFT_COLOR,
                                        linewidth=1.6, label="scaling-tax SOFT"))
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right", fontsize=7.0)

    _setup_cell_xaxis(ax, cells_ordered)
    ax.set_ylabel("Speedup factor (× upstream)")
    title_extra = ""
    if scaling_tax and scaling_tax.get("applicable"):
        title_extra = (f"   ·   tax verdict: {scaling_tax.get('hard_fails', 0)} HARD, "
                       f"{scaling_tax.get('soft_flags', 0)} SOFT")
    ax.set_title(f"B   Speedup factor (× upstream){title_extra}", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)




def _panel_memory(ax, cells_ordered, dev_tiers, ood_tiers, plt):
    """Panel C: per-cell paired bars — upstream peak memory vs turbo peak memory.

    Mirrors Panel A's layout (paired bars, log scale when dynamic range > 20×)
    but on the memory axis. Without baseline_peak_mb (e.g., legacy tasks where
    reference.{py,R} did not record peak_mb) the baseline bar is omitted and
    only the turbo bar is drawn — annotated with `(no baseline)`. With both
    bars present, the label above is the relative change vs baseline (e.g.
    `−42% (3.2 G)`) so memory regressions can't hide under absolute-MB framing.
    """
    bar_w = 0.36
    all_mb = []
    any_baseline = False
    for i, c in enumerate(cells_ordered):
        base = c.get("baseline_peak_mb", 0.0) or 0.0
        turbo = c.get("peak_mb_median", 0.0) or 0.0
        is_oom = bool(c.get("oom"))
        all_mb += [v for v in (base, turbo) if v > 0]
        color = _tier_color(c["tier"], dev_tiers, ood_tiers)
        if base > 0:
            any_baseline = True
            ax.bar(i - bar_w / 2, base, width=bar_w,
                   color=_BASELINE_COLOR, edgecolor=_BASELINE_EDGE,
                   linewidth=0.6, zorder=2)
            # Baseline label directly above gray bar.
            ax.text(i - bar_w / 2, base, _fmt_mb(base),
                    ha="center", va="bottom",
                    fontsize=6.4, color="#606060")
        if is_oom:
            # OOM: keep baseline bar (it succeeded) but turbo slot becomes
            # a hatched dark block at baseline's height with white "OOM" text.
            oom_h = base if base > 0 else 1024.0  # 1 GB visual placeholder
            ax.bar(i + (bar_w / 2 if base > 0 else 0), oom_h,
                   width=bar_w if base > 0 else bar_w * 1.6,
                   color=_OOM_COLOR, edgecolor="#202020", linewidth=0.6,
                   hatch="xx", zorder=2)
            ax.text(i + (bar_w / 2 if base > 0 else 0), oom_h * 0.5, "OOM",
                    ha="center", va="center",
                    fontsize=7.0, color="white", fontweight="bold")
        elif turbo > 0:
            ax.bar(i + (bar_w / 2 if base > 0 else 0), turbo,
                   width=bar_w if base > 0 else bar_w * 1.6,
                   color=color, edgecolor="#202020", linewidth=0.6,
                   alpha=0.95 if c.get("in_current_run", True) else 0.55,
                   hatch=None if c.get("in_current_run", True) else "//",
                   zorder=2)
            # Turbo label directly above its own bar — relative change first
            # (the headline number), absolute value second.
            if base > 0:
                delta = (turbo - base) / base * 100.0
                sign = "+" if delta >= 0 else "−"
                lbl = f"{sign}{abs(delta):.0f}%\n{_fmt_mb(turbo)}"
            else:
                lbl = f"{_fmt_mb(turbo)}\n(no baseline)"
            ax.text(i + (bar_w / 2 if base > 0 else 0), turbo, lbl,
                    ha="center", va="bottom",
                    fontsize=6.8, color="#202020", linespacing=1.0,
                    fontweight="bold")

    if all_mb and (max(all_mb) / max(min(all_mb), 1.0)) > 20:
        ax.set_yscale("log")
        ax.set_ylim(top=max(all_mb) * 1.8)
    elif all_mb:
        ax.set_ylim(0, max(all_mb) * 1.24)

    _setup_cell_xaxis(ax, cells_ordered)
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title("C   Peak memory: upstream vs turbo", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    handles = []
    if any_baseline:
        handles.append(Patch(facecolor=_BASELINE_COLOR, edgecolor=_BASELINE_EDGE,
                             label="upstream baseline"))
    handles.append(Patch(facecolor="#777777", edgecolor="#202020",
                         label="turbo (this build)"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.2, ncol=2)




def _panel_concordance(ax, cells_ordered, metrics_spec, plt):
    """Panel D: metrics × cells grid, per-metric color scaling.

    Each metric (each row) gets its own color ramp: pass cells use a green
    ramp, fail cells use a red ramp, both saturated within the row's actual
    spread. This avoids the "all pale yellow" trap of a shared ±50% scale —
    when every cell on a row passes (even with tiny margins), they all read
    as deep green; only true failures show up red. Cell label = actual value.
    """
    if not metrics_spec:
        ax.text(0.5, 0.5,
                "no metrics declared in task.yaml",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("D   Concordance metrics", loc="left")
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        return
    if not cells_ordered:
        ax.set_axis_off()
        return

    import numpy as np
    from matplotlib.patches import Rectangle

    n_cells = len(cells_ordered)
    n_metrics = len(metrics_spec)

    # Per-cell, per-metric raw values + pass/fail decisions.
    actual = np.full((n_metrics, n_cells), np.nan)
    margin = np.full((n_metrics, n_cells), np.nan)
    is_pass = np.zeros((n_metrics, n_cells), dtype=bool)
    yticks_lbl = []

    for mi, m in enumerate(metrics_spec):
        name = m["name"]
        comp = m["comparator"]
        thresh = m.get("threshold", m.get("absolute_floor"))
        if "threshold" in m:
            thr_label = f"{comp} {thresh}"
        else:
            thr_label = f"{comp} {thresh} · noise×{m.get('noise_multiplier', 2.0):g}"
        yticks_lbl.append(f"{name}\n({thr_label})")
        if thresh is None:
            continue
        try:
            thresh_f = float(thresh)
        except (TypeError, ValueError):
            continue
        for ci, c in enumerate(cells_ordered):
            v = c["metrics_worst"].get(name)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            actual[mi, ci] = vf
            raw = vf - thresh_f if comp == "gte" else thresh_f - vf
            margin[mi, ci] = raw
            is_pass[mi, ci] = (raw >= 0)

    # Per-metric saturated green/red palette.
    # Pass: light green → deep green. Fail: light red → deep red.
    # Unconditional minimum saturation so even a "barely passing" cell still
    # reads clearly as green (not pale yellow). Within the row, larger margin
    # = deeper color so the eye can still rank cells against each other.
    PASS_LIGHT = np.array([0.78, 0.91, 0.78])  # #C7E8C7
    PASS_DEEP  = np.array([0.07, 0.46, 0.20])  # #117733
    FAIL_LIGHT = np.array([0.97, 0.84, 0.82])  # #F8D7D1
    FAIL_DEEP  = np.array([0.80, 0.20, 0.07])  # #CC3311
    EMPTY      = np.array([0.96, 0.96, 0.96])
    PASS_FLOOR = 0.55  # min saturation along PASS_LIGHT→PASS_DEEP axis
    FAIL_FLOOR = 0.55

    # Pre-compute which cells are OOM so they get a uniform dark fill across
    # all metric rows (overrides per-metric pass/fail coloring).
    oom_cells = [bool(c.get("oom")) for c in cells_ordered]

    for mi in range(n_metrics):
        row_margin = margin[mi]
        # Exclude OOM cells from the row's pass/fail spread (they're not a
        # measurement) so the row's color scale stays anchored on real data.
        row_pass_mask = is_pass[mi] & np.isfinite(row_margin) & ~np.array(oom_cells)
        row_fail_mask = (~is_pass[mi]) & np.isfinite(row_margin) & ~np.array(oom_cells)
        pass_vals = row_margin[row_pass_mask]
        fail_vals = -row_margin[row_fail_mask]
        pass_lo, pass_hi = (pass_vals.min(), pass_vals.max()) if pass_vals.size else (0, 0)
        fail_lo, fail_hi = (fail_vals.min(), fail_vals.max()) if fail_vals.size else (0, 0)

        for ci in range(n_cells):
            v = actual[mi, ci]
            if oom_cells[ci]:
                # OOM column: dark fill, white "OOM" text added in the label
                # loop below. Keeps the cell visually distinct from "missing
                # data" (gray) and from "fail" (red).
                ax.add_patch(Rectangle(
                    (ci - 0.5, mi - 0.5), 1.0, 1.0,
                    facecolor=_OOM_COLOR, edgecolor="none", zorder=1,
                    hatch="xx",
                ))
                continue
            if not np.isfinite(v):
                color = EMPTY
            elif is_pass[mi, ci]:
                if pass_hi > pass_lo:
                    t = (row_margin[ci] - pass_lo) / (pass_hi - pass_lo)
                else:
                    t = 1.0
                t = PASS_FLOOR + (1.0 - PASS_FLOOR) * t
                color = PASS_LIGHT * (1 - t) + PASS_DEEP * t
            else:
                magnitude = -row_margin[ci]
                if fail_hi > fail_lo:
                    t = (magnitude - fail_lo) / (fail_hi - fail_lo)
                else:
                    t = 1.0
                t = FAIL_FLOOR + (1.0 - FAIL_FLOOR) * t
                color = FAIL_LIGHT * (1 - t) + FAIL_DEEP * t
            ax.add_patch(Rectangle(
                (ci - 0.5, mi - 0.5), 1.0, 1.0,
                facecolor=tuple(color), edgecolor="none", zorder=1,
            ))

    # Cell labels: the actual measured value. Auto-pick precision so the
    # number always reads cleanly regardless of the metric's scale.
    for mi in range(n_metrics):
        for ci in range(n_cells):
            if oom_cells[ci]:
                # Only label the middle row(s) of the OOM column to avoid
                # repetition. With ≤2 metrics, label every row; otherwise
                # only label the middle row.
                if n_metrics <= 2 or mi == n_metrics // 2:
                    ax.text(ci, mi, "OOM", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold")
                continue
            v = actual[mi, ci]
            if not np.isfinite(v):
                ax.text(ci, mi, "—", ha="center", va="center",
                        fontsize=7, color="#888")
                continue
            if abs(v) >= 100:
                txt = f"{v:.0f}"
            elif abs(v) >= 1:
                txt = f"{v:.3f}"
            elif abs(v) >= 0.01:
                txt = f"{v:.4f}"
            else:
                txt = f"{v:.2e}"
            # White on red (failure), dark on green (pass) — high contrast in
            # both directions regardless of where the cell sits in its ramp.
            label_color = "white" if not is_pass[mi, ci] else "#101010"
            ax.text(ci, mi, txt, ha="center", va="center",
                    fontsize=6.8, color=label_color, fontweight="bold")

    # Tier-group dividers: thick white lines between tier columns.
    last_tier = None
    for i, c in enumerate(cells_ordered):
        if last_tier is not None and c["tier"] != last_tier:
            ax.axvline(i - 0.5, color="white", linewidth=1.8, zorder=3)
        last_tier = c["tier"]

    ax.set_xlim(-0.5, n_cells - 0.5)
    ax.set_ylim(n_metrics - 0.5, -0.5)  # invert so first metric is on top
    ax.set_xticks(range(n_cells))
    ax.set_xticklabels(
        [_cell_xtick_label(c) for c in cells_ordered],
        fontsize=7.2, linespacing=1.2,
    )
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(yticks_lbl, fontsize=7.5)
    ax.set_title("D   Concordance: per-metric color (green = pass, red = fail; label = value)",
                 loc="left")
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)




def _panel_cv(ax, cells_ordered, n_reps, plt):
    """Panel E: per-cell run-to-run CV%. Single row, color-graded bar chart.

    Reps≥2 only — for single-rep matrices, CV is undefined and the panel shows
    a placeholder. Bar height = CV%; bars ≥ 30% are flagged red (host-pressure
    noise the 3-rep gate is meant to catch); the 30% threshold is dashed.
    """
    if n_reps < 2:
        ax.text(0.5, 0.5,
                f"single rep per cell — CV undefined\n(rerun with `--reps 3` for variance)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("E   Run-to-run CV%", loc="left")
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        return

    cvs = [c.get("cv_pct") for c in cells_ordered]
    bar_w = 0.7
    any_value = False
    for i, (c, cv) in enumerate(zip(cells_ordered, cvs)):
        if cv is None:
            ax.text(i, 1.0, "—", ha="center", va="bottom",
                    fontsize=8, color="#888")
            continue
        any_value = True
        color = _FAIL_COLOR if cv >= 30.0 else "#4477AA"
        ax.bar(i, cv, width=bar_w, color=color, edgecolor="#202020",
               linewidth=0.6, zorder=2,
               alpha=0.95 if c.get("in_current_run", True) else 0.55,
               hatch=None if c.get("in_current_run", True) else "//")
        ax.text(i, cv, f"{cv:.1f}%", ha="center", va="bottom",
                fontsize=7.2, fontweight="bold",
                color=_FAIL_COLOR if cv >= 30.0 else "#202020")

    finite = [v for v in cvs if v is not None]
    ymax = max([35.0] + [v * 1.2 for v in finite]) if any_value else 35.0
    ax.set_ylim(0, ymax)
    ax.axhline(30.0, color=_FAIL_COLOR, linestyle="--", linewidth=0.7, zorder=1)
    ax.text(len(cells_ordered) - 0.4, 30.0, " 30% (host-noise flag) ",
            va="bottom", ha="right", fontsize=6.8, color=_FAIL_COLOR)

    _setup_cell_xaxis(ax, cells_ordered)
    ax.set_ylabel("CV % (across reps)")
    ax.set_title("E   Run-to-run CV%   (≥30% = host noise)", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)




def _panel_verdict(ax, cells_ordered, scaling_tax, plt):
    """Panel F: per-cell PASS/FAIL/CRASH strip + scaling-tax footer.

    Top half is a single-row colored strip (one block per cell, dev → OOD
    order with the same x positions as A/B/C/D/E). Bottom half is a multi-line
    text block: dev geom mean + per-OOD-cell tax verdict, so the reader sees
    the headline number from `_format_scaling_tax` without leaving the figure.
    """
    n_cells = len(cells_ordered)
    # Empty axis frame, then build the strip + text by hand.
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)

    # Strip: PASS / FAIL / CRASH / OOM / context cells (no verdict computed).
    color_for = {
        "PASS": _PASS_COLOR, "FAIL": _FAIL_COLOR, "CRASH": _CRASH_COLOR,
        "OOM": _OOM_COLOR,
    }
    for i, c in enumerate(cells_ordered):
        if c.get("oom"):
            v = "OOM"
        elif c.get("any_crash"):
            v = "CRASH"
        elif c.get("verdict") == "FAIL":
            v = "FAIL"
        elif c.get("verdict") == "PASS":
            v = "PASS"
        else:
            v = "—"
        face = color_for.get(v, "#CCCCCC")
        edge = "#202020"
        # OOM gets a hatched fill so it's visually distinct from CRASH (the
        # hatching is the "we tried, didn't fit" signal).
        oom_hatch = "xx" if v == "OOM" else None
        ctx_hatch = None if c.get("in_current_run", True) else "//"
        hatch = oom_hatch or ctx_hatch
        ax.add_patch(plt.Rectangle((i - 0.45, 0.7), 0.9, 0.6,
                                   facecolor=face, edgecolor=edge,
                                   linewidth=0.7, zorder=2,
                                   alpha=0.95 if c.get("in_current_run", True) else 0.55,
                                   hatch=hatch))
        ax.text(i, 1.0, v, ha="center", va="center",
                fontsize=7.4, fontweight="bold",
                color="white" if v in ("FAIL", "CRASH", "OOM", "PASS") else "#202020")
        ax.text(i, 0.45, _cell_xtick_label(c), ha="center", va="top",
                fontsize=7.0, color="#202020", linespacing=1.1)

    # Strip-level tier dividers.
    last_tier = None
    for i, c in enumerate(cells_ordered):
        if last_tier is not None and c["tier"] != last_tier:
            ax.axvline(i - 0.5, color="#D0D0D0", linewidth=0.7, zorder=1)
        last_tier = c["tier"]

    # Footer text: scaling-tax summary.
    footer_lines = []
    if scaling_tax and scaling_tax.get("applicable"):
        geom = scaling_tax.get("dev_geom_mean")
        geom_by_thread = scaling_tax.get("dev_geom_by_thread") or {}
        if geom_by_thread:
            per_thread = ", ".join(
                f"{t}t={geom_by_thread[t]:.1f}×" for t in sorted(geom_by_thread)
            )
            footer_lines.append(
                f"Dev-tier speedup factor (overall geomean): {geom:.1f}×  "
                f"|  per-thread (used for tax): {per_thread}"
            )
        else:
            footer_lines.append(f"Dev-tier speedup factor (geometric mean): {geom:.1f}×")
        if scaling_tax.get("ood_results"):
            footer_lines.append("OOD scaling tax:")
            for r in scaling_tax["ood_results"]:
                v = r["verdict"]
                if r["factor"] is None or r["factor"] <= 0:
                    factor_str, tax_str = "0×", "∞"
                else:
                    factor_str = f"{r['factor']:.1f}×"
                    tax_str = f"{r['tax']:.1f}×"
                sym = {"PASS": "✓", "SOFT": "⚠", "HARD": "✗", "ZERO": "✗"}.get(v, "·")
                footer_lines.append(
                    f"   {sym}  {r['tier']:<11} {r['thread']}t   "
                    f"speedup={factor_str:>7}   tax={tax_str:>7}   {v}"
                )
        footer_lines.append(
            f"Overall: {scaling_tax['hard_fails']} HARD, "
            f"{scaling_tax['soft_flags']} SOFT  "
            f"(out of {len(scaling_tax['ood_results'])} OOD cell{'s' if len(scaling_tax['ood_results']) != 1 else ''})"
        )
    elif scaling_tax:
        reason = scaling_tax.get("reason",
                                 "matrix does not have both dev and OOD tiers")
        footer_lines.append("Scaling tax: N/A — " + reason + ".")
        footer_lines.append("Include both dev tiers and at least one ood_* tier")
        footer_lines.append("(e.g. `--tiers tiny,medium,ood_large,ood_xlarge`)")
        footer_lines.append("for the generalization verdict.")
    ax.text(-0.45, 0.18, "\n".join(footer_lines),
            ha="left", va="top", fontsize=7.6, family="monospace",
            color="#202020")

    # Legend — verdict colors, plus the hatch convention for context cells.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_PASS_COLOR, edgecolor="#202020", label="PASS"),
        Patch(facecolor=_FAIL_COLOR, edgecolor="#202020", label="FAIL"),
        Patch(facecolor=_CRASH_COLOR, edgecolor="#202020", label="CRASH"),
    ]
    has_oom = any(c.get("oom") for c in cells_ordered)
    if has_oom:
        handles.append(Patch(facecolor=_OOM_COLOR, edgecolor="#202020", hatch="xx",
                             label="OOM (tier did not fit)"))
    has_context = any(not c.get("in_current_run", True) for c in cells_ordered)
    if has_context:
        handles.append(Patch(facecolor="#CCCCCC", edgecolor="#202020", hatch="//",
                             label="prior-phase cell (verify.tsv)"))
    ax.legend(handles=handles, loc="upper right",
              fontsize=7.0, ncol=len(handles), bbox_to_anchor=(1.0, 1.0))

    ax.set_xlim(-0.6, n_cells - 0.4)
    ax.set_ylim(-0.4, 1.5)
    ax.set_title("F   Verdict per cell + scaling-tax summary", loc="left")
