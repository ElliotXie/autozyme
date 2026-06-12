"""`zyme plot` — convergence plots from results.tsv."""

from pathlib import Path

from zyme.utils import die, info, task_dir_from_args
from zyme.parsers.results_tsv import (
    row_phase,
)
from zyme.commands._shared import _read_results_rows




def cmd_plot(args):
    """Render a convergence curve from results.tsv.

    Style mirrors `Figure/convergence/convergence_panel.py`:
      - baseline + keep rows = "Accepted", connected by a solid line
      - discard rows           = faint red X markers (off-line)
      - crash rows             = red X anchored at baseline level
      - rerun / pending rows   = excluded (stability measurements / undecided)

    One figure per dataset; auto log-scale when the runtime range spans >20×.
    """
    task_dir = task_dir_from_args(args)
    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        die(f"no results.tsv in {task_dir}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        die("matplotlib is required for `zyme plot`: pip install matplotlib")

    rows = _read_results_rows(results_tsv)
    if not rows:
        die("results.tsv has no data rows")

    plot_phase = getattr(args, "phase", "optimize")
    if plot_phase != "all":
        rows = [r for r in rows if row_phase(r) == plot_phase
                # baseline rows are stamped phase=optimize but should appear
                # on the plot regardless of what phase is being plotted.
                or r.get("status") == "baseline"]

    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r.get("dataset") or "default", []).append(r)

    if args.dataset:
        if args.dataset not in by_dataset:
            die(f"dataset '{args.dataset}' not in results.tsv. "
                f"Available: {sorted(by_dataset)}")
        targets = {args.dataset: by_dataset[args.dataset]}
    else:
        targets = by_dataset

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (task_dir / "figure")
    out_dir.mkdir(parents=True, exist_ok=True)

    task_name = task_dir.name
    written = []
    for dataset, drows in targets.items():
        decision = [r for r in drows
                    if r.get("status") in ("baseline", "keep", "discard", "crash")]
        if not decision:
            info(f"skipping {dataset}: no decision rows")
            continue

        def _round_key(r):
            try:
                return float(r.get("round") or 0)
            except ValueError:
                return float("inf")
        decision.sort(key=_round_key)

        title = args.title or f"{task_name} — {dataset}"
        out_base = out_dir / f"convergence_{task_name}_{dataset}"
        _render_convergence(decision, title, out_base, args.log, plt, ticker)
        written.append(out_base)

    if not written:
        die("no figures written (no decision rows in results.tsv)")
    info(f"wrote {len(written)} figure(s):")
    for b in written:
        info(f"  {b}.png / .pdf / .svg")




def _render_convergence(rows, title, out_base, log_mode, plt, ticker):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "figure.dpi": 300,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
    })

    n = len(rows)
    xs = list(range(1, n + 1))
    ys = []
    for r in rows:
        try:
            ys.append(float(r.get("speed_sec") or 0))
        except ValueError:
            ys.append(0.0)
    statuses = [r.get("status", "") for r in rows]

    keep_x = [x for x, s in zip(xs, statuses) if s in ("keep", "baseline")]
    keep_y = [y for y, s in zip(ys, statuses) if s in ("keep", "baseline")]
    disc_x = [x for x, s in zip(xs, statuses) if s == "discard"]
    disc_y = [y for y, s in zip(ys, statuses) if s == "discard"]
    crash_x = [x for x, s in zip(xs, statuses) if s == "crash"]

    fig, ax = plt.subplots(figsize=(4.4, 3.0))

    if len(keep_x) >= 2:
        ax.plot(keep_x, keep_y, color="#2166ac", linewidth=1.5, zorder=2, alpha=0.85)
    if keep_x:
        ax.scatter(keep_x, keep_y, s=30, color="#2166ac", edgecolors="white",
                   linewidths=0.4, zorder=4, label="Accepted")
    if disc_x:
        ax.scatter(disc_x, disc_y, s=20, color="#d6604d", marker="x",
                   linewidths=0.5, zorder=3, alpha=0.4, label="Rejected")

    baseline_y = next(
        (y for y, s in zip(ys, statuses) if s == "baseline"),
        keep_y[0] if keep_y else None,
    )

    if crash_x and baseline_y:
        ax.scatter(crash_x, [baseline_y] * len(crash_x), s=24, color="#cc0000",
                   marker="x", linewidths=0.7, zorder=3, alpha=0.55, label="Crashed")

    if baseline_y:
        ax.axhline(y=baseline_y, color="#cccccc", linestyle=":", linewidth=0.6, zorder=1)

    ax.set_xlabel("Experiment round")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(title, fontweight="bold", pad=8)

    ax.set_xlim(0.5, n + 2)
    step = max(1, n // 7)
    ax.set_xticks(list(range(1, n + 1, step)))

    use_log = (log_mode == "on")
    if log_mode == "auto":
        ys_pos = [y for y in ys if y > 0]
        if ys_pos and (max(ys_pos) / min(ys_pos)) > 20:
            use_log = True
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}")
        )

    if baseline_y and keep_y and keep_y[-1] > 0 and keep_x[-1] != keep_x[0]:
        final_best = keep_y[-1]
        speedup = baseline_y / final_best

        def fmt_rt(v):
            return f"{v:.0f}" if v >= 10 else f"{v:.1f}"
        ax.annotate(
            f"{fmt_rt(baseline_y)}s → {fmt_rt(final_best)}s ({speedup:.1f}×)",
            xy=(keep_x[-1], final_best),
            xytext=(14, 0),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#2166ac",
            va="center",
        )

    ax.legend(fontsize=7, loc="best", framealpha=0.92, edgecolor="none",
              handletextpad=0.4, borderpad=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=300, facecolor="white")
    fig.savefig(f"{out_base}.pdf", facecolor="white")
    fig.savefig(f"{out_base}.svg", facecolor="none", transparent=True)
    plt.close(fig)
