"""Evidence-card rendering: stderr summary the agent reads.

Produces:
    [profile] backend=<b> kind=<k> unit=<u> total_wall=<X>s tier=<t>
    [profile] Top N hotspots:
      1. <label>            self=<S>s (<P>%)  <backend-specific extras>
      2. ...
    [profile] Notes:
      - <note 1>
      - <note 2>
    [profile] Artifacts: profile_history/<run>/<raw>, ...
    [profile] Profile dir: profile_history/<run>

The format is line-oriented and grep-friendly so the agent can pull
specific facts deterministically.
"""
import sys


TOP_N_RENDERED = 10


def evidence_card(profile_data: dict, profile_dir: str | None = None) -> None:
    """Print the human/agent-readable evidence card to stderr."""
    lines = []
    backend = profile_data.get("backend", "?")
    lang = profile_data.get("lang", "?")
    tier = profile_data.get("tier", "?")
    totals = profile_data.get("totals") or {}
    hotspots = profile_data.get("hotspots") or []
    actionable = profile_data.get("actionable_hotspots") or []
    call_chains = profile_data.get("call_chains") or []
    notes = profile_data.get("notes") or []
    artifacts = profile_data.get("artifacts") or {}

    header_bits = [f"backend={backend}", f"lang={lang}", f"tier={tier}"]
    if totals.get("wall_s") is not None:
        header_bits.append(f"wall={totals['wall_s']:.3f}s")
    if totals.get("cpu_s") is not None:
        header_bits.append(f"cpu={totals['cpu_s']:.3f}s")
    if totals.get("peak_mb"):
        header_bits.append(f"peak={totals['peak_mb']:.1f}MB")
    lines.append("[profile] " + " ".join(header_bits))

    # Layer breakdown: rendered FIRST when present so the agent reads
    # "where is the time going by editable scope" before drilling into any
    # specific function. Each line: layer_group | self_time | pct.
    layer_breakdown = profile_data.get("layer_breakdown") or []
    if layer_breakdown:
        lines.append(
            "[profile] Layer breakdown "
            "(read FIRST — answers 'what scope is editable?'):"
        )
        for lb in layer_breakdown:
            lines.append(
                f"  {lb.get('layer_group','?'):<24} "
                f"self={lb.get('self_time_s',0.0):>8.3f}s "
                f"({lb.get('pct',0.0):>5.1f}%)"
            )

    # Per-layer top: shows the dominant function in each layer group,
    # even when the global top hotspots are dominated by one layer (e.g.
    # primitives 70%). Lets the agent see the editable hotspot in `task`
    # layer that would otherwise be buried in the long tail.
    per_layer_top = profile_data.get("per_layer_top") or {}
    if per_layer_top:
        # Render in deterministic layer-priority order
        layer_order = ["task", "library", "base-r", "base-py",
                       "primitive", "builtin", "anonymous", "unknown"]
        order_keys = [k for k in layer_order if k in per_layer_top] + \
                     [k for k in per_layer_top if k not in layer_order]
        lines.append("[profile] Per-layer top (dominant function within each layer):")
        for lg in order_keys:
            entries = per_layer_top.get(lg) or []
            if not entries:
                continue
            top = entries[0]
            n_calls = top.get("n_calls")
            ncs = f" n_calls={n_calls}" if n_calls else ""
            lines.append(
                f"  [{lg:<10}] {(top.get('label') or '?')[:48]:<48} "
                f"self={top.get('self_pct',0.0):>5.2f}%{ncs}"
            )

    # Python/native split — when Scalene backend provides py% / native%
    # per hotspot, show the aggregate so the agent knows at a glance how
    # much time is Python dispatch overhead vs compiled compute. Works
    # universally with any C extension (numpy, scipy, torch, TF, numba).
    pns = profile_data.get("python_native_split")
    if pns:
        lines.append(
            "[profile] Python/native split "
            "(how much is Python overhead vs compiled C compute?):"
        )
        lines.append(
            f"  python  {pns['python_s']:>8.3f}s ({pns['python_pct']:>5.1f}%)  "
            "← dispatch, glue, interpreter overhead"
        )
        lines.append(
            f"  native  {pns['native_s']:>8.3f}s ({pns['native_pct']:>5.1f}%)  "
            "← compiled C/Fortran/BLAS compute"
        )
        if pns.get("system_pct", 0) > 1:
            lines.append(
                f"  system  {pns['system_s']:>8.3f}s ({pns['system_pct']:>5.1f}%)"
            )

    if actionable:
        lines.append(
            f"[profile] Top {min(TOP_N_RENDERED, len(actionable))} actionable targets "
            "(optimize these first; raw hotspots below are evidence):"
        )
        for h in actionable[:TOP_N_RENDERED]:
            lines.append("  " + _format_hotspot(h, backend))
        if hotspots:
            lines.append(
                f"[profile] Raw profiler hotspots ({min(5, len(hotspots))} shown; "
                "may be runtime/native frames, not direct patch targets):"
            )
            for h in hotspots[:5]:
                lines.append("  " + _format_hotspot(h, backend))
    elif hotspots:
        lines.append(f"[profile] Top {min(TOP_N_RENDERED, len(hotspots))} hotspots:")
        for h in hotspots[:TOP_N_RENDERED]:
            lines.append("  " + _format_hotspot(h, backend))
    else:
        lines.append("[profile] No hotspots extracted (raw artifact may be missing or empty)")

    # Override summary: deterministic, fork-safe per-override timing.
    # Always shown when present — it's complementary signal to hotspots,
    # especially valuable on fast-fork tasks where profiler hotspots are
    # parent-side noise.
    overrides = profile_data.get("override_summary") or []
    if overrides:
        lines.append(f"[profile] Override timing ({len(overrides)} tracked):")
        for ov in overrides[:5]:
            workers_suffix = (
                f"  [aggregated across {ov['n_workers']} workers]"
                if ov.get("n_workers", 1) > 1 else ""
            )
            lines.append(
                f"  {ov['name'][:50]:<50}  "
                f"calls={ov['calls']:>6}  "
                f"total={ov['total_s']:>8.3f}s  "
                f"mean={ov['mean_s']*1000:>8.3f}ms"
                f"{workers_suffix}"
            )

    markers = profile_data.get("override_markers") or []
    if markers:
        lines.append(f"[profile] Active override markers without timing ({len(markers)}):")
        for marker in markers[:5]:
            lines.append(f"  {marker.get('name', '?')}")

    if call_chains:
        lines.append(
            f"[profile] Representative call chains ({min(3, len(call_chains))} shown):"
        )
        for chain in call_chains[:3]:
            lines.append("  " + _format_call_chain(chain))

    if notes:
        lines.append("[profile] Notes:")
        for n in notes:
            lines.append(f"  - {n}")

    if artifacts:
        art_strs = []
        for kind, path in artifacts.items():
            art_strs.append(f"{kind}={path}")
        lines.append("[profile] Artifacts: " + ", ".join(art_strs))

    if profile_dir:
        lines.append(f"[profile] Profile dir: {profile_dir}")

    print("\n".join(lines), file=sys.stderr, flush=True)


def _format_hotspot(h: dict, backend: str) -> str:
    rank = h.get("rank", "?")
    label = h.get("label") or "?"
    self_s = h.get("self_time_s")
    self_pct = h.get("self_pct")
    calls = h.get("calls")
    raw = h.get("raw") or {}

    label = label[:60].ljust(60)
    parts = [f"{rank:2d}.", label]

    if self_s is not None:
        parts.append(f"self={self_s:7.3f}s")
    if self_pct is not None:
        parts.append(f"({self_pct:5.1f}%)")
    if calls:
        parts.append(f"calls={calls}")

    # Enrichment fields (schema v2): layer tag + per-function call count.
    # n_calls (exact, cProfile) or n_calls_est (approx, Rprof stack walks).
    layer = h.get("layer")
    if layer:
        parts.append(f"layer={layer}")
    n_calls = h.get("n_calls")
    if n_calls is None:
        n_calls = h.get("n_calls_est")
    if n_calls is not None and not calls:  # avoid double-counting cProfile
        parts.append(f"n_calls={n_calls}")

    # Source file:line — surfaced by native backend when the hotspot's func
    # was resolved against <task>/upstream_repo/. Lets the agent open the
    # right file directly instead of grepping the upstream tree by symbol.
    source_loc = raw.get("source_location")
    if source_loc:
        parts.append(f"src={source_loc}")

    source = raw.get("source")
    if source:
        parts.append(f"source={source}")
    if source == "override_summary":
        mean_s = raw.get("mean_s")
        workers = raw.get("n_workers")
        if mean_s is not None:
            parts.append(f"mean={float(mean_s) * 1000:.3f}ms")
        if workers and workers > 1:
            parts.append(f"workers={workers}")
    elif source == "override_marker":
        parts.append("timing=unavailable")

    # Backend-specific extras: only the ones that add new info beyond
    # self_time / self_pct.
    if backend == "full":
        py_pct = raw.get("cpu_python_pct")
        nt_pct = raw.get("cpu_native_pct")
        peak_mb = raw.get("mem_peak_mb")
        if py_pct is not None and nt_pct is not None:
            parts.append(f"py={py_pct:4.1f}% native={nt_pct:4.1f}%")
        if peak_mb:
            parts.append(f"peak_mb={peak_mb:.1f}")
    elif backend == "mem":
        bytes_ = raw.get("alloc_bytes")
        if bytes_:
            parts.append(f"alloc={_humanize_bytes(bytes_)}")
        cnt = raw.get("alloc_count")
        if cnt:
            parts.append(f"n={cnt}")
    elif backend == "cpu" and raw.get("mem_total_mb") not in (None, ""):
        parts.append(f"mem={raw['mem_total_mb']:.1f}MB")

    return " ".join(parts)


def _format_call_chain(chain: dict) -> str:
    rank = chain.get("rank", "?")
    owner = chain.get("nearest_actionable_frame")
    owner_part = f" owner={owner}" if owner else ""
    frames = [str(f) for f in (chain.get("chain") or [])]
    if len(frames) > 5:
        shown = [frames[0], "...", *frames[-3:]]
    else:
        shown = frames
    body = " -> ".join(f[:50] for f in shown) if shown else str(chain.get("leaf") or "?")
    samples = chain.get("samples")
    sample_part = f" samples={samples}" if samples is not None else ""
    return f"{rank}. {body}{owner_part}{sample_part}"


def _humanize_bytes(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
