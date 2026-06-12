"""diff.py — `zyme profile --diff <a> <b>`: compare two profile.json snapshots.

Two questions an agent asks every iteration:
  1. "Did my last patch change the hotspot ranking?" — hotspot delta
  2. "Did the override I just installed get faster?" — override_summary delta

This subcommand answers both by reading two profile.json files, typically
from `profile_history/<run-id>/profile.json`.

Output: text table to stderr/stdout (or JSON via --json).

Schema-wise we expect the canonical schema_version="1" produced by
parsers.normalize / native.parse_and_normalize. Cross-backend diffs
(e.g. cpu vs full) are allowed but flagged in notes — backend choice
affects which frames are captured, so a function appearing in one but
not the other may just be a backend-coverage difference, not a real
optimization gain/loss.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_profile(spec: str) -> tuple[dict, str]:
    """Resolve `spec` to a profile dict + a display label.

    `spec` can be:
      - absolute or relative path to a profile.json file
      - path to a profile_history/<run-id>/ directory
      - path to a legacy profile_history/*.json file
      - "history:<short_substring>" — fuzzy match against profile_history/
        filenames in cwd's task. Picks the most recent match.
      - "current" / "head" — alias for latest profile_history profile
    """
    p = Path(spec)
    if spec in ("current", "head"):
        p = _latest_history_profile()
    elif spec.startswith("history:"):
        needle = spec[len("history:"):].lower()
        hist = Path("profile_history")
        if not hist.is_dir():
            raise FileNotFoundError(
                f"history: spec given but no profile_history/ in cwd"
            )
        candidates = sorted(
            (f for f in hist.iterdir() if needle in f.name.lower()),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"no profile_history file matched substring {needle!r}"
            )
        p = _profile_path_from_history_entry(candidates[0])
    elif p.is_dir():
        p = p / "profile.json"
    if not p.exists():
        # Back-compat for pre-profile_history-dir tasks.
        if spec in ("current", "head"):
            legacy = Path("pipeline") / "profile.json"
            if legacy.exists():
                p = legacy
            else:
                raise FileNotFoundError(
                    f"profile not found: {spec} (latest profile_history missing; "
                    f"legacy {legacy} missing)"
                )
        else:
            raise FileNotFoundError(f"profile not found: {spec} (resolved to {p})")
    return json.loads(p.read_text()), str(p)


def _latest_history_profile() -> Path:
    hist = Path("profile_history")
    if not hist.is_dir():
        return Path("pipeline") / "profile.json"
    candidates = []
    for entry in hist.iterdir():
        profile = _profile_path_from_history_entry(entry)
        if profile.exists():
            candidates.append(profile)
    if not candidates:
        return Path("pipeline") / "profile.json"
    return sorted(candidates, key=lambda f: f.stat().st_mtime, reverse=True)[0]


def _profile_path_from_history_entry(entry: Path) -> Path:
    if entry.is_dir():
        return entry / "profile.json"
    return entry


def _hotspot_index(profile_data: dict) -> dict[str, dict]:
    """Build label → hotspot dict for diff lookup.

    Labels are normalized: backend-specific prefixes (e.g. lib name
    in native) are kept since they're meaningful identifiers; we don't
    try to normalize across backends.
    """
    return {
        h["label"]: h
        for h in (profile_data.get("hotspots") or [])
        if h.get("label")
    }


def _override_index(profile_data: dict) -> dict[str, dict]:
    return {
        o["name"]: o
        for o in (profile_data.get("override_summary") or [])
        if o.get("name")
    }


def _fmt_delta(a: float | None, b: float | None,
               unit: str = "s", precision: int = 3) -> str:
    """Format a delta. Positive = b is larger, negative = b is smaller."""
    if a is None or b is None:
        return "—"
    delta = b - a
    pct = (delta / a * 100.0) if a > 0 else None
    sign = "+" if delta >= 0 else ""
    pct_str = f" ({sign}{pct:.1f}%)" if pct is not None else ""
    return f"{sign}{delta:.{precision}f}{unit}{pct_str}"


def diff_profiles(a_data: dict, b_data: dict,
                  a_label: str = "A", b_label: str = "B") -> dict:
    """Compute structured diff. Returns a dict for both rendering and JSON."""
    a_hot = _hotspot_index(a_data)
    b_hot = _hotspot_index(b_data)
    a_ov = _override_index(a_data)
    b_ov = _override_index(b_data)

    # Hotspot deltas: appeared / disappeared / both-present (with delta).
    appeared = [b_hot[k] for k in b_hot if k not in a_hot]
    disappeared = [a_hot[k] for k in a_hot if k not in b_hot]
    common: list[dict] = []
    for k in a_hot:
        if k in b_hot:
            ah, bh = a_hot[k], b_hot[k]
            common.append({
                "label": k,
                "a_pct": ah.get("self_pct"),
                "b_pct": bh.get("self_pct"),
                "a_self_s": ah.get("self_time_s"),
                "b_self_s": bh.get("self_time_s"),
                "a_rank": ah.get("rank"),
                "b_rank": bh.get("rank"),
            })
    common.sort(key=lambda x: (
        -(x["b_pct"] or 0) if x["b_pct"] is not None else 0,
        -(x["a_pct"] or 0) if x["a_pct"] is not None else 0,
    ))

    # Override deltas: per-name comparison of calls/total/mean.
    override_diffs: list[dict] = []
    for name in sorted(set(a_ov) | set(b_ov)):
        a, b = a_ov.get(name), b_ov.get(name)
        override_diffs.append({
            "name": name,
            "a_calls": a.get("calls") if a else None,
            "b_calls": b.get("calls") if b else None,
            "a_total_s": a.get("total_s") if a else None,
            "b_total_s": b.get("total_s") if b else None,
            "a_mean_s": a.get("mean_s") if a else None,
            "b_mean_s": b.get("mean_s") if b else None,
            "a_workers": a.get("n_workers") if a else None,
            "b_workers": b.get("n_workers") if b else None,
        })

    # Backend / tier metadata for context.
    notes = []
    if a_data.get("backend") != b_data.get("backend"):
        notes.append(
            f"WARNING: cross-backend diff "
            f"({a_data.get('backend')} → {b_data.get('backend')}). "
            f"Hotspot label sets may differ structurally; treat appeared/"
            f"disappeared as backend-coverage differences, not real changes."
        )
    if a_data.get("tier") != b_data.get("tier"):
        notes.append(
            f"WARNING: tier mismatch "
            f"({a_data.get('tier')} → {b_data.get('tier')}). "
            f"Self-times not comparable."
        )

    return {
        "a_label": a_label,
        "b_label": b_label,
        "a_meta": {k: a_data.get(k) for k in
                   ("backend", "lang", "tier", "hypothesis", "timestamp")},
        "b_meta": {k: b_data.get(k) for k in
                   ("backend", "lang", "tier", "hypothesis", "timestamp")},
        "a_totals": a_data.get("totals") or {},
        "b_totals": b_data.get("totals") or {},
        "common_hotspots": common,
        "appeared_in_b": appeared,
        "disappeared_from_a": disappeared,
        "override_diffs": override_diffs,
        "notes": notes,
    }


def render_diff(d: dict) -> str:
    """Format the structured diff for terminal display."""
    out: list[str] = []
    a_lab, b_lab = d["a_label"], d["b_label"]
    out.append(f"[diff] {a_lab}  →  {b_lab}")
    am, bm = d["a_meta"], d["b_meta"]
    out.append(
        f"  {a_lab}: backend={am.get('backend')} tier={am.get('tier')} "
        f"hypothesis={(am.get('hypothesis') or '')[:40]!r}"
    )
    out.append(
        f"  {b_lab}: backend={bm.get('backend')} tier={bm.get('tier')} "
        f"hypothesis={(bm.get('hypothesis') or '')[:40]!r}"
    )
    a_wall = (d["a_totals"] or {}).get("wall_s")
    b_wall = (d["b_totals"] or {}).get("wall_s")
    if a_wall is not None and b_wall is not None:
        out.append(
            f"  wall: {a_wall:.3f}s → {b_wall:.3f}s "
            f"(delta {_fmt_delta(a_wall, b_wall)})"
        )

    for n in d["notes"]:
        out.append(f"  ⚠ {n}")

    common = d["common_hotspots"]
    if common:
        out.append("")
        out.append("[diff] Hotspots present in BOTH (sorted by B's pct):")
        out.append(f"  {'label':<55}  {'rank':>10}  {'self_pct':>20}")
        for h in common[:10]:
            rank_str = f"{h['a_rank']}→{h['b_rank']}"
            pct_str = (
                f"{h['a_pct']:.1f}% → {h['b_pct']:.1f}%"
                if h["a_pct"] is not None and h["b_pct"] is not None
                else "—"
            )
            out.append(f"  {h['label'][:55]:<55}  {rank_str:>10}  {pct_str:>20}")

    if d["appeared_in_b"]:
        out.append("")
        out.append(f"[diff] Appeared in {b_lab} (not in {a_lab}):")
        for h in d["appeared_in_b"][:5]:
            out.append(f"  + {h.get('label')[:60]}  "
                       f"({h.get('self_pct')}% self)")

    if d["disappeared_from_a"]:
        out.append("")
        out.append(f"[diff] Disappeared from {a_lab} (gone in {b_lab}):")
        for h in d["disappeared_from_a"][:5]:
            out.append(f"  - {h.get('label')[:60]}  "
                       f"({h.get('self_pct')}% self)")

    if d["override_diffs"]:
        out.append("")
        out.append("[diff] Override timing changes:")
        out.append(f"  {'name':<45}  {'calls':>15}  {'total':>20}  {'mean':>20}")
        for od in d["override_diffs"]:
            calls_str = f"{od['a_calls']}→{od['b_calls']}"
            total_str = _fmt_delta(od["a_total_s"], od["b_total_s"])
            mean_str = _fmt_delta(od["a_mean_s"], od["b_mean_s"], precision=4)
            out.append(f"  {od['name'][:45]:<45}  {calls_str:>15}  "
                       f"{total_str:>20}  {mean_str:>20}")
    return "\n".join(out)


def cmd_profile_diff(args) -> None:
    """Entry point for `zyme profile --diff a b`."""
    try:
        a_data, a_path = _load_profile(args.diff_a)
        b_data, b_path = _load_profile(args.diff_b)
    except FileNotFoundError as e:
        from zyme.utils import die
        die(str(e))

    diff = diff_profiles(
        a_data, b_data,
        a_label=Path(a_path).stem[:30],
        b_label=Path(b_path).stem[:30],
    )

    if getattr(args, "json", False):
        print(json.dumps(diff, indent=2, default=str))
    else:
        print(render_diff(diff), file=sys.stderr)
