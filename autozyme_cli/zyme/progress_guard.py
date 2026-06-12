"""Progress plateau guard for optimization loops.

Two tiers, both agent-visible. The shape of the message changes with how
much speedup the agent has already banked.

Tier 1 — round 5+ no-keep, gentle nudge.
  late_plateau (best_keep_pct >= 95) — "try one focused angle on the meat,
    or document genuine convergence".
  early_stuck  (best_keep_pct <  50) — "you're probably not on the real
    hot path; re-profile".
  middle band 50–95% is silent at Tier 1 (healthy progress).

Tier 2 — round 10+ no-keep, action mandate; supersedes Tier 1.
  escalate  (best_keep_pct <  50) — "small wins for 10 rounds means you're
    being timid; stop treating in-scope rewrites as out-of-scope, attack
    the real meat".
  decide    (best_keep_pct 50–90) — "real wins but stuck; 5-round
    structural-attempt budget, then verdict (resume or accept convergence)".
  terminate (best_keep_pct >= 90) — "strong wins, marginal value < token
    cost; document and exit".

A structured record is also appended to .zyme/convergence_silent.log every
time Tier 2 is in effect, regardless of console cooldown, so the operator
/ sweep coordinator can detect deeply-stuck tasks programmatically. The
historical "silent from agent" behavior was removed because withholding
the signal made the trigger inert — the agent kept burning rounds with no
new guidance — see the Tier-2 escalation/decide/terminate split for
rationale.
"""

import json
import time
from pathlib import Path

from zyme.commands._shared import _read_results_rows
from zyme.parsers.results_tsv import row_phase


LATE_PLATEAU_SPEEDUP_PCT = 95.0
EARLY_STUCK_SPEEDUP_PCT_MAX = 50.0
TIER1_NO_KEEP_ROUNDS = 5
TIER2_NO_KEEP_ROUNDS = 10
TIER2_GAIN_PCT_OVER_10 = 2.0   # absolute speedup_pct increase between best 10
                                #  rounds ago vs now; <2 → counts toward Tier-2
                                #  trigger in decide / terminate bands

# Tier-2 band boundaries (independent from Tier-1 regime thresholds above so the
# two tiers can be tuned separately). At round 10+ of no-keep:
#   best <  TIER2_ESCALATE_SPEEDUP_PCT_MAX  → escalate  (you're being timid)
#   best in [escalate, terminate)           → decide    (5-round structural budget)
#   best >= TIER2_TERMINATE_SPEEDUP_PCT     → terminate (stop wasting tokens)
TIER2_ESCALATE_SPEEDUP_PCT_MAX = 50.0
TIER2_TERMINATE_SPEEDUP_PCT = 90.0
TIER2_DECIDE_BUDGET_ROUNDS = 5      # structural-attempt budget shown to agent
TIER2_REMINDER_COOLDOWN = 5         # rounds between Tier-2 re-prints

STATE_FILE_NAME = "progress_guard.json"
SILENT_LOG_NAME = "convergence_silent.log"
TERMINAL_DECISION_STATUSES = {"keep", "discard", "crash", "rollback", "oom"}


def maybe_print_plateau_reminder(task_dir: Path) -> None:
    """Print a plateau reminder when warranted, with per-tier cooldown.

    Tier 2 (round 10+, any speedup band) — supersedes Tier 1, prints the
    band-specific escalation (escalate / decide / terminate) and also
    appends an audit entry to `.zyme/convergence_silent.log` so the operator
    / phase-3 sweep coordinator can still detect deeply-stuck tasks
    programmatically. Own cooldown: TIER2_REMINDER_COOLDOWN rounds.

    Tier 1 (round 5+, late_plateau or early_stuck only) — middle band stays
    silent. Cooldown: TIER1_NO_KEEP_ROUNDS rounds.
    """
    state = plateau_state(task_dir)
    if state is None:
        return

    prior = _read_state(task_dir) or {}
    current_round = state["current_round"]

    # ---- Tier 2 (agent-visible escalation + operator audit log) ----
    if state.get("tier2_band"):
        # Audit log appends every time Tier-2 is in effect, regardless of
        # cooldown, so operator-side sweep tooling sees the continuous signal.
        _append_silent_log(task_dir, state)

        last_tier2 = prior.get("last_tier2_warned_round")
        if (
            isinstance(last_tier2, int)
            and current_round - last_tier2 < TIER2_REMINDER_COOLDOWN
        ):
            return  # console cooldown still active; log already appended

        print()
        print(_reminder_for(state))
        new_state = {**prior, "last_tier2_warned_round": current_round}
        _write_state(task_dir, new_state)
        return

    # ---- Tier 1 (agent-visible reminder, early_stuck / late_plateau only) ----
    if state.get("regime") == "middle":
        return
    if state["tail_len"] < TIER1_NO_KEEP_ROUNDS:
        return

    last_tier1 = prior.get("last_tier1_warned_round")
    # Back-compat with state files written by the pre-Tier-2-rewrite version.
    if last_tier1 is None:
        last_tier1 = prior.get("last_warned_round")
    if (
        isinstance(last_tier1, int)
        and current_round - last_tier1 < TIER1_NO_KEEP_ROUNDS
    ):
        return

    print()
    print(_reminder_for(state))
    new_state = {**prior, "last_tier1_warned_round": current_round}
    _write_state(task_dir, new_state)


def current_plateau_reminder(task_dir: Path) -> str | None:
    """Return current plateau reminder text without updating cooldown state.

    Tier-2 (round 10+, any band) supersedes Tier-1. Middle band at Tier-1 only
    is silent — that's the healthy-progress zone.
    """
    state = plateau_state(task_dir)
    if state is None:
        return None
    if state.get("tier2_band"):
        return _reminder_for(state)
    if state.get("regime") == "middle":
        return None
    if state["tail_len"] < TIER1_NO_KEEP_ROUNDS:
        return None
    return _reminder_for(state)


def _reminder_for(state: dict) -> str:
    band = state.get("tier2_band")
    if band == "escalate":
        return format_tier2_escalate(state)
    if band == "decide":
        return format_tier2_decide(state)
    if band == "terminate":
        return format_tier2_terminate(state)
    if state.get("regime") == "early_stuck":
        return format_early_stuck_reminder(state)
    return format_plateau_reminder(state)


def plateau_state(task_dir: Path) -> dict | None:
    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        return None

    rows = [
        r for r in _read_results_rows(results_tsv)
        if row_phase(r) == "optimize"
    ]
    keep_rows = [r for r in rows if r.get("status") == "keep"]
    if not keep_rows:
        return None

    best_keep_pct = None
    for row in keep_rows:
        pct = _float(row.get("speedup_pct"))
        if pct is not None:
            best_keep_pct = pct if best_keep_pct is None else max(best_keep_pct, pct)
    if best_keep_pct is None:
        return None

    # Tier-1 regime (controls the round-5 reminder). Middle band 50-95% is
    # intentionally silent at Tier-1 — that's healthy progress.
    if best_keep_pct >= LATE_PLATEAU_SPEEDUP_PCT:
        regime = "late_plateau"
    elif best_keep_pct < EARLY_STUCK_SPEEDUP_PCT_MAX:
        regime = "early_stuck"
    else:
        regime = "middle"

    decisions = [
        r for r in rows
        if r.get("status") in TERMINAL_DECISION_STATUSES and _round_num(r) > 0
    ]
    if not decisions:
        return None

    no_keep_tail = []
    for row in reversed(decisions):
        if row.get("status") == "keep":
            break
        no_keep_tail.append(row)
    tail_len = len(no_keep_tail)

    latest = decisions[-1]
    latest_keep = keep_rows[-1]

    # Tier-2 band (round-10 escalation). Three bands keyed off current best
    # speedup — they share the round-count trigger but emit different action:
    #   escalate  → "you have small wins, you're being timid, attack primitives"
    #   decide    → "real wins, 5-round structural-only budget, then verdict"
    #   terminate → "strong wins, marginal value < token cost, exit"
    # The <2pp-gain-over-10 trigger only applies to decide/terminate; in
    # escalate the agent is barely making progress anyway so the gain signal
    # is uninformative.
    gain_pct_over_10 = None
    if len(decisions) >= 10:
        recent_10 = decisions[-10:]
        recent_best = max(
            (_float(r.get("speedup_pct")) for r in recent_10),
            default=None,
            key=lambda v: -1 if v is None else v,
        )
        prior_pool = decisions[:-10]
        prior_best = max(
            (_float(r.get("speedup_pct")) for r in prior_pool),
            default=None,
            key=lambda v: -1 if v is None else v,
        )
        if recent_best is not None and prior_best is not None:
            gain_pct_over_10 = recent_best - prior_best

    tier2_band = None
    if best_keep_pct < TIER2_ESCALATE_SPEEDUP_PCT_MAX:
        if tail_len >= TIER2_NO_KEEP_ROUNDS:
            tier2_band = "escalate"
    elif best_keep_pct < TIER2_TERMINATE_SPEEDUP_PCT:
        if tail_len >= TIER2_NO_KEEP_ROUNDS or (
            gain_pct_over_10 is not None and gain_pct_over_10 < TIER2_GAIN_PCT_OVER_10
        ):
            tier2_band = "decide"
    else:
        if tail_len >= TIER2_NO_KEEP_ROUNDS or (
            gain_pct_over_10 is not None and gain_pct_over_10 < TIER2_GAIN_PCT_OVER_10
        ):
            tier2_band = "terminate"

    return {
        "regime": regime,
        "best_keep_pct": best_keep_pct,
        "tail_len": tail_len,
        "current_round": _round_num(latest),
        "latest_keep_round": _round_num(latest_keep),
        "latest_keep_pct": _float(latest_keep.get("speedup_pct")),
        "gain_pct_over_10": gain_pct_over_10,
        "tier2_band": tier2_band,
        # legacy key kept so external readers (tests, sweep coordinator) that
        # only looked for tier2_triggered still see the bool
        "tier2_triggered": tier2_band is not None,
    }


def format_plateau_reminder(state: dict) -> str:
    """Tier-1 agent-visible reminder. Tone: not surrender — try the meat, then judge."""
    latest_keep_pct = state.get("latest_keep_pct")
    latest_keep = (
        f"latest keep was round {state['latest_keep_round']}"
        + (f" at +{latest_keep_pct:.1f}%" if latest_keep_pct is not None else "")
    )
    return (
        "[progress plateau reminder]\n"
        f"  Best kept speedup is +{state['best_keep_pct']:.1f}% and the last "
        f"{state['tail_len']} decision rounds have no keep ({latest_keep}).\n"
        "  Before continuing:\n"
        "    1. Run `zyme profile --backend cpu` if your last profile is "
        "stale (>10 rounds old). Bottleneck may have shifted.\n"
        "    2. Consider one focused angle on the meat — kernel rewrite "
        "(Rcpp / numba / Cython), call-chain restructure to drop a redundant "
        "intermediate, peer-library swap. Bigger structural changes have "
        "higher variance but real upside; small knob tweaks at this stage rarely "
        "compound.\n"
        "    3. Do NOT reach for measurement-side tricks — pre-warming on "
        "production data, allocating proportional scratch before `t0`, "
        "stubbing internal-function outputs to constants, gate-edge knob "
        "bisection. Those produce hacks that fail audit, not real gains.\n"
        "    4. Genuine convergence is a valid outcome. If you've tried the "
        "meat and the function truly has no remaining headroom (e.g. a "
        "vendored ② primitive like BLAS / UMAP / FAISS dominates remaining "
        "wall-time), that is OK. Document why in `memory/discoveries.md` "
        "and move on — don't manufacture gains."
    )


def format_early_stuck_reminder(state: dict) -> str:
    """Tier-1 early-stuck reminder. Tone: you're probably not on the hot path."""
    latest_keep_pct = state.get("latest_keep_pct")
    latest_keep = (
        f"latest keep was round {state['latest_keep_round']}"
        + (f" at +{latest_keep_pct:.1f}%" if latest_keep_pct is not None else "")
    )
    return (
        "[early-stuck reminder]\n"
        f"  Best kept speedup is only +{state['best_keep_pct']:.1f}% and the last "
        f"{state['tail_len']} decision rounds have no keep ({latest_keep}).\n"
        "  At this stage a dry spell usually means you're not on the real hot "
        "path, not that you're out of headroom. Before continuing:\n"
        "    1. Re-run `zyme profile --backend cpu`. Don't trust a stale "
        "profile — the bottleneck may not be where you think.\n"
        "    2. Confirm the function you're editing actually dominates wall "
        "time. If the top frame is an adjacent-package primitive (② in the "
        "editable-scope rules — e.g. BLAS / irlba / FAISS / a Seurat workflow "
        "step), your current scope is wrong; step back to ① / ③.\n"
        "    3. Attack the real core — pick the heaviest in-scope frame and "
        "rewrite it structurally (Rcpp / numba / Cython, call-chain "
        "restructure to drop a redundant intermediate). Knob-tweaking at the "
        "periphery rarely compounds when the best you've found is still "
        "below 50%.\n"
        "    4. Anti-hack rule still applies — no pre-warming on production "
        "data, no scratch allocation before t0, no stubbing internal-function "
        "outputs to constants."
    )


def format_tier2_escalate(state: dict) -> str:
    """Tier-2 escalation for best <50%, round 10+ no-keep.

    Tone: you have small wins and you're being timid. The actual failure mode
    at this stage is almost never headroom — it's the agent treating in-scope
    rewrites as out-of-scope, or knob-tweaking a function that isn't the hot
    path. Lift the brake; demand a structural swing.
    """
    latest_keep_pct = state.get("latest_keep_pct")
    latest_keep = (
        f"latest keep was round {state['latest_keep_round']}"
        + (f" at +{latest_keep_pct:.1f}%" if latest_keep_pct is not None else "")
    )
    return (
        "[plateau escalation — round 10, small wins, stop being timid]\n"
        f"  Best kept speedup is only +{state['best_keep_pct']:.1f}% and the last "
        f"{state['tail_len']} decision rounds have no keep ({latest_keep}).\n"
        "  At this point a long dry spell almost never means you're out of "
        "headroom. The two real failure modes:\n"
        "    1. You're knob-tweaking a function that isn't actually the hot "
        "path. Re-run `zyme profile --backend cpu` — do not trust a stale "
        "profile. If the top frame is not what you've been editing, switch "
        "targets THIS round.\n"
        "    2. You're treating in-scope rewrites as out-of-scope. Per the "
        "scope rule in the iterate prompt, the following are IN scope and "
        "you should be attacking them:\n"
        "       • Rewriting the target package's own native kernels (Rcpp / "
        "numba / Cython) — including C++ kernels already shipped inside the "
        "target package. Encouraged when profile justifies.\n"
        "       • Peer-library substitution at the same architectural level "
        "(RANN::nn2 ↔ fields::rdist, hnswlib ↔ faiss, etc.).\n"
        "       • Eliminate / reuse / reorder calls to general-purpose "
        "primitives (BLAS / NormalizeData / pca). Only the primitive's "
        "internal math is off-limits — the call pattern around it is yours.\n"
        "  Take one ambitious structural swing in the next round. The "
        "anti-hack rules still apply — no measurement-side tricks, no "
        "pre-warming on production data, no scratch allocation before t0."
    )


def format_tier2_decide(state: dict) -> str:
    """Tier-2 decision window for best 50-90%, round 10+ no-keep.

    Tone: you have real wins; you're either one structural attack away from
    a bigger jump or you've hit the genuine ceiling. Force the question
    through a 5-round structural-only budget — after that, verdict.
    """
    latest_keep_pct = state.get("latest_keep_pct")
    latest_keep = (
        f"latest keep was round {state['latest_keep_round']}"
        + (f" at +{latest_keep_pct:.1f}%" if latest_keep_pct is not None else "")
    )
    return (
        f"[plateau decision window — {TIER2_DECIDE_BUDGET_ROUNDS}-round "
        "structural budget]\n"
        f"  Best kept speedup is +{state['best_keep_pct']:.1f}% — real wins, "
        f"but {state['tail_len']} rounds with no keep. You're between two "
        "failure modes:\n"
        "    • One structural attack you haven't tried — there's a bigger "
        "jump waiting.\n"
        "    • Genuine ceiling at this scope.\n"
        f"  The next {TIER2_DECIDE_BUDGET_ROUNDS} rounds are a "
        "structural-attempt budget. Use them for:\n"
        "    1. Native kernel rewrite of the target's own algorithm "
        "(Rcpp / numba / Cython), including in-package C++ kernels.\n"
        "    2. Peer-library swap at the same architectural level.\n"
        "    3. Call-chain restructure — eliminate / reuse / reorder calls "
        "to general-purpose primitives.\n"
        "  Knob-tweak rounds (param flips, dispatch micro-opts) are still "
        "allowed but do not change the verdict.\n"
        f"  After {TIER2_DECIDE_BUDGET_ROUNDS} structural attempts:\n"
        "    - Any keep → resume normal loop.\n"
        "    - All attempts fail → accept convergence at current best, write "
        "`memory/discoveries.md` naming the ceiling (which call dominates, "
        "what constraint blocks further wins), and exit."
    )


def format_tier2_terminate(state: dict) -> str:
    """Tier-2 terminate for best ≥90%, round 10+ no-keep.

    Tone: you've banked a strong result. 10 rounds of failure at this level
    means marginal value < token cost. Stop, document, ship.
    """
    latest_keep_pct = state.get("latest_keep_pct")
    latest_keep = (
        f"latest keep was round {state['latest_keep_round']}"
        + (f" at +{latest_keep_pct:.1f}%" if latest_keep_pct is not None else "")
    )
    gain = state.get("gain_pct_over_10")
    gain_clause = (
        f" and cumulative gain over the last 10 rounds is +{gain:.1f}pp"
        if gain is not None and gain < TIER2_GAIN_PCT_OVER_10
        else ""
    )
    return (
        "[plateau terminate — strong wins, marginal value exhausted]\n"
        f"  Best kept speedup is +{state['best_keep_pct']:.1f}% "
        f"({latest_keep}){gain_clause}. The last {state['tail_len']} "
        "decision rounds have no keep. At this level, additional rounds "
        "cost more in tokens than they return in speed.\n"
        "  Action:\n"
        "    1. `best` is already locked from the latest keep — no further "
        "`zyme accept` is needed.\n"
        "    2. Write `memory/discoveries.md`: which call dominates the "
        "remaining wall-time, what constraint blocks further wins (a "
        "general-purpose primitive's internal math, an irreducible "
        "algorithmic step, a vendored kernel you tried and could not beat).\n"
        "    3. Exit the loop. Move on.\n"
        "  Do not manufacture gains. Pre-warming on production data, "
        "allocating scratch before t0, stubbing internal-function outputs "
        "to constants, gate-edge knob bisection — these WILL fail audit "
        "and erase the strong result you already have."
    )


def _append_silent_log(task_dir: Path, state: dict) -> None:
    """Tier-2 operator audit log — runs in parallel with the agent-visible
    Tier-2 escalation print.

    The agent now sees the Tier-2 reminder (escalate / decide / terminate)
    directly via maybe_print_plateau_reminder. This log is the machine-
    readable counterpart: one JSON line per round in which Tier-2 is
    triggered, used by sweep coordinators to detect deeply-stuck tasks and
    by phase-3 audit to spot patterns across runs.

    Despite the legacy SILENT_LOG_NAME the entries are no longer hidden
    from the agent in any meaningful sense — the variable / filename are
    retained for back-compat with operator scripts that read the file.
    """
    path = task_dir / ".zyme" / SILENT_LOG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_round": state["current_round"],
        "best_keep_pct": state["best_keep_pct"],
        "tail_len": state["tail_len"],
        "gain_pct_over_10": state.get("gain_pct_over_10"),
        "reason": (
            "tier2_consecutive_discards" if state["tail_len"] >= TIER2_NO_KEEP_ROUNDS
            else "tier2_low_gain_over_10"
        ),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _state_path(task_dir: Path) -> Path:
    return task_dir / ".zyme" / STATE_FILE_NAME


def _read_state(task_dir: Path):
    path = _state_path(task_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_state(task_dir: Path, state: dict) -> None:
    path = _state_path(task_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_num(row: dict) -> int:
    try:
        return int(float(row.get("round") or 0))
    except (TypeError, ValueError):
        return 0
