"""`zyme status` — one-screen task state summary."""


from zyme.utils import (
    git, task_dir_from_args, zyme_state,
)
from zyme.parsers.task_yaml import parse_datasets
from zyme.parsers.results_tsv import (
    row_phase, get_baseline_speed,
    find_best_speed_at_dataset, best_cv_at_dataset,
    phase_cv_at_dataset, count_decision_rounds,
)
from zyme.commands._shared import _read_results_rows


# Per-phase round budgets. Mirrors the caps documented in `zyme run --phase`.
# `all` falls back to the optimize cap because mixed counters don't share a
# meaningful ceiling — the per-phase rows above are what readers should trust.
_PHASE_CAPS = {"optimize": 100, "validate": 30, "memory": 50, "all": 100}


def _phase_cap(phase):
    return _PHASE_CAPS.get(phase, 100)


# Severity ranking for audit verdict rollup. First match wins ⇒ verdict is the
# highest-severity finding's severity. Empty findings ⇒ PASS.
_AUDIT_SEV_ORDER = ("FAIL", "LIKELY_HACK", "LIKELY_HACK_INVITED", "WEAK")


def _read_validate_tsv(task_dir):
    """Return the latest validate run per phase from `validate.tsv`.

    Returns {phase: {timestamp, agent, model, sev_counts, report_path}} or
    {} if no validate.tsv exists. A single TSV may carry many invocations
    (append-only); we group rows by phase and keep the most recent
    (timestamp, agent, model) invocation per phase.
    """
    tsv = task_dir / "validate.tsv"
    if not tsv.is_file():
        return {}
    try:
        with tsv.open(encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            if "phase" not in header or "timestamp" not in header or "severity" not in header:
                return {}
            cols = {name: i for i, name in enumerate(header)}
            rows = []
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    continue
                rows.append({name: parts[idx] for name, idx in cols.items()})
    except OSError:
        return {}

    # Pass 1: find the latest key per phase.
    latest_key = {}
    for r in rows:
        phase = r.get("phase", "")
        ts = r.get("timestamp", "")
        if not phase or not ts:
            continue
        key = (ts, r.get("validator_agent", ""), r.get("validator_model", ""))
        if phase not in latest_key or key > latest_key[phase]:
            latest_key[phase] = key

    # Pass 2: tally severity counts for the rows matching the latest key.
    out = {}
    for phase, key in latest_key.items():
        sev_counts = {}
        report_path = ""
        for r in rows:
            if r.get("phase") != phase:
                continue
            rkey = (r.get("timestamp", ""), r.get("validator_agent", ""),
                    r.get("validator_model", ""))
            if rkey != key:
                continue
            sev = (r.get("severity") or "").strip()
            if sev:
                sev_counts[sev] = sev_counts.get(sev, 0) + 1
            if not report_path:
                report_path = r.get("report_path", "")
        out[phase] = {
            "timestamp": key[0],
            "agent": key[1],
            "model": key[2],
            "sev_counts": sev_counts,
            "report_path": report_path,
        }
    return out


def _audit_verdict(sev_counts):
    """Rollup: highest-severity finding wins; empty ⇒ PASS."""
    for s in _AUDIT_SEV_ORDER:
        if sev_counts.get(s, 0):
            return s
    return "PASS"


def _print_audit_section(task_dir):
    """One-screen audit summary. Reads validate.tsv. Read-only."""
    audits = _read_validate_tsv(task_dir)
    if not audits:
        print("Audit: not run  (use `zyme validate iterate` to audit hack-hardness)")
        print()
        return
    if len(audits) == 1:
        phase, a = next(iter(audits.items()))
        verdict = _audit_verdict(a["sev_counts"])
        n = sum(a["sev_counts"].values())
        sev_str = ", ".join(f"{s}: {a['sev_counts'][s]}"
                            for s in _AUDIT_SEV_ORDER if a["sev_counts"].get(s, 0))
        meta = f"{a['agent']}/{a['model']}" if a["agent"] or a["model"] else "—"
        print(f"Audit ({phase}): verdict {verdict}  findings {n}"
              + (f" ({sev_str})" if sev_str else ""))
        print(f"  ran {a['timestamp']}  {meta}")
        if a["report_path"]:
            print(f"  report: {a['report_path']}")
        print()
        return
    # Both init and iterate present.
    print("Audit:")
    print(f"  {'phase':<8} {'verdict':<12} {'findings':<10} ran (agent/model)  report")
    for phase in ("init", "iterate"):
        if phase not in audits:
            continue
        a = audits[phase]
        verdict = _audit_verdict(a["sev_counts"])
        n = sum(a["sev_counts"].values())
        meta = f"{a['agent']}/{a['model']}" if a["agent"] or a["model"] else "—"
        ts = a["timestamp"]
        report = a["report_path"] or "—"
        print(f"  {phase:<8} {verdict:<12} {n:<10} {ts}  {meta}  {report}")
    print()




def cmd_status(args):
    """Print a one-screen snapshot of where the task is right now.

    Read-only, no side effects. Answers the questions agents repeatedly
    grep `results.tsv` for: budget remaining, current best per tier,
    per-tier speedup vs baseline, rolling CV, the patch stack, and the
    last few decisions. Designed to be cheap to run between rounds.
    """
    task_dir = task_dir_from_args(args)
    results_tsv = task_dir / "results.tsv"
    z, best_ref, _ = zyme_state(task_dir)

    head_sha = git("rev-parse", "HEAD", cwd=task_dir, check=False)
    head_short = head_sha[:7] if head_sha else "—"
    porcelain = git("status", "--porcelain", cwd=task_dir, check=False)
    porcelain_lines = [ln for ln in porcelain.splitlines() if ln.strip()]
    has_modified = any(not ln.startswith("??") for ln in porcelain_lines)
    has_untracked = any(ln.startswith("??") for ln in porcelain_lines)
    if has_modified and has_untracked:
        head_clean = "DIRTY (modified+untracked)"
    elif has_modified:
        head_clean = "DIRTY (modified)"
    elif has_untracked:
        # Untracked-only is benign for zyme operations (rev-parse, results.tsv,
        # best.ref) — don't flag with the alarming `DIRTY` label that suggests
        # uncommitted edits to tracked files.
        head_clean = "clean (untracked-only)"
    else:
        head_clean = "clean"

    best_short = "—"
    best_long = ""
    if best_ref.exists():
        best_long = best_ref.read_text().strip()
        if best_long:
            best_short = best_long[:7]

    print(f"Task: {task_dir.name}")
    print(f"HEAD: {head_short} ({head_clean})")
    print(f"Best: {best_short}" + ("  (no best yet)" if best_short == "—" else ""))
    print()

    _print_audit_section(task_dir)

    if not results_tsv.exists():
        print("(no results.tsv yet — no rounds recorded)")
        return

    rows = _read_results_rows(results_tsv)
    show_phase = getattr(args, "phase", "optimize")
    if show_phase == "all":
        narrative_rows = rows
    else:
        narrative_rows = [r for r in rows if row_phase(r) == show_phase]

    decisions = count_decision_rounds(results_tsv, phase=show_phase if show_phase != "all" else "optimize")
    cap = _phase_cap(show_phase)
    if show_phase == "validate":
        print(f"Scale-fix rounds: {decisions}/{cap}  ({max(0, cap - decisions)} remaining)")
    elif show_phase == "memory":
        print(f"Memory rounds: {decisions}/{cap}  ({max(0, cap - decisions)} remaining)")
    else:
        label = "Decision rounds" if show_phase == "optimize" else "Decision rounds (all phases)"
        print(f"{label}: {decisions}/{cap}  ({max(0, cap - decisions)} remaining)")

    # Phase 3 progress callout — surfaces validate-phase activity in the default
    # (optimize-only) view so the user knows it exists without a flag.
    if show_phase == "optimize":
        validate_decisions = count_decision_rounds(results_tsv, phase="validate")
        if validate_decisions > 0:
            print(f"  + Phase 3 (validate-scaling): {validate_decisions}/30 scale-fix rounds  "
                  f"[`zyme status --phase validate` for detail]")

    counts = {"keep": 0, "discard": 0, "crash": 0, "pending": 0, "rollback": 0}
    for r in narrative_rows:
        s = r.get("status", "")
        if s in counts:
            counts[s] += 1
    print(
        f"Outcomes: keeps={counts['keep']}  discards={counts['discard']}  "
        f"crashes={counts['crash']}  pending={counts['pending']}"
        + (f"  rollbacks={counts['rollback']}" if counts['rollback'] else "")
    )
    if show_phase == "optimize":
        try:
            from zyme import progress_guard
            reminder = progress_guard.current_plateau_reminder(task_dir)
        except Exception:
            reminder = None
        if reminder:
            print()
            print(reminder)
    print()

    # Per-tier table. For validate phase, measurements live in verify.tsv (not
    # results.tsv), so we pre-read verify.tsv once and use its data in the loop.
    tiers = parse_datasets(task_dir / "task.yaml")

    # Build per-tier verify summary from verify.tsv (validate phase view only).
    # Keys: tier name → {t1_speeds, n_pass, n_fail, n_crash, n_cells, commit}.
    verify_summary: dict = {}
    if show_phase == "validate":
        verify_tsv = task_dir / "verify.tsv"
        if verify_tsv.exists():
            _vlines = verify_tsv.read_text().splitlines()
            if _vlines:
                _vh = _vlines[0].split("\t")
                _vc = {_n: _i for _i, _n in enumerate(_vh)}
                if {"thread", "tier", "speed_sec", "status"}.issubset(_vc):
                    _raw, _all_commits = [], []
                    for _vl in _vlines[1:]:
                        _vp = _vl.split("\t")
                        if len(_vp) < len(_vh):
                            continue
                        _vph = _vp[_vc["phase"]] if "phase" in _vc else "optimize"
                        if _vph != "validate":
                            continue
                        _vcm = _vp[_vc["commit"]] if "commit" in _vc else ""
                        _vt = _vp[_vc["tier"]]
                        try:
                            _vth = int(_vp[_vc["thread"]])
                            _vsp = float(_vp[_vc["speed_sec"]])
                        except ValueError:
                            continue
                        _vst = _vp[_vc["status"]]
                        _raw.append({"commit": _vcm, "tier": _vt, "thread": _vth,
                                     "speed": _vsp, "status": _vst})
                        if _vcm and _vcm not in _all_commits:
                            _all_commits.append(_vcm)
                    _latest_cm = _all_commits[-1] if _all_commits else None
                    from collections import defaultdict as _dd
                    _by_tier = _dd(list)
                    for _r in _raw:
                        if _r["commit"] == _latest_cm:
                            _by_tier[_r["tier"]].append(_r)
                    for _vtier, _vrows in _by_tier.items():
                        _t1sp = [_r["speed"] for _r in _vrows
                                 if _r["thread"] == 1 and _r["speed"] > 0]
                        verify_summary[_vtier] = {
                            "t1_speeds": sorted(_t1sp),
                            "n_pass": sum(1 for _r in _vrows if _r["status"] == "pass"),
                            "n_fail": sum(1 for _r in _vrows if _r["status"] == "fail"),
                            "n_crash": sum(1 for _r in _vrows if _r["status"] == "crash"),
                            "n_cells": len(_vrows),
                            "commit": _latest_cm,
                        }

    if tiers:
        print("Per tier:")
        print(f"  {'tier':<10} {'baseline':>10} {'best(t=1)':>10} {'speedup':>10} {'CV':>8} {'n':>4}")
        for t in tiers:
            name = t["name"]
            baseline = get_baseline_speed(task_dir, name, thread=1)

            if show_phase == "validate":
                # Use verify.tsv thread=1 data for the best/CV/n columns.
                vsum = verify_summary.get(t["tier"])
                if vsum and vsum["t1_speeds"]:
                    import statistics as _stat
                    best_med = _stat.median(vsum["t1_speeds"])
                    cv = (_stat.stdev(vsum["t1_speeds"]) / _stat.mean(vsum["t1_speeds"]) * 100.0
                          if len(vsum["t1_speeds"]) >= 2 else None)
                    n = vsum["n_cells"]
                elif vsum:
                    best_med, cv, n = None, None, vsum["n_cells"]
                else:
                    best_med, cv, n = phase_cv_at_dataset(task_dir, name, phase="validate")
            else:
                # In optimize phase, best.ref points at the Phase 2 keep —
                # within the current active mode.
                best_med = find_best_speed_at_dataset(task_dir, name, thread=1)
                _, cv, n = best_cv_at_dataset(task_dir, name, thread=1)

            base_str = f"{baseline:.3f}s" if baseline else "—"
            best_str = f"{best_med:.3f}s" if best_med else "—"
            if baseline and best_med:
                pct = (1 - best_med / baseline) * 100.0
                speedup_str = f"{pct:+.1f}%"
            else:
                speedup_str = "—"
            cv_str = f"{cv:.1f}%" if cv is not None else "—"
            print(f"  {t['tier']:<10} {base_str:>10} {best_str:>10} {speedup_str:>10} {cv_str:>8} {n:>4}")
        print()

    # Verify matrix summary (validate phase only) — shows per-tier PASS/FAIL
    # aggregates from verify.tsv so the agent doesn't have to grep the file.
    if show_phase == "validate" and verify_summary:
        _commit_for_display = next(iter(verify_summary.values()))["commit"] or "?"
        print(f"Verify matrix (verify.tsv, latest commit {_commit_for_display}):")
        print(f"  {'tier':<12} {'t=1 speed':>10} {'speedup':>10} {'pass/total':>12}")
        for t in tiers:
            vsum = verify_summary.get(t["tier"])
            if not vsum:
                continue
            baseline = get_baseline_speed(task_dir, t["name"], thread=1)
            if vsum["t1_speeds"]:
                import statistics as _stat
                t1_med = _stat.median(vsum["t1_speeds"])
                t1_str = f"{t1_med:.3f}s"
                if baseline and baseline > 0:
                    pct = (1 - t1_med / baseline) * 100.0
                    spd_str = f"{pct:+.1f}%"
                else:
                    spd_str = "—"
            else:
                t1_str = "—"
                spd_str = "—"
            total = vsum["n_cells"]
            n_pass = vsum["n_pass"]
            n_fail = vsum["n_fail"]
            n_crash = vsum["n_crash"]
            extra = f" ({n_crash} crash)" if n_crash else ""
            print(f"  {t['tier']:<12} {t1_str:>10} {spd_str:>10} "
                  f"{f'{n_pass}/{total}':>12}{extra}")
        print()

    # Patch stack: every keep, oldest first.
    keeps = [r for r in narrative_rows if r.get("status") == "keep"]
    if keeps:
        print(f"Patch stack ({len(keeps)} keep{'s' if len(keeps) != 1 else ''}, oldest first):")
        print(f"  {'round':<6} {'commit':<8} {'speedup':>9}  description")
        for r in keeps:
            rnd = r.get("round", "?")
            sha = (r.get("commit") or "?")[:7]
            try:
                pct = float(r.get("speedup_pct") or 0)
                pct_str = f"{pct:+.1f}%"
            except ValueError:
                pct_str = "—"
            desc = (r.get("description") or r.get("hypothesis") or "").strip()
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"  {rnd:<6} {sha:<8} {pct_str:>9}  {desc}")
        print()

    # Last N decision rows (status ∈ {keep, discard, crash, pending, rollback}).
    # Skip status=rerun (those are stability measurements, not decisions).
    decision_rows = [r for r in narrative_rows if r.get("status") in
                     ("keep", "discard", "crash", "pending", "rollback")]
    n_show = max(1, args.last)
    recent = decision_rows[-n_show:]
    if recent:
        print(f"Last {len(recent)} decision{'s' if len(recent) != 1 else ''}:")
        print(f"  {'round':<6} {'status':<8} {'tier':<10} {'speedup':>9}  hypothesis")
        for r in recent:
            rnd = r.get("round", "?")
            status = r.get("status", "?")
            tier = (r.get("dataset") or "?")
            if len(tier) > 10:
                tier = tier[:10]
            try:
                pct = float(r.get("speedup_pct") or 0)
                pct_str = f"{pct:+.1f}%" if status != "crash" else "—"
            except ValueError:
                pct_str = "—"
            hyp = (r.get("hypothesis") or "").strip()
            if len(hyp) > 70:
                hyp = hyp[:67] + "..."
            print(f"  {rnd:<6} {status:<8} {tier:<10} {pct_str:>9}  {hyp}")



