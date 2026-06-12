"""`zyme cost` — per-task time + token + USD accounting (multi-agent).

Reads ``.zyme/audit.jsonl`` (time) plus per-agent token sources: a captured
``.zyme/agent_usage.jsonl`` ledger (written by ``zyme cost-capture``), Claude
Code transcripts (via ``cc_session``), and Codex rollouts (matched by cwd). USD
is estimated via the dispatch price table. Cursor persists no usage on disk, so
its tokens come only from capture. Read-only; not recorded in the audit log.
"""

import sys
import json

from zyme.utils import info, task_dir_from_args
from zyme.cost import compute_task_cost, parse_stream_usage, append_agent_usage


_SOURCE_LABEL = {
    "captured": "captured",
    "claude_transcript": "transcripts",
    "codex_rollout": "rollouts",
}


def _fmt_min(m: float) -> str:
    """Minutes as a compact human string: 3.2m / 1h44m."""
    if m < 60:
        return f"{m:.1f}m"
    h, rem = divmod(int(round(m)), 60)
    return f"{h}h{rem:02d}m"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_usd(x) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


_PHASE_ORDER = ["init", "iterate", "validate", "package", "meta"]


def _print_agent(a: dict) -> None:
    """One agent's token + cost block."""
    src = _SOURCE_LABEL.get(a.get("source"), a.get("source") or "?")
    if a.get("source") == "captured":
        sess = f"{a.get('n_records', a['n_sessions'])} captured run(s)"
    else:
        sess = (f"{a['n_sessions']} session(s): "
                f"{a['n_scoped']} scoped, {a['n_shared']} shared")
    print(f"  [{a['agent']}]  {sess}  ({src})   model: {a.get('dominant_model') or '?'}")
    print(f"    input        : {_fmt_int(a['input_tokens']):>15}")
    print(f"    output       : {_fmt_int(a['output_tokens']):>15}")
    print(f"    cache read   : {_fmt_int(a['cache_read_tokens']):>15}")
    print(f"    cache write  : {_fmt_int(a['cache_write_tokens']):>15}")
    print(f"    total        : {_fmt_int(a['total_tokens']):>15}")
    cost = a.get("cost")
    if cost is None:
        print(f"    cost         : n/a")
    else:
        print(f"    cost         : {_fmt_usd(cost['total_usd'])}"
              f"   (price: {cost['model_price_id']})")


def cmd_cost(args):
    """Show how much time and how many tokens/dollars a task has consumed.

    TIME comes from every zyme invocation's audit row (`ts` + `duration_s`):
    calendar span, total CLI subprocess wall, agent active-minutes (from the
    session logs, idle gaps removed), and a per-phase CLI-wall breakdown.

    TOKENS + USD come from the driving agent's session logs — Claude Code
    transcripts (via `cc_session`) and Codex rollouts (matched by cwd). USD uses
    the dispatch price table. Cursor is unsupported (no on-disk token
    telemetry); its time metrics still work.
    """
    task_dir = task_dir_from_args(args)
    rep = compute_task_cost(
        task_dir,
        gap_min=args.gap_min,
        model_override=args.model,
    )

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return

    t = rep["time"]
    if t["n_invocations"] == 0 and not rep["agents"]:
        info(f"(no audit log at {task_dir / '.zyme' / 'audit.jsonl'} — nothing to account)")
        return

    print(f"task: {rep['task']}")
    print(f"  {task_dir}")
    print()

    # ----- TIME -----
    print("TIME")
    print(f"  calendar span     : {_fmt_min(t['calendar_span_min'])}"
          f"   ({t['first_ts']}  →  {t['last_ts']})")
    print(f"  agent active       : {_fmt_min(t['active_min'])}"
          f"   (session logs, idle gaps >{args.gap_min:g}m removed)")
    print(f"  CLI subprocess wall: {_fmt_min(t['cli_wall_min'])}"
          f"   (benchmark + git, across {t['n_invocations']} invocations)")
    print()
    print(f"  {'phase':<10} {'calls':>6} {'cli_wall':>10}")
    for phase in _PHASE_ORDER:
        b = t["by_phase"].get(phase)
        if not b:
            continue
        print(f"  {phase:<10} {b['n_calls']:>6} {_fmt_min(b['cli_wall_s'] / 60):>10}")
    print()

    # ----- TOKENS + COST -----
    print("TOKENS")
    agents = rep["agents"]
    if not agents:
        print("  (no Claude/Codex sessions matched — token/cost unavailable; see notes)")
    else:
        for a in agents:
            _print_agent(a)
        cost = rep["cost"]
        if cost is not None:
            print()
            print("COST (estimated)")
            if len(cost["by_agent"]) > 1:
                for ag, usd in cost["by_agent"].items():
                    print(f"  {ag:<8}: {_fmt_usd(usd)}")
            print(f"  TOTAL   : {_fmt_usd(cost['total_usd'])}")

    if rep["warnings"]:
        print()
        print("NOTES")
        for w in rep["warnings"]:
            print(f"  - {w}")


def cmd_cost_capture(args):
    """Record one agent run's token usage into .zyme/agent_usage.jsonl.

    Reads the agent's `--output-format stream-json` output (stdin by default, or
    a file via --from), extracts the final token usage + model, and appends a
    normalized record that `zyme cost` reads as the authoritative per-run source.
    This is how Cursor — which persists no usage on disk — gets token/cost, and
    it gives exact, unambiguous numbers for any agent.

    By default stdin is echoed through to stdout (so a live pipe still shows the
    agent's output); pass --quiet to suppress passthrough.

    Examples:
      cursor-agent -p --output-format stream-json "<task>" | zyme cost-capture --agent cursor
      zyme cost-capture --agent codex --from run.stream.jsonl
    """
    task_dir = task_dir_from_args(args)

    if args.from_file:
        with open(args.from_file, "r", encoding="utf-8", errors="replace") as fh:
            lines = list(fh)
        record = parse_stream_usage(lines, args.agent)
    else:
        # Stream stdin: tee to stdout (unless --quiet) while collecting lines.
        collected = []

        def _gen():
            for line in sys.stdin:
                if not args.quiet:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                collected.append(line)
                yield line
        record = parse_stream_usage(_gen(), args.agent)

    path = append_agent_usage(task_dir, record)
    tok_total = (record["input_tokens"] + record["output_tokens"]
                 + record["cache_read_tokens"] + record["cache_write_tokens"])
    if record["n_results"] == 0 or tok_total == 0:
        info(f"cost-capture: no usage found in {args.agent} stream "
             f"(nothing token-bearing) — wrote a zero record to {path}")
    else:
        info(f"cost-capture[{args.agent}]: {tok_total:,} tokens "
             f"(model {record.get('model') or '?'}) → {path}")
