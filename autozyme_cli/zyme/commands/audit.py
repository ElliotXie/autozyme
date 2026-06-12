"""`zyme audit` — per-task command audit log reader; impl in zyme.audit."""

import json
from datetime import datetime

from zyme.utils import info, task_dir_from_args




def _render_cc_tool(t: dict) -> str:
    """One-line render of a CC tool_use event for the audit table."""
    name = t.get("tool", "?")
    sub = " (subagent)" if t.get("subagent") else ""
    if name == "Read":
        s = f"Read {t.get('path', '')}"
        if t.get("offset") or t.get("limit"):
            s += f" [{t.get('offset', 0)}:+{t.get('limit', '')}]"
        return s + sub
    if name == "Edit":
        ra = " all" if t.get("replace_all") else ""
        return f"Edit{ra} {t.get('path', '')} (-{t.get('old_chars', 0)}/+{t.get('new_chars', 0)} chars){sub}"
    if name == "Write":
        return f"Write {t.get('path', '')} ({t.get('bytes', 0)} bytes){sub}"
    if name == "Bash":
        line = f"Bash $ {t.get('cmd', '')}"
        return line + sub
    if name in ("Grep", "Glob"):
        s = f"{name} {t.get('pattern', '')}"
        if t.get("path"):
            s += f" in {t['path']}"
        return s + sub
    if name == "Agent":
        return f"Agent[{t.get('agent', '?')}] {t.get('desc', '')}" + sub
    if name == "WebFetch":
        return f"WebFetch {t.get('url', '')}" + sub
    if name == "WebSearch":
        return f"WebSearch {t.get('query', '')}" + sub
    inp = t.get("input", "")
    return f"{name} {inp}{sub}"




def cmd_audit(args):
    """Show this task's command audit log.

    Each `zyme` invocation in a task directory appends one JSON row to
    `<task>/.zyme/audit.jsonl`. This subcommand renders the tail as a table
    (default) or streams the raw JSONL (--json).

    Read-only; the audit subcommand itself is NOT recorded.
    """
    from zyme.audit import read_audit
    task_dir = task_dir_from_args(args)
    last_n = args.last if args.last and args.last > 0 else None
    rows = read_audit(task_dir, last_n=last_n)
    if not rows:
        info(f"(no audit entries at {task_dir / '.zyme' / 'audit.jsonl'})")
        return

    if args.json:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return

    # Table view: timestamp (local) | cmd | dur | exit | argv-tail | outputs
    print(f"{'time':<19}  {'cmd':<10} {'dur':>7}  {'exit':>4}  argv / outputs")
    print("-" * 100)
    for r in rows:
        # Render UTC ts as local time for human consumption.
        try:
            ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
            ts_local = ts.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_local = r.get("ts", "")[:19]
        cmd = r.get("cmd", "?")
        dur = r.get("duration_s", 0.0)
        exitc = r.get("exit_code", "?")
        argv = r.get("argv", [])
        # Drop the leading "zyme" / script path and the subcommand name.
        argv_tail = " ".join(argv[2:]) if len(argv) > 2 else ""
        if len(argv_tail) > 60:
            argv_tail = argv_tail[:57] + "..."
        line = f"{ts_local:<19}  {cmd:<10} {dur:>6.2f}s  {str(exitc):>4}  {argv_tail}"
        print(line)
        outs = r.get("outputs") or {}
        if outs:
            parts = []
            for kind in ("created", "modified", "deleted"):
                if outs.get(kind):
                    parts.append(f"{kind}={','.join(outs[kind])}")
            if parts:
                print(f"{'':<19}  {'':<10} {'':>7}  {'':>4}  → " + "; ".join(parts))
        if r.get("error"):
            print(f"{'':<19}  {'':<10} {'':>7}  {'':>4}  ! {r['error']}")
        # CC tools that ran in the lead-up to this zyme call.
        cc_tools = r.get("cc_tools") or []
        for t in cc_tools:
            print(f"{'':<19}  {'':<10} {'':>7}  {'':>4}    cc: {_render_cc_tool(t)}")
    print()
    print(f"({len(rows)} entr{'y' if len(rows) == 1 else 'ies'} from {task_dir / '.zyme' / 'audit.jsonl'})")
