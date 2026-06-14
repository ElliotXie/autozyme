"""Per-task time + token + cost accounting for a zyme task (multi-agent).

Two independent signals are merged:

1. TIME — from ``.zyme/audit.jsonl``. Every zyme CLI invocation writes a row
   with ``ts`` and ``duration_s`` (see ``audit.py``). From these we derive the
   task's calendar span, total CLI subprocess wall time, and a per-phase
   breakdown. The gaps *between* command timestamps are the agent's
   thinking/editing time; ``duration_s`` alone is only the subprocess time.

2. TOKENS + COST — from the driving agent's session logs. Supported agents:

   * Claude Code — when zyme runs inside it (``CLAUDECODE=1``), ``audit.py``
     stamps the row with ``cc_session`` = the transcript stem. We resolve those
     to ``~/.claude/projects/*/<session>.jsonl`` and sum per-message ``usage``.
     A session whose transcript dir encodes *this* task is "scoped" (counted in
     full); a workspace-root session is "shared" (windowed to this task's audit
     span to avoid attributing other tasks' tokens).

   * Codex — rollout logs at ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``
     record the session ``cwd`` (→ which task), the ``model`` per turn, and a
     cumulative ``token_count.total_token_usage``. We map by cwd: a rollout
     whose cwd *is* the task is scoped (final cumulative total); a rollout at an
     ancestor dir is shared (cumulative delta over the audit window). No env
     bridge is needed — mapping is retroactive over existing rollouts.

   Cursor stores conversations as opaque content-addressed blobs in per-session
   ``store.db`` files with no exposed token telemetry, so Cursor token/cost is
   not supported (time still works for any agent).

USD is estimated via :mod:`zyme.dispatch.pricing` using each agent's own model.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from zyme.audit import read_audit
from zyme.dispatch.pricing import estimate_usage_cost, find_model_price


CC_TRANSCRIPT_ROOT = Path.home() / ".claude" / "projects"
CODEX_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"
# Forward-capture ledger: one JSON line per agent run, written by
# `zyme cost-capture` from the agent's --output-format stream-json. This is the
# only token source for agents that persist no usage on disk (e.g. Cursor), and
# the most precise source for any agent (no scoped/shared attribution guess).
AGENT_USAGE_FILENAME = "agent_usage.jsonl"
DEFAULT_GAP_MIN = 10.0

_TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
)


# --------------------------------------------------------------------------- #
# cmd -> lifecycle phase
# --------------------------------------------------------------------------- #
# Coarse mapping from the zyme subcommand recorded in audit.jsonl to a
# lifecycle phase. Mirrors scan.PHASE_ORDER plus a "meta" bucket for read-only
# / housekeeping commands. Note: the `reflect` and `bootstrap` phases run no
# zyme command, so they never appear here — their time shows up as gaps.
_PHASE_BY_CMD = {
    "init": "init",
    "init-check": "init",
    "run": "iterate",
    "dryrun": "iterate",
    "accept": "iterate",
    "reject": "iterate",
    "rollback": "iterate",
    "iterate": "iterate",
    "profile": "iterate",
    "baseline": "iterate",
    "inspect-parallelism": "iterate",
    "validate": "validate",
    "attest": "validate",
    "attest-sweep": "validate",
    "verify": "validate",
    "backfill": "validate",
    "package": "package",
    "publish-speedups": "package",
}


def cmd_to_phase(cmd: str) -> str:
    """Map an audit ``cmd`` string (e.g. "baseline reference") to a phase."""
    head = (cmd or "").strip().split()[0] if cmd else ""
    return _PHASE_BY_CMD.get(head, "meta")


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _empty_tokens() -> dict[str, int]:
    return {f: 0 for f in _TOKEN_FIELDS}


def active_minutes(events: list[datetime], gap_min: float) -> tuple[float, float]:
    """Sum inter-event gaps ≤ threshold (active min) + raw span, both in minutes."""
    if len(events) < 2:
        return 0.0, 0.0
    thresh = gap_min * 60
    active = sum(
        d for a, b in zip(events, events[1:])
        if 0 <= (d := (b - a).total_seconds()) <= thresh
    )
    raw = (events[-1] - events[0]).total_seconds()
    return active / 60, raw / 60


def _pick_model(model_counts: dict[str, int]) -> str | None:
    """Most-frequent model across all messages/turns."""
    if not model_counts:
        return None
    return max(model_counts.items(), key=lambda kv: kv[1])[0]


def _resolve_price_name(model: str | None) -> str | None:
    """Map a raw model id to something pricing.find_model_price accepts,
    falling back to the family's `*-latest` entry on a version we don't list."""
    if not model:
        return None
    if find_model_price(model) is not None:
        return model
    # Fall back to the family's price row. Return the table's `id` (with the
    # `anthropic:` prefix) so find_model_price actually matches it: the bare
    # `claude-opus-4-latest` string is NOT itself a key/alias in MODEL_PRICES.
    m = model.lower()
    if m.startswith("claude-opus"):
        return "anthropic:claude-opus-4-latest"
    if m.startswith("claude-sonnet"):
        return "anthropic:claude-sonnet-4-latest"
    if m.startswith("claude-haiku"):
        return "anthropic:claude-haiku-4.5"
    return model  # let pricing return None; caller notes "no price"


# --------------------------------------------------------------------------- #
# Claude Code transcripts
# --------------------------------------------------------------------------- #
def _encode_cwd(cwd: Path) -> str:
    """Mirror Claude Code's project-dir encoding: `/` and `_` → `-`."""
    return str(cwd).replace("/", "-").replace("_", "-")


def resolve_transcript(session_id: str) -> Path | None:
    """Find ``~/.claude/projects/*/<session_id>.jsonl`` (None if missing)."""
    if not CC_TRANSCRIPT_ROOT.exists():
        return None
    matches = glob.glob(str(CC_TRANSCRIPT_ROOT / "*" / f"{session_id}.jsonl"))
    return Path(matches[0]) if matches else None


def collect_session(
    jsonl: Path,
    lo: datetime | None = None,
    hi: datetime | None = None,
) -> dict[str, Any]:
    """Pull timestamps + per-message usage from one Claude transcript.

    When ``lo``/``hi`` are given, only events/usage with a timestamp inside the
    window are counted (used to bound shared workspace-root sessions to this
    task's activity). De-duplicates usage by (requestId, message id).
    """
    events: list[datetime] = []
    tok = _empty_tokens()
    n_assistant = 0
    model_counts: dict[str, int] = {}
    seen_usage: set[tuple[str, str]] = set()
    for line in jsonl.read_text(errors="replace").splitlines():
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        ts = parse_ts(e.get("timestamp", ""))
        if ts is not None and (lo is not None or hi is not None):
            if lo is not None and ts < lo:
                continue
            if hi is not None and ts > hi:
                continue
        if ts is not None:
            events.append(ts)
        if e.get("type") != "assistant":
            continue
        msg = e.get("message")
        if not isinstance(msg, dict):
            continue
        u = msg.get("usage") or {}
        model = msg.get("model")
        if not u or model == "<synthetic>":
            continue
        key = (str(e.get("requestId") or ""), str(msg.get("id") or ""))
        if key in seen_usage:
            continue
        seen_usage.add(key)
        tok["input_tokens"] += u.get("input_tokens", 0) or 0
        tok["output_tokens"] += u.get("output_tokens", 0) or 0
        tok["cache_read_tokens"] += u.get("cache_read_input_tokens", 0) or 0
        tok["cache_write_tokens"] += u.get("cache_creation_input_tokens", 0) or 0
        if model:
            model_counts[model] = model_counts.get(model, 0) + 1
        n_assistant += 1
    events.sort()
    return {
        "events": events,
        "n_assistant": n_assistant,
        "model_counts": model_counts,
        **tok,
        "total_tokens": sum(tok.values()),
    }


def _gather_claude_sessions(task_dir: Path, rows: list[dict]) -> dict[str, dict]:
    """Map each ``cc_session`` in audit rows to {path, scoped, audit_ts}."""
    task_proj = _encode_cwd(task_dir.resolve())
    out: dict[str, dict] = {}
    for r in rows:
        sid = r.get("cc_session")
        if not sid:
            continue
        info = out.get(sid)
        if info is None:
            path = resolve_transcript(sid)
            info = {"path": path, "scoped": False, "audit_ts": []}
            if path is not None:
                info["scoped"] = path.parent.name == task_proj
            out[sid] = info
        ts = parse_ts(r.get("ts", ""))
        if ts is not None:
            info["audit_ts"].append(ts)
    for info in out.values():
        info["audit_ts"].sort()
    return out


def _collect_claude(
    task_dir: Path, rows: list[dict], gap_min: float,
) -> dict[str, Any] | None:
    """Aggregate Claude Code token usage for the task. None if no sessions."""
    sessions = _gather_claude_sessions(task_dir, rows)
    if not sessions:
        return None
    tok = _empty_tokens()
    model_counts: dict[str, int] = {}
    events: list[datetime] = []
    n_scoped = n_shared = n_missing = 0
    pad = timedelta(minutes=gap_min)
    for info in sessions.values():
        path = info["path"]
        if path is None or not path.exists():
            n_missing += 1
            continue
        if info["scoped"]:
            n_scoped += 1
            data = collect_session(path)
        else:
            n_shared += 1
            ats = info["audit_ts"]
            lo = (ats[0] - pad) if ats else None
            hi = (ats[-1] + pad) if ats else None
            data = collect_session(path, lo=lo, hi=hi)
        for f in _TOKEN_FIELDS:
            tok[f] += data[f]
        for m, c in data["model_counts"].items():
            model_counts[m] = model_counts.get(m, 0) + c
        events.extend(data["events"])
    warnings: list[str] = []
    if n_missing:
        warnings.append(
            f"claude: {n_missing} referenced transcript(s) not found under "
            f"{CC_TRANSCRIPT_ROOT} — their tokens are not counted."
        )
    if n_shared:
        warnings.append(
            f"claude: {n_shared} shared (workspace-root) session(s) windowed to "
            f"this task's audit span (±{gap_min:g}m); may include adjacent work."
        )
    return {
        "agent": "claude",
        "source": "claude_transcript",
        **tok,
        "total_tokens": sum(tok.values()),
        "events": sorted(events),
        "model_counts": model_counts,
        "n_scoped": n_scoped,
        "n_shared": n_shared,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Codex rollouts
# --------------------------------------------------------------------------- #
def _codex_pricing_tokens(total_usage: dict | None) -> dict[str, int]:
    """Map Codex total_token_usage → pricing token fields.

    Codex ``input_tokens`` is inclusive of ``cached_input_tokens`` (the cached
    portion is billed at the discounted cached rate), and ``output_tokens``
    already includes reasoning tokens. Codex reports no cache *write*.
    """
    tu = total_usage or {}
    inp = int(tu.get("input_tokens", 0) or 0)
    cached = int(tu.get("cached_input_tokens", 0) or 0)
    out = int(tu.get("output_tokens", 0) or 0)
    return {
        "input_tokens": max(inp - cached, 0),
        "cache_read_tokens": cached,
        "output_tokens": out,
        "cache_write_tokens": 0,
    }


def collect_codex_rollout(jsonl: Path) -> dict[str, Any] | None:
    """Parse one Codex rollout. Returns {cwd, model_counts, events,
    token_events: [(ts, total_token_usage)]} or None if unparseable."""
    cwd: str | None = None
    model_counts: dict[str, int] = {}
    events: list[datetime] = []
    token_events: list[tuple[datetime, dict]] = []
    try:
        text = jsonl.read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        ts = parse_ts(e.get("timestamp", ""))
        if ts is not None:
            events.append(ts)
        payload = e.get("payload") if isinstance(e.get("payload"), dict) else {}
        etype = e.get("type")
        if etype == "session_meta":
            cwd = cwd or payload.get("cwd")
        elif etype == "turn_context":
            cwd = cwd or payload.get("cwd")
            model = payload.get("model")
            if model:
                model_counts[model] = model_counts.get(model, 0) + 1
        elif etype == "event_msg" and payload.get("type") == "token_count":
            info = payload.get("info")
            if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                if ts is not None:
                    token_events.append((ts, info["total_token_usage"]))
    if cwd is None:
        return None
    token_events.sort(key=lambda kv: kv[0])
    events.sort()
    return {
        "cwd": cwd,
        "model_counts": model_counts,
        "events": events,
        "token_events": token_events,
    }


def _codex_windowed_usage(
    token_events: list[tuple[datetime, dict]],
    lo: datetime | None,
    hi: datetime | None,
) -> dict[str, int]:
    """Tokens consumed in (lo, hi] from a cumulative counter.

    token_events are (ts, cumulative total_token_usage), sorted. Full-session
    usage = the final cumulative total (lo=hi=None). Windowed usage = cumulative
    at hi minus cumulative just before lo, field-wise (floored at 0)."""
    if not token_events:
        return _empty_tokens()

    def cum_at(bound, *, strict) -> dict:
        chosen = None
        for ts, tu in token_events:
            if bound is None or (ts < bound if strict else ts <= bound):
                chosen = tu
            else:
                break
        return chosen or {}

    end = cum_at(hi, strict=False)
    start = cum_at(lo, strict=True) if lo is not None else {}
    end_p = _codex_pricing_tokens(end)
    start_p = _codex_pricing_tokens(start) if start else _empty_tokens()
    return {f: max(end_p[f] - start_p[f], 0) for f in _TOKEN_FIELDS}


def _iter_codex_rollouts(root: Path):
    if not root.exists():
        return
    yield from sorted(root.glob("*/*/*/rollout-*.jsonl"))


def _collect_codex(
    task_dir: Path,
    first_ts: datetime | None,
    last_ts: datetime | None,
    gap_min: float,
) -> dict[str, Any] | None:
    """Aggregate Codex token usage for the task by matching rollout cwd.

    Only *scoped* rollouts (session cwd == task_dir) are counted — those are
    unambiguously this task, and we take the final cumulative total_token_usage.

    Rollouts whose cwd is an *ancestor* (e.g. the workspace root) are NOT
    attributable to a single task: every task's Codex work shares the same cwd,
    so windowing them by a multi-day audit span would dump unrelated tasks'
    tokens onto whichever task is queried. We only tally how many such ambiguous
    rollouts overlap this task's audit window, for a note. (Run Codex with the
    task dir as cwd to get clean per-task attribution.)

    None if no scoped rollout matched."""
    task_real = task_dir.resolve()
    tok = _empty_tokens()
    model_counts: dict[str, int] = {}
    events: list[datetime] = []
    n_scoped = 0
    n_ambiguous = 0
    pad = timedelta(minutes=gap_min)
    lo = (first_ts - pad) if first_ts else None
    hi = (last_ts + pad) if last_ts else None

    for path in _iter_codex_rollouts(CODEX_SESSIONS_ROOT):
        data = collect_codex_rollout(path)
        if data is None:
            continue
        try:
            cwd_real = Path(data["cwd"]).resolve()
        except Exception:
            continue
        if cwd_real == task_real:
            usage = _codex_windowed_usage(data["token_events"], None, None)
            if sum(usage.values()) == 0 and not data["events"]:
                continue
            n_scoped += 1
            for f in _TOKEN_FIELDS:
                tok[f] += usage[f]
            for m, c in data["model_counts"].items():
                model_counts[m] = model_counts.get(m, 0) + c
            events.extend(data["events"])
        elif task_real.is_relative_to(cwd_real) and lo is not None and hi is not None:
            # Ancestor cwd: only count it as "ambiguous overlap" if it was
            # actually active during this task's audit window.
            if any(lo <= t <= hi for t in data["events"]):
                n_ambiguous += 1

    if n_scoped == 0 and n_ambiguous == 0:
        return None
    warnings: list[str] = []
    if n_ambiguous:
        warnings.append(
            f"codex: {n_ambiguous} rollout(s) ran from an ancestor dir (e.g. the "
            f"workspace root) during this task's window and can't be attributed "
            f"to one task — excluded. Run Codex with the task dir as cwd for full "
            f"capture."
        )
    return {
        "agent": "codex",
        "source": "codex_rollout",
        **tok,
        "total_tokens": sum(tok.values()),
        "events": sorted(events),
        "model_counts": model_counts,
        "n_scoped": n_scoped,
        "n_shared": 0,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# forward capture (zyme cost-capture) — the agent_usage.jsonl ledger
# --------------------------------------------------------------------------- #
def _find_session_id(obj: Any) -> str:
    """Best-effort session id from a stream-json object (claude/cursor/codex)."""
    if isinstance(obj, dict):
        for key in ("session_id", "sessionId", "conversation_id",
                    "conversationId", "thread_id", "threadId"):
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in obj.values():
            got = _find_session_id(v)
            if got:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_session_id(v)
            if got:
                return got
    return ""


def parse_stream_usage(lines, agent: str) -> dict[str, Any]:
    """Extract one run's token usage from an agent's stream-json output.

    Handles both shapes:
      * Claude / Cursor — a ``type=result`` line with
        ``usage={inputTokens,outputTokens,cacheReadTokens,cacheWriteTokens}``
        (summed across results), plus model from the ``type=system`` line.
      * Codex — ``event_msg`` ``token_count`` events whose
        ``info.total_token_usage`` is cumulative; the final one is the total.

    Returns a normalized record (no timestamp; the caller stamps it)."""
    models: list[str] = []
    session_id = ""
    res_tok = _empty_tokens()
    n_results = 0
    duration_ms = 0
    cost_usd: float | None = None
    codex_last: dict | None = None

    for line in lines:
        line = line.strip() if isinstance(line, str) else line
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        t = o.get("type")
        payload = o.get("payload") if isinstance(o.get("payload"), dict) else {}
        if t == "system" and o.get("model"):
            models.append(o["model"])
        if t == "turn_context" and payload.get("model"):
            models.append(payload["model"])
        if t == "session_meta" and payload.get("id") and not session_id:
            session_id = str(payload["id"])
        if not session_id:
            session_id = _find_session_id(o)
        if t == "result":
            u = o.get("usage")
            if isinstance(u, dict):
                n_results += 1
                res_tok["input_tokens"] += int(u.get("inputTokens", 0) or 0)
                res_tok["output_tokens"] += int(u.get("outputTokens", 0) or 0)
                res_tok["cache_read_tokens"] += int(u.get("cacheReadTokens", 0) or 0)
                res_tok["cache_write_tokens"] += int(u.get("cacheWriteTokens", 0) or 0)
            c = o.get("total_cost_usd")
            if c is None:
                c = o.get("cost_usd")
            if c is not None:
                try:
                    cost_usd = (cost_usd or 0.0) + float(c)
                except (TypeError, ValueError):
                    pass
            duration_ms += int(o.get("duration_ms", 0) or 0)
        if t == "event_msg" and payload.get("type") == "token_count":
            info = payload.get("info")
            if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                codex_last = info["total_token_usage"]

    if n_results > 0:
        tok = res_tok
    elif codex_last is not None:
        tok = _codex_pricing_tokens(codex_last)
        n_results = 1
    else:
        tok = _empty_tokens()
    return {
        "agent": agent,
        "model": _pick_model({m: 1 for m in models}) if models else None,
        "session_id": session_id or None,
        **tok,
        "duration_ms": duration_ms or None,
        "cost_usd": cost_usd,
        "n_results": n_results,
    }


def append_agent_usage(task_dir: Path, record: dict[str, Any]) -> Path:
    """Append a usage record (with a UTC ts) to <task>/.zyme/agent_usage.jsonl."""
    z = Path(task_dir) / ".zyme"
    z.mkdir(exist_ok=True)
    path = z / AGENT_USAGE_FILENAME
    row = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _collect_captured(task_dir: Path) -> list[dict[str, Any]]:
    """Read agent_usage.jsonl → one aggregated per-agent dict per agent."""
    path = Path(task_dir) / ".zyme" / AGENT_USAGE_FILENAME
    if not path.exists():
        return []
    per_agent: dict[str, dict] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        agent = r.get("agent") or "unknown"
        a = per_agent.get(agent)
        if a is None:
            a = {"agent": agent, "source": "captured", **_empty_tokens(),
                 "events": [], "model_counts": {}, "n_records": 0}
            per_agent[agent] = a
        for f in _TOKEN_FIELDS:
            a[f] += int(r.get(f, 0) or 0)
        if r.get("model"):
            a["model_counts"][r["model"]] = a["model_counts"].get(r["model"], 0) + 1
        ts = parse_ts(r.get("ts", ""))
        if ts is not None:
            a["events"].append(ts)
        a["n_records"] += 1
    out = []
    for a in per_agent.values():
        a["total_tokens"] = sum(a[f] for f in _TOKEN_FIELDS)
        a["events"].sort()
        a["n_scoped"] = a["n_records"]   # captured runs are unambiguous
        a["n_shared"] = 0
        a["warnings"] = []
        out.append(a)
    return out


# --------------------------------------------------------------------------- #
# top-level report
# --------------------------------------------------------------------------- #
def _finalize_agent(per: dict, model_override: str | None) -> dict:
    """Attach dominant model + USD cost to a per-agent token dict."""
    model = model_override or _pick_model(per["model_counts"])
    price_name = model_override or _resolve_price_name(model)
    has_tokens = per["total_tokens"] > 0
    cost = estimate_usage_cost(per, price_name) if has_tokens else None
    per["dominant_model"] = model
    per["cost"] = cost
    per["n_sessions"] = per["n_scoped"] + per["n_shared"]
    return per


def compute_task_cost(
    task_dir: Path,
    gap_min: float = DEFAULT_GAP_MIN,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Build the full time + token + cost report for one task (all agents)."""
    task_dir = Path(task_dir).resolve()
    rows = read_audit(task_dir)
    warnings: list[str] = []

    # ----- TIME (from audit.jsonl) -----
    ts_all = sorted(t for t in (parse_ts(r.get("ts", "")) for r in rows) if t)
    first_ts = ts_all[0] if ts_all else None
    last_ts = ts_all[-1] if ts_all else None
    by_phase: dict[str, dict[str, float]] = {}
    total_wall = 0.0
    for r in rows:
        phase = cmd_to_phase(r.get("cmd", ""))
        dur = float(r.get("duration_s") or 0.0)
        total_wall += dur
        b = by_phase.setdefault(phase, {"n_calls": 0, "cli_wall_s": 0.0})
        b["n_calls"] += 1
        b["cli_wall_s"] += dur
    calendar_span_min = (
        (last_ts - first_ts).total_seconds() / 60 if len(ts_all) >= 2 else 0.0
    )

    # ----- TOKENS per agent -----
    # Captured (agent_usage.jsonl) is authoritative and unambiguous: for any
    # agent captured at run time, skip its retroactive scraper (avoids double
    # counting and beats scoped/shared attribution). Scrapers fill in agents
    # that were never captured.
    captured = _collect_captured(task_dir)
    captured_agents = {a["agent"] for a in captured}
    claude = (None if "claude" in captured_agents
              else _collect_claude(task_dir, rows, gap_min))
    codex = (None if "codex" in captured_agents
             else _collect_codex(task_dir, first_ts, last_ts, gap_min))

    agents: list[dict] = []
    for per in (*captured, claude, codex):
        if per is None:
            continue
        warnings.extend(per.pop("warnings", []))
        # A collector may return note-only (e.g. Codex with only ambiguous
        # ancestor-cwd rollouts): warning captured above, but nothing to count.
        if per["n_scoped"] + per["n_shared"] == 0:
            continue
        a = _finalize_agent(per, model_override)
        if a["total_tokens"] > 0 and a["cost"] is None and a["dominant_model"]:
            warnings.append(
                f"{a['agent']}: no price entry for model "
                f"'{a['dominant_model']}'; USD not estimated."
            )
        agents.append(a)

    # ----- combined tokens / cost / active-min -----
    combined_tok = _empty_tokens()
    all_events: list[datetime] = []
    for a in agents:
        for f in _TOKEN_FIELDS:
            combined_tok[f] += a[f]
        all_events.extend(a["events"])
    combined_total = sum(combined_tok.values())
    all_events.sort()
    active_min, raw_span_min = active_minutes(all_events, gap_min)

    cost_by_agent = {a["agent"]: a["cost"]["total_usd"]
                     for a in agents if a.get("cost")}
    combined_cost = (
        {"total_usd": sum(cost_by_agent.values()), "by_agent": cost_by_agent}
        if cost_by_agent else None
    )

    if not agents and rows:
        warnings.append(
            "No token data: no captured runs (.zyme/agent_usage.jsonl), Claude "
            "transcripts, or Codex rollouts matched this task. Cursor persists no "
            "usage on disk — capture it at run time with `zyme cost-capture`. Time "
            "metrics are still valid."
        )

    return {
        "task": task_dir.name,
        "task_dir": str(task_dir),
        "time": {
            "n_invocations": len(rows),
            "first_ts": first_ts.isoformat() if first_ts else None,
            "last_ts": last_ts.isoformat() if last_ts else None,
            "calendar_span_min": calendar_span_min,
            "cli_wall_min": total_wall / 60,
            "active_min": active_min,
            "raw_session_span_min": raw_span_min,
            "by_phase": by_phase,
        },
        "agents": [
            {k: v for k, v in a.items() if k != "events"} for a in agents
        ],
        "tokens": {
            **combined_tok,
            "total_tokens": combined_total,
            "agents": [a["agent"] for a in agents],
            "n_sessions": sum(a["n_sessions"] for a in agents),
        },
        "cost": combined_cost,
        "warnings": warnings,
    }
