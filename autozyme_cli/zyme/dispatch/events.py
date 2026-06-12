"""Parse agent JSONL streams into normalized dispatch events.

Solves the "stdout fully buffered until exit" problem from the bash-driven
dispatch — by streaming JSON we get one event per agent action (text, tool
call, tool result, completion) and can surface live progress.

Normalized event shapes (the only thing the rest of the pipeline cares about):

  {kind: "claude_text",        ts, snippet, char_count}
  {kind: "claude_tool_use",    ts, tool, summary}
  {kind: "claude_tool_result", ts, tool, ok, summary}
  {kind: "claude_done",        ts, num_turns, duration_ms, total_cost_usd, is_error}
  {kind: "zyme_run",           ts, hypothesis, raw_command}
  {kind: "zyme_accept",        ts, description, raw_command}
  {kind: "zyme_reject",        ts, description, raw_command}

The `zyme_*` events are recognized by inspecting Bash tool_use commands —
they're the in-band progress signal the master uses to maintain
`accepts` / `rejects` / `round` counters in state.json.
"""
import json
import re
import shlex
from typing import Iterable

from zyme.dispatch.state import utcnow_iso

_SNIPPET_LEN = 240
_SUMMARY_LEN = 200


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def parse_agent_line(raw: str, *, agent: str = "claude") -> list[dict]:
    if agent == "codex":
        return parse_codex_line(raw)
    if agent == "cursor":
        return parse_cursor_line(raw)
    return parse_claude_line(raw)


def parse_claude_line(raw: str) -> list[dict]:
    """Parse one stream-json line from claude. Returns 0 or more events.

    Returns [] for unparseable lines, partial-message stream_event lines
    (we use full assistant/user/result events for atomicity), and
    anything we don't recognize. Robust to schema drift — unknown shapes
    are silently ignored, not raised.
    """
    raw = raw.strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []

    session_id = _find_session_id(obj)

    t = obj.get("type")
    if t == "assistant":
        return _attach_session_id(_parse_assistant(obj), session_id)
    if t == "user":
        return _attach_session_id(_parse_user(obj), session_id)
    if t == "result":
        return _attach_session_id([_parse_result(obj)], session_id)
    if t == "system" and session_id:
        return [{
            "ts": utcnow_iso(),
            "kind": "agent_session",
            "agent": "claude",
            "session_id": session_id,
            "subtype": obj.get("subtype"),
        }]
    # type=system, type=stream_event, etc — ignored for normalized stream
    return []


def parse_codex_line(raw: str) -> list[dict]:
    """Parse one Codex CLI `codex exec --json` line.

    Codex's JSONL schema has changed across CLI versions. This parser is
    deliberately forgiving: it emits specific text/tool/done events for known
    fields and otherwise stores a compact generic codex_event so dispatch
    status still shows progress.
    """
    raw = raw.strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return [{
            "ts": utcnow_iso(),
            "kind": "codex_text",
            "snippet": _truncate(raw, _SNIPPET_LEN),
            "char_count": len(raw),
        }]
    if not isinstance(obj, dict):
        return []

    session_id = _find_session_id(obj)

    msg = obj.get("msg") if isinstance(obj.get("msg"), dict) else {}
    item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
    t = str(
        obj.get("type")
        or obj.get("event")
        or obj.get("kind")
        or msg.get("type")
        or item.get("type")
        or ""
    )
    lower_t = t.lower()
    nested_t = str(msg.get("type") or item.get("type") or "").lower()

    text = _find_text(obj)
    if text and nested_t in ("agent_message", "assistant_message", "message"):
        return _attach_session_id([{
            "ts": utcnow_iso(),
            "kind": "claude_text",
            "snippet": _truncate(text, _SNIPPET_LEN),
            "char_count": len(text),
        }], session_id)

    if text and any(key in lower_t for key in ("message", "text", "delta", "assistant", "output")):
        return _attach_session_id([{
            "ts": utcnow_iso(),
            "kind": "claude_text",
            "snippet": _truncate(text, _SNIPPET_LEN),
            "char_count": len(text),
        }], session_id)

    tool = (
        obj.get("tool") or obj.get("name") or obj.get("call_id")
        or msg.get("tool") or msg.get("name") or msg.get("call_id")
        or item.get("tool") or item.get("name") or item.get("call_id")
    )
    if (not tool
            and (lower_t in ("exec_command_begin", "tool_call", "command_begin")
                 or nested_t in ("exec_command_begin", "tool_call", "command_begin"))):
        tool = "shell"
    event_label = f"{lower_t} {nested_t}"
    if tool and any(key in event_label for key in ("tool", "exec", "command", "call")):
        summary = _find_command(obj) or text or _truncate(json.dumps(obj, separators=(",", ":")), _SUMMARY_LEN)
        out = [{
            "ts": utcnow_iso(),
            "kind": "claude_tool_use",
            "tool": str(tool),
            "summary": _truncate(summary, _SUMMARY_LEN),
        }]
        extra = _zyme_event_from_bash(summary)
        if extra is not None:
            out.append(extra)
        return _attach_session_id(out, session_id)

    if any(key in lower_t for key in ("done", "complete", "completed", "result", "finished")):
        return _attach_session_id([{
            "ts": utcnow_iso(),
            "kind": "claude_done",
            "num_turns": obj.get("num_turns") or obj.get("turns"),
            "duration_ms": obj.get("duration_ms"),
            "total_cost_usd": obj.get("total_cost_usd") or obj.get("cost_usd"),
            "is_error": bool(obj.get("is_error") or obj.get("error")),
            "subtype": t,
        }], session_id)

    if lower_t == "error":
        return _attach_session_id([{
            "ts": utcnow_iso(),
            "kind": "claude_done",
            "num_turns": obj.get("num_turns") or obj.get("turns"),
            "duration_ms": obj.get("duration_ms"),
            "total_cost_usd": obj.get("total_cost_usd") or obj.get("cost_usd"),
            "is_error": True,
            "subtype": t,
            "message": text or _find_text(msg),
        }], session_id)

    summary = text or _find_command(obj) or _truncate(json.dumps(obj, separators=(",", ":")), _SUMMARY_LEN)
    return _attach_session_id([{
        "ts": utcnow_iso(),
        "kind": "codex_event",
        "event": t or "?",
        "summary": _truncate(summary, _SUMMARY_LEN),
    }], session_id)


def parse_cursor_line(raw: str) -> list[dict]:
    """Parse Cursor agent output.

    Cursor's `--output-format stream-json` currently mirrors Claude-style
    assistant/user/result events, but beta builds may drift. Try that parser
    first, then fall back to the generic Codex-style parser so plain text and
    unknown JSON are still visible.
    """
    raw_stripped = raw.strip()
    if raw_stripped:
        try:
            obj = json.loads(raw_stripped)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and obj.get("type") == "system" and obj.get("model"):
            return [{
                "ts": utcnow_iso(),
                "kind": "agent_model",
                "agent": "cursor",
                "model": str(obj.get("model") or ""),
                "subtype": obj.get("subtype"),
                "session_id": _find_session_id(obj),
            }]

    claude_events = parse_claude_line(raw)
    if claude_events:
        return claude_events
    events = parse_codex_line(raw)
    for ev in events:
        if ev.get("kind") == "codex_event":
            ev["kind"] = "cursor_event"
        elif ev.get("kind") == "codex_text":
            ev["kind"] = "cursor_text"
    return events


def _find_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return ""
    for key in ("text", "message", "content", "delta", "output", "summary"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val
    msg = obj.get("msg") or obj.get("item")
    if isinstance(msg, dict):
        return _find_text(msg)
    if isinstance(obj.get("content"), list):
        parts = []
        for block in obj["content"]:
            txt = _find_text(block)
            if txt:
                parts.append(txt)
        return "\n".join(parts)
    return ""


def _find_command(obj) -> str:
    if not isinstance(obj, dict):
        return ""
    for key in ("command", "cmd", "arguments", "args"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list):
            if (len(val) >= 3
                    and str(val[0]).rsplit("/", 1)[-1] in ("sh", "bash", "zsh")
                    and str(val[1]) in ("-c", "-lc")):
                return str(val[2])
            return " ".join(str(x) for x in val)
    for key in ("tool_input", "input", "params", "msg", "item"):
        val = obj.get(key)
        if isinstance(val, dict):
            nested = _find_command(val)
            if nested:
                return nested
    return ""


def _find_session_id(obj) -> str:
    """Return a resumable conversation/session id from agent JSON, if present."""
    if isinstance(obj, dict):
        for key in (
            "session_id",
            "sessionId",
            "conversation_id",
            "conversationId",
            "thread_id",
            "threadId",
        ):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for val in obj.values():
            found = _find_session_id(val)
            if found:
                return found
    elif isinstance(obj, list):
        for val in obj:
            found = _find_session_id(val)
            if found:
                return found
    return ""


def _attach_session_id(events: list[dict], session_id: str) -> list[dict]:
    if not session_id:
        return events
    for ev in events:
        ev.setdefault("session_id", session_id)
    return events


# ---------------------------------------------------------------------------
# assistant: text blocks + tool_use blocks
# ---------------------------------------------------------------------------

def _parse_assistant(obj: dict) -> list[dict]:
    msg = obj.get("message") or {}
    blocks = msg.get("content") or []
    out: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            text = b.get("text") or ""
            if text.strip():
                out.append({
                    "ts": utcnow_iso(),
                    "kind": "claude_text",
                    "snippet": _truncate(text, _SNIPPET_LEN),
                    "char_count": len(text),
                })
        elif bt == "tool_use":
            tool = b.get("name") or "?"
            inp = b.get("input") or {}
            summary = _summarize_tool_input(tool, inp)
            out.append({
                "ts": utcnow_iso(),
                "kind": "claude_tool_use",
                "tool": tool,
                "summary": summary,
            })
            # If this is a Bash invocation of zyme run/accept/reject, emit
            # an extra in-band event so the master can update counters.
            if tool == "Bash":
                cmd = (inp.get("command") or "").strip()
                extra = _zyme_event_from_bash(cmd)
                if extra is not None:
                    out.append(extra)
    return out


# ---------------------------------------------------------------------------
# user: tool_result blocks (the response side of a tool_use)
# ---------------------------------------------------------------------------

def _parse_user(obj: dict) -> list[dict]:
    msg = obj.get("message") or {}
    blocks = msg.get("content") or []
    out: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") != "tool_result":
            continue
        ok = not bool(b.get("is_error"))
        content = b.get("content")
        text = _flatten_tool_result_content(content)
        out.append({
            "ts": utcnow_iso(),
            "kind": "claude_tool_result",
            "ok": ok,
            "summary": _truncate(text, _SUMMARY_LEN),
        })
    return out


def _flatten_tool_result_content(content) -> str:
    """tool_result.content can be a string or a list of {type,text} blocks.

    Flatten to a single short string for the summary field.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text") or "")
                elif "text" in b:
                    parts.append(str(b.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return ""


# ---------------------------------------------------------------------------
# result: terminal event with totals
# ---------------------------------------------------------------------------

def _parse_result(obj: dict) -> dict:
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
    return {
        "ts": utcnow_iso(),
        "kind": "claude_done",
        "num_turns": obj.get("num_turns"),
        "duration_ms": obj.get("duration_ms"),
        "duration_api_ms": obj.get("duration_api_ms"),
        "total_cost_usd": obj.get("total_cost_usd"),
        "request_id": obj.get("request_id"),
        "usage": usage,
        "is_error": bool(obj.get("is_error")),
        "subtype": obj.get("subtype"),
    }


# ---------------------------------------------------------------------------
# Bash → zyme_run / zyme_accept / zyme_reject recognition
# ---------------------------------------------------------------------------

def _zyme_event_from_bash(cmd: str) -> dict | None:
    """Detect `zyme run|accept|reject` invocations inside a Bash command.

    We try shlex first, falling back to a regex if the command can't be
    tokenized (heredocs, unbalanced quotes, etc). Looks for the verb as
    the first token of any segment split on `&&` / `;` / `|` boundaries.
    """
    for seg in _split_command_segments(cmd):
        verb_match = _ZYME_VERB_RE.match(seg)
        if not verb_match:
            continue
        verb = verb_match.group(1)
        # Try to extract the first quoted/positional argument as the
        # hypothesis or description. shlex handles quoting correctly.
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        # Drop "zyme" + verb
        rest = tokens[2:] if len(tokens) >= 2 else []
        text_arg = _extract_text_arg(rest)
        kind = {"run": "zyme_run", "accept": "zyme_accept", "reject": "zyme_reject"}[verb]
        ev = {
            "ts": utcnow_iso(),
            "kind": kind,
            "raw_command": _truncate(seg, _SUMMARY_LEN),
        }
        if verb == "run":
            ev["hypothesis"] = text_arg
        else:
            ev["description"] = text_arg
        return ev
    return None


_ZYME_VERB_RE = re.compile(r"^\s*(?:[\w/\.\-]*?\b)?zyme\s+(run|accept|reject)\b")


def _split_command_segments(cmd: str) -> list[str]:
    """Split a Bash command on `&&` / `;` boundaries, ignoring `||` / `|`.

    Naive (doesn't respect quoting) but good enough for recognizing
    `zyme run "..."` as the first verb of a chained command. Worst case
    we miss a zyme call inside a complex pipeline — acceptable.
    """
    parts = re.split(r"\s*&&\s*|\s*;\s*", cmd)
    return [p.strip() for p in parts if p.strip()]


def _extract_text_arg(tokens: list[str]) -> str:
    """Pick the most likely "free-text" argument from a token list.

    For `zyme run`, that's typically the first positional (the
    hypothesis). For `zyme accept -m "..."`, it's whatever follows
    `-m` / `--description`. We try the flagged form first, then fall
    back to the first non-flag token.
    """
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-m", "--description") and i + 1 < len(tokens):
            return tokens[i + 1]
        i += 1
    for tok in tokens:
        if not tok.startswith("-") and tok not in ("--rerun", "--setup"):
            return tok
    return ""


# ---------------------------------------------------------------------------
# Tool-input summarization (one short line per tool call, for logs)
# ---------------------------------------------------------------------------

def _summarize_tool_input(tool: str, inp: dict) -> str:
    if not isinstance(inp, dict):
        return ""
    if tool == "Bash":
        return _truncate(inp.get("command") or "", _SUMMARY_LEN)
    if tool in ("Read", "Write"):
        return _truncate(inp.get("file_path") or "", _SUMMARY_LEN)
    if tool == "Edit":
        path = inp.get("file_path") or ""
        old = (inp.get("old_string") or "").splitlines()[:1]
        return _truncate(f"{path} :: {old[0] if old else ''}", _SUMMARY_LEN)
    if tool == "Grep":
        return _truncate(f"{inp.get('pattern','')} in {inp.get('path','')}", _SUMMARY_LEN)
    if tool == "TodoWrite":
        todos = inp.get("todos") or []
        return f"{len(todos)} todo(s)"
    # Fallback: dump compact JSON keys to give some signal
    keys = ",".join(sorted(inp.keys())[:4])
    return _truncate(keys, _SUMMARY_LEN)


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ⏎ ").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# State updater — applied to one task's state entry per normalized event
# ---------------------------------------------------------------------------

def apply_event_to_task_state(task_state: dict, event: dict) -> None:
    """Mutate `task_state` (one entry from state.queue) for one event.

    Tracks: n_events, last_event_at, last_event_kind, accepts, rejects,
    round (= accepts + rejects), and clears `stalled` whenever any
    event arrives.
    """
    task_state["n_events"] = (task_state.get("n_events") or 0) + 1
    task_state["last_event_at"] = event.get("ts")
    task_state["last_event_kind"] = event.get("kind")
    task_state["stalled"] = False
    if event.get("session_id"):
        task_state["agent_session_id"] = str(event.get("session_id"))
    kind = event.get("kind")
    if kind == "zyme_accept":
        task_state["accepts"] = (task_state.get("accepts") or 0) + 1
    elif kind == "zyme_reject":
        task_state["rejects"] = (task_state.get("rejects") or 0) + 1
    if kind in ("zyme_accept", "zyme_reject"):
        task_state["round"] = (task_state.get("accepts") or 0) + (task_state.get("rejects") or 0)


def parse_lines(lines: Iterable[str]) -> list[dict]:
    """Convenience: feed an iterable of raw lines, get all normalized events."""
    out = []
    for line in lines:
        out.extend(parse_claude_line(line))
    return out
