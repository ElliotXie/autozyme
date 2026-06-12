"""Per-task command audit log.

Writes one JSON object per `zyme` invocation to `<task_dir>/.zyme/audit.jsonl`.
Captures: timestamp, subcommand, argv, cwd, duration, exit code, error msg
(if any), and a delta of well-known output files (mtime/size before/after).

When invoked from Claude Code (env CLAUDECODE=1), also extracts the
tool_use events that happened in the CC session between the previous audit
row and this invocation — gives a unified timeline of "what CC did to decide
on this zyme call." Disable with ZYME_AUDIT_NO_CC=1.

Goal: cheap, append-only, per-task. Workspace-level commands (scan, dispatch,
prompt, bench, inspect-parallelism) skip naturally because
their cwd is the workspace root — no `task.yaml` there.

NOT captured: phase-internal LLM calls, full subprocess stdout, or anything
that already lives in results.tsv / phase outputs. Audit is a thin index, not
a replay log.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# stderr_tail: how many trailing lines to capture per invocation. 30 is enough
# to include die()'s message plus surrounding context (e.g. git stderr printed
# right before a wrapped failure), without blowing audit row size for tasks
# that emit progress on stderr.
_STDERR_TAIL_LINES = 30

# Files we snapshot before/after each command. Cheap stat() calls; we report
# which of these were created / modified / deleted by the command.
_TRACKED_OUTPUTS = (
    "results.tsv",
    "verify.tsv",
    "task.yaml",
    ".zyme/best.ref",
    ".zyme/round.counter",
    ".zyme/baselines_stash.tsv",
)


def _resolve_task_dir(args) -> Path | None:
    """Best-effort task-dir resolution. Returns None if not a task dir.

    Mirrors utils.task_dir_from_args but returns None instead of die()ing,
    and DOES NOT require task.yaml at call time — for `init`, task.yaml
    appears mid-command. Caller re-checks at the end.
    """
    raw = getattr(args, "task_dir", None) or os.getcwd()
    try:
        return Path(raw).resolve()
    except Exception:
        return None


def _snapshot(task_dir: Path) -> dict[str, tuple[float, int] | None]:
    """Stat each tracked file. Returns {rel_path: (mtime, size) | None}."""
    snap: dict[str, tuple[float, int] | None] = {}
    for rel in _TRACKED_OUTPUTS:
        p = task_dir / rel
        try:
            st = p.stat()
            snap[rel] = (st.st_mtime, st.st_size)
        except FileNotFoundError:
            snap[rel] = None
        except OSError:
            snap[rel] = None
    return snap


def _diff_snapshots(before: dict, after: dict) -> dict[str, list[str]]:
    """Compare two snapshots; bucket changes."""
    created, modified, deleted = [], [], []
    for rel in _TRACKED_OUTPUTS:
        b, a = before.get(rel), after.get(rel)
        if b is None and a is not None:
            created.append(rel)
        elif b is not None and a is None:
            deleted.append(rel)
        elif b is not None and a is not None and b != a:
            modified.append(rel)
    out = {}
    if created:
        out["created"] = created
    if modified:
        out["modified"] = modified
    if deleted:
        out["deleted"] = deleted
    return out


def _is_task_dir(p: Path) -> bool:
    return (p / "task.yaml").exists()


def _short_exc(exc: BaseException) -> str:
    """One-line error summary for the audit row.

    Form: `<ExceptionClass>: <first line of message>`. Empty message → just
    the class. Long messages truncated to 200 chars.
    """
    cls = type(exc).__name__
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    line = f"{cls}: {msg}" if msg else cls
    if len(line) > 200:
        line = line[:197] + "..."
    return line


class _StderrTee:
    """Tee writes to original stderr and capture a tail ring buffer.

    Audit records `stderr_tail` on failure so postmortem analysis can see
    why die() fired without having to re-run the command. Subprocess output
    written direct to fd 2 (no `capture` in subprocess.run) bypasses this —
    accepted: most zyme error paths route through die() or the git wrapper,
    both of which write via sys.stderr.
    """

    def __init__(self, original, max_lines: int = _STDERR_TAIL_LINES):
        self._original = original
        self._lines: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._partial = ""

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        self._original.write(s)
        self._partial += s
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._lines.append(line)
        return len(s)

    def flush(self):
        self._original.flush()

    def isatty(self):
        return getattr(self._original, "isatty", lambda: False)()

    def fileno(self):
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)

    def tail(self) -> list[str]:
        out = list(self._lines)
        if self._partial:
            out.append(self._partial)
        return out


class AuditContext:
    """Captures pre-state at __enter__; writes a record at __exit__.

    Used as a context manager around `args.func(args)` in cli.main(). Tolerates
    SystemExit (from die()), normal returns, and unhandled exceptions. Never
    raises out of __exit__ — audit failures must not break the user's command.
    """

    def __init__(self, args, argv: list[str], cmd_name: str):
        self.args = args
        self.argv = list(argv)
        self.cmd_name = cmd_name
        self.cwd = Path(os.getcwd()).resolve()
        # Task dir at entry (may not be a task yet — init creates task.yaml mid-run).
        self.task_dir_entry = _resolve_task_dir(args)
        self.start_ts = time.time()
        self.start_iso = datetime.fromtimestamp(self.start_ts, tz=timezone.utc).isoformat()
        self.snap_before: dict | None = None
        # Lower bound for CC-tools window: previous audit row's ts.
        # Read at __init__ so we capture the boundary BEFORE this row is written.
        self.last_audit_ts: str | None = None
        if self.task_dir_entry is not None and _is_task_dir(self.task_dir_entry):
            try:
                self.snap_before = _snapshot(self.task_dir_entry)
            except Exception:
                self.snap_before = None
            self.last_audit_ts = _last_audit_ts(self.task_dir_entry)

    def __enter__(self):
        self._tee = _StderrTee(sys.stderr)
        sys.stderr = self._tee
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore real stderr first so any audit-side error surfaces cleanly.
        if getattr(self, "_tee", None) is not None and sys.stderr is self._tee:
            sys.stderr = self._tee._original
        try:
            self._write(exc_type, exc)
        except Exception:
            # Audit must never mask the real error. Swallow + emit a stderr
            # breadcrumb that doesn't pollute normal output.
            sys.stderr.write("[zyme audit] failed to record this invocation\n")
        # Don't suppress exceptions — let cli.main propagate normally.
        return False

    def _write(self, exc_type, exc):
        # Resolve task_dir at exit time too (handles `zyme init` creating task.yaml).
        task_dir = self.task_dir_entry
        if task_dir is None or not _is_task_dir(task_dir):
            # Workspace-level command, or pre-init failure — nothing to audit.
            return

        end_ts = time.time()
        snap_after = _snapshot(task_dir)
        outputs = _diff_snapshots(self.snap_before or {}, snap_after)

        # Exit code: 0 on clean return, the SystemExit code on die(), -1 on
        # uncaught exception. die() always uses SystemExit so this covers it.
        if exc_type is None:
            exit_code: int = 0
            error: str | None = None
        elif exc_type is SystemExit:
            code = getattr(exc, "code", 0)
            exit_code = int(code) if isinstance(code, int) else 1
            # die() writes the message to stderr and SystemExit carries only
            # the int code — no useful payload to record. Leave error blank.
            error = None
        else:
            exit_code = -1
            error = _short_exc(exc) if exc else (exc_type.__name__ if exc_type else None)

        record = {
            "ts": self.start_iso,
            "cmd": self.cmd_name,
            "argv": self.argv,
            "cwd": str(self.cwd),
            "duration_s": round(end_ts - self.start_ts, 3),
            "exit_code": exit_code,
        }
        if error:
            record["error"] = error
        if outputs:
            record["outputs"] = outputs
        # stderr_tail: on any failure (die() or uncaught), include the captured
        # tail so postmortem analysis doesn't require re-running. Skipped on
        # success — successful commands often print noisy progress on stderr
        # and the audit row stays leaner.
        if exit_code != 0 and getattr(self, "_tee", None) is not None:
            tail = self._tee.tail()
            if tail:
                record["stderr_tail"] = "\n".join(tail).strip()

        # CC-tools window: tools CC ran since the prior audit row, up to this
        # invocation's start. Best-effort — failures here must not break audit.
        if _is_cc_session():
            try:
                transcript = _find_session_transcript(self.cwd)
                if transcript is not None:
                    cc_tools = _collect_cc_tools(
                        transcript, task_dir, self.last_audit_ts, self.start_iso,
                    )
                    if cc_tools:
                        record["cc_session"] = transcript.stem
                        record["cc_tools"] = cc_tools
            except Exception:
                pass

        # Append-only JSONL. Atomic per-line on POSIX since the line is short
        # and we open/close per call (no long-lived fd).
        log_path = task_dir / ".zyme" / "audit.jsonl"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Claude Code transcript scraping
# ---------------------------------------------------------------------------
# When zyme is invoked from inside Claude Code (env CLAUDECODE=1), we can
# scrape the session's local transcript at ~/.claude/projects/<encoded-cwd>/
# <session-id>.jsonl to find tool_use events that happened between the prior
# audit row and this invocation. That window = "what CC did before deciding
# to call this zyme command." Embedded into the audit row as `cc_tools`.
#
# We DO NOT capture: tool_result payloads (verbose; user can read transcript
# directly), text/thinking blocks (not actions), TodoWrite/ToolSearch events
# (pure agent bookkeeping noise). Per-tool input is truncated.

CC_TRANSCRIPT_ROOT = Path.home() / ".claude" / "projects"

# Tools whose mere invocation is noise — pure agent bookkeeping, not actions
# the user would care to audit.
_CC_TOOLS_SKIP = {"TodoWrite", "ToolSearch"}


def _is_cc_session() -> bool:
    """True when zyme was launched by Claude Code's Bash tool."""
    if os.environ.get("ZYME_AUDIT_NO_CC") == "1":
        return False
    return os.environ.get("CLAUDECODE") == "1"


def _encode_cwd(cwd: Path) -> str:
    """Mirror Claude Code's project-dir encoding: `/` and `_` → `-`.

    Examples observed under ~/.claude/projects/:
      /Users/me/my_project → -Users-me-my-project
      /Users/me/my_project/sub_field/test_x
        → -Users-me-my-project-sub-field-test-x
    """
    return str(cwd).replace("/", "-").replace("_", "-")


def _find_session_transcript(cwd: Path) -> Path | None:
    """Resolve this CC session's transcript file.

    Primary: env CLAUDE_CODE_SESSION_ID identifies the file uniquely; glob
    ~/.claude/projects/*/<session_id>.jsonl for the one match. CC's project
    dir is encoded from where CC was *launched* (e.g. workspace root), not
    cwd at zyme call — so the encoded-cwd path does NOT necessarily exist
    when zyme is invoked deep in a subdir.

    Fallback: pick the globally-most-recent .jsonl, used only when the env
    var is unset or no match found.
    """
    if not CC_TRANSCRIPT_ROOT.exists():
        return None
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if sid:
        matches = list(CC_TRANSCRIPT_ROOT.glob(f"*/{sid}.jsonl"))
        if matches:
            return matches[0]
    candidates: list[Path] = []
    for d in CC_TRANSCRIPT_ROOT.iterdir():
        if not d.is_dir():
            continue
        candidates.extend(d.glob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _last_audit_ts(task_dir: Path) -> str | None:
    """Read the most recent audit.jsonl row's `ts`. None if file empty/absent.

    Used as the lower bound of the CC-tools window so each row only covers
    the slice of CC activity since the previous zyme call.
    """
    p = task_dir / ".zyme" / "audit.jsonl"
    if not p.exists():
        return None
    try:
        with p.open("rb") as fh:
            # Read the last line cheaply: seek to end, walk back for newline.
            fh.seek(0, 2)
            size = fh.tell()
            if size == 0:
                return None
            # Cap reverse scan at 64 KB — audit rows are tiny.
            chunk = min(size, 65536)
            fh.seek(size - chunk)
            tail = fh.read(chunk).decode("utf-8", errors="ignore")
        last_line = tail.strip().splitlines()[-1] if tail.strip() else ""
        if not last_line:
            return None
        obj = json.loads(last_line)
        return obj.get("ts")
    except Exception:
        return None


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _summarize_tool_call(tu: dict, ts: str, sidechain: bool) -> dict | None:
    """Compress one tool_use content block into an audit-friendly dict.

    Per-tool truncation strips bulk content (file bodies, edit diffs) but
    keeps identifying params (path, line counts, command first line, etc.).
    Returns None for tools we deliberately skip.
    """
    name = tu.get("name", "?")
    if name in _CC_TOOLS_SKIP:
        return None
    inp = tu.get("input") or {}
    if not isinstance(inp, dict):
        inp = {}

    out: dict = {"ts": ts, "tool": name}
    if sidechain:
        out["subagent"] = True

    if name == "Read":
        out["path"] = inp.get("file_path", "")
        if inp.get("offset") is not None:
            out["offset"] = inp["offset"]
        if inp.get("limit") is not None:
            out["limit"] = inp["limit"]
    elif name == "Write":
        out["path"] = inp.get("file_path", "")
        out["bytes"] = len(inp.get("content", "") or "")
    elif name == "Edit":
        out["path"] = inp.get("file_path", "")
        out["old_chars"] = len(inp.get("old_string", "") or "")
        out["new_chars"] = len(inp.get("new_string", "") or "")
        if inp.get("replace_all"):
            out["replace_all"] = True
    elif name == "Bash":
        cmd = (inp.get("command") or "").split("\n", 1)[0]
        out["cmd"] = _truncate(cmd, 160)
        if inp.get("description"):
            out["desc"] = _truncate(str(inp["description"]), 80)
    elif name in ("Grep", "Glob"):
        if inp.get("pattern") is not None:
            out["pattern"] = _truncate(str(inp["pattern"]), 120)
        if inp.get("path"):
            out["path"] = inp["path"]
    elif name == "Agent":
        out["agent"] = inp.get("subagent_type", "general-purpose")
        if inp.get("description"):
            out["desc"] = _truncate(str(inp["description"]), 80)
    elif name == "WebFetch":
        if inp.get("url"):
            out["url"] = _truncate(str(inp["url"]), 160)
    elif name == "WebSearch":
        if inp.get("query"):
            out["query"] = _truncate(str(inp["query"]), 160)
    else:
        # Generic — keep input as truncated JSON. Unknown / new tools land here.
        try:
            out["input"] = _truncate(json.dumps(inp, ensure_ascii=False), 200)
        except (TypeError, ValueError):
            pass
    return out


def _collect_cc_tools(
    transcript: Path,
    task_dir: Path,
    since_ts: str | None,
    until_ts: str,
) -> list[dict]:
    """Walk transcript, return tool_use summaries in (since_ts, until_ts).

    Filters by `cwd startswith task_dir` so events from a parent-dir CC
    session that happened to touch other tasks don't bleed in. ISO timestamps
    are compared lexicographically — safe because both ends are normalized
    UTC ('Z' or '+00:00') and same precision.
    """
    out: list[dict] = []
    task_dir_str = str(task_dir)
    try:
        with transcript.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = obj.get("timestamp")
                if not ts:
                    continue
                # Normalize 'Z' suffix to '+00:00' for comparison.
                if ts.endswith("Z"):
                    ts_cmp = ts[:-1] + "+00:00"
                else:
                    ts_cmp = ts
                since_cmp = since_ts
                if since_cmp and since_cmp.endswith("Z"):
                    since_cmp = since_cmp[:-1] + "+00:00"
                until_cmp = until_ts
                if until_cmp.endswith("Z"):
                    until_cmp = until_cmp[:-1] + "+00:00"
                if since_cmp and ts_cmp <= since_cmp:
                    continue
                if ts_cmp >= until_cmp:
                    continue

                cwd_evt = obj.get("cwd")
                if cwd_evt and not str(cwd_evt).startswith(task_dir_str):
                    continue

                msg = obj.get("message")
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue

                sidechain = bool(obj.get("isSidechain"))
                for c in content:
                    if not isinstance(c, dict) or c.get("type") != "tool_use":
                        continue
                    summary = _summarize_tool_call(c, ts, sidechain)
                    if summary:
                        out.append(summary)
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_audit(task_dir: Path, last_n: int | None = None) -> list[dict]:
    """Read audit.jsonl. Returns most-recent-last. last_n caps from the tail."""
    p = task_dir / ".zyme" / "audit.jsonl"
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if last_n is not None and last_n > 0:
        rows = rows[-last_n:]
    return rows
