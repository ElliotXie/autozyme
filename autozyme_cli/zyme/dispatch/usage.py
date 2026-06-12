"""Token/cost telemetry for `zyme dispatch` workspaces.

Agent CLIs expose usage inconsistently. Cursor currently puts token counts in
raw stream-json `type=result` lines, while Claude often exposes USD cost on the
same terminal event. This module treats raw logs as the source of truth and
falls back to parsed per-task events when raw logs are unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zyme.dispatch.pricing import estimate_usage_cost
from zyme.dispatch.state import state_path, task_log_dir, read_state


TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)

RAW_TOKEN_KEYS = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "cacheReadTokens": "cache_read_tokens",
    "cacheWriteTokens": "cache_write_tokens",
}


def collect_usage(
    workspace: Path,
    *,
    token_budget: int | None = None,
    budget_basis: str = "total",
    price_model: str | None = None,
) -> dict[str, Any]:
    """Return token/cost telemetry for a dispatch workspace.

    `workspace` is the directory containing `.zyme_dispatch/`. `token_budget`,
    when provided, is compared against either total telemetry tokens including
    cache reads (`budget_basis="total"`) or non-cache tokens
    (`budget_basis="non-cache"`).
    """
    workspace = Path(workspace).resolve()
    if budget_basis not in {"total", "non-cache"}:
        raise ValueError("budget_basis must be 'total' or 'non-cache'")

    state = read_state(state_path(workspace)) or {}
    tasks: dict[str, dict[str, Any]] = {}

    for task in state.get("queue") or []:
        name = str(task.get("name") or "").strip()
        if not name:
            continue
        bucket = _task_bucket(name)
        bucket["status"] = task.get("status")
        bucket["task_dir"] = task.get("task_dir")
        _add_unique(bucket["models"], task.get("actual_model"))
        _add_unique(bucket["sessions"], task.get("agent_session_id"))
        tasks[name] = bucket

    logs_dir = task_log_dir(workspace)
    if logs_dir.is_dir():
        for raw_log in sorted(logs_dir.glob("*.out")):
            name = raw_log.name[:-4]
            bucket = tasks.setdefault(name, _task_bucket(name))
            bucket["source_files"].append(str(raw_log))
            _read_raw_log(raw_log, bucket)

    if logs_dir.is_dir():
        for event_log in sorted(logs_dir.glob("*.events.ndjson")):
            name = event_log.name[:-14]
            bucket = tasks.setdefault(name, _task_bucket(name))
            if bucket["raw_result_events"] > 0:
                continue
            bucket["source_files"].append(str(event_log))
            _read_event_log(event_log, bucket)

    finalized_tasks = {name: _finalize_task(tasks[name]) for name in sorted(tasks)}
    for task in finalized_tasks.values():
        _attach_estimated_cost(
            task,
            candidates=_pricing_candidates(
                override=price_model,
                task=task,
                state_model=state.get("model"),
            ),
        )

    summary = {
        "workspace": str(workspace),
        "agent": state.get("agent"),
        "model": state.get("model"),
        "effort": state.get("effort"),
        "max_rounds": state.get("max_rounds"),
        "force_mode": state.get("force_mode"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "price_model_override": price_model,
        "tasks": finalized_tasks,
        "totals": {},
        "budget": None,
        "notes": [],
    }

    summary["totals"] = _totals(summary["tasks"].values())
    if token_budget is not None:
        budget = int(token_budget)
        basis_field = "total_tokens" if budget_basis == "total" else "non_cache_tokens"
        used = summary["totals"][basis_field]
        summary["budget"] = {
            "tokens": budget,
            "basis": budget_basis,
            "used_tokens": used,
            "used_pct": (used / budget * 100.0) if budget > 0 else None,
            "remaining_tokens": budget - used,
        }

    if summary.get("agent") == "cursor" and summary["totals"].get("cost_usd") is None:
        summary["notes"].append(
            "Cursor emits token telemetry, but USD cost is n/a unless the CLI "
            "emits pricing."
        )
    unpriced = [
        name for name, task in summary["tasks"].items()
        if task.get("requests") and task.get("estimated_cost_usd") is None
    ]
    if unpriced:
        summary["notes"].append(
            "No model price match for: " + ", ".join(unpriced)
            + ". Use --price-model to override."
        )
    pricing_notes = []
    for task in summary["tasks"].values():
        for note in task.get("pricing_notes") or []:
            if note not in pricing_notes:
                pricing_notes.append(note)
    for note in pricing_notes:
        summary["notes"].append(f"Pricing assumption: {note}")
    return summary


def render_usage(summary: dict[str, Any]) -> str:
    """Render `collect_usage()` output as a compact human-readable report."""
    totals = summary.get("totals") or {}
    lines = [
        "# dispatch usage",
        f"workspace: {summary.get('workspace')}",
        f"agent: {summary.get('agent') or '-'}",
        f"model: {summary.get('model') or '-'}",
    ]
    if summary.get("max_rounds") is not None:
        lines.append(
            f"max_rounds: {summary.get('max_rounds')} "
            f"(force_mode={bool(summary.get('force_mode'))})"
        )
    lines.extend([
        "",
        "totals:",
        f"  requests          : {_fmt_int(totals.get('requests', 0))}",
        f"  input tokens      : {_fmt_int(totals.get('input_tokens', 0))}",
        f"  output tokens     : {_fmt_int(totals.get('output_tokens', 0))}",
        f"  cache read tokens : {_fmt_int(totals.get('cache_read_tokens', 0))}",
        f"  cache write tokens: {_fmt_int(totals.get('cache_write_tokens', 0))}",
        f"  non-cache tokens  : {_fmt_int(totals.get('non_cache_tokens', 0))}",
        f"  total tokens      : {_fmt_int(totals.get('total_tokens', 0))}",
        f"  wall duration     : {_fmt_duration_ms(totals.get('duration_ms', 0))}",
        f"  api duration      : {_fmt_duration_ms(totals.get('duration_api_ms', 0))}",
        f"  cost usd          : {_fmt_cost(totals.get('cost_usd'))}",
        f"  est. cost usd     : {_fmt_cost(totals.get('estimated_cost_usd'))}",
    ])

    budget = summary.get("budget")
    if budget:
        pct = budget.get("used_pct")
        pct_s = "n/a" if pct is None else f"{pct:.1f}%"
        remaining = budget.get("remaining_tokens")
        lines.append(
            "  token budget      : "
            f"{_fmt_int(budget.get('used_tokens', 0))} / {_fmt_int(budget.get('tokens', 0))} "
            f"({pct_s}, basis={budget.get('basis')}, "
            f"remaining={_fmt_int(remaining)})"
        )

    tasks = summary.get("tasks") or {}
    lines.append("")
    if not tasks:
        lines.append("(no task usage found)")
    else:
        lines.append("tasks:")
        lines.append(
            f"{'task':<28} {'req':>5} {'input':>12} {'output':>12} "
            f"{'cache_read':>12} {'total':>12} {'api_dur':>9} {'est_cost':>10} price"
        )
        for name, task in tasks.items():
            price = task.get("pricing_model_id") or "-"
            lines.append(
                f"{name[:28]:<28} {task.get('requests', 0):>5} "
                f"{_fmt_int(task.get('input_tokens', 0)):>12} "
                f"{_fmt_int(task.get('output_tokens', 0)):>12} "
                f"{_fmt_int(task.get('cache_read_tokens', 0)):>12} "
                f"{_fmt_int(task.get('total_tokens', 0)):>12} "
                f"{_fmt_duration_ms(task.get('duration_api_ms', 0)):>9} "
                f"{_fmt_cost(task.get('estimated_cost_usd')):>10} {price}"
            )

    notes = summary.get("notes") or []
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"note: {note}")
    return "\n".join(lines)


def _task_bucket(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": None,
        "task_dir": None,
        "requests": 0,
        "done_events": 0,
        "raw_result_events": 0,
        "usage_events": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "non_cache_tokens": 0,
        "total_tokens": 0,
        "duration_ms": 0,
        "duration_api_ms": 0,
        "cost_usd": None,
        "estimated_cost_usd": None,
        "estimated_cost_breakdown": None,
        "pricing_model_id": None,
        "pricing_model": None,
        "pricing_notes": [],
        "sessions": [],
        "models": [],
        "source_files": [],
    }


def _read_raw_log(path: Path, bucket: dict[str, Any]) -> None:
    for obj in _iter_json_objects(path):
        if obj.get("type") == "system":
            _add_unique(bucket["models"], obj.get("model"))
            _add_unique(bucket["sessions"], _find_session_id(obj))
            continue
        if obj.get("type") != "result":
            continue
        bucket["raw_result_events"] += 1
        _consume_terminal_event(obj, bucket)


def _read_event_log(path: Path, bucket: dict[str, Any]) -> None:
    for obj in _iter_json_objects(path):
        if obj.get("kind") == "agent_model":
            _add_unique(bucket["models"], obj.get("model"))
            _add_unique(bucket["sessions"], obj.get("session_id"))
            continue
        if obj.get("kind") != "claude_done":
            continue
        _consume_terminal_event(obj, bucket)


def _consume_terminal_event(obj: dict[str, Any], bucket: dict[str, Any]) -> None:
    bucket["requests"] += 1
    bucket["done_events"] += 1
    _add_unique(bucket["sessions"], _find_session_id(obj))
    _add_unique(bucket["models"], obj.get("model"))
    bucket["duration_ms"] += _as_int(obj.get("duration_ms")) or 0
    bucket["duration_api_ms"] += _as_int(obj.get("duration_api_ms")) or 0
    cost = obj.get("total_cost_usd")
    if cost is None:
        cost = obj.get("cost_usd")
    _add_cost(bucket, cost)

    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return
    bucket["usage_events"] += 1
    for raw_key, field in RAW_TOKEN_KEYS.items():
        bucket[field] += _as_int(usage.get(raw_key)) or 0


def _iter_json_objects(path: Path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _find_session_id(obj: Any) -> str:
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


def _finalize_task(bucket: dict[str, Any]) -> dict[str, Any]:
    bucket = dict(bucket)
    bucket["models"] = sorted(x for x in bucket.get("models", []) if x)
    bucket["sessions"] = sorted(x for x in bucket.get("sessions", []) if x)
    bucket["source_files"] = sorted(dict.fromkeys(bucket.get("source_files", [])))
    bucket["non_cache_tokens"] = (
        bucket["input_tokens"] + bucket["output_tokens"] + bucket["cache_write_tokens"]
    )
    bucket["total_tokens"] = bucket["non_cache_tokens"] + bucket["cache_read_tokens"]
    return bucket


def _totals(tasks) -> dict[str, Any]:
    total = _task_bucket("TOTAL")
    estimated_cost = 0.0
    estimated_count = 0
    for task in tasks:
        for field in (
            "requests",
            "done_events",
            "raw_result_events",
            "usage_events",
            "duration_ms",
            "duration_api_ms",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "non_cache_tokens",
            "total_tokens",
        ):
            total[field] += int(task.get(field) or 0)
        _add_cost(total, task.get("cost_usd"))
        for model in task.get("models") or []:
            _add_unique(total["models"], model)
        for session in task.get("sessions") or []:
            _add_unique(total["sessions"], session)
        est = _as_float(task.get("estimated_cost_usd"))
        if est is not None:
            estimated_cost += est
            estimated_count += 1
    if estimated_count:
        total["estimated_cost_usd"] = estimated_cost
    return _finalize_task(total)


def _pricing_candidates(
    *,
    override: str | None,
    task: dict[str, Any],
    state_model: str | None,
) -> list[str]:
    candidates = []
    if override:
        candidates.append(override)
    candidates.extend(task.get("models") or [])
    if state_model:
        candidates.append(state_model)
    out = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _attach_estimated_cost(task: dict[str, Any], *, candidates: list[str]) -> None:
    estimate = None
    for model in candidates:
        estimate = estimate_usage_cost(task, model)
        if estimate is not None:
            break
    if estimate is None:
        return
    task["estimated_cost_usd"] = estimate["total_usd"]
    task["estimated_cost_breakdown"] = {
        "input_usd": estimate["input_usd"],
        "output_usd": estimate["output_usd"],
        "cache_read_usd": estimate["cache_read_usd"],
        "cache_write_usd": estimate["cache_write_usd"],
    }
    task["pricing_model_id"] = estimate["model_price_id"]
    task["pricing_model"] = estimate["model"]
    task["pricing_source_url"] = estimate["source_url"]
    task["pricing_source_checked_at"] = estimate["source_checked_at"]
    task["pricing_notes"] = estimate["notes"]


def _add_unique(values: list[str], value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text and text not in values:
        values.append(text)


def _add_cost(bucket: dict[str, Any], value: Any) -> None:
    cost = _as_float(value)
    if cost is None:
        return
    bucket["cost_usd"] = (
        cost if bucket.get("cost_usd") is None else bucket["cost_usd"] + cost
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_duration_ms(value: Any) -> str:
    ms = _as_int(value) or 0
    if ms <= 0:
        return "0s"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{hours}h{rem_minutes:02d}m"


def _fmt_cost(value: Any) -> str:
    cost = _as_float(value)
    if cost is None:
        return "n/a"
    return f"${cost:.4f}"
