"""`zyme validate` — LLM-based adversarial audit for hack detection.

Run at phase end (init or iterate), NOT per round. Spawns one of three agent
backends (claude / cursor / codex) headless, points it at a prompt under
`autozyme_cli/zyme/prompts/validate_{init,iterate}.md`, and captures the agent's
markdown report. The report is written to
`<autozyme-framework>/validation/<task>/v<N>/validate_{phase}_<ts>.md`, mirrored
to `<task_dir>/.zyme/validate/v<N>/` so portable task snapshots carry it, and
each Finding section is parsed into a TSV row in `<task_dir>/validate.tsv`.

Design notes:
- Open-ended: the prompts include A–G failure modes as few-shot examples,
  but the agent is explicitly allowed to propose new categories.
- Multi-backend: reuses zyme.dispatch.master.{find_agent_binary,_build_agent_cmd}
  so claude / cursor / codex all work through the same code path.
- File-output, not stdout-capture: the CLI computes the report path up front,
  mkdir's its parent, and passes the path to the agent via the launch message.
  The agent uses its Write tool to land the markdown report on disk; we read
  it back after the subprocess exits. This avoids paying output-token cost
  for the entire report, sidesteps stream-json schema drift across backends,
  and survives mid-stream truncation. The stderr stream is still tailed for
  live UX (one preview line per assistant turn), but is not the source of truth.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# `_build_agent_cmd` is currently private to dispatch.master. TODO: promote
# to a public `build_agent_cmd` once a second consumer (this file) lands
# and the dispatch refactor opens. Using the private name unblocks v0.
from zyme.dispatch.master import _build_agent_cmd, find_agent_binary
from zyme.utils import die, task_dir_from_args


# ---------------------------------------------------------------------------
# Public command entrypoints (cli.py wires these via set_defaults(func=...))
# ---------------------------------------------------------------------------

def cmd_validate_init(args) -> int:
    rc = _run_validate("init", args)
    if rc == 0 and getattr(args, "register_template", None):
        _auto_register_iterate_template(args)
    return rc


def cmd_validate_iterate(args) -> int:
    return _run_validate("iterate", args)


def _auto_register_iterate_template(args) -> None:
    """Register the validated task as an iterate-stage bench template.

    Called only when --register-template NAME is passed to `validate init` and
    the audit passes. Wraps cmd_bench_register_template so the user doesn't
    have to run a second command. Registration failure is non-fatal: validate
    already succeeded, so we print a warning and return rather than dying.
    """
    import argparse
    from zyme.commands.bench import cmd_bench_register_template

    task_dir = task_dir_from_args(args)
    name = args.register_template
    reg_args = argparse.Namespace(
        task_dir=str(task_dir),
        name=name,
        stage="iterate",
        at="post_init",
        commit=None,
        force=getattr(args, "register_force", False),
    )
    print(
        f"\nzyme validate init: registering '{name}' as iterate-stage bench template …",
        flush=True,
    )
    try:
        cmd_bench_register_template(reg_args)
    except SystemExit:
        sys.stderr.write(
            f"  [warn] bench register-template failed for '{name}' (see above). "
            "Validate result is still recorded.\n"
        )


def _run_validate(phase: str, args) -> int:
    task_dir = task_dir_from_args(args)
    _check_preconditions(phase, task_dir)

    prompt_path = _resolve_prompt_path(phase)
    report_path = _resolve_report_path(
        phase,
        task_dir,
        getattr(args, "out", None),
        getattr(args, "version", None),
    )
    # mkdir parent BEFORE launching the agent so its Write tool doesn't need
    # to figure out directory creation.
    report_path.parent.mkdir(parents=True, exist_ok=True)

    agent = args.agent
    if agent == "auto":
        from zyme.dispatch.master import detect_agent_binary
        agent, _ = detect_agent_binary()

    print(f"zyme validate {phase}: launching {agent} on {task_dir}", flush=True)
    print(f"  report target: {report_path}", flush=True)
    _invoke_validator(
        prompt_path=prompt_path,
        report_path=report_path,
        task_dir=task_dir,
        agent=agent,
        model=getattr(args, "model", None),
        effort=getattr(args, "effort", "high"),
    )

    if not report_path.is_file():
        die(
            f"validator agent exited but did not write the report to {report_path}. "
            "Re-run with --agent <other-backend> or check the agent's stderr."
        )
    report = report_path.read_text(encoding="utf-8")
    local_mirror = _mirror_report_locally(report_path, task_dir)
    findings = _parse_findings(report)
    tsv_path = task_dir / "validate.tsv"
    _append_tsv(
        tsv_path=tsv_path,
        phase=phase,
        agent=agent,
        model=getattr(args, "model", None),
        findings=findings,
        report_path=report_path,
    )

    _print_summary(phase, task_dir, agent, findings, report_path, local_mirror)
    return 0


def _mirror_report_locally(report_path: Path, task_dir: Path) -> Path:
    """Copy the framework-side report into <task_dir>/.zyme/validate/ so the
    audit travels with portable task snapshots alongside validate.tsv. If the
    framework path uses v<N>/ versioning, preserve it under the mirror."""
    mirror_dir = task_dir / ".zyme" / "validate"
    parent_name = report_path.parent.name
    if re.fullmatch(r"v\d+", parent_name):
        mirror_dir = mirror_dir / parent_name
    mirror_dir.mkdir(parents=True, exist_ok=True)
    dest = mirror_dir / report_path.name
    shutil.copy2(report_path, dest)
    return dest


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def _check_preconditions(phase: str, task_dir: Path) -> None:
    if not (task_dir / "task.yaml").is_file():
        die(f"task.yaml not found in {task_dir}")

    # init: reference must have produced at least one tier of outputs
    ref_outputs = task_dir / "reference_outputs"
    has_nested = ref_outputs.is_dir() and any(ref_outputs.iterdir())
    has_flat = bool(list(task_dir.glob("reference_output_*")))
    if not (has_nested or has_flat):
        die("no reference outputs found; run `zyme baseline reference` first")

    eval_py = task_dir / "evaluate.py"
    eval_r = task_dir / "evaluate.R"
    pipeline_eval_py = task_dir / "pipeline" / "evaluate.py"
    pipeline_eval_r = task_dir / "pipeline" / "evaluate.R"
    if not any(p.is_file() for p in (eval_py, eval_r, pipeline_eval_py, pipeline_eval_r)):
        die("no evaluate.{py,R} found in task dir or pipeline/")

    if phase == "iterate":
        results = task_dir / "results.tsv"
        if not results.is_file():
            die("results.tsv not found; no iterate rounds recorded yet")
        if not _results_has_data_row(results):
            die("results.tsv has no data rows; no iterate work to audit")
        best_ref = task_dir / ".zyme" / "best.ref"
        if not best_ref.is_file():
            die(".zyme/best.ref missing; no accepted iterate round")


def _results_has_data_row(results_tsv: Path) -> bool:
    """Cheap check: at least one non-header row exists. The `.zyme/best.ref`
    file is the authoritative "iterate happened" signal; we just want to
    confirm results.tsv isn't an empty header file."""
    try:
        with results_tsv.open("r", encoding="utf-8") as f:
            _header = f.readline()
            for line in f:
                if line.strip():
                    return True
        return False
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_PROMPT_FILE = {
    "init": "validate_init.md",
    "iterate": "validate_iterate.md",
}


def _resolve_prompt_path(phase: str) -> Path:
    # autozyme_cli/zyme/commands/validate.py -> autozyme_cli/zyme/prompts/
    zyme_root = Path(__file__).resolve().parent.parent
    p = zyme_root / "prompts" / _PROMPT_FILE[phase]
    if not p.is_file():
        die(f"validate prompt not found at {p}")
    return p


def _resolve_report_path(
    phase: str,
    task_dir: Path,
    out_override: str | None,
    version: str | None = None,
) -> Path:
    if out_override:
        return Path(out_override).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    # autozyme_cli/zyme/commands/ -> autozyme_cli/zyme/ -> autozyme_cli/ -> autozyme-framework/
    framework_root = Path(__file__).resolve().parent.parent.parent.parent
    task_val_dir = framework_root / "validation" / task_dir.name

    if version is None:
        # Auto-detect: re-runs of the current prompt round land in the highest
        # existing v<N>/. Bump to a new v<N+1>/ explicitly via --version when
        # the validator prompt changes.
        existing = []
        if task_val_dir.is_dir():
            for d in task_val_dir.iterdir():
                if d.is_dir() and re.fullmatch(r"v\d+", d.name):
                    existing.append(int(d.name[1:]))
        ver = f"v{max(existing) if existing else 1}"
    else:
        v = str(version).lstrip("vV")
        if not v.isdigit():
            die(f"--version must be an integer or v<int> (got {version!r})")
        ver = f"v{int(v)}"

    return task_val_dir / ver / f"validate_{phase}_{ts}.md"


# ---------------------------------------------------------------------------
# Agent invocation (multi-backend)
# ---------------------------------------------------------------------------

def _invoke_validator(
    *,
    prompt_path: Path,
    report_path: Path,
    task_dir: Path,
    agent: str,
    model: str | None,
    effort: str,
) -> None:
    """Spawn the agent, stream a stderr preview for UX, return when the
    subprocess exits. The agent is expected to land the markdown report on
    disk at `report_path` via its Write tool — the caller validates that
    afterwards."""
    if agent == "auto":
        from zyme.dispatch.master import detect_agent_binary
        agent, _binary = detect_agent_binary()
    find_agent_binary(agent)  # raises RuntimeError early if not on PATH

    # Build the launch message so the agent (a) reads the validator prompt
    # and (b) knows where to write its report. Passing `message` overrides
    # dispatch.master's default `read and follow <prompt>` string.
    launch_message = (
        f"Read and follow the validator prompt at {prompt_path}. "
        f"Write your final markdown report — following the schema in that prompt — "
        f"to the absolute path {report_path} using your Write tool. "
        "Create the file even if you only have one finding. "
        "Do not print the report content to stdout — your driver reads it from disk. "
        "End your turn after the file is written."
    )

    cmd = _build_agent_cmd(
        agent=agent,
        prompt=str(prompt_path),   # kept for backward-compat with dispatch helpers
        model=model,
        effort=effort,
        message=launch_message,
    )

    proc = subprocess.Popen(
        cmd,
        cwd=str(task_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,  # line-buffered
    )

    # Tail stdout for live preview only — the report itself lands on disk.
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        text = _extract_assistant_text(raw_line, agent)
        if text:
            preview = re.sub(r"\s+", " ", text)[:120]
            sys.stderr.write(f"  [validator] {preview}\n")
            sys.stderr.flush()

    proc.wait()
    err = proc.stderr.read() if proc.stderr else ""
    if proc.returncode != 0:
        die(
            f"validator agent {agent} exited with code {proc.returncode}\n"
            f"stderr:\n{err}"
        )


def _extract_assistant_text(raw_line: str, agent: str) -> str:
    """Extract the full assistant-text content from one stream-json/--json line.

    Returns "" if the line is not an assistant text event.

    Schemas:
    - claude / cursor: {type: "assistant", message: {content: [{type:"text", text:"..."}]}}
    - codex --json:    shape varies; commonly {msg: {type: "agent_message", message: "..."}}
                       or {type: "agent_message", content: "..."} or item-wrapped.
    """
    raw = raw_line.strip()
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(obj, dict):
        return ""

    # claude / cursor: "assistant" type with content blocks
    if obj.get("type") == "assistant":
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
        blocks = msg.get("content") or []
        parts = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text")
                if isinstance(t, str):
                    parts.append(t)
        if parts:
            return "\n".join(parts)

    # codex: walk msg / item wrappers for agent_message events
    inner = obj.get("msg") if isinstance(obj.get("msg"), dict) else (
        obj.get("item") if isinstance(obj.get("item"), dict) else obj
    )
    if isinstance(inner, dict):
        inner_type = str(
            inner.get("type") or inner.get("kind") or inner.get("event") or ""
        ).lower()
        if "agent_message" in inner_type or "assistant_message" in inner_type:
            for key in ("message", "text", "content"):
                v = inner.get(key)
                if isinstance(v, str) and v.strip():
                    return v
                if isinstance(v, list):
                    parts = []
                    for b in v:
                        if isinstance(b, dict):
                            t = b.get("text") or b.get("message") or b.get("content")
                            if isinstance(t, str):
                                parts.append(t)
                        elif isinstance(b, str):
                            parts.append(b)
                    if parts:
                        return "\n".join(parts)

    return ""


# ---------------------------------------------------------------------------
# Markdown report → TSV findings parser
# ---------------------------------------------------------------------------

# Matches: "### Finding 1 — LIKELY_HACK — Cat C - timer hoist"
# Also tolerates ASCII '-' as separator and missing surrounding whitespace.
_FINDING_HEADER = re.compile(
    r"^###\s+Finding\s+(\d+)\s*[—\-–]+\s*([A-Za-z_]+)\s*[—\-–]+\s*(.+?)\s*$",
    re.MULTILINE,
)

_FILE_REF_PAT = re.compile(
    r"\*\*(?:File:line evidence|File evidence|Setup artifact)\*\*\s*:\s*`?([^\n`]+?)`?\s*$",
    re.MULTILINE,
)
_MECHANISM_PAT = re.compile(
    r"\*\*Mechanism\*\*\s*:\s*(.+?)(?=\n\*\*[A-Z][^*\n]*\*\*\s*:|\n\n###\s|\n\n##\s|\Z)",
    re.DOTALL,
)


def _parse_findings(report: str) -> list[dict]:
    """Extract Finding sections from a validator markdown report.

    Returns list of dicts: {finding_id, severity, category, file_ref, line_ref, mechanism_summary}.
    Tolerant of missing fields; severity unrecognized -> kept as-is (string).
    """
    findings: list[dict] = []
    headers = list(_FINDING_HEADER.finditer(report))
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(report)
        body = report[start:end]
        file_ref_raw = _match_or_empty(_FILE_REF_PAT, body)
        file_ref, line_ref = _split_file_line(file_ref_raw)
        mech_raw = _match_or_empty(_MECHANISM_PAT, body)
        findings.append({
            "finding_id": int(h.group(1)),
            "severity": h.group(2).strip().upper(),
            "category": h.group(3).strip(),
            "file_ref": file_ref,
            "line_ref": line_ref,
            "mechanism_summary": _flatten_one_line(mech_raw, max_chars=400),
        })
    return findings


def _match_or_empty(pat: re.Pattern, body: str) -> str:
    m = pat.search(body)
    return m.group(1).strip() if m else ""


def _split_file_line(ref: str) -> tuple[str, str]:
    """`pipeline/run.py:218-265` -> ("pipeline/run.py", "218-265")."""
    if not ref:
        return "", ""
    ref = ref.strip().strip("`").strip()
    if ":" in ref:
        file_part, _, line_part = ref.rpartition(":")
        if re.fullmatch(r"[\d,\-\s]+", line_part):
            return file_part.strip(), line_part.strip()
    return ref, ""


def _flatten_one_line(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) > max_chars:
        flat = flat[:max_chars - 1] + "…"
    return flat


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------

_TSV_HEADER = [
    "timestamp", "phase", "validator_agent", "validator_model",
    "finding_id", "severity", "category",
    "file_ref", "line_ref", "mechanism_summary", "report_path",
]


def _append_tsv(
    *,
    tsv_path: Path,
    phase: str,
    agent: str,
    model: str | None,
    findings: list[dict],
    report_path: Path,
) -> None:
    is_new = not tsv_path.is_file()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model_str = model or "(default)"
    rel_report = _safe_relative(report_path, tsv_path.parent)

    with tsv_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("\t".join(_TSV_HEADER) + "\n")
        for fd in findings:
            row = [
                ts, phase, agent, model_str,
                str(fd["finding_id"]),
                fd["severity"],
                _tsv_safe(fd["category"]),
                _tsv_safe(fd["file_ref"]),
                _tsv_safe(fd["line_ref"]),
                _tsv_safe(fd["mechanism_summary"]),
                rel_report,
            ]
            f.write("\t".join(row) + "\n")


def _safe_relative(p: Path, base: Path) -> str:
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(p.resolve())


def _tsv_safe(s: str) -> str:
    """TSV cell escape: replace separators / newlines with spaces."""
    return (s or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")


# ---------------------------------------------------------------------------
# User-facing summary
# ---------------------------------------------------------------------------

_SEV_ORDER = (
    "FAIL", "LIKELY_HACK", "LIKELY_HACK_INVITED",
    "WEAK",
)


def _print_summary(
    phase: str,
    task_dir: Path,
    agent: str,
    findings: list[dict],
    report_path: Path,
    local_mirror: Path | None = None,
) -> None:
    counts: dict[str, int] = {}
    for fd in findings:
        counts[fd["severity"]] = counts.get(fd["severity"], 0) + 1
    print(f"\nzyme validate {phase} complete — agent={agent} task={task_dir.name}")
    print(f"  findings: {len(findings)}")
    for sev in _SEV_ORDER:
        if sev in counts:
            print(f"    {sev}: {counts[sev]}")
    # Unknown severities (validator went off-schema)
    other = [k for k in counts if k not in _SEV_ORDER]
    for sev in sorted(other):
        print(f"    {sev}: {counts[sev]} (off-schema)")
    print(f"  report: {report_path}")
    if local_mirror is not None:
        print(f"  mirror: {local_mirror}")
    print(f"  tsv:    {task_dir / 'validate.tsv'}")
