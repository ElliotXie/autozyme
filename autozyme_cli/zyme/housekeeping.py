"""Mechanical housekeeping scanner — flags surface-level dead code in
`pipeline/run.{R,py}` and prints a reminder via `zyme accept`.

Three metrics fire individually; size threshold + cooldown gate frequency:
  M2  dead `fast_*` helpers (defined but never installed and never called)
  M5  dead captured originals (`.orig_*` / `_orig_*` defined but never called)
  M10 stale round-history comments (`removed in round X` / `reverted in round Y`)
plus an unconditional file-size signal (M1 lines > 500).

Cooldown: after a reminder fires, we silence further reminders for
COOLDOWN_ROUNDS rounds. The cooldown clock is also reset by:
  - any accept whose hypothesis starts with `[housekeeping]` (the agent did
    the cleanup, no need to re-warn)
  - `zyme accept --dismiss-housekeeping` (agent judged findings intentional)

Mechanically scannable only — does NOT see internal dead branches inside
live functions, near-duplicate algorithms, etc. The reminder text tells
the agent to read the whole file; this module is the alarm, not the
diagnosis.
"""
import json
import re
from pathlib import Path

COOLDOWN_ROUNDS = 15
LINE_THRESHOLD = 500
# State lives under .zyme/ alongside the rest of the framework's task state
# (best.ref, baselines_stash.tsv, round.counter, verify_probe.cache). The
# legacy location was task_dir/.zyme_housekeeping.json — `_state_path`
# auto-migrates that file on first access so existing tasks don't break.
STATE_FILE_NAME = "housekeeping.json"
LEGACY_STATE_FILE = ".zyme_housekeeping.json"
HOUSEKEEPING_TAG = "[housekeeping]"


# ---------------------------------------------------------------------------
# Public interface (called by commands.py)
# ---------------------------------------------------------------------------

def maybe_print_reminder(task_dir: Path, current_round: int, hypothesis: str = "") -> None:
    """Run scan + cooldown gate; print reminder to stdout if appropriate.

    `hypothesis` is the round's hypothesis string. If it starts with
    `[housekeeping]`, this round IS the cleanup response — silence and
    reset the cooldown.
    """
    pipeline = _find_pipeline(task_dir)
    if pipeline is None:
        return

    if hypothesis.lstrip().startswith(HOUSEKEEPING_TAG):
        # Agent did a housekeeping round; reset cooldown regardless of findings.
        _write_state(task_dir, {"last_warned_round": current_round})
        return

    state = _read_state(task_dir)
    if state is not None:
        last = state.get("last_warned_round", -10**9)
        if current_round - last < COOLDOWN_ROUNDS:
            return  # cooldown active

    findings = _scan(pipeline)
    if not findings.fires():
        return

    print()  # spacer between accept output and reminder
    print(findings.format_reminder(pipeline))
    _write_state(task_dir, {"last_warned_round": current_round})


def mark_dismissed(task_dir: Path, current_round: int) -> None:
    """Called by `zyme accept --dismiss-housekeeping`. Resets cooldown."""
    _write_state(task_dir, {"last_warned_round": current_round})


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _find_pipeline(task_dir: Path):
    for ext in ("R", "py"):
        p = task_dir / "pipeline" / f"run.{ext}"
        if p.exists():
            return p
    return None


def _state_path(task_dir: Path) -> Path:
    """Resolve the housekeeping state-file path.

    Canonical location: `task_dir/.zyme/housekeeping.json`. If the legacy
    file (`task_dir/.zyme_housekeeping.json`) exists from a pre-migration
    task, move it into the canonical location first — keeps the cooldown
    state intact across the rename and removes the orphan untracked file
    that used to dirty `git status` after every `zyme accept`.
    """
    canonical = task_dir / ".zyme" / STATE_FILE_NAME
    legacy = task_dir / LEGACY_STATE_FILE
    if legacy.exists() and not canonical.exists():
        canonical.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(canonical)
    return canonical


def _read_state(task_dir: Path):
    f = _state_path(task_dir)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return None


def _write_state(task_dir: Path, state: dict) -> None:
    f = _state_path(task_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class Findings:
    def __init__(self):
        self.dead_fast = []     # list of (line_no, name)
        self.dead_orig = []
        self.stale = []         # list of (line_no, line_text)
        self.total_lines = 0
        self.n_install = 0
        self.n_getfromns = 0
        self.n_cpp_blocks = 0

    def fires(self) -> bool:
        return bool(self.dead_fast or self.dead_orig or self.stale
                    or self.total_lines > LINE_THRESHOLD)

    def format_reminder(self, pipeline_path: Path) -> str:
        lines = [
            f"[housekeeping reminder] Mechanical scan of {pipeline_path.name} "
            f"({self.total_lines} lines, {self.n_install} install_override, "
            f"{self.n_cpp_blocks} cpp blocks):"
        ]
        if self.dead_fast:
            lines.append(f"  - {len(self.dead_fast)} dead fast_* helpers:")
            for ln, name in self.dead_fast:
                lines.append(f"      L{ln}  {name}")
        if self.dead_orig:
            lines.append(f"  - {len(self.dead_orig)} dead .orig_* captures:")
            for ln, name in self.dead_orig:
                lines.append(f"      L{ln}  {name}")
        if self.stale:
            lines.append(f"  - {len(self.stale)} stale round-history comments:")
            for ln, text in self.stale[:3]:
                lines.append(f"      L{ln}  {text[:80]}")
            if len(self.stale) > 3:
                lines.append(f"      ... and {len(self.stale) - 3} more")
        if not (self.dead_fast or self.dead_orig or self.stale):
            lines.append(f"  - file > {LINE_THRESHOLD} lines, "
                         "consider a full read for internal bloat")

        lines.append(
            f"\nSee 2_iterate.md § Housekeeping. Open a [housekeeping] round, "
            f"or pass --dismiss-housekeeping to your next zyme accept "
            f"(silences {COOLDOWN_ROUNDS} rounds)."
        )
        return "\n".join(lines)


def _scan(pipeline_path: Path) -> Findings:
    src = pipeline_path.read_text()
    ext = pipeline_path.suffix.lstrip(".")
    f = Findings()
    f.total_lines = src.count("\n") + 1
    f.n_install = len(re.findall(r"install_(?:global_)?override\s*\(", src))
    f.n_getfromns = len(re.findall(r"getFromNamespace\s*\(", src))
    f.n_cpp_blocks = len(re.findall(r"cppFunction\s*\(|sourceCpp\s*\(", src))

    # M2: dead fast_* helpers
    if ext == "R":
        fast_def_re = re.compile(
            r"^\s*((?:fast_|\.fast_)[\w.]+)\s*<-\s*function", re.M)
    else:
        fast_def_re = re.compile(
            r"^\s*def\s+((?:fast_|_fast_)[\w]+)", re.M)

    inst_targets = set()
    for m in re.finditer(
            r"install_(?:global_)?override\s*\(([^)]*)\)", src, re.DOTALL):
        for n in re.findall(r"(_?\.?fast_[\w.]+)", m.group(1)):
            inst_targets.add(n)
    # Direct attribute assignment (Py): X.method = fast_xxx — be strict so we
    # don't match `# === fast_xxx` separator comments.
    for m in re.finditer(r"^\s*[\w.]+\s*=\s*(_?fast_[\w]+)\b", src, re.M):
        inst_targets.add(m.group(1))

    for m in fast_def_re.finditer(src):
        name = m.group(1)
        line_no = src[: m.start()].count("\n") + 1
        pat = r"(?<![\w.])" + re.escape(name) + r"(?![\w.])"
        refs = len(re.findall(pat, src))
        if name not in inst_targets and refs <= 1:
            f.dead_fast.append((line_no, name))

    # M5: dead captured originals
    if ext == "R":
        orig_def_re = re.compile(
            r"^\s*((?:\.|_)?orig_[\w.]+)\s*<-", re.M)
    else:
        orig_def_re = re.compile(
            r"^\s*((?:_)?orig_[\w]+)\s*=", re.M)

    for m in orig_def_re.finditer(src):
        name = m.group(1)
        line_no = src[: m.start()].count("\n") + 1
        pat = r"(?<![\w.])" + re.escape(name) + r"(?![\w.])"
        refs = len(re.findall(pat, src))
        if refs <= 1:
            f.dead_orig.append((line_no, name))

    # M10: stale round-history comments
    stale_re = re.compile(
        r"#.*(?:removed in round|reverted in round|deprecated in round|"
        r"dead code in round|see active_opts)", re.I)
    src_lines = src.split("\n")
    for m in stale_re.finditer(src):
        line_no = src[: m.start()].count("\n") + 1
        f.stale.append((line_no, src_lines[line_no - 1].strip()))

    return f
