"""Cross-module helpers used by 2+ command modules."""

from pathlib import Path


# Framework package root: zyme/. Resolved once here so each command module can
# `from zyme.commands._shared import FRAMEWORK_ROOT` instead of recomputing
# `Path(__file__).resolve().parent.parent` per file. _shared.py is one level
# under zyme/, so .parent.parent walks from .../zyme/commands/_shared.py to
# .../zyme/.
FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent


_MEMORY_HEADERS = {
    "discoveries.md": (
        "# discoveries.md — {task_name}\n\n"
        "**Append-only log of non-obvious technical facts** uncovered while optimizing this target. "
        "The agent reads this first on session start and skips paying the discovery cost again. "
        "The compaction agent preserves entries here verbatim — do not summarize.\n"
    ),
    "active_opts.md": (
        "# active_opts.md — {task_name}\n\n"
        "**Stack of accepted optimizations that compose into the current `best`.** Overwrite-in-place "
        "when an opt gets superseded — this file should always describe the *current* state of "
        "`pipeline/run.{{py,R}}`, not its history.\n"
    ),
    "dead_ends.md": (
        "# dead_ends.md — {task_name}\n\n"
        "**One entry per falsified angle.** Overwrite-in-place: when you try a variation of an "
        "existing dead end, update its entry with the new variation tried + why it also failed, "
        "rather than create a new entry. Goal: prevent re-trying the same angle in slightly different clothes.\n"
    ),
}


_FAMILY_SKELETON = (
    "# Method family — <fill in: package or workflow name>\n\n"
    "**Verdict:** TBD — fill during init Step 1.\n\n"
    "<one short paragraph explaining the call chain and the verdict — for \"single\", note the "
    "public API surface is one function and call out that internal helpers are private; for "
    "\"split\", list each recommended sibling with role, estimated cost share, and the reason it qualifies>\n\n"
    "## Suggested next CLI invocations  (only if siblings recommended)\n\n"
    "```\n"
    "cd <new_task_dir> && python <path>/bin/zyme init <upstream_url> <function> --dataset <path>\n"
    "...\n"
    "```\n"
)




def _write_memory_skeleton(task_dir: Path, task_name: str) -> None:
    """Write the three narrative memory files (discoveries.md / active_opts.md /
    dead_ends.md) with task-name-substituted headers. Reused by `zyme init` and
    `zyme bench register-template` (memory/ is gitignored — not in git archive)."""
    mem = task_dir / "memory"
    mem.mkdir(exist_ok=True)
    for name, template_str in _MEMORY_HEADERS.items():
        (mem / name).write_text(
            template_str.format(task_name=task_name), encoding="utf-8"
        )




def _read_results_rows(results_tsv):
    """Parse results.tsv → list of dicts. Tolerates missing columns (legacy schemas)."""
    text = results_tsv.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for raw in lines[1:]:
        parts = raw.split("\t")
        rows.append({h: (parts[i] if i < len(parts) else "")
                     for i, h in enumerate(header)})
    return rows
