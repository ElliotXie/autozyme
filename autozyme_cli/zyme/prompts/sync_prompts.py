#!/usr/bin/env python3
"""Drift checker for the Bio / OtherField prompt sets.

Walks every `*.md` under `prompts/Bio` and `prompts/OtherField`, extracts each
`zyme <cmd> ...` invocation, and:

  1. Validates the subcommand + flags against the installed CLI's argparse
     spec (catches dead flags after CLI changes).

  2. For each parallel file pair (Bio/X.md ↔ OtherField/X.md), reports drift
     in flag usage for shared subcommands.

  3. Surfaces known sync patterns as suggestions (e.g., `--rerun` without
     `--n` should usually pair with `--n 1` in scale prompts).

Use as a periodic check after CLI flag changes. Not an auto-rewriter — the
two prompt sets diverge intentionally (Bio uses `zyme verify`; OtherField
doesn't), so blanket text-sync is wrong. The report tells YOU where to look.

Usage:
    python prompts/sync_prompts.py             # report drift + invalid commands
    python prompts/sync_prompts.py --strict    # exit 1 on any drift or error
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

try:
    from zyme.cli import build_parser
except ImportError:
    sys.exit(
        "zyme not importable. Install with `pip install -e autozyme_cli/` "
        "from the monorepo root, or run from inside autozyme_cli/."
    )

PROMPTS_ROOT = Path(__file__).resolve().parent
FIELDS = ("Bio", "OtherField")

# Match `zyme <cmd> [args...]`. The args run until end of line, a closing
# backtick (inline-code form), a trailing markdown table delimiter ` |`, or
# a comment marker. We deliberately stop at `|` to avoid eating the table
# cell separator that follows the command in the commands-table rows.
ZYME_INVOCATION_RE = re.compile(
    r"\bzyme\s+([a-z][a-z\-]*)\b([^\n`|]*)"
)

# Known sync invariants. Each tuple:
#   (subcommand, must_have_flags, satisfying_flags, file_glob, "why")
# An invocation matches the rule if it's a real call to `subcommand` (see
# command_like_context filter), names every flag in `must_have_flags`, and
# is in a file whose name matches `file_glob` ('*' = anywhere). It violates
# the rule if it doesn't include any flag from `satisfying_flags`.
SYNC_INVARIANTS = [
    # Scale-validation prompts: --rerun should pair with explicit --n
    # so per-call cost at xlarge OOD wall times is visible at the call site.
    ("run", ["--rerun"], ["--n"], "3_*_scaling.md",
     "`--rerun` defaults to 1 rep; in scale prompts, specify `--n 1` "
     "(single-shot) or `--n 3` (variance) so cost is explicit."),
    # `zyme verify` in scale prompts should pass `--phase validate` so
    # packaging-time matrices at the same commit don't share rows. Only
    # match real calls (those carrying --tiers/--threads/--cells/--reps).
    ("verify", ["--tiers"], ["--phase"], "3_*_scaling.md",
     "Verify in scale prompts should be tagged `--phase validate` so "
     "packaging-time matrices at the same commit don't share rows."),
    ("verify", ["--threads"], ["--phase"], "3_*_scaling.md",
     "Verify in scale prompts should be tagged `--phase validate`."),
    ("verify", ["--cells"], ["--phase"], "3_*_scaling.md",
     "Verify in scale prompts should be tagged `--phase validate`."),
]


def command_like_context(line: str, match_start: int) -> bool:
    """True if the regex match looks like an actual CLI invocation, not prose.

    Heuristics: match is at line start (possibly after indent + `$`), or
    inside backticks/fenced code, or in the first cell of a markdown table
    row. We exclude matches that follow prose markers like "triggered by",
    "after `", arrows, etc.
    """
    prefix = line[:match_start]
    # Common prose patterns we want to exclude.
    prose_markers = (
        "triggered by ", " after `", " before `", " calls ", " calls.",
        " uses ", " runs ", " invokes ", " via ", "by `",
    )
    lower_prefix = prefix.lower()
    for marker in prose_markers:
        if marker in lower_prefix:
            return False
    # Markdown flow arrows: `→ zyme bench →`
    if "→" in prefix and "zyme " not in prefix.split("→")[-1]:
        return False
    # Strip leading list / quote / indent markers, and a single `$ ` shell
    # prompt; also strip a leading backtick (inline code) and a leading
    # `|` (markdown table cell).
    stripped = prefix.lstrip(" \t").lstrip("|").lstrip(" ").lstrip("`").lstrip(" `$").lstrip()
    # If after stripping, the prefix is empty, the command is at "start of
    # significant content" — looks like a real invocation.
    if stripped == "":
        return True
    # If the immediate preceding char is a backtick or `$ `, also treat
    # as command-like.
    if prefix.rstrip().endswith("`") or prefix.rstrip().endswith("$"):
        return True
    return False


def in_fenced_code(text: str, line_idx: int) -> bool:
    """True if the line at `line_idx` is inside a ``` fenced block."""
    lines = text.splitlines()
    fence_count = sum(1 for ln in lines[:line_idx] if ln.strip().startswith("```"))
    return fence_count % 2 == 1


def file_matches_glob(filename: str, glob: str) -> bool:
    """fnmatch-equivalent for the small set of patterns we use."""
    if glob == "*":
        return True
    import fnmatch
    return fnmatch.fnmatch(filename, glob)


def all_subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Return {subcommand_name: subparser}."""
    out = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            out.update(action.choices)
    return out


def all_flags_for(subparser: argparse.ArgumentParser) -> set[str]:
    """Return every long-form flag accepted by a subparser ('--xxx')."""
    flags = set()
    for action in subparser._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                flags.add(opt)
    return flags


def nested_subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Return nested subcommand choices for `parser`, if any."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def clean_token(tok: str) -> str:
    """Trim markdown optional-arg punctuation around a shell token."""
    return tok.strip("[](),.;")


def flag_from_token(tok: str) -> str | None:
    cleaned = clean_token(tok)
    if not cleaned.startswith("--"):
        return None
    return cleaned.split("=", 1)[0]


def flag_scope_for(cmd: str, tokens: list[str], subparsers: dict) -> tuple[set[str], str]:
    """Return valid flags for a top-level command plus any nested subcommand.

    Example: `zyme baseline record --tier tiny` should validate `--tier`
    against the `baseline record` subparser, not the parent `baseline` parser.
    We do not fully parse required positional args here; this checker is only
    meant to catch stale flag names in prompts.
    """
    current = subparsers[cmd]
    path = [cmd]
    flags = set(all_flags_for(current))
    remaining = tokens

    while True:
        choices = nested_subcommands(current)
        if not choices:
            break
        match = None
        for i, tok in enumerate(remaining):
            cleaned = clean_token(tok)
            if cleaned in choices:
                match = (i, cleaned)
                break
        if match is None:
            break
        idx, subcmd = match
        current = choices[subcmd]
        path.append(subcmd)
        flags.update(all_flags_for(current))
        remaining = remaining[idx + 1:]

    return flags, " ".join(path)


def extract_invocations(md_path: Path) -> list[tuple[int, str, str, str]]:
    """Yield (line_no, cmd, args_str, raw_line) for every `zyme <cmd>` in the file.

    Filters out prose mentions ("triggered by zyme reminder", "after `zyme bench`")
    using the command-like-context heuristic.
    """
    text = md_path.read_text()
    lines = text.splitlines()
    invocations = []
    for i, line in enumerate(lines, start=1):
        for m in ZYME_INVOCATION_RE.finditer(line):
            if not command_like_context(line, m.start()):
                continue
            cmd = m.group(1)
            tail = m.group(2).strip()
            invocations.append((i, cmd, tail, line.strip()))
    return invocations


def validate_invocation(cmd: str, args_str: str, parser: argparse.ArgumentParser,
                        subparsers: dict) -> tuple[bool, str]:
    """Try to parse the command. Return (valid, reason)."""
    if cmd not in subparsers:
        return False, f"unknown subcommand `{cmd}`"
    # Tokenize the args. Replace shell-meta placeholders (`<x>`, `'<y>'`)
    # with sentinel strings so argparse doesn't choke on missing values.
    sanitized = re.sub(r"<[^>\s]+>", "PLACEHOLDER", args_str)
    sanitized = re.sub(r"\\\n", " ", sanitized)
    try:
        tokens = shlex.split(sanitized)
    except ValueError as e:
        return False, f"shlex parse failed: {e}"
    # Strip continuation backslashes / line breaks / leading shell prompts.
    tokens = [t for t in tokens if t and t not in ("\\",)]
    # Only check that flags are recognized by the parser for this command path;
    # don't require values to be valid (the prompt has `<placeholder>`
    # everywhere). Nested command families like `baseline record` and `bench
    # register-template` need their child parser's flags.
    valid_flags, cmd_path = flag_scope_for(cmd, tokens, subparsers)
    bad = []
    for tok in tokens:
        flag = flag_from_token(tok)
        if flag and flag not in valid_flags:
            bad.append(flag)
    if bad:
        return False, f"unknown flag(s) for `{cmd_path}`: {', '.join(sorted(set(bad)))}"
    return True, ""


def collect_per_file(field: str) -> dict[str, list[tuple[int, str, str, str]]]:
    """Return {basename: [(line, cmd, args, raw), ...]} for a field's prompts."""
    out = {}
    field_dir = PROMPTS_ROOT / field
    if not field_dir.exists():
        return out
    for md in sorted(field_dir.glob("*.md")):
        out[md.name] = extract_invocations(md)
    return out


def parallel_basenames(bio: dict, other: dict) -> list[str]:
    """Files that exist in both fields. Used for cross-field drift checks."""
    common = set(bio.keys()) & set(other.keys())
    # 3_*_scaling.md is the named odd-pair (3_validate vs 3_expand). Match by
    # numeric prefix so we still pair them.
    by_prefix_bio = {n.split("_")[0]: n for n in bio.keys()}
    by_prefix_other = {n.split("_")[0]: n for n in other.keys()}
    paired = []
    for prefix in sorted(set(by_prefix_bio) & set(by_prefix_other)):
        b = by_prefix_bio[prefix]
        o = by_prefix_other[prefix]
        paired.append((b, o))
    return paired


def cmd_signature(invocations, cmd_filter=None):
    """Group invocations by subcommand. Return {cmd: {flag_set frozenset}}."""
    by_cmd: dict[str, set] = {}
    for _, cmd, args, _ in invocations:
        if cmd_filter and cmd != cmd_filter:
            continue
        flags = frozenset(
            tok.split("=", 1)[0]
            for tok in shlex.split(re.sub(r"<[^>\s]+>", "PH", args)) if tok.startswith("--")
        )
        by_cmd.setdefault(cmd, set()).add(flags)
    return by_cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 on any drift or invalid command (CI-friendly).")
    ap.add_argument("--cmd", default=None,
                    help="Limit drift report to a single subcommand "
                         "(e.g., `--cmd run` to inspect just `zyme run` patterns).")
    args = ap.parse_args()

    parser = build_parser()
    subs = all_subcommands(parser)

    bio = collect_per_file("Bio")
    other = collect_per_file("OtherField")

    invalid_count = 0
    drift_count = 0

    # Section 1: invalid commands per file.
    print("=" * 70)
    print("Section 1: command validity (against installed CLI argparse spec)")
    print("=" * 70)
    for field, files in (("Bio", bio), ("OtherField", other)):
        for basename, invocs in files.items():
            file_invalid = []
            for line_no, cmd, argstr, raw in invocs:
                ok, reason = validate_invocation(cmd, argstr, parser, subs)
                if not ok:
                    file_invalid.append((line_no, cmd, argstr, reason, raw))
            if file_invalid:
                rel = f"{field}/{basename}"
                print(f"\n  {rel}:")
                for ln, cmd, argstr, reason, raw in file_invalid:
                    print(f"    L{ln}  zyme {cmd} {argstr[:60]}")
                    print(f"          → {reason}")
                    print(f"          (line: {raw[:100]})")
                    invalid_count += 1

    if invalid_count == 0:
        print("\n  ✓ All `zyme ...` invocations parse cleanly.")

    # Section 2: cross-field drift in shared subcommands.
    print()
    print("=" * 70)
    print("Section 2: cross-field flag drift (Bio ↔ OtherField parallel files)")
    print("=" * 70)
    for bio_name, other_name in parallel_basenames(bio, other):
        bio_sig = cmd_signature(bio[bio_name], cmd_filter=args.cmd)
        other_sig = cmd_signature(other[other_name], cmd_filter=args.cmd)
        all_cmds = set(bio_sig) | set(other_sig)
        differences = []
        for cmd in sorted(all_cmds):
            bio_sets = bio_sig.get(cmd, set())
            other_sets = other_sig.get(cmd, set())
            # Treat as drift if the SETS of flag-sets diverge.
            if bio_sets != other_sets:
                differences.append((cmd, bio_sets, other_sets))
        if differences:
            print(f"\n  {bio_name} ↔ {other_name}:")
            for cmd, b, o in differences:
                print(f"    `zyme {cmd}` flag-set drift:")
                only_bio = b - o
                only_other = o - b
                shared = b & o
                if shared:
                    for s in sorted(shared, key=lambda x: sorted(x)):
                        flags_str = " ".join(sorted(s)) if s else "(no flags)"
                        print(f"      both         : {flags_str}")
                if only_bio:
                    for s in sorted(only_bio, key=lambda x: sorted(x)):
                        flags_str = " ".join(sorted(s)) if s else "(no flags)"
                        print(f"      Bio only     : {flags_str}")
                if only_other:
                    for s in sorted(only_other, key=lambda x: sorted(x)):
                        flags_str = " ".join(sorted(s)) if s else "(no flags)"
                        print(f"      OtherField   : {flags_str}")
                drift_count += 1

    if drift_count == 0:
        print("\n  ✓ No flag-set drift detected across parallel prompt files.")

    # Section 3: known sync invariant violations.
    print()
    print("=" * 70)
    print("Section 3: sync invariants (known patterns)")
    print("=" * 70)
    invariant_hits = 0
    # Lines whose surrounding prose explains a default behavior are
    # documentation, not literal invocations — `zyme run --rerun` "runs 3 reps
    # by default" should not be flagged as missing `--n` when the next clause
    # is teaching the reader what the default IS.
    docs_signal = re.compile(
        r"\b(default|defaults|by default|behaves|means|re-measure|opts out)\b",
        re.IGNORECASE,
    )
    for field, files in (("Bio", bio), ("OtherField", other)):
        for basename, invocs in files.items():
            for line_no, cmd, argstr, raw in invocs:
                for inv_cmd, must_have, satisfying, file_glob, why in SYNC_INVARIANTS:
                    if cmd != inv_cmd:
                        continue
                    if not file_matches_glob(basename, file_glob):
                        continue
                    flags_in_args = set(re.findall(r"--[a-z\-]+", argstr))
                    if must_have and not all(f in flags_in_args for f in must_have):
                        continue
                    if inv_cmd == "verify" and field == "OtherField":
                        continue
                    if any(f in flags_in_args for f in satisfying):
                        continue
                    if docs_signal.search(raw):
                        continue  # documentation prose, not a literal invocation
                    rel = f"{field}/{basename}"
                    print(f"\n  {rel}  L{line_no}:")
                    print(f"    zyme {cmd} {argstr[:80]}")
                    print(f"    Missing one of: {', '.join(satisfying)}")
                    print(f"    Why: {why}")
                    invariant_hits += 1

    if invariant_hits == 0:
        print("\n  ✓ All sync invariants satisfied.")

    # Summary.
    print()
    print("=" * 70)
    total = invalid_count + drift_count + invariant_hits
    if total == 0:
        print("RESULT: ✓ all checks pass")
    else:
        print(f"RESULT: {invalid_count} invalid command(s), "
              f"{drift_count} drift block(s), {invariant_hits} invariant violation(s)")
    print("=" * 70)

    if args.strict and total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
