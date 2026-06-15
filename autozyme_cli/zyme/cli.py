"""zyme CLI entry point — argparse setup + main dispatch."""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from zyme import __version__
from zyme.audit import AuditContext


# Lifecycle grouping for the top-level `zyme --help` listing. Membership ⇒
# one row per command under the named section; commands not listed here fall
# under "Other" so a forgotten add still shows up. Order within a group is the
# display order. Keep this in sync with build_parser() below — the smoke test
# `tests/test_cli_smoke.py` enforces every registered subcommand is reachable.
_TOP_LEVEL_GROUPS = [
    ("Core loop",   ["init", "run", "dryrun", "accept", "reject", "rollback", "iterate"]),
    ("Measurement", ["baseline", "verify", "attest", "attest-sweep", "backfill", "validate"]),
    ("Analysis",    ["status", "plot", "scan", "registry", "audit", "cost", "cost-capture"]),
    ("Workflow",    ["dispatch", "bench", "prompt"]),
    ("Packaging",   ["package"]),
    ("Utility",     ["inspect-parallelism", "datasets", "publish-speedups"]),
]


class _GroupedHelpFormatter(argparse.HelpFormatter):
    """Render the top-level subcommand list with lifecycle section headers.

    Argparse's default listing dumps every subcommand under one positional-args
    block; with 18+ entries that's a wall of text and agents have to scan it
    linearly. This formatter intercepts the subparsers action and prints the
    same data grouped by lifecycle role (Core loop, Measurement, …) so the
    structure is visible at a glance. Falls back to the default rendering for
    every other action.
    """
    def _format_action(self, action):
        if not isinstance(action, argparse._SubParsersAction):
            return super()._format_action(action)
        # Map name → (subparser, help text) by walking _choices_actions.
        help_by_name = {}
        for choice_action in action._choices_actions:
            help_by_name[choice_action.dest] = choice_action.help or ""
        registered = set(action.choices.keys())
        listed = {n for _, names in _TOP_LEVEL_GROUPS for n in names}
        leftover = sorted(registered - listed)
        groups = list(_TOP_LEVEL_GROUPS)
        if leftover:
            groups.append(("Other", leftover))

        lines = []
        # Match argparse's indentation: 2-space indent for section header,
        # 4-space indent for command rows; column-align the help text to ~22.
        for header, names in groups:
            present = [n for n in names if n in registered]
            if not present:
                continue
            lines.append(f"  {header}:")
            for n in present:
                hlp = help_by_name.get(n, "")
                # Truncate help to one line so the section grid stays readable.
                hlp = hlp.split("\n", 1)[0]
                lines.append(f"    {n:<22} {hlp}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def _parse_token_budget(value: str) -> int:
    raw = str(value).strip().replace(",", "").replace("_", "").lower()
    multiplier = 1
    for suffix, mult in (("k", 1_000), ("m", 1_000_000), ("b", 1_000_000_000)):
        if raw.endswith(suffix):
            raw = raw[:-1]
            multiplier = mult
            break
    try:
        tokens = int(float(raw) * multiplier)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "token budget must be a positive integer, optionally suffixed with k/m/b"
        ) from e
    if tokens <= 0:
        raise argparse.ArgumentTypeError("token budget must be positive")
    return tokens


from zyme.commands import (
    cmd_init, cmd_init_check, cmd_init_attest,
    cmd_run, cmd_accept, cmd_reject, cmd_dryrun,
    cmd_record_baseline, cmd_record_noise, cmd_reference, cmd_promote_baseline,
    cmd_baseline_list, cmd_baseline_show, cmd_baseline_rebench,
    cmd_rollback,
    cmd_iterate,
    cmd_verify,
    cmd_attest,
    cmd_attest_sweep,
    cmd_backfill,
    cmd_publish_speedups,
    cmd_plot,
    cmd_status,
    cmd_scan,
    cmd_inspect_parallelism,
    cmd_dispatch, cmd_dispatch_status, cmd_dispatch_usage, cmd_dispatch_prices,
    cmd_dispatch_resume, cmd_dispatch_logs, cmd_dispatch_stop, cmd_dispatch_wait,
    cmd_prompt_save, cmd_prompt_list, cmd_prompt_show, cmd_prompt_diff,
    cmd_prompt_use, cmd_prompt_annotate,
    cmd_bench_register_template, cmd_bench_list_templates,
    cmd_bench_doctor, cmd_bench_status, cmd_bench_init, cmd_bench_start,
    cmd_bench_usage, cmd_bench_prices, cmd_bench_list,
    cmd_registry_rebuild, cmd_registry_query, cmd_registry_suggest, cmd_registry_list,
    cmd_audit,
    cmd_cost, cmd_cost_capture,
    cmd_validate_init, cmd_validate_iterate,
    cmd_profile,
    cmd_report,
    add_datasets_subparser,
    cmd_package_lint, cmd_package_check_intercept, cmd_package_check_versions,
    cmd_package_preflight,
    cmd_package_smoke_parity, cmd_package_sync_manifests,
)


def build_parser():
    p = argparse.ArgumentParser(
        prog="zyme",
        description="autozyme-framework CLI",
        formatter_class=_GroupedHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"zyme {__version__}")
    sub = p.add_subparsers(
        dest="cmd",
        required=True,
        title="commands",
        metavar="<command>",
    )

    # Common parent parser: `--task-dir` is shared by every task-scoped command.
    # Workspace-level commands (init, scan, dispatch, …) don't inherit it.
    task_dir_parent = argparse.ArgumentParser(add_help=False)
    task_dir_parent.add_argument(
        "--task-dir", dest="task_dir", default=None,
        help="Task directory (default: cwd).",
    )

    pi = sub.add_parser(
        "init",
        help="Scaffold a new task in the current directory (cwd basename = task name)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Scaffold from a GitHub URL; CLI clones into upstream_repo/, picks main entry point\n"
            "  mkdir test_findmarkers && cd test_findmarkers\n"
            "  zyme init https://github.com/satijalab/seurat\n\n"
            "  # Local repo + explicit function + dataset hint\n"
            "  zyme init /path/to/upstream_repo FindAllMarkers --dataset /data/pbmc.rds\n\n"
            "  # Non-bio task (uses the OtherField prompt set instead of Bio)\n"
            "  zyme init https://github.com/astropy/astropy --field OtherField\n"
        ),
    )
    pi.add_argument("target_repo", help="URL or local path of the upstream library being optimized")
    pi.add_argument("target_function", nargs="?", default=None,
                    help="(optional) Function name / entry point. If omitted, the init agent picks the obvious main entry from target_repo.")
    pi.add_argument("--dataset", dest="dataset", default=None,
                    help="(optional) Local path to dev dataset. Persisted as `task.yaml::dataset_hint`; "
                         "init agent uses it as the source path. If omitted, the init agent searches "
                         "(framework's recommended-datasets file → online → ask user).")
    pi.add_argument("--field", dest="field", default="Bio",
                    help="Prompt set to copy into the task: Bio (default — single-cell / bioinformatics framing) "
                         "or OtherField (generic performance-engineer framing). New sets can be added by creating "
                         "a sibling directory under framework/prompts/.")
    pi.add_argument("--language", choices=["python", "R"], default=None,
                    help="Source language of the target. If omitted, CLI sniffs target_repo (DESCRIPTION → R, "
                         "pyproject.toml/setup.py → Python). If sniff fails, init proceeds without renaming "
                         ".template files and warns; the init agent renames them.")
    pi.add_argument("--no-clone", dest="no_clone", action="store_true",
                    help="Don't auto-clone target_repo when it's a URL. Default: CLI clones the URL into "
                         "upstream_repo/ at init time. Local paths are never cloned.")
    pi.add_argument("--no-bench-snapshot", dest="no_bench_snapshot",
                    action="store_true",
                    help="Skip the auto-registration of an init-stage bench template "
                         "(default: ON — every `zyme init` also drops a snapshot at "
                         "PromptLab/bench_templates/init/<task_name>/, both as a rollback "
                         "anchor and as a seed for future init-prompt benchmarking).")
    pi.set_defaults(func=cmd_init)

    pic = sub.add_parser(
        "init-check",
        parents=[task_dir_parent],
        help="End-of-init self-check: per-tier baseline / reference_output / noise "
             "tickbox + scaffold checks. Run after `zyme init` + reference + noise "
             "to verify everything landed coherently. Exits 1 if any required check "
             "fails so handoff scripts can gate on it.",
    )
    pic.add_argument("--parity", action="store_true",
                     help="Also run pipeline + evaluate against each tier's "
                          "reference_outputs/<tier>/ and assert all metrics are "
                          "perfect (gte→1.0, lte→0.0). Catches save/load asymmetry "
                          "bugs between reference.{R,py} and pipeline/run.{R,py} "
                          "before they leak into iterate rounds.")
    pic.set_defaults(func=cmd_init_check)

    pia = sub.add_parser(
        "init-attest",
        help="Scaffold a post-publication attest task under postpublication/ "
             "(clones an existing task's structure for an agent to fill in)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Post-publication workflow: write the accelerator override in the\n"
            "shipped patch (e.g. autozyme_r/inst/patches/seurat/patch.R), then\n"
            "`init-attest` a task, fill it, attest, and sync the patch code to\n"
            "release. The task is NOT wired into any attest manifest, so\n"
            "publish auto-skips and paper numbers stay frozen.\n\n"
            "Examples:\n"
            "  # clone find_markers' shape into postpublication/find_neighbors_single/\n"
            "  zyme init-attest find_neighbors_single --target 'Seurat::FindNeighbors'\n\n"
            "  # clone a different existing task as the starting point\n"
            "  zyme init-attest my_task --like find_markers --patch seurat\n"
        ),
    )
    pia.add_argument("name", help="New task name (becomes postpublication/<name>/)")
    pia.add_argument("--like", default="find_markers",
                     help="Existing postpublication task to clone structure from "
                          "(task.yaml, attest/smoke.R, evaluate.R, .gitignore, data "
                          "symlink). Default: find_markers.")
    pia.add_argument("--target", default=None,
                     help="(optional) target_function to substitute into task.yaml "
                          "(e.g. 'Seurat::FindMarkers').")
    pia.add_argument("--patch", default="seurat",
                     help="Patch name used in the printed attest command's --name "
                          "(default: seurat).")
    pia.add_argument("--dest", default=None,
                     help="Parent dir for the task (default: framework/postpublication).")
    pia.add_argument("--force", action="store_true",
                     help="Overwrite <name>/ if it already exists.")
    pia.set_defaults(func=cmd_init_attest)

    pr = sub.add_parser(
        "run", parents=[task_dir_parent],
        help="Commit pipeline change with hypothesis, run pipeline + evaluate, log + snapshot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Fresh round at the default (first-listed) tier\n"
            "  zyme run \"vectorize wilcoxon\"\n\n"
            "  # Fresh round, decision row at tiny + free secondary measurements at medium/large\n"
            "  zyme run \"sparse densify\" --dataset tiny --extra-tiers medium,large\n\n"
            "  # Re-measure HEAD (status=rerun, no budget consumed; default 1 rep)\n"
            "  zyme run --rerun --dataset medium\n\n"
            "  # Borderline call: 3 reps with mean ± stdev to separate signal from noise\n"
            "  zyme run --rerun --dataset tiny --n 3\n\n"
            "  # Setup commit (e.g. wire ZYME_THREADS) — advances best.ref so reject won't wipe it\n"
            "  zyme run --setup \"wire ZYME_THREADS through pipeline/run.R\"\n\n"
            "  # Phase 3 fix-loop (auto-prefixes hypothesis with [scale-fix])\n"
            "  zyme run \"fix BLAS oversubscription at thread=8/large\" --phase validate --dataset large\n"
        ),
    )
    pr.add_argument("hypothesis", nargs="?", default=None,
                    help="One-line hypothesis (becomes commit message). Keep ≤120 chars; details go in -m description "
                         "after accept/reject. Wrap in single quotes if it contains parens, slashes, or `()`. "
                         "Omit when using --rerun.")
    pr.add_argument("--rerun", action="store_true",
                    help="Re-measure the committed HEAD (not the working tree — uncommitted edits are NOT reflected). "
                         "Doesn't consume round budget. Combines with --extra-tiers for free multi-tier remeasurement.")
    pr.add_argument("--setup", default=None, metavar="MSG",
                    help="Setup-commit mode: stage `pipeline/`, `setup/`, `task.yaml`, "
                         "`reference.{py,R}`, and `evaluate.{py,R}` (if changed), "
                         "commit with `setup: <MSG>` as the message, AND advance `best.ref` to the new SHA. "
                         "NO pipeline run, NO results.tsv row, NO round consumed. Mutually exclusive with --rerun "
                         "and the positional hypothesis. "
                         "Use only for task-definition repairs that must land on HEAD before iteration or "
                         "verify/validate work (threading wiring, new tier entries, tier-param table extensions, "
                         "or evaluator repairs). Refuses to run while a pending decision row exists. "
                         "DO NOT replace with plain `git commit -m 'setup: ...'`: that does NOT advance `best.ref`, "
                         "so the first `zyme reject` in the next fix-loop will hard-reset past the setup commit "
                         "and silently wipe your wiring. --setup is the only safe way to land setup state.")
    pr.add_argument("--n", dest="n_reps", type=int, default=None,
                    help="(--rerun only) Repeat the rerun N times. Default = 1 (one quick "
                         "re-check). Pass --n 3 (or more) for borderline calls where "
                         "mean ± stdev is needed to separate signal from noise.")
    pr.add_argument("--dataset", dest="dataset", default=None,
                    help="Primary tier (or dataset name) to run against. "
                         "FRESH ROUND: this is the decision row that counts toward the "
                         "50-round budget (status=pending). WITH --rerun: this is one of "
                         "the tiers being re-measured (status=rerun, no budget consumed). "
                         "Default = first dataset listed in task.yaml.")
    pr.add_argument("--extra-tiers", dest="extra_tiers", default=None,
                    help="Comma-separated tier/name list of ADDITIONAL free measurements "
                         "(e.g. medium,large). Always written as status=rerun, never counts "
                         "toward the round budget — meaning is the same regardless of --rerun. "
                         "Use to bundle multi-tier observations into one shot: "
                         "FRESH ROUND `--dataset tiny --extra-tiers medium,large` records "
                         "tiny as the decision row plus free medium/large measurements at "
                         "the same commit. WITH --rerun, all of (--dataset, --extra-tiers) "
                         "are free reruns of HEAD.")
    pr.add_argument("--phase", choices=["optimize", "validate", "memory"], default="optimize",
                    help="Which phase this run belongs to. `optimize` (default) is the "
                         "50-round speed-up budget. `memory` is the 50-round "
                         "memory-optimization loop (separate counter). `validate` is "
                         "the Phase 3 fix-loop budget (30 rounds, separate counter). "
                         "Stamped on every row written; cmd_status / cmd_plot filter by phase.")
    pr.add_argument("--thread", type=int, default=None,
                    help="Thread count for this run. Default reads "
                         "task.yaml::baseline_threads[0] (the canonical regime declared "
                         "at init); falls back to 1 only when that field is absent. "
                         "Pipeline subprocess sees ZYME_THREADS=<N>. Every row written "
                         "is tagged with this thread, and speedup_pct divides by the "
                         "same-thread baseline.")
    pr.add_argument("--yes", "-y", dest="yes", action="store_true",
                    help="Skip the safety gate that blocks --rerun when projected wall time "
                         "exceeds 10 min. Without --yes you must drop reps (--n 1) or tiers "
                         "(--extra-tiers) until the projection fits, OR pass --yes to acknowledge "
                         "the cost. Catches the 'forgot --n 1 on OOD tier' footgun.")
    pr.add_argument("--bypass-hoist", dest="bypass_hoist", default=None, metavar="REASON",
                    help="Bypass the hoist_audit block for THIS round only. REASON is appended "
                         "to .zyme/hoist_log.jsonl alongside the violation details. Use when you "
                         "genuinely believe the check misfired (synthetic dummy with non-production "
                         "args, runtime knob a user would also flip, etc.). For permanent "
                         "exemptions tied to the task's contract, declare task.yaml::hoist_exempt "
                         "instead. Repeated bypasses on the same pattern are a signal to fix the "
                         "code or promote to hoist_exempt.")
    pr.set_defaults(func=cmd_run)

    pd = sub.add_parser("dryrun", parents=[task_dir_parent],
                        help="Run pipeline only (skip evaluate, no commit, no results.tsv) — for syntactic / compilation checks")
    pd.add_argument("--dataset", dest="dataset", default=None,
                    help="Tier or dataset name (default = first listed in task.yaml)")
    pd.set_defaults(func=cmd_dryrun)

    ppr = sub.add_parser(
        "profile", parents=[task_dir_parent],
        help="Capture CPU+memory profile for one pipeline run (default: "
             "Scalene/profvis with cpu fallback). Diagnostic — does NOT "
             "write results.tsv, commit, or run evaluate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Backends:\n"
            "  cpu     cProfile (Python) / Rprof (R). Stdlib, deterministic, "
            "function-level CPU.\n"
            "  full    Scalene (Python) / profvis (R). Line-level CPU+mem+"
            "native split. Optional install — falls back to cpu when missing.\n"
            "  mem     memray (Python) / Rprof+memory (R). Allocation tracking "
            "with native attribution. Refuses to fall back (different signal axis).\n"
            "  native  macOS sample(1) external observer. Sees BLAS/Cython/Rcpp "
            "internals that all in-process profilers miss. Multi-process aware "
            "(tracks forked workers via pgrep). macOS only.\n\n"
            "Outputs:\n"
            "  profile_history/<run-id>/profile.json  normalized hotspots\n"
            "  profile_history/<run-id>/<backend artifact> raw artifact (profile.out / Rprof.out / "
            "scalene.json / memray.bin / native_sample_<pid>.txt)\n"
            "  profile_history/<run-id>/run.log       pipeline stdout/stderr\n"
            "  --no-archive writes only profile_history/current/ scratch\n\n"
            "Examples:\n"
            "  zyme profile                       # default: backend=full\n"
            "  zyme profile --backend cpu         # zero-dep cprofile/Rprof\n"
            "  zyme profile --backend mem         # memray allocations\n"
            "  zyme profile --backend native      # macOS sample, see BLAS/Rcpp\n"
            "  zyme profile --json | jq .hotspots # machine-readable\n"
        ),
    )
    ppr.add_argument("hypothesis", nargs="?", default="",
                     help="Optional label for this profile run "
                          "(recorded in profile.json metadata).")
    ppr.add_argument("--backend", choices=["full", "cpu", "mem", "native"],
                     default="full",
                     help="full=Scalene/profvis (line-level CPU+mem+native split). "
                          "cpu=cProfile/Rprof (cheap, function-level). "
                          "mem=memray/Rprof+memory (allocation tracking). "
                          "native=macOS sample(1) external observer (sees BLAS/Rcpp internals). "
                          "Default: full.")
    ppr.add_argument("--dataset", dest="dataset", default=None,
                     help="Tier or dataset name (default = first listed in task.yaml).")
    ppr.add_argument("--no-archive", dest="no_archive", action="store_true",
                     help="Skip timestamped history; write overwritten "
                          "profile_history/current/ scratch only.")
    ppr.add_argument("--json", action="store_true",
                     help="Emit profile.json to stdout instead of evidence card to stderr "
                          "(pipe-friendly: `zyme profile --json | jq`).")
    ppr.add_argument("--diff", dest="diff_a", default=None, metavar="A",
                     help="Compare two profile.json snapshots instead of "
                          "running a new profile. A and B can be paths, or "
                          "shorthand: 'current' (= latest profile_history "
                          "profile), 'history:<substring>' (most-recent "
                          "matching entry in profile_history/). Use with "
                          "--diff-against B.")
    ppr.add_argument("--diff-against", dest="diff_b", default=None, metavar="B",
                     help="Second snapshot for --diff. Required when --diff is used.")
    ppr.set_defaults(func=cmd_profile)

    paa = sub.add_parser(
        "accept", parents=[task_dir_parent],
        help="Mark current attempt keep; advance best",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Accept with metric values in the description (recommended for keep rows)\n"
            "  zyme accept -m \"RANN::nn2 k=20 → +12% speed, knn_overlap=0.998\"\n\n"
            "  # Accept and silence the housekeeping reminder you've judged intentional\n"
            "  zyme accept -m \"sparse path validated\" --dismiss-housekeeping\n"
        ),
    )
    paa.add_argument("-m", "--description", default="",
                     help="Short description for results.tsv. Include speedup, mechanism, "
                          "and (for keeps) the metric values. ✓ \"RANN::nn2 k=20 → +12% speed, "
                          "knn_overlap=0.998\". ✗ \"sparse helped\".")
    paa.add_argument("--dismiss-housekeeping", dest="dismiss_housekeeping",
                     action="store_true",
                     help="Acknowledge most recent housekeeping reminder without "
                          "acting on it (you've judged the flagged findings as "
                          "intentional). Silences the next 15 rounds.")
    paa.set_defaults(func=cmd_accept)

    prj = sub.add_parser(
        "reject", parents=[task_dir_parent],
        help="Mark current attempt discard; HEAD hard-resets to best",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Standard reject with failure-mode description\n"
            "  zyme reject -m \"vectorized wilcoxon → top20=0.81 < 0.95, drops zero-variance genes\"\n\n"
            "  # Crash-then-fix: keep working-tree edits (the in-progress fix) instead of wiping them\n"
            "  zyme reject -m \"crashed at chunk 3; fixing index bounds\" --keep-tree\n"
        ),
    )
    prj.add_argument("-m", "--description", default="",
                     help="Short description for results.tsv. Include the failure mode and "
                          "metric values that fell short. ✓ \"vectorized wilcoxon → top20=0.81 "
                          "< 0.95, broadcast drops zero-variance genes\". ✗ \"didn't work\".")
    prj.add_argument("--keep-tree", dest="keep_tree", action="store_true",
                     help="Reset HEAD to best with --mixed (keep working tree changes) instead "
                          "of --hard. Designed for crash-then-fix: when the rejected round "
                          "crashed and you've edited the working tree to fix the bug, "
                          "--keep-tree preserves that fix so you don't have to retype it.")
    prj.add_argument("--force", action="store_true",
                     help="Skip reject's safety gates: (1) 'crash + uncommitted changes' "
                          "(wipes working tree even if it looks like an unsaved crash fix); "
                          "(2) 'cross-task commits in best..HEAD' (allows hard-reset past "
                          "sibling-task commits in a shared repo). Only use when you're "
                          "certain neither working-tree changes nor sibling tasks will be hurt.")
    prj.set_defaults(func=cmd_reject)

    prb_back = sub.add_parser(
        "rollback",
        parents=[task_dir_parent],
        help="Demote the most recent accepted (keep) commit; re-point best.ref to the prior keep. "
             "Does NOT rewrite git history — the rolled-back commit stays in the log, just flipped "
             "to status=rollback in results.tsv. Use when a previously-accepted round is later "
             "revealed to regress (at a larger tier, under different load, after a measurement bug). "
             "If host load (not code) is the suspect, rerun first — rollback is the wrong tool.",
    )
    prb_back.add_argument("-m", "--description", default="",
                          help="Optional rollback reason; appended to the demoted row's description.")
    prb_back.add_argument("--dry-run", dest="dry_run", action="store_true",
                          help="Report what would change (rolled-back round, new best.ref target) "
                               "without modifying results.tsv, best.ref, or HEAD. Re-run without "
                               "--dry-run to apply.")
    prb_back.set_defaults(func=cmd_rollback)

    # ---- iterate (auto-resume babysitter) --------------------------------
    piter = sub.add_parser(
        "iterate", parents=[task_dir_parent],
        help="Launch Claude with the iterate prompt and auto-resume when it "
             "exits before --max-rounds. Same resume logic as dispatch force_mode, "
             "without multi-task orchestration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run iterate with default 50 rounds\n"
            "  zyme iterate\n\n"
            "  # Custom round cap and prompt\n"
            "  zyme iterate --max-rounds 30 --prompt prompts/2.5_iterate_memory.md\n\n"
            "  # Use a specific model\n"
            "  zyme iterate --model claude-opus-4-7\n"
        ),
    )
    piter.add_argument("--max-rounds", dest="max_rounds", type=int, default=50,
                       help="Stop after this many completed decision rounds (default: 50).")
    piter.add_argument("--no-progress-limit", dest="no_progress_limit", type=int, default=3,
                       help="Give up after this many consecutive resumes with zero new rounds "
                            "(default: 3).")
    piter.add_argument("--prompt", default="prompts/2_iterate.md",
                       help="Iterate prompt file, relative to task dir "
                            "(default: prompts/2_iterate.md).")
    piter.add_argument("--model", default=None,
                       help="Model to pass to claude (e.g. claude-opus-4-7).")
    piter.add_argument("--effort", default=None,
                       help="Reasoning effort to pass to claude (e.g. max).")
    piter.add_argument("--stall-threshold", dest="stall_threshold", type=int,
                       default=900,
                       help="Seconds of silence before flagging a stall (default: 900).")
    piter.set_defaults(func=cmd_iterate)

    # ---- baseline family -------------------------------------------------
    # Group everything that touches the upstream-reference baseline table:
    # write paths (record / reference / noise / promote / rebench) and read
    # paths (list / show). All operate inside one task; all inherit --task-dir.
    pbase = sub.add_parser(
        "baseline",
        help="Baseline management. Primary command: `baseline reference` — runs the "
             "upstream reference and auto-records its timing. Also: `baseline noise` "
             "(intrinsic-noise calibration), `baseline show` / `baseline list` (read), "
             "`baseline record` (low-level manual entry, prefer `reference` instead).",
    )
    pbase_sub = pbase.add_subparsers(dest="baseline_cmd", required=True)

    pb_ref = pbase_sub.add_parser(
        "reference",
        parents=[task_dir_parent],
        help="PRIMARY: run reference.{py,R} for one tier with the right env vars and "
             "auto-record the baseline. One-step replacement for the manual "
             "`ZYME_DATA_PATH=... python reference.py` + `baseline record ...` pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run reference at one tier, auto-record speed_sec / peak_mb to results.tsv\n"
            "  zyme baseline reference --tier tiny\n\n"
            "  # Threaded baseline — sets ZYME_THREADS=8 for the subprocess\n"
            "  zyme baseline reference --tier medium --thread 8\n\n"
            "  # Re-record after a host upgrade (skip the >2× sanity gate)\n"
            "  zyme baseline reference --tier large --force\n"
        ),
    )
    pb_ref.add_argument("--thread", type=int, default=None,
                        help="Thread count to run reference.{R,py} at. Default reads "
                             "task.yaml::baseline_threads[0]; falls back to 1 only when "
                             "that field is absent. Sets ZYME_THREADS=<N> in the subprocess "
                             "env; the reference script reads it and branches into a "
                             "parallel path when N>1.")
    pb_ref.add_argument("--tier", required=True,
                        help="Tier label (or dataset name) from task.yaml.")
    pb_ref.add_argument("--reps", type=int, default=1,
                        help="Number of times to run reference.{py,R} for wall-time noise "
                             "calibration. Default 1 (single measurement). Pass --reps 5 "
                             "on `tiny` and --reps 3 on `medium` during init to populate "
                             ".zyme/baseline_noise.json — the agent reads this CV after each "
                             "`zyme run` to judge whether a delta is decisive or noise. "
                             "Mean is recorded as the baseline; per-rep stdev/CV go into "
                             "the noise file.")
    pb_ref.add_argument("--force", action="store_true",
                        help="Skip the >2× sanity gate when a prior baseline exists and the new value differs significantly.")
    pb_ref.add_argument("--accept-synthesis", dest="accept_synthesis", action="store_true",
                        help="Bypass the reference.{py,R} fingerprint check when "
                             "`task.yaml::synthesis: <reason>` is declared. See "
                             "`zyme baseline record --help` for details.")
    pb_ref.set_defaults(func=cmd_reference)

    pb_rec = pbase_sub.add_parser(
        "record",
        parents=[task_dir_parent],
        help="LOW-LEVEL: record a manually-measured upstream timing for one (tier, "
             "thread). PREFER `zyme baseline reference --tier <TIER>` — it runs the "
             "reference script and records the timing in one shot, with no copy-paste. "
             "Use `record` only when you genuinely cannot run reference.{py,R} from "
             "this CLI (e.g. you measured on another host) or when marking --oom.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "PREFER `zyme baseline reference` for the common case. Examples below are\n"
            "for the residual cases where reference.{py,R} ran outside this CLI.\n\n"
            "Examples:\n"
            "  # Manual entry from a reference run you already captured elsewhere\n"
            "  zyme baseline record --tier tiny --speed-sec 12.3 --peak-mb 850\n\n"
            "  # Auto-fill speed_sec / peak_mb from a saved reference log\n"
            "  zyme baseline record --tier medium --from-log /tmp/ref_medium.log\n\n"
            "  # Mark a tier as OOM (upstream reference itself didn't fit on this host)\n"
            "  zyme baseline record --tier ood_xlarge --oom\n\n"
            "  # Multi-thread baseline (record once per thread point you'll evaluate at)\n"
            "  zyme baseline record --tier medium --thread 8 --speed-sec 4.7\n"
        ),
    )
    pb_rec.add_argument("--thread", type=int, default=None,
                        help="Thread count this baseline was measured at. Default reads "
                             "task.yaml::baseline_threads[0]; falls back to 1 only when "
                             "that field is absent. Sanity gate keys on (dataset, thread): "
                             "different speeds at different thread counts are expected and "
                             "don't trigger the 2× check.")
    pb_rec.add_argument("--tier", default=None,
                        help="Tier label (e.g. tiny, medium, large) or dataset name; must exist in task.yaml. "
                             "Required unless --oom is passed.")
    pb_rec.add_argument("--name", default=None,
                        help="(optional) Dataset name for cross-check; rejected if it disagrees with task.yaml.")
    pb_rec.add_argument("--speed-sec", dest="speed_sec", type=float, default=None,
                        help="Measured upstream wall time in seconds. Required unless --from-log / --oom is used.")
    pb_rec.add_argument("--peak-mb", dest="peak_mb", type=float, default=0.0,
                        help="(optional) Measured peak memory in MB; default 0.0 if not captured.")
    pb_rec.add_argument("--metrics", default="{}",
                        help="(optional) JSON string of concordance metrics. "
                             "When omitted (or '{}'), auto-fills identity values "
                             "from task.yaml metrics: comparators (gte→1.0, lte→0.0). "
                             "Pass an explicit JSON object to override.")
    pb_rec.add_argument("--from-log", dest="from_log", default=None, metavar="PATH",
                        help="Read speed_sec / peak_mb from a captured stdout log file (e.g. saved "
                             "from a manual `python reference.py` run). Mutually exclusive with explicit "
                             "--speed-sec; auto-fills both fields from the log's `speed_sec:` / `peak_mb:` lines.")
    pb_rec.add_argument("--force", action="store_true",
                        help="Skip the >2× sanity gate that compares the new value against the prior baseline. "
                             "Use only when the change is intentional (genuine dataset swap, host upgrade, etc.).")
    pb_rec.add_argument("--oom", action="store_true",
                        help="Record this tier's baseline as OOM (upstream reference itself "
                             "did not fit on this host — exit 137 / 144 from the OS or zyme's "
                             "memory watchdog). Skips the speed/peak parse: writes a row with "
                             "speed_sec=0, peak_mb=0, status=oom. `zyme verify` then auto-marks "
                             "every cell of that tier as OOM in verify.tsv (no subprocess run), "
                             "and the figure renders the tier as a hatched OOM block in panels "
                             "A/B/C/D/F. --speed-sec / --peak-mb are ignored when --oom is set.")
    pb_rec.add_argument("--accept-synthesis", dest="accept_synthesis", action="store_true",
                        help="Bypass the reference.{py,R} fingerprint check when "
                             "`task.yaml::synthesis: <reason>` is declared. Use only when "
                             "the input synthesis is genuinely intrinsic to the workload "
                             "(PDE solver, generated signals, license-blocked data). "
                             "The fingerprint check otherwise rejects `[x]*N`, `np.tile`, "
                             "and repeat-loops in reference.py — see 1_init.md step 2.")
    pb_rec.set_defaults(func=cmd_record_baseline)

    pb_noise = pbase_sub.add_parser(
        "noise",
        parents=[task_dir_parent],
        help="Calibrate intrinsic noise for one tier (stochastic algorithms only). "
             "Runs reference.{py,R} with one or more noise-calibration seeds, "
             "compares each against the primary reference output via evaluate.{py,R}, "
             "and writes worst-seed per-metric noise values to "
             "`task.yaml::intrinsic_noise[tier]`. "
             "Concordance gates downstream then use noise-relative thresholds: "
             "`max(absolute_floor, multiplier × intrinsic_noise[tier])`.",
    )
    pb_noise.add_argument("--tier", required=True,
                          help="Tier label (or dataset name) from task.yaml.")
    pb_noise.add_argument("--thread", type=int, default=None,
                          help="Thread count to run the calibration reference at. Default "
                               "reads task.yaml::baseline_threads[0]; falls back to 1 only "
                               "when that field is absent. Sets ZYME_THREADS=<N> in the "
                               "subprocess env.")
    pb_noise.add_argument("--seeds", default=None,
                          help="Comma-separated calibration seeds, e.g. 43,44,45. "
                               "Default reads task.yaml::random_seeds.noise_calibration; "
                               "falls back to 43,44,45. Including the primary seed (42) "
                               "in this list reuses the existing primary reference output "
                               "for that seed (no extra reference run, drift = 0) — useful "
                               "for trading one calibration run against an extra free data "
                               "point (e.g. --seeds 42,43,44 runs only 2 reference scripts).")
    pb_noise.set_defaults(func=cmd_record_noise)

    pb_prom = pbase_sub.add_parser(
        "promote",
        parents=[task_dir_parent],
        help="One-shot drain of any leftover .zyme/baselines_stash.tsv entries "
             "into results.tsv. Baselines are now written directly; this command "
             "exists only to absorb the residual stash file when migrating an old "
             "task. Idempotent and safe to run with no stash.",
    )
    pb_prom.set_defaults(func=cmd_promote_baseline)

    pb_reb = pbase_sub.add_parser(
        "rebench",
        parents=[task_dir_parent],
        help="Audit / fairness retrofit: re-time reference.{R,py} at each "
             "(tier, thread) and update verify.tsv's baseline_speed + speedup_pct "
             "columns in place. DOES NOT re-run pipeline — only the divisor changes. "
             "Use after editing reference.{R,py} to read ZYME_THREADS.",
    )
    pb_reb.add_argument("--threads", default=None,
                        help="Comma-separated thread counts to bench (e.g. 1,4,8). "
                             "Default: read task.yaml::baseline_threads (or [1] if absent).")
    pb_reb.add_argument("--tiers", default=None,
                        help="Comma-separated tiers, or 'all' (default). "
                             "Limits which tiers to re-bench.")
    pb_reb.add_argument("--reps", type=int, default=1,
                        help="Repetitions per (tier, thread) cell; reports the median. Default: 1.")
    pb_reb.add_argument("--replicated", action="store_true",
                        help="Outcome-B fairness retrofit shortcut: run reference exactly ONCE "
                             "per tier (at thread=1) and replicate the speed_sec/peak_mb to "
                             "every requested thread. Use when upstream is known-serial (no "
                             "parallelism knob — outcome B in `M_thread_baseline_fairness.md`); "
                             "saves wall-time vs re-running the same serial reference at every "
                             "thread, which would otherwise just measure cold-cache noise. "
                             "Replicated rows are description-tagged so downstream readers can "
                             "tell them apart from real per-thread measurements.")
    pb_reb.add_argument("--verify-tsv", dest="verify_tsv", default=None,
                        help="Path to verify.tsv to update (default: <task_dir>/verify.tsv). "
                             "If absent, only results.tsv baselines are recorded.")
    pb_reb.add_argument("--no-plot", dest="no_plot", action="store_true",
                        help="Skip re-rendering the verify figure.")
    pb_reb.set_defaults(func=cmd_baseline_rebench)

    pb_list = pbase_sub.add_parser(
        "list",
        parents=[task_dir_parent],
        help="Show the current baseline rows in results.tsv (the truth `zyme verify` "
             "compares against). Pass --history to see the append-only audit log at "
             ".zyme/baselines_history.tsv instead.",
    )
    pb_list.add_argument("--history", action="store_true",
                         help="Show the append-only audit history at "
                              ".zyme/baselines_history.tsv instead of the current "
                              "results.tsv baseline rows. Useful when investigating "
                              "how a baseline value changed across re-records.")
    pb_list.set_defaults(func=cmd_baseline_list)

    pb_show = pbase_sub.add_parser(
        "show",
        parents=[task_dir_parent],
        help="Print recorded baseline rows for one tier (across all thread points) from results.tsv.",
    )
    pb_show.add_argument("tier",
                         help="Tier label (or dataset name) to look up.")
    pb_show.set_defaults(func=cmd_baseline_show)

    pv = sub.add_parser(
        "verify",
        parents=[task_dir_parent],
        help="Run pipeline at thread × tier matrix; verify concordance + speedup hold "
             "(packaging-time threading robustness check, used by 4_package.md step 2). "
             "Verification, not iteration: writes verify.tsv, doesn't touch results.tsv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default matrix: threads 1,4,8 × all tiers in task.yaml, 1 rep each\n"
            "  zyme verify\n\n"
            "  # Sparse grid: only specific (thread, tier) combinations\n"
            "  zyme verify --cells 1:tiny,4:large,8:medium\n\n"
            "  # Final stability gate before packaging: 3 reps per cell\n"
            "  zyme verify --threads 1,4,8 --tiers medium,large --reps 3\n\n"
            "  # Top up an earlier 1-rep matrix to 3 reps without redoing rep 1\n"
            "  zyme verify --threads 1,4,8 --tiers medium,large --reps 3 --write-mode topup\n\n"
            "  # Append a new OOD tier alongside an existing dev-tier matrix\n"
            "  zyme verify --tiers ood_xlarge --threads 1,4,8 --reps 1 --write-mode append\n\n"
            "  # Re-render verify.png/.pdf from rows already in verify.tsv (no subprocess runs)\n"
            "  zyme verify --render-only\n"
        ),
    )
    pv.add_argument("--threads", default="1,4,8",
                    help="Comma-separated thread counts to test. Default: 1,4,8. "
                         "Combined with --tiers as a cartesian product (ignored if --cells is given).")
    pv.add_argument("--tiers", default=None,
                    help="Comma-separated tier names (e.g. tiny,medium). Default: all tiers in task.yaml. "
                         "Combined with --threads as a cartesian product (ignored if --cells is given).")
    pv.add_argument("--cells", default=None,
                    help="Sparse grid: comma-separated `thread:tier` pairs (e.g. `1:tiny,4:large,8:medium`). "
                         "Use when you want specific (thread, tier) combinations without the full cartesian "
                         "product. Overrides --threads / --tiers when given.")
    pv.add_argument("--reps", type=int, default=1,
                    help="Repetitions per cell. Default 1 (fast feedback during threading iteration). "
                         "Use --reps 3 as the final stability gate before packaging — speed_sec is reported "
                         "as median(min–max), and EVERY rep's concordance must pass thresholds (not just median).")
    pv.add_argument("--output", default="verify.tsv",
                    help="Output TSV filename (in task root). Default: verify.tsv.")
    pv.add_argument("--write-mode", dest="write_mode",
                    choices=["overwrite", "append", "topup"],
                    default="overwrite",
                    help="How to merge with existing verify.tsv. "
                         "`overwrite` (default): truncate and rewrite from scratch. "
                         "`append`: keep existing rows; run every requested cell fresh "
                         "(no row reuse). Use to add cells outside an existing matrix "
                         "(e.g. ood_xlarge after a dev-tier matrix). "
                         "`topup`: reuse rows at the current HEAD commit + phase; only "
                         "run reps not yet recorded for each (thread,tier) cell, topping "
                         "up to --reps. Use after a fast 1-rep matrix when you want the "
                         "same cells extended to 3 reps without redoing rep 1. Old-schema "
                         "verify.tsv (no commit column) is rejected — delete it first.")
    pv.add_argument("--no-plot", dest="no_plot", action="store_true",
                    help="Skip rendering the speedup matrix as PNG/PDF/SVG (default: render alongside the TSV).")
    pv.add_argument("--render-only", dest="render_only", action="store_true",
                    help="Skip the matrix execution entirely; just rebuild verify.png/.pdf/.svg from "
                         "rows already in verify.tsv at the current HEAD commit + --phase. Useful when "
                         "you want to refresh the figure after editing task.yaml thresholds, after a "
                         "framework plot-rendering update, or to re-render a verify.tsv produced on "
                         "another machine. Honors --tiers / --threads / --cells as a filter on which "
                         "rows are included; defaults to all rows at the matching commit + phase. "
                         "No subprocess runs, no probe, no watchdog — fast and safe to repeat.")
    pv.add_argument("--skip-probe", dest="skip_probe", action="store_true",
                    help="Skip the threading-wired probe that runs before the matrix. The probe "
                         "spends ~1 minute at the smallest tier checking that thread=1 vs thread=max "
                         "speeds differ ≥1.5×; if not, ZYME_THREADS isn't actually wired through "
                         "pipeline/run.{py,R} and the matrix would be a silent no-op. Use this flag "
                         "only when you're sure threading is wired (e.g., test environments). "
                         "(After a passing probe at a given commit, subsequent runs at the same "
                         "commit and thread set auto-skip via .zyme/verify_probe.cache — no flag needed.)")
    pv.add_argument("--probe-tier", dest="probe_tier", default=None,
                    help="Override the auto-selected probe tier (default: smallest tier in the matrix). "
                         "Useful when the matrix only contains a single large tier — point the probe at a "
                         "small dev tier so wiring validation doesn't take as long as the matrix itself. "
                         "Tier name must exist in task.yaml (does NOT need to be in --tiers/--cells).")
    pv.add_argument("--force-probe", dest="force_probe", action="store_true",
                    help="Re-run the probe even if .zyme/verify_probe.cache already records a pass for "
                         "this commit + thread set. Use after touching framework/helpers.{py,R} or "
                         "any other code that could regress threading wiring without changing the "
                         "task's HEAD commit.")
    pv.add_argument("--allow-not-applicable-threads",
                    dest="allow_not_applicable_threads",
                    action="store_true",
                    help="Bypass the default guard for task.yaml `threading: not_applicable`. "
                         "Without this flag, verify forces such tasks to thread=1 and skips "
                         "requested 4t/8t cells. Use only for diagnostics or when actively "
                         "reclassifying the task's threading policy; the command prints a warning.")
    pv.add_argument("--phase", choices=["optimize", "validate"], default="optimize",
                    help="Which phase this verify matrix belongs to. Stamped on every row in "
                         "verify.tsv as a `phase` column; --write-mode topup filters reused rows "
                         "by both commit AND phase so a Phase 3 (validate) matrix and a Phase 4 "
                         "(packaging, optimize) matrix at the same commit don't collide. "
                         "Default: optimize.")
    pv.add_argument("--ram-floor", dest="ram_floor", default="auto",
                    help="Minimum free RAM (GB) required before each cell starts. Default: auto = "
                         "tier's recorded peak_mb × 1.5 + 2 GB OS headroom (falls back to 6 GB "
                         "when no baseline peak_mb is recorded). If free RAM falls below the "
                         "floor, the matrix aborts with a clear message naming the cell. Pass a "
                         "number to override (e.g. --ram-floor 10), or 0 to disable. This is the "
                         "pre-flight gate; the watchdog (--mem-cap-gb) is the in-cell enforcer.")
    pv.add_argument("--mem-cap-gb", dest="mem_cap_gb", default="auto",
                    help="Per-subprocess memory cap (GB). The watchdog polls each cell's "
                         "process-group RSS every 2s and SIGKILLs the group if it exceeds the "
                         "cap — turning a host-killing OOM thrash into a clean per-cell crash "
                         "(status=crash, crash_msg=killed-by-mem-watchdog). Matrix continues to "
                         "the next cell. Default: auto = total system RAM × 0.7. Pass a number "
                         "to override (e.g. --mem-cap-gb 12), or 0 to disable (let the OS "
                         "handle OOM — risks Mac thrash; only safe on Linux with cgroups). "
                         "POSIX-only.")
    pv.add_argument("--cell-settle-s", dest="cell_settle_s", type=float, default=2.0,
                    help="Seconds to sleep between cells so the kernel can finalize page reclaim "
                         "after the previous cell exited. Cheap (default 2s × n_cells), prevents "
                         "macOS swap-file-still-being-written → next-cell-thrashes race. Set 0 "
                         "to disable.")
    pv.set_defaults(func=cmd_verify)

    pat = sub.add_parser(
        "attest",
        parents=[task_dir_parent],
        help="Produce the paper-headline speedup for a packaged patch by "
             "shelling out to `autozyme.verify_patch()` (Python) or "
             "`autozyme::verify_patch()` (R). Spawns fresh subprocesses at "
             "5 tiers × 2 reps (auto-escalates to 3 reps if reps disagree "
             ">20%), appends per-tier rows to package_verify.tsv. "
             "On success, merges publishable passing rows and crash/OOM "
             "sentinels into the bundled package speedups snapshot. "
             "Distinct from `zyme verify`: verify = threading robustness "
             "(Phase 3, verify.tsv); attest = final headline number "
             "(Phase 4, package_verify.tsv).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default: auto-infer patch name from task.yaml::target_function,\n"
            "  # 5 tiers × 2 reps (with auto-escalation), append to package_verify.tsv\n"
            "  zyme attest\n\n"
            "  # Explicit patch name (when target_function doesn't match)\n"
            "  zyme attest --name mgcv\n\n"
            "  # CI smoke: only tiny tier, 1 rep\n"
            "  zyme attest --tiers tiny --reps 1\n\n"
            "  # Higher-confidence final number: 3 reps fixed (no auto-escalation)\n"
            "  zyme attest --reps 3\n\n"
            "  # Batch: attest several tasks sequentially with a summary at the end\n"
            "  zyme attest test_decontx test_slingshot test_RCTD\n\n"
            "  # Print the underlying command without running\n"
            "  zyme attest --dry-run\n"
        ),
    )
    pat.add_argument("task_dirs", nargs="*",
                     help="Zero or more task directories to attest sequentially. "
                          "When empty, falls back to --task-dir / cwd (single task). "
                          "Batch mode (>=2 dirs) runs each in turn, continues on "
                          "per-task failure, and emits a final summary.")
    pat.add_argument("--name", default=None,
                     help="Registered patch name (key for autozyme.verify_patch). "
                          "Default: infer from task.yaml::target_function "
                          "(e.g. `mgcv::gam` -> `mgcv`). Only valid for a single task.")
    pat.add_argument("--tiers", default=None,
                     help="Comma-separated tier names to verify "
                          "(e.g. `tiny,large,ood_xlarge`). Default: the package's "
                          "5-tier default (tiny, medium, large, ood_large, ood_xlarge). "
                          "Mutually exclusive with --skip-tiers.")
    pat.add_argument("--skip-tiers", dest="skip_tiers", default=None,
                     help="Comma-separated tier names to EXCLUDE from the 5-tier "
                          "default (e.g. `--skip-tiers ood_xlarge` runs the four "
                          "smaller tiers). Convenient when running parallel batches "
                          "to avoid memory contention on the heaviest tier. "
                          "Mutually exclusive with --tiers.")
    pat.add_argument("--reps", type=int, default=2,
                     help="Reps per tier. Default 2 — when reps==2, an auto-escalation "
                          "adds 1 more if the two reps' speedup_x disagree by >20%%. "
                          "Pass --reps 3+ to disable auto-escalation (fixed sample size).")
    pat.add_argument("--threads", default=None,
                     help="Comma-separated thread counts to attest "
                          "(e.g. `1,4,full`). `full`/`max`/`all` map to "
                          "os.cpu_count(). Default: task.yaml::baseline_threads[0] "
                          "or existing ZYME_THREADS.")
    pat.add_argument("--allow-not-applicable-threads",
                     dest="allow_not_applicable_threads",
                     action="store_true",
                     help="Bypass the default guard for task.yaml `threading: not_applicable`. "
                          "Without this flag, attest forces such tasks to thread=1 and skips "
                          "requested 4t/8t rows. Use only for diagnostics or when actively "
                          "reclassifying the task's threading policy; the command prints a warning "
                          "and automatic publish will keep those rows.")
    pat.add_argument("--retry-oom", dest="retry_oom",
                     action="store_true",
                     help="Bypass cached same-machine OOM sentinels in "
                          "package_verify.tsv. By default, attest skips a tier "
                          "when the latest attempt for the same platform, CPU, "
                          "RAM, thread count, patch, and tier is recorded as OOM.")
    pat.add_argument("--lang", choices=["py", "R"], default=None,
                     help="Override language detection. Default: detect from "
                          "pipeline/run.{py,R}.")
    pat.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="Print the underlying autozyme.verify_patch invocation "
                          "without running it.")
    pat.add_argument("--rerun-baseline", dest="rerun_baseline",
                     action="store_true",
                     help="Force fresh baseline measurement at every tier, "
                          "ignoring any cached entries in "
                          ".zyme/baseline_noise.json or on-disk "
                          "reference_output_<tier>/ artifacts. Use when you "
                          "suspect environmental drift (CPU governor, BLAS "
                          "update, different ZYME_THREADS) or want a clean "
                          "adversarial number. Default: cache is consulted; "
                          "on hit, baseline runs at most 1 confirmation rep.")
    pat.add_argument("--baseline-confirm-sigma", dest="baseline_confirm_sigma",
                     type=float, default=3.0,
                     help="Confirmation-rep tolerance, in standard deviations "
                          "of the cached mean. After the single confirmation "
                          "baseline rep, if |measured-cached| > K*cached_stdev "
                          "(σ floored at 1%% of mean), the cache is treated as "
                          "invalid and a fresh full baseline runs. "
                          "Default 3.0. Ignored under --rerun-baseline.")
    pat.add_argument("--no-baseline-confirm", dest="no_baseline_confirm",
                     action="store_true",
                     help="Skip the single confirmation baseline rep when the "
                          "cache hits — trust the cached timing as-is and run "
                          "only the patched reps. Saves another ~baseline_sec "
                          "per tier at the cost of detecting NO environmental "
                          "drift. Ignored under --rerun-baseline.")
    pat.add_argument("--no-preflight", dest="no_preflight",
                     action="store_true",
                     help="Skip the automatic `zyme package preflight` gate "
                          "(lint + portability scan + smoke-parity) that "
                          "runs before measurement. Use for CI re-runs after "
                          "preflight has already passed once, or when you "
                          "deliberately want to attest a known-failing patch.")
    pat.add_argument("--patched-only", dest="patched_only",
                     action="store_true",
                     help="Measure ONLY the patched variant — no baseline run, "
                          "no concordance check. For cells whose baseline OOMs "
                          "on this machine but whose patch fits: records the "
                          "patched wall-time + peak RSS (speedup/pass left blank, "
                          "NA). Correctness is verified separately on a box that "
                          "can run the baseline.")
    pat.set_defaults(func=cmd_attest)

    pasw = sub.add_parser(
        "attest-sweep",
        help="Fill missing (patch, tier, threads) attest coverage across all "
             "packaged patches. Walks autozyme_py + autozyme_r, reads each "
             "task's package_verify.tsv to find Windows cells that haven't "
             "been measured yet at the requested (tier, threads) matrix, and "
             "runs `zyme attest` for each gap. Resumable: every combo "
             "re-checks coverage before launching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Plan only — print what's missing without running\n"
            "  zyme attest-sweep --plan\n\n"
            "  # Run defaults (5 tiers x threads {1,4,8}, 2 reps each)\n"
            "  zyme attest-sweep\n\n"
            "  # Restrict matrix\n"
            "  zyme attest-sweep --tiers tiny,medium --threads 1,8 --reps 1\n\n"
            "  # Only a few patches\n"
            "  zyme attest-sweep --only cell2location,cellchat\n\n"
            "  # Skip a flaky/expensive patch\n"
            "  zyme attest-sweep --skip xclim,scvelo\n"
        ),
    )
    pasw.add_argument("--plan", action="store_true",
                      help="Print the missing-combo plan and exit without running.")
    pasw.add_argument("--tiers", default=None,
                      help="Comma-separated tiers to target. "
                           "Default: tiny,medium,large,ood_large,ood_xlarge.")
    pasw.add_argument("--threads", default=None,
                      help="Comma-separated thread counts to target. Default: 1,4,8.")
    pasw.add_argument("--reps", type=int, default=2,
                      help="Reps per combo passed through to `zyme attest`. Default 2.")
    pasw.add_argument("--timeout", type=int, default=3600,
                      help="Per-combo timeout in seconds (0 disables). Default 3600.")
    pasw.add_argument("--limit", type=int, default=0,
                      help="Run only the first N planned combos (0 = no cap).")
    pasw.add_argument("--skip", default=None,
                      help="Comma-separated patch names to skip.")
    pasw.add_argument("--only", default=None,
                      help="Comma-separated patch names to restrict to.")
    pasw.set_defaults(func=cmd_attest_sweep)

    # -----------------------------------------------------------------
    # backfill — bring under-replicated speedup cells up to a target
    # -----------------------------------------------------------------
    pbf = sub.add_parser(
        "backfill",
        help="Read each patch's speedups_finalized.tsv, find cells with "
             "n_reps below target, and drive `zyme attest` + publish + "
             "re-finalize to fill them in. Tier-aware concurrency: "
             "small/medium run 2 at a time, large/OOD serial.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Plan only — see what would run\n"
            "  zyme backfill --plan\n\n"
            "  # Bring all win cells up to n=2 (default), 2-way for small\n"
            "  zyme backfill\n\n"
            "  # Plus stabilize n=2 cells with >10% variance up to n=3\n"
            "  zyme backfill --also-stabilize\n\n"
            "  # Just one patch, just tiny tier — for smoke testing\n"
            "  zyme backfill --only sccoda --tiers tiny --limit 2\n"
        ),
    )
    pbf.add_argument("--plan", action="store_true",
                     help="Print the planned work and exit without running.")
    pbf.add_argument("--target-reps", dest="target_reps", type=int, default=2,
                     help="Per-variant rep target. Default 2.")
    pbf.add_argument("--also-stabilize", dest="also_stabilize",
                     action="store_true",
                     help="Also target unstable n=2 cells (2-rep diff > "
                          "--variance-pct) for a third rep (n=3).")
    pbf.add_argument("--variance-pct", dest="variance_pct", type=float,
                     default=10.0,
                     help="With --also-stabilize: percent diff threshold. "
                          "Default 10.")
    pbf.add_argument("--platform", choices=("win", "mac"), default="win",
                     help="Which platform's cells to fill. Default win "
                          "(the runner only fills cells matching the "
                          "current platform — running on Mac will not "
                          "fill Windows cells).")
    pbf.add_argument("--tiers", default=None,
                     help="Comma-separated tiers to consider. Default: all.")
    pbf.add_argument("--threads", default=None,
                     help="Comma-separated thread counts to consider. "
                          "Default: all.")
    pbf.add_argument("--skip", default=None,
                     help="Comma-separated patch names to skip (in addition "
                          "to the built-in default skip set).")
    pbf.add_argument("--only", default=None,
                     help="Comma-separated patch names to restrict to.")
    pbf.add_argument("--small-workers", dest="small_workers", type=int,
                     default=2,
                     help="Concurrency for small/medium tier work. Default 2.")
    pbf.add_argument("--large-workers", dest="large_workers", type=int,
                     default=1,
                     help="Concurrency for large/OOD tier work. Default 1.")
    pbf.add_argument("--timeout", type=int, default=3600,
                     help="Per-cell timeout (sec). 0 disables. Default 3600.")
    pbf.add_argument("--limit", type=int, default=0,
                     help="Run only the first N planned items (0 = no cap).")
    pbf.add_argument("--framework-root", dest="framework_root", default=None,
                     help="Override autodetect.")
    pbf.add_argument("--plain", action="store_true",
                     help="Force plain stdout progress (skip rich dashboard).")
    pbf.set_defaults(func=cmd_backfill)

    # -----------------------------------------------------------------
    # validate — LLM-based adversarial audit (phase-end, not per round)
    # -----------------------------------------------------------------
    pvalp = sub.add_parser(
        "validate",
        help="LLM-based adversarial audit for hack detection. Run ONCE at "
             "phase end (init or iterate), not per round. Spawns a headless "
             "agent (claude/cursor/codex) against prompts/validate_{init,iterate}.md, "
             "writes a markdown report to audit_reports/ and per-finding rows "
             "to validate.tsv. Complements deterministic checks "
             "(`zyme audit`, hoist_audit, argdiff) with semantic cross-evidence "
             "review that static lints cannot perform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Audit a task right after iterate converges\n"
            "  cd test_core_singlecell/test_decontx && zyme validate iterate\n\n"
            "  # Audit init setup before iterating (catches dataset/metric design issues)\n"
            "  zyme validate init --task-dir test_core_singlecell/test_decontx\n\n"
            "  # Different backend\n"
            "  zyme validate iterate --agent cursor\n"
            "  zyme validate iterate --agent codex --model gpt-5-codex\n\n"
            "  # Override report output path (default: audit_reports/validate_<phase>_<ts>.md)\n"
            "  zyme validate iterate --out /tmp/decontx_audit.md\n"
        ),
    )
    pvalsub = pvalp.add_subparsers(dest="validate_subcmd", required=True,
                                   metavar="{init,iterate}")

    def _add_validate_common(p):
        p.add_argument("--task-dir", dest="task_dir", default=None,
                       help="Task directory (default: cwd).")
        p.add_argument("--agent", choices=["auto", "claude", "cursor", "codex"], default="auto",
                       help="LLM agent backend. Default: auto (detect first available).")
        p.add_argument("--model", default=None,
                       help="Override agent default model "
                            "(e.g. claude-opus-4-7, composer-2.5, gpt-5-codex). "
                            "Default: agent's own default.")
        p.add_argument("--effort", default="max",
                       help="Reasoning effort: low|medium|high|max. "
                            "Mapped per backend (codex: low/medium/high; claude/cursor: passthrough). "
                            "Default: high.")
        p.add_argument("--out", default=None,
                       help="Override markdown report output path. "
                            "Default: <framework>/validation/<task_name>/v<N>/validate_<phase>_<UTC-ts>.md.")
        p.add_argument("--version", default=None,
                       help="Validation round subfolder (e.g. v2 or 2). Bump this "
                            "when the validator prompt changes. "
                            "Default: highest existing v<N> in validation/<task>/, or v1.")

    pvi = pvalsub.add_parser(
        "init",
        help="Audit init-phase setup: dataset realism, metric coverage, "
             "threshold sanity, target-function scope. Run after "
             "`zyme baseline reference` but before iterating, to catch "
             "setup choices that would invite hacking downstream.",
    )
    _add_validate_common(pvi)
    pvi.add_argument(
        "--register-template", dest="register_template", default=None, metavar="NAME",
        help="After a successful audit, register this task as an iterate-stage bench "
             "template named NAME. Equivalent to `zyme bench register-template --as NAME "
             "--stage iterate --at post_init` run after validate completes. Data dirs are "
             "symlinked (not copied), so the template stays small.",
    )
    pvi.add_argument(
        "--register-force", dest="register_force", action="store_true",
        help="Pass --force to the template registration step (overwrite if NAME already exists).",
    )
    pvi.set_defaults(func=cmd_validate_init)

    pviter = pvalsub.add_parser(
        "iterate",
        help="Audit iterate-phase pipeline behavior: timer-window hoist, "
             "IO function hijack, gate-threshold relaxation, output truncation, "
             "narrative-vs-code consistency, seed pinning. Run after "
             "iterate converges (honest stop / budget out).",
    )
    _add_validate_common(pviter)
    pviter.set_defaults(func=cmd_validate_iterate)

    pps = sub.add_parser(
        "publish-speedups",
        help="Merge each task's package_verify.tsv into its matching patch's "
             "raw bundled snapshot slot (autozyme_r/inst/patches/<name>/speedups.tsv or "
             "autozyme_py/src/autozyme/<name>/speedups.tsv). Release packages ship "
             "speedups_finalized.tsv only; end users of the "
             "installed package access these via `autozyme::speedups(name)` "
             "(R) / `autozyme.speedups(name)` (Py). `zyme attest` auto-merges "
             "passing rows; this command is the repair/backfill/manual refresh "
             "path. The task-side TSV stays the live source of truth. "
             "Default --write-mode is "
             "`merge`: rows from other hosts (e.g. Windows attest data) are "
             "preserved across publishes from this host.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Refresh bundled speedups for every patch (merge with existing)\n"
            "  zyme publish-speedups\n\n"
            "  # One task only (task dir name or patch stem)\n"
            "  zyme publish-speedups test_decontx\n"
            "  zyme publish-speedups decontx\n\n"
            "  # Paper snapshot: latest row per tier, all 5 tiers must pass\n"
            "  zyme publish-speedups --select latest-per-tier "
             "--require-all-tiers --require-all-pass\n\n"
            "  # Latest attest batch only (rows sharing max timestamp)\n"
            "  zyme publish-speedups test_decontx --select latest-run\n\n"
            "  # Force legacy overwrite (wipes other hosts' rows!)\n"
            "  zyme publish-speedups --write-mode overwrite\n\n"
            "  # Last 5 data rows only\n"
            "  zyme publish-speedups test_decontx --select tail --tail 5\n\n"
            "  # See what would change without writing\n"
            "  zyme publish-speedups --dry-run\n"
        ),
    )
    pps.add_argument(
        "tasks", nargs="*",
        help="Optional task dir names or patch stems to publish. "
             "When omitted, every patch in the lifted-from index is considered.",
    )
    pps.add_argument(
        "--select",
        choices=("full", "latest-per-tier", "latest-run", "tail"),
        default="full",
        help="Which rows to copy from package_verify.tsv. "
             "full=entire file (default); latest-per-tier=one row per "
             "(platform,tier), latest timestamp wins (matches "
             "autozyme.speedups()); latest-run=rows from the newest "
             "timestamp batch; tail=last N rows (--tail, default 5).",
    )
    pps.add_argument(
        "--tail", type=int, default=None,
        help="With --select tail: number of trailing data rows to keep "
             "(default 5).",
    )
    pps.add_argument(
        "--tiers", default=None,
        help="Comma-separated tier subset to include (e.g. small,medium,large). "
             "Also defines which tiers --require-all-tiers checks.",
    )
    pps.add_argument(
        "--platform", choices=("win", "mac", "unknown"), default=None,
        help="Keep only rows from this platform (system_os bucket).",
    )
    pps.add_argument(
        "--all-pass-only", dest="all_pass_only", action="store_true",
        help="Drop rows where all_pass is not true before selecting.",
    )
    pps.add_argument(
        "--require-all-tiers", dest="require_all_tiers", action="store_true",
        help="Skip a task when the selected snapshot lacks a valid row for "
             "every default tier (or every tier in --tiers).",
    )
    pps.add_argument(
        "--require-all-pass", dest="require_all_pass", action="store_true",
        help="Skip a task when any selected row has all_pass != true.",
    )
    pps.add_argument(
        "--write-mode", dest="write_mode",
        choices=("merge", "overwrite", "append"),
        default="merge",
        help="How to combine selected rows with the bundled TSV. "
             "`merge` (default): row-level union keyed by (tier, system_os, "
             "system_cpu, system_threads, framework_version); newer "
             "timestamp wins per key — preserves other hosts' rows so "
             "publishing from Mac no longer wipes Windows attest data. "
             "`overwrite`: legacy behavior, truncate and rewrite from the "
             "selected rows only. `append`: keep existing rows, append "
             "selected rows below (no dedup; can produce duplicates).",
    )
    pps.add_argument(
        "--allow-not-applicable-threads",
        dest="allow_not_applicable_threads",
        action="store_true",
        help="Bypass the default guard for task.yaml `threading: not_applicable`. "
             "Without this flag, publish-speedups publishes only thread=1 rows "
             "and prunes stale 4t/8t rows from the bundled snapshot. Use only "
             "for diagnostics or task threading-policy reclassification; the "
             "command prints a warning.",
    )
    pps.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="List what would be copied without writing.")
    pps.set_defaults(func=cmd_publish_speedups)

    ps = sub.add_parser(
        "status",
        parents=[task_dir_parent],
        help="One-screen snapshot of task state: HEAD/best, decision rounds remaining, "
             "per-tier baseline/best/speedup/CV, patch stack, and last N decisions. "
             "Read-only; safe to run between rounds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default optimize-phase narrative (Phase 3 / memory rounds appear as callouts)\n"
            "  zyme status\n\n"
            "  # Phase 3 fix-loop view (validate cap = 30 rounds)\n"
            "  zyme status --phase validate\n\n"
            "  # Memory-loop view (memory cap = 50 rounds)\n"
            "  zyme status --phase memory\n"
        ),
    )
    ps.add_argument("--last", type=int, default=5,
                    help="Show this many most-recent decision rows (default 5).")
    ps.add_argument("--phase", choices=["optimize", "validate", "memory", "all"],
                    default="optimize",
                    help="Filter narrative to one phase. `optimize` (default) shows the "
                         "Phase 2 optimization history; a one-line callout flags any "
                         "Phase 3 activity. `validate` shows the Phase 3 scale-fix loop. "
                         "`memory` shows the memory-optimization loop (matches `run "
                         "--phase memory` rows). `all` shows everything.")
    ps.set_defaults(func=cmd_status)

    prep = sub.add_parser(
        "report",
        parents=[task_dir_parent],
        help="Render a self-contained HTML optimization report from "
             "results.tsv + memory/. Includes interactive charts, patch stack, "
             "dead ends, discoveries, and (if present) agent-curated narrative "
             "from memory/report_narrative.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Render to ./report.html\n"
            "  zyme report\n\n"
            "  # Render and open in default browser\n"
            "  zyme report --open\n\n"
            "  # Custom output path\n"
            "  zyme report -o /tmp/my_report.html\n\n"
            "  # Use a narrative file from a non-default path\n"
            "  zyme report --narrative path/to/narrative.md\n"
        ),
    )
    prep.add_argument("-o", "--output", default=None,
                      help="Output HTML path (default: <task>/report.html).")
    prep.add_argument("--narrative", default=None,
                      help="Path to narrative md (default: <task>/memory/report_narrative.md). "
                           "When absent, mechanical defaults are used for all narrative slots.")
    prep.add_argument("--open", action="store_true",
                      help="Open the rendered report in the default browser.")
    prep.set_defaults(func=cmd_report)

    pp = sub.add_parser(
        "plot",
        parents=[task_dir_parent],
        help="Render a convergence curve from results.tsv "
             "(accepted points connected; rejected as faint X markers).",
    )
    pp.add_argument("--dataset", default=None,
                    help="Dataset name to plot. Default: render one figure per dataset present.")
    pp.add_argument("--output-dir", dest="output_dir", default=None,
                    help="Directory to write convergence_<task>_<dataset>.{png,pdf,svg}. Default: <task_dir>/figure/.")
    pp.add_argument("--log", choices=["auto", "on", "off"], default="auto",
                    help="Y-axis log scale. 'auto' = log if max/min runtime > 20.")
    pp.add_argument("--title", default=None,
                    help="Override figure title (default: '<task> — <dataset>').")
    pp.add_argument("--phase", choices=["optimize", "validate", "memory", "all"],
                    default="optimize",
                    help="Filter rows by phase. Baselines always appear regardless. "
                         "Default `optimize` keeps Phase 3 scale-fix and memory-loop rows "
                         "out of the convergence curve. `memory` plots the memory-"
                         "optimization loop separately.")
    pp.set_defaults(func=cmd_plot)

    psc = sub.add_parser(
        "scan",
        help="Workspace-wide dashboard: for every task.yaml under PATHS, "
             "report which lifecycle phase (scaffold/init/iterate/scaling/package) "
             "is done plus per-phase reflection status. Read-only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default workspace scan; writes <framework>/SCAN.md alongside the table\n"
            "  zyme scan\n\n"
            "  # Show only tasks stuck at the iterate phase\n"
            "  zyme scan --filter iterate\n\n"
            "  # Suppress reflect columns (just lifecycle phases)\n"
            "  zyme scan --phase-only\n\n"
            "  # Stream NDJSON (one task per line) for agent parsing\n"
            "  zyme scan --json | jq 'select(.phase==\"package\")'\n\n"
            "  # Dataset audit: per-task local/external/missing + shared-path roster.\n"
            "  # Writes <framework>/DATASETS.md. Run before moving a task to spot\n"
            "  # files that won't travel with the task directory.\n"
            "  zyme scan --dataset\n\n"
            "  # Attest coverage: which patches have speedups_finalized.tsv, which tiers\n"
            "  # are filled. Pure package-side scan; doesn't need the workspace.\n"
            "  zyme scan --attest\n\n"
            "  # Speedup coverage lint: per-patch finalized TSV check (status, n_reps,\n"
            "  # rep variance on sec/mem, baseline-era drift, baseline/patched pairing,\n"
            "  # baseline/patched symmetry, platform symmetry, thread coverage). Writes\n"
            "  # <framework>/COVERAGE.md.\n"
            "  zyme scan --coverage\n"
            "  zyme scan --coverage --detail            # per-issue listing\n"
            "  zyme scan --coverage --strict            # exit 1 if any FAIL/WARN\n\n"
            "  # Portability hazard scan after scaling; writes each task's\n"
            "  # .zyme/portability_scan.json and prints skip/run-3.5 verdict.\n"
            "  zyme scan --portability path/to/task\n"
            "  zyme scan --portability --needs-3-5\n"
        ),
    )
    psc.add_argument("paths", nargs="*",
                     help="Dirs to scan. Default: autodetect workspace root and "
                          "scan every sibling dir of autozyme-framework/ that "
                          "contains tasks.")
    psc.add_argument("--max-depth", type=int, default=3,
                     help="Max recursion depth searching for task.yaml. Default 3 "
                          "covers <workspace>/<category>/<task>/task.yaml.")
    psc.add_argument("--filter", dest="phase_filter",
                     choices=["scaffold", "init", "iterate", "scaling", "package", "unknown"],
                     default=None,
                     help="Show only tasks whose furthest-completed phase matches.")
    psc.add_argument("--phase-only", action="store_true",
                     help="Suppress reflect columns.")
    psc.add_argument("--reflect-only", action="store_true",
                     help="Suppress phase columns; show reflect grid only.")
    psc.add_argument("--json", action="store_true",
                     help="Emit one JSON object per task (line-delimited) instead of a table.")
    psc.add_argument("--active-minutes", type=float, default=15,
                     help="Recent file-activity window for scan's active column "
                          "(default 15). Live dispatch state is always reported; "
                          "use 0 to disable the recent-activity fallback.")
    psc.add_argument("--framework-root", dest="framework_root", default=None,
                     help="Override autodetected framework root (used to resolve "
                          "package patches and reflection feedback files).")
    psc.add_argument("--export", dest="export_path", default=None,
                     help="Write a markdown summary to this path. Default: "
                          "<framework>/SCAN.md. The exported file always "
                          "reflects the full unfiltered set, regardless of "
                          "--filter. Skipped when --json or --no-export is set.")
    psc.add_argument("--no-export", dest="no_export", action="store_true",
                     help="Skip writing the markdown summary.")
    psc.add_argument("--dataset", action="store_true",
                     help="Dataset-focused scan: for each task report its tier "
                          "datasets, mark whether each path is local to the task "
                          "dir (moves with the task), external (won't move), or "
                          "missing on disk. Skips the lifecycle table and writes "
                          "<framework>/DATASETS.md instead of SCAN.md. Useful "
                          "before relocating tasks or auditing shared-data "
                          "storage in /datasets/.")
    psc.add_argument("--attest", action="store_true",
                     help="Attest-coverage audit: enumerate every patch in "
                          "autozyme_r/inst/patches/ and autozyme_py/src/autozyme/, "
                          "and for each report whether speedups_finalized.tsv is present and "
                          "which workload tiers (small/medium/large/ood_large/"
                          "ood_xlarge) have valid attest data. Pure package-side "
                          "scan — does NOT walk the workspace or task dirs. "
                          "Writes <framework>/ATTEST.md instead of SCAN.md.")
    psc.add_argument("--coverage", action="store_true",
                     help="Speedup coverage lint: read every "
                          "inst/patches/<patch>/speedups_finalized.tsv (and Py "
                          "equivalent), check each patch's rows against 5 "
                          "inside-out rules (valid status, n_reps>=3 for "
                          "patched / >=2 for baseline, baseline/patched "
                          "symmetry, cross-platform symmetry, thread coverage). "
                          "No external metadata — each patch's own TSV is the "
                          "source of truth. Default exit 0 (report-only); "
                          "writes <framework>/COVERAGE.md.")
    psc.add_argument("--detail", action="store_true",
                     help="With --coverage: list every issue per patch under "
                          "the summary table (default is summary counts only).")
    psc.add_argument("--strict", action="store_true",
                     help="With --coverage: exit 1 if any WARN or FAIL is "
                          "reported. Use as a release gate.")
    psc.add_argument("--patched-min-reps", type=int, default=2,
                     help="With --coverage: minimum n_reps for patched rows. "
                          "Default 2 — empirical audit shows patched kernels "
                          "are mostly deterministic; cells with genuinely "
                          "unstable n=2 are caught by --rep-variance-pct.")
    psc.add_argument("--baseline-min-reps", type=int, default=2,
                     help="With --coverage: minimum n_reps for baseline rows. "
                          "Default 2 (baseline represents the upstream divisor; "
                          "statistical power lives on the patched side).")
    psc.add_argument("--rep-variance-pct", type=float, default=10.0,
                     help="With --coverage: warn when an n=2 cell's two sec "
                          "reps differ by more than this percent (default 10). "
                          "Catches unstable cells where a third rep would "
                          "actually change the conclusion.")
    psc.add_argument("--rep-variance-abs-sec", type=float, default=2.0,
                     help="With --coverage: absolute floor (seconds) for "
                          "high_rep_variance — both percent AND this gap must be "
                          "exceeded (default 2.0, so a 12%% / 1s swing on an 8s "
                          "op is ignored — a 3rd rep wouldn't move the median).")
    psc.add_argument("--mem-variance-pct", type=float, default=20.0,
                     help="With --coverage: warn when an n=2 cell's two peak-RSS "
                          "reps differ by more than this percent (default 20 — "
                          "RSS is noisier than sec, so a higher bar than "
                          "--rep-variance-pct).")
    psc.add_argument("--mem-variance-abs-mb", type=float, default=1024.0,
                     help="With --coverage: absolute floor (MB) for "
                          "high_mem_rep_variance — both percent AND this absolute "
                          "gap must be exceeded (default 1024 = 1GB, so a 20%% "
                          "swing on a 0.5GB baseline is ignored as RSS noise).")
    psc.add_argument("--baseline-era-pct", type=float, default=20.0,
                     help="With --coverage: warn when raw baseline rows for one "
                          "cell span multiple dates and per-date peak medians "
                          "differ by more than this percent (default 20 — peak "
                          "RSS is noisy, same bar as --mem-variance-pct).")
    psc.add_argument("--baseline-era-abs-mb", type=float, default=1024.0,
                     help="With --coverage: with --baseline-era-pct, also "
                          "require at least this many MB spread between era "
                          "medians (default 1024 = 1GB, ignores small-RSS drift).")
    cov_plat = psc.add_mutually_exclusive_group()
    cov_plat.add_argument("--mac-only", action="store_true",
                          help="With --coverage: lint macOS rows only "
                               "(finalized platform=macOS + speedups.mac.tsv "
                               "era checks). Writes COVERAGE.mac.md.")
    cov_plat.add_argument("--win-only", action="store_true",
                          help="With --coverage: lint Windows rows only "
                               "(finalized platform=Windows + speedups.win.tsv "
                               "era checks). Writes COVERAGE.win.md.")
    psc.add_argument("--skip-experimental", dest="skip_experimental",
                     action="store_true",
                     help="With --coverage: skip fusion-experiment patches "
                          "(tradeseq/mast/infercnv/find_all_markers) and "
                          "orchestrate headline patches (scanpy_*/seurat_*), "
                          "which are measured separately, not via plain attest.")
    psc.add_argument("--portability", action="store_true",
                     help="Portability hazard scan: statically inspect each task's "
                          "pipeline/run.{R,py} (and packaged patch if present), "
                          "classify cross-platform debt, and write "
                          "<task>/.zyme/portability_scan.json. Use after scaling "
                          "to decide whether to dispatch 3.5_portability.")
    psc.add_argument("--needs-3-5", dest="needs_3_5", action="store_true",
                     help="With --portability: show only tasks whose verdict "
                          "requires the 3.5 portability phase.")
    psc.set_defaults(func=cmd_scan)

    preg = sub.add_parser(
        "registry",
        help="Searchable registry of optimized functions, hot dependency engines, and similar task opts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Rebuild from every task in the workspace; writes autozyme-framework/registry/\n"
            "  zyme registry rebuild\n\n"
            "  # Query known optimized functions or dependency bottlenecks\n"
            "  zyme registry query Seurat::NormalizeData\n"
            "  zyme registry query mgcv gam\n\n"
            "  # For a fresh task, surface reusable patches and similar active_opts.md files\n"
            "  zyme registry suggest --task task.yaml\n\n"
            "  # Add a profile to bias suggestions toward current hotspots\n"
            "  zyme registry suggest --task . --profile profile_history/current/profile.json\n"
        ),
    )
    preg_sub = preg.add_subparsers(dest="registry_cmd", required=True)

    preg_rebuild = preg_sub.add_parser(
        "rebuild",
        help="Rebuild registry/functions.tsv and per-function detail cards from task artifacts.",
    )
    preg_rebuild.add_argument("paths", nargs="*",
                              help="Roots to scan. Default: autodetected workspace root.")
    preg_rebuild.add_argument("--max-depth", type=int, default=4,
                              help="Max recursion depth searching for task.yaml. Default 4.")
    preg_rebuild.add_argument("--framework-root", default=None,
                              help="Framework root. Default: autodetect from cwd/workspace.")
    preg_rebuild.add_argument("--registry-root", default=None,
                              help="Output root. Default: <framework>/registry.")
    preg_rebuild.add_argument("--include-bench", action="store_true",
                              help="Include bench_runs/ task replicas. Default skips them to avoid duplicate entries.")
    preg_rebuild.set_defaults(func=cmd_registry_rebuild)

    preg_query = preg_sub.add_parser(
        "query",
        help="Search registry entries by function symbol, package, task name, or tag.",
    )
    preg_query.add_argument("terms", nargs="+",
                            help="Search terms, e.g. `mgcv gam` or `Seurat::NormalizeData`.")
    preg_query.add_argument("--registry-root", default=None,
                            help="Registry root. Default: <framework>/registry.")
    preg_query.add_argument("--limit", type=int, default=12,
                            help="Maximum matches to print. Default 12.")
    preg_query.set_defaults(func=cmd_registry_query)

    preg_suggest = preg_sub.add_parser(
        "suggest",
        help="Suggest reusable optimized functions, dependency bottlenecks, and similar active_opts for a task/profile.",
    )
    preg_suggest.add_argument("--task", default=None,
                              help="Task dir or task.yaml path to inspect.")
    preg_suggest.add_argument("--profile", default=None,
                              help="Optional profile.json path to bias suggestions toward current hotspots.")
    preg_suggest.add_argument("--registry-root", default=None,
                              help="Registry root. Default: <framework>/registry.")
    preg_suggest.add_argument("--limit", type=int, default=8,
                              help="Maximum entries per section. Default 8.")
    preg_suggest.set_defaults(func=cmd_registry_suggest)

    preg_list = preg_sub.add_parser(
        "list",
        help="List registry entries, optionally filtered by kind.",
    )
    preg_list.add_argument("--kind", choices=["target_function", "workflow_primitive", "numerical_engine"],
                           default=None,
                           help="Filter by registry kind.")
    preg_list.add_argument("--registry-root", default=None,
                           help="Registry root. Default: <framework>/registry.")
    preg_list.add_argument("--limit", type=int, default=80,
                           help="Maximum rows to print. Default 80.")
    preg_list.set_defaults(func=cmd_registry_list)

    pip = sub.add_parser(
        "inspect-parallelism",
        help="Scan an upstream repo for parallelism backends (mclapply, "
             "OpenMP, BLAS, mp.Pool, joblib, RcppParallel, numba, torch, "
             "CUDA, MPI, ...). Emits a structured inventory + a draft "
             "`parallelism_profile` YAML block for the init agent to "
             "review and paste into task.yaml. Read-only.",
    )
    pip.add_argument("repo",
                     help="Path to the upstream repo checkout (typically "
                          "<task_dir>/upstream_repo). Must be a directory.")
    pip.add_argument("--target", default=None,
                     help="Target function name (e.g. 'blockwiseModules'). "
                          "Filters USER-CONTROLLABLE KNOBS to functions "
                          "reachable from the target's call chain. "
                          "Auto-read from task.yaml if omitted.")
    pip.add_argument("--max-hits", type=int, default=5,
                     help="Max source-line hits to print per backend. "
                          "Higher values are useful when investigating where "
                          "a backend lives across many files. Default 5.")
    pip.set_defaults(func=cmd_inspect_parallelism)

    # ---- dispatch family --------------------------------------------------
    # Workspace-level: dispatch a prompt across multiple tasks. Run / status /
    # logs / stop are nested under one parent so `zyme dispatch --help` lists
    # the family without polluting the top-level command list.
    pdis = sub.add_parser(
        "dispatch",
        help="Multi-task dispatch: run / status / usage / logs / stop",
    )
    pdis_sub = pdis.add_subparsers(dest="dispatch_cmd", required=True)

    pdis_run = pdis_sub.add_parser(
        "run",
        help="Run a prompt across multiple tasks via an agent CLI, gating on RAM "
             "and disk floors. Sequential. Foreground unless --detach. "
             "Manager (caller) decides task list and prompt; CLI handles "
             "resource gates, daemon, stream-json parsing, stall detection, "
             "and ETA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Foreground dispatch over 3 tasks; Ctrl-C cleanly stops\n"
            "  zyme dispatch run test_findmarkers test_singler test_slingshot \\\n"
            "      --prompt prompts/2_iterate.md\n\n"
            "  # Detached daemon, custom RAM floor, lower-effort model\n"
            "  zyme dispatch run task_a task_b --prompt prompts/3_validate_scaling.md \\\n"
            "      --detach --ram-floor 12g --effort medium\n\n"
            "  # Smoke-run two completed decision rounds, then stop the agent\n"
            "  zyme dispatch run task_a --prompt prompts/2_iterate.md --agent codex --max-rounds 2\n\n"
            "  # Dry-run: print the launch plan without launching anything\n"
            "  zyme dispatch run task_a task_b --prompt prompts/2_iterate.md --dry-run\n\n"
            "Companion commands:\n"
            "  zyme dispatch status         # one-screen state of the active run\n"
            "  zyme dispatch usage          # token/cost telemetry from agent logs\n"
            "  zyme dispatch logs <task>    # parsed events for one worker\n"
            "  zyme dispatch stop           # SIGTERM master + workers\n"
        ),
    )
    pdis_run.add_argument("tasks", nargs="+",
                          help="Task names or task directory paths. Names are "
                               "resolved against --workspace; paths can be "
                               "relative or absolute.")
    pdis_run.add_argument("--prompt", required=True,
                          help="Path (relative to each task dir) of the prompt "
                               "file to dispatch. Sent to the agent as `read and "
                               "follow <prompt>`. Required.")
    pdis_run.add_argument("--workspace", default=None,
                          help="Workspace root (parent containing the task "
                               "directories). Default: cwd.")
    pdis_run.add_argument("--ram-floor", default="10g",
                          help="Minimum free RAM before launching each task "
                               "(default 10g). Suffix g/m/k accepted.")
    pdis_run.add_argument("--disk-floor", default="auto",
                          help="Minimum free disk before launching each task. "
                               "'auto' (default) estimates from each task.yaml's "
                               "dataset sizes ×2. Suffix g/m/k accepted to override.")
    pdis_run.add_argument("--agent", choices=["auto", "claude", "codex", "cursor"], default="auto",
                          help="Agent CLI to launch (default: auto-detect).")
    pdis_run.add_argument("--model", default=None,
                          help="Model passed to the agent. Default: claude-opus-4-7[1m] for claude (1M context); "
                               "composer-2 for cursor; agent default for codex.")
    pdis_run.add_argument("--effort", default="max",
                          choices=["low", "medium", "high", "xhigh", "max"],
                          help="Effort level passed to claude (default max).")
    pdis_run.add_argument("--stall-threshold", type=int, default=900,
                          help="Seconds of stream-json silence before a task is "
                               "flagged as stalled (default 900 = 15min). The task "
                               "is NOT killed — just flagged in status.")
    pdis_run.add_argument("--max-rounds", type=int, default=None,
                          help="Stop each running agent after this many completed "
                               "decision rounds appear in results.tsv.")
    pdis_run.add_argument("--force-mode", action="store_true",
                          help="With --max-rounds, resume a cleanly exited agent "
                               "session until the target round count is reached. "
                               "Requires the agent stream to expose a session id.")
    pdis_run.add_argument("--reflect", action="store_true",
                          help="After each task finishes successfully, resume the same "
                               "agent session with a post-run reflection prompt.")
    pdis_run.add_argument("--reflect-prompt", default=None,
                          help="Prompt path relative to each task for --reflect "
                               "(default prompts/5_reflect.md).")
    pdis_run.add_argument("--reflection-root", default=None,
                          help="Directory for --reflect outputs. Default: "
                               "<autozyme-framework>/reflections/. This root "
                               "contains prompt_reflect_feedback/ and "
                               "zyme_cli_feedback/.")
    pdis_run.add_argument("--reflect-category", default="iteration",
                          help="Reflection category label for --reflect (default iteration).")
    pdis_run.add_argument("--detach", action="store_true",
                          help="Daemonize (POSIX double-fork). Caller returns "
                               "immediately. Default: foreground (Ctrl-C cleanly stops).")
    pdis_run.add_argument("--dry-run", dest="dry_run", action="store_true",
                          help="Resolve tasks and print the launch plan without "
                               "starting anything.")
    pdis_run.set_defaults(func=cmd_dispatch)

    pdis_st = pdis_sub.add_parser(
        "status",
        help="One-screen status of the active or last dispatch run. "
             "Read-only; safe to run any time.",
    )
    pdis_st.add_argument("--workspace", default=None,
                         help="Workspace root. Default: cwd.")
    pdis_st.set_defaults(func=cmd_dispatch_status)

    pdis_us = pdis_sub.add_parser(
        "usage",
        help="Summarize token/cost telemetry for the active or last dispatch run.",
    )
    pdis_us.add_argument("--workspace", default=None,
                         help="Workspace root. Default: cwd.")
    pdis_us.add_argument("--token-budget", type=_parse_token_budget, default=None,
                         help="Optional token budget to compare against usage. "
                              "Suffix k/m/b accepted.")
    pdis_us.add_argument("--budget-basis", choices=["total", "non-cache"], default="total",
                         help="Budget basis: total includes cache reads; "
                              "non-cache is input+output+cache writes.")
    pdis_us.add_argument("--price-model", default=None,
                         help="Override model price id/name for USD estimation.")
    pdis_us.add_argument("--json", dest="json_output", action="store_true",
                         help="Emit raw JSON instead of a text summary.")
    pdis_us.set_defaults(func=cmd_dispatch_usage)

    pdis_price = pdis_sub.add_parser(
        "prices",
        help="List built-in model prices used for usage cost estimates.",
    )
    pdis_price.add_argument("--json", dest="json_output", action="store_true",
                            help="Emit raw JSON instead of a table.")
    pdis_price.set_defaults(func=cmd_dispatch_prices)

    pdis_res = pdis_sub.add_parser(
        "resume",
        help="Send one prompt/message to a recorded dispatch agent session.",
    )
    pdis_res.add_argument("task", nargs="?",
                          help="Task name to resume. Optional only when the dispatch has one task.")
    pdis_res.add_argument("--workspace", default=None,
                          help="Workspace root. Default: cwd.")
    pdis_res_mode = pdis_res.add_mutually_exclusive_group(required=True)
    pdis_res_mode.add_argument("--prompt", default=None,
                              help="Prompt path relative to the task dir to send as `read and follow <prompt>`.")
    pdis_res_mode.add_argument("--message", default=None,
                              help="Literal message to send to the resumed session.")
    pdis_res.add_argument("--stall-threshold", type=int, default=None,
                          help="Seconds of JSON-stream silence before marking the resumed session stalled.")
    pdis_res.add_argument("--dry-run", action="store_true",
                          help="Print the resume plan without launching the agent.")
    pdis_res.set_defaults(func=cmd_dispatch_resume)

    pdis_lg = pdis_sub.add_parser(
        "logs",
        help="Tail parsed events for one task in the current dispatch. "
             "Each line is one normalized event (text snippet, tool call, "
             "zyme run/accept/reject, etc.) — not raw stream-json.",
    )
    pdis_lg.add_argument("task", help="Task name (basename of task directory).")
    pdis_lg.add_argument("--workspace", default=None,
                         help="Workspace root. Default: cwd.")
    pdis_lg.add_argument("--follow", "-f", action="store_true",
                         help="Stream new events as they arrive (Ctrl-C to stop).")
    pdis_lg.add_argument("-n", dest="last_n", type=int, default=10,
                         help="Show last N events (default 10).")
    pdis_lg.add_argument("--full", action="store_true",
                         help="Show all events (no cap).")
    pdis_lg.set_defaults(func=cmd_dispatch_logs)

    pdis_sp = pdis_sub.add_parser(
        "stop",
        help="Cleanly stop the active dispatch master + any running claude. "
             "Sends SIGTERM, waits up to 30s for graceful exit, then escalates.",
    )
    pdis_sp.add_argument("--workspace", default=None,
                         help="Workspace root. Default: cwd.")
    pdis_sp.set_defaults(func=cmd_dispatch_stop)

    pdis_wt = pdis_sub.add_parser(
        "wait",
        help="Block until the dispatch master exits. Use with run_in_background "
             "in agent workflows.",
    )
    pdis_wt.add_argument("--workspace", default=None,
                         help="Workspace root. Default: cwd.")
    pdis_wt.add_argument("--poll", type=int, default=60,
                         help="Poll interval in seconds (default: 60).")
    pdis_wt.add_argument("--timeout", type=int, default=None,
                         help="Max wait in seconds. Exit 124 on timeout.")
    pdis_wt.add_argument("-v", "--verbose", action="store_true",
                         help="Print progress on each poll.")
    pdis_wt.set_defaults(func=cmd_dispatch_wait)

    # ---- prompt registry --------------------------------------------------
    # First nested subcommand group in this CLI. Registry stores versioned
    # snapshots of phase prompts (autozyme_cli/zyme/prompts/<field>/<N>_<slot>.md)
    # under PromptLab/prompt_snapshots/ when PromptLab exists, each with a card.yaml carrying free-form
    # dimension labels (aggressiveness, thread_breadth, ...) and a hypothesis.
    # Lets the user A/B prompt edits without losing history or losing
    # attribution of which prompt produced which results.
    pp = sub.add_parser(
        "prompt",
        help="Prompt registry: snapshot/list/diff/use/annotate live phase prompts",
    )
    pp_sub = pp.add_subparsers(dest="prompt_cmd", required=True)

    ppsave = pp_sub.add_parser(
        "save",
        help="Snapshot the current live prompt as a new registry version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Snapshot the current iterate prompt with a label and hypothesis\n"
            "  zyme prompt save autozyme_cli/zyme/prompts/Bio/2_iterate.md \\\n"
            "      --as aggr10_v1 --label aggressiveness=10 \\\n"
            "      -m \"raise aggressiveness to surface weaker signals\"\n\n"
            "  # Multi-dimensional labels for later A/B comparison\n"
            "  zyme prompt save autozyme_cli/zyme/prompts/Bio/2_iterate.md \\\n"
            "      --as broad_threads_v2 --label aggressiveness=8 --label thread_breadth=8\n"
        ),
    )
    ppsave.add_argument(
        "live_path",
        help="Path to a live prompt file under autozyme_cli/zyme/prompts/<field>/, "
             "e.g. autozyme_cli/zyme/prompts/Bio/2_iterate.md",
    )
    ppsave.add_argument(
        "--as", dest="name", required=True,
        help="Short human-readable name for this version (e.g. aggr10_v1). "
             "Becomes part of the prompt_id.",
    )
    ppsave.add_argument(
        "--label", dest="labels", action="append", default=[],
        help="Declared dimension label, repeat: --label aggressiveness=10 "
             "--label thread_breadth=8. Numeric values stored as float; "
             "non-numeric stored as string. Free-form key namespace.",
    )
    ppsave.add_argument(
        "-m", "--message", dest="message", default="",
        help="Hypothesis: what you expect this prompt version to do differently. "
             "Multi-line via shell quoting. Stored as the card's `hypothesis` field.",
    )
    ppsave.add_argument(
        "--notes", dest="notes", default="",
        help="(optional) Longer free-form notes; stored as the card's `notes` field.",
    )
    ppsave.set_defaults(func=cmd_prompt_save)

    pplist = pp_sub.add_parser("list", help="List snapshots in the registry")
    pplist.add_argument("--field", default=None,
                        help="Filter by field (Bio / OtherField).")
    pplist.add_argument("--slot", default=None,
                        help="Filter by slot (iterate / init / bootstrap / ...).")
    pplist.add_argument(
        "--label", dest="label_filters", action="append", default=[],
        help="Filter by label expression: --label aggressiveness>=8. "
             "Operators: =, >=, <=, >, <. Comparison operators require numeric RHS.",
    )
    pplist.set_defaults(func=cmd_prompt_list)

    ppshow = pp_sub.add_parser(
        "show",
        help="Print one snapshot's card.yaml (and optionally prompt.md) to stdout",
    )
    ppshow.add_argument("prompt_id",
                        help="prompt_id, registered name, or unique substring.")
    ppshow.add_argument("--card-only", action="store_true",
                        help="Print card.yaml only; suppress the prompt.md body.")
    ppshow.set_defaults(func=cmd_prompt_show)

    ppdiff = pp_sub.add_parser(
        "diff",
        help="git diff --no-index between two snapshot prompt.md files",
    )
    ppdiff.add_argument("id_a", help="First snapshot (prompt_id, name, or substring).")
    ppdiff.add_argument("id_b", help="Second snapshot.")
    ppdiff.set_defaults(func=cmd_prompt_diff)

    ppuse = pp_sub.add_parser(
        "use",
        help="Restore a registry snapshot as the live prompt file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Install a snapshot by short name\n"
            "  zyme prompt use aggr10_v1\n\n"
            "  # By full prompt_id (when names are ambiguous)\n"
            "  zyme prompt use zyme_iterate_20260508_aggr10_v1_a1b2c3d4\n\n"
            "  # Force-overwrite even if the live file has unsaved edits\n"
            "  zyme prompt use baseline_v1 --force\n"
        ),
    )
    ppuse.add_argument("prompt_id")
    ppuse.add_argument(
        "--force", action="store_true",
        help="Overwrite the live file even if it has unsaved changes "
             "(content sha differs from the active snapshot).",
    )
    ppuse.set_defaults(func=cmd_prompt_use)

    ppann = pp_sub.add_parser(
        "annotate",
        help="Add labels / update notes on an existing snapshot (post-hoc)",
    )
    ppann.add_argument("prompt_id")
    ppann.add_argument("--label", dest="labels", action="append", default=[],
                       help="Label to merge into the card's labels mapping. Repeat.")
    ppann.add_argument("--note", default=None,
                       help="Replace the notes field. Pass empty string to clear.")
    ppann.set_defaults(func=cmd_prompt_annotate)

    # ---- bench scaffolding ------------------------------------------------
    # Two-stage: register-template freezes a known-good task at a lifecycle
    # commit; init scaffolds N replicate task folders from that template with
    # a chosen prompt installed and .zyme_meta.yaml dropped. See commands.py
    # `Bench scaffolding` section for the full design.
    pb = sub.add_parser(
        "bench",
        help="Bench scaffolding: register clean task templates + scaffold replicate runs",
    )
    pb_sub = pb.add_subparsers(dest="bench_cmd", required=True)

    pbreg = pb_sub.add_parser(
        "register-template",
        help="Snapshot a task at its `zyme init` or post-init commit into bench_templates/<stage>/<name>/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Register a post-init snapshot for benchmarking the iterate prompt\n"
            "  zyme bench register-template /path/to/test_findmarkers --as findmarkers\n\n"
            "  # Register a bare-init snapshot for benchmarking the init prompt itself\n"
            "  zyme bench register-template /path/to/test_findmarkers --as findmarkers \\\n"
            "      --stage init --at init\n\n"
            "  # Pin to an explicit commit when auto-detection can't find init/post_init\n"
            "  zyme bench register-template /path/to/test_findmarkers --as findmarkers \\\n"
            "      --commit a1b2c3d --stage iterate\n"
        ),
    )
    pbreg.add_argument("task_dir",
                       help="Path to a fully-initialized task directory (must be a git repo with task.yaml).")
    pbreg.add_argument("--as", dest="name", required=True,
                       help="Template name (e.g. findallmarker). Becomes "
                            "bench_templates/<stage>/<name>/.")
    pbreg.add_argument(
        "--stage", default="iterate",
        help="Which prompt stage this template is used to bench. Common values: "
             "iterate (default), init, scaling, package. Free-form — typos fail "
             "loudly when a suite later looks for a template under that subdir.",
    )
    pbreg.add_argument(
        "--at", choices=["init", "post_init"], default="post_init",
        help="Lifecycle commit to snapshot. 'init' = bare scaffold from `zyme init` "
             "(used with --stage init to test the init prompt itself). 'post_init' "
             "(default) = parent of the first iterate-style commit, i.e. the state "
             "after the init prompt finished its work but before any optimization "
             "rounds (used with --stage iterate).",
    )
    pbreg.add_argument(
        "--commit", default=None, metavar="SHA",
        help="Explicit commit sha (or 'HEAD'), bypassing --at auto-detection. "
             "Use when the task's git history has been squashed/reset and "
             "neither init nor post_init commits can be found. Pair with "
             "--stage iterate (which auto-resets pipeline/run.{R,py} to match "
             "reference.{R,py}, simulating post-init state).",
    )
    pbreg.add_argument("--force", action="store_true",
                       help="Overwrite an existing template with the same name.")
    pbreg.set_defaults(func=cmd_bench_register_template)

    pblst = pb_sub.add_parser("list-templates",
                              help="List registered bench templates, grouped by stage")
    pblst.add_argument("--stage", default=None,
                       help="Filter to one stage (default: show all stages).")
    pblst.set_defaults(func=cmd_bench_list_templates)

    pbdoc = pb_sub.add_parser(
        "doctor",
        help="Validate suite templates, symlink sources, datasets, and clean iterate start state",
    )
    pbdoc.add_argument("suite_id",
                       help="Suite manifest id (matches autozyme_cli/bench_suites/<id>.yaml).")
    pbdoc.add_argument("--only", action="append", default=[],
                       help="Limit to one or more suite task ids/templates. Repeat or comma-separate.")
    pbdoc.set_defaults(func=cmd_bench_doctor)

    pbstat = pb_sub.add_parser(
        "status",
        help="Show prompt snapshots, template readiness, and bench-run progress",
    )
    pbstat.add_argument("suite_id", nargs="?",
                        help="Optional suite id to focus status on one benchmark suite.")
    pbstat.add_argument("--only", action="append", default=[],
                        help="With a suite id, limit template status to task ids/templates.")
    pbstat.add_argument("--root", default=None,
                        help="Override the bench-runs root directory.")
    pbstat.add_argument("--field", default=None,
                        help="When no suite is given, filter prompt snapshots by field.")
    pbstat.add_argument("--slot", default=None,
                        help="When no suite is given, filter prompt snapshots by slot.")
    pbstat.add_argument("--limit", type=int, default=12,
                        help="Max prompt snapshots / bench runs to show (default: 12).")
    pbstat.set_defaults(func=cmd_bench_status)

    pbinit = pb_sub.add_parser(
        "init",
        help="Scaffold N replicate task folders for a benchmark suite with a chosen prompt installed",
    )
    pbinit.add_argument("suite_id",
                        help="Suite manifest id (matches autozyme_cli/bench_suites/<id>.yaml).")
    pbinit.add_argument("--prompt", dest="prompt_id", required=True,
                        help="Registry prompt_id (or unique name/substring) to install in each task.")
    pbinit.add_argument("--reps", type=int, default=None,
                        help="Replicates per task. Default: suite's `default_reps` (or 1).")
    pbinit.add_argument("--only", action="append", default=[],
                        help="Limit to one or more suite task ids/templates. Repeat or comma-separate.")
    pbinit.add_argument("--out", default=None,
                        help="Output dir. Default: <workspace>/bench_runs/<suite_id>__<prompt_name>__<timestamp>/ — bakes in the suite (and thus task name for single-task suites) so `ls bench_runs/` is self-describing.")
    pbinit.add_argument("--purpose", default=None,
                        help="Short human-readable experiment purpose written to EXPERIMENT.md.")
    pbinit.add_argument("--force", action="store_true",
                        help="Overwrite an existing output dir (DESTRUCTIVE — wipes bench_runs/<name>/).")
    pbinit.add_argument("--name", default=None,
                        help="Custom replicate folder name (replaces default `<task_id>_r<n>`). "
                             "Implies single-replicate mode: requires --reps 1 and exactly one task "
                             "(via single-task suite or --only <id>). Skips suite-level scaffolding "
                             "(no bench_manifest.yaml, no EXPERIMENT.md, no autozyme-framework symlink "
                             "in --out) — drops a self-contained replicate at `<--out>/<name>/`. "
                             "Lets `--out` point at an existing dir (only the named subdir must be free, "
                             "or pass --force to wipe just that subdir). Default --out under this mode "
                             "is cwd. Use for adding an agent's replicate next to existing siblings.")
    pbinit.set_defaults(func=cmd_bench_init)

    pbstart = pb_sub.add_parser(
        "start",
        help="Launch agent workers for an existing bench run via dispatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  zyme bench start zyme_iterate_..._20260510_120000 --agent codex --detach\n"
            "  zyme bench start zyme_iterate_... --agent cursor --max-rounds 2 --force-mode --only findallmarker\n"
            "  zyme bench start /path/to/bench_runs/run --agent claude --only findallmarker\n\n"
            "Use `zyme dispatch status --workspace <bench_run>` to watch a detached run."
        ),
    )
    pbstart.add_argument("run",
                         help="Bench run dir or name under bench_runs/ (must contain bench_manifest.yaml).")
    pbstart.add_argument("--root", default=None,
                         help="Override bench-runs root for name lookup.")
    pbstart.add_argument("--only", action="append", default=[],
                         help="Limit to one or more bench task ids/templates/rep dirs. Repeat or comma-separate.")
    pbstart.add_argument("--agent", choices=["auto", "claude", "codex", "cursor"], default="auto",
                         help="Agent CLI to launch (default: auto-detect).")
    pbstart.add_argument("--prompt", default=None,
                         help="Prompt path relative to each task dir. Default: inferred from bench_manifest.")
    pbstart.add_argument("--ram-floor", default="10g",
                         help="Minimum free RAM before launching each task (default 10g).")
    pbstart.add_argument("--disk-floor", default="auto",
                         help="Minimum free disk before launching each task. 'auto' estimates from task.yaml.")
    pbstart.add_argument("--model", default=None,
                         help="Model passed to the agent. Default: claude-opus-4-7[1m] for claude (1M context); composer-2 for cursor; agent default for codex.")
    pbstart.add_argument("--effort", default="max",
                         choices=["low", "medium", "high", "xhigh", "max"],
                         help="Effort level passed to claude; ignored by codex unless configured externally.")
    pbstart.add_argument("--stall-threshold", type=int, default=900,
                         help="Seconds of JSON-stream silence before a task is flagged stalled (default 900).")
    pbstart.add_argument("--max-rounds", type=int, default=None,
                         help="Stop each running agent after this many completed decision rounds appear in results.tsv.")
    pbstart.add_argument("--force-mode", action="store_true",
                         help="With --max-rounds, resume a cleanly exited agent session until the target is reached.")
    pbstart.add_argument("--reflect", action="store_true",
                         help="After each task finishes successfully, resume the same agent session with prompts/5_reflect.md and write experiment-local reflections.")
    pbstart.add_argument("--reflect-prompt", default=None,
                         help="Prompt path relative to each task for --reflect (default prompts/5_reflect.md).")
    pbstart.add_argument("--reflect-category", default=None,
                         help="Reflection category label. Default inferred from the bench suite stage.")
    pbstart.add_argument("--detach", action="store_true",
                         help="Daemonize. Default: foreground (Ctrl-C cleanly stops).")
    pbstart.add_argument("--dry-run", action="store_true",
                         help="Resolve the run and print the launch plan without starting agents.")
    pbstart.set_defaults(func=cmd_bench_start)

    pbusage = pb_sub.add_parser(
        "usage",
        help="Summarize token/cost telemetry for a bench run",
    )
    pbusage.add_argument("run",
                         help="Bench run dir or name under bench_runs/ (must contain bench_manifest.yaml).")
    pbusage.add_argument("--root", default=None,
                         help="Override bench-runs root for name lookup.")
    pbusage.add_argument("--token-budget", type=_parse_token_budget, default=None,
                         help="Optional token budget to compare against usage. "
                              "Suffix k/m/b accepted.")
    pbusage.add_argument("--budget-basis", choices=["total", "non-cache"], default="total",
                         help="Budget basis: total includes cache reads; "
                              "non-cache is input+output+cache writes.")
    pbusage.add_argument("--price-model", default=None,
                         help="Override model price id/name for USD estimation.")
    pbusage.add_argument("--json", dest="json_output", action="store_true",
                         help="Emit raw JSON instead of a text summary.")
    pbusage.set_defaults(func=cmd_bench_usage)

    pbprice = pb_sub.add_parser(
        "prices",
        help="List built-in model prices used for usage cost estimates",
    )
    pbprice.add_argument("--json", dest="json_output", action="store_true",
                         help="Emit raw JSON instead of a table.")
    pbprice.set_defaults(func=cmd_bench_prices)

    pbruns = pb_sub.add_parser(
        "list",
        help="List bench runs (default: <workspace>/bench_runs/)",
    )
    pbruns.add_argument("--root", default=None,
                        help="Override the bench-runs root directory.")
    pbruns.set_defaults(func=cmd_bench_list)

    # ---- audit (per-task command log) -------------------------------------
    pa = sub.add_parser(
        "audit",
        parents=[task_dir_parent],
        help="Show this task's command audit log (.zyme/audit.jsonl). "
             "Every `zyme` invocation in a task dir appends one row: "
             "timestamp, subcommand, argv, duration, exit code, output deltas.",
    )
    pa.add_argument("--last", type=int, default=20,
                    help="Show last N entries (default 20). Pass 0 for all.")
    pa.add_argument("--json", action="store_true",
                    help="Emit raw JSONL instead of the table view.")
    pa.set_defaults(func=cmd_audit)

    # ---- cost (per-task time + token + USD accounting) --------------------
    pcost = sub.add_parser(
        "cost",
        parents=[task_dir_parent],
        help="Per-task time + token + estimated USD. Time from "
             ".zyme/audit.jsonl (calendar span, agent active-min, CLI wall, "
             "per-phase). Tokens from Claude Code transcripts (via cc_session) "
             "and Codex rollouts (matched by cwd); USD via the dispatch price "
             "table. Read-only; not itself audited.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  zyme cost                 # summary for the task in cwd\n"
            "  zyme cost --json          # machine-readable report\n"
            "  zyme cost --gap-min 15    # treat gaps >15m as idle (default 10)\n"
            "  zyme cost --model gpt-5.5 # force a price model\n\n"
            "Token sources, in precedence: captured runs (.zyme/agent_usage.jsonl,\n"
            "written by `zyme cost-capture` — authoritative, any agent), then\n"
            "Claude transcripts (via cc_session), then Codex rollouts (cwd-matched,\n"
            "scoped only; ancestor-cwd rollouts are excluded/noted). Cursor has no\n"
            "on-disk usage, so capture it. Time works for any agent."
        ),
    )
    pcost.add_argument("--gap-min", type=float, default=10.0,
                       help="Inter-event gap (minutes) above which time counts "
                            "as idle and is excluded from agent active-min "
                            "(default 10).")
    pcost.add_argument("--model", default=None,
                       help="Override the model used for USD pricing "
                            "(default: most-frequent model in the transcripts).")
    pcost.add_argument("--json", action="store_true",
                       help="Emit the full report as JSON.")
    pcost.set_defaults(func=cmd_cost)

    # ---- cost-capture (forward token capture for any agent) --------------
    pcap = sub.add_parser(
        "cost-capture",
        parents=[task_dir_parent],
        help="Record one agent run's token usage into .zyme/agent_usage.jsonl "
             "from its --output-format stream-json output (stdin or --from). "
             "The authoritative per-run source `zyme cost` reads; the only way "
             "to get Cursor tokens (it persists no usage on disk).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # live pipe (stdin is echoed through unless --quiet)\n"
            "  cursor-agent -p --output-format stream-json \"<task>\" | \\\n"
            "      zyme cost-capture --agent cursor\n\n"
            "  # post-hoc from a saved stream-json log\n"
            "  zyme cost-capture --agent codex --from run.stream.jsonl\n\n"
            "Parses the result `usage` (Claude/Cursor) or cumulative token_count\n"
            "(Codex). Captured agents take precedence over the retroactive\n"
            "transcript/rollout scrapers in `zyme cost`."
        ),
    )
    pcap.add_argument("--agent", required=True,
                      choices=["cursor", "codex", "claude"],
                      help="Which agent produced the stream (labels the record).")
    pcap.add_argument("--from", dest="from_file", default=None,
                      help="Read stream-json from this file instead of stdin.")
    pcap.add_argument("--quiet", action="store_true",
                      help="Do not echo stdin through to stdout (live-pipe mode).")
    pcap.set_defaults(func=cmd_cost_capture)

    # ---- package: packaging-time utilities ------------------------------
    # Lint, intercept-checking, preflight gate, smoke-parity, and manifest
    # sync. See commands/package/ for the per-subcommand implementations.
    ppkg = sub.add_parser(
        "package",
        help="Patch packaging utilities: lint, check-intercept, preflight, "
             "smoke-parity, sync-manifests",
    )
    ppkg_sub = ppkg.add_subparsers(dest="package_cmd", required=True)

    ppkg_lint = ppkg_sub.add_parser(
        "lint",
        help="Static checks against CAVEATS-derived rules on patch files. "
             "Default: lint all patches in autozyme_r + autozyme_py.",
    )
    ppkg_lint.add_argument("--patch", default=None,
                           help="Lint only this patch name (matches both R and Python).")
    ppkg_lint.add_argument("--all", action="store_true",
                           help="Lint every patch (default when --patch is omitted).")
    ppkg_lint.set_defaults(func=cmd_package_lint)

    ppkg_chk = ppkg_sub.add_parser(
        "check-intercept", parents=[task_dir_parent],
        help="Run smoke under an instrumented dispatcher; FAIL if any "
             "registered target's fast fn was not called.",
    )
    ppkg_chk.add_argument("--patch", default=None,
                          help="Patch name (default: infer from task.yaml::target_function).")
    ppkg_chk.add_argument("--tier", default="tiny",
                          help="Tier to run smoke on (default: tiny).")
    ppkg_chk.add_argument("--lang", choices=["py", "R"], default=None,
                          help="Override language detection.")
    ppkg_chk.set_defaults(func=cmd_package_check_intercept)

    ppkg_pre = ppkg_sub.add_parser(
        "preflight", parents=[task_dir_parent],
        help="Run lint + portability scan + smoke-parity. The same gate "
             "that `zyme attest` runs by default.",
    )
    ppkg_pre.add_argument("--continue", dest="continue_on_fail",
                          action="store_true",
                          help="Continue past failures so you see every "
                               "problem in one pass.")
    ppkg_pre.add_argument("--skip-parity", action="store_true",
                          help="Skip the smoke-parity step (cheaper; "
                               "useful when smoke is intentionally being iterated).")
    ppkg_pre.set_defaults(func=cmd_package_preflight)

    ppkg_par = ppkg_sub.add_parser(
        "smoke-parity", parents=[task_dir_parent],
        help="Run smoke at one tier; FAIL if evaluate metrics fall below "
             "task.yaml thresholds (smoke is timing a different region than pipeline/run).",
    )
    ppkg_par.add_argument("--patch", default=None,
                          help="Patch name (default: infer from task.yaml).")
    ppkg_par.add_argument("--tier", default="tiny",
                          help="Tier to run smoke on (default: tiny).")
    ppkg_par.add_argument("--lang", choices=["py", "R"], default=None,
                          help="Override language detection.")
    ppkg_par.set_defaults(func=cmd_package_smoke_parity)

    ppkg_ver = ppkg_sub.add_parser(
        "check-versions",
        help="Flag patches whose tested_against pin has drifted from the "
             "installed upstream version. Exits non-zero on drift / missing.",
    )
    ppkg_ver.add_argument("--only-drift", action="store_true",
                          help="Hide patches in `ok` status; show only "
                               "drift + missing rows.")
    ppkg_ver.add_argument("--framework-root", default=None,
                          help="Override framework root (default: auto-detect).")
    ppkg_ver.set_defaults(func=cmd_package_check_versions)

    ppkg_sync = ppkg_sub.add_parser(
        "sync-manifests",
        help="Reconcile UPSTREAMS / .zyme_upstreams against register_patch() "
             "calls. Default is dry-run; pass --apply to write changes.",
    )
    ppkg_sync.add_argument("--apply", action="store_true",
                           help="Write the reconciled manifests in place.")
    ppkg_sync.add_argument("--framework-root", default=None,
                           help="Override framework root (default: auto-detect).")
    ppkg_sync.set_defaults(func=cmd_package_sync_manifests)

    add_datasets_subparser(sub)

    return p


def _full_cmd_name(args):
    """Compose the audit-log command label, including the nested verb.

    `zyme baseline record` → "baseline record" (not just "baseline").
    Falls back to args.cmd alone when the family has no sub-verb.
    """
    cmd = getattr(args, "cmd", None)
    for nested_attr in ("baseline_cmd", "dispatch_cmd", "prompt_cmd",
                        "bench_cmd", "package_cmd"):
        verb = getattr(args, nested_attr, None)
        if verb:
            return f"{cmd} {verb}"
    return cmd


_GIT_LOCKED_COMMANDS = {"run", "dryrun", "accept", "reject", "rollback", "profile"}


def _find_git_root(start: Path) -> Path | None:
    cur = start.resolve()
    while True:
        if (cur / ".git").exists():
            return cur
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent


@contextlib.contextmanager
def _workspace_git_lock(args):
    """Serialize zyme commands that mutate or depend on the shared git HEAD."""
    if getattr(args, "cmd", None) not in _GIT_LOCKED_COMMANDS:
        yield
        return
    if os.environ.get("ZYME_GIT_LOCK_HELD") == "1":
        yield
        return

    task_dir = Path(getattr(args, "task_dir", None) or os.getcwd())
    git_root = _find_git_root(task_dir)
    if git_root is None:
        yield
        return

    lock_path = git_root / ".zyme_git.lock"
    payload = {
        "pid": os.getpid(),
        "cmd": _full_cmd_name(args),
        "task_dir": str(task_dir.resolve()),
        "started": time.time(),
    }
    waited_notice_at = 0.0
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps(payload).encode("utf-8"))
            os.close(fd)
            fd = -1
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                holder = lock_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                age = 0.0
                holder = "<unreadable>"
            if age > 6 * 60 * 60:
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            now = time.time()
            if now - waited_notice_at >= 30:
                print(
                    "[zyme] waiting for shared git lock "
                    f"{lock_path} held by {holder}",
                    file=sys.stderr,
                    flush=True,
                )
                waited_notice_at = now
            time.sleep(2)

    try:
        yield
    finally:
        try:
            if lock_path.exists():
                current = lock_path.read_text(encoding="utf-8", errors="replace")
                if f'"pid": {os.getpid()}' in current:
                    lock_path.unlink()
        except OSError:
            pass


def main():
    # Windows defaults stdout/stderr to cp1252; any non-ASCII byte (e.g. arrows
    # in info() messages) raises UnicodeEncodeError mid-command. Reconfigure to
    # utf-8 with replacement fallback. Pipes/captured streams may not expose
    # reconfigure() — skip them and let print fall back to the default codec.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    parser = build_parser()
    args = parser.parse_args()
    # Audit: wraps args.func(args) and appends one row to <task>/.zyme/audit.jsonl
    # when cwd (or --task-dir) is a zyme task. Workspace-level commands skip
    # naturally because their cwd has no task.yaml. Audit failures never mask
    # real errors. The `audit`/`cost` subcommands aren't recorded (read-only
    # analysis; recording them would pollute the very log they read).
    if getattr(args, "cmd", None) in ("audit", "cost", "cost-capture"):
        args.func(args)
        return
    with AuditContext(args, sys.argv, _full_cmd_name(args)):
        with _workspace_git_lock(args):
            args.func(args)


if __name__ == "__main__":
    main()
