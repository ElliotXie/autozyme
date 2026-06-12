"""cmd_profile — orchestrate one profile run end-to-end.

Flow:
  1. Resolve task_dir, language, tier.
  2. Resolve effective backend (full → cpu fallback if scalene/profvis missing;
     mem refuses to fall back).
  3. Build extra_env + (Python full only) scalene-wrapper python args.
  4. Invoke runner.run_task with skip_evaluate=True.
  5. Parse the backend-native artifact into a normalized profile dict.
  6. Write profile_history/<run-id>/profile.json and run.log.
     `--no-archive` uses profile_history/current/ as overwritten scratch.
  7. Emit either evidence card to stderr (default) or JSON to stdout (--json).

This command is a DIAGNOSTIC: it does NOT write to results.tsv, does NOT
commit, does NOT advance round counters. It exists purely so the agent
gets richer hotspot evidence than `zyme run`'s timing summary.
"""
import ast
import json
import re
import sys
from pathlib import Path

from zyme.utils import die, info, task_dir_from_args, detect_lang
from zyme.parsers.task_yaml import resolve_tiers, parse_executor
from zyme.runner import run_task

from zyme.commands.profile import backends, parsers, render, archive, native
from zyme.commands.profile import enrich as profile_enrich
from zyme.commands.profile.diff import cmd_profile_diff


def cmd_profile(args):
    # --diff sub-mode: bypass the normal profile-capture flow entirely.
    # Two profile.json paths in, delta table out. Useful after agent has
    # accepted a patch and wants to see what changed in the hot path.
    if getattr(args, "diff_a", None) is not None:
        return cmd_profile_diff(args)

    task_dir = task_dir_from_args(args)
    lang = detect_lang(task_dir)
    pipeline_dir = task_dir / "pipeline"
    executor = parse_executor(task_dir / "task.yaml")

    # Resolve tier (single-tier — profile is diagnostic, multi-tier sweeps
    # are the agent's job to schedule).
    requested = [args.dataset] if getattr(args, "dataset", None) else None
    try:
        tiers = resolve_tiers(task_dir / "task.yaml", requested)
    except ValueError as e:
        die(str(e))
    tier_entry = tiers[0]

    # Resolve backend with fallback.
    try:
        effective_backend, fallback_warn = backends.resolve(
            args.backend, lang, executor=executor,
        )
    except RuntimeError as e:
        die(str(e))
    # In --json mode, route status messages to stderr so stdout is pure JSON.
    json_mode = getattr(args, "json", False)
    say = (lambda m: print(f"[zyme] {m}", file=sys.stderr)) if json_mode else info

    if fallback_warn:
        # Loud banner: agents (and humans) have been fooled into thinking
        # "two passes done" when both passes silently fell back to cpu and
        # the only visible difference was the timestamp in the output dir
        # name. Surround the message with separators so it's impossible to
        # miss in a stream of other output.
        bar = "!" * 72
        say(bar)
        for line in str(fallback_warn).splitlines():
            say(line)
        say(bar)

    profile_dir, profile_rel = archive.prepare_run_dir(
        task_dir,
        backend=effective_backend,
        tier=tier_entry["tier"],
        archive=not getattr(args, "no_archive", False),
    )

    # Build env + (Python full mode only) scalene wrapper args.
    # Note: native backend skips ZYME_PROFILE entirely — the helper isn't
    # involved (sample(1) is an external observer, not an in-process
    # context manager).
    extra_env: dict[str, str] = {}
    pipeline_python_args = None
    side_observer = None
    mem_scope = None

    if effective_backend != "native":
        extra_env["ZYME_PROFILE"] = "1"
        extra_env["ZYME_PROFILE_BACKEND"] = effective_backend
        extra_env["ZYME_PROFILE_DIR"] = str(profile_dir)

    if effective_backend == "full" and lang == "py":
        scalene_outfile = profile_dir / "scalene.json"
        pipeline_python_args = backends.scalene_pipeline_args(scalene_outfile)
        extra_env["ZYME_SCALENE_ACTIVE"] = "1"
    elif effective_backend == "mem" and lang == "py":
        memray_outfile = profile_dir / "memray.bin"
        if _python_pipeline_uses_with_profile(pipeline_dir / "run.py"):
            # Scoped path: helpers.with_profile() starts memray. This keeps
            # allocation attribution focused on the target-method region
            # when pipeline/run.py has one. No external wrapper, because
            # memray permits only one active tracker per process.
            extra_env["ZYME_MEMRAY_SCOPED"] = "1"
            mem_scope = "with_profile"
        else:
            # Compatibility path: older Python tasks may not have a
            # with_profile() region. Capture the whole script so backend=mem
            # still produces a useful profile instead of silently missing
            # memray.bin.
            pipeline_python_args = backends.memray_pipeline_args(memray_outfile)
            extra_env["ZYME_MEMRAY_ACTIVE"] = "1"
            mem_scope = "whole_process"
    elif effective_backend == "native":
        side_observer = native.NativeSampler(out_dir=profile_dir)

    say(
        f"profile: tier={tier_entry['tier']} ({tier_entry['name']}) "
        f"backend={effective_backend} lang={lang} out={profile_rel}"
    )

    # Run.
    log = run_task(
        task_dir,
        dataset_entry=tier_entry,
        extra_env=extra_env,
        pipeline_python_args=pipeline_python_args,
        skip_evaluate=True,
        side_observer=side_observer,
    )
    # In --json mode, route the runner's log to stderr so stdout stays a
    # clean JSON document (pipe-friendly: `zyme profile --json | jq`).
    log_stream = sys.stderr if getattr(args, "json", False) else sys.stdout
    print(log, flush=True, file=log_stream)
    archive.write_run_log(profile_dir, log)

    # Extract totals from the pipeline log (emit_summary lines: speed_sec /
    # peak_mb / cpu_sec). These come from the same code path as `zyme run`,
    # so they're directly comparable to results.tsv numbers.
    totals = _extract_totals(log)

    # Parse + normalize. Native backend has its own normalizer (the data
    # source is a pile of sample-output files, not a single artifact).
    if effective_backend == "native":
        profile_data = native.parse_and_normalize(
            out_dir=profile_dir,
            sampler=side_observer,
            lang=lang,
            tier=tier_entry["tier"],
            hypothesis=getattr(args, "hypothesis", "") or "",
            totals=totals,
            artifact_prefix=profile_rel,
        )
    else:
        profile_data = parsers.normalize(
            backend=effective_backend,
            lang=lang,
            artifact_dir=profile_dir,
            hypothesis=getattr(args, "hypothesis", "") or "",
            tier=tier_entry["tier"],
            artifact_prefix=profile_rel,
            executor=executor,
            totals=totals,
        )

    if effective_backend == "mem" and lang == "py":
        if mem_scope == "with_profile" and "scope=with_profile" in log:
            profile_data.setdefault("notes", []).append(
                "scope=with_profile: memray Tracker was started inside "
                "helpers.with_profile(), so allocations are limited to the "
                "target profiled region."
            )
        elif mem_scope == "whole_process":
            profile_data.setdefault("notes", []).append(
                "scope=whole_process: pipeline/run.py did not expose a "
                "with_profile() region, so zyme used the external memray "
                "wrapper around the entire script."
            )

    actionable, actionable_notes = parsers.build_actionable_hotspots(profile_data)
    profile_data["actionable_hotspots"] = actionable
    profile_data.setdefault("notes", []).extend(actionable_notes)
    profile_data["call_chains"] = parsers.annotate_call_chains(profile_data)

    # Enrich: add `layer` + `n_calls`/`n_calls_est` per hotspot, plus
    # top-level `layer_breakdown` and `per_layer_top`. Tells the agent
    # which buckets are editable (task / library / primitive) and surfaces
    # the dominant function PER layer — answers "what should I attack at
    # what scope" up front instead of forcing the agent to mentally tag
    # every hotspot. Best-effort: any failure leaves the profile unchanged.
    try:
        target_pkg = profile_enrich.parse_target_pkg_from_task_yaml(task_dir)
        # The Rprof.out (when present) is the canonical raw artifact under
        # the run dir. Native backend doesn't produce one; Py uses
        # profile.out (cProfile) and gets its call counts from there.
        rprof_path = (profile_dir / "Rprof.out") if lang == "R" else None
        profile_enrich.enrich(
            profile_data,
            target_pkg=target_pkg,
            rprof_path=rprof_path,
            executor=executor,
            task_dir=task_dir,
        )
    except Exception as e:  # never fail the profile run for an enrichment hiccup
        profile_data.setdefault("notes", []).append(
            f"enrichment skipped: {type(e).__name__}: {e}"
        )

    # Top-hotspot opacity hint: when the #1 actionable frame is something the
    # current backend literally cannot see inside (`.Call` for Rprof, parent-
    # side multiprocessing waits for cProfile), the rank is honest but not
    # actionable. Suggest the agent rerun with `--backend native` (which walks
    # the OS-level sample stack and CAN see the C++/Cython / worker frames).
    opaque_hint = _top_hotspot_opaque_hint(actionable, effective_backend)
    if opaque_hint:
        profile_data.setdefault("notes", []).append(opaque_hint)

    # Keep every profile artifact, raw log, and normalized JSON in the same
    # profile_history run directory. pipeline/ remains source-only.
    profile_data.setdefault("artifacts", {})["run_log"] = f"{profile_rel}/run.log"
    profile_data.setdefault("artifacts", {})["profile_json"] = f"{profile_rel}/profile.json"
    archive.write_profile_json(profile_dir, profile_data)

    # Point profile_history/latest at this run so agents can read
    # profile_history/latest/profile.json without knowing the timestamp.
    # Only meaningful for archived runs (--no-archive uses /current already).
    if not getattr(args, "no_archive", False):
        archive.update_latest_symlink(task_dir / "profile_history", profile_dir)

    # Emit either JSON-to-stdout (machine-readable) or evidence card (stderr).
    if getattr(args, "json", False):
        print(json.dumps(profile_data, indent=2, default=str))
    else:
        render.evidence_card(profile_data, profile_dir=profile_rel)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_SUMMARY_LINE = re.compile(r"^([A-Za-z_]+):\s+([0-9.eE+-]+)\s*$", re.MULTILINE)


def _extract_totals(log: str) -> dict:
    """Pull speed_sec / peak_mb / cpu_sec from the runner's summary block."""
    out: dict = {}
    for m in _SUMMARY_LINE.finditer(log):
        key, val = m.group(1), m.group(2)
        try:
            v = float(val)
        except ValueError:
            continue
        if key == "speed_sec":
            out["wall_s"] = v
        elif key == "peak_mb":
            out["peak_mb"] = v
        elif key == "cpu_sec":
            out["cpu_s"] = v
    return out


def _python_pipeline_uses_with_profile(run_py: Path) -> bool:
    """Best-effort check for a scoped Python profiling region.

    We parse the AST and look for a call expression, not just the symbol
    name, so a plain import or explanatory comment doesn't force scoped
    memray.
    """
    try:
        text = run_py.read_text()
    except OSError:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "with_profile":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "with_profile":
            return True
    return False


# Patterns marking actionable_hotspots[0] entries that are technically the
# top of the profile but represent frames the current backend cannot see
# inside — the answer is to switch backend, not to optimize the frame.
_OPAQUE_NATIVE_RE = re.compile(
    r"(?:\.Call\b|<native\s*code>|psynch_cvwait|workq_kernreturn|"
    r"std::__1::condition_variable)",
    re.IGNORECASE,
)
_OPAQUE_WORKER_WAIT_RE = re.compile(
    r"multiprocessing\.(?:pool|connection)|"
    r"\bpool\b.*\b(?:wait|get|recv|map)\b|"
    r"\bworker_handler\b|\b_handle_results\b",
    re.IGNORECASE,
)


def _top_hotspot_opaque_hint(actionable: list, backend: str) -> str | None:
    """If `actionable[0]` looks opaque to the current backend, return a hint."""
    if not actionable or backend == "native":
        return None
    top = actionable[0]
    label = str(top.get("label") or "")
    raw_blob = " ".join(
        str(v) for v in (top.get("raw") or {}).values()
        if isinstance(v, (str, int, float))
    )
    text = f"{label} {raw_blob}"
    if _OPAQUE_NATIVE_RE.search(text):
        return (
            f"top actionable_hotspot ({label!r}) is an opaque native frame "
            f"the `{backend}` backend cannot see inside. Rerun with "
            f"`zyme profile --backend native --dataset <tier> --json` to "
            f"walk the C++ / Cython / BLAS stack and identify the real "
            f"function consuming time."
        )
    if _OPAQUE_WORKER_WAIT_RE.search(text):
        return (
            f"top actionable_hotspot ({label!r}) is a multiprocessing "
            f"parent-side wait frame — the parent is blocked while workers "
            f"do the real work, which `{backend}` can't see. Rerun with "
            f"`zyme profile --backend native --dataset <tier> --json` "
            f"(walks the native stack across all sampled threads) to see "
            f"where workers spend time."
        )
    return None
