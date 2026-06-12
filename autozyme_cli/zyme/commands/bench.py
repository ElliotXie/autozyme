"""`zyme bench` family — benchmark template + run scaffolding."""

import os
import sys
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from zyme.utils import (
    die, info, git,
)
from zyme.commands._shared import FRAMEWORK_ROOT, _write_memory_skeleton

# ============================================================================
# Bench scaffolding
# ============================================================================
# Two-stage workflow:
#   1. `zyme bench register-template <task_dir> --as <name> --at {init|post_init}`
#      Snapshots a known-good task at a specific lifecycle commit (right after
#      `zyme init`, or right after the init-prompt phase) into
#      autozyme_cli/bench_templates/<name>/. Uses `git archive` so only tracked
#      files come along — datasets, upstream_repo/, results.tsv, memory/,
#      prompts/ are all gitignored and stay out.
#   2. `zyme bench init <suite_id> --prompt <prompt_id> [--reps N]`
#      For each (task, rep) in the suite, copies the registered template into
#      bench_runs/<auto_name>/<task>_r<n>/, drops the prompt snapshot into
#      prompts/<N_slot>.md, writes .zyme_meta.yaml, and `git init` + initial
#      commit so subsequent `zyme run` calls work.
# Suites are defined in autozyme_cli/bench_suites/<suite_id>.yaml (committed).

BENCH_TEMPLATES_DIRNAME = "bench_templates"

BENCH_SUITES_DIRNAME = "bench_suites"

ZYME_META_FILENAME = ".zyme_meta.yaml"

EXPERIMENT_FILENAME = "EXPERIMENT.md"


# Dirs CO-LOCATED with the template (small, per-replicate state worth carrying):
#   .zyme/             — baselines_stash.tsv, best.ref, round.counter, etc.
#   reference_outputs/ — new layout: cached evaluate outputs per tier
# Plus reference_output_*/ glob — legacy per-tier dirs (tiny/medium/large).
_TEMPLATE_COPY_DIRS = (".zyme", "reference_outputs")

_TEMPLATE_COPY_GLOBS = ("reference_output_*",)

_TEMPLATE_COPY_FILES = ("results.tsv", "pipeline/profile.json")

_TEMPLATE_ZYME_COPY_FILES = (
    "baselines_history.tsv",
    "baselines_stash.tsv",
    "version_check.json",
)

_FRESH_MEMORY_STAGES = {"init", "iterate"}


# Dirs SYMLINKED at bench-init time (large, read-only, shared across replicates):
#   data/      — the actual datasets (often 10s-100s of MB)
#   upstream/  — older convention used by some legacy tasks (e.g. cellchat)
# Symlinks point back at the source task; iterate.md forbids modifying
# upstream code, so practically safe.
#
# upstream_repo/ is NOT in this list — it's recorded as a clone URL + commit
# in bench_template.yaml's top-level `upstream_repo:` block and freshly
# cloned at bench-init time. That makes templates portable across hosts and
# survives deletion of the original source task. See _detect_upstream_clone.
_TEMPLATE_SYMLINK_DIRS = ("data", "upstream")


# Patterns the per-task .gitignore uses to exclude state we deliberately
# hard-copied into the template. The template's .gitignore must NOT block
# these (else PromptLab git won't track them and the template isn't
# portable). Bench-init restores the strict task semantics.
_TEMPLATE_GITIGNORE_DROP_PATTERNS = (
    ".zyme/", ".zyme",
    "reference_outputs/", "reference_output/", "reference_output_*/",
    "results.tsv", "pipeline/profile.json",
    "memory/discoveries.md", "memory/active_opts.md", "memory/dead_ends.md",
    "memory/transfer_kit.md", "memory/best_state.md",
)


# A real iterate-round commit's message always starts with one of these tags
# (iterate.md "Concordance budget" enforces this). Setup commits use `setup:`
# prefix; pre-iterate test runs typically have free-form messages like
# "test setup". So commit-message scan beats results.tsv scan, which can
# include free-form crash rows from pre-iterate sanity tests.
_ITERATE_TAG_RE = re.compile(r"^\[(conservative|algorithmic|housekeeping)\]")




def _bench_templates_root() -> Path:
    """Resolve the bench-templates root (parent of stage subdirs).

    Layout: <root>/<stage>/<template_name>/. Stages map to which prompt the
    template is used to bench (iterate / init / scaling / package / ...);
    a single source task can be registered as both an `init`-stage template
    (bare scaffold for testing init prompts) and an `iterate`-stage one
    (post-init state for testing iterate prompts).

    Prefers <workspace>/PromptLab/bench_templates/ when PromptLab exists —
    templates carry user-specific absolute paths (data/, upstream_repo/
    symlink sources) and don't belong in the framework wheel. Falls back to
    autozyme_cli/bench_templates/ for fresh installs / CI.
    """
    # FRAMEWORK_ROOT = .../autozyme-framework/autozyme_cli/zyme/
    # workspace      = parent of autozyme-framework/
    workspace_root = FRAMEWORK_ROOT.parent.parent.parent
    promptlab = workspace_root / "PromptLab"
    if promptlab.exists() and promptlab.is_dir():
        return promptlab / "bench_templates"
    return FRAMEWORK_ROOT.parent / BENCH_TEMPLATES_DIRNAME




def _bench_template_path(stage: str, name: str) -> Path:
    """Full path to a bench template: <root>/<stage>/<name>/."""
    return _bench_templates_root() / stage / name




def _bench_suites_root() -> Path:
    return FRAMEWORK_ROOT.parent / BENCH_SUITES_DIRNAME




def _default_bench_runs_root() -> Path:
    """Workspace-root /bench_runs/. Workspace = parent of autozyme-framework/."""
    # FRAMEWORK_ROOT = .../autozyme-framework/autozyme_cli/zyme/
    # workspace      = .../autozyme-framework/.. = .../
    return FRAMEWORK_ROOT.parent.parent.parent / "bench_runs"




def _emit_simple_yaml(d: dict) -> str:
    """One-level YAML emit: top-level scalars + at most one level of nested
    mapping per key. Reuses registry's scalar formatter for quoting safety."""
    from zyme import registry
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            if not v:
                lines.append(f"{k}: {{}}")
            else:
                lines.append(f"{k}:")
                for sk, sv in v.items():
                    lines.append(f"  {sk}: {registry._yaml_scalar(sv)}")
        elif isinstance(v, list):
            if not v:
                lines.append(f"{k}: []")
            else:
                # render simple lists as `- item` lines
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {registry._yaml_scalar(item)}")
        else:
            lines.append(f"{k}: {registry._yaml_scalar(v)}")
    return "\n".join(lines) + "\n"




def _read_simple_yaml(path: Path) -> dict:
    """Read flat YAML + one-level nested mappings. Reuses registry's parser
    (which handles `key:` followed by indented `subkey: value` lines)."""
    from zyme import registry
    return registry.read_card_yaml(path)


def _write_experiment_doc(
    out_root: Path,
    *,
    suite: dict,
    selected_tasks: list[dict],
    prompt_id: str,
    prompt_card: dict,
    reps: int,
    created_at: str,
    purpose: str | None = None,
) -> None:
    """Write the run-root markdown card that explains the experiment intent."""
    task_list = ", ".join(f"`{t['id']}`" for t in selected_tasks)
    prompt_name = prompt_card.get("name") or ""
    hypothesis = prompt_card.get("hypothesis") or ""
    purpose_text = purpose or (
        f"Benchmark prompt `{prompt_id}` on suite `{suite['id']}` "
        f"for the `{suite['stage']}` stage."
    )
    lines = [
        "# Experiment",
        "",
        f"Purpose: {purpose_text}",
        "",
        "Design:",
        f"- Suite: `{suite['id']}`",
        f"- Stage: `{suite['stage']}`",
        f"- Field/slot: `{suite['field']}` / `{suite['prompt_slot']}`",
        f"- Prompt: `{prompt_id}`",
    ]
    if prompt_name:
        lines.append(f"- Prompt name: `{prompt_name}`")
    if hypothesis:
        lines.append(f"- Prompt hypothesis: {hypothesis}")
    lines.extend([
        f"- Replicates per task: {reps}",
        f"- Tasks: {task_list}",
        f"- Created: {created_at}",
        "",
        (
            "Primary readout: decision-round outcomes in each task's "
            "`results.tsv`, plus dispatch telemetry under `.zyme_dispatch/`."
        ),
        "",
        (
            "Agent/model details are appended by `zyme bench start` when the "
            "run is launched."
        ),
    ])
    (out_root / EXPERIMENT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_experiment_dispatch_doc(
    run_dir: Path,
    *,
    agent: str,
    model: str | None,
    effort: str,
    prompt_rel: str,
    tasks: list[dict],
    max_rounds: int | None,
    force_mode: bool,
    reflect: bool = False,
    reflect_prompt: str | None = None,
    reflection_root: Path | None = None,
    reflect_category: str | None = None,
    detach: bool = False,
) -> None:
    path = run_dir / EXPERIMENT_FILENAME
    if path.exists():
        text = path.read_text(encoding="utf-8").rstrip()
    else:
        text = "# Experiment\n\nPurpose: benchmark run launched by `zyme bench start`."
    task_list = ", ".join(f"`{t['name']}`" for t in tasks)
    launched_at = datetime.now().isoformat(timespec="seconds")
    block = [
        "",
        "## Dispatch",
        "",
        f"- Launched: {launched_at}",
        f"- Agent: `{agent}`",
        f"- Model: `{model or '(agent default)'}`",
        f"- Effort: `{effort}`",
        f"- Prompt file: `{prompt_rel}`",
        f"- Tasks: {task_list}",
        f"- Max rounds: {max_rounds if max_rounds is not None else '(none)'}",
        f"- Force mode: {bool(force_mode)}",
        f"- Reflect: {bool(reflect)}",
    ]
    if reflect:
        block.extend([
            f"- Reflect prompt: `{reflect_prompt or 'prompts/5_reflect.md'}`",
            f"- Reflection root: `{reflection_root or run_dir / 'reflections'}`",
            f"- Reflect category: `{reflect_category or 'iteration'}`",
        ])
    block.append(f"- Detached: {bool(detach)}")
    path.write_text(text + "\n" + "\n".join(block) + "\n", encoding="utf-8")


def _split_csv(values) -> list:
    """Normalize repeated comma-list CLI args into a de-duped list."""
    out = []
    seen = set()
    for raw in values or []:
        for part in str(raw).split(","):
            val = part.strip()
            if val and val not in seen:
                out.append(val)
                seen.add(val)
    return out


def _select_suite_tasks(suite: dict, only_specs) -> list:
    """Return suite tasks filtered by --only task id/template name."""
    wanted = _split_csv(only_specs)
    if not wanted:
        return list(suite["tasks"])
    selected = [
        t for t in suite["tasks"]
        if t.get("id") in wanted or t.get("template") in wanted
    ]
    matched = set()
    for t in selected:
        if t.get("id") in wanted:
            matched.add(t.get("id"))
        if t.get("template") in wanted:
            matched.add(t.get("template"))
    missing = [w for w in wanted if w not in matched]
    if missing:
        valid = sorted({x for t in suite["tasks"] for x in (t.get("id"), t.get("template")) if x})
        die(f"--only did not match suite task/template: {', '.join(missing)}. "
            f"Valid choices: {', '.join(valid)}")
    return selected




# `path: ...` lines in task.yaml that start with `..` (relative going up)
# don't resolve in bench tasks — the bench task lives at a different
# filesystem depth than the source, so `../datasets/...` becomes nonsense.
# Rewrite to absolute paths resolved against the SOURCE task dir; the source
# task.yaml itself is never modified.
_TASK_YAML_PATH_LINE = re.compile(
    r'(path:\s*)(["\']?)(\.\.[^"\'\s,}]+)(["\']?)'
)


def _rewrite_relative_data_paths(dest: Path, source_task: Path) -> list:
    """Rewrite `path: ../...` entries in dest/task.yaml to absolute paths
    resolved against the SOURCE task dir. Returns list of (relative, absolute)
    pairs that were rewritten.

    Source task.yaml is NOT modified — only the snapshot in the template.
    """
    task_yaml = dest / "task.yaml"
    if not task_yaml.exists():
        return []
    text = task_yaml.read_text(encoding="utf-8")
    rewrites = []

    def _sub(m):
        prefix, q1, rel, q2 = m.group(1), m.group(2), m.group(3), m.group(4)
        absolute = (source_task / rel).resolve()
        rewrites.append((rel, str(absolute)))
        return f'{prefix}{q1}{absolute}{q2}'

    new_text = _TASK_YAML_PATH_LINE.sub(_sub, text)
    if rewrites:
        task_yaml.write_text(new_text, encoding="utf-8")
    return rewrites


def _reference_text_as_round0_pipeline(ref_text: str, ext: str) -> str:
    """Convert a reference script into a round-0 pipeline script.

    Round 0 should call the same upstream function as reference, but it is not
    a byte-for-byte copy: pipeline/run executes from the pipeline/ directory and
    must write test outputs there, while reference writes cached ground truth.
    """
    text = ref_text.replace("[reference]", "[pipeline]")
    text = text.replace("reference.R", "pipeline/run.R")
    text = text.replace("reference.py", "pipeline/run.py")

    if ext == "R":
        text = re.sub(
            r"(?m)^(TASK_DIR\s*<-\s*)SCRIPT_DIR\s*$",
            r"\1dirname(SCRIPT_DIR)",
            text,
        )
        text = text.replace('"ZYME_REFERENCE_DIR"', '"ZYME_TEST_DIR"')
        text = re.sub(
            r'file\.path\(TASK_DIR,\s*"reference_outputs",\s*TIER\)',
            'file.path(SCRIPT_DIR, sprintf("output_%s", TIER))',
            text,
        )
        text = re.sub(
            r'file\.path\(TASK_DIR,\s*sprintf\("reference_output_%s",\s*TIER\)\)',
            'file.path(SCRIPT_DIR, sprintf("output_%s", TIER))',
            text,
        )
        text = text.replace(
            'file.path(TASK_DIR, "reference_output")',
            'file.path(SCRIPT_DIR, "output")',
        )
    elif ext == "py":
        text = re.sub(
            r"(?m)^(TASK_DIR\s*=\s*)SCRIPT_DIR\s*$",
            r"\1os.path.dirname(SCRIPT_DIR)",
            text,
        )
        text = text.replace('"ZYME_REFERENCE_DIR"', '"ZYME_TEST_DIR"')
        text = text.replace("'ZYME_REFERENCE_DIR'", "'ZYME_TEST_DIR'")
        text = text.replace(
            'os.path.join(TASK_DIR, "reference_outputs", TIER)',
            'os.path.join(SCRIPT_DIR, f"output_{TIER}")',
        )
        text = text.replace(
            "os.path.join(TASK_DIR, 'reference_outputs', TIER)",
            "os.path.join(SCRIPT_DIR, f'output_{TIER}')",
        )
        text = text.replace(
            'os.path.join(TASK_DIR, f"reference_output_{TIER}")',
            'os.path.join(SCRIPT_DIR, f"output_{TIER}")',
        )
        text = text.replace(
            "os.path.join(TASK_DIR, f'reference_output_{TIER}')",
            "os.path.join(SCRIPT_DIR, f'output_{TIER}')",
        )
        text = text.replace(
            'os.path.join(TASK_DIR, "reference_output")',
            'os.path.join(SCRIPT_DIR, "output")',
        )
        text = text.replace(
            "os.path.join(TASK_DIR, 'reference_output')",
            "os.path.join(SCRIPT_DIR, 'output')",
        )

    return text


def _has_active_install_override(text: str) -> bool:
    return re.search(r"(?m)^[^#\n]*install_override\(", text) is not None


def _reset_pipeline_to_reference(dest: Path) -> list:
    """For iterate-stage templates: reset pipeline/run.{R,py} to a round-0
    upstream baseline (i.e. no optimization has been applied yet).

    Necessary when the snapshot commit is mid-iterate or post-iterate
    (pipeline/run.{R,py} carries opts) — without this, the bench task starts
    from the source's converged state, biasing every iterate-prompt experiment.

    Returns list of (run_path, ref_path) pairs that were reset.
    """
    reset = []
    for ext in ("R", "py"):
        run_p = dest / "pipeline" / f"run.{ext}"
        ref_p = dest / f"reference.{ext}"
        if not (run_p.exists() and ref_p.exists()):
            continue
        run_text = run_p.read_text(encoding="utf-8")
        ref_text = ref_p.read_text(encoding="utf-8")
        if run_text != ref_text and not _has_active_install_override(run_text):
            continue
        pipeline_text = _reference_text_as_round0_pipeline(ref_text, ext)
        if run_text == pipeline_text:
            continue
        run_p.write_text(pipeline_text, encoding="utf-8")
        reset.append((f"pipeline/run.{ext}", f"reference.{ext}"))
    return reset


def _is_ood_tier_dirname(name: str) -> bool:
    """Convention: tier names starting with `ood_` (or equal to `ood`) belong
    to the phase-3 (validate_scaling) regime — out-of-distribution tiers added
    via `setup:` commits during scale validation, never used during iterate.
    """
    return name == "ood" or name.startswith("ood_")


def _copy_runtime_state(source: Path, dest: Path, stage: str) -> tuple:
    """Copy small per-replicate state dirs from source task into the template.

    For `stage='iterate'` templates, OOD subdirs under reference_outputs/ and
    legacy reference_output_ood_*/ dirs are skipped — those artifacts come
    from phase-3 validate_scaling, not the iterate regime we're benchmarking.
    Other stages copy everything.

    Returns (copied_names, skipped_ood_names) for the summary line.
    """
    copied = []
    skipped_ood = []
    skip_ood = (stage == "iterate")

    for name in _TEMPLATE_COPY_DIRS:
        src = source / name
        if not (src.exists() and src.is_dir()):
            continue
        if name == ".zyme":
            out_dir = dest / name
            out_dir.mkdir(exist_ok=True)
            any_copied = False
            for fname in _TEMPLATE_ZYME_COPY_FILES:
                fsrc = src / fname
                if fsrc.exists() and fsrc.is_file():
                    shutil.copy2(fsrc, out_dir / fname)
                    any_copied = True
            if any_copied:
                copied.append(name)
            else:
                out_dir.rmdir()
            continue
        if name == "reference_outputs" and skip_ood:
            # Selectively copy non-OOD tier subdirs only.
            (dest / name).mkdir(exist_ok=True)
            any_kept = False
            for tier_dir in sorted(src.iterdir()):
                if not tier_dir.is_dir():
                    continue
                if _is_ood_tier_dirname(tier_dir.name):
                    skipped_ood.append(f"reference_outputs/{tier_dir.name}")
                    continue
                shutil.copytree(tier_dir, dest / name / tier_dir.name)
                any_kept = True
            if any_kept:
                copied.append(name)
            else:
                # Empty parent dir is harmless but uninformative — drop it.
                (dest / name).rmdir()
        else:
            shutil.copytree(src, dest / name)
            copied.append(name)

    for pattern in _TEMPLATE_COPY_GLOBS:
        for src in sorted(source.glob(pattern)):
            if not src.is_dir() or (dest / src.name).exists():
                continue
            # Legacy layout: `reference_output_<tier>/`. Skip OOD tier names.
            if skip_ood and pattern == "reference_output_*":
                tier = src.name.removeprefix("reference_output_")
                if _is_ood_tier_dirname(tier):
                    skipped_ood.append(src.name)
                    continue
            shutil.copytree(src, dest / src.name)
            copied.append(src.name)
    return copied, skipped_ood


def _copy_runtime_files(source: Path, dest: Path) -> list:
    copied = []
    for rel in _TEMPLATE_COPY_FILES:
        src = source / rel
        if not src.exists() or not src.is_file():
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        copied.append(rel)
    return copied


def _copy_memory_state(source: Path, dest: Path) -> list:
    src = source / "memory"
    out = dest / "memory"
    if not src.is_dir():
        return []
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    copied = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            copied.append(str(path.relative_to(dest)))
    return copied




def _detect_symlink_sources(source: Path) -> dict:
    """Record absolute source paths for dirs that should be symlinked at
    bench-init time. Returns {name: abs_path_str}."""
    out = {}
    for name in _TEMPLATE_SYMLINK_DIRS:
        src = source / name
        if src.exists() and src.is_dir():
            out[name] = str(src.resolve())
    return out


_REPO_URL_PREFIXES = ("http://", "https://", "git@", "git://", "ssh://")


def _detect_upstream_clone(source: Path) -> dict | None:
    """Return clone metadata for the upstream_repo, or None if we can't.

    Records the clone URL + the upstream's current HEAD SHA so that
    `bench init` can re-clone fresh on any host without depending on the
    source task dir surviving. URL resolution preference:
      1. task.yaml's `target_repo:` field, if it's a URL
      2. `<source>/upstream_repo/.git`'s `remote.origin.url`
    Returns None when neither yields a URL (e.g. task was inited from a
    pure local path with no origin) — caller falls back to the legacy
    symlink behavior with a warning.
    """
    upstream = source / "upstream_repo"
    if not upstream.is_dir() or not (upstream / ".git").exists():
        return None

    clone_url = None
    task_yaml = source / "task.yaml"
    if task_yaml.exists():
        for line in task_yaml.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("target_repo:"):
                val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if val.startswith(_REPO_URL_PREFIXES):
                    clone_url = val
                break

    if clone_url is None:
        try:
            url = git("config", "--get", "remote.origin.url", cwd=upstream).strip()
        except SystemExit:
            url = ""
        if url.startswith(_REPO_URL_PREFIXES):
            clone_url = url

    if clone_url is None:
        return None

    sha = _git_head_sha(upstream)
    if not sha:
        return None

    return {"clone_url": clone_url, "commit": sha}


# Convention: per-task datasets are mirrored to elliotxie/autozyme-datasets
# under per_task/<name>/. register-template infers the HF coords from the
# source task's `data` symlink target. Shared / external datasets (not in
# per_task/) fall through to legacy symlink_sources.
_HF_DATASET_REPO = "elliotxie/autozyme-datasets"
_HF_DATASET_REPO_TYPE = "dataset"


def _detect_hf_data(source: Path) -> dict | None:
    """Infer HF dataset coords from the source task's `data` symlink target.

    Returns {"hf_repo", "hf_repo_type", "hf_path"} if the symlink resolves
    into `<workspace>/datasets/per_task/<name>/`, else None. Templates with
    HF coords are portable across hosts — bench init re-downloads from HF
    when the local copy is absent. Non-per_task data (shared/, external)
    keeps using symlink_sources for now.
    """
    data = source / "data"
    if not data.is_symlink():
        return None
    target = data.resolve()
    parts = target.parts
    try:
        idx = parts.index("per_task")
    except ValueError:
        return None
    if idx == 0 or parts[idx - 1] != "datasets":
        return None
    if idx + 1 >= len(parts):
        return None
    name = parts[idx + 1]
    return {
        "hf_repo": _HF_DATASET_REPO,
        "hf_repo_type": _HF_DATASET_REPO_TYPE,
        "hf_path": f"per_task/{name}",
    }


def _resolve_datasets_root() -> Path:
    """Where HF data lands on this host. AUTOZYME_DATASETS_ROOT wins; else
    `<workspace>/datasets/` (workspace = framework parent dir).

    FRAMEWORK_ROOT = .../autozyme-framework/autozyme_cli/zyme/, so three
    `.parent` walks land at the workspace root (matches `_workspace_root` and
    `_default_bench_runs_root`)."""
    env = os.environ.get("AUTOZYME_DATASETS_ROOT")
    if env:
        return Path(env).resolve()
    return (FRAMEWORK_ROOT.parent.parent.parent / "datasets").resolve()


def _hf_download(repo_id: str, repo_type: str, path_prefix: str,
                 local_root: Path) -> None:
    """Download all files under path_prefix from an HF repo into local_root.

    Mirrors the layout used by data.link.json snapshots: <local_root>/<path_prefix>/...
    Lets `hf download`'s own progress + file list stream straight to the terminal
    (no --quiet) so a stalled or slow download is visible. Raises
    subprocess.CalledProcessError on failure. Caller decides how to surface that.
    """
    local_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf", "download", repo_id,
        "--repo-type", repo_type,
        "--include", f"{path_prefix}/*",
        "--local-dir", str(local_root),
    ]
    subprocess.run(cmd, check=True)




def _rewrite_gitignore_for_template(dest: Path) -> None:
    """Rewrite the template's .gitignore so PromptLab git can track the dirs
    we explicitly hard-copied (.zyme/, memory/, reference_outputs/, etc.).
    Stashes the original at .gitignore.task so bench-init can restore the
    strict per-task semantics in scaffolded bench tasks.
    """
    gi = dest / ".gitignore"
    if not gi.exists():
        return
    original = gi.read_text(encoding="utf-8")
    (dest / ".gitignore.task").write_text(original, encoding="utf-8")

    drop = set(_TEMPLATE_GITIGNORE_DROP_PATTERNS)
    kept = []
    for line in original.splitlines():
        stripped = line.strip()
        # Drop both bare patterns and trailing-slash variants. Comments and
        # blank lines pass through.
        if stripped in drop:
            continue
        kept.append(line)
    header = (
        "# (template-context .gitignore — original task .gitignore stashed at\n"
        "#  .gitignore.task; bench init restores it in each replicate)\n"
    )
    gi.write_text(header + "\n".join(kept).rstrip() + "\n", encoding="utf-8")




def _find_init_commit(task_dir: Path) -> str:
    """Return the SHA of the first commit whose message starts with 'zyme init:'."""
    out = git("log", "--reverse", "--format=%H %s", cwd=task_dir)
    for line in out.splitlines():
        sha, _, msg = line.partition(" ")
        if msg.startswith("zyme init:"):
            return sha
    return None




def _find_post_init_commit(task_dir: Path) -> str:
    """Parent of the first tagged-iterate commit; HEAD if none.

    A tagged-iterate commit = first commit on HEAD's history whose message
    starts with `[conservative]`, `[algorithmic]`, or `[housekeeping]`.
    Walks `git log --reverse` from the root.
    """
    out = git("log", "--reverse", "--format=%H %s", cwd=task_dir)
    first_iter_commit = None
    for line in out.splitlines():
        sha, _, msg = line.partition(" ")
        if _ITERATE_TAG_RE.match(msg):
            first_iter_commit = sha
            break
    if first_iter_commit is None:
        return git("rev-parse", "HEAD", cwd=task_dir)
    parent = git("rev-parse", f"{first_iter_commit}^", cwd=task_dir)
    return parent or None




def _git_head_sha(repo_dir: Path) -> str:
    try:
        return git("rev-parse", "HEAD", cwd=repo_dir)
    except SystemExit:
        return ""




def cmd_bench_register_template(args):
    """Snapshot a task at a lifecycle commit into bench_templates/<name>/."""
    task_dir = Path(args.task_dir).resolve()
    if not (task_dir / "task.yaml").exists():
        die(f"not a zyme task (no task.yaml): {task_dir}")
    if not (task_dir / ".git").exists():
        die(f"not a git repo: {task_dir}")

    # --commit is an explicit escape hatch for tasks with broken/squashed
    # histories where neither --at init nor --at post_init can find a tagged
    # commit (e.g. a task that was reset to a single "Initialize workspace"
    # commit before iterate started). Resolves a sha (or "HEAD") via git
    # rev-parse and uses it directly. Pair with --reset-pipeline (auto when
    # stage=iterate) to ensure pipeline/run is at post-init state.
    if args.commit is not None:
        sha = git("rev-parse", args.commit, cwd=task_dir)
        if not sha:
            die(f"--commit '{args.commit}' did not resolve to a sha in {task_dir}")
    elif args.at == "init":
        sha = _find_init_commit(task_dir)
        if sha is None:
            die(f"no 'zyme init: ...' commit found in {task_dir}. "
                f"Pass --commit <sha-or-HEAD> to override.")
    elif args.at == "post_init":
        sha = _find_post_init_commit(task_dir)
        if sha is None:
            die(f"could not resolve post_init commit in {task_dir}. "
                f"Pass --commit <sha-or-HEAD> to override.")
    else:
        die(f"--at must be 'init' or 'post_init', got '{args.at}'")

    short = sha[:8]
    msg = git("log", "-1", "--format=%s", sha, cwd=task_dir)

    dest = _bench_template_path(args.stage, args.name)
    if dest.exists():
        if not args.force:
            die(f"template '{args.name}' already exists at {dest}. "
                f"Pass --force to overwrite.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # git archive | tar x. Streaming via subprocess pipe keeps the tarball
    # out of memory for large repos.
    info(f"git archive {short} ('{msg[:60]}') → {dest}")
    archive_proc = subprocess.Popen(
        ["git", "-C", str(task_dir), "archive", "--format=tar", sha],
        stdout=subprocess.PIPE,
    )
    extract_proc = subprocess.Popen(
        ["tar", "xf", "-", "-C", str(dest)],
        stdin=archive_proc.stdout,
    )
    archive_proc.stdout.close()  # let archive proc receive SIGPIPE if extract dies
    extract_rc = extract_proc.wait()
    archive_rc = archive_proc.wait()
    if archive_rc != 0 or extract_rc != 0:
        shutil.rmtree(dest, ignore_errors=True)
        die(f"git archive | tar failed: archive rc={archive_rc} extract rc={extract_rc}")

    # Rewrite relative `path: ../...` entries in the template's task.yaml to
    # absolute paths resolved against the SOURCE task dir. Bench tasks live at
    # a different filesystem depth than the source, so `../datasets/...` would
    # otherwise dangle. Source task.yaml is NOT modified — only the snapshot.
    path_rewrites = _rewrite_relative_data_paths(dest, task_dir)

    # For iterate-stage templates, ensure pipeline/run.{R,py} matches the
    # untouched reference.{R,py}. The git-archive snapshot may be mid-iterate
    # (when the user passes --commit, or when post_init heuristic falls back
    # to HEAD on broken histories), in which case pipeline/run carries opts
    # the bench agent shouldn't start with.
    pipeline_reset = []
    if args.stage == "iterate":
        pipeline_reset = _reset_pipeline_to_reference(dest)

    # Carry per-replicate state dirs (.zyme/baselines_stash.tsv, cached
    # reference outputs) — small enough to copy into the template, valuable
    # enough that bench tasks shouldn't have to regenerate them. For
    # stage='iterate', OOD tier outputs are skipped (phase-3 artifacts).
    copied_state, skipped_ood = _copy_runtime_state(task_dir, dest, args.stage)
    copied_files = _copy_runtime_files(task_dir, dest)

    # Init/iterate templates start as new optimization sessions, so their
    # memory should be fresh. Later stages resume the optimized task in a new
    # session and must inherit the iteration memory just like staying in the
    # same folder would.
    copied_memory = []
    if args.stage in _FRESH_MEMORY_STAGES:
        preserved_discoveries = None
        discoveries_path = dest / "memory" / "discoveries.md"
        if args.stage == "iterate" and not pipeline_reset and discoveries_path.exists():
            text = discoveries_path.read_text(encoding="utf-8")
            if "## DISCOVERY:" in text:
                preserved_discoveries = text
        _write_memory_skeleton(dest, task_dir.name)
        if preserved_discoveries is not None:
            discoveries_path.write_text(preserved_discoveries, encoding="utf-8")
    else:
        copied_memory = _copy_memory_state(task_dir, dest)
        if not copied_memory:
            _write_memory_skeleton(dest, task_dir.name)

    # Record symlink sources for large read-only data dirs (data/, legacy
    # upstream/). Bench init creates symlinks back to these instead of copying.
    symlink_sources = _detect_symlink_sources(task_dir)

    # Record `data` as an HF-portable block when the source's data symlink
    # points into `<workspace>/datasets/per_task/<name>/` (the HF-mirrored
    # convention). Drop it from symlink_sources so bench init treats the HF
    # block as authoritative — local path on this host is still tried first.
    hf_data = _detect_hf_data(task_dir)
    if hf_data is not None:
        symlink_sources.pop("data", None)

    # Record upstream_repo as a clone URL + commit SHA so bench init can
    # `git clone` fresh on any host. Stale local path → portability bug;
    # the source task dir can be deleted after registration once this is set.
    upstream_clone = _detect_upstream_clone(task_dir)
    if upstream_clone is None and (task_dir / "upstream_repo").is_dir():
        # Fall back to recording the local path under symlink_sources so old
        # bench-init logic still finds it. Warn so the user knows the template
        # is host-bound until they wire up an origin URL.
        symlink_sources["upstream_repo"] = str((task_dir / "upstream_repo").resolve())
        info(
            "WARN: upstream_repo has no clone URL (no target_repo URL in "
            "task.yaml and no git origin). Recording local path as symlink "
            "source — template is NOT portable. Add a `git remote add origin "
            "<url>` in upstream_repo/ and re-register to fix."
        )

    # Rewrite the .gitignore so PromptLab git tracks the state dirs we just
    # hard-copied. Strict per-task .gitignore is stashed at .gitignore.task
    # for bench-init to restore in each replicate.
    _rewrite_gitignore_for_template(dest)

    # Record provenance. NOT a dataclass — flat YAML so users can grep / edit.
    meta = {
        "name": args.name,
        "stage": args.stage,
        "source_task": str(task_dir),
        "source_task_name": task_dir.name,
        "source_commit": sha,
        "source_commit_short": short,
        "source_commit_message": msg,
        # When --commit was passed, the lifecycle category is unknown — record
        # that explicitly rather than misreporting --at's default.
        "restore_point": (f"explicit_commit:{args.commit}"
                          if args.commit is not None else args.at),
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }
    framework_repo = FRAMEWORK_ROOT.parent.parent  # autozyme-framework/
    fw_sha = _git_head_sha(framework_repo)
    if fw_sha:
        meta["framework_sha"] = fw_sha
    if copied_state:
        meta["copied_state_dirs"] = copied_state
    if copied_files:
        meta["copied_state_files"] = copied_files
    if copied_memory:
        meta["copied_memory_files"] = copied_memory
    meta["memory_policy"] = (
        "fresh_skeleton" if args.stage in _FRESH_MEMORY_STAGES
        else "inherit_source_task_memory"
    )
    if symlink_sources:
        meta["symlink_sources"] = symlink_sources
    if hf_data is not None:
        meta["data"] = hf_data
    if upstream_clone is not None:
        meta["upstream_repo"] = upstream_clone
    (dest / "bench_template.yaml").write_text(_emit_simple_yaml(meta), encoding="utf-8")

    # Surface what landed for sanity-checking
    file_count = sum(1 for p in dest.rglob("*") if p.is_file())
    info(f"template '{args.name}' registered ({file_count} files)")
    info(f"  stage:  {args.stage}")
    info(f"  source: {task_dir}")
    info(f"  commit: {short} ('{msg[:60]}')")
    info(f"  restore point: {meta['restore_point']}")
    info(f"  → {dest}")
    if copied_state:
        info(f"  copied state: {' '.join(copied_state)}")
    if copied_memory:
        info(f"  copied memory: {len(copied_memory)} file(s)")
    if skipped_ood:
        info(f"  skipped (phase-3 OOD, stage='{args.stage}'): {' '.join(skipped_ood)}")
    if pipeline_reset:
        info(f"  reset pipeline → reference (post-init simulation):")
        for run_p, ref_p in pipeline_reset:
            info(f"    {run_p} ← {ref_p}")
    if path_rewrites:
        info(f"  rewrote {len(path_rewrites)} relative data path(s) → absolute (template only):")
        for rel, absolute in path_rewrites:
            info(f"    {rel}  →  {absolute}")
    if symlink_sources:
        info(f"  symlink sources (created at bench-init time):")
        for n, p in symlink_sources.items():
            info(f"    {n} -> {p}")
    if hf_data is not None:
        info(f"  data (HF-portable; downloaded at bench-init time if missing):")
        info(f"    {hf_data['hf_repo']} :: {hf_data['hf_path']}")
    if upstream_clone is not None:
        info(f"  upstream_repo (cloned at bench-init time):")
        info(f"    {upstream_clone['clone_url']} @ {upstream_clone['commit'][:12]}")
    notes = []
    if args.at == "post_init" and ".zyme" not in copied_state:
        notes.append(
            "no .zyme/ in source task — baselines won't carry over. "
            "After `zyme bench init`, run `zyme reference --tier <t>` in each "
            "bench task before iterating."
        )
    for n in notes:
        info(f"  note: {n}")




def _read_suite_yaml(path: Path) -> dict:
    """Parse a bench-suite manifest. Supports flat key:value plus a top-level
    `tasks:` block of `- key: value` list-of-mappings."""
    from zyme import registry
    text = path.read_text(encoding="utf-8")
    out: dict = {}
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        # Top-level key (no leading whitespace)
        if line[:1] != " " and line[:1] != "-":
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # Could be a nested mapping or a `- ` list. Look ahead.
                # Skip blank lines.
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                if j < n and lines[j].lstrip().startswith("- "):
                    # List of mappings.
                    items = []
                    cur: dict = None
                    i = j
                    while i < n:
                        nxt = lines[i]
                        ns = nxt.lstrip()
                        if not nxt.strip() or ns.startswith("#"):
                            i += 1
                            continue
                        # End of list when we hit a non-indented, non-dash line
                        leading = len(nxt) - len(ns)
                        if leading == 0 and not ns.startswith("- "):
                            break
                        if ns.startswith("- "):
                            if cur is not None:
                                items.append(cur)
                            cur = {}
                            inner = ns[2:]  # strip "- "
                            ik, _, iv = inner.partition(":")
                            cur[ik.strip()] = registry._parse_scalar(iv.strip())
                            i += 1
                        else:
                            # continuation of current item
                            ck, _, cv = ns.partition(":")
                            if cur is None:
                                cur = {}
                            cur[ck.strip()] = registry._parse_scalar(cv.strip())
                            i += 1
                    if cur is not None:
                        items.append(cur)
                    out[key] = items
                    continue
                else:
                    # Empty mapping
                    out[key] = {}
                    i = j
                    continue
            else:
                out[key] = registry._parse_scalar(val)
                i += 1
                continue
        i += 1
    return out




def _bench_suite_path(suite_id: str) -> Path:
    return _bench_suites_root() / f"{suite_id}.yaml"




def _validate_suite(suite: dict, suite_id: str) -> None:
    for k in ("id", "stage", "prompt_slot", "field", "tasks"):
        if k not in suite:
            die(f"suite '{suite_id}' missing required key: {k}")
    if not isinstance(suite["tasks"], list) or not suite["tasks"]:
        die(f"suite '{suite_id}' has empty or malformed `tasks:` list")
    for t in suite["tasks"]:
        for k in ("id", "template"):
            if k not in t:
                die(f"suite '{suite_id}' task entry missing '{k}': {t}")


def _parse_flow_mapping(text: str) -> dict:
    """Parse the simple `{key: value, ...}` flow mappings used in task.yaml."""
    from zyme import registry
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    out = {}
    parts = []
    buf = []
    quote = None
    escape = False
    for ch in s:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\" and quote:
            buf.append(ch)
            escape = True
            continue
        if ch in ("'", '"'):
            if quote == ch:
                quote = None
            elif quote is None:
                quote = ch
            buf.append(ch)
            continue
        if ch == "," and quote is None:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    for part in parts:
        if not part or ":" not in part:
            continue
        k, _, v = part.partition(":")
        out[k.strip()] = registry._parse_scalar(v.strip())
    return out


def _parse_task_yaml_datasets(task_yaml: Path) -> list:
    """Extract dataset tier/name/path triples from task.yaml.

    Supports the limited task.yaml shapes zyme emits: flow-style rows
    (`- {tier: tiny, path: ...}`) and small block mappings.
    """
    from zyme import registry
    if not task_yaml.exists():
        return []
    lines = task_yaml.read_text(encoding="utf-8").splitlines()
    datasets = []
    in_block = False
    cur = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        leading = len(raw) - len(raw.lstrip())
        if not in_block:
            if leading == 0 and stripped == "datasets:":
                in_block = True
            continue
        if leading == 0 and not stripped.startswith("-"):
            break
        if stripped.startswith("- {"):
            if cur:
                datasets.append(cur)
                cur = None
            datasets.append(_parse_flow_mapping(stripped[2:].strip()))
            continue
        if stripped.startswith("- "):
            if cur:
                datasets.append(cur)
            cur = {}
            inner = stripped[2:].strip()
            if inner and ":" in inner:
                k, _, v = inner.partition(":")
                cur[k.strip()] = registry._parse_scalar(v.strip())
            continue
        if cur is not None and ":" in stripped:
            k, _, v = stripped.partition(":")
            cur[k.strip()] = registry._parse_scalar(v.strip())
    if cur:
        datasets.append(cur)
    return datasets


def _resolve_template_dataset_path(template_dir: Path, dataset_path: str, meta: dict) -> Path:
    """Resolve a task.yaml dataset path in template context."""
    p = Path(str(dataset_path))
    if p.is_absolute():
        return p
    rel = str(dataset_path).strip()
    if rel.startswith("./"):
        rel = rel[2:]
    parts = Path(rel).parts
    symlink_sources = meta.get("symlink_sources") or {}
    if parts and parts[0] in symlink_sources:
        return Path(symlink_sources[parts[0]]).joinpath(*parts[1:])
    # HF-portable `data:` block — resolve `data/foo.rds` against the local
    # download destination `<datasets_root>/<hf_path>/foo.rds`.
    hf_data = meta.get("data")
    if hf_data and parts and parts[0] == "data":
        local_root = _resolve_datasets_root() / hf_data.get("hf_path", "")
        return local_root.joinpath(*parts[1:])
    return template_dir / rel


def _reference_output_dir_for(template_dir: Path, tier: str) -> Path:
    current = template_dir / "reference_outputs" / tier
    if current.exists():
        return current
    legacy = template_dir / f"reference_output_{tier}"
    if legacy.exists():
        return legacy
    return current


def _template_check_lines(suite: dict, task: dict, template_dir: Path) -> tuple:
    """Return (errors, warnings) for one template in a suite context."""
    errors = []
    warnings = []
    manifest = template_dir / "bench_template.yaml"
    if not manifest.exists():
        errors.append(f"missing bench_template.yaml at {template_dir}")
        return errors, warnings
    meta = _read_simple_yaml(manifest)
    if meta.get("stage") and meta.get("stage") != suite["stage"]:
        errors.append(f"manifest stage={meta.get('stage')} but suite stage={suite['stage']}")

    task_yaml = template_dir / "task.yaml"
    if not task_yaml.exists():
        errors.append("missing task.yaml")

    refs = [p for p in (template_dir / "reference.R", template_dir / "reference.py") if p.exists()]
    if not refs:
        errors.append("missing reference.R/reference.py")

    if suite["stage"] == "iterate":
        for ext in ("R", "py"):
            ref_p = template_dir / f"reference.{ext}"
            run_p = template_dir / "pipeline" / f"run.{ext}"
            if ref_p.exists() or run_p.exists():
                if not ref_p.exists():
                    errors.append(f"pipeline/run.{ext} exists but reference.{ext} is missing")
                elif not run_p.exists():
                    errors.append(f"reference.{ext} exists but pipeline/run.{ext} is missing")
                else:
                    run_text = run_p.read_text(encoding="utf-8")
                    ref_text = ref_p.read_text(encoding="utf-8")
                    if run_text == ref_text:
                        errors.append(
                            f"pipeline/run.{ext} is a literal reference copy; "
                            "round-0 pipeline must write to pipeline output dirs"
                        )
                    if re.search(r"(?m)^[^#\n]*install_override\(", run_text):
                        errors.append(
                            f"pipeline/run.{ext} contains an active install_override; "
                            "iterate template starts optimized"
                        )
        for polluted in (".zyme/best.ref", ".zyme/round.counter", ".zyme/audit.jsonl",
                         ".zyme/edit_audit.log", ".zyme/housekeeping.json",
                         ".zyme/verify_probe.cache"):
            if (template_dir / polluted).exists():
                errors.append(f"template carries runtime state {polluted}")
        pipeline_dir = template_dir / "pipeline"
        output_dirs = sorted(pipeline_dir.glob("output_*")) if pipeline_dir.exists() else []
        if output_dirs:
            errors.append("template carries pipeline output dirs: " +
                          ", ".join(p.relative_to(template_dir).as_posix() for p in output_dirs))

    for name, src in (meta.get("symlink_sources") or {}).items():
        if not Path(src).exists():
            errors.append(f"symlink source missing: {name} -> {src}")

    ur = meta.get("upstream_repo")
    if ur is not None:
        if not isinstance(ur, dict):
            errors.append(f"upstream_repo must be a mapping, got {type(ur).__name__}")
        else:
            if not ur.get("clone_url"):
                errors.append("upstream_repo block missing clone_url")
            if not ur.get("commit"):
                errors.append("upstream_repo block missing commit (need a pinned SHA)")

    hd = meta.get("data")
    if hd is not None:
        if not isinstance(hd, dict):
            errors.append(f"data block must be a mapping, got {type(hd).__name__}")
        else:
            for key in ("hf_repo", "hf_repo_type", "hf_path"):
                if not hd.get(key):
                    errors.append(f"data block missing {key}")

    datasets = _parse_task_yaml_datasets(task_yaml)
    if task_yaml.exists() and not datasets:
        warnings.append("no datasets parsed from task.yaml")
    for ds in datasets:
        tier = str(ds.get("tier") or "?")
        if suite["stage"] == "iterate" and _is_ood_tier_dirname(tier):
            continue
        dpath = ds.get("path")
        if not dpath:
            errors.append(f"dataset {tier} missing path")
            continue
        resolved = _resolve_template_dataset_path(template_dir, str(dpath), meta)
        if not resolved.exists():
            errors.append(f"dataset {tier} path missing: {dpath} -> {resolved}")
        ref_dir = _reference_output_dir_for(template_dir, tier)
        if not ref_dir.exists():
            errors.append(f"reference output missing for tier {tier}: {ref_dir}")
        elif not any(p.is_file() for p in ref_dir.iterdir()):
            errors.append(f"reference output empty for tier {tier}: {ref_dir}")

    return errors, warnings


def cmd_bench_doctor(args):
    """Validate templates and data sources for a benchmark suite."""
    suite_path = _bench_suite_path(args.suite_id)
    if not suite_path.exists():
        die(f"suite manifest not found: {suite_path}. "
            f"Available: {sorted(p.stem for p in _bench_suites_root().glob('*.yaml'))}")
    suite = _read_suite_yaml(suite_path)
    _validate_suite(suite, args.suite_id)
    tasks = _select_suite_tasks(suite, args.only)

    print(f"# bench doctor: {suite['id']}")
    print(f"stage={suite['stage']} field={suite['field']} slot={suite['prompt_slot']}")
    print(f"templates_root={_bench_templates_root()}")
    print()

    any_errors = False
    for t in tasks:
        tpath = _bench_template_path(suite["stage"], t["template"])
        label = f"{t['id']} (template={t['template']})"
        if not tpath.exists():
            any_errors = True
            print(f"FAIL {label}")
            print(f"  - missing template dir: {tpath}")
            continue
        errors, warnings = _template_check_lines(suite, t, tpath)
        if errors:
            any_errors = True
            print(f"FAIL {label}")
            for msg in errors:
                print(f"  - {msg}")
            for msg in warnings:
                print(f"  ! {msg}")
        elif warnings:
            print(f"WARN {label}")
            for msg in warnings:
                print(f"  ! {msg}")
        else:
            print(f"OK   {label}")

    print()
    if any_errors:
        print("doctor found blocking issues")
        sys.exit(1)
    print("doctor passed")




def cmd_bench_list_templates(args):
    """List registered bench templates, grouped by stage."""
    root = _bench_templates_root()
    if not root.exists():
        info(f"(no templates registered yet at {root})")
        return

    requested_stage = getattr(args, "stage", None)

    by_stage = {}
    for stage_dir in sorted(root.iterdir()):
        if not stage_dir.is_dir():
            continue
        # Skip a legacy template that lives directly under root (no stage subdir).
        # Surfaced separately below so the user notices and migrates.
        if (stage_dir / "bench_template.yaml").exists():
            by_stage.setdefault("(unstaged-legacy)", []).append(
                (stage_dir.name, _read_simple_yaml(stage_dir / "bench_template.yaml"))
            )
            continue
        if requested_stage and stage_dir.name != requested_stage:
            continue
        for entry in sorted(stage_dir.iterdir()):
            mp = entry / "bench_template.yaml"
            if entry.is_dir() and mp.exists():
                by_stage.setdefault(stage_dir.name, []).append(
                    (entry.name, _read_simple_yaml(mp))
                )

    if not by_stage:
        msg = f"(no templates registered yet at {root}"
        if requested_stage:
            msg += f" for stage '{requested_stage}'"
        info(msg + ")")
        return

    total = 0
    for stage_name in sorted(by_stage):
        rows = by_stage[stage_name]
        print(f"\n=== stage: {stage_name} ({len(rows)} template(s)) ===")
        print(f"{'name':<24} {'restore_point':<12} {'source_task':<32} {'commit':<10} registered_at")
        print("-" * 110)
        for name, m in rows:
            short = m.get("source_commit_short")
            if not isinstance(short, str):
                short = m.get("source_commit") or short
            print(f"{name:<24} {m.get('restore_point', '?'):<12} "
                  f"{(m.get('source_task_name') or '')[:32]:<32} "
                  f"{str(short or '')[:10]:<10} "
                  f"{m.get('registered_at', '')}")
        total += len(rows)
    print()
    print(f"({total} total template(s) under {root})")


def _bench_task_progress(task_dir: Path) -> tuple:
    """Return (has_results, decisions, last_status) for a bench task."""
    results = task_dir / "results.tsv"
    if not results.exists():
        return False, 0, "-"
    lines = [ln for ln in results.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) <= 1:
        return True, 0, "-"
    header = lines[0].split("\t")
    try:
        status_idx = header.index("status")
    except ValueError:
        status_idx = None
    decisions = 0
    last_status = "-"
    for ln in lines[1:]:
        parts = ln.split("\t")
        status = parts[status_idx] if status_idx is not None and status_idx < len(parts) else ""
        if status and status != "baseline":
            decisions += 1
            last_status = status
    return True, decisions, last_status


_TERMINAL_DECISION_STATUSES = {"keep", "discard", "crash", "rollback", "oom"}
_BEST_SEEN_STATUSES = {"keep", "pending", "rerun"}


def _bench_task_progress_detail(task_dir: Path) -> dict:
    """Summarize one bench task from results.tsv for live monitoring."""
    out = {
        "has_results": False,
        "decision_rounds": 0,
        "pending_rounds": 0,
        "last_round": None,
        "last_status": "-",
        "best_keep_pct": None,
        "best_seen_pct": None,
    }
    results = task_dir / "results.tsv"
    if not results.exists():
        return out
    out["has_results"] = True
    try:
        lines = [ln for ln in results.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return out
    if len(lines) <= 1:
        return out

    header = lines[0].split("\t")

    def idx(name: str, default=None):
        try:
            return header.index(name)
        except ValueError:
            return default

    round_idx = idx("round", 0)
    status_idx = idx("status", 6)
    speedup_idx = idx("speedup_pct", 4)
    phase_idx = idx("phase", None)
    completed = set()
    pending = set()
    best_keep = None
    best_seen = None

    for ln in lines[1:]:
        parts = ln.split("\t")
        if round_idx is None or round_idx >= len(parts):
            continue
        try:
            round_num = int(parts[round_idx].strip())
        except ValueError:
            continue
        if round_num < 1:
            continue
        phase = "optimize"
        if phase_idx is not None and phase_idx < len(parts) and parts[phase_idx].strip():
            phase = parts[phase_idx].strip()
        if phase != "optimize":
            continue
        status = parts[status_idx].strip() if status_idx is not None and status_idx < len(parts) else ""
        speedup = None
        if speedup_idx is not None and speedup_idx < len(parts):
            try:
                speedup = float(parts[speedup_idx])
            except ValueError:
                speedup = None

        out["last_round"] = round_num
        out["last_status"] = status or "-"
        if status in _TERMINAL_DECISION_STATUSES:
            completed.add(round_num)
        elif status == "pending" or not status:
            pending.add(round_num)
        if speedup is not None and status == "keep":
            best_keep = speedup if best_keep is None else max(best_keep, speedup)
        if speedup is not None and status in _BEST_SEEN_STATUSES:
            best_seen = speedup if best_seen is None else max(best_seen, speedup)

    out["decision_rounds"] = len(completed)
    out["pending_rounds"] = len(pending - completed)
    out["best_keep_pct"] = best_keep
    out["best_seen_pct"] = best_seen
    return out


def _fmt_speedup_pct(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.1f}%"


def _dispatch_task_states(run_dir: Path) -> dict:
    """Return {task_name: state_row} for a bench run dispatch, if present."""
    try:
        from zyme.dispatch.state import pid_alive, read_state, state_path
    except Exception:
        return {}
    state = read_state(state_path(run_dir))
    if not state:
        return {}
    pid = state.get("master_pid")
    alive = pid is not None and pid_alive(pid)
    out = {}
    for task in state.get("queue") or []:
        row = dict(task)
        row["dispatch_alive"] = alive
        row["agent"] = state.get("agent", "")
        row["model"] = task.get("actual_model") or state.get("model", "")
        out[str(task.get("name") or "")] = row
    return out


def _dispatch_summary(task_states: dict) -> str:
    if not task_states:
        return "not_started"
    counts = {}
    alive = False
    for task in task_states.values():
        status = task.get("status") or "?"
        counts[status] = counts.get(status, 0) + 1
        alive = alive or bool(task.get("dispatch_alive"))
    order = ["running", "resource_wait", "pending", "done", "failed", "stopped"]
    parts = [f"{counts[k]} {k}" for k in order if counts.get(k)]
    if not parts:
        parts = [f"{v} {k}" for k, v in sorted(counts.items())]
    suffix = "" if alive else " stale"
    return ", ".join(parts) + suffix


def _filter_task_dirs_for_status(task_dirs: list[Path], only_specs=None) -> list[Path]:
    wanted = set(_split_csv(only_specs))
    if not wanted:
        return task_dirs
    out = []
    for task_dir in task_dirs:
        meta = _read_simple_yaml(task_dir / ZYME_META_FILENAME)
        keys = {
            task_dir.name,
            str(meta.get("bench_task") or ""),
            str(meta.get("template_name") or ""),
        }
        if keys & wanted:
            out.append(task_dir)
    return out


def _resolve_bench_run(run_ref: str, root: Path = None) -> Path:
    """Resolve a bench run by absolute/relative path or name under bench_runs."""
    raw = Path(run_ref)
    candidates = []
    if raw.is_absolute() or "/" in run_ref:
        candidates.append(raw)
    else:
        runs_root = Path(root).resolve() if root else _default_bench_runs_root()
        candidates.append(runs_root / run_ref)
    for cand in candidates:
        p = cand.resolve()
        if (p / "bench_manifest.yaml").is_file():
            return p
    tried = ", ".join(str(c.resolve()) for c in candidates)
    die(f"bench run not found: {run_ref} (looked for bench_manifest.yaml at {tried})")


def _prompt_file_for_bench_run(run_dir: Path, manifest: dict) -> str:
    """Return prompt path relative to each bench task dir."""
    from zyme import registry
    prompt_id = manifest.get("prompt_id")
    if prompt_id:
        try:
            found = registry.find_snapshot(FRAMEWORK_ROOT, prompt_id)
        except ValueError:
            found = None
        if found is not None:
            _f, _s, _pid, card = found
            filename = Path(card.get("source_path") or "").name
            if filename:
                return f"prompts/{filename}"

    slot = manifest.get("prompt_slot")
    first_task = next((p for p in sorted(run_dir.iterdir())
                       if p.is_dir() and (p / ZYME_META_FILENAME).exists()), None)
    if first_task is not None and slot:
        matches = sorted((first_task / "prompts").glob(f"*_{slot}.md"))
        if matches:
            return f"prompts/{matches[0].name}"
    die(f"could not resolve prompt file for bench run {run_dir}")


def _reflect_category_for_stage(stage: str | None) -> str:
    value = (stage or "").strip().lower()
    if value in ("init", "initialization", "transfer_init"):
        return "initialization"
    if value in ("iterate", "iteration", "memory"):
        return "iteration"
    if value in ("scaling", "validate_scaling", "expand_scaling"):
        return "scaling"
    if value in ("package", "packaging"):
        return "packaging"
    return value or "iteration"


def _bench_run_task_dirs(run_dir: Path, only_specs=None) -> list[dict]:
    """Return dispatch task dicts for one bench run."""
    wanted = _split_csv(only_specs)
    out = []
    matched = set()
    for task_dir in sorted(p for p in run_dir.iterdir()
                           if p.is_dir() and (p / ZYME_META_FILENAME).exists()):
        meta = _read_simple_yaml(task_dir / ZYME_META_FILENAME)
        keys = {
            task_dir.name,
            str(meta.get("bench_task") or ""),
            str(meta.get("template_name") or ""),
        }
        if wanted and not (keys & set(wanted)):
            continue
        matched.update(keys & set(wanted))
        out.append({
            "name": task_dir.name,
            "task_dir": str(task_dir.resolve()),
        })
    missing = [w for w in wanted if w not in matched]
    if missing:
        valid = []
        for task_dir in sorted(p for p in run_dir.iterdir()
                               if p.is_dir() and (p / ZYME_META_FILENAME).exists()):
            meta = _read_simple_yaml(task_dir / ZYME_META_FILENAME)
            valid.append(task_dir.name)
            if meta.get("bench_task"):
                valid.append(str(meta["bench_task"]))
            if meta.get("template_name"):
                valid.append(str(meta["template_name"]))
        die(f"--only did not match bench task/template: {', '.join(missing)}. "
            f"Valid choices: {', '.join(sorted(set(valid)))}")
    if not out:
        die(f"no bench task dirs found in {run_dir}")
    return out


def cmd_bench_start(args):
    """Start agent workers for an existing bench run."""
    from zyme.dispatch import (
        DEFAULT_CODEX_MODEL, DEFAULT_CURSOR_MODEL, DEFAULT_DISK_FLOOR_FALLBACK_GB,
        DEFAULT_RAM_FLOOR_GB, DEFAULT_REFLECT_PROMPT,
        find_agent_binary, start_dispatch,
    )
    from zyme.dispatch.resources import parse_size_gb
    from zyme.dispatch.state import (
        master_log_path, pid_alive, pid_path, read_pid, state_path,
    )

    run_dir = _resolve_bench_run(args.run, Path(args.root) if args.root else None)
    manifest = _read_simple_yaml(run_dir / "bench_manifest.yaml")
    prompt_rel = args.prompt or _prompt_file_for_bench_run(run_dir, manifest)
    reflect_prompt = args.reflect_prompt or DEFAULT_REFLECT_PROMPT
    reflect_category = args.reflect_category or _reflect_category_for_stage(manifest.get("prompt_slot"))
    reflection_root = run_dir / "reflections"
    tasks = _bench_run_task_dirs(run_dir, args.only)

    missing_prompt = []
    for t in tasks:
        p = Path(t["task_dir"]) / prompt_rel
        if not p.is_file():
            missing_prompt.append(f"{t['name']}: {p}")
    if missing_prompt:
        die("prompt file missing for:\n  " + "\n  ".join(missing_prompt))
    if args.reflect:
        missing_reflect_prompt = []
        for t in tasks:
            p = Path(t["task_dir"]) / reflect_prompt
            if not p.is_file():
                missing_reflect_prompt.append(f"{t['name']}: {p}")
        if missing_reflect_prompt:
            die("reflect prompt file missing for:\n  " + "\n  ".join(missing_reflect_prompt))

    prior = read_pid(pid_path(run_dir))
    if prior is not None and pid_alive(prior):
        die(f"another bench dispatch is already running in {run_dir} "
            f"(PID {prior}). Run `zyme dispatch stop --workspace {run_dir}` first.")

    if args.agent == "auto":
        from zyme.dispatch.master import detect_agent_binary
        args.agent, _ = detect_agent_binary()
    try:
        find_agent_binary(args.agent)
    except RuntimeError as e:
        die(str(e))
    model = args.model
    if args.agent == "claude" and not model:
        model = "claude-opus-4-7[1m]"
    if args.agent == "codex" and not model:
        model = DEFAULT_CODEX_MODEL
    if args.agent == "cursor" and not model:
        model = DEFAULT_CURSOR_MODEL

    try:
        parsed_ram_floor = parse_size_gb(args.ram_floor)
        ram_floor_gb = DEFAULT_RAM_FLOOR_GB if parsed_ram_floor is None else parsed_ram_floor
    except ValueError as e:
        die(str(e))
    if args.disk_floor == "auto":
        disk_floor_gb = None
    else:
        try:
            disk_floor_gb = parse_size_gb(args.disk_floor)
        except ValueError as e:
            die(str(e))

    if args.dry_run:
        print(f"bench run : {run_dir}")
        print(f"suite     : {manifest.get('bench_suite', '')}")
        print(f"prompt id : {manifest.get('prompt_id', '')}")
        print(f"prompt    : {prompt_rel}")
        print(f"agent     : {args.agent}")
        print(f"model     : {model or '(agent default)'} (effort={args.effort})")
        print(f"max rounds: {args.max_rounds or '(none)'}")
        print(f"force mode: {bool(args.force_mode)}")
        print(f"reflect   : {bool(args.reflect)}"
              + (f" ({reflect_prompt}, category={reflect_category})" if args.reflect else ""))
        print(f"ram floor : {ram_floor_gb:.1f} GB")
        print(f"disk floor: "
              + ("auto (per-task estimate ×2, fallback "
                 f"{DEFAULT_DISK_FLOOR_FALLBACK_GB} GB)"
                 if disk_floor_gb is None else f"{disk_floor_gb:.1f} GB"))
        print(f"detach    : {args.detach}")
        print(f"queue     ({len(tasks)} tasks):")
        for i, t in enumerate(tasks, 1):
            print(f"  {i:2d}. {t['name']}  ({t['task_dir']})")
        return

    _append_experiment_dispatch_doc(
        run_dir,
        agent=args.agent,
        model=model,
        effort=args.effort,
        prompt_rel=prompt_rel,
        tasks=tasks,
        max_rounds=args.max_rounds,
        force_mode=args.force_mode,
        reflect=args.reflect,
        reflect_prompt=reflect_prompt,
        reflection_root=reflection_root,
        reflect_category=reflect_category,
        detach=args.detach,
    )

    try:
        pid = start_dispatch(
            workspace=run_dir,
            tasks=tasks,
            prompt=prompt_rel,
            agent=args.agent,
            model=model,
            effort=args.effort,
            ram_floor_gb=ram_floor_gb,
            disk_floor_gb=disk_floor_gb,
            detach=args.detach,
            stall_threshold_s=args.stall_threshold,
            max_rounds=args.max_rounds,
            force_mode=args.force_mode,
            reflect=args.reflect,
            reflect_prompt=reflect_prompt,
            reflection_root=reflection_root,
            reflect_category=reflect_category,
        )
    except RuntimeError as e:
        die(str(e))

    if args.detach:
        info(f"bench dispatch started (PID {pid}, daemonized)")
        info(f"  run   : {run_dir}")
        info(f"  state : {state_path(run_dir)}")
        info(f"  log   : {master_log_path(run_dir)}")
        info(f"  watch : zyme dispatch status --workspace {run_dir}")
        info(f"  stop  : zyme dispatch stop --workspace {run_dir}")


def cmd_bench_usage(args):
    """Print token/cost telemetry for a bench run's dispatch workspace."""
    import json
    from zyme.dispatch import collect_usage, render_usage

    run_dir = _resolve_bench_run(args.run, Path(args.root) if args.root else None)
    summary = collect_usage(
        run_dir,
        token_budget=args.token_budget,
        budget_basis=args.budget_basis,
        price_model=args.price_model,
    )
    if args.json_output:
        print(json.dumps(summary, indent=2, sort_keys=False))
    else:
        print(render_usage(summary))


def cmd_bench_prices(args):
    """Print the built-in model price registry."""
    import json
    from zyme.dispatch import list_prices, render_price_table
    if args.json_output:
        print(json.dumps(list_prices(), indent=2, sort_keys=False))
    else:
        print(render_price_table())


def _iter_bench_runs(root: Path, suite_id: str = None):
    if not root.exists():
        return
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        manifest = entry / "bench_manifest.yaml"
        if not manifest.exists():
            continue
        m = _read_simple_yaml(manifest)
        if suite_id and m.get("bench_suite") != suite_id:
            continue
        task_dirs = [p for p in sorted(entry.iterdir())
                     if p.is_dir() and (p / ZYME_META_FILENAME).exists()]
        yield entry, m, task_dirs


def cmd_bench_status(args):
    """Show prompt snapshots, template readiness, and bench-run progress."""
    from zyme import registry

    suite = None
    if args.suite_id:
        suite_path = _bench_suite_path(args.suite_id)
        if not suite_path.exists():
            die(f"suite manifest not found: {suite_path}. "
                f"Available: {sorted(p.stem for p in _bench_suites_root().glob('*.yaml'))}")
        suite = _read_suite_yaml(suite_path)
        _validate_suite(suite, args.suite_id)

    print("# bench status")
    print(f"templates_root={_bench_templates_root()}")
    print(f"bench_runs_root={Path(args.root).resolve() if args.root else _default_bench_runs_root()}")
    if suite:
        print(f"suite={suite['id']} stage={suite['stage']} field={suite['field']} slot={suite['prompt_slot']}")
    print()

    snap_field = suite.get("field") if suite else args.field
    snap_slot = suite.get("prompt_slot") if suite else args.slot
    snapshots = list(registry.iter_snapshots(FRAMEWORK_ROOT, field=snap_field, slot=snap_slot))
    snapshots.sort(key=lambda x: str(x[3].get("created_at") or ""), reverse=True)
    print("## prompt snapshots")
    if snapshots:
        print(f"{'A':1} {'field':<6} {'slot':<14} {'name':<26} {'created':<19} id")
        for f, s, pid, card in snapshots[:args.limit]:
            active = registry.read_active_lock(FRAMEWORK_ROOT, f, s)
            marker = "*" if pid == active else " "
            print(f"{marker} {f:<6} {s:<14} {(card.get('name') or '')[:26]:<26} "
                  f"{(card.get('created_at') or '')[:19]:<19} {pid}")
        if len(snapshots) > args.limit:
            print(f"... {len(snapshots) - args.limit} more (use --limit to show more)")
    else:
        print("(no snapshots)")
    print()

    print("## templates")
    if suite:
        tasks = _select_suite_tasks(suite, args.only)
        print(f"{'task':<22} {'template':<22} status")
        for t in tasks:
            tpath = _bench_template_path(suite["stage"], t["template"])
            if not tpath.exists():
                print(f"{t['id']:<22} {t['template']:<22} missing")
                continue
            errors, warnings = _template_check_lines(suite, t, tpath)
            if errors:
                status = f"FAIL ({len(errors)} issue{'s' if len(errors) != 1 else ''})"
            elif warnings:
                status = f"WARN ({len(warnings)} warning{'s' if len(warnings) != 1 else ''})"
            else:
                status = "OK"
            print(f"{t['id']:<22} {t['template']:<22} {status}")
    else:
        root = _bench_templates_root()
        if root.exists():
            for stage_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                n = len([p for p in stage_dir.iterdir()
                         if p.is_dir() and (p / "bench_template.yaml").exists()])
                print(f"{stage_dir.name:<14} {n} template(s)")
        else:
            print("(no templates)")
    print()

    print("## bench runs")
    runs_root = Path(args.root).resolve() if args.root else _default_bench_runs_root()
    rows = list(_iter_bench_runs(runs_root, suite.get("id") if suite else None))
    if not rows:
        print(f"(no bench runs at {runs_root})")
        return
    print(f"{'run':<48} {'suite':<14} {'prompt':<34} {'tasks':<7} {'dispatch':<34} {'best_seen':>10}")
    for run_dir, m, task_dirs in rows[:args.limit]:
        task_dirs = _filter_task_dirs_for_status(task_dirs, args.only)
        states = _dispatch_task_states(run_dir)
        best_seen = None
        for task_dir in task_dirs:
            detail = _bench_task_progress_detail(task_dir)
            if detail["best_seen_pct"] is not None:
                best_seen = (
                    detail["best_seen_pct"] if best_seen is None
                    else max(best_seen, detail["best_seen_pct"])
                )
        print(f"{run_dir.name[:48]:<48} {m.get('bench_suite', ''):<14} "
              f"{(m.get('prompt_id') or '')[:34]:<34} {len(task_dirs):<7} "
              f"{_dispatch_summary(states)[:34]:<34} {_fmt_speedup_pct(best_seen):>10}")
    if len(rows) > args.limit:
        print(f"... {len(rows) - args.limit} more (use --limit to show more)")

    if suite:
        print()
        print("## task progress")
        print(f"{'run':<32} {'task':<24} {'agent':<8} {'model':<18} {'status':<14} "
              f"{'round':>5} {'done':>5} {'pend':>5} {'best_keep':>10} {'best_seen':>10} last")
        for run_dir, _m, task_dirs in rows[:args.limit]:
            task_dirs = _filter_task_dirs_for_status(task_dirs, args.only)
            states = _dispatch_task_states(run_dir)
            for task_dir in task_dirs:
                detail = _bench_task_progress_detail(task_dir)
                state = states.get(task_dir.name, {})
                status = state.get("status") or "not_started"
                agent = state.get("agent") or "-"
                model = state.get("model") or "-"
                round_num = detail["last_round"] or state.get("round") or "-"
                print(
                    f"{run_dir.name[:32]:<32} {task_dir.name[:24]:<24} "
                    f"{agent[:8]:<8} {model[:18]:<18} {status[:14]:<14} "
                    f"{str(round_num):>5} {detail['decision_rounds']:>5} "
                    f"{detail['pending_rounds']:>5} "
                    f"{_fmt_speedup_pct(detail['best_keep_pct']):>10} "
                    f"{_fmt_speedup_pct(detail['best_seen_pct']):>10} "
                    f"{detail['last_status']}"
                )




def _git_init_with_initial_commit(repo_dir: Path, message: str) -> str:
    """`git init` + add everything + initial commit. Returns the new SHA."""
    git("init", "--quiet", cwd=repo_dir)
    git("add", "-A", cwd=repo_dir)
    # Need a user.email/name for the commit; try local config first, fall back
    # to env vars to avoid clobbering user's global git identity.
    env_overrides = {}
    try:
        git("config", "user.email", cwd=repo_dir)
    except SystemExit:
        env_overrides["GIT_AUTHOR_EMAIL"] = "bench@autozyme.local"
        env_overrides["GIT_COMMITTER_EMAIL"] = "bench@autozyme.local"
        env_overrides["GIT_AUTHOR_NAME"] = "zyme bench"
        env_overrides["GIT_COMMITTER_NAME"] = "zyme bench"
    if env_overrides:
        env = dict(os.environ)
        env.update(env_overrides)
        res = subprocess.run(
            ["git", "commit", "--quiet", "-m", message],
            cwd=str(repo_dir), env=env, capture_output=True, text=True,
        )
        if res.returncode != 0:
            sys.stderr.write(res.stderr or "")
            die(f"git commit failed in {repo_dir} (exit {res.returncode})")
    else:
        git("commit", "--quiet", "-m", message, cwd=repo_dir)
    return git("rev-parse", "HEAD", cwd=repo_dir)


def _ensure_bench_framework_link(out_root: Path) -> tuple[str, str]:
    """Mirror the normal workspace layout inside a bench run.

    Real task parents (core_singlecell/, general_bio/, non_bio/) contain an
    `autozyme-framework -> ../autozyme-framework` link so task scripts can walk
    upward and find shared helpers. Bench runs live elsewhere, so create the
    same parent-level link at `<bench_run>/autozyme-framework`.
    """
    src = FRAMEWORK_ROOT.parent.parent.resolve()
    link = out_root / "autozyme-framework"
    if link.exists() or link.is_symlink():
        if link.resolve() == src:
            return ("already_exists", str(src))
        return ("conflict", str(link))
    link.symlink_to(src, target_is_directory=True)
    return ("created", str(src))




def _scaffold_bench_task(template_dir: Path, dest: Path, prompt_snapshot_dir: Path,
                          prompt_card: dict, field_prompts_dir: Path,
                          meta: dict) -> tuple:
    """Materialize one replicate task folder.

    Returns (initial_commit_sha, symlink_report) where symlink_report is a list
    of (name, source_path, status) triples. Status ∈ {created, missing_source,
    already_exists}.
    """
    if dest.exists():
        die(f"destination already exists: {dest}")
    # 1. Copy the template tree (tracked files + per-replicate state copied
    #    during register-template: .zyme/, reference_output*/, memory/).
    shutil.copytree(template_dir, dest)

    # Capture template provenance, then strip the manifest from the dest (the
    # agent doesn't need it; bench_manifest.yaml at run-root has the same info).
    template_meta_path = dest / "bench_template.yaml"
    template_meta = _read_simple_yaml(template_meta_path) if template_meta_path.exists() else {}
    if template_meta_path.exists():
        template_meta_path.unlink()

    # Restore strict per-task .gitignore (the template's .gitignore was
    # rewritten at register-template time so PromptLab git could track the
    # hard-copied state dirs; here we put back the original so bench tasks
    # behave like normal tasks — .zyme/ regenerated, results.tsv ignored, etc.)
    task_gitignore_stash = dest / ".gitignore.task"
    if task_gitignore_stash.exists():
        (dest / ".gitignore").write_text(
            task_gitignore_stash.read_text(encoding="utf-8"), encoding="utf-8"
        )
        task_gitignore_stash.unlink()

    # 2. Populate prompts/ from the live field prompts (full lifecycle set),
    #    then overwrite the slot under test with the registry snapshot.
    prompts_dir = dest / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    for src in sorted(field_prompts_dir.glob("*.md")):
        if src.name == "README.md":
            continue
        shutil.copy2(src, prompts_dir / src.name)

    target_prompt_filename = Path(prompt_card["source_path"]).name
    target_prompt_path = prompts_dir / target_prompt_filename
    target_prompt_path.write_text(
        (prompt_snapshot_dir / "prompt.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # 3. Defensive memory/ skeleton (older templates may not have one; newer
    #    register-template writes it but we don't trust the template wholesale).
    if not (dest / "memory" / "discoveries.md").exists():
        _write_memory_skeleton(dest, dest.name)

    # 4. Materialize large read-only deps that the template recorded:
    #    - upstream_repo: cloned from URL @ commit (template-portable)
    #    - data/, legacy upstream/: symlinked back to the source task dir
    #    Failures land in symlink_report; caller surfaces them in bulk.
    symlink_report = []

    upstream_clone = template_meta.get("upstream_repo")
    if upstream_clone and upstream_clone.get("clone_url"):
        target = dest / "upstream_repo"
        url = upstream_clone["clone_url"]
        sha = upstream_clone.get("commit") or ""
        if target.exists() or target.is_symlink():
            symlink_report.append(("upstream_repo", url, "already_exists"))
        else:
            try:
                git("clone", url, str(target))
                if sha:
                    git("checkout", sha, cwd=target)
                symlink_report.append(("upstream_repo", url, "cloned"))
            except SystemExit:
                shutil.rmtree(target, ignore_errors=True)
                symlink_report.append(("upstream_repo", url, "clone_failed"))

    # Materialize `data/` from the HF-portable block. Local-first: if the
    # expected `<datasets_root>/<hf_path>/` already exists, just symlink.
    # Otherwise `hf download` to land it there, then symlink. AUTOZYME_DATASETS_ROOT
    # lets users redirect to a shared cache. Tasks that omit `data:` (shared
    # checkpoints, external data) keep using symlink_sources below.
    hf_data = template_meta.get("data")
    if hf_data and hf_data.get("hf_repo") and hf_data.get("hf_path"):
        target = dest / "data"
        datasets_root = _resolve_datasets_root()
        local_target = datasets_root / hf_data["hf_path"]
        if target.exists() or target.is_symlink():
            symlink_report.append(("data", str(local_target), "already_exists"))
        else:
            downloaded_now = False
            if not local_target.exists():
                info(f"  hf: downloading {hf_data['hf_repo']}:{hf_data['hf_path']} "
                     f"→ {local_target}/")
                try:
                    _hf_download(hf_data["hf_repo"], hf_data["hf_repo_type"],
                                 hf_data["hf_path"], datasets_root)
                    downloaded_now = True
                except subprocess.CalledProcessError:
                    info(f"  hf: download FAILED for "
                         f"{hf_data['hf_repo']}:{hf_data['hf_path']} "
                         f"(check `hf auth login` + network; "
                         f"datasets_root={datasets_root})")
                    symlink_report.append(
                        ("data", f"hf://{hf_data['hf_repo']}/{hf_data['hf_path']}",
                         "hf_download_failed"))
            if local_target.exists():
                target.symlink_to(local_target)
                symlink_report.append(
                    ("data", str(local_target),
                     "downloaded" if downloaded_now else "created"))
            elif not any(r[0] == "data" for r in symlink_report):
                symlink_report.append(("data", str(local_target), "missing_source"))

    for name, src_path in (template_meta.get("symlink_sources") or {}).items():
        target = dest / name
        if target.exists() or target.is_symlink():
            symlink_report.append((name, src_path, "already_exists"))
            continue
        src = Path(src_path)
        if not src.exists():
            symlink_report.append((name, src_path, "missing_source"))
            continue
        target.symlink_to(src)
        symlink_report.append((name, src_path, "created"))

    # 4b. Append explicit gitignore entries for the deps we just materialized.
    #     The template's .gitignore has `data/`, `upstream_repo/` (with
    #     trailing slashes), which match directory entries but NOT symlinks —
    #     git records a symlink as a file even when it points at a directory.
    #     Without this, the initial commit would track the (host-specific)
    #     symlink as a blob, breaking template portability. Cloned upstream
    #     repos are also gitignored: they're host-local and shouldn't be in
    #     the replicate's git history.
    materialized_locally = [n for n, _, st in symlink_report
                            if st in ("created", "cloned", "downloaded")]
    if materialized_locally:
        gi = dest / ".gitignore"
        existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
        addition = "\n# bench-init deps (host-specific; never commit)\n" + \
                   "\n".join(f"/{n}" for n in materialized_locally) + "\n"
        gi.write_text(existing.rstrip() + "\n" + addition, encoding="utf-8")

    # 5. Drop .zyme_meta.yaml at task root.
    (dest / ZYME_META_FILENAME).write_text(_emit_simple_yaml(meta), encoding="utf-8")

    # 6. git init + initial commit so subsequent `zyme run` works. The
    #    .gitignore inherited from the template (plus the bench-symlink lines
    #    appended above) keeps all gitignored state out.
    sha = _git_init_with_initial_commit(
        dest, f"zyme bench: {meta['bench_suite']}/{meta['bench_task']}_r{meta['replicate']}"
    )
    return sha, symlink_report




def cmd_bench_init(args):
    """Scaffold N replicate task folders for a benchmark suite with a chosen prompt."""
    from zyme import registry

    suite_path = _bench_suite_path(args.suite_id)
    if not suite_path.exists():
        die(f"suite manifest not found: {suite_path}. "
            f"Available: {sorted(p.stem for p in _bench_suites_root().glob('*.yaml'))}")
    suite = _read_suite_yaml(suite_path)
    _validate_suite(suite, args.suite_id)
    selected_tasks = _select_suite_tasks(suite, args.only)

    # Resolve prompt snapshot
    try:
        found = registry.find_snapshot(FRAMEWORK_ROOT, args.prompt_id)
    except ValueError as e:
        die(str(e))
    if found is None:
        die(f"prompt snapshot not found: {args.prompt_id}")
    p_field, p_slot, prompt_id, prompt_card = found

    # Sanity: prompt's slot/field should match the suite (otherwise the user
    # is testing the wrong slot).
    if p_slot != suite["prompt_slot"]:
        die(f"prompt {prompt_id} is for slot '{p_slot}' but suite "
            f"'{args.suite_id}' tests slot '{suite['prompt_slot']}'.")
    if p_field != suite["field"]:
        die(f"prompt {prompt_id} is field '{p_field}' but suite "
            f"'{args.suite_id}' is field '{suite['field']}'.")

    snap_dir = registry.snapshot_dir_for(FRAMEWORK_ROOT, p_field, p_slot, prompt_id)
    field_prompts_dir = registry.live_prompts_dir(FRAMEWORK_ROOT, p_field)

    # Resolve all required templates BEFORE creating any output (fail fast).
    # Templates are looked up under <root>/<suite.stage>/<name>/.
    suite_stage = suite["stage"]
    missing = []
    template_paths = {}
    for t in selected_tasks:
        tpath = _bench_template_path(suite_stage, t["template"])
        if not tpath.exists() or not (tpath / "bench_template.yaml").exists():
            missing.append((t["id"], t["template"], str(tpath)))
        else:
            template_paths[t["id"]] = tpath
    if missing:
        msg = ["missing bench templates:"]
        for tid, tname, path in missing:
            msg.append(f"  - task '{tid}' expects template '{tname}' at {path}")
        msg.append("")
        msg.append(f"Register them first: `zyme bench register-template "
                   f"<task_dir> --as <template_name> --stage {suite_stage} --at post_init`")
        die("\n".join(msg))

    reps = args.reps or suite.get("default_reps") or 1

    # --name: single-replicate "drop into existing dir" mode. Skips suite-level
    # manifest / EXPERIMENT.md / framework symlink so the replicate lands as a
    # self-contained sibling next to anything already at <out>/.
    single_name = args.name
    if single_name:
        if len(selected_tasks) != 1:
            die(f"--name requires exactly one task selected (got {len(selected_tasks)}). "
                f"Use --only <task_id> to filter.")
        if reps != 1:
            die(f"--name is single-replicate only; got --reps {reps}.")

    # Output directory
    if args.out:
        out_root = Path(args.out).resolve()
    elif single_name:
        out_root = Path.cwd().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Default name bakes in suite id + prompt short name so `ls bench_runs/`
        # is self-describing. suite["id"] carries the task name for single-task
        # suites (e.g. init_test_fgsea) and a meaningful bundle name for
        # multi-task ones (e.g. iterate_core). Full prompt_id is recorded in
        # bench_manifest.yaml, so the dir name doesn't need it.
        prompt_short = prompt_card.get("name") or prompt_id
        out_root = _default_bench_runs_root() / f"{suite['id']}__{prompt_short}__{ts}"

    if single_name:
        # Only the named subdir must be free; out_root itself may pre-exist.
        named_dest = out_root / single_name
        if named_dest.exists():
            if not args.force:
                die(f"replicate dir already exists: {named_dest}. "
                    f"Pass --force to overwrite (DESTRUCTIVE — wipes only that subdir).")
            shutil.rmtree(named_dest)
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        if out_root.exists():
            if not args.force:
                die(f"bench output dir already exists: {out_root}. "
                    f"Pass --force to overwrite (DESTRUCTIVE).")
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True)

    framework_repo = FRAMEWORK_ROOT.parent.parent
    fw_sha = _git_head_sha(framework_repo)
    if single_name:
        fw_link_status, fw_link_target = (None, None)
    else:
        fw_link_status, fw_link_target = _ensure_bench_framework_link(out_root)
        if fw_link_status == "conflict":
            die(f"bench output dir has conflicting autozyme-framework entry: {fw_link_target}")
    created_at = datetime.now().isoformat(timespec="seconds")

    # Suite-level manifest at out_root (standard mode only — --name is for
    # drop-in single replicates and doesn't own the parent dir).
    if not single_name:
        suite_manifest = {
            "bench_suite": suite["id"],
            "prompt_id": prompt_id,
            "prompt_sha256": prompt_card.get("content_sha256", ""),
            "prompt_slot": p_slot,
            "prompt_field": p_field,
            "reps": reps,
            "tasks": [t["id"] for t in selected_tasks],
            "framework_sha": fw_sha,
            "framework_path": str(framework_repo.resolve()),
            "created_at": created_at,
        }
        if args.purpose:
            suite_manifest["purpose"] = args.purpose
        (out_root / "bench_manifest.yaml").write_text(
            _emit_simple_yaml(suite_manifest), encoding="utf-8"
        )
        _write_experiment_doc(
            out_root,
            suite=suite,
            selected_tasks=selected_tasks,
            prompt_id=prompt_id,
            prompt_card=prompt_card,
            reps=reps,
            created_at=created_at,
            purpose=args.purpose,
        )

    if single_name:
        info(f"scaffolding 1 replicate as '{single_name}' "
             f"(--name mode, no suite-level scaffolding)")
    else:
        info(f"scaffolding {len(selected_tasks)} task(s) × {reps} rep(s) "
             f"= {len(selected_tasks) * reps} bench tasks")
    info(f"  prompt:    {prompt_id}")
    info(f"  suite:     {suite['id']}")
    if len(selected_tasks) != len(suite["tasks"]):
        info(f"  only:      {', '.join(t['id'] for t in selected_tasks)}")
    if args.purpose:
        info(f"  purpose:   {args.purpose}")
    info(f"  out:       {out_root}")
    if fw_link_status == "created":
        info(f"  symlink:   autozyme-framework -> {fw_link_target}")

    created = []
    missing_symlinks = []  # accumulate cross-task symlink failures for one summary
    for t in selected_tasks:
        tpath = template_paths[t["id"]]
        for r in range(1, reps + 1):
            dest = out_root / (single_name if single_name else f"{t['id']}_r{r}")
            meta = {
                "bench_suite": suite["id"],
                "bench_task": t["id"],
                "replicate": r,
                "prompt_slot": p_slot,
                "prompt_field": p_field,
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_card.get("content_sha256", ""),
                "framework_sha": fw_sha,
                "template_name": t["template"],
                "created_at": created_at,
            }
            sha, symlink_report = _scaffold_bench_task(
                template_dir=tpath, dest=dest,
                prompt_snapshot_dir=snap_dir, prompt_card=prompt_card,
                field_prompts_dir=field_prompts_dir, meta=meta,
            )
            created.append((dest, sha))
            # Per-task line: mention deps materialized inline; failures
            # surface as one bulk warning at end so the user sees them clearly.
            link_summary = ""
            if symlink_report:
                by_status = {}
                for n, _, st in symlink_report:
                    by_status.setdefault(st, []).append(n)
                parts = []
                if by_status.get("created"):
                    parts.append(f"symlinks: {', '.join(by_status['created'])}")
                if by_status.get("cloned"):
                    parts.append(f"cloned: {', '.join(by_status['cloned'])}")
                if by_status.get("downloaded"):
                    parts.append(f"hf: {', '.join(by_status['downloaded'])}")
                if parts:
                    link_summary = f" [+{'; '.join(parts)}]"
                for n, sp, st in symlink_report:
                    if st in ("missing_source", "clone_failed", "hf_download_failed"):
                        missing_symlinks.append((dest, n, sp, st))
            info(f"  ✓ {dest.relative_to(out_root)} (initial commit {sha[:8]}){link_summary}")

    if missing_symlinks:
        info("WARN: some bench-init deps failed — tasks will fail until you fix:")
        reasons = {
            "missing_source": "source not found",
            "clone_failed": "git clone failed",
            "hf_download_failed": "hf download failed (run `hf auth login`?)",
        }
        for dest, name, src, st in missing_symlinks:
            info(f"  {dest.relative_to(out_root)}/{name} → {src} ({reasons[st]})")

    info(f"done. Open each folder in Claude Code and run the iterate session.")
    info(f"  → {out_root}")




def cmd_bench_list(args):
    """List bench runs under the default bench_runs/ root (or --root)."""
    root = Path(args.root).resolve() if args.root else _default_bench_runs_root()
    if not root.exists():
        info(f"(no bench runs at {root})")
        return
    rows = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        manifest = entry / "bench_manifest.yaml"
        if not manifest.exists():
            continue
        m = _read_simple_yaml(manifest)
        # Count task folders
        task_dirs = [p for p in entry.iterdir()
                     if p.is_dir() and (p / ZYME_META_FILENAME).exists()]
        rows.append((entry.name, m, len(task_dirs)))
    if not rows:
        info(f"(no bench runs at {root})")
        return
    print(f"{'run':<58} {'suite':<14} {'tasks':<6} created_at")
    print("-" * 110)
    for name, m, n in rows:
        print(f"{name[:58]:<58} {m.get('bench_suite', ''):<14} "
              f"{n:<6} {m.get('created_at', '')}")
    print()
    print(f"({len(rows)} run(s) at {root})")
