"""`zyme init` — scaffold a new task in the current directory."""

import os
import shutil
from pathlib import Path

from zyme.utils import die, info, git
from zyme.commands._shared import (
    FRAMEWORK_ROOT,
    _FAMILY_SKELETON,
    _write_memory_skeleton,
)




def _is_repo_url(s):
    """Return True if s looks like a clone-able URL (vs a local path)."""
    return s.startswith(("http://", "https://", "git@", "git://", "ssh://"))




def _detect_language(*candidate_dirs):
    """Sniff python vs R from a list of candidate directories.

    Returns 'python', 'R', or None. First candidate that yields a verdict wins.
    """
    for d in candidate_dirs:
        if d is None or not d.exists() or not d.is_dir():
            continue
        if (d / "DESCRIPTION").exists():
            return "R"
        if (d / "pyproject.toml").exists() or (d / "setup.py").exists():
            return "python"
    return None




def cmd_init(args):
    """Scaffold a new task in the current directory.

    cwd IS the task. Task name = cwd basename. Files from `templates/task_template/`
    are copied flat into cwd. If cwd is not already a git repo, one is initialized
    and the scaffold is committed; otherwise the existing repo is used as-is.
    Errors if cwd already has a task.yaml (refuses to clobber).

    Beyond the template copy, this command also: clones target_repo into
    upstream_repo/ when it's a URL (unless --no-clone); detects language
    (python vs R) from upstream/local repo or --language; renames the
    matching .template files (deleting the wrong-language siblings); creates
    data/, setup/, reference_outputs/ scaffolding directories; writes
    memory/{discoveries,active_opts,dead_ends}.md with task-name-substituted
    headers; writes family.md skeleton. The init agent's job is now content,
    not scaffolding.
    """
    task_dir = Path(os.getcwd()).resolve()
    task_name = task_dir.name

    if (task_dir / "task.yaml").exists():
        die(f"task already exists in {task_dir}: task.yaml present (cwd is already a task)")

    template = FRAMEWORK_ROOT / "templates" / "task_template"
    if not template.exists():
        die(f"template not found: {template}")

    # Copy template contents (files + sub-dirs) into cwd flat — NOT the template root itself
    for item in template.iterdir():
        target = task_dir / item.name
        if target.exists():
            die(f"refusing to clobber existing {target}")
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    # Auto-clone target_repo into upstream_repo/ when it's a URL. The init
    # agent used to do this in step 1; deterministic CLI clone is more
    # reliable and lets us sniff language right after.
    upstream_repo_dir = task_dir / "upstream_repo"
    if _is_repo_url(args.target_repo) and not args.no_clone:
        info(f"cloning {args.target_repo} → upstream_repo/ ...")
        try:
            git("clone", args.target_repo, str(upstream_repo_dir))
        except Exception as e:
            info(f"[clone] failed: {e}")
            info("[clone] init agent will need to clone manually in step 1")

    # Detect language: explicit --language wins; else sniff upstream_repo/
    # (cloned above) or the local target_repo path. None → leave .template
    # files alone and warn (init agent renames). DESCRIPTION → R;
    # pyproject.toml/setup.py → python.
    target_local = Path(args.target_repo)
    sniff_dirs = [
        upstream_repo_dir if upstream_repo_dir.exists() else None,
        target_local if target_local.exists() else None,
    ]
    language = args.language or _detect_language(*sniff_dirs)
    if language is None:
        info("[language] could not auto-detect — .template files left in place; "
             "init agent will rename based on target language. "
             "Pass --language python|R next time to skip this warning.")

    # Rename .template → .{py,R} for the chosen language; delete the wrong-
    # language siblings so the agent isn't tempted to fill the wrong one.
    if language is not None:
        keep_ext = "py" if language == "python" else "R"
        drop_ext = "R" if language == "python" else "py"
        rename_targets = [
            (task_dir / f"evaluate.{keep_ext}.template",  task_dir / f"evaluate.{keep_ext}"),
            (task_dir / f"reference.{keep_ext}.template", task_dir / f"reference.{keep_ext}"),
            (task_dir / "pipeline" / f"run.{keep_ext}.template",
             task_dir / "pipeline" / f"run.{keep_ext}"),
        ]
        drop_targets = [
            task_dir / f"evaluate.{drop_ext}.template",
            task_dir / f"reference.{drop_ext}.template",
            task_dir / "pipeline" / f"run.{drop_ext}.template",
        ]
        for src, dst in rename_targets:
            if src.exists():
                src.rename(dst)
        for p in drop_targets:
            if p.exists():
                p.unlink()

    # Scaffold the directory structure the init agent (and downstream phases)
    # will populate. data/ for binary inputs, setup/ for prep scripts that
    # generate them, reference_outputs/ for per-tier reference dumps, memory/
    # for narrative state files. All four were previously created ad-hoc by
    # the agent following 1_init.md prose; deterministic CLI scaffolding
    # makes the prompt smaller and the layout consistent.
    for sub in ("data", "setup", "reference_outputs", "memory"):
        (task_dir / sub).mkdir(exist_ok=True)

    # Write the three narrative files with task-name-substituted headers.
    # These are gitignored (see .gitignore) — they're local narrative state,
    # not code — but the agent expects them to exist on session start.
    _write_memory_skeleton(task_dir, task_name)

    # Write family.md skeleton. The init agent fills in verdict + paragraph
    # during step 1 after reading upstream source.
    (task_dir / "family.md").write_text(_FAMILY_SKELETON, encoding="utf-8")

    # Each task is its own self-contained git repo (PRD §8 double-pointer
    # workflow: attempts on default branch, .zyme/best.ref tracks best so far).
    # No autozyme/* branch — nothing to isolate from in a fresh per-task repo.
    inside_repo = git("rev-parse", "--is-inside-work-tree", cwd=task_dir, check=False)
    git_initialized = inside_repo != "true"
    if git_initialized:
        git("init", cwd=task_dir)

    # Fill `task.yaml`'s structured fields directly — `target_repo` /
    # `target_function` come from CLI args, `task` slug from cwd basename.
    # `signature` is left as a placeholder for the init agent to fill after
    # reading upstream source. `dataset_hint:` is added only when --dataset
    # was passed; the init agent reads it during step 2 then removes it.
    task_yaml_path = task_dir / "task.yaml"
    yaml_text = task_yaml_path.read_text(encoding="utf-8")
    yaml_text = yaml_text.replace("<TASK_NAME>", task_name)
    yaml_text = yaml_text.replace("<TARGET_REPO_URL_OR_LOCAL_PATH>", args.target_repo)
    if args.target_function:
        yaml_text = yaml_text.replace("<PKG::FUNC>", args.target_function)
    # K2: task.yaml no longer carries a reference.<EXT> reference — the
    # single `reference.{R,py}` is resolved at runtime. Nothing to substitute.
    if args.dataset:
        # Insert dataset_hint right after the signature: line, transient field.
        yaml_text = yaml_text.replace(
            "signature: <CALL_SIGNATURE>",
            f"signature: <CALL_SIGNATURE>\ndataset_hint: {args.dataset}",
            1,
        )
    task_yaml_path.write_text(yaml_text, encoding="utf-8")

    # Copy every phase prompt from the chosen prompt set into the task's
    # prompts/ dir. Straight copy — agents read per-task facts from task.yaml,
    # not from substituted prompt placeholders. Skip README — framework-
    # internal, not a runnable prompt.
    field = getattr(args, "field", None) or "Bio"
    src_dir = FRAMEWORK_ROOT / "prompts" / field
    if not src_dir.exists():
        available = sorted(p.name for p in (FRAMEWORK_ROOT / "prompts").iterdir() if p.is_dir())
        die(f"unknown --field '{field}'. Available prompt sets: {', '.join(available)}")
    prompts_dir = task_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    customized = []
    for src in sorted(src_dir.glob("*.md")):
        if src.name == "README.md":
            continue
        dest = prompts_dir / src.name
        shutil.copy2(src, dest)
        customized.append(dest)

    # Situational prompts (transfer-init, thread-fairness audit) live in a
    # `situational/` subdir so they don't clutter the task's top-level
    # prompts/ — they fire only for specific workflows, not the main 0-5 loop.
    # Copy them tucked away under prompts/situational/.
    sit_src = src_dir / "situational"
    if sit_src.is_dir():
        sit_dest = prompts_dir / "situational"
        sit_dest.mkdir(exist_ok=True)
        for src in sorted(sit_src.glob("*.md")):
            dest = sit_dest / src.name
            shutil.copy2(src, dest)
            customized.append(dest)

    # Initial commit captures the full scaffold (template files + prompts).
    # Only commit when we created the repo; if user dropped the task into an
    # existing repo, leave their working tree alone for them to commit.
    if git_initialized:
        git("add", "-A", cwd=task_dir)
        git("commit", "-m", f"zyme init: {task_name}", cwd=task_dir)

    info(f"task '{task_name}' scaffolded in {task_dir}")
    info(f"git: {'initialized new repo' if git_initialized else 'using existing repo'}")
    info(f"language: {language if language else 'not auto-detected (init agent will rename .template files)'}")
    info(f"prompt set: {field}")
    info(f"prompts written to {prompts_dir}/:")
    for p in customized:
        info(f"  - {p.relative_to(prompts_dir)}")

    # Auto-register an init-stage bench template. Two purposes at once:
    #   (a) Rollback anchor — the pristine bare scaffold the init prompt has
    #       not yet touched. `git reset --hard <init_sha>` recovers it; this
    #       template is a second copy that survives even if the user deletes
    #       the source task.
    #   (b) Seed for prompt A/B benchmarking later (`zyme bench init` needs a
    #       across init-stage prompts needs a fixed starting point per task).
    # Opt out with --no-bench-snapshot. Skip if the repo wasn't freshly init'd
    # (commit history we'd snapshot off of isn't ours), or if a template with
    # this name already exists (precious — never silently overwrite).
    if git_initialized and not getattr(args, "no_bench_snapshot", False):
        from argparse import Namespace
        from zyme.commands.bench import (
            _bench_template_path, cmd_bench_register_template,
        )
        template_path = _bench_template_path("init", task_name)
        print()
        if template_path.exists():
            info(f"[bench-snapshot] init-stage template '{task_name}' already "
                 f"exists at {template_path}; skipping (--force on the manual "
                 f"`zyme bench register-template` if you want to overwrite)")
        else:
            info("[bench-snapshot] auto-registering init-stage template "
                 "(pass --no-bench-snapshot to opt out next time)")
            try:
                cmd_bench_register_template(Namespace(
                    task_dir=str(task_dir),
                    name=task_name,
                    stage="init",
                    at="init",
                    commit=None,
                    force=False,
                ))
            except SystemExit:
                info(f"[bench-snapshot] register-template failed; init "
                     f"scaffolding itself succeeded. Re-run manually later: "
                     f"`zyme bench register-template {task_dir} --as {task_name} "
                     f"--stage init --at init`")
    print()
    # Build the post-iteration prompt chain dynamically — different prompt
    # sets (Bio vs OtherField) name step 3 differently (e.g. validate_scaling
    # vs expand_scaling), so don't hardcode the filename.
    post_iterate = sorted(
        p.name for p in customized
        if p.name not in ("1_init.md", "2_iterate.md")
    )
    chain = " → ".join(f"prompts/{n}" for n in post_iterate) if post_iterate else "(none)"

    print("Workflow:")
    print(f"  1. cd {task_dir}")
    print("  2. Open Claude Code; paste prompts/1_init.md → agent fills task + runs baselines")
    print("  3. Review agent's chat summary; say 'ok' to confirm")
    print("  4. Open another Claude Code session; paste prompts/2_iterate.md → agent enters the optimization loop")
    print(f"  5. After convergence, run {chain}")
