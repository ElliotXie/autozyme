# zyme — bootstrap (step 0)

You are helping a user start a new autozyme optimization task. Your job: confirm the target, check the environment, scaffold the task with `zyme init`, and hand off.

## Step 0: Verify `zyme` CLI is available

Run `zyme --version`. If it fails:

- If `autozyme-framework/` exists locally (or as a sibling/parent dir): tell the user to run `pip install -e <path-to>/autozyme-framework/autozyme_cli/`
- Otherwise: `pip install "git+https://github.com/ElliotXie/autozyme.git#subdirectory=autozyme_cli"`

Do not proceed until `zyme --version` succeeds.

## Step 1: Confirm target

Restate which package and which function you understood the task to be (e.g. "you want to optimize `Seurat::FindAllMarkers`, right?"). Have the user confirm. Don't guess silently and proceed.

## Step 2: Confirm dataset (optional)

Ask whether the user has a specific dataset path. If yes, verify the file exists — pass it via `--dataset`. If no, omit `--dataset` entirely — dataset selection is handled later by the init agent. Don't try to find one here.

## Step 3: Infer field

Decide which prompt set to use based on the target:

- **Bio** — target is from R/Bioconductor, Seurat, Scanpy, scRNA-seq, spatial transcriptomics, or any bioinformatics/computational biology ecosystem.
- **OtherField** — everything else (astronomy, seismology, climate, physics, generic scientific computing).

State your inference to the user (e.g. "Seurat is bioinformatics, so I'll use the Bio prompt set"). Only ask for confirmation if genuinely ambiguous.

## Step 4: Confirm task folder location

Propose where to scaffold the task:

- If cwd is the framework root or a workspace root: propose creating `./test_<function_name>/` as a subdirectory.
- If cwd is already an empty / fresh task dir: use cwd.

The task dir must have `autozyme-framework/` reachable as a sibling or ancestor — the R/Python runtime helpers are sourced from there. If the user cloned the framework repo, the task dir should be a sibling of that clone (e.g. `autozyme-framework/` and `test_findallmarkers/` at the same level).

Confirm before `cd`-ing or running `zyme init`.

## Step 5: Run `zyme init`

```
zyme init <target_repo> [<target_function>] [--dataset <path>] [--field OtherField]
```

- `--field` defaults to Bio; pass `--field OtherField` only for non-bio targets.
- `target_repo` is the upstream library's GitHub URL (e.g. `https://github.com/satijalab/seurat`).

## What `zyme init` does

- Refuses if `task.yaml` already exists (no clobber).
- Copies task template files, clones `target_repo` into `upstream_repo/`, sniffs language (R vs Python), renames `.template` files accordingly.
- Creates scaffolding dirs: `data/`, `setup/`, `reference_outputs/`, `memory/`, `prompts/`.
- Copies all phase prompts from the chosen field into `<task_dir>/prompts/`.
- Runs `git init` and commits the scaffold.

What it does NOT do: download data, run reference, or fill placeholders — that's `1_init.md`'s job.

## Step 6: Hand off

After `zyme init` succeeds, relay the workflow to the user:

> Done. Open a fresh Claude Code session in `<task_dir>/` and paste the contents of `prompts/1_init.md` to start phase 1 (scaffold fill + baselines).
>
> After phase 1 completes, open another fresh session and paste `prompts/2_iterate.md` to start the optimization loop.

Then **stop**. Do NOT read `1_init.md` yourself — it's a separate phase designed for a fresh agent context.
