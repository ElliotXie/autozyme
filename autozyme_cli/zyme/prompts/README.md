# prompts/

Phase prompts for the autozyme loop. Two parallel sets:

- **[Bio/](Bio/)** — single-cell / bioinformatics targets (default). `zyme init` reads from here unless `--field OtherField`.
- **[OtherField/](OtherField/)** — non-biology targets (ML kernels, scientific libraries in any domain). Near-copy of `Bio/` with generic performance-engineer framing.

`Bio/` is the reference. Structural changes (new flag, new tag, new helper, new flow) land in `Bio/` first, then propagate to `OtherField/` minus the biology-specific examples. See each folder's `README.md` for the per-phase agent map and main-path diagram, and `../../PROMPT_PHILOSOPHY.md` for editing rules.

## Top-level field-agnostic prompts

A small set of prompts lives at the top level of `prompts/` (NOT inside `Bio/` or `OtherField/`). These run outside the optimization loop — they audit / validate / report rather than drive iteration, and their content is field-agnostic so we keep one copy.

| File | Used by | What it is |
|---|---|---|
| [validate_init.md](validate_init.md) | `zyme validate init` | LLM-based adversarial audit of init-phase setup (dataset realism, metric coverage, threshold sanity, target-function scope). Run once after `zyme baseline reference`, before iterating. |
| [validate_iterate.md](validate_iterate.md) | `zyme validate iterate` | LLM-based adversarial audit of iterate-phase pipeline behavior (timer-window hoist, IO function hijack, gate-threshold relaxation, output truncation, narrative-vs-code consistency, seed pinning). Run once after iterate converges. |

These are **not** scanned by `sync_prompts.py` (it only walks `Bio/` and `OtherField/`) and **not** copied into per-task `prompts/` directories by `sync_prompts_to_tasks.py`. They live with the framework and are read directly by `zyme validate` from the framework install at runtime.

Convention for adding more top-level prompts: name them after the `zyme` subcommand or skill that consumes them (e.g. a future `zyme reflect-corpus` would read `prompts/reflect_corpus.md`), and document them here.

## Keeping the prompts in sync

Two helpers live alongside the prompt folders so an agent editing here sees them. After any prompt edit — and especially after CLI flag changes in [../zyme/cli.py](../zyme/cli.py) — run them:

| Script | What it checks / does |
|---|---|
| [sync_prompts.py](sync_prompts.py) | Drift checker. Walks every `*.md` under `Bio/` and `OtherField/`, validates each `zyme <cmd> ...` invocation against the installed CLI's argparse spec (catches dead flags), reports cross-field flag-set drift in parallel files, and surfaces known sync invariants (e.g. `--rerun` should pair with explicit `--n` in scale prompts). Read-only; not an auto-rewriter. Run with `--strict` in CI. |
| [sync_prompts_to_tasks.py](sync_prompts_to_tasks.py) | Propagates framework prompt edits into every existing task at `<workspace>/test_*/prompts/` and `<workspace>/{test_,}{core_singlecell,general_bio,non_bio}/test_*/prompts/`. Detects task flavor (Bio vs OtherField) from `1_init.md` content first, then scaling-prompt filename. Splices `1_init.md` (preserves the task's `## Your inputs for this run` block, takes everything else from the framework); straight-overwrites all other prompt files. Use `--dry-run` first. |

Typical workflow after a prompt edit:

```bash
python autozyme_cli/zyme/prompts/sync_prompts.py                    # report drift / invalid flags
python autozyme_cli/zyme/prompts/sync_prompts_to_tasks.py --dry-run # preview
python autozyme_cli/zyme/prompts/sync_prompts_to_tasks.py           # apply
```

On Windows, use the `autozyme_cli/zyme/prompts/...` path above. The old
`autozyme_cli/prompts` developer symlink can check out as a plain text file
when Git symlink support is disabled.
