# templates/

Skeletons consumed by `zyme init`. The init agent copies `task_template/` to `tasks/<name>/` and fills placeholders.

## Files

| File | What | Filled by |
|---|---|---|
| `task_template/` | Per-task directory skeleton | `zyme init` (copies wholesale, then fills) |
| `task_template/task.yaml` | Task config schema | `zyme init` |
| `task_template/reference.py.template` | Baseline runner pattern | `zyme init` (extracts from upstream repo) |
| `task_template/evaluate.py.template` | Concordance check pattern | `zyme init` (writes metrics inline based on the function's mathematical nature) |
| `task_template/pipeline/run.py.template` | Agent's monkey-patch entry | `zyme init` (skeleton; first round = unmodified baseline) |
| `task_template/discoveries.md` | Append-only log of non-obvious technical facts (load-bearing) | template content kept; main agent appends DISCOVERY blocks |
| `task_template/active_opts.md` | Stack of accepted optimizations composing into `best` (overwrite-in-place) | template content kept; main agent updates per accept |
| `task_template/dead_ends.md` | Falsified angles (overwrite-in-place when re-tried) | template content kept; main agent updates per discard |
| `task_template/round_log.md` | Optional terse prose log for things that don't fit a TSV cell | unused at init; main agent appends sparingly |
| `task_template/results.tsv` | Header-only leaderboard (round, commit, dataset, speed_sec, speedup_pct, peak_mb, status, metrics_json, hypothesis, description) | `zyme run` appends rows |
| `task_template/.gitignore` | Standard ignores (artifacts/, reference_output/, *.pyc) | static |
| `task_template/README.md` | Per-task documentation: target / call chain / metrics / concordance budget / anticipated angles / baseline | `zyme init` (replaces template placeholders) |

## R parity

For R-based tasks (Seurat, Bioconductor packages), `.py.template` files have `.R.template` siblings with the same shape. v1 ships Python first; R is mirror. Add R templates when first R task is added.

## What lives elsewhere

- The **prompts** that drive the init/main/expand/package/reflect agents → [prompts/](../prompts/)
