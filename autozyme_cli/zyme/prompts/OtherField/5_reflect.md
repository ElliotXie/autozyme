# Reflect — autozyme-framework

Two questions. Be honest, not diplomatic. Empty answers OK if you genuinely have nothing — don't pad. Each answer goes into its OWN file under its own folder so the two channels stay separately browsable.

1. **Prompt feedback** — what in the prompt you just ran was unclear, missing, redundant, or off? What would you change?
   → write to `<autozyme-framework>/reflections/prompt_reflect_feedback/<category>_<task>.md`

2. **CLI (`zyme`) feedback** — what was awkward, missing, or surprising? What would you change?
   → write to `<autozyme-framework>/reflections/zyme_cli_feedback/<category>_<task>.md`

Where `<task>` = `task:` field from `task.yaml` and `<category>` is one of:

- `initialization` — you were setting up a fresh task or porting a sibling
- `iteration` — you were running the optimization loop
- `scaling` — you were validating / expanding to larger datasets
- `packaging` — you were bundling a converged task into an installable package

If a target file already exists, suffix `__YYYY-MM-DD`. Skip writing one of the two files entirely if you genuinely have nothing for that channel — don't create empty stubs.

If `<category>` is `initialization` or `iteration`, run `zyme validate init` (or `zyme validate iterate`) before committing. The audit report lands under `<autozyme-framework>/validation/<task>/v<N>/`, is mirrored to `<task_dir>/.zyme/validate/`, and per-finding rows are appended to `<task_dir>/validate.tsv`. Skip for `scaling` / `packaging`.

Then commit and push.
