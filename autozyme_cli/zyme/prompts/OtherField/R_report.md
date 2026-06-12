# R_report — generate the final HTML report

Terminal one-shot. Runs once after phase 4 package + any phase 6.x transfer have converged. For mid-stream previews run `zyme report` directly (mechanical-only, no agent needed).

The renderer (`zyme report`) owns charts, accordions, theme, animations. Your job: (1) normalize memory files so the parser can read them, (2) fill `memory/report_narrative.md`, (3) render, (4) eyeball.

## Step 1 — Survey

Read: `task.yaml`, `memory/best_state.md`, `memory/active_opts.md`, `memory/discoveries.md`, `memory/dead_ends.md`, `results.tsv`. Identify the production tier (largest baseline), the target function signature + language, and a one-line field/domain tagline (e.g. "Python · Astropy · time-series photometry"; "Python · ObsPy · earthquake waveforms"; "Python · xclim · bias correction").

## Step 2 — Normalize memory files to canonical schema

Rewrite every entry in `memory/active_opts.md`, `memory/discoveries.md`, `memory/dead_ends.md` to match the schemas below. **Format conversion only — preserve content verbatim** (every observation, mechanism, caveat, number, commit hash survives). Don't summarize, compress, or invent. Field labels are mandatory; the parser extracts by them.

`memory/active_opts.md` (one section per accepted optimization, matching `best_state.md` patch stack):

```markdown
## <opt_name>
Round: N (commit <sha>)
Mechanism: <preserve original; multi-line OK>
Speedup contribution: <numbers verbatim; "—" if not recorded>
Concordance: <metric: value pairs verbatim>
Location: <file:line / function>
Notes: <caveats, scale-dependence, ordering — verbatim>
```

`memory/discoveries.md`:

```markdown
## DISCOVERY: <title verbatim>
Triggered: round N <context verbatim, or "(no round attribution)" if unknowable>
Cause: <first paragraph verbatim>
Implication: <remaining paragraphs verbatim>
```

`memory/dead_ends.md`:

```markdown
## <angle name>
Tried: rounds N1, N2 (commits <sha1>, <sha2>)
Variations: <verbatim>
Why it doesn't work: <verbatim>
What would change the verdict: <verbatim>
```

If `best_state.md` lists an accepted round with no matching entry in `active_opts.md`, add a minimal new entry rather than skip. Commit the normalized files (`report: normalize memory files`) before Step 3.

## Step 3 — Draft `memory/report_narrative.md`

Fill every section. Missing sections fall back to generic mechanical defaults that degrade the report.

```markdown
## hero_tagline
<one short sentence — becomes the <h1>. **Use the `{speedup}` placeholder
for the speed number — do not hardcode it.** The renderer fills it with the
**median** speedup across all tiers (not the best tier — that would mismatch
the stat tile below the headline). Embed as `<em>{speedup}×</em>` for the
gold italic styling. `<br>` for line break. Other placeholders available:
{speedup_pct} / {baseline} / {baseline_min} / {final} / {tier} / {task}.>

## target_signature
<pre-formatted code block with syntax-highlight spans. Classes:
  .tk-ns  namespace/package    .tk-fn  function name
  .tk-op  operator             .tk-arg argument name
  .tk-val literal value        .tk-str string literal
Preserve indentation; multi-line OK.>

## target_meta
<short line under the code: "Language · Ecosystem · domain". HTML allowed
for `<span class="dot-sep">·</span>` separator dots.>

## tier_descriptions
<one `name: description` line per dataset in `task.yaml`.>

## why_faster_intro
<optional one-line lead under "Why is it faster?". Omit unless you have a
single illuminating story to call out; otherwise the renderer synthesizes
one from round_angles.>

## round_angles
<One line per accepted round, classified against the 17-angle taxonomy
below. Format:

    round <N>: <angle-num> <angle-name> [<tier>] [+ <sec-num> <sec-name>] — <one-line rationale>

Tier is `T1`/`T2`/`T3`. Add `+ <sec>` only for genuine composite mechanisms
(e.g. Python→numba port + parallelize). Use `FLAG: <self-label>` for
genuinely out-of-taxonomy rounds (build fixes, portability, measurement-only).

T1 LOCAL / SAME-OUTPUT (bit-exact or fp-noise <1e-10)
  1  eliminate          skip dead / redundant / unused work
  2  overhead           cheaper primitive; bypass dispatch; fewer copies
  3  reuse              cache, memoize, hoist loop-invariants
  4  setup-process      pre-warm / pre-compile inside the process
  5  setup-disk         cross-process disk cache
T2 STRUCTURAL / SAME-MATH (math-equivalent restructuring)
  6  representation     sparse/dense, dtype, layout, streaming
  7a analytical         closed-form / math identity / drop state-independent
  7b better-algorithm   complexity downgrade / fundamentally different algorithm
  7c fusion             fuse multiple kernels/loops into one
  7d vectorize          per-row apply → batched primitive; library backend swap
  8  compile-source     write Rcpp / numba / Cython for the hotpath
  9  compile-build      compiler flags / byte-compile / alt BLAS
  10 parallel           multi-process / multi-thread / GPU
T3 LOSSY / BOUNDED (output drifts within tolerance bands)
  11 convergence        loosen iteration cap / eps / early stop
  12 precision          float32 / low-precision intermediates / -ffast-math
  13 approximation      Wald instead of LR / top-K / prefilter / drop correction
  14 subsample          fewer draws / candidates / poses

Reference: `autozyme-framework/paper/figures/fig3/tier_taxonomy_and_mechanisms.md`.>
```

Keep `report_narrative.md` under ~150 lines.

## Step 4 — Render

```bash
zyme report --open
```

Reads `results.tsv` + `memory/*.md` + narrative, writes self-contained `report.html` (~150 KB), opens in browser.

## Step 5 — Verify

Spot-check:
- Hero speedup matches `best_state.md` "Latest keep".
- Target card syntax highlighting renders (not raw `<span>` tags).
- All `task.yaml` datasets appear in the tier chart.
- Donut tier counts sum to the keep count.
- Top 3 discoveries are **diverse** (not near-duplicates of the same finding).
- Dark mode toggle doesn't make anything illegible.

If anything looks off, edit `memory/report_narrative.md` and re-run.

## Don't

- Don't hallucinate concordance values, round numbers, or commit SHAs — those come from `results.tsv`.
- Don't expand the narrative schema with new sections.
- Don't run `R_report` between iterations.
