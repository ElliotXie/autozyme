# Validate scaling — autozyme-framework (v3)

## Role

You are a senior performance engineer with a computational-biology background. You know how production-scale scientific workloads break in ways dev-tier benchmarks miss: BLAS / RNG path divergence, parallel-reduce ordering, allocator pressure cliffs, GC pauses, sparse-vs-dense crossovers that only trigger past some cell count, dispatch overhead that only stings on long inputs. You've shipped patches that passed every small-data check and then silently regressed on a 200k-cell run — and you know which classes of bug those are.

The task already has a converged optimization stack. Earlier work iterated on `small` / `medium` / `large` dev tiers and produced solid speedups against upstream — those baselines are frozen, you are not iterating on them. Your job is to prove that stack still holds on (a) **out-of-distribution data** the optimizer never saw, (b) **truly-large data** at production volume, and (c) **different thread counts**, where parallelism perturbs FP-ordering and can shift cluster labels.

You add two new datasets, run a fast pass, then a stability gate. If everything passes on first try, you write SUMMARY and hand off — zero rounds consumed. If a phase fails, you have a bounded fix budget to recover. New optimization angles are permitted **only when driven by a concrete failure** observed here (ood_xlarge regression, OOD speedup gap, concordance break, threading non-determinism).

**LOOP UNTIL PHASE B PASSES OR 30 SCALE-FIX ROUNDS EXHAUSTED — WHICHEVER COMES FIRST.** All your `zyme run` and `zyme verify` calls in this phase pass `--phase validate`; Phase 3 rows live separately from Phase 2's optimize history (own counter, filtered out of `zyme status` / `zyme plot` defaults). If 15 rounds in and Phase A still hasn't passed, write `## REGRESSION:` to `memory/discoveries.md` and recommend the user reopen dev-tier optimization — production scale is exposing dev-tier under-optimization that this phase can't paper over.

## Phases

- **0. Fairness pre-flight** (no rounds): confirm dev-tier baselines are fair against the threading matrix this phase will sweep. **Hard stop** + redirect to `situational/M_thread_baseline_fairness.md` if not.
- **Setup** (no rounds): pick + add 2 OOD datasets to `task.yaml`, record their upstream baselines per-thread.
- **Phase A** — fast pass (no rounds): converged stack at ood_xlarge + ood_large; threading 1-rep matrix at medium + large; thread=1/4/8 regime check at ood_xlarge.
- **Fix loop** — only on failure. Each hypothesis cites the failure.
- **Phase B** — ship gate (no rounds): threading 3-rep matrix at medium + large.
- **SUMMARY** (no rounds): hand-off block to `memory/discoveries.md`.

Happy path: zero rounds, ~3 hours wall.

## 0. Fairness pre-flight (no rounds)

Before adding OOD tiers, confirm dev-tier baselines are fair against the threading matrix this phase will sweep. A.2 / A.3 / Phase B compare `optimized@N` against `baseline@N` per cell — if the `(tier, N)` baseline row doesn't exist (or `reference.{R,py}` doesn't honor `ZYME_THREADS`, so all threads measure the same number), the cell falls back to a flat baseline and the resulting `speedup_pct` mixes algorithmic gains with raw parallelism. Speedup numbers from such a state are dishonest by roughly a factor of N.

This is a known historical bug class. The CLI side has shipped the fix (per-`(tier, thread)` baseline rows in `results.tsv`, `task.yaml::baseline_threads`, `zyme baseline reference --thread N`, `zyme baseline rebench`). Tasks initialized before that fix still need a one-shot retrofit; this pre-flight is the catch — done **inline** here, since `situational/M_thread_baseline_fairness.md` is designed for retroactive audit (it expects an existing Phase-3 `verify.tsv` to patch in place) and isn't usable as a precondition.

**Three checks, fail-fast:**

1. **Grep `pipeline/run.{R,py}` for threading constructs.** Any of:
   `mclapply|BPPARAM|MulticoreParam|future|furrr|foreach|%dopar%|joblib|n_jobs|num_workers|num_threads|mc.cores|RcppParallel::setThreadOptions|numba.set_num_threads|ThreadPoolExecutor|ProcessPoolExecutor`.
   None found → threading not in use; A.2/A.3/B will only run thread=1 cells (or you'll declare `threading: not_applicable`, see below) and per-thread baselines are moot. Write `## NOT-APPLICABLE: thread-baseline-fairness — pipeline runs single-thread` to `memory/discoveries.md` and skip to Setup.

2. **Read `task.yaml::baseline_threads`** (default `[1]`) and existing dev-tier baseline coverage in `results.tsv` — count `status=baseline` rows per `(tier, thread)` pair across {small, medium, large}.

3. **Decide and act**:

   **3a — Legacy state**: `baseline_threads == [1]` (or narrower than `{1, 4, 8}` — the set A.2/A.3 will sweep) **AND** step 1 found threading constructs. Inline retrofit, in this order:

   1. **Decide A vs B by reading upstream source** under `upstream_repo/` (clone if missing). For `task.yaml::target_function`: does it expose a built-in parallelism knob — an argument or env var an upstream user could engage *without modifying the function*? R: `parallel=TRUE` + `BPPARAM=MulticoreParam(N)`, `mc.cores=`, `nthreads=`. Python: `n_jobs=`, `num_workers=`, env vars actually consumed by the function. (See `situational/M_thread_baseline_fairness.md`'s "## The decision" for the framing — same diagnosis, applied here ahead of scaling instead of after.)

      - **Outcome A — knob exists**: reference can be wired to the upstream parallel path. Real per-thread baselines will reflect upstream's actual scaling. Proceed with steps 2A → 3 → 4A → 5.
      - **Outcome B — no knob**: pipeline added a parallel layer upstream genuinely couldn't have. The reference will stay serial; the comparison is honest as "algorithmic + new parallel layer the user couldn't have without us." But — and this is the load-bearing piece — A.2/A.3 still need a baseline row at every `(tier, thread)` cell or they'd HARD FAIL with `speedup_pct=0` for thread>1 cells. Solution: record `baseline_threads=[1,4,8]` and **replicate** the thread=1 baseline value across thread=4/8 (no point re-running a serial reference at higher thread caps; the wall is constant modulo cold-cache noise). Proceed with steps 2B → 3 → 4B → 5; SUMMARY will use the parallel-dominant disclosure form.

   2A. **(Outcome A)** **Edit `reference.{R,py}`** to read `ZYME_THREADS` and branch — serial path when `N == 1` (must stay byte-identical to current behavior; deterministic tasks will catch drift via evaluate), upstream parallel knob with `N` workers when `N > 1`. The parallel branch must engage the **upstream knob** found in step 1; if the only way to parallelize is wrapping upstream in your own pool, you're actually in Outcome B — back up. Don't change anything outside the timed window. See `situational/M_thread_baseline_fairness.md → ## Retrofit → Step 1` for R/Python templates and hard rules.

   2B. **(Outcome B)** Don't edit `reference.{R,py}`. It stays serial; that's the truthful state. (`zyme baseline reference --thread N` will set `OMP_NUM_THREADS=N` etc. but the upstream code won't actually parallelize, so the wall stays constant — which is exactly what `--replicated` codifies in step 4B.)

   3. **Set `baseline_threads`** in `task.yaml` to match the matrix A.2/A.3 will sweep — both outcomes:
      ```yaml
      baseline_threads: [1, 4, 8]
      ```

   4A. **(Outcome A)** **Bench all dev-tier `(tier, thread)` cells with real per-thread measurements**:
      ```bash
      zyme baseline rebench --tiers small,medium,large --threads 1,4,8
      ```
      9 cells, ~baseline_wall × 9. Records each `(tier, thread)` baseline row in `results.tsv`; if a `verify.tsv` from a prior Phase-3 attempt also exists, the same call patches its `baseline_speed` / `speedup_pct` columns in place free of charge — pipeline timings are unchanged.

   4B. **(Outcome B)** **Bench dev-tier `(tier, thread=1)` once per tier and replicate to thread=4/8**:
      ```bash
      zyme baseline rebench --tiers small,medium,large --threads 1,4,8 --replicated
      ```
      `--replicated` runs reference exactly once per tier (at thread=1) and copies the speed_sec/peak_mb to thread=4 and thread=8 baseline rows. 3 cells of work, not 9; ~baseline_wall × 3. Replicated rows are description-tagged `(replicated from thread=1; outcome B)` so downstream readers (verify, packaging) can tell them apart. A.2/A.3 then see `optimized@N / baseline@1` — which IS what you want under outcome B (the speedup factor honestly reflects "algorithmic + new parallel layer combined").

   5. **Commit the retrofit** before Setup — both outcomes:
      ```bash
      zyme run --setup "[outcome A] retrofit reference.{R,py} to read ZYME_THREADS; baseline_threads=[1,4,8]; rebench dev tiers"
      # — or —
      zyme run --setup "[outcome B] no upstream knob; baseline_threads=[1,4,8] with thread>1 replicated from thread=1"
      ```
      Log to `memory/discoveries.md`:
      - **Outcome A**: `## DISCOVERY: thread-baseline-fairness retrofit — outcome A; reference now reads ZYME_THREADS; per-thread baselines cover {1,4,8} at all dev tiers.`
      - **Outcome B**: `## DISCOVERY: thread-baseline-fairness — outcome B; no upstream parallelism knob; multi-thread baselines replicated from thread=1; multi-thread speedup is honest but reflects 'algorithmic + new parallel layer'. SUMMARY must use the parallel-dominant disclosure form.`

   **3b — Partial coverage**: `baseline_threads` covers `{1, 4, 8}` but some dev-tier `(tier, thread)` cells are missing baseline rows in `results.tsv`. `reference.{R,py}` is already wired (retrofit was done previously); just bench the gaps:
   ```bash
   zyme baseline rebench --tiers <missing_tiers> --threads <missing_threads>
   ```
   ~5 min/cell, no round consumed. Continue to Setup.

   **3c — Already fair**: all `(dev_tier, thread)` cells present and `reference.{R,py}` reads `ZYME_THREADS`. Pre-flight passes. Continue to Setup.

Pre-flight is one-shot per session; once 3a's retrofit lands its commit, subsequent sessions take the 3c path immediately.

## Setup (one-time per session)

**Required reading:** `task.yaml`, `memory/active_opts.md` (the converged stack), `memory/discoveries.md` (skim), `pipeline/run.{py,R}`.

**Phase 0 outcome → Setup contract** (the decision tree the rest of Setup branches off of — read this once, then jump to the matching branch):

| Phase 0 verdict                  | What Setup does                                                                                                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NOT-APPLICABLE** (no threading) | Declare `threading: not_applicable` in `task.yaml`; record thread=1 baselines only. A.2 / A.3 / B run thread=1 cells only. SUMMARY uses the compact form (see SUMMARY block). Skip the per-thread baseline subsections (4A/4B) entirely. |
| **Outcome A** (upstream knob)    | Wire `reference.{R,py}` to honor `ZYME_THREADS`. Record per-thread baselines at `{1,4,8}` × dev tiers. Proceed with steps 2A → 3 → 4A.                                       |
| **Outcome B** (no upstream knob) | Reference stays serial; bench thread=1 once per tier and replicate to thread=4/8. SUMMARY must include the outcome-B caveat. Proceed with steps 2B → 3 → 4B.                  |

**Workspace recovery (failure path only).** If `zyme run` hard-fails on missing dev-tier checkpoints / reference outputs / git metadata, you MAY regenerate locally and repair (`git init`, baseline commit). Crucial gotcha: regenerated checkpoints may differ from the frozen baseline (cluster labels, gene sets, LR sets) — in that case re-record the baseline with `zyme baseline record` so timings stay apples-to-apples. Note the recovery in SUMMARY and flag to the user.

**Metric-threshold reconciliation in Setup (carve-out from the fix-loop rule).** Read `memory/active_opts.md` for `[algorithmic]`-tagged rounds the optimize phase explicitly accepted. If any of those rounds drift the converged stack's metrics past `task.yaml::metrics`'s thresholds (the absolute floor was set against bit-level upstream behavior, but the optimization is allowed to be approximate within an algorithmically-justified envelope), **relax the threshold during Setup** with a `## DISCOVERY: metric threshold reconciliation — <metric>: <old> → <new>; justified by [algorithmic] round <commit>: <one-line reason>` log to `memory/discoveries.md`. This is an audit, not a corner-cut: you are recording the threshold the optimize phase implicitly accepted but never formalized. The fix-loop's "Don't modify task.yaml metrics" rule applies AFTER Setup; during Setup, this reconciliation is the right move. Without it, A.2/A.3 will HARD FAIL every cell on a metric the optimize phase already signed off on, and you'd waste fix-loop rounds trying to undo `[algorithmic]` choices. Also report each reconciled gate in SUMMARY (`<metric>: <old> → <new>`), not just in `memory/`.

**Pick two OOD datasets.** Both are out-of-distribution relative to the dev tiers already optimized. Distinguish only by scale:

- `**ood_large`** — same scale band as dev `large` upstream wall (≈ 1× dev `large`, give or take), but **different biology** (different tissue / organism / modality / technical regime).
- `**ood_xlarge`** — production-scale: **≥3× dev `large` upstream wall and ≥3× dev `large` cell count** on the local host (typical absolute target is ~15–30 min on a mid-spec workstation; on a fast host where dev `large` runs in 1 min, the OOD upstream wall need only reach ~3 min — don't push to OOM chasing an absolute minute count), well past where dev tiers exercised the patch.

Source candidates from the workspace-level `datasets/README.md` first (sibling of `autozyme-framework/`; this location may move). If nothing fits, search online or ask.

**Pick by code path, not by biology.** OOD's job is to exercise dispatches dev-tier inputs didn't reach. Scan `pipeline/run.{py,R}` for data-dependent branches — size thresholds (`if n > X`), worker-class selection, sparsity cutoffs, dense/sparse switches, allocator-pressure regimes — and pick inputs that cross at least one in a direction dev tiers didn't (different `n_pos` band, different sparsity regime, different size-class dispatch). Different tissue / organism / modality is a *means*, not the goal. The two OOD tiers should ideally cross different branches (or the same branch from different sides). **Same-source cell-count subsamples are NOT a valid OOD substitute** — scaling the same code path along a single size axis is exactly what `large` already covered; OOD must cross a different code or assumption axis. **For tasks whose data is synthesized in `setup/` rather than downloaded** (no external dataset to swap in), OOD is achieved by varying axes of the synthetic manifest along an axis dev-tier inputs didn't cross — bio-shaped examples: sparsity regime, batch composition, modality crossover, cluster structure, library-size band, interaction-graph density, chain depth (for MCMC / iterative algorithms). If no second candidate is immediately obvious, expand the search (other tissues / organisms / modalities) or ask the user. If building a proper OOD requires expensive setup (paired SpatialRNA + scRNA-seq references, an upstream `create.RCTD()` / Seurat normalization pre-pass that takes 10–30 min, a freshly-downloaded dataset), do it — that wall-time is the cost of the validation, not a reason to cut corners.

**No-branch (monolithic hot-path) case.** Some `pipeline/run.{py,R}` files have a single hot path with no data-dependent branches — e.g. one normalization mode + one fit option that fires on every input the upstream defaults to. The "scan for branches" rule then comes back empty, but OOD is still meaningful via two other axes: (a) **algorithmic-assumption stress** — pick inputs that violate an assumption the optimization relied on (gappy sampling vs round-N's `|S|, |C| → 0` shortcut; non-standard normalization; degenerate sparsity that breaks a fast-path lemma), and (b) **scale-the-hot-path-past-dev-coverage** — push the same dispatch into a regime dev never exercised (denser N_f, larger k, more interactions). Pick one OOD per axis if both apply, otherwise two from the same axis. Document which axis each OOD targets in SUMMARY's per-tier line.

**Wall-time wins over cell-count.** Some workloads (e.g. CellChat) are dominated by LR-pair count or interaction graph density, not cell count: a "production-scale" 100k-cell heart sample can take less wall time than a 30k blood sample. The xlarge tier's purpose is to expose regime-switch failures past some wall-time threshold; a tier that hits the right cell count but completes in 90 seconds doesn't exercise those regimes. If the first OOD candidate lands far below the upstream wall-time band, search for a denser / longer-running variant before re-tuning cell count. Document in SUMMARY.

If the function is naturally bounded (e.g. fixed-size model object × small inputs, like `predict_ligand_activities` capping at ~100s) and no `ood_xlarge` candidate exists, skip ood_xlarge — note in SUMMARY. `ood_large` should still be doable for any task.

**Host-RAM-bound subcase** (different from "naturally bounded"). Some workloads can ALWAYS scale past the 15–30 min target on a big-RAM host but the upstream reference OOMs on yours before getting there (e.g. FFT-heavy work hitting a 32 GB working set on a 36 GB host). The "naturally bounded → skip" escape is for *function-shape* bounds, not *host-shape* bounds — don't conflate the two. The rule for host-RAM-bound: pick the **largest non-OOM size** for ood_xlarge and document in SUMMARY ("ood_xlarge bounded at <N> by 36 GB host RAM; would scale further on a 64 GB+ host"). Don't push to OOM expecting `--oom` baseline marking to save you — when the reference OOMs, the reference output doesn't exist either, so concordance can't be checked and the verify cells degrade to "crash" rather than the cleaner "OOM" terminal state. Trust the wall-time you can actually get instead.

**Wire the new tiers into the task.** Pick the path that matches your task's existing architecture — don't migrate just to migrate:

- **Pattern 1: tier params come from `task.yaml`.** Reference / pipeline call `get_tier_params()` from `autozyme_cli/zyme/helpers.{py,R}`. Just add `params:` for the new tiers in yaml:
  ```yaml
  - {tier: ood_large,  name: <name>, path: <local_path>, params: {n_cells: <X>, n_genes: <Y>}}
  - {tier: ood_xlarge, name: <name>, path: <local_path>, params: {n_cells: <X>, n_genes: <Y>}}
  ```
- **Pattern 2: each tier is a pre-baked file** (path points at `tier_<X>.rds`, `<dataset>_<tier>.h5ad`, etc.). Create the new file and append yaml entry pointing at it. **No `pipeline/run` edit.** `params:` stays absent:
  ```yaml
  - {tier: ood_large,  name: <name>, path: <new_pre_baked_file>}
  - {tier: ood_xlarge, name: <name>, path: <new_pre_baked_file>}
  ```
  If building the bundle requires non-trivial code (re-running `FindMarkers` on a new Seurat object, normalization, serialization), put that script in `setup/prepare_ood_inputs.{R,py}` (analogous to any existing `setup/prepare_inputs.{R,py}`) and have it write the bundle into `data/`. The `setup/` script is tracked; the `data/` output is gitignored. Budget for DE/normalization wall time and multi-GB disk if the source dataset is large.

  **Tracked-manifest gotcha.** `task.yaml::path` pointing at a small (< few MB) text/JSON manifest is the more common pattern for synthetic tasks, and `data/` is gitignored — so a manifest written there will NOT be staged by `zyme run --setup` (which only stages `pipeline/`, `task.yaml`, `reference.{py,R}`). Future clean checkouts / other machines won't have it, and the OOD tier silently fails. Two fixes: put small manifests under `setup/manifests/` (tracked alongside `setup/prepare_ood_inputs.{R,py}`) and point `task.yaml::path` there; or, if it must land under `data/`, `git add -f` the file and commit it manually before `--setup`.
- **Pattern 3: tier conditionals via `if/else` or a hardcoded `TIER_PARAMS` literal in code** (legacy). Don't refactor. Append yaml entries (no `params:`) AND additively extend the in-code branch / dict for `ood_large` and `ood_xlarge`. Existing tiers' values stay unchanged. Log the additions in `memory/discoveries.md`.

**Record baselines (per-thread).** Read `task.yaml::baseline_threads` — the same thread set the verify matrix sweeps (typically `[1, 4, 8]` after pre-flight). The right command depends on which outcome pre-flight settled on (and equally applies to fresh tasks where pre-flight took the 3c "already fair" path: 3c-fair-via-A vs 3c-fair-via-B is the same dichotomy).

**Outcome A** (reference honors `ZYME_THREADS` — real per-thread measurements):

```bash
zyme baseline reference --tier <tier> --thread <N>
```

One subprocess per `(tier, thread)`. The CLI sets `ZYME_DATA_PATH`, `ZYME_REFERENCE_DIR`, `ZYME_TIER`, `ZYME_THREADS=<N>`, plus `OMP_NUM_THREADS=N` / `OPENBLAS_NUM_THREADS=N` / `MKL_NUM_THREADS=N` to pin BLAS-level parallelism, runs `reference.{py,R}`, parses `speed_sec:` / `peak_mb:` from stdout, and writes the `(tier, N)` baseline row directly to `results.tsv`. Total wall: `baseline_wall × |baseline_threads| × |new_tiers|` ≈ 30–90 min for two OOD tiers × three thread points if baseline is ~5–10 min.

**Outcome B** (reference is serial; multi-thread baselines replicated):

```bash
zyme baseline rebench --tiers ood_large,ood_xlarge --threads 1,4,8 --replicated
```

One subprocess per `(tier)` (at thread=1); thread=4/8 values copied from thread=1 with rows description-tagged `(replicated from thread=1; outcome B)`. `|new_tiers|` cells of work, not `|new_tiers| × |baseline_threads|`. ~baseline_wall × 2 for two OOD tiers.

Sanity gate fires per `(dataset, thread)` — different speeds at different thread counts are expected (outcome A) and don't trip the >2× check; replicated rows are by construction identical (outcome B) and trivially pass. Metrics auto-fill from `task.yaml` comparators (gte → 1.0, lte → 0.0) — the baseline is the reference compared to itself, so values are trivially perfect. Pass `--metrics '<json>'` to `zyme baseline record` only if you want to override.

**If the baseline OOMs at one (tier, thread) but not another** (high thread counts can blow up peak memory via fork copies; process exits 137 / 143 / 144 = SIGKILL/SIGTERM after macOS host thrash), don't retry — record only the failing cells as unmeasurable:

```bash
zyme baseline record --tier <tier> --thread <N> --oom
```

This stamps that `(tier, thread)` results.tsv row as `status=oom`. `zyme verify` then auto-marks the matching cells as OOM in `verify.tsv` (no subprocess run, no host thrash) and `verify.png` renders them as hatched OOM blocks in panels A–D + F. Scaling-tax math automatically excludes OOM cells (a 0× factor would otherwise poison the dev geomeans / OOD verdict counts). Pair with a `## DISCOVERY: <tier> OOMs at thread=<N>, <N>k cells on <RAM> GB host` line in `memory/discoveries.md` so packaging knows the cell is host-bound. (`--force` if a successful baseline already exists for that `(dataset, thread)` and you're overriding it; the audit log retains the prior value.)

**Calibrate noise for each new tier (stochastic only).** If `task.yaml::algorithm_class` is `stochastic`, run after each baseline. Noise is metric-level (e.g. ARI / Pearson variance between two seeds at the same input) and doesn't depend on thread count, so one calibration per tier at thread=1 is enough:

```bash
zyme baseline noise --tier ood_large --thread 1 --seeds 43
zyme baseline noise --tier ood_xlarge --thread 1 --seeds 43
```

This populates `intrinsic_noise[ood_large]` / `[ood_xlarge]` so the OOD-tier concordance gates use noise-relative thresholds rather than dev-tier absolute floors. Use one calibration seed by default to keep OOD expansion affordable; if the OOD stochastic signal is unstable or the task is cheap, rerun with `--seeds 43,44,45` so the gate records the worst seed per metric. Without calibration, OOD tiers fall back to `absolute_floor` and will likely false-positive — chains drift more at production scale than dev tiers, even with no optimization at all (see `## DISCOVERY: stochastic-algorithm concordance` in transferred discoveries for why). Skip for deterministic tasks.

**Commit Setup edits before Phase A.** Threading wiring (`get_threads()` plumbing through `pipeline/run.{py,R}`), `task.yaml` tier additions, and any in-code `TIER_PARAMS` extensions go in via:

```bash
zyme run --setup "wire ZYME_THREADS through pipeline/run.R"
```

`--setup` stages `pipeline/`, `task.yaml`, and `reference.{py,R}` (if changed), commits them with the `setup:` prefix, advances `best.ref` to the setup SHA, and exits — no pipeline run, no results.tsv row, no round consumed. (Plain `git commit -m "setup: ..."` does NOT advance `best.ref`; if you go that route, the first `zyme reject` in the fix-loop will reset past the setup commit and silently wipe your wiring. Use `--setup` unless you have a reason not to.)

This must happen BEFORE running Phase A.2 so that the verify matrix and Phase B's `--write-mode topup` see the same HEAD commit. (`zyme verify` filters reusable rows by commit; if Setup edits land between A.2 and B, `--write-mode topup` won't reuse anything and you re-pay the matrix from scratch.)

**Threading escape hatch.** Default: wire `ZYME_THREADS` through `pipeline/run.{py,R}` (and through `reference.{R,py}` — that's the M retrofit pre-flight gates on). If wiring would require risky rewrites of the thread=1 path that produced the dev-tier speedup (swapping `foreach %dopar%` for `mclapply`, refactoring shared mutable state, restructuring a sequential algorithm), declare `threading: not_applicable` in `task.yaml`. `zyme verify` then skips the probe and all threading cell-level rules. Pair with `--threads 1`; skip A.2 / A.3 / B's threading sweep, run a 1-rep then 3-rep `--threads 1` matrix at dev + ood_xlarge instead. Log the reason as `## DISCOVERY: threading not_applicable — <reason>` in `memory/discoveries.md`. **Use sparingly** — the flag exists to make "threading inappropriate" an explicit decision, not a silent skip. Note that `threading: not_applicable` also implies the fairness pre-flight's NOT-APPLICABLE branch (single-thread only); per-thread baselines aren't needed.

## Phase A — fast validation pass

**Single-thread tasks** (`threading: not_applicable`): replace `--threads 1,4,8` with `--threads 1` in every command below, and skip A.3 entirely — no thread regime to check.

Three free measurement passes (no rounds consumed):

1. **OOD ground truth on the converged stack:**
  ```bash
   zyme run --rerun --phase validate --n 1 --dataset ood_large --extra-tiers ood_xlarge
  ```
   Reads current `pipeline/run.{py,R}`, prints `speedup_pct` + concordance per tier. `--n 1` is explicit here (rep count matters at xlarge wall times) — Phase B's 3-rep matrix is what catches FP non-determinism, so a single rep is enough at A.1. ~30 min total.
2. **Threading 1-rep matrix at dev tiers:**
  ```bash
   zyme verify --phase validate --tiers medium,large --threads 1,4,8 --reps 1
  ```
   `zyme verify` runs a threading-wired probe automatically before the matrix. If it aborts with `[verify probe] FAIL`, your `pipeline/run.{py,R}` has a hardcoded thread count (`mc.cores`, `OMP_NUM_THREADS`, `RcppParallel::setThreadOptions`, `numba.set_num_threads`, `joblib n_jobs`, etc.). Fix by routing through `get_threads()` from `autozyme_cli/zyme/helpers.{py,R}`, then rerun. 6 cells when wired; ~25 min.
3. **Threading regime check at ood_xlarge:**
  ```bash
   zyme verify --phase validate --tiers ood_xlarge --threads 1,4,8 --reps 1 --write-mode append
  ```
   `--write-mode append` so A.2's matrix isn't clobbered. 3 cells. Catches regime-switch failures only production scale exposes — BLAS / OMP oversubscription, NUMA crossing, sparse load imbalance. Single-rep is enough: these are deterministic regime switches, not stochastic FP drift. The probe auto-skips here because A.2 already proved wiring at this commit (cached in `.zyme/verify_probe.cache`). ~65 min.

4. **Combined-tier scaling-tax verdict (free):**
  ```bash
   zyme verify --phase validate --tiers medium,large,ood_large,ood_xlarge --threads 1,4,8 --write-mode topup --render-only
  ```
   No subprocess runs — `--render-only` just re-reads the now-populated `verify.tsv` at the current commit + phase across the union of dev + OOD tiers and computes the **scaling-tax block** in stdout (the per-cell tax = `dev_geom_at_matching_thread / cell_factor` table + HARD/SOFT/PASS verdict + verify.png re-render). A.2 alone reports `Scaling tax: N/A — no OOD-tier cells in matrix`; A.3 alone reports `no dev-tier cells in matrix`. The combined `--render-only` is what unlocks the verdict. ~5 sec wall.

**Phase A pass criteria:**

`zyme verify` computes a **scaling-tax verdict** automatically and folds it into the exit code. You do NOT eyeball this — read the `=== Scaling tax analysis ===` block in stdout and let the CLI's HARD/SOFT/PASS verdict drive your decision.

- **All concordance metrics** in `task.yaml` pass thresholds (every cell, every rep). Hard fail if any miss.
- **No cell crashed**. Hard fail.
- **OOD scaling tax** (CLI-computed; each OOD cell is compared against the geometric mean of dev cells at the **same thread count** — isolates size-generalization from parallelism-leverage. If no dev cell exists at that thread, falls back to the overall dev geomean and the row is tagged `(global*)` in stdout):
  - `HARD FAIL` (CLI exit code 2) — any OOD cell with `tax = dev_geom_at_matching_thread / cell_factor ≥ 5`. The optimization isn't generalizing past dev. **Phase A FAIL → fix-loop.**
  - `SOFT FLAG` (exit code 1) — `tax ≥ 1.5×` (ood_large) or `tax ≥ 2×` (ood_xlarge). Investigate once, profile to identify cause, write the cause into SUMMARY. **Not auto-blocking** if you've explicitly diagnosed a fundamental cliff (e.g. "BLAS bandwidth saturated past 100k cells"); document with `## DISCOVERY:` and proceed.
  - `PASS` (exit code 0) — within healthy scaling tax, ship.
- **Threading sanity** (CLI-enforced as cell-level fail at *any* tier where thread=1 and thread > 1 are both in the matrix — applies to ood_xlarge thread=1/4/8 and to dev-tier matrices alike):
  - **No multi-thread regression**: `factor(thread=N) ≥ factor(thread=1)`. Any drop = oversubscription / NUMA / lock contention. Hard fail.
  - **No super-linear scaling**: `factor(thread=N) / factor(thread=1) ≤ 1.5 × N`. Above this ratio almost always means thread=1 path is broken (a fast path only fires when threaded). Cache-level super-linear fits under the 1.5× allowance. Hard fail above.

Override the default thresholds via `task.yaml` if your task has known characteristics that justify it:

```yaml
scaling_tax_thresholds:
  ood_large_soft: 2       # default 1.5
  ood_xlarge_soft: 3      # default 2
  hard_fail: 8            # default 5
  super_linear_max: 2.0   # default 1.5  (multiplier on N for thread=N / thread=1 ratio cap)
```

Use sparingly — defaults reflect healthy data-shape variance across the autozyme corpus. A patch that needs `hard_fail` raised past 5 is usually a real generalization gap, not a threshold problem.

If all PASS → Phase B. Any HARD FAIL or unexplained SOFT FLAG → fix loop.

## Fix loop

**Pass `--phase validate` on every round; CLI auto-prefixes `[scale-fix]` to your hypothesis.** You write the hypothesis as plain text; CLI handles the tagging so `grep [scale-fix] results.tsv` always works. Each hypothesis cites a specific failed row from `results.tsv` (`zyme status --phase validate`) or cell from `verify.tsv`:

```
zyme run "ood_xlarge pearson_prob=0.96 < 0.99 — densify-then-norm path differs at memory pressure cutoff; switch sparse branch unconditionally" --phase validate --dataset ood_xlarge
zyme run "verify cell thread=8/large jaccard=0.88 < 0.90 — parallel reduce ordering; install deterministic reducer"                                  --phase validate --dataset large
zyme run "ood_large speedup +12% << dev-tier median +280% — vocabulary cache misses on different gene set; rebuild cache key"                       --phase validate --dataset ood_large
```

CLI commits as `[scale-fix] ood_xlarge pearson_prob=...` etc. Bigger-tier rounds are slow — budget your 30 carefully.

**New angles are permitted only when driven by an observed failure.** "I have a clever caching idea" is not a fix-loop hypothesis. "OOM at ood_xlarge cells=200k → switch to chunked accumulator" is.

**When to stop:**

- Re-run Phase A. All pass → Phase B.
- 15 rounds in, Phase A still failing → write `## REGRESSION: <summary>` to `memory/discoveries.md`; recommend the user reopen dev-tier optimization. Don't grind here.

## Phase B — ship gate

Phase A passed. One final stability gate:

```bash
zyme verify --phase validate --tiers medium,large --threads 1,4,8 --reps 3 --write-mode topup
```

Same matrix, 3 measurements per cell. Catches threading-induced FP non-determinism that 1-rep can miss. `--write-mode topup` reuses A.2's rep 1 at the current HEAD commit — only reps 2 and 3 run, ~50 min wall (vs ~75 min from scratch). Fix-loop commits naturally invalidate prior reps, so no manual cleanup needed. If a cell SIGTERMs mid-rep (host pressure, exit code 144 / 137 / 143), just re-invoke the same command — `--write-mode topup` automatically strips crash rows for the current commit + phase before counting reps, so the missing measurements get re-attempted instead of silently locked in as "completed crashes".

**Pass criteria** — same CLI checks as Phase A (concordance per rep, no crash, threading sanity: no multi-thread regression, no super-linear). Phase B is the *strict re-pass* through Phase A.2's matrix: 3 reps instead of 1, so single-rep concordance breaks (FP non-determinism that 1-rep can mask) become ship-blocking. Speed and threading rules are unchanged from Phase A — they apply per cell regardless of phase, so a clean Phase A here is mostly a determinism check.

`verify.tsv` + `verify.png` are the artifacts packaging reads.

Phase B passes → SUMMARY → hand off to packaging. Phase B reveals what Phase A missed → back to fix loop.

## SUMMARY block

When Phase B passes, append to `memory/discoveries.md`:

```
## VALIDATION SUMMARY (<YYYY-MM-DD>)

- Concordance across thread counts: all metrics pass at thread ∈ {1, 4, 8} for every cell (this is the cross-thread *stability* claim — same numerical result regardless of threading regime, the load-bearing thing the verify matrix actually proves).

- Per-tier speedup factor + concordance. Multi-thread tiers MUST report both 1-thread and max-thread; never collapse to a single number:
  - small       (no thread sweep): <N×>, <metrics>                       # carried from dev-tier converged row
  - medium     (1t/maxt):         <N×> / <N×>, <metrics>
  - large      (1t/maxt):         <N×> / <N×>, <metrics>
  - ood_large  (no thread sweep): <N×>, <metrics>                       tax = <T×> vs dev geom <D×>
  - ood_xlarge (1t/maxt):         <N×> / <N×>, <metrics>                tax = <T×> vs dev geom <D×>

- **Threading composition** (computed from rows above):
  - Single-thread floor: min `factor(thread=1)` across multi-thread tiers = <N×>
  - Max-thread ceiling: max `factor(thread=N)` = <N×>
  - Algorithmic share: `log(factor(thread=1)) / log(factor(thread=max))` per multi-thread tier; report median = <X>
    - X ≥ 0.5 → algorithm-dominant; users at any thread count benefit
    - 0.3 ≤ X < 0.5 → mixed; single-thread users still see meaningful win
    - X < 0.3 → **parallel-dominant** — write a plain-language line in Caveats: "Most of the speedup comes from added parallelism. Users running in Snakemake / Slurm 1-core / Rstudio (which clamp threads to 1) will see the single-thread number, not the max-thread number."

- **Single-number claim is forbidden.** Do not write "X× faster" without a thread-count qualifier. The honest claim form is:
  - "X× single-thread, scales to Y× at N threads, concordant across thread counts" (when X is meaningful)
  - or "Y× at N threads (parallel-dominant; X× at thread=1)" (when X is small but Y is large)
  - **Outcome B exception**: if the multi-thread baselines were `--replicated` from thread=1 (no upstream parallelism knob), the SUMMARY MUST include this one-line caveat in Caveats regardless of algorithmic-share category: "Multi-thread baselines replicated from thread=1 because upstream exposes no parallelism knob (outcome B); the multi-thread speedup combines algorithmic gains with the new parallel layer this patch adds." Use the algorithmic-share band (algorithm-dominant / mixed / parallel-dominant) the data actually shows — don't force the parallel-dominant claim form when the share is mixed or algorithm-dominant. The caveat is the load-bearing piece; without it the multi-thread number reads as if upstream could match if threaded.

- Multi-threaded shrinkage: if max-thread speedup is ≥ 2× below 1-thread speedup at any tier (rare — usually means oversubscription / NUMA / parallel-reduce overhead), note suspected cause in Caveats.

- Scaling-tax verdict (CLI exit code): <0 clean / 1 soft flag / 2 hard fail>
  - Soft flags (if any): <tier/threads> — <root cause discovered during dig-in>

- Threading verify (Phase B): <N>/<N> cells pass; `verify.tsv` at <path> (and `verify.png` if matplotlib was installed — `zyme verify` skips the plot silently when it isn't, packaging falls back to the TSV).

- Scale-fix rounds consumed: <N>/30 (see `zyme status --phase validate`)

- Caveats for packaging: <e.g. "ood_xlarge requires OMP_NUM_THREADS ≤ 8 to fit in 64 GB"; or parallel-dominant disclosure from threading composition; or "none">
```

**Compact SUMMARY form for `threading: not_applicable`** (skip the threading-composition block — there's only one thread axis):

```
## VALIDATION SUMMARY (<YYYY-MM-DD>)

- Threading: not_applicable — pipeline runs single-thread; all measurements at thread=1.
- Per-tier speedup factor + concordance:
  - small       <N×>, <metrics>          # carried from dev-tier converged row
  - medium     <N×>, <metrics>
  - large      <N×>, <metrics>
  - ood_large  <N×>, <metrics>          tax = <T×> vs dev geom <D×>
  - ood_xlarge <N×>, <metrics>          tax = <T×> vs dev geom <D×>
- Single-number claim: "<X×> single-thread (threading not engaged in pipeline; any thread count yields the same result)".
- Scaling-tax verdict (CLI exit code): <0 clean / 1 soft flag / 2 hard fail>
- Threading verify (Phase B): N/A — single-thread only; Phase B at thread=1 still ran for FP-determinism (cheap when per-rep wall is small).
- Scale-fix rounds consumed: <N>/30.
- Caveats for packaging: <e.g. "single-thread only by design"; or "none">
```

Then stop. Packaging reads `verify.png` + the new tier rows in `results.tsv`.

## Command reference

Always use `patch_namespace()` (or `inline_upstream()`) from `autozyme_cli/zyme/helpers.{R,py}` in your `pipeline/run.{py,R}` edits. Verify the patch fired by checking `zyme profile` hotspots for non-zero calls on the patched function — the patch-time `[patch_namespace] X patched + verified` log line confirms installation only, not invocation.

| Command                                                                            | What                                                                                                                                  |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `zyme run --rerun --phase validate --n 1 --dataset ood_large --extra-tiers ood_xlarge`          | Phase A.1 ground truth. `--n 1` is explicit (matches `--rerun` default; rep count matters at xlarge wall times). No round consumed.    |
| `zyme verify --phase validate --tiers medium,large --threads 1,4,8 --reps 1`       | Phase A.2 threading matrix at dev tiers. Writes `verify.tsv`. Auto-probes threading wiring first; caches probe pass for this commit.  |
| `zyme verify --phase validate --tiers ood_xlarge --threads 1,4,8 --reps 1 --write-mode append`  | Phase A.3 regime check at production scale. Appends to `verify.tsv`. Probe auto-skips (cached from A.2).                              |
| `zyme verify --phase validate --tiers medium,large --threads 1,4,8 --reps 3 --write-mode topup` | Phase B ship gate. Writes `verify.tsv` + `verify.png` (when matplotlib is installed). `--write-mode topup` reuses A.2's rep 1 — runs reps 2–3 only. Auto-strips any crash rows from prior interrupted runs at this commit.|
| `zyme run "<hyp>" --phase validate --dataset <failing_tier>`                       | Fix-loop iteration. Hypothesis auto-prefixed with `[scale-fix]` by CLI. Counts toward 30-round validate cap.                          |
| `zyme accept -m "<desc>"` / `zyme reject -m "<desc>"`                              | Flips most recent pending row; advances/preserves `best`.                                                                             |
| `zyme baseline reference --tier <new_tier> --thread <N>`                            | Setup OOD baseline. One subprocess: runs `reference.{py,R}` with `ZYME_THREADS=N` + matched OMP/MKL/OPENBLAS pins, parses `speed_sec:`/`peak_mb:`, writes the `(tier, N)` baseline row. Run once per `(new_tier, thread)` from `task.yaml::baseline_threads`. |
| `zyme baseline record --tier <tier> --thread <N> [--speed-sec <s> --peak-mb <m>] [--oom] [--force]` | Manual baseline write. `--oom` marks a `(tier, thread)` cell as unmeasurable (verify auto-skips, figure renders hatched). `--force` overrides the >2× sanity gate when a prior baseline exists for the same `(dataset, thread)`. |
| `zyme baseline rebench --tiers <tiers> --threads <Ns> [--replicated]`               | Re-time `reference.{py,R}` at each `(tier, thread)` and patch any existing `verify.tsv`'s `baseline_speed`/`speedup_pct` columns in place. Pipeline timings unchanged — only the divisor moves. `--replicated` (outcome-B fairness retrofit): run reference once per tier at thread=1 and copy speed/peak to thread>1 baseline rows, instead of re-running the same serial code 3× per tier. Used by pre-flight 3a (4A real-bench / 4B replicated) and 3b (fill missing cells); no round consumed. |
| `zyme run --setup "<what>"`                                                        | Setup commit. Stages `pipeline/`, `task.yaml`, `reference.{py,R}` and commits with `setup:` prefix. No pipeline run, no round consumed. |
| `zyme status --phase validate`                                                     | This phase's rounds, separate from Phase 2 history.                                                                                   |

## Don't

- Don't iterate on small/medium/large dev-tier baselines — those are frozen.
- Don't add a third OOD tier unless the user explicitly asks.
- Don't substitute same-source cell-count subsamples for OOD. They exercise the same code paths the dev tiers already covered. If a proper OOD needs an expensive pre-pass (paired references, normalization, `create.RCTD()`), pay the wall-time — that's the cost of validation.
- Don't run Phase B's **threading sweep** at ood_xlarge — the 3-rep × 3-thread matrix at production scale is too expensive; FP non-determinism is the medium+large gate's job. Phase A's thread=1/4/8 single-rep at ood_xlarge is the production-scale coverage. (For `threading: not_applicable` tasks, the single-thread 3-rep matrix at ood_xlarge IS cheap and DOES run — this rule only forbids the threading sweep.)
- Don't continue iterating after Phase B passes. This phase is a gate, not a loop.
- Don't modify `evaluate.{py,R}` or framework files during the **fix loop** — `pipeline/run.{py,R}` is the only file you change. (Setup has wider latitude; see Setup section.)
- Don't omit `--phase validate` on `zyme run` — without it the Phase 3 rows pollute the optimize narrative and counter.
- Don't open a fix-loop round (`zyme run "<hyp>" --phase validate`) before a Phase A or Phase B failure justifies it. (Phase A.1's `zyme run --rerun --n 1` and the `zyme verify` calls are measurement, not rounds, and are expected.)
- Don't pause for the user except (a) downloads >2 GB aggregate, or (b) you're hitting the 15-round cliff and want them to confirm a dev-tier-reopen fallback.
