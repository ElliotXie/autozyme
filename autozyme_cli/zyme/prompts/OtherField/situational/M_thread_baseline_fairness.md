# M — Thread baseline fairness audit

You are an expert performance engineer. This task may have an unfair-baseline bug; your job is one round of audit, diagnose, and (if applicable) retrofit. Do not iterate, do not start a new optimization round.

Examples below sometimes reference biology / R libraries because the framework grew up against that domain. Substitute the analogues from your target.

## The pattern

`reference.{R,py}` was written without threading. Later in iteration someone added a parallel construct inside `pipeline/run.{R,py}` — `mclapply`, `BPPARAM`, `future`, `joblib`, `multiprocessing`, custom thread pools, …. Result: optimized side gets N cores, baseline gets 1. `speedup_pct` mixes algorithmic gains with raw parallelism — the headline is dishonest by roughly a factor of N.

`zyme verify` does NOT catch this. Verify re-times the pipeline across thread counts but compares against the same single-threaded reference. The smoking gun lives in `verify.tsv`: for one tier across the thread axis, `baseline_speed` should change as `thread` increases IF the reference honors threading. If it's flat (within ~1.1×), the reference doesn't see the threads — confirmed.

## When this prompt does NOT apply

Exit immediately if any are true. Write one line to `memory/discoveries.md` (`## NOT-APPLICABLE: thread-baseline-fairness — <reason> — <date>`) and end your turn.

- `pipeline/run.{R,py}` has no threading: grep `mclapply|parallel::|BPPARAM|MulticoreParam|future|furrr|foreach|%dopar%|joblib|n_jobs|num_workers|num_threads|workers=|mc.cores|OMP_NUM_THREADS|MKL_NUM_THREADS|RcppParallel::setThreadOptions|numba.set_num_threads|torch.set_num_threads|tf.config.threading|rayon|tbb`.
- Threading was already in `pipeline/run` at the very first commit (`git log -p pipeline/run.{R,py}` shows it from the start). That's a known asymmetry from day one, not the bug we fix here.
- `verify.tsv` already shows `baseline_speed` changing ≥1.3× between thread=1 and thread=max for the same tier. The reference is already honoring threading.

## The decision

Read upstream source under `upstream_repo/` (clone if missing). For `task.yaml::target_function`:

> Does it expose a built-in mechanism for parallelism — an argument or honored env var that an upstream user could engage **without modifying the function**?

Common shapes: argument-level (`n_jobs=`, `nthreads=`, `mc.cores=`, `parallel=TRUE` + `BPPARAM=`), pool injection (`cluster=`, `executor=`), process-scope setter called before the work (`torch.set_num_threads`, `RcppParallel::setThreadOptions`, `numba.set_num_threads`), env-var driven (`OMP_NUM_THREADS` actually consumed by the function — many compiled libraries respect it; many wrappers don't).

**A — yes, a knob exists.** Original baseline was unfair (we never engaged it). Retrofit below.

**B — no knob.** Pipeline added a parallel layer upstream simply does not have. The speedup is honestly "algorithmic + new parallel layer the user couldn't have without us." No retrofit possible. Document and exit (see Document section below).

Be ruthless about A vs B. Don't fabricate a knob. If the only way to parallelize is to wrap upstream in your own `mclapply` / pool, that's B.

## Retrofit (A only)

K2 has no `mode` axis. The fix is purely about giving the reference its own thread-aware path and recording per-thread baselines.

### Step 1: edit `reference.{R,py}` to read `ZYME_THREADS`

Inside the **timed window**, read `ZYME_THREADS` and branch — serial path when `N == 1`, the upstream parallel knob with `N` workers when `N > 1`.

Argument-level:

```python
import os
N = int(os.environ.get("ZYME_THREADS", "1"))
result = upstream_target(X, y, n_jobs=N) if N > 1 else upstream_target(X, y)
```

Process-scope setter (call BEFORE the timed window):

```python
import os
import torch
N = int(os.environ.get("ZYME_THREADS", "1"))
if N > 1:
    torch.set_num_threads(N)
# inside timed window:
result = upstream_target(X)
```

Env-var driven (set at top, BEFORE importing the library):

```python
import os
N = int(os.environ.get("ZYME_THREADS", "1"))
os.environ["OMP_NUM_THREADS"] = str(N)
import upstream_lib
result = upstream_lib.target(X)
```

R / BiocParallel:

```r
N <- as.integer(Sys.getenv("ZYME_THREADS", "1"))
if (N > 1L) {
    suppressPackageStartupMessages(library(BiocParallel))
    register(MulticoreParam(workers = N))
    result <- upstream_target(X, parallel = TRUE)
} else {
    result <- upstream_target(X)
}
```

Hard rules:

- The serial branch (`N == 1`) MUST stay byte-identical in behavior to what the script did pre-edit. For deterministic tasks, evaluate.{R,py} will catch any drift on the next pipeline run.
- The parallel branch must engage the **upstream knob** (the one you found in the decision step). If the only way to parallelize is wrapping upstream in your own pool, you're in Outcome B — stop and re-read.
- Don't change anything outside the timed window — same imports (except adding the threading library if needed), same data loading, same output writing.

Commit:

```bash
git add reference.{R,py} && git commit -m "reference: read ZYME_THREADS, engage <upstream knob> when N>1"
```

### Step 2: declare baseline thread points in `task.yaml`

Pick the thread points you want fair baselines at. At minimum match what `zyme verify` sweeps (typically 1, 4, 8). Add to `task.yaml`:

```yaml
baseline_threads: [1, 4, 8]
```

(If `baseline_threads:` is already there, just edit the list.)

### Step 3: re-bench baselines + repoint verify.tsv

Single command. Re-times reference at each (tier, thread) and updates the existing `verify.tsv`'s `baseline_speed` and `speedup_pct` columns at every cell whose `(tier, thread)` matches. **Pipeline timings are NOT re-run** — they were already correct, only the divisor was wrong.

```bash
zyme baseline rebench --tiers all --threads 1,4,8
```

For specific tiers / different reps:

```bash
zyme baseline rebench --tiers small,medium,large --threads 1,4,8 --reps 1
```

The CLI:
1. Records each (tier, thread) baseline as a row in `results.tsv` (creates / updates as needed).
2. Walks `verify.tsv`, for each cell where `(tier, thread)` matches a freshly-recorded baseline: writes the new `baseline_speed` and recomputes `speedup_pct`.
3. Re-renders `verify.png/pdf/svg` from the updated TSV.

Cells where the reference OOMs at a high thread count (forking blows up peak memory): record those manually as OOM:

```bash
zyme baseline record --tier <T> --thread <N> --oom
```

## Verify the fix

Inspect the freshly-rendered `verify.tsv` / figure:

- `baseline_speed` must visibly change between thread=1 and thread=N at any given tier — smoking gun gone.
- `speedup_pct` typically drops substantially compared to before. The drop is the parallelism contribution; what remains is the honest algorithmic-only speedup. If what remains is ≤1×, the optimization was almost entirely parallelism — important to surface.

You don't need to re-run `zyme verify`. The pipeline numbers in `verify.tsv` are unchanged (correct as committed); only the baseline column was repointed.

## Document (always required, A and B)

Append one section to `memory/discoveries.md` under today's date covering:

1. The threading construct in `pipeline/run` (file:line + what it wraps).
2. Upstream knob found, or "none — Outcome B".
3. (A only) Per-tier honesty for each thread point: `before <X>× → after <Y>×`, with parallel ~`<X/Y>×` and algorithmic ~`<Y>×` split out.
4. Which speedup the project's headline number for this task should reference (1-thread? 8-thread?).

End your turn with a 3-5 line summary. Don't start a new optimization round, don't touch `pipeline/run` or `task.yaml::metrics`.
