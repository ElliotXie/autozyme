# autozyme `astropy_boxleastsquares` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `astropy.timeseries.BoxLeastSquares.autopower`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `BoxLeastSquares(t=<data>, y=<data>, dy=<data>).autopower(duration=<data 3-element array, e.g. [0.018, 0.036, 0.072] days>)  # all algorithmic args left at upstream defaults: objective=None(->"likelihood"), method=None(->"fast"), oversample=10, minimum_n_transit=3, minimum_period=None, maximum_period=None, frequency_factor=1.0`
- **Supported scope:** The patch replaces the native kernel astropy.timeseries.periodograms.bls.methods.bls_fast (the hot function under BoxLeastSquares.autopower/.power for method='fast'). It is an embarrassingly-parallel split of the upstream bls_fast over the trial-period axis: it slices period[chunk] into n_workers disjoint contiguous ranges, calls the captured upstream _orig_bls_fast (bls_impl) on each slice with the SAME (t, y, ivar, duration, oversample, use_likelihood) it received, then re-assembles the 7-tuple result by field-wise np.concatenate in original period order. Because each trial period's BLS computation in bls_impl is independent of every other period, this reproduces upstream bit-exactly (speedups_finalized.tsv shows pearson_power=1.0, q99_rel_diff_power=0.0, rel_peak_period_err=0.0, top10_peak_jaccard=1.0). Crucially, the patch receives oversample and use_likelihood as already-resolved arguments and passes them through unchanged, so it correctly handles ANY objective ('likelihood' or 'snr'), ANY oversample>=1, ANY duration array, and any period grid produced by autopower/autoperiod (any minimum_n_transit/minimum_period/maximum_period/frequency_factor). method='slow' is NOT patched (only bls_fast is in targets=), so method='slow' transparently uses unmodified upstream bls_slow. The parallel path only engages when len(period) >= 2*16384 (32768) AND the resolved worker count (_thread_count) >= 2; otherwise (single-thread env or small grid) it returns the original bls_fast result unchanged. Threads resolve from ZYME_THREADS/AUTOZYME_THREADS/AUTOZYMER_THREADS/OMP_NUM_THREADS, else os.cpu_count()+6.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

