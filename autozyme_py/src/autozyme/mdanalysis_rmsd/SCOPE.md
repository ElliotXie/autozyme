# autozyme `mdanalysis_rmsd` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `MDAnalysis.analysis.rms.RMSD.run`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `R = rms.RMSD(u, ref, select="backbone", ref_frame=0); R.run()  — run() called with NO arguments (no start/stop/step/frames, default serial backend). u = mda.Universe(adk.psf, [adk_dims.dcd]*n_concat) where n_concat in {7500(small),22500(medium),45000(large)} repeat pattern, or variant_cycle for ood tiers. Inputs are <data> (AdK DIMS PSF+DCD).`
- **Supported scope:** Correct ONLY for the exact benchmark shape: RMSD over a Universe whose trajectory is a DCDReader or a ChainReader-of-DCDReaders, with NO groupselections (single select-RMSD, output is the 3-column [frame, time, rmsd] array), and run() invoked with no frame slicing (start=None, stop=None, step=None, frames=None) on the default serial backend. The fast_compute bulk path streams frames sequentially from each segment's DCDFile.readframes() and fills output column 0 with np.arange(n) and column 1 with a precomputed cumulative-time vector. This is bit-exact to upstream (pearson_r=1.0, max_abs_diff=0.0 in finalized rows) at ~3.3-4.1x speedup. The has_groups branch (groupselections present) is handled by delegating per-frame to the original _single_frame, so that case is preserved. A chunked per-frame fallback covers non-ChainReader / non-DCD readers (e.g. XTC) and is also generic-correct for the no-slicing, no-groups case.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

