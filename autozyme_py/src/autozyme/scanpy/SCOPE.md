# autozyme `scanpy` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `scanpy.pp.highly_variable_genes`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `sc.pp.highly_variable_genes(<data>, n_top_genes=2000, flavor="seurat") — <data> = log-normalized sparse CSR AnnData (pbmc68k etc.)`
- **Supported scope:** The dispatcher _patched_hvg routes by flavor and batch_key. The BENCHMARKED config (flavor="seurat", batch_key=None, sparse CSR log1p-normalized input, numba available) is handled by _fast_hvg_seurat. That fast path correctly supports: flavor="seurat" only; sparse input (auto-converted to CSR float32); both selection modes — n_top_genes (argpartition top-N on normalized dispersion) AND the cutoff mode with min_mean/max_mean/min_disp/max_disp (all four honored, lines 296-299); n_bins (honored, passed into kernel); layer= (reads adata.layers[layer]); subset= and inplace= (both honored, lines 304-321); it stores log1p(mean) for means to match upstream scanpy seurat contract (line 302). Separately, flavor in {seurat_v3, seurat_v3_paper} WITH batch_key set routes to _fast_hvg_seurat_v3_batch (a heavily guarded CSR-raw-counts batch path), but that is NOT the benchmarked path. The eval metric is hvg_jaccard>=0.95 (set overlap of selected genes), tolerant of small numeric drift.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

## `scanpy.pp.highly_variable_genes`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `sc.pp.highly_variable_genes(<data>, n_top_genes=2000, flavor="seurat_v3_paper", batch_key=<dataset batch_key e.g. "donor_id">)`
- **Supported scope:** The v3-batch fast path (_fast_hvg_seurat_v3_batch) correctly handles: flavor in {"seurat_v3","seurat_v3_paper"} WITH a non-None batch_key, on a CSR-sparse raw-count matrix (adata.X or a named layer), with n_top_genes a concrete int (not None), subset=False, inplace=True, and no extra/unknown kwargs. It honors span (passed into loess) and check_values (drives the non-integer warning). It supports multiple batches: per-batch loess fit, per-batch clipped variance, median-rank aggregation, and the seurat_v3 vs seurat_v3_paper lexsort tiebreak ordering (lines 519-522). It writes highly_variable, highly_variable_rank, means (overall), variances (overall), variances_norm, highly_variable_nbatches to adata.var and uns["hvg"]. Numba (parallel prange) must be importable and skmisc.loess must be importable. Requires >=2 obs, >=1 var, all batch sizes >=2, valid (non-negative) batch codes, and >=2 non-constant genes per batch. Anything outside this is delegated verbatim to the captured upstream original (__autozyme_original__). A separate flavor="seurat" non-batch path (_fast_hvg_seurat) also exists but is not the benchmarked target here.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

## `scanpy.pp.normalize_total + scanpy.pp.log1p`

- **In-scope output equivalence:** tolerance
- **Validated at:** `sc.pp.normalize_total(<data>, target_sum=1e4); sc.pp.log1p(<data>)`
- **Supported scope:** Fast path handles the standard single-cell tutorial pair on sparse data: X that is sparse-convertible (converted to CSR float32 with int32 indices when nnz <= 2^31-1), inplace=True, copy in {True, False}, and no extra keyword args. Both target_sum modes are implemented: explicit target_sum (e.g. 1e4) routes to the fused per-row sum+scale kernel _fused_normalize_only; target_sum=None routes to _row_sums + a median-based target then _scale_only. log1p applies a parallel numba np.log1p over CSR .data (natural log, base=None), and falls back to np.log1p(out) for dense X. Numeric output of the .X matrix matches upstream on the benchmarked default-dtype CSR counts path (this is what the concordance metric checks).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

## `scanpy.pp.regress_out`

- **In-scope output equivalence:** tolerance
- **Validated at:** `sc.pp.regress_out(<data>, ['total_counts', 'pct_counts_mt'], n_jobs=1)  — data = log-normalized sparse CSR AnnData (pbmc68k_prepped.h5ad, small tier); layer=None, copy=False (both upstream defaults). task.yaml signature line: sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'], n_jobs=1)`
- **Supported scope:** Fast path is correct for non-categorical, numeric (ordinal) regressor keys with a non-empty keys list. It correctly handles: (a) sparse CSR/CSBase X via a sparse-aware regressors.T@X GEMM computed before densification (lines 154-158) — this is the benchmarked path; (b) rank-deficient / singular gram (e.g. an all-zero pct_counts_mt covariate) via np.linalg.pinv closed-form OLS, bypassing upstream's per-gene statsmodels GLM fallback (lines 137-145); (c) the layer= argument (reads/writes via _get_obs_rep/_set_obs_rep, lines 95/176) and copy= (line 88). It matches upstream's target_dtype rule for integer/float32/float64 X (lines 110-116). For the DENSE + non-singular case it intentionally defers to the unmodified upstream original (lines 137-143), so it is never wrong there. Math is OLS-equivalent (pearson per gene 1.000000, q99_abs_diff_X <= 1.2e-5 per docstring; task thresholds pearson>=0.9999, q99<=0.001, max<=0.01).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

## `scanpy.pp.scale`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `sc.pp.scale(<data:AnnData>, max_value=10)  # i.e. zero_center=True (default), copy=False, layer=None, obsm=None, mask_obs=None all default; only max_value deviates`
- **Supported scope:** Fast numba path activates ONLY when ALL hold: numba is importable; data is an anndata.AnnData; zero_center=True; layer is None; obsm is None; mask_obs is None; and adata.X is a scipy CSR sparse matrix (sparse.isspmatrix_csr). On this path it computes per-gene mean and unbiased (ddof=1) variance directly from the CSR data/indices via a numba accumulation kernel, densifies X once to float32, and applies a fused (mean-subtract, divide-by-std, symmetric clip) numba kernel. max_value is fully supported: None -> +inf (no clip), or a finite value -> symmetric clip to [-max_value, +max_value] (the 2026-05-21 fix restored two-sided clipping to match upstream; the benchmark/old run.py used upper-only clip but the SHIPPED kernel is symmetric). copy=True (returns a scaled copy) and copy=False (in-place, returns None) are both handled. std==0 columns are set to 1.0 (matching upstream constant-gene handling). It also writes adata.var['mean'/'var'/'std'] like upstream.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

## `scanpy.tl.leiden`

- **In-scope output equivalence:** bounded
- **Validated at:** `sc.tl.leiden(<data>, resolution=1.0, flavor="igraph", n_iterations=2)`
- **Supported scope:** Fast path correctly handles the igraph-flavor Leiden case: flavor="igraph" (the default for the patch arg, _leiden.py:347), directed=False, restrict_to=None, partition_type=None. It honors resolution (passes through to community_leiden, _leiden.py:388-389), use_weights (default True; sets weight attr, :386-387), random_state (default 0; applied via set_igraph_random_state in both direct and fork paths, :278/:303), key_added, copy, adjacency override, neighbors_key/obsp graph selection (_choose_graph, :379-380), and objective_function via clustering_args (defaults to "modularity", :390). It builds a deduplicated upper-triangle simple graph with 2x weights (vs upstream's multi-edge graph) and runs igraph community_leiden in a forked child on Unix / directly on Windows; output partition is similar but not bit-identical to upstream (ARI ~0.90-0.98, benchmark threshold ari>=0.90). Effectively reproduces upstream's flavor="igraph", n_iterations=2 result, which is exactly the benchmarked configuration.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

## `scanpy.tl.pca`

- **In-scope output equivalence:** tolerance
- **Validated at:** `sc.tl.pca(<data>, n_comps=50, svd_solver="arpack")`
- **Supported scope:** The fast path computes a full, zero-centered PCA via a Gram matrix (X^T X scaled by 1/(n_cells-1)) plus a symmetric eigendecomposition, then projects centered X onto the top n_comps eigenvectors. It correctly handles: dense or sparse adata.X (sparse is densified via toarray, line 159); any n_comps valid for the matrix; copy=True/False; and the implicit upstream zero_center=True / arpack-or-auto / full deterministic-solver case (output is mathematically equivalent up to sign, which is what the eval metric min_pc_cor>=0.95 checks). use_highly_variable is honored explicitly (lines 121-145): when an HVG annotation is present (or use_highly_variable=True) and the mask is a proper subset, it recurses on the HVG-subset matrix and lifts PCs back into full var space with zeros for non-HVG genes; use_highly_variable=False forces the full-gene path. Matrices with n_genes>8000 fall through to upstream ARPACK (line 154-156) where all kwargs are forwarded, and zyme=False (line 105-106) forwards everything to upstream. On macOS it uses Apple Accelerate cblas_sgemm + numpy.linalg.eigh; elsewhere numpy BLAS + scipy.linalg.eigh(driver="evr"). All computation is done in float32 internally regardless of requested dtype.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

## `scanpy.tl.rank_genes_groups`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `sc.tl.rank_genes_groups(<data adata>, groupby="leiden", method="wilcoxon", n_genes=250)  # all other args at default: groups="all", reference="rest", corr_method="benjamini-hochberg", tie_correct=False, pts=False, use_raw=None, layer=None, rankby_abs=False, key_added=None, copy=False, mask_var=None. Data is a CSC-presorted sparse log1p-normalized AnnData (run.py monkeypatches read_h5ad to convert CSR->CSC + presort columns outside the timed section). task.yaml declares groupby="leiden"; smoke recipe in __init__.py uses groupby="celltype", use_raw=True.`
- **Supported scope:** Fast path runs (and is benchmarked) only for: method=="wilcoxon" AND reference=="rest" AND tie_correct==False AND numba available AND X is a SciPy sparse matrix (CSBase, converted to CSC internally at line 508). Within that: groups=="all" or an explicit sequence of group names; corr_method in {benjamini-hochberg, bonferroni, other(=no adjustment)}; n_genes=None (full output, descending-score sort) or n_genes<n_genes_total (top-N branch); rankby_abs True/False; pts True/False; use_raw/layer/mask_var/key_added/copy honored; groupby column must be categorical (uses .cat.categories/.cat.codes). logfoldchanges reverse the log1p transform via expm1, reading the log base from adata.uns['log1p']['base'] (defaults to natural log if absent).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

