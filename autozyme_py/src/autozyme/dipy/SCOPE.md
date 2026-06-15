# autozyme `dipy` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `dipy.reconst.dti.TensorModel.fit`

- **In-scope output equivalence:** tolerance
- **Validated at:** `dti.TensorModel(gtab, fit_method="WLS", return_S0_hat=True).fit(data=<data>, mask=<mask>)  where <mask> is a median_otsu brain mask; <data> is a Stanford-HARDI / CFIN / Sherbrooke DWI volume tiled per tier (small=8x, medium=30x, large=60x, ood_large=60x CFIN, ood_xlarge=180x Sherbrooke). min_signal left at upstream default (None -> dti.MIN_POSITIVE_SIGNAL).`
- **Supported scope:** Correctly handles fit_method="WLS" (the upstream default) tensor fits. (1) Masked WLS fit (mask not None) uses the chunked fast path: per-chunk (step=1250) masked-voxel unpacking, batched 7x7 normal-equations solve via hand-rolled Cholesky (_chol7_solve), with per-voxel SVD-pinv fallback (_pinv_wls_fit) for any voxel whose normal matrix is not safely positive-definite — so singular/ill-conditioned voxels are guarded and matched to upstream. (2) Unmasked WLS fit (mask=None) is also accelerated via the patched dti.wls_fit_tensor through the upstream-style reshape path. (3) return_S0_hat True/False both handled (line 235-250, 280-299). (4) weights= passed through to the kernel (line 144-145). (5) return_lower_triangular handled (line 177-178). (6) return_leverages=True routed to an upstream-equivalent SVD-pinv solve (line 165-172). All numerics are float64, matching upstream (no float32 downcast). Empty mask and single-voxel masks work. The supported numeric domain is the standard rank-2 diffusion tensor: a 7-column design matrix (6 tensor + 1 log-S0) producing the (...,12) eig_from_lo_tri output contract.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

