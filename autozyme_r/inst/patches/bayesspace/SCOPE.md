# autozyme `bayesspace` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `BayesSpace::spatialCluster`

- **In-scope output equivalence:** bounded
- **Validated at:** `BayesSpace::spatialCluster(<data sce>, q=4, platform="ST", d=7, init.method="mclust", model="t", gamma=2, nrep=50000, burn.in=1000)  # headline "small"/tiny tier; other benchmarked tiers vary q/platform/d/gamma/nrep/burn.in, e.g. medium = (q=8, platform="Visium", d=15, gamma=3, nrep=10000, burn.in=1000), large = (q=8, "Visium", d=15, gamma=3, nrep=20000, burn.in=2000), ood_xlarge = (q=12, "Visium", d=20, gamma=3, nrep=30000, burn.in=3000)`
- **Supported scope:** The patch overrides ONLY BayesSpace internal iterate_t, the Gibbs/MH MCMC inner loop invoked when spatialCluster is called with model="t". Within that path the fast kernel (fast_iterate_t_impl in src/bayesspace.cpp) reproduces the full upstream t-model math for arbitrary n (spots), d (PC dims), q>=2 (clusters), gamma, nrep, thin, burn.in, and any df_j neighbor structure (handles empty neighbor lists). It is platform-agnostic because platform only affects how spatialCluster builds df_j (the neighbor list) before iterate_t runs, and any q,d,gamma,nrep are honored. It is NOT bit-exact: BLAS-batched rooti projection introduces fp reordering and, more importantly, the proposal draw was changed from Rcpp::sample to R::unif_rand, which rotates the entire RNG trajectory — equilibrium distribution preserved, individual trajectory diverges. Correctness is gated statistically (ARI/NMI permutation-invariant clustering similarity, noise_multiplier widened 5%), not element-wise. Validated tiers span platform in {ST, Visium}, q in 4-12, d in 7-20, nrep 10000-50000.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

