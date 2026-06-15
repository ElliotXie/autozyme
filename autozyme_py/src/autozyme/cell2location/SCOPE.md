# autozyme `cell2location` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## Tested environment

The patched `forward` binds cell2location 0.1.5's long PyroModule class name and its `forward(x_data, idx, batch_index)` signature, both of which change across releases, so keep cell2location pinned to 0.1.5. Validated on macOS arm64, Python 3.10.20:

```bash
pip install "cell2location==0.1.5" "scvi-tools==1.3.3" "pyro-ppl==1.9.1" "torch==2.11.0" "numpy==2.2.6" "anndata==0.11.4" "scanpy==1.11.5"
```

numpy 2.x is supported (numpy<2 is NOT required). The `autozyme` package declares no hard dependencies by design: it overlays whatever cell2location you already have. Install the stack above, or `pip install "autozyme[cell2location]"`.

## `cell2location.models.Cell2location.train`

- **In-scope output equivalence:** tolerance
- **Validated at:** `mod = cell2location.models.Cell2location(<data adata_vis>, cell_state_df=<inf_aver signature>, N_cells_per_location=30, detection_alpha=20); mod.train(max_epochs=300, batch_size=None, train_size=1, lr=0.002, accelerator="cpu", enable_progress_bar=False). Model setup via Cell2location.setup_anndata(adata=<data>, batch_key="sample") yielding n_batch=1 (single sample). reference.py:70-88; pipeline/run.py:483-500.`
- **Supported scope:** The fast path is correct ONLY for the single-batch, full-batch SVI regime that the benchmark uses. Specifically: (1) n_batch == 1 — fast_forward checks `if self.n_batch != 1: return _orig_forward(...)` (line 132), so multi-sample data correctly falls back to upstream. (2) Full-batch training, i.e. batch_size=None / train_size=1, so the entire count matrix x_data is passed every step as the SAME tensor object — this is required by the lgamma cache keyed on value.data_ptr() (lines 46-49, 123-128) which assumes lgamma(value+1) is constant across all 300 epochs. (3) The (alpha, mu)-parameterized GammaPoisson data likelihood as constructed in the LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlpha model forward — _GPLogProbFn (lines 73-119) hardcodes the explicit forward+backward for exactly that NegativeBinomial/GammaPoisson form (alpha = 1/alpha_g_inverse^2, mu = (w_sf @ (cell_state*m_g) + s_g_gene_add)*detection_y_s). (4) The fall-back upstream path (n_batch != 1) is still accelerated by fast_gp_log_prob, which is mathematically equivalent to GammaPoisson.log_prob and adds a row-redundancy shortcut only when concentration rows are bit-identical (verified via torch.equal, lines 53-62) — otherwise it computes the full result. (5) The validation-disable toggle is save/restored locally around the n_batch=1 forward (lines 135-139, 268), so it has no process-global side effect. Verified bit-close: pearson_loss=1.0, pearson_w_sf=1.0, tiny max_abs diffs across all benchmarked tiers (speedups_finalized.tsv).
- **Out-of-scope behavior:** ⚠ **Documented approximation.** Correct for the validated configuration below; results may differ outside it and there is no automatic fall-back, so stay within the stated scope (or deactivate the patch).
- **Approximation details:** n_batch != 1 (multiple samples / batch_key with >1 level): fast_forward is skipped entirely — guarded fallback to upstream forward (safe). ; batch_size != None or train_size < 1 (minibatch SVI): UNGUARDED. The lgamma cache is keyed on value.data_ptr() and assumes x_data content is constant for a given pointer. Under minibatching, PyTorch/scvi-tools may reuse the same buffer address for different minibatch contents, so a stale lg_v1 (lgamma(value+1)) could be applied to the wrong counts — silently wrong likelihood. No guard checks batch_size. ; dropout_p != 0: x_data is passed through self.dropout(x_data) (line 258-259) BEFORE _gp_log_prob_alpha_mu, but the lgamma cache key uses the ORIGINAL x_data data_ptr while lg_v1 is computed from the pre-dropout value at first call — and dropout produces a new tensor each step. The cache uses the value tensor passed in; in the n_batch=1 path x_data is overwritten by dropout output, so on the first step lg_v1=lgamma(dropout(x)+1) is cached against the dropout tensor's data_ptr and reused for subsequent (differently-dropped) steps. GUARDED 2026-06-09: fast_forward now falls back to upstream when dropout_p != 0, so this path is no longer silently wrong. Benchmark/default dropout_p=0 is unaffected. ; num_particles != 1 / scale_elbo != 1.0: these train()-signature args are not handled by the patch; behavior is whatever the upstream guide/ELBO wiring does with the patched forward — untested, the _GPLogProbFn returns a single summed scalar likelihood and may not interact correctly with num_particles>1 vector ELBO. ; Any cell2location model class other than LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel: the patch only re-binds forward on that one class; other model variants are untouched (effectively safe/no-op). ; training_wo_observed=True: handled (likelihood block skipped via line 255 guard) — matches upstream structure. ; GPU (accelerator!='cpu'): device string is part of the cache key so caching is per-device, but the patch was only benchmarked/intended for CPU; not validated on CUDA. ; cell2location version != 0.1.5: tested_against / tested_upstream_versions pin 0.1.5; the long PyroModule class name and forward signature (x_data, idx, batch_index) are version-fragile.

