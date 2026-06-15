# autozyme `sarsen` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `sarsen.apps.terrain_correction`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `sarsen.apps.terrain_correction(product, <data DEM .tif>, output_urlpath=<tmp GTC.tif>, correct_radiometry=None, chunks=None) — where product = sentinel1.Sentinel1SarProduct(<GRD .SAFE>, "IW/VV"). Headline tiers small/medium/large all use correct_radiometry=None (GTC); the two ood tiers additionally pass correct_radiometry="gamma_nearest". interp_method left at default "nearest"; grouping_area_factor default (3.0,3.0); all other args default.`
- **Supported scope:** Correct (bit-exact, pass_rate 1.0, max_abs_diff 0.0) for: GRD products (GroundRangeSarProduct) processed in-memory with chunks=None, interp_method="nearest" (the upstream default), correct_radiometry in {None (GTC), "gamma_nearest"} — note "gamma_nearest" still goes through upstream do_terrain_correction radiometry chain but calls the patched simulate_acquisition/orbit/geocoding helpers, and verifies bit-exact at ood_large. The patch replaces 9 internal helpers (transform_dem_3d, convert_to_dem_3d, slant_range_time_to_ground_range, the three OrbitPolyfitInterpolator polyval fits, the two zero_doppler Newton kernels, simulate_acquisition, GroundRangeSarProduct.interp_sar, Sentinel1SarProduct.beta_nought) plus a beta_nought process-local cache; the public terrain_correction wrapper only scopes xr.set_options(use_bottleneck=False) and delegates to upstream. Verified upstream version sarsen 0.9.6.dev5+g6c5e37d1d on the Rome DEM / S1B GRD IW/VV product.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

