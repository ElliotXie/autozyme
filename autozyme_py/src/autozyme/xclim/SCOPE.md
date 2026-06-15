# autozyme `xclim` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `xclim.indices.growing_season_length`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `growing_season_length(<data tas>, thresh="5.0 degC", window=6, mid_date="07-01", freq="YS", op=">=")`
- **Supported scope:** Fast numba two-phase per-cell scan handles: op in {">", ">=", "gt", "ge"}; mid_date a valid "MM-DD" string that exists in the calendar; freq="YS" (annual, year-start); a DataArray with a "time" dimension on a daily axis (noleap OR Gregorian/standard calendars both work — year boundaries derived from times.year diffs, mid offsets from month/day matching). Within this scope the kernel faithfully reproduces upstream xclim season/season_length semantics: start = first run of `window` consecutive condition-true days whose run-start index is < mid_date; end = first run of `window` consecutive condition-false days at/after max(start, mid_date); length = end-beg, or (year_end - beg) when a start exists but no end is found. Verified against upstream xclim 0.60.0 run_length.season / first_run_before_date / first_run_after_date source: the start upper bound mid+window-1, the run_start<mid acceptance, the search_start=max(beg,mid) end search, and the size-minus-start fallback all match. Benchmarked tiers hit pct_exact=1.0 / max_abs_diff=0.
- **Out-of-scope behavior:** ⚠ **Documented approximation.** Correct for the validated configuration below; results may differ outside it and there is no automatic fall-back, so stay within the stated scope (or deactivate the patch).
- **Approximation details:** op not in {>, >=, gt, ge} (e.g. <, <=, ==, !=): explicitly falls back to upstream (line 105) — safe ; mid_date is None: explicitly falls back to upstream (line 105) — safe ; freq != 'YS' (e.g. 'YS-JUL' Southern-Hemisphere usage): explicitly falls back to upstream (line 105) — safe ; UNGUARDED: input is cast to float32 (arr_flat dtype=np.float32, line 118) and threshold to np.float32 (line 135). float64 temperatures within ~0.01 K of the threshold can flip the comparison vs upstream's float64 compare — no guard, silently applied on every supported call ; GUARDED 2026-06-09: mid_date that does not exist in a given year's calendar (e.g. '02-29' on a noleap year) now falls back to upstream (previously it silently forced length 0 for that year) ; UNGUARDED: output attrs units='d' hardcoded (line 146) and name=tas.name or 'growing_season_length' (line 147) — assumes daily steps; not equivalent to upstream dynamic to_agg_units for non-daily input ; Implicit: hardcoded time_dim='time' (line 113) — KeyError if no 'time' dim (same as upstream, no new failure) ; Implicit: tas.transpose(...).values (line 115) forces full in-memory materialization — loses upstream dask laziness for large lazy arrays

