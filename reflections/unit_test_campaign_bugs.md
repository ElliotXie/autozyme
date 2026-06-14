# Bugs surfaced by the unit-test coverage campaign

Found 2026-06-13 while raising unit-test coverage across `autozyme_cli`, `autozyme_py`,
`autozyme_r`. Each is DOCUMENTED in a test (asserting current behavior) rather than fixed, so
the suite stays green and a maintainer decides intent. Severity is my estimate.

## UPDATE 2026-06-14 — low-priority / latent batch FIXED
The following were fixed in source (smallest-correct change) and their pin tests flipped from
"asserts the bug" to "asserts the corrected behavior"; each fix was independently verified:
- **B2** report.py JS block -> raw string (`r"""`); no more SyntaxWarning, output byte-identical.
- **B4** scan_parallelism knob-default regex now matches Python `True`/`False` too.
- **B5** scan_parallelism `_parse_python_deps` now parses pyproject.toml via `tomllib` (no
  key/value leak), with a hardened quoted regex (no cross-quote pairing) as the fallback.
- **B6** fingerprint `_R_REP_PATTERN` now allows one level of nested parens (`rep(readRDS(...), N)`).
- **B7** fingerprint `_check_list_mul` no longer flags `[a] * [b]` (multiplier must be scalar-like).
- **B8** cost `_resolve_price_name` returns the `anthropic:`-prefixed row id so the `*-latest`
  fallback actually resolves to a price.
- **B9** scanpy HVG kernels size per-thread buffers by `numba.get_num_threads()` (no OOB segfault).
  SPEEDUP-SAFE: production callers already pass `n_threads = numba.get_num_threads()`, so this is
  byte-identical in benchmarks; only the untimed JIT warmups (n_threads=1) are affected. Output
  bit-exact.
- **B10** `_threads.safe_set_num_threads` now also catches `ValueError` (an over-request when
  OMP/env asks for more threads than the numba pool max) and stays at the current count. Fixed in
  the shared helper, NOT in the patch hot path -- `_normalize.py` is byte-identical to before.
  SPEEDUP-SAFE: in every benchmark config the target <= pool max, so `set_num_threads(target)`
  succeeds exactly as before; the new catch only fires in a misconfigured env that previously
  crashed.
- **B11** R `.zyme_intercept_write` emits a single backslash before embedded quotes -> valid JSON.
- **B15** package_verify_tsv `tier_dataset_map_from_task_yaml` also catches `yaml.YAMLError` (and
  non-mapping docs) -> returns `{}` on malformed YAML as documented.
- **B16** verify `_parse_metrics_json_field` normalizes a non-object JSON value to `{}`.
- **B17** task_yaml `_split_top_level_commas` is quote-aware -> comma-bearing mode values survive
  a write->read roundtrip.
- **B18** validate `_FINDING_HEADER` accepts case-insensitive severities (normalized to upper).
- **B19** profile/parsers `_split_memray_location` keeps a Windows drive-letter colon in the file
  part (split func off the left, line off the right).

STILL OPEN (real bugs, pinned, awaiting maintainer decision): B1, B21 (see entries below).

## UPDATE 2026-06-14 (round 2) — investigated the 7 "critical" bugs, fixed 4
A parallel deep investigation confirmed each with a real repro and checked every one against the
finalized speedups.tsv. KEY FINDING: NONE of the 7 affect any published speedup or output-
equivalence number (the buggy paths are all non-benchmarked or non-numeric tooling). Actions:
- **B13** mdanalysis weighted RMSD: FIXED. Both einsum sites (`__init__.py:252` DCD-bulk and `:295`
  non-bulk) changed from `"cij,j->ci"` to `"cij,i->cj"` (contract the atom axis). Weighted RMSD no
  longer crashes and now matches vanilla; the two strict-xfail tests were converted to real parity
  tests. Unweighted (benchmarked) path is untouched. No speedup/numeric change.
- **B22** fipy source-term crash: FIXED. `fast_binary_buildAndAddMatrices` now only writes the
  binary cache when `tmpRHSvector_diff_for_cache is not None`, so an out-of-scope
  `transient == diffusion + source` equation falls back to the full upstream path instead of
  crashing on the 2nd solve. The xfail test became a parity test. RUN-VERIFIED in the scanpy310
  conda env (fipy 4.0.2, the version the patch targets): the source-term equation now falls back
  and its result matches vanilla. (On the base anaconda env fipy's editable install target was
  removed, so its tests skip there.)
- **B14** check_intercept ignored executor.python: FIXED. `_resolve_python_for` passes
  `task_dir / "task.yaml"` (the file) to parse_executor instead of the directory, so a declared
  custom interpreter is honored on the attest preflight path. Pin test flipped.
- **B3** scan_parallelism path priority: FIXED. `_path_priority` set check now uses lowercase `"r"`
  (rel_path is already lowercased), so R/ source ranks production (0) again. Pin test flipped.
- **B12** FindAllMarkers Wilcoxon: INVESTIGATED, NOT A BUG. The kernel is bit-exact to presto
  (vanilla Seurat's default Wilcoxon backend): 300-trial fuzz, end-to-end, and a forced tie-trigger
  all show max p-value diff = 0 vs vanilla Seurat. The "missing" t^3-t term on the final tie run
  matches presto's convention; the ~1e-3 gap is only vs base-R wilcox.test/textbook, which Seurat
  does not use. Changing the code would BREAK parity with the very baseline the project claims
  faithfulness to. Left as-is intentionally. (test-cpp-markers.R already pins this at tol 5e-3.)

## CRITICAL — stale `restore()` references (FIXED in test/CI files)

### B0. `restore()` was renamed to `deactivate()` but stale calls remain and ERROR at runtime (HIGH)
`restore` does not exist anywhere in the autozyme R package: not defined in any `R/*.R`, not in
`NAMESPACE` (only `deactivate`/`deactivate_all` are exported), not in the namespace, and
`find("restore")` is empty. `autozyme::restore(...)` raises "'restore' is not an exported object
from 'namespace:autozyme'"; bare `restore(...)` raises "could not find function 'restore'".
Stale calls remained in:
- `tests/testthat/test-core.R:41` (un-guarded) -> the "register + activate + restore roundtrip"
  test ERRORED.
- `tests/testthat/test-contract-clusterprofiler.R:79` -> that test ERRORED.
- `tests/testthat/test-contract-all-patches-activate.R:63` (un-guarded) -> ALL 17 patch
  round-trip tests ERRORED (this is the coverage-drift guard test).
- `tests/smoke_seurat_normalize.R:113`, `tests/smoke_seurat_phase4.R:126` (standalone smoke).
- `.github/workflows/tier-a-pr.yml:276,284` (the R direct-smoke step) -> the R CI job would
  fail if `CI_ENABLED` were on (likely why this went unnoticed).
These errored silently in prior tallies because testthat records a code-execution error in the
`error` column, not `failed`. FIX APPLIED: replaced `restore(` -> `deactivate(` in all the
above (the documented, exported replacement). After the fix, all 19+ affected tests pass.

## Python — autozyme_cli

### B1. report.py: `conc_chip` / `tldr` are dead outputs (LOW-MED)
In `render_report` (~lines 1268-1287), `conc_chip` and `tldr` are computed but never
interpolated into the final `HTML` string. The `.hero .conc-chips` / `.hero .tldr` CSS exists
but nothing emits those elements, so the concordance chips and the BRCA-Wu tldr never render.
Either wire them into the template or delete the dead computation + CSS.
- Test: `tests/test_report_unit.py::test_brca_wu_concordance_branch_executes`

### B2. report.py: SyntaxWarning on embedded JS regex (LOW)
`report.py:2048` emits `SyntaxWarning: invalid escape sequence '\s'` — the `JS = """..."""`
block contains `\s` regex literals in a non-raw string. Make it a raw string (`r"""`).

### B3. scan_parallelism._path_priority: lowercases then tests uppercase "R" (MED)
`rel_path.lower()` is computed, then `parts[0] in {"R","src","inst"}` is checked. After
lowercasing, an R-package path `R/foo.R` becomes `r/foo.R`, which is NOT in the set, so R
production source is ranked priority 5 (generic) instead of 0 (production). Only `src/`/`inst/`
(already lowercase) reach 0. Effect: R `R/` source is not preferred when picking the "via" hint.
- Test: `tests/test_scan_parallelism_unit.py::TestPathPriority::test_uppercase_R_dir_quirk`

### B4. scan_parallelism: Python boolean knob defaults not captured (LOW-MED)
The knob-default regex's boolean branch is `[A-Z][A-Z]+`, matching only R-style `FALSE`/`TRUE`.
Python `False`/`True` are silently dropped, so a Python `parallel: bool = False` signature
yields no knob row (numeric + `func()` defaults still captured).
- Test: `tests/test_scan_parallelism_unit.py::test_python_lowercase_bool_default_not_captured`

### B5. scan_parallelism._parse_python_deps: leaks TOML keys as deps (LOW)
The requirements-style regex matches TOML assignment keys (`name =`, `license =`,
`dependencies =`) as dependencies; the metadata-exclusion set is only applied to the quoted-
string regex. Consistent with the module's stated "false positives are cheap" stance.

### B6. fingerprint._R_REP_PATTERN: misses `rep(<call>, N)` (LOW)
`rep(readRDS('a.rds'), 7)` is not flagged because the capture group is `[^,()]+?` (no parens).
Source already labels the R checks "partial coverage", so a known limitation.
- Test: `tests/test_fingerprint_unit.py::test_rep_nested_call_arg_NOT_flagged_partial_coverage`

### B7. fingerprint._check_list_mul: `[a] * [b]` false positive (LOW)
Flags two single-element lists multiplied as `list_mul` without checking the multiplier is
scalar-like. Rare in reference scripts; minor false-positive surface.
- Test: `tests/test_fingerprint_unit.py::test_two_lists_multiplied_is_flagged`

### B8. cost._resolve_price_name: returns unpriced `*-latest` ids (LOW)
Fallback returns family ids like `claude-sonnet-4-latest` that are not themselves aliases in
`dispatch.pricing.MODEL_PRICES`, so an explicit `--model-override` of a bare `*-latest` yields
no price. Works only because the input model is usually a real alias.

## Python — autozyme_py

### B9. scanpy HVG numba kernels: segfault if `n_threads` arg < live pool (LATENT/MED)
`_hvg_one_pass`, `_v3_batch_*` index per-thread buffers by `numba.get_thread_id()` under
`boundscheck=False`. Passing `n_threads` smaller than the live `numba.get_num_threads()` pool
writes out of bounds -> segfault. Production always passes `numba.get_num_threads()`, so it is
safe today, but it is a footgun for any future caller that hardcodes a smaller `n_threads`.
Consider clamping internally or asserting `n_threads >= get_num_threads()`.

### B10. scanpy _normalize: thread count not clamped to numba pool max (LATENT/LOW)
`_n_threads()` resolves from `OMP_NUM_THREADS`/cpu and the patched path calls
`numba.set_num_threads(that)`. If `NUMBA_NUM_THREADS` is pinned below that value, numba raises
`ValueError: number of threads must be between 1 and N`. Only bites when the numba pool is
capped smaller than OMP/cpu (e.g. a misconfigured test env); clamp to `numba.get_num_threads()`.

## R — autozyme_r

### B11. intercept_probe.R `.zyme_intercept_write`: double-escapes quotes -> invalid JSON (LOW)
`gsub('"','\\\\"',k,fixed=TRUE)` emits two backslashes before the quote (`\\"`), not the single
backslash JSON requires (`\"`). A patch key containing a literal `"` produces invalid JSON. Low
impact (keys are `pkg::fn`, never quote-bearing in practice).
- Test: `tests/testthat/test-unit-intercept-probe.R`

### WAVE 2 additions (2026-06-13)

### B13. mdanalysis_rmsd.fast_compute: weighted center-of-mass einsum is wrong (MED, REAL)
`fast_compute` does `np.einsum("cij,j->ci", buf, w)` where `buf` is `(chunk, n_atoms, 3)` and
`w` is length `n_atoms`. This contracts the size-3 COORDINATE axis against `w` (lengths only
match by accident when n_atoms==3) and otherwise raises `ValueError: operands could not be
broadcast`. It should be `"cij,i->cj"` (contract the atom axis `i`). `fast_single_frame` uses
the correct `np.dot(w, buf)`. So weighted RMSD over a full in-order (non-DCD-bulk) trajectory
crashes. The benchmark only ran unweighted backbone RMSD, so it never tripped.
- Test: `tests/contract/test_mdanalysis_rmsd_e2e.py::test_weighted_com_path_matches_vanilla` (strict xfail)
- **SECOND SITE (wave 3)**: the same wrong einsum appears in the DCD **bulk** path at
  `fast_compute` line ~252 (`np.einsum("cij,j->ci", buf_seg_f64, w)`), distinct from the
  non-bulk MemoryReader site at ~295. Both should be `"cij,i->cj"`. Pinned separately in
  `tests/contract/test_mdanalysis_rmsd_w3.py::test_dcd_bulk_weighted_path_is_broken` (strict xfail).

### B21. infercnv fast_smooth_window is wrong for odd-tail_length windows (MED, REAL — wave 4)
The patched center smoother (`inst/patches/infercnv/patch.R:150-164`, matrixStats double-
`colCumsums` box filter) uses `w = tail_length + 1` and divides by `w*w`. This matches upstream's
true window mean only when `tail_length = (window_length-1)/2` is EVEN (wl=5,9,13 -> bit-exact)
but DIVERGES when tail_length is odd: wl=3 -> max_diff 1.34, wl=7 -> 0.64, wl=11 -> 0.28, in the
CENTER rows (not the edge taper). infercnv's default `window_length=101` has even tail_length (50)
and is additionally routed to a separate C++ kernel, so production never hits this, but any caller
passing wl in {3,7,11,...} gets incorrect smoothing. Pinned (current behavior) in
`tests/testthat/test-w4-contract-deepen.R` with a comment so a future fix flips the expectation.

### B22. fipy fast_binary_buildAndAddMatrices crashes on 3-term eq with source on cache hit (MED, REAL — wave 4)
For a 3-term equation `TransientTerm() == DiffusionTerm(coeff=const) + source`, `self.other`
becomes a nested `_BinaryTerm` (diffusion+source), so the diffusion sub-term is never captured and
`tmpRHSvector_diff_for_cache` stays None. On the 2nd solve the cache-hit branch does
`RHSvector = b_diff + b_trans` with `b_diff=None` -> `TypeError: unsupported operand 'NoneType' + 'float'`.
The same equation solves fine with the patch DISABLED. It is outside the patch's documented target
(DiffusionTerm with constant coeff, no source), but the patch should fall back to the slow path
rather than crash. Pinned strict-xfail in `tests/contract/test_fipy_w4.py::test_source_term_equation_crashes_on_cache_hit`.

### B20. Coverage-drift: patch `milor` has no `test-contract-milor.R` (FIXED — wave 4)
`tests/testthat/test-coverage-gaps.R` fails one assertion: "patches without a contract test:
milor". The `milor` patch is registered but ships no per-API contract test, which the CI
coverage-drift check (`.github/scripts/ci_check_coverage.R`) is meant to catch. Pre-existing
(present with or without the campaign's files). FIXED in wave 4: added
`tests/testthat/test-contract-milor.R` (a real per-API contract test for the miloR
`calcNhoodDistance`/`.calc_distance` targets). The CI coverage-drift check
(`.github/scripts/ci_check_coverage.R`) now reports "OK: 17 patches -> 28 contract tests, fully
accounted for". (Fixture note: vanilla `calcNhoodDistance(zyme=FALSE)` leaves `nhoodDistances()`
empty due to miloR's broken `nhoodDistances<-` setter, which the patch bypasses; parity is pinned
via a manual `dist()` reference + the bit-exact internal `.calc_distance` target.)

### NOT-A-BUG note: 8 "errors" in test-python-env.R / test-contract-RunPCA.R / RunCCA.R
Under a file-by-file harness (`library(autozyme)` + `testthat::test_file`) these error with
`could not find function ".az_py_bind"` because they reference INTERNAL functions
(`.az_py_bind`, `.az_py_probe`, ...) by BARE name. Those resolve under the canonical runner
(`R CMD check` / `tests/testthat.R` -> `test_check("autozyme")`, whose test-env parent IS the
package namespace) but not under `test_file` (parent = globalenv). Confirmed `.az_py_bind` IS in
the namespace, so these PASS under the canonical runner -- not bugs. (Contrast B0: `restore` is
in NO scope, so it fails even under `test_check`.) The canonical runner can't be run fully
locally because `test-shared-infra.R` segfaults on this Mac, hence the file-by-file approach.

### B14. commands/package/check_intercept._resolve_python_for: passes dir, not task.yaml (LOW-MED)
Passes the task DIRECTORY to `parse_executor()`, which expects the task.yaml FILE (it calls
`.read_text()` -> `IsADirectoryError`). A broad `except Exception: pass` swallows it, so a task's
declared `executor.python` is SILENTLY IGNORED and the interpreter always falls back to
`sys.executable`. `smoke_parity.py` imports the same helper and inherits the quirk.
- Test: `tests/commands/test_package_cmd.py::TestResolvePythonFor::test_parse_executor_arg_quirk_falls_back`

### B15. parsers/package_verify_tsv.tier_dataset_map_from_task_yaml: wrong except set (LOW)
Catches only `(OSError, ValueError)`, but `yaml.safe_load` raises `yaml.YAMLError` on malformed
task.yaml, which is not a `ValueError`. The docstring promises `{}` on bad input; instead the
exception propagates uncaught. (`tests/parsers/test_package_verify_tsv_deep.py`)

### B16. commands/verify._parse_metrics_json_field: not guaranteed to return a dict (LOW)
Returns whatever `json.loads` yields, so a bare JSON scalar (`"plain"`) returns a `str` and a
JSON array returns a `list` rather than `{}`. A downstream `.get()` would `AttributeError`.
Harmless today (the column always holds an object).
- Test: `tests/commands/test_verify_cmd.py::test_quirk_valid_json_scalar_passes_through`

### B17. parsers/task_yaml.parse_modes_block: comma-splitter is quote-unaware (LOW)
`write_mode_entry` quotes a mode value containing commas, but `parse_modes_block`'s
`_split_top_level_commas` ignores quotes, so a comma-bearing mode value does NOT survive a
write->read roundtrip (split mid-value). Low impact (mode blocks are deprecated legacy).

### B18. commands/validate._FINDING_HEADER requires UPPERCASE severity (LOW)
The regex is `[A-Z_]+`, so a finding written with a lowercase severity (e.g. `fail`) is silently
dropped from the TSV. (`tests/commands/test_validate_cmd.py::test_lowercase_severity_is_dropped`)

### B19. commands/profile/parsers._split_memray_location: Windows drive colon mis-split (LOW, doc)
The docstring claims it keeps a Windows drive colon in the file part, but `rsplit(":", 2)` puts
the leftover drive colon in the FUNC field. POSIX memray locations (the real case) work fine.

### Dead/defensive code noted (not bugs, flagged for cleanup)
- `commands/profile/enrich.py:74` `tf.startswith("<")` is unreachable (regex charset excludes `<`).
- `commands/init_check.py` task.yaml-missing exit-2 guard is dead (the dir guard rejects first).
- `commands/registry._score` never returns 0 for a packaged-status entry (status prior +2).

### B12. FindAllMarkers Wilcoxon: missing tie-correction for the final tie run (MED)
`parallel_all_in_one_dgc` / `turbo_all_in_one_wilcox` (`seurat_markers.cpp:208`) omit the
`t^3 - t` tie-correction term for the LAST tie run of each feature (the `next < m` guard skips
it). Against a textbook tie-corrected Wilcoxon, p-values differ by ~1e-3 on tiny data. The
statistic is otherwise a valid normal-approx Wilcoxon (nnz + expm1 sums are bit-exact). Worth a
maintainer's look: intentional (matching the lifted prototype) or a latent off-by-one.
- Test: `tests/testthat/test-cpp-markers.R`
