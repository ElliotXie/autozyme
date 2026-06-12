# autozyme — cross-patch traps registry

Living list of non-obvious failure modes hit while lifting tasks into patches.
**Before lifting a new task**, scan for upstream pkgs that match any pattern
below and apply the relevant fix preemptively. Append a new entry whenever a
fresh trap costs more than ~30 minutes to diagnose.

Format: **Symptom — Trigger — Fix — Seen in**.

---

## 1. `MAST::generic(...)` returns wrong class via `callName()` reflection

- **Symptom**: `MAST::lrTest(zfit, MAST::CoefficientHypothesis("groupB"))` fails
  with `"undefined class MAST::CoefficientHypothesis"` or returns garbage class
  names.
- **Trigger**: Upstream uses `callName()` / `match.call()` reflection on the
  call form to infer a class or method name, AND we invoke it with the
  `pkg::fn(...)` qualified prefix.
- **Fix**: Inside the smoke recipe (or fast fn body), call
  `suppressPackageStartupMessages(library(MAST))` once, then use the
  unqualified name. Do not rely on `requireNamespace()` — that loads the
  namespace but does NOT attach it to the search path, so `callName()` still
  sees the prefixed form.
- **Seen in**: `inst/patches/mast.R` (CoefficientHypothesis).

## 2. S4 method patches need the generic *function object*, not its name

- **Symptom**: `setMethod("getCurves", sig, fn)` raises
  `"could not find function 'getCurves'"` even though `slingshot::getCurves`
  exists and `requireNamespace("slingshot")` returned TRUE.
- **Trigger**: register_patch with `kind = "s4"` target. autozyme's namespace
  doesn't `import()` upstream S4 generics, so setMethod's name-based lookup
  fails up the parent chain.
- **Fix**: Pass the generic *object* (`utils::getFromNamespace(fn_name, pkg)`)
  to `setMethod`, AND install via `where = globalenv()` — both upstream and
  autozyme namespaces are locked, so global is the only writable env that
  also wins S4 dispatch.
- **Seen in**: `R/core.R::.activate_one` (s4 branch); `inst/patches/slingshot.R`.

## 3. S4 slot access fails with "<Class> is not a defined class" under requireNamespace alone

- **Symptom**: A smoke `call` (or fast fn body) that does `pkg::fn(obj)` where
  `obj` is an S4 instance defined by `pkg` raises
  `Error: "<ClassName>" is not a defined class` — even though
  `requireNamespace(pkg)` returned TRUE and `methods::isClass("<ClassName>")`
  returns TRUE in the calling session.
- **Trigger**: spacexr-style packages whose S4 class table is fully registered
  in the runtime methods cache only after `library(pkg)` attaches the package
  (not on plain namespace load). `@` slot access from inside the pkg's own
  fns then uses `topenv(parent.frame())` to look up the class, which finds
  spacexr's namespace but apparently misses class entries that
  `library()`-time attachment registers via `methods::cacheMetaData`.
  Manifests as soon as the upstream function does `obj@slot <- ...`.
- **Fix**: Inside the smoke `load` (or smoke `call`) recipe, do
  `suppressPackageStartupMessages(library(<pkg>))` once. Cheap; runs untimed
  in `load` so it does not pollute the speedup ratio. Identical shape to the
  MAST `callName()` caveat above — same prescription, different underlying
  reflection target.
- **Seen in**: `inst/patches/rctd.R` (spacexr RCTD S4 class).

## 4. Rcpp `cppFunction` inline compile triggers GC corruption

- **Symptom**: `Error: NULL value passed as symbol address`, often after the
  first GC cycle or first parallel call. Replicable but timing-dependent.
- **Trigger**: A patch was scaffolded with `Rcpp::cppFunction("...")` at file
  load time (the original "lift it inline" approach).
- **Fix**: Move the C++ to `src/<kernel>.cpp` with `// [[Rcpp::depends(...)]]`,
  add `LinkingTo: Rcpp, RcppParallel, RcppArmadillo` (as needed) to
  DESCRIPTION, and `useDynLib(autozyme, .registration = TRUE)` to NAMESPACE.
  Use `tools/scaffold_cpp_patch.R` — it generates the boilerplate. On macOS
  also add `-Wl,-rpath,$(R_PACKAGE_DIR)/libs` to `src/Makevars` if the kernel
  links RcppParallel.
- **Seen in**: `src/score_ligands.cpp` (was inline in nichenetr at lift time);
  `src/soft_assignment.cpp`; `src/cpp_updateState.cpp`.

## 5. Stale upstream API calls (e.g. `GetAssayData(slot=)`) break the BASELINE call

- **Symptom**: `verify_patch()` aborts during baseline with
  `"The slot argument of GetAssayData() was deprecated in SeuratObject 5.0.0
  and is now defunct. Please use the layer argument instead."` — even though
  the task's pipeline ran fine against the same upstream code.
- **Trigger**: Unmaintained upstream pkg (scriabin pinned to a 2022 API)
  calls a downstream fn whose signature changed. The task's `pipeline/run.R`
  works around this by overriding the deprecated fn in `globalenv()` AND
  source-loading the upstream R files into globalenv, so the lookup chain
  (globalenv -> search path) hits the shim. The patch's `smoke$call` does
  not — it invokes the INSTALLED upstream via `upstream::fn(...)`, whose
  body resolves the deprecated dep via the upstream pkg's own imports table
  (parent.env(asNamespace("upstream"))), which the globalenv shim never sees.
- **Fix**: In smoke `load` (untimed, runs once per tier — same cost
  baseline+patched, no fair-comparison violation), install the translation
  shim directly into upstream's imports env via assign + lock/unlock:
  ```r
  imports_env <- parent.env(asNamespace("scriabin"))
  was_locked <- bindingIsLocked("GetAssayData", imports_env)
  if (was_locked) unlockBinding("GetAssayData", imports_env)
  assign("GetAssayData", shim, envir = imports_env)
  if (was_locked) lockBinding("GetAssayData", imports_env)
  ```
  Tag the shim with an attribute so a re-run of `load` doesn't re-wrap.
- **Seen in**: `inst/patches/scriabin.R` (`.zyme_scriabin_install_shim`).

## 6. Formula-LHS `eval(formula[[2]], envir = parent.frame(), enclos = environment(formula))` misses smoke-frame bindings under the dispatcher wrapper

- **Symptom**: A patched namespace fn (e.g. vegan's `adonis2`) fails inside
  its call with `Error: object '<lhs_name>' not found` when invoked from a
  smoke `call` recipe that pre-computes the LHS and binds it as a local in
  the smoke body. Baseline (no `activate()`) works; only the patched
  subprocess path errors.
- **Trigger**: Upstream's body resolves the formula LHS via
  `eval(formula[[2]], envir = parent.frame(), enclos = environment(formula))`.
  `enclos` is **silently ignored** when `envir` is an environment — only
  `parent.frame()` and its lexical parent chain are searched. Under
  autozyme's `.wrap_namespace_fast` dispatcher, the fast fn's
  `parent.frame()` is the WRAPPER's frame, not the smoke `call` function's
  frame, so a local binding in smoke `call` (or even an env attached to
  the formula) is unreachable.
- **Fix**: Inside smoke `call`, assign the LHS object (and any other names
  the formula references, e.g. `data` arg) into `.GlobalEnv` with
  `on.exit` cleanup:
  ```r
  assign("dist_mat", inputs$dist_mat, envir = globalenv())
  on.exit(rm("dist_mat", envir = globalenv()), add = TRUE)
  ```
  The wrapper's enclosing env walks autozyme ns -> imports -> base ->
  R_GlobalEnv, so the LHS resolves. Constant cost on both baseline and
  patched (microseconds), so it does not bias the speedup ratio.
- **Seen in**: `inst/patches/vegan.R` (adonis2 smoke recipe; `dist_mat` LHS).

## 7. `data.table` `:=` / `[.data.table` rejects calls from autozyme namespace ("not data.table-aware")

- **Symptom**: A patched fast fn that uses `dt[, col := expr]` or `dt[i, j, by=]`
  inside a function defined in `inst/patches/<name>.R` fails at activation
  time with
  `Error in '[.data.table'(maf, , ':='(Chromosome, ...)): [ was called on a
  data.table in an environment that is not data.table-aware (i.e. cedta()),
  but ':=' was used`.
- **Trigger**: data.table's `[.data.table` calls `cedta(n=2)` which inspects
  `topenv(parent.frame())` — the calling function's namespace — and accepts
  only packages whose Imports field mentions `data.table` OR whose namespace
  defines `.datatable.aware = TRUE`. Patches in `inst/patches/` run with
  `topenv() = asNamespace("autozyme")`; injecting `.datatable.aware <- TRUE`
  into autozyme's namespace at file-source time fails because the namespace
  is locked. The fix has to be at package build time.
- **Fix**: Add `data.table` to `DESCRIPTION`'s `Imports:` field AND add
  `importFrom(data.table, ":=")` (or any other data.table symbol) to
  `NAMESPACE`. The bare `Imports:` entry by itself does NOT register the
  package in `getNamespaceImports()` — only an `importFrom` does. Verify
  with `Rscript -e 'library(autozyme); print(names(getNamespaceImports(asNamespace("autozyme"))))'`
  → expect to see `"data.table"` in the result. Then `[.data.table` and `:=`
  resolve normally from any patch that lives in `inst/patches/`.
- **Seen in**: `inst/patches/maftools.R` (read.maf / validateMaf / summarizeMaf
  patches; all three perform `dt[, col := ...]` inside autozyme's namespace).

## 8. Upstream that overrides base `cor()` only on `library()` attach (WGCNA)

- **Symptom**: `verify_patch` baseline (activate=FALSE) subprocess errors with
  `Error in (function (x, y = NULL, use = "everything", method = ...) :
  unused arguments (weights.x = NULL, weights.y = NULL, cosine = FALSE)` —
  raised from inside `WGCNA::blockwiseModules`'s body when it calls bare
  `cor(...)`. Patched subprocess also crashes the same way until the smoke
  recipe attaches the upstream.
- **Trigger**: Upstream's `.onAttach` (or top-level on-load code) replaces
  the global `cor` with its own extended-signature variant. Inside the
  upstream's exported function bodies, the bare `cor(...)` lookup walks the
  search path until it finds the override — present only when `library(pkg)`
  has been called. `requireNamespace(pkg)` loads but does NOT attach, so the
  override is invisible and `stats::cor` (which lacks the extra args) is
  what resolves. Same shape as CAVEATS #1 (MAST callName reflection) and #3
  (RCTD S4 cache) — those are reflection-on-call-form; this one is
  reflection-on-search-path.
- **Fix**: Inside the smoke `load` recipe, call
  `suppressPackageStartupMessages(library(WGCNA))` once. Untimed (load runs
  before timing starts), runs the same in baseline and patched, no
  fair-comparison bias. Note that the patch's `requireNamespace("WGCNA")`
  gate at file scope is NOT enough — it only loads the namespace.
- **Seen in**: `inst/patches/wgcna.R` (blockwiseModules smoke load).

---

For Python-side equivalents (TF+numba threading, obspy entry-point cache, etc.)
see `../autozyme_py/CAVEATS.md`.
