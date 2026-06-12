# Package — autozyme-framework

You are the **packaging agent**. Lift one converged task into a registered patch in the unified accelerator package.

Run once per converged task. Do not iterate new optimizations. Translate the working `pipeline/run.{R,py}` into `autozyme_{r,py}`, install it, and prove via `zyme attest` that the package patch reproduces the task metrics.

Read `CAVEATS.md` and `memory/discoveries.md` before editing. Existing patches in `inst/patches/` and `src/autozyme/` are sibling references when convention is unclear.

## Steps

1. **Translate** — read `pipeline/run.{R,py}`; lift only live replacements routed by `patch_namespace()`, `inline_upstream()`, or `install_override()`. Drop dead earlier-round artifacts. Identify patch kind (namespace function, class method, S4 method, C++ kernel + glue).
2. **Release contract** — define attested signature, supported fast path, fallback surface (see below). Add fallback gates for unsupported branches.
3. **Write the patch + smoke** — see **Patch Locations**, **Registering**, **Smoke Contract**.
4. **`zyme package sync-manifests --apply`** — regenerates `UPSTREAMS` / `.zyme_upstreams`; do not hand-edit.
5. **Install** — `R CMD INSTALL autozyme_r` or `pip install -e autozyme_py`.
6. **`zyme package preflight`** — runs lint + portability scan + smoke-parity. Each finding's message contains the fix; act on it and re-run until clean. This is the default `zyme attest` gate.
7. **`zyme attest`** — final headline measurement. All rows must pass. Run `zyme package check-intercept --patch <name>` first when the patched timing looks identical to baseline (count=0 → patch didn't fire). Pass `--no-preflight` only after step 6 is green.

## Release Contract

Before timing, define:

1. **Attested signature** — the exact call from `task.yaml::signature` and the smoke `call`.
2. **Supported fast path** — argument values, object classes, data layers, and thread knobs the patch actually implements.
3. **Fallback surface** — every argument value or branch outside the supported path.

The fast path must not be wider than the attested signature plus explicitly validated harmless variations. Read upstream signatures (not only `pipeline/run`): R `args(getS3method(...))` / `getMethod(...)`; Python `inspect.signature` + branches on kwargs / object type.

| Mismatch | Required action |
|---|---|
| Patch intercepts branches not in the attested signature | Add explicit fallback gates before benchmarking. |
| Patch changes defaults or silently rewrites user parameters | Restore upstream defaults or fall back. |
| Intentionally broader contract | Add task fixtures and metrics for that broader contract first. |
| Escape flag (`zyme=FALSE`) present | Verify escape returns upstream-equivalent output. |

## Portability

When preflight reports portability `hits[]`, triage each hit and apply fixes to `patch.R` (not to `pipeline/run.{R,py}`) so the dev-platform hot path stays behind an OS-specific branch.

- **Portable kernel** — repeated numeric op over rows / features / genes / cells / blocks → RcppParallel, BLAS/crossprod, `Matrix` (R) or NumPy/SciPy/Numba (Python).
- **Package-specific worker** — heavy per-task, state exportable once → PSOCK/future only with isolated small+medium dev evidence of no regression. Do not replace `mclapply` with generic PSOCK just because Windows cannot fork; prefer `.zyme_mclapply` serial fallback on non-Unix when there is no kernel candidate.
- **Not a portability target** — arbitrary closures, S4/dataframe glue, enrichment/list logic, I/O orchestration. Leave unchanged.

Add a disable switch (`AUTOZYME_PORTABLE_KERNEL=0` or task-specific env) that falls back to the dev path; on runtime failure on dev, fall back silently unless it's a correctness bug.

Append to `memory/discoveries.md`:

```
## DISCOVERY: portability (pkg) - <date>
- Scan verdict: <before -> after>
- Fix: <portable backend, serial fallback, or blocked reason>
- Disable switch: <ENV/option>
```

## Cross-platform

Single machine, no other-platform attest possible:

- `zyme package preflight` passing is the cross-platform ship gate for **correctness** (scan statically detects fork/PSOCK/platform-only failures).
- Cross-platform **speed** cannot be measured here — record as `pending`. CI (e.g. GitHub Actions on `ubuntu-latest` + `windows-latest`) can fill in with `zyme attest --no-preflight` after the patch ships.

## Patch Locations

- R: `autozyme_r/inst/patches/<name>/patch.R`; C++ kernels: `autozyme_r/src/<name>.cpp`.
- Python: `autozyme_py/src/autozyme/<name>/__init__.py`.
- Lazy-loaded patch locations, not eager core files (`R/`, top-level imports).
- One patch per converged task. `<name>` is the registry key for `activate()` / `deactivate()` / `verify_patch()`.
- Update `CHANGELOG.md` (and `manifest.yml` when that patch directory uses one).

## Registering

R:

```r
register_patch(
  name = "<patch_name>",
  upstream = "<upstream_pkg>",
  targets = list(
    fn = fast_fn,
    generic = list(kind = "s4", signature = "<S4_class>", fn = fast_generic)
  ),
  smoke = list(load = ..., call = ..., save = ...),
  tested_against = "<upstream_pkg> <X.Y.Z>",
  tested_upstream_versions = list("<upstream_pkg>" = c("<X.Y.Z>"))
)
```

Python:

```python
register_patch(
    name="<patch_name>",
    targets=[("upstream.module", "fn_name", fast_fn), ...],
    smoke={"load": ..., "call": ..., "save": ...},
    tested_against="<upstream_pkg> <X.Y.Z>",
    tested_upstream_versions={"<upstream_pkg>": ["<X.Y.Z>"]},
)
```

Use `packageVersion()` / `importlib.metadata.version()` for `tested_against`. `tested_upstream_versions` must name versions CI can recreate. For normal releases, verify the exact `pkg==version` / package-manager install works in a clean environment. For a GitHub/default-branch source or other unreleased dev build, keep the real dev/local version string and add/update the CI install spec so the exact commit is installed. Do not write a non-existent release number just because the local package metadata looks release-like.

## Smoke Contract

`verify_patch` runs `load` untimed, times `call`, then runs `save` for evaluation.

| Work | Where |
|---|---|
| File reads, object construction, preprocessing | `load` |
| The upstream public API call the patch targets | `call` |
| Output serialization matching `evaluate.{R,py}` | `save` |

`call` must invoke the upstream public API, not a patched internal helper. `save` accepts `tier` and `...` (R) or `**kwargs` (Python). Read `evaluate.{R,py}` first; it dictates filenames and shape. Task-local `attest/smoke.{R,py}` overrides patch smoke when present.

## Footguns (not lint-covered — agent judgment)

- **`zyme = TRUE` convention** — namespace-function patches add `zyme = TRUE`; `zyme = FALSE` returns upstream. If upstream already has `zyme`, use a non-clashing escape name.
- **S4 disabled dispatch** — `register_patch` handles it automatically for namespace functions. R S4 patches must check `autozyme::is_disabled()` themselves.
- **Multiple bind sites** — Python re-exports / aliases must all be registered if users can reach the fn through multiple paths.
- **Reflection on call form** — some upstreams inspect `match.call()` / `inspect.signature`; use `library(pkg)` in smoke `load` and the unqualified name rather than `pkg::fn(...)`.

## C++ Scaffolding (R native kernels)

Put the kernel in `autozyme_r/src/<name>.cpp` with `// [[Rcpp::depends(X)]]` tags. Run `autozyme_r/tools/scaffold_cpp_patch.R`. Reference compiled functions by bare name from `patch.R`. No inline `Rcpp::cppFunction()` / `sourceCpp(code=…)` — caught by `R-RCPP-INLINE` lint.

## Maintenance

`zyme package check-versions` lists every patch whose `tested_against` pin differs from the installed upstream — run it when upstream packages have been upgraded since the patch was lifted. Drift is not automatically a failure (the API may not have moved), but each drift row is a candidate for re-attest before relying on the speedup number.

## Constraints

- Do not edit task code: `pipeline/`, `evaluate.{R,py}`, `task.yaml`, `data/`, `reference.{R,py}`.
- Do not add unused dependencies.
- Do not release/upload (`R CMD build`, PyPI); the user handles external publication.
- Do not loosen task thresholds when metrics fail — fix the patch or the contract scope.

## Output

- Patch file path (+ C++ kernel paths if any).
- `zyme attest` table: tier, baseline_sec, patched_sec, speedup_x, all_pass. Cross-platform rows listed as `pending` when not measured.
- Portability scan verdict (before → after) + disable switch when portability fixes were applied.
- New `CAVEATS.md` entry when a fresh trap cost >30 minutes to diagnose.
