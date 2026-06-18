# autozyme

Drop-in accelerators for scientific R packages. Each accelerator is a validated, monkey-patched fast replacement for a hot function in an upstream library (Seurat, nichenetr, slingshot, etc.). Install once; existing scripts run faster automatically.

## Install

```r
# from GitHub (the R package is in the autozyme_r/ subdir)
remotes::install_github("ElliotXie/autozyme", subdir = "autozyme_r")

# or from a local checkout
R CMD INSTALL autozyme_r
```

The package compiles C++ (Rcpp) kernels at install. This only really trips up **Windows** (no built-in compiler): install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version first, or the build fails. macOS (Xcode Command Line Tools) and Linux (gcc) normally already have a compiler.

`autozyme` itself does not Depend on any upstream package. A patch activates only if its upstream is installed in your library.

## Use

```r
library(autozyme)
# autozyme 0.1.0: activated { ... }; skipped { ... } (upstream not installed)

nichenetr::predict_ligand_activities(...)   # uses fast version transparently
```

### Supported parameter scope

Each accelerator is validated for a specific parameter envelope. Supported
parameters (`npcs`, resolution, the data itself, …) can be set freely; a handful
of parameters per patch are not on the fast path and **fall back to the upstream
implementation automatically** (correct result, just not accelerated). A few
documented approximations are validated only at their stated configuration. Each
patch ships its own `SCOPE.md` in its package directory
(`inst/patches/<patch>/SCOPE.md`) documenting its supported scope and
out-of-scope behavior.

### Opt-out levels

```r
# per-call (only namespace-fn patches; class-method / S4 patches don't expose this)
fn(..., zyme = FALSE)

# per-block (works for ALL patches incl. S4 — hard kill switch)
autozyme::with_disabled({
  fn(...)        # uses captured original
  other_fn(...)  # also original
})

# per-package
autozyme::restore("nichenetr")
autozyme::activate("nichenetr")

# session-wide
autozyme::restore_all()

# never activate (set before library())
Sys.setenv(AUTOZYME_DISABLED = "1"); library(autozyme)
```

### Parallel workers

`activate()` binds patches into the upstream namespace of the **current process**. Fork-based workers — `parallel::mclapply`, `BiocParallel::MulticoreParam`, `future::multicore` — inherit the bindings automatically. Snow-style worker processes — `parallel::makeCluster("PSOCK")`, `BiocParallel::SnowParam`, `future::multisession` — start fresh R sessions and do **not** inherit.

For snow-style clusters, re-activate inside each worker:

```r
cl <- parallel::makeCluster(4)
parallel::clusterEvalQ(cl, {
  library(autozyme)
  autozyme::activate("nichenetr")
})
# ... parLapply / parSapply ... goes here
parallel::stopCluster(cl)
```

For `BiocParallel::SnowParam`, pass an equivalent `RNGseed` / setup expression to `bpworkers()`.

### Threading

```r
autozyme::set_threads(8)
```

### Python setup (Seurat PCA / CCA / integration)

Most accelerators are pure C++/R and need nothing extra. The Seurat `RunPCA`,
`RunCCA`, and integration fast paths offload linear algebra to NumPy + SciPy via
[reticulate](https://rstudio.github.io/reticulate/); everything else works
without Python.

The first time you run `autozyme::activate("seurat")` **interactively**, autozyme
offers to set up a small dedicated Python env (a one-time download). Once it
exists, Python is warmed up at `activate()` so your first `RunPCA` call pays no
startup cost. In **non-interactive** setups (CI, Docker builds, headless
servers), run the setup once yourself:

```r
autozyme::install_python_deps()   # builds the "r-autozyme" virtualenv (numpy, scipy)
```

To point at an existing interpreter instead, set `AUTOZYME_PYTHON` (or the
standard `RETICULATE_PYTHON`) to a Python that has numpy + scipy — it takes
priority over the managed env. If no usable Python is found, the PCA/CCA paths
transparently fall back to upstream Seurat (all other speedups stay active) and
print how to enable them.

## Status and introspection

```r
autozyme::dashboard()                       # one-screen status of patches + upstream availability
autozyme::list_patches()                    # all shipped patches
autozyme::list_patches(installed = TRUE)    # only patches whose upstream is installed
autozyme::status()                          # named character vector: c(nichenetr = "active", ...)
autozyme::inspect("nichenetr")              # detailed binding info for one patch
autozyme::env_snapshot()                    # structured list for provenance / Methods capture
```

## Citation

`autozyme` releases on GitHub get a Zenodo DOI. Cite the version + the patch name(s) you activated. The activation marker logged to stderr is the canonical provenance record, e.g.

```
[autozyme] activated nichenetr -> 1 target(s) in nichenetr 2.2.1.1
```

In Methods: "We used autozyme v0.1.0 with the `nichenetr` patch (tested against nichenetr 2.2.1.1)."

## License

MIT
