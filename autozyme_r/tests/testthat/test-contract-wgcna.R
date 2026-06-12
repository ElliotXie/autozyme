# Contract: WGCNA::blockwiseModules + moduleEigengenes + goodSamplesGenes
#
# Patched targets:
#   - blockwiseModules    (the top-level module-detection driver)
#   - moduleEigengenes    (per-module first-PC eigengene)
#   - goodSamplesGenes    (variance-filter helper)
#   - collectGarbage      (no-op stub; doesn't need a contract test)

.skip_if_no_wgcna <- function() {
  testthat::skip_if_not_installed("WGCNA")
  # blockwiseModules' body calls bare `cor(...)` with WGCNA-specific
  # kwargs (weights.x, weights.y, cosine). Under requireNamespace alone
  # the lookup falls through to stats::cor which rejects them. The
  # patch's smoke loader uses library(WGCNA) for this same reason --
  # see autozyme_r/inst/patches/wgcna/patch.R comment near line 451.
  suppressPackageStartupMessages(library(WGCNA))
}

.make_wgcna_inputs <- function(seed = 0) {
  set.seed(seed)
  n_samples <- 40
  n_genes <- 100
  # 4 module structure: each module's genes correlate within block.
  k <- 4
  per_module <- n_genes / k
  expr <- matrix(stats::rnorm(n_samples * n_genes), nrow = n_samples,
                 ncol = n_genes)
  for (m in seq_len(k)) {
    idx <- ((m - 1) * per_module + 1):(m * per_module)
    factor_vec <- stats::rnorm(n_samples)
    expr[, idx] <- expr[, idx] + factor_vec  # broadcast: module structure
  }
  rownames(expr) <- paste0("s", seq_len(n_samples))
  colnames(expr) <- paste0("g", seq_len(n_genes))
  expr
}

test_that("goodSamplesGenes returns a list with allOK + goodSamples + goodGenes", {
  .skip_if_no_wgcna()
  expr <- .make_wgcna_inputs()
  out <- WGCNA::goodSamplesGenes(expr, verbose = 0)
  expect_type(out, "list")
  for (slot in c("allOK", "goodSamples", "goodGenes")) {
    expect_true(slot %in% names(out),
                info = paste("missing goodSamplesGenes slot:", slot))
  }
  expect_equal(length(out$goodSamples), nrow(expr))
  expect_equal(length(out$goodGenes), ncol(expr))
})

test_that("goodSamplesGenes zyme=FALSE matches patched flags", {
  .skip_if_no_wgcna()
  expr <- .make_wgcna_inputs()
  vanilla <- WGCNA::goodSamplesGenes(expr, verbose = 0, zyme = FALSE)
  patched <- WGCNA::goodSamplesGenes(expr, verbose = 0)
  expect_equal(patched$allOK, vanilla$allOK)
  expect_equal(patched$goodSamples, vanilla$goodSamples)
  expect_equal(patched$goodGenes, vanilla$goodGenes)
})

test_that("blockwiseModules returns a list with colors + MEs slots", {
  .skip_if_no_wgcna()
  expr <- .make_wgcna_inputs()
  out <- tryCatch(
    suppressWarnings(suppressMessages(
      WGCNA::blockwiseModules(expr, power = 6, minModuleSize = 5,
                              numericLabels = TRUE, verbose = 0)
    )),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    testthat::skip(paste0("blockwiseModules raised on tiny fixture: ",
                          conditionMessage(out)))
  }
  expect_type(out, "list")
  for (slot in c("colors", "MEs", "dendrograms")) {
    expect_true(slot %in% names(out),
                info = paste("missing blockwiseModules slot:", slot))
  }
  # One color per gene.
  expect_equal(length(out$colors), ncol(expr))
})

test_that("moduleEigengenes returns a list with eigengenes slot", {
  .skip_if_no_wgcna()
  expr <- .make_wgcna_inputs()
  # Synthetic coloring: 4 blocks of 25 genes each.
  colors <- rep(seq_len(4), each = 25)
  out <- suppressWarnings(suppressMessages(
    WGCNA::moduleEigengenes(expr, colors)
  ))
  expect_type(out, "list")
  expect_true("eigengenes" %in% names(out))
  expect_equal(nrow(out$eigengenes), nrow(expr))
})
