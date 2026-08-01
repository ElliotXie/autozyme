# Contract: release-locked scDblFinder default workflow accelerator.

.skip_if_no_exact_scdblfinder <- function() {
  testthat::skip_if_not_installed("scDblFinder")
  testthat::skip_if_not_installed("Matrix")
  if (!identical(as.character(utils::packageVersion("scDblFinder")), "1.27.6")) {
    testthat::skip("contract is pinned to scDblFinder 1.27.6")
  }
  autozyme::activate("scdblfinder")
}

test_that("scdblfinder release guard covers body and formals for five targets", {
  .skip_if_no_exact_scdblfinder()
  registry <- get(".zyme_registry", envir = asNamespace("autozyme"))
  targets <- registry[["scdblfinder"]]$targets
  patch_env <- environment(targets$scDblFinder)

  expect_true(get(".scdblfinder_release_ok", envir = patch_env))
  expect_identical(
    get(".scdblfinder_actual_hashes", envir = patch_env),
    get(".scdblfinder_expected_hashes", envir = patch_env)
  )
  expect_setequal(
    names(targets),
    c("scDblFinder", ".defaultProcessing", ".evaluateKNN", "cxds2",
      "createDoublets")
  )
})

test_that("scdblfinder public-driver rewrite is narrow and callable", {
  .skip_if_no_exact_scdblfinder()
  registry <- get(".zyme_registry", envir = asNamespace("autozyme"))
  patch_env <- environment(registry[["scdblfinder"]]$targets$scDblFinder)
  driver <- get(".driver_scDblFinder", envir = patch_env)
  original <- get(".orig_scDblFinder", envir = patch_env)
  rewrite <- get(".sel_features_rewrite", envir = patch_env)

  expect_identical(rewrite$count, 1L)
  expect_identical(formals(driver), formals(original))
  expect_true(is.function(driver))

  driver_body <- paste(deparse(body(driver), width.cutoff = 500L), collapse = "\n")
  expect_match(driver_body, "identical\\(sel_features, row.names\\(sce\\)\\)")
  expect_match(driver_body, "gc\\(verbose = FALSE\\)")
  expect_match(driver_body, "gc\\(verbose = FALSE, full = TRUE\\)")
})

test_that("scdblfinder internal calls fall back outside public context", {
  .skip_if_no_exact_scdblfinder()

  set.seed(7)
  pca <- matrix(rnorm(30L * 4L), nrow = 30L, ncol = 4L,
                dimnames = list(paste0("c", seq_len(30L)), NULL))
  ctype <- factor(rep(c("real", "doublet"), c(20L, 10L)))
  origins <- factor(rep(NA_character_, 30L))
  patched_knn <- scDblFinder:::.evaluateKNN(
    pca, ctype, origins, expected = NULL, k = c(3L, 5L), BNPARAM = NULL
  )
  vanilla_knn <- autozyme::with_disabled(scDblFinder:::.evaluateKNN(
    pca, ctype, origins, expected = NULL, k = c(3L, 5L), BNPARAM = NULL
  ))
  expect_identical(patched_knn, vanilla_knn)

  x <- Matrix::sparseMatrix(
    i = c(1L, 2L, 4L, 1L, 3L, 5L),
    j = c(1L, 1L, 2L, 3L, 4L, 6L),
    x = c(1, 2, 1, 3, 2, 4), dims = c(5L, 6L)
  )
  expect_identical(
    scDblFinder::cxds2(x),
    autozyme::with_disabled(scDblFinder::cxds2(x))
  )
})

test_that("scdblfinder transient context is nest-safe and restored", {
  .skip_if_no_exact_scdblfinder()
  registry <- get(".zyme_registry", envir = asNamespace("autozyme"))
  patch_env <- environment(registry[["scdblfinder"]]$targets$scDblFinder)
  active <- get(".scdblfinder_context_active", envir = patch_env)
  enter <- get(".scdblfinder_context_enter", envir = patch_env)
  exit <- get(".scdblfinder_context_exit", envir = patch_env)

  expect_false(active())
  outer <- enter()
  expect_true(active())
  inner <- enter()
  expect_true(active())
  exit(inner)
  expect_true(active())
  exit(outer)
  expect_false(active())
})

test_that("scdblfinder normalization boundary falls back under public context", {
  .skip_if_no_exact_scdblfinder()
  registry <- get(".zyme_registry", envir = asNamespace("autozyme"))
  patch_env <- environment(registry[["scdblfinder"]]$targets$scDblFinder)
  enter <- get(".scdblfinder_context_enter", envir = patch_env)
  exit <- get(".scdblfinder_context_exit", envir = patch_env)
  max_cols <- get(".scdblfinder_norm_max_cols", envir = patch_env)
  original_default <- get(".orig_defaultProcessing", envir = patch_env)
  on.exit(assign(".orig_defaultProcessing", original_default, envir = patch_env),
          add = TRUE)
  assign(
    ".orig_defaultProcessing",
    function(e, dims = NULL, doNorm = NULL) {
      list(fallback = TRUE, ncol = ncol(e), dims = dims, doNorm = doNorm)
    },
    envir = patch_env
  )

  x <- Matrix::sparseMatrix(
    i = 1L, j = 1L, x = 1, dims = c(2L, max_cols + 1L)
  )
  old <- enter()
  on.exit(exit(old), add = TRUE)
  expect_identical(
    scDblFinder:::.defaultProcessing(x, dims = 20L),
    list(fallback = TRUE, ncol = max_cols + 1L, dims = 20L, doNorm = NULL)
  )
  exit(old)
})

test_that("scdblfinder guarded internal fast paths match upstream", {
  .skip_if_no_exact_scdblfinder()
  registry <- get(".zyme_registry", envir = asNamespace("autozyme"))
  patch_env <- environment(registry[["scdblfinder"]]$targets$scDblFinder)
  enter <- get(".scdblfinder_context_enter", envir = patch_env)
  exit <- get(".scdblfinder_context_exit", envir = patch_env)

  set.seed(11)
  pca <- matrix(rnorm(40L * 5L), nrow = 40L, ncol = 5L,
                dimnames = list(paste0("c", seq_len(40L)), NULL))
  ctype <- factor(rep(c("real", "doublet"), c(27L, 13L)))
  origins <- factor(rep(NA_character_, 40L))
  vanilla_knn <- autozyme::with_disabled(scDblFinder:::.evaluateKNN(
    pca, ctype, origins, expected = NULL, k = c(3L, 7L), BNPARAM = NULL
  ))

  old <- enter()
  fast_knn <- scDblFinder:::.evaluateKNN(
    pca, ctype, origins, expected = NULL, k = c(3L, 7L), BNPARAM = NULL
  )
  exit(old)
  expect_identical(fast_knn, vanilla_knn)

  x <- Matrix::rsparsematrix(20L, 30L, density = 0.12)
  x@x <- abs(x@x) + 1
  vanilla_cxds <- autozyme::with_disabled(scDblFinder::cxds2(x))
  old <- enter()
  fast_cxds <- scDblFinder::cxds2(x)
  exit(old)
  expect_identical(fast_cxds, vanilla_cxds)

  counts <- Matrix::rsparsematrix(40L, 24L, density = 0.18)
  counts@x <- as.numeric(pmax(1L, round(abs(counts@x) * 5)))
  pairs <- cbind(seq_len(12L), 13:24)
  clusters <- factor(rep(letters[1:3], each = 8L))
  set.seed(193)
  vanilla_doublets <- autozyme::with_disabled(
    scDblFinder::createDoublets(
      counts, pairs, clusters = clusters,
      adjustSize = 0.25, halfSize = 0.25, resamp = 0.25
    )
  )
  vanilla_rng <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  set.seed(193)
  old <- enter()
  fast_doublets <- scDblFinder::createDoublets(
    counts, pairs, clusters = clusters,
    adjustSize = 0.25, halfSize = 0.25, resamp = 0.25
  )
  exit(old)
  fast_rng <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  expect_identical(fast_doublets, vanilla_doublets)
  expect_identical(fast_rng, vanilla_rng)
})

test_that("scdblfinder public wrapper scopes context to one supported call", {
  .skip_if_no_exact_scdblfinder()
  testthat::skip_if_not_installed("SingleCellExperiment")
  testthat::skip_if_not_installed("BiocParallel")
  registry <- get(".zyme_registry", envir = asNamespace("autozyme"))
  target <- registry[["scdblfinder"]]$targets$scDblFinder
  patch_env <- environment(target)
  active <- get(".scdblfinder_context_active", envir = patch_env)
  original_driver <- get(".driver_scDblFinder", envir = patch_env)
  original_upstream <- get(".orig_scDblFinder", envir = patch_env)
  on.exit(assign(".driver_scDblFinder", original_driver, envir = patch_env),
          add = TRUE)
  on.exit(assign(".orig_scDblFinder", original_upstream, envir = patch_env),
          add = TRUE)

  probe_driver <- function(sce, clusters = NULL, samples = NULL,
                           clustCor = NULL, artificialDoublets = NULL,
                           knownDoublets = NULL,
                           knownUse = c("discard", "positive"), dbr = NULL,
                           dbr.sd = NULL, dbr.per1k = 0.008, nfeatures = 1352,
                           dims = 20, k = NULL, removeUnidentifiable = TRUE,
                           includePCs = 19, propRandom = 0, propMarkers = 0,
                           aggregateFeatures = FALSE,
                           returnType = c("sce", "table", "full", "counts", "scores"),
                           BNPARAM = NULL, score = c("xgb", "weighted", "ratio"),
                           processing = "default", metric = "logloss",
                           nrounds = 0.25, max_depth = 4, iter = 3,
                           trainingFeatures = NULL, unident.th = NULL,
                           multiSampleMode = c("split", "singleModel",
                                               "singleModelSplitThres", "asOne"),
                           threshold = TRUE, verbose = TRUE,
                           BPPARAM = BiocParallel::SerialParam(
                             progressbar = verbose
                           ),
                           ...) {
    list(
      active = active(), cells = ncol(sce), verbose = verbose,
      bp_class = class(BPPARAM)[1L], workers = BiocParallel::bpnworkers(BPPARAM)
    )
  }
  assign(".driver_scDblFinder", probe_driver, envir = patch_env)
  assign(".orig_scDblFinder", probe_driver, envir = patch_env)

  counts <- Matrix::sparseMatrix(
    i = rep(1:3, 4L), j = rep(1:4, each = 3L), x = 1,
    dims = c(3L, 4L)
  )
  sce <- SingleCellExperiment::SingleCellExperiment(list(counts = counts))
  result <- target(
    sce, verbose = FALSE,
    BPPARAM = BiocParallel::SerialParam(progressbar = FALSE)
  )
  expected_fast <- list(
    active = TRUE, cells = 4L, verbose = FALSE,
    bp_class = "SerialParam", workers = 1L
  )
  expect_identical(result, expected_fast)
  expect_false(active())

  # Omitting BPPARAM exercises the unchanged upstream formal expression.  It
  # must resolve SerialParam from the patch environment and still enter the
  # supported public fast context.
  expect_identical(target(sce, verbose = FALSE), expected_fast)
  expect_false(active())

  # A dense counts assay is out of scope.  With BPPARAM omitted, the wrapper
  # must still resolve the formal and forward to the captured upstream target
  # with no transient context left active.
  dense_sce <- SingleCellExperiment::SingleCellExperiment(
    list(counts = matrix(1, nrow = 3L, ncol = 4L))
  )
  expected_fallback <- probe_driver(dense_sce, verbose = FALSE)
  expect_identical(target(dense_sce, verbose = FALSE), expected_fallback)
  expect_false(active())

  # Logical compressed sparse matrices are CsparseMatrix instances but were
  # not part of the five attested tiers.  They must not enter the dgC fast
  # context even though they expose an x slot and contain no NA values.
  logical_counts <- counts > 0
  expect_s4_class(logical_counts, "lgCMatrix")
  logical_sce <- SingleCellExperiment::SingleCellExperiment(
    list(counts = logical_counts)
  )
  expected_logical <- probe_driver(logical_sce, verbose = FALSE)
  expect_identical(target(logical_sce, verbose = FALSE), expected_logical)
  expect_false(active())
})

test_that("scdblfinder RNG snapshots restore retry state atomically", {
  .skip_if_no_exact_scdblfinder()
  registry <- get(".zyme_registry", envir = asNamespace("autozyme"))
  patch_env <- environment(registry[["scdblfinder"]]$targets$scDblFinder)
  snapshot <- get(".scdblfinder_rng_snapshot", envir = patch_env)
  restore <- get(".scdblfinder_rng_restore", envir = patch_env)

  set.seed(917)
  state <- snapshot()
  expected <- runif(5L)
  restore(state)
  expect_identical(runif(5L), expected)

  old_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
  rm(".Random.seed", envir = globalenv())
  absent <- snapshot()
  runif(1L)
  restore(absent)
  expect_false(exists(".Random.seed", envir = globalenv(), inherits = FALSE))
})
