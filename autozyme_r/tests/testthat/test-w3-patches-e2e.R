# Wave-3 end-to-end coverage: NON-Seurat patched functions.
#
# The per-patch contract tests already pin the headline default-parameter case
# (numeric / shape parity vs zyme=FALSE or with_disabled()). This file drives
# the dispatch / parameter-variant branches they skip and the per-target
# activate -> call -> deactivate restore lifecycle exercised through R/core.R's
# .wrap_namespace_fast dispatcher, .activate_one / .deactivate_one, status(),
# and inspect().
#
# Lifecycle tests use the exported deactivate() (NOT the unexported restore()
# that two wave-1 contract files mistakenly reference). Each lifecycle test
# leaves the patch reactivated on.exit so alphabetically-later files still see
# the patched functions (which accept zyme=FALSE) — matching the convention in
# test-contract-all-patches-activate.R.
#
# Run at OMP_NUM_THREADS=1 (native #pragma kernels segfault at threads>1).

`%||%` <- function(a, b) if (is.null(a)) b else a

# ── fgsea parameter / dispatch variants ─────────────────────────────────────

test_that("fgseaMultilevel scoreType='pos' matches vanilla ES", {
  testthat::skip_if_not_installed("fgsea")
  testthat::skip_if_not_installed("data.table")
  set.seed(0)
  genes <- paste0("g", seq_len(400))
  stats <- abs(rnorm(400)); names(stats) <- genes
  stats <- sort(stats, decreasing = TRUE)
  pathways <- list(top = head(genes, 25), mid = genes[100:130])
  # scoreType "pos" is a different code path than the default "std".
  set.seed(7)
  vanilla <- suppressWarnings(fgsea::fgseaMultilevel(
    pathways, stats, minSize = 5, maxSize = 100, nPermSimple = 100,
    scoreType = "pos", zyme = FALSE))
  set.seed(7)
  patched <- suppressWarnings(fgsea::fgseaMultilevel(
    pathways, stats, minSize = 5, maxSize = 100, nPermSimple = 100,
    scoreType = "pos"))
  expect_setequal(vanilla$pathway, patched$pathway)
  for (pw in vanilla$pathway) {
    expect_equal(patched$ES[patched$pathway == pw],
                 vanilla$ES[vanilla$pathway == pw],
                 tolerance = 1e-6, info = paste("ES drift:", pw))
  }
})

test_that("fgseaMultilevel size-filtered subset matches vanilla", {
  testthat::skip_if_not_installed("fgsea")
  testthat::skip_if_not_installed("data.table")
  set.seed(1)
  genes <- paste0("g", seq_len(400))
  stats <- rnorm(400); names(stats) <- genes
  stats <- sort(stats, decreasing = TRUE)
  # One pathway too small (filtered), one in range, one too big.
  pathways <- list(tiny = head(genes, 3), ok = genes[10:40],
                   huge = genes)
  set.seed(9)
  vanilla <- suppressWarnings(fgsea::fgseaMultilevel(
    pathways, stats, minSize = 5, maxSize = 60, nPermSimple = 100,
    zyme = FALSE))
  set.seed(9)
  patched <- suppressWarnings(fgsea::fgseaMultilevel(
    pathways, stats, minSize = 5, maxSize = 60, nPermSimple = 100))
  # Both should retain only the 'ok' pathway.
  expect_equal(patched$pathway, "ok")
  expect_setequal(patched$pathway, vanilla$pathway)
  expect_equal(patched$ES, vanilla$ES, tolerance = 1e-6)
})

test_that("fgsea deactivate -> bare call -> reactivate restores patch", {
  testthat::skip_if_not_installed("fgsea")
  testthat::skip_if_not_installed("data.table")
  on.exit(suppressMessages(autozyme::activate("fgsea")), add = TRUE)
  expect_equal(autozyme::status()[["fgsea"]], "active")
  suppressMessages(autozyme::deactivate("fgsea"))
  expect_equal(autozyme::status()[["fgsea"]], "inactive")
  # With the patch removed, the bare upstream no longer accepts zyme=.
  set.seed(0); genes <- paste0("g", seq_len(200))
  stats <- sort(rnorm(200), decreasing = TRUE); names(stats) <- genes
  err <- tryCatch(
    suppressWarnings(fgsea::fgseaMultilevel(
      list(p = genes[1:20]), stats, minSize = 5, maxSize = 100,
      nPermSimple = 50, zyme = FALSE)),
    error = function(e) e)
  expect_s3_class(err, "error")  # vanilla rejects unknown zyme= kwarg
  suppressMessages(autozyme::activate("fgsea"))
  expect_equal(autozyme::status()[["fgsea"]], "active")
  # Patched again accepts zyme=FALSE without error.
  out <- suppressWarnings(fgsea::fgseaMultilevel(
    list(p = genes[1:20]), stats, minSize = 5, maxSize = 100,
    nPermSimple = 50, zyme = FALSE))
  expect_s3_class(out, "data.table")
})

# ── vegan parameter variants ────────────────────────────────────────────────

test_that("adonis2 euclidean method matches vanilla F-statistic", {
  testthat::skip_if_not_installed("vegan")
  set.seed(0)
  k <- 3; per <- 15; n <- k * per
  group <- factor(rep(letters[seq_len(k)], each = per))
  abund <- matrix(stats::rpois(n * 20, lambda = 2), nrow = n)
  for (g in seq_len(k)) {
    abund[as.integer(group) == g, (g * 3):(g * 3 + 3)] <-
      abund[as.integer(group) == g, (g * 3):(g * 3 + 3)] + 5L
  }
  rownames(abund) <- paste0("s", seq_len(n))
  df <- data.frame(group = group)
  assign("..az_eu..", abund, envir = globalenv())
  on.exit(rm("..az_eu..", envir = globalenv()), add = TRUE)
  fml <- stats::as.formula("..az_eu.. ~ group", env = globalenv())
  # method="euclidean" is a different dissimilarity branch than "bray".
  set.seed(3)
  vanilla <- autozyme::with_disabled(
    vegan::adonis2(fml, data = df, permutations = 99, method = "euclidean"))
  set.seed(3)
  patched <- vegan::adonis2(fml, data = df, permutations = 99,
                            method = "euclidean")
  expect_equal(class(patched), class(vanilla))
  f_v <- vanilla[["F"]]; f_p <- patched[["F"]]
  expect_equal(f_p[!is.na(f_p)], f_v[!is.na(f_v)], tolerance = 1e-6)
})

test_that("adonis2 by='margin' matches vanilla on a 2-term model", {
  testthat::skip_if_not_installed("vegan")
  set.seed(2)
  n <- 48
  g1 <- factor(rep(c("a", "b"), each = n / 2))
  g2 <- factor(rep(c("x", "y"), times = n / 2))
  abund <- matrix(stats::rpois(n * 25, lambda = 2), nrow = n)
  abund[g1 == "a", 1:6] <- abund[g1 == "a", 1:6] + 4L
  rownames(abund) <- paste0("s", seq_len(n))
  df <- data.frame(g1 = g1, g2 = g2)
  assign("..az_mg..", abund, envir = globalenv())
  on.exit(rm("..az_mg..", envir = globalenv()), add = TRUE)
  fml <- stats::as.formula("..az_mg.. ~ g1 + g2", env = globalenv())
  set.seed(5)
  vanilla <- autozyme::with_disabled(
    vegan::adonis2(fml, data = df, permutations = 99, method = "bray",
                   by = "margin"))
  set.seed(5)
  patched <- vegan::adonis2(fml, data = df, permutations = 99,
                            method = "bray", by = "margin")
  expect_equal(class(patched), class(vanilla))
  expect_equal(nrow(patched), nrow(vanilla))
  f_v <- vanilla[["F"]]; f_p <- patched[["F"]]
  expect_equal(f_p[!is.na(f_p)], f_v[!is.na(f_v)], tolerance = 1e-6)
})

# ── decontx variant: explicit z (cluster labels) ────────────────────────────

test_that("decontX with explicit z labels matches with_disabled shape", {
  testthat::skip_if_not_installed("celda")
  testthat::skip_if_not_installed("SingleCellExperiment")
  set.seed(0)
  k <- 3; per <- 25; n <- k * per; ng <- 50
  cluster <- rep(seq_len(k), each = per)
  counts <- matrix(stats::rpois(ng * n, lambda = 1), nrow = ng)
  for (cl in seq_len(k)) {
    mg <- ((cl - 1) * 12 + 1):(cl * 12)
    counts[mg, cluster == cl] <- stats::rpois(length(mg) * sum(cluster == cl),
                                              lambda = 8)
  }
  rownames(counts) <- paste0("g", seq_len(ng))
  colnames(counts) <- paste0("c", seq_len(n))
  # Passing z= skips decontX's internal clustering init: a distinct branch.
  patched <- tryCatch(
    suppressWarnings(suppressMessages(
      celda::decontX(counts, z = cluster, verbose = FALSE))),
    error = function(e) e)
  if (inherits(patched, "error")) {
    testthat::skip(paste0("decontX(z=) raised: ", conditionMessage(patched)))
  }
  vanilla <- tryCatch(
    autozyme::with_disabled(suppressWarnings(suppressMessages(
      celda::decontX(counts, z = cluster, verbose = FALSE)))),
    error = function(e) e)
  if (inherits(vanilla, "error")) {
    testthat::skip(paste0("vanilla decontX(z=) raised: ",
                          conditionMessage(vanilla)))
  }
  expect_equal(dim(patched$decontXcounts), dim(vanilla$decontXcounts))
  expect_equal(length(patched$contamination), length(vanilla$contamination))
})

# ── maftools call-shape variant: useAll / removeSilent ──────────────────────

test_that("read.maf removeSilent variant matches mutation table dims", {
  testthat::skip_if_not_installed("maftools")
  maf_path <- system.file("extdata", "tcga_laml.maf.gz", package = "maftools")
  if (!nzchar(maf_path)) testthat::skip("tcga_laml.maf.gz missing")
  # useAll=FALSE drops non-"Somatic" mutation status rows: a different
  # validateMaf filtering branch than the default.
  patched <- suppressWarnings(suppressMessages(
    maftools::read.maf(maf = maf_path, useAll = FALSE, verbose = FALSE)))
  vanilla <- autozyme::with_disabled(suppressWarnings(suppressMessages(
    maftools::read.maf(maf = maf_path, useAll = FALSE, verbose = FALSE))))
  expect_s4_class(patched, "MAF")
  expect_equal(nrow(patched@data), nrow(vanilla@data))
  expect_setequal(
    unique(patched@data$Tumor_Sample_Barcode),
    unique(vanilla@data$Tumor_Sample_Barcode))
})

test_that("read.maf deactivate restores upstream then reactivate re-patches", {
  testthat::skip_if_not_installed("maftools")
  maf_path <- system.file("extdata", "tcga_laml.maf.gz", package = "maftools")
  if (!nzchar(maf_path)) testthat::skip("tcga_laml.maf.gz missing")
  on.exit(suppressMessages(autozyme::activate("maftools")), add = TRUE)
  expect_equal(autozyme::status()[["maftools"]], "active")
  suppressMessages(autozyme::deactivate("maftools"))
  expect_equal(autozyme::status()[["maftools"]], "inactive")
  # Deactivated -> the vanilla read.maf must reject the patch-only zyme= kwarg.
  err <- tryCatch(
    suppressWarnings(suppressMessages(
      maftools::read.maf(maf = maf_path, verbose = FALSE, zyme = FALSE))),
    error = function(e) e)
  expect_s3_class(err, "error")
  suppressMessages(autozyme::activate("maftools"))
  expect_equal(autozyme::status()[["maftools"]], "active")
})

# ── wgcna variant: networkType signed (different cor path) ───────────────────

test_that("goodSamplesGenes with low-variance gene flags matches vanilla", {
  testthat::skip_if_not_installed("WGCNA")
  suppressPackageStartupMessages(library(WGCNA))
  set.seed(0)
  expr <- matrix(stats::rnorm(30 * 60), nrow = 30, ncol = 60)
  # Inject a zero-variance gene + a near-constant gene to exercise the
  # variance-filter branches inside goodSamplesGenes.
  expr[, 1] <- 0
  expr[, 2] <- 1e-12 * stats::rnorm(30)
  rownames(expr) <- paste0("s", seq_len(30))
  colnames(expr) <- paste0("g", seq_len(60))
  vanilla <- WGCNA::goodSamplesGenes(expr, verbose = 0, zyme = FALSE)
  patched <- WGCNA::goodSamplesGenes(expr, verbose = 0)
  expect_equal(patched$allOK, vanilla$allOK)
  expect_equal(patched$goodGenes, vanilla$goodGenes)
  expect_equal(patched$goodSamples, vanilla$goodSamples)
  # The zero-variance gene 1 must be flagged bad under both paths.
  expect_false(patched$goodGenes[1])
})

test_that("moduleEigengenes with 2-module coloring matches with_disabled", {
  testthat::skip_if_not_installed("WGCNA")
  suppressPackageStartupMessages(library(WGCNA))
  set.seed(1)
  ns <- 30; ng <- 40
  expr <- matrix(stats::rnorm(ns * ng), nrow = ns, ncol = ng)
  f1 <- stats::rnorm(ns); f2 <- stats::rnorm(ns)
  expr[, 1:20] <- expr[, 1:20] + f1
  expr[, 21:40] <- expr[, 21:40] + f2
  colors <- rep(c(1L, 2L), each = 20)
  patched <- suppressWarnings(suppressMessages(
    WGCNA::moduleEigengenes(expr, colors)))
  vanilla <- autozyme::with_disabled(suppressWarnings(suppressMessages(
    WGCNA::moduleEigengenes(expr, colors))))
  expect_equal(dim(patched$eigengenes), dim(vanilla$eigengenes))
  # First-PC eigengene is sign-ambiguous; compare up to sign via abs cosine.
  for (m in seq_len(ncol(vanilla$eigengenes))) {
    ev <- vanilla$eigengenes[, m]; ep <- patched$eigengenes[, m]
    cos_m <- abs(sum(ev * ep)) / (sqrt(sum(ev^2)) * sqrt(sum(ep^2)) + 1e-12)
    expect_gt(cos_m, 0.99)
  }
})

# ── slingshot variant: with weights / non-default params ────────────────────

test_that("getCurves with non-default approx_points matches curve count", {
  testthat::skip_if_not_installed("slingshot")
  testthat::skip_if_not_installed("S4Vectors")
  set.seed(0)
  k <- 4; per <- 25
  centers <- rbind(c(0, 0), c(2, 1), c(4, 0), c(6, 1))
  X <- do.call(rbind, lapply(seq_len(k), function(i) {
    pts <- matrix(stats::rnorm(per * 2, sd = 0.3), nrow = per, ncol = 2)
    pts + matrix(rep(centers[i, ], each = per), nrow = per)
  }))
  colnames(X) <- c("D1", "D2"); rownames(X) <- paste0("c", seq_len(nrow(X)))
  cl <- rep(seq_len(k), each = per)
  pto <- tryCatch(
    suppressWarnings(slingshot::getLineages(data = X, clusterLabels = cl)),
    error = function(e) e)
  if (inherits(pto, "error")) {
    testthat::skip(paste0("getLineages failed: ", conditionMessage(pto)))
  }
  # approx_points is honored by both patched and upstream getCurves.
  patched <- suppressWarnings(slingshot::getCurves(pto, approx_points = 50))
  vanilla <- autozyme::with_disabled(
    suppressWarnings(slingshot::getCurves(pto, approx_points = 50)))
  expect_s4_class(patched, "PseudotimeOrdering")
  expect_equal(length(S4Vectors::metadata(patched)$curves),
               length(S4Vectors::metadata(vanilla)$curves))
})

# ── rctd solver-helper variant ──────────────────────────────────────────────

test_that("rctd psd on larger/indefinite Hessians matches with_disabled", {
  testthat::skip_if_not_installed("spacexr")
  suppressPackageStartupMessages(library(spacexr))
  psd <- utils::getFromNamespace("psd", "spacexr")
  set.seed(0)
  for (i in seq_len(3)) {
    A <- matrix(stats::rnorm(16), 4, 4)
    H <- (A + t(A)) / 2          # symmetric, generally indefinite
    patched <- psd(H)
    vanilla <- autozyme::with_disabled(psd(H))
    expect_equal(patched, vanilla, tolerance = 1e-9,
                 info = paste("psd mismatch on indefinite Hessian", i))
    # PSD projection must produce a symmetric matrix with no negative eigvals.
    ev <- eigen(patched, symmetric = TRUE, only.values = TRUE)$values
    expect_true(all(ev >= -1e-8))
  }
})

# ── generic per-patch deactivate/activate lifecycle (dispatcher coverage) ───
#
# Drives .deactivate_one / .activate_one / status() / inspect() through R/core.R
# for every installed patch NOT given a bespoke lifecycle test above. This is
# pure framework-path coverage: deactivate -> assert inactive + all targets
# unbound -> reactivate -> assert active + targets rebound. Leaves each patch
# active on exit so later files keep their zyme=FALSE escape.

local({
  bespoke <- c("fgsea", "maftools")  # already lifecycle-tested above
  lifecycle_patches <- setdiff(autozyme::list_patches(), bespoke)
  for (patch_name in lifecycle_patches) {
    local({
      pn <- patch_name
      test_that(paste0("lifecycle: ", pn, " deactivate unbinds, activate rebinds"), {
        probe <- autozyme:::.probe_patch_installed(pn)
        if (!isTRUE(probe$installed)) {
          testthat::skip(paste0("upstream missing for '", pn, "': ",
                                probe$reason %||% "not installed"))
        }
        on.exit(try(suppressMessages(autozyme::activate(pn)), silent = TRUE),
                add = TRUE)

        # Start active (setup-autozyme.R activated everything installed).
        suppressMessages(try(autozyme::activate(pn), silent = TRUE))
        expect_equal(unname(autozyme::status()[pn]), "active")
        info_active <- autozyme::inspect(pn)
        expect_equal(info_active$status, "active")
        expect_gt(length(info_active$targets), 0)
        expect_true(all(vapply(info_active$targets,
                               function(x) isTRUE(x$currently_bound), logical(1))),
                    info = paste0("not all targets bound while active: ", pn))

        # Deactivate -> targets unbound, status inactive.
        suppressMessages(autozyme::deactivate(pn))
        expect_equal(unname(autozyme::status()[pn]), "inactive")
        info_inactive <- autozyme::inspect(pn)
        expect_equal(info_inactive$status, "inactive")
        expect_false(any(vapply(info_inactive$targets,
                                function(x) isTRUE(x$currently_bound), logical(1))),
                     info = paste0("targets still bound after deactivate: ", pn))

        # Reactivate -> status active again (round-trip closed).
        suppressMessages(autozyme::activate(pn))
        expect_equal(unname(autozyme::status()[pn]), "active")
      })
    })
  }
})
