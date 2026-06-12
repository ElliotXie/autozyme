# Patch for celda::decontX.
#
# Lifted from autozyme task `test_decontx`. Four namespace overrides on
# `celda` that, together, accelerate the full decontX call without hoisting
# input-dependent initialization outside the patched API:
#
#   1. decontXLogLik             -> no-op (returns 0). LL is diagnostic; the
#                                   EM convergence test in celda uses max|Δθ|.
#   2. .decontxInitializeZ       -> same UMAP/dbscan init, but pass the
#                                   runner-selected thread count to
#                                   scater::calculateUMAP instead of celda's
#                                   hardcoded single-thread call.
#   3. calculateNativeMatrix     -> sparse R implementation matching celda's
#                                   native-count calculation, so decontXcounts
#                                   remains a real output.
#   4. decontXEM                 -> Rcpp wrapper around fast_decontXEM_cpp.
#                                   std::thread parallelism with per-thread
#                                   phi/native accumulators; threshold
#                                   nC >= 500 (round 61).
#
# The C++ kernel lives in src/decontx.cpp (RcppEigen + std::thread, no OMP).
# `// [[Rcpp::depends(RcppEigen)]]` is set there; scaffold_cpp_patch added
# RcppEigen to DESCRIPTION's LinkingTo.

if (requireNamespace("celda",         quietly = TRUE) &&
    requireNamespace("Matrix",        quietly = TRUE) &&
    requireNamespace("MCMCprecision", quietly = TRUE) &&
    requireNamespace("SeuratObject",  quietly = TRUE) &&
    requireNamespace("scater",        quietly = TRUE) &&
    requireNamespace("SingleCellExperiment", quietly = TRUE) &&
    requireNamespace("withr",         quietly = TRUE) &&
    requireNamespace("dbscan",        quietly = TRUE) &&
    requireNamespace("yaml",          quietly = TRUE)) {

  # File-scope captures.
  .decontx_orig_decontX        <- utils::getFromNamespace("decontX",            "celda")

  # 1. decontXLogLik: restore upstream C++ call. The original is a single
  #    .Call into celda's C++ kernel — fast enough that a no-op saves nothing
  #    measurable, but returning 0 breaks downstream LL-based model selection.
  .orig_decontXLogLik <- utils::getFromNamespace("decontXLogLik", "celda")
  fast_decontXLogLik <- function(counts, theta, eta, phi, z, pseudocount, zyme = TRUE) {
    .orig_decontXLogLik(counts, theta, eta, phi, z, pseudocount)
  }

  # 2. .decontxInitializeZ: keep the init inside the decontX call, but let
  #    scater/uwot use the current AutoZyme thread budget.
  fast_decontxInitializeZ <- function(object, varGenes = 5000, dbscanEps = 1,
                                      estimateCellTypes = TRUE, seed = 12345) {
    if (!methods::is(object, "SingleCellExperiment")) {
      sce <- SingleCellExperiment::SingleCellExperiment(
        assays = list(counts = object)
      )
    } else {
      sce <- object
    }
    sce <- scater::logNormCounts(sce, log = TRUE)
    n_threads <- autozyme::auto_threads(cap = 8L)
    call_umap <- function() {
      scater::calculateUMAP(
        sce, ntop = varGenes,
        n_threads = n_threads,
        exprs_values = "logcounts"
      )
    }
    resUmap <- if (!is.null(seed)) withr::with_seed(seed, call_umap()) else call_umap()
    z <- NULL
    if (isTRUE(estimateCellTypes)) {
      totalClusters <- 1
      iter <- 1
      while (totalClusters <= 1 && dbscanEps > 0 && iter < 10) {
        resDbscan <- dbscan::dbscan(resUmap, dbscanEps)
        dbscanEps <- dbscanEps - (0.25 * dbscanEps)
        totalClusters <- length(unique(resDbscan$cluster))
        iter <- iter + 1
      }
      if (totalClusters == 1) {
        cl <- stats::kmeans(t(SingleCellExperiment::logcounts(sce)), 2)
        z <- cl$cluster
      } else {
        z <- resDbscan$cluster
      }
    }
    list(z = z, umap = resUmap)
  }

  # 3. calculateNativeMatrix: real sparse native matrix, matching celda's
  #    normp weighting. This keeps res$decontXcounts meaningful for users.
  fast_calculateNativeMatrix <- function(counts, theta, eta, phi, z, pseudocount) {
    if (!methods::is(counts, "dgCMatrix")) {
      counts <- methods::as(counts, "CsparseMatrix")
    }
    native <- counts
    nr <- nrow(counts)
    i_idx <- counts@i + 1L
    j_idx <- rep.int(seq_len(ncol(counts)), diff(counts@p))
    k_idx <- as.integer(z)[j_idx]
    lin <- i_idx + nr * (k_idx - 1L)
    phi_lk <- phi[lin]
    eta_lk <- eta[lin]
    theta_j <- theta[j_idx]
    pnative <- (phi_lk + pseudocount) * (theta_j + pseudocount)
    pcontamin <- (eta_lk + pseudocount) * (1 - theta_j + pseudocount)
    native@x <- counts@x * (pnative / (pnative + pcontamin))
    native
  }

  # 4. decontXEM: Rcpp kernel. Note this signature MUST match upstream's
  #    decontXEM (no zyme= kwarg) because celda calls it internally via
  #    positional args from .decontXoneBatch. The auto-wrapper still gives
  #    us the with_disabled() short-circuit; per-call zyme= isn't useful for
  #    internal-only fns.
  fast_decontXEM <- function(counts, counts_colsums, theta,
                             estimate_eta, eta, phi, z,
                             estimate_delta, delta, pseudocount) {
    n_threads <- autozyme::auto_threads(cap = 8L)
    fast_decontXEM_cpp(counts, counts_colsums, theta,
                       estimate_eta, eta, phi, as.integer(z),
                       estimate_delta, delta, pseudocount,
                       as.integer(n_threads))
  }

  # ---------------------------------------------------------------------------
  # Smoke recipe
  #
  # Fair-comparison boundary: `decontX(counts, ...)` is the user-facing API
  # the patch claims to optimize. Input-dependent UMAP/dbscan initialization
  # stays inside that call for both baseline and patched runs.
  # ---------------------------------------------------------------------------
  .decontx_smoke_load <- function(task_dir, tier) {
    suppressPackageStartupMessages({
      library(SeuratObject)
      library(celda)
    })
    n <- autozyme::auto_threads(cap = 8L)
    Sys.setenv(OMP_NUM_THREADS      = as.character(n),
               OPENBLAS_NUM_THREADS = as.character(n),
               MKL_NUM_THREADS      = as.character(n))
    if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
      RhpcBLASctl::blas_set_num_threads(n)
      RhpcBLASctl::omp_set_num_threads(n)
    }

    task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
    ds <- Filter(function(d) identical(d$tier, tier), task$datasets)
    if (length(ds) == 0L) stop("no dataset for tier '", tier, "' in task.yaml")
    data_path <- resolve_dataset_path(task_dir, ds[[1]]$path)
    seu <- readRDS(data_path)
    counts <- SeuratObject::LayerData(seu, assay = "RNA", layer = "counts")

    SEED       <- 12345L
    MAX_ITER   <- 500L
    DELTA      <- c(10, 10)
    VAR_GENES  <- 5000L
    DBSCAN_EPS <- 1.0

    list(counts = counts,
         seed = SEED, max_iter = MAX_ITER, delta = DELTA,
         var_genes = VAR_GENES, dbscan_eps = DBSCAN_EPS)
  }

  .decontx_smoke_call <- function(inputs, tier = "tiny") {
    res <- celda::decontX(
      inputs$counts, z = NULL, batch = NULL,
      maxIter = inputs$max_iter, delta = inputs$delta,
      estimateDelta = TRUE,
      varGenes = inputs$var_genes, dbscanEps = inputs$dbscan_eps,
      seed = inputs$seed, verbose = FALSE)
    list(res = res, counts = inputs$counts)
  }

  .decontx_smoke_save <- function(result, dir, tier = "tiny", ...) {
    res    <- result$res
    counts <- result$counts
    cells  <- colnames(counts)
    est    <- res$estimates[[1]]
    colsums_vec <- Matrix::colSums(res$decontXcounts)
    out <- list(
      cells         = cells,
      contamination = setNames(as.numeric(res$contamination), cells),
      z             = setNames(as.integer(as.character(res$z)), cells),
      colsums       = setNames(as.numeric(colsums_vec), cells),
      iteration     = est$iteration
    )
    saveRDS(out, file.path(dir, "result.rds"))
  }

  register_patch(
    name = "decontx",
    upstream = "celda",
    targets = list(
      decontXLogLik         = fast_decontXLogLik,
      .decontxInitializeZ   = fast_decontxInitializeZ,
      calculateNativeMatrix = fast_calculateNativeMatrix,
      decontXEM             = fast_decontXEM
    ),
    smoke = list(
      load = .decontx_smoke_load,
      call = .decontx_smoke_call,
      save = .decontx_smoke_save
    ),
    tested_against = "celda 1.24.0",
    tested_upstream_versions = list(celda = "1.24.0")
  )
}
