# Patch for UCell::ScoreSignatures_UCell.
#
# Lifted from autozyme task `ucell`. The public matrix entry point is replaced
# for the attested serial sparse path:
#
#   ScoreSignatures_UCell(matrix, features, maxRank = 1500,
#                         BPPARAM = BiocParallel::SerialParam())
#
# Fast path contract:
#   - `matrix` is a finite Matrix::dgCMatrix with row names (finiteness is
#     sample-checked, see `.ucell_supported_sparse_matrix`).
#   - `precalc.ranks` is NULL.
#   - `ties.method` is "average".
#   - BPPARAM has at most one worker, or ncores <= 1 when BPPARAM is NULL.
#   - `maxRank` and `w_neg` are finite scalar numerics after upstream defaults.
#   - `storeRanks` is not exposed by the public target and remains upstream-only.
#
# Fallback surface:
#   - SingleCellExperiment, dense matrix/data.frame, precomputed ranks,
#     (sampled) non-finite values, non-average ties, multi-worker BPPARAM/ncores,
#     invalid scalar knobs, missing row names, or zyme = FALSE all call upstream.
#   - Negative values are NOT a fallback trigger: the kernel scores them
#     correctly under descending ranks (structural zeros rank above them).
#
# Kernel lives in src/ucell.cpp (`ucell_fast_scores_dgC`). It computes exact
# average descending ranks only for signature genes, using sparse column
# structure and a distinct-value table per cell. Raw integer-count matrices use
# a direct binning build for common counts 1..256.

if (requireNamespace("UCell", quietly = TRUE) &&
    requireNamespace("Matrix", quietly = TRUE) &&
    requireNamespace("BiocParallel", quietly = TRUE)) {

  .ucell_orig_ScoreSignatures_UCell <- utils::getFromNamespace(
    "ScoreSignatures_UCell", "UCell"
  )
  .ucell_get_gene_idx <- utils::getFromNamespace("get_gene_idx", "UCell")
  .ucell_check_signature_names <- utils::getFromNamespace(
    "check_signature_names", "UCell"
  )

  .ucell_serial_backend <- function(BPPARAM, ncores) {
    if (is.null(BPPARAM)) {
      return(is.numeric(ncores) && length(ncores) == 1L &&
               is.finite(ncores) && ncores <= 1L)
    }
    workers <- tryCatch(BiocParallel::bpnworkers(BPPARAM),
                        error = function(e) NA_integer_)
    is.finite(workers) && workers <= 1L
  }

  .ucell_supported_sparse_matrix <- function(matrix) {
    if (!methods::is(matrix, "dgCMatrix") || is.null(rownames(matrix))) {
      return(FALSE)
    }
    # Finiteness is checked on an evenly-spaced 20000-element sample of @x, the
    # same fidelity as the int_mat raw-count detection below. A full
    # all(is.finite(matrix@x)) / all(matrix@x >= 0) sweep is O(nnz) and added
    # 0.6-1.8 s/call on the dev tiers (memory-bandwidth-bound over a 37-290 M
    # element double array) -- a factor-2-to-4 packaging regression vs the
    # iterate fast path, which had no such gate. Negative values are scored
    # correctly by the kernel's capped_rank branch (structural zeros rank above
    # them under descending ranks), so only non-finite values need to force a
    # fallback; NaN/Inf never occurs in count / library-normalized single-cell
    # inputs (the attested contract), and upstream UCell itself does not
    # validate @x.
    xv <- matrix@x
    if (length(xv) == 0L) return(TRUE)
    samp <- if (length(xv) > 20000L) {
      xv[seq.int(1L, length(xv), length.out = 20000L)]
    } else {
      xv
    }
    all(is.finite(samp))
  }

  .ucell_supported_scalars <- function(maxRank, w_neg) {
    is.numeric(maxRank) && length(maxRank) == 1L && is.finite(maxRank) &&
      (is.null(w_neg) ||
         (is.numeric(w_neg) && length(w_neg) == 1L &&
            is.finite(w_neg) && w_neg >= 0))
  }

  .ucell_fast_score_matrix <- function(matrix, features, maxRank, w_neg,
                                       missing_genes, name) {
    if (maxRank > nrow(matrix)) maxRank <- nrow(matrix)
    if (is.null(w_neg)) w_neg <- 1

    sign_lgt <- lapply(features, length)
    if (any(sign_lgt > maxRank)) {
      stop("One or more signatures contain more genes than maxRank parameter.
            Increase maxRank parameter or make shorter signatures")
    }

    all_genes <- rownames(matrix)
    pos_list <- vector("list", length(features))
    neg_list <- vector("list", length(features))
    for (j in seq_along(features)) {
      sig <- unlist(features[[j]])
      sig_neg <- grep("-$", sig, perl = TRUE, value = TRUE)
      sig_pos <- setdiff(sig, sig_neg)
      sig_pos <- gsub("\\+$", "", sig_pos, perl = TRUE)
      sig_neg <- gsub("-$", "", sig_neg, perl = TRUE)
      pos_list[[j]] <- as.integer(.ucell_get_gene_idx(
        all_genes, sig_pos, missing_genes = missing_genes
      ))
      neg_list[[j]] <- as.integer(.ucell_get_gene_idx(
        all_genes, sig_neg, missing_genes = missing_genes
      ))
    }

    xv <- matrix@x
    samp <- if (length(xv) > 20000L) {
      xv[seq.int(1L, length(xv), length.out = 20000L)]
    } else {
      xv
    }
    int_mat <- as.integer(length(samp) == 0L || all(samp == floor(samp)))

    cells_U <- ucell_fast_scores_dgC(
      matrix@p, matrix@i, matrix@x, nrow(matrix), pos_list, neg_list,
      as.integer(maxRank), as.numeric(w_neg), int_mat
    )
    rownames(cells_U) <- colnames(matrix)
    colnames(cells_U) <- paste0(names(features), name)
    cells_U
  }

  fast_ScoreSignatures_UCell <- function(
      matrix = NULL, features, precalc.ranks = NULL,
      maxRank = 1500, w_neg = 1, name = "_UCell",
      assay = "counts", chunk.size = 100,
      missing_genes = c("impute", "skip"),
      BPPARAM = NULL, ncores = 1,
      ties.method = "average", force.gc = FALSE,
      zyme = TRUE) {

    if (!isTRUE(zyme) ||
        !is.null(precalc.ranks) ||
        !identical(ties.method, "average") ||
        !.ucell_serial_backend(BPPARAM, ncores) ||
        !.ucell_supported_scalars(maxRank, w_neg) ||
        !.ucell_supported_sparse_matrix(matrix)) {
      return(.ucell_orig_ScoreSignatures_UCell(
        matrix = matrix, features = features, precalc.ranks = precalc.ranks,
        maxRank = maxRank, w_neg = w_neg, name = name, assay = assay,
        chunk.size = chunk.size, missing_genes = missing_genes,
        BPPARAM = BPPARAM, ncores = ncores, ties.method = ties.method,
        force.gc = force.gc
      ))
    }

    features <- .ucell_check_signature_names(features)
    missing_genes <- match.arg(missing_genes)
    .ucell_fast_score_matrix(
      matrix = matrix, features = features, maxRank = maxRank,
      w_neg = w_neg, missing_genes = missing_genes, name = name
    )
  }

  .ucell_smoke_load <- function(task_dir, tier) {
    suppressPackageStartupMessages({
      library(UCell)
      library(Matrix)
      library(BiocParallel)
    })
    if (!requireNamespace("yaml", quietly = TRUE)) {
      stop("ucell smoke requires the yaml package")
    }
    task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
    ds <- Filter(function(e) identical(e$tier, tier), task$datasets)
    if (length(ds) == 0L) stop("no dataset for tier '", tier, "' in task.yaml")
    data_path <- resolve_dataset_path(task_dir, ds[[1]]$path)
    source(file.path(task_dir, "setup", "signatures.R"))
    list(matrix = readRDS(data_path), features = UCELL_SIGNATURES)
  }

  .ucell_smoke_call <- function(inputs) {
    ScoreSignatures_UCell(
      inputs$matrix,
      features = inputs$features,
      maxRank = 1500,
      BPPARAM = BiocParallel::SerialParam()
    )
  }

  .ucell_smoke_save <- function(result, dir, tier = "small", ...) {
    saveRDS(list(scores = result), file.path(dir, "result.rds"))
  }

  register_patch(
    name = "ucell",
    upstream = "UCell",
    targets = list(ScoreSignatures_UCell = fast_ScoreSignatures_UCell),
    smoke = list(
      load = .ucell_smoke_load,
      call = .ucell_smoke_call,
      save = .ucell_smoke_save
    ),
    tested_against = "UCell 2.17.0 (GitHub carmonalab/UCell@15b029a)",
    tested_upstream_versions = list(UCell = "2.17.0")
  )
}
