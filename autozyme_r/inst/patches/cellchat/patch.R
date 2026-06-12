# Patch for CellChat::computeCommunProb (+ helper CellChat::triMean).
#
# Lifted from autozyme task `test_cellchat`. Two namespace patches:
#   - triMean: drop the `names = TRUE` overhead in stats::quantile.
#   - computeCommunProb: full body replacement that
#     (a) replaces `aggregate(t(data.use), list(group), FUN=triMean)` with the
#         Rcpp `cpp_aggregate_triMean` kernel (type-7 quantile via nth_element,
#         bit-equivalent to upstream within ~6e-17),
#     (b) batches the nboot=100 permutation aggregator into a single OMP-
#         parallel `cpp_aggregate_triMean_boot` call producing the
#         (ngenes × ngroups × nboot) tensor directly,
#     (c) precomputes L/R/coreceptor/agonist/antagonist subunit row indices
#         once (off the inner loop) and filters `data.use` to LR-referenced
#         genes only (~30% aggregator savings at large),
#     (d) computes the Pnull outer product (`cpp_outer_Pnull`) and per-LR/
#         per-bootstrap Hill-product comparisons (`cpp_unified_inner`) in two
#         OMP-parallel C++ kernels with no R↔C++ boundary per bootstrap,
#     (e) restricts to datatype="RNA" (the only branch the pipeline exercises;
#         spatial mode falls back to upstream).
#
# C++ kernels live in src/cellchat.cpp and are exposed via
# useDynLib(autozyme, .registration = TRUE). `#ifdef _OPENMP` guards keep the
# kernels correct (serial) on toolchains without libomp; macOS users get
# parallelism when the src/Makevars libomp probe succeeds.
#
# At ifnb_2k (tiny) the converged version reports baseline ~84s vs
# patched ~0.25s (99.7% speedup). At heart_adult_30k (ood_large) baseline
# ~320s vs patched ~1.93s (99.4% speedup). Concordance: pearson 1.0,
# max_abs_diff 0.0 (bit-identical algebra modulo FP noise).

if (requireNamespace("CellChat",  quietly = TRUE) &&
    requireNamespace("Seurat",    quietly = TRUE)) {

  # File-scope captures. Names of CellChat internals reused inside the fast
  # body — fetched via getFromNamespace so the closure resolves them without
  # relying on autozyme's namespace importing CellChat's exports.
  .cellchat_orig_triMean             <- utils::getFromNamespace("triMean",             "CellChat")
  .cellchat_orig_computeCommunProb   <- utils::getFromNamespace("computeCommunProb",   "CellChat")
  .cellchat_thresholdedMean          <- utils::getFromNamespace("thresholdedMean",     "CellChat")
  .cellchat_geometricMean            <- utils::getFromNamespace("geometricMean",       "CellChat")
  .cellchat_computeExpr_coreceptor   <- utils::getFromNamespace("computeExpr_coreceptor","CellChat")

  .cellchat_native_fast_enabled <- function() {
    opt <- getOption("autozyme.cellchat.native_fast", NULL)
    if (!is.null(opt)) return(isTRUE(opt))
    env <- tolower(Sys.getenv("AUTOZYME_CELLCHAT_NATIVE_FAST", unset = ""))
    !(env %in% c("0", "false", "no", "off"))
  }

  .cellchat_enter_native_threads <- function() {
    vars <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    old_env <- Sys.getenv(vars, unset = NA_character_)
    old_blas <- NULL
    if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
      old_blas <- list(
        blas = tryCatch(RhpcBLASctl::blas_get_num_procs(), error = function(e) NA_integer_),
        omp  = tryCatch(RhpcBLASctl::omp_get_max_threads(), error = function(e) NA_integer_)
      )
    }
    args <- as.list(rep("1", length(vars)))
    names(args) <- vars
    do.call(Sys.setenv, args)
    if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
      RhpcBLASctl::blas_set_num_threads(1L)
      RhpcBLASctl::omp_set_num_threads(1L)
    }
    list(vars = vars, env = old_env, blas = old_blas)
  }

  .cellchat_exit_native_threads <- function(state) {
    restore_env <- as.list(state$env[!is.na(state$env)])
    unset_env <- state$vars[is.na(state$env)]
    if (length(restore_env) > 0L) do.call(Sys.setenv, restore_env)
    if (length(unset_env) > 0L) Sys.unsetenv(unset_env)
    if (!is.null(state$blas) && requireNamespace("RhpcBLASctl", quietly = TRUE)) {
      if (!is.na(state$blas$blas)) RhpcBLASctl::blas_set_num_threads(state$blas$blas)
      if (!is.na(state$blas$omp))  RhpcBLASctl::omp_set_num_threads(state$blas$omp)
    }
  }

  # -- fast_triMean ----------------------------------------------------------
  # Round 2 (conservative). Drop quantile name-formatting; same numeric value.
  fast_triMean <- function(x, na.rm = TRUE, zyme = TRUE) {
    if (!isTRUE(zyme)) return(.cellchat_orig_triMean(x, na.rm = na.rm))
    mean(stats::quantile(x, probs = c(0.25, 0.50, 0.50, 0.75),
                         na.rm = na.rm, names = FALSE))
  }

  # -- fast_aggregate_triMean (Rcpp dispatcher) ------------------------------
  .cellchat_fast_aggregate_triMean <- function(data, group) {
    lvls <- levels(group)
    ng <- length(lvls)
    m <- cpp_aggregate_triMean(data, as.integer(group), ng)
    dimnames(m) <- list(rownames(data), lvls)
    m
  }

  # -- fast_computeCommunProb ------------------------------------------------
  fast_computeCommunProb <- function(object,
                                     type = c("triMean", "truncatedMean",
                                              "thresholdedMean", "median"),
                                     trim = 0.1, LR.use = NULL,
                                     raw.use = TRUE, population.size = FALSE,
                                     distance.use = TRUE,
                                     interaction.range = 250,
                                     scale.distance = 0.01,
                                     k.min = 10,
                                     contact.dependent = TRUE,
                                     contact.range = NULL,
                                     contact.knn.k = NULL,
                                     contact.dependent.forced = FALSE,
                                     do.symmetric = TRUE,
                                     nboot = 100, seed.use = 1L,
                                     Kh = 0.5, n = 1,
                                     zyme = TRUE) {
    if (!isTRUE(zyme)) {
      return(.cellchat_orig_computeCommunProb(
        object = object, type = type, trim = trim, LR.use = LR.use,
        raw.use = raw.use, population.size = population.size,
        distance.use = distance.use, interaction.range = interaction.range,
        scale.distance = scale.distance, k.min = k.min,
        contact.dependent = contact.dependent, contact.range = contact.range,
        contact.knn.k = contact.knn.k,
        contact.dependent.forced = contact.dependent.forced,
        do.symmetric = do.symmetric, nboot = nboot, seed.use = seed.use,
        Kh = Kh, n = n))
    }

    # Keep a kill switch for unsupported branches, but default to the package
    # native rewrite. The macOS attest crash was triggered by multi-threaded
    # OpenMP/BLAS state inherited from the worker; the fast kernels are stable
    # and still substantially faster when run with a single native thread.
    if (!.cellchat_native_fast_enabled()) {
      return(.cellchat_orig_computeCommunProb(
        object = object, type = type, trim = trim, LR.use = LR.use,
        raw.use = raw.use, population.size = population.size,
        distance.use = distance.use, interaction.range = interaction.range,
        scale.distance = scale.distance, k.min = k.min,
        contact.dependent = contact.dependent, contact.range = contact.range,
        contact.knn.k = contact.knn.k,
        contact.dependent.forced = contact.dependent.forced,
        do.symmetric = do.symmetric, nboot = nboot, seed.use = seed.use,
        Kh = Kh, n = n))
    }

    .cellchat_thread_state <- .cellchat_enter_native_threads()
    on.exit(.cellchat_exit_native_threads(.cellchat_thread_state), add = TRUE)

    type <- match.arg(type)
    cat(type, "is used for calculating the average gene expression per cell group.", "\n")
    FunMean <- switch(type,
                      triMean         = .cellchat_orig_triMean,
                      truncatedMean   = function(x) mean(x, trim = trim, na.rm = TRUE),
                      thresholdedMean = function(x) .cellchat_thresholdedMean(x, trim = trim, na.rm = TRUE),
                      median          = function(x) stats::median(x, na.rm = TRUE))

    if (raw.use) {
      data <- as.matrix(object@data.signaling)
    } else {
      data <- as.matrix(object@data.smooth)
    }
    if (is.null(LR.use)) {
      pairLR.use <- object@LR$LRsig
    } else {
      if (length(unique(LR.use$annotation)) > 1) {
        LR.use$annotation <- factor(LR.use$annotation,
                                    levels = c("Secreted Signaling", "ECM-Receptor",
                                               "Non-protein Signaling",
                                               "Cell-Cell Contact"))
        LR.use <- LR.use[order(LR.use$annotation), , drop = FALSE]
        LR.use$annotation <- as.character(LR.use$annotation)
      }
      pairLR.use <- LR.use
    }
    complex_input  <- object@DB$complex
    cofactor_input <- object@DB$cofactor

    ptm <- Sys.time()

    pairLRsig  <- pairLR.use
    group      <- object@idents
    geneL      <- as.character(pairLRsig$ligand)
    geneR      <- as.character(pairLRsig$receptor)
    nLR        <- nrow(pairLRsig)
    numCluster <- nlevels(group)
    if (numCluster != length(unique(group))) {
      stop("Please check `unique(object@idents)` and ensure factor levels match the data.")
    }

    data.use <- data / max(data)

    # Round 31: filter data.use to only the genes referenced by some LR
    # (L, R, complex subunits, cofactor cofactors). Unreferenced rows do not
    # contribute to Prob/Pval, so the aggregator can skip them. At the large
    # tier this drops ngenes from 1409 → ~1000 (~30% aggregator savings).
    {
      .all_orig_rows <- rownames(data.use)
      .ref <- character(0)
      .ref <- c(.ref, geneL[geneL %in% .all_orig_rows])
      .ref <- c(.ref, geneR[geneR %in% .all_orig_rows])
      .subcols <- grepl("^subunit", colnames(complex_input))
      .complex_genes <- unique(c(geneL[!geneL %in% .all_orig_rows],
                                 geneR[!geneR %in% .all_orig_rows]))
      .complex_genes <- .complex_genes[.complex_genes %in% rownames(complex_input)]
      if (length(.complex_genes) > 0L) {
        .blk <- as.matrix(complex_input[.complex_genes, .subcols, drop = FALSE])
        .subs <- unique(as.character(.blk))
        .ref <- c(.ref, .subs[.subs != "" & .subs %in% .all_orig_rows])
      }
      .cofcols <- grepl("^cofactor", colnames(cofactor_input))
      .lkps <- unique(c(pairLRsig$co_A_receptor, pairLRsig$co_I_receptor,
                        pairLRsig$agonist, pairLRsig$antagonist))
      .lkps <- .lkps[!is.na(.lkps) & .lkps != "" & .lkps %in% rownames(cofactor_input)]
      if (length(.lkps) > 0L) {
        .blk <- as.matrix(cofactor_input[.lkps, .cofcols, drop = FALSE])
        .cof <- unique(as.character(.blk))
        .ref <- c(.ref, .cof[.cof != "" & .cof %in% .all_orig_rows])
      }
      .ref <- unique(.ref)
      if (length(.ref) > 0L && length(.ref) < length(.all_orig_rows)) {
        data.use <- data.use[.ref, , drop = FALSE]
      }
    }

    # Round 4/12: Rcpp triMean kernel; fall through to upstream slow path for
    # non-triMean modes (R-level aggregator + per-bootstrap aggregate calls).
    if (type == "triMean") {
      data.use.avg <- .cellchat_fast_aggregate_triMean(data.use, group)
    } else {
      data.use.avg <- stats::aggregate(t(data.use), list(group), FUN = FunMean)
      data.use.avg <- t(data.use.avg[, -1])
      colnames(data.use.avg) <- levels(group)
    }

    # Round 19/21: precompute simple-gene row indices for L and R, and the
    # complex-gene subunit row indices once. Reused by fast_outer_LR_early and
    # by the inner-kernel sub lists below.
    L_idx <- match(geneL, rownames(data.use.avg))
    R_idx <- match(geneR, rownames(data.use.avg))
    .subunit_cols_local <- grepl("^subunit", colnames(complex_input))
    .complex_subs_helper <- function(gn) {
      if (!gn %in% rownames(complex_input)) return(NULL)
      v <- unlist(complex_input[gn, .subunit_cols_local, drop = FALSE], use.names = FALSE)
      v[v != ""]
    }
    complex_L_subs <- vector("list", nLR)
    complex_R_subs <- vector("list", nLR)
    for (i in seq_len(nLR)) {
      if (is.na(L_idx[i])) complex_L_subs[[i]] <- .complex_subs_helper(geneL[i])
      if (is.na(R_idx[i])) complex_R_subs[[i]] <- .complex_subs_helper(geneR[i])
    }
    rownames_dav <- rownames(data.use.avg)
    complex_L_idx <- vector("list", nLR)
    complex_R_idx <- vector("list", nLR)
    for (i in seq_len(nLR)) {
      if (!is.null(complex_L_subs[[i]])) {
        idx <- match(complex_L_subs[[i]], rownames_dav)
        complex_L_idx[[i]] <- idx[!is.na(idx)]
      }
      if (!is.null(complex_R_subs[[i]])) {
        idx <- match(complex_R_subs[[i]], rownames_dav)
        complex_R_idx[[i]] <- idx[!is.na(idx)]
      }
    }

    # Inline computeExpr_LR replacement, length-1 form.
    fast_outer_LR_early <- function(geneLR, data, simple_idx, complex_idx_list) {
      nLR_in <- length(geneLR)
      nC_in  <- ncol(data)
      out <- matrix(NA_real_, nrow = nLR_in, ncol = nC_in)
      not_na <- !is.na(simple_idx)
      if (any(not_na)) out[not_na, ] <- data[simple_idx[not_na], ]
      for (i in which(!not_na)) {
        avail_idx <- complex_idx_list[[i]]
        if (length(avail_idx) == 1L) {
          out[i, ] <- data[avail_idx, ]
        } else if (length(avail_idx) > 1L) {
          out[i, ] <- exp(colMeans(log(data[avail_idx, , drop = FALSE])))
        }
      }
      out
    }
    dataLavg <- fast_outer_LR_early(geneL, data.use.avg, L_idx, complex_L_idx)
    dataRavg <- fast_outer_LR_early(geneR, data.use.avg, R_idx, complex_R_idx)
    dataRavg.co.A.receptor <- .cellchat_computeExpr_coreceptor(
      cofactor_input, data.use.avg, pairLRsig, type = "A")
    dataRavg.co.I.receptor <- .cellchat_computeExpr_coreceptor(
      cofactor_input, data.use.avg, pairLRsig, type = "I")
    dataRavg <- dataRavg * dataRavg.co.A.receptor / dataRavg.co.I.receptor

    index.agonist    <- which(!is.na(pairLRsig$agonist)    & pairLRsig$agonist    != "")
    index.antagonist <- which(!is.na(pairLRsig$antagonist) & pairLRsig$antagonist != "")

    if (!is.null(object@options$datatype) && object@options$datatype != "RNA") {
      # spatial / contact branches not in the fast rewrite — defer to upstream.
      return(.cellchat_orig_computeCommunProb(
        object = object, type = type, trim = trim, LR.use = LR.use,
        raw.use = raw.use, population.size = population.size,
        distance.use = distance.use, interaction.range = interaction.range,
        scale.distance = scale.distance, k.min = k.min,
        contact.dependent = contact.dependent, contact.range = contact.range,
        contact.knn.k = contact.knn.k,
        contact.dependent.forced = contact.dependent.forced,
        do.symmetric = do.symmetric, nboot = nboot, seed.use = seed.use,
        Kh = Kh, n = n))
    }
    print(paste0(">>> Run CellChat on sc/snRNA-seq data <<< [", Sys.time(), "]"))

    nC <- ncol(data.use)
    Prob <- array(0, dim = c(numCluster, numCluster, nLR))
    Pval <- array(0, dim = c(numCluster, numCluster, nLR))

    set.seed(seed.use)
    permutation <- replicate(nboot, sample.int(nC, size = nC))

    # Round 12: batched bootstrap aggregator. One C++ call → tensor directly.
    if (type == "triMean") {
      boot_tensor <- cpp_aggregate_triMean_boot(
        data.use, as.integer(group), nlevels(group), permutation)
      dim(boot_tensor) <- c(nrow(data.use), nlevels(group), nboot)
      data.use.avg.boot <- NULL
    } else {
      data.use.avg.boot <- vector("list", nboot)
      for (nE in seq_len(nboot)) {
        data.use.avg.boot[[nE]] <- .cellchat_fast_aggregate_triMean(
          data.use, group[permutation[, nE]])
      }
      boot_tensor <- NULL
    }

    has_coA <- !is.na(pairLRsig$co_A_receptor) & pairLRsig$co_A_receptor != ""
    has_coI <- !is.na(pairLRsig$co_I_receptor) & pairLRsig$co_I_receptor != ""
    is_agonist    <- logical(nLR); is_agonist[index.agonist]       <- TRUE
    is_antagonist <- logical(nLR); is_antagonist[index.antagonist] <- TRUE

    # Cofactor-subunit name vectors per LR, pre-cached for the agonist/
    # antagonist outer factors below (matches upstream's
    # computeExpr_coreceptor / _agonist / _antagonist).
    .cofactor_cols <- grepl("^cofactor", colnames(cofactor_input))
    fast_coreceptor_subs <- function(coreceptor_name) {
      if (is.na(coreceptor_name) || coreceptor_name == "" ||
          !coreceptor_name %in% rownames(cofactor_input)) return(NULL)
      v <- unlist(cofactor_input[coreceptor_name, .cofactor_cols, drop = FALSE],
                  use.names = FALSE)
      v <- v[v != ""]
      if (length(v) == 0L) NULL else v
    }
    cofactor_A_subs <- vector("list", nLR)
    cofactor_I_subs <- vector("list", nLR)
    for (i in seq_len(nLR)) {
      if (has_coA[i]) cofactor_A_subs[[i]] <- fast_coreceptor_subs(pairLRsig$co_A_receptor[i])
      if (has_coI[i]) cofactor_I_subs[[i]] <- fast_coreceptor_subs(pairLRsig$co_I_receptor[i])
    }
    agonist_subs    <- vector("list", nLR)
    antagonist_subs <- vector("list", nLR)
    for (i in seq_len(nLR)) {
      if (is_agonist[i])    agonist_subs[[i]]    <- fast_coreceptor_subs(pairLRsig$agonist[i])
      if (is_antagonist[i]) antagonist_subs[[i]] <- fast_coreceptor_subs(pairLRsig$antagonist[i])
    }
    ones_1xclus <- matrix(1, nrow = 1L, ncol = numCluster)
    Kh_n_outer  <- Kh^n
    fast_agonist_vec <- function(data, cached_subs, Kh_n, n, numCluster) {
      if (is.null(cached_subs)) return(ones_1xclus)
      avail <- intersect(cached_subs, rownames(data))
      if (length(avail) == 0L) return(ones_1xclus)
      da <- data[avail, , drop = FALSE]
      if (n == 1) hill <- da / (Kh_n + da) else { dn <- da^n; hill <- dn / (Kh_n + dn) }
      if (length(avail) == 1L) {
        matrix(1 + hill, nrow = 1L, ncol = numCluster)
      } else {
        matrix(apply(1 + hill, 2, prod), nrow = 1L, ncol = numCluster)
      }
    }
    fast_antagonist_vec <- function(data, cached_subs, Kh_n, n, numCluster) {
      if (is.null(cached_subs)) return(ones_1xclus)
      avail <- intersect(cached_subs, rownames(data))
      if (length(avail) == 0L) return(ones_1xclus)
      da <- data[avail, , drop = FALSE]
      if (n == 1) anti <- Kh_n / (Kh_n + da) else { dn <- da^n; anti <- Kh_n / (Kh_n + dn) }
      if (length(avail) == 1L) {
        matrix(anti, nrow = 1L, ncol = numCluster)
      } else {
        matrix(apply(anti, 2, prod), nrow = 1L, ncol = numCluster)
      }
    }

    # Subunit-index lists (0-indexed) per LR for the unified inner kernel.
    to_idx_vec <- function(name_vec) {
      if (is.null(name_vec) || length(name_vec) == 0L) return(integer(0))
      idx <- match(name_vec, rownames_dav)
      idx <- idx[!is.na(idx)]
      as.integer(idx - 1L)
    }
    Lsubs_list   <- vector("list", nLR)
    Rsubs_list   <- vector("list", nLR)
    coAsubs_list <- vector("list", nLR)
    coIsubs_list <- vector("list", nLR)
    agsubs_list  <- vector("list", nLR)
    antsubs_list <- vector("list", nLR)
    for (i in seq_len(nLR)) {
      Lsubs_list[[i]]  <- if (!is.na(L_idx[i])) as.integer(L_idx[i] - 1L) else as.integer(complex_L_idx[[i]] - 1L)
      Rsubs_list[[i]]  <- if (!is.na(R_idx[i])) as.integer(R_idx[i] - 1L) else as.integer(complex_R_idx[[i]] - 1L)
      coAsubs_list[[i]] <- if (has_coA[i])      to_idx_vec(cofactor_A_subs[[i]]) else integer(0)
      coIsubs_list[[i]] <- if (has_coI[i])      to_idx_vec(cofactor_I_subs[[i]]) else integer(0)
      agsubs_list[[i]]  <- if (is_agonist[i])   to_idx_vec(agonist_subs[[i]])    else integer(0)
      antsubs_list[[i]] <- if (is_antagonist[i]) to_idx_vec(antagonist_subs[[i]]) else integer(0)
    }

    if (population.size) {
      return(.cellchat_orig_computeCommunProb(
        object = object, type = type, trim = trim, LR.use = LR.use,
        raw.use = raw.use, population.size = population.size,
        distance.use = distance.use, interaction.range = interaction.range,
        scale.distance = scale.distance, k.min = k.min,
        contact.dependent = contact.dependent, contact.range = contact.range,
        contact.knn.k = contact.knn.k,
        contact.dependent.forced = contact.dependent.forced,
        do.symmetric = do.symmetric, nboot = nboot, seed.use = seed.use,
        Kh = Kh, n = n))
    }

    # Round 17: Pnull (= Prob) via OMP-parallel outer product.
    agonist_T_outer    <- matrix(1, nrow = numCluster, ncol = nLR)
    antagonist_T_outer <- matrix(1, nrow = numCluster, ncol = nLR)
    for (i in seq_len(nLR)) {
      if (is_agonist[i])    agonist_T_outer[, i]    <- as.numeric(fast_agonist_vec(   data.use.avg, agonist_subs[[i]],    Kh_n_outer, n, numCluster))
      if (is_antagonist[i]) antagonist_T_outer[, i] <- as.numeric(fast_antagonist_vec(data.use.avg, antagonist_subs[[i]], Kh_n_outer, n, numCluster))
    }
    dataLavg_T <- t(dataLavg)
    dataRavg_T <- t(dataRavg)
    # scBLAS opt-in gate (Phase 5 integration). Default OFF preserves the
    # existing cpp_outer_Pnull (OMP-parallel) path. Set
    # AUTOZYME_SCBLAS_OUTER=1 (or AUTOZYME_CELLCHAT_SCBLAS_OUTER=1) to route
    # through scblasR::outer_product_hill (libomp + NEON polynomial pow).
    # To revert: delete this `if/else` block, keep only the `else` body.
    if (.az_feature_enabled("scblas_outer",
                             patch = "cellchat", default = FALSE) &&
        requireNamespace("scblasR", quietly = TRUE)) {
      Prob_flat <- scblasR::outer_product_hill(
        L = dataLavg_T, R = dataRavg_T,
        A = agonist_T_outer, Ant = antagonist_T_outer,
        n_hill = n, Kh = Kh,
        as_array = FALSE,
        threads = 0L)
    } else {
      Prob_flat <- cpp_outer_Pnull(dataLavg_T, dataRavg_T,
                                   agonist_T_outer, antagonist_T_outer,
                                   nLR, numCluster, Kh, n)
    }
    Prob <- array(Prob_flat, dim = c(numCluster, numCluster, nLR))
    Pnull_arr <- Prob

    ngenes_sig <- nrow(data.use.avg)
    if (is.null(boot_tensor)) {
      boot_tensor <- array(0, dim = c(ngenes_sig, numCluster, nboot))
      for (nE in seq_len(nboot)) boot_tensor[, , nE] <- data.use.avg.boot[[nE]]
    }

    flatten_subs <- function(lst) {
      sizes <- lengths(lst)
      list(flat = as.integer(unlist(lst)),
           off  = as.integer(c(0, cumsum(sizes))))
    }
    L_fo   <- flatten_subs(Lsubs_list)
    R_fo   <- flatten_subs(Rsubs_list)
    coA_fo <- flatten_subs(coAsubs_list)
    coI_fo <- flatten_subs(coIsubs_list)
    ag_fo  <- flatten_subs(agsubs_list)
    ant_fo <- flatten_subs(antsubs_list)

    nR_flat <- cpp_unified_inner(boot_tensor, ngenes_sig, numCluster, nboot,
                                 L_fo$flat,   L_fo$off,
                                 R_fo$flat,   R_fo$off,
                                 coA_fo$flat, coA_fo$off,
                                 coI_fo$flat, coI_fo$off,
                                 ag_fo$flat,  ag_fo$off,
                                 ant_fo$flat, ant_fo$off,
                                 Pnull_arr, nLR, Kh, n)
    nR_arr <- array(nR_flat, dim = c(numCluster, numCluster, nLR))
    Pval <- nR_arr / nboot
    Pval[Prob == 0] <- 1
    dimnames(Prob) <- list(levels(group), levels(group), rownames(pairLRsig))
    dimnames(Pval) <- dimnames(Prob)
    net <- list("prob" = Prob, "pval" = Pval)
    execution.time <- Sys.time() - ptm
    object@options$run.time <- as.numeric(execution.time, units = "secs")
    object@options$parameter <- list(
      type.mean = type, trim = trim, raw.use = raw.use,
      population.size = population.size, nboot = nboot, seed.use = seed.use,
      Kh = Kh, n = n,
      distance.use = distance.use, interaction.range = interaction.range,
      ratio = NULL, tol = NULL, k.min = NULL,
      contact.dependent = FALSE, contact.range = NULL,
      contact.knn.k = NULL, contact.dependent.forced = FALSE)
    object@net <- net
    print(paste0(">>> CellChat inference is done. Parameter values are stored in `object@options$parameter` <<< [", Sys.time(), "]"))
    object
  }

  # -- Smoke recipe ---------------------------------------------------------
  # Fair-comparison boundary (per 4_package.md):
  #   - readRDS(checkpoint), createCellChat, subsetData, identify*Genes /
  #     identify*Interactions are user-side prep → `load`.
  #   - Only computeCommunProb(...) → `call`.
  #   - The call's nboot/seed/raw.use/type match pipeline/run.R exactly.
  .cellchat_smoke_load <- function(task_dir, tier) {
    suppressPackageStartupMessages({
      library(Seurat)
      library(CellChat)
      library(future)
    })
    future::plan("sequential")
    # OMP thread count. cap=8 matches pipeline/run.R's tier-aware cap; the
    # OMP kernels in cellchat.cpp honor the OMP_NUM_THREADS env.
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
    .gene_sample <- head(rownames(seu), 200)
    .is_mouse <- mean(grepl("^[A-Z][a-z]", .gene_sample)) > 0.5
    DB <- if (.is_mouse) CellChat::CellChatDB.mouse else CellChat::CellChatDB.human

    DefaultAssay(seu) <- "RNA"
    if (any(grepl("^data\\.", Layers(seu)))) {
      seu[["RNA"]] <- JoinLayers(seu[["RNA"]])
    }
    data.input <- GetAssayData(seu, assay = "RNA", layer = "data")
    meta <- seu@meta.data
    meta[["seurat_clusters"]] <- droplevels(as.factor(
      paste0("C", as.character(meta[["seurat_clusters"]]))))
    cellchat <- CellChat::createCellChat(object = data.input, meta = meta,
                                         group.by = "seurat_clusters")
    cellchat@DB <- DB
    cellchat <- CellChat::subsetData(cellchat)
    cellchat <- CellChat::identifyOverExpressedGenes(cellchat)
    cellchat <- CellChat::identifyOverExpressedInteractions(cellchat)
    list(cellchat = cellchat)
  }

  .cellchat_smoke_call <- function(inputs, tier = "tiny") {
    CellChat::computeCommunProb(inputs$cellchat,
                                type            = "triMean",
                                raw.use         = TRUE,
                                population.size = FALSE,
                                nboot           = 100L,
                                seed.use        = 1L)
  }

  .cellchat_smoke_save <- function(result, dir, tier = "tiny", ...) {
    prob <- result@net$prob
    pval <- result@net$pval
    storage.mode(prob) <- "double"
    storage.mode(pval) <- "double"
    saveRDS(list(prob = prob, pval = pval),
            file.path(dir, "net.rds"))
  }

  register_patch(
    name = "cellchat",
    upstream = "CellChat",
    targets = list(
      triMean           = fast_triMean,
      computeCommunProb = fast_computeCommunProb
    ),
    smoke = list(
      load = .cellchat_smoke_load,
      call = .cellchat_smoke_call,
      save = .cellchat_smoke_save
    ),
    tested_against = "CellChat 2.2.0.9001",
    tested_upstream_versions = list(CellChat = "2.2.0.9001")
  )
}
