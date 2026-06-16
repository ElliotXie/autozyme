# Patch for spacexr::run.RCTD (doublet mode).
#
# Lifted from autozyme task `test_RCTD`. Nine co-registered fast functions
# all sit on spacexr's namespace and form a single coherent optimization
# of the doublet-mode pixel fitting loop:
#
#   - calc_log_l_vec       -> C++ spline kernel
#   - get_der_fast         -> C++ vectorized grad/hess (non-bulk only)
#   - solveWLS             -> C++ p=1 / p=2 closed-form QP step
#   - solveIRWLS.weights   -> C++ active-set IRWLS for p>2 non-bulk
#   - psd                  -> closed-form 1x1/2x2 PSD shortcut
#   - process_bead_doublet -> C++-fused candidate scoring + pair refit
#   - decompose_sparse     -> C++ IRWLS for p<=2 sparse pairs
#   - gather_results       -> tight Rcpp-friendly column packing
#   - process_beads_batch  -> mclapply over pixels w/ precomputed row_idx mat
#
# Two extra helpers (fast_calc_Q_all, fast_get_d1_d2) are kept at file scope
# for downstream callers but NOT registered — upstream's calc_Q_all has no
# safe binding point and the helpers are only invoked via the registered
# fast_* paths.
#
# Shared globals (Q_mat, SQ_mat, X_vals, K_val): spacexr::set_likelihood_vars
# writes these into .GlobalEnv via <<- from inside spacexr's namespace
# (writes can't reach the locked package ns and fall through to globalenv).
# Our fast_* functions resolve them by bare name; lookup walks
# patch_env -> autozyme ns -> imports -> base -> R_GlobalEnv and finds them.
# Forked mclapply workers inherit the globals via copy-on-write.

if (requireNamespace("spacexr",   quietly = TRUE) &&
    requireNamespace("Matrix",    quietly = TRUE) &&
    requireNamespace("quadprog",  quietly = TRUE) &&
    requireNamespace("parallel",  quietly = TRUE)) {

  # Snap spacexr's namespace once. Fast functions reference patched siblings
  # (process_bead_doublet, decompose_sparse, etc.) and unpatched internals
  # (decompose_full, solveOLS, ...) via $-lookup at call time — so when
  # restore() flips upstream back, the patched fns are no longer reachable
  # via that path either (the sibling lookups return upstream's originals,
  # not our captured fast_ versions). This is what we want for verify_patch
  # which toggles restore <-> activate within one process.
  .spacexr_ns <- asNamespace("spacexr")

  # Originals captured at file scope — only used as fallback in the rare
  # bulk_mode branches that the patch chose not to accelerate. The autozyme
  # dispatcher already routes is_disabled() to upstream for the namespace-fn
  # patches, so we don't need is_disabled() checks here.
  .orig_get_der_fast       <- utils::getFromNamespace("get_der_fast",       "spacexr")
  .orig_solveWLS           <- utils::getFromNamespace("solveWLS",           "spacexr")
  .orig_solveIRWLS.weights <- utils::getFromNamespace("solveIRWLS.weights", "spacexr")
  .orig_psd                <- utils::getFromNamespace("psd",                "spacexr")

  # ---- Helpers (file scope, not registered) ----------------------------

  fast_calc_Q_all <- function(Y, lambda) {
    row_idx <- attr(Y, "zyme_row_idx", exact = TRUE)
    if (is.null(row_idx)) {
      row_idx <- pmin.int(Y, K_val) + 1L
    }
    epsilon <- 1e-4
    X_max <- max(X_vals)
    delta <- 1e-6
    lambda <- pmin.int(pmax.int(lambda, epsilon), X_max - epsilon)

    l <- floor((lambda / delta)^(1 / 2))
    m <- pmin.int(l - 9, 40) + pmax.int(ceiling(sqrt(pmax.int(l - 48.7499, 0) * 4)) - 2, 0)
    ti1 <- X_vals[m]
    ti  <- X_vals[m + 1L]
    hi  <- ti - ti1

    q_idx <- row_idx + (m - 1L) * nrow(Q_mat)
    fti1 <- Q_mat[q_idx]
    fti  <- Q_mat[q_idx + nrow(Q_mat)]
    zi1  <- SQ_mat[q_idx]
    zi   <- SQ_mat[q_idx + nrow(SQ_mat)]

    diff1 <- lambda - ti1
    diff2 <- ti - lambda
    diff3 <- fti / hi - zi * hi / 6
    diff4 <- fti1 / hi - zi1 * hi / 6
    zdi  <- zi / hi
    zdi1 <- zi1 / hi

    d0_vec <- zdi * (diff1)^3 / 6 + zdi1 * (diff2)^3 / 6 + diff3 * diff1 + diff4 * diff2
    d1_vec <- zdi * (diff1)^2 / 2 - zdi1 * (diff2)^2 / 2 + diff3 - diff4
    d2_vec <- zdi * (diff1) + zdi1 * (diff2)
    list(d0_vec = d0_vec, d1_vec = d1_vec, d2_vec = d2_vec)
  }

  fast_get_d1_d2 <- function(Y, lambda) {
    row_idx <- attr(Y, "zyme_row_idx", exact = TRUE)
    rctd_cpp_get_d1_d2(Y, lambda, row_idx, Q_mat, SQ_mat, X_vals, K_val)
  }

  # ---- Registered fast functions ---------------------------------------

  fast_calc_log_l_vec <- function(lambda, Y, return_vec = FALSE) {
    row_idx <- attr(Y, "zyme_row_idx", exact = TRUE)
    if (return_vec) {
      return(rctd_cpp_calc_log_l_vec(lambda, Y, row_idx, Q_mat, SQ_mat, X_vals, K_val))
    }
    rctd_cpp_calc_log_l_sum(lambda, Y, row_idx, Q_mat, SQ_mat, X_vals, K_val)
  }

  fast_get_der_fast <- function(S, B, gene_list, prediction, bulk_mode = FALSE) {
    if (bulk_mode) {
      return(.orig_get_der_fast(S, B, gene_list, prediction, bulk_mode = bulk_mode))
    }
    row_idx <- attr(B, "zyme_row_idx", exact = TRUE)
    rctd_cpp_get_der_fast_nonbulk(S, B, prediction, row_idx, Q_mat, SQ_mat, X_vals, K_val)
  }

  fast_solveWLS <- function(S, B, initialSol, nUMI, bulk_mode = FALSE, constrain = FALSE) {
    if (!bulk_mode && !constrain && dim(S)[2L] == 1L) {
      row_idx <- attr(B, "zyme_row_idx", exact = TRUE)
      solution <- rctd_cpp_solve_wls_p1(S, B, row_idx, initialSol[1L], nUMI, Q_mat, SQ_mat, X_vals, K_val)
      names(solution) <- colnames(S)
      return(solution)
    }
    if (!bulk_mode && !constrain && dim(S)[2L] == 2L) {
      row_idx <- attr(B, "zyme_row_idx", exact = TRUE)
      solution <- rctd_cpp_solve_wls_p2(S, B, row_idx, initialSol, nUMI, Q_mat, SQ_mat, X_vals, K_val)
      names(solution) <- colnames(S)
      return(solution)
    }
    if (!bulk_mode && !constrain && dim(S)[2L] > 2L) {
      solution <- pmax(initialSol, 0)
      prediction <- abs(S %*% solution)
      threshold <- max(1e-4, nUMI * 1e-7)
      prediction[prediction < threshold] <- threshold
      # get_der_fast in spacexr ns is our patched version when activated
      # (and the original when restored). Use $-lookup so dispatch is live.
      derivatives <- .spacexr_ns$get_der_fast(S, B, rownames(S), prediction, bulk_mode = FALSE)
      eig <- eigen(derivatives$hess, symmetric = TRUE)
      eig_values <- pmax(eig$values, 1e-3)
      D_mat <- eig$vectors %*% (eig_values * t(eig$vectors))
      norm_factor <- max(eig_values)
      D_mat <- D_mat / norm_factor + 1e-7 * diag(length(eig_values))
      d_vec <- as.numeric(-derivatives$grad) / norm_factor
      step <- quadprog::solve.QP(D_mat, d_vec, diag(length(d_vec)), -solution, meq = 0)$solution
      solution <- solution + 0.3 * step
      names(solution) <- colnames(S)
      return(solution)
    }
    .orig_solveWLS(S, B, initialSol, nUMI, bulk_mode = bulk_mode, constrain = constrain)
  }

  fast_solveIRWLS.weights <- function(S, B, nUMI, OLS = FALSE, constrain = TRUE, verbose = FALSE,
                                      n.iter = 50, MIN_CHANGE = 0.001, bulk_mode = FALSE,
                                      solution = NULL) {
    if (bulk_mode) {
      return(.orig_solveIRWLS.weights(
        S, B, nUMI, OLS = OLS, constrain = constrain, verbose = verbose,
        n.iter = n.iter, MIN_CHANGE = MIN_CHANGE, bulk_mode = bulk_mode,
        solution = solution
      ))
    }
    B[B > K_val] <- K_val
    solution <- numeric(dim(S)[2L])
    solution[] <- 1 / length(solution)
    if (OLS) {
      solution <- .spacexr_ns$solveOLS(S, B, solution, constrain = constrain)
      return(list(weights = solution, converged = TRUE))
    }
    if (!bulk_mode && !constrain && ncol(S) > 2L) {
      row_idx <- attr(B, "zyme_row_idx", exact = TRUE)
      fit <- rctd_cpp_irwls_full_nonbulk(
        S, B, row_idx, nUMI, n.iter, MIN_CHANGE,
        Q_mat, SQ_mat, X_vals, K_val
      )
      weights <- fit$weights
      names(weights) <- colnames(S)
      return(list(weights = weights, converged = fit$converged))
    }
    names(solution) <- colnames(S)

    iterations <- 0L
    change <- 1
    while (change > MIN_CHANGE && iterations < n.iter) {
      new_solution <- .spacexr_ns$solveWLS(S, B, solution, nUMI,
                                           constrain = constrain, bulk_mode = bulk_mode)
      change <- sum(abs(new_solution - solution))
      if (verbose) {
        print(paste("Change:", change))
        print(solution)
      }
      solution <- new_solution
      iterations <- iterations + 1L
    }
    list(weights = solution, converged = (change <= MIN_CHANGE))
  }

  fast_psd <- function(H, epsilon = 1e-3) {
    if (length(H) == 1L) {
      return(matrix(max(as.numeric(H), epsilon), nrow = 1L, ncol = 1L))
    }
    if (is.matrix(H) && nrow(H) == 2L && ncol(H) == 2L) {
      a <- H[1L, 1L]
      b <- H[1L, 2L]
      c <- H[2L, 2L]
      disc <- sqrt((a - c) * (a - c) + 4 * b * b)
      lambda1 <- (a + c + disc) / 2
      lambda2 <- (a + c - disc) / 2
      f1 <- max(lambda1, epsilon)
      f2 <- max(lambda2, epsilon)
      if (disc < 1e-12) {
        return(matrix(c(f1, 0, 0, f1), nrow = 2L, ncol = 2L))
      }
      scale <- (f1 - f2) / (lambda1 - lambda2)
      return(f2 * diag(2L) + scale * (H - lambda2 * diag(2L)))
    }
    .orig_psd(H, epsilon = epsilon)
  }

  fast_process_bead_doublet <- function(cell_type_info, gene_list, UMI_tot, bead, class_df = NULL,
                                        constrain = TRUE, verbose = FALSE, MIN.CHANGE = 0.001,
                                        CONFIDENCE_THRESHOLD = 10, DOUBLET_THRESHOLD = 25) {
    attr(bead, "zyme_row_idx") <- rctd_cpp_row_idx1(bead, K_val)
    cell_type_profiles <- cell_type_info[[1]][gene_list, ]
    cell_type_profiles <- cell_type_profiles * UMI_tot
    cell_type_profiles <- data.matrix(cell_type_profiles)
    QL_score_cutoff <- CONFIDENCE_THRESHOLD
    doublet_like_cutoff <- DOUBLET_THRESHOLD
    results_all <- .spacexr_ns$decompose_full(cell_type_profiles, UMI_tot, bead,
                                              constrain = constrain, verbose = verbose,
                                              MIN_CHANGE = MIN.CHANGE)
    all_weights <- results_all$weights
    conv_all <- results_all$converged
    initial_weight_thresh <- 0.01
    candidates <- names(which(all_weights > initial_weight_thresh))
    if (length(candidates) == 0) {
      candidates <- cell_type_info[[2]][1:min(3, cell_type_info[[3]])]
    }
    if (length(candidates) == 1) {
      if (candidates[1] == cell_type_info[[2]][1]) {
        candidates <- c(candidates, cell_type_info[[2]][2])
      } else {
        candidates <- c(candidates, cell_type_info[[2]][1])
      }
    }
    if (length(candidates) > 7L) {
      sorted_candidates <- names(sort(all_weights[candidates], decreasing = TRUE))
      keep_n <- if (all_weights[sorted_candidates[7L]] < 0.025) 6L else 7L
      keep_candidates <- sorted_candidates[seq_len(keep_n)]
      candidates <- candidates[candidates %in% keep_candidates]
    }
    candidate_cols <- match(candidates, colnames(cell_type_profiles))
    if (!constrain) {
      candidate_scores <- rctd_cpp_score_sparse_candidates(
        cell_type_profiles, bead, candidate_cols, attr(bead, "zyme_row_idx", exact = TRUE),
        UMI_tot, MIN.CHANGE, Q_mat, SQ_mat, X_vals, K_val
      )
      score_mat <- candidate_scores$score_mat
      rownames(score_mat) <- candidates
      colnames(score_mat) <- candidates
      singlet_scores <- candidate_scores$singlet_scores
      names(singlet_scores) <- candidates
      min_score <- candidate_scores$min_score
      first_type <- candidates[candidate_scores$min_i]
      second_type <- candidates[candidate_scores$min_j]
    } else {
      score_mat <- matrix(0, nrow = length(candidates), ncol = length(candidates))
      rownames(score_mat) <- candidates
      colnames(score_mat) <- candidates
      singlet_scores <- numeric(length(candidates))
      names(singlet_scores) <- candidates
      for (type in candidates) {
        singlet_scores[type] <- .spacexr_ns$get_singlet_score(
          cell_type_profiles, bead, UMI_tot, type, constrain,
          MIN.CHANGE = MIN.CHANGE
        )
      }
      min_score <- 0
      first_type <- NULL
      second_type <- NULL
      for (i in 1:(length(candidates) - 1)) {
        type1 <- candidates[i]
        for (j in (i + 1):length(candidates)) {
          type2 <- candidates[j]
          score <- .spacexr_ns$decompose_sparse(
            cell_type_profiles, UMI_tot, bead, type1, type2,
            score_mode = TRUE, constrain = constrain, verbose = verbose,
            MIN.CHANGE = MIN.CHANGE
          )
          score_mat[i, j] <- score
          score_mat[j, i] <- score
          if (is.null(second_type) || score < min_score) {
            first_type <- type1
            second_type <- type2
            min_score <- score
          }
        }
      }
    }
    first_class <- FALSE
    second_class <- FALSE
    type1_pres <- .spacexr_ns$check_pairs_type(
      cell_type_profiles, bead, UMI_tot, score_mat, min_score, first_type,
      class_df, QL_score_cutoff, constrain, singlet_scores,
      MIN.CHANGE = MIN.CHANGE
    )
    type2_pres <- .spacexr_ns$check_pairs_type(
      cell_type_profiles, bead, UMI_tot, score_mat, min_score, second_type,
      class_df, QL_score_cutoff, constrain, singlet_scores,
      MIN.CHANGE = MIN.CHANGE
    )
    if (!type1_pres$all_pairs_class && !type2_pres$all_pairs_class) {
      spot_class <- "reject"
      singlet_score <- min_score + 2 * doublet_like_cutoff
    } else if (type1_pres$all_pairs_class && !type2_pres$all_pairs_class) {
      first_class <- !type1_pres$all_pairs
      singlet_score <- type1_pres$singlet_score
      spot_class <- "doublet_uncertain"
    } else if (!type1_pres$all_pairs_class && type2_pres$all_pairs_class) {
      first_class <- !type2_pres$all_pairs
      singlet_score <- type2_pres$singlet_score
      temp <- first_type
      first_type <- second_type
      second_type <- temp
      spot_class <- "doublet_uncertain"
    } else {
      spot_class <- "doublet_certain"
      singlet_score <- min(type1_pres$singlet_score, type2_pres$singlet_score)
      first_class <- !type1_pres$all_pairs
      second_class <- !type2_pres$all_pairs
      if (type2_pres$singlet_score < type1_pres$singlet_score) {
        temp <- first_type
        first_type <- second_type
        second_type <- temp
        first_class <- !type2_pres$all_pairs
        second_class <- !type1_pres$all_pairs
      }
    }
    if (singlet_score - min_score < doublet_like_cutoff) {
      spot_class <- "singlet"
    }
    if (!constrain) {
      doublet_results <- rctd_cpp_fit_sparse_pair(
        cell_type_profiles, bead, match(c(first_type, second_type), colnames(cell_type_profiles)),
        attr(bead, "zyme_row_idx", exact = TRUE), UMI_tot, 50L, MIN.CHANGE,
        Q_mat, SQ_mat, X_vals, K_val
      )
      doublet_weights <- doublet_results$weights
      names(doublet_weights) <- c(first_type, second_type)
      doublet_weights <- doublet_weights / sum(doublet_weights)
      conv_doublet <- doublet_results$converged
    } else {
      doublet_results <- .spacexr_ns$decompose_sparse(
        cell_type_profiles, UMI_tot, bead, first_type, second_type,
        constrain = constrain, MIN.CHANGE = MIN.CHANGE
      )
      doublet_weights <- doublet_results$weights
      conv_doublet <- doublet_results$converged
    }
    spot_class <- factor(spot_class, c("reject", "singlet", "doublet_certain", "doublet_uncertain"))
    list(
      all_weights = all_weights, spot_class = spot_class,
      first_type = first_type, second_type = second_type,
      doublet_weights = doublet_weights, min_score = min_score,
      singlet_score = singlet_score, conv_all = conv_all,
      conv_doublet = conv_doublet, score_mat = score_mat,
      singlet_scores = singlet_scores, first_class = first_class,
      second_class = second_class
    )
  }

  fast_decompose_sparse <- function(cell_type_profiles, nUMI, bead, type1 = NULL, type2 = NULL,
                                    score_mode = FALSE, plot = FALSE, custom_list = NULL,
                                    verbose = FALSE, constrain = TRUE, MIN.CHANGE = 0.001) {
    if (is.null(custom_list)) {
      cell_types <- c(type1, type2)
    } else {
      cell_types <- custom_list
    }
    if (is.matrix(cell_type_profiles)) {
      reg_data <- cell_type_profiles[, cell_types, drop = FALSE]
    } else {
      reg_data <- data.matrix(cell_type_profiles[, cell_types])
    }
    if (score_mode) {
      n.iter <- 25
    } else {
      n.iter <- 50
    }
    if (!constrain && ncol(reg_data) <= 2L) {
      row_idx <- attr(bead, "zyme_row_idx", exact = TRUE)
      fit <- rctd_cpp_irwls_sparse_p12(
        reg_data, bead, row_idx, nUMI, n.iter, MIN.CHANGE,
        Q_mat, SQ_mat, X_vals, K_val
      )
      weights <- fit$weights
      names(weights) <- colnames(reg_data)
      if (score_mode) {
        return(fit$score)
      }
      weights <- weights / sum(weights)
      return(list(weights = weights, converged = fit$converged))
    }
    results <- .spacexr_ns$solveIRWLS.weights(
      reg_data, bead, nUMI, OLS = FALSE, constrain = constrain,
      verbose = verbose, n.iter = n.iter, MIN_CHANGE = MIN.CHANGE
    )
    if (!score_mode) {
      results$weights <- results$weights / sum(results$weights)
      return(results)
    }
    prediction <- reg_data %*% results$weights
    .spacexr_ns$calc_log_l_vec(prediction, bead)
  }

  fast_gather_results <- function(RCTD, results) {
    cell_type_names <- RCTD@cell_type_info$renorm[[2]]
    barcodes <- colnames(RCTD@spatialRNA@counts)
    N <- length(results)
    weights <- matrix(0, nrow = N, ncol = length(cell_type_names))
    weights_doublet <- matrix(0, nrow = N, ncol = 2L)
    rownames(weights) <- barcodes
    rownames(weights_doublet) <- barcodes
    colnames(weights) <- cell_type_names
    colnames(weights_doublet) <- c("first_type", "second_type")
    empty_cell_types <- factor(character(N), levels = cell_type_names)
    spot_levels <- c("reject", "singlet", "doublet_certain", "doublet_uncertain")
    results_df <- data.frame(
      spot_class = factor(character(N), levels = spot_levels),
      first_type = empty_cell_types, second_type = empty_cell_types,
      first_class = logical(N), second_class = logical(N),
      min_score = numeric(N), singlet_score = numeric(N),
      conv_all = logical(N), conv_doublet = logical(N)
    )
    score_mat <- vector("list", N)
    singlet_scores <- vector("list", N)
    for (i in seq_len(N)) {
      if (i %% 1000 == 0) {
        print(paste("gather_results: finished", i))
      }
      weights_doublet[i, ] <- results[[i]]$doublet_weights
      weights[i, ] <- results[[i]]$all_weights
      results_df[i, "spot_class"] <- results[[i]]$spot_class
      results_df[i, "first_type"] <- results[[i]]$first_type
      results_df[i, "second_type"] <- results[[i]]$second_type
      results_df[i, "first_class"] <- results[[i]]$first_class
      results_df[i, "second_class"] <- results[[i]]$second_class
      results_df[i, "min_score"] <- results[[i]]$min_score
      results_df[i, "singlet_score"] <- results[[i]]$singlet_score
      results_df[i, "conv_all"] <- results[[i]]$conv_all
      results_df[i, "conv_doublet"] <- results[[i]]$conv_doublet
      score_mat[[i]] <- results[[i]]$score_mat
      singlet_scores[[i]] <- results[[i]]$singlet_scores
    }
    rownames(results_df) <- barcodes
    RCTD@results <- list(
      results_df = results_df, weights = weights,
      weights_doublet = weights_doublet, score_mat = score_mat,
      singlet_scores = singlet_scores
    )
    RCTD
  }

  # fast_decompose_batch: bypass upstream's PSOCK + foreach orchestrator inside
  # choose_sigma_c. Upstream uses `if (max_cores > 1) { foreach %dopar% }` here,
  # but the actual work is tiny (N_fit <= ~1000 spots, each ~5ms of IRWLS via
  # the patched solveIRWLS.weights). PSOCK startup on Windows is ~10-15s for
  # cluster spinup + per-worker library load + data export — net LOSS vs serial.
  # On Mac fork is essentially free, so we keep mclapply there.
  # This removes the upstream "PSOCK tax" charged to both baseline and patched
  # whenever a user runs RCTD with max_cores > 1 (the default).
  fast_decompose_batch <- function(nUMI, cell_type_means, beads, gene_list,
                                   constrain = TRUE, OLS = FALSE,
                                   max_cores = 8, MIN.CHANGE = 0.001) {
    N <- dim(beads)[1L]
    cores <- max(1L, as.integer(max_cores))
    df <- .spacexr_ns$decompose_full
    worker <- function(i) {
      df(
        data.matrix(cell_type_means[gene_list, ] * nUMI[i]),
        nUMI[i], beads[i, ], constrain = constrain,
        OLS = OLS, MIN_CHANGE = MIN.CHANGE
      )
    }
    if (cores > 1L && Sys.info()[["sysname"]] != "Windows" && N >= 200L) {
      .zyme_mclapply(seq_len(N), worker, mc.cores = cores, mc.preschedule = TRUE)
    } else {
      lapply(seq_len(N), worker)
    }
  }

  fast_process_beads_batch <- function(cell_type_info, gene_list, puck, class_df = NULL, constrain = TRUE,
                                       MAX_CORES = 8, MIN.CHANGE = 0.001,
                                       CONFIDENCE_THRESHOLD = 10, DOUBLET_THRESHOLD = 25) {
    beads <- t(as.matrix(puck@counts[gene_list, ]))
    bead_row_idx <- rctd_cpp_row_idx_mat(beads, K_val)
    N <- nrow(beads)
    # Capture the (possibly patched) process_bead_doublet at batch entry so
    # forked workers all see the same binding. Reaching into spacexr ns each
    # time would still work but adds an environment lookup per pixel.
    pbd <- .spacexr_ns$process_bead_doublet
    worker <- function(i) {
      bead <- beads[i, ]
      attr(bead, "zyme_row_idx") <- bead_row_idx[i, ]
      pbd(
        cell_type_info, gene_list, puck@nUMI[i], bead,
        class_df = class_df, constrain = constrain, MIN.CHANGE = MIN.CHANGE,
        CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD = DOUBLET_THRESHOLD
      )
    }
    cores <- max(1L, as.integer(MAX_CORES))
    sysname <- Sys.info()[["sysname"]]

    # [port] Windows can't fork, so mclapply silently falls through to serial.
    # Empirically the patched C++ kernels are so fast per-pixel (~8ms) that
    # PSOCK startup (~10-15s for makePSOCKcluster + library load + activate
    # per worker) plus 4-chunk closure serialization (~50 MB of beads /
    # row_idx for ood-scale data) only amortizes for very large datasets.
    # Measured large (N=4033) serial=35s vs PSOCK-4w=50s; ood_xlarge
    # (N=10000) serial=125s vs PSOCK-4w=~70s. Threshold sits between.
    # Aligns with tradeseq's cross-platform PSOCK orchestrator pattern
    # (tradeseq/patch.R:407-507) but with a much higher work threshold.
    PSOCK_MIN_PIXELS <- 6000L

    if (cores > 1L && sysname != "Windows") {
      .zyme_mclapply(seq_len(N), worker, mc.cores = cores, mc.preschedule = TRUE)
    } else if (cores > 1L && sysname == "Windows" && N >= PSOCK_MIN_PIXELS) {
      # Cap workers: at most 4 useful workers — past that, both per-worker
      # data transfer (a chunk of beads + bead_row_idx) and BLAS/OS context
      # switching erode parallel scaling.
      n_workers <- min(cores, 4L, as.integer(ceiling(N / 2000)))
      parent_lib_paths <- .libPaths()

      cl <- parallel::makePSOCKcluster(n_workers)
      on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)

      # Per-worker init: lib path sync, BLAS pin to 1, load packages, activate
      # rctd patch so spacexr's process_bead_doublet routes to fast_*.
      parallel::clusterCall(cl, function(lp) {
        .libPaths(lp)
        Sys.setenv(OMP_NUM_THREADS = "1",
                   OPENBLAS_NUM_THREADS = "1",
                   MKL_NUM_THREADS = "1")
        suppressPackageStartupMessages({
          requireNamespace("spacexr",  quietly = TRUE)
          requireNamespace("autozyme", quietly = TRUE)
        })
        autozyme::activate("rctd")
        invisible(NULL)
      }, parent_lib_paths)

      # Shared likelihood globals: spacexr::set_likelihood_vars wrote these via
      # <<- into the parent's .GlobalEnv; push the same to each worker's
      # .GlobalEnv so fast_* kernels resolve them by bare name.
      parallel::clusterExport(cl,
        varlist = c("Q_mat", "SQ_mat", "X_vals", "K_val"),
        envir = .GlobalEnv)

      # Push per-pixel batch data + scalar params to workers' .GlobalEnv ONCE
      # via clusterExport, then have parLapply transmit only the index chunks.
      # Without this, every parLapply chunk re-serializes the captured beads /
      # bead_row_idx matrices, costing 4 × ~50 MB at ood_xlarge.
      .puck_nUMI <- puck@nUMI
      env_share <- new.env(parent = emptyenv())
      env_share$beads             <- beads
      env_share$bead_row_idx      <- bead_row_idx
      env_share$nUMI              <- .puck_nUMI
      env_share$cell_type_info    <- cell_type_info
      env_share$gene_list         <- gene_list
      env_share$class_df          <- class_df
      env_share$constrain         <- constrain
      env_share$MIN.CHANGE        <- MIN.CHANGE
      env_share$CONFIDENCE_THRESHOLD <- CONFIDENCE_THRESHOLD
      env_share$DOUBLET_THRESHOLD    <- DOUBLET_THRESHOLD
      parallel::clusterExport(cl,
        varlist = c("beads", "bead_row_idx", "nUMI", "cell_type_info",
                    "gene_list", "class_df", "constrain", "MIN.CHANGE",
                    "CONFIDENCE_THRESHOLD", "DOUBLET_THRESHOLD"),
        envir = env_share)

      # Worker closure is tiny: only the index chunk gets re-transmitted.
      chunk_idx <- parallel::splitIndices(N, n_workers)
      chunk_worker <- function(idx_vec) {
        pbd <- asNamespace("spacexr")$process_bead_doublet
        lapply(idx_vec, function(i) {
          bead <- beads[i, ]
          attr(bead, "zyme_row_idx") <- bead_row_idx[i, ]
          pbd(cell_type_info, gene_list, nUMI[i], bead,
              class_df = class_df, constrain = constrain,
              MIN.CHANGE = MIN.CHANGE,
              CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD,
              DOUBLET_THRESHOLD = DOUBLET_THRESHOLD)
        })
      }
      chunk_results <- parallel::parLapply(cl, chunk_idx, chunk_worker)
      unlist(chunk_results, recursive = FALSE, use.names = FALSE)
    } else {
      lapply(seq_len(N), worker)
    }
  }

  # ---- Smoke triplet ---------------------------------------------------
  #
  # Fair-comparison rule: `load` runs once (untimed); `call` runs twice
  # (baseline + patched, timed). The .rds files in data/ already contain a
  # pre-built RCTD object (the task's _prepare_ood_tiers.R built them via
  # create.RCTD()). So `load` just reads the .rds and forwards it; `call`
  # is exactly one `spacexr::run.RCTD(...)`. set.seed(1) is inside `call`
  # to match the task's pipeline; the seed prep itself is fast and matches
  # the run.RCTD path that consumes it.
  #
  # `max_cores` lives on the rctd object's `@config` — both baseline and
  # patched runs use whatever's already there (the task's data files were
  # built with max_cores=1). The patch's fast_process_beads_batch honors
  # that field, so we don't override it here.

  .rctd_extract_output <- function(rctd) {
    results_df <- rctd@results$results_df
    list(
      weights = as.matrix(rctd@results$weights),
      weights_doublet = as.matrix(rctd@results$weights_doublet),
      spot_class = as.character(results_df$spot_class),
      first_type = as.character(results_df$first_type),
      second_type = as.character(results_df$second_type),
      min_score = as.numeric(results_df$min_score),
      singlet_score = as.numeric(results_df$singlet_score),
      rownames = rownames(results_df),
      cell_types = colnames(rctd@results$weights),
      metadata = list(
        spots = nrow(results_df),
        genes_reg = length(rctd@internal_vars$gene_list_reg),
        genes_bulk = length(rctd@internal_vars$gene_list_bulk),
        mode = rctd@config$RCTDmode,
        sigma = rctd@internal_vars$sigma
      )
    )
  }

  register_patch(
    name = "rctd",
    upstream = "spacexr",
    targets = list(
      calc_log_l_vec       = fast_calc_log_l_vec,
      get_der_fast         = fast_get_der_fast,
      solveWLS             = fast_solveWLS,
      solveIRWLS.weights   = fast_solveIRWLS.weights,
      psd                  = fast_psd,
      process_bead_doublet = fast_process_bead_doublet,
      decompose_sparse     = fast_decompose_sparse,
      gather_results       = fast_gather_results,
      process_beads_batch  = fast_process_beads_batch,
      decompose_batch      = fast_decompose_batch
    ),
    smoke = list(
      load = function(task_dir, tier) {
        # Attach spacexr so the RCTD S4 class is fully registered (S4 slot
        # access via @ inside run.RCTD uses topenv lookup that does NOT find
        # the class via plain requireNamespace alone — same shape as the
        # MAST callName() trap in CAVEATS.md). Cheap, runs once untimed.
        suppressPackageStartupMessages(library(spacexr))
        task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
        ds <- Filter(function(d) d$tier == tier, task$datasets)
        if (length(ds) == 0L) stop("no dataset for tier '", tier, "' in task.yaml")
        readRDS(resolve_dataset_path(task_dir, ds[[1]]$path))
      },
      call = function(rctd_obj) {
        set.seed(1)
        spacexr::run.RCTD(rctd_obj, doublet_mode = "doublet")
      },
      save = function(result, dir, tier = "tiny", ...) {
        out <- .rctd_extract_output(result)
        saveRDS(out, file.path(dir, "rctd_output.rds"))
      }
    ),
    tested_against = "spacexr 2.2.1",
    tested_upstream_versions = list(spacexr = "2.2.1")
  )
}
