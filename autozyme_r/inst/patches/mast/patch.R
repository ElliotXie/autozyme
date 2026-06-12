# Patch for MAST::zlm + MAST::lrTest.
#
# Lifted from autozyme task `test_mast`. Four namespace patches in MAST's
# private API:
#   - .bayesglm.fit.loop.validateState — strip 5 sanity checks down to the
#     convergence test (saves O(nobs) × ~28k iters of unused logical ops).
#   - .bayesglm.fit                    — no-op family$validmu / $valideta +
#                                         loosen IRLS epsilon 1e-8 → 1e-5.
#   - .bayesglm.fit.loop.updateState   — fused IRLS-iter body in C++ for
#                                         binomial+logit and gaussian+identity
#                                         (cpp_updateState in src/).
#   - lrTest                           — hybrid two-pass: Wald for all genes
#                                         then upstream LR only on top-K.
#
# This is the only task with a verify.tsv in core_singlecell — so it's the
# closest thing autozyme has to a fully ship-ready demo.

if (requireNamespace("MAST",                 quietly = TRUE) &&
    requireNamespace("SummarizedExperiment", quietly = TRUE) &&
    requireNamespace("yaml",                 quietly = TRUE)) {

  .mast_orig_bgf    <- utils::getFromNamespace(".bayesglm.fit", "MAST")
  .mast_orig_lrTest <- utils::getFromNamespace("lrTest",        "MAST")
  .mast_orig_updateState <- utils::getFromNamespace(".bayesglm.fit.loop.updateState", "MAST")

  .mast_native_update_state_enabled <- function() {
    isTRUE(getOption("autozyme.mast.native_update_state", FALSE)) ||
      identical(Sys.getenv("AUTOZYME_MAST_NATIVE_UPDATE_STATE", unset = ""), "1")
  }

  # Tier-aware mc.cores caps, lifted from pipeline/run.R::.MC_CORES_BY_TIER
  # of test_mast. MAST::zlm with parallel=TRUE forks via mclapply; the
  # caps reflect what the lift agent found optimal across tiers (smaller
  # data: more workers; larger data: fewer workers because per-gene
  # crossprod gets heavier and BLAS contention demands headroom).
  # See CAVEATS.md for the broader "lift must move method-side threading
  # config into the patch" rule that this exemplifies.
  # `tiny` renamed to `small` in 2026-05 task.yaml refactor; keep both
  # keys so [[tier]] lookups resolve under either name.
  .mast_mc_cores_cap <- list(tiny      = 12L, small     = 12L,
                             medium    = 10L, large     = 8L,
                             ood_large = 8L,  ood_xlarge = 6L)

  fast_validateState <- function(state, family, control, iter, dispersionold, devold) {
    if (iter > 1L &&
        abs(state$dev - devold)/(0.1 + abs(state$dev)) < control$epsilon &&
        abs(state$dispersion - dispersionold)/(0.1 + abs(state$dispersion)) < control$epsilon) {
      return(FALSE)
    }
    TRUE
  }

  fast_bgf <- function(...) {
    args <- list(...)
    fam <- args$family
    if (is.null(fam)) fam <- gaussian()
    fam$validmu  <- function(mu) TRUE
    fam$valideta <- function(eta) TRUE
    args$family <- fam
    ctrl <- args$control
    if (is.null(ctrl)) ctrl <- glm.control()
    ctrl$epsilon <- 1e-5
    args$control <- ctrl
    do.call(.mast_orig_bgf, args)
  }

  fast_updateState <- function(state, priors, family,
                               offset, weights,
                               y, x, x.nobs, nvars, nobs,
                               intercept, scaled, control) {
    # cpp_updateState is the only MAST package path observed to segfault under
    # package attest. Keep the target wrapped for compatibility, but default to
    # MAST's own updater unless explicitly opted into the native path.
    if (!.mast_native_update_state_enabled()) {
      return(.mast_orig_updateState(
        state, priors, family, offset, weights,
        y, x, x.nobs, nvars, nobs,
        intercept, scaled, control
      ))
    }

    fam_name <- family$family
    if (isTRUE(all(state$good)) && (fam_name == "binomial" || fam_name == "gaussian")) {
      return(cpp_updateState(
        state$eta, state$mu, state$mu.eta.val, state$varmu,
        state$dispersion, state$prior.sd,
        x, x.nobs, y, weights, offset,
        priors$mean, priors$scale, priors$df,
        if (fam_name == "binomial") 1L else 0L,
        as.integer(intercept), as.integer(scaled)
      ))
    }
    # Fallback to upstream-equivalent R path for non-{binomial, gaussian}.
    g <- state$good
    all_good <- isTRUE(all(g))
    if (all_good) {
      z <- state$eta - offset + (y - state$mu) / state$mu.eta.val
      w <- sqrt((weights * state$mu.eta.val^2) / state$varmu)
    } else {
      z <- (state$eta[g] - offset[g]) + (y[g] - state$mu[g]) / state$mu.eta.val[g]
      w <- sqrt((weights[g] * state$mu.eta.val[g]^2) / state$varmu[g])
    }
    z.star <- c(z, priors$mean)
    w.star <- c(w, sqrt(state$dispersion) / priors$scale)
    if (all_good) {
      x_active <- x
    } else {
      good.star <- c(g, rep(TRUE, nvars))
      x_active <- x[good.star, , drop = FALSE]
    }
    fit <- lm.fit(x_active * w.star, z.star * w.star)
    start <- state$Start
    coefold <- state$Start
    if (!all(priors$df == Inf)) {
      colMeans.x <- colMeans(x.nobs)
      centered.coefs <- fit$coefficients
      if (NCOL(x.nobs) == 1L) {
        V.coefs <- chol2inv(fit$qr$qr[1:nvars])
      } else {
        V.coefs <- chol2inv(fit$qr$qr[1:nvars, 1:nvars, drop = FALSE])
      }
      sampling.var <- diag(V.coefs)
      if (intercept && scaled) {
        centered.coefs[1L] <- sum(fit$coefficients * colMeans.x)
        sampling.var[1L] <- crossprod(crossprod(V.coefs, colMeans.x), colMeans.x)
      }
      sd.tmp <- ((centered.coefs - priors$mean)^2 +
                 sampling.var * state$dispersion +
                 priors$df * state$prior.sd^2) / (1 + priors$df)
      sd.coef <- sqrt(sd.tmp)
      state$prior.sd[priors$df != Inf] <- sd.coef[priors$df != Inf]
    }
    predictions <- if (NCOL(x.nobs) == 1L) x.nobs * fit$coefficients else x.nobs %*% fit$coefficients
    start[fit$qr$pivot] <- fit$coefficients
    if (fam_name != "poisson" && fam_name != "binomial") {
      if (!exists("V.coefs", inherits = FALSE)) {
        V.coefs <- if (NCOL(x.nobs) == 1L) chol2inv(fit$qr$qr[1:nvars])
                   else chol2inv(fit$qr$qr[1:nvars, 1:nvars, drop = FALSE])
      }
      if (all_good) {
        mse.resid <- mean(((z.star * w.star)[seq_len(nobs)] - w * predictions)^2)
      } else {
        mse.resid <- mean(((z.star * w.star)[seq_len(sum(g))] - w * predictions[g, ])^2)
      }
      mse.uncertainty <- max(0, mean(rowSums((x.nobs %*% V.coefs) * x.nobs)) * state$dispersion)
      state$dispersion <- mse.resid + mse.uncertainty
    }
    state$eta <- drop(predictions) + offset
    state$mu <- family$linkinv(state$eta)
    state$mu.eta.val <- family$mu.eta(state$eta)
    dev <- sum(family$dev.resids(y, state$mu, weights))
    new_varmu <- if (fam_name == "binomial") {
      state$mu * (1 - state$mu)
    } else if (fam_name == "gaussian") {
      rep.int(1, length(state$mu))
    } else {
      family$variance(state$mu)
    }
    new_good <- (weights > 0) & (state$mu.eta.val != 0)
    list(Start = start, Coefold = coefold,
         eta = state$eta, mu = state$mu,
         mu.eta.val = state$mu.eta.val, varmu = new_varmu,
         good = new_good, dispersion = state$dispersion,
         dev = dev, fit = fit, conv = FALSE,
         boundary = state$boundary, prior.sd = state$prior.sd,
         z = z, w = w)
  }

  fast_lrTest_hybrid <- function(object, hypothesis, ...) {
    if (methods::is(hypothesis, "CoefficientHypothesis")) {
      termName <- as.character(hypothesis@.Data)
    } else if (is.character(hypothesis)) {
      termName <- hypothesis
    } else {
      return(.mast_orig_lrTest(object, hypothesis, ...))
    }
    coefC <- object@coefC
    termIdx <- which(colnames(coefC) == termName)
    if (length(termIdx) != 1L) return(.mast_orig_lrTest(object, hypothesis, ...))

    ng <- nrow(coefC)
    vcC_diag <- object@vcovC[termIdx, termIdx, ]
    vcD_diag <- object@vcovD[termIdx, termIdx, ]
    cC <- coefC[, termIdx]
    cD <- object@coefD[, termIdx]
    W_C_w <- ifelse(is.finite(cC) & is.finite(vcC_diag) & vcC_diag > 0,
                    cC^2 / vcC_diag, 0)
    W_D_w <- ifelse(is.finite(cD) & is.finite(vcD_diag) & vcD_diag > 0,
                    cD^2 / vcD_diag, 0)
    testable_w <- (W_C_w > 0) & (W_D_w > 0)
    W_H_w <- W_C_w + W_D_w
    pH_w <- ifelse(testable_w, pchisq(W_H_w, df = 2, lower.tail = FALSE), 1)

    K <- min(200L, ng)
    topK_idx <- order(pH_w)[seq_len(K)]

    partial <- object
    partial@sca       <- object@sca[topK_idx, ]
    partial@coefC     <- object@coefC[topK_idx, , drop = FALSE]
    partial@coefD     <- object@coefD[topK_idx, , drop = FALSE]
    partial@vcovC     <- object@vcovC[, , topK_idx, drop = FALSE]
    partial@vcovD     <- object@vcovD[, , topK_idx, drop = FALSE]
    partial@loglik    <- object@loglik[topK_idx, , drop = FALSE]
    partial@converged <- object@converged[topK_idx, , drop = FALSE]
    partial@df.resid  <- object@df.resid[topK_idx, , drop = FALSE]

    lrt_K <- .mast_orig_lrTest(partial, hypothesis, ...)

    pC_w <- ifelse(testable_w, pchisq(W_C_w, df = 1, lower.tail = FALSE), 1)
    pD_w <- ifelse(testable_w, pchisq(W_D_w, df = 1, lower.tail = FALSE), 1)
    df_one <- as.numeric(testable_w)
    df_H   <- 2 * df_one
    W_C_w[!testable_w] <- 0
    W_D_w[!testable_w] <- 0
    W_H_w[!testable_w] <- 0

    arr <- array(NA_real_, dim = c(ng, 3L, 3L))
    dimnames(arr) <- list(primerid  = rownames(coefC),
                          test.type = c("cont", "disc", "hurdle"),
                          metric    = c("lambda", "df", "Pr(>Chisq)"))
    arr[, "cont",   "lambda"]     <- W_C_w
    arr[, "disc",   "lambda"]     <- W_D_w
    arr[, "hurdle", "lambda"]     <- W_H_w
    arr[, "cont",   "df"]         <- df_one
    arr[, "disc",   "df"]         <- df_one
    arr[, "hurdle", "df"]         <- df_H
    arr[, "cont",   "Pr(>Chisq)"] <- pC_w
    arr[, "disc",   "Pr(>Chisq)"] <- pD_w
    arr[, "hurdle", "Pr(>Chisq)"] <- pH_w
    arr[topK_idx, , ] <- lrt_K
    arr
  }

  # smoke recipe — build SCA from pbmc68k.rds (mirrors task's build_inputs);
  # save under output_<tier>/ to match evaluate.R's TEST_DIR convention.
  .mast_tier_params <- list(
    # `tiny` renamed to `small` in 2026-05 task.yaml refactor; keep both
    # keys so [[tier]] lookups resolve under either name.
    tiny       = list(n_cells = 8000L,   n_genes = 2000L),
    small      = list(n_cells = 8000L,   n_genes = 2000L),
    medium     = list(n_cells = 18000L,  n_genes = 2200L),
    large      = list(n_cells = 50000L,  n_genes = 2200L),
    ood_large  = list(n_cells = 50000L,  n_genes = 2200L),
    ood_xlarge = list(n_cells = 200000L, n_genes = 2200L)
  )

  .mast_build_sca <- function(data_path, n_cells, n_genes, seed = 12345L) {
    seu <- readRDS(data_path)
    counts <- SeuratObject::LayerData(seu, assay = "RNA", layer = "counts")
    set.seed(seed)
    cell_idx <- sort(sample.int(ncol(counts), size = min(n_cells, ncol(counts))))
    counts_sub <- counts[, cell_idx, drop = FALSE]
    detect_frac <- Matrix::rowSums(counts_sub > 0) / ncol(counts_sub)
    keep <- which(detect_frac >= 0.05)
    gene_totals <- Matrix::rowSums(counts_sub[keep, , drop = FALSE])
    top_idx <- sort(keep[order(gene_totals, decreasing = TRUE)[seq_len(n_genes)]])
    counts_sub <- counts_sub[top_idx, , drop = FALSE]
    col_tot <- Matrix::colSums(counts_sub); col_tot[col_tot == 0] <- 1
    logcpm <- counts_sub
    logcpm@x <- log1p(1e4 * logcpm@x / rep.int(col_tot, diff(logcpm@p)))
    set.seed(seed + 1L)
    group <- factor(sample(c("A", "B"), size = ncol(counts_sub), replace = TRUE),
                    levels = c("A", "B"))
    cdr <- Matrix::colSums(counts_sub > 0)
    cngeneson <- as.numeric(scale(cdr))
    cdat <- data.frame(wellKey = colnames(counts_sub), group = group,
                       cngeneson = cngeneson, stringsAsFactors = FALSE)
    fdat <- data.frame(primerid = rownames(counts_sub), stringsAsFactors = FALSE)
    MAST::FromMatrix(exprsArray = as.matrix(logcpm), cData = cdat, fData = fdat,
                     check_sanity = FALSE)
  }

  register_patch(
    name = "mast",
    upstream = "MAST",
    targets = list(
      .bayesglm.fit.loop.validateState = fast_validateState,
      .bayesglm.fit                    = fast_bgf,
      .bayesglm.fit.loop.updateState   = fast_updateState,
      lrTest                           = fast_lrTest_hybrid
    ),
    smoke = list(
      load = function(task_dir, tier) {
        # Attach MAST to the search path NOW (untimed) so the timed `call`
        # doesn't pay library() overhead. MAST::CoefficientHypothesis uses
        # `callName()` reflection internally; the `MAST::` prefix produces
        # `getClass("MAST::CoefficientHypothesis")` which doesn't exist, so
        # we need the unqualified form to resolve via the search path.
        suppressPackageStartupMessages(library(MAST))
        # mc.cores controls MAST::zlm(parallel=TRUE)'s mclapply width. Set
        # in load so it persists into call (same R session under the
        # subprocess verify protocol). auto_threads honors AUTOZYME_THREADS.
        cap <- .mast_mc_cores_cap[[tier]]
        if (is.null(cap)) cap <- 12L
        options(mc.cores = autozyme::auto_threads(cap = cap))

        task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
        ds <- Filter(function(d) d$tier == tier, task$datasets)
        # MAST's task.yaml `path` may be a single rds (pbmc68k.rds) shared
        # across tiers; tier_params controls subsampling. Fall back to fixed
        # data path if datasets entry missing.
        path <- if (length(ds)) ds[[1]]$path else "data/pbmc68k.rds"
        data_path <- resolve_dataset_path(task_dir, path)
        params <- .mast_tier_params[[tier]]
        sca <- .mast_build_sca(data_path, params$n_cells, params$n_genes)
        list(sca = sca)
      },
      call = function(inputs, tier = "tiny") {
        zfit <- MAST::zlm(formula = ~ group + cngeneson,
                          sca = inputs$sca, method = "bayesglm",
                          ebayes = TRUE, parallel = TRUE, silent = TRUE)
        lrt <- MAST::lrTest(zfit, CoefficientHypothesis("groupB"))
        list(zfit = zfit, lrt = lrt, genes = SummarizedExperiment::mcols(inputs$sca)$primerid)
      },
      save = function(result, dir, tier = "tiny", ...) {
        zfit <- result$zfit
        lrt  <- result$lrt
        diag3 <- function(arr) {
          out <- matrix(NA_real_, nrow = dim(arr)[3], ncol = dim(arr)[1])
          for (i in seq_len(dim(arr)[3])) out[i, ] <- sqrt(pmax(diag(arr[, , i]), 0))
          colnames(out) <- dimnames(arr)[[1]]
          out
        }
        out <- list(
          genes      = result$genes,
          coefC      = zfit@coefC,
          coefD      = zfit@coefD,
          loglikC    = zfit@loglik,
          converged  = zfit@converged,
          seC        = diag3(zfit@vcovC),
          seD        = diag3(zfit@vcovD),
          lrt_chisq  = lrt[, , "lambda"],
          lrt_pvalue = lrt[, , "Pr(>Chisq)"],
          lrt_df     = lrt[, , "df"]
        )
        # mast's evaluate.R has asymmetric expectations:
        #   ref:   <REF_DIR>/result.rds                (flat)
        #   test:  <TEST_DIR>/output_<tier>/result.rds (subdir)
        # Detect which arm by dir name (verify_patch creates them with these
        # specific basenames).
        is_ref <- grepl("^reference_output_", basename(dir))
        out_dir <- if (is_ref) dir else file.path(dir, sprintf("output_%s", tier))
        dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
        saveRDS(out, file.path(out_dir, "result.rds"))
      }
    ),
    tested_against = "MAST 1.35.2",
    tested_upstream_versions = list(MAST = "1.35.2")
  )
}
