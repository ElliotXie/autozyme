# Patch for tradeSeq (v4 — fast native NB-GAM engine).
#
# Lifted from autozyme task `test_tradeseq_fitgam`.
#
# Replaces tradeSeq's internal `.fitGAM` (the workhorse of `fitGAM()`) with a
# self-contained C++/OpenMP negative-binomial penalized-GAM fitter. This is a
# ground-up rewrite of the earlier (v1-v3) patch, which merely re-orchestrated
# per-gene `mgcv::gam()` calls (prefit-G reuse, hoisted formula, fork/PSOCK
# pool, mgcv namespace crossprod rewrites) and stayed bit-exact but only ~1.1-
# 1.3x on a single thread.
#
# The v4 engine (autozyme:::fastgam_fit, src/fastgam.cpp = fitgam_grind
# fastgam_v4_syrk) exploits the fact that the design matrix X, the penalty S and
# the offset are IDENTICAL for every gene on the shared pseudotime/lineage
# design. It builds that design once, reparameterizes into the rank-identifiable
# subspace (so every inner solve is a small PD Cholesky), and selects the two
# hyperparameters (shared smoothing lambda via id=1, NB dispersion theta) on the
# exact Laplace-REML criterion mgcv optimizes — with an analytic gradient
# driving a damped Newton (Nelder-Mead fallback for the rare lambda->inf gene).
# Genes are fit independently in a gene-level OpenMP loop with no R calls.
#
# Equivalence class: this is an ALGORITHMIC (approximate) patch, NOT bit-exact.
# It reproduces mgcv::gam(family="nb") to ~1e-5 on the fitted linear predictor:
# fitted eta within 1% for ~100% of genes (single-lineage) / ~95.5% (2-lineage),
# contrast SEs (which drive the Wald tests) within 5% at median ~0.4%. Validated
# against mgcv on bench_500s, real Paul/Nestorowa, a 1->4 lineage synthetic
# sweep and the 5 HF autozyme datasets (see fitgam_grind/opt/README.md +
# results.csv). Speedups: 5-6x single-thread, 14-75x at 8 threads on the
# single-lineage common case; 1.9-2.0x / 14x on the harder 2-lineage tiers.
#
# The fast path fires only for the standard NB-GAM workload (zyme=TRUE,
# conditions=NULL, family="nb", sce=TRUE, aic=FALSE, no weights, vector offset,
# default U). Anything else falls back to stock tradeSeq `.fitGAM` (correct, no
# speedup). Set zyme=FALSE for a byte-for-byte upstream fit.

if (requireNamespace("tradeSeq",             quietly = TRUE) &&
    requireNamespace("mgcv",                 quietly = TRUE) &&
    requireNamespace("S4Vectors",            quietly = TRUE) &&
    requireNamespace("SummarizedExperiment", quietly = TRUE)) {

  orig_fitGAM_internal <- utils::getFromNamespace(".fitGAM",      "tradeSeq")
  .checks              <- utils::getFromNamespace(".checks",      "tradeSeq")
  .assignCells         <- utils::getFromNamespace(".assignCells", "tradeSeq")
  .findKnots           <- utils::getFromNamespace(".findKnots",   "tradeSeq")
  .get_offset          <- utils::getFromNamespace(".get_offset",  "tradeSeq")

  # --- shared-design context builders (ported from fitgam_grind fastgam_proto.R
  # / fastgam_reduced.R). Built ONCE per fitGAM call; pure base-R linear algebra.

  # From an mgcv G (fit = FALSE) object: assemble the block-diagonal penalty
  # Sbar, its positive-eigenvalue rank / log-det, and the null-space dimension.
  .fg_make_ctx <- function(G) {
    p <- ncol(G$X)
    Sbar <- matrix(0, p, p)
    for (i in seq_along(G$S)) {
      idx <- G$off[i] + seq_len(nrow(G$S[[i]])) - 1L
      Sbar[idx, idx] <- Sbar[idx, idx] + G$S[[i]]
    }
    ev <- eigen(Sbar, symmetric = TRUE, only.values = TRUE)$values
    thresh <- max(ev) * 1e-7
    rankS <- sum(ev > thresh)
    logdetSpos <- sum(log(ev[ev > thresh]))
    list(X = G$X, Sbar = Sbar, offset = as.numeric(G$offset),
         p = p, n = nrow(G$X), rankS = rankS, logdetSpos = logdetSpos)
  }

  # Reduced (rank-identifiable) reparameterization: Z is an orthonormal basis of
  # the design's row space, so Xr = X Z has full column rank and every per-gene
  # solve in C++ is a small PD Cholesky (no eigen). Rank/null-space are read from
  # the SVD of X, not hardcoded, so this is general to 1..k lineages.
  .fg_make_ctx_reduced <- function(ctx) {
    X <- ctx$X; p <- ctx$p
    sv <- svd(X, nu = 0, nv = p)
    tol <- max(dim(X)) * max(sv$d) * .Machine$double.eps
    q <- sum(sv$d > tol)
    Z <- sv$v[, seq_len(q), drop = FALSE]
    Xr <- X %*% Z
    Sr <- crossprod(Z, ctx$Sbar %*% Z)
    Sr <- (Sr + t(Sr)) / 2
    ev <- eigen(Sr, symmetric = TRUE, only.values = TRUE)$values
    thresh <- max(ev) * 1e-8
    rankS <- sum(ev > thresh)
    logdetSpos <- sum(log(ev[ev > thresh]))
    list(Z = Z, Xr = Xr, Sr = Sr, offset = ctx$offset, n = ctx$n, q = q,
         rankS = rankS, logdetSpos = logdetSpos, MpEff = q - rankS)
  }

  # --- the fast `.fitGAM` replacement. Same call surface as tradeSeq's internal
  # `.fitGAM`; the extra `zyme` toggle lets a caller force the upstream path.
  fast_fitGAM_v4 <- function(counts, U = NULL, pseudotime, cellWeights,
                             conditions,
                             genes = seq_len(nrow(counts)),
                             weights = NULL, offset = NULL,
                             nknots = 6, verbose = TRUE,
                             parallel = FALSE,
                             BPPARAM = BiocParallel::bpparam(),
                             aic = FALSE,
                             control = mgcv::gam.control(),
                             sce = TRUE, family = "nb", gcv = FALSE,
                             zyme = TRUE) {

    # gene id resolution (identical to upstream)
    if (methods::is(genes, "character")) {
      if (!all(genes %in% rownames(counts))) {
        stop("The genes ID is not present in the models object.")
      }
      if (any(duplicated(genes))) {
        stop("The genes vector contains duplicates.")
      }
      id <- match(genes, rownames(counts))
    } else {
      id <- genes
    }

    # Fast path envelope. Anything outside it (multi-condition, non-NB family,
    # aic/gcv accounting, per-gene weights/offset, custom U, sce=FALSE model
    # return, empty gene set) delegates to stock tradeSeq — correct, no speedup.
    eligible <- isTRUE(zyme) && is.null(conditions) &&
      identical(family, "nb") && isTRUE(sce) && !isTRUE(aic) &&
      is.null(weights) && is.null(dim(offset)) && is.null(U) &&
      length(id) > 0L
    if (!eligible) {
      return(orig_fitGAM_internal(
        counts = counts, U = U, pseudotime = pseudotime,
        cellWeights = cellWeights, conditions = conditions, genes = genes,
        weights = weights, offset = offset, nknots = nknots, verbose = verbose,
        parallel = parallel, BPPARAM = BPPARAM, aic = aic, control = control,
        sce = sce, family = family, gcv = gcv
      ))
    }

    if (is.null(dim(pseudotime))) {
      pseudotime <- matrix(pseudotime, nrow = length(pseudotime))
    }
    if (is.null(dim(cellWeights))) {
      cellWeights <- matrix(cellWeights, nrow = length(cellWeights))
    }

    # cell->lineage assignment + library-size offset. `.assignCells` draws
    # rmultinom from the ambient RNG state exactly as upstream does at this point
    # (the caller seeds before fitGAM), so wSamp matches the baseline fit.
    .checks(pseudotime, cellWeights, U, counts, conditions)
    wSamp  <- .assignCells(cellWeights)
    offset <- .get_offset(offset, counts)
    U      <- matrix(rep(1, nrow(pseudotime)), ncol = 1)
    nLin   <- ncol(pseudotime)

    # knot placement: single-lineage uses quantile probs with the upstream
    # duplicate->seq repair; multi-lineage delegates to tradeSeq::.findKnots.
    if (nLin == 1L) {
      knotLocs <- stats::quantile(
        pseudotime[, 1], probs = (0:(nknots - 1)) / (nknots - 1)
      )
      if (any(duplicated(knotLocs))) {
        knotLocs <- seq(min(pseudotime[, 1]), max(pseudotime[, 1]),
                        length = nknots)
      }
      knotLocs[1]      <- min(pseudotime[, 1])
      knotLocs[nknots] <- max(pseudotime[, 1])
      knotList <- list(t1 = knotLocs)
    } else {
      knotList <- .findKnots(nknots, pseudotime, wSamp)
    }

    # shared design: data frame + smooth formula (k=nknots resolved from fenv).
    fenv <- new.env(parent = globalenv())
    assign("nknots", nknots, envir = fenv)
    dat <- list()
    for (ii in seq_len(nLin)) {
      dat[[paste0("t", ii)]] <- pseudotime[, ii]
      dat[[paste0("l", ii)]] <- 1 * (wSamp[, ii] == 1)
    }
    dat$U <- U
    dat$offset <- offset
    smoothForm <- stats::as.formula(paste0(
      "y ~ -1 + U + ",
      paste(vapply(seq_len(nLin), function(ii) {
        paste0("s(t", ii, ", by=l", ii, ", bs='cr', id=1, k=nknots)")
      }, FUN.VALUE = ""), collapse = "+"),
      " + offset(offset)"
    ))
    environment(smoothForm) <- fenv

    counts_for_gam <- as.matrix(counts)[id, , drop = FALSE]
    storage.mode(counts_for_gam) <- "double"
    Gn <- nrow(counts_for_gam)

    # Two one-off mgcv setups (cheap, on gene 1 only):
    #   Gpre = gam(fit = FALSE)  -> the shared X / S / offset the engine needs.
    #   m1   = one real gam fit  -> the canonical lpmatrix / model frame / knot
    #                               points stored verbatim in the fitGAM output.
    # Both use STOCK mgcv (no namespace overrides), so the metadata the output
    # carries is byte-identical to what upstream fitGAM would store.
    dat1 <- dat
    dat1$y <- as.numeric(counts_for_gam[1, ])
    Gpre <- suppressWarnings(mgcv::gam(
      smoothForm, family = "nb", knots = knotList,
      control = control, fit = FALSE, data = dat1
    ))
    m1 <- suppressWarnings(mgcv::gam(
      smoothForm, family = "nb", knots = knotList,
      control = control, data = dat1
    ))

    ctx <- .fg_make_ctx(Gpre)
    cr  <- .fg_make_ctx_reduced(ctx)
    Z   <- cr$Z
    q   <- ncol(Z)

    Xlp        <- stats::predict(m1, type = "lpmatrix")
    dm         <- m1$model[, -1]
    knotPoints <- m1$smooth[[1]]$xp
    coefNames  <- colnames(Xlp)

    # gene-level OpenMP NB-GAM fit. Result is independent of nthreads (each gene
    # is a separate solve with no cross-gene reduction), so the thread count is a
    # pure performance knob — honor the attest/user thread budget.
    nthreads <- auto_threads()
    Y <- t(counts_for_gam)               # cells x genes
    storage.mode(Y) <- "double"
    res <- fastgam_fit(
      cr$Xr, cr$Sr, cr$offset, Y,
      rankS = cr$rankS, logdetSpos = cr$logdetSpos, MpEff = cr$MpEff,
      start_llam = 5.0, start_lth = 2.5, nthreads = as.integer(nthreads)
    )

    # back-project reduced coefficients / Hessians to the full coefficient space.
    betaAll <- t(Z %*% res$a)            # genes x ncoef
    betaAllDf <- as.data.frame(betaAll)
    rownames(betaAllDf) <- rownames(counts)[id]
    colnames(betaAllDf) <- coefNames

    SigmaAll <- lapply(seq_len(Gn), function(g) {
      out <- tryCatch({
        H  <- matrix(res$H[, g], q, q)
        Vp <- Z %*% chol2inv(chol(H)) %*% t(Z)
        dimnames(Vp) <- list(coefNames, coefNames)
        Vp
      }, error = function(e) NA)
      out
    })

    # A gene that exhausts the per-gene solve budget in C++ is flagged; treat it
    # as non-converged (upstream's convention on a failed/warned gene fit).
    converged <- as.logical(res$capped < 0.5)

    return(list(beta = betaAllDf,
                Sigma = SigmaAll,
                X = Xlp,
                dm = dm,
                knotPoints = knotPoints,
                converged = converged))
  }

  .tradeseq_summarize_fit <- function(fit) {
    trade  <- as.data.frame(SummarizedExperiment::rowData(fit)$tradeSeq)
    beta   <- as.matrix(trade$beta)
    sigma  <- do.call(rbind, lapply(trade$Sigma, function(x) as.numeric(x)))
    x_mat  <- as.matrix(SummarizedExperiment::colData(fit)$tradeSeq$X)
    dm_mat <- as.matrix(SummarizedExperiment::colData(fit)$tradeSeq$dm)
    list(
      beta       = beta,
      sigma      = sigma,
      converged  = as.logical(trade$converged),
      X          = x_mat,
      dm         = dm_mat,
      knots      = as.numeric(S4Vectors::metadata(fit)$tradeSeq$knots),
      gene_names = rownames(fit),
      cell_names = colnames(fit)
    )
  }

  register_patch(
    name = "tradeseq",
    upstream = "tradeSeq",
    targets = list(.fitGAM = fast_fitGAM_v4),
    smoke = list(
      load = function(task_dir, tier) {
        task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
        ds <- Filter(function(d) d$tier == tier, task$datasets)
        if (length(ds) == 0L) stop("no dataset for tier '", tier, "' in task.yaml")
        obj <- readRDS(resolve_dataset_path(task_dir, ds[[1]]$path))
        storage.mode(obj$counts)      <- "integer"
        storage.mode(obj$pseudotime)  <- "double"
        storage.mode(obj$cellWeights) <- "double"
        obj
      },
      call = function(inputs) {
        # Baseline fairness: tradeSeq::fitGAM has its own gene-parallel layer
        # (parallel = TRUE + BPPARAM) that any upstream user can reach. The
        # honest baseline at thread=N therefore runs N workers on the backend
        # that platform actually offers -- else the patched-vs-baseline speedup
        # at t>1 silently pockets the parallel scaling the baseline was denied.
        # MulticoreParam forks, so on Windows it yields a single worker; there
        # the reachable backend is SnowParam, which is what bpparam() returns.
        # Read the same ZYME_THREADS the attest harness sets for the patched
        # side. parallel=TRUE splits the per-gene fits and recombines in gene
        # order -> numerically identical to serial, so the saved concordance
        # output is unchanged.
        n_threads <- suppressWarnings(as.integer(Sys.getenv("ZYME_THREADS", unset = "1")))
        if (is.na(n_threads) || n_threads < 1L) n_threads <- 1L
        suppressWarnings(tradeSeq::fitGAM(
          counts      = inputs$counts,
          pseudotime  = inputs$pseudotime,
          cellWeights = inputs$cellWeights,
          nknots      = inputs$nknots,
          verbose     = FALSE,
          parallel    = n_threads > 1L,
          BPPARAM     = if (n_threads <= 1L) BiocParallel::SerialParam()
                        else if (.Platform$OS.type == "windows")
                          BiocParallel::SnowParam(workers = n_threads)
                        else BiocParallel::MulticoreParam(workers = n_threads),
          sce         = TRUE
        ))
      },
      save = function(result, dir, ...) {
        out <- .tradeseq_summarize_fit(result)
        saveRDS(out, file.path(dir, "fitgam_output.rds"), compress = "xz")
      }
    ),
    tested_against = "tradeSeq 1.22.0",
    tested_upstream_versions = list(tradeSeq = "1.22.0")
  )
}
