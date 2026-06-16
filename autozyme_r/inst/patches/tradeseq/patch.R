# Patch for tradeSeq.
#
# Lifted from autozyme task `test_tradeseq_fitgam`. Replaces tradeSeq's
# internal `.fitGAM` (the workhorse of `fitGAM()`) with a hoisted-formula /
# prefit-G / fork-pool variant that avoids per-gene formula reconstruction
# and lets sklearn's mgcv::gam reuse a single G-prefit across all genes.
#
# tradeSeq's `.fitGAM` references several other internal helpers
# (.checks / .assignCells / .findKnots / .get_offset). The original script
# patched this via `environment(fast_) <- asNamespace("tradeSeq")` which
# would clobber our package's enclosure. Instead: capture every internal
# at file scope inside the requireNamespace gate, so the fast function's
# closure resolves them via lexical scoping without env-mutation.

if (requireNamespace("tradeSeq",             quietly = TRUE) &&
    requireNamespace("mgcv",                 quietly = TRUE) &&
    requireNamespace("BiocParallel",         quietly = TRUE) &&
    requireNamespace("S4Vectors",            quietly = TRUE) &&
    requireNamespace("SummarizedExperiment", quietly = TRUE) &&
    requireNamespace("pbapply",              quietly = TRUE)) {

  orig_fitGAM_internal <- utils::getFromNamespace(".fitGAM",      "tradeSeq")
  .checks              <- utils::getFromNamespace(".checks",      "tradeSeq")
  .assignCells         <- utils::getFromNamespace(".assignCells", "tradeSeq")
  .findKnots           <- utils::getFromNamespace(".findKnots",   "tradeSeq")
  .get_offset          <- utils::getFromNamespace(".get_offset",  "tradeSeq")

  # mgcv internals we cross-patch. Keep only the bit-exact subset: a few
  # `t(X) %*% Y` -> crossprod rewrites in gam.fit4, plus NB-family helper
  # closures whose outputs match mgcv on the audited tiers. The faster dDeta
  # and nb$ls replacements are intentionally not installed: tiny local
  # floating-point differences there were amplified by mgcv's optimizer into
  # OOD coefficient drift.
  orig_nb        <- utils::getFromNamespace("nb",       "mgcv")
  orig_dDeta     <- utils::getFromNamespace("dDeta",    "mgcv")
  orig_gam_fit4  <- utils::getFromNamespace("gam.fit4", "mgcv")

  # --- fast_nb: Rcpp-backed NB family safe subset. Replaces R closures for
  # Dd, linkinv, mu.eta, dev.resids. Keeps the original ls closure because
  # the Rcpp scalar-sum version was not bit-exact on OOD inputs. Closure
  # captures the family's own .Theta env so the dev.resids(theta=NULL)
  # fallback keeps working.
  fast_nb <- function(theta = NULL, link = "log") {
    # orig_nb does substitute(link) internally, so a forwarded `link = link`
    # delivers the symbol "link" rather than its value. do.call evaluates the
    # arg list first, so the resulting call carries the string literal.
    fam <- do.call(orig_nb, list(theta = theta, link = link))
    fam_env <- environment(fam$dev.resids)
    # scBLAS opt-in gate (Phase 5 integration). Checked ONCE at family
    # construction — not per IRLS step — so gate evaluation cost is one
    # Sys.getenv + one requireNamespace per fit. Default OFF preserves the
    # existing autozyme nb_Dd_cpp / linkinv_log_cpp / nb_dev_resids_cpp paths
    # bit-for-bit. Set AUTOZYME_SCBLAS_NB=1 (or AUTOZYME_TRADESEQ_SCBLAS_NB=1)
    # to route the 3 NB callbacks (Dd, linkinv, dev.resids) through scblasR.
    # To revert: delete this `if (use_scblas) ... else ...` block and keep
    # only the `else` body inline.
    use_scblas <- .az_feature_enabled("scblas_nb",
                                       patch = "tradeseq", default = FALSE) &&
                  requireNamespace("scblasR", quietly = TRUE)
    if (use_scblas) {
      # mgcv passes `theta` on the LOG scale to Dd / dev.resids. Call the
      # lower-level Rcpp entry (scblasR:::scblasR_nb_derivs) which also takes
      # theta_log directly — saves an exp()/log() round-trip per IRLS callback
      # and skips the user-facing R wrapper's argument validation costs.
      sc_nb_derivs    <- scblasR:::scblasR_nb_derivs
      sc_nb_linkinv   <- scblasR:::scblasR_nb_linkinv
      sc_nb_dev_resid <- scblasR:::scblasR_nb_dev_resids
      sc_nb_ls        <- scblasR:::scblasR_nb_ls
      fam$Dd <- function(y, mu, theta, wt, level = 0) {
        sc_nb_derivs(y, mu, theta, wt, as.integer(level))
      }
      if (identical(link, "log")) {
        fam$linkinv <- function(eta) sc_nb_linkinv(eta)
        fam$mu.eta  <- function(eta) sc_nb_linkinv(eta)
        fam$dev.resids <- function(y, mu, wt, theta = NULL) {
          if (is.null(theta)) theta <- get(".Theta", envir = fam_env)
          sc_nb_dev_resid(y, mu, wt, theta)
        }
      }
      # `fam$ls` was historically left as the mgcv R-level closure because
      # autozyme's earlier nb_ls_cpp was not bit-exact on OOD inputs.
      # scblasR's nb_ls uses long-double accumulators (matching R's sum()
      # precision) and matches mgcv to ≤ 75 ulps relative across n / θ
      # configurations — safe to replace. Profile of the R-level closure
      # showed ~9% self time in gam.fit4's IRLS loop; replacing it with the
      # C-level version is the biggest unclaimed slice in tradeseq.
      fam$ls <- function(y, w, theta, scale) {
        sc_nb_ls(y, w, theta, scale)
      }
    } else {
      fam$Dd <- function(y, mu, theta, wt, level = 0) {
        nb_Dd_cpp(y, mu, theta, wt, level)
      }
      if (identical(link, "log")) {
        fam$linkinv <- function(eta) linkinv_log_cpp(eta)
        fam$mu.eta  <- function(eta) linkinv_log_cpp(eta)
        fam$dev.resids <- function(y, mu, wt, theta = NULL) {
          if (is.null(theta)) theta <- get(".Theta", envir = fam_env)
          nb_dev_resids_cpp(y, mu, wt, theta)
        }
      }
    }
    fam
  }

  # --- fast_dDeta: scbblas-accelerated chain rule for NB log link --------
  # mgcv::dDeta is the chain-rule function that takes mu-space derivatives
  # (from fam$Dd) and converts them to eta-space. The default R-level impl
  # is ~88 lines doing R-vector arithmetic and (with profile data) accounts
  # for ~14% of total gam() time on tradeseq workloads. autozyme historically
  # avoided patching it due to OOD coefficient drift in their earlier C port.
  # scblasR's nb_dDeta_log fuses the fam$Dd + chain-rule into a single
  # C99/NEON pass — preserving long-double accumulator discipline.
  #
  # We invoke fast_dDeta only when fam is an NB family with link == "log".
  # Other links / non-NB families fall through to the original mgcv::dDeta
  # so this replacement is a pure superset (covers the hot tradeseq path
  # without touching anything else mgcv users might be doing).
  fast_dDeta <- local({
    sc_nb_dDeta <- if (requireNamespace("scblasR", quietly = TRUE))
      scblasR:::scblasR_nb_dDeta else NULL
    function(y, mu, wt, theta, fam, deriv = 0) {
      if (is.null(sc_nb_dDeta) ||
          !identical(fam$link, "log") ||
          is.null(fam$family) ||
          !grepl("Negative Binomial", fam$family, ignore.case = TRUE)) {
        return(orig_dDeta(y, mu, wt, theta, fam, deriv = deriv))
      }
      # scblasR_nb_dDeta returns the full eta-space derivative struct in one
      # C99 call. Field naming maps as: scbblas uses `Deta_Deta2` /
      # `Deta_EDeta2` (underscores) where mgcv uses `Deta.Deta2` /
      # `Deta.EDeta2` (dots).
      sc <- sc_nb_dDeta(y, mu, wt, theta, as.integer(deriv))
      d <- list(
        Deta        = sc$Deta,
        Deta2       = sc$Deta2,
        EDeta2      = sc$EDeta2,
        Deta.Deta2  = sc$Deta_Deta2,
        Deta.EDeta2 = sc$Deta_EDeta2,
        Dth         = 0, Detath = 0, Deta3 = 0, Deta2th = 0,
        EDeta2th    = 0, EDeta3 = 0,
        Deta4       = 0, Dth2 = 0, Detath2 = 0, Deta3th = 0, Deta2th2 = 0
      )
      if (deriv > 0) {
        d$Dth      <- sc$Dth
        d$Detath   <- sc$Detath
        d$Deta3    <- sc$Deta3
        d$Deta2th  <- sc$Deta2th
        d$EDeta2th <- sc$EDeta2th
        # EDeta3 not provided by scbblas (autozyme nb_Dd_cpp also doesn't
        # supply r$EDmu3 — mgcv leaves d$EDeta3 = 0 in that case).
      }
      if (deriv > 1) {
        d$Deta4    <- sc$Deta4
        d$Dth2     <- sc$Dth2
        d$Detath2  <- sc$Detath2
        d$Deta2th2 <- sc$Deta2th2
        d$Deta3th  <- sc$Deta3th
      }
      d$good <- as.logical(sc$good)
      d
    }
  })

  # --- fast_gam.fit4: textual substitution. `t(X) %*% Y` -> crossprod helper.
  # On Windows the helper can opt into autozyme's dynamic BLAS backend, but
  # tradeSeq defaults it off: the tiny-tier guardrail showed NumPy/SciPy
  # OpenBLAS regressed these small mgcv crossprod calls by >10x. The 5 patches
  # still keep the base crossprod rewrite, and users can force the dynamic path
  # with AUTOZYME_TRADESEQ_BLAS=1 or options(autozyme.tradeseq.dynamic_blas=TRUE)
  # before activation for local experiments. The decision is made once while
  # building patched gam.fit4; the default path emits plain crossprod(...) so
  # mgcv's hot loop does not pay an option/env check per IRLS step.
  # Sentinel warns on miss so a future mgcv minor bump surfaces the drift
  # loudly instead of silently degrading speedup.
  fast_gam_fit4 <- local({
    src_text <- paste(deparse(orig_gam_fit4, width.cutoff = 500L,
                              control = c("keepInteger", "keepNA")),
                      collapse = "\n")
    use_dynamic_blas <- .Platform$OS.type == "windows" &&
      .az_dynamic_blas_enabled("tradeseq", default = FALSE)
    cp <- function(x, y) {
      if (isTRUE(use_dynamic_blas)) {
        sprintf("autozyme:::.az_windows_crossprod(%s, %s)", x, y)
      } else {
        sprintf("crossprod(%s, %s)", x, y)
      }
    }
    patches <- list(
      c("t(start) %*% St %*% start",
        cp("start", "St %*% start")),
      c("t(null.coef) %*% St %*% null.coef",
        cp("null.coef", "St %*% null.coef")),
      c("t(T) %*% null.coef", cp("T", "null.coef")),
      c("t(T) %*% start",     cp("T", "start")),
      c("2 * t(x[good, , drop = FALSE]) %*% ((w[good] * (x %*% start)[good] - wz[good]))",
        paste0("2 * ", cp("x[good, , drop = FALSE]",
                          "(w[good] * (x %*% start)[good] - wz[good])")))
    )
    for (p in patches) {
      before_len <- nchar(src_text)
      src_text <- gsub(p[1], p[2], src_text, fixed = TRUE)
      after_len <- nchar(src_text)
      if (identical(before_len, after_len)) {
        warning(sprintf(
          "[autozyme/tradeseq] gam.fit4 textual patch did not match: '%s' (mgcv %s drift?)",
          p[1], as.character(utils::packageVersion("mgcv"))
        ), call. = FALSE)
      }
    }
    patched <- eval(parse(text = src_text))
    environment(patched) <- environment(orig_gam_fit4)
    patched
  })

  .tradeseq_install_mgcv_overrides <- function() {
    utils::assignInNamespace("nb",       fast_nb,       ns = asNamespace("mgcv"))
    # scBLAS opt-in gate (Phase 5 polish 2026-05-31). Same env var as fast_nb;
    # when ON, install fast_dDeta (scbblas-accelerated chain-rule). When OFF,
    # keep orig_dDeta. To revert: just always install orig_dDeta here.
    if (.az_feature_enabled("scblas_nb",
                             patch = "tradeseq", default = FALSE) &&
        requireNamespace("scblasR", quietly = TRUE)) {
      utils::assignInNamespace("dDeta",  fast_dDeta,  ns = asNamespace("mgcv"))
    } else {
      utils::assignInNamespace("dDeta",  orig_dDeta,  ns = asNamespace("mgcv"))
    }
    utils::assignInNamespace("gam.fit4", fast_gam_fit4, ns = asNamespace("mgcv"))
  }
  .tradeseq_restore_mgcv_overrides <- function() {
    utils::assignInNamespace("nb",       orig_nb,       ns = asNamespace("mgcv"))
    utils::assignInNamespace("dDeta",    orig_dDeta,    ns = asNamespace("mgcv"))
    utils::assignInNamespace("gam.fit4", orig_gam_fit4, ns = asNamespace("mgcv"))
  }

  fast_fitGAM_hoist_formula <- function(counts, U = NULL, pseudotime, cellWeights,
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
    if (!isTRUE(zyme) || !is.null(conditions)) {
      return(orig_fitGAM_internal(
        counts = counts, U = U, pseudotime = pseudotime,
        cellWeights = cellWeights, conditions = conditions, genes = genes,
        weights = weights, offset = offset, nknots = nknots, verbose = verbose,
        parallel = parallel, BPPARAM = BPPARAM, aic = aic, control = control,
        sce = sce, family = family, gcv = gcv
      ))
    }

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

    if (parallel) {
      BiocParallel::register(BPPARAM)
      if (verbose) {
        BPPARAM$tasks <- as.integer(40)
        BPPARAM$progressbar <- TRUE
      }
    }

    if (is.null(dim(pseudotime))) {
      pseudotime <- matrix(pseudotime, nrow = length(pseudotime))
    }
    if (is.null(dim(cellWeights))) {
      cellWeights <- matrix(cellWeights, nrow = length(cellWeights))
    }

    .checks(pseudotime, cellWeights, U, counts, conditions)
    wSamp <- .assignCells(cellWeights)

    for (ii in seq_len(ncol(pseudotime))) {
      assign(paste0("t", ii), pseudotime[, ii])
    }
    for (ii in seq_len(ncol(pseudotime))) {
      assign(paste0("l", ii), 1 * (wSamp[, ii] == 1))
    }

    offset <- .get_offset(offset, counts)

    if (is.null(U)) {
      U <- matrix(rep(1, nrow(pseudotime)), ncol = 1)
    }

    # Upstream `.fitGAM` does NOT bump nthreads; the original Mac
    # pipeline/run.R did because forked workers shared one thread budget.
    # On Win OpenBLAS, forcing nthreads >= 2 here changes BLAS reduction
    # order vs baseline (which uses mgcv::gam.control()$nthreads = 1) and
    # amplifies into max_abs_diff ~0.5 on the multi-lineage OOD tier where
    # the shared-penalty IRLS surface is bumpy. Keep parity with baseline.
    # control$nthreads <- max(2L, control$nthreads)  # removed 2026-05-22

    if (ncol(pseudotime) == 1L) {
      knotLocs <- stats::quantile(
        pseudotime[, 1],
        probs = (0:(nknots - 1)) / (nknots - 1)
      )
      if (any(duplicated(knotLocs))) {
        knotLocs <- seq(min(pseudotime[, 1]), max(pseudotime[, 1]),
                        length = nknots)
      }
      knotLocs[1] <- min(pseudotime[, 1])
      knotLocs[nknots] <- max(pseudotime[, 1])
      knotList <- list(t1 = knotLocs)
    } else {
      knotList <- .findKnots(nknots, pseudotime, wSamp)
    }
    smoothForm_template <- stats::as.formula(
      paste0(
        "y ~ -1 + U + ",
        paste(vapply(seq_len(ncol(pseudotime)), function(ii) {
          paste0("s(t", ii, ", by=l", ii, ", bs='cr', id=1, k=nknots)")
        }, FUN.VALUE = "formula"), collapse = "+"),
        " + offset(offset)"
      )
    )

    counts_for_gam <- as.matrix(counts)[id, , drop = FALSE]
    use_prefit_G <- is.null(weights) && is.null(dim(offset)) &&
      identical(family, "nb") && length(id) > 0
    if (use_prefit_G) {
      y <- unname(counts_for_gam[1, ])
      s <- mgcv::s
      prefit_G <- suppressWarnings(mgcv::gam(
        smoothForm_template, family = family, knots = knotList,
        weights = weights, control = control, fit = FALSE
      ))
      prefit_G$cl$fit <- NULL
    }

    teller <- 0
    converged <- rep(TRUE, length(genes))
    counts_to_Gam <- function(y) {
      teller <<- teller + 1
      nknots <- nknots
      if (!is.null(weights)) weights <- weights[teller, ]
      if (!is.null(dim(offset))) offset <- offset[teller, ]
      smoothForm <- smoothForm_template
      environment(smoothForm) <- environment()
      s <- mgcv::s
      m <- suppressWarnings(try(withCallingHandlers({
        if (use_prefit_G) {
          G <- prefit_G
          G$y <- y
          G$mf[[1]] <- y
          G$family <- mgcv::nb()
          mgcv::gam(G = G, control = control)
        } else {
          mgcv::gam(smoothForm, family = family, knots = knotList,
                    weights = weights, control = control)
        }
      },
      error = function(e) {
        converged[teller] <<- FALSE
        return(structure("Fitting errored",
                         class = c("try-error", "character")))
      },
      warning = function(w) {
        converged[teller] <<- FALSE
      }), silent = TRUE))
      return(m)
    }

    compactFits <- NULL
    if (parallel) {
      gamList <- BiocParallel::bplapply(
        as.data.frame(t(as.matrix(counts)[id, ])),
        counts_to_Gam, BPPARAM = BPPARAM
      )
    } else {
      adaptive_worker_count <- if (nrow(counts_for_gam) <= 1000L) {
        11L
      } else if (nrow(counts_for_gam) <= 3000L) {
        12L
      } else {
        11L
      }
      worker_count <- auto_threads(cap = adaptive_worker_count)

      # On Windows, the PSOCK cost model is dominated by fixed per-worker
      # startup. Empirically: spawn ~1.5s/worker (sequential), then a parallel
      # clusterCall that pushes ~2-5s of init (libPaths + namespace rebinds +
      # OMP env), plus clusterExport of the 500KB prefit_G to each worker.
      # Total overhead ~= 2*n_workers + ~10s.
      # Per-gene work is roughly nrow * 0.07s baseline. The total wall-clock
      # minimizing n_workers solves
      #   t(n) = 2n + 10 + baseline/n,  t'(n)=0  =>  n* = sqrt(baseline/2)
      # baseline ~ nrow * 0.07 => n* ~ sqrt(nrow * 0.035).
      # Below 200 genes the work is too small to justify even spawning a
      # cluster at all — go serial.
      if (.Platform$OS.type == "windows" && worker_count > 1L) {
        if (nrow(counts_for_gam) < 200L) {
          worker_count <- 1L
        } else {
          optimal <- as.integer(round(sqrt(nrow(counts_for_gam) * 0.035)))
          worker_count <- min(worker_count, max(2L, optimal))
        }
      }

      if (use_prefit_G && !verbose && sce && !aic && worker_count > 1L) {
        worker_control <- control
        worker_control$nthreads <- 1L
        # Use fully-qualified namespace calls (mgcv::nb, mgcv::gam, stats::coef)
        # instead of cached aliases. parLapply does NOT preserve a closure's
        # enclosing locals across worker boundaries — only vars explicitly
        # exported via clusterExport land in the worker's global env.
        # Aliases like `nb_fn <- mgcv::nb` were silently dropped, leading to
        # "could not find function 'nb_fn'" once the worker tried to fit.
        # Qualified calls also pick up the WORKER's own activated fast_nb,
        # which is what we want.
        fit_gene_index <- function(gene_index) {
          gene_converged <- TRUE
          y <- unname(counts_for_gam[gene_index, ])
          m <- suppressWarnings(try(withCallingHandlers({
            G <- prefit_G
            G$y <- y
            G$mf[[1]] <- y
            G$family <- mgcv::nb()
            mgcv::gam(G = G, control = worker_control)
          },
          error = function(e) {
            gene_converged <<- FALSE
            return(structure("Fitting errored",
                             class = c("try-error", "character")))
          },
          warning = function(w) {
            gene_converged <<- FALSE
          }), silent = TRUE))
          if (methods::is(m, "try-error")) {
            return(list(error = TRUE, coef = NA, Vp = NA, meta = NULL,
                        converged = gene_converged))
          }
          meta <- NULL
          if (gene_index == 1L) {
            meta <- list(
              X = stats::predict(m, type = "lpmatrix"),
              dm = m$model[, -1],
              knotPoints = m$smooth[[1]]$xp
            )
          }
          list(error = FALSE, coef = stats::coef(m), Vp = m$Vp, meta = meta,
               converged = gene_converged)
        }
        if (.Platform$OS.type != "windows") {
          # Fork path: workers inherit the assignInNamespace state from the
          # parent process (the on_activate hook already ran), so prefit_G
          # and the mgcv overrides are visible without extra wiring.
          compactFits <- .zyme_mclapply(
            seq_len(nrow(counts_for_gam)), fit_gene_index,
            mc.cores = worker_count, mc.preschedule = TRUE
          )
        } else {
          # PSOCK path: each child is a fresh R session and does NOT inherit
          # (a) the assignInNamespace mgcv overrides, or (b) the parent's
          # .libPaths(). Propagate libPaths so workers find the *same*
          # autozyme installation (important when running from a non-default
          # library, e.g. R CMD INSTALL --library=<tmp>), then push the
          # already-built fast_nb / original dDeta / fast_gam_fit4 closures and
          # do the assignInNamespace rebinds directly in each worker. This
          # SKIPS the full autozyme::activate("tradeseq") flow — which would
          # re-source the patch file and re-run the gam.fit4 textual
          # substitution (deparse + 5 gsub + parse + eval) in every worker,
          # costing ~2-5s/worker for no benefit since the parent already did
          # this work and we can just transmit the result.
          patch_env <- environment(sys.function())
          fast_nb_obj       <- patch_env$fast_nb
          orig_dDeta_obj    <- patch_env$orig_dDeta
          fast_gam_fit4_obj <- patch_env$fast_gam_fit4
          parent_lib_paths  <- .libPaths()

          cl <- parallel::makePSOCKcluster(min(worker_count,
                                                nrow(counts_for_gam)))
          on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)
          parallel::clusterCall(cl, function(lp, nb_obj, dDeta_obj, gf4_obj) {
            .libPaths(lp)
            Sys.setenv(OMP_NUM_THREADS = "1",
                       OPENBLAS_NUM_THREADS = "1",
                       MKL_NUM_THREADS = "1")
            requireNamespace("autozyme", quietly = TRUE)
            requireNamespace("mgcv",     quietly = TRUE)
            utils::assignInNamespace("nb",       nb_obj,    ns = asNamespace("mgcv"))
            utils::assignInNamespace("dDeta",    dDeta_obj, ns = asNamespace("mgcv"))
            utils::assignInNamespace("gam.fit4", gf4_obj,   ns = asNamespace("mgcv"))
            invisible(NULL)
          }, parent_lib_paths, fast_nb_obj, orig_dDeta_obj, fast_gam_fit4_obj)
          parallel::clusterExport(cl,
            varlist = c("prefit_G", "counts_for_gam", "worker_control"),
            envir = environment())
          compactFits <- parallel::parLapply(cl,
            seq_len(nrow(counts_for_gam)), fit_gene_index)
        }
        gamList <- NULL
        converged <- vapply(compactFits, `[[`, "converged",
                              FUN.VALUE = TRUE)
      } else {
        counts_to_Gam_row <- function(gene_index) {
          counts_to_Gam(unname(counts_for_gam[gene_index, ]))
        }
        if (verbose) {
          gamList <- pbapply::pblapply(
            seq_len(nrow(counts_for_gam)),
            counts_to_Gam_row
          )
        } else {
          gamList <- lapply(
            seq_len(nrow(counts_for_gam)),
            counts_to_Gam_row
          )
        }
      }
    }

    if (aic) {
      aicVals <- unlist(lapply(gamList, function(x) {
        if (class(x)[1] == "try-error") return(NA)
        x$aic
      }))
      if (gcv) {
        gcvVals <- unlist(lapply(gamList, function(x) {
          if (class(x)[1] == "try-error") return(NA)
          x$gcv.ubre
        }))
        return(list(aicVals, gcvVals))
      } else {
        return(aicVals)
      }
    }

    if (sce) {
      if (!is.null(compactFits)) {
        valid_beta <- which(!vapply(compactFits, `[[`, "error",
                                    FUN.VALUE = TRUE))[1]
        coef_template <- compactFits[[valid_beta]]$coef
        betaAll <- matrix(NA_real_, nrow = length(compactFits),
                          ncol = length(coef_template))
        colnames(betaAll) <- names(coef_template)
        for (model_index in seq_along(compactFits)) {
          if (!compactFits[[model_index]]$error) {
            betaAll[model_index, ] <- compactFits[[model_index]]$coef
          }
        }
        betaAllDf <- as.data.frame(betaAll)
        rownames(betaAllDf) <- rownames(counts)[id]

        SigmaAll <- lapply(compactFits, function(m) {
          if (m$error) NA else m$Vp
        })

        meta_id <- which(vapply(compactFits, function(m) {
          !is.null(m$meta)
        }, FUN.VALUE = TRUE))[1]
        X <- compactFits[[meta_id]]$meta$X
        dm <- compactFits[[meta_id]]$meta$dm
        knotPoints <- compactFits[[meta_id]]$meta$knotPoints
      } else {
        betaAll <- lapply(gamList, function(m) {
          if (methods::is(m, "try-error")) {
            NA
          } else {
            beta <- matrix(stats::coef(m), ncol = 1)
            rownames(beta) <- names(stats::coef(m))
            beta
          }
        })
        betaAllDf <- data.frame(t(do.call(cbind, betaAll)))
        rownames(betaAllDf) <- rownames(counts)[id]

        SigmaAll <- lapply(gamList, function(m) {
          if (methods::is(m, "try-error")) NA else m$Vp
        })

        element <- min(which(!is.na(SigmaAll)))
        m <- gamList[[element]]
        X <- stats::predict(m, type = "lpmatrix")
        dm <- m$model[, -1]
        knotPoints <- m$smooth[[1]]$xp
      }

      return(list(beta = betaAllDf,
                  Sigma = SigmaAll,
                  X = X,
                  dm = dm,
                  knotPoints = knotPoints,
                  converged = converged))
    } else {
      return(gamList)
    }
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
    targets = list(.fitGAM = fast_fitGAM_hoist_formula),
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
        suppressWarnings(tradeSeq::fitGAM(
          counts      = inputs$counts,
          pseudotime  = inputs$pseudotime,
          cellWeights = inputs$cellWeights,
          nknots      = inputs$nknots,
          verbose     = FALSE,
          parallel    = FALSE,
          sce         = TRUE
        ))
      },
      save = function(result, dir, ...) {
        out <- .tradeseq_summarize_fit(result)
        saveRDS(out, file.path(dir, "fitgam_output.rds"), compress = "xz")
      }
    ),
    on_activate   = .tradeseq_install_mgcv_overrides,
    on_deactivate = .tradeseq_restore_mgcv_overrides,
    # tested_against is parsed as "<pkg> <ver>" by the activation marker;
    # the mgcv 1.9.4 pin is enforced by the gam.fit4 textual-substitution
    # sentinel inside this file, so we only record the upstream here.
    tested_against = "tradeSeq 1.22.0",
    tested_upstream_versions = list(tradeSeq = "1.22.0")
  )
}
