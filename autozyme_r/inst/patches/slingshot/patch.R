# Patch for slingshot::getCurves.
#
# Lifted from autozyme task `test_slingshot`. Uses S4 method dispatch
# (signature = "PseudotimeOrdering") rather than namespace function patching
# — slingshot::getCurves is a generic with method-table dispatch.
#
# This is the first patch in autozyme to exercise the s4-method target
# kind in register_patch. The optimization parallelizes the per-lineage
# smoother + project_to_curve loop in getCurves' main while-loop using
# parallel::mclapply, plus caches smooth.spline's spar across iterations.

if (requireNamespace("slingshot",            quietly = TRUE) &&
    requireNamespace("SingleCellExperiment", quietly = TRUE) &&
    requireNamespace("S4Vectors",            quietly = TRUE) &&
    requireNamespace("SummarizedExperiment", quietly = TRUE) &&
    requireNamespace("TrajectoryUtils",      quietly = TRUE) &&
    requireNamespace("princurve",            quietly = TRUE) &&
    requireNamespace("matrixStats",          quietly = TRUE) &&
    requireNamespace("methods",              quietly = TRUE)) {

  # Pre-resolve hot lookups once to avoid namespace lookup per iter.
  .slingshot_proj_to_curve <- princurve::project_to_curve
  .slingshot_cw_means      <- matrixStats::colWeightedMeans
  .slingshot_row_maxs      <- matrixStats::rowMaxs
  .slingshot_row_mins      <- matrixStats::rowMins

  # Capture the original S4 method at file-scope so fast_getCurves can fall
  # back inside `with_disabled({...})` blocks. S4 patches don't go through
  # the namespace-fn dispatcher in .activate_one — the wrapper there can't
  # cleanly preserve generic dispatch — so we hand-honor is_disabled() here.
  .slingshot_orig_getCurves <- methods::getMethod(
    "getCurves", "PseudotimeOrdering",
    where = asNamespace("slingshot"), optional = TRUE
  )

  fast_getCurves <- function(data, shrink = TRUE, extend = "y", reweight = TRUE,
                             reassign = TRUE, thresh = 0.001, maxit = 15, stretch = 2,
                             approx_points = NULL, smoother = "smooth.spline",
                             shrink.method = "cosine", allow.breaks = TRUE, ...) {
    if (autozyme::is_disabled() && !is.null(.slingshot_orig_getCurves)) {
      return(.slingshot_orig_getCurves(
        data, shrink = shrink, extend = extend, reweight = reweight,
        reassign = reassign, thresh = thresh, maxit = maxit, stretch = stretch,
        approx_points = approx_points, smoother = smoother,
        shrink.method = shrink.method, allow.breaks = allow.breaks, ...
      ))
    }
    proj_to_curve <- .slingshot_proj_to_curve
    cw_means      <- .slingshot_cw_means
    row_maxs      <- .slingshot_row_maxs
    row_mins      <- .slingshot_row_mins

    pto <- data
    if (!all(c("lineages", "mst", "slingParams") %in%
             names(S4Vectors::metadata(pto)))) {
      stop("Lineage information is missing or incomplete. Either run ",
           "getLineages() first or use the slingshot() function.")
    }
    X <- slingshot::slingReducedDim(pto)
    clusterLabels <- slingshot::slingClusterLabels(pto)
    lineages <- slingshot::slingLineages(pto)

    assign_slingParams <- get(".slingParams<-", envir = asNamespace("slingshot"))
    pto <- assign_slingParams(pto, c(
      slingshot::slingParams(pto),
      shrink = shrink, extend = extend, reweight = reweight,
      reassign = reassign, thresh = thresh, maxit = maxit,
      stretch = stretch, approx_points = approx_points,
      smoother = smoother, shrink.method = shrink.method,
      allow.breaks = allow.breaks))

    shrink <- as.numeric(shrink)
    if (shrink < 0 | shrink > 1)
      stop("'shrink' parameter must be logical or numeric between 0 and 1")
    if (is.null(approx_points)) {
      approx_points <- if (nrow(X) > 150) 150 else FALSE
    }

    smootherFcn <- local({
      ss_cache <- new.env(parent = emptyenv())
      ss_cache$last_x    <- NULL
      ss_cache$last_w    <- NULL
      ss_cache$last_spar <- NULL
      switch(smoother,
        loess = function(lambda, xj, w = NULL, ...)
          loess(xj ~ lambda, weights = w, ...)$fitted,
        smooth.spline = function(lambda, xj, w = NULL, ..., df = 5,
                                  tol = 1e-4) {
          can_cache <- !is.null(ss_cache$last_x) &&
                       identical(ss_cache$last_x, lambda) &&
                       identical(ss_cache$last_w, w)
          if (can_cache) {
            fit <- tryCatch(
              stats::smooth.spline(lambda, xj, w = w, ...,
                                   spar = ss_cache$last_spar,
                                   tol = tol, keep.data = FALSE),
              error = function(e) NULL)
            if (!is.null(fit)) return(fit$y[match(lambda, fit$x)])
          }
          fit <- tryCatch(
            stats::smooth.spline(lambda, xj, w = w, ..., df = df, tol = tol,
                                  keep.data = FALSE),
            error = function(e)
              stats::smooth.spline(lambda, xj, w = w, ..., df = df,
                                    tol = tol, keep.data = FALSE, spar = 1))
          ss_cache$last_x    <- lambda
          ss_cache$last_w    <- w
          ss_cache$last_spar <- fit$spar
          fit$y[match(lambda, fit$x)]
        })
    })

    X.original <- X
    clusterLabels.original <- clusterLabels
    clusterLabels <- clusterLabels[, colnames(clusterLabels) != -1, drop = FALSE]
    L <- length(lineages)
    clusters <- colnames(clusterLabels)
    d <- dim(X); n <- d[1]; p <- d[2]
    nclus <- length(clusters)
    centers <- t(vapply(clusters, function(clID) {
      w <- clusterLabels[, clID]
      cw_means(X, w = w)
    }, rep(0, ncol(X))))
    if (p == 1) {
      centers <- t(centers)
      rownames(centers) <- clusters
    }
    rownames(centers) <- clusters
    W <- SummarizedExperiment::assay(pto, "weights")
    W.orig <- W
    D <- W; D[,] <- NA

    C <- as.matrix(vapply(lineages[seq_len(L)], function(lin)
      vapply(clusters, function(clID) as.numeric(clID %in% lin), 0),
      rep(0, nclus)))
    rownames(C) <- clusters
    segmnts <- unique(C[rowSums(C) > 1, , drop = FALSE])
    segmnts <- segmnts[order(rowSums(segmnts), decreasing = FALSE), , drop = FALSE]
    avg.order <- list()
    for (i in seq_len(nrow(segmnts))) {
      idx <- segmnts[i, ] == 1
      avg.order[[i]] <- colnames(segmnts)[idx]
      new.col <- rowMeans(segmnts[, idx, drop = FALSE])
      segmnts <- cbind(segmnts[, !idx, drop = FALSE], new.col)
      colnames(segmnts)[ncol(segmnts)] <- paste("average", i, sep = "")
    }

    pcurves <- list()
    resample_curve <- function(c, ap) {
      if (ap <= 0) return(c)
      xin <- c$lambda[c$ord]
      xout <- seq(min(xin), max(xin), length.out = ap)
      new_s <- vapply(seq_len(ncol(c$s)),
                      function(jj) approx(xin, c$s[c$ord, jj], xout,
                                          ties = "ordered")$y,
                      numeric(ap))
      list(s = new_s, ord = seq_len(ap), lambda = c$lambda,
           dist_ind = c$dist_ind)
    }
    for (l in seq_len(L)) {
      idx <- W[, l] > 0
      line.initial <- centers[clusters %in% lineages[[l]], , drop = FALSE]
      line.initial <- line.initial[match(lineages[[l]], rownames(line.initial)), , drop = FALSE]
      K <- nrow(line.initial)
      if (K == 1) {
        pca <- prcomp(X[idx, , drop = FALSE])
        ctr <- line.initial
        line.initial <- rbind(ctr - 10 * pca$sdev[1] * pca$rotation[, 1],
                              ctr,
                              ctr + 10 * pca$sdev[1] * pca$rotation[, 1])
        curve <- proj_to_curve(X[idx, , drop = FALSE], s = line.initial,
                                stretch = 9999)
        if (approx_points > 0) {
          curve_rs <- resample_curve(curve, approx_points)
          pcurve <- proj_to_curve(X, s = curve_rs$s, stretch = 0)
        } else {
          pcurve <- proj_to_curve(X, s = curve$s[curve$ord, , drop = FALSE], stretch = 0)
        }
        if (approx_points > 0) {
          xout_lambda <- seq(min(pcurve$lambda), max(pcurve$lambda),
                             length.out = approx_points)
          ord_pc <- pcurve$ord
          lambda_pc_ord <- pcurve$lambda[ord_pc]
          s_pc <- pcurve$s
          pcurve$s <- vapply(seq_len(p), function(jj)
            approx(x = lambda_pc_ord, y = s_pc[ord_pc, jj],
                   xout = xout_lambda, ties = "ordered")$y,
            numeric(approx_points))
          pcurve$ord <- seq_len(approx_points)
        }
        pcurve$dist_ind <- abs(pcurve$dist_ind)
        pcurve$lambda <- pcurve$lambda - min(pcurve$lambda, na.rm = TRUE)
        pcurve$w <- W[, l]
        pcurves[[l]] <- pcurve
        D[, l] <- abs(pcurve$dist_ind)
        next
      }

      if (extend == "y") {
        curve <- proj_to_curve(X[idx, , drop = FALSE], s = line.initial, stretch = 9999)
        curve$dist_ind <- abs(curve$dist_ind)
      } else if (extend == "n") {
        curve <- proj_to_curve(X[idx, , drop = FALSE], s = line.initial, stretch = 0)
        curve$dist_ind <- abs(curve$dist_ind)
      } else if (extend == "pc1") {
        cl1.idx <- clusterLabels[, lineages[[l]][1], drop = FALSE] > 0
        pc1.1 <- prcomp(X[cl1.idx, ])
        pc1.1 <- pc1.1$rotation[, 1] * pc1.1$sdev[1]^2
        leg1 <- line.initial[2, ] - line.initial[1, ]
        if (sum(pc1.1 * leg1) > 0) pc1.1 <- -pc1.1
        cl2.idx <- clusterLabels[, lineages[[l]][K], drop = FALSE] > 0
        pc1.2 <- prcomp(X[cl2.idx, ])
        pc1.2 <- pc1.2$rotation[, 1] * pc1.2$sdev[1]^2
        leg2 <- line.initial[K - 1, ] - line.initial[K, ]
        if (sum(pc1.2 * leg2) > 0) pc1.2 <- -pc1.2
        line.initial <- rbind(line.initial, line.initial[K, ] + pc1.2)
        line.initial <- rbind(line.initial[1, ] + pc1.1, line.initial)
        curve <- proj_to_curve(X[idx, , drop = FALSE], s = line.initial, stretch = 9999)
        curve$dist_ind <- abs(curve$dist_ind)
      }

      if (approx_points > 0) {
        curve_rs <- resample_curve(curve, approx_points)
        pcurve <- proj_to_curve(X, s = curve_rs$s, stretch = 0)
      } else {
        pcurve <- proj_to_curve(X, s = curve$s[curve$ord, , drop = FALSE], stretch = 0)
      }
      if (approx_points > 0) {
        xout_lambda <- seq(min(pcurve$lambda), max(pcurve$lambda),
                           length.out = approx_points)
        ord_pc <- pcurve$ord
        lambda_pc_ord <- pcurve$lambda[ord_pc]
        s_pc <- pcurve$s
        pcurve$s <- vapply(seq_len(p), function(jj)
          approx(x = lambda_pc_ord, y = s_pc[ord_pc, jj],
                 xout = xout_lambda, ties = "ordered")$y,
          numeric(approx_points))
        pcurve$ord <- seq_len(approx_points)
      }
      pcurve$dist_ind <- abs(pcurve$dist_ind)
      pcurve$lambda <- pcurve$lambda - min(pcurve$lambda, na.rm = TRUE)
      pcurve$w <- W[, l]
      pcurves[[l]] <- pcurve
      D[, l] <- abs(pcurve$dist_ind)
    }

    dist.new <- sum(abs(D[W > 0]), na.rm = TRUE)
    it <- 0
    hasConverged <- FALSE

    n_workers <- min(L, max(1L, autozyme::auto_threads(cap = 16L)))
    dots <- list(...)

    while (!hasConverged && it < maxit) {
      it <- it + 1
      dist.old <- dist.new

      if (reweight | reassign) {
        ordD <- order(D)
        W.prob <- W / rowSums(W)
        WrnkD <- cumsum(W.prob[ordD]) / sum(W.prob)
        Z <- D
        Z[ordD] <- WrnkD
      }
      if (reweight) {
        Z.prime <- 1 - Z^2
        Z.prime[W == 0] <- NA
        W0 <- W
        W <- Z.prime / row_maxs(Z.prime, na.rm = TRUE)
        W[is.nan(W)] <- 1
        W[is.na(W)]  <- 0
        W[W > 1] <- 1
        W[W < 0] <- 0
        W[W0 == 0] <- 0
      }
      if (reassign) {
        idx <- Z < .5
        W[idx] <- 1
        ridx <- row_maxs(Z, na.rm = TRUE) > .9 & row_mins(W, na.rm = TRUE) < .1
        W0 <- W[ridx, ]
        Z0 <- Z[ridx, ]
        W0[!is.na(Z0) & Z0 > .9 & W0 < .1] <- 0
        W[ridx, ] <- W0
      }

      dots_empty <- length(dots) == 0L
      new_pcurves <- .zyme_mclapply(seq_len(L), function(l) {
        pcurve <- pcurves[[l]]
        s <- pcurve$s
        pcurve_lambda <- pcurve$lambda
        pcurve_w      <- pcurve$w
        ordL <- order(pcurve_lambda)
        lambda_ord    <- pcurve_lambda[ordL]
        if (approx_points > 0) {
          xout_lambda <- seq(min(pcurve_lambda),
                             max(pcurve_lambda[which(pcurve_w > 0)]),
                             length.out = approx_points)
        }
        for (jj in seq_len(p)) {
          yjj <- if (dots_empty) {
            smootherFcn(pcurve_lambda, X[, jj], w = pcurve_w)[ordL]
          } else {
            do.call(smootherFcn,
                    c(list(pcurve_lambda, X[, jj], w = pcurve_w), dots))[ordL]
          }
          if (approx_points > 0) {
            yjj <- approx(x = lambda_ord, y = yjj, xout = xout_lambda,
                          ties = "ordered")$y
          }
          s[, jj] <- yjj
        }
        new.pcurve <- proj_to_curve(X, s = s, stretch = stretch)
        new.pcurve$lambda <- new.pcurve$lambda -
            min(new.pcurve$lambda[which(W[, l] > 0)])
        if (approx_points > 0) {
          xout_lambda <- seq(min(new.pcurve$lambda),
                             max(new.pcurve$lambda[which(W[, l] > 0)]),
                             length.out = approx_points)
          ord_pc       <- new.pcurve$ord
          lambda_pc_ord <- new.pcurve$lambda[ord_pc]
          s_pc          <- new.pcurve$s
          new.pcurve$s <- vapply(seq_len(p), function(jj)
            approx(x = lambda_pc_ord, y = s_pc[ord_pc, jj],
                   xout = xout_lambda, ties = "ordered")$y,
            numeric(approx_points))
          new.pcurve$ord <- seq_len(approx_points)
        }
        new.pcurve$dist_ind <- abs(new.pcurve$dist_ind)
        new.pcurve$w <- W[, l]
        new.pcurve
      }, mc.cores = n_workers, mc.preschedule = FALSE)

      if (any(vapply(new_pcurves, function(x) inherits(x, "try-error"), logical(1)))) {
        err_msg <- vapply(new_pcurves[vapply(new_pcurves, function(x)
          inherits(x, "try-error"), logical(1))], conditionMessage, character(1))[1]
        stop("getCurves: per-lineage worker errored: ", err_msg)
      }
      pcurves <- new_pcurves
      D[,] <- vapply(pcurves, function(p) p$dist_ind, rep(0, nrow(X)))

      if (shrink > 0) {
        if (max(rowSums(C)) > 1) {
          segmnts <- unique(C[rowSums(C) > 1, , drop = FALSE])
          segmnts <- segmnts[order(rowSums(segmnts), decreasing = FALSE), , drop = FALSE]
          seg.mix <- segmnts
          avg.lines  <- list()
          pct.shrink <- list()
          percent_shrink_fn <- get(".percent_shrinkage", envir = asNamespace("slingshot"))

          avg_curves_fn <- function(pcurves_in, Xx, stretch = 2, approx_points = FALSE) {
            n <- nrow(pcurves_in[[1]]$s)
            p <- ncol(pcurves_in[[1]]$s)
            max.shared.lambda <- min(vapply(pcurves_in, function(pcv) max(pcv$lambda), 0))
            lambdas.combine <- seq(0, max.shared.lambda, length.out = n)
            pcurves.dense <- lapply(pcurves_in, function(pcv) {
              if (approx_points > 0) {
                xin <- seq(min(pcv$lambda), max(pcv$lambda), length.out = approx_points)
              } else {
                xin <- pcv$lambda
              }
              ordv <- pcv$ord
              xin_ord <- xin[ordv]
              pcv_s <- pcv$s
              vapply(seq_len(p), function(jj)
                approx(xin_ord, pcv_s[ordv, jj], xout = lambdas.combine,
                       ties = "ordered")$y,
                numeric(n))
            })
            if (length(pcurves.dense) >= 2L) {
              arr <- simplify2array(pcurves.dense)
              avg <- rowMeans(arr, dims = 2)
            } else {
              avg <- pcurves.dense[[1]]
            }
            avg.curve <- proj_to_curve(Xx, avg, stretch = stretch)
            if (approx_points > 0) {
              xout_lambda <- seq(min(avg.curve$lambda), max(avg.curve$lambda),
                                 length.out = approx_points)
              ord_pc <- avg.curve$ord
              lambda_pc_ord <- avg.curve$lambda[ord_pc]
              s_pc <- avg.curve$s
              avg.curve$s <- vapply(seq_len(p), function(jj)
                approx(x = lambda_pc_ord, y = s_pc[ord_pc, jj],
                       xout = xout_lambda, ties = "ordered")$y,
                numeric(approx_points))
              avg.curve$ord <- seq_len(approx_points)
            }
            avg.curve$w <- rowSums(vapply(pcurves_in, function(p_) p_$w, rep(0, nrow(Xx))))
            avg.curve
          }

          shrink_to_avg_fn <- function(pcurve, avg.curve, pct, X,
                                        approx_points = FALSE, stretch = 2) {
            n <- nrow(pcurve$s)
            p_ <- ncol(pcurve$s)
            if (approx_points > 0) {
              lam   <- seq(min(pcurve$lambda),    max(pcurve$lambda),    length.out = approx_points)
              avlam <- seq(min(avg.curve$lambda), max(avg.curve$lambda), length.out = approx_points)
            } else {
              lam   <- pcurve$lambda
              avlam <- avg.curve$lambda
            }
            avg_s <- avg.curve$s
            pcv_s <- pcurve$s
            one_minus_pct <- 1 - pct
            s <- vapply(seq_len(p_), function(jj) {
              avg.jj <- approx(x = avlam, y = avg_s[, jj], xout = lam,
                               rule = 2, ties = mean)$y
              avg.jj * pct + pcv_s[, jj] * one_minus_pct
            }, numeric(n))
            w <- pcurve$w
            pcurve <- proj_to_curve(X, as.matrix(s[pcurve$ord, , drop = FALSE]), stretch = stretch)
            pcurve$w <- w
            if (approx_points > 0) {
              xout_lambda <- seq(min(pcurve$lambda), max(pcurve$lambda), length.out = approx_points)
              ord_pc <- pcurve$ord
              lambda_pc_ord <- pcurve$lambda[ord_pc]
              s_pc <- pcurve$s
              pcurve$s <- vapply(seq_len(p_), function(jj)
                approx(x = lambda_pc_ord, y = s_pc[ord_pc, jj],
                       xout = xout_lambda, ties = "ordered")$y,
                numeric(approx_points))
              pcurve$ord <- seq_len(approx_points)
            }
            pcurve
          }

          for (i in seq_along(avg.order)) {
            ns <- avg.order[[i]]
            to.avg <- lapply(ns, function(n) {
              if (grepl("Lineage", n))
                return(pcurves[[as.numeric(gsub("Lineage", "", n))]])
              if (grepl("average", n))
                return(avg.lines[[as.numeric(gsub("average", "", n))]])
            })
            avg <- avg_curves_fn(to.avg, X, stretch = stretch, approx_points = approx_points)
            avg.lines[[i]] <- avg
            common.ind <- rowMeans(vapply(to.avg, function(crv) crv$w > 0,
                                           rep(TRUE, nrow(X)))) == 1
            pct.shrink[[i]] <- lapply(to.avg, function(crv)
              percent_shrink_fn(crv, common.ind, approx_points = approx_points,
                                method = shrink.method))
            new.avg.order <- avg.order
            all.zero <- vapply(pct.shrink[[i]], function(pij) all(pij == 0), TRUE)
            if (any(all.zero)) {
              if (allow.breaks) {
                new.avg.order[[i]] <- NULL
                message("Curves for ", ns[1], " and ", ns[2],
                  " appear to be going in opposite directions. ",
                  "No longer forcing them to share an initial point. ",
                  "To manually override this, set allow.breaks = FALSE.")
              }
              pct.shrink[[i]] <- lapply(pct.shrink[[i]], function(pij) {
                pij[] <- 0; pij
              })
            }
          }

          for (j in rev(seq_along(avg.lines))) {
            ns <- avg.order[[j]]
            avg <- avg.lines[[j]]
            to.shrink <- lapply(ns, function(n) {
              if (grepl("Lineage", n))
                return(pcurves[[as.numeric(gsub("Lineage", "", n))]])
              if (grepl("average", n))
                return(avg.lines[[as.numeric(gsub("average", "", n))]])
            })
            shrunk <- lapply(seq_along(ns), function(jj) {
              crv <- to.shrink[[jj]]
              shrink_to_avg_fn(crv, avg, pct.shrink[[j]][[jj]] * shrink, X,
                                approx_points = approx_points, stretch = stretch)
            })
            for (jj in seq_along(ns)) {
              n <- ns[jj]
              if (grepl("Lineage", n))
                pcurves[[as.numeric(gsub("Lineage", "", n))]] <- shrunk[[jj]]
              if (grepl("average", n))
                avg.lines[[as.numeric(gsub("average", "", n))]] <- shrunk[[jj]]
            }
          }
          avg.order <- new.avg.order
        }
      }
      D[,] <- vapply(pcurves, function(p) p$dist_ind, rep(0, nrow(X)))
      dist.new <- sum(D[W > 0], na.rm = TRUE)
      hasConverged <- (abs(dist.old - dist.new) <= thresh * dist.old)
    }

    if (reweight | reassign) {
      ordD <- order(D)
      W.prob <- W / rowSums(W)
      WrnkD <- cumsum(W.prob[ordD]) / sum(W.prob)
      Z <- D
      Z[ordD] <- WrnkD
    }
    if (reweight) {
      Z.prime <- 1 - Z^2
      Z.prime[W == 0] <- NA
      W0 <- W
      W <- Z.prime / row_maxs(Z.prime, na.rm = TRUE)
      W[is.nan(W)] <- 1
      W[is.na(W)]  <- 0
      W[W > 1] <- 1
      W[W < 0] <- 0
      W[W0 == 0] <- 0
    }
    if (reassign) {
      idx <- Z < .5
      W[idx] <- 1
      ridx <- row_maxs(Z, na.rm = TRUE) > .9 & row_mins(W, na.rm = TRUE) < .1
      W0 <- W[ridx, ]
      Z0 <- Z[ridx, ]
      W0[!is.na(Z0) & Z0 > .9 & W0 < .1] <- 0
      W[ridx, ] <- W0
    }

    for (l in seq_len(L)) {
      class(pcurves[[l]]) <- "principal_curve"
      pcurves[[l]]$w <- W[, l]
    }
    names(pcurves) <- TrajectoryUtils::pathnames(pto)

    assign_slingCurves <- get(".slingCurves<-", envir = asNamespace("slingshot"))
    pto <- assign_slingCurves(pto, pcurves)

    pst <- vapply(pcurves, function(pc) {
      t <- pc$lambda; t[pc$w == 0] <- NA; t
    }, rep(0, nrow(X)))
    rownames(pst) <- rownames(pto)
    colnames(pst) <- colnames(pto)
    SummarizedExperiment::assay(pto, "pseudotime") <- pst

    scw <- vapply(pcurves, function(pc) pc$w, rep(0, nrow(X)))
    rownames(scw) <- rownames(pto)
    colnames(scw) <- colnames(pto)
    SummarizedExperiment::assay(pto, "weights") <- scw

    methods::validObject(pto)
    pto
  }

  register_patch(
    name = "slingshot",
    upstream = "slingshot",
    targets = list(
      getCurves = list(
        kind = "s4",
        signature = "PseudotimeOrdering",
        fn = fast_getCurves
      )
    ),
    smoke = list(
      load = function(task_dir, tier) {
        task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
        ds <- Filter(function(d) d$tier == tier, task$datasets)
        if (length(ds) == 0L) stop("no dataset for tier '", tier, "' in task.yaml")
        readRDS(resolve_dataset_path(task_dir, ds[[1]]$path))
      },
      call = function(sce) {
        slingshot::slingshot(sce, clusterLabels = "seurat_clusters", reducedDim = "PCA")
      },
      save = function(sce_out, dir, ...) {
        pto <- SummarizedExperiment::colData(sce_out)$slingshot
        pst      <- slingshot::slingPseudotime(pto)
        weights  <- slingshot::slingCurveWeights(pto)
        mst_adj  <- as.matrix(igraph::as_adjacency_matrix(slingshot::slingMST(pto)))
        lineages <- slingshot::slingLineages(pto)
        saveRDS(pst,      file.path(dir, "pseudotime.rds"))
        saveRDS(weights,  file.path(dir, "weights.rds"))
        saveRDS(mst_adj,  file.path(dir, "mst_adj.rds"))
        saveRDS(lineages, file.path(dir, "lineages.rds"))
      }
    ),
    tested_against = "slingshot 2.16.0",
    tested_upstream_versions = list(slingshot = "2.16.0")
  )
}
