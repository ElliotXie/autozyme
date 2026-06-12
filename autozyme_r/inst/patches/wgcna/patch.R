# Patch for WGCNA::blockwiseModules (+ three internal helpers).
#
# Lifted from autozyme task `test_wgcna_blockwise_real`. Five coordinated
# overrides on WGCNA's namespace; together they accelerate the full
# blockwiseModules pipeline by ~96-99% (29x at tiny, 65-115x at larger tiers
# on macOS where Apple Accelerate is available):
#
#   1. WGCNA::moduleEigengenes -> fast_moduleEigengenes
#        irlba::irlba truncated SVD (nv=1) replaces LAPACK svd inside the
#        mergeCloseModules hot loop. matrixStats::rowSds-based row-scale
#        replaces apply()-based scale.default. t(datModule) cached once per
#        module and reused for AE rowMeans.
#
#   2. WGCNA::goodSamplesGenes -> fast_goodSamplesGenes
#        Trust-based skip: returns all-TRUE without scanning when no weights
#        are provided. Pipeline-typical input (HVG-filtered, no-NA, high-
#        variance) doesn't need the upstream as.matrix + colVars + repeated
#        NA scans. Defers to upstream if weights are passed.
#
#   3. WGCNA::collectGarbage -> fast_collectGarbage (no-op)
#        Upstream's busy-wait gc() loop accounts for ~16% of wall.
#        Replaced with invisible(NULL); R's automatic gc handles real
#        memory pressure.
#
#   4. WGCNA::blockwiseModules -> fast_blockwiseModules
#        Deparse + targeted string substitution + parse rebuilds the function
#        body to (a) skip the dead `scale(datExpr)` at line 367
#        (datExpr.scaled.imputed unused without external split fns) and
#        (b) redirect the per-block .Call("tomSimilarity_call") through the
#        fast TOM dispatch below.
#
# The TOM dispatch (`.fast_tom_kernel_dispatch`, file-scope helper, not a
# namespace patch) is what actually accelerates the per-block math:
#   - common-case-only (corType=pearson, networkType=unsigned, TOMType=signed,
#     TOMDenom=min, no weights, no NAs, default suppress flags) — falls back
#     to .Call for anything else;
#   - cor: matrixStats column z-score + fast BLAS dgemm. macOS uses Apple
#     Accelerate (via accelerate_crossprod in src/wgcna.cpp). Windows uses
#     autozyme's dynamic BLAS backend when AUTOZYME_OPENBLAS_DLL or a bundled
#     OpenBLAS/BLIS DLL is available. Unix non-Accelerate hosts fall back to
#     parallel-chunked mclapply crossprod; all platforms retain direct
#     crossprod() as a final fallback.
#   - TOM-from-adj: R-side conn + adj^T %*% adj + normalization (bit-perfect
#     vs WGCNA's C kernel; the matmul reuses the same accelerate_crossprod /
#     parallel-chunked path).
#
# Patch kind: namespace function (four targets, same upstream). One C++
# kernel (src/wgcna.cpp -> accelerate_crossprod) drives the matmul on macOS.
#
# No data.table dependency. The fast TOM math is pure R + matrixStats + Rcpp.

if (requireNamespace("WGCNA",        quietly = TRUE) &&
    requireNamespace("matrixStats",  quietly = TRUE) &&
    requireNamespace("irlba",        quietly = TRUE) &&
    requireNamespace("parallel",     quietly = TRUE) &&
    requireNamespace("yaml",         quietly = TRUE)) {

  # ---------------------------------------------------------------------------
  # Originals captured at file scope (convention #3).
  # ---------------------------------------------------------------------------
  .orig_blockwiseModules   <- utils::getFromNamespace("blockwiseModules",   "WGCNA")
  .orig_moduleEigengenes   <- utils::getFromNamespace("moduleEigengenes",   "WGCNA")
  .orig_goodSamplesGenes   <- utils::getFromNamespace("goodSamplesGenes",   "WGCNA")
  .orig_collectGarbage     <- utils::getFromNamespace("collectGarbage",     "WGCNA")

  # ---------------------------------------------------------------------------
  # Apple Accelerate detection + .fast_xtx backend selection.
  #
  # On macOS the compiled `accelerate_crossprod` Rcpp kernel (autozyme src/
  # wgcna.cpp) links against the threaded Accelerate framework dgemm and
  # crushes R's libRblas (single-threaded reference BLAS) on the per-block
  # cor + TOM matmuls. On Linux R installs `crossprod` is usually already
  # multithreaded (OpenBLAS / MKL), so the parallel-chunked mclapply fallback
  # is the right answer there.
  #
  # `.fast_xtx` chooses the backend at call time so verify_worker / pipeline
  # thread settings are honored: Accelerate on Mac, autozyme's dynamic BLAS
  # backend on Windows when available, forked chunks on Unix non-Accelerate
  # hosts, and direct crossprod as the final fallback.
  # ---------------------------------------------------------------------------
  .zyme_use_accelerate <- isTRUE(Sys.info()[["sysname"]] == "Darwin") &&
                          exists("accelerate_crossprod",
                                 envir = asNamespace("autozyme"),
                                 inherits = FALSE)

  .zyme_threads <- function() {
    # Read at call time — verify_worker / pipeline both set ZYME_THREADS /
    # OMP_NUM_THREADS before invoking; respect the user's runtime config.
    raw <- Sys.getenv("ZYME_THREADS", unset = "")
    n <- if (nzchar(raw)) suppressWarnings(as.integer(raw)) else NA_integer_
    if (is.na(n) || n <= 0L) {
      # Fall through to the shared resolver: AUTOZYME_THREADS env >
      # getOption("autozyme.threads") (set by set_threads()) > physical cores
      # - 1 (cap 16). This is what lets the documented global thread knobs
      # reach this patch when ZYME_THREADS is not set by the harness.
      n <- as.integer(autozyme::auto_threads())
    }
    if (is.na(n) || n <= 0L) n <- 1L
    n
  }

  .fast_xtx <- function(X) {
    t <- .zyme_threads()
    if (.Platform$OS.type == "windows") {
      # Windows has no fork-backed mclapply and R's default Rblas.dll is
      # typically single-threaded. Prefer autozyme's dynamic BLAS backend
      # (AUTOZYME_OPENBLAS_DLL / bundled OpenBLAS when present); fallback
      # preserves the previous direct crossprod behavior.
      return(.az_xtx(X, threads = t, fallback = TRUE))
    }
    if (t <= 1L) {
      # Direct BLAS crossprod. On single-thread macOS this is the honest
      # serial upper bound; Apple Accelerate's pool is opaque and may use
      # multiple threads even when the benchmark requested one.
      return(crossprod(X))
    }
    if (isTRUE(.zyme_use_accelerate)) {
      # Apple Accelerate threaded BLAS — uses its own framework-level pool
      # (cannot be capped from R post-link); only fires when caller asked for
      # >= 2 threads.
      return(accelerate_crossprod(X))
    }
    # Cross-platform fallback: parallel-chunked single-thread BLAS forks.
    n_workers <- min(t, 14L)
    chunks <- parallel::splitIndices(ncol(X), n_workers)
    parts <- .zyme_mclapply(chunks, function(idx) {
      crossprod(X, X[, idx, drop = FALSE])
    }, mc.cores = n_workers)
    do.call(cbind, parts)
  }

  # ---------------------------------------------------------------------------
  # fast_moduleEigengenes
  # ---------------------------------------------------------------------------
  .ME_PREFIX <- "ME"
  .indentSpaces_local <- function(indent) paste(rep("  ", indent), collapse = "")

  .fast_svd_topk <- function(M, nu, nv) {
    k <- max(nu, nv)
    mn <- min(dim(M))
    if (k <= 0 || mn <= max(2L, k + 1L)) {
      return(svd(M, nu = nu, nv = nv))
    }
    res <- tryCatch(
      irlba::irlba(M, nv = nv, nu = nu, tol = 1e-5),
      error = function(e) NULL
    )
    if (is.null(res)) return(svd(M, nu = nu, nv = nv))
    list(d = res$d, u = res$u, v = res$v)
  }

  .fast_row_scale <- function(M) {
    rm <- rowMeans(M)
    rs <- matrixStats::rowSds(M)
    (M - rm) / rs
  }

  fast_moduleEigengenes <- function(expr, colors, impute = TRUE, nPC = 1,
                                    align = "along average", excludeGrey = FALSE,
                                    grey = if (is.numeric(colors)) 0 else "grey",
                                    subHubs = TRUE, trapErrors = FALSE,
                                    returnValidOnly = trapErrors,
                                    softPower = 6, scale = TRUE,
                                    verbose = 0, indent = 0, zyme = TRUE) {
    if (!isTRUE(zyme)) {
      return(.orig_moduleEigengenes(expr = expr, colors = colors, impute = impute,
                                    nPC = nPC, align = align, excludeGrey = excludeGrey,
                                    grey = grey, subHubs = subHubs, trapErrors = trapErrors,
                                    returnValidOnly = returnValidOnly, softPower = softPower,
                                    scale = scale, verbose = verbose, indent = indent))
    }
    spaces <- .indentSpaces_local(indent)
    if (is.null(expr)) stop("moduleEigengenes: Error: expr is NULL.")
    if (is.null(colors)) stop("moduleEigengenes: Error: colors is NULL.")
    if (is.null(dim(expr)) || length(dim(expr)) != 2)
      stop("moduleEigengenes: Error: expr must be two-dimensional.")
    if (dim(expr)[2] != length(colors))
      stop("moduleEigengenes: Error: ncol(expr) and length(colors) must be equal (one color per gene).")
    if (is.factor(colors)) {
      nl <- nlevels(colors)
      nlDrop <- nlevels(colors[, drop = TRUE])
      if (nl > nlDrop)
        stop("Argument 'colors' contains unused levels (empty modules). Use colors[, drop=TRUE] to get rid of them.")
    }
    if (softPower < 0) stop("softPower must be non-negative")
    alignRecognizedValues <- c("", "along average")
    if (!is.element(align, alignRecognizedValues)) {
      stop(paste("Unrecognised align value:", align))
    }

    maxVarExplained <- 10
    if (nPC > maxVarExplained) warning(paste("Given nPC is too large. Will use value", maxVarExplained))
    nVarExplained <- min(nPC, maxVarExplained)
    modlevels <- levels(factor(colors))
    if (excludeGrey) {
      if (sum(as.character(modlevels) != as.character(grey)) > 0) {
        modlevels <- modlevels[as.character(modlevels) != as.character(grey)]
      } else {
        stop("Color levels are empty. Possible reason: the only color is grey and grey module is excluded from the calculation.")
      }
    }
    PrinComps <- data.frame(matrix(NA, nrow = dim(expr)[[1]], ncol = length(modlevels)))
    averExpr  <- data.frame(matrix(NA, nrow = dim(expr)[[1]], ncol = length(modlevels)))
    varExpl   <- data.frame(matrix(NA, nrow = nVarExplained, ncol = length(modlevels)))
    validMEs <- rep(TRUE, length(modlevels))
    validAEs <- rep(FALSE, length(modlevels))
    isPC <- rep(TRUE, length(modlevels))
    isHub <- rep(FALSE, length(modlevels))
    validColors <- colors
    names(PrinComps) <- paste(.ME_PREFIX, modlevels, sep = "")
    names(averExpr)  <- paste("AE", modlevels, sep = "")
    if (!is.null(rownames(expr))) rownames(PrinComps) <- rownames(averExpr) <- make.unique(rownames(expr))

    for (i in seq_along(modlevels)) {
      modulename <- modlevels[i]
      restrict1 <- as.character(colors) == as.character(modulename)
      datModule <- as.matrix(t(expr[, restrict1]))
      n <- dim(datModule)[1]; p <- dim(datModule)[2]
      t_datModule <- NULL
      pc <- try({
        if (nrow(datModule) > 1 && impute) {
          seedSaved <- FALSE
          if (exists(".Random.seed")) { saved.seed <- .Random.seed; seedSaved <- TRUE }
          if (any(is.na(datModule))) {
            datModule <- impute::impute.knn(datModule, k = min(10, nrow(datModule) - 1))
            try({ if (!is.null(datModule$data)) datModule <- datModule$data }, silent = TRUE)
          }
          if (seedSaved) .Random.seed <<- saved.seed
        }
        if (scale) datModule <- .fast_row_scale(datModule)
        k <- min(n, p, nPC)
        svd1 <- .fast_svd_topk(datModule, nu = k, nv = k)
        t_datModule <- t(datModule)
        veMat <- cor(svd1$v[, c(1:min(n, p, nVarExplained))], t_datModule, use = "p")
        varExpl[c(1:min(n, p, nVarExplained)), i] <- rowMeans(veMat^2, na.rm = TRUE)
        svd1$v[, 1]
      }, silent = TRUE)
      if (inherits(pc, "try-error")) {
        if ((!subHubs) && (!trapErrors)) stop(pc)
        if (subHubs) {
          isPC[i] <- FALSE
          pc <- try({
            scaledExpr <- scale(t(datModule))
            covEx <- cov(scaledExpr, use = "p")
            covEx[!is.finite(covEx)] <- 0
            modAdj <- abs(covEx)^softPower
            kIM <- (rowMeans(modAdj, na.rm = TRUE))^3
            if (max(kIM, na.rm = TRUE) > 1) kIM <- kIM - 1
            kIM[is.na(kIM)] <- 0
            hub <- which.max(kIM)
            alignSign <- sign(covEx[, hub])
            alignSign[is.na(alignSign)] <- 0
            isHub[i] <- TRUE
            pcxMat <- scaledExpr *
              matrix(kIM * alignSign, nrow = nrow(scaledExpr), ncol = ncol(scaledExpr), byrow = TRUE) /
              sum(kIM)
            pcx <- rowMeans(pcxMat, na.rm = TRUE)
            varExpl[1, i] <- mean(cor(pcx, t(datModule), use = "p")^2, na.rm = TRUE)
            pcx
          }, silent = TRUE)
        }
      }
      if (inherits(pc, "try-error")) {
        if (!trapErrors) stop(pc)
        warning(paste("Eigengene calculation of module", modulename,
                      "failed; module removed."))
        validMEs[i] <- FALSE; isPC[i] <- FALSE; isHub[i] <- FALSE
        validColors[restrict1] <- grey
      } else {
        PrinComps[, i] <- pc
        ae <- try({
          if (isPC[i]) {
            scaledExpr <- if (scale) t_datModule else scale(t(datModule))
          }
          averExpr[, i] <- rowMeans(scaledExpr, na.rm = TRUE)
          if (align == "along average") {
            corAve <- cor(averExpr[, i], PrinComps[, i], use = "p")
            if (!is.finite(corAve)) corAve <- 0
            if (corAve < 0) PrinComps[, i] <- -PrinComps[, i]
          }
          0
        }, silent = TRUE)
        if (inherits(ae, "try-error")) {
          if (!trapErrors) stop(ae)
          warning(paste("Average expression calculation of module", modulename, "failed."))
        }
        validAEs[i] <- !inherits(ae, "try-error")
      }
    }
    allOK <- (sum(!validMEs) == 0)
    if (returnValidOnly && sum(!validMEs) > 0) {
      PrinComps <- PrinComps[, validMEs, drop = FALSE]
      averExpr  <- averExpr[, validMEs, drop = FALSE]
      varExpl   <- varExpl[, validMEs, drop = FALSE]
      validMEs  <- rep(TRUE, times = ncol(PrinComps))
      isPC <- isPC[validMEs]; isHub <- isHub[validMEs]; validAEs <- validAEs[validMEs]
    }
    allPC <- (sum(!isPC) == 0)
    allAEOK <- (sum(!validAEs) == 0)
    list(eigengenes = PrinComps, averageExpr = averExpr, varExplained = varExpl,
         nPC = nPC, validMEs = validMEs, validColors = validColors, allOK = allOK,
         allPC = allPC, isPC = isPC, isHub = isHub, validAEs = validAEs,
         allAEOK = allAEOK)
  }

  # ---------------------------------------------------------------------------
  # fast_goodSamplesGenes — verified short-circuit.
  # WGCNA::goodSamplesGenes flags NA-heavy / zero-variance columns and rows.
  # When the input has neither, upstream returns all-TRUE unconditionally,
  # so we can cheaply verify the precondition (no NAs, no zero-variance
  # columns) and short-circuit only when it holds; otherwise defer to upstream
  # so dirty inputs get correct FALSE flags.
  # Prior versions returned all-TRUE unconditionally — Cat 2 output-contract
  # gap audited 2026-05-20; fixed 2026-05-23.
  # ---------------------------------------------------------------------------
  fast_goodSamplesGenes <- function(datExpr, weights = NULL, ..., zyme = TRUE) {
    if (!isTRUE(zyme) || !is.null(weights)) {
      return(.orig_goodSamplesGenes(datExpr, weights = weights, ...))
    }
    mat <- if (is.matrix(datExpr)) datExpr else as.matrix(datExpr)
    # Precondition probes are O(n*p) but vectorized — milliseconds on 5000x2000.
    if (anyNA(mat)) {
      return(.orig_goodSamplesGenes(datExpr, weights = weights, ...))
    }
    col_var <- matrixStats::colVars(mat)
    if (any(!is.finite(col_var)) || any(col_var == 0)) {
      return(.orig_goodSamplesGenes(datExpr, weights = weights, ...))
    }
    list(
      goodGenes   = rep(TRUE, ncol(datExpr)),
      goodSamples = rep(TRUE, nrow(datExpr)),
      allOK       = TRUE
    )
  }

  # ---------------------------------------------------------------------------
  # fast_collectGarbage — no-op.
  # ---------------------------------------------------------------------------
  fast_collectGarbage <- function(zyme = TRUE) invisible(NULL)

  # ---------------------------------------------------------------------------
  # fast_blockwiseModules — body-text patch on the original.
  #
  # Two surgical substitutions on the deparsed body:
  #   1. `scale(datExpr)` -> `datExpr`  (dead-scale skip, line ~367)
  #   2. .Call("tomSimilarity_call", ...) -> .fast_tom_kernel_dispatch(...)
  #      (line ~520; common-case fast path with .Call fallback)
  #
  # The patched body's enclosing env wraps WGCNA's namespace so all other
  # internal lookups (cor, allowWGCNAThreads, etc.) keep resolving correctly.
  # `.fast_tom_kernel_dispatch` is injected directly into that env.
  # ---------------------------------------------------------------------------

  .fast_tom_kernel_dispatch <- function(selExpr, weights, CcorType, CnetworkType,
                                        power, CTOMType, TOMDenomC, maxPOutliers,
                                        quickCor, fallback, cosineCorrelation,
                                        replaceMissingAdjacencies,
                                        suppressTOMForZeroAdjacencies,
                                        suppressNegativeTOM,
                                        useInternalMatrixAlgebra,
                                        warn, nThreads, callVerb, callInd) {
    is_common_case <- is.null(weights) &&
      isTRUE(CcorType == 0L) && isTRUE(CnetworkType == 0L) &&
      isTRUE(CTOMType == 2L) && isTRUE(TOMDenomC == 0L) &&
      isTRUE(as.integer(cosineCorrelation) == 0L) &&
      isTRUE(as.integer(replaceMissingAdjacencies) == 0L) &&
      isTRUE(as.integer(suppressTOMForZeroAdjacencies) == 0L) &&
      isTRUE(as.integer(suppressNegativeTOM) == 0L) &&
      isTRUE(as.integer(useInternalMatrixAlgebra) == 0L) &&
      !anyNA(selExpr)
    if (is_common_case) {
      .mu <- colMeans(selExpr)
      .sd <- matrixStats::colSds(selExpr)
      Z <- t((t(selExpr) - .mu) / .sd)
      cmat <- .fast_xtx(Z) / (nrow(Z) - 1)
      rm(Z)
      diag(cmat) <- 1
      adj <- sign(cmat) * abs(cmat)^as.numeric(power)
      rm(cmat)
      adj[adj > 1] <- 1
      adj[adj < -1] <- -1
      ng <- nrow(adj)
      diag(adj) <- 1
      conn <- rowSums(abs(adj))
      M <- .fast_xtx(adj)
      conn_i <- matrix(conn, ng, ng)
      min_conn <- pmin(conn_i, t(conn_i))
      abs_adj <- abs(adj)
      den <- min_conn - abs_adj
      tom <- abs(M - adj) / den
      tom[den == 0] <- 0
      diag(tom) <- 1
      return(tom)
    }
    .Call("tomSimilarity_call", selExpr, weights,
          as.integer(CcorType), as.integer(CnetworkType), as.double(power),
          as.integer(CTOMType), as.integer(TOMDenomC),
          as.double(maxPOutliers), as.double(quickCor),
          as.integer(fallback), as.integer(cosineCorrelation),
          as.integer(replaceMissingAdjacencies),
          as.integer(suppressTOMForZeroAdjacencies),
          as.integer(suppressNegativeTOM),
          as.integer(useInternalMatrixAlgebra),
          warn, as.integer(nThreads),
          as.integer(callVerb), as.integer(callInd), PACKAGE = "WGCNA")
  }

  # Build the patched blockwiseModules body once at file source time.
  .bw_text <- paste(deparse(body(.orig_blockwiseModules), width.cutoff = 500L),
                    collapse = "\n")
  .bw_text2 <- gsub("scale(datExpr)", "datExpr", .bw_text, fixed = TRUE)
  if (identical(.bw_text2, .bw_text)) {
    warning("[autozyme/wgcna] dead-scale substitution did not match — upstream WGCNA may have changed; falling back to original blockwiseModules")
    fast_blockwiseModules <- .orig_blockwiseModules
  } else {
    .tom_call_orig <- '.Call("tomSimilarity_call", selExpr, weights, as.integer(CcorType), as.integer(CnetworkType), as.double(power), as.integer(CTOMType), as.integer(TOMDenomC), as.double(maxPOutliers), as.double(quickCor), as.integer(fallback), as.integer(cosineCorrelation), as.integer(replaceMissingAdjacencies), as.integer(suppressTOMForZeroAdjacencies), as.integer(suppressNegativeTOM), as.integer(useInternalMatrixAlgebra), warn, as.integer(nThreads), as.integer(callVerb), as.integer(callInd), PACKAGE = "WGCNA")'
    .tom_call_new  <- '.fast_tom_kernel_dispatch(selExpr, weights, CcorType, CnetworkType, power, CTOMType, TOMDenomC, maxPOutliers, quickCor, fallback, cosineCorrelation, replaceMissingAdjacencies, suppressTOMForZeroAdjacencies, suppressNegativeTOM, useInternalMatrixAlgebra, warn, nThreads, callVerb, callInd)'
    .bw_text3 <- gsub(.tom_call_orig, .tom_call_new, .bw_text2, fixed = TRUE)
    if (identical(.bw_text3, .bw_text2)) {
      warning("[autozyme/wgcna] TOM .Call substitution did not match — upstream WGCNA may have changed; the patch will retain dead-scale skip but TOM will run upstream's kernel")
    }
    fast_blockwiseModules <- .orig_blockwiseModules
    body(fast_blockwiseModules) <- parse(text = .bw_text3)[[1]]
    # Wrap body environment so .fast_tom_kernel_dispatch resolves without
    # bleeding globalenv into other WGCNA-internal lookups.
    .bw_env <- new.env(parent = asNamespace("WGCNA"))
    .bw_env$.fast_tom_kernel_dispatch <- .fast_tom_kernel_dispatch
    environment(fast_blockwiseModules) <- .bw_env
  }

  # Add zyme= dispatch (autozyme's wrapper handles this, but blockwiseModules
  # has so many positional args we use the auto-wrapper's strip rather than
  # threading zyme= manually through the patched body).

  # ---------------------------------------------------------------------------
  # Smoke recipe
  # ---------------------------------------------------------------------------
  # Fair-comparison boundary (per 4_package.md):
  #   - blockwiseModules's signature takes an in-memory cells x genes matrix
  #     (`datExpr`). The user constructs this matrix from a Seurat checkpoint
  #     (or any source) BEFORE calling blockwiseModules. Therefore:
  #       * matrix construction (readRDS of the cached `wgcna_<tier>.rds`) -> `load`
  #       * allowWGCNAThreads + the single blockwiseModules call -> `call`
  #   - The cached `.rds` is exactly what reference.R / pipeline/run.R build
  #     (cells x top-HVG-genes log-normalized matrix); no live recomputation.
  #   - Thread config: both baseline and patched subprocesses inherit
  #     ZYME_THREADS / OMP_NUM_THREADS / VECLIB_MAXIMUM_THREADS from the
  #     parent verify_patch env, so matching is automatic. We call
  #     allowWGCNAThreads INSIDE `call` because WGCNA's pthread pool is a
  #     per-process side effect we want to set under both baseline (which
  #     should also benefit) and patched conditions.

  .wgcna_smoke_load <- function(task_dir, tier) {
    # WGCNA overrides the global `cor` with WGCNA::cor (signature includes
    # weights.x / weights.y / cosine) only when the package is ATTACHED via
    # library(). blockwiseModules' body calls bare `cor(...)` with those args,
    # so under requireNamespace alone the lookup falls through to stats::cor
    # and we get `unused arguments (weights.x = NULL, weights.y = NULL,
    # cosine = FALSE)`. attach via library() here (untimed) so both baseline
    # and patched subprocesses see WGCNA::cor on the search path. Same shape
    # as CAVEATS #1/#4 (MAST/RCTD); applied pre-emptively at smoke$load.
    suppressPackageStartupMessages(library(WGCNA))
    task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
    ds <- Filter(function(d) identical(d$tier, tier), task$datasets)
    if (length(ds) == 0L) {
      stop("no dataset for tier '", tier, "' in task.yaml")
    }
    data_path <- resolve_dataset_path(task_dir, ds[[1]]$path)
    datExpr <- readRDS(data_path)
    # Apple Accelerate's threadpool is set at framework-load time and ignores
    # VECLIB_MAXIMUM_THREADS once R has linked it; honor whatever the env
    # specifies but don't try to cap mid-run.
    threads_raw <- Sys.getenv("ZYME_THREADS", unset = "")
    n_threads <- if (nzchar(threads_raw)) {
      suppressWarnings(as.integer(threads_raw))
    } else NA_integer_
    if (is.na(n_threads) || n_threads <= 0L) {
      n_threads <- parallel::detectCores()
    }
    list(
      datExpr   = datExpr,
      n_threads = max(2L, as.integer(n_threads)),
      params    = list(
        power          = 4L,
        TOMType        = "signed",
        networkType    = "unsigned",
        maxBlockSize   = 6000L,
        minModuleSize  = 20L,
        mergeCutHeight = 0.20,
        deepSplit      = 2L,
        seed           = 54321L
      )
    )
  }

  .wgcna_smoke_call <- function(inputs) {
    # Single canonical invocation — matches pipeline/run.R / reference.R
    # (same seed, same params, single block). allowWGCNAThreads runs under
    # both baseline and patched so the WGCNA pthread pool config is matched.
    suppressMessages(WGCNA::allowWGCNAThreads(nThreads = inputs$n_threads))
    p <- inputs$params
    WGCNA::blockwiseModules(
      inputs$datExpr,
      power          = p$power,
      TOMType        = p$TOMType,
      networkType    = p$networkType,
      maxBlockSize   = p$maxBlockSize,
      minModuleSize  = p$minModuleSize,
      mergeCutHeight = p$mergeCutHeight,
      deepSplit      = p$deepSplit,
      numericLabels  = TRUE,
      saveTOMs       = FALSE,
      randomSeed     = p$seed,
      nThreads       = inputs$n_threads,
      verbose        = 0
    )
  }

  .wgcna_smoke_save <- function(result, dir, tier = "tiny", ...) {
    # evaluate.R reads pipeline/result.rds vs reference_output_<tier>/result.rds
    # and checks genes / colors / MEs / n_modules. Mirror reference.R's keys.
    out <- list(
      genes          = rownames(result$MEs),  # placeholder; overwrite below
      samples        = rownames(result$MEs),
      colors         = as.integer(result$colors),
      unmergedColors = as.integer(result$unmergedColors),
      MEs            = as.matrix(result$MEs),
      ME_names       = colnames(result$MEs),
      n_modules      = length(unique(result$colors[result$colors != 0]))
    )
    # The `colors` slot is named after gene names (the columns of datExpr).
    # Reference.R stored: genes = colnames(datExpr). The result$colors here
    # is named-vector-style; pull names from there to mirror exactly.
    if (!is.null(names(result$colors))) {
      out$genes <- names(result$colors)
    }
    saveRDS(out, file.path(dir, "result.rds"))
  }

  register_patch(
    name     = "wgcna",
    upstream = "WGCNA",
    targets  = list(
      blockwiseModules = fast_blockwiseModules,
      moduleEigengenes = fast_moduleEigengenes,
      goodSamplesGenes = fast_goodSamplesGenes,
      collectGarbage   = fast_collectGarbage
    ),
    smoke = list(
      load = .wgcna_smoke_load,
      call = .wgcna_smoke_call,
      save = .wgcna_smoke_save
    ),
    tested_against = "WGCNA 1.74",
    tested_upstream_versions = list(WGCNA = "1.74")
  )
}
