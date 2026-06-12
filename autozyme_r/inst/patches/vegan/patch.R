# Patch for vegan::adonis2 (PERMANOVA via dbRDA).
#
# Lifted from autozyme task `test_vegan_adonis2`. Three coordinated overrides
# on vegan's namespace, all targeting the by="terms" dbRDA permutation path:
#
#   1. vegan::adonis0
#        -> fast_adonis0: when there are no conditioning variables, build the
#           dbRDA fit directly from the doubly-centered Gram (initDBRDA) +
#           a single qr() on the centered model matrix. Trace of the Gram
#           gives total inertia; .colSums on Q^T G Q gives fit inertia.
#           Skips the full ordi*-style residual eigen path.
#
#   2. vegan::adonis2
#        -> fast_adonis2: when the LHS is a count matrix and method="bray",
#           swap vegan's serial Bray-Curtis for parallelDist::parDist
#           (multi-threaded). Falls back to upstream for non-bray methods,
#           extra `...` args (preserves upstream semantics), or when
#           parallelDist is not installed. Threads come from
#           getOption("autozyme.threads", parallel::detectCores(logical=FALSE)).
#
#   3. vegan::permutest.cca
#        -> fast_permutest_cca: the by="terms", model="reduced", dbRDA-only,
#           non-partial, non-CCA path. Builds the permuted-Q block in bounded
#           permutation batches (Qbatch: N x (max_rank * batch_perms)), fanning
#           the trace reduction E %*% Qbatch across forked mclapply workers
#           (Unix only) and reusing the buffer across batches, then collapses
#           per-perm term traces from cumulative .colSums output. Bounds peak
#           memory to O(batch_perms) instead of O(nperm). Bypasses
#           upstream's per-permutation qr.fitted loop entirely. Falls back
#           to orig_permutest_cca for any unsupported branch (model != reduced,
#           by != terms, first=TRUE, partial, classical-CCA, non-dbrda).
#
# Patch kind: namespace function (three targets, same upstream). No S4, no
# C++ — pure R + parallelDist + mclapply.

if (requireNamespace("vegan",        quietly = TRUE) &&
    requireNamespace("parallel",     quietly = TRUE) &&
    requireNamespace("parallelDist", quietly = TRUE) &&
    requireNamespace("permute",      quietly = TRUE) &&
    requireNamespace("yaml",         quietly = TRUE)) {

  # Originals + internal helpers captured at file scope via getFromNamespace.
  # Fast fns reference these via lexical closure — see CAVEATS.md / convention #3.
  .orig_adonis0       <- utils::getFromNamespace("adonis0",       "vegan")
  .orig_adonis2       <- utils::getFromNamespace("adonis2",       "vegan")
  .orig_permutest_cca <- utils::getFromNamespace("permutest.cca", "vegan")

  .ordiYbar_vegan           <- utils::getFromNamespace("ordiYbar",           "vegan")
  .getPermuteMatrix_vegan   <- utils::getFromNamespace("getPermuteMatrix",   "vegan")
  .initDBRDA_vegan          <- utils::getFromNamespace("initDBRDA",          "vegan")
  .ordiParseFormula_vegan   <- utils::getFromNamespace("ordiParseFormula",   "vegan")
  .ordiTerminfo_vegan       <- utils::getFromNamespace("ordiTerminfo",       "vegan")
  .addLingoes_vegan         <- utils::getFromNamespace("addLingoes",         "vegan")
  .addCailliez_vegan        <- utils::getFromNamespace("addCailliez",        "vegan")

  # ----- Thread resolution ------------------------------------------------
  # Match the task's pipeline/run.R semantics: a user-passed `parallel` arg
  # (cluster object or integer) takes precedence; otherwise consult
  # getOption("autozyme.threads") (set by zyme run on iteration phase) and
  # finally fall back to detectCores(logical=FALSE).
  .resolve_zyme_threads <- function(parallel_arg = NULL) {
    if (inherits(parallel_arg, "cluster"))
      return(parallel_arg)
    default <- parallel::detectCores(logical = FALSE)
    if (is.na(default) || default < 1L)
      default <- parallel::detectCores(logical = TRUE)
    if (is.na(default) || default < 1L)
      default <- 1L
    if (!is.null(parallel_arg)) {
      n <- suppressWarnings(as.integer(parallel_arg[[1L]]))
      if (!is.na(n) && n > 0L)
        default <- n
    }
    max(1L, as.integer(getOption("autozyme.threads", default)))
  }

  .vegan_gemm <- function(A, B, threads = 0L) {
    if (.Platform$OS.type == "windows") {
      return(.az_gemm(A, B, threads = threads, fallback = TRUE,
                      patch = "vegan"))
    }
    A %*% B
  }

  # ============================================================
  # fast_adonis0: build dbRDA fit directly from Gram + single qr()
  # ============================================================
  fast_adonis0 <- function(lhs, X = NULL, Z = NULL, zyme = TRUE, ...) {
    dots <- list(...)
    # Delegate to vanilla on any of:
    #   - explicit zyme=FALSE per-call escape
    #   - global autozyme::with_disabled({...}) context
    #   - unrecognized kwargs arriving via `...` -- happens when vanilla
    #     adonis2 (invoked from our zyme=FALSE branch of fast_adonis2)
    #     internally calls adonis0(formula, data=, method=) and lands on
    #     this still-patched symbol in vegan's namespace (Bug 7)
    #   - explicit Z block (original fast-path guard)
    if (!isTRUE(zyme) || autozyme::is_disabled() ||
        length(dots) > 0L || (!is.null(Z) && ncol(Z))) {
      args <- c(list(lhs),
                if (!is.null(X)) list(X) else list(),
                if (!is.null(Z)) list(Z) else list(),
                dots)
      return(do.call(.orig_adonis0, args))
    }

    G <- .initDBRDA_vegan(lhs)
    total_chi <- sum(diag(G))
    sol <- list(Ybar = G, tot.chi = total_chi, adjust = 1,
                method = "adonis")

    if (!is.null(Z))
      X <- cbind(Z, X)
    X <- scale(X, scale = FALSE)
    qrhs <- qr(X)
    rank <- qrhs$rank

    if (rank > 0) {
      qmat <- qr.Q(qrhs, complete = FALSE)[, seq_len(rank), drop = FALSE]
      gq <- .vegan_gemm(G, qmat, threads = .resolve_zyme_threads(NULL))
      fit_chi <- sum(.colSums(qmat * gq, nrow(G), rank, na.rm = FALSE))
      CCA <- list(rank = rank, qrank = rank, tot.chi = fit_chi, QR = qrhs)
    } else {
      fit_chi <- 0
      CCA <- NULL
    }

    CA <- list(rank = nrow(lhs) - max(qrhs$rank, 0) - 1,
               u = matrix(0, nrow = nrow(lhs)),
               tot.chi = total_chi - fit_chi)
    sol$CCA <- CCA
    sol$CA <- CA
    class(sol) <- c("adonis2", "dbrda", "rda", "cca")
    sol
  }

  # ============================================================
  # fast_adonis2: parallelDist Bray swap + dbRDA fast core
  # ============================================================
  fast_adonis2 <- function(formula, data, permutations = 999, method = "bray",
                           sqrt.dist = FALSE, add = FALSE, by = NULL,
                           parallel = getOption("mc.cores"),
                           na.action = na.fail, strata = NULL, ...,
                           zyme = TRUE) {
    dots <- list(...)
    if (!isTRUE(zyme) || !identical(method, "bray") || length(dots) > 0L ||
        !requireNamespace("parallelDist", quietly = TRUE)) {
      return(.orig_adonis2(formula = formula, data = data,
                          permutations = permutations, method = method,
                          sqrt.dist = sqrt.dist, add = add, by = by,
                          parallel = parallel, na.action = na.action,
                          strata = strata, ...))
    }

    if (missing(data))
      data <- parent.frame()
    else
      data <- eval(match.call()$data, parent.frame(),
                   enclos = environment(formula))
    formula <- formula(terms(formula, data = data))
    if (!is.null(by))
      by <- match.arg(by, c("terms", "margin", "onedf"))
    lhs <- eval(formula[[2]], envir = parent.frame(),
                enclos = environment(formula))
    if ((is.matrix(lhs) || is.data.frame(lhs)) &&
        isSymmetric(unname(as.matrix(lhs))))
      lhs <- as.dist(lhs)
    if (!inherits(lhs, "dist")) {
      run_parallel <- .resolve_zyme_threads(parallel)
      distance_threads <- if (inherits(run_parallel, "cluster")) {
        1L
      } else {
        max(1L, as.integer(run_parallel))
      }
      lhs <- parallelDist::parDist(as.matrix(lhs), method = "bray",
                                   threads = distance_threads)
    }

    d <- .ordiParseFormula_vegan(formula = formula,
                                 data = data,
                                 na.action = na.action,
                                 subset = NULL,
                                 X = lhs)
    if (is.null(d$Y))
      stop("needs explanatory variables on the right-hand-side")
    lhs <- d$X
    if (!is.null(d$na.action))
      lhs <- lhs[-d$na.action, -d$na.action, drop = FALSE]
    if (sqrt.dist)
      lhs <- sqrt(lhs)
    if (is.logical(add) && add)
      add <- "lingoes"
    if (is.character(add)) {
      add <- match.arg(add, c("lingoes", "cailliez"))
      if (add == "lingoes") {
        ac <- .addLingoes_vegan(as.matrix(lhs))
        lhs <- sqrt(lhs^2 + 2 * ac)
      } else if (add == "cailliez") {
        ac <- .addCailliez_vegan(as.matrix(lhs))
        lhs <- lhs + ac
      }
    }
    # Resolve adonis0 by namespace lookup so the active vegan::adonis0 binding
    # (our fast_adonis0 when patch is active, orig otherwise) is what runs.
    sol <- getFromNamespace("adonis0", "vegan")(lhs, d$Y, d$Z)
    sol$formula <- match.call()
    sol$terms <- d$terms
    sol$terminfo <- .ordiTerminfo_vegan(d, data)
    perm <- .getPermuteMatrix_vegan(permutations, NROW(lhs), strata = strata)
    run_parallel <- .resolve_zyme_threads(parallel)
    out <- anova(sol, permutations = perm, by = by, parallel = run_parallel)
    att <- attributes(out)
    out <- rbind(out, "Total" = c(nobs(sol) - 1, sol$tot.chi, NA, NA))
    out <- cbind(out[, 1:2], "R2" = out[, 2] / sol$tot.chi, out[, 3:4])
    att$heading[2] <- deparse(match.call(), width.cutoff = 500L)
    att$names <- names(out)
    att$row.names <- rownames(out)
    attributes(out) <- att
    out
  }

  # ============================================================
  # fast_permutest_cca: dbRDA + by="terms" + model="reduced" path
  # ============================================================
  fast_permutest_cca <- function(x, permutations = permute::how(nperm = 99),
                                 model = c("reduced", "direct", "full"),
                                 by = NULL, first = FALSE, strata = NULL,
                                 parallel = getOption("mc.cores"), ...,
                                 zyme = TRUE) {
    if (!isTRUE(zyme) || is.null(x$CCA) || x$CCA$rank == 0) {
      return(.orig_permutest_cca(x = x, permutations = permutations,
                                model = model, by = by, first = first,
                                strata = strata, parallel = parallel, ...))
    }

    if (!is.null(by)) {
      if (first)
        stop("'by' cannot be used with option 'first=TRUE'")
      by <- match.arg(by, c("onedf", "terms"))
      if (by == "terms" && is.null(x$terminfo))
        stop("by='terms' needs a model fitted with a formula")
    }
    model <- match.arg(model)

    w <- attr(.ordiYbar_vegan(x, "initial"), "RW")
    is_cca <- !is.null(w)
    is_partial <- !is.null(x$pCCA) && x$pCCA$rank > 0
    is_db <- inherits(x, c("dbrda"))

    if (!identical(model, "reduced") || !identical(by, "terms") ||
        isTRUE(first) || is_partial || is_cca || !is_db) {
      return(.orig_permutest_cca(x = x, permutations = permutations,
                                model = model, by = by, first = first,
                                strata = strata, parallel = parallel, ...))
    }

    Q <- x$CCA$QR
    Chi.z <- x$CCA$tot.chi
    names(Chi.z) <- "Model"
    q <- x$CCA$qrank

    partXbar <- .ordiYbar_vegan(x, "partial")
    ass <- x$terminfo$assign
    if (is.null(ass))
      stop("update() old ordination result object")
    pivot <- Q$pivot
    ass <- ass[pivot[seq_len(x$CCA$qrank)]]
    effects <- cumsum(rle(ass)$length)
    termlabs <- labels(terms(x$terminfo))
    termlabs <- termlabs[unique(ass)]
    q <- diff(c(0, effects))

    F.0 <- numeric(length(effects))
    for (k in seq_along(effects)) {
      fv <- qr.fitted(Q, partXbar, k = effects[k])
      F.0[k] <- sum(diag(fv))
    }

    Chi.xz <- x$CA$tot.chi
    names(Chi.xz) <- "Residual"
    r <- nobs(x) - Q$rank - 1
    Chi.tot <- Chi.z + Chi.xz

    Chi.z <- diff(c(0, F.0))
    F.0 <- Chi.z / q * r / Chi.xz

    E <- .ordiYbar_vegan(x, "partial")
    N <- nrow(E)
    permutations <- .getPermuteMatrix_vegan(permutations, N, strata = strata)
    nperm <- nrow(permutations)

    p <- length(effects)
    max_rank <- max(effects)
    qmat <- qr.Q(Q, complete = FALSE)[, seq_len(max_rank), drop = FALSE]
    total_cols <- max_rank * nperm
    run_parallel <- .resolve_zyme_threads(parallel)
    workers <- if (inherits(run_parallel, "cluster")) {
      1L
    } else {
      max(1L, as.integer(run_parallel))
    }
    workers <- min(workers, total_cols)

    # Chunked permutation. The per-permutation trace is column-separable --
    # trace_cols[j] = colSums(q_j * (E %*% q_j)) depends only on column j -- so
    # the nperm permutations can be processed in bounded batches with bit-
    # identical results (the same separability the mclapply path already relies
    # on). Before this rewrite the full qbig (N x max_rank*nperm) AND its
    # product eqbig were materialised at once: ~2 * N * max_rank * nperm * 8
    # bytes (~240 MB at N=5000, max_rank=3, nperm=999; and the O(nperm) cliff
    # the audit flagged -- ~15 GB at max_rank=20, nperm=9999). Per batch the
    # live working set is ~2 * N * max_rank * batch_perms * 8 bytes (~30 MB at
    # batch_perms=128) and the qbatch buffer is reused; the E %*% qbatch GEMM
    # stays large enough to keep BLAS/mclapply efficiency.
    batch_perms <- max(1L, min(
      as.integer(getOption("autozyme.vegan.perm_batch", 128L)), nperm))

    trace_cols <- numeric(total_cols)
    inv_perm <- integer(N)
    batch_starts <- seq.int(1L, nperm, by = batch_perms)

    for (b0 in batch_starts) {
      b1 <- min(b0 + batch_perms - 1L, nperm)
      nb <- b1 - b0 + 1L
      nb_cols <- max_rank * nb
      qbatch <- matrix(0, nrow = N, ncol = nb_cols)
      for (j in seq_len(nb)) {
        perm <- permutations[b0 + j - 1L, ]
        inv_perm[perm] <- seq_len(N)
        cc <- ((j - 1L) * max_rank + 1L):(j * max_rank)
        qbatch[, cc] <- qmat[inv_perm, , drop = FALSE]
      }
      out_cols <- ((b0 - 1L) * max_rank + 1L):(b1 * max_rank)
      if (.Platform$OS.type == "unix" && workers > 1L && nb_cols > 1L) {
        nchunk <- min(workers, nb_cols)
        chunks <- parallel::splitIndices(nb_cols, nchunk)
        trace_cols[out_cols] <- unlist(.zyme_mclapply(chunks, function(cols) {
          qchunk <- qbatch[, cols, drop = FALSE]
          eqchunk <- .vegan_gemm(E, qchunk, threads = workers)
          .colSums(qchunk * eqchunk, N, length(cols), na.rm = FALSE)
        }, mc.cores = nchunk), use.names = FALSE)
      } else {
        eqbatch <- .vegan_gemm(E, qbatch, threads = workers)
        trace_cols[out_cols] <- .colSums(qbatch * eqbatch, N, nb_cols, na.rm = FALSE)
      }
    }
    trace_by_rank <- matrix(trace_cols, nrow = max_rank, ncol = nperm)
    if (max_rank > 1) {
      for (k in 2:max_rank)
        trace_by_rank[k, ] <- trace_by_rank[k, ] + trace_by_rank[k - 1L, ]
    }
    term_chi <- matrix(0, nrow = nperm, ncol = p)
    previous <- numeric(nperm)
    for (k in seq_along(effects)) {
      current <- trace_by_rank[effects[k], ]
      term_chi[, k] <- current - previous
      previous <- current
    }

    den <- Chi.tot - rowSums(term_chi)
    if (p > 1) {
      num <- sweep(term_chi, 2, q, "/")
      F.perm <- sweep(num, 1, den / r, "/")
    } else {
      num <- term_chi[, 1]
      F.perm <- matrix((num / q) / (den / r), ncol = 1)
    }

    Call <- match.call()
    Call[[1]] <- as.name("permutest")
    sol <- list(call = Call, testcall = x$call, model = model,
                F.0 = F.0, F.perm = F.perm, chi = c(Chi.z, Chi.xz),
                num = num, den = den, df = c(q, r), nperm = nperm,
                method = x$method, first = first, termlabels = termlabs)
    sol$Random.seed <- attr(permutations, "seed")
    sol$control <- attr(permutations, "control")
    if (!missing(strata)) {
      sol$strata <- deparse(substitute(strata))
      sol$stratum.values <- strata
    }
    class(sol) <- "permutest.cca"
    sol
  }

  register_patch(
    name     = "vegan",
    upstream = "vegan",
    targets  = list(
      adonis0       = fast_adonis0,
      adonis2       = fast_adonis2,
      permutest.cca = fast_permutest_cca
    ),
    smoke = list(
      load = function(task_dir, tier) {
        # Fair-comparison rule (per 4_package.md): user-side prep goes in
        # `load`, untimed. The user runs adonis2() on either a count matrix
        # or a precomputed dist — both are documented inputs. We construct
        # the Bray-Curtis dist here so the timed `call` measures ONLY the
        # PERMANOVA permutation engine (adonis0 + permutest.cca), which is
        # the algorithmic core this patch optimizes. The Bray distance
        # speedup (parallelDist swap) IS a real win in fast_adonis2's hot
        # path, but timing it would mix two distinct optimizations into one
        # ratio; we report the cleaner number here. The smoke `call` mirrors
        # pipeline/run.R's `adonis2(counts ~ group + x, ...)` shape but with
        # a precomputed `dist` LHS, which adonis2 accepts as a documented
        # input form (the `if (!inherits(lhs, "dist"))` branch is skipped).
        task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
        ds <- Filter(function(d) identical(d$tier, tier), task$datasets)
        if (length(ds) == 0L) {
          stop("no dataset for tier '", tier, "' in task.yaml")
        }
        ds <- ds[[1]]
        data_path <- if (startsWith(ds$path, "./")) {
          file.path(task_dir, sub("^\\./", "", ds$path))
        } else if (substr(ds$path, 1L, 1L) == "/") {
          ds$path
        } else {
          file.path(task_dir, ds$path)
        }
        if (!file.exists(data_path)) {
          stop(sprintf("dataset for tier '%s' not found at %s", tier, data_path))
        }
        dat <- readRDS(data_path)
        # Pre-compute Bray-Curtis distance (vegan::vegdist matches what
        # adonis2 would build internally; using vegan here keeps baseline
        # and patched calls receiving bit-identical inputs).
        dist_mat <- vegan::vegdist(dat$counts, method = "bray")
        list(
          dist_mat = dist_mat,
          meta     = dat$meta,
          perms    = 999L,
          seed     = 1234L
        )
      },
      call = function(inputs) {
        # Single canonical invocation — matches pipeline/run.R / reference.R
        # exactly (same seed, same perms, same formula, by="terms"). LHS is
        # a `dist` so adonis2's parallelDist branch is bypassed; the timed
        # region is the dbRDA fit + permutation trace loop.
        #
        # vegan::adonis2 resolves the formula's LHS via
        # `eval(formula[[2]], envir = parent.frame(), ...)`. `enclos` is
        # silently ignored when `envir` is an environment — only the frame
        # and its lexical parent chain are searched. Under the autozyme
        # dispatcher wrapper, parent.frame() inside fast_adonis2 is the
        # wrapper's frame whose enclosing env walks autozyme ns -> base ->
        # R_GlobalEnv. Bind `dist_mat` (and `meta`) in .GlobalEnv so both
        # baseline (adonis2 is the original; parent.frame is smoke$call —
        # locals would also work) AND patched paths resolve them. Cleanup
        # on exit keeps the global namespace pristine. The assign overhead
        # is constant (microseconds) on both sides, so it does not bias
        # the speedup ratio.
        set.seed(inputs$seed)
        assign("dist_mat", inputs$dist_mat, envir = globalenv())
        assign("meta",     inputs$meta,     envir = globalenv())
        on.exit({
          if (exists("dist_mat", envir = globalenv(), inherits = FALSE))
            rm("dist_mat", envir = globalenv())
          if (exists("meta", envir = globalenv(), inherits = FALSE))
            rm("meta", envir = globalenv())
        }, add = TRUE)
        vegan::adonis2(dist_mat ~ group + x, data = meta,
                       permutations = inputs$perms, by = "terms")
      },
      save = function(result, dir, tier = "tiny", ...) {
        # evaluate.R reads pipeline/adonis2.rds — mirror reference.R's keys
        # exactly so the symmetric reader picks up Df / SumOfSqs / R2 / F /
        # Pr / F.perm / rownames / vegan_version / perms / seed.
        out <- list(
          Df            = result$Df,
          SumOfSqs      = result$SumOfSqs,
          R2            = result$R2,
          F             = result$F,
          Pr            = result$`Pr(>F)`,
          F.perm        = attr(result, "F.perm"),
          rownames      = rownames(result),
          vegan_version = as.character(utils::packageVersion("vegan")),
          perms         = 999L,
          seed          = 1234L
        )
        # verify_worker passes both pipeline_dir and ref_dir; evaluate.R
        # reads ref from ZYME_REFERENCE_DIR/adonis2.rds and test from
        # pipeline/adonis2.rds. The worker creates the trailing `pipeline`
        # / `reference_output_<tier>` subdir as needed — write directly.
        saveRDS(out, file.path(dir, "adonis2.rds"))
      }
    ),
    tested_against = "vegan 2.8.0",
    tested_upstream_versions = list(vegan = "2.8.0")
  )
}
