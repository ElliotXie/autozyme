# Patch for fgsea::fgseaMultilevel (+ preparePathwaysAndStats + calcGseaStat).
#
# Lifted from autozyme task `test_fgsea` (converged iter speedup_pct=83.5%
# ~6x at tiny). Three coordinated namespace-fn overrides on fgsea, plus one
# compiled C++ kernel pair (autozyme src/fgsea.cpp) driving the hot loops:
#
#   1. fgsea:::calcGseaStat -> fast_calcGseaStat
#        Strip per-call match.arg + stopifnot (fgseaMultilevel's initial
#        scoring already passes valid scoreType + sorted-unique selectedStats),
#        lift the scoreType if/else outside the math. Bit-exact ES.
#
#   2. fgsea:::preparePathwaysAndStats -> fast_preparePathwaysAndStats
#        Cache pathway -> raw-name-index map across cluster calls. names(stats)
#        is identical across clusters in the typical multi-cluster workflow,
#        so per-pathway fmatch+unique+na.omit (~17% of wall) is invariant —
#        compute once, reuse. Per-cluster cost: sort + ^gseaParam (unchanged)
#        + an O(n) inverse-permutation build + a vectorized inv[idx] remap
#        per pathway (cheaper than the original ~9420 fmatch calls per
#        cluster). Falls back to upstream for malformed input.
#
#   3. fgsea::fgseaMultilevel -> fast_fgseaMultilevel
#        - Replace the `lapply(pathwaysFiltered, calcGseaStat, ...,
#          returnLeadingEdge=TRUE)` initial-scoring loop with one C++ call
#          to calcEsLeBatchCpp (autozyme src/fgsea.cpp).
#        - Replace the per-(size-group) multilevel C++ call with
#          fastFgseaMultilevelBatchCpp — a vendored fgsea 1.34.2 EsRuler
#          batched across size groups with a std::thread worker pool. Each
#          group is independent (own RNG seeded with shared seed) -> bit-
#          exact vs per-group entry.
#        - Replace qbeta + multilevelError + trigamma per-pathway scalar
#          calls with precomputed integer-indexed lookup tables (~160K calls
#          per cluster collapsed to vectorized table lookups).
#
# Patch kind: namespace function (three targets, same upstream). C++ kernels
# (calcEsLeBatchCpp + fastFgseaMultilevelBatchCpp) live in autozyme/src/
# fgsea.cpp, exposed through useDynLib(autozyme).
#
# Threading: the multilevel batch kernel uses std::thread workers. Thread
# count is read from auto_threads() at call time (env / option / hardware
# default), capped at 8 (the converged sweet spot from rounds 36-44 — the
# multilevel inner loop is small per group, so adding workers past ~8 hits
# diminishing returns).

if (requireNamespace("fgsea",        quietly = TRUE) &&
    requireNamespace("BiocParallel", quietly = TRUE) &&
    requireNamespace("data.table",   quietly = TRUE) &&
    requireNamespace("fastmatch",    quietly = TRUE)) {

  # ---------------------------------------------------------------------------
  # Originals + internal helpers captured at file scope (convention #3).
  # ---------------------------------------------------------------------------
  .orig_fgseaMultilevel        <- utils::getFromNamespace("fgseaMultilevel",        "fgsea")
  .orig_preparePathwaysAndStats <- utils::getFromNamespace("preparePathwaysAndStats", "fgsea")
  .orig_calcGseaStat            <- utils::getFromNamespace("calcGseaStat",            "fgsea")

  .fgsea_setUpBPPARAM   <- utils::getFromNamespace("setUpBPPARAM",   "fgsea")
  .fgsea_fgseaSimpleImpl <- utils::getFromNamespace("fgseaSimpleImpl", "fgsea")
  .fgsea_multilevelError <- utils::getFromNamespace("multilevelError", "fgsea")

  # ---------------------------------------------------------------------------
  # fast_calcGseaStat — strip per-call match.arg + stopifnot, lift scoreType
  # branch outside the math. Bit-exact ES (no RNG).
  # ---------------------------------------------------------------------------
  fast_calcGseaStat <- function(stats, selectedStats, gseaParam = 1,
                                returnAllExtremes = FALSE,
                                returnLeadingEdge = FALSE,
                                scoreType = "std", zyme = TRUE) {
    if (!isTRUE(zyme)) {
      return(.orig_calcGseaStat(stats = stats, selectedStats = selectedStats,
                                gseaParam = gseaParam,
                                returnAllExtremes = returnAllExtremes,
                                returnLeadingEdge = returnLeadingEdge,
                                scoreType = scoreType))
    }
    S <- selectedStats
    if (is.unsorted(S, strictly = TRUE)) S <- sort(S)
    N <- length(stats)
    m <- length(S)
    if (m == N) stop("GSEA statistic is not defined when all genes are selected")

    rAdj <- abs(stats[S])
    if (gseaParam != 1) rAdj <- rAdj^gseaParam
    NR <- sum(rAdj)

    if (NR == 0) {
      rCumSum <- seq_len(m) / m
      bottoms_offset <- 1 / m
    } else {
      rCumSum <- cumsum(rAdj) / NR
      bottoms_offset <- rAdj / NR
    }

    tops <- rCumSum - (S - seq_len(m)) / (N - m)
    bottoms <- tops - bottoms_offset

    maxP <- max(tops); minP <- min(bottoms)

    if (scoreType == "std") {
      geneSetStatistic <- if (maxP == -minP) 0 else if (maxP > -minP) maxP else minP
    } else if (scoreType == "pos") {
      geneSetStatistic <- maxP
    } else {
      geneSetStatistic <- minP
    }

    if (!returnAllExtremes && !returnLeadingEdge) return(geneSetStatistic)

    res <- list(res = geneSetStatistic)
    if (returnAllExtremes) {
      res$tops <- tops
      res$bottoms <- bottoms
    }
    if (returnLeadingEdge) {
      if (scoreType == "std") {
        leadingEdge <- if (maxP > -minP) {
          S[seq_len(which.max(tops))]
        } else if (maxP < -minP) {
          rev(S[which.min(bottoms):m])
        } else {
          integer(0)
        }
      } else if (scoreType == "pos") {
        leadingEdge <- S[seq_len(which.max(tops))]
      } else {
        leadingEdge <- rev(S[which.min(bottoms):m])
      }
      res$leadingEdge <- leadingEdge
    }
    res
  }

  # ---------------------------------------------------------------------------
  # Cross-call caches. Lifted from pipeline/run.R — populated lazily on first
  # invocation, reused across subsequent clusters in the same R session.
  # Held in patch-scope envs (autozyme namespace), inherited by mclapply
  # children via copy-on-write fork.
  # ---------------------------------------------------------------------------
  .fgsea_pp_cache    <- new.env(parent = emptyenv())
  .fgsea_qbeta_cache <- new.env(parent = emptyenv())
  .fgsea_me_cache    <- new.env(parent = emptyenv())

  .fgsea_qbeta_tables <- function(nPermSimple) {
    key <- as.character(nPermSimple)
    if (is.null(.fgsea_qbeta_cache[[key]])) {
      k <- 0:nPermSimple
      ql <- qbeta(0.025, shape1 = k,      shape2 = nPermSimple - k + 1L)
      qr <- qbeta(0.975, shape1 = k + 1L, shape2 = nPermSimple - k)
      .fgsea_qbeta_cache[[key]] <- list(left = ql, right = qr)
    }
    .fgsea_qbeta_cache[[key]]
  }

  .fgsea_me_tables <- function(nPermSimple, sampleSize) {
    key <- paste(nPermSimple, sampleSize, sep = "_")
    if (is.null(.fgsea_me_cache[[key]])) {
      k <- 0:nPermSimple
      pvals <- (k + 1L) / (nPermSimple + 1L)
      me <- .fgsea_multilevelError(pvals, sampleSize)
      trig_n <- trigamma(k + 1L)
      .fgsea_me_cache[[key]] <- list(me = me, trig_n = trig_n,
                                      trig_total = trigamma(nPermSimple + 1L))
    }
    .fgsea_me_cache[[key]]
  }

  # ---------------------------------------------------------------------------
  # fast_preparePathwaysAndStats — cache pathway -> raw-name-index map across
  # clusters; per-cluster inv-permutation remap is cheap.
  # ---------------------------------------------------------------------------
  fast_preparePathwaysAndStats <- function(pathways, stats, minSize, maxSize,
                                           gseaParam, scoreType, zyme = TRUE) {
    if (!isTRUE(zyme)) {
      return(.orig_preparePathwaysAndStats(pathways, stats, minSize, maxSize,
                                            gseaParam, scoreType))
    }
    raw_names <- names(stats)
    if (is.null(raw_names) || any(is.na(raw_names)) || any(raw_names == "") ||
        anyDuplicated(raw_names)) {
      return(.orig_preparePathwaysAndStats(pathways, stats, minSize, maxSize,
                                            gseaParam, scoreType))
    }

    cache_key <- list(n  = length(raw_names),
                      h1 = raw_names[1L],
                      h2 = raw_names[length(raw_names)],
                      np = length(pathways),
                      pf = if (length(pathways)) names(pathways)[1L] else NULL,
                      pl = if (length(pathways)) names(pathways)[length(pathways)] else NULL,
                      pw_addr = data.table::address(pathways))

    if (!isTRUE(identical(.fgsea_pp_cache$key, cache_key))) {
      .fgsea_pp_cache$pathway_idx_raw <- lapply(pathways, function(p) {
        unique(stats::na.omit(fastmatch::fmatch(p, raw_names)))
      })
      .fgsea_pp_cache$pathway_sizes <- lengths(.fgsea_pp_cache$pathway_idx_raw)
      # Hold references so cached address keys cannot be reused for unrelated
      # objects during this R session.
      .fgsea_pp_cache$pathways_ref <- pathways
      .fgsea_pp_cache$key <- cache_key
    }

    if (any(!is.finite(stats))) stop("Not all stats values are finite numbers")
    ties <- sum(duplicated(stats[stats != 0]))
    if (ties != 0) {
      warning("There are ties in the preranked stats (",
              paste(round(ties * 100 / length(stats), digits = 2)),
              "% of the list).\n",
              "The order of those tied genes will be arbitrary, which may produce unexpected results.")
    }
    if (all(stats > 0) & scoreType == "std") {
      warning("All values in the stats vector are greater than zero and scoreType is \"std\", ",
              "maybe you should switch to scoreType = \"pos\".")
    }
    prepared_stats <- sort(stats, decreasing = TRUE)
    prepared_stats <- abs(prepared_stats)^gseaParam

    ord <- order(stats, decreasing = TRUE)
    inv_ord <- integer(length(ord))
    inv_ord[ord] <- seq_along(ord)

    pathway_sizes <- .fgsea_pp_cache$pathway_sizes
    minSize_eff <- max(minSize, 1)
    maxSize_eff <- min(maxSize, length(raw_names) - 1L)
    toKeep <- which(minSize_eff <= pathway_sizes & pathway_sizes <= maxSize_eff)

    pathway_idx_raw <- .fgsea_pp_cache$pathway_idx_raw
    pathwaysFiltered <- lapply(pathway_idx_raw[toKeep], function(idx) inv_ord[idx])
    pathwaysSizes <- pathway_sizes[toKeep]

    list(filtered = pathwaysFiltered,
         sizes    = pathwaysSizes,
         stats    = prepared_stats)
  }

  # ---------------------------------------------------------------------------
  # multilevelImpl override — batched call to fastFgseaMultilevelBatchCpp
  # (compiled in autozyme/src/fgsea.cpp). Each multilevel-size-group has its
  # own EsRuler; the batch kernel parallelizes across groups with a std::thread
  # pool. Per-group RNG is seeded with the shared `seed`, so output is bit-
  # exact regardless of dispatch order.
  # ---------------------------------------------------------------------------
  # Cap at 8 — the converged sweet spot. Inner per-group multilevel work is
  # small, so additional workers hit diminishing returns above ~8.
  .fgsea_inner_thread_cap <- 8L

  .fgsea_multilevelImpl <- function(multilevelPathwaysList, stats, sampleSize,
                                    seed, eps, sign = FALSE, BPPARAM = NULL) {
    size <- ES <- NULL
    groupES <- lapply(multilevelPathwaysList, function(x) x[, ES])
    sizes   <- vapply(multilevelPathwaysList, function(x) unique(x[, size]), integer(1))
    eps_per <- vapply(multilevelPathwaysList, function(x) eps * min(x$denomProb), numeric(1))
    nthr <- auto_threads(cap = .fgsea_inner_thread_cap)
    fastFgseaMultilevelBatchCpp(groupES, sizes, eps_per, stats, sampleSize, seed,
                                sign, nthreads = nthr)
  }

  # ---------------------------------------------------------------------------
  # fast_fgseaMultilevel — top-level replacement. Routes through the patched
  # preparePathwaysAndStats (via the namespace chain, so it picks up our
  # version when activated) and the C++ kernels above.
  # ---------------------------------------------------------------------------
  fast_fgseaMultilevel <- function(pathways, stats, sampleSize = 101, minSize = 1,
                                   maxSize = length(stats) - 1, eps = 1e-50,
                                   scoreType = c("std", "pos", "neg"),
                                   nproc = 0, gseaParam = 1, BPPARAM = NULL,
                                   nPermSimple = 1000, absEps = NULL,
                                   zyme = TRUE) {
    if (!isTRUE(zyme)) {
      return(.orig_fgseaMultilevel(pathways = pathways, stats = stats,
        sampleSize = sampleSize, minSize = minSize, maxSize = maxSize,
        eps = eps, scoreType = scoreType, nproc = nproc, gseaParam = gseaParam,
        BPPARAM = BPPARAM, nPermSimple = nPermSimple, absEps = absEps))
    }
    scoreType <- match.arg(scoreType)
    # Resolve via namespace so the active preparePathwaysAndStats (our patched
    # version when activated, original otherwise) is what runs.
    pp <- utils::getFromNamespace("preparePathwaysAndStats", "fgsea")(
      pathways, stats, minSize, maxSize, gseaParam, scoreType)
    pathwaysFiltered <- pp$filtered
    pathwaysSizes    <- pp$sizes
    stats <- pp$stats
    m <- length(pathwaysFiltered)
    if (m == 0) {
      return(data.table::data.table(pathway = character(), pval = numeric(),
                                    padj = numeric(), log2err = numeric(),
                                    ES = numeric(), NES = numeric(),
                                    size = integer(), leadingEdge = list()))
    }
    if (!is.null(absEps)) {
      warning("You are using deprecated argument `absEps`. Use `eps` argument instead. `absEps` was assigned to `eps`.")
      eps <- absEps
    }
    if (sampleSize < 3) {
      warning("sampleSize is too small, so sampleSize = 3 is set.")
      sampleSize <- max(3, sampleSize)
    }
    log2err <- nMoreExtreme <- pathway <- pval <- padj <- NULL
    nLeZero <- nGeZero <- leZeroMean <- geZeroMean <- nLeEs <- nGeEs <- isCpGeHalf <- NULL
    ES <- NES <- size <- leadingEdge <- modeFraction <- denomProb <- NULL

    minSize <- max(minSize, 1)
    eps <- max(0, min(1, eps))
    if (sampleSize %% 2 == 0) sampleSize <- sampleSize + 1

    # === HOT SPOT: lapply(calcGseaStat, ...) -> single C++ batch call ===
    # Guard: if all stats are zero, NR=0 for every pathway — the C++ kernel
    # would divide by zero. Fall back to the R-level calcGseaStat which
    # handles NR==0 explicitly (uniform 1/k increments).
    if (all(stats == 0)) {
      esle_list <- lapply(pathwaysFiltered, function(pw) {
        fast_calcGseaStat(stats, pw, gseaParam = 1,
                          returnLeadingEdge = TRUE, scoreType = scoreType)
      })
      pathwayScores <- vapply(esle_list, function(x) x$res, numeric(1))
      stat_names <- names(stats)
      leadingEdges <- lapply(esle_list, function(x) stat_names[x$leadingEdge])
    } else {
    esle <- calcEsLeBatchCpp(stats, pathwaysFiltered, scoreType)
    pathwayScores <- esle$es
    stat_names    <- names(stats)
    leadingEdges  <- lapply(esle$le, function(idx) stat_names[idx])
    }

    seeds <- sample.int(10^9, 1)
    BPPARAM <- .fgsea_setUpBPPARAM(nproc = nproc, BPPARAM = BPPARAM)

    simpleFgseaRes <- .fgsea_fgseaSimpleImpl(
      pathwayScores = pathwayScores, pathwaysSizes = pathwaysSizes,
      pathwaysFiltered = pathwaysFiltered, leadingEdges = leadingEdges,
      permPerProc = nPermSimple, seeds = seeds, toKeepLength = m,
      stats = stats, BPPARAM = BiocParallel::SerialParam(), scoreType = scoreType)

    switch(scoreType,
           std = simpleFgseaRes[, modeFraction := ifelse(ES >= 0, nGeZero, nLeZero)],
           pos = simpleFgseaRes[, modeFraction := nGeZero],
           neg = simpleFgseaRes[, modeFraction := nLeZero])

    simpleFgseaRes[, leZeroMean := NULL]
    simpleFgseaRes[, geZeroMean := NULL]
    simpleFgseaRes[, nLeEs := NULL]
    simpleFgseaRes[, nGeEs := NULL]
    simpleFgseaRes[, nLeZero := NULL]
    simpleFgseaRes[, nGeZero := NULL]

    simpleFgseaRes[modeFraction < 10, pval := as.numeric(NA)]
    simpleFgseaRes[modeFraction < 10, padj := as.numeric(NA)]
    simpleFgseaRes[modeFraction < 10, NES := as.numeric(NA)]

    if (any(simpleFgseaRes$modeFraction < 10)) {
      warning("There were ", paste(sum(simpleFgseaRes$modeFraction < 10)),
              " pathways for which P-values were not calculated properly due to ",
              "unbalanced (positive and negative) gene-level statistic values. ",
              "For such pathways pval, padj, NES, log2err are set to NA. ",
              "You can try to increase the value of the argument nPermSimple (for example set it nPermSimple = ",
              paste0(format(nPermSimple * 10, scientific = FALSE), ")"))
    }

    naSimpleRes <- simpleFgseaRes[is.na(pval)]
    naSimpleRes[, padj := as.numeric(NA)]
    naSimpleRes[, log2err := as.numeric(NA)]
    naSimpleRes[, modeFraction := NULL]

    simpleFgseaRes <- simpleFgseaRes[!is.na(pval)]

    qb <- .fgsea_qbeta_tables(nPermSimple)
    mt <- .fgsea_me_tables(nPermSimple, sampleSize)
    nme <- simpleFgseaRes$nMoreExtreme
    leftBorder  <- log2(qb$left[nme + 1L])
    rightBorder <- log2(qb$right[nme + 1L])
    crudeEstimator <- log2((simpleFgseaRes$nMoreExtreme + 1) / (nPermSimple + 1))
    simpleError <- 0.5 * pmax(crudeEstimator - leftBorder, rightBorder - crudeEstimator)
    multError <- mt$me[nme + 1L]

    if (all(multError >= simpleError)) {
      simpleFgseaRes[, log2err := 1 / log(2) * sqrt(mt$trig_n[nMoreExtreme + 1L] - mt$trig_total)]
      simpleFgseaRes[, modeFraction := NULL]
      simpleFgseaRes <- data.table::rbindlist(list(simpleFgseaRes, naSimpleRes), use.names = TRUE)
      data.table::setorder(simpleFgseaRes, pathway)
      simpleFgseaRes[, "nMoreExtreme" := NULL]
      data.table::setcolorder(simpleFgseaRes, c("pathway", "pval", "padj", "log2err",
                                                 "ES", "NES", "size", "leadingEdge"))
      simpleFgseaRes <- simpleFgseaRes[]
      return(simpleFgseaRes)
    }

    dtSimpleFgsea <- simpleFgseaRes[multError >= simpleError]
    dtSimpleFgsea[, log2err := 1 / log(2) * sqrt(mt$trig_n[nMoreExtreme + 1L] - mt$trig_total)]
    dtSimpleFgsea[, modeFraction := NULL]

    dtMultilevel <- simpleFgseaRes[multError < simpleError]
    dtMultilevel[, "denomProb" := (modeFraction + 1) / (nPermSimple + 1)]
    multilevelPathwaysList <- split(dtMultilevel, by = "size")
    indxs <- sample(1:length(multilevelPathwaysList))
    multilevelPathwaysList <- multilevelPathwaysList[indxs]

    seed <- sample.int(1e9, size = 1)
    sign <- if (scoreType %in% c("pos", "neg")) TRUE else FALSE
    cpp.res <- .fgsea_multilevelImpl(multilevelPathwaysList, stats, sampleSize,
                                     seed, eps, sign = sign, BPPARAM = BPPARAM)
    cpp.res <- data.table::rbindlist(cpp.res)

    result <- data.table::rbindlist(multilevelPathwaysList)
    result[, pval := pmin(1, cpp.res$cppMPval / denomProb)]
    result[, isCpGeHalf := cpp.res$cppIsCpGeHalf]
    result[, log2err := .fgsea_multilevelError(pval, sampleSize = sampleSize)]
    result[isCpGeHalf == FALSE, log2err := NA]

    if (!all(result$isCpGeHalf)) {
      warning("For some of the pathways the P-values were likely overestimated. ",
              "For such pathways log2err is set to NA.")
    }

    result[, isCpGeHalf := NULL]
    result[, modeFraction := NULL]
    result[, denomProb := NULL]

    result <- data.table::rbindlist(list(result, dtSimpleFgsea, naSimpleRes), use.names = TRUE)
    result[, nMoreExtreme := NULL]
    result[pval < eps, c("pval", "log2err") := list(eps, NA)]
    result[, padj := p.adjust(pval, method = "BH")]

    if (nrow(result[pval == eps & is.na(log2err)])) {
      warning("For some pathways, in reality P-values are less than ", paste(eps),
              ". You can set the `eps` argument to zero for better estimation.")
    }

    data.table::setcolorder(result, c("pathway", "pval", "padj", "log2err",
                                      "ES", "NES", "size", "leadingEdge"))
    data.table::setorder(result, pathway)
    result <- result[]
    result
  }

  # ---------------------------------------------------------------------------
  # Smoke recipe
  # ---------------------------------------------------------------------------
  # Fair-comparison boundary (per 4_package.md):
  #   - The task target is `fgsea::fgsea` and the workflow is one fgsea call
  #     per cluster across N clusters. pipeline/run.R parallelizes across
  #     clusters with mclapply; the cross-cluster pathway-prep cache is part
  #     of the patch's optimization surface, so the timed region must include
  #     more than one cluster for the cache amortization to show.
  #   - The bundle (per-cluster ranked stats + shared pathways) is read from
  #     a pre-baked .rds in load — user-side prep (the runner already did
  #     wilcoxauc + msigdbr), invariant whether patched or not.
  #   - The cluster loop itself + mclapply dispatch + all N fgsea calls -> call.
  #     Baseline runs the same outer mclapply + per-cluster fgsea (the patch
  #     does not change the outer loop, only the upstream's internal hot path).
  #     mc.cores is read from auto_threads() so the env-driven thread budget
  #     is matched on both baseline and patched subprocesses.
  #   - SEED is set inside every per-cluster invocation (mirrors reference.R)
  #     so the per-call RNG path is deterministic.

  .SEED     <- 42L
  .MIN_SIZE <- 15L
  .MAX_SIZE <- 500L

  .fgsea_smoke_load <- function(task_dir, tier) {
    task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
    ds <- Filter(function(d) identical(d$tier, tier), task$datasets)
    if (length(ds) == 0L) {
      stop("no dataset for tier '", tier, "' in task.yaml")
    }
    data_path <- resolve_dataset_path(task_dir, ds[[1]]$path)
    bundle <- readRDS(data_path)
    list(
      bundle    = bundle,
      n_workers = max(1L, as.integer(auto_threads(cap = 14L))),
      min_size  = .MIN_SIZE,
      max_size  = .MAX_SIZE,
      seed      = .SEED
    )
  }

  .fgsea_smoke_call <- function(inputs) {
    bundle    <- inputs$bundle
    n_workers <- min(inputs$n_workers, length(bundle$stats_list))
    seed      <- inputs$seed
    min_size  <- inputs$min_size
    max_size  <- inputs$max_size

    one_cluster <- function(cid) {
      set.seed(seed)
      res <- suppressWarnings(fgsea::fgsea(
        pathways = bundle$pathways,
        stats    = bundle$stats_list[[cid]],
        minSize  = min_size,
        maxSize  = max_size,
        BPPARAM  = BiocParallel::SerialParam(),
        eps      = 9e-6
      ))
      res$cluster <- cid
      res
    }
    cluster_ids <- names(bundle$stats_list)
    # LPT-ish dispatch (heaviest var first) — preserves the iter run's
    # ordering and gives mclapply better load balance under mc.preschedule=FALSE.
    cluster_var <- vapply(bundle$stats_list, var, numeric(1))
    dispatch_order <- order(-cluster_var)
    dispatch_ids   <- cluster_ids[dispatch_order]

    set.seed(seed)
    if (n_workers > 1L && Sys.info()[["sysname"]] != "Windows" &&
        length(dispatch_ids) > 1L) {
      results_list <- .zyme_mclapply(
        dispatch_ids, one_cluster,
        mc.cores       = n_workers,
        mc.preschedule = FALSE
      )
    } else {
      results_list <- lapply(dispatch_ids, one_cluster)
    }
    names(results_list) <- dispatch_ids
    results <- results_list[cluster_ids]
    data.table::rbindlist(results, use.names = TRUE)
  }

  .fgsea_smoke_save <- function(result, dir, tier = "tiny", ...) {
    # evaluate.R reads pipeline/fgsea_results.rds vs
    # reference_output_<tier>/fgsea_results.rds and compares ES / NES / pval
    # / padj per (cluster, pathway). Mirror reference.R's keys.
    saveRDS(result, file.path(dir, "fgsea_results.rds"))
  }

  register_patch(
    name     = "fgsea",
    upstream = "fgsea",
    targets  = list(
      fgseaMultilevel         = fast_fgseaMultilevel,
      preparePathwaysAndStats = fast_preparePathwaysAndStats,
      calcGseaStat            = fast_calcGseaStat
    ),
    smoke = list(
      load = .fgsea_smoke_load,
      call = .fgsea_smoke_call,
      save = .fgsea_smoke_save
    ),
    tested_against = "fgsea 1.34.2",
    tested_upstream_versions = list(fgsea = "1.34.2")
  )
}
