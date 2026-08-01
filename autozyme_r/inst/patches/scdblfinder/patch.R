# scDblFinder default-workflow accelerator.
#
# Lifted from autozyme task `test_scdblfinder`.
#
# Four namespace targets are registered as one release-locked unit.  The
# internal fast functions are active only while the patched public
# scDblFinder() driver is servicing the attested default call.  Direct calls
# to internals, unsupported arguments or inputs, global disablement, version
# drift, and source/formals drift all execute the captured upstream functions.

if (requireNamespace("scDblFinder", quietly = TRUE) &&
    requireNamespace("BiocParallel", quietly = TRUE) &&
    requireNamespace("BiocNeighbors", quietly = TRUE) &&
    requireNamespace("BiocSingular", quietly = TRUE) &&
    requireNamespace("DelayedArray", quietly = TRUE) &&
    requireNamespace("digest", quietly = TRUE) &&
    requireNamespace("Matrix", quietly = TRUE) &&
    requireNamespace("scater", quietly = TRUE) &&
    requireNamespace("scrapper", quietly = TRUE) &&
    requireNamespace("SingleCellExperiment", quietly = TRUE) &&
    requireNamespace("SummarizedExperiment", quietly = TRUE)) {

  # Preserve the upstream public formal exactly (`BPPARAM =
  # SerialParam(progressbar = verbose)`) while making its unqualified symbol
  # resolvable from this patch's isolated source environment.
  SerialParam <- BiocParallel::SerialParam

  .scdblfinder_ns <- asNamespace("scDblFinder")
  .orig_scDblFinder <- utils::getFromNamespace("scDblFinder", "scDblFinder")
  .orig_defaultProcessing <- utils::getFromNamespace(
    ".defaultProcessing", "scDblFinder"
  )
  .orig_evaluateKNN <- utils::getFromNamespace(".evaluateKNN", "scDblFinder")
  .orig_cxds2 <- utils::getFromNamespace("cxds2", "scDblFinder")

  # Hash exactly the width=500 deparse used by the task's source audit.  R
  # versions without tools::sha256sum fail closed and leave upstream active.
  .scdblfinder_sha256_text <- function(text) {
    tools_ns <- asNamespace("tools")
    if (!exists("sha256sum", envir = tools_ns, inherits = FALSE)) {
      return(NA_character_)
    }
    path <- tempfile("autozyme-scdblfinder-hash-")
    on.exit(unlink(path), add = TRUE)
    con <- file(path, open = "wb")
    tryCatch(
      writeBin(charToRaw(enc2utf8(text)), con),
      finally = close(con)
    )
    unname(get("sha256sum", envir = tools_ns, inherits = FALSE)(path))
  }

  .scdblfinder_fn_hashes <- function(fn) {
    list(
      body = .scdblfinder_sha256_text(paste(
        deparse(body(fn), width.cutoff = 500L), collapse = "\n"
      )),
      formals = .scdblfinder_sha256_text(paste(
        deparse(formals(fn), width.cutoff = 500L), collapse = "\n"
      ))
    )
  }

  .scdblfinder_expected_hashes <- list(
    scDblFinder = list(
      body = "0654126963c6721692462a8daef66df0bc76a326c9904568adbe8619081ee439",
      formals = "0f0e3a279d19128904cfa36ff48ad69d64861791cacfdba04744fafa223abcc6"
    ),
    .defaultProcessing = list(
      body = "098fe3bebf80d80d9bc0f2183a40d22d63abf9036a12d5dd0cda8413402012d8",
      formals = "7850c1a62fe3788ee08fc504d9583a8ab72d041afda1f476821154773b811644"
    ),
    .evaluateKNN = list(
      body = "4bb7c4aa7da82e36f36293c8fcb1bb57111a8d2a3ce71fe94c618650f83f75b8",
      formals = "c7ee10058c056a4782a69b8abfe274ed18fe10c5d6e5ddf794ffc5966477e9c4"
    ),
    cxds2 = list(
      body = "684b255fce06358f39a3a82ae0283fa7e6d2867d954b51a411c0c2401898d6f7",
      formals = "f4e8f2a6ce72d8b57abbea2b8badcbf1a78ba75a75ac0adfa7089a50892e5402"
    )
  )
  .scdblfinder_actual_hashes <- list(
    scDblFinder = .scdblfinder_fn_hashes(.orig_scDblFinder),
    .defaultProcessing = .scdblfinder_fn_hashes(.orig_defaultProcessing),
    .evaluateKNN = .scdblfinder_fn_hashes(.orig_evaluateKNN),
    cxds2 = .scdblfinder_fn_hashes(.orig_cxds2)
  )
  .scdblfinder_release_ok <- identical(
    as.character(utils::packageVersion("scDblFinder")), "1.27.6"
  ) && identical(.scdblfinder_actual_hashes, .scdblfinder_expected_hashes)

  # Process-local, nest-safe transient context.  It is intentionally entered
  # only by the supported public call, never by direct internal calls.
  .scdblfinder_context <- new.env(parent = emptyenv())
  .scdblfinder_context$depth <- 0L
  .scdblfinder_context_enter <- function() {
    old <- .scdblfinder_context$depth
    .scdblfinder_context$depth <- old + 1L
    old
  }
  .scdblfinder_context_exit <- function(old) {
    .scdblfinder_context$depth <- as.integer(old)
    invisible(NULL)
  }
  .scdblfinder_context_active <- function() {
    isTRUE(.scdblfinder_release_ok) &&
      .scdblfinder_context$depth > 0L &&
      !autozyme::is_disabled()
  }
  .scdblfinder_rng_snapshot <- function() {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      return(list(existed = TRUE, value = get(
        ".Random.seed", envir = globalenv(), inherits = FALSE
      )))
    }
    list(existed = FALSE, value = NULL)
  }
  .scdblfinder_rng_restore <- function(snapshot) {
    if (isTRUE(snapshot$existed)) {
      assign(".Random.seed", snapshot$value, envir = globalenv())
    } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
    invisible(NULL)
  }
  .scdblfinder_with_context_suspended <- function(expr) {
    old <- .scdblfinder_context$depth
    .scdblfinder_context$depth <- 0L
    on.exit(.scdblfinder_context$depth <- old, add = TRUE)
    force(expr)
  }

  .scdblfinder_is_exact_dgCMatrix <- function(x) {
    cls <- unname(as.character(class(x)))
    length(cls) == 1L && identical(cls[[1L]], "dgCMatrix")
  }
  .scdblfinder_norm_max_cols <- 35000L

  # Forward formal promises by name.  This avoids both do.call()'s materialized
  # call and re-evaluation of user expressions from match.call().
  .scdblfinder_invoke <- function(
    target,
    sce, clusters = NULL, samples = NULL, clustCor = NULL,
    artificialDoublets = NULL, knownDoublets = NULL,
    knownUse = c("discard", "positive"), dbr = NULL, dbr.sd = NULL,
    dbr.per1k = 0.008, nfeatures = 1352, dims = 20, k = NULL,
    removeUnidentifiable = TRUE, includePCs = 19, propRandom = 0,
    propMarkers = 0, aggregateFeatures = FALSE,
    returnType = c("sce", "table", "full", "counts", "scores"),
    BNPARAM = NULL, score = c("xgb", "weighted", "ratio"),
    processing = "default", metric = "logloss", nrounds = 0.25,
    max_depth = 4, iter = 3, trainingFeatures = NULL, unident.th = NULL,
    multiSampleMode = c("split", "singleModel", "singleModelSplitThres", "asOne"),
    threshold = TRUE, verbose = TRUE,
    BPPARAM = BiocParallel::SerialParam(progressbar = verbose), ...
  ) {
    target(
      sce = sce, clusters = clusters, samples = samples, clustCor = clustCor,
      artificialDoublets = artificialDoublets, knownDoublets = knownDoublets,
      knownUse = knownUse, dbr = dbr, dbr.sd = dbr.sd,
      dbr.per1k = dbr.per1k, nfeatures = nfeatures, dims = dims, k = k,
      removeUnidentifiable = removeUnidentifiable, includePCs = includePCs,
      propRandom = propRandom, propMarkers = propMarkers,
      aggregateFeatures = aggregateFeatures, returnType = returnType,
      BNPARAM = BNPARAM, score = score, processing = processing,
      metric = metric, nrounds = nrounds, max_depth = max_depth, iter = iter,
      trainingFeatures = trainingFeatures, unident.th = unident.th,
      multiSampleMode = multiSampleMode, threshold = threshold,
      verbose = verbose, BPPARAM = BPPARAM, ...
    )
  }

  .scdblfinder_public_supported <- function(mc, sce, verbose, BPPARAM) {
    if (!isTRUE(.scdblfinder_release_ok)) return(FALSE)
    supplied <- names(mc)[-1L]
    if (is.null(supplied) ||
        any(!supplied %in% c("sce", "verbose", "BPPARAM")) ||
        !identical(verbose, FALSE)) {
      return(FALSE)
    }
    if (!methods::is(sce, "SingleCellExperiment") ||
        ncol(sce) > 33000L || ncol(sce) < 1L) {
      return(FALSE)
    }
    counts <- tryCatch(
      SummarizedExperiment::assay(sce, "counts"),
      error = function(e) NULL
    )
    if (is.null(counts) || !.scdblfinder_is_exact_dgCMatrix(counts) ||
        anyNA(counts@x)) {
      return(FALSE)
    }
    libsize <- tryCatch(Matrix::colSums(counts), error = function(e) NULL)
    if (is.null(libsize) || any(!is.finite(libsize)) || any(libsize <= 0)) {
      return(FALSE)
    }
    workers <- tryCatch(BiocParallel::bpnworkers(BPPARAM), error = function(e) NA_integer_)
    serial <- methods::is(BPPARAM, "SerialParam")
    multicore <- .Platform$OS.type != "windows" &&
      methods::is(BPPARAM, "MulticoreParam")
    (serial || multicore) && workers %in% c(1L, 4L, 8L)
  }

  .scdblfinder_rewrite_sel_features <- function(fn) {
    body_text <- paste(deparse(body(fn), width.cutoff = 500L), collapse = "\n")
    from <- "sel_features <- selFeatures(sce[sel_features, ], cl, nfeatures = nfeatures, propMarkers = propMarkers)"
    to <- paste(
      "sel_features <- if (identical(sel_features, row.names(sce)))",
      "selFeatures(sce, cl, nfeatures = nfeatures, propMarkers = propMarkers)",
      "else selFeatures(sce[sel_features, ], cl, nfeatures = nfeatures, propMarkers = propMarkers)"
    )
    matches <- gregexpr(from, body_text, fixed = TRUE)[[1L]]
    count <- if (identical(matches, -1L)) 0L else length(matches)
    if (count == 1L) {
      body_text <- sub(from, to, body_text, fixed = TRUE)
      body(fn) <- parse(text = body_text, keep.source = FALSE)[[1L]]
    }
    list(fn = fn, count = count)
  }

  # Keep upstream gc() behavior.  The only public-driver rewrite is the exact
  # full-row feature-selection subset bypass accepted in task best 34b1fc7.
  .driver_scDblFinder <- .orig_scDblFinder
  .sel_features_rewrite <- .scdblfinder_rewrite_sel_features(.driver_scDblFinder)
  if (.sel_features_rewrite$count != 1L) .scdblfinder_release_ok <- FALSE
  if (.sel_features_rewrite$count == 1L) {
    .driver_scDblFinder <- .sel_features_rewrite$fn
  }

  # Preserve the complete upstream formal signature; the wrapper only chooses
  # between an exact upstream call and the release-locked driver clone.
  fast_scDblFinder <- .orig_scDblFinder
  body(fast_scDblFinder) <- quote({
    .mc <- match.call(expand.dots = FALSE)
    if (!.scdblfinder_public_supported(.mc, sce, verbose, BPPARAM)) {
      return(.scdblfinder_with_context_suspended(
        .scdblfinder_invoke(
          .orig_scDblFinder, sce, clusters, samples, clustCor,
          artificialDoublets, knownDoublets, knownUse, dbr, dbr.sd,
          dbr.per1k, nfeatures, dims, k, removeUnidentifiable, includePCs,
          propRandom, propMarkers, aggregateFeatures, returnType, BNPARAM,
          score, processing, metric, nrounds, max_depth, iter,
          trainingFeatures, unident.th, multiSampleMode, threshold, verbose,
          BPPARAM, ...
        )
      ))
    }
    .old_context <- .scdblfinder_context_enter()
    on.exit(.scdblfinder_context_exit(.old_context), add = TRUE)
    .scdblfinder_invoke(
      .driver_scDblFinder, sce, clusters, samples, clustCor,
      artificialDoublets, knownDoublets, knownUse, dbr, dbr.sd,
      dbr.per1k, nfeatures, dims, k, removeUnidentifiable, includePCs,
      propRandom, propMarkers, aggregateFeatures, returnType, BNPARAM,
      score, processing, metric, nrounds, max_depth, iter, trainingFeatures,
      unident.th, multiSampleMode, threshold, verbose, BPPARAM, ...
    )
  })
  environment(fast_scDblFinder) <- environment()

  fast_default_processing <- function(e, dims = NULL, doNorm = NULL) {
    fallback <- function() .scdblfinder_with_context_suspended(
      .orig_defaultProcessing(e, dims = dims, doNorm = doNorm)
    )
    if (!.scdblfinder_context_active() || !is.null(doNorm) ||
        length(dims) != 1L || !isTRUE(as.numeric(dims) == 20) ||
        !.scdblfinder_is_exact_dgCMatrix(e) || anyNA(e@x) ||
        ncol(e) > .scdblfinder_norm_max_cols) {
      return(fallback())
    }
    sf <- tryCatch(
      scrapper::centerSizeFactors(Matrix::colSums(e)),
      error = function(err) NULL
    )
    if (is.null(sf) || any(!is.finite(sf)) || any(sf <= 0)) return(fallback())
    rng <- .scdblfinder_rng_snapshot()
    tryCatch({
      normalized <- scrapper::normalizeCounts(e, sf, delayed = FALSE)
      normalized <- DelayedArray::DelayedArray(normalized)
      pca <- scater::calculatePCA(
        normalized,
        ncomponents = dims,
        subset_row = seq_len(nrow(normalized)),
        ntop = nrow(normalized),
        BSPARAM = BiocSingular::IrlbaParam()
      )
      if (is.list(pca)) pca <- pca$x
      row.names(pca) <- colnames(e)
      rm(normalized)
      invisible(gc(verbose = FALSE))
      pca
    }, error = function(err) {
      .scdblfinder_rng_restore(rng)
      fallback()
    })
  }

  fast_evaluate_knn <- function(pca, ctype, origins, expected = NULL, k,
                                BNPARAM = NULL) {
    fallback <- function() .scdblfinder_with_context_suspended(
      .orig_evaluateKNN(pca, ctype, origins, expected = expected, k = k,
                        BNPARAM = BNPARAM)
    )
    if (!.scdblfinder_context_active() || !is.null(expected) ||
        !is.null(BNPARAM) || length(k) < 1L || any(!is.finite(k)) ||
        any(k < 1L) || any(k %% 1 != 0)) {
      return(fallback())
    }
    rng <- .scdblfinder_rng_snapshot()
    tryCatch({
      knn <- suppressWarnings(BiocNeighbors::findKNN(
        as.matrix(pca), max(k), BNPARAM = BNPARAM
      ))
      hasOrigins <- length(unique(origins)) > 1L
      knn$type <- matrix(as.integer(ctype)[knn$index] - 1L,
                         nrow = nrow(knn$index))
      if (hasOrigins) {
        knn$orig <- matrix(origins[knn$index], nrow = nrow(knn[[1L]]))
      }
      if (any(w <- knn$distance == 0)) {
        knn$distance[w] <- min(knn$distance[knn$distance[, 1L] > 0, 1L])
      }
      md <- max(knn$distance[, 1L])
      dr <- t(vapply(seq_len(nrow(knn$distance)),
                    FUN.VALUE = numeric(2L), FUN = function(x) {
        w <- knn$type[x, ] == 1L
        dA <- ifelse(length(wA <- which(w)) == 0L, 2 * md,
                     knn$distance[x, wA[1L]])
        dB <- ifelse(length(wB <- which(!w)) == 0L, 2 * md,
                     knn$distance[x, wB[1L]])
        c(dA, dB)
      }))
      dw <- sqrt(max(k) - seq_len(max(k))) / knn$distance
      dw <- dw / rowSums(dw)
      d <- data.frame(
        row.names = row.names(pca), type = ctype, cluster = NA,
        weighted = rowSums(knn$type * dw),
        distanceToNearest = knn$distance[, 1L],
        distanceToNearestDoublet = dr[, 1L],
        distanceToNearestReal = dr[, 2L],
        nearestClass = knn$type[, 1L]
      )
      if (hasOrigins) {
        d <- cbind(d, utils::getFromNamespace(
          ".getMostLikelyOrigins", "scDblFinder"
        )(knn, origins))
      }
      for (ki in k) d[[paste0("ratio.k", ki)]] <- NA_real_
      running_ratio <- numeric(nrow(knn$type))
      last_k <- 0L
      for (ki in sort(unique(as.integer(k)))) {
        cols <- seq.int(last_k + 1L, ki)
        running_ratio <- running_ratio + rowSums(knn$type[, cols, drop = FALSE])
        d[[paste0("ratio.k", ki)]] <- running_ratio / ki
        last_k <- ki
      }
      list(knn = knn, d = d)
    }, error = function(err) {
      .scdblfinder_rng_restore(rng)
      fallback()
    })
  }

  fast_cxds2 <- function(x, whichDbls = c(), ntop = 500, binThresh = NULL) {
    fallback <- function() .scdblfinder_with_context_suspended(
      .orig_cxds2(x, whichDbls = whichDbls, ntop = ntop,
                  binThresh = binThresh)
    )
    if (!.scdblfinder_context_active() ||
        !.scdblfinder_is_exact_dgCMatrix(x) || anyNA(x@x) ||
        length(ntop) != 1L || !isTRUE(as.numeric(ntop) == 500) ||
        !is.null(binThresh)) {
      return(fallback())
    }
    rng <- .scdblfinder_rng_snapshot()
    tryCatch({
      pNonZero <- length(x@x) / length(x)
      if (pNonZero > 0.5) {
        pNonZero <- rowSums(x > 0) / ncol(x)
        x <- x[utils::head(order(pNonZero), ntop), ]
        binThresh <- max(
          1L,
          as.numeric(stats::quantile(x@x, mean(pNonZero) * 0.5))
        )
      } else {
        binThresh <- 1L
      }
      Bp <- x <- x >= binThresh
      ps <- rowMeans(x)
      if (nrow(x) > ntop) {
        hvg <- order(ps * (1 - ps), decreasing = TRUE)[seq_len(ntop)]
        Bp <- x <- x[hvg, ]
        ps <- ps[hvg]
      }
      if (length(whichDbls) > 0L) Bp <- Bp[, -whichDbls]
      prb <- outer(ps, 1 - ps)
      prb <- prb + t(prb)
      obs <- Bp %*% (1 - Matrix::t(Bp))
      obs <- obs + Matrix::t(obs)
      S <- suppressWarnings(stats::pbinom(
        as.matrix(obs) - 1, prob = prb, size = ncol(Bp),
        lower.tail = FALSE, log.p = TRUE
      ))
      if (all(S == 0)) return(rep(0L, ncol(x)))
      if (any(w <- is.infinite(S))) {
        smin <- min(S[!is.infinite(S)])
        S[S < smin] <- smin
      }
      s <- -Matrix::colSums(x * (S %*% x))
      s <- s - min(s)
      s / max(s)
    }, error = function(err) {
      .scdblfinder_rng_restore(rng)
      fallback()
    })
  }

  .scdblfinder_as_plain_df <- function(x) {
    as.data.frame(x, optional = TRUE, stringsAsFactors = FALSE)
  }

  .scdblfinder_assay_contract <- function(x, fingerprint_fn) {
    nms <- SummarizedExperiment::assayNames(x)
    stats::setNames(lapply(nms, function(nm) {
      assay <- SummarizedExperiment::assay(x, nm)
      fp <- fingerprint_fn(assay)
      list(
        class = paste(class(assay), collapse = "|"),
        dim = as.integer(dim(assay)),
        fingerprint = digest::digest(list(
          class = class(assay),
          dim = as.integer(dim(assay)),
          dimnames = dimnames(assay),
          serialized_size_bytes = fp$serialized_size_bytes,
          serialized_sha256 = fp$serialized_sha256
        ), algo = "sha256")
      )
    }), nms)
  }

  .scdblfinder_extract_output <- function(input, result, contract_env) {
    cd <- as.data.frame(SummarizedExperiment::colData(result))
    rd <- as.data.frame(SummarizedExperiment::rowData(result))
    input_cd <- .scdblfinder_as_plain_df(SummarizedExperiment::colData(input))
    input_rd <- .scdblfinder_as_plain_df(SummarizedExperiment::rowData(input))
    output_cd_all <- .scdblfinder_as_plain_df(SummarizedExperiment::colData(result))
    output_rd_all <- .scdblfinder_as_plain_df(SummarizedExperiment::rowData(result))
    original_cd_names <- colnames(input_cd)
    original_rd_names <- colnames(input_rd)
    output_original_cd <- if (all(original_cd_names %in% colnames(output_cd_all))) {
      output_cd_all[, original_cd_names, drop = FALSE]
    } else {
      output_cd_all[, intersect(original_cd_names, colnames(output_cd_all)), drop = FALSE]
    }
    output_original_rd <- if (all(original_rd_names %in% colnames(output_rd_all))) {
      output_rd_all[, original_rd_names, drop = FALSE]
    } else {
      output_rd_all[, intersect(original_rd_names, colnames(output_rd_all)), drop = FALSE]
    }
    scd_cd_names <- grep("^scDblFinder\\.", colnames(output_cd_all), value = TRUE)
    scd_rd_names <- grep("^scDblFinder\\.", colnames(output_rd_all), value = TRUE)
    scd_coldata <- output_cd_all[, scd_cd_names, drop = FALSE]
    scd_rowdata <- output_rd_all[, scd_rd_names, drop = FALSE]

    selected <- rd[["scDblFinder.selected"]]
    if (is.null(selected)) selected <- rep(NA, nrow(result))
    names(selected) <- rownames(result)

    score <- cd[["scDblFinder.score"]]
    names(score) <- rownames(cd)

    klass <- as.character(cd[["scDblFinder.class"]])
    names(klass) <- rownames(cd)

    weighted <- cd[["scDblFinder.weighted"]]
    if (!is.null(weighted)) names(weighted) <- rownames(cd)

    cxds <- cd[["scDblFinder.cxds_score"]]
    if (!is.null(cxds)) names(cxds) <- rownames(cd)

    list(
      cell_ids = rownames(cd),
      feature_ids = rownames(result),
      input_cell_ids = colnames(input),
      input_feature_ids = rownames(input),
      input_assays = .scdblfinder_assay_contract(
        input, contract_env$stream_serialized_fingerprint
      ),
      output_assays = .scdblfinder_assay_contract(
        result, contract_env$stream_serialized_fingerprint
      ),
      input_coldata = input_cd,
      output_original_coldata = output_original_cd,
      input_rowdata = input_rd,
      output_original_rowdata = output_original_rd,
      scdblfinder_coldata = scd_coldata,
      scdblfinder_rowdata = scd_rowdata,
      scdblfinder_col_fields = scd_cd_names,
      scdblfinder_row_fields = scd_rd_names,
      score = score,
      class = klass,
      weighted = weighted,
      cxds_score = cxds,
      selected_features = selected,
      threshold = S4Vectors::metadata(result)$scDblFinder.threshold,
      stats = S4Vectors::metadata(result)$scDblFinder.stats,
      full_sce_contract = contract_env$full_sce_contract(result),
      n_cells = ncol(result),
      n_features = nrow(result)
    )
  }

  register_patch(
    name = "scdblfinder",
    upstream = "scDblFinder",
    targets = list(
      scDblFinder = fast_scDblFinder,
      .defaultProcessing = fast_default_processing,
      .evaluateKNN = fast_evaluate_knn,
      cxds2 = fast_cxds2
    ),
    smoke = list(
      load = function(task_dir, tier) {
        task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
        ds <- Filter(function(d) identical(d$tier, tier), task$datasets)
        if (length(ds) == 0L) {
          stop("no dataset for tier '", tier, "' in task.yaml")
        }
        sce <- readRDS(resolve_dataset_path(task_dir, ds[[1L]]$path))
        if (!methods::is(sce, "SingleCellExperiment")) {
          stop("scDblFinder smoke input must be a SingleCellExperiment")
        }
        if (ncol(sce) > 33000L) {
          stop("scDblFinder smoke input exceeds task max_cells=33000")
        }
        contract_env <- new.env(parent = globalenv())
        sys.source(file.path(task_dir, "setup", "full_sce_contract.R"),
                   envir = contract_env)
        seed <- task$random_seeds$primary
        if (is.null(seed)) seed <- 42L
        set.seed(as.integer(seed))
        list(sce = sce, contract_env = contract_env)
      },
      call = function(inputs) {
        result <- scDblFinder::scDblFinder(
          inputs$sce,
          verbose = FALSE,
          BPPARAM = BiocParallel::SerialParam(progressbar = FALSE)
        )
        list(
          input = inputs$sce,
          result = result,
          contract_env = inputs$contract_env
        )
      },
      save = function(result, dir, tier = "small", ...) {
        output <- .scdblfinder_extract_output(
          result$input,
          result$result,
          result$contract_env
        )
        saveRDS(output, file.path(dir, "scdblfinder_output.rds"))
      }
    ),
    tested_against = "scDblFinder 1.27.6",
    tested_upstream_versions = list(scDblFinder = "1.27.6")
  )
}
