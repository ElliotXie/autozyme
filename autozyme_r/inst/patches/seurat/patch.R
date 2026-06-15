# Patch for Seurat.
#
# Phase 1 + 2: NormalizeData, FindVariableFeatures (Seurat + StdAssay + VST),
# ScaleData, FindNeighbors, FindAllMarkers.
#
# Subsequent phases will add RunPCA / RunCCA / Integration (Python-backed),
# and SCTransform.
#
# C++ kernels live in src/seurat_*.cpp and are exported as turbo_* in
# autozyme's namespace via Rcpp::compileAttributes.

if (requireNamespace("Seurat", quietly = TRUE) &&
    requireNamespace("SeuratObject", quietly = TRUE) &&
    requireNamespace("Matrix", quietly = TRUE)) {

  # ── Originals (captured at patch source time) ────────────────────────────
  .seurat_orig_NormalizeData_Seurat <- utils::getFromNamespace(
    "NormalizeData.Seurat", "Seurat")
  .seurat_orig_FindVariableFeatures_Seurat <- utils::getFromNamespace(
    "FindVariableFeatures.Seurat", "Seurat")
  .seurat_orig_FindVariableFeatures_StdAssay <- utils::getFromNamespace(
    "FindVariableFeatures.StdAssay", "Seurat")
  .seurat_orig_VST_dgCMatrix <- utils::getFromNamespace(
    "VST.dgCMatrix", "Seurat")
  .seurat_orig_ScaleData_Seurat <- utils::getFromNamespace(
    "ScaleData.Seurat", "Seurat")
  .seurat_orig_FindNeighbors_Seurat <- utils::getFromNamespace(
    "FindNeighbors.Seurat", "Seurat")
  .seurat_orig_FindAllMarkers <- utils::getFromNamespace(
    "FindAllMarkers", "Seurat")
  .seurat_orig_FindMarkers_Seurat <- utils::getFromNamespace(
    "FindMarkers.Seurat", "Seurat")
  # Translates the seurat-zyme back-compat `turbo=` kwarg into autozyme's
  # canonical `zyme=`. Returns the effective zyme flag.
  .seurat_zyme_flag <- function(zyme, turbo) {
    if (!is.null(turbo)) isTRUE(turbo) else isTRUE(zyme)
  }

  # ── NormalizeData ────────────────────────────────────────────────────────

  fast_NormalizeData_Seurat <- function(object,
                                        assay = NULL,
                                        normalization.method = "LogNormalize",
                                        scale.factor = 10000,
                                        margin = 1,
                                        block.size = NULL,
                                        verbose = TRUE,
                                        zyme = TRUE,
                                        turbo = NULL,
                                        ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    fallback <- function() {
      .seurat_orig_NormalizeData_Seurat(
        object, assay = assay, normalization.method = normalization.method,
        scale.factor = scale.factor, margin = margin, block.size = block.size,
        verbose = verbose, ...)
    }

    dots_call <- match.call(expand.dots = FALSE)$...
    fast_path_ok <- isTRUE(zyme) &&
      identical(normalization.method, "LogNormalize") &&
      is.numeric(scale.factor) && length(scale.factor) == 1L &&
      is.finite(scale.factor) &&
      is.numeric(margin) && length(margin) == 1L &&
      is.finite(margin) && margin == 1 &&
      is.null(block.size) &&
      length(dots_call) == 0L
    if (!fast_path_ok) {
      return(fallback())
    }

    assay <- if (is.null(assay)) {
      SeuratObject::DefaultAssay(object)
    } else {
      if (!is.character(assay) || length(assay) != 1L || is.na(assay)) {
        return(fallback())
      }
      assay
    }
    assays <- methods::slot(object, "assays")
    if (!assay %in% names(assays)) {
      return(fallback())
    }
    assay_obj <- assays[[assay]]
    assay_slots <- methods::slotNames(assay_obj)
    if (!inherits(assay_obj, "StdAssay") ||
        !all(c("layers", "cells", "features") %in% assay_slots)) {
      return(fallback())
    }
    counts_layers <- tryCatch(
      SeuratObject::Layers(assay_obj, search = "counts"),
      error = function(e) NULL)
    if (!identical(counts_layers, "counts")) {
      return(fallback())
    }
    layers <- methods::slot(assay_obj, "layers")
    counts <- layers[["counts"]]
    if (is.null(counts)) {
      return(fallback())
    }
    if (!inherits(counts, "dgCMatrix")) {
      counts <- tryCatch(
        methods::as(counts, "dgCMatrix"),
        error = function(e) NULL)
      if (is.null(counts)) {
        return(fallback())
      }
    }

    data_mat <- counts
    data_mat@x <- counts@x + 0  # deep copy @x; @i / @p stay shared (read-only)
    # scBLAS opt-in gate (Phase 5 integration). Default OFF preserves the
    # existing path bit-for-bit. Set AUTOZYME_SCBLAS_SPARSE_LOG1P=1 (or the
    # patch-scoped AUTOZYME_SEURAT_NORMALIZE_SCBLAS_SPARSE_LOG1P=1) to route
    # this normalization step through scblasR::sparse_log1p_normalize.
    # To revert: delete this `if/else` block and keep the bare call to
    # `seurat_log_normalize_dgc(data_mat, scale.factor, 100L)`.
    if (.az_feature_enabled("scblas_sparse_log1p",
                             patch = "seurat_normalize", default = FALSE) &&
        requireNamespace("scblasR", quietly = TRUE)) {
      tryCatch(
        scblasR::sparse_log1p_normalize(data_mat, scale = scale.factor,
                                         clone = FALSE, threads = 0L),
        error = function(e) seurat_log_normalize_dgc(data_mat,
                                                       scale.factor, 100L)
      )
    } else {
      seurat_log_normalize_dgc(data_mat, scale.factor, 100L)
    }
    layers[["data"]] <- data_mat
    methods::slot(assay_obj, "layers") <- layers

    cm <- methods::slot(assay_obj, "cells")
    if (!"data" %in% colnames(cm)) {
      new_col <- matrix(TRUE, nrow = nrow(cm), ncol = 1,
                        dimnames = list(rownames(cm), "data"))
      methods::slot(cm, ".Data") <- cbind(cm, new_col)
      methods::slot(assay_obj, "cells") <- cm
    }
    fm <- methods::slot(assay_obj, "features")
    if (!"data" %in% colnames(fm)) {
      new_col <- matrix(TRUE, nrow = nrow(fm), ncol = 1,
                        dimnames = list(rownames(fm), "data"))
      methods::slot(fm, ".Data") <- cbind(fm, new_col)
      methods::slot(assay_obj, "features") <- fm
    }
    assays[[assay]] <- assay_obj
    methods::slot(object, "assays") <- assays
    SeuratObject::LogSeuratCommand(object)
  }

  # ── FindVariableFeatures (3 dispatch entries) ────────────────────────────

  fast_VST_dgCMatrix <- function(data, nselect = 2000L, span = 0.3,
                                  clip = NULL, verbose = TRUE,
                                  zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    if (!isTRUE(zyme)) {
      return(.seurat_orig_VST_dgCMatrix(
        data = data, nselect = nselect, span = span,
        clip = clip, verbose = verbose, ...))
    }
    nfeatures <- nrow(data); n_cells <- ncol(data)
    mv <- turbo_FastSparseRowMeanVar(p = data@p, i = data@i, x = data@x,
                                      nrow = nfeatures, ncol = n_cells)
    mu <- mv$mean; variance <- mv$variance; nnz_per_row <- mv$nnz

    var_expected <- numeric(nfeatures)
    not.const <- variance > 0
    log_mean_nc <- log10(mu[not.const])
    log_var_nc <- log10(variance[not.const])
    fit_result <- stats:::simpleLoess(
      y = log_var_nc, x = matrix(log_mean_nc, ncol = 1),
      weights = rep.int(1, length(log_mean_nc)),
      span = span, degree = 2L, parametric = FALSE, drop.square = FALSE,
      normalize = FALSE, statistics = "approximate", surface = "interpolate",
      cell = 0.2, iterations = 1L, iterTrace = FALSE, trace.hat = "approximate")
    var_expected[not.const] <- 10 ^ fit_result$fitted

    sd_vec <- sqrt(var_expected)
    vmax <- if (is.null(clip)) sqrt(n_cells) else clip
    var_std <- turbo_FastSparseRowVarStd(
      p = data@p, i = data@i, x = data@x,
      nrow = nfeatures, ncol = n_cells,
      mu = mu, sd = sd_vec, vmax = vmax, nnzPerRow = nnz_per_row)

    hvf.info <- SeuratObject::EmptyDF(n = nfeatures)
    hvf.info$mean <- mu
    hvf.info$variance <- variance
    hvf.info$variance.expected <- var_expected
    hvf.info$variance.standardized <- var_std
    hvf.info$variable <- FALSE
    hvf.info$rank <- NA
    vf <- head(order(var_std, decreasing = TRUE), n = nselect)
    hvf.info$variable[vf] <- TRUE
    hvf.info$rank[vf] <- seq_along(vf)
    hvf.info
  }

  fast_FindVariableFeatures_StdAssay <- function(object, method = NULL,
                                                  nfeatures = 2000L,
                                                  layer = NULL, span = 0.3,
                                                  clip = NULL, key = NULL,
                                                  verbose = TRUE,
                                                  selection.method = "vst",
                                                  zyme = TRUE, turbo = NULL,
                                                  ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    if (!isTRUE(zyme) || !identical(selection.method, "vst")) {
      # upstream .StdAssay dispatches to VST.dgCMatrix (also patched) — wrap
      # to make sure the whole baseline call tree honors zyme=FALSE.
      return(with_disabled(.seurat_orig_FindVariableFeatures_StdAssay(
        object = object, method = method, nfeatures = nfeatures,
        layer = layer, span = span, clip = clip, key = key,
        verbose = verbose, selection.method = selection.method, ...)))
    }
    # Fast path needs a v5 Assay5 with a unified "counts" layer. Multi-layer
    # v5 assays (after `split()`) have counts.<group> instead; v3 Assay has no
    # layers slot at all. Either case → turbo kernel deref NULL. Defer.
    if (!inherits(object, "Assay5") ||
        !("counts" %in% SeuratObject::Layers(object))) {
      return(with_disabled(.seurat_orig_FindVariableFeatures_StdAssay(
        object = object, method = method, nfeatures = nfeatures,
        layer = layer, span = span, clip = clip, key = key,
        verbose = verbose, selection.method = selection.method, ...)))
    }
    data <- methods::slot(object, "layers")[["counts"]]
    hvf.info <- fast_VST_dgCMatrix(data, nselect = nfeatures, span = span,
                                    clip = clip, verbose = verbose)
    colnames(hvf.info) <- paste("vf_vst_counts", colnames(hvf.info), sep = "_")
    rownames(hvf.info) <- SeuratObject::Features(object, layer = "counts")
    object[["var.features"]] <- NULL
    object[["var.features.rank"]] <- NULL
    object[[names(hvf.info)]] <- NULL
    object[[names(hvf.info)]] <- hvf.info
    SeuratObject::VariableFeatures(object) <-
      SeuratObject::VariableFeatures(object, nfeatures = nfeatures, method = "vst")
    object
  }

  fast_FindVariableFeatures_Seurat <- function(object, assay = NULL,
                                                selection.method = "vst",
                                                loess.span = 0.3,
                                                clip.max = "auto",
                                                mean.function = NULL,
                                                dispersion.function = NULL,
                                                num.bin = 20,
                                                binning.method = "equal_width",
                                                nfeatures = 2000,
                                                mean.cutoff = c(0.1, 8),
                                                dispersion.cutoff = c(1, Inf),
                                                verbose = TRUE,
                                                zyme = TRUE, turbo = NULL,
                                                ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    if (!isTRUE(zyme) || !identical(selection.method, "vst")) {
      # upstream .Seurat → FVF.StdAssay → VST.dgCMatrix; all are patched.
      return(with_disabled(.seurat_orig_FindVariableFeatures_Seurat(
        object = object, assay = assay, selection.method = selection.method,
        loess.span = loess.span, clip.max = clip.max,
        mean.function = mean.function, dispersion.function = dispersion.function,
        num.bin = num.bin, binning.method = binning.method,
        nfeatures = nfeatures, mean.cutoff = mean.cutoff,
        dispersion.cutoff = dispersion.cutoff, verbose = verbose, ...)))
    }
    assay <- if (is.null(assay)) SeuratObject::DefaultAssay(object) else assay[1L]
    assay_obj <- object[[assay]]
    # Dispatcher routes everything to the StdAssay fast kernel, which assumes
    # a v5 Assay5 with a unified "counts" layer. v3 Assay (e.g. `pbmc_small`)
    # lacks the `layers` slot; v5 split assays expose only counts.<group>.
    # Both crash the turbo path → fall back to upstream which handles each.
    if (!inherits(assay_obj, "Assay5") ||
        !("counts" %in% SeuratObject::Layers(assay_obj))) {
      return(with_disabled(.seurat_orig_FindVariableFeatures_Seurat(
        object = object, assay = assay, selection.method = selection.method,
        loess.span = loess.span, clip.max = clip.max,
        mean.function = mean.function, dispersion.function = dispersion.function,
        num.bin = num.bin, binning.method = binning.method,
        nfeatures = nfeatures, mean.cutoff = mean.cutoff,
        dispersion.cutoff = dispersion.cutoff, verbose = verbose, ...)))
    }
    assay_obj <- fast_FindVariableFeatures_StdAssay(
      object = assay_obj,
      nfeatures = nfeatures,
      span = loess.span,
      clip = if (identical(clip.max, "auto")) NULL else clip.max,
      verbose = verbose)
    methods::slot(object, "assays")[[assay]] <- assay_obj
    object
  }

  # ── ScaleData ────────────────────────────────────────────────────────────

  fast_ScaleData_Seurat <- function(object, features = NULL, assay = NULL,
                                     vars.to.regress = NULL, split.by = NULL,
                                     model.use = "linear", use.umi = FALSE,
                                     do.scale = TRUE, do.center = TRUE,
                                     scale.max = 10, block.size = 1000,
                                     min.cells.to.block = 3000, verbose = TRUE,
                                     zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    # Fast path only handles the default scale+center+linear case.
    fast_path_ok <- isTRUE(zyme) && is.null(vars.to.regress) &&
                    is.null(split.by) && identical(model.use, "linear") &&
                    isFALSE(use.umi) && isTRUE(do.scale) && isTRUE(do.center)
    if (!fast_path_ok) {
      return(.seurat_orig_ScaleData_Seurat(
        object = object, features = features, assay = assay,
        vars.to.regress = vars.to.regress, split.by = split.by,
        model.use = model.use, use.umi = use.umi,
        do.scale = do.scale, do.center = do.center, scale.max = scale.max,
        block.size = block.size, min.cells.to.block = min.cells.to.block,
        verbose = verbose, ...))
    }
    assay_name <- if (is.null(assay)) SeuratObject::DefaultAssay(object) else assay[1L]
    assay_obj <- methods::slot(object, "assays")[[assay_name]]
    # Fast turbo_scale_sparse_full needs a v5 Assay5 with a unified "data"
    # layer. v3 Assay has no `layers` slot; v5 split assays expose data.<group>
    # instead. Both crash → fall back to upstream which handles multi-layer.
    if (!inherits(assay_obj, "Assay5") ||
        !("data" %in% SeuratObject::Layers(assay_obj))) {
      return(.seurat_orig_ScaleData_Seurat(
        object = object, features = features, assay = assay,
        vars.to.regress = vars.to.regress, split.by = split.by,
        model.use = model.use, use.umi = use.umi,
        do.scale = do.scale, do.center = do.center, scale.max = scale.max,
        block.size = block.size, min.cells.to.block = min.cells.to.block,
        verbose = verbose, ...))
    }
    data_mat <- methods::slot(assay_obj, "layers")[["data"]]
    features <- if (is.null(features)) SeuratObject::VariableFeatures(object) else features
    # Match upstream ScaleData.Assay: VariableFeatures(object) %||% rownames.
    if (length(features) == 0) {
      features <- rownames(assay_obj)
    }

    all_genes <- rownames(assay_obj)
    features <- intersect(features, all_genes)
    features <- features[order(match(features, all_genes))]
    idx <- match(features, all_genes) - 1L

    result <- turbo_scale_sparse_full(data_mat, idx, scale.max)
    dimnames(result) <- list(features, colnames(object))

    assay_obj@layers[["scale.data"]] <- result
    if (!"scale.data" %in% colnames(assay_obj@cells)) {
      cm <- assay_obj@cells
      methods::slot(assay_obj@cells, ".Data") <- cbind(cm,
        matrix(TRUE, nrow = nrow(cm), ncol = 1,
               dimnames = list(rownames(cm), "scale.data")))
      fm <- assay_obj@features
      methods::slot(assay_obj@features, ".Data") <- cbind(fm,
        matrix(rownames(fm) %in% features, nrow = nrow(fm), ncol = 1,
               dimnames = list(rownames(fm), "scale.data")))
    }
    methods::slot(object, "assays")[[assay_name]] <- assay_obj
    object
  }

  # ── FindNeighbors ────────────────────────────────────────────────────────

  fast_FindNeighbors_Seurat <- function(object, reduction = "pca",
                                         dims = 1:10, assay = NULL,
                                         features = NULL, k.param = 20,
                                         return.neighbor = FALSE,
                                         compute.SNN = !return.neighbor,
                                         prune.SNN = 1/15,
                                         nn.method = "annoy", n.trees = 50,
                                         annoy.metric = "euclidean", nn.eps = 0,
                                         verbose = TRUE, do.plot = FALSE,
                                         graph.name = NULL, l2.norm = FALSE,
                                         cache.index = FALSE,
                                         zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    fast_path_ok <- isTRUE(zyme) && inherits(object, "Seurat") &&
                    identical(nn.method, "annoy") &&
                    identical(annoy.metric, "euclidean") &&
                    isFALSE(return.neighbor) && isFALSE(l2.norm) &&
                    # Scope guard (additive): the fast path always builds the graph
                    # from Embeddings(reduction)[, dims] with a fixed annoy search,
                    # so params it does not honor fall back to upstream rather than
                    # being silently ignored. All default values pass unchanged.
                    is.null(features) && !is.null(dims) &&
                    isFALSE(cache.index) && isFALSE(do.plot) &&
                    isTRUE(nn.eps == 0)
    if (!fast_path_ok) {
      return(.seurat_orig_FindNeighbors_Seurat(
        object = object, reduction = reduction, dims = dims, assay = assay,
        features = features, k.param = k.param, return.neighbor = return.neighbor,
        compute.SNN = compute.SNN, prune.SNN = prune.SNN, nn.method = nn.method,
        n.trees = n.trees, annoy.metric = annoy.metric, nn.eps = nn.eps,
        verbose = verbose, do.plot = do.plot, graph.name = graph.name,
        l2.norm = l2.norm, cache.index = cache.index, ...))
    }
    assay <- SeuratObject::DefaultAssay(object[[reduction]])
    data.use <- SeuratObject::Embeddings(object[[reduction]])[, dims]
    cell.names <- rownames(data.use); n.cells <- nrow(data.use)
    n.cores <- max(1L, as.integer(Sys.getenv(
      "OMP_NUM_THREADS", parallel::detectCores())))

    nn.idx <- turbo_annoy_build_search(data.use, k.param, n.trees, n.cores)

    j <- as.numeric(t(nn.idx))
    i <- rep(seq_len(n.cells), each = k.param)
    nn.matrix <- Matrix::sparseMatrix(i = i, j = j, x = 1,
                                       dims = c(n.cells, n.cells),
                                       dimnames = list(cell.names, cell.names))
    nn.matrix <- methods::as(nn.matrix, "Graph")
    SeuratObject::DefaultAssay(nn.matrix) <- assay

    if (compute.SNN) {
      snn.matrix <- Seurat:::ComputeSNN(nn_ranked = nn.idx, prune = prune.SNN)
      rownames(snn.matrix) <- cell.names
      colnames(snn.matrix) <- cell.names
      snn.matrix <- SeuratObject::as.Graph(snn.matrix)
      SeuratObject::DefaultAssay(snn.matrix) <- assay
    }
    graph.name <- if (is.null(graph.name)) paste0(assay, "_", c("nn", "snn")) else graph.name
    object[[graph.name[1]]] <- nn.matrix
    if (compute.SNN && length(graph.name) >= 2) {
      object[[graph.name[2]]] <- snn.matrix
    }
    object
  }

  # ── FindAllMarkers ───────────────────────────────────────────────────────
  # Two implementations live behind the same Seurat::FindAllMarkers override.
  # The dispatcher `fast_FindAllMarkers` at the bottom routes between them
  # based on Sys.getenv("AUTOZYME_MODE"):
  #
  #   default (AUTOZYME_MODE != "for_paper"):
  #     fast_FindAllMarkers_fusion — ships to end users. One fused
  #     RcppParallel kernel computes nnz/expm1/rank/pval per feature in a
  #     single pass; fastest path on every dataset we measured. Kernel:
  #     `parallel_all_in_one_dgc` in src/seurat_markers.cpp.
  #
  #   AUTOZYME_MODE=for_paper:
  #     fast_FindAllMarkers_for_paper — V3 filter-then-rank pipeline whose
  #     numbers are reported in the paper. Computes pct/lfc first, then
  #     only ranks features that pass the gate. Kernels: `count_sum_by_group_dgc`
  #     + `portable_rank_dgc` + `rank_sum_by_group_dgc` in
  #     src/for_paper_markers.cpp. No presto dependency.
  #
  # Shared fast-path gating (test.use=='wilcox', slot=='data', no node /
  # latent.vars / mean.fxn / fc.name / densify / only.pos / max.cells.per.ident
  # / min.diff.pct > -Inf / base != 2 / extra args / group.by != 'ident') is
  # checked once in the dispatcher; both implementations skip the gate.

  fast_FindAllMarkers_fusion <- function(object, assay = NULL, features = NULL,
                                   group.by = NULL, logfc.threshold = 0.1,
                                   test.use = "wilcox", slot = "data",
                                   min.pct = 0.01, min.diff.pct = -Inf,
                                   node = NULL, verbose = TRUE,
                                   only.pos = FALSE,
                                   max.cells.per.ident = Inf, random.seed = 1,
                                   latent.vars = NULL, min.cells.feature = 3,
                                   min.cells.group = 3, mean.fxn = NULL,
                                   fc.name = NULL, base = 2,
                                   return.thresh = 1e-2, densify = FALSE,
                                   zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    fallback <- function() {
      .seurat_orig_FindAllMarkers(
        object = object, assay = assay, features = features, group.by = group.by,
        logfc.threshold = logfc.threshold, test.use = test.use, slot = slot,
        min.pct = min.pct, min.diff.pct = min.diff.pct, node = node,
        verbose = verbose, only.pos = only.pos,
        max.cells.per.ident = max.cells.per.ident, random.seed = random.seed,
        latent.vars = latent.vars, min.cells.feature = min.cells.feature,
        min.cells.group = min.cells.group, mean.fxn = mean.fxn, fc.name = fc.name,
        base = base, return.thresh = return.thresh, densify = densify, ...)
    }

    dots <- list(...)
    fast_path_ok <- isTRUE(zyme) &&
      identical(test.use, "wilcox") &&
      identical(slot, "data") &&
      is.null(features) &&
      is.null(node) &&
      is.null(latent.vars) &&
      is.null(mean.fxn) &&
      is.null(fc.name) &&
      !isTRUE(only.pos) &&
      !isTRUE(densify) &&
      is.infinite(max.cells.per.ident) &&
      identical(min.diff.pct, -Inf) &&
      isTRUE(base == 2) &&
      length(dots) == 0L &&
      (is.null(group.by) || identical(group.by, "ident"))
    if (!fast_path_ok) {
      return(fallback())
    }

    assay <- if (is.null(assay)) SeuratObject::DefaultAssay(object) else assay
    assay.obj <- object[[assay]]
    if (length(SeuratObject::Layers(assay.obj, search = slot)) > 1L) {
      stop(slot, " layers are not joined. Please run JoinLayers")
    }
    data.use <- assay.obj@layers[[slot]]
    if (is.null(data.use) || !inherits(data.use, "dgCMatrix")) {
      return(fallback())
    }
    dimnames(data.use) <- list(rownames(assay.obj), colnames(assay.obj))

    all.features <- rownames(data.use)
    all.cells <- colnames(data.use)
    cell.idents <- SeuratObject::Idents(object)
    if (is.null(names(cell.idents)) || !all(all.cells %in% names(cell.idents))) {
      return(fallback())
    }
    cell.idents <- cell.idents[all.cells]
    idents.all <- sort(unique(cell.idents))
    group.labels <- as.character(idents.all)
    grp.factor <- factor(as.character(cell.idents), levels = group.labels)
    group.sizes <- tabulate(as.integer(grp.factor), nbins = length(group.labels))
    # Small clusters (< min.cells.group) are skipped per-cluster inside the
    # main loop below (matching Seurat's own behavior — it warns + skips, not
    # errors). DO NOT fall back globally just because one cluster is too small,
    # otherwise huge datasets with rare clusters (e.g. heart_adult has a
    # 2-cell cluster) get pushed onto the slow baseline path.

    N <- length(all.cells)
    n.features <- length(all.features)
    if (is.null(fc.name)) fc.name <- paste0("avg_log", base, "FC")
    pseudocount.use <- 1
    n.groups <- length(idents.all)

    # Parallel C++ kernel: features × clusters matrices of pval / sum / detected.
    # (Worker computes pval in-kernel via R::pnorm5; no R-side z/pnorm loop.)
    #
    # scBLAS opt-in gate (Phase 5 integration). Default OFF preserves the
    # existing parallel_all_in_one_dgc (RcppParallel/tbb) path. Set
    # AUTOZYME_SCBLAS_WILCOX=1 (or AUTOZYME_SEURAT_MARKERS_SCBLAS_WILCOX=1)
    # to route through scblasR's wilcoxon_rank_sum (libomp + NEON polynomial
    # log; bench: 1.39-1.53× over autozyme RcppParallel-at-4t at threads=8
    # on 1k-5k feature configs). To revert: delete the `if/else` block,
    # keep only the `else` body.
    if (.az_feature_enabled("scblas_wilcox",
                             patch = "seurat_markers", default = FALSE) &&
        requireNamespace("scblasR", quietly = TRUE)) {
      sc.res <- scblasR::wilcoxon_rank_sum(
        mat       = data.use,
        groups    = grp.factor,
        n_groups  = length(group.sizes),
        threads   = 0L        # 0 = use omp_get_max_threads() (matches the
                              # behavior of autozyme's RcppParallel default)
      )
      sum.by.group       <- unname(sc.res$sum)
      detected.by.group  <- unname(sc.res$detected)
      pval.by.group      <- unname(sc.res$pval)
      rm(sc.res)
    } else {
      native.res <- parallel_all_in_one_dgc(
        x_sexp      = data.use,
        groups      = as.integer(grp.factor),   # 1-based, matches kernel
        group_sizes = as.integer(group.sizes)
      )
      sum.by.group       <- native.res$sum_by_group
      detected.by.group  <- native.res$detected_by_group
      pval.by.group      <- native.res$pval_by_group
      rm(native.res)
    }
    total.sum          <- rowSums(sum.by.group)
    total.detected     <- rowSums(detected.by.group)

    gde.list <- vector("list", n.groups)
    for (i in seq_len(n.groups)) {
      cl <- group.labels[i]
      n1 <- group.sizes[i]; n2 <- N - n1
      if (n1 < min.cells.group || n2 < min.cells.group) next
      sums.1   <- sum.by.group[, i]
      counts.1 <- detected.by.group[, i]
      counts.2 <- total.detected - counts.1
      fc <- log((sums.1 + pseudocount.use) / n1, base = base) -
            log((total.sum - sums.1 + pseudocount.use) / n2, base = base)
      pct.1.full <- round(counts.1 / n1, digits = 3)
      pct.2.full <- round(counts.2 / n2, digits = 3)
      pass <- (pmax(pct.1.full, pct.2.full) >= min.pct) &
              (abs(fc) >= logfc.threshold)
      feat.idx <- which(pass)
      if (length(feat.idx) == 0L) next
      de.results <- data.frame(
        p_val      = pval.by.group[feat.idx, i],
        fc[feat.idx],
        pct.1.full[feat.idx],
        pct.2.full[feat.idx],
        row.names  = all.features[feat.idx],
        check.names = FALSE
      )
      colnames(de.results) <- c("p_val", fc.name, "pct.1", "pct.2")
      de.results <- de.results[order(de.results$p_val,
                                      -abs(de.results$pct.1 - de.results$pct.2)), ,
                                drop = FALSE]
      de.results$p_val_adj <- p.adjust(de.results$p_val,
                                        method = "bonferroni", n = n.features)
      gde <- de.results[de.results$p_val < return.thresh, , drop = FALSE]
      if (nrow(gde) > 0) {
        gde$cluster <- factor(rep(cl, nrow(gde)), levels = group.labels)
        gde$gene <- rownames(gde)
        gde.list[[i]] <- gde
      }
    }
    gde.all <- do.call(rbind, gde.list[!vapply(gde.list, is.null, logical(1))])
    if (is.null(gde.all)) gde.all <- data.frame()
    if (nrow(gde.all) == 0L) {
      warning("No DE genes identified", call. = FALSE, immediate. = TRUE)
    }
    gde.all
  }

  # V3 filter-then-rank pipeline used for the paper's headline speedup. Kept
  # behind AUTOZYME_MODE=for_paper so end users still get the (faster)
  # fusion path by default. Kernels live in src/for_paper_markers.cpp — no
  # presto dependency; the three exports replace the presto:::nnzeroGroups /
  # sumGroups / rank_matrix / compute_ustat / compute_pval pipeline that the
  # standalone V3 prototype used.
  fast_FindAllMarkers_for_paper <- function(object, assay = NULL, features = NULL,
                                            group.by = NULL, logfc.threshold = 0.1,
                                            test.use = "wilcox", slot = "data",
                                            min.pct = 0.01, min.diff.pct = -Inf,
                                            node = NULL, verbose = TRUE,
                                            only.pos = FALSE,
                                            max.cells.per.ident = Inf, random.seed = 1,
                                            latent.vars = NULL, min.cells.feature = 3,
                                            min.cells.group = 3, mean.fxn = NULL,
                                            fc.name = NULL, base = 2,
                                            return.thresh = 1e-2, densify = FALSE,
                                            zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    fallback <- function() {
      .seurat_orig_FindAllMarkers(
        object = object, assay = assay, features = features, group.by = group.by,
        logfc.threshold = logfc.threshold, test.use = test.use, slot = slot,
        min.pct = min.pct, min.diff.pct = min.diff.pct, node = node,
        verbose = verbose, only.pos = only.pos,
        max.cells.per.ident = max.cells.per.ident, random.seed = random.seed,
        latent.vars = latent.vars, min.cells.feature = min.cells.feature,
        min.cells.group = min.cells.group, mean.fxn = mean.fxn, fc.name = fc.name,
        base = base, return.thresh = return.thresh, densify = densify, ...)
    }

    dots <- list(...)
    # Same gate as fusion so both modes benchmark the same supported scope.
    fast_path_ok <- isTRUE(zyme) &&
      identical(test.use, "wilcox") &&
      identical(slot, "data") &&
      is.null(features) &&
      is.null(node) &&
      is.null(latent.vars) &&
      is.null(mean.fxn) &&
      is.null(fc.name) &&
      !isTRUE(only.pos) &&
      !isTRUE(densify) &&
      is.infinite(max.cells.per.ident) &&
      identical(min.diff.pct, -Inf) &&
      isTRUE(base == 2) &&
      length(dots) == 0L &&
      (is.null(group.by) || identical(group.by, "ident"))
    if (!fast_path_ok) {
      return(fallback())
    }

    assay <- if (is.null(assay)) SeuratObject::DefaultAssay(object) else assay
    assay.obj <- object[[assay]]
    if (length(SeuratObject::Layers(assay.obj, search = slot)) > 1L) {
      stop(slot, " layers are not joined. Please run JoinLayers")
    }
    data.use <- assay.obj@layers[[slot]]
    if (is.null(data.use) || !inherits(data.use, "dgCMatrix")) {
      return(fallback())
    }
    dimnames(data.use) <- list(rownames(assay.obj), colnames(assay.obj))

    all.features <- rownames(data.use)
    all.cells <- colnames(data.use)
    cell.idents <- SeuratObject::Idents(object)
    if (is.null(names(cell.idents)) || !all(all.cells %in% names(cell.idents))) {
      return(fallback())
    }
    cell.idents <- cell.idents[all.cells]
    idents.all <- sort(unique(cell.idents))
    group.labels <- as.character(idents.all)
    grp.factor <- factor(as.character(cell.idents), levels = group.labels)
    group.sizes <- tabulate(as.integer(grp.factor), nbins = length(group.labels))

    N <- length(all.cells)
    n.features <- length(all.features)
    if (is.null(fc.name)) fc.name <- paste0("avg_log", base, "FC")
    pseudocount.use <- 1
    n.groups <- length(idents.all)
    sizes.rest <- N - group.sizes
    valid_mask <- group.sizes >= min.cells.group & sizes.rest >= min.cells.group

    # 1. per-(feature, cluster) nnz + expm1 sum — single C++ pass over X.
    native.cs <- count_sum_by_group_dgc(
      x_sexp  = data.use,
      groups  = as.integer(grp.factor),
      ngroups = as.integer(n.groups))
    nnz.by.group <- native.cs$nnz_by_group
    sum.by.group <- native.cs$sum_by_group
    total.nnz    <- rowSums(nnz.by.group)
    total.sum    <- rowSums(sum.by.group)
    rm(native.cs)

    # 2. pct + lfc + gate -> shrink feature set before ranking.
    pct.1.mat <- round(sweep(nnz.by.group, 2, group.sizes, "/"), 3)
    pct.2.mat <- round(sweep(total.nnz - nnz.by.group, 2, sizes.rest, "/"), 3)
    mean1.mat <- log(sweep(sum.by.group + pseudocount.use, 2, group.sizes, "/"),
                     base = base)
    mean2.mat <- log(sweep(total.sum - sum.by.group + pseudocount.use, 2,
                           sizes.rest, "/"), base = base)
    logfc.mat <- mean1.mat - mean2.mat

    alpha.min  <- pmax(pct.1.mat, pct.2.mat)
    pass.mat   <- alpha.min >= min.pct
    alpha.diff <- alpha.min - pmin(pct.1.mat, pct.2.mat)
    pass.mat   <- pass.mat & (alpha.diff >= min.diff.pct)
    if (only.pos) {
      pass.mat <- pass.mat & (logfc.mat >= logfc.threshold)
    } else {
      pass.mat <- pass.mat & (abs(logfc.mat) >= logfc.threshold)
    }
    for (k in which(!valid_mask)) pass.mat[, k] <- FALSE

    row.any   <- rowSums(pass.mat) > 0
    union.idx <- which(row.any)
    if (length(union.idx) == 0L) {
      warning("No DE genes identified", call. = FALSE, immediate. = TRUE)
      return(data.frame())
    }

    # 3. rank columns of the cells × |union| subset.
    data.subset <- data.use[union.idx, , drop = FALSE]
    t.data.sub  <- Matrix::t(data.subset)
    ranked      <- portable_rank_dgc(t.data.sub@x, t.data.sub@p,
                                     nrow(t.data.sub))
    t.data.sub@x <- ranked$x
    tie.sum      <- ranked$tie_sum

    # 4. rank-sum -> U -> z -> p (closed-form, all R after the two kernels).
    n1n2 <- as.numeric(group.sizes) * as.numeric(sizes.rest)
    rank.sum.mat <- rank_sum_by_group_dgc(
      x_sexp      = t.data.sub,
      groups      = as.integer(grp.factor),
      group_sizes = as.integer(group.sizes))
    ustat.mat <- sweep(rank.sum.mat, 2,
                       as.numeric(group.sizes) *
                         (as.numeric(group.sizes) + 1) / 2, "-")
    rhs.vec <- ((N^3 - N) - tie.sum) / (12 * (N^2 - N))
    z.mat   <- sweep(ustat.mat, 2, 0.5 * n1n2, "-")
    z.mat   <- z.mat - sign(z.mat) * 0.5
    sigma.mat <- sqrt(outer(rhs.vec, n1n2))
    z.mat   <- z.mat / sigma.mat
    pval.mat <- 2 * pnorm(-abs(z.mat))

    # 5. per-cluster output assembly (mirrors fusion's loop).
    gde.list <- vector("list", n.groups)
    for (i in seq_len(n.groups)) {
      if (!valid_mask[i]) next
      cl     <- group.labels[i]
      cl.idx <- which(pass.mat[, i])
      if (length(cl.idx) == 0L) next
      sub.row <- match(cl.idx, union.idx)
      de.results <- data.frame(
        p_val = pval.mat[sub.row, i],
        logfc.mat[cl.idx, i],
        pct.1.mat[cl.idx, i],
        pct.2.mat[cl.idx, i],
        row.names   = all.features[cl.idx],
        check.names = FALSE)
      colnames(de.results) <- c("p_val", fc.name, "pct.1", "pct.2")
      de.results <- de.results[order(de.results$p_val,
                                     -abs(de.results$pct.1 - de.results$pct.2)), ,
                               drop = FALSE]
      de.results$p_val_adj <- p.adjust(de.results$p_val,
                                       method = "bonferroni", n = n.features)
      gde <- de.results[de.results$p_val < return.thresh, , drop = FALSE]
      if (nrow(gde) > 0) {
        gde$cluster <- factor(rep(cl, nrow(gde)), levels = group.labels)
        gde$gene    <- rownames(gde)
        gde.list[[i]] <- gde
      }
    }
    gde.all <- do.call(rbind, gde.list[!vapply(gde.list, is.null, logical(1))])
    if (is.null(gde.all)) gde.all <- data.frame()
    if (nrow(gde.all) == 0L) {
      warning("No DE genes identified", call. = FALSE, immediate. = TRUE)
    }
    gde.all
  }

  # ── fast_FindAllMarkers_for_paper_omp ────────────────────────────────────
  # Produced autonomously by the AutoZyme framework's 6.3 cross-platform port
  # phase on Windows R 4.5.0 (clean-room autozyme_porttest experiment,
  # 2026-05-31). The dev-platform pipeline used `parallel::mclapply` for outer
  # chunking; on Windows that hard-errors (no fork). The agent built three
  # OpenMP-threaded native kernels — `omp_sum_nnz_expm1_groups`,
  # `omp_filter_pct_lfc`, `omp_subset_transpose_rank_pval` — in
  # `src/for_paper_markers_omp.cpp` that replace the presto:::* calls
  # in-process, bit-identical to the dev reference.
  fast_FindAllMarkers_for_paper_omp <- function(
      object, assay = NULL, features = NULL,
      logfc.threshold = 0.1, test.use = "wilcox", slot = "data",
      min.pct = 0.01, min.diff.pct = -Inf,
      verbose = TRUE, only.pos = FALSE,
      max.cells.per.ident = Inf, random.seed = 1,
      min.cells.feature = 3, min.cells.group = 3,
      base = 2, return.thresh = 1e-2, densify = FALSE,
      zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    if (!isTRUE(zyme)) {
      return(.seurat_orig_FindAllMarkers(
        object = object, assay = assay, features = features,
        logfc.threshold = logfc.threshold, test.use = test.use, slot = slot,
        min.pct = min.pct, min.diff.pct = min.diff.pct,
        verbose = verbose, only.pos = only.pos,
        max.cells.per.ident = max.cells.per.ident, random.seed = random.seed,
        min.cells.feature = min.cells.feature, min.cells.group = min.cells.group,
        base = base, return.thresh = return.thresh, densify = densify, ...))
    }
    if (test.use != "wilcox") {
      stop("fast_FindAllMarkers_for_paper_omp only supports test.use='wilcox'")
    }
    if (is.finite(max.cells.per.ident)) {
      stop("fast_FindAllMarkers_for_paper_omp does not support max.cells.per.ident")
    }
    dots <- list(...)
    if (!is.null(dots$group.by)) {
      stop("fast_FindAllMarkers_for_paper_omp does not support group.by")
    }
    if (!is.null(dots$node)) {
      stop("fast_FindAllMarkers_for_paper_omp does not support node")
    }

    assay <- assay %||% Seurat::DefaultAssay(object)
    assay_obj <- object[[assay]]
    data_layer <- if (inherits(assay_obj, "Assay5")) {
      assay_obj@layers[[slot]]
    } else {
      Seurat::GetAssayData(assay_obj, layer = slot)
    }
    n_genes_total <- nrow(data_layer)
    idents_all <- sort(unique(Seurat::Idents(object)))

    all_cells <- colnames(assay_obj)
    idents_vec <- Seurat::Idents(object)[all_cells]
    cell_cluster <- as.character(idents_vec)
    n_total <- length(all_cells)

    all_gene_names <- rownames(assay_obj)
    if (is.null(features)) {
      features <- all_gene_names
    } else {
      feat_idx <- match(features, all_gene_names)
      if (any(is.na(feat_idx))) {
        features <- features[!is.na(feat_idx)]
        feat_idx <- feat_idx[!is.na(feat_idx)]
      }
      data_layer <- data_layer[feat_idx, , drop = FALSE]
      n_genes_total <- nrow(data_layer)
    }

    cluster_levels <- as.character(idents_all)
    n_clusters <- length(cluster_levels)
    y_factor <- factor(cell_cluster, levels = cluster_levels)
    cluster_sizes <- as.numeric(table(y_factor))
    sizes_rest <- n_total - cluster_sizes
    valid_mask <- cluster_sizes >= min.cells.group & sizes_rest >= min.cells.group
    valid_clusters <- which(valid_mask)
    n1n2 <- cluster_sizes * sizes_rest

    # Step 1: per-(feature, group) nnz count + expm1 sum, single OMP pass.
    grp_res <- omp_sum_nnz_expm1_groups(data_layer, as.integer(y_factor),
                                        n_clusters)
    nnz_mat <- grp_res$nnz
    expm1_sums <- grp_res$sum_expm1
    rm(grp_res)
    total_nnz <- rowSums(nnz_mat)
    total_expm1 <- rowSums(expm1_sums)

    # Step 2: per-(feature, group) pct1/pct2/lfc + per-feature row_any.
    flt <- omp_filter_pct_lfc(nnz_mat, expm1_sums, total_nnz, total_expm1,
                              cluster_sizes, sizes_rest, valid_mask,
                              min.pct, min.diff.pct, logfc.threshold,
                              only.pos, base)
    pct1_mat <- flt$pct1
    pct2_mat <- flt$pct2
    logfc_mat <- flt$lfc
    pass_mat <- flt$pass
    row_any <- flt$row_any
    rm(flt)
    rownames(pct1_mat) <- features
    rownames(pct2_mat) <- features
    rownames(logfc_mat) <- features

    union_idx <- which(row_any)
    all_union <- features[union_idx]

    if (length(all_union) == 0) {
      warning("No DE genes identified", call. = FALSE, immediate. = TRUE)
      return(data.frame())
    }

    # Step 3: CSR transpose + row-subset + rank + ustat + pval, fused OMP.
    pval_mat <- omp_subset_transpose_rank_pval(
      data_layer, as.integer(union_idx), as.integer(y_factor),
      n_clusters, cluster_sizes)
    colnames(pval_mat) <- cluster_levels
    rownames(pval_mat) <- all_union

    res_pval <- numeric(0); res_logfc <- numeric(0)
    res_pct1 <- numeric(0); res_pct2 <- numeric(0)
    res_padj <- numeric(0); res_cluster <- integer(0); res_gene <- character(0)

    for (ki in seq_along(valid_clusters)) {
      k <- valid_clusters[ki]
      cl_feats <- features[pass_mat[, k]]
      if (length(cl_feats) == 0) next
      pv <- pval_mat[cl_feats, k]
      lfc <- logfc_mat[cl_feats, k]
      p1 <- pct1_mat[cl_feats, k]
      p2 <- pct2_mat[cl_feats, k]
      if (only.pos) {
        keep <- lfc > 0
        pv <- pv[keep]; lfc <- lfc[keep]; p1 <- p1[keep]; p2 <- p2[keep]
        cl_feats <- cl_feats[keep]
      }
      ord <- order(pv, -abs(p1 - p2))
      pv <- pv[ord]; lfc <- lfc[ord]; p1 <- p1[ord]; p2 <- p2[ord]
      cl_feats <- cl_feats[ord]
      padj <- p.adjust(pv, method = "bonferroni", n = n_genes_total)
      keep <- pv < return.thresh
      if (any(keep)) {
        res_pval <- c(res_pval, pv[keep])
        res_logfc <- c(res_logfc, lfc[keep])
        res_pct1 <- c(res_pct1, p1[keep])
        res_pct2 <- c(res_pct2, p2[keep])
        res_padj <- c(res_padj, padj[keep])
        res_cluster <- c(res_cluster, rep(k, sum(keep)))
        res_gene <- c(res_gene, cl_feats[keep])
      }
    }

    if (length(res_gene) == 0) {
      warning("No DE genes identified", call. = FALSE, immediate. = TRUE)
      return(data.frame())
    }

    data.frame(
      p_val = res_pval, avg_log2FC = res_logfc,
      pct.1 = res_pct1, pct.2 = res_pct2,
      p_val_adj = res_padj,
      cluster = idents_all[res_cluster],
      gene = res_gene,
      row.names = make.unique(res_gene),
      stringsAsFactors = FALSE)
  }

  # Dispatcher — Sys.getenv("AUTOZYME_MODE") chooses the implementation.
  # Default (any value other than "for_paper" / "for_paper_omp",
  # case-insensitive): fusion (ships to end users).
  #   AUTOZYME_MODE=for_paper      → V3 filter-then-rank (count_sum + portable
  #                                   rank + rank_sum kernels). Original
  #                                   paper-reported speed.
  #   AUTOZYME_MODE=for_paper_omp  → V3 path rebuilt by framework's 6.3 port
  #                                   phase using OMP-threaded kernels.
  #
  # Platform note for for_paper_omp: the OMP kernels were authored on Windows
  # (no fork). Raw OpenMP thread-spawn SEGFAULTs inside R on macOS arm64
  # (homebrew libomp + R; crashes the moment >1 worker thread is created,
  # independent of the kernel body). Linux + Windows R run raw OpenMP fine.
  # On macOS we therefore fall back to the single-thread `for_paper` kernel —
  # which is numerically identical AND is exactly how the macOS markers numbers
  # in the paper were produced (find_all_markers/v3 iterated entirely at
  # thread=1; the ~150x speedup is algorithmic, not from parallelism).
  fast_FindAllMarkers <- function(object, ...) {
    mode <- tolower(Sys.getenv("AUTOZYME_MODE", unset = ""))
    if (identical(mode, "for_paper")) {
      fast_FindAllMarkers_for_paper(object, ...)
    } else if (identical(mode, "for_paper_omp")) {
      if (identical(Sys.info()[["sysname"]], "Darwin")) {
        if (!isTRUE(getOption("autozyme.markers_omp_mac_notice"))) {
          options(autozyme.markers_omp_mac_notice = TRUE)
          message("autozyme: AUTOZYME_MODE=for_paper_omp on macOS -> using the ",
                  "single-thread for_paper kernel (raw OpenMP segfaults in R on ",
                  "macOS arm64; result is identical). Use Linux/Windows for the ",
                  "OMP multi-thread path.")
        }
        fast_FindAllMarkers_for_paper(object, ...)
      } else {
        fast_FindAllMarkers_for_paper_omp(object, ...)
      }
    } else {
      fast_FindAllMarkers_fusion(object, ...)
    }
  }

  # ── FindMarkers (one-vs-rest + two-group) ────────────────────────────────
  # Generalizes the FindAllMarkers fusion kernel to G=2 via the same compiled
  # `parallel_all_in_one_dgc` sweep with a 2-level factor:
  #   * ident.2 = NULL  -> one-vs-rest (ident.1 vs all other cells, full matrix)
  #   * ident.2 = level(s) -> two-group (subset to ident.1 cells + ident.2
  #     level(s), ident.1 vs ident.2)
  # Bit-exact vs Seurat/presto on shared genes (validated both shapes incl.
  # multi-level ident.2: neglog10_p spearman = 1.0, q99 |avg_log2FC| diff
  # < 1e-15, top50 jaccard = 1.0). This also accelerates FindConservedMarkers,
  # which calls FindMarkers per grouping.var level with an explicit ident.2.
  # Any other shape (group.by/subset.ident/reduction/latent.vars, non-wilcox
  # test, only.pos, non-default base/min.diff.pct/max.cells.per.ident, explicit
  # zyme=FALSE) falls back to the original method untouched.
  fast_FindMarkers_Seurat <- function(object, ident.1 = NULL, ident.2 = NULL,
                                      latent.vars = NULL, group.by = NULL,
                                      subset.ident = NULL, assay = NULL,
                                      reduction = NULL, ...) {
    dots <- list(...)
    fallback <- function() {
      .seurat_orig_FindMarkers_Seurat(
        object = object, ident.1 = ident.1, ident.2 = ident.2,
        latent.vars = latent.vars, group.by = group.by,
        subset.ident = subset.ident, assay = assay, reduction = reduction, ...)
    }

    test.use     <- dots[["test.use"]]            %||% "wilcox"
    slot         <- dots[["slot"]]                %||% "data"
    logfc.thr    <- dots[["logfc.threshold"]]     %||% 0.1
    min.pct      <- dots[["min.pct"]]             %||% 0.01
    base         <- dots[["base"]]                %||% 2
    only.pos     <- dots[["only.pos"]]            %||% FALSE
    min.diff.pct <- dots[["min.diff.pct"]]        %||% -Inf
    mcpi         <- dots[["max.cells.per.ident"]] %||% Inf
    mcg          <- dots[["min.cells.group"]]     %||% 3
    mcf          <- dots[["min.cells.feature"]]   %||% 3
    pseudocount.use <- dots[["pseudocount.use"]]  %||% 1

    unsupported <- isFALSE(dots[["zyme"]]) || isFALSE(dots[["turbo"]]) ||
      !is.null(latent.vars) || !is.null(group.by) ||
      !is.null(subset.ident) || !is.null(reduction) ||
      !is.null(dots[["features"]]) || !is.null(dots[["mean.fxn"]]) ||
      !is.null(dots[["fc.name"]]) ||
      !identical(test.use, "wilcox") || !identical(slot, "data") ||
      isTRUE(only.pos) || base != 2 || is.finite(mcpi) || min.diff.pct > -Inf ||
      is.null(ident.1) || length(ident.1) != 1L ||
      # min.cells.group is enforced here as the hard-coded `< 3` fallback below;
      # min.cells.feature pre-filters features in stock Seurat but not here.
      # A non-default value of either changes the result (or turns Seurat's
      # too-few-cells stop() into a silent result), so defer to upstream.
      !identical(as.numeric(mcg), 3) || !identical(as.numeric(mcf), 3) ||
      # pseudocount.use feeds log() directly; a non-finite or non-scalar value
      # would silently corrupt every fold-change. (A valid positive scalar,
      # incl. the default 1, is supported.)
      !is.numeric(pseudocount.use) || length(pseudocount.use) != 1L ||
      !is.finite(pseudocount.use) || pseudocount.use <= 0
    if (unsupported) return(fallback())

    assay <- assay %||% SeuratObject::DefaultAssay(object)
    assay.obj <- object[[assay]]
    if (length(SeuratObject::Layers(assay.obj, search = slot)) > 1) return(fallback())
    data.use <- SeuratObject::LayerData(object, layer = slot, assay = assay)
    if (is.null(data.use) || !inherits(data.use, "dgCMatrix")) return(fallback())

    idents <- SeuratObject::Idents(object)
    cellnames <- colnames(data.use)
    if (is.null(names(idents)) || !all(cellnames %in% names(idents))) return(fallback())
    idents <- as.character(idents[cellnames])
    if (!(as.character(ident.1) %in% idents)) return(fallback())

    in1 <- idents == as.character(ident.1)
    if (is.null(ident.2)) {
      # one-vs-rest: ident.1 vs all other cells (full matrix)
      gf <- factor(ifelse(in1, "x", "rest"), levels = c("x", "rest"))
    } else {
      # two-group: restrict to ident.1 cells + ident.2 level(s)
      ident.2.chr <- as.character(ident.2)
      if (!all(ident.2.chr %in% idents)) return(fallback())
      in2 <- idents %in% ident.2.chr
      if (any(in1 & in2)) return(fallback())  # idents must be disjoint
      use <- in1 | in2
      data.use <- data.use[, use, drop = FALSE]
      gf <- factor(ifelse(in1[use], "x", "rest"), levels = c("x", "rest"))
    }
    gsizes <- tabulate(as.integer(gf), nbins = 2L)
    n.1 <- gsizes[1]; n.2 <- gsizes[2]
    if (n.1 < 3L || n.2 < 3L) return(fallback())

    feature.names <- rownames(data.use)
    n.features.total <- nrow(data.use)
    res <- parallel_all_in_one_dgc(data.use, as.integer(gf), gsizes)

    # Match FoldChange / fast_FindAllMarkers_fusion exactly: honor
    # pseudocount.use (read + validated above) instead of hard-coding +1, and
    # gate min.pct on the ROUNDED pct (pmax(round(count/n,3)) >= min.pct), not
    # raw counts — raw counts are stricter at the boundary and would drop genes
    # Seurat and the fusion path keep.
    sums.1 <- res$sum_by_group[, 1]
    total.sum <- res$sum_by_group[, 1] + res$sum_by_group[, 2]
    counts.1 <- res$detected_by_group[, 1]
    counts.2 <- res$detected_by_group[, 2]
    fc <- log((sums.1 + pseudocount.use) / n.1, base = base) -
          log((total.sum - sums.1 + pseudocount.use) / n.2, base = base)
    pct.1 <- round(counts.1 / n.1, 3)
    pct.2 <- round(counts.2 / n.2, 3)

    features.use <- which(pmax(pct.1, pct.2) >= min.pct & abs(fc) >= logfc.thr)
    if (length(features.use) == 0L) return(fallback())

    out <- data.frame(
      p_val      = res$pval_by_group[features.use, 1],
      avg_log2FC = fc[features.use],
      pct.1      = pct.1[features.use],
      pct.2      = pct.2[features.use],
      row.names  = feature.names[features.use], check.names = FALSE)
    out <- out[order(out$p_val, -abs(out$pct.1 - out$pct.2)), , drop = FALSE]
    out$p_val_adj <- p.adjust(out$p_val, method = "bonferroni", n = n.features.total)
    out
  }

  # ── Phase 3: Python-backed patches (RunPCA, RunCCA, Integration) ─────────
  #
  # The PCA / CCA fast paths offload SVD / partial eigendecomposition to
  # NumPy + SciPy via reticulate. Python initialization is **lazy**: we do
  # not require Python at activate('seurat') time; we try to bring it up on
  # the first PCA/CCA fast-path call. If Python (numpy + scipy) cannot be
  # located, the patch silently delegates to upstream Seurat — single-cell
  # users who never call PCA/CCA never pay the Python tax.

  .seurat_py_state <- new.env(parent = emptyenv())
  .seurat_py_state$ready  <- FALSE
  .seurat_py_state$failed <- FALSE  # once failed, don't retry per-call
  .seurat_py_state$threadpoolctl <- NULL

  # Discovery + binding of a numpy/scipy-capable Python lives in the package
  # (R/python_env.R: .az_py_bind) -- a dedicated reproducible virtualenv,
  # env-var overrides (AUTOZYME_PYTHON / RETICULATE_PYTHON), and a numpy probe
  # *before* binding, with NO hardcoded conda env names. Here we only memoize
  # per session and emit the one actionable fallback message.
  .seurat_init_python <- function() {
    if (isTRUE(.seurat_py_state$ready))  return(TRUE)
    if (isTRUE(.seurat_py_state$failed)) return(FALSE)
    if (isTRUE(.az_py_bind())) {
      .seurat_py_state$ready <- TRUE
      return(TRUE)
    }
    err <- .az_py_state$err
    message(sprintf(
      paste0("[autozyme] Seurat PCA/CCA fast path needs Python (numpy+scipy); ",
             "falling back to upstream. (%s) Run ",
             "autozyme::install_python_deps() to enable."),
      if (length(err) == 0L || is.na(err)) "unavailable" else err))
    .seurat_py_state$failed <- TRUE
    FALSE
  }

  # Activation hook (runs once, at first activate("seurat")): warm Python up now
  # if the env is ready -- so the first RunPCA/RunCCA pays zero startup cost --
  # else offer to set it up (interactive) or print one actionable note.
  .seurat_on_activate <- function() {
    .az_py_warmup_or_notify("seurat", "PCA/CCA/integration")
    invisible(NULL)
  }

  .seurat_python_thread_guard <- function() {
    ctl_mod <- .seurat_py_state$threadpoolctl
    if (is.null(ctl_mod)) {
      ctl_mod <- tryCatch(
        reticulate::import("threadpoolctl", convert = FALSE),
        error = function(e) FALSE)
      .seurat_py_state$threadpoolctl <- ctl_mod
    }
    if (identical(ctl_mod, FALSE)) return(NULL)
    guard <- tryCatch(
      ctl_mod$threadpool_limits(limits = as.integer(1)),
      error = function(e) NULL)
    if (!is.null(guard)) guard$`__enter__`()
    guard
  }

  # ── RunPCA ───────────────────────────────────────────────────────────────

  # TRUE when the compiled native PCA/CCA kernel + a fast BLAS backend resolve
  # (macOS Accelerate / OpenBLAS via dlopen). Honors AUTOZYME_NATIVE_PCA=0
  # (checked C++-side). Old installs without the kernel return FALSE -> scipy.
  .seurat_native_ok <- function() {
    exists("native_pca_available", envir = asNamespace("autozyme"),
           inherits = FALSE) &&
      isTRUE(tryCatch(native_pca_available(), error = function(e) FALSE))
  }

  fast_RunPCA_default <- function(object, assay = NULL, npcs = 50,
                                   rev.pca = FALSE, weight.by.var = TRUE,
                                   verbose = TRUE, ndims.print = 1:5,
                                   nfeatures.print = 30,
                                   reduction.key = "PC_", seed.use = 42,
                                   approx = TRUE,
                                   .feature.names = rownames(object),
                                   .cell.names = colnames(object),
                                   zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    # Scope guard: approx=FALSE asks for the exact (prcomp) decomposition; the
    # Gram+eigh fast path does not honor it, so fall back. Additive — for the
    # default approx=TRUE call this is FALSE and the fast path runs unchanged.
    if (!isTRUE(zyme) || isTRUE(rev.pca) || !isTRUE(approx)) {
      return(.seurat_orig_RunPCA_default(
        object = object, assay = assay, npcs = npcs, rev.pca = rev.pca,
        weight.by.var = weight.by.var, verbose = verbose,
        ndims.print = ndims.print, nfeatures.print = nfeatures.print,
        reduction.key = reduction.key, seed.use = seed.use, approx = approx,
        ...))
    }
    if (!is.null(seed.use)) set.seed(seed = seed.use)
    npcs <- min(npcs, nrow(object) - 1)
    nfeatures <- nrow(object); n <- ncol(object)

    # Native, Python-free fast path (macOS Accelerate / OpenBLAS via dlopen):
    # Gram + partial eigh, bit-exact to the scipy path up to sign. Falls through
    # to scipy on any error or when no fast BLAS is found.
    if (.seurat_native_ok()) {
      .native_res <- tryCatch({
        obj <- object
        if (!is.matrix(obj)) obj <- as.matrix(obj)
        storage.mode(obj) <- "double"
        nv <- native_pca_run(obj, as.integer(npcs), isTRUE(weight.by.var))
        feature.loadings <- nv$loadings
        cell.embeddings  <- nv$embeddings
        rownames(feature.loadings) <- .feature.names
        colnames(feature.loadings) <- paste0(reduction.key, 1:npcs)
        rownames(cell.embeddings)  <- .cell.names
        colnames(cell.embeddings)  <- colnames(feature.loadings)
        SeuratObject::CreateDimReducObject(
          embeddings = cell.embeddings, loadings = feature.loadings,
          assay = assay, stdev = as.numeric(nv$sdev), key = reduction.key)
      }, error = function(e) NULL)
      if (!is.null(.native_res)) return(.native_res)
    }

    # scipy fast path (fallback). If Python is unavailable too, use upstream.
    if (!.seurat_init_python()) {
      return(.seurat_orig_RunPCA_default(
        object = object, assay = assay, npcs = npcs, rev.pca = rev.pca,
        weight.by.var = weight.by.var, verbose = verbose,
        ndims.print = ndims.print, nfeatures.print = nfeatures.print,
        reduction.key = reduction.key, seed.use = seed.use, approx = approx,
        ...))
    }
    np <- reticulate::import("numpy", convert = FALSE)
    X_py <- np$asarray(object, dtype = "float64")
    XtX  <- X_py$`__matmul__`(np$ascontiguousarray(np$transpose(X_py)))

    if (Sys.info()[["sysname"]] == "Darwin") {
      # macOS Accelerate BLAS segfaults inside scipy eigh's partial path;
      # use numpy.linalg.eigh (full) and slice the top npcs.
      np_linalg  <- reticulate::import("numpy.linalg", convert = FALSE)
      eig_result <- np_linalg$eigh(XtX)
      all_eigvals_py <- eig_result[[0]]
      all_eigvecs_py <- eig_result[[1]]
      idx_start <- as.integer(nfeatures - npcs)
      eigvals_py <- all_eigvals_py[reticulate::py_eval(
        sprintf("slice(%d, None)", idx_start), convert = FALSE)]
      eigvecs_py <- all_eigvecs_py[, reticulate::py_eval(
        sprintf("slice(%d, None)", idx_start), convert = FALSE)]
    } else {
      scipy_linalg <- reticulate::import("scipy.linalg", convert = FALSE)
      eig_result <- scipy_linalg$eigh(
        XtX,
        subset_by_index = reticulate::tuple(
          as.integer(nfeatures - npcs), as.integer(nfeatures - 1L)),
        overwrite_a = TRUE, driver = "evr")
      eigvals_py <- eig_result[[0]]
      eigvecs_py <- eig_result[[1]]
    }
    eigvals     <- as.numeric(reticulate::py_to_r(np$flip(eigvals_py)$copy()))
    eigvecs_rev <- np$ascontiguousarray(np$flip(eigvecs_py, axis = 1L))
    cell_embed_py <- np$ascontiguousarray(np$transpose(X_py))$`__matmul__`(eigvecs_rev)
    feature.loadings <- as.matrix(reticulate::py_to_r(eigvecs_rev))
    cell.embeddings  <- as.matrix(reticulate::py_to_r(cell_embed_py))

    d <- sqrt(pmax(eigvals, 0))
    sdev <- as.numeric(d / sqrt(max(1, n - 1)))
    if (!weight.by.var) cell.embeddings <- sweep(cell.embeddings, 2, d, `/`)

    rownames(feature.loadings) <- .feature.names
    colnames(feature.loadings) <- paste0(reduction.key, 1:npcs)
    rownames(cell.embeddings)  <- .cell.names
    colnames(cell.embeddings)  <- colnames(feature.loadings)
    SeuratObject::CreateDimReducObject(
      embeddings = cell.embeddings, loadings = feature.loadings,
      assay = assay, stdev = sdev, key = reduction.key)
  }

  fast_RunPCA_StdAssay <- function(object, assay = NULL, features = NULL,
                                    layer = "scale.data", npcs = 50,
                                    rev.pca = FALSE, weight.by.var = TRUE,
                                    verbose = TRUE, ndims.print = 1:5,
                                    nfeatures.print = 30,
                                    reduction.key = "PC_", seed.use = 42,
                                    zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    # Scope guard: the fast path extracts `layer` as-is and feeds it to the
    # Gram+eigh kernel (validated only on the centered+scaled "scale.data"
    # layer), and does not honor approx=FALSE (exact prcomp). A non-default
    # layer or approx=FALSE falls back to the FULL upstream StdAssay chain HERE
    # (not at the .default layer, which only receives a bare matrix and would
    # error on missing dimnames), so the output contract is preserved. The
    # default layer="scale.data" / approx=TRUE call is FALSE here (unchanged).
    if (!isTRUE(zyme) ||
        (!is.null(layer) && !identical(as.character(layer), "scale.data")) ||
        isFALSE(list(...)[["approx"]])) {
      # upstream .StdAssay → RunPCA.default (patched); wrap to honor zyme=FALSE
      return(with_disabled(.seurat_orig_RunPCA_StdAssay(
        object = object, assay = assay, features = features, layer = layer,
        npcs = npcs, rev.pca = rev.pca, weight.by.var = weight.by.var,
        verbose = verbose, ndims.print = ndims.print,
        nfeatures.print = nfeatures.print, reduction.key = reduction.key,
        seed.use = seed.use, ...)))
    }
    layer <- SeuratObject::Layers(object, search = layer)
    data.use <- methods::slot(object, "layers")[[layer]]
    feature.names <- SeuratObject::Features(object, layer = layer)
    cell.names <- colnames(object)
    if (is.null(features)) features <- SeuratObject::VariableFeatures(object)
    if (!isTRUE(setequal(features, feature.names))) {
      features <- features[!is.na(features)]
      idx <- match(features, feature.names, nomatch = 0L)
      data.use <- data.use[idx[idx > 0L], , drop = FALSE]
      feature.names <- feature.names[idx[idx > 0L]]
    }
    # Defer to the upstream S3 dispatcher; with the patch live this hits
    # fast_RunPCA_default via the namespace rebind.
    Seurat::RunPCA(
      object = data.use, assay = assay, npcs = npcs, rev.pca = rev.pca,
      weight.by.var = weight.by.var, verbose = verbose,
      ndims.print = ndims.print, nfeatures.print = nfeatures.print,
      reduction.key = reduction.key, seed.use = seed.use,
      .feature.names = feature.names, .cell.names = cell.names, ...)
  }

  # ── RunCCA ───────────────────────────────────────────────────────────────

  fast_RunCCA_default <- function(object1, object2, standardize = TRUE,
                                   num.cc = 20, seed.use = 42, verbose = FALSE,
                                   zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    if (!isTRUE(zyme)) {
      return(.seurat_orig_RunCCA_default(
        object1 = object1, object2 = object2, standardize = standardize,
        num.cc = num.cc, seed.use = seed.use, verbose = verbose, ...))
    }
    if (!is.null(seed.use)) set.seed(seed.use)
    cells1 <- colnames(object1); cells2 <- colnames(object2)
    if (standardize) {
      object1 <- Seurat:::Standardize(mat = object1, display_progress = FALSE)
      object2 <- Seurat:::Standardize(mat = object2, display_progress = FALSE)
    }
    if (inherits(object1, "sparseMatrix")) object1 <- as.matrix(object1)
    if (inherits(object2, "sparseMatrix")) object2 <- as.matrix(object2)

    # Native, Python-free CCA SVD: form A = X1^T X2 on a fast BLAS + irlba Krylov
    # top-k. Matches scipy svds; falls through to scipy on error / no BLAS / no
    # irlba. object1/object2 are already standardized above.
    if (.seurat_native_ok() && requireNamespace("irlba", quietly = TRUE)) {
      .native_res <- tryCatch({
        storage.mode(object1) <- "double"; storage.mode(object2) <- "double"
        A  <- native_cca_formA(object1, object2)
        nc <- min(as.integer(num.cc), nrow(A) - 1L, ncol(A) - 1L)
        ir <- irlba::irlba(A, nv = nc, nu = nc)
        ord <- order(-ir$d)
        U <- ir$u[, ord, drop = FALSE]; V <- ir$v[, ord, drop = FALSE]
        cca.data <- rbind(U, V)
        colnames(cca.data) <- paste0("CC", seq_len(nc))
        rownames(cca.data) <- c(cells1, cells2)
        cca.data <- apply(cca.data, 2, function(x) { if (sign(x[1]) == -1) x <- x * -1; x })
        list(ccv = cca.data, d = ir$d[ord])
      }, error = function(e) NULL)
      if (!is.null(.native_res)) return(.native_res)
    }

    # scipy fast path (fallback). object1/object2 already standardized, so pass
    # standardize = FALSE to the upstream fallback. On Darwin scipy needs the
    # threadpoolctl thread-guard.
    if (!.seurat_init_python()) {
      return(.seurat_orig_RunCCA_default(
        object1 = object1, object2 = object2, standardize = FALSE,
        num.cc = num.cc, seed.use = seed.use, verbose = verbose, ...))
    }
    guard <- NULL
    if (Sys.info()[["sysname"]] == "Darwin") {
      guard <- .seurat_python_thread_guard()
      if (is.null(guard)) {
        return(.seurat_orig_RunCCA_default(
          object1 = object1, object2 = object2, standardize = FALSE,
          num.cc = num.cc, seed.use = seed.use, verbose = verbose, ...))
      }
      on.exit(guard$`__exit__`(NULL, NULL, NULL), add = TRUE)
    }

    np <- reticulate::import("numpy", convert = FALSE)
    scipy_sparse_linalg <- reticulate::import("scipy.sparse.linalg", convert = FALSE)
    X1 <- np$asfortranarray(np$asarray(object1, dtype = "float32"))
    X2 <- np$asfortranarray(np$asarray(object2, dtype = "float32"))
    A <- np$ascontiguousarray(
      np$transpose(X1)$`__matmul__`(X2), dtype = "float64")
    svd_result <- scipy_sparse_linalg$svds(
      A, k = as.integer(num.cc), which = "LM", solver = "propack")
    U  <- reticulate::py_to_r(svd_result[[0]])
    s  <- reticulate::py_to_r(svd_result[[1]])
    Vt <- reticulate::py_to_r(svd_result[[2]])
    idx <- order(-s); U <- U[, idx, drop = FALSE]; s <- s[idx]
    V <- t(Vt[idx, , drop = FALSE])

    cca.data <- rbind(U, V)
    colnames(cca.data) <- paste0("CC", seq_len(num.cc))
    rownames(cca.data) <- c(cells1, cells2)
    cca.data <- apply(cca.data, 2, function(x) {
      if (sign(x[1]) == -1) x <- x * -1; x
    })
    list(ccv = cca.data, d = s)
  }

  fast_RunCCA_Seurat <- function(object1, object2, assay1 = NULL, assay2 = NULL,
                                  num.cc = 20, features = NULL,
                                  renormalize = FALSE, rescale = FALSE,
                                  compute.gene.loadings = TRUE,
                                  add.cell.id1 = NULL, add.cell.id2 = NULL,
                                  verbose = TRUE,
                                  zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    # Scope guard (additive): the fast CCA path is validated only for the
    # default renormalize=FALSE / rescale=FALSE / compute.gene.loadings=TRUE
    # combination (what IntegrateLayers uses). Any deviation, which the fast
    # path does not honor, falls back. Defaults pass unchanged.
    if (!isTRUE(zyme) || isTRUE(renormalize) || isTRUE(rescale) ||
        !isTRUE(compute.gene.loadings)) {
      # upstream .Seurat → RunCCA.default (patched); wrap to honor zyme=FALSE
      return(with_disabled(.seurat_orig_RunCCA_Seurat(
        object1 = object1, object2 = object2, assay1 = assay1, assay2 = assay2,
        num.cc = num.cc, features = features, renormalize = renormalize,
        rescale = rescale, compute.gene.loadings = compute.gene.loadings,
        add.cell.id1 = add.cell.id1, add.cell.id2 = add.cell.id2,
        verbose = verbose, ...)))
    }
    op <- options(Seurat.object.assay.version = "v3",
                   Seurat.object.assay.calcn   = FALSE)
    on.exit(options(op), add = TRUE)
    if (is.null(assay1)) assay1 <- SeuratObject::DefaultAssay(object1)
    if (is.null(assay2)) assay2 <- SeuratObject::DefaultAssay(object2)
    if (is.null(features)) {
      features <- union(SeuratObject::VariableFeatures(object1),
                        SeuratObject::VariableFeatures(object2))
    }
    data.use1 <- SeuratObject::GetAssayData(object1, assay = assay1,
                                             layer = "scale.data")
    data.use2 <- SeuratObject::GetAssayData(object2, assay = assay2,
                                             layer = "scale.data")
    features <- Seurat:::CheckFeatures(
      data.use = data.use1, features = features,
      object.name = "object1", verbose = FALSE)
    features <- Seurat:::CheckFeatures(
      data.use = data.use2, features = features,
      object.name = "object2", verbose = FALSE)
    data1 <- data.use1[features, ]; data2 <- data.use2[features, ]

    # Matrix inputs → upstream dispatch hits .default, which is patched.
    cca.results <- Seurat::RunCCA(object1 = data1, object2 = data2,
                                   standardize = TRUE, num.cc = num.cc,
                                   verbose = FALSE)

    combined.object <- merge(x = object1, y = object2,
                              merge.data = FALSE, ...)
    rownames(cca.results$ccv) <- SeuratObject::Cells(combined.object)
    colnames(data1) <- SeuratObject::Cells(combined.object)[1:ncol(data1)]
    colnames(data2) <- SeuratObject::Cells(combined.object)[
      (ncol(data1) + 1):length(SeuratObject::Cells(combined.object))]
    combined.object[["cca"]] <- SeuratObject::CreateDimReducObject(
      embeddings = cca.results$ccv[colnames(combined.object), ],
      assay = assay1, key = "CC_")
    combined.object[["cca"]]@assay.used <- SeuratObject::DefaultAssay(
      combined.object)
    combined.object <- SeuratObject::SetAssayData(
      combined.object, new.data = cbind(data1, data2), layer = "scale.data")
    combined.object
  }

  # ── Integration: FindIntegrationAnchors / FindWeights / CCAIntegration ──

  fast_FindIntegrationAnchors <- function(object.list = NULL, assay = NULL,
                                           reference = NULL,
                                           anchor.features = 2000,
                                           scale = TRUE,
                                           normalization.method = c("LogNormalize", "SCT"),
                                           sct.clip.range = NULL,
                                           reduction = c("cca", "rpca", "jpca", "rlsi"),
                                           l2.norm = TRUE, dims = 1:30,
                                           k.anchor = 5, k.filter = 200,
                                           k.score = 30, max.features = 200,
                                           nn.method = "annoy",
                                           n.trees = 50, eps = 0, verbose = TRUE,
                                           zyme = TRUE, turbo = NULL) {
    # Keep the upstream default n.trees=50. Users can still explicitly pass a
    # smaller value such as 10 for speed; we must not silently change it.
    zyme <- .seurat_zyme_flag(zyme, turbo)
    if (!isTRUE(zyme)) {
      # internally calls RunCCA / ScaleData / FindAnchors (all patched).
      return(with_disabled(.seurat_orig_FindIntegrationAnchors(
        object.list = object.list, assay = assay, reference = reference,
        anchor.features = anchor.features, scale = scale,
        normalization.method = normalization.method,
        sct.clip.range = sct.clip.range, reduction = reduction,
        l2.norm = l2.norm, dims = dims, k.anchor = k.anchor,
        k.filter = k.filter, k.score = k.score, max.features = max.features,
        nn.method = nn.method, n.trees = n.trees, eps = eps,
        verbose = verbose)))
    }
    normalization.method <- match.arg(normalization.method)
    reduction <- match.arg(reduction)
    if (is.null(assay)) assay <- sapply(object.list, SeuratObject::DefaultAssay)
    object.list <- lapply(object.list, function(obj) {
      methods::slot(obj, "tools")$Integration <- NULL; obj
    })
    object.list <- Seurat:::CheckDuplicateCellNames(object.list = object.list)

    if (is.numeric(anchor.features) && normalization.method != "SCT") {
      anchor.features <- Seurat::SelectIntegrationFeatures(
        object.list = object.list, nfeatures = anchor.features, assay = assay)
    }
    if (scale) {
      object.list <- lapply(object.list, function(object)
        Seurat::ScaleData(object = object, features = anchor.features,
                           verbose = FALSE))
    }

    slot.use <- "data"
    combinations <- expand.grid(seq_along(object.list), seq_along(object.list))
    combinations <- combinations[combinations$Var1 < combinations$Var2, ,
                                  drop = FALSE]
    objects.ncell <- sapply(object.list, ncol)
    offsets <- as.vector(cumsum(c(0, objects.ncell)))[seq_along(object.list)]

    find_pairwise_anchors <- function(row) {
      i <- combinations[row, 1]; j <- combinations[row, 2]
      o1 <- object.list[[i]]; o2 <- object.list[[j]]
      suppressWarnings(o1[["ToIntegrate"]] <- o1[[assay[i]]])
      SeuratObject::DefaultAssay(o1) <- "ToIntegrate"
      o1 <- Seurat::DietSeurat(o1, assays = "ToIntegrate",
                                scale.data = TRUE, dimreducs = reduction)
      suppressWarnings(o2[["ToIntegrate"]] <- o2[[assay[j]]])
      SeuratObject::DefaultAssay(o2) <- "ToIntegrate"
      o2 <- Seurat::DietSeurat(o2, assays = "ToIntegrate",
                                scale.data = TRUE, dimreducs = reduction)
      object.pair <- Seurat::RunCCA(
        object1 = o1, object2 = o2,
        assay1 = "ToIntegrate", assay2 = "ToIntegrate",
        features = anchor.features, num.cc = max(dims),
        renormalize = FALSE, rescale = FALSE, verbose = FALSE)
      if (l2.norm) {
        object.pair <- Seurat::L2Dim(object = object.pair,
                                      reduction = reduction)
        red <- paste0(reduction, ".l2")
      } else {
        red <- reduction
      }
      anchors <- Seurat:::FindAnchors_v3(
        object.pair = object.pair,
        assay = c("ToIntegrate", "ToIntegrate"),
        slot = slot.use,
        cells1 = colnames(o1), cells2 = colnames(o2),
        internal.neighbors = list(NULL, NULL),
        reduction = red, reduction.2 = character(),
        nn.reduction = red, dims = dims,
        k.anchor = k.anchor, k.filter = k.filter, k.score = k.score,
        max.features = max.features, nn.method = nn.method,
        n.trees = n.trees, eps = eps, verbose = FALSE)
      anchors[, 1] <- anchors[, 1] + offsets[i]
      anchors[, 2] <- anchors[, 2] + offsets[j]
      anchors
    }
    all.anchors <- lapply(seq_len(nrow(combinations)), find_pairwise_anchors)
    all.anchors <- do.call("rbind", all.anchors)
    all.anchors <- rbind(all.anchors, all.anchors[, c(2, 1, 3)])
    all.anchors <- Seurat:::AddDatasetID(
      anchor.df = all.anchors, offsets = offsets, obj.lengths = objects.ncell)
    command <- SeuratObject::LogSeuratCommand(
      object = object.list[[1]], return.command = TRUE)
    methods::new(Class = "IntegrationAnchorSet",
      object.list = object.list,
      reference.objects = if (is.null(reference)) seq_along(object.list)
                          else reference,
      anchors = all.anchors, offsets = offsets,
      anchor.features = anchor.features, command = command)
  }

  fast_FindWeights <- function(object, reduction = NULL, assay = NULL,
                                integration.name = "integrated", dims = 1:10,
                                features = NULL, k = 300, sd.weight = 1,
                                nn.method = "annoy", n.trees = 50, eps = 0,
                                reverse = FALSE, verbose = TRUE,
                                zyme = TRUE, turbo = NULL) {
    # Keep the upstream default n.trees=50 and honor explicit caller values.
    zyme <- .seurat_zyme_flag(zyme, turbo)
    .seurat_orig_FindWeights(
      object = object, reduction = reduction, assay = assay,
      integration.name = integration.name, dims = dims, features = features,
      k = k, sd.weight = sd.weight, nn.method = nn.method,
      n.trees = n.trees,
      eps = eps, reverse = reverse, verbose = verbose)
  }

  fast_CCAIntegration <- function(object = NULL, assay = NULL, layers = NULL,
                                   orig = NULL, new.reduction = "integrated.dr",
                                   reference = NULL, features = NULL,
                                   normalization.method = c("LogNormalize", "SCT"),
                                   dims = 1:30, k.filter = NA,
                                   scale.layer = "scale.data",
                                   dims.to.integrate = NULL, k.weight = 100,
                                   weight.reduction = NULL, sd.weight = 1,
                                   sample.tree = NULL, preserve.order = FALSE,
                                   verbose = TRUE,
                                   zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    if (!isTRUE(zyme)) {
      # internally calls FindIntegrationAnchors + FindWeights (both patched).
      return(with_disabled(.seurat_orig_CCAIntegration(
        object = object, assay = assay, layers = layers, orig = orig,
        new.reduction = new.reduction, reference = reference,
        features = features, normalization.method = normalization.method,
        dims = dims, k.filter = k.filter, scale.layer = scale.layer,
        dims.to.integrate = dims.to.integrate, k.weight = k.weight,
        weight.reduction = weight.reduction, sd.weight = sd.weight,
        sample.tree = sample.tree, preserve.order = preserve.order,
        verbose = verbose, ...)))
    }
    .seurat_orig_CCAIntegration(
      object = object, assay = assay, layers = layers, orig = orig,
      new.reduction = new.reduction, reference = reference,
      features = features, normalization.method = normalization.method,
      dims = dims, k.filter = k.filter, scale.layer = scale.layer,
      dims.to.integrate = dims.to.integrate, k.weight = k.weight,
      weight.reduction = weight.reduction, sd.weight = sd.weight,
      sample.tree = sample.tree, preserve.order = preserve.order,
      verbose = verbose, ...)
  }

  # ── Phase 4: SCTransform ─────────────────────────────────────────────────
  #
  # SCTransform is the trickiest patch in the seurat suite for three reasons:
  #
  #   1. The C++ residual kernels (src/seurat_sctransform.cpp) need a PSOCK
  #      worker pool on Windows for parallel GLM fitting. The pool is heavy
  #      to spin up (~10s loading glmGamPoi/sctransform in each worker), so
  #      we lazy-init on first SCT call, not at activate() time.
  #
  #   2. seurat::SCTransform.default invokes `sctransform::vst` directly via
  #      do.call(sctransform::vst, ...). The namespace prefix bypasses any
  #      local binding, so to substitute turbo_vst we MUST modify
  #      sctransform's namespace. Unlike seurat-zyme (which leaves the
  #      override in place persistently), we do this *scoped* — push the
  #      patch right before the do.call, pop in on.exit. Cleaner for CRAN
  #      (no persistent cross-package mutation) and easier to reason about.
  #
  #   3. The fast SCT internally calls Seurat::SCTransform(matrix, ...) from
  #      the .Seurat wrapper; that dispatches to .default which is also
  #      patched. The zyme=FALSE escape uses with_disabled() to make the
  #      entire upstream call tree honor the baseline request.

  .seurat_sct_state <- new.env(parent = emptyenv())
  .seurat_sct_state$cluster      <- NULL
  .seurat_sct_state$N_VST_WORKERS <- NULL
  .seurat_sct_state$initialized  <- FALSE

  .seurat_sct_init <- function() {
    if (isTRUE(.seurat_sct_state$initialized)) return(invisible(NULL))
    .seurat_sct_state$initialized <- TRUE  # set up front so failures don't retry

    if (.Platform$OS.type != "windows") {
      .seurat_sct_state$N_VST_WORKERS <- 1L
      return(invisible(NULL))
    }
    available <- as.integer(Sys.getenv(
      "OMP_NUM_THREADS", parallel::detectCores(logical = FALSE)))
    if (is.na(available) || available < 1) available <- 1L
    requested <- getOption("autozyme.sct_workers", default = NULL)
    if (!is.null(requested) && requested == 0L) {
      .seurat_sct_state$N_VST_WORKERS <- 1L; return(invisible(NULL))
    }
    if (available < 2) {
      .seurat_sct_state$N_VST_WORKERS <- 1L; return(invisible(NULL))
    }
    n_workers <- if (!is.null(requested) && requested > 0) {
      min(requested, available - 1L)
    } else {
      min(available - 1L, 8L)
    }
    .seurat_sct_state$N_VST_WORKERS <- n_workers
    .seurat_sct_state$cluster <- tryCatch({
      cl <- parallel::makeCluster(n_workers,
                                   port = 20000L + sample.int(1000L, 1L))
      parallel::clusterCall(cl, function() {
        suppressPackageStartupMessages({
          if (requireNamespace("glmGamPoi", quietly = TRUE))
            library(glmGamPoi)
          library(sctransform)
        })
      })
      message(sprintf(
        "[autozyme] SCTransform PSOCK ready (%d workers / %d cores)",
        n_workers, available))
      cl
    }, error = function(e) {
      message(sprintf(
        "[autozyme] SCT PSOCK init failed (%s); GLM fitting will be sequential",
        conditionMessage(e)))
      NULL
    })
    invisible(.seurat_sct_state$cluster)
  }

  .seurat_sct_cleanup <- function() {
    if (!is.null(.seurat_sct_state$cluster)) {
      tryCatch(parallel::stopCluster(.seurat_sct_state$cluster),
               error = function(e) NULL)
      .seurat_sct_state$cluster <- NULL
    }
    .seurat_sct_state$initialized <- FALSE
    invisible(NULL)
  }

  .seurat_sct_default_clip <- function(n_cells) {
    c(-sqrt(n_cells / 30), sqrt(n_cells / 30))
  }

  .seurat_sct_is_default_clip <- function(clip.range, n_cells) {
    if (!is.numeric(clip.range) || length(clip.range) != 2L ||
        any(!is.finite(clip.range))) {
      return(FALSE)
    }
    isTRUE(all.equal(as.numeric(clip.range),
                     .seurat_sct_default_clip(n_cells),
                     tolerance = sqrt(.Machine$double.eps),
                     check.attributes = FALSE))
  }

  .seurat_sct_supported_default_path <- function(n_cells,
                                                 reference.SCT.model,
                                                 do.correct.umi,
                                                 ncells,
                                                 residual.features,
                                                 variable.features.n,
                                                 variable.features.rv.th,
                                                 vars.to.regress,
                                                 latent.data,
                                                 do.scale,
                                                 do.center,
                                                 clip.range,
                                                 vst.flavor,
                                                 conserve.memory,
                                                 return.only.var.genes,
                                                 extra_args) {
    is.null(reference.SCT.model) &&
      isTRUE(do.correct.umi) &&
      is.numeric(ncells) && length(ncells) == 1L && is.finite(ncells) &&
      ncells > 0 &&
      is.null(residual.features) &&
      !is.null(variable.features.n) &&
      is.numeric(variable.features.n) && length(variable.features.n) == 1L &&
      is.finite(variable.features.n) && variable.features.n > 0 &&
      identical(variable.features.rv.th, 1.3) &&
      is.null(vars.to.regress) &&
      is.null(latent.data) &&
      isFALSE(do.scale) &&
      isTRUE(do.center) &&
      .seurat_sct_is_default_clip(clip.range, n_cells) &&
      identical(vst.flavor, "v2") &&
      isFALSE(conserve.memory) &&
      isTRUE(return.only.var.genes) &&
      length(extra_args) == 0L
  }

  # Parallel get_model_pars: bin-level GLM fitting across PSOCK workers.
  # Falls back to upstream sctransform:::get_model_pars for offset methods
  # or when no cluster is up.
  .seurat_sct_get_model_pars <- function(genes_step1, bin_size, umi,
                                          model_str, cells_step1, method,
                                          data_step1, theta_given,
                                          theta_estimation_fun,
                                          exclude_poisson = FALSE,
                                          fix_intercept = FALSE,
                                          fix_slope = FALSE,
                                          use_geometric_mean = TRUE,
                                          use_geometric_mean_offset = FALSE,
                                          verbosity = 0) {
    row_var <- utils::getFromNamespace("row_var", "sctransform")
    if (startsWith(method, "offset") || is.null(.seurat_sct_state$cluster)) {
      return(sctransform:::get_model_pars(
        genes_step1, bin_size, umi, model_str, cells_step1, method,
        data_step1, theta_given, theta_estimation_fun, exclude_poisson,
        fix_intercept, fix_slope, use_geometric_mean,
        use_geometric_mean_offset, verbosity))
    }
    bin_ind <- ceiling(seq_along(genes_step1) / bin_size)
    max_bin <- max(bin_ind); model_pars <- list()
    cl <- .seurat_sct_state$cluster
    for (i in seq_len(max_bin)) {
      genes_bin_regress <- genes_step1[bin_ind == i]
      umi_bin <- as.matrix(umi[genes_bin_regress, cells_step1, drop = FALSE])
      n_genes_bin <- nrow(umi_bin)
      n_workers <- min(.seurat_sct_state$N_VST_WORKERS, n_genes_bin)
      gene_indices <- split(seq_len(n_genes_bin), ceiling(
        seq_len(n_genes_bin) / (n_genes_bin / n_workers + 1e-10)))
      chunk_list <- lapply(gene_indices, function(idx)
        umi_bin[idx, , drop = FALSE])
      worker_env <- new.env(parent = globalenv())
      worker_env$model_str      <- model_str
      worker_env$data_step1     <- data_step1
      worker_env$exclude_poisson <- exclude_poisson
      worker_fn <- function(chunk) {
        sctransform:::fit_glmGamPoi_offset(
          umi = chunk, model_str = model_str,
          data = data_step1, allow_inf_theta = exclude_poisson)
      }
      environment(worker_fn) <- worker_env
      par_results <- parallel::parLapply(cl, chunk_list, worker_fn)
      model_pars[[i]] <- do.call(rbind, par_results)
    }
    model_pars <- do.call(rbind, model_pars)
    rownames(model_pars) <- genes_step1
    colnames(model_pars)[1] <- "theta"
    if (exclude_poisson) {
      umi_step1 <- umi[genes_step1, , drop = FALSE]
      genes_amean_step1 <- Matrix::rowMeans(umi_step1)
      genes_var_step1 <- row_var(umi_step1)
      predicted_theta <- genes_amean_step1^2 /
        (genes_var_step1 - genes_amean_step1)
      actual_theta <- model_pars[genes_step1, "theta"]
      diff_theta <- predicted_theta / actual_theta
      model_pars <- cbind(model_pars, diff_theta)
      diff_theta_index <- rownames(model_pars[
        model_pars[genes_step1, "diff_theta"] < 0.001, ])
      model_pars[diff_theta_index, 1] <- Inf
      model_pars <- model_pars[, -dim(model_pars)[2]]
    }
    model_pars
  }

  # Drop-in replacement for sctransform::vst. Sparse nnz counting via
  # tabulate, rowSums/ncol over rowMeans, skip gc, residual_type = none.
  .seurat_sct_fast_vst <- function(umi, cell_attr = NULL,
                                    latent_var = c("log_umi"),
                                    batch_var = NULL,
                                    latent_var_nonreg = NULL,
                                    n_genes = 2000, n_cells = 5000,
                                    method = "poisson", do_regularize = TRUE,
                                    theta_given = NULL,
                                    theta_estimation_fun = "theta.ml",
                                    exclude_poisson = FALSE,
                                    use_geometric_mean = TRUE,
                                    use_geometric_mean_offset = FALSE,
                                    fix_intercept = FALSE, fix_slope = FALSE,
                                    scale_factor = NULL, vst.flavor = NULL,
                                    verbosity = 2, verbose = NULL,
                                    show_progress = NULL,
                                    residual_type = "pearson",
                                    return_cell_attr = FALSE,
                                    return_gene_attr = TRUE,
                                    return_corrected_umi = FALSE,
                                    min_cells = 5, gmean_eps = 1,
                                    theta_regularization = "od_factor",
                                    bin_size = 500, min_variance = -Inf,
                                    bw_adjust = 3,
                                    res_clip_range = c(-sqrt(ncol(umi)),
                                                        sqrt(ncol(umi)))) {
    make_cell_attr     <- utils::getFromNamespace("make_cell_attr", "sctransform")
    reg_model_pars     <- utils::getFromNamespace("reg_model_pars", "sctransform")
    get_model_formula  <- utils::getFromNamespace("get_model_formula", "sctransform")
    row_gmean          <- utils::getFromNamespace("row_gmean", "sctransform")
    row_var            <- utils::getFromNamespace("row_var", "sctransform")
    clip_matrix_values <- utils::getFromNamespace("clip_matrix_values", "sctransform")
    get_model_pars     <- .seurat_sct_get_model_pars

    if (!is.null(vst.flavor)) {
      if (vst.flavor == "v2") {
        glmGamPoi_check <- requireNamespace("glmGamPoi", quietly = TRUE)
        method <- "glmGamPoi_offset"
        if (!glmGamPoi_check) method <- "nb_offset"
        exclude_poisson <- TRUE
        if (min_variance == -Inf) min_variance <- "umi_median"
        if (is.null(n_cells)) n_cells <- 2000
      }
    }
    arguments <- as.list(environment())
    arguments <- arguments[!names(arguments) %in% c("umi", "cell_attr")]
    if (startsWith(method, "offset")) {
      cell_attr <- NULL; latent_var <- c("log_umi"); batch_var <- NULL
      latent_var_nonreg <- NULL; n_genes <- NULL; n_cells <- NULL
      do_regularize <- FALSE
      if (is.null(theta_given)) theta_given <- 100
      else                       theta_given <- theta_given[1]
    }
    times <- list(start_time = Sys.time())
    cell_attr <- make_cell_attr(umi, cell_attr, latent_var, batch_var,
                                latent_var_nonreg, verbosity)
    if (inherits(umi, "dgCMatrix")) {
      genes_cell_count <- tabulate(umi@i + 1L, nbins = nrow(umi))
      names(genes_cell_count) <- rownames(umi)
    } else {
      genes_cell_count <- Matrix::rowSums(umi >= 0.01)
    }
    genes <- rownames(umi)[genes_cell_count >= min_cells]
    umi <- umi[genes, ]
    if (use_geometric_mean) {
      genes_log_gmean <- log10(row_gmean(umi, eps = gmean_eps))
    } else {
      genes_log_gmean <- log10(Matrix::rowMeans(umi))
    }
    if (!do_regularize && !is.null(n_genes)) n_genes <- NULL
    if (!is.null(n_cells) && n_cells < ncol(umi)) {
      cells_step1 <- sample(colnames(umi), size = n_cells)
      if (inherits(umi, "dgCMatrix")) {
        umi_sub <- umi[, cells_step1]
        genes_cell_count_step1 <- tabulate(umi_sub@i + 1L,
                                            nbins = nrow(umi_sub))
        names(genes_cell_count_step1) <- rownames(umi_sub)
        rm(umi_sub)
      } else {
        genes_cell_count_step1 <- Matrix::rowSums(umi[, cells_step1] > 0)
      }
      genes_step1 <- rownames(umi)[genes_cell_count_step1 >= min_cells]
      if (use_geometric_mean) {
        genes_log_gmean_step1 <- log10(row_gmean(umi[genes_step1, ],
                                                  eps = gmean_eps))
      } else {
        genes_log_gmean_step1 <- log10(Matrix::rowMeans(umi[genes_step1, ]))
      }
    } else {
      cells_step1 <- colnames(umi); genes_step1 <- genes
      genes_log_gmean_step1 <- genes_log_gmean
    }
    genes_amean <- NULL; genes_var <- NULL
    if (do_regularize && exclude_poisson) {
      genes_amean <- Matrix::rowSums(umi) / ncol(umi)
      genes_var <- row_var(umi)
      overdispersion_factor <- genes_var - genes_amean
      overdispersion_factor_step1 <- overdispersion_factor[genes_step1]
      is_overdispersed <- overdispersion_factor_step1 > 0
      genes_step1 <- genes_step1[is_overdispersed]
      genes_log_gmean_step1 <- genes_log_gmean[genes_step1]
    }
    data_step1 <- cell_attr[cells_step1, , drop = FALSE]
    if (!is.null(n_genes) && n_genes < length(genes_step1)) {
      log_gmean_dens <- density(x = genes_log_gmean_step1, bw = "nrd",
                                 adjust = 1)
      sampling_prob <- 1 / (approx(x = log_gmean_dens$x,
                                    y = log_gmean_dens$y,
                                    xout = genes_log_gmean_step1)$y +
                            .Machine$double.eps)
      genes_step1 <- sample(genes_step1, size = n_genes,
                            prob = sampling_prob)
      if (use_geometric_mean) {
        genes_log_gmean_step1 <- log10(row_gmean(umi[genes_step1, ],
                                                  eps = gmean_eps))
      } else {
        genes_log_gmean_step1 <- log10(Matrix::rowMeans(umi[genes_step1, ]))
      }
    }
    model_str <- paste0("y ~ ", paste(latent_var, collapse = " + "))
    if (verbosity > 0) {
      message("Variance stabilizing transformation of count matrix of size ",
              nrow(umi), " by ", ncol(umi))
      message("Model formula is ", model_str)
    }
    times$get_model_pars <- Sys.time()
    model_pars <- get_model_pars(
      genes_step1, bin_size, umi, model_str, cells_step1,
      method, data_step1, theta_given, theta_estimation_fun,
      exclude_poisson, fix_intercept, fix_slope, use_geometric_mean,
      use_geometric_mean_offset, verbosity)
    min_theta <- 1e-7
    if (any(model_pars[, "theta"] < min_theta)) {
      model_pars[, "theta"] <- pmax(model_pars[, "theta"], min_theta)
    }
    times$reg_model_pars <- Sys.time()
    if (do_regularize) {
      model_pars_fit <- reg_model_pars(
        model_pars, genes_log_gmean_step1, genes_log_gmean, cell_attr,
        batch_var, cells_step1, genes_step1, umi, bw_adjust, gmean_eps,
        theta_regularization, genes_amean, genes_var, exclude_poisson,
        fix_intercept, fix_slope, use_geometric_mean,
        use_geometric_mean_offset, verbosity)
      model_pars_outliers <- attr(model_pars_fit, "outliers")
    } else {
      model_pars_fit <- model_pars
      model_pars_outliers <- rep(FALSE, nrow(model_pars))
    }
    regressor_data <- model.matrix(get_model_formula(model_str), cell_attr)
    times$get_residuals <- Sys.time()
    res <- matrix(NA, nrow = 0, ncol = 0)
    rv <- list(y = res, model_str = model_str, model_pars = model_pars,
               model_pars_outliers = model_pars_outliers,
               model_pars_fit = model_pars_fit,
               model_str_nonreg = "", model_pars_nonreg = c(),
               arguments = arguments,
               genes_log_gmean_step1 = genes_log_gmean_step1,
               cells_step1 = cells_step1, cell_attr = cell_attr)
    rm(res)
    rv$y <- clip_matrix_values(rv$y, res_clip_range)
    if (!return_cell_attr) rv[["cell_attr"]] <- NULL
    times$get_gene_attr <- Sys.time()
    times$done <- Sys.time()
    rv$times <- times
    rv
  }

  # ── SCTransform.default ──────────────────────────────────────────────────

  fast_SCTransform_default <- function(object, cell.attr,
                                        reference.SCT.model = NULL,
                                        do.correct.umi = TRUE, ncells = 5000,
                                        residual.features = NULL,
                                        variable.features.n = 3000,
                                        variable.features.rv.th = 1.3,
                                        vars.to.regress = NULL,
                                        latent.data = NULL,
                                        do.scale = FALSE, do.center = TRUE,
                                        clip.range = c(
                                          -sqrt(ncol(object) / 30),
                                           sqrt(ncol(object) / 30)),
                                        vst.flavor = "v2",
                                        conserve.memory = FALSE,
                                        return.only.var.genes = TRUE,
                                        seed.use = 1448145, verbose = TRUE,
                                        zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    extra_args <- list(...)
    fast_path_ok <- isTRUE(zyme) &&
      .seurat_sct_supported_default_path(
        n_cells = ncol(object),
        reference.SCT.model = reference.SCT.model,
        do.correct.umi = do.correct.umi,
        ncells = ncells,
        residual.features = residual.features,
        variable.features.n = variable.features.n,
        variable.features.rv.th = variable.features.rv.th,
        vars.to.regress = vars.to.regress,
        latent.data = latent.data,
        do.scale = do.scale,
        do.center = do.center,
        clip.range = clip.range,
        vst.flavor = vst.flavor,
        conserve.memory = conserve.memory,
        return.only.var.genes = return.only.var.genes,
        extra_args = extra_args)
    if (!fast_path_ok) {
      return(with_disabled(.seurat_orig_SCTransform_default(
        object = object, cell.attr = cell.attr,
        reference.SCT.model = reference.SCT.model,
        do.correct.umi = do.correct.umi, ncells = ncells,
        residual.features = residual.features,
        variable.features.n = variable.features.n,
        variable.features.rv.th = variable.features.rv.th,
        vars.to.regress = vars.to.regress, latent.data = latent.data,
        do.scale = do.scale, do.center = do.center, clip.range = clip.range,
        vst.flavor = vst.flavor, conserve.memory = conserve.memory,
        return.only.var.genes = return.only.var.genes, seed.use = seed.use,
        verbose = verbose, ...)))
    }
    .seurat_sct_init()  # lazy PSOCK spinup (first SCT call only)

    if (!is.null(seed.use)) set.seed(seed.use)
    vst.args <- list(...)
    object <- SeuratObject::as.sparse(object)
    umi <- object
    if (!is.null(vst.flavor) && vst.flavor == "v1") vst.flavor <- NULL
    vst.args[["vst.flavor"]]           <- vst.flavor
    vst.args[["umi"]]                  <- umi
    vst.args[["cell_attr"]]            <- cell.attr
    vst.args[["verbosity"]]            <- as.numeric(verbose) * 1
    vst.args[["return_cell_attr"]]     <- TRUE
    vst.args[["return_gene_attr"]]     <- FALSE
    vst.args[["return_corrected_umi"]] <- FALSE
    vst.args[["residual_type"]]        <- "none"
    vst.args[["n_cells"]]              <- min(ncells, ncol(umi))
    vst.args[["bin_size"]]             <- 15000

    # Scoped namespace patches: sctransform::vst → fast_vst, and a cached
    # sctransform::row_gmean (eliminates two redundant gmean computations
    # over the full matrix). Both are restored on.exit so we leave no
    # persistent cross-package mutation.
    orig_sct_vst      <- utils::getFromNamespace("vst",       "sctransform")
    orig_sct_rowgmean <- utils::getFromNamespace("row_gmean", "sctransform")
    .gmean_cache <- new.env(parent = emptyenv())
    .gmean_cache$full_result <- NULL
    .gmean_cache$full_ncol   <- NULL
    cached_row_gmean <- function(x, eps = 1) {
      if (inherits(x, "dgCMatrix")) {
        nc <- ncol(x)
        if (!is.null(.gmean_cache$full_result) &&
            nc == .gmean_cache$full_ncol) {
          rn <- rownames(x)
          if (!is.null(rn) && all(rn %in% names(.gmean_cache$full_result))) {
            return(.gmean_cache$full_result[rn])
          }
        }
        result <- orig_sct_rowgmean(x, eps = eps)
        if (is.null(.gmean_cache$full_result) ||
            length(result) > length(.gmean_cache$full_result)) {
          .gmean_cache$full_result <- result
          .gmean_cache$full_ncol   <- nc
        }
        return(result)
      }
      orig_sct_rowgmean(x, eps = eps)
    }
    utils::assignInNamespace("vst",       .seurat_sct_fast_vst, ns = "sctransform")
    utils::assignInNamespace("row_gmean", cached_row_gmean,     ns = "sctransform")
    on.exit({
      utils::assignInNamespace("vst",       orig_sct_vst,      ns = "sctransform")
      utils::assignInNamespace("row_gmean", orig_sct_rowgmean, ns = "sctransform")
    }, add = TRUE)

    vst.out <- do.call(sctransform::vst, args = vst.args)

    get_model_formula <- utils::getFromNamespace("get_model_formula", "sctransform")
    regressor_data_orig <- model.matrix(
      get_model_formula(vst.out$model_str), vst.out$cell_attr)
    cell_attr_corr <- vst.out$cell_attr
    latent_var <- vst.out$arguments$latent_var
    cell_attr_corr[, latent_var] <- apply(
      cell_attr_corr[, latent_var, drop = FALSE], 2,
      function(x) rep(median(x), length(x)))
    regressor_data_corr <- model.matrix(
      get_model_formula(vst.out$model_str), cell_attr_corr)
    delta_reg <- regressor_data_corr - regressor_data_orig
    model_pars_fit <- vst.out$model_pars_fit
    genes <- rownames(umi)[rownames(umi) %in% rownames(model_pars_fit)]
    sample_coefs <- model_pars_fit[1, -1, drop = FALSE]
    corr_factor <- as.numeric(exp(delta_reg %*% t(sample_coefs)))

    min_var_raw <- vst.out$arguments$min_variance
    if (identical(min_var_raw, "umi_median")) {
      x_vals <- umi@x; n <- length(x_vals); half <- n %/% 2L
      if (n %% 2L == 1L) {
        min_var <- (sort(x_vals, partial = half + 1L)[half + 1L] / 5)^2
      } else {
        sorted_partial <- sort(x_vals, partial = c(half, half + 1L))
        min_var <- ((sorted_partial[half] + sorted_partial[half + 1L]) /
                    2 / 5)^2
      }
      rm(x_vals)
    } else {
      min_var <- as.numeric(min_var_raw)
    }

    slope_col <- ncol(regressor_data_orig)
    cell_mu_base <- as.numeric(exp(
      model_pars_fit[1, slope_col + 1] * regressor_data_orig[, slope_col]))
    res.clip.range <- c(-sqrt(ncol(umi)), sqrt(ncol(umi)))
    n_cells <- ncol(umi); col_names <- colnames(umi)
    all_gene_names <- rownames(umi)

    csr <- turbo_csc_to_csr(umi@i, umi@p, umi@x, nrow(umi), ncol(umi))

    BIN_SIZE <- 7000
    bin_ind <- ceiling(seq_along(genes) / BIN_SIZE)
    max_bin <- max(bin_ind)
    corrected_list <- vector("list", max_bin)
    res_var  <- numeric(length(genes)); names(res_var)  <- genes
    res_mean <- numeric(length(genes)); names(res_mean) <- genes
    for (b in seq_len(max_bin)) {
      genes_bin <- genes[bin_ind == b]
      intercepts_bin <- as.numeric(model_pars_fit[genes_bin, 2])
      theta_bin <- model_pars_fit[genes_bin, 1]
      gene_global_idx <- match(genes_bin, all_gene_names) - 1L
      result <- turbo_stats_correct_sparse(
        intercepts_bin, cell_mu_base,
        csr$row_ptr, csr$col_idx, csr$vals, gene_global_idx,
        theta_bin, corr_factor,
        min_var, res.clip.range[1], res.clip.range[2], do.correct.umi)
      res_mean[genes_bin] <- result$res_mean
      res_var[genes_bin]  <- result$res_var
      if (do.correct.umi) {
        corrected_list[[b]] <- methods::new("dgCMatrix",
          i = result$csc_i, p = result$csc_p, x = result$csc_x,
          Dim = c(length(genes_bin), n_cells),
          Dimnames = list(genes_bin, col_names))
      }
      rm(result)
    }
    if (do.correct.umi) {
      vst.out$umi_corrected <- do.call(rbind, corrected_list)
      rm(corrected_list)
    } else {
      vst.out$umi_corrected <- umi
    }

    gene_attr <- data.frame(residual_mean = res_mean,
                             residual_variance = res_var, row.names = genes)
    vst.out$gene_attr <- gene_attr
    feature.variance <- sort(res_var, decreasing = TRUE)
    top.features <- names(feature.variance)[
      1:min(variable.features.n, length(feature.variance))]
    feat_positions <- match(top.features, all_gene_names)
    top.features <- top.features[order(feat_positions)]

    varfeat_intercepts <- as.numeric(model_pars_fit[top.features, 2])
    varfeat_theta      <- model_pars_fit[top.features, 1]
    varfeat_gene_idx   <- match(top.features, all_gene_names) - 1L
    rm(umi)
    scale.data <- turbo_fused_resid_center_sparse(
      varfeat_intercepts, cell_mu_base,
      csr$row_ptr, csr$col_idx, csr$vals, varfeat_gene_idx, varfeat_theta,
      min_var, res.clip.range[1], res.clip.range[2],
      clip.range[1], clip.range[2])
    rm(csr)
    dimnames(scale.data) <- list(top.features, col_names)
    vst.out$y <- scale.data
    vst.out$variable_features <- top.features
    vst.out
  }

  # ── SCTransform.Seurat ───────────────────────────────────────────────────

  fast_SCTransform_Seurat <- function(object, assay = "RNA",
                                       new.assay.name = "SCT",
                                       reference.SCT.model = NULL,
                                       do.correct.umi = TRUE, ncells = 5000,
                                       residual.features = NULL,
                                       variable.features.n = 3000,
                                       variable.features.rv.th = 1.3,
                                       vars.to.regress = NULL,
                                       do.scale = FALSE, do.center = TRUE,
                                       clip.range = c(
                                         -sqrt(ncol(object[[assay]]) / 30),
                                          sqrt(ncol(object[[assay]]) / 30)),
                                       vst.flavor = "v2",
                                       conserve.memory = FALSE,
                                       return.only.var.genes = TRUE,
                                       seed.use = 1448145, verbose = TRUE,
                                       zyme = TRUE, turbo = NULL, ...) {
    zyme <- .seurat_zyme_flag(zyme, turbo)
    if (is.null(assay) || length(assay) != 1L || identical(assay, "SCT")) {
      return(with_disabled(.seurat_orig_SCTransform_Seurat(
        object = object, assay = assay, new.assay.name = new.assay.name,
        reference.SCT.model = reference.SCT.model,
        do.correct.umi = do.correct.umi, ncells = ncells,
        residual.features = residual.features,
        variable.features.n = variable.features.n,
        variable.features.rv.th = variable.features.rv.th,
        vars.to.regress = vars.to.regress,
        do.scale = do.scale, do.center = do.center, clip.range = clip.range,
        vst.flavor = vst.flavor, conserve.memory = conserve.memory,
        return.only.var.genes = return.only.var.genes,
        seed.use = seed.use, verbose = verbose, ...)))
    }
    extra_args <- list(...)
    fast_path_ok <- isTRUE(zyme) &&
      .seurat_sct_supported_default_path(
        n_cells = ncol(object[[assay]]),
        reference.SCT.model = reference.SCT.model,
        do.correct.umi = do.correct.umi,
        ncells = ncells,
        residual.features = residual.features,
        variable.features.n = variable.features.n,
        variable.features.rv.th = variable.features.rv.th,
        vars.to.regress = vars.to.regress,
        latent.data = NULL,
        do.scale = do.scale,
        do.center = do.center,
        clip.range = clip.range,
        vst.flavor = vst.flavor,
        conserve.memory = conserve.memory,
        return.only.var.genes = return.only.var.genes,
        extra_args = extra_args)
    if (!fast_path_ok) {
      return(with_disabled(.seurat_orig_SCTransform_Seurat(
        object = object, assay = assay, new.assay.name = new.assay.name,
        reference.SCT.model = reference.SCT.model,
        do.correct.umi = do.correct.umi, ncells = ncells,
        residual.features = residual.features,
        variable.features.n = variable.features.n,
        variable.features.rv.th = variable.features.rv.th,
        vars.to.regress = vars.to.regress,
        do.scale = do.scale, do.center = do.center, clip.range = clip.range,
        vst.flavor = vst.flavor, conserve.memory = conserve.memory,
        return.only.var.genes = return.only.var.genes,
        seed.use = seed.use, verbose = verbose, ...)))
    }
    if (!is.null(seed.use)) set.seed(seed.use)
    cell.attr <- methods::slot(object, "meta.data")[
      colnames(object[[assay]]), ]
    umi <- SeuratObject::GetAssayData(object[[assay]], layer = "counts")
    vst.out <- Seurat::SCTransform(
      object = umi, cell.attr = cell.attr,
      reference.SCT.model = reference.SCT.model,
      do.correct.umi = do.correct.umi, ncells = ncells,
      residual.features = residual.features,
      variable.features.n = variable.features.n,
      variable.features.rv.th = variable.features.rv.th,
      vars.to.regress = vars.to.regress, latent.data = NULL,
      do.scale = do.scale, do.center = do.center, clip.range = clip.range,
      vst.flavor = vst.flavor, conserve.memory = conserve.memory,
      return.only.var.genes = return.only.var.genes,
      seed.use = seed.use, verbose = verbose, ...)
    rm(umi)

    assay.out <- SeuratObject::CreateAssayObject(counts = vst.out$umi_corrected)
    data_layer <- vst.out$umi_corrected
    vst.out$umi_corrected <- NULL
    data_layer@x <- log1p(data_layer@x)
    SeuratObject::VariableFeatures(assay.out) <- vst.out$variable_features
    methods::slot(assay.out, "data") <- data_layer
    rm(data_layer)
    methods::slot(assay.out, "scale.data") <- vst.out$y
    vst.out$y <- NULL
    vst.out$arguments$sct.clip.range <- clip.range
    vst.out$arguments <- vst.out$arguments[
      !vapply(vst.out$arguments, is.null, logical(1))]
    SeuratObject::Misc(assay.out, slot = "vst.out") <- vst.out
    rm(vst.out)
    old_validate <- getOption("Seurat.object.validate", default = TRUE)
    on.exit(options(Seurat.object.validate = old_validate), add = TRUE)
    options(Seurat.object.validate = FALSE)
    assay.out <- methods::as(assay.out, "SCTAssay")
    SCTAssay_fn <- utils::getFromNamespace("SCTAssay", "Seurat")
    assay.out <- SCTAssay_fn(assay.out, assay.orig = assay)
    methods::slot(methods::slot(assay.out, "SCTModel.list")[[1]],
                   "umi.assay") <- assay
    SeuratObject::Key(assay.out) <- tolower(paste0(new.assay.name, "_"))
    assays_list <- methods::slot(object, "assays")
    assays_list[[new.assay.name]] <- assay.out
    methods::slot(object, "assays") <- assays_list
    rm(assays_list, assay.out)
    methods::slot(object, "active.assay") <- new.assay.name
    SeuratObject::LogSeuratCommand(object)
  }

  # ── Capture Phase 3 originals (after fast fns are defined) ──────────────

  .seurat_orig_RunPCA_StdAssay <- utils::getFromNamespace(
    "RunPCA.StdAssay", "Seurat")
  .seurat_orig_RunPCA_default <- utils::getFromNamespace(
    "RunPCA.default", "Seurat")
  .seurat_orig_RunCCA_default <- utils::getFromNamespace(
    "RunCCA.default", "Seurat")
  .seurat_orig_RunCCA_Seurat <- utils::getFromNamespace(
    "RunCCA.Seurat", "Seurat")
  .seurat_orig_FindIntegrationAnchors <- utils::getFromNamespace(
    "FindIntegrationAnchors", "Seurat")
  .seurat_orig_FindWeights <- utils::getFromNamespace(
    "FindWeights", "Seurat")
  .seurat_orig_CCAIntegration <- utils::getFromNamespace(
    "CCAIntegration", "Seurat")
  .seurat_orig_SCTransform_Seurat <- utils::getFromNamespace(
    "SCTransform.Seurat", "Seurat")
  .seurat_orig_SCTransform_default <- utils::getFromNamespace(
    "SCTransform.default", "Seurat")

  # ── Register ─────────────────────────────────────────────────────────────

  register_patch(
    name = "seurat",
    upstream = "Seurat",
    targets = list(
      NormalizeData.Seurat            = fast_NormalizeData_Seurat,
      FindVariableFeatures.Seurat     = fast_FindVariableFeatures_Seurat,
      FindVariableFeatures.StdAssay   = fast_FindVariableFeatures_StdAssay,
      VST.dgCMatrix                   = fast_VST_dgCMatrix,
      ScaleData.Seurat                = fast_ScaleData_Seurat,
      FindNeighbors.Seurat            = fast_FindNeighbors_Seurat,
      FindAllMarkers                  = fast_FindAllMarkers,
      FindMarkers.Seurat              = fast_FindMarkers_Seurat,
      RunPCA.StdAssay                 = fast_RunPCA_StdAssay,
      RunPCA.default                  = fast_RunPCA_default,
      RunCCA.default                  = fast_RunCCA_default,
      RunCCA.Seurat                   = fast_RunCCA_Seurat,
      FindIntegrationAnchors          = fast_FindIntegrationAnchors,
      FindWeights                     = fast_FindWeights,
      CCAIntegration                  = fast_CCAIntegration,
      SCTransform.Seurat              = fast_SCTransform_Seurat,
      SCTransform.default             = fast_SCTransform_default
    ),
    smoke = list(
      # Representative task: Seurat::FindAllMarkers on a clustered object.
      # Canonical signature from optimized_task/test_seurat_scanpy/find_all_markers/v2/
      # (task.yaml + reference.R) — `FindAllMarkers(object, verbose = FALSE)`
      # with all defaults. The checkpoint .rds already has Idents() populated,
      # so no separate clustering step is timed.
      load = function(task_dir, tier) {
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
        list(obj = readRDS(data_path))
      },
      call = function(inputs) {
        # Suppress the "Seurat is now using presto for wilcox" startup message
        # in fresh subprocesses; it would otherwise fragment timing output.
        options(Seurat.presto.wilcox.msg = FALSE)
        markers <- Seurat::FindAllMarkers(object = inputs$obj, verbose = FALSE)
        # Carry dataset shape + cluster set alongside the markers so save()
        # can mirror reference.R's result.rds schema. The three lookups are
        # O(1) and run on BOTH baseline and patched, so they do not bias the
        # speedup ratio.
        list(
          markers    = markers,
          n_cells    = ncol(inputs$obj),
          n_genes    = nrow(inputs$obj),
          identities = levels(Seurat::Idents(inputs$obj))
        )
      },
      save = function(result, dir, tier = "tiny", ...) {
        # evaluate.R reads result.rds with keys: markers, n_cells, n_genes,
        # identities. Mirror reference.R exactly (find_all_markers/v2/
        # reference.R:73-79). The verify_worker creates `dir` (either
        # pipeline/output_<tier>/ or reference_output_<tier>/) so write
        # directly.
        saveRDS(result, file.path(dir, "result.rds"))
      }
    ),
    tested_against = "Seurat 5.4.0",
    tested_upstream_versions = list(Seurat = c("5.2.1", "5.4.0")),
    on_activate = .seurat_on_activate,
    on_deactivate = .seurat_sct_cleanup
  )
}
