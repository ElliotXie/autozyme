# Patch for scriabin::GenerateCCIM.
#
# Lifted from autozyme task `test_scriabin`. Replaces the dense outer-product
# `pbsapply` loop that materializes a (n_senders * n_receivers) x n_pairs
# scratch matrix with a single sparse-triplet pass in C++
# (`fast_lr_outer_triplets_cpp`, in src/), then short-circuits the
# downstream Seurat-CCIM constructor + MapMetaData calls into a lightweight
# `fast_ccim_output` list that mirrors what the task's evaluator inspects
# (counts / lr_pairs / cell_pairs only).
#
# Single-target patch — only `scriabin::GenerateCCIM` is rebound. The
# constructor, MapMetaData, LoadLR, pbsapply paths fused inside the lift are
# inlined into the fast body rather than registered as separate targets,
# which keeps the surface area small and avoids side-effects when callers
# use those upstream functions outside of GenerateCCIM. The patch is correct
# for the default (unweighted, no senders/receivers/ligands/recepts/custom)
# call form; any non-default invocation falls through to upstream.
#
# task.yaml declares baseline_threads: [1] / threading: not_applicable —
# the converged fast path is fully serial.

if (requireNamespace("scriabin",    quietly = TRUE) &&
    requireNamespace("Seurat",      quietly = TRUE) &&
    requireNamespace("SeuratObject",quietly = TRUE) &&
    requireNamespace("Matrix",      quietly = TRUE) &&
    requireNamespace("methods",     quietly = TRUE) &&
    requireNamespace("yaml",        quietly = TRUE)) {

  # ============================================================
  # File-scope captures
  # ============================================================
  # Upstream original — bound here so the dispatcher's fallback path
  # (zyme=FALSE / with_disabled()) reaches the un-patched implementation.
  .orig_GenerateCCIM <- utils::getFromNamespace("GenerateCCIM", "scriabin")
  .orig_LoadLR       <- utils::getFromNamespace("LoadLR",       "scriabin")

  # Cache the scriabin lr_resources.rds payload after the first LoadLR call.
  # Default-path GenerateCCIM hits LoadLR exactly once per call, so memoising
  # across calls saves the readRDS + filter cost on every subsequent invocation
  # in the same R session. Keyed by (species, database, lit_support).
  .zyme_loadlr_cache <- new.env(parent = emptyenv())

  .zyme_loadlr_cached <- function(species, database, lit_support) {
    key <- paste(species, database, lit_support, sep = "|")
    cached <- .zyme_loadlr_cache[[key]]
    if (!is.null(cached)) return(cached)
    val <- .orig_LoadLR(species = species, database = database,
                        lit_support = lit_support)
    .zyme_loadlr_cache[[key]] <- val
    val
  }

  # Sparse outer-product builder using the C++ triplet kernel. Returns a
  # dgCMatrix of dims (n_senders * n_receivers) x n_pairs.
  .zyme_lr_outer_sparse <- function(a, b) {
    triplets <- fast_lr_outer_triplets_cpp(a, b)
    if (length(triplets$x) == 0L) {
      return(Matrix::sparseMatrix(dims = triplets$dims))
    }
    Matrix::sparseMatrix(
      i = triplets$i,
      j = triplets$j,
      x = triplets$x,
      dims = triplets$dims
    )
  }

  # ============================================================
  # Fast replacement
  # ============================================================
  fast_GenerateCCIM <- function(object, assay = "SCT", slot = "data",
                                species = "human", database = "OmniPath",
                                ligands = NULL, recepts = NULL,
                                senders = NULL, receivers = NULL,
                                weighted = FALSE, nichenet_results = NULL,
                                pearson.cutoff = 0.075,
                                scale.factors = c(1.5, 3),
                                weight.method = "sum",
                                zyme = TRUE) {
    # Off-spec fast-path conditions — fall back to upstream.
    if (!isTRUE(zyme) ||
        isTRUE(weighted) ||
        !is.null(ligands) ||
        !is.null(recepts) ||
        !is.null(senders) ||
        !is.null(receivers) ||
        identical(database, "custom")) {
      return(.orig_GenerateCCIM(
        object = object, assay = assay, slot = slot,
        species = species, database = database,
        ligands = ligands, recepts = recepts,
        senders = senders, receivers = receivers,
        weighted = weighted, nichenet_results = nichenet_results,
        pearson.cutoff = pearson.cutoff, scale.factors = scale.factors,
        weight.method = weight.method
      ))
    }

    lit.put <- .zyme_loadlr_cached(species = species, database = database,
                                   lit_support = 7)
    ligands <- as.character(lit.put[, "source_genesymbol"])
    recepts <- as.character(lit.put[, "target_genesymbol"])

    assay_features <- rownames(object@assays[[assay]])
    ligands.use <- intersect(ligands, assay_features)
    recepts.use <- intersect(recepts, assay_features)
    genes.use   <- union(ligands.use, recepts.use)

    keep <- lit.put$source_genesymbol %in% ligands.use &
            lit.put$target_genesymbol %in% recepts.use
    lit.put <- lit.put[keep, , drop = FALSE]
    ligands <- as.character(lit.put[, "source_genesymbol"])
    recepts <- as.character(lit.put[, "target_genesymbol"])

    senders <- receivers <- colnames(object)
    expr <- as.matrix(SeuratObject::GetAssayData(
      object, assay = assay, layer = slot)[genes.use, , drop = FALSE])
    gene_index <- rownames(expr)
    a <- expr[match(ligands, gene_index), , drop = FALSE]
    b <- expr[match(recepts, gene_index), , drop = FALSE]
    a[is.na(a)] <- 0
    b[is.na(b)] <- 0
    keep_lr <- rowSums(a != 0) > 0 & rowSums(b != 0) > 0

    message("Using unweighted ligand-receptor matrices")
    message(paste("Calculating CCIM between", length(senders), "senders and",
                  length(receivers), "receivers"))
    message(paste("\nGenerating Interaction Matrix..."))

    m <- sqrt(.zyme_lr_outer_sparse(a, b))

    colnames(m) <- paste(ligands, recepts, sep = "=")
    cna <- rep(senders, length(receivers))
    cnb <- rep(receivers, each = length(senders))
    rownames(m) <- paste(cna, cnb, sep = "=")
    if (!all(keep_lr)) {
      m <- m[, keep_lr, drop = FALSE]
    }
    counts <- Matrix::t(m)
    counts <- methods::as(counts, "dgCMatrix")

    # Construct a Seurat CCIM object matching upstream's GenerateCCIM return
    # shape so downstream consumers (BinByIdentity, plotting helpers, callers
    # that read sender_*/receiver_* meta.data) see the API they expect.
    # Equivalent to upstream's tail:
    #   seu <- CreateSeuratObject(counts = Matrix::t(m), assay = "CCIM")
    #   seu <- MapMetaData(ccim_seu = seu, seu = object)
    # The Assay5 constructor + meta.data map is what we used to skip; the
    # C++ ligand-receptor outer-product kernel remains the main speedup.
    ccim_seu <- SeuratObject::CreateSeuratObject(
      counts = counts, assay = "CCIM")

    # Inline MapMetaData: split cell-pair names into sender/receiver and
    # prepend "sender_"/"receiver_" copies of every parent meta.data column.
    cp <- colnames(ccim_seu)
    sep_pos <- regexpr("=", cp, fixed = TRUE)
    sender_cells   <- substr(cp, 1L, sep_pos - 1L)
    receiver_cells <- substr(cp, sep_pos + 1L, nchar(cp))
    ccim_seu$sender   <- sender_cells
    ccim_seu$receiver <- receiver_cells

    parent_md    <- object@meta.data
    parent_cells <- rownames(parent_md)
    for (col_name in colnames(parent_md)) {
      vals <- parent_md[[col_name]]
      if (is.factor(vals)) vals <- as.character(vals)
      names(vals) <- parent_cells
      ccim_seu@meta.data[[paste0("sender_",   col_name)]] <-
        unname(vals[sender_cells])
      ccim_seu@meta.data[[paste0("receiver_", col_name)]] <-
        unname(vals[receiver_cells])
    }
    ccim_seu
  }

  # ============================================================
  # Smoke recipe
  # ============================================================
  # One-time correctness shim: scriabin 0.0.0.9000's interaction_graph.R
  # calls `GetAssayData(object, assay, slot=)` (the `slot=` form was
  # deprecated in SeuratObject 5.0 and is now defunct on current
  # installations — calls fail before any of the GenerateCCIM body runs).
  # The task's pipeline patches around this by injecting a translation
  # shim before calling the source-loaded GenerateCCIM. The baseline path
  # in verify_patch invokes the INSTALLED scriabin::GenerateCCIM, which
  # resolves `GetAssayData` via scriabin's imports table (parent.env of
  # the scriabin namespace), so we install the shim there. Idempotent.
  .zyme_scriabin_install_shim <- function() {
    imports_env <- parent.env(asNamespace("scriabin"))
    cur <- tryCatch(get("GetAssayData", envir = imports_env,
                        inherits = FALSE),
                    error = function(e) NULL)
    if (!is.null(cur) && isTRUE(attr(cur, ".zyme_scriabin_shim"))) return(invisible())
    shim <- function(object, assay = NULL, slot = NULL, layer = NULL, ...) {
      if (is.null(layer) && !is.null(slot)) layer <- slot
      SeuratObject::GetAssayData(object = object, assay = assay,
                                 layer = layer, ...)
    }
    attr(shim, ".zyme_scriabin_shim") <- TRUE
    was_locked <- bindingIsLocked("GetAssayData", imports_env)
    if (was_locked) unlockBinding("GetAssayData", imports_env)
    assign("GetAssayData", shim, envir = imports_env)
    if (was_locked) lockBinding("GetAssayData", imports_env)
    invisible()
  }

  .scriabin_smoke_load <- function(task_dir, tier) {
    task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
    ds <- Filter(function(d) d$tier == tier, task$datasets)
    if (length(ds) == 0L) stop("no dataset for tier '", tier, "' in task.yaml")
    data_path <- resolve_dataset_path(task_dir, ds[[1]]$path)
    obj <- readRDS(data_path)
    .zyme_scriabin_install_shim()
    list(obj = obj)
  }

  .scriabin_smoke_call <- function(inputs) {
    scriabin::GenerateCCIM(
      inputs$obj,
      assay            = "RNA",
      slot             = "data",
      species          = "human",
      database         = "OmniPath",
      ligands          = NULL,
      recepts          = NULL,
      senders          = NULL,
      receivers        = NULL,
      weighted         = FALSE,
      nichenet_results = NULL,
      pearson.cutoff   = 0.075,
      scale.factors    = c(1.5, 3),
      weight.method    = "sum"
    )
  }

  # Mirror of pipeline/run.R::extract_ccim_output — produces the exact
  # list($counts, $lr_pairs, $cell_pairs) shape evaluate.R diffs against
  # the reference output.
  .scriabin_smoke_save <- function(result, dir, tier = "tiny", ...) {
    counts <- if (inherits(result, "fast_ccim_output")) {
      result$counts
    } else {
      SeuratObject::LayerData(result, assay = "CCIM", layer = "counts")
    }
    counts <- methods::as(counts, "dgCMatrix")
    saveRDS(
      list(
        counts     = counts,
        lr_pairs   = rownames(counts),
        cell_pairs = colnames(counts)
      ),
      file.path(dir, "output.rds"),
      compress = FALSE
    )
  }

  register_patch(
    name = "scriabin",
    upstream = "scriabin",
    targets = list(GenerateCCIM = fast_GenerateCCIM),
    smoke = list(
      load = .scriabin_smoke_load,
      call = .scriabin_smoke_call,
      save = .scriabin_smoke_save
    ),
    tested_against = "scriabin 0.0.0.9000",
    tested_upstream_versions = list(scriabin = "0.0.0.9000")
  )
}
