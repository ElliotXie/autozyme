# Patch for SoupX::adjustCounts / SoupX::expandClusters.
#
# Lifted from autozyme task `test_adjustcounts`. The attested public call is
# `SoupX::adjustCounts(sc, verbose = 0)`: default subtraction, no stochastic
# rounding, no extra expandClusters arguments, and automatic clusters from
# `sc$metaData` when present. Dense/triplet/row-compressed inputs and wider API
# branches delegate to SoupX.
#
# The C++ kernels live in src/soupx.cpp.

if (requireNamespace("SoupX", quietly = TRUE) &&
    requireNamespace("Matrix", quietly = TRUE) &&
    requireNamespace("yaml", quietly = TRUE)) {

  .soupx_orig_adjustCounts <- utils::getFromNamespace("adjustCounts", "SoupX")
  .soupx_orig_expandClusters <- utils::getFromNamespace("expandClusters", "SoupX")
  .soupx_version <- as.character(utils::packageVersion("SoupX"))

  .soupx_upstream_adjustCounts <- .soupx_orig_adjustCounts
  .soupx_upstream_env <- new.env(parent = environment(.soupx_orig_adjustCounts))
  .soupx_upstream_env$adjustCounts <- .soupx_upstream_adjustCounts
  .soupx_upstream_env$expandClusters <- .soupx_orig_expandClusters
  environment(.soupx_upstream_adjustCounts) <- .soupx_upstream_env

  .soupx_can_use_dgC_fast_path <- function(sc) {
    inherits(sc$toc, "dgCMatrix")
  }

  .soupx_no_extra_args <- function(args) {
    length(args) == 0L
  }

  .soupx_positive_Tsparse_from_dgC <- function(mat) {
    keep <- mat@x > 0
    col_index <- rep.int(seq_len(ncol(mat)), diff(mat@p))
    Matrix::sparseMatrix(
      i = mat@i[keep] + 1L,
      j = col_index[keep],
      x = mat@x[keep],
      dims = dim(mat),
      dimnames = dimnames(mat),
      giveCsparse = FALSE
    )
  }

  .soupx_expandClusters_impl <- function(clustSoupCnts, cellObsCnts, clusters,
                                         cellWeights, verbose = 1,
                                         corrected = FALSE) {
    if (verbose > 0) {
      message(sprintf("Expanding counts from %d clusters to %d cells.",
                      ncol(clustSoupCnts), ncol(cellObsCnts)))
    }

    alloc_fn <- utils::getFromNamespace("alloc", "SoupX")
    clusters <- as.character(clusters)
    if (!is.null(names(clusters))) {
      clusters <- clusters[colnames(cellObsCnts)]
    }
    cluster_levels <- colnames(clustSoupCnts)
    cluster_id <- match(clusters, cluster_levels)
    if (anyNA(cluster_id)) {
      stop("Cluster labels in clusters do not match clustSoupCnts columns.")
    }

    obs <- cellObsCnts

    if (isTRUE(corrected)) {
      out <- obs
      out@x <- soupx_expand_corrected_x_cpp(
        obs@p, obs@i, obs@x, as.matrix(clustSoupCnts), cluster_id, cellWeights
      )
      return(out)
    }

    out_i <- vector("list", length(cluster_levels))
    out_j <- vector("list", length(cluster_levels))
    out_x <- vector("list", length(cluster_levels))

    for (cluster_pos in seq_along(cluster_levels)) {
      cell_idx <- which(cluster_id == cluster_pos)
      entry_lens <- diff(obs@p)[cell_idx]
      has_entries <- entry_lens > 0L
      if (!any(has_entries)) {
        next
      }
      entry_idx <- sequence(entry_lens[has_entries],
                            obs@p[cell_idx[has_entries]] + 1L)
      entry_cell <- rep.int(cell_idx[has_entries], entry_lens[has_entries])

      n_soup <- as.numeric(clustSoupCnts[, cluster_pos])
      gene <- obs@i[entry_idx] + 1L
      entry_x <- obs@x[entry_idx]
      gene_totals <- rowsum(entry_x, gene, reorder = FALSE)
      lim_tot <- numeric(nrow(cellObsCnts))
      lim_tot[as.integer(rownames(gene_totals))] <- gene_totals[, 1L]

      soup_for_entry <- n_soup[gene]
      keep <- soup_for_entry > 0
      if (!any(keep)) {
        next
      }

      kept_gene <- gene[keep]
      kept_cell <- entry_cell[keep]
      x <- entry_x[keep]

      partial <- n_soup[kept_gene] > 0 & n_soup[kept_gene] < lim_tot[kept_gene]
      if (any(partial)) {
        partial_local <- which(partial)
        partial_order <- order(kept_gene[partial_local])
        partial_local <- partial_local[partial_order]
        partial_gene <- kept_gene[partial_local]
        run_start <- c(1L, which(diff(partial_gene) != 0L) + 1L)
        run_end <- c(run_start[-1L] - 1L, length(partial_local))
        for (run_pos in seq_along(run_start)) {
          local_idx <- partial_local[run_start[run_pos]:run_end[run_pos]]
          gene_pos <- kept_gene[local_idx[1L]]
          x[local_idx] <- alloc_fn(
            n_soup[gene_pos], x[local_idx], cellWeights[kept_cell[local_idx]]
          )
        }
      }

      full_soup <- n_soup[kept_gene] >= lim_tot[kept_gene]
      x[full_soup] <- entry_x[keep][full_soup]

      out_i[[cluster_pos]] <- kept_gene
      out_j[[cluster_pos]] <- kept_cell
      out_x[[cluster_pos]] <- x
    }

    Matrix::sparseMatrix(
      i = unlist(out_i, use.names = FALSE),
      j = unlist(out_j, use.names = FALSE),
      x = unlist(out_x, use.names = FALSE),
      dims = dim(cellObsCnts),
      dimnames = dimnames(cellObsCnts)
    )
  }

  fast_expandClusters <- function(clustSoupCnts, cellObsCnts, clusters,
                                  cellWeights, verbose = 1, ...,
                                  zyme = TRUE) {
    dots <- list(...)
    if (!isTRUE(zyme) || !.soupx_no_extra_args(dots) ||
        !inherits(cellObsCnts, "dgCMatrix")) {
      return(.soupx_orig_expandClusters(
        clustSoupCnts, cellObsCnts, clusters, cellWeights, verbose = verbose
      ))
    }
    .soupx_expandClusters_impl(
      clustSoupCnts, cellObsCnts, clusters, cellWeights, verbose = verbose,
      corrected = FALSE
    )
  }

  fast_adjustCounts <- function(sc, clusters = NULL,
                                method = c("subtraction", "soupOnly", "multinomial"),
                                roundToInt = FALSE, verbose = 1, tol = 1e-3,
                                pCut = 0.01, ..., zyme = TRUE) {
    dots <- list(...)
    if (!isTRUE(zyme)) {
      return(.soupx_orig_adjustCounts(
        sc, clusters = clusters, method = method, roundToInt = roundToInt,
        verbose = verbose, tol = tol, pCut = pCut, ...
      ))
    }

    method <- match.arg(method)
    if (!is(sc, "SoupChannel")) {
      stop("sc must be an object of type SoupChannel")
    }
    if (!"rho" %in% colnames(sc$metaData)) {
      stop("Contamination fractions must have already been calculated/set.")
    }

    if (!identical(method, "subtraction") ||
        !identical(roundToInt, FALSE) ||
        !isTRUE(all.equal(tol, 1e-3)) ||
        !isTRUE(all.equal(pCut, 0.01)) ||
        !.soupx_no_extra_args(dots)) {
      return(.soupx_orig_adjustCounts(
        sc, clusters = clusters, method = method, roundToInt = roundToInt,
        verbose = verbose, tol = tol, pCut = pCut, ...
      ))
    }

    if (is.null(clusters)) {
      if ("clusters" %in% colnames(sc$metaData)) {
        clusters <- stats::setNames(as.character(sc$metaData$clusters),
                                    rownames(sc$metaData))
      } else {
        warning("Clustering data not found.  Adjusting counts at cell level.  You will almost certainly get better results if you cluster data first.")
        clusters <- FALSE
      }
    } else {
      return(.soupx_orig_adjustCounts(
        sc, clusters = clusters, method = method, roundToInt = roundToInt,
        verbose = verbose, tol = tol, pCut = pCut
      ))
    }

    if (clusters[1] == FALSE) {
      if (!.soupx_can_use_dgC_fast_path(sc)) {
        return(.soupx_upstream_adjustCounts(
          sc, clusters = FALSE, method = method, roundToInt = roundToInt,
          verbose = verbose, tol = tol, pCut = pCut
        ))
      }

      out <- sc$toc
      out@x <- soupx_adjust_counts_no_cluster_x_cpp(
        out@p, out@i, out@x,
        sc$metaData[colnames(sc$toc), "nUMIs"] *
          sc$metaData[colnames(sc$toc), "rho"],
        sc$soupProfile$est, nrow(sc$toc)
      )
      return(.soupx_positive_Tsparse_from_dgC(out))
    }

    if (!all(colnames(sc$toc) %in% names(clusters))) {
      stop("Invalid cluster specification.  clusters must be a named vector with all column names in the table of counts appearing.")
    }

    if (!.soupx_can_use_dgC_fast_path(sc)) {
      return(.soupx_upstream_adjustCounts(
        sc, clusters = clusters, method = method, roundToInt = roundToInt,
        verbose = verbose, tol = tol, pCut = pCut
      ))
    }

    clusters_ordered <- as.character(clusters[colnames(sc$toc)])
    cluster_levels <- sort(unique(clusters_ordered))
    cluster_id <- match(clusters_ordered, cluster_levels)
    out <- soupx_cluster_soup_from_cells_cpp(
      sc$toc@p, sc$toc@i, sc$toc@x,
      cluster_id,
      sc$metaData[colnames(sc$toc), "nUMIs"] *
        sc$metaData[colnames(sc$toc), "rho"],
      sc$soupProfile$est,
      nrow(sc$toc), length(cluster_levels)
    )
    dimnames(out) <- list(rownames(sc$toc), cluster_levels)

    .soupx_expandClusters_impl(out, sc$toc, clusters_ordered,
                               sc$metaData$nUMIs * sc$metaData$rho,
                               verbose = verbose, corrected = TRUE)
  }

  .soupx_smoke_load <- function(task_dir, tier) {
    suppressPackageStartupMessages({
      library(Matrix)
      library(SoupX)
    })
    task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
    ds <- Filter(function(d) identical(d$tier, tier), task$datasets)
    if (length(ds) == 0L) {
      stop("no dataset for tier '", tier, "' in task.yaml")
    }
    data_path <- resolve_dataset_path(task_dir, ds[[1]]$path)
    sc <- readRDS(data_path)
    if (!is(sc, "SoupChannel")) {
      stop("Input is not a SoupChannel: ", data_path)
    }
    if (!"rho" %in% colnames(sc$metaData)) {
      stop("SoupChannel missing metaData$rho")
    }
    if (is.null(sc$soupProfile)) {
      stop("SoupChannel missing soupProfile")
    }
    list(sc = sc)
  }

  .soupx_smoke_call <- function(inputs) {
    SoupX::adjustCounts(inputs$sc, verbose = 0)
  }

  .soupx_smoke_save <- function(result, dir, tier = "tiny", ...) {
    saveRDS(list(corrected_counts = result), file.path(dir, "result.rds"))
  }

  register_patch(
    name = "soupx",
    upstream = "SoupX",
    targets = list(
      adjustCounts = fast_adjustCounts,
      expandClusters = fast_expandClusters
    ),
    smoke = list(
      load = .soupx_smoke_load,
      call = .soupx_smoke_call,
      save = .soupx_smoke_save
    ),
    tested_against = paste("SoupX", .soupx_version),
    tested_upstream_versions = list(SoupX = .soupx_version)
  )
}
