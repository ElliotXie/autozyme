# Patch for nichenetr.
#
# Lifted from autozyme task `test_nichenet`. Replaces nichenetr's
# per-ligand evaluation loop in `predict_ligand_activities` (single = TRUE
# path) with a parallel C++ kernel (`score_ligands_cpp`, in src/), falling
# back to a vectorized R inner loop on Windows or for very small ligand sets.
#
# Wrapped in requireNamespace gates so package load is a no-op when nichenetr
# (or any patch dep) is not installed.

if (requireNamespace("nichenetr", quietly = TRUE) &&
    requireNamespace("caTools",   quietly = TRUE) &&
    requireNamespace("dplyr",     quietly = TRUE)) {

  .nichenetr_orig_predict_ligand_activities <- utils::getFromNamespace(
    "predict_ligand_activities", "nichenetr"
  )

  .nichenetr_fast_score_selected_metrics <- function(prediction, response) {
    if (any(is.infinite(prediction))) {
      message("Warning: Inf values detected in prediction vector. Replacing with max value + 1e-1.")
      max_value <- max(prediction[is.finite(prediction)], na.rm = TRUE)
      prediction[is.infinite(prediction)] <- max_value + 0.1
    }

    pred_order <- order(prediction, decreasing = TRUE)
    prediction_sorted <- prediction[pred_order]
    response_sorted <- response[pred_order]
    tp_cum <- cumsum(response_sorted)
    fp_cum <- cumsum(!response_sorted)
    duplicate_cutoff <- rev(duplicated(rev(prediction_sorted)))
    tp <- c(0, tp_cum[!duplicate_cutoff])
    fp <- c(0, fp_cum[!duplicate_cutoff])

    tpr <- tp / sum(response)
    fpr <- fp / sum(!response)
    precision <- tp / (fp + tp)
    recall <- tpr
    recall[is.na(recall)] <- 0
    precision[is.na(precision)] <- 1

    aupr <- caTools::trapz(recall, precision)
    c(
      auroc = caTools::trapz(fpr, tpr),
      aupr = aupr,
      aupr_corrected = aupr - (sum(response) / length(response)),
      pearson = stats::cor(prediction, response)
    )
  }

  fast_predict_ligand_activities <- function(geneset,
                                             background_expressed_genes,
                                             ligand_target_matrix,
                                             potential_ligands,
                                             single = TRUE,
                                             zyme = TRUE,
                                             ...) {
    if (!isTRUE(zyme) || !isTRUE(single)) {
      return(.nichenetr_orig_predict_ligand_activities(
        geneset = geneset,
        background_expressed_genes = background_expressed_genes,
        ligand_target_matrix = ligand_target_matrix,
        potential_ligands = potential_ligands,
        single = single,
        ...
      ))
    }

    background <- background_expressed_genes[
      (background_expressed_genes %in% geneset) == FALSE
    ]
    response <- c(setNames(rep(FALSE, length(background)), background),
                  setNames(rep(TRUE,  length(geneset)),    geneset))
    matrix_rows <- rownames(ligand_target_matrix)
    idx <- match(names(response), matrix_rows)
    keep <- !is.na(idx)
    if (!any(keep)) {
      stop("Gene names in response don't accord to gene names in ligand-target matrix (did you consider differences human-mouse namings?)")
    }

    response <- response[keep]
    row_idx <- idx[keep]
    pos_row_idx <- sort(row_idx[response])
    neg_row_idx <- sort(row_idx[!response])
    col_idx <- match(potential_ligands, colnames(ligand_target_matrix))
    if (any(is.na(col_idx))) {
      stop("ligand should be in ligand_target_matrix")
    }

    n_ligands <- length(potential_ligands)

    # auto_threads() honors AUTOZYME_THREADS env / set_threads() (capped at 14);
    # the n_ligands and physical-core mins remain as oversubscription guards.
    n_workers <- min(
      autozyme::auto_threads(cap = 14L),
      n_ligands,
      parallel::detectCores(logical = FALSE)
    )

    if (n_ligands >= 64L) {
      score_mat <- score_ligands_cpp(ligand_target_matrix,
                                     as.integer(pos_row_idx),
                                     as.integer(neg_row_idx),
                                     as.integer(col_idx), n_workers)
    } else {
      ligand_target_matrix <- ligand_target_matrix[row_idx, col_idx, drop = FALSE]
      score_list <- lapply(seq_len(n_ligands), function(j) {
        .nichenetr_fast_score_selected_metrics(ligand_target_matrix[, j], response)
      })
      score_mat <- do.call(rbind, score_list)
    }

    dplyr::tibble(
      test_ligand    = potential_ligands,
      auroc          = score_mat[, 1],
      aupr           = score_mat[, 2],
      aupr_corrected = score_mat[, 3],
      pearson        = score_mat[, 4]
    )
  }

  register_patch(
    name = "nichenetr",
    upstream = "nichenetr",
    targets = list(predict_ligand_activities = fast_predict_ligand_activities),
    smoke = list(
      load = function(task_dir, tier) {
        readRDS(resolve_dataset_path(
          task_dir, sprintf("data/inputs_%s.rds", tier)))
      },
      call = function(inputs) {
        suppressWarnings(nichenetr::predict_ligand_activities(
          geneset                    = inputs$geneset,
          background_expressed_genes = inputs$background_expressed_genes,
          ligand_target_matrix       = inputs$ligand_target_matrix,
          potential_ligands          = inputs$potential_ligands
        ))
      },
      save = function(result, dir, ...) {
        ordered <- dplyr::arrange(result, test_ligand)
        saveRDS(list(ligand_activities = ordered, n_ligands = nrow(ordered)),
                file.path(dir, "result.rds"))
      }
    ),
    tested_against = "nichenetr 2.2.1.1",
    tested_upstream_versions = list(nichenetr = "2.2.1.1")
  )
}
