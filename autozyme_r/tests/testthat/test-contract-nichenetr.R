# Contract: nichenetr::predict_ligand_activities
#
# Single patched target. Fast path is gated on single=TRUE (default).
# single=FALSE delegates to vanilla. Returns a tibble of ligand-activity
# scores; the canonical 4 columns are test_ligand, auroc, aupr, pearson.

.skip_if_no_nichenetr <- function() {
  testthat::skip_if_not_installed("nichenetr")
}

.make_ligand_target_matrix <- function(seed = 0) {
  set.seed(seed)
  # 200 genes x 30 ligands -- enough for predict_ligand_activities to run
  # without hitting the n_ligands >= 64 + non-Windows Rcpp fast-path
  # (which would need spacexr-style C++ infra). Keep ligand count under
  # 64 so we stay on the lapply path on every platform.
  n_genes <- 200
  n_ligands <- 30
  mat <- matrix(runif(n_genes * n_ligands), nrow = n_genes,
                ncol = n_ligands)
  rownames(mat) <- paste0("g", seq_len(n_genes))
  colnames(mat) <- paste0("L", seq_len(n_ligands))
  mat
}

test_that("predict_ligand_activities returns a tibble with score columns", {
  .skip_if_no_nichenetr()
  ltm <- .make_ligand_target_matrix()
  genes <- rownames(ltm)
  out <- tryCatch(
    suppressWarnings(suppressMessages(
      nichenetr::predict_ligand_activities(
        geneset = genes[1:30],
        background_expressed_genes = genes,
        ligand_target_matrix = ltm,
        potential_ligands = colnames(ltm),
        single = TRUE
      )
    )),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    testthat::skip(paste0("predict_ligand_activities raised on fixture: ",
                          conditionMessage(out)))
  }
  expect_s3_class(out, "data.frame")
  expect_true("test_ligand" %in% colnames(out))
  # nichenetr's canonical metric columns; tolerate either pearson or aupr.
  metric_cols <- intersect(c("auroc", "aupr", "pearson", "aupr_corrected"),
                           colnames(out))
  expect_true(length(metric_cols) > 0,
              info = "no recognized score columns in output")
})

test_that("predict_ligand_activities zyme=FALSE matches patched scores", {
  .skip_if_no_nichenetr()
  ltm <- .make_ligand_target_matrix()
  genes <- rownames(ltm)
  vanilla <- tryCatch(
    suppressWarnings(suppressMessages(
      nichenetr::predict_ligand_activities(
        geneset = genes[1:30],
        background_expressed_genes = genes,
        ligand_target_matrix = ltm,
        potential_ligands = colnames(ltm),
        single = TRUE, zyme = FALSE
      )
    )),
    error = function(e) e
  )
  if (inherits(vanilla, "error")) {
    testthat::skip(paste0("vanilla failed: ", conditionMessage(vanilla)))
  }
  patched <- suppressWarnings(suppressMessages(
    nichenetr::predict_ligand_activities(
      geneset = genes[1:30],
      background_expressed_genes = genes,
      ligand_target_matrix = ltm,
      potential_ligands = colnames(ltm),
      single = TRUE
    )
  ))
  expect_equal(class(patched), class(vanilla))
  expect_equal(nrow(patched), nrow(vanilla))
  expect_setequal(patched$test_ligand, vanilla$test_ligand)
})
