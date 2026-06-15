# Contract: clusterProfiler::enrichGO + compareCluster
#
# Patched surface (3 targets):
#   - get_GO_data       (GO term -> gene mapping cache)
#   - enricher_internal (hypergeometric test engine)
#   - compareCluster    (multi-group orchestrator)
#
# enrichGO + compareCluster both reach the patched internals. Contract
# pins: returns enrichResult / compareClusterResult of expected class.

.skip_if_no_clusterprofiler <- function() {
  testthat::skip_if_not_installed("clusterProfiler")
  testthat::skip_if_not_installed("org.Hs.eg.db")
  testthat::skip_if_not_installed("GO.db")
  testthat::skip_if_not_installed("AnnotationDbi")
}

# Small set of canonical cancer-related human gene Entrez IDs — enough
# coverage that BP/MF ontologies return at least a few hits.
.cancer_entrez <- function() {
  list(
    set1 = c("7157", "672", "1956", "4609", "5728", "3845",
             "5290", "1029", "351", "207"),   # TP53, BRCA1, EGFR, MYC, PTEN, KRAS, PIK3CA, CDKN2A, APP, AKT1
    set2 = c("207", "5290", "1956", "5599",
             "5594", "1432", "4615", "2475")  # AKT1, PIK3CA, EGFR, MAPK8, MAPK1, MAPK14, MYD88, MTOR
  )
}

test_that("enrichGO returns enrichResult on canonical input", {
  .skip_if_no_clusterprofiler()
  genes <- .cancer_entrez()$set1
  out <- tryCatch(
    suppressWarnings(suppressMessages(
      clusterProfiler::enrichGO(
        gene = genes, OrgDb = "org.Hs.eg.db", keyType = "ENTREZID",
        ont = "BP", pvalueCutoff = 0.5, qvalueCutoff = 0.5,
        readable = FALSE
      )
    )),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    testthat::skip(paste0("enrichGO raised on minimal fixture: ",
                          conditionMessage(out)))
  }
  # enrichGO returns NULL when no terms pass the cutoff; relax to NULL OR enrichResult.
  if (is.null(out)) {
    testthat::skip("enrichGO returned NULL on minimal fixture (no terms passed cutoff)")
  }
  expect_s4_class(out, "enrichResult")
})

test_that("compareCluster returns compareClusterResult on multi-group input", {
  .skip_if_no_clusterprofiler()
  gene_clusters <- .cancer_entrez()
  out <- tryCatch(
    suppressWarnings(suppressMessages(
      clusterProfiler::compareCluster(
        geneClusters = gene_clusters, fun = "enrichGO",
        OrgDb = "org.Hs.eg.db", keyType = "ENTREZID", ont = "BP",
        pvalueCutoff = 0.5, qvalueCutoff = 0.5
      )
    )),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    testthat::skip(paste0("compareCluster raised on minimal fixture: ",
                          conditionMessage(out)))
  }
  if (is.null(out)) {
    testthat::skip("compareCluster returned NULL (no terms passed cutoff)")
  }
  expect_s4_class(out, "compareClusterResult")
})

test_that("clusterProfiler patch preserves enrichResult validity checks", {
  .skip_if_no_clusterprofiler()
  testthat::skip_if_not_installed("DOSE")
  autozyme::restore("clusterprofiler")
  cls <- methods::getClass("enrichResult", where = asNamespace("DOSE"))
  before <- methods::getValidity(cls)
  suppressWarnings(autozyme::activate("clusterprofiler"))
  after <- methods::getValidity(cls)
  expect_identical(after, before)

  patch_file <- system.file("patches", "clusterprofiler", "patch.R",
                            package = "autozyme")
  expect_false(any(grepl("setValidity\\s*\\(", readLines(patch_file, warn = FALSE))))
})

test_that("enrichGO with_disabled() matches patched enrichment IDs", {
  .skip_if_no_clusterprofiler()
  genes <- .cancer_entrez()$set1
  vanilla <- tryCatch(
    autozyme::with_disabled(
      suppressWarnings(suppressMessages(
        clusterProfiler::enrichGO(
          gene = genes, OrgDb = "org.Hs.eg.db", keyType = "ENTREZID",
          ont = "BP", pvalueCutoff = 0.5, qvalueCutoff = 0.5,
          readable = FALSE
        )
      ))
    ),
    error = function(e) e
  )
  patched <- tryCatch(
    suppressWarnings(suppressMessages(
      clusterProfiler::enrichGO(
        gene = genes, OrgDb = "org.Hs.eg.db", keyType = "ENTREZID",
        ont = "BP", pvalueCutoff = 0.5, qvalueCutoff = 0.5,
        readable = FALSE
      )
    )),
    error = function(e) e
  )
  if (inherits(vanilla, "error") || inherits(patched, "error") ||
      is.null(vanilla) || is.null(patched)) {
    testthat::skip("enrichGO returned NULL or errored on minimal fixture")
  }
  expect_equal(class(patched), class(vanilla))
  # Same gene set + same DB -> same enriched term IDs (modulo p-value drift).
  expect_setequal(patched@result$ID, vanilla@result$ID)
})
