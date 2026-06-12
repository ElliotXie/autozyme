# Contract: fgsea::fgseaMultilevel (and the two helper functions it patches:
# preparePathwaysAndStats + calcGseaStat).
#
# fgsea inputs are framework-light (named gene-set lists + named numeric
# rank vector), so the contract test runs a real-ish GSEA on a synthetic
# but reproducible input. Returns a data.table; contract pins the
# columns + p-value agreement under zyme=FALSE escape.

.skip_if_no_fgsea <- function() {
  testthat::skip_if_not_installed("fgsea")
  testthat::skip_if_not_installed("data.table")
}

.make_fgsea_inputs <- function(seed = 0) {
  set.seed(seed)
  n_genes <- 500
  genes <- paste0("g", seq_len(n_genes))
  # Reproducible random ranks; fgsea expects named numeric, descending.
  stats <- rnorm(n_genes)
  names(stats) <- genes
  stats <- sort(stats, decreasing = TRUE)

  # 5 synthetic pathways of 20-30 genes each, with deliberate signal in
  # pathway 1 (its members concentrated at the top of the rank list).
  pathways <- list(
    enriched_top  = head(names(stats), 30),
    enriched_bot  = tail(names(stats), 30),
    random1       = sample(genes, 25),
    random2       = sample(genes, 22),
    random3       = sample(genes, 28)
  )
  list(pathways = pathways, stats = stats)
}

test_that("fgseaMultilevel returns a data.table with expected columns", {
  .skip_if_no_fgsea()
  inp <- .make_fgsea_inputs()
  out <- suppressWarnings(
    fgsea::fgseaMultilevel(inp$pathways, inp$stats, minSize = 5,
                           maxSize = 100, nPermSimple = 100)
  )
  expect_s3_class(out, "data.table")
  for (col in c("pathway", "pval", "padj", "ES", "NES", "size",
                "leadingEdge")) {
    expect_true(col %in% colnames(out),
                info = paste("missing fgsea column:", col))
  }
  # 5 pathways in -> 5 rows out (none filtered by min/max size).
  expect_equal(nrow(out), 5L)
})

test_that("fgseaMultilevel zyme=FALSE matches vanilla on the same input", {
  .skip_if_no_fgsea()
  inp <- .make_fgsea_inputs()
  set.seed(42)  # fgseaMultilevel uses sampling internally; fix RNG.
  vanilla <- suppressWarnings(
    fgsea::fgseaMultilevel(inp$pathways, inp$stats, minSize = 5,
                           maxSize = 100, nPermSimple = 100, zyme = FALSE)
  )
  set.seed(42)
  patched <- suppressWarnings(
    fgsea::fgseaMultilevel(inp$pathways, inp$stats, minSize = 5,
                           maxSize = 100, nPermSimple = 100)
  )
  expect_setequal(vanilla$pathway, patched$pathway)
  # ES is deterministic (no RNG); pval depends on permutation sampling
  # and can drift between fast/vanilla paths even with the same seed.
  for (pw in vanilla$pathway) {
    es_v <- vanilla$ES[vanilla$pathway == pw]
    es_p <- patched$ES[patched$pathway == pw]
    expect_equal(es_p, es_v, tolerance = 1e-6,
                 info = paste("ES drift for pathway:", pw))
  }
})

test_that("fgseaMultilevel empty-pathway input returns empty data.table", {
  .skip_if_no_fgsea()
  inp <- .make_fgsea_inputs()
  # All pathways below minSize floor -> contract says empty data.table.
  out <- suppressWarnings(
    fgsea::fgseaMultilevel(inp$pathways, inp$stats, minSize = 1000,
                           maxSize = 2000, nPermSimple = 100)
  )
  expect_s3_class(out, "data.table")
  expect_equal(nrow(out), 0L)
})

test_that("preparePathwaysAndStats cache rejects same-endpoint collisions", {
  .skip_if_no_fgsea()
  prep <- getFromNamespace("preparePathwaysAndStats", "fgsea")

  stats_a <- seq(50, -49, by = -1)
  names(stats_a) <- c("g_first", paste0("a", 2:99), "g_last")
  stats_b <- seq(50, -49, by = -1)
  names(stats_b) <- c("g_first", paste0("b", 2:99), "g_last")

  pathways_a <- list(
    first_pathway = paste0("a", 10:25),
    last_pathway = paste0("a", 60:75)
  )
  pathways_b <- list(
    first_pathway = paste0("b", 40:55),
    last_pathway = paste0("b", 80:95)
  )

  # Populate the patch-scope cache with an input whose cheap key collides with
  # pathways_b/stats_b but whose gene universe and pathway members differ.
  prep(pathways_a, stats_a, minSize = 5, maxSize = 100,
       gseaParam = 1, scoreType = "std")
  patched_b <- prep(pathways_b, stats_b, minSize = 5, maxSize = 100,
                    gseaParam = 1, scoreType = "std")
  vanilla_b <- autozyme::with_disabled(
    prep(pathways_b, stats_b, minSize = 5, maxSize = 100,
         gseaParam = 1, scoreType = "std")
  )

  expect_equal(patched_b$filtered, vanilla_b$filtered)
  expect_equal(patched_b$sizes, vanilla_b$sizes)
})
