# Unit tests for R/subsets.R registry data + the core.R API that reads it
# (list_subsets / subset_patches). Pure data, no upstream needed.

test_that("the three canonical subsets exist and are character vectors", {
  ns <- asNamespace("autozyme")
  subs <- ns$.zyme_subsets
  expect_true(is.list(subs))
  expect_setequal(names(subs),
                  c("scrna_signaling", "scrna_trajectory", "scrna_spatial"))
  for (s in subs) {
    expect_type(s, "character")
    expect_true(length(s) >= 1L)
  }
})

test_that("every subset contains seurat as its anchor patch", {
  ns <- asNamespace("autozyme")
  for (members in ns$.zyme_subsets) {
    expect_true("seurat" %in% members)
  }
})

test_that("list_subsets() returns sorted subset names", {
  s <- list_subsets()
  expect_type(s, "character")
  expect_equal(s, sort(s))
  expect_true(all(c("scrna_signaling", "scrna_trajectory", "scrna_spatial") %in% s))
})

test_that("subset_patches() returns the registered members", {
  expect_equal(subset_patches("scrna_signaling"),
               c("seurat", "cellchat", "nichenetr", "scriabin"))
  expect_equal(subset_patches("scrna_spatial"),
               c("seurat", "bayesspace", "infercnv", "rctd"))
})

test_that("subset_patches() errors on an unknown subset", {
  expect_error(subset_patches("not_a_subset"), "no subset named")
})

test_that(".zyme_conflicts is an (empty) list", {
  ns <- asNamespace("autozyme")
  expect_true(is.list(ns$.zyme_conflicts))
})

test_that(".zyme_upstreams maps known patches to upstream packages", {
  ns <- asNamespace("autozyme")
  up <- ns$.zyme_upstreams
  expect_true(is.list(up))
  expect_equal(up[["seurat"]], "Seurat")
  expect_equal(up[["rctd"]], "spacexr")
  expect_equal(up[["decontx"]], "celda")
  # Every value is a non-empty character scalar (or vector).
  for (v in up) {
    expect_type(v, "character")
    expect_true(all(nzchar(v)))
  }
})

test_that("every subset member has an entry in .zyme_upstreams", {
  ns <- asNamespace("autozyme")
  members <- unique(unlist(ns$.zyme_subsets, use.names = FALSE))
  expect_true(all(members %in% names(ns$.zyme_upstreams)))
})
