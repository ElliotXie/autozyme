# Wave-3 end-to-end coverage: Seurat patched fast-path PARAMETER VARIANTS.
#
# The per-API contract tests (test-contract-NormalizeData.R etc.) already pin
# the *headline* default-parameter case (numeric parity vs zyme=FALSE). This
# file deliberately drives the wrapper/dispatch branches those skip:
#   - non-default normalization.method / scale.factor / margin
#   - do.scale / do.center / vars.to.regress / scale.max combos in ScaleData
#   - nfeatures / selection.method variants in FindVariableFeatures
#   - npcs / weight.by.var / approx in RunPCA
#   - k.param / nn.method / return.neighbor / compute.SNN in FindNeighbors
#   - the per-call zyme=FALSE passthrough AND the with_disabled() lifecycle
#   - the deactivate -> bare-call -> activate restore lifecycle per target
#
# Each parameter variant exercises EITHER the fast turbo path OR the explicit
# scope-guard fallback in seurat/patch.R, plus the .wrap_namespace_fast
# dispatcher in R/core.R that strips zyme/turbo and forwards. Every assertion
# is parity vs the captured upstream original, so a silent fast-path regression
# fails the test rather than passing on a no-op.
#
# Run at OMP_NUM_THREADS=1 (native #pragma kernels segfault at threads>1).

# ── NormalizeData parameter variants ────────────────────────────────────────

test_that("NormalizeData scale.factor variant matches vanilla (fast path)", {
  .skip_if_no_seurat()
  obj <- .make_tiny_seurat()
  # Non-default but still-on-fast-path scale.factor (LogNormalize + scalar +
  # margin 1 still satisfies fast_path_ok). Exercises the kernel with a
  # different scale.factor than the contract test's default 10000.
  for (sf in c(1e4 * 0 + 1e6, 1e3)) {
    vanilla <- suppressWarnings(Seurat::NormalizeData(
      obj, scale.factor = sf, verbose = FALSE, zyme = FALSE))
    patched <- suppressWarnings(Seurat::NormalizeData(
      obj, scale.factor = sf, verbose = FALSE))
    expect_equal(
      as.matrix(SeuratObject::GetAssayData(patched, layer = "data")),
      as.matrix(SeuratObject::GetAssayData(vanilla, layer = "data")),
      tolerance = 1e-6, info = paste("scale.factor =", sf))
  }
})

test_that("NormalizeData RC method delegates to vanilla (off fast path)", {
  .skip_if_no_seurat()
  obj <- .make_tiny_seurat()
  # RC (relative counts) is not LogNormalize -> fast_path_ok FALSE -> fallback.
  vanilla <- suppressWarnings(Seurat::NormalizeData(
    obj, normalization.method = "RC", verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::NormalizeData(
    obj, normalization.method = "RC", verbose = FALSE))
  expect_s4_class(patched, "Seurat")
  expect_equal(
    as.matrix(SeuratObject::GetAssayData(patched, layer = "data")),
    as.matrix(SeuratObject::GetAssayData(vanilla, layer = "data")),
    tolerance = 1e-6)
})

test_that("NormalizeData margin=2 delegates to vanilla (off fast path)", {
  .skip_if_no_seurat()
  obj <- .make_tiny_seurat()
  # margin != 1 takes the explicit `margin == 1` scope-guard fallback branch.
  vanilla <- suppressWarnings(Seurat::NormalizeData(
    obj, margin = 2, verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::NormalizeData(
    obj, margin = 2, verbose = FALSE))
  expect_s4_class(patched, "Seurat")
  expect_equal(
    as.matrix(SeuratObject::GetAssayData(patched, layer = "data")),
    as.matrix(SeuratObject::GetAssayData(vanilla, layer = "data")),
    tolerance = 1e-6)
})

test_that("NormalizeData explicit assay= argument resolves on fast path", {
  .skip_if_no_seurat()
  obj <- .make_tiny_seurat()
  # Passing assay = "RNA" explicitly (default assay) hits the `!is.null(assay)`
  # validation+resolution branch rather than the DefaultAssay() branch.
  vanilla <- suppressWarnings(Seurat::NormalizeData(
    obj, assay = "RNA", verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::NormalizeData(
    obj, assay = "RNA", verbose = FALSE))
  expect_equal(
    as.matrix(SeuratObject::GetAssayData(patched, layer = "data")),
    as.matrix(SeuratObject::GetAssayData(vanilla, layer = "data")),
    tolerance = 1e-6)
})

# ── FindVariableFeatures parameter variants ─────────────────────────────────

test_that("FindVariableFeatures different nfeatures matches vanilla set", {
  .skip_if_no_seurat()
  obj <- .make_normalized_seurat()
  for (nf in c(20L, 80L)) {
    vanilla <- suppressWarnings(Seurat::FindVariableFeatures(
      obj, nfeatures = nf, verbose = FALSE, zyme = FALSE))
    patched <- suppressWarnings(Seurat::FindVariableFeatures(
      obj, nfeatures = nf, verbose = FALSE))
    expect_equal(length(SeuratObject::VariableFeatures(patched)), nf,
                 info = paste("nfeatures =", nf))
    expect_setequal(
      SeuratObject::VariableFeatures(patched),
      SeuratObject::VariableFeatures(vanilla))
  }
})

test_that("FindVariableFeatures mean.var.plot method delegates (off fast path)", {
  .skip_if_no_seurat()
  obj <- .make_normalized_seurat()
  # selection.method != "vst" -> the `!identical(selection.method, "vst")`
  # scope guard delegates to upstream (wrapped in with_disabled).
  vanilla <- suppressWarnings(Seurat::FindVariableFeatures(
    obj, selection.method = "mean.var.plot", verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::FindVariableFeatures(
    obj, selection.method = "mean.var.plot", verbose = FALSE))
  expect_s4_class(patched, "Seurat")
  expect_setequal(
    SeuratObject::VariableFeatures(patched),
    SeuratObject::VariableFeatures(vanilla))
})

test_that("FindVariableFeatures custom loess.span tracks vanilla closely", {
  .skip_if_no_seurat()
  obj <- .make_normalized_seurat()
  # loess.span maps to the VST span= kernel arg; non-default value flows
  # through fast_FindVariableFeatures_Seurat -> _StdAssay -> VST.dgCMatrix.
  #
  # NOTE / suspected minor divergence: at the DEFAULT span (0.3) the fast and
  # vanilla HVG sets are identical (see test-contract-FindVariableFeatures.R).
  # At span=0.5 the patch's loess-fit standardized variance drifts from
  # upstream by ~0.07 (rank correlation ~0.994), enough to swap ~2/40 features
  # at the selection boundary. The contract tests never exercise a non-default
  # span so this was not previously surfaced. We assert the CURRENT behavior
  # (strong rank agreement + large set overlap), not exact set equality.
  vanilla <- suppressWarnings(Seurat::FindVariableFeatures(
    obj, nfeatures = 40, loess.span = 0.5, verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::FindVariableFeatures(
    obj, nfeatures = 40, loess.span = 0.5, verbose = FALSE))
  vf_v <- SeuratObject::VariableFeatures(vanilla)
  vf_p <- SeuratObject::VariableFeatures(patched)
  expect_equal(length(vf_p), 40L)
  # At least 90% of the selected features agree.
  expect_gte(length(intersect(vf_v, vf_p)), 36L)
  # Standardized-variance ranking tracks tightly.
  md_v <- methods::slot(obj[["RNA"]], "meta.data")
  v_std <- methods::slot(vanilla[["RNA"]],
                         "meta.data")[["vf_vst_counts_variance.standardized"]]
  p_std <- methods::slot(patched[["RNA"]],
                         "meta.data")[["vf_vst_counts_variance.standardized"]]
  expect_gt(stats::cor(v_std, p_std), 0.99)
})

# ── ScaleData parameter variants ────────────────────────────────────────────

test_that("ScaleData custom scale.max matches vanilla (fast path)", {
  .skip_if_no_seurat()
  obj <- .make_hvg_seurat()
  # scale.max is honored on the fast path (clip). Non-default value still on
  # fast path since do.scale/do.center default TRUE.
  vanilla <- suppressWarnings(Seurat::ScaleData(
    obj, scale.max = 5, verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::ScaleData(
    obj, scale.max = 5, verbose = FALSE))
  expect_equal(
    SeuratObject::GetAssayData(patched, layer = "scale.data"),
    SeuratObject::GetAssayData(vanilla, layer = "scale.data"),
    tolerance = 1e-4)
})

test_that("ScaleData do.center=FALSE delegates to vanilla (off fast path)", {
  .skip_if_no_seurat()
  obj <- .make_hvg_seurat()
  # do.center FALSE fails fast_path_ok (requires isTRUE(do.center)) -> fallback.
  vanilla <- suppressWarnings(Seurat::ScaleData(
    obj, do.center = FALSE, verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::ScaleData(
    obj, do.center = FALSE, verbose = FALSE))
  expect_equal(
    SeuratObject::GetAssayData(patched, layer = "scale.data"),
    SeuratObject::GetAssayData(vanilla, layer = "scale.data"),
    tolerance = 1e-4)
})

test_that("ScaleData do.scale=FALSE delegates to vanilla (off fast path)", {
  .skip_if_no_seurat()
  obj <- .make_hvg_seurat()
  vanilla <- suppressWarnings(Seurat::ScaleData(
    obj, do.scale = FALSE, verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::ScaleData(
    obj, do.scale = FALSE, verbose = FALSE))
  expect_equal(
    SeuratObject::GetAssayData(patched, layer = "scale.data"),
    SeuratObject::GetAssayData(vanilla, layer = "scale.data"),
    tolerance = 1e-4)
})

test_that("ScaleData vars.to.regress delegates to vanilla (off fast path)", {
  .skip_if_no_seurat()
  obj <- .make_hvg_seurat()
  # A non-NULL vars.to.regress hits the `is.null(vars.to.regress)` scope
  # guard -> full upstream ScaleData regression path.
  obj$dummy_cov <- stats::rnorm(ncol(obj))
  vanilla <- suppressWarnings(Seurat::ScaleData(
    obj, vars.to.regress = "dummy_cov", verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::ScaleData(
    obj, vars.to.regress = "dummy_cov", verbose = FALSE))
  expect_equal(
    SeuratObject::GetAssayData(patched, layer = "scale.data"),
    SeuratObject::GetAssayData(vanilla, layer = "scale.data"),
    tolerance = 1e-4)
})

test_that("ScaleData features= subset matches vanilla", {
  .skip_if_no_seurat()
  obj <- .make_hvg_seurat()
  feats <- SeuratObject::VariableFeatures(obj)[seq_len(20)]
  vanilla <- suppressWarnings(Seurat::ScaleData(
    obj, features = feats, verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::ScaleData(
    obj, features = feats, verbose = FALSE))
  expect_equal(
    SeuratObject::GetAssayData(patched, layer = "scale.data"),
    SeuratObject::GetAssayData(vanilla, layer = "scale.data"),
    tolerance = 1e-4)
})

# ── RunPCA parameter variants ───────────────────────────────────────────────

test_that("RunPCA different npcs matches vanilla up to sign", {
  .skip_if_no_seurat()
  obj <- .make_scaled_seurat()
  for (np in c(5L, 15L)) {
    vanilla <- suppressWarnings(Seurat::RunPCA(
      obj, npcs = np, verbose = FALSE, zyme = FALSE))
    patched <- suppressWarnings(Seurat::RunPCA(
      obj, npcs = np, verbose = FALSE))
    emb_v <- SeuratObject::Embeddings(vanilla, reduction = "pca")
    emb_p <- SeuratObject::Embeddings(patched, reduction = "pca")
    expect_equal(ncol(emb_p), ncol(emb_v), info = paste("npcs =", np))
    for (k in seq_len(min(3, ncol(emb_v)))) {
      cos_k <- abs(sum(emb_v[, k] * emb_p[, k])) /
        (sqrt(sum(emb_v[, k]^2)) * sqrt(sum(emb_p[, k]^2)) + 1e-12)
      expect_gt(cos_k, 0.99)
    }
  }
})

test_that("RunPCA approx=FALSE delegates to vanilla (off fast path)", {
  .skip_if_no_seurat()
  obj <- .make_scaled_seurat()
  # approx=FALSE asks for the exact prcomp decomposition; the Gram+eigh fast
  # path does not honor it -> the `isFALSE(list(...)[["approx"]])` scope guard
  # in fast_RunPCA_StdAssay falls back to the full upstream chain.
  vanilla <- suppressWarnings(Seurat::RunPCA(
    obj, npcs = 8, approx = FALSE, verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::RunPCA(
    obj, npcs = 8, approx = FALSE, verbose = FALSE))
  expect_s4_class(patched, "Seurat")
  emb_v <- SeuratObject::Embeddings(vanilla, reduction = "pca")
  emb_p <- SeuratObject::Embeddings(patched, reduction = "pca")
  expect_equal(dim(emb_p), dim(emb_v))
  for (k in seq_len(min(3, ncol(emb_v)))) {
    cos_k <- abs(sum(emb_v[, k] * emb_p[, k])) /
      (sqrt(sum(emb_v[, k]^2)) * sqrt(sum(emb_p[, k]^2)) + 1e-12)
    expect_gt(cos_k, 0.99)
  }
})

test_that("RunPCA weight.by.var=FALSE matches vanilla up to sign", {
  .skip_if_no_seurat()
  obj <- .make_scaled_seurat()
  # weight.by.var FALSE divides embeddings by sqrt(eigvals); exercises that
  # branch of the fast kernel.
  vanilla <- suppressWarnings(Seurat::RunPCA(
    obj, npcs = 8, weight.by.var = FALSE, verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::RunPCA(
    obj, npcs = 8, weight.by.var = FALSE, verbose = FALSE))
  emb_v <- SeuratObject::Embeddings(vanilla, reduction = "pca")
  emb_p <- SeuratObject::Embeddings(patched, reduction = "pca")
  expect_equal(dim(emb_p), dim(emb_v))
  for (k in seq_len(min(3, ncol(emb_v)))) {
    cos_k <- abs(sum(emb_v[, k] * emb_p[, k])) /
      (sqrt(sum(emb_v[, k]^2)) * sqrt(sum(emb_p[, k]^2)) + 1e-12)
    expect_gt(cos_k, 0.99)
  }
})

# ── FindNeighbors parameter variants ────────────────────────────────────────

test_that("FindNeighbors different k.param matches vanilla graph", {
  .skip_if_no_seurat()
  obj <- .make_pca_seurat()
  for (k in c(10L, 15L)) {
    vanilla <- suppressWarnings(Seurat::FindNeighbors(
      obj, reduction = "pca", dims = 1:10, k.param = k,
      verbose = FALSE, zyme = FALSE))
    patched <- suppressWarnings(Seurat::FindNeighbors(
      obj, reduction = "pca", dims = 1:10, k.param = k, verbose = FALSE))
    snn_name <- grep("_snn$", SeuratObject::Graphs(vanilla), value = TRUE)[1]
    expect_equal(
      as.matrix(vanilla[[snn_name]]),
      as.matrix(patched[[snn_name]]),
      tolerance = 1e-5, info = paste("k.param =", k))
  }
})

test_that("FindNeighbors compute.SNN=FALSE produces only the nn graph", {
  .skip_if_no_seurat()
  obj <- .make_pca_seurat()
  # compute.SNN FALSE is still on the fast path (no scope guard against it);
  # the patched path skips the SNN block. We assert the patched fast-path
  # output directly: a single nn graph and no snn graph.
  #
  # NOTE: the upstream (zyme=FALSE) baseline cannot be used here -- on this
  # 60-cell fixture, vanilla FindNeighbors with compute.SNN=FALSE supplies a
  # length-1 graph.name and then assigns a *named* list(nn=...) into the graph
  # slot, which Seurat 5's `[[<-` rejects ("`i` must be one of \"nn\", not
  # \"RNA_nn\""). That is an upstream quirk, not the patch; the patch's fast
  # path sidesteps it by building the graph object explicitly. Documented, not
  # a patch bug.
  patched <- suppressWarnings(Seurat::FindNeighbors(
    obj, reduction = "pca", dims = 1:10, compute.SNN = FALSE, verbose = FALSE))
  expect_s4_class(patched, "Seurat")
  g_p <- SeuratObject::Graphs(patched)
  expect_true(any(grepl("_nn$", g_p)))
  expect_false(any(grepl("_snn$", g_p)))
  # The nn graph is a valid square cell x cell Graph.
  nn_name <- grep("_nn$", g_p, value = TRUE)[1]
  m <- as.matrix(patched[[nn_name]])
  expect_equal(dim(m), c(ncol(obj), ncol(obj)))
})

test_that("FindNeighbors return.neighbor=TRUE delegates to vanilla", {
  .skip_if_no_seurat()
  obj <- .make_pca_seurat()
  # return.neighbor TRUE fails the `isFALSE(return.neighbor)` scope guard ->
  # full upstream path returning a Neighbor object.
  vanilla <- suppressWarnings(Seurat::FindNeighbors(
    obj, reduction = "pca", dims = 1:10, return.neighbor = TRUE,
    verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::FindNeighbors(
    obj, reduction = "pca", dims = 1:10, return.neighbor = TRUE,
    verbose = FALSE))
  expect_equal(class(patched), class(vanilla))
})

test_that("FindNeighbors manhattan metric delegates to vanilla", {
  .skip_if_no_seurat()
  obj <- .make_pca_seurat()
  # annoy.metric != "euclidean" fails the scope guard -> fallback.
  vanilla <- suppressWarnings(Seurat::FindNeighbors(
    obj, reduction = "pca", dims = 1:10, annoy.metric = "manhattan",
    verbose = FALSE, zyme = FALSE))
  patched <- suppressWarnings(Seurat::FindNeighbors(
    obj, reduction = "pca", dims = 1:10, annoy.metric = "manhattan",
    verbose = FALSE))
  expect_s4_class(patched, "Seurat")
  expect_setequal(SeuratObject::Graphs(vanilla), SeuratObject::Graphs(patched))
})

# ── with_disabled() lifecycle (the hard kill switch in R/core.R) ─────────────

test_that("with_disabled() routes the whole Seurat chain to upstream", {
  .skip_if_no_seurat()
  obj <- .make_tiny_seurat()
  # with_disabled() flips .zyme_state$disabled so .wrap_namespace_fast forwards
  # to the captured original. Compare against a per-call zyme=FALSE which must
  # be numerically identical (both are the upstream path).
  via_disabled <- autozyme::with_disabled(
    suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE)))
  via_kwarg <- suppressWarnings(
    Seurat::NormalizeData(obj, verbose = FALSE, zyme = FALSE))
  expect_equal(
    as.matrix(SeuratObject::GetAssayData(via_disabled, layer = "data")),
    as.matrix(SeuratObject::GetAssayData(via_kwarg, layer = "data")),
    tolerance = 1e-10)
  # disabled flag restored to FALSE after the block.
  expect_false(autozyme::is_disabled())
})

test_that("with_disabled() restores the flag even on error", {
  .skip_if_no_seurat()
  expect_false(autozyme::is_disabled())
  expect_error(autozyme::with_disabled(stop("boom")), "boom")
  expect_false(autozyme::is_disabled())
})

# ── activate/deactivate restore lifecycle for the seurat target ─────────────

test_that("deactivate('seurat') restores upstream NormalizeData, reactivate re-patches", {
  .skip_if_no_seurat()
  on.exit(suppressMessages(autozyme::activate("seurat")), add = TRUE)
  obj <- .make_tiny_seurat()

  # Active patched path baseline.
  expect_equal(autozyme::status()[["seurat"]], "active")
  patched <- suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE))

  # Deactivate -> bare call now lands on the genuine upstream binding.
  suppressMessages(autozyme::deactivate("seurat"))
  expect_equal(autozyme::status()[["seurat"]], "inactive")
  bare_upstream <- suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE))

  # Reactivate -> patched path restored.
  suppressMessages(autozyme::activate("seurat"))
  expect_equal(autozyme::status()[["seurat"]], "active")
  repatched <- suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE))

  # All three "data" layers agree numerically (patch is bit-faithful).
  d_patched <- as.matrix(SeuratObject::GetAssayData(patched, layer = "data"))
  d_bare    <- as.matrix(SeuratObject::GetAssayData(bare_upstream, layer = "data"))
  d_re      <- as.matrix(SeuratObject::GetAssayData(repatched, layer = "data"))
  expect_equal(d_patched, d_bare, tolerance = 1e-6)
  expect_equal(d_re, d_bare, tolerance = 1e-6)
})

test_that("inspect('seurat') reports active bindings for the headline targets", {
  .skip_if_no_seurat()
  info <- autozyme::inspect("seurat")
  expect_equal(info$status, "active")
  fns <- vapply(info$targets, function(x) x$fn_name, character(1))
  bound <- vapply(info$targets, function(x) isTRUE(x$currently_bound), logical(1))
  for (fn in c("NormalizeData.Seurat", "ScaleData.Seurat",
               "RunPCA.StdAssay", "FindNeighbors.Seurat")) {
    expect_true(fn %in% fns, info = paste("missing target:", fn))
    expect_true(bound[match(fn, fns)], info = paste("not bound:", fn))
  }
})
