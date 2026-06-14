# Unit tests for R/core.R surfaces NOT covered by the existing test-core.R
# (which covers status/set_threads/register+activate+restore/conflict/disjoint).
# Here: the kill switch (is_disabled / with_disabled), the namespace-fast
# dispatcher wrapper, target/kind classification, name resolution + did-you-mean,
# list_patches(installed), deactivate(_all), inspect / env_snapshot / dashboard,
# and the subset resolution path. All use the harmless `tools` upstream or pure
# registry inspection so nothing heavy is touched.

# ---- is_disabled / with_disabled (process kill switch) ---------------------

test_that("is_disabled is FALSE by default and TRUE inside with_disabled", {
  expect_false(is_disabled())
  expect_true(with_disabled(is_disabled()))
  expect_false(is_disabled())               # restored on exit
})

test_that("with_disabled returns the value of its expression", {
  expect_equal(with_disabled(40 + 2), 42)
})

test_that("with_disabled restores the flag even when the body errors", {
  expect_error(with_disabled(stop("boom")), "boom")
  expect_false(is_disabled())
})

test_that("is_disabled honors AUTOZYME_DISABLE / AUTOZYME_DISABLED env vars", {
  old <- Sys.getenv(c("AUTOZYME_DISABLE", "AUTOZYME_DISABLED"), unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
  }, add = TRUE)
  Sys.unsetenv(c("AUTOZYME_DISABLE", "AUTOZYME_DISABLED"))
  expect_false(is_disabled())
  Sys.setenv(AUTOZYME_DISABLE = "1")
  expect_true(is_disabled())
  Sys.setenv(AUTOZYME_DISABLE = "0", AUTOZYME_DISABLED = "yes")
  expect_true(is_disabled())
})

# ---- .wrap_namespace_fast / .zyme_strip_and_forward ------------------------

test_that(".wrap_namespace_fast calls fast normally and original when disabled", {
  ns <- asNamespace("autozyme")
  orig <- function(x) paste0(x, "_orig")
  fast <- function(x) paste0(x, "_fast")
  w <- ns$.wrap_namespace_fast(fast, orig)
  expect_identical(w("a"), "a_fast")
  expect_identical(with_disabled(w("a")), "a_orig")
})

test_that(".wrap_namespace_fast strips zyme=/turbo= before forwarding to original", {
  ns <- asNamespace("autozyme")
  # original takes only `x`; passing zyme/turbo must NOT reach it.
  orig <- function(x) paste0(x, "_orig")
  fast <- function(x, ...) paste0(x, "_fast")
  w <- ns$.wrap_namespace_fast(fast, orig)
  expect_identical(with_disabled(w("a", zyme = FALSE, turbo = TRUE)), "a_orig")
})

test_that(".zyme_strip_and_forward drops zyme/turbo and forwards the rest", {
  ns <- asNamespace("autozyme")
  f <- function(a, b) a + b
  expect_equal(ns$.zyme_strip_and_forward(f, a = 1, b = 2, zyme = FALSE), 3)
  expect_equal(ns$.zyme_strip_and_forward(f, 5, 6, turbo = TRUE), 11)
})

# ---- target classification -------------------------------------------------

test_that(".target_is_s4 distinguishes s4 target lists from plain functions", {
  ns <- asNamespace("autozyme")
  expect_true(ns$.target_is_s4(list(kind = "s4", signature = "X", fn = function() 1)))
  expect_false(ns$.target_is_s4(function() 1))
  expect_false(ns$.target_is_s4(list(kind = "namespace")))
})

# ---- did-you-mean + name resolution ----------------------------------------

test_that(".did_you_mean suggests a near-match and is empty when none", {
  ns <- asNamespace("autozyme")
  expect_match(ns$.did_you_mean("scoda", c("sccoda", "seurat")), "sccoda")
  expect_identical(ns$.did_you_mean("zzzzzz", c("seurat", "vegan")), "")
  expect_identical(ns$.did_you_mean("x", character(0)), "")
})

test_that(".resolve_activation_target expands a subset to its members", {
  ns <- asNamespace("autozyme")
  expect_identical(ns$.resolve_activation_target("scrna_spatial"),
                   c("seurat", "bayesspace", "infercnv", "rctd"))
})

test_that(".resolve_activation_target resolves a registered test patch by name", {
  ns <- asNamespace("autozyme")
  register_patch("res_demo", "tools", list(file_ext = function(x) x))
  on.exit(rm("res_demo", envir = ns$.zyme_registry), add = TRUE)
  expect_identical(ns$.resolve_activation_target("res_demo"), "res_demo")
})

test_that(".resolve_activation_target errors with a did-you-mean on a typo", {
  expect_error(activate("seruat"), "neither a registered patch nor a subset")
})

test_that(".resolve_activation_target dedups a mixed vector of names", {
  ns <- asNamespace("autozyme")
  register_patch("vec_demo", "tools", list(file_ext = function(x) x))
  on.exit(rm("vec_demo", envir = ns$.zyme_registry), add = TRUE)
  out <- ns$.resolve_activation_target(c("vec_demo", "vec_demo"))
  expect_identical(out, "vec_demo")
})

# ---- list_patches / probe --------------------------------------------------

test_that("list_patches() returns sorted unique names incl. test registrations", {
  ns <- asNamespace("autozyme")
  register_patch("aaa_zzz_demo", "tools", list(file_ext = function(x) x))
  on.exit(rm("aaa_zzz_demo", envir = ns$.zyme_registry), add = TRUE)
  lp <- list_patches()
  expect_type(lp, "character")
  expect_identical(lp, sort(unique(lp)))
  expect_true("aaa_zzz_demo" %in% lp)
})

test_that("list_patches(installed=TRUE) is a subset of all patches", {
  all_n <- list_patches()
  inst <- list_patches(installed = TRUE)
  expect_true(all(inst %in% all_n))
  expect_lte(length(inst), length(all_n))
})

test_that(".probe_patch_installed reports installed status + caches it", {
  ns <- asNamespace("autozyme")
  res <- ns$.probe_patch_installed("vegan")
  expect_type(res, "list")
  expect_true(is.logical(res$installed))
  # second call returns the memoized identical result
  expect_identical(ns$.probe_patch_installed("vegan"), res)
})

# ---- deactivate / deactivate_all idempotency -------------------------------

test_that("deactivate on a registered-but-inactive patch is a silent no-op", {
  ns <- asNamespace("autozyme")
  register_patch("deact_demo", "tools", list(file_ext = function(x) x))
  on.exit(rm("deact_demo", envir = ns$.zyme_registry), add = TRUE)
  expect_silent(deactivate("deact_demo"))
  expect_identical(status()[["deact_demo"]], "inactive")
})

test_that("activate then deactivate round-trips a tools patch back to original", {
  ns <- asNamespace("autozyme")
  original_file_ext <- tools::file_ext
  register_patch("rt_demo", "tools",
                 list(file_ext = function(x) paste0(original_file_ext(x), "_rt")))
  on.exit({
    try(deactivate("rt_demo"), silent = TRUE)
    if (exists("rt_demo", envir = ns$.zyme_registry)) rm("rt_demo", envir = ns$.zyme_registry)
  }, add = TRUE)

  expect_true(activate("rt_demo"))
  expect_identical(tools::file_ext("foo.R"), "R_rt")
  deactivate("rt_demo")
  expect_identical(tools::file_ext("foo.R"), "R")
  expect_identical(status()[["rt_demo"]], "inactive")
})

test_that("deactivate is idempotent (calling twice does not error)", {
  ns <- asNamespace("autozyme")
  original_file_ext <- tools::file_ext
  register_patch("idem_demo", "tools",
                 list(file_ext = function(x) paste0(original_file_ext(x), "_i")))
  on.exit({
    if (exists("idem_demo", envir = ns$.zyme_registry)) rm("idem_demo", envir = ns$.zyme_registry)
  }, add = TRUE)
  activate("idem_demo")
  deactivate("idem_demo")
  expect_silent(deactivate("idem_demo"))    # second call no-op
  expect_identical(tools::file_ext("a.txt"), "txt")
})

# ---- inspect / env_snapshot / dashboard ------------------------------------

test_that("inspect() returns a structured view for a registered tools patch", {
  ns <- asNamespace("autozyme")
  # inspect()'s installed-probe resolves the upstream by the PATCH NAME (via
  # .zyme_upstreams, falling back to name == package). A fabricated name would
  # probe as "uninstalled". Register under "tools" so the probe finds the
  # base `tools` package and the structured-view branch is exercised.
  register_patch("tools", "tools", list(file_path_sans_ext = function(x) x))
  on.exit(rm("tools", envir = ns$.zyme_registry), add = TRUE)
  info <- inspect("tools")
  expect_identical(info$name, "tools")
  expect_true(info$status %in% c("active", "inactive"))
  expect_identical(info$upstream, "tools")
  expect_equal(length(info$targets), 1L)
  expect_identical(info$targets[[1]]$fn_name, "file_path_sans_ext")
  expect_identical(info$targets[[1]]$kind, "namespace")
})

test_that("inspect() reports 'uninstalled' for a name whose upstream is absent", {
  ns <- asNamespace("autozyme")
  register_patch("insp_absent_demo", "tools", list(file_ext = function(x) x))
  on.exit(rm("insp_absent_demo", envir = ns$.zyme_registry), add = TRUE)
  # name 'insp_absent_demo' has no .zyme_upstreams entry; falls back to a
  # package of that name, which does not exist -> uninstalled view.
  info <- inspect("insp_absent_demo")
  expect_identical(info$status, "uninstalled")
  expect_true(!is.null(info$error))
})

test_that("inspect() rejects a name that resolves to multiple patches (subset)", {
  expect_error(inspect("scrna_spatial"), "single patch name")
})

test_that("env_snapshot() returns JSON-serializable provenance fields", {
  snap <- env_snapshot()
  expect_setequal(names(snap),
                  c("autozyme_version", "r_version", "platform", "patches"))
  expect_type(snap$autozyme_version, "character")
  expect_true(is.list(snap$patches))
  # each patch entry has at least name + status/error
  for (p in snap$patches) {
    expect_true("name" %in% names(p))
    expect_true(any(c("status", "error") %in% names(p)))
  }
})

test_that("dashboard() prints and returns the snapshot invisibly", {
  out <- capture.output(snap <- dashboard())
  expect_true(any(grepl("patches discovered", out)))
  expect_true(any(grepl("subsets", out)))
  expect_setequal(names(snap),
                  c("autozyme_version", "r_version", "platform", "patches"))
})

# ---- register_patch validation paths ---------------------------------------

test_that("register_patch rejects malformed targets / smoke / metadata", {
  expect_error(register_patch("bad1", "tools", list()))            # no names
  expect_error(register_patch("bad2", "tools", list(x = 1),
                              smoke = list(load = function() 1)))   # incomplete smoke
  expect_error(register_patch("bad3", "tools", list(x = function() 1),
                              tested_upstream_versions = list("no name")))
})

test_that("register_patch accepts valid on_activate / on_deactivate hooks", {
  ns <- asNamespace("autozyme")
  register_patch("hook_demo", "tools", list(file_ext = function(x) x),
                 on_activate = function() invisible(NULL),
                 on_deactivate = function() invisible(NULL))
  on.exit(rm("hook_demo", envir = ns$.zyme_registry), add = TRUE)
  p <- ns$.zyme_registry[["hook_demo"]]
  expect_true(is.function(p$on_activate))
  expect_true(is.function(p$on_deactivate))
})
