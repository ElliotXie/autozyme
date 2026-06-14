test_that("status() returns character on empty registry", {
  expect_true(is.character(status()))
})

test_that("set_threads writes env vars and option", {
  set_threads(3)
  expect_equal(Sys.getenv("OMP_NUM_THREADS"), "3")
  expect_equal(Sys.getenv("OPENBLAS_NUM_THREADS"), "3")
  expect_equal(getOption("autozyme.threads"), 3L)
})

test_that("set_threads rejects bad input", {
  expect_error(set_threads(0))
  expect_error(set_threads(-1))
})

test_that("register + activate + deactivate roundtrip on a synthetic upstream", {
  # Use 'tools' (base R, always available) as a synthetic upstream.
  # Capture original at definition time — referencing tools::file_ext from
  # inside the body would resolve via namespace at call time and hit the
  # patched version, recursing infinitely.
  original_file_ext <- tools::file_ext
  fast_file_ext <- function(x) paste0(original_file_ext(x), "_fast")

  register_patch(
    name = "tools_demo",
    upstream = "tools",
    targets = list(file_ext = fast_file_ext)
  )
  on.exit({
    try(deactivate("tools_demo"), silent = TRUE)
    if (exists("tools_demo", envir = autozyme:::.zyme_registry)) {
      rm("tools_demo", envir = autozyme:::.zyme_registry)
    }
  }, add = TRUE)

  expect_true(activate("tools_demo"))
  expect_equal(status()[["tools_demo"]], "active")
  expect_equal(tools::file_ext("foo.R"), "R_fast")

  deactivate("tools_demo")
  expect_equal(status()[["tools_demo"]], "inactive")
  expect_equal(tools::file_ext("foo.R"), "R")
})

test_that("unknown patch name raises", {
  expect_error(deactivate("does_not_exist"))
  expect_error(activate("does_not_exist"))
})

test_that("register_patch rejects conflicting (upstream, attr)", {
  register_patch("conflict_a", "tools", list(file_ext = function(x) x))
  on.exit(rm("conflict_a", envir = autozyme:::.zyme_registry), add = TRUE)
  expect_error(
    register_patch("conflict_b", "tools", list(file_ext = function(x) x)),
    "already claimed"
  )
})

test_that("register_patch allows disjoint targets on same upstream", {
  register_patch("disjoint_ext", "tools", list(file_ext = function(x) x))
  register_patch("disjoint_path", "tools", list(file_path_sans_ext = function(x) x))
  on.exit({
    rm("disjoint_ext",  envir = autozyme:::.zyme_registry)
    rm("disjoint_path", envir = autozyme:::.zyme_registry)
  }, add = TRUE)
  s <- status()
  expect_true("disjoint_ext"  %in% names(s))
  expect_true("disjoint_path" %in% names(s))
})
