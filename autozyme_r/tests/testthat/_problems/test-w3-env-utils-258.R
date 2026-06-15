# Extracted from test-w3-env-utils.R:258

# prequel ----------------------------------------------------------------------
ns <- asNamespace("autozyme")
.w3_with_env <- function(vars, code) {
  old <- vapply(names(vars), function(n) Sys.getenv(n, unset = NA_character_),
                character(1))
  on.exit({
    for (n in names(vars)) {
      v <- old[[n]]
      if (is.na(v)) Sys.unsetenv(n) else do.call(Sys.setenv, setNames(list(v), n))
    }
  }, add = TRUE)
  for (n in names(vars)) {
    v <- vars[[n]]
    if (is.na(v)) Sys.unsetenv(n) else do.call(Sys.setenv, setNames(list(v), n))
  }
  force(code)
}

# test -------------------------------------------------------------------------
skip_if_not_installed("reticulate")
local_mocked_bindings(
    interactive       = function() TRUE,
    .package = "base")
local_mocked_bindings(
    .az_py_bind        = function() TRUE,               # already usable after setup
    install_python_deps = function(...) invisible(TRUE),
    .package = "autozyme")
calls <- 0L
local_mocked_bindings(
    .az_py_bind = function() { calls <<- calls + 1L; calls > 1L },  # 1st FALSE, 2nd TRUE
    install_python_deps = function(...) invisible(TRUE),
    .package = "autozyme")
local_mocked_bindings(askYesNo = function(...) TRUE, .package = "utils")
.w3_with_env(list(AUTOZYME_NO_PROMPT = NA), {
    res <- ns$.az_py_warmup_or_notify("seurat", "PCA")
    expect_true(res)                          # line 231 invisible(TRUE)
    expect_gte(calls, 2L)                     # re-bound after install
  })
