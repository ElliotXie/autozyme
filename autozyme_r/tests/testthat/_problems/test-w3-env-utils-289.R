# Extracted from test-w3-env-utils.R:289

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
local_mocked_bindings(interactive = function() TRUE, .package = "base")
local_mocked_bindings(.az_py_bind = function() FALSE,
                        install_python_deps = function(...) stop("setup-failed-w3"),
                        .package = "autozyme")
local_mocked_bindings(askYesNo = function(...) TRUE, .package = "utils")
.w3_with_env(list(AUTOZYME_NO_PROMPT = NA), {
    # The install error is caught (lines 227-230 warning), then the final nudge.
    expect_warning(
      expect_message(res <- ns$.az_py_warmup_or_notify("seurat", "PCA"),
                     "acceleration is OFF"),
      "setup-failed-w3")
    expect_false(res)
  })
