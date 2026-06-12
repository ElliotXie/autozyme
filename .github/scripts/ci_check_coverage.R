#!/usr/bin/env Rscript
# Tier A meta-test: every registered R patch has a contract test.
#
# Fails CI with a clear diff when a patch is added without a corresponding
# test-contract-<name>.R, or when a contract test exists for a patch that's
# been removed. R-side mirror of .github/scripts/ci_check_coverage.py.

# load_all instead of library() so this works in CI before the autozyme
# package itself is R-CMD-INSTALLed (setup-r-dependencies installs deps
# only, not the package). pkgload comes in via devtools' tree which
# setup-r-dependencies already pulls.
if (requireNamespace("autozyme", quietly = TRUE)) {
  suppressMessages(library(autozyme))
} else {
  suppressMessages(pkgload::load_all("autozyme_r", quiet = TRUE))
}

# Patches that register multiple sub-targets under one patch name and are
# intentionally split into multiple contract test files.
KNOWN_SPLITS <- list(
  seurat = c(
    "FindAllMarkers", "FindIntegrationAnchors", "FindNeighbors",
    "FindVariableFeatures", "NormalizeData", "RunCCA", "RunPCA",
    "ScaleData", "SCTransform", "seurat-s3-stragglers"
  )
)

# Contract tests that don't map 1:1 to a patch (meta / cross-cutting).
META_TESTS <- c("all-patches-activate", "zzz-options")

patches <- autozyme::list_patches()

# Run from repo root in CI; resolve test dir relative to this script.
script_path <- normalizePath(sub("^--file=", "",
                                 grep("^--file=", commandArgs(trailingOnly = FALSE),
                                      value = TRUE)[1]))
repo_root <- dirname(dirname(dirname(script_path)))
test_dir <- file.path(repo_root, "autozyme_r", "tests", "testthat")
test_files <- sub("^test-contract-", "",
                  sub("\\.R$", "",
                      list.files(test_dir, pattern = "^test-contract-.*\\.R$")))

expected <- character()
for (p in patches) {
  if (p %in% names(KNOWN_SPLITS)) {
    expected <- c(expected, KNOWN_SPLITS[[p]])
  } else {
    expected <- c(expected, p)
  }
}
expected <- unique(expected)

missing <- setdiff(expected, test_files)
extra <- setdiff(test_files, c(expected, META_TESTS))

ok <- TRUE
if (length(missing) > 0L) {
  message(sprintf("ERROR: patches without contract test: %s",
                  paste(sort(missing), collapse = ", ")))
  message("  -> add autozyme_r/tests/testthat/test-contract-<name>.R for each.")
  ok <- FALSE
}
if (length(extra) > 0L) {
  message(sprintf("ERROR: contract tests for unknown patches: %s",
                  paste(sort(extra), collapse = ", ")))
  message("  -> remove the stale test, re-add the patch, or extend ",
          "KNOWN_SPLITS / META_TESTS in .github/scripts/ci_check_coverage.R.")
  ok <- FALSE
}

if (!ok) {
  quit(status = 1)
}

cat(sprintf("OK: %d patches -> %d contract tests, fully accounted for.\n",
            length(patches), length(test_files)))
