# Universal activation smoke for every patch shipped in autozyme.r.
#
# Mirror of autozyme_py/tests/contract/test_all_patches_activate.py.
# Goal: minimum-viable coverage for every registered patch -- does
# activate -> status -> restore round-trip without crashing?
#
# Skips cleanly when a patch's upstream isn't installed. A failure here
# means the patch is broken at registration / source / binding level,
# which is the cheapest class of bug to catch and worth tripping every
# single CI run.
#
# For deep per-API contracts (see test-contract-NormalizeData.R etc.),
# each Seurat-style API gets its own file. Adding a new patch? Ship at
# least this universal smoke automatically (no edit needed -- it
# enumerates from list_patches()), plus ideally a per-API file.

# Define %||% before usage inside loop closures (R evaluates body at
# call time, but having it module-top keeps the file self-contained).
`%||%` <- function(a, b) if (is.null(a)) b else a

local({
  all_patches <- autozyme::list_patches()

  test_that("list_patches() returns a non-empty character vector", {
    expect_type(all_patches, "character")
    expect_gt(length(all_patches), 0)
  })

  for (patch_name in all_patches) {
    # Closure captures patch_name -- force binding via local() so each
    # test_that block sees its own value rather than the loop's last.
    local({
      pn <- patch_name

      test_that(paste0("activate('", pn, "') round-trips cleanly"), {
        # Skip if the upstream isn't installed locally / in CI. The
        # framework's .probe_patch_installed handles version-string
        # checking and missing-namespace; we trust it.
        probe <- autozyme:::.probe_patch_installed(pn)
        if (!isTRUE(probe$installed)) {
          testthat::skip(paste0("upstream missing for '", pn, "': ",
                                probe$reason %||% "not installed"))
        }
        # End-of-test cleanup: leave the patch in the SAME state
        # setup-autozyme.R set (activated for every installed upstream).
        # If we just restored to "inactive" and walked away, every
        # alphabetically-later test file would see vanilla functions
        # rejecting `zyme = FALSE` -- breaking suite-wide test_dir runs.
        on.exit(try(autozyme::activate(pn), silent = TRUE), add = TRUE)

        # Activation
        result <- autozyme::activate(pn)
        expect_true(isTRUE(result),
                    info = paste0("activate('", pn, "') returned ",
                                  deparse(result),
                                  " despite probe reporting installed"))
        st <- autozyme::status()
        expect_equal(unname(st[pn]), "active",
                     info = paste0("status after activate -> ",
                                   unname(st[pn])))

        # Restore (the round-trip assertion)
        autozyme::restore(pn)
        st <- autozyme::status()
        expect_equal(unname(st[pn]), "inactive",
                     info = paste0("status after restore -> ",
                                   unname(st[pn])))
      })

    })
  }

  test_that("activate() on registered patch with missing upstream returns FALSE", {
    pn <- "missing_upstream_probe"
    on.exit({
      try(autozyme::restore(pn), silent = TRUE)
      if (exists(pn, envir = autozyme:::.zyme_registry, inherits = FALSE)) {
        rm(list = pn, envir = autozyme:::.zyme_registry)
      }
      if (exists(pn, envir = autozyme:::.zyme_probe_cache, inherits = FALSE)) {
        rm(list = pn, envir = autozyme:::.zyme_probe_cache)
      }
    }, add = TRUE)

    autozyme::register_patch(
      name = pn,
      upstream = "autozymeDefinitelyMissingRPackage",
      targets = list(noop = function(...) {
        stop("missing-upstream patch should not activate", call. = FALSE)
      })
    )

    probe <- autozyme:::.probe_patch_installed(pn)
    expect_false(isTRUE(probe$installed))

    result <- tryCatch(autozyme::activate(pn), error = function(e) e)
    expect_false(inherits(result, "error"))
    expect_false(isTRUE(result),
                 info = paste0("activate('", pn,
                               "') with missing upstream returned ",
                               deparse(result)))
  })
})
