.onLoad <- function(libname, pkgname) {
  if (.Platform$OS.type == "windows" &&
      !nzchar(Sys.getenv("PROCESSOR_ARCHITECTURE", unset = ""))) {
    Sys.setenv(PROCESSOR_ARCHITECTURE = "AMD64")
  }
  if (isTRUE(.az_truthy(Sys.getenv("AUTOZYME_DISABLED", unset = ""))) ||
      isTRUE(.az_truthy(Sys.getenv("AUTOZYME_DISABLE", unset = "")))) {
    return(invisible(NULL))
  }
  ver <- utils::packageVersion(pkgname)
  # Raise future's per-export serialization cap. The default 500 MB rejects
  # any realistic Seurat object passed to future_lapply / multisession workers,
  # which is the first parallelism path R users try in iteration rounds on
  # Windows. Only raise -- never lower a user override.
  fg_max <- getOption("future.globals.maxSize")
  if (is.null(fg_max) || !is.finite(fg_max) || fg_max < 16 * 1024^3) {
    options(future.globals.maxSize = 16 * 1024^3)
  }
  # Patches enumerate from inst/patches/ -- see .zyme_available_patches() in
  # core.R. We do not source them here: each is loaded lazily on first
  # activate() to avoid eagerly probing every upstream namespace.
  n_patches <- length(.zyme_available_patches())
  n_subsets <- length(.zyme_subsets)
  if (n_patches > 0L) {
    packageStartupMessage(sprintf(
      "autozyme %s: %d patches available, %d subsets -- autozyme::activate(<name>) to enable",
      ver, n_patches, n_subsets
    ))
  }
}
