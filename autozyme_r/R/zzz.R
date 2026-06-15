.onLoad <- function(libname, pkgname) {
  if (.Platform$OS.type == "windows" &&
      !nzchar(Sys.getenv("PROCESSOR_ARCHITECTURE", unset = ""))) {
    Sys.setenv(PROCESSOR_ARCHITECTURE = "AMD64")
  }
  # Windows: if the package ships a bundled OpenBLAS stack
  # (inst/libs/x64/libopenblas.dll + libgfortran/libgcc/libquadmath/
  # libwinpthread/libgomp), prepend that directory to PATH so
  # native_pca.cpp's runtime LoadLibraryA("libopenblas.dll") finds it and
  # its transitive deps resolve. This makes the Python-free RunPCA /
  # RunCCA fast path the default on a vanilla Win 10/11 + R >= 4.2 machine
  # with no Rtools / no conda. If the bundle is absent (source install
  # without the binaries), the native_pca resolver silently falls back to
  # user env (AUTOZYME_OPENBLAS_DLL / OPENBLAS_DLL) or PATH, then ultimately
  # to scipy — exact pre-bundle behavior.
  if (.Platform$OS.type == "windows") {
    libs_dir <- system.file("libs", .Platform$r_arch, package = pkgname)
    if (nzchar(libs_dir) &&
        file.exists(file.path(libs_dir, "libopenblas.dll"))) {
      cur_path <- Sys.getenv("PATH")
      if (!grepl(libs_dir, cur_path, fixed = TRUE)) {
        Sys.setenv(PATH = paste(libs_dir, cur_path, sep = ";"))
      }
    }
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
