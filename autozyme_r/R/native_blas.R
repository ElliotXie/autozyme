# Shared native BLAS helpers for patches that can express their hot path as
# dense DGEMM/crossprod. Patches call the unexported helpers; users can inspect
# backend resolution with blas_info().

.az_blas_disabled <- function() {
  identical(Sys.getenv("AUTOZYME_BLAS_DISABLE", unset = ""), "1")
}

.az_split_paths <- function(paths) {
  paths <- paths[!is.na(paths) & nzchar(paths)]
  if (!length(paths)) return(character())
  unique(unlist(strsplit(paths, .Platform$path.sep, fixed = TRUE),
                use.names = FALSE))
}

.az_norm_paths <- function(paths) {
  paths <- paths[!is.na(paths) & nzchar(paths)]
  if (!length(paths)) return(character())
  unique(normalizePath(paths, winslash = "/", mustWork = FALSE))
}

.az_blas_patterns <- function() {
  if (.Platform$OS.type == "windows") {
    c(".*openblas.*\\.dll$", ".*blis.*\\.dll$", ".*scipy.*blas.*\\.dll$")
  } else if (identical(Sys.info()[["sysname"]], "Darwin")) {
    c(".*openblas.*\\.dylib$", ".*blis.*\\.dylib$", ".*scipy.*blas.*\\.dylib$")
  } else {
    c(".*openblas.*\\.so(\\.[0-9]+)*$", ".*blis.*\\.so(\\.[0-9]+)*$",
      ".*scipy.*blas.*\\.so(\\.[0-9]+)*$")
  }
}

.az_list_blas_files <- function(dirs, recursive = FALSE) {
  dirs <- unique(dirs[nzchar(dirs) & dir.exists(dirs)])
  if (!length(dirs)) return(character())
  out <- character()
  for (d in dirs) {
    for (pat in .az_blas_patterns()) {
      out <- c(out, list.files(d, pattern = pat, recursive = recursive,
                               full.names = TRUE, ignore.case = TRUE))
    }
  }
  unique(out)
}

.az_conda_roots <- function() {
  path_dirs <- .az_split_paths(Sys.getenv("PATH", unset = ""))
  conda_hits <- path_dirs[grepl("(^|/|\\\\)(mini)?conda|anaconda|mambaforge|miniforge",
                                path_dirs, ignore.case = TRUE)]
  roots <- c(Sys.getenv("CONDA_PREFIX", unset = ""),
             Sys.getenv("MAMBA_ROOT_PREFIX", unset = ""),
             Sys.getenv("CONDA_ROOT", unset = ""))
  for (p in conda_hits) {
    parts <- strsplit(normalizePath(p, winslash = "/", mustWork = FALSE),
                      "/", fixed = TRUE)[[1L]]
    hit <- grep("(mini)?conda|anaconda|mambaforge|miniforge", parts,
                ignore.case = TRUE)
    if (length(hit)) {
      roots <- c(roots, paste(parts[seq_len(hit[1L])], collapse = "/"))
    }
  }
  prefix <- Sys.getenv("CONDA_PREFIX", unset = "")
  if (nzchar(prefix) && basename(dirname(prefix)) == "envs") {
    roots <- c(roots, dirname(dirname(prefix)))
  }
  if (.Platform$OS.type == "windows") {
    roots <- c(roots,
               file.path(Sys.getenv("USERPROFILE", unset = ""), "miniconda3"),
               file.path(Sys.getenv("USERPROFILE", unset = ""), "anaconda3"),
               "C:/ProgramData/miniconda3", "C:/ProgramData/anaconda3",
               "D:/miniconda3", "D:/anaconda3")
  }
  unique(normalizePath(roots[nzchar(roots)], winslash = "/", mustWork = FALSE))
}

.az_conda_blas_dirs <- function() {
  roots <- .az_conda_roots()
  envs <- unique(c(roots, unlist(lapply(roots, function(root) {
    Sys.glob(file.path(root, "envs", "*"))
  }), use.names = FALSE)))
  site_dirs <- unlist(lapply(envs, function(env) {
    file.path(env, "Lib", "site-packages",
              c("numpy.libs", "numpy/.libs", "scipy.libs", "scipy/.libs"))
  }), use.names = FALSE)
  c(file.path(envs, "Library", "bin"), site_dirs)
}

.az_blas_paths <- function(include_explicit = TRUE) {
  if (.az_blas_disabled()) return(character())

  explicit <- character()
  if (isTRUE(include_explicit)) {
    explicit <- .az_split_paths(c(
      getOption("autozyme.openblas.dll", NULL),
      Sys.getenv("AUTOZYME_OPENBLAS_DLL", unset = ""),
      Sys.getenv("AUTOZYME_BLAS_DLL", unset = ""),
      Sys.getenv("OPENBLAS_DLL", unset = "")
    ))
  }

  pkg_libs <- system.file("libs", package = "autozyme")
  auto <- c(
    .az_list_blas_files(pkg_libs, recursive = TRUE),
    .az_list_blas_files(.az_conda_blas_dirs(), recursive = FALSE),
    .az_list_blas_files(.az_split_paths(Sys.getenv("PATH", unset = "")),
                        recursive = FALSE)
  )

  # Generic blas.dll/libblas is intentionally not auto-discovered. It is
  # accepted only when the user points an explicit env var or option at it.
  .az_norm_paths(c(explicit, auto))
}

.az_blas_info <- function(paths = .az_blas_paths()) {
  if (.az_blas_disabled()) {
    return(list(available = FALSE, backend = NA_character_,
                path = NA_character_, abi = NA_character_,
                dgemm_symbol = NA_character_, thread_control = FALSE,
                threads = NA_integer_, disabled = TRUE))
  }
  info <- az_blas_info(paths)
  info$disabled <- FALSE
  info
}

#' Inspect autozyme's optional dynamic BLAS backend
#'
#' @return A list describing the loaded dynamic BLAS backend, or
#'   \code{available = FALSE} when none is available.
#' @export
blas_info <- function() {
  .az_blas_info()
}

.az_as_dense_double <- function(x) {
  if (is.matrix(x)) {
    if (!is.double(x)) storage.mode(x) <- "double"
    return(x)
  }
  if (is.numeric(x) && is.null(dim(x))) {
    return(matrix(as.numeric(x), ncol = 1L))
  }
  NULL
}

.az_base_gemm <- function(A, B, transA = FALSE, transB = FALSE) {
  lhs <- if (isTRUE(transA)) t(A) else A
  rhs <- if (isTRUE(transB)) t(B) else B
  lhs %*% rhs
}

.az_gemm <- function(A, B, transA = FALSE, transB = FALSE,
                     threads = 0L, fallback = TRUE, patch = NULL) {
  transA <- isTRUE(transA)
  transB <- isTRUE(transB)
  threads <- suppressWarnings(as.integer(threads))
  if (is.na(threads)) threads <- 0L

  A_native <- .az_as_dense_double(A)
  B_native <- .az_as_dense_double(B)
  can_native <- .az_dynamic_blas_enabled(patch = patch, default = TRUE) &&
    !is.null(A_native) && !is.null(B_native)

  if (can_native) {
    res <- tryCatch(
      az_blas_gemm(A_native, B_native, transA = transA, transB = transB,
                   threads = threads, dll_paths = .az_blas_paths()),
      error = function(e) {
        if (isTRUE(fallback)) return(NULL)
        stop(e)
      }
    )
    if (!is.null(res)) return(res)
  } else if (!isTRUE(fallback)) {
    stop("az_gemm: inputs must be dense numeric matrices or vectors")
  }
  .az_base_gemm(A, B, transA = transA, transB = transB)
}

.az_crossprod <- function(X, Y = NULL, threads = 0L, fallback = TRUE,
                          patch = NULL) {
  if (is.null(Y)) {
    .az_gemm(X, X, transA = TRUE, transB = FALSE,
             threads = threads, fallback = fallback, patch = patch)
  } else {
    .az_gemm(X, Y, transA = TRUE, transB = FALSE,
             threads = threads, fallback = fallback, patch = patch)
  }
}

.az_xtx <- function(X, threads = 0L, fallback = TRUE, patch = NULL) {
  .az_crossprod(X, threads = threads, fallback = fallback, patch = patch)
}

.az_default_blas_threads <- function() {
  raw <- Sys.getenv("ZYME_THREADS", unset = "")
  n <- if (nzchar(raw)) suppressWarnings(as.integer(raw)) else NA_integer_
  if (is.na(n) || n <= 0L) {
    n <- suppressWarnings(as.integer(getOption("autozyme.threads", NA_integer_)))
  }
  if (is.na(n) || n <= 0L) n <- auto_threads()
  if (is.na(n) || n <= 0L) 1L else n
}

.az_dynamic_blas_enabled <- function(patch = NULL, default = TRUE) {
  if (.az_blas_disabled()) return(FALSE)
  global <- .az_first_truthy(c(
    Sys.getenv("AUTOZYME_DYNAMIC_BLAS", unset = NA_character_),
    .az_option_values("dynamic_blas")
  ))
  if (!is.na(global) && !isTRUE(global)) return(FALSE)
  if (!is.null(patch) && nzchar(patch)) {
    key <- gsub("[^A-Za-z0-9]+", "_", toupper(patch))
    env <- Sys.getenv(paste0("AUTOZYME_", key, "_BLAS"), unset = "")
    parsed <- .az_truthy(env)
    if (!is.na(parsed)) return(parsed)

    opt <- getOption(paste0("autozyme.", patch, ".dynamic_blas"), NULL)
    parsed <- .az_truthy(opt)
    if (!is.na(parsed)) return(parsed)
  }

  .az_feature_enabled("dynamic_blas", patch = patch, default = default)
}

.az_windows_crossprod <- function(X, Y = NULL, threads = .az_default_blas_threads(),
                                  fallback = TRUE, enabled = TRUE,
                                  patch = NULL) {
  if (.Platform$OS.type == "windows" && isTRUE(enabled)) {
    return(.az_crossprod(X, Y, threads = threads, fallback = fallback,
                         patch = patch))
  }
  crossprod(X, Y)
}
