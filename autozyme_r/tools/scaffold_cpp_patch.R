# Idempotent Rcpp scaffolding for an autozyme C++ patch.
#
# Source this file, then call:
#   scaffold_cpp_patch(cpp_file = "/path/to/x.cpp", pkg_dir = "/path/to/autozyme")
#
# What it does (every step is idempotent — safe to re-run):
#   1. Copy <cpp_file> into <pkg_dir>/src/
#   2. DESCRIPTION:  ensure Imports/LinkingTo include Rcpp + RcppParallel,
#                    SystemRequirements includes "GNU make"
#   3. NAMESPACE:    ensure useDynLib + importFrom(Rcpp, evalCpp) +
#                    importFrom(RcppParallel, RcppParallelLibs)
#   4. src/Makevars: create with the macOS rpath fix if missing
#   5. Run Rcpp::compileAttributes(pkg_dir) to (re)generate RcppExports.{cpp,R}
#
# After this, R CMD INSTALL <pkg_dir> picks up the C++ kernel. R/<task>.R can
# then call the exported function directly (it lives in autozyme's namespace
# — no inline cppFunction!).

scaffold_cpp_patch <- function(cpp_file,
                               pkg_dir = ".",
                               cpp_basename = basename(cpp_file),
                               quiet = FALSE) {
  cpp_file <- normalizePath(cpp_file, mustWork = TRUE)
  pkg_dir  <- normalizePath(pkg_dir,  mustWork = TRUE)
  log_ <- function(...) if (!quiet) message("[scaffold] ", ...)

  # Sanity: this is an R package source tree.
  desc_path <- file.path(pkg_dir, "DESCRIPTION")
  if (!file.exists(desc_path)) {
    stop("DESCRIPTION not found at ", desc_path,
         " — pkg_dir must point to an R package source tree")
  }
  pkg_name <- read.dcf(desc_path, fields = "Package")[1, 1]
  if (is.na(pkg_name)) stop("DESCRIPTION has no Package: field")

  # 1. Copy .cpp into src/.
  src_dir <- file.path(pkg_dir, "src")
  if (!dir.exists(src_dir)) {
    dir.create(src_dir)
    log_("created src/")
  }
  cpp_lines <- readLines(cpp_file, warn = FALSE)
  if (!any(grepl("\\[\\[Rcpp::export\\]\\]", cpp_lines))) {
    warning("No `// [[Rcpp::export]]` tag found in ", cpp_file,
            " — no functions will be exposed to R.")
  }

  # Scan for `// [[Rcpp::depends(Foo, Bar)]]` to discover required Rcpp-* deps.
  # Always include Rcpp itself. Anything else (RcppParallel, RcppEigen,
  # RcppArmadillo, ...) gets added to LinkingTo + Imports.
  dep_pat <- "\\[\\[Rcpp::depends\\(([^)]+)\\)\\]\\]"
  matches <- regmatches(cpp_lines, regexec(dep_pat, cpp_lines))
  declared <- character(0)
  for (m in matches) {
    if (length(m) >= 2L) {
      parts <- trimws(strsplit(m[2], ",")[[1]])
      declared <- c(declared, parts[nzchar(parts)])
    }
  }
  cpp_deps <- unique(c("Rcpp", declared))
  needs_parallel <- "RcppParallel" %in% cpp_deps
  log_("detected cpp deps: ", paste(cpp_deps, collapse = ", "))

  dest <- file.path(src_dir, cpp_basename)
  same_file <- normalizePath(cpp_file, mustWork = TRUE) ==
               (if (file.exists(dest)) normalizePath(dest, mustWork = TRUE) else "")
  if (same_file) {
    log_(cpp_basename, " is already at src/, skipping copy")
  } else {
    file.copy(cpp_file, dest, overwrite = TRUE)
    log_("copied ", cpp_basename, " -> src/")
  }

  # 2. DESCRIPTION updates (idempotent).
  ensure_pkg_field <- function(desc_mat, field, pkgs) {
    current <- if (field %in% colnames(desc_mat)) desc_mat[1, field] else NA_character_
    cur_list <- if (is.na(current)) character(0)
                else trimws(strsplit(current, ",")[[1]])
    cur_list <- cur_list[nzchar(cur_list)]
    bare <- sub("\\s*\\(.*", "", cur_list)
    add <- pkgs[!pkgs %in% bare]
    if (length(add) == 0L) return(list(desc_mat = desc_mat, changed = FALSE))
    combined <- c(cur_list, add)
    new_value <- paste(combined, collapse = ",\n    ")
    if (field %in% colnames(desc_mat)) {
      desc_mat[1, field] <- new_value
    } else {
      desc_mat <- cbind(desc_mat,
                        matrix(new_value, nrow = 1L,
                               dimnames = list(NULL, field)))
    }
    list(desc_mat = desc_mat, changed = TRUE)
  }

  desc <- read.dcf(desc_path)
  any_changed <- FALSE
  desc_specs <- list(
    list(field = "Imports",   pkgs = cpp_deps),
    list(field = "LinkingTo", pkgs = cpp_deps)
  )
  if (needs_parallel) {
    desc_specs <- c(desc_specs, list(
      list(field = "SystemRequirements", pkgs = "GNU make")
    ))
  }
  for (spec in desc_specs) {
    res <- ensure_pkg_field(desc, spec$field, spec$pkgs)
    desc <- res$desc_mat
    if (res$changed) any_changed <- TRUE
  }
  if (any_changed) {
    write.dcf(desc, desc_path, indent = 4, width = 80, keep.white = colnames(desc))
    log_("DESCRIPTION updated (Imports / LinkingTo / SystemRequirements)")
  } else {
    log_("DESCRIPTION already complete")
  }

  # 3. NAMESPACE updates (idempotent, plain text).
  ns_path <- file.path(pkg_dir, "NAMESPACE")
  ns_lines <- if (file.exists(ns_path)) readLines(ns_path) else character(0)
  required <- c(
    sprintf("useDynLib(%s, .registration = TRUE)", pkg_name),
    "importFrom(Rcpp, evalCpp)"
  )
  if (needs_parallel) {
    # RcppParallel ships a runtime .so (tbbParallelFor); forcing its namespace
    # to load before our .so is mandatory or dyn.load fails to resolve symbols.
    required <- c(required, "importFrom(RcppParallel, RcppParallelLibs)")
  }
  added <- character(0)
  for (line in required) {
    if (!any(trimws(ns_lines) == line)) {
      ns_lines <- c(ns_lines, line)
      added <- c(added, line)
    }
  }
  if (length(added)) {
    writeLines(ns_lines, ns_path)
    log_("NAMESPACE: added ", length(added), " line(s)")
  } else {
    log_("NAMESPACE already has useDynLib / importFrom lines")
  }

  # 4. src/Makevars — only needed when RcppParallel is in the dep set
  # (it ships an external library with @rpath quirks on macOS). Header-only
  # deps like RcppEigen / RcppArmadillo work without a Makevars.
  makevars_path <- file.path(src_dir, "Makevars")
  if (needs_parallel) {
    if (!file.exists(makevars_path)) {
      writeLines(c(
        "RCPPPARALLEL_LIBDIR = $(shell \"${R_HOME}/bin/Rscript\" -e \"cat(system.file('lib', package='RcppParallel'))\")",
        "PKG_CXXFLAGS = $(shell \"${R_HOME}/bin/Rscript\" -e \"RcppParallel::CxxFlags()\") -DRCPP_PARALLEL_USE_TBB=1",
        "PKG_LIBS = $(shell \"${R_HOME}/bin/Rscript\" -e \"RcppParallel::RcppParallelLibs()\") -Wl,-rpath,$(RCPPPARALLEL_LIBDIR)",
        "CXX_STD = CXX17"
      ), makevars_path)
      log_("created src/Makevars (RcppParallel rpath fix)")
    } else {
      log_("src/Makevars already exists, leaving alone")
    }
  } else if (!file.exists(makevars_path)) {
    log_("no Makevars needed (header-only deps)")
  }

  # 5. Regenerate RcppExports via compileAttributes.
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("Rcpp must be installed to run compileAttributes()")
  }
  Rcpp::compileAttributes(pkg_dir)
  log_("compileAttributes() completed — RcppExports.{cpp,R} regenerated")

  invisible(list(
    pkg = pkg_name,
    src_file = dest,
    desc_path = desc_path,
    namespace_path = ns_path,
    makevars_path = makevars_path
  ))
}
