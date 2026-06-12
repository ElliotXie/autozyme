# helpers.R — Shared utilities for autozyme tasks (R/Seurat/Bioconductor).
#
# This file is READ-ONLY during experiments. Tasks source it in their
# pipeline/run.R for install_override / time_it / peak memory tracking.
#
# Convention: pipeline/run.R prints `speed_sec: <float>` and `peak_mb: <float>`
# to stdout; the zyme runner greps these for the summary.

# The autozyme-framework directory always has this name. Code that needs to
# locate the framework programmatically (find_framework_root) reads it from
# here. Task-side bootstraps inline the literal string because they run
# before helpers.R has been sourced.
FRAMEWORK_DIR_NAME <- "autozyme-framework"


# Walk up from `start` to locate the autozyme-framework root.
# Identifies the framework by a directory named FRAMEWORK_DIR_NAME whose
# `autozyme_cli/` child exists. Returns absolute path; stops with `stop()`
# if not found within `max_depth` levels.
find_framework_root <- function(start = NULL, max_depth = 8L) {
  if (is.null(start)) start <- getwd()
  cur <- normalizePath(start, mustWork = FALSE)
  for (i in seq_len(max_depth + 1L)) {
    cand <- file.path(cur, FRAMEWORK_DIR_NAME)
    if (dir.exists(cand) && dir.exists(file.path(cand, "autozyme_cli"))) {
      return(normalizePath(cand))
    }
    parent <- dirname(cur)
    if (identical(parent, cur)) break
    cur <- parent
  }
  stop(sprintf("%s not found above %s (searched %d levels)",
               FRAMEWORK_DIR_NAME, start, max_depth))
}


# Track which overrides have emitted [override active] (one-shot marker
# for `with_override_timing` deprecated path). The full per-call timing
# pipeline that previously fed `[override summary]` lines was removed —
# use `zyme profile` for hit counts and call timing.
.zyme_overrides_seen <- new.env(parent = emptyenv())


record_override_timing <- function(full_name, elapsed_s) {
  # Deprecated no-op: the override_summary pipeline that consumed this has
  # been removed. Use `zyme profile` for timing data. Kept so existing task
  # code that calls it doesn't error.
  invisible(NULL)
}


with_override_timing <- function(full_name, expr) {
  # Deprecated: previously fed per-call timing into `[override summary]`.
  # Now just emits a one-shot `[override active]` marker (agent sanity
  # check that the lexically-shadowing wrapper actually fired) and forwards
  # the expression. For timing, use `zyme profile` or `time_it()`.
  if (!isTRUE(get0(full_name, envir = .zyme_overrides_seen, inherits = FALSE))) {
    message(sprintf("[override active] %s", full_name))
    assign(full_name, TRUE, envir = .zyme_overrides_seen)
  }
  force(expr)
}


# Base / stdlib R packages: `assignInNamespace` from a function environment
# is rejected ("locked binding" / "cannot change value"). Used by both
# `install_override` and `patch_namespace` to short-circuit with a helpful
# diagnostic instead of R's stock locked-binding message.
.zyme_base_pkgs <- c("base", "stats", "graphics", "grDevices",
                     "utils", "datasets", "methods", "tools", "compiler")


# install_override — DEPRECATED alias for `patch_namespace`.
#
# Previously this wrapped `new_func` in a per-call `marker_fn` that emitted
# `[override active] pkg::fn` (first call) and `[override summary] ...`
# (every call) for the framework's profile pipeline to parse. That pipeline
# has been removed — use `zyme profile` for hit counts and timing.
#
# New code should call `patch_namespace(func_name, package_name, new_func)`
# directly. This alias preserves the (name, pkg, new_func) signature so
# existing pipeline/run.{R} call sites keep working without modification;
# it will be removed in a future release.
install_override <- function(func_name, package_name, new_func) {
  patch_namespace(func_name, package_name, new_func)
}


# Backwards-compat alias — old code still calls inject_override.
# Note the argument order shift: install_override is (name, pkg, fn);
# inject_override was (name, fn, pkg).
inject_override <- function(func_name, new_func, package_name = "Seurat") {
  install_override(func_name, package_name, new_func)
}


# patch_namespace — bare namespace replacement, NO instrumentation.
#
# Same patching semantics as install_override (assignInNamespace + package: env
# alias + verify, with the same locked-binding diagnostic), but does NOT wrap
# `new_func` in marker_fn. Use when:
#   - the override fires inside a hot loop and per-call instrumentation tax
#     would dominate (R: ~100µs/call cat+flush in mclapply children);
#   - you want to know hit-count / timing via `zyme profile` (cProfile/Rprof)
#     instead of the framework's parsed log lines.
#
# CAVEAT (R-specific): if the patched function is called from PARENT process
# BEFORE `parallel::mclapply` forks, every forked worker inherits a dirty MAST
# namespace and may pay a measurable lookup-cache penalty. Validated on MAST
# r18: +171% wall-time even with bare assignInNamespace. For that "pre-fork
# one-shot" pattern, prefer inline_upstream() which never mutates upstream
# namespace.
#
# Args identical to install_override; returns the original function invisibly.
patch_namespace <- function(func_name, package_name, new_func) {
  full_name <- paste0(package_name, "::", func_name)

  if (package_name %in% .zyme_base_pkgs) {
    stop(
      sprintf("patch_namespace: replacing %s::%s is not supported.\n", package_name, func_name),
      "  Base/stdlib R namespaces refuse `assignInNamespace` from a function\n",
      "  environment in modern R. Use inline_upstream() to clone the caller\n",
      "  and shim its environment instead.",
      call. = FALSE
    )
  }

  ns <- tryCatch(asNamespace(package_name), error = function(e) NULL)
  if (is.null(ns)) {
    swap_hint <- ""
    swapped_ns <- tryCatch(asNamespace(func_name), error = function(e) NULL)
    if (!is.null(swapped_ns)) {
      swap_hint <- sprintf(
        "\n  Did you swap the arguments? Try: patch_namespace(\"%s\", \"%s\", ...)",
        package_name, func_name)
    }
    stop(sprintf("patch_namespace: package '%s' is not loadable.%s",
                 package_name, swap_hint))
  }
  original <- tryCatch(get(func_name, envir = ns, inherits = FALSE),
                       error = function(e) NULL)
  if (is.null(original)) {
    swap_hint <- ""
    swapped <- tryCatch(get(package_name, envir = ns, inherits = FALSE),
                        error = function(e) NULL)
    if (!is.null(swapped)) {
      swap_hint <- sprintf(
        "\n  Did you swap the arguments? Try: patch_namespace(\"%s\", \"%s\", ...)",
        package_name, func_name)
    }
    stop(sprintf("patch_namespace: '%s' not found in '%s' (replaces existing bindings only).%s",
                 func_name, package_name, swap_hint))
  }

  # 1. Patch the package's namespace (private + exported names resolve here).
  tryCatch(
    utils::assignInNamespace(func_name, new_func, ns = package_name),
    error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("locked binding|locked environment|cannot add bindings|cannot change value",
                msg, ignore.case = TRUE)) {
        stop(
          sprintf("patch_namespace: cannot patch %s — namespace binding is locked.\n", full_name),
          sprintf("  Underlying error: %s\n", msg),
          "  Try inline_upstream(\"<calling_pkg>::<calling_fn>\") instead — clone\n",
          "  the caller, mutate its shim env, and call the clone in pipeline/run.\n",
          "  That avoids assignInNamespace entirely.",
          call. = FALSE
        )
      }
      stop(e)
    }
  )

  # 2. Patch the exported package environment if `library(<pkg>)` was called.
  pkg_env_name <- paste0("package:", package_name)
  if (pkg_env_name %in% search()) {
    env <- as.environment(pkg_env_name)
    if (exists(func_name, envir = env, inherits = FALSE)) {
      tryCatch({
        if (bindingIsLocked(func_name, env)) unlockBinding(func_name, env)
        assign(func_name, new_func, envir = env)
        lockBinding(func_name, env)
      }, error = function(e) {
        message(sprintf("[patch_namespace] WARN: package-env patch failed for %s: %s",
                        full_name, e$message))
      })
    }
  }

  # 3. Verify (fail loud — silent override failure is the worst possible bug).
  bound <- tryCatch(getFromNamespace(func_name, package_name),
                    error = function(e) NULL)
  if (!identical(bound, new_func)) {
    stop(sprintf("patch_namespace(%s) FAILED: namespace binding did not update",
                 full_name))
  }

  message(sprintf("[patch_namespace] %s patched + verified", full_name))
  invisible(original)
}


# patch_call_site — replace a .Call / .Fortran / .External inside an R function.
#
# When a function calls .Call("c_kernel", ...) directly and there is no R
# wrapper between the caller and the C kernel, patch_namespace cannot help.
# This helper deparses the function body, replaces the matching call site
# with your replacement expression, reparses, and installs via patch_namespace.
#
# Args:
#   package_name: package that owns the function (e.g. "WGCNA")
#   func_name:    function to patch (e.g. "blockwiseModules")
#   pattern:      string to find in the deparsed body (e.g. '.Call("tomSimilarity_call"')
#   replacement:  string to substitute (e.g. 'fast_tom(')
#
# Returns the original function invisibly, like patch_namespace.
patch_call_site <- function(package_name, func_name, pattern, replacement) {
  full_name <- paste0(package_name, "::", func_name)
  ns <- tryCatch(asNamespace(package_name), error = function(e) NULL)
  if (is.null(ns))
    stop(sprintf("patch_call_site: package '%s' is not loadable", package_name))
  original <- tryCatch(get(func_name, envir = ns, inherits = FALSE),
                       error = function(e) NULL)
  if (is.null(original) || !is.function(original))
    stop(sprintf("patch_call_site: '%s' not found (or not a function) in '%s'",
                 func_name, package_name))

  body_text <- deparse(body(original), width.cutoff = 500L)
  if (!any(grepl(pattern, body_text, fixed = TRUE)))
    stop(sprintf("patch_call_site: pattern %s not found in body of %s",
                 dQuote(pattern, q = FALSE), full_name))

  patched_text <- gsub(pattern, replacement, body_text, fixed = TRUE)
  new_body <- tryCatch(
    parse(text = paste(patched_text, collapse = "\n"))[[1]],
    error = function(e) stop(sprintf(
      "patch_call_site: reparse failed after substitution in %s: %s",
      full_name, e$message))
  )

  patched_fn <- original
  body(patched_fn) <- new_body
  environment(patched_fn) <- new.env(parent = environment(original))

  patch_namespace(func_name, package_name, patched_fn)
  invisible(original)
}


# inline_upstream — clone an upstream function with an editable shim env.
#
# Returns a clone of `qualified_name` whose enclosing environment is a fresh
# env (parent = original's environment). Mutating that env adds shim overrides
# WITHOUT touching the upstream package namespace.
#
# Args:
#   qualified_name: "pkg::fn" (e.g. "MAST::zlm").
#
# Returns: a function. `attr(., "shim")` exposes the mutable shim env for the
# caller to add overrides via `attr(cloned, "shim")$inner_helper <- fast_fn`.
#
# Coverage: catches name lookups in the cloned function's OWN body. Inner
# helpers defined in the upstream namespace use THEIR own environment chain
# (not the shim) — for those, use patch_namespace.
#
# Edge notes:
#   - Byte-compiled functions: shim still works (R looks up names at runtime).
#   - S4 generics: shim can short-circuit the generic entirely by shadowing
#     the generic's name; for method-specific overrides, use setMethod().
#
# Example (the MAST ebayes case that bare patch_namespace can't safely solve):
#   my_zlm <- inline_upstream("MAST::zlm")
#   attr(my_zlm, "shim")$ebayes <- fast_ebayes
#   result <- my_zlm(formula, sca)  # uses fast_ebayes; MAST namespace untouched
inline_upstream <- function(qualified_name) {
  parts <- strsplit(qualified_name, "::", fixed = TRUE)[[1]]
  if (length(parts) != 2L || any(!nzchar(parts))) {
    stop(sprintf("inline_upstream: expected 'package::function', got %s",
                 sQuote(qualified_name)))
  }
  pkg <- parts[1]
  fn  <- parts[2]

  ns <- tryCatch(asNamespace(pkg), error = function(e) NULL)
  if (is.null(ns)) {
    stop(sprintf("inline_upstream: package '%s' not loadable", pkg))
  }
  original <- tryCatch(get(fn, envir = ns, inherits = FALSE),
                       error = function(e) NULL)
  if (is.null(original)) {
    stop(sprintf("inline_upstream: '%s' not found in '%s'", fn, pkg))
  }
  if (!is.function(original)) {
    stop(sprintf("inline_upstream: %s is not a function (got %s)",
                 qualified_name, class(original)[1]))
  }

  cloned <- original
  shim <- new.env(parent = environment(original))
  environment(cloned) <- shim
  attr(cloned, "shim") <- shim
  cloned
}


# Time a single call. Returns list(result=..., elapsed=<seconds>).
time_it <- function(expr_func, ...) {
  t0 <- Sys.time()
  res <- expr_func(...)
  list(result = res, elapsed = as.numeric(difftime(Sys.time(), t0, units = "secs")))
}


# getrusage(RUSAGE_SELF).ru_maxrss — true peak RSS via Rcpp.
# Compiled once per session (Rcpp caches the .so by code hash).
# Returns NULL if Rcpp is unavailable.
.zyme_getrusage_maxrss_mb <- function() {
  if (!requireNamespace("Rcpp", quietly = TRUE)) return(NULL)
  if (!exists(".zyme_ru_maxrss_fn", envir = .GlobalEnv)) {
    tryCatch({
      Rcpp::sourceCpp(code = '
#include <Rcpp.h>
#include <sys/resource.h>
// [[Rcpp::export(name = ".zyme_ru_maxrss_impl")]]
double zyme_ru_maxrss_impl() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1.0;
#ifdef __APPLE__
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#else
    return (double)ru.ru_maxrss / 1024.0;
#endif
}
', verbose = FALSE)
      assign(".zyme_ru_maxrss_fn", TRUE, envir = .GlobalEnv)
    }, error = function(e) {
      assign(".zyme_ru_maxrss_fn", FALSE, envir = .GlobalEnv)
    })
  }
  if (isTRUE(get(".zyme_ru_maxrss_fn", envir = .GlobalEnv))) {
    return(.zyme_ru_maxrss_impl())
  }
  NULL
}

# Peak memory in MB (best-effort).
#
# Fallback chain:
#   1. Windows: ps::ps_memory_info()$peak_wset — true peak working set.
#   2. Linux: /proc/self/status VmHWM — true peak RSS (KiB).
#   3. macOS / other Unix: getrusage(RUSAGE_SELF).ru_maxrss via Rcpp.
#      Kernel-tracked true peak RSS including ALL allocations.
#      Rcpp::sourceCpp caches the .so — first call ~2s, subsequent 0.
#   4. gc() max-used — R-managed heap only, undercounts native. Last resort.
#
# BUG FIXED (2026-05-27): the old macOS path used ps::rss which is CURRENT
# RSS at call time, NOT peak. If the pipeline gc'd large temporaries before
# emit_summary(), the "peak" was artificially low (e.g. 1.9 GB reported when
# true peak was 14 GB). getrusage gives the real answer.
peak_memory_mb <- function() {
  tryCatch({
    if (.Platform$OS.type == "windows") {
      if (requireNamespace("ps", quietly = TRUE)) {
        info <- ps::ps_memory_info(ps::ps_handle())
        return(as.numeric(info[["peak_wset"]]) / 1024^2)
      }
      .zyme_warn_gc_fallback()
      g <- gc(reset = FALSE)
      return(sum(g[, ncol(g)]))
    }
    if (file.exists("/proc/self/status")) {
      lines <- readLines("/proc/self/status")
      peak <- grep("^VmHWM:", lines, value = TRUE)
      if (length(peak) > 0) {
        val_kb <- as.numeric(sub("VmHWM:\\s+(\\d+).*", "\\1", peak[1]))
        return(val_kb / 1024)
      }
    }
    # macOS / other Unix without /proc: use getrusage for true peak.
    val <- .zyme_getrusage_maxrss_mb()
    if (!is.null(val) && val > 0) return(val)
    .zyme_warn_gc_fallback()
    g <- gc(reset = FALSE)
    sum(g[, ncol(g)])
  }, error = function(e) 0.0)
}

.zyme_warn_gc_fallback <- function() {
  # Warn once per R session (not per peak_memory_mb call).
  if (isTRUE(getOption("zyme.peak_mb_gc_warned", FALSE))) return(invisible())
  options(zyme.peak_mb_gc_warned = TRUE)
  message(
    "[peak_mb] warning: ps package not available; falling back to gc() ",
    "for peak memory. This counts only R-managed heap and UNDERCOUNTS ",
    "Rcpp / RcppParallel / FAISS / native-library allocations (often by ",
    "GBs on autozyme tasks). Install ps with `install.packages(\"ps\")` ",
    "to get accurate peak working set."
  )
}


# with_profile — wrap a block in a profiler when ZYME_PROFILE=1 is set.
#
# Backend selected by ZYME_PROFILE_BACKEND (default cpu):
#   cpu  → Rprof, sampling 5ms, line-level, no mem (writes Rprof.out)
#   full → Rprof at 1ms with memory + profvis HTML render (writes
#          Rprof.out + profvis.html; profvis is opt-in install)
#   mem  → Rprof at 5ms with memory.profiling=TRUE, emphasis on
#          mem.total in stderr render (writes Rprof.out)
#
# All three write Rprof.out to ZYME_PROFILE_DIR when set by `zyme profile`;
# otherwise they fall back to getwd() for direct helper-level debugging.
# The CLI parses it the same way and produces a normalized profile.json.
# Differences are in capture interval, memory inclusion, and the optional
# profvis viewer.
#
# When ZYME_PROFILE is unset (the normal `zyme run` path), with_profile is
# a no-op pass-through — zero overhead, zero behavior change.
#
# For sub-block timing inside opaque native code (Rcpp / BLAS / dispatch),
# use with_subprofile() — it stays useful regardless of backend.
with_profile <- function(expr) {
  if (!nzchar(Sys.getenv("ZYME_PROFILE", ""))) {
    return(force(expr))
  }
  backend <- tolower(Sys.getenv("ZYME_PROFILE_BACKEND", "cpu"))
  if (backend == "cpu") {
    .with_profile_cpu(expr)
  } else if (backend == "full") {
    .with_profile_full(expr)
  } else if (backend == "mem") {
    .with_profile_mem(expr)
  } else {
    message(sprintf("[profile] unknown ZYME_PROFILE_BACKEND=%s; falling back to cpu",
                    backend))
    .with_profile_cpu(expr)
  }
}


.with_profile_cpu <- function(expr) {
  prof_path <- .profile_output_path("Rprof.out")
  Rprof(filename = prof_path, interval = 0.005, line.profiling = TRUE,
        memory.profiling = FALSE)
  on.exit({
    Rprof(NULL)
    .summarize_rprof(prof_path, kind = "cpu")
  }, add = TRUE)
  message(sprintf("[profile active] backend=Rprof kind=sampling unit=wall_seconds interval=5ms writing to %s",
                  prof_path))
  force(expr)
}


.with_profile_full <- function(expr) {
  prof_path <- .profile_output_path("Rprof.out")
  has_profvis <- requireNamespace("profvis", quietly = TRUE)
  has_htmlwidgets <- requireNamespace("htmlwidgets", quietly = TRUE)
  Rprof(filename = prof_path, interval = 0.001, line.profiling = TRUE,
        memory.profiling = TRUE)
  on.exit({
    Rprof(NULL)
    .summarize_rprof(prof_path, kind = "full")
    if (has_profvis && has_htmlwidgets && file.exists(prof_path)) {
      tryCatch({
        widget <- profvis::profvis(prof_input = prof_path)
        htmlwidgets::saveWidget(widget,
                                .profile_output_path("profvis.html"),
                                selfcontained = FALSE)
        message("[profile] profvis.html rendered for human inspection")
      }, error = function(e) {
        message(sprintf("[profile] profvis render failed: %s",
                        conditionMessage(e)))
      })
    } else if (!has_profvis) {
      message("[profile] profvis not installed; install with ",
              "`install.packages(\"profvis\")` for an HTML view. ",
              "CLI still parses Rprof.out for the evidence card.")
    }
  }, add = TRUE)
  message(sprintf("[profile active] backend=Rprof+profvis kind=sampling+mem unit=mixed interval=1ms writing to %s",
                  prof_path))
  force(expr)
}


.with_profile_mem <- function(expr) {
  prof_path <- .profile_output_path("Rprof.out")
  Rprof(filename = prof_path, interval = 0.005, line.profiling = TRUE,
        memory.profiling = TRUE)
  on.exit({
    Rprof(NULL)
    .summarize_rprof(prof_path, kind = "mem")
  }, add = TRUE)
  message(sprintf("[profile active] backend=Rprof kind=sampling+mem unit=mixed interval=5ms writing to %s",
                  prof_path))
  force(expr)
}


.profile_output_path <- function(filename) {
  out_dir <- Sys.getenv("ZYME_PROFILE_DIR", unset = getwd())
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  }
  file.path(out_dir, filename)
}


.summarize_rprof <- function(prof_path, kind = "cpu") {
  if (!file.exists(prof_path)) return(invisible())
  summ <- tryCatch(summaryRprof(prof_path, memory = "both"),
                   error = function(e) NULL)
  if (is.null(summ) || is.null(summ$by.self) || nrow(summ$by.self) == 0) {
    return(invisible())
  }
  top <- utils::head(summ$by.self, 20)
  message(sprintf("[profile] top %d by self.time (kind=%s):", nrow(top), kind))
  has_mem <- !is.null(top$mem.total)
  for (i in seq_len(nrow(top))) {
    mem_str <- if (has_mem && !is.na(top$mem.total[i])) {
      sprintf("  mem=%6.1f MB", top$mem.total[i])
    } else { "" }
    message(sprintf("  %-50s %7.3fs (%5.1f%%)%s",
                    rownames(top)[i],
                    top$self.time[i],
                    top$self.pct[i],
                    mem_str))
  }
}


# with_subprofile — named sub-block timer when ZYME_PROFILE=1 is set.
#
# When the main with_profile sampler points at an opaque native call
# (Rcpp / BLAS / dispatch chain) and you need to bisect time inside it,
# wrap segments with with_subprofile() to get clean named timings:
#
#   with_subprofile("normalize", { ndata <- normalize(counts) })
#   with_subprofile("scale",     { scaled <- scale(ndata) })
#   with_subprofile("pca",       { pcs    <- run_pca(scaled) })
#
# On exit, prints `[subprofile] <name>: <elapsed>s` to stderr. No-op
# when ZYME_PROFILE is unset — safe to leave in committed pipeline code.
with_subprofile <- function(name, expr) {
  if (nzchar(Sys.getenv("ZYME_PROFILE", ""))) {
    t0 <- Sys.time()
    on.exit({
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      message(sprintf("[subprofile] %s: %.3fs", name, elapsed))
    }, add = TRUE)
  }
  force(expr)
}


# Print the standard summary that the zyme runner parses.
emit_summary <- function(speed_sec, peak_mb = NA, cpu_sec = NA, ...) {
  # cpu_sec defaults to user + sys CPU time of this R process (proc.time()
  # entries 1 + 2). Together with speed_sec (wall time) it lets the agent
  # distinguish a real algorithm win from a "throw threads at it" win:
  # parallel-only wins show wall_time / cpu_time approaching the thread
  # count, while algorithmic wins show cpu_time dropping in lockstep with
  # wall_time.
  cat(sprintf("speed_sec: %.6f\n", speed_sec))
  if (is.na(peak_mb)) peak_mb <- peak_memory_mb()
  cat(sprintf("peak_mb: %.1f\n", peak_mb))
  if (is.na(cpu_sec)) {
    pt <- proc.time()
    cpu_sec <- as.numeric(pt["user.self"]) + as.numeric(pt["sys.self"])
  }
  cat(sprintf("cpu_sec: %.6f\n", cpu_sec))
  metrics <- list(...)
  for (k in names(metrics)) {
    v <- metrics[[k]]
    if (is.numeric(v)) {
      cat(sprintf("%s: %.6f\n", k, v))
    } else {
      cat(sprintf("%s: %s\n", k, as.character(v)))
    }
  }
  # In-process thread budget self-check. Advisory only — these signals
  # (options(mc.cores), BLAS thread count) are declarations, not actual
  # usage. The runner's PGID poller is the authoritative gate that
  # measures real CPU consumption and decides REJECT vs WARN per
  # iterate.md §Parallelism's case A / case B split (see
  # runner.py::run_task's thread budget audit). This helper just surfaces
  # the configuration up-front so the agent sees the intent before the
  # round runs to completion.
  .check_thread_budget()
}

.check_thread_budget <- function() {
  raw <- Sys.getenv("ZYME_THREADS", unset = "")
  if (!nzchar(raw)) return(invisible())
  zt <- suppressWarnings(as.integer(raw))
  if (is.na(zt) || zt < 1L) return(invisible())
  violations <- character(0)
  cur_mc <- getOption("mc.cores")
  if (!is.null(cur_mc)) {
    cur_mc_i <- suppressWarnings(as.integer(cur_mc))
    if (!is.na(cur_mc_i) && cur_mc_i > zt) {
      violations <- c(violations,
                      sprintf("options(mc.cores)=%d > ZYME_THREADS=%d",
                              cur_mc_i, zt))
    }
  }
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    blas_n <- tryCatch(RhpcBLASctl::blas_get_num_procs(),
                       error = function(e) NA_integer_)
    if (!is.na(blas_n) && as.integer(blas_n) > zt) {
      violations <- c(violations,
                      sprintf("BLAS threads=%d > ZYME_THREADS=%d",
                              as.integer(blas_n), zt))
    }
  }
  if (length(violations) == 0L) return(invisible())
  message("[zyme] thread budget declaration over budget — ",
          paste(violations, collapse = "; "))
  message("[zyme] This is a *declaration* check (configured cores), not ",
          "actual usage; the runner's PGID poller is authoritative.")
  message("[zyme] If the engaged knob is in task.yaml::upstream_parallelism ",
          "(case A in iterate.md §Parallelism) AND it will actually drive ",
          "fork workers (mclapply/bplapply), re-record baseline:")
  message("[zyme]   zyme baseline reference --tier <t> --reps 3 --force")
  message("[zyme] Otherwise (case B: a parallel layer upstream doesn't ",
          "expose) this is a legitimate contribution — continue and ",
          "document in memory/discoveries.md.")
}

# ---------------------------------------------------------------------------
# Tier-config + threading helpers (mirrors helpers.py)
# ---------------------------------------------------------------------------
# These let task code read per-tier subset sizes from `task.yaml` and respect
# the thread count zyme injects (`ZYME_THREADS`). Both are opt-in.

.find_task_yaml <- function() {
  cur <- normalizePath(getwd(), mustWork = FALSE)
  repeat {
    candidate <- file.path(cur, "task.yaml")
    if (file.exists(candidate)) return(candidate)
    parent <- dirname(cur)
    if (parent == cur) return(NULL)
    cur <- parent
  }
}

.split_top_level_commas <- function(s) {
  parts <- character(0)
  depth <- 0L
  buf <- character(0)
  chars <- strsplit(s, "", fixed = TRUE)[[1]]
  for (ch in chars) {
    if (ch == "{") {
      depth <- depth + 1L
      buf <- c(buf, ch)
    } else if (ch == "}") {
      depth <- depth - 1L
      buf <- c(buf, ch)
    } else if (ch == "," && depth == 0L) {
      parts <- c(parts, paste(buf, collapse = ""))
      buf <- character(0)
    } else {
      buf <- c(buf, ch)
    }
  }
  if (length(buf) > 0L) parts <- c(parts, paste(buf, collapse = ""))
  parts
}

.coerce_scalar <- function(v) {
  v <- trimws(v)
  if (nchar(v) >= 2L && substr(v, 1, 1) == substr(v, nchar(v), nchar(v)) &&
      substr(v, 1, 1) %in% c('"', "'")) {
    return(substr(v, 2, nchar(v) - 1))
  }
  num <- suppressWarnings(as.numeric(v))
  if (!is.na(num)) {
    int_val <- suppressWarnings(as.integer(v))
    if (!is.na(int_val) && as.character(int_val) == v) return(int_val)
    return(num)
  }
  v
}

.parse_inline_value <- function(v) {
  v <- trimws(v)
  if (nchar(v) >= 2L && substr(v, 1, 1) == "{" &&
      substr(v, nchar(v), nchar(v)) == "}") {
    inner <- substr(v, 2, nchar(v) - 1)
    out <- list()
    for (kv in .split_top_level_commas(inner)) {
      kv <- trimws(kv)
      if (!nzchar(kv)) next
      # Accept dotted keys (burn.in, na.rm, BPPARAM) — R packages routinely
      # expose them. Bare \\w+ silently dropped these.
      m <- regmatches(kv, regexec("^\\s*([A-Za-z0-9_.]+)\\s*:\\s*(.+)$", kv))[[1]]
      if (length(m) < 3L) {
        message(sprintf(
          "[parse_params] WARN: dropping malformed entry %s (no `key: value` shape)",
          dQuote(kv, q = FALSE)))
        next
      }
      out[[m[2]]] <- .parse_inline_value(m[3])
    }
    return(out)
  }
  .coerce_scalar(v)
}

.parse_datasets_minimal <- function(task_yaml_path) {
  if (!file.exists(task_yaml_path)) return(list())
  text <- readLines(task_yaml_path, warn = FALSE)
  in_datasets <- FALSE
  out <- list()
  for (raw in text) {
    line <- sub("\\s+$", "", raw)
    if (!nzchar(line) || grepl("^\\s*#", line)) next
    if (grepl("^datasets:\\s*$", line)) {
      in_datasets <- TRUE
      next
    }
    if (in_datasets) {
      if (grepl("^\\S", line) && !startsWith(line, "- ")) {
        in_datasets <- FALSE
        next
      }
      stripped <- trimws(line)
      if (!startsWith(stripped, "- ")) next
      inner <- trimws(substring(stripped, 3))
      if (nchar(inner) >= 2L && substr(inner, 1, 1) == "{" &&
          substr(inner, nchar(inner), nchar(inner)) == "}") {
        inner <- substr(inner, 2, nchar(inner) - 1)
      }
      entry <- list()
      for (kv in .split_top_level_commas(inner)) {
        kv <- trimws(kv)
        if (!nzchar(kv)) next
        m <- regmatches(kv, regexec("^\\s*([A-Za-z0-9_.]+)\\s*:\\s*(.+)$", kv))[[1]]
        if (length(m) < 3L) {
          message(sprintf(
            "[parse_datasets] WARN: dropping malformed entry %s (no `key: value` shape)",
            dQuote(kv, q = FALSE)))
          next
        }
        key <- m[2]
        val <- trimws(m[3])
        if (nchar(val) >= 2L && substr(val, 1, 1) == "{" &&
            substr(val, nchar(val), nchar(val)) == "}") {
          entry[[key]] <- .parse_inline_value(val)
        } else {
          if (nchar(val) >= 2L && substr(val, 1, 1) == substr(val, nchar(val), nchar(val)) &&
              substr(val, 1, 1) %in% c('"', "'")) {
            val <- substr(val, 2, nchar(val) - 1)
          }
          entry[[key]] <- val
        }
      }
      if (!is.null(entry$name) && !is.null(entry$path)) {
        if (is.null(entry$tier)) entry$tier <- "tiny"
        if (is.null(entry$params)) entry$params <- list()
        out[[length(out) + 1L]] <- entry
      }
    }
  }
  out
}

# Read tier-specific parameters from task.yaml's datasets[].params field.
#
# Args:
#   tier: tier name. Defaults to Sys.getenv("ZYME_TIER").
#
# Returns: named list of params (e.g. list(n_cells=50000L, n_genes=2200L)).
# Empty list when the tier exists but declares no params.
#
# Errors with a clear message when the tier or task.yaml is missing.
get_tier_params <- function(tier = NULL) {
  if (is.null(tier) || identical(tier, "")) {
    tier <- Sys.getenv("ZYME_TIER", unset = "")
    if (!nzchar(tier)) {
      stop("get_tier_params(): tier= not given and ZYME_TIER env var not set. ",
           "Either pass tier explicitly or run via zyme.")
    }
  }
  yaml_path <- .find_task_yaml()
  if (is.null(yaml_path)) {
    stop("get_tier_params(): no task.yaml found walking up from cwd.")
  }
  entries <- .parse_datasets_minimal(yaml_path)
  by_tier <- setNames(entries, vapply(entries, function(e) e$tier, character(1)))
  if (!tier %in% names(by_tier)) {
    stop(sprintf("get_tier_params(): tier '%s' not found in %s. Available tiers: %s",
                 tier, yaml_path, paste(sort(names(by_tier)), collapse = ", ")))
  }
  params <- by_tier[[tier]]$params
  if (is.null(params)) list() else params
}

# Read thread count from ZYME_THREADS env var, with fallback.
#
# Args:
#   default: integer to return when ZYME_THREADS is not set. Falls back
#            further to parallel::detectCores() if default is NULL.
#
# Returns: integer thread count.
#
# Use this anywhere the task currently hardcodes mc.cores / RcppParallel
# threads / etc. `zyme verify` sets ZYME_THREADS per matrix cell; routing
# through this helper makes the matrix actually exercise different thread
# counts.
get_threads <- function(default = NULL) {
  raw <- Sys.getenv("ZYME_THREADS", unset = "")
  if (nzchar(raw)) {
    n <- suppressWarnings(as.integer(raw))
    if (!is.na(n) && n > 0L) return(n)
  }
  if (!is.null(default)) return(as.integer(default))
  if (requireNamespace("parallel", quietly = TRUE)) {
    n <- parallel::detectCores(logical = TRUE)
    if (!is.na(n) && n > 0L) return(as.integer(n))
  }
  1L
}


# ============================================================
# auto_structure_check_all_slots — F3 defense (Output Truncation / Stub)
# ============================================================
# Default-on, opt-out structural check for every top-level slot in the
# reference output. Three boolean metrics per slot (1.0 = pass, 0.0 = fail):
#
#   <slot>_present         test has the slot (catches dropped output)
#   <slot>_shape_match     test slot has same shape/length as ref
#   <slot>_non_degenerate  if ref has variance, test does too
#                          (catches the "stub to constant" hack —
#                           override returns 0 / NA / single value while
#                           ref legitimately varies)
#
# This is NOT a value comparison. Pearson / max_abs_diff / Jaccard / ARI
# remain init agent's responsibility, declared explicitly in task.yaml::metrics
# for the slots that need them.
#
# What this DOES catch:
#   ✓ Output truncation (dipy_dti model_params 12 → 3): shape_match=0
#   ✓ Stub-to-constant (decontXLogLik=0, lifelines stale=0): non_degenerate=0
#   ✓ Dropped slot (override returns NULL / missing field): present=0
#
# What this does NOT catch (stays init agent's responsibility):
#   ✗ Stale-but-plausible values (e.g. variance matrix with subtly wrong
#     numbers) — needs explicit pearson_variance / max_abs_diff_variance
#   ✗ Algorithmic drift within reasonable tolerance — needs explicit Pearson
#
# Waive specific slots via `waived = c("slot_a", ...)` only when the slot
# legitimately differs across runs (random init log, iteration history of
# different length). Pair every waiver with a `# diagnostic: <reason>`
# comment in evaluate.R.
#
# Returns a named list of numeric (1.0/0.0/NA) metrics. NA when a check is
# undefined (e.g. unrecognized type); treat NA as a soft fail in audit.
auto_structure_check_all_slots <- function(ref, test, waived = character(0)) {
  if (is.null(ref) || is.null(test)) return(list())
  out <- list()
  ref_slots <- if (is.list(ref)) names(ref) else character(0)
  for (slot in ref_slots) {
    if (slot %in% waived) next
    if (is.null(slot) || !nzchar(slot)) next
    r <- ref[[slot]]
    if (is.null(r)) next

    # `t` may be NULL (test dropped the slot) or have wrong type.
    t <- if (is.list(test)) test[[slot]] else NULL

    # 1. presence
    if (is.null(t)) {
      out[[paste0(slot, "_present")]] <- 0
      next
    }
    out[[paste0(slot, "_present")]] <- 1

    # Recurse into nested lists and data.frames. For data.frames, names(r)
    # gives column names so each column is checked individually — catches
    # in-df hacks (column deletion, column-to-constant, type swap) that the
    # whole-df shape check would miss.
    if (is.list(r) && is.list(t)) {
      nested <- auto_structure_check_all_slots(r, t, waived = waived)
      for (nm in names(nested)) {
        out[[paste0(slot, ".", nm)]] <- nested[[nm]]
      }
      next
    }

    # 2. shape match
    r_dim <- if (is.null(dim(r))) length(r) else dim(r)
    t_dim <- if (is.null(dim(t))) length(t) else dim(t)
    if (!identical(as.integer(r_dim), as.integer(t_dim))) {
      out[[paste0(slot, "_shape_match")]] <- 0
      next
    }
    out[[paste0(slot, "_shape_match")]] <- 1

    # 3. non-degenerate (ref-has-variance → test-has-variance)
    out[[paste0(slot, "_non_degenerate")]] <- .non_degenerate_check(r, t)
  }
  out
}


# Helper: returns 1 if test has appropriate variance given ref, 0 if test
# is degenerate while ref isn't. NA for unrecognized types.
.non_degenerate_check <- function(r, t) {
  # Numeric
  if (is.numeric(r) || is.complex(r)) {
    r_vec <- as.numeric(r); t_vec <- tryCatch(as.numeric(t),
                                              error = function(e) NULL)
    if (is.null(t_vec)) return(NA_real_)
    r_finite <- r_vec[is.finite(r_vec)]
    t_finite <- t_vec[is.finite(t_vec)]
    # If ref is empty or all-NaN, can't make a claim — pass.
    if (length(r_finite) == 0L) return(1)
    r_unique <- length(unique(r_finite))
    # Ref degenerate (constant or all-NaN-equivalent) → test must match exactly
    if (r_unique <= 1L) {
      return(if (isTRUE(all.equal(r_vec, t_vec))) 1 else 0)
    }
    # Ref non-degenerate → test must also be non-degenerate
    if (length(t_finite) == 0L) return(0)
    t_unique <- length(unique(t_finite))
    return(if (t_unique <= 1L) 0 else 1)
  }
  # Categorical / logical
  if (is.factor(r) || is.character(r) || is.logical(r)) {
    r_lab <- as.character(r); t_lab <- tryCatch(as.character(t),
                                                error = function(e) NULL)
    if (is.null(t_lab)) return(NA_real_)
    r_unique <- length(unique(r_lab[!is.na(r_lab)]))
    if (r_unique <= 1L) {
      return(if (identical(r_lab, t_lab)) 1 else 0)
    }
    t_unique <- length(unique(t_lab[!is.na(t_lab)]))
    return(if (t_unique <= 1L) 0 else 1)
  }
  NA_real_
}


# Emit auto_structure metrics compactly: one summary line when all perfect,
# individual `name: value` lines only for failures (runner ignores `#` lines).
emit_auto_structure_summary <- function(struct_metrics) {
  if (length(struct_metrics) == 0L) return(invisible(NULL))
  fail_names <- names(struct_metrics)[vapply(struct_metrics, function(v) {
    is.numeric(v) && !is.na(v) && v < 1.0
  }, logical(1))]
  n_total <- length(struct_metrics)
  n_fail  <- length(fail_names)
  if (n_fail == 0L) {
    cat(sprintf("# auto_structure: %d/%d perfect\n", n_total, n_total))
  } else {
    cat(sprintf("# auto_structure: %d/%d pass, %d FAIL:\n",
                n_total - n_fail, n_total, n_fail))
    for (nm in fail_names) {
      v <- struct_metrics[[nm]]
      cat(sprintf("%s: %.6f\n", nm, v))
    }
  }
}
