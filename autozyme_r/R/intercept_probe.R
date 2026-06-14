# Intercept-counting probe for `zyme package check-intercept`.
#
# Sourced into the R worker process when env var ZYME_INSTRUMENT_INTERCEPTS=1
# is set. Wraps every registered target's fast function with a counter so
# `zyme package check-intercept` can confirm the patch actually fired.
#
# Counters write to the path in ZYME_INTERCEPT_OUT as a small JSON-shaped
# string on R's exit (no jsonlite dependency — emit manually).
#
# Must run BEFORE `autozyme::activate(<patch>)`: once activate has resolved
# its dispatchers, swapping the fast fn afterwards is too late.

.zyme_intercept <- new.env(parent = emptyenv())
.zyme_intercept$counts <- list()
.zyme_intercept$installed <- FALSE

.zyme_intercept_key <- function(upstream, attr) paste0(upstream, "::", attr)

.zyme_intercept_inc <- function(key) {
  cur <- .zyme_intercept$counts[[key]]
  if (is.null(cur)) cur <- 0L
  .zyme_intercept$counts[[key]] <- cur + 1L
}

.zyme_intercept_wrap <- function(fn, key) {
  force(fn); force(key)
  wrapped <- function(...) {
    .zyme_intercept_inc(key)
    fn(...)
  }
  attr(wrapped, ".zyme_intercept_key") <- key
  wrapped
}

# Walk a patch's registry entry and replace each target's fast fn with a
# counted wrapper. The registry stores fast fns in `targets`; we mutate in
# place so the subsequent `activate()` resolves to the wrapped variant.
.zyme_intercept_install_for <- function(patch_name) {
  registry <- get0(".zyme_registry", envir = asNamespace("autozyme"),
                   inherits = FALSE)
  if (is.null(registry) || is.null(registry[[patch_name]])) return(invisible())
  entry <- registry[[patch_name]]
  upstream <- entry$upstream
  targets <- entry$targets
  if (is.null(targets)) return(invisible())
  # Targets is a named list: each element either a function (namespace target)
  # or a list with $kind / $signature / $fn for s4. Both shapes get wrapped.
  for (nm in names(targets)) {
    t <- targets[[nm]]
    key <- .zyme_intercept_key(upstream, nm)
    if (is.function(t)) {
      targets[[nm]] <- .zyme_intercept_wrap(t, key)
    } else if (is.list(t) && is.function(t$fn)) {
      t$fn <- .zyme_intercept_wrap(t$fn, key)
      targets[[nm]] <- t
    }
  }
  entry$targets <- targets
  registry[[patch_name]] <- entry
  assign(".zyme_registry", registry, envir = asNamespace("autozyme"))
  invisible()
}

.zyme_intercept_write <- function() {
  out_path <- Sys.getenv("ZYME_INTERCEPT_OUT", "")
  if (!nzchar(out_path)) return(invisible())
  counts <- .zyme_intercept$counts
  if (length(counts) == 0L) {
    cat("{}", file = out_path)
    return(invisible())
  }
  # Hand-emit JSON to avoid jsonlite dependency in the probe — keep object
  # ordered for deterministic diffs.
  keys <- sort(names(counts))
  pieces <- vapply(keys, function(k) {
    sprintf('"%s": %d', gsub('"', '\\"', k, fixed = TRUE),
            as.integer(counts[[k]]))
  }, character(1))
  cat("{\n  ", paste(pieces, collapse = ",\n  "), "\n}\n",
      sep = "", file = out_path)
  invisible()
}

# Install once. Idempotent so multiple sources from a CLI shim do not re-wrap.
.zyme_intercept_install <- function(patch_name = NULL) {
  if (isTRUE(.zyme_intercept$installed)) {
    if (!is.null(patch_name)) .zyme_intercept_install_for(patch_name)
    return(invisible())
  }
  .zyme_intercept$installed <- TRUE
  reg.finalizer(.zyme_intercept,
                function(e) try(.zyme_intercept_write(), silent = TRUE),
                onexit = TRUE)
  if (!is.null(patch_name)) .zyme_intercept_install_for(patch_name)
  invisible()
}

# Convenience entry point matching the Python helper's install_from_env().
.zyme_intercept_install_from_env <- function() {
  if (Sys.getenv("ZYME_INSTRUMENT_INTERCEPTS", "") != "1") return(FALSE)
  patch_name <- Sys.getenv("ZYME_INTERCEPT_PATCH", "")
  .zyme_intercept_install(if (nzchar(patch_name)) patch_name else NULL)
  TRUE
}
