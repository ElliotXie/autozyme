.zyme_registry <- new.env(parent = emptyenv())

# Process-local kill switch. When TRUE, dispatcher wrappers installed by
# .activate_one short-circuit to the captured upstream original. Toggled
# only via with_disabled() so on.exit restores reliably even on error.
.zyme_state <- new.env(parent = emptyenv())
.zyme_state$disabled <- FALSE

#' Test whether autozyme patches are currently context-disabled
#'
#' Returns TRUE inside a \code{with_disabled({...})} block, FALSE otherwise.
#' Patched fast functions and the namespace-fn dispatcher consult this to
#' decide whether to fall through to the captured upstream original.
#' @export
is_disabled <- function() {
  isTRUE(.zyme_state$disabled) ||
    isTRUE(.az_truthy(Sys.getenv("AUTOZYME_DISABLE", unset = ""))) ||
    isTRUE(.az_truthy(Sys.getenv("AUTOZYME_DISABLED", unset = "")))
}

#' Run code with all activated patches temporarily disabled
#'
#' Hard kill switch: every patched fn (namespace-fn dispatchers and S4 patches
#' that consult \code{is_disabled()}) forwards to the captured upstream
#' original within \code{expr}. The original \code{disabled} state is restored
#' on exit, including on error. This is the only opt-out path that works for
#' patches whose fast function can't take a per-call \code{zyme = FALSE} kwarg
#' (e.g. S4 methods where the generic's signature is fixed by upstream).
#'
#' Note: not thread-local. Across mclapply() forks the child inherits the
#' parent's flag (the env is copied), which is usually what you want; across
#' fresh-process workers (BiocParallel SnowParam, etc.) the flag does NOT
#' propagate — use \code{deactivate()} for those.
#' @param expr Expression to evaluate.
#' @return The value of \code{expr}.
#' @export
with_disabled <- function(expr) {
  old <- .zyme_state$disabled
  .zyme_state$disabled <- TRUE
  on.exit(.zyme_state$disabled <- old, add = TRUE)
  force(expr)
}

# Wrap a namespace-fn fast replacement so that calls short-circuit to the
# captured original when is_disabled() is TRUE. The wrapper strips our
# `zyme=` / `turbo=` kwargs before forwarding so original (which doesn't
# know about them) doesn't see "unused argument (zyme = ...)". The wrapper
# closes over both `fast_fn` and `original` directly — does not look them
# up by name — so rebinding upstream's namespace afterwards doesn't cause
# recursion.
#
# NOTE: the disabled branch *cannot* use `do.call(original, list(...))`.
# do.call inlines list elements as values into a literal call object;
# downstream code in `original` that touches `match.call()` / `sys.call()`
# (Seurat v5's NormalizeData.Seurat etc. do this) then walks a call whose
# args carry the full materialized object, which is pathologically slow on
# large inputs — 58x slowdown observed on a 208k-cell Seurat object
# (12s → >700s) at 2026-05-14. The helper below uses R's normal calling
# convention (promise references) and named-arg matching to consume
# zyme/turbo without ever materializing a list of all positional args.
.zyme_strip_and_forward <- function(.autozyme_fn, ..., zyme = NULL,
                                    turbo = NULL) {
  .autozyme_fn(...)
}

.wrap_namespace_fast <- function(fast_fn, original) {
  force(fast_fn); force(original)
  function(...) {
    if (is_disabled()) {
      return(.zyme_strip_and_forward(original, ...))
    }
    fast_fn(...)
  }
}

# Patches live in inst/patches/<name>.R and are sourced lazily on activate so
# library(autozyme) doesn't trigger requireNamespace() probes for every
# upstream. See .ensure_registered() below.
.zyme_available_patches <- function() {
  patches_dir <- system.file("patches", package = "autozyme")
  if (!nzchar(patches_dir) || !dir.exists(patches_dir)) return(character(0))
  # Two layouts coexist (migration in progress):
  #   legacy: inst/patches/<name>.R
  #   folder: inst/patches/<name>/patch.R  + speedups_finalized.tsv
  legacy <- sub("\\.R$", "", list.files(patches_dir, pattern = "\\.R$"))
  subdirs <- list.dirs(patches_dir, full.names = FALSE, recursive = FALSE)
  folder <- subdirs[vapply(subdirs, function(d) {
    file.exists(file.path(patches_dir, d, "patch.R"))
  }, logical(1))]
  sort(unique(c(legacy, folder)))
}

.ensure_registered <- function(name) {
  if (!is.null(.zyme_registry[[name]])) return(TRUE)
  if (!(name %in% .zyme_available_patches())) {
    stop(sprintf(
      "no patch named '%s'; see list_patches() / list_subsets()", name))
  }
  # Prefer folder layout (inst/patches/<name>/patch.R) when present;
  # fall back to legacy single-file (inst/patches/<name>.R).
  patch_file <- system.file("patches", name, "patch.R", package = "autozyme")
  if (!nzchar(patch_file)) {
    patch_file <- system.file("patches", paste0(name, ".R"),
                              package = "autozyme")
  }
  if (!nzchar(patch_file)) {
    stop(sprintf("patch file for '%s' missing from inst/patches/", name))
  }
  # Each patch gets its own environment with autozyme's namespace as parent
  # so register_patch / Rcpp exports / utils.R helpers resolve, while the
  # patch's local definitions (.foo_orig_*, fast_*) stay isolated.
  patch_env <- new.env(parent = asNamespace("autozyme"))
  sys.source(patch_file, envir = patch_env)
  # If the file's outer requireNamespace() gate failed, register_patch was
  # never called and the registry slot stays NULL — caller decides what to do.
  !is.null(.zyme_registry[[name]])
}

#' Register a patch
#'
#' Called by patch files at load time to declare a fast replacement.
#'
#' @param name Canonical patch name (typically the upstream package name).
#' @param upstream Upstream package name (string).
#' @param targets Named list whose names are upstream functions to patch and
#'   whose values are the fast replacement functions.
#' @param smoke Optional list with three closures used by `verify_patch()`:
#'   \describe{
#'     \item{\code{load(task_dir, tier)}}{returns the input bundle.}
#'     \item{\code{call(inputs)}}{invokes the upstream API once.}
#'     \item{\code{save(result, dir)}}{writes the result into \code{dir}
#'       in the format the task's \code{evaluate.R} expects.}
#'   }
#'   These three pieces lift directly from the task's \code{pipeline/run.R}
#'   (data load / function call / save) with framework wrapping stripped.
#' @param tested_against Optional free-form string like \code{"nichenetr 2.1.7"}
#'   naming the upstream version the patch was lifted against. The activation
#'   marker compares it to the installed version and warns on drift.
#' @param tested_upstream_versions Optional named list mapping upstream
#'   package names to character vectors of versions the patch is known to
#'   parity-pass on. Consumed by Tier C CI to build a per-version drift
#'   matrix. Example: \code{list(Seurat = c("5.0.3", "5.1.0"))}.
#' @param on_activate Optional zero-arg function invoked by \code{activate(name)}
#'   AFTER the targets have been rebound. Use it to apply side patches the
#'   framework can't model as a single \code{upstream}/\code{targets} pair —
#'   e.g. rebinding helper internals in a sibling package (tradeseq patching
#'   mgcv's nb/dDeta/gam.fit4). Must be idempotent: activate may be called
#'   multiple times. Errors are caught and warned so a broken hook can't
#'   trap the user mid-activation.
#' @param on_deactivate Optional zero-arg function invoked by
#'   \code{deactivate(name)} after upstream bindings are restored. Use it to
#'   tear down any side state the patch built up at runtime — e.g. stop
#'   PSOCK clusters, free caches, undo scoped namespace patches in other
#'   packages. The callback runs once per deactivate; the registry forgets it
#'   on session exit. Errors inside the callback are caught and warned so a
#'   broken hook can't trap the user in an active state.
#' @export
register_patch <- function(name, upstream, targets, smoke = NULL,
                            tested_against = NULL,
                            tested_upstream_versions = NULL,
                            on_activate = NULL,
                            on_deactivate = NULL) {
  stopifnot(is.character(name), length(name) == 1L)
  stopifnot(is.character(upstream), length(upstream) == 1L)
  stopifnot(is.list(targets), !is.null(names(targets)),
            all(nzchar(names(targets))))
  if (!is.null(smoke)) {
    stopifnot(is.list(smoke),
              all(c("load", "call", "save") %in% names(smoke)),
              all(vapply(smoke[c("load", "call", "save")], is.function, logical(1))))
  }
  if (!is.null(tested_against)) {
    stopifnot(is.character(tested_against), length(tested_against) == 1L)
  }
  if (!is.null(tested_upstream_versions)) {
    if (!is.list(tested_upstream_versions) ||
        is.null(names(tested_upstream_versions)) ||
        !all(nzchar(names(tested_upstream_versions)))) {
      stop("tested_upstream_versions must be a named list of character vectors",
           call. = FALSE)
    }
    for (pkg in names(tested_upstream_versions)) {
      v <- tested_upstream_versions[[pkg]]
      if (!is.character(v) || length(v) == 0L) {
        stop(sprintf("tested_upstream_versions[['%s']] must be a non-empty character vector",
                     pkg), call. = FALSE)
      }
    }
  }
  if (!is.null(on_activate)) {
    stopifnot(is.function(on_activate))
  }
  if (!is.null(on_deactivate)) {
    stopifnot(is.function(on_deactivate))
  }

  # Reject overlap on (upstream, attr): two patches cannot coexist on the same
  # upstream symbol — at most one binding can be live at a time.
  new_claims <- paste(upstream, names(targets), sep = "::")
  for (other_name in ls(.zyme_registry)) {
    if (identical(other_name, name)) next
    other <- .zyme_registry[[other_name]]
    other_claims <- paste(other$upstream, names(other$targets), sep = "::")
    conflicts <- intersect(new_claims, other_claims)
    if (length(conflicts) > 0L) {
      stop(sprintf(
        "patch '%s' targets { %s }, already claimed by patch '%s'",
        name, paste(conflicts, collapse = ", "), other_name
      ))
    }
  }

  .zyme_registry[[name]] <- list(
    name = name,
    upstream = upstream,
    targets = targets,
    smoke = smoke,
    tested_against = tested_against,
    tested_upstream_versions = tested_upstream_versions,
    on_activate = on_activate,
    on_deactivate = on_deactivate,
    originals = list(),
    injected = FALSE
  )
  invisible(NULL)
}

.rebind <- function(ns, fn_name, value) {
  was_locked <- bindingIsLocked(fn_name, ns)
  if (was_locked) unlockBinding(fn_name, ns)
  assign(fn_name, value, envir = ns)
  if (was_locked) lockBinding(fn_name, ns)
  # S3 dispatch caches resolved methods in .__S3MethodsTable__.; if the patched
  # name is a method (e.g. NormalizeData.Seurat), update the table too so live
  # UseMethod() calls pick up the replacement. No-op for non-S3 names.
  s3_table <- ns[[".__S3MethodsTable__."]]
  if (!is.null(s3_table) && exists(fn_name, envir = s3_table, inherits = FALSE)) {
    assign(fn_name, value, envir = s3_table)
  }
  # If the upstream package is attached, its exported functions live in BOTH
  # the package namespace and the package:XXX env on the search path. Bare
  # name resolution (e.g. `FindAllMarkers(obj)` after `library(Seurat)`)
  # finds the search-path binding FIRST, so missing this update silently
  # bypasses the patch for non-namespace-prefixed calls — the most common
  # call style. Mirror the namespace rebind into package:XXX.
  pkg_name <- environmentName(ns)
  pkg_env_name <- paste0("package:", pkg_name)
  if (pkg_env_name %in% search()) {
    # Warn (don't silently swallow) when the package: env mirror fails. The
    # namespace rebind above already succeeded, so `pkg::fn(...)` calls hit
    # the patch — but `library(pkg); fn(...)` would resolve to the unmirrored
    # search-path binding and silently bypass the patch. Surfacing the error
    # lets the user act on the half-applied state instead of debugging a
    # "patch isn't applying" ghost.
    tryCatch({
      pkg_env <- as.environment(pkg_env_name)
      if (exists(fn_name, envir = pkg_env, inherits = FALSE)) {
        if (bindingIsLocked(fn_name, pkg_env)) unlockBinding(fn_name, pkg_env)
        assign(fn_name, value, envir = pkg_env)
        lockBinding(fn_name, pkg_env)
      }
    }, error = function(e) warning(sprintf(
      "[autozyme] could not mirror '%s' into 'package:%s' env: %s. The namespace binding was patched, but bare `%s(...)` calls after library(%s) will bypass the patch. Either reload autozyme after attaching %s, or use the fully-qualified `%s::%s(...)` form.",
      fn_name, pkg_name, conditionMessage(e),
      fn_name, pkg_name, pkg_name, pkg_name, fn_name
    ), call. = FALSE))
  }
}

# Each target value is either:
#   (a) a function — old-style namespace patch via assign + lockBinding;
#   (b) a list with $kind = "s4" + $signature + $fn — S4 method dispatch via
#       methods::setMethod, captured/restored via methods::getMethod.
.target_is_s4 <- function(t) {
  is.list(t) && identical(t[["kind"]], "s4")
}

.emit_activation_marker <- function(p) {
  if (nzchar(Sys.getenv("AUTOZYME_QUIET"))) return(invisible(NULL))
  installed <- tryCatch(
    as.character(utils::packageVersion(p$upstream)),
    error = function(e) NA_character_
  )
  ver_str <- if (is.na(installed)) {
    sprintf("%s (version unknown)", p$upstream)
  } else {
    sprintf("%s %s", p$upstream, installed)
  }
  drift <- ""
  if (!is.null(p$tested_against)) {
    # tested_against is "<pkg> <ver>". Known-good = that literal PLUS any
    # versions recorded in tested_upstream_versions for the named pkg; only
    # warn when the installed version is none of them (so recording multiple
    # validated versions actually suppresses the warning, not just the single
    # primary literal).
    parts <- strsplit(p$tested_against, " ", fixed = TRUE)[[1]]
    if (length(parts) >= 2L) {
      tested_pkg <- parts[1]
      tested_ver <- paste(parts[-1], collapse = " ")
      known <- tested_ver
      if (!is.null(p$tested_upstream_versions) &&
          !is.null(p$tested_upstream_versions[[tested_pkg]])) {
        known <- c(known, p$tested_upstream_versions[[tested_pkg]])
      }
      if (identical(tested_pkg, p$upstream) &&
          !is.na(installed) && !(installed %in% known)) {
        drift <- sprintf(
          " - WARN: lifted against %s %s, installed %s (may be unstable)",
          tested_pkg, tested_ver, installed
        )
      }
    }
  }
  message(sprintf(
    "[autozyme] activated %s -> %d target(s) in %s%s",
    p$name, length(p$targets), ver_str, drift
  ))
}

.activate_one <- function(name) {
  if (!.ensure_registered(name)) return(FALSE)
  p <- .zyme_registry[[name]]
  if (isTRUE(p$injected)) return(TRUE)
  if (!requireNamespace(p$upstream, quietly = TRUE)) return(FALSE)
  ns <- asNamespace(p$upstream)
  for (fn_name in names(p$targets)) {
    target <- p$targets[[fn_name]]
    if (.target_is_s4(target)) {
      # S4 method patch: pass the generic *function object* (not its name)
      # to setMethod, fetched via getFromNamespace. By name, setMethod looks
      # up the generic via the calling env's parent chain — autozyme's
      # namespace doesn't import upstream's generics, so the lookup fails.
      # Install to .GlobalEnv: both upstream and autozyme namespaces are
      # locked. Global takes priority in S4 dispatch lookup.
      sig <- target$signature
      fn  <- target$fn
      generic <- utils::getFromNamespace(fn_name, p$upstream)
      orig <- methods::getMethod(fn_name, sig, optional = TRUE)
      p$originals[[fn_name]] <- list(kind = "s4", signature = sig, fn = orig,
                                      generic = generic)
      methods::setMethod(generic, sig, fn, where = globalenv())
    } else {
      # Namespace function patch. Wrap the fast fn in a dispatcher that
      # honors with_disabled(); see .wrap_namespace_fast for details.
      orig <- get(fn_name, envir = ns, inherits = FALSE)
      p$originals[[fn_name]] <- orig
      .rebind(ns, fn_name, .wrap_namespace_fast(target, orig))
    }
  }
  p$injected <- TRUE
  .zyme_registry[[name]] <- p
  if (is.function(p$on_activate)) {
    tryCatch(p$on_activate(), error = function(e) {
      warning(sprintf("[autozyme] on_activate hook for '%s' failed: %s",
                      name, conditionMessage(e)), call. = FALSE)
    })
  }
  .emit_activation_marker(p)
  TRUE
}

.deactivate_one <- function(name) {
  p <- .zyme_registry[[name]]
  if (is.null(p) || !isTRUE(p$injected)) return(invisible(NULL))
  ns <- asNamespace(p$upstream)
  for (fn_name in names(p$originals)) {
    orig <- p$originals[[fn_name]]
    if (is.list(orig) && identical(orig[["kind"]], "s4")) {
      sig <- orig$signature
      generic <- orig$generic
      if (is.null(orig$fn)) {
        methods::removeMethod(generic, sig, where = globalenv())
      } else {
        methods::setMethod(generic, sig, orig$fn, where = globalenv())
      }
    } else {
      .rebind(ns, fn_name, orig)
    }
  }
  p$originals <- list()
  p$injected <- FALSE
  .zyme_registry[[name]] <- p
  if (is.function(p$on_deactivate)) {
    tryCatch(p$on_deactivate(), error = function(e) {
      warning(sprintf("[autozyme] on_deactivate hook for '%s' failed: %s",
                      name, conditionMessage(e)), call. = FALSE)
    })
  }
  invisible(NULL)
}

#' Inject all available patches whose upstream is installed
#'
#' Sources each patch in \code{inst/patches/} and activates it if the upstream
#' \code{requireNamespace()} gate passes; otherwise it stays skipped. Also
#' covers any test-time patches registered directly via \code{register_patch()}.
#' @return list with `activated` and `skipped` character vectors.
#' @export
inject_all <- function() {
  activated <- character(0)
  skipped <- character(0)
  for (name in list_patches()) {
    if (.activate_one(name)) {
      activated <- c(activated, name)
    } else {
      skipped <- c(skipped, name)
    }
  }
  list(activated = activated, skipped = skipped)
}

.check_conflicts <- function(newly_activating) {
  live <- character(0)
  for (nm in ls(.zyme_registry)) {
    p <- .zyme_registry[[nm]]
    if (!is.null(p) && isTRUE(p$injected)) live <- c(live, nm)
  }
  live <- unique(c(live, newly_activating))
  for (entry in .zyme_conflicts) {
    if (all(entry$pair %in% live)) {
      warning(sprintf(
        "autozyme: activating %s together is known to interact badly. %s",
        paste(sQuote(sort(entry$pair)), collapse = " + "), entry$reason
      ), call. = FALSE)
    }
  }
}

#' Activate one patch, a subset, or a vector of patches/subsets
#'
#' @param name Either a registered patch name, a subset name (see
#'   \code{list_subsets()}), or a character vector mixing them.
#' @return For a single patch: invisible TRUE/FALSE. For a vector or subset:
#'   invisible named logical vector mapping patch name to activation result.
#' @export
activate <- function(name) {
  resolved <- .resolve_activation_target(name)
  .check_conflicts(resolved)
  if (length(resolved) == 1L && is.character(name) && length(name) == 1L &&
      !name %in% names(.zyme_subsets)) {
    return(invisible(.activate_one(resolved[1])))
  }
  out <- vapply(resolved, function(n) {
    tryCatch(.activate_one(n), error = function(e) FALSE)
  }, logical(1))
  names(out) <- resolved
  invisible(out)
}

.did_you_mean <- function(name, candidates) {
  # Use base::agrep approximate-match. max.distance=0.3 catches single typos
  # in short strings (e.g. "scoda" -> "sccoda") without being too lax.
  if (!length(candidates)) return("")
  hits <- agrep(name, candidates, max.distance = 0.3, value = TRUE)
  hits <- setdiff(hits, name)
  if (!length(hits)) return("")
  if (length(hits) == 1L) {
    sprintf(" Did you mean '%s'?", hits[1])
  } else {
    sprintf(" Did you mean one of %s?", paste(shQuote(head(hits, 2)), collapse = ", "))
  }
}

.resolve_activation_target <- function(name) {
  if (is.character(name) && length(name) == 1L) {
    if (name %in% names(.zyme_subsets)) {
      return(.zyme_subsets[[name]])
    }
    available <- union(.zyme_available_patches(), ls(.zyme_registry))
    if (name %in% available) {
      return(name)
    }
    candidates <- unique(c(available, names(.zyme_subsets)))
    stop(sprintf(
      "'%s' is neither a registered patch nor a subset.%s See list_patches() / list_subsets().",
      name, .did_you_mean(name, candidates)
    ), call. = FALSE)
  }
  if (is.character(name)) {
    out <- character(0)
    for (n in name) out <- c(out, .resolve_activation_target(n))
    return(unique(out))
  }
  stop("activate() expects character; got ", class(name)[1])
}

# Cache for probe results. Memoized per session; clears when R restarts.
.zyme_probe_cache <- new.env(parent = emptyenv())

.probe_patch_installed <- function(name) {
  if (!is.null(.zyme_probe_cache[[name]])) return(.zyme_probe_cache[[name]])
  upstreams <- .zyme_upstreams[[name]]
  if (is.null(upstreams)) upstreams <- name  # fallback: pkg name == patch name
  missing <- character(0)
  for (pkg in upstreams) {
    if (!requireNamespace(pkg, quietly = TRUE)) missing <- c(missing, pkg)
  }
  result <- if (length(missing)) {
    list(installed = FALSE,
         error = sprintf("upstream not installed: %s", paste(missing, collapse = ", ")))
  } else {
    list(installed = TRUE, error = NA_character_)
  }
  .zyme_probe_cache[[name]] <- result
  result
}

#' List available patches
#'
#' Lists every patch shipped in \code{inst/patches/} plus any test-time patches
#' registered directly via \code{register_patch()}.
#'
#' @param installed When TRUE, return only patches whose upstream package is
#'   importable in the current R session. Probes via
#'   \code{requireNamespace(..., quietly = TRUE)}, memoized per session.
#'   Default (FALSE) returns every shipped patch regardless of upstream
#'   availability — same behavior as before.
#' @export
list_patches <- function(installed = FALSE) {
  all_names <- sort(unique(c(.zyme_available_patches(), ls(.zyme_registry))))
  if (!installed) return(all_names)
  Filter(function(n) .probe_patch_installed(n)$installed, all_names)
}

#' List curated subset names
#' @export
list_subsets <- function() {
  sort(names(.zyme_subsets))
}

#' Get the patches in a subset
#'
#' Named \code{subset_patches} (not \code{subset}) to avoid masking
#' \code{base::subset}. Python-side equivalent is \code{autozyme.subset()}.
#' @param name Subset name.
#' @export
subset_patches <- function(name) {
  if (!name %in% names(.zyme_subsets)) {
    stop(sprintf("no subset named '%s'; see list_subsets()", name))
  }
  .zyme_subsets[[name]]
}

#' Deactivate one patch, a subset, or a vector of patches/subsets
#'
#' Symmetric with \code{activate()}. Unknown names raise with a did-you-mean
#' suggestion. Names that resolve but were never activated are skipped
#' silently (idempotent deactivate).
#'
#' @param name A patch name, a subset name, or a character vector mixing them.
#' @export
deactivate <- function(name) {
  resolved <- .resolve_activation_target(name)
  for (n in resolved) {
    if (!is.null(.zyme_registry[[n]])) .deactivate_one(n)
  }
  invisible(NULL)
}

#' Deactivate every patch, rebinding upstream to its original implementation
#' @export
deactivate_all <- function() {
  for (name in ls(.zyme_registry)) .deactivate_one(name)
  invisible(NULL)
}

#' Show the activation state of every available patch
#'
#' Returns "active" for patches whose targets are currently bound into
#' upstream, "inactive" otherwise. "inactive" covers both registered-but-not-
#' activated patches and patches still lazy in \code{inst/patches/}.
#' @return Named character vector mapping patch name -> "active" | "inactive".
#' @export
status <- function() {
  names_ <- list_patches()
  if (length(names_) == 0L) return(character(0))
  out <- vapply(names_, function(n) {
    p <- .zyme_registry[[n]]
    if (!is.null(p) && isTRUE(p$injected)) "active" else "inactive"
  }, character(1))
  names(out) <- names_
  out
}

#' Inspect a patch's bindings
#'
#' Returns a structured view of which upstream symbols are claimed by the
#' patch, which fast functions they map to, and the current binding state.
#' Forces source of the patch file if it hasn't been registered yet, so this
#' triggers \code{requireNamespace()} on the upstream (do not call from
#' performance-sensitive code).
#'
#' @param name Patch name.
#' @return A list with elements: name, status, tested_against,
#'   installed_version, targets (one entry per (upstream, attr) pair).
#' @export
inspect <- function(name) {
  resolved <- .resolve_activation_target(name)
  if (length(resolved) != 1L) {
    stop(sprintf("inspect() takes a single patch name; got '%s' (resolves to %s)",
                 name, paste(resolved, collapse = ", ")), call. = FALSE)
  }
  n <- resolved[1]
  probe <- .probe_patch_installed(n)
  if (!probe$installed) {
    return(list(name = n, status = "uninstalled", error = probe$error, targets = list()))
  }
  ok <- tryCatch(.ensure_registered(n), error = function(e) FALSE)
  if (!isTRUE(ok)) {
    return(list(name = n, status = "uninstalled",
                error = "patch file failed to register; see .ensure_registered",
                targets = list()))
  }
  p <- .zyme_registry[[n]]
  installed_ver <- tryCatch(
    as.character(utils::packageVersion(p$upstream)),
    error = function(e) NA_character_
  )
  target_views <- lapply(names(p$targets), function(fn_name) {
    target <- p$targets[[fn_name]]
    is_s4 <- is.list(target) && identical(target[["kind"]], "s4")
    list(
      fn_name        = fn_name,
      kind           = if (is_s4) "s4" else "namespace",
      signature      = if (is_s4) target$signature else NA_character_,
      currently_bound = isTRUE(fn_name %in% names(p$originals))
    )
  })
  list(
    name              = n,
    status            = if (isTRUE(p$injected)) "active" else "inactive",
    tested_against    = if (is.null(p$tested_against)) NA_character_ else p$tested_against,
    installed_version = installed_ver,
    upstream          = p$upstream,
    targets           = target_views
  )
}

#' Structured environment snapshot for provenance / Methods capture
#'
#' Returns autozyme version, R version, platform, and per-patch state
#' (tested vs installed upstream versions). Uses cheap
#' \code{requireNamespace()} probes — does NOT source any patch file.
#' For patches that have been registered (activate or inspect was called),
#' the snapshot includes their declared \code{tested_against}.
#'
#' @return Nested list, JSON-serializable, suitable for dumping into a
#'   reproducibility log.
#' @export
env_snapshot <- function() {
  all_names <- sort(unique(c(.zyme_available_patches(), ls(.zyme_registry))))
  patches <- lapply(all_names, function(n) {
    probe <- .probe_patch_installed(n)
    if (!probe$installed) {
      return(list(name = n, status = "uninstalled", error = probe$error))
    }
    upstreams <- .zyme_upstreams[[n]]
    if (is.null(upstreams)) upstreams <- n
    versions <- vapply(upstreams, function(pkg) {
      tryCatch(as.character(utils::packageVersion(pkg)),
               error = function(e) NA_character_)
    }, character(1))
    names(versions) <- upstreams
    p <- .zyme_registry[[n]]
    list(
      name               = n,
      status             = if (!is.null(p) && isTRUE(p$injected)) "active" else "inactive",
      tested_against     = if (is.null(p) || is.null(p$tested_against)) NA_character_ else p$tested_against,
      installed_versions = as.list(versions)
    )
  })
  list(
    autozyme_version = as.character(utils::packageVersion("autozyme")),
    r_version        = paste(R.version$major, R.version$minor, sep = "."),
    platform         = R.version$platform,
    patches          = patches
  )
}

#' One-screen dashboard of patches and subsets
#'
#' Prints a tabular summary: which patches autozyme ships, which have their
#' upstream installed, and which subsets are partially / fully installable.
#' Uses cheap probes — does not source patch files.
#' @return Invisibly, the same data \code{env_snapshot()} returns.
#' @export
dashboard <- function() {
  snap <- env_snapshot()
  cat(sprintf("autozyme %s\n", snap$autozyme_version))
  cat(sprintf("%d patches discovered:\n", length(snap$patches)))
  name_w <- max(c(nchar(vapply(snap$patches, `[[`, character(1), "name")), 1L))
  installed_count <- 0L
  for (entry in snap$patches) {
    if (identical(entry$status, "uninstalled")) {
      cat(sprintf("  x %s  %s\n", format(entry$name, width = name_w), entry$error))
    } else {
      installed_count <- installed_count + 1L
      vers <- entry$installed_versions
      ver_str <- paste(names(vers),
                       vapply(vers, function(v) if (is.na(v)) "(version unknown)" else v, character(1)),
                       collapse = ", ")
      cat(sprintf("  v %s  %s\n", format(entry$name, width = name_w), ver_str))
    }
  }
  cat(sprintf("  (%d/%d with upstream installed)\n\n", installed_count, length(snap$patches)))
  subs <- list_subsets()
  cat(sprintf("%d subsets:\n", length(subs)))
  for (s in subs) {
    members <- .zyme_subsets[[s]]
    inst <- sum(vapply(members, function(m) .probe_patch_installed(m)$installed, logical(1)))
    cat(sprintf("  %s: %s  (%d/%d installed)\n",
                s, paste(members, collapse = ", "), inst, length(members)))
  }
  cat("\nActivate with: autozyme::activate('<name>') / autozyme::activate('<subset>')\n")
  invisible(snap)
}
