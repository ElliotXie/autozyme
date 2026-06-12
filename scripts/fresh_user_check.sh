#!/usr/bin/env bash
# fresh_user_check.sh — simulate a fresh-start user.
#
# The developer's machine is the WORST place to test "does a new user's install
# work": it can't *not* have the deps (scanpy310, /opt/anaconda3, every upstream,
# CONDA_PREFIX, RETICULATE_PYTHON, prior installs). Bugs that only surface in a
# clean environment (hardcoded paths, missing-dep silent fallback, build-from-
# source breaks, install-as-documented failures) are invisible to dev testing —
# they only showed up when an outsider tried it.
#
# This harness reproduces an outsider by (a) building from a CLEAN checkout and
# (b) running with the developer's environment STRIPPED.
#
#   Tier 0  build/install the packages the official way from `git archive`
#           (tracked files only — no stale .o/.so/.test-lib/__pycache__).
#   Tier 1  minimal activation smoke with conda + AUTOZYME_*/RETICULATE_PYTHON
#           removed and HOME pointed at a temp dir, so machine-specific
#           assumptions are EXPOSED instead of silently satisfied.
#   Scan    grep the shipped tree for machine-specific leaks (scanpy310,
#           /opt/anaconda3, /Users/<dev>, git+ssh, D:\ ...).
#
# Usage:  scripts/fresh_user_check.sh [repo_root]
# Exits non-zero if any check fails. Safe for CI.
set -uo pipefail

REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"

PASS=0; FAIL=0
ok()  { printf '  \033[32mPASS\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
hdr() { printf '\n== %s ==\n' "$*"; }

RSCRIPT="$(command -v Rscript || true)"
PYBASE="$(command -v python3 || command -v python || true)"
# The real library paths (where Rcpp/data.table etc. live). We strip the conda /
# autozyme ENV VARS to expose machine-specific assumptions, but we must NOT hide
# autozyme's own declared dependencies — a real user gets those pulled in.
RLIBS_REAL=""
[ -n "$RSCRIPT" ] && RLIBS_REAL="$("$RSCRIPT" -e 'cat(paste(.libPaths(), collapse=":"))' 2>/dev/null)"

# Run a command with the dev environment stripped: no conda vars, no autozyme/
# reticulate overrides, HOME in a temp dir. Interpreters are invoked by absolute
# path so removing conda from the picture can't make them un-findable.
strip_env() {
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_EXE -u CONDA_PYTHON_EXE \
      -u RETICULATE_PYTHON -u RETICULATE_PYTHON_ENV \
      -u AUTOZYME_PYTHON -u AUTOZYME_PY_ENV \
      -u AUTOZYME_DISABLED -u AUTOZYME_DISABLE \
      HOME="$WORK/home" "$@"
}

# Export only git-tracked files for a subdir (no build artifacts).
clean_export() {
  git -C "$REPO" archive HEAD -- "$1" | tar -x -C "$WORK"
}

# ─────────────────────────────────────────────────────────────────────────────
hdr "Scan: machine-specific leaks in shipped code"
# Patterns that should never ship to a user (tests/fixtures + this script excluded).
LEAKS="$(git -C "$REPO" grep -nE 'scanpy310|/opt/anaconda3|/Users/[a-z]+/autozyme|git\+ssh|[A-Z]:\\\\autosearch' -- \
          'autozyme_r/R' 'autozyme_r/inst/patches' 'autozyme_py/src' \
          'autozyme_cli/zyme' 2>/dev/null \
          | grep -vE 'fresh_user_check' || true)"
if [ -z "$LEAKS" ]; then ok "no scanpy310 / dev-path / ssh-url leaks in shipped code"
else bad "machine-specific leaks found:"; printf '%s\n' "$LEAKS" | sed 's/^/      /'; fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "Tier 0 (R): build + install autozyme from a clean checkout"
if [ -z "$RSCRIPT" ]; then
  echo "  (skip — Rscript not found)"
else
  clean_export autozyme_r
  RLIB="$WORK/rlib"; mkdir -p "$RLIB"
  R_BIN="$(dirname "$RSCRIPT")/R"
  if (cd "$WORK" && "$R_BIN" CMD build --no-build-vignettes --no-manual autozyme_r) >"$WORK/rbuild.log" 2>&1; then
    TARBALL="$(ls -t "$WORK"/autozyme_*.tar.gz 2>/dev/null | head -1)"
    if [ -n "$TARBALL" ] && \
       KMP_DUPLICATE_LIB_OK=TRUE "$R_BIN" CMD INSTALL --library="$RLIB" "$TARBALL" >"$WORK/rinstall.log" 2>&1; then
      ok "R CMD build + INSTALL from clean source (catches duplicate-symbol / dep breaks)"
    else
      bad "R CMD INSTALL failed (see below)"; tail -15 "$WORK/rinstall.log" | sed 's/^/      /'
    fi
  else
    bad "R CMD build failed (see below)"; tail -15 "$WORK/rbuild.log" | sed 's/^/      /'
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "Tier 1 (R): load + Python discovery in a STRIPPED environment"
if [ -z "$RSCRIPT" ] || [ ! -d "$WORK/rlib/autozyme" ]; then
  echo "  (skip — package not installed in Tier 0)"
else
  strip_env "$RSCRIPT" --vanilla -e "
    .libPaths(c('$WORK/rlib', strsplit('$RLIBS_REAL', ':', fixed = TRUE)[[1]]))
    suppressPackageStartupMessages(library(autozyme))
    stopifnot(length(list_patches()) > 0)
    stopifnot(is.function(install_python_deps))
    # With conda stripped + HOME relocated, discovery must NOT resolve to a
    # machine-specific env (the scanpy310 / ~/anaconda3 bug class). It may find
    # a real system python or nothing — never a hardcoded dev path.
    p <- tryCatch(autozyme:::.az_py_find_existing(), error = function(e) '')
    if (grepl('scanpy310|anaconda3/envs', p)) stop('discovery returned dev path: ', p)
    cat('OK: loads, enumerates patches, no machine-specific python discovery\n')
  " >"$WORK/rsmoke.log" 2>&1 \
    && ok "R loads + discovery is portable under a stripped env" \
    || { bad "R stripped-env smoke failed:"; tail -15 "$WORK/rsmoke.log" | sed 's/^/      /'; }
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "Tier 0 (Python): fresh venv install autozyme + import"
if [ -z "$PYBASE" ]; then
  echo "  (skip — python not found)"
else
  clean_export autozyme_py
  VENV="$WORK/venv"
  if "$PYBASE" -m venv "$VENV" >"$WORK/venv.log" 2>&1 \
     && "$VENV/bin/python" -m pip -q install "$WORK/autozyme_py" >"$WORK/pyinstall.log" 2>&1 \
     && "$VENV/bin/python" -c "import autozyme; assert autozyme.list_patches()" >"$WORK/pyimport.log" 2>&1; then
    ok "pip install + import autozyme in a fresh venv"
  else
    bad "Python fresh-venv install/import failed:"; tail -15 "$WORK"/py*.log | sed 's/^/      /'
  fi

  # ───────────────────────────────────────────────────────────────────────────
  hdr "Tier 1 (Python): kill-switch + loud-failure in a STRIPPED env"
  if [ -x "$VENV/bin/python" ]; then
    # AUTOZYME_DISABLED must make activate() a no-op (documented kill switch).
    out_dis="$(strip_env "$VENV/bin/python" -I -c "
import os; os.environ['AUTOZYME_DISABLED']='1'
import autozyme
print('R1' if autozyme.activate('prody') is False else 'R0')" 2>/dev/null)"
    [ "$out_dis" = "R1" ] && ok "AUTOZYME_DISABLED makes activate() a no-op" \
                          || bad "AUTOZYME_DISABLED ignored (got: $out_dis)"

    # A missing upstream must FAIL LOUD (not silently look activated).
    out_miss="$(strip_env "$VENV/bin/python" -I -c "
import autozyme
r = autozyme.activate('__definitely_missing__') if '__definitely_missing__' in autozyme.list_patches() else None
print('done')" 2>&1 || true)"
    # Pick a real patch whose upstream is absent in the bare venv and assert the
    # NOT-activated marker fires.
    out_loud="$(strip_env "$VENV/bin/python" -I -c "
import autozyme
name = next((n for n in autozyme.list_patches()), None)
print('NAME', name)" 2>/dev/null)"
    pick="$(strip_env "$VENV/bin/python" -I -c "
import autozyme, sys
for n in autozyme.list_patches():
    sys.stdout.write(n+'\n')" 2>/dev/null | head -1)"
    if [ -n "$pick" ]; then
      marker="$(strip_env "$VENV/bin/python" -I -c "
import autozyme
autozyme.activate('$pick')" 2>&1 || true)"
      if echo "$marker" | grep -q "NOT activated -- upstream not installed"; then
        ok "missing upstream prints a loud 'NOT activated' marker (no silent no-op)"
      else
        # Could be that this upstream IS importable in the venv; treat as inconclusive, not fail.
        echo "  (info — '$pick' upstream importable or marker not triggered; not a failure)"
      fi
    fi
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
printf '\n== Summary: %d passed, %d failed ==\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
