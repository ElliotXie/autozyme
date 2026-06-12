#!/usr/bin/env bash
# Build source tarball and run R CMD check --as-cran locally.
# Mirrors what .github/workflows/r-build.yml does in CI.
# Run from anywhere; resolves paths relative to this script.

set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "$PKG_DIR")"
cd "$PARENT_DIR"

echo "==> Cleaning prior build artifacts"
rm -f autozyme_*.tar.gz
rm -rf autozyme.Rcheck

echo "==> Regenerating Rcpp exports"
Rscript -e 'Rcpp::compileAttributes("autozyme_r")'

echo "==> R CMD build autozyme_r"
R CMD build autozyme_r

TARBALL=$(ls -1 autozyme_*.tar.gz | head -1)
echo "==> Tarball: $TARBALL"

echo "==> R CMD check --as-cran --no-manual $TARBALL"
R CMD check --as-cran --no-manual "$TARBALL"

echo
echo "Artifacts:"
ls -lh autozyme_*.tar.gz
echo "(check log at autozyme.Rcheck/00check.log)"
