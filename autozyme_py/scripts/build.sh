#!/usr/bin/env bash
# Build sdist + wheel locally and run twine check.
# Mirrors what .github/workflows/python-build.yml does in CI.
# Run from anywhere; resolves paths relative to this script.

set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PKG_DIR"

echo "==> Cleaning dist/ + *.egg-info"
rm -rf dist build src/autozyme.egg-info

echo "==> Installing build tooling"
python -m pip install --quiet --upgrade pip build twine

echo "==> Building sdist + wheel"
python -m build

echo "==> twine check"
twine check dist/*

echo
echo "Built artifacts:"
ls -lh dist/
