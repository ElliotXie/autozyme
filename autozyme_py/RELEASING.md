# Releasing `autozyme` to PyPI

The PyPI project name is **`autozyme`** (currently unclaimed: the first publish
takes the name). Distribution is automated by
[`.github/workflows/release.yml`](../.github/workflows/release.yml): it builds
cross-platform wheels with cibuildwheel plus an sdist, and uploads them via PyPI
**Trusted Publishing** (OIDC, no stored token).

This package has a native C extension (`_native_scanpy`). It is **optional**:
`setup.py`'s `_OptionalBuildExt` falls back to stock scanpy if the kernel will
not compile, so an install never fails. The wheels exist so users get the native
speedup without a local compiler.

## What gets built

| Platform | Wheels | Notes |
|----------|--------|-------|
| Linux | manylinux `x86_64`, CPython 3.10-3.13 | `aarch64` and musllinux are off (slow QEMU / no Alpine validation); enable in `[tool.cibuildwheel.linux]` + a QEMU step when needed. |
| macOS | `x86_64` + `arm64`, CPython 3.10-3.13 | Built from one arm64 runner; the non-native arch is built but not tested. |
| Windows | `win_amd64`, CPython 3.10-3.13 | The MSVC `/std:c11` C path was not CI-validated before this workflow. If it fails to compile, the wheel still ships (pure-Python fallback). **Confirm `_native_scanpy*.pyd` is in the Windows wheel** by checking the build log; if absent, the Windows wheel has no native acceleration. |
| (any) | sdist | Fallback for platforms without a wheel; compiles on install (or falls back). |

The build matrix is configured in `[tool.cibuildwheel]` in `pyproject.toml`.

## One-time setup (before the first publish)

1. **PyPI Trusted Publisher** -- on https://pypi.org, project `autozyme`
   (use "pending publisher" since the project does not exist yet), add a
   GitHub Actions publisher:
   - Owner: `ElliotXie`  ·  Repo: `autozyme`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
2. **TestPyPI Trusted Publisher** -- same on https://test.pypi.org, environment
   name `testpypi` (only needed for the rehearsal step).
3. **GitHub environments** -- in the `ElliotXie/autozyme` repo settings, create
   environments `pypi` and `testpypi` (names must match steps 1-2). Optionally
   add a required reviewer on `pypi` for a manual approval gate before any
   upload.

## Rehearse on TestPyPI (recommended, uploads nothing to real PyPI)

1. Actions tab -> "Publish autozyme (Python) to PyPI" -> Run workflow ->
   `publish_target = testpypi`.
2. Install from TestPyPI into a clean env and smoke it:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ autozyme
   python -c "import autozyme; print(autozyme.__version__, len(autozyme.list_patches()))"
   ```
   (TestPyPI versions are immutable too; bump the version if you re-rehearse.)

## Dry run anytime (builds the full matrix, uploads nothing)

Actions tab -> Run workflow -> leave `publish_target = none`. Use this to verify
the Linux/Windows builds (which cannot be reproduced on the Mac) stay green.

## Publish a real release

1. Make sure the working tree is current: speedups finalized, framework synced
   to the release repo (`python scripts/sync_release.py` reports IN SYNC), CI
   green.
2. Set the version in **both** `autozyme_py/pyproject.toml` and
   `src/autozyme/__init__.py` (`__version__`) -- they must match. PyPI versions
   are immutable; a typo means burning a version number.
3. Commit, then tag and push from the **release repo** (the one wired to
   GitHub Actions):
   ```bash
   git tag py-v0.3.0
   git push origin py-v0.3.0
   ```
   The `py-v<version>` tag triggers the workflow; the `publish_pypi` job guards
   that the tag version matches the built sdist before uploading.
4. Confirm at https://pypi.org/project/autozyme/ and smoke a clean
   `pip install autozyme`.

## Notes

- **Version bumps** are duplicated in py/r/cli; `scripts/check_versions.py`
  (run in tier-a CI) enforces they agree.
- **`SCOPE.md` files are not shipped** in the wheel/sdist (by design); they live
  next to each patch's source in the repo. The README points users there.
- The committed `src/autozyme/_native_scanpy*.so` files are a convenience for
  source-tree use; they do **not** leak into built wheels/sdists (each wheel
  ships only its freshly compiled extension).
