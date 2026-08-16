"""Curated patch bundles and known cross-patch conflicts.

Subsets group patches that:
  (a) are commonly used together in the same downstream workflow, AND
  (b) are known to import + warm-up safely in the same process.

Bundle naming reflects realistic co-activation -- the cross-domain CI
matrix would burn minutes (and produce false-positive bugs) trying to
co-activate scriabin (single-cell) with xclim (climate), so bundles are
scoped per-domain and CI exercises each bundle as one unit.

Subset membership decisions (the *why* matters more than the list, since
the list will grow):

  - "scrna_core": the everyday scanpy + scvelo pair -- AnnData
    backbone and the most common co-activation.
  - "scrna_spatial": scanpy + cell2location + squidpy_cooccurrence --
    spatial transcriptomics workflow; both cell2location and squidpy
    depend on scanpy.
  - "molecular_dynamics": mdanalysis_rmsd + prody -- protein structural
    analysis; both operate on MDAnalysis Universe objects.
  - "climate" stays separate -- xclim sets NUMBA_NUM_THREADS=1 at import
    and CANNOT cohabit safely with scanpy's numba kernels (see CONFLICTS).

Adding a new patch: drop it into the most specific subset(s) it belongs
to. If it conflicts with anything already in a subset, give it a NEW
subset and add an entry to CONFLICTS below.
"""
from __future__ import annotations

SUBSETS: dict[str, list[str]] = {
    "scrna_core":         ["scanpy", "scvelo"],
    "scrna_spatial":      ["scanpy", "cell2location", "squidpy_cooccurrence"],
    "molecular_dynamics": ["mdanalysis_rmsd", "prody"],
    "climate":            ["xclim"],
}

# Declarative manifest: patch name -> top-level upstream packages it depends on.
# `list_patches(installed=True)` / dashboard / env_snapshot consult this via
# `importlib.util.find_spec` to answer "is upstream available?" WITHOUT executing
# the patch's heavy imports (TF, torch, numba). A patch is "installed" iff all
# of its upstreams have a findable spec.
#
# Keep in sync with each patch's `register_patch(targets=...)`. New patches
# packaged via 4_package.md must add their entry here.
# AUTOZYME-GENERATED-UPSTREAMS-BEGIN
UPSTREAMS: dict[str, list[str]] = {
    "_test_json": ["json"],
    "astropy_boxleastsquares": ["astropy"],
    "cell2location": ["cell2location", "pyro"],
    "cellphonedb": ["cellphonedb"],
    "dipy": ["dipy"],
    "fipy": ["fipy"],
    "lifelines": ["lifelines", "numba"],
    "mdanalysis_rmsd": ["MDAnalysis"],
    "obspy": ["obspy"],
    "prody": ["prody"],
    "sarsen": ["sarsen", "xarray_sentinel"],
    "scanpy": ["scanpy"],
    "scvelo": ["scvelo"],
    "squidpy_cooccurrence": ["squidpy"],
    "statsmodels": ["statsmodels"],
    "xclim": ["xclim"],
}
# AUTOZYME-GENERATED-UPSTREAMS-END

# Pairs of patches known to interact badly when activated in the same process.
# `activate()` consults this and warns (does not raise) when a user lights up
# both sides — they might still want the combo with knobs tuned manually.
# Each entry: (frozenset of patch names, human reason).
CONFLICTS: list[tuple[frozenset[str], str]] = []
