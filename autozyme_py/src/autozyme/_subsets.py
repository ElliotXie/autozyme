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

  - "scrna_core": the everyday scanpy + scvelo + sccoda trio -- AnnData
    backbone, pulls torch (sccoda's TF) lazily. Most common combo.
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
    "scrna_core":         ["scanpy", "sccoda", "scvelo"],
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
UPSTREAMS: dict[str, list[str]] = {
    "cell2location": ["cell2location", "pyro"],
    "sccoda":        ["sccoda", "tensorflow", "tensorflow_probability", "tf_keras"],
    "xclim":         ["xclim"],
    "obspy":         ["obspy"],
    "prody":         ["prody"],
    "scvelo":        ["scvelo"],
    "lifelines":     ["lifelines"],
    "mdanalysis_rmsd": ["MDAnalysis"],
    "statsmodels":   ["statsmodels"],
    "dipy":          ["dipy"],
    "fipy":          ["fipy"],
    "sarsen":        ["sarsen", "xarray_sentinel"],
    "scanpy":        ["scanpy"],
    "cellphonedb":   ["cellphonedb"],
    "squidpy_cooccurrence": ["squidpy"],
    "astropy_boxleastsquares": ["astropy"],
    # Test-only synthetic patch (see autozyme/_test_json/__init__.py).
    # Underscore prefix excludes it from _AVAILABLE / list_patches().
    "_test_json":    ["json"],
}

# Pairs of patches known to interact badly when activated in the same process.
# `activate()` consults this and warns (does not raise) when a user lights up
# both sides — they might still want the combo with knobs tuned manually.
# Each entry: (frozenset of patch names, human reason).
CONFLICTS: list[tuple[frozenset[str], str]] = [
    (
        frozenset({"sccoda", "xclim"}),
        "sccoda's TensorFlow threading layer can deadlock xclim's numba "
        "parallel JIT. Set NUMBA_NUM_THREADS=1 before importing xclim if "
        "you need both in one process.",
    ),
]
