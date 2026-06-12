"""Contract tests for ``sc.pp.scale`` — kwarg-heavy patched API.

``fast_scale`` has 6 kwargs (zero_center, max_value, copy, layer, obsm,
mask_obs) plus the autozyme-injected ``zyme``. Only a subset triggers the
fast numba path; the rest delegate to upstream. Contract test walks the
combinations that exercise (a) the fast path, (b) each delegation gate.
"""
from __future__ import annotations

import pytest


_FAST_PATH_CASES = [
    # (copy, max_value) -- zero_center=True default, no layer/obsm/mask_obs
    (False, None),
    (True,  None),
    (False, 10.0),
    (True,  10.0),
]


@pytest.mark.parametrize("copy,max_value", _FAST_PATH_CASES)
def test_scale_fast_path_return_type(tiny_adata, copy, max_value):
    """Fast-path kwarg combos must match upstream's return type."""
    import autozyme
    import scanpy as sc

    a_fast = tiny_adata.copy()
    a_vanilla = tiny_adata.copy()

    with autozyme.disabled():
        ref = sc.pp.scale(a_vanilla, copy=copy, max_value=max_value)
    out = sc.pp.scale(a_fast, copy=copy, max_value=max_value)

    assert type(out) is type(ref), (
        f"copy={copy}, max_value={max_value}: "
        f"patched={type(out).__name__}, vanilla={type(ref).__name__}"
    )


def test_scale_copy_true_returns_distinct(tiny_adata):
    import scanpy as sc

    out = sc.pp.scale(tiny_adata, copy=True)
    assert out is not None
    assert out is not tiny_adata


def test_scale_copy_false_returns_none(tiny_adata):
    import scanpy as sc

    assert sc.pp.scale(tiny_adata) is None


# Delegation-gate tests: these kwargs force fast_scale to drop through to
# upstream. The contract is that delegation preserves the upstream return
# type / identity.

def test_scale_zero_center_false_delegates(tiny_adata):
    """zero_center=False is not implemented in fast path -> upstream."""
    import autozyme
    import scanpy as sc

    a_fast = tiny_adata.copy()
    a_vanilla = tiny_adata.copy()
    with autozyme.disabled():
        ref = sc.pp.scale(a_vanilla, zero_center=False, copy=True)
    out = sc.pp.scale(a_fast, zero_center=False, copy=True)
    assert type(out) is type(ref)
    assert out is not a_fast


def test_scale_dense_input_delegates(tiny_dense_adata):
    """Non-CSR (dense) input is not on the fast path -> upstream."""
    import autozyme
    import scanpy as sc

    a_fast = tiny_dense_adata.copy()
    a_vanilla = tiny_dense_adata.copy()
    with autozyme.disabled():
        ref = sc.pp.scale(a_vanilla, copy=True)
    out = sc.pp.scale(a_fast, copy=True)
    assert type(out) is type(ref)


def test_scale_zyme_false_delegates(tiny_adata):
    import autozyme
    import numpy as np
    import scanpy as sc

    a_escape = tiny_adata.copy()
    a_vanilla = tiny_adata.copy()
    with autozyme.disabled():
        sc.pp.scale(a_vanilla)
    sc.pp.scale(a_escape, zyme=False)

    np.testing.assert_allclose(
        np.asarray(a_escape.X), np.asarray(a_vanilla.X),
        rtol=1e-5, atol=1e-5,
    )
