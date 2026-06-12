"""Contract tests for ``sc.pp.log1p``.

Same latent ``copy=True`` bug shape as ``normalize_total`` — the patched
fast path mutated in-place and returned None regardless of ``copy=``.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("copy", [False, True])
def test_log1p_return_type_matches_vanilla(tiny_adata, copy):
    import autozyme
    import scanpy as sc

    a_fast = tiny_adata.copy()
    a_vanilla = tiny_adata.copy()

    with autozyme.disabled():
        ref = sc.pp.log1p(a_vanilla, copy=copy)
    out = sc.pp.log1p(a_fast, copy=copy)

    assert type(out) is type(ref), (
        f"copy={copy}: patched returned {type(out).__name__}, "
        f"vanilla returned {type(ref).__name__}"
    )


def test_log1p_copy_true_returns_distinct_object(tiny_adata):
    """copy=True must return a NEW AnnData -- the bug fixed alongside normalize_total."""
    import scanpy as sc

    out = sc.pp.log1p(tiny_adata, copy=True)
    assert out is not None
    assert out is not tiny_adata


def test_log1p_copy_false_returns_none(tiny_adata):
    import scanpy as sc

    out = sc.pp.log1p(tiny_adata)
    assert out is None


def test_log1p_zyme_false_delegates_to_upstream(tiny_adata):
    import autozyme
    import numpy as np
    import scanpy as sc

    a_escape = tiny_adata.copy()
    a_vanilla = tiny_adata.copy()
    with autozyme.disabled():
        sc.pp.log1p(a_vanilla)
    sc.pp.log1p(a_escape, zyme=False)

    np.testing.assert_allclose(
        a_escape.X.data, a_vanilla.X.data, rtol=1e-6, atol=1e-6,
    )


def test_normalize_then_log1p_chain_with_copy_true(tiny_adata):
    """The exact pattern from the May 2026 user report — was AttributeError."""
    import scanpy as sc

    b = sc.pp.normalize_total(tiny_adata, copy=True)
    c = sc.pp.log1p(b)  # mutates b in place, returns None
    assert c is None
    assert b is not None
    assert b is not tiny_adata
