"""Contract tests for ``sc.pp.normalize_total`` after ``autozyme.activate("scanpy")``.

Walks the kwarg cartesian product and asserts the patched return shape
matches vanilla scanpy. Regression target: the May 2026 ``copy=True``
returned-None bug (silent data loss, breaks ``b = sc.pp.normalize_total(
a, copy=True); sc.pp.log1p(b)``).

Template for adding a new patched API — copy this file, change the
imports + parametrize matrix to cover the new function's kwargs.
"""
from __future__ import annotations

import pytest


# kwarg matrix: every combination scanpy supports.
_RETURN_CASES = [
    # (copy, inplace, target_sum)
    (False, True, None),       # default: mutates, returns None
    (False, True, 1e4),        # mutates with explicit target_sum
    (True,  True, None),       # returns a new AnnData -- the BUG we hit
    (True,  True, 1e4),
    (False, False, None),      # returns a dict (delegated to upstream)
]


@pytest.mark.parametrize("copy,inplace,target_sum", _RETURN_CASES)
def test_normalize_total_return_type_matches_vanilla(
    tiny_adata, copy, inplace, target_sum
):
    """Patched return type must match vanilla scanpy for every kwarg combo."""
    import autozyme
    import scanpy as sc

    a_fast = tiny_adata.copy()
    a_vanilla = tiny_adata.copy()

    with autozyme.disabled():
        ref = sc.pp.normalize_total(
            a_vanilla, target_sum=target_sum, copy=copy, inplace=inplace,
        )
    out = sc.pp.normalize_total(
        a_fast, target_sum=target_sum, copy=copy, inplace=inplace,
    )

    assert type(out) is type(ref), (
        f"copy={copy}, inplace={inplace}: patched returned {type(out).__name__}, "
        f"vanilla returned {type(ref).__name__}"
    )


def test_normalize_total_copy_true_returns_distinct_object(tiny_adata):
    """copy=True must return a NEW AnnData, not None and not the input."""
    import scanpy as sc

    out = sc.pp.normalize_total(tiny_adata, copy=True)
    assert out is not None, "copy=True must not return None (the May 2026 bug)"
    assert out is not tiny_adata, "copy=True must not return the input handle"
    # The standard chain pattern must work:
    sc.pp.log1p(out)  # used to crash with AttributeError on NoneType


def test_normalize_total_copy_false_returns_none(tiny_adata):
    """copy=False / inplace=True (default) must mutate and return None."""
    import scanpy as sc

    original_id = id(tiny_adata)
    out = sc.pp.normalize_total(tiny_adata)
    assert out is None
    assert id(tiny_adata) == original_id  # same object, mutated in place


def test_normalize_total_zyme_false_delegates_to_upstream(tiny_adata):
    """Per-call ``zyme=False`` must produce vanilla-identical output."""
    import autozyme
    import numpy as np
    import scanpy as sc

    a_escape = tiny_adata.copy()
    a_vanilla = tiny_adata.copy()
    with autozyme.disabled():
        sc.pp.normalize_total(a_vanilla, target_sum=1e4)
    sc.pp.normalize_total(a_escape, target_sum=1e4, zyme=False)

    np.testing.assert_allclose(
        a_escape.X.data, a_vanilla.X.data, rtol=1e-6, atol=1e-6,
    )


def test_normalize_total_copy_true_with_zyme_false(tiny_adata):
    """copy=True + zyme=False (full upstream delegation) must still copy."""
    import scanpy as sc

    out = sc.pp.normalize_total(tiny_adata, copy=True, zyme=False)
    assert out is not None
    assert out is not tiny_adata
