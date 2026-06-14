"""Unit tests for autozyme.scanpy._regress_out.

fast_regress_out is mostly an end-to-end OLS-residualization wrapper over
scanpy internals, so most tests need scanpy. We exercise:
  - the dispatch guards (zyme=False, empty keys, categorical key, dense+
    non-singular gram defer)
  - numeric parity of the OLS residuals against a hand-rolled numpy reference
    on the sparse and singular-gram patch-engaged paths.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _regress_out as RO


def _orig_callable():
    """A stub upstream that records it was called and leaves adata untouched."""
    calls = {"n": 0}

    def fn(adata, keys, *, layer=None, n_jobs=None, copy=False):
        calls["n"] += 1
        return None

    return fn, calls


# --------------------------------------------------------------------------
# Dispatch: zyme=False
# --------------------------------------------------------------------------

def test_zyme_false_delegates(monkeypatch):
    fn, calls = _orig_callable()
    monkeypatch.setattr(RO, "_orig_regress_out", lambda: fn)
    RO.fast_regress_out("ADATA", "key", zyme=False)
    assert calls["n"] == 1


# --------------------------------------------------------------------------
# Patch-engaged numeric path (needs scanpy for sanitize/get/set helpers)
# --------------------------------------------------------------------------

def _adata(X, obs):
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    return ad.AnnData(X, obs=pd.DataFrame(obs, index=[str(i) for i in range(X.shape[0])]))


def _ols_residuals_reference(X_dense, regressors):
    """Reference OLS residuals: X - R @ pinv(R'R) R' X, column by column.

    Matches the math fast_regress_out implements (pinv closed-form).
    """
    gram = regressors.T @ regressors
    coeff = np.linalg.pinv(gram) @ (regressors.T @ X_dense)
    return X_dense - regressors @ coeff


def test_regress_out_sparse_matches_numpy_reference():
    pytest.importorskip("scanpy")
    rng = np.random.default_rng(0)
    n, m = 30, 8
    dense = rng.random((n, m)).astype(np.float32)
    dense[dense < 0.4] = 0.0  # make it sparse-ish
    X = sparse.csr_matrix(dense)
    cov = rng.random(n).astype(np.float32)
    a = _adata(X, {"cov": cov})

    # Reference uses regressors = [1, cov].
    regressors = np.empty((n, 2), dtype=np.float32)
    regressors[:, 0] = 1.0
    regressors[:, 1] = cov
    ref = _ols_residuals_reference(dense, regressors)

    RO.fast_regress_out(a, "cov")
    got = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_regress_out_singular_gram_path_matches_reference():
    """A constant-zero covariate makes R'R singular -> pinv path engaged.

    Upstream would detour through the per-gene statsmodels GLM fallback; the
    patch uses pinv. Residuals must still match the closed-form reference.
    """
    pytest.importorskip("scanpy")
    rng = np.random.default_rng(1)
    n, m = 25, 6
    dense = rng.random((n, m)).astype(np.float32)
    X = sparse.csr_matrix(dense)
    zero_cov = np.zeros(n, dtype=np.float32)  # constant-zero -> singular gram
    a = _adata(X, {"pct_mt": zero_cov})

    regressors = np.empty((n, 2), dtype=np.float32)
    regressors[:, 0] = 1.0
    regressors[:, 1] = zero_cov
    ref = _ols_residuals_reference(dense, regressors)

    RO.fast_regress_out(a, "pct_mt")
    got = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_regress_out_copy_returns_new():
    pytest.importorskip("scanpy")
    rng = np.random.default_rng(2)
    X = sparse.csr_matrix(rng.random((20, 5)).astype(np.float32))
    a = _adata(X, {"cov": rng.random(20).astype(np.float32)})
    before = a.X.copy()
    out = RO.fast_regress_out(a, "cov", copy=True)
    assert out is not None and out is not a
    np.testing.assert_allclose(a.X.toarray(), before.toarray())


def test_regress_out_empty_keys_delegates(monkeypatch):
    pytest.importorskip("scanpy")
    rng = np.random.default_rng(3)
    X = sparse.csr_matrix(rng.random((10, 4)).astype(np.float32))
    a = _adata(X, {"cov": rng.random(10).astype(np.float32)})
    fn, calls = _orig_callable()
    monkeypatch.setattr(RO, "_orig_regress_out", lambda: fn)
    RO.fast_regress_out(a, [])  # empty keys -> defer to upstream
    assert calls["n"] == 1


def test_regress_out_categorical_key_delegates(monkeypatch):
    pytest.importorskip("scanpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(4)
    X = sparse.csr_matrix(rng.random((12, 4)).astype(np.float32))
    a = _adata(X, {"batch": pd.Categorical(["a", "b"] * 6)})
    fn, calls = _orig_callable()
    monkeypatch.setattr(RO, "_orig_regress_out", lambda: fn)
    RO.fast_regress_out(a, "batch")  # categorical regressor -> defer
    assert calls["n"] == 1


def test_regress_out_dense_nonsingular_defers(monkeypatch):
    """Dense + non-singular gram: upstream's numba kernel already does this,
    so the patch defers (guaranteed never-worse contract)."""
    pytest.importorskip("scanpy")
    rng = np.random.default_rng(5)
    X = rng.random((20, 5)).astype(np.float32)  # dense .X
    a = _adata(X, {"cov": rng.random(20).astype(np.float32)})
    fn, calls = _orig_callable()
    monkeypatch.setattr(RO, "_orig_regress_out", lambda: fn)
    RO.fast_regress_out(a, "cov")
    assert calls["n"] == 1


def test_regress_out_str_key_normalized_to_list():
    """A single str key is internally wrapped to [key]; just confirm it runs
    end-to-end and residualizes (slope ~ removed)."""
    pytest.importorskip("scanpy")
    rng = np.random.default_rng(6)
    n = 40
    cov = rng.random(n).astype(np.float32)
    # gene linearly driven by cov + noise -> residuals nearly uncorrelated.
    g = (2.0 * cov + 0.01 * rng.random(n)).astype(np.float32)
    dense = np.stack([g, rng.random(n).astype(np.float32)], axis=1)
    X = sparse.csr_matrix(dense)
    a = _adata(X, {"cov": cov})
    RO.fast_regress_out(a, "cov")
    got = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    # Residual of gene 0 should be near-uncorrelated with cov.
    r = np.corrcoef(got[:, 0], cov)[0, 1]
    assert abs(r) < 0.2
