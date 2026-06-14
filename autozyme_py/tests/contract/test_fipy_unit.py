"""Unit tests for the pure fingerprint/cache helpers in autozyme.fipy.

The contract test (test_fipy.py) drives Term.solve end to end through the FiPy
FV stack. Here we test the self-contained content-fingerprint helpers that gate
the matrix/LU caches; these are the correctness-critical pieces (a bad
fingerprint means a stale cached operator -> wrong solution):

  - _matrix_signature       blake2b CSR fingerprint
  - _array_state_sig        numeric/object content fingerprint
  - _iter_constraints / _constraint_state_sig / _boundary_conditions_sig
  - _binary_state_sig / _term_state_sig

fipy must import for the module to load; these helpers use only numpy/hashlib +
fipy.tools.numerix (numpy under the hood).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("fipy")

from autozyme import fipy as azfipy


# --------------------------------------------------------------------------
# _matrix_signature
# --------------------------------------------------------------------------
def _csr(n=6, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.4, format="csr", random_state=rng)
    A.data = rng.standard_normal(A.nnz)
    return A.tocsr()


def test_matrix_signature_deterministic():
    A = _csr(seed=1)
    assert azfipy._matrix_signature(A) == azfipy._matrix_signature(A.copy())


def test_matrix_signature_sensitive_to_data():
    A = _csr(seed=2)
    B = A.copy()
    B.data = B.data.copy()
    B.data[0] += 1e-9
    assert azfipy._matrix_signature(A) != azfipy._matrix_signature(B)


def test_matrix_signature_sensitive_to_structure():
    A = _csr(seed=3)
    B = _csr(seed=4)  # different sparsity pattern
    assert azfipy._matrix_signature(A) != azfipy._matrix_signature(B)


def test_matrix_signature_returns_16_bytes():
    sig = azfipy._matrix_signature(_csr(seed=5))
    assert isinstance(sig, bytes) and len(sig) == 16


# --------------------------------------------------------------------------
# _array_state_sig
# --------------------------------------------------------------------------
def test_array_state_sig_equal_for_equal_arrays():
    a = np.arange(10.0)
    assert azfipy._array_state_sig(a) == azfipy._array_state_sig(a.copy())


def test_array_state_sig_detects_inplace_mutation():
    a = np.arange(10.0)
    s1 = azfipy._array_state_sig(a)
    a2 = a.copy()
    a2[3] *= 2.0
    assert azfipy._array_state_sig(a2) != s1


def test_array_state_sig_scalar_and_none():
    assert azfipy._array_state_sig(3.5) == azfipy._array_state_sig(3.5)
    assert azfipy._array_state_sig(3.5) != azfipy._array_state_sig(3.6)
    # None is representable as a numeric array path; just must be stable.
    assert azfipy._array_state_sig(None) == azfipy._array_state_sig(None)


def test_array_state_sig_dtype_sensitive():
    a32 = np.ones(4, dtype=np.float32)
    a64 = np.ones(4, dtype=np.float64)
    assert azfipy._array_state_sig(a32) != azfipy._array_state_sig(a64)


def test_array_state_sig_object_array():
    # Object arrays go through the repr-based branch and must be stable + distinct.
    a = np.array(["x", "y"], dtype=object)
    b = np.array(["x", "z"], dtype=object)
    assert azfipy._array_state_sig(a) == azfipy._array_state_sig(
        np.array(["x", "y"], dtype=object)
    )
    assert azfipy._array_state_sig(a) != azfipy._array_state_sig(b)


# --------------------------------------------------------------------------
# _iter_constraints / _constraint_state_sig
# --------------------------------------------------------------------------
def _constraint(value, where):
    return SimpleNamespace(value=value, where=where)


def test_iter_constraints_dedupes_by_identity():
    c = _constraint(np.ones(3), np.array([True, False, True]))
    var = SimpleNamespace(constraints=[c, c], _constraints=[c], faceConstraints=[])
    got = list(azfipy._iter_constraints(var))
    assert len(got) == 1 and got[0] is c


def test_iter_constraints_collects_all_attrs():
    c1 = _constraint(np.ones(2), np.array([True, True]))
    c2 = _constraint(np.zeros(2), np.array([False, True]))
    var = SimpleNamespace(constraints=[c1], _constraints=[], faceConstraints=[c2])
    got = list(azfipy._iter_constraints(var))
    assert set(map(id, got)) == {id(c1), id(c2)}


def test_constraint_state_sig_changes_on_value_mutation():
    c = _constraint(np.array([1.0, 2.0, 3.0]), np.array([True, False, True]))
    var = SimpleNamespace(constraints=[c], _constraints=[], faceConstraints=[])
    s1 = azfipy._constraint_state_sig(var)
    c.value = np.array([1.0, 2.0, 9.0])  # in-place value change
    s2 = azfipy._constraint_state_sig(var)
    assert s1 != s2


def test_constraint_state_sig_changes_on_where_mutation():
    c = _constraint(np.array([1.0, 2.0]), np.array([True, False]))
    var = SimpleNamespace(constraints=[c], _constraints=[], faceConstraints=[])
    s1 = azfipy._constraint_state_sig(var)
    c.where = np.array([False, True])
    assert azfipy._constraint_state_sig(var) != s1


# --------------------------------------------------------------------------
# _boundary_conditions_sig
# --------------------------------------------------------------------------
def test_boundary_conditions_sig_empty():
    assert azfipy._boundary_conditions_sig(()) == ()
    assert azfipy._boundary_conditions_sig(None) == ()


def test_boundary_conditions_sig_value_sensitive():
    bc = SimpleNamespace(value=np.array([1.0]), faces=np.array([0, 1]))
    s1 = azfipy._boundary_conditions_sig([bc])
    bc2 = SimpleNamespace(value=np.array([2.0]), faces=np.array([0, 1]))
    s2 = azfipy._boundary_conditions_sig([bc2])
    assert s1 != s2


# --------------------------------------------------------------------------
# _binary_state_sig / _term_state_sig
# --------------------------------------------------------------------------
def test_binary_state_sig_reflects_subterm_coeff_change():
    var = SimpleNamespace(constraints=[], _constraints=[], faceConstraints=[])
    term = SimpleNamespace(coeff=np.array([1.0]))
    other = SimpleNamespace(coeff=np.array([2.0]))
    self = SimpleNamespace(term=term, other=other)
    s1 = azfipy._binary_state_sig(self, var, ())
    other.coeff = np.array([3.0])  # mutate sub-term coefficient
    s2 = azfipy._binary_state_sig(self, var, ())
    assert s1 != s2


def test_term_state_sig_folds_subterm_coeffs():
    var = SimpleNamespace(constraints=[], _constraints=[], faceConstraints=[])
    term = SimpleNamespace(coeff=np.array([0.5]))
    other = SimpleNamespace(coeff=np.array([0.7]))
    self = SimpleNamespace(coeff=None, term=term, other=other)
    s1 = azfipy._term_state_sig(self, var, ())
    term.coeff = np.array([0.9])  # mutate one sub-term coefficient
    s2 = azfipy._term_state_sig(self, var, ())
    assert s1 != s2
    # Same state -> same signature (cache-hit path).
    self2 = SimpleNamespace(
        coeff=None,
        term=SimpleNamespace(coeff=np.array([0.9])),
        other=SimpleNamespace(coeff=np.array([0.7])),
    )
    var2 = SimpleNamespace(constraints=[], _constraints=[], faceConstraints=[])
    assert azfipy._term_state_sig(self2, var2, ()) == s2
