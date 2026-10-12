"""Tests for src/pca_methods.py — validates eigenpair extraction methods."""

from __future__ import annotations

import numpy as np
import pytest

from src.pca_methods import (
    EigenpairsResult,
    calc_chol,
    compare_eigenpairs,
    eigenpairs_efficient,
    eigenpairs_power_iterations,
)


@pytest.fixture
def ret_small():
    """50 stocks × 20 return observations (N=50, M+1=20)."""
    rng = np.random.default_rng(7)
    N, T = 50, 20
    F = rng.standard_normal((3, T))
    B = rng.standard_normal((N, 3))
    noise = rng.standard_normal((N, T)) * 0.3
    return B @ F + noise


def test_calc_chol_shape():
    """calc_chol should return an M×M matrix."""
    for m in [5, 10, 19]:
        p = calc_chol(m)
        assert p.shape == (m, m), f"Wrong shape for m={m}"


def test_efficient_eigenvalues_descending(ret_small):
    """Efficient method: eigenvalues should be descending."""
    result = eigenpairs_efficient(ret_small, calc_cor=True)
    diffs = np.diff(result.values)
    assert np.all(diffs <= 1e-8), "Eigenvalues not descending"


def test_efficient_eigenvectors_orthonormal(ret_small):
    """Efficient method: eigenvectors should be orthonormal."""
    result = eigenpairs_efficient(ret_small, calc_cor=True)
    M = result.vectors.shape[1]
    gram = result.vectors.T @ result.vectors
    np.testing.assert_allclose(gram, np.eye(M), atol=1e-6)


def test_efficient_vs_numpy_eigh(ret_small):
    """Efficient eigenpairs should match numpy.linalg.eigh on same data."""
    from src.risk_model import _sample_cov_eigh

    result_eff = eigenpairs_efficient(ret_small, calc_cor=True)
    eigvals_ref, eigvecs_ref, _ = _sample_cov_eigh(ret_small, use_cor=True)

    M = result_eff.vectors.shape[1]
    ref = EigenpairsResult(values=eigvals_ref[:M], vectors=eigvecs_ref[:, :M])
    cmp = compare_eigenpairs(result_eff, ref, atol=1e-5)

    assert cmp["pass"], (
        f"Efficient vs numpy: max_val_diff={cmp['max_eigenvalue_diff']:.2e}, "
        f"max_vec_diff={cmp['max_eigenvector_diff']:.2e}"
    )


def test_power_iter_convergence(ret_small):
    """Power iterations should converge for small test data."""
    result = eigenpairs_power_iterations(ret_small, calc_cor=True, k=5, prec=1e-3, seed=0)
    assert len(result.values) == 5
    assert result.n_iterations is not None
    assert np.all(result.n_iterations >= 1), "Each component needs at least 1 iteration"


def test_power_iter_positive_eigenvalues(ret_small):
    """Power iteration eigenvalues should be positive."""
    result = eigenpairs_power_iterations(ret_small, calc_cor=True, k=5, prec=1e-3, seed=0)
    assert np.all(result.values > 0)


def test_power_iter_vs_numpy(ret_small):
    """Power iterations should agree with numpy.linalg.eigh up to tolerance."""
    from src.risk_model import _sample_cov_eigh

    result_pow = eigenpairs_power_iterations(ret_small, calc_cor=True, k=5, prec=1e-3, seed=0)
    eigvals_ref, eigvecs_ref, _ = _sample_cov_eigh(ret_small, use_cor=True)

    ref = EigenpairsResult(values=eigvals_ref[:5], vectors=eigvecs_ref[:, :5])
    cmp = compare_eigenpairs(result_pow, ref, atol=1e-2)
    assert cmp["pass"], (
        f"Power vs numpy: max_val_diff={cmp['max_eigenvalue_diff']:.2e}"
    )


def test_power_iter_first_component_unit_init(ret_small):
    """First component should converge from unit initialization (no seed variation)."""
    result1 = eigenpairs_power_iterations(ret_small, k=1, prec=1e-3, seed=0)
    result2 = eigenpairs_power_iterations(ret_small, k=1, prec=1e-3, seed=99)
    # First component initialization is deterministic (unit vector), so should match
    np.testing.assert_allclose(
        np.abs(result1.values[0] - result2.values[0]),
        0.0, atol=1e-8,
        err_msg="First eigenvalue should be deterministic",
    )


def test_compare_eigenpairs_identical():
    """compare_eigenpairs should pass when comparing a result to itself."""
    rng = np.random.default_rng(5)
    ret = rng.standard_normal((30, 10))
    result = eigenpairs_efficient(ret, calc_cor=True)
    cmp = compare_eigenpairs(result, result, atol=1e-10)
    assert cmp["pass"]
    np.testing.assert_allclose(cmp["max_eigenvalue_diff"], 0.0, atol=1e-12)


def test_efficient_elapsed_positive(ret_small):
    """Elapsed time should be non-negative."""
    result = eigenpairs_efficient(ret_small)
    assert result.elapsed_sec >= 0.0
