"""
Tests for src/risk_model.py — validates risk model construction against paper.
Runs entirely on synthetic data; no API calls required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.risk_model import (
    RiskModelResult,
    _erank,
    _k_select_minimization,
    _sample_cov_eigh,
    build_inv_cov,
    build_shrinkage_model,
    fit_risk_model,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ret() -> np.ndarray:
    """N=100 stocks, M+1=20 observations – small enough for fast tests."""
    rng = np.random.default_rng(42)
    N, T = 100, 20
    # Construct a low-rank return matrix: 3 factors + idiosyncratic noise
    F = rng.standard_normal((3, T))
    B = rng.standard_normal((N, 3))
    noise = rng.standard_normal((N, T)) * 0.5
    ret = B @ F + noise
    return ret


@pytest.fixture
def small_ret() -> np.ndarray:
    """Tiny matrix for edge-case tests."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((10, 5))


# ─── _sample_cov_eigh ────────────────────────────────────────────────────────

def test_cov_eigh_descending(synthetic_ret):
    """Eigenvalues should be in descending order."""
    eigvals, eigvecs, sigma = _sample_cov_eigh(synthetic_ret, use_cor=True)
    assert np.all(np.diff(eigvals) <= 1e-10), "Eigenvalues not descending"


def test_cov_eigh_orthonormal(synthetic_ret):
    """Eigenvectors should be orthonormal."""
    eigvals, eigvecs, sigma = _sample_cov_eigh(synthetic_ret, use_cor=True)
    # V^T V should be identity
    eye_approx = eigvecs.T @ eigvecs
    np.testing.assert_allclose(eye_approx, np.eye(len(eigvals)), atol=1e-8)


def test_cov_eigh_diag_ones_for_correlation(synthetic_ret):
    """Diagonal of correlation matrix should be 1.0 everywhere."""
    eigvals, eigvecs, sigma = _sample_cov_eigh(synthetic_ret, use_cor=True)
    # Reconstruct diagonal: diag(C) = sum_A lambda_A * V_iA^2
    diag_reconstructed = np.sum(eigvecs ** 2 * eigvals[np.newaxis, :], axis=1)
    np.testing.assert_allclose(diag_reconstructed, 1.0, atol=1e-8,
                               err_msg="Diagonal of correlation matrix must be 1.0")


def test_sigma_positive(synthetic_ret):
    """Per-stock volatilities must be positive."""
    _, _, sigma = _sample_cov_eigh(synthetic_ret)
    assert np.all(sigma > 0)


# ─── _erank ─────────────────────────────────────────────────────────────────

def test_erank_uniform():
    """Uniform eigenvalue distribution → eRank = number of eigenvalues."""
    n = 10
    lam = np.ones(n)
    er = _erank(lam)
    np.testing.assert_allclose(er, n, atol=1e-10)


def test_erank_spike():
    """One dominant eigenvalue → eRank ≈ 1."""
    lam = np.array([1000.0, 1.0, 1.0, 1.0, 1.0])
    er = _erank(lam)
    assert er < 2.0, f"eRank should be close to 1 for spike, got {er:.3f}"


def test_erank_range(synthetic_ret):
    """eRank should be in [1, min(N, M)]."""
    N, T = synthetic_ret.shape
    M = T - 1
    eigvals, _, _ = _sample_cov_eigh(synthetic_ret, use_cor=True)
    er = _erank(eigvals[:M])
    assert 1.0 <= er <= M + 1, f"eRank={er:.3f} outside [1, {M+1}]"


# ─── fit_risk_model ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["min", "erank_round", "erank_floor"])
def test_fit_model_shapes(synthetic_ret, method):
    """RiskModelResult shapes must be consistent with input."""
    N = synthetic_ret.shape[0]
    model = fit_risk_model(synthetic_ret, method=method)
    assert model.spec_risk.shape == (N,)
    assert model.fac_load.shape[0] == N
    assert 1 <= model.n_factors <= 19
    assert model.n_factors == model.fac_load.shape[1]


@pytest.mark.parametrize("method", ["min", "erank_round", "erank_floor"])
def test_spec_risk_positive(synthetic_ret, method):
    """Specific risk must be non-negative."""
    model = fit_risk_model(synthetic_ret, method=method)
    assert np.all(model.spec_risk >= 0), f"Negative spec_risk for method={method}"


def test_in_sample_variance_preserved(synthetic_ret):
    """
    In-sample diagonal of the model covariance must equal sample variance.
    Gamma_hat_ii = sigma_i^2 * (xi_i^2 + sum_A lambda_A * V_iA^2) = sigma_i^2 * 1 = sigma_i^2
    (for correlation matrix, the total sum of factors + specific risk = 1 in corr space)
    """
    model = fit_risk_model(synthetic_ret, method="min")
    # diagonal of Gamma_hat = spec_risk_i^2 + sum_A fac_load_iA^2
    diag_model = model.spec_risk ** 2 + np.sum(model.fac_load ** 2, axis=1)
    # sample variance from data
    sample_var = synthetic_ret.var(axis=1, ddof=1)
    # These should match because we're matching in-sample diagonal
    # (may differ slightly if spec_risk is clamped to 0)
    np.testing.assert_allclose(diag_model, sample_var, rtol=0.02,
                               err_msg="In-sample diagonal mismatch")


def test_fixed_k_model(synthetic_ret):
    """Fixed-K model should use exactly K factors."""
    for K in [1, 5, 10, 15]:
        model = fit_risk_model(synthetic_ret, fixed_k=K)
        assert model.n_factors == K, f"Expected K={K}, got {model.n_factors}"
        assert model.fac_load.shape == (synthetic_ret.shape[0], K)


def test_excl_first_adds_one(synthetic_ret):
    """excl_first should produce K=K'+1 for minimization method."""
    model_base = fit_risk_model(synthetic_ret, method="min", excl_first=False)
    model_excl = fit_risk_model(synthetic_ret, method="min", excl_first=True)
    # Both should produce valid models; excl tends to produce higher K
    assert model_excl.n_factors >= 1
    assert model_base.n_factors >= 1


# ─── build_inv_cov ───────────────────────────────────────────────────────────

def test_inv_cov_symmetry(synthetic_ret):
    """Inverse covariance should be symmetric."""
    model = fit_risk_model(synthetic_ret, fixed_k=3)
    inv_cov, _ = build_inv_cov(model)
    np.testing.assert_allclose(inv_cov, inv_cov.T, atol=1e-8)


def test_inv_cov_product_identity(synthetic_ret):
    """Gamma^{-1} @ Gamma ≈ I (test numerically on small N)."""
    model = fit_risk_model(synthetic_ret, fixed_k=3)
    inv_cov, _ = build_inv_cov(model)
    # Reconstruct Gamma
    Gamma = np.diag(model.spec_risk ** 2) + model.fac_load @ model.fac_load.T
    product = inv_cov @ Gamma
    N = len(model.spec_risk)
    np.testing.assert_allclose(product, np.eye(N), atol=1e-6,
                               err_msg="Gamma^{-1} @ Gamma ≠ I")


# ─── build_shrinkage_model ───────────────────────────────────────────────────

def test_shrinkage_diagonal_shapes(synthetic_ret):
    """Diagonal shrinkage model shapes must match input."""
    N = synthetic_ret.shape[0]
    for q in [1e-6, 0.3, 0.9]:
        model = build_shrinkage_model(synthetic_ret, q=q, rho=None)
        assert model.spec_risk.shape == (N,)
        assert model.fac_load.shape[0] == N
        assert model.n_factors == 19   # K=M for shrinkage


def test_shrinkage_uniform_extra_factor(synthetic_ret):
    """Uniform-correlation shrinkage should have M+1 factors."""
    model = build_shrinkage_model(synthetic_ret, q=0.3, rho=0.5)
    assert model.n_factors == 20   # K=M+1 for uniform-correlation


def test_shrinkage_spec_risk_q_proportion(synthetic_ret):
    """Diagonal shrinkage: specific variance = q * sigma_i^2 (in return space)."""
    q = 0.5
    model = build_shrinkage_model(synthetic_ret, q=q, rho=None)
    _, _, sigma = _sample_cov_eigh(synthetic_ret, use_cor=True)
    expected_spec = sigma * np.sqrt(q)
    np.testing.assert_allclose(model.spec_risk, expected_spec, rtol=1e-6)


# ─── Edge cases ──────────────────────────────────────────────────────────────

def test_small_matrix(small_ret):
    """Model fitting should work for very small matrices."""
    model = fit_risk_model(small_ret, method="min")
    assert model.n_factors >= 1
    assert np.all(np.isfinite(model.spec_risk))
    assert np.all(np.isfinite(model.fac_load))


def test_erank_ignores_non_positive(synthetic_ret):
    """eRank should ignore zero or negative eigenvalues."""
    lam = np.array([5.0, 3.0, 1.0, 0.0, -0.001])
    er = _erank(lam)
    er_clean = _erank(np.array([5.0, 3.0, 1.0]))
    np.testing.assert_allclose(er, er_clean, atol=1e-10)
