"""Tests for src/optimizer.py — validates Sharpe-ratio optimization."""

from __future__ import annotations

import numpy as np
import pytest

from src.optimizer import compute_holdings, optimize_bounded, optimize_unbounded
from src.risk_model import fit_risk_model


@pytest.fixture
def model_and_alpha():
    """Small 50-stock problem for optimizer tests."""
    rng = np.random.default_rng(0)
    N, T = 50, 20
    ret = rng.standard_normal((N, T)) * 0.01
    model = fit_risk_model(ret, method="min", fixed_k=3)
    alpha = rng.standard_normal(N) * 0.001  # overnight returns
    return model, alpha


def test_unbounded_dollar_neutral(model_and_alpha):
    """Unbounded holdings must sum to zero (dollar neutrality)."""
    model, alpha = model_and_alpha
    H = optimize_unbounded(alpha, model, investment_level=1e6)
    np.testing.assert_allclose(H.sum(), 0.0, atol=1.0,
                               err_msg="Holdings not dollar-neutral")


def test_unbounded_gross_exposure(model_and_alpha):
    """Gross exposure should equal investment level."""
    model, alpha = model_and_alpha
    inv = 1_000_000.0
    H = optimize_unbounded(alpha, model, investment_level=inv)
    np.testing.assert_allclose(np.abs(H).sum(), inv, rtol=1e-6)


def test_unbounded_shape(model_and_alpha):
    """Holdings must have the same shape as alpha."""
    model, alpha = model_and_alpha
    H = optimize_unbounded(alpha, model)
    assert H.shape == alpha.shape


def test_bounded_dollar_neutral(model_and_alpha):
    """Bounded holdings must sum to zero."""
    model, alpha = model_and_alpha
    bounds = np.full(len(alpha), 50_000.0)
    H = optimize_bounded(alpha, model, bounds, investment_level=1_000_000.0)
    np.testing.assert_allclose(H.sum(), 0.0, atol=10.0,
                               err_msg="Bounded holdings not dollar-neutral")


def test_bounded_respects_box(model_and_alpha):
    """Each holding must satisfy |H_i| <= bound_i."""
    model, alpha = model_and_alpha
    bounds = np.full(len(alpha), 30_000.0)
    H = optimize_bounded(alpha, model, bounds, investment_level=1_000_000.0)
    violations = np.abs(H) - bounds
    assert violations.max() < 1.0, f"Bound violated by {violations.max():.2f}"


def test_compute_holdings_dispatch(model_and_alpha):
    """compute_holdings dispatches correctly based on bounds argument."""
    model, alpha = model_and_alpha
    H_unb = compute_holdings(alpha, model, bounds=None)
    bounds = np.full(len(alpha), 50_000.0)
    H_bnd = compute_holdings(alpha, model, bounds=bounds)
    # Both return finite arrays of the correct shape
    assert np.all(np.isfinite(H_unb))
    assert np.all(np.isfinite(H_bnd))
    assert H_unb.shape == alpha.shape
    assert H_bnd.shape == alpha.shape


def test_zero_alpha_zero_holdings(model_and_alpha):
    """Zero alpha should produce zero holdings (or near-zero)."""
    model, _ = model_and_alpha
    alpha = np.zeros(len(model.spec_risk))
    H = optimize_unbounded(alpha, model)
    np.testing.assert_allclose(H, 0.0, atol=1e-6)
