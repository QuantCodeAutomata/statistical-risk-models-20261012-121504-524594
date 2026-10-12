"""Tests for src/metrics.py — validates P&L and performance metric computations."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import PerformanceMetrics, aggregate_metrics, compute_daily_pnl, metrics_to_dict


# ─── compute_daily_pnl ────────────────────────────────────────────────────────

def test_pnl_sign_long_gain():
    """Long position in rising stock → positive P&L."""
    H = np.array([1_000_000.0])      # long $1M
    unadj_open = np.array([100.0])
    unadj_close = np.array([101.0])  # +1%
    pnl, shares = compute_daily_pnl(H, unadj_open, unadj_close)
    assert pnl[0] > 0


def test_pnl_sign_short_gain():
    """Short position in falling stock → positive P&L."""
    H = np.array([-1_000_000.0])     # short $1M
    unadj_open = np.array([100.0])
    unadj_close = np.array([99.0])   # -1%
    pnl, shares = compute_daily_pnl(H, unadj_open, unadj_close)
    assert pnl[0] > 0


def test_pnl_magnitude():
    """P&L = H * (close/open - 1): exact calculation."""
    H = np.array([2_000_000.0])
    unadj_open = np.array([50.0])
    unadj_close = np.array([51.0])
    pnl, shares = compute_daily_pnl(H, unadj_open, unadj_close)
    expected = 2_000_000.0 * (51.0 / 50.0 - 1.0)
    np.testing.assert_allclose(pnl[0], expected, rtol=1e-12)


def test_shares_formula():
    """Shares = 2 * |H| / open (both legs of a round-trip)."""
    H = np.array([500_000.0, -300_000.0])
    open_ = np.array([20.0, 30.0])
    _, shares = compute_daily_pnl(H, open_, np.array([21.0, 29.0]))
    expected_shares = 2.0 * np.abs(H) / open_
    np.testing.assert_allclose(shares, expected_shares, rtol=1e-12)


def test_invalid_prices_handled():
    """NaN or zero open should produce zero P&L and zero shares."""
    H = np.array([1_000_000.0, 1_000_000.0])
    unadj_open = np.array([0.0, np.nan])
    unadj_close = np.array([1.0, 1.0])
    pnl, shares = compute_daily_pnl(H, unadj_open, unadj_close)
    np.testing.assert_array_equal(pnl, 0.0)
    np.testing.assert_array_equal(shares, 0.0)


# ─── aggregate_metrics ───────────────────────────────────────────────────────

def test_roc_formula():
    """ROC_annual = 252 * mean(daily_pnl) / I * 100."""
    inv = 20_000_000.0
    daily_pnl = np.full(252, 10_000.0)   # constant $10k/day
    daily_shares = np.full(252, 5000.0)
    m = aggregate_metrics(daily_pnl, daily_shares, inv)
    expected_roc = 252.0 * 10_000.0 / inv * 100.0
    np.testing.assert_allclose(m.roc_annual, expected_roc, rtol=1e-10)


def test_sr_formula():
    """SR_annual = (mean/std) * sqrt(252)."""
    daily_pnl = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
    daily_shares = np.ones(50)
    m = aggregate_metrics(daily_pnl, daily_shares)
    expected = daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(m.sr_annual, expected, rtol=1e-10)


def test_cps_formula():
    """CPS = total_pnl / total_shares * 100 (cents per share, Section 3.4.3)."""
    daily_pnl = np.array([100.0, 200.0, 300.0])
    daily_shares = np.array([50.0, 100.0, 150.0])
    m = aggregate_metrics(daily_pnl, daily_shares)
    # multiply by 100 to convert dollars-per-share to cents-per-share
    expected_cps = (100 + 200 + 300) / (50 + 100 + 150) * 100.0
    np.testing.assert_allclose(m.cps, expected_cps, rtol=1e-10)


def test_zero_pnl_zero_shares():
    """All-zero P&L → zero metrics, no division errors."""
    daily_pnl = np.zeros(100)
    daily_shares = np.zeros(100)
    m = aggregate_metrics(daily_pnl, daily_shares)
    assert m.cps == 0.0
    assert m.total_pnl == 0.0


def test_single_day():
    """Works for a single trading day."""
    m = aggregate_metrics(np.array([500.0]), np.array([200.0]))
    assert m.total_pnl == 500.0
    assert m.n_days == 1


# ─── metrics_to_dict ─────────────────────────────────────────────────────────

def test_metrics_to_dict_keys():
    """metrics_to_dict should return expected keys."""
    m = aggregate_metrics(np.ones(100), np.ones(100))
    d = metrics_to_dict(m)
    assert "ROC (%)" in d
    assert "SR" in d
    assert "CPS (cents)" in d
    assert "Total PnL" in d
    assert "Total Shares" in d
    assert "Days" in d


def test_roc_percent_not_fraction():
    """ROC should be a percentage, not a fraction (should be order 10–100 for this input)."""
    inv = 20_000_000.0
    # $100k/day average → ROC ≈ 252 * 100k / 20M * 100 = 126%
    daily_pnl = np.full(252, 100_000.0)
    m = aggregate_metrics(daily_pnl, np.ones(252), inv)
    assert m.roc_annual > 10.0, f"ROC looks like fraction, not percent: {m.roc_annual}"
