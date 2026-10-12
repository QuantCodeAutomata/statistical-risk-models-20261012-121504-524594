"""
Performance metrics for intraday mean-reversion strategy (Section 3.4.4).

Daily P&L:  Pi_is = H_is * (P^C_is / P^O_is - 1)
Shares:     Q_is  = 2 * |H_is| / P^O_is

Annualized ROC  = 252 * mean(daily_pnl) / I
Annualized SR   = mean(daily_pnl) / std(daily_pnl, ddof=1) * sqrt(252)
CPS             = total_pnl / total_shares
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import INVESTMENT_LEVEL


@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics over the full backtest."""

    roc_annual: float   # Annualized return on capital (percent)
    sr_annual: float    # Annualized Sharpe ratio
    cps: float          # Cents per share (paper Section 3.4.3: total P&L / total shares * 100)
    total_pnl: float    # Total P&L over backtest (dollars)
    total_shares: float # Total shares traded
    n_days: int         # Number of trading days with non-zero positions

    def __repr__(self) -> str:
        return (
            f"ROC={self.roc_annual:.2f}%  SR={self.sr_annual:.2f}  "
            f"CPS={self.cps:.4f}  days={self.n_days}"
        )


def compute_daily_pnl(
    holdings: np.ndarray,
    unadj_open: np.ndarray,
    unadj_close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute daily P&L and shares traded for one day.

    P&L:    Pi_i = H_i * (P^C_i / P^O_i - 1)
    Shares: Q_i  = 2 * |H_i| / P^O_i

    Parameters
    ----------
    holdings   : (N,) dollar holdings H_i
    unadj_open : (N,) unadjusted open prices P^O_i
    unadj_close: (N,) unadjusted close prices P^C_i

    Returns
    -------
    pnl_vec    : (N,) per-stock P&L
    shares_vec : (N,) per-stock shares traded
    """
    valid = (unadj_open > 0) & np.isfinite(unadj_open) & np.isfinite(unadj_close)
    with np.errstate(divide="ignore", invalid="ignore"):
        oc_ret = np.where(unadj_open > 0, unadj_close / unadj_open - 1.0, 0.0)
        inv_open = np.where(unadj_open > 0, 1.0 / unadj_open, 0.0)
    pnl_vec = np.where(valid, holdings * oc_ret, 0.0)
    shares_vec = np.where(valid & (holdings != 0), 2.0 * np.abs(holdings) * inv_open, 0.0)
    return pnl_vec, shares_vec


def aggregate_metrics(
    daily_pnl: np.ndarray,
    daily_shares: np.ndarray,
    investment_level: float = INVESTMENT_LEVEL,
) -> PerformanceMetrics:
    """
    Aggregate daily P&L vectors into performance metrics (Section 3.4.4).

    Parameters
    ----------
    daily_pnl    : (T,) total portfolio P&L per day
    daily_shares : (T,) total shares traded per day
    investment_level : I = gross portfolio value ($)

    Returns
    -------
    PerformanceMetrics
    """
    T = len(daily_pnl)
    active = np.abs(daily_pnl) > 0

    mean_pnl = daily_pnl.mean()
    std_pnl = daily_pnl.std(ddof=1) if T > 1 else 1.0

    roc_annual = 252.0 * mean_pnl / investment_level * 100.0   # percent
    sr_annual = (mean_pnl / std_pnl * np.sqrt(252.0)) if std_pnl > 0 else 0.0

    total_pnl = daily_pnl.sum()
    total_shares = daily_shares.sum()
    # Paper reports CPS in cents per share (Section 3.4.3): multiply $/share by 100
    cps = (total_pnl / total_shares * 100.0) if total_shares > 0 else 0.0

    return PerformanceMetrics(
        roc_annual=roc_annual,
        sr_annual=sr_annual,
        cps=cps,
        total_pnl=total_pnl,
        total_shares=total_shares,
        n_days=int(active.sum()),
    )


def metrics_to_dict(m: PerformanceMetrics) -> dict:
    """Convert PerformanceMetrics to plain dict for DataFrame rows."""
    return {
        "ROC (%)": round(m.roc_annual, 4),
        "SR": round(m.sr_annual, 4),
        "CPS (cents)": round(m.cps, 4),
        "Total PnL": round(m.total_pnl, 2),
        "Total Shares": round(m.total_shares, 0),
        "Days": m.n_days,
    }
