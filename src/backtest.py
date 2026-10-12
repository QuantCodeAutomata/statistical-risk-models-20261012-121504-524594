"""
Shared backtest engine for intraday mean-reversion experiments.
Implements the common pipeline in Section 3.4 of Kakushadze & Yu (2016).

Calling convention:
    results = run_backtest(price_arrays, model_fn, bounded=False)

where model_fn(ret_window) → RiskModelResult.

Expected return (Section 3.4.4):
    E_is = ln(P^AO_is / P^AC_{i,s-1})   [overnight: prev adj close → today adj open]

Realized P&L (Section 3.4.4):
    Pi_is = H_is * (P^C_is / P^O_is - 1)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import (
    ADDV_DAYS,
    BACKTEST_DAYS,
    BACKTEST_END_DATE,
    BOUNDS_MULTIPLIER,
    INVESTMENT_LEVEL,
    M,
    M_PLUS_1,
    REBAL_PERIOD,
    UNIVERSE_SIZE,
    USE_CORRELATION,
)
from src.data_loader import compute_addv, get_trading_dates
from src.metrics import PerformanceMetrics, aggregate_metrics, compute_daily_pnl
from src.optimizer import compute_holdings
from src.risk_model import RiskModelResult

logger = logging.getLogger(__name__)

ModelFn = Callable[[np.ndarray], RiskModelResult]


def _select_universe(addv_row: pd.Series, size: int = UNIVERSE_SIZE) -> list[str]:
    """
    Select the top `size` stocks by ADDV for a given rebalancing date.
    Returns list of ticker strings with valid (finite, positive) ADDV.
    """
    valid = addv_row.dropna()
    valid = valid[valid > 0]
    if len(valid) == 0:
        return []
    return valid.nlargest(size).index.tolist()


def run_backtest(
    price_arrays: dict[str, pd.DataFrame],
    model_fn: ModelFn,
    bounded: bool = False,
    backtest_end: str = BACKTEST_END_DATE,
    n_days: int = BACKTEST_DAYS,
    rebal_period: int = REBAL_PERIOD,
    universe_size: int = UNIVERSE_SIZE,
    m_obs: int = M,
    investment_level: float = INVESTMENT_LEVEL,
    bounds_multiplier: float = BOUNDS_MULTIPLIER,
    verbose: bool = True,
) -> tuple[PerformanceMetrics, dict]:
    """
    Run the full intraday mean-reversion backtest.

    Parameters
    ----------
    price_arrays   : dict with keys unadj_open, unadj_close, unadj_vol,
                     adj_open, adj_close (date×ticker DataFrames)
    model_fn       : callable(ret_matrix: np.ndarray) → RiskModelResult
                     where ret_matrix is (N_universe, M+1) of close-to-close log returns
    bounded        : whether to apply position bounds |H_i| <= BOUNDS_MULTIPLIER * ADDV_i
    backtest_end   : last backtest date (inclusive)
    n_days         : number of trading days in the backtest
    rebal_period   : rebalancing frequency in trading days
    universe_size  : maximum number of stocks in the universe
    m_obs          : M (number of close-to-close return observations = M+1 - 1 = 19)
    investment_level : gross portfolio value I
    bounds_multiplier : factor applied to ADDV for position bounds

    Returns
    -------
    metrics : PerformanceMetrics over full backtest
    info    : dict with daily P&L series, K history, avg eRank, avg correlation
    """
    unadj_open = price_arrays["unadj_open"]
    unadj_close = price_arrays["unadj_close"]
    unadj_vol = price_arrays["unadj_vol"]
    adj_open = price_arrays["adj_open"]
    adj_close = price_arrays["adj_close"]

    # Select backtest dates
    trading_dates = get_trading_dates(price_arrays, backtest_end, n_days)
    all_dates = adj_close.index

    # Pre-compute ADDV for universe selection
    addv_df = compute_addv(unadj_close, unadj_vol, window=ADDV_DAYS)

    # ── Pre-convert DataFrames to numpy for O(1) per-day access ──────────────
    # This avoids pandas .loc overhead (≈20ms/day) in the inner trading loop.
    all_tickers_list = adj_close.columns.tolist()
    all_dates_list = all_dates.tolist()
    ticker_to_col = {t: i for i, t in enumerate(all_tickers_list)}
    date_to_row: dict = {d: i for i, d in enumerate(all_dates_list)}

    ao_np = adj_open.values.astype(float)     # (T_all, N_all)
    ac_np = adj_close.values.astype(float)
    uo_np = unadj_open.values.astype(float)
    uc_np = unadj_close.values.astype(float)

    addv_tickers_list = addv_df.columns.tolist()
    addv_dates_list = addv_df.index.tolist()
    addv_ticker_to_col = {t: i for i, t in enumerate(addv_tickers_list)}
    addv_date_to_row: dict = {d: i for i, d in enumerate(addv_dates_list)}
    addv_np = addv_df.values.astype(float)

    # Pre-compute the previous-date row index in all_dates for every trading day
    prev_row_cache: list = []
    for d in trading_dates:
        d_ts = pd.Timestamp(d)
        ridx = int(all_dates.searchsorted(d_ts, side="left")) - 1
        prev_row_cache.append(ridx if ridx >= 0 else -1)

    T = len(trading_dates)
    daily_pnl = np.zeros(T)
    daily_shares = np.zeros(T)
    k_history: list = []
    erank_history: list = []
    avg_corr_history: list = []

    # State: rebalanced every rebal_period trading days
    current_universe_cols: Optional[np.ndarray] = None   # integer columns into all_tickers
    current_universe_tickers: list[str] = []              # ticker strings (same order)
    current_model: Optional[RiskModelResult] = None
    current_bounds_arr: Optional[np.ndarray] = None
    next_rebal_idx = 0

    for t, date in enumerate(trading_dates):
        date_ts = pd.Timestamp(date)

        # ── Rebalance? ────────────────────────────────────────────────────────
        if t >= next_rebal_idx:
            # 1. Universe: top stocks by ADDV using the most recent available ADDV date
            if date_ts in addv_date_to_row:
                a_ridx = addv_date_to_row[date_ts]
            else:
                a_ridx = int(addv_df.index.searchsorted(date_ts, side="left")) - 1
                if a_ridx < 0:
                    next_rebal_idx = t + rebal_period
                    continue

            addv_vals_row = addv_np[a_ridx, :]               # (N_addv,)
            valid_mask = np.isfinite(addv_vals_row) & (addv_vals_row > 0)
            if valid_mask.sum() < 10:
                next_rebal_idx = t + rebal_period
                continue

            valid_idx = np.where(valid_mask)[0]
            top_local = valid_idx[np.argsort(addv_vals_row[valid_idx])[::-1][:universe_size]]
            rebal_tickers = [addv_tickers_list[i] for i in top_local]

            # 2. M+2 price window ending just before current date (gives M+1 returns)
            d_row = date_to_row.get(date_ts)
            if d_row is None or d_row < m_obs + 2:
                next_rebal_idx = t + rebal_period
                continue

            row_start = d_row - m_obs - 2   # inclusive; M+2 rows: [row_start, d_row)
            # Filter rebal_tickers to those present in adj_close
            rc_idx = [ticker_to_col[tk] for tk in rebal_tickers if tk in ticker_to_col]
            rc_tickers = [tk for tk in rebal_tickers if tk in ticker_to_col]
            if not rc_idx:
                next_rebal_idx = t + rebal_period
                continue

            ac_win = ac_np[row_start:d_row, rc_idx]     # (M+2, N_rebal)
            no_nan = np.isfinite(ac_win).all(axis=0)
            ok = np.where(no_nan)[0]
            if len(ok) < 10:
                next_rebal_idx = t + rebal_period
                continue

            final_global_cols = [rc_idx[i] for i in ok]
            current_universe_tickers = [rc_tickers[i] for i in ok]
            current_universe_cols = np.array(final_global_cols, dtype=np.intp)

            ac_prices = ac_win[:, ok]                    # (M+2, N_valid)
            log_ret = np.log(ac_prices[1:] / ac_prices[:-1])  # (M+1, N_valid)
            ret_matrix = log_ret.T                       # (N_valid, M+1)

            # 3. Fit risk model
            try:
                current_model = model_fn(ret_matrix)
            except Exception as exc:
                logger.warning("Risk model failed at %s: %s", date, exc)
                current_model = None
                next_rebal_idx = t + rebal_period
                continue

            # 4. ADDV bounds (bounded case)
            if bounded:
                b_arr = np.array(
                    [addv_vals_row[addv_ticker_to_col[tk]] if tk in addv_ticker_to_col else 0.0
                     for tk in current_universe_tickers], dtype=float
                )
                current_bounds_arr = np.where(np.isfinite(b_arr), bounds_multiplier * b_arr, 0.0)
            else:
                current_bounds_arr = None

            # Diagnostics
            if current_model is not None:
                k_history.append(current_model.n_factors)
                if current_model.erank is not None:
                    erank_history.append(current_model.erank)
                n_u = len(current_universe_tickers)
                corr = np.corrcoef(ret_matrix)
                avg_c = (corr.sum() - n_u) / (n_u * (n_u - 1)) if n_u > 1 else 0.0
                avg_corr_history.append(float(avg_c))

            next_rebal_idx = t + rebal_period

        # ── Daily optimization and P&L ────────────────────────────────────────
        if current_model is None or current_universe_cols is None:
            continue

        d_row = date_to_row.get(date_ts)
        prev_row = prev_row_cache[t]
        if d_row is None or prev_row < 0:
            continue

        univ_cols = current_universe_cols          # (N_univ,) int array
        ao_vals = ao_np[d_row, univ_cols]          # adj open today
        ac_vals = ac_np[prev_row, univ_cols]       # adj close yesterday

        valid = np.isfinite(ao_vals) & np.isfinite(ac_vals) & (ao_vals > 0) & (ac_vals > 0)
        if valid.sum() < 5:
            continue

        v_idx = np.where(valid)[0]                 # local indices into universe

        sub_spec = current_model.spec_risk[v_idx]
        sub_fac = current_model.fac_load[v_idx, :]
        from src.risk_model import RiskModelResult as RMR
        sub_model = RMR(spec_risk=sub_spec, fac_load=sub_fac,
                        n_factors=current_model.n_factors,
                        sigma=current_model.sigma[v_idx])

        # Overnight expected return (Eq. 21); sign-flip for mean-reversion (Eq. 27)
        E = -np.log(ao_vals[v_idx] / ac_vals[v_idx])

        bounds_today = current_bounds_arr[v_idx] if (bounded and current_bounds_arr is not None) else None
        H = compute_holdings(E, sub_model, bounds_today, investment_level)

        # Realized P&L using unadjusted prices (Section 3.4.4)
        global_v = univ_cols[v_idx]
        uo_vals = uo_np[d_row, global_v]
        uc_vals = uc_np[d_row, global_v]
        ok_oc = np.isfinite(uo_vals) & np.isfinite(uc_vals) & (uo_vals > 0)
        if ok_oc.sum() == 0:
            continue

        pnl_vec, shares_vec = compute_daily_pnl(H[ok_oc], uo_vals[ok_oc], uc_vals[ok_oc])
        daily_pnl[t] = pnl_vec.sum()
        daily_shares[t] = shares_vec.sum()

    metrics = aggregate_metrics(daily_pnl, daily_shares, investment_level)
    info = {
        "daily_pnl": daily_pnl,
        "daily_shares": daily_shares,
        "k_history": k_history,
        "erank_history": erank_history,
        "avg_corr_history": avg_corr_history,
        "mean_k": float(np.mean(k_history)) if k_history else np.nan,
        "mean_erank": float(np.mean(erank_history)) if erank_history else np.nan,
        "mean_avg_corr": float(np.mean(avg_corr_history)) if avg_corr_history else np.nan,
    }

    if verbose:
        logger.info(
            "Backtest complete: %s | K=%.1f | eRank=%.2f | avg_corr=%.3f",
            metrics,
            info["mean_k"],
            info["mean_erank"],
            info["mean_avg_corr"],
        )

    return metrics, info
