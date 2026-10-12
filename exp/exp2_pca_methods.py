"""
Exp 2: Computational comparison of PCA methods (Section 4.1–4.2, Table 4).
Replicates the iteration-count comparison between power iterations and the
efficient no-iterations eigenpairs method of Kakushadze & Yu (2016).

Expected: total power-iteration count ≈ 5000 for N=2339, M=19, prec=1e-3.
The efficient method recovers the same eigenpairs with O(M^2 N) cost.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import M, M_PLUS_1, POWER_ITER_PREC, POWER_ITER_RUNS
from src.data_loader import build_price_arrays, download_all_bars, fetch_universe_tickers
from src.pca_methods import (
    compare_eigenpairs,
    eigenpairs_efficient,
    eigenpairs_power_iterations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _get_sample_return_matrix(price_arrays: dict, n_stocks: int = 2339) -> np.ndarray:
    """
    Build one (N, M+1) return matrix representative of the paper's N=2339, M+1=20.
    Uses the most recent M+2 adj_close prices up to backtest end.
    """
    from src.config import BACKTEST_END_DATE
    adj_close = price_arrays["adj_close"]
    end = pd.Timestamp(BACKTEST_END_DATE)
    prior = adj_close.index[adj_close.index <= end]

    # Need M+2 = 21 price dates to form M+1 = 20 returns
    price_dates = prior[-(M_PLUS_1 + 1):]
    ac = adj_close.loc[price_dates]

    # Keep stocks with no missing values in this window
    valid = ac.notna().all(axis=0)
    ac_valid = ac.loc[:, valid]

    # Cap at n_stocks
    ac_valid = ac_valid.iloc[:, :n_stocks]

    prices = ac_valid.values   # (M+2, N)
    ret = np.log(prices[1:] / prices[:-1]).T   # (N, M+1)

    logger.info(
        "Return matrix for exp2: shape=%s, N=%d, M+1=%d",
        ret.shape, ret.shape[0], ret.shape[1],
    )
    return ret


def run_exp2():
    """Run power-iteration benchmark and efficient method comparison."""
    logger.info("=== Exp 2: PCA methods comparison ===")

    # Load data
    tickers = fetch_universe_tickers(cache=True)
    panel_unadj, panel_adj = download_all_bars(tickers, cache=True)
    price_arrays = build_price_arrays(panel_unadj, panel_adj)

    ret = _get_sample_return_matrix(price_arrays, n_stocks=2400)
    N, T = ret.shape
    logger.info("Using return matrix N=%d, M+1=%d", N, T)

    # ── Part A: Power iterations benchmark (Table 4) ──────────────────────────
    logger.info("--- Part A: Power iterations (10 runs) ---")
    all_iter_counts = []
    all_total_iters = []

    for run in range(POWER_ITER_RUNS):
        result_pow = eigenpairs_power_iterations(
            ret, calc_cor=True, k=M, prec=POWER_ITER_PREC, seed=run
        )
        total_iters = int(result_pow.n_iterations.sum())
        all_iter_counts.append(result_pow.n_iterations.copy())
        all_total_iters.append(total_iters)
        logger.info(
            "  Run %2d: total_iter=%d, per_component=%s",
            run + 1,
            total_iters,
            result_pow.n_iterations.tolist(),
        )

    iter_matrix = np.vstack(all_iter_counts)  # (10, M)
    mean_total = float(np.mean(all_total_iters))
    std_total = float(np.std(all_total_iters, ddof=1))

    logger.info(
        "Power iterations: mean total = %.1f ± %.1f (over %d runs)",
        mean_total, std_total, POWER_ITER_RUNS
    )
    logger.info("Paper reports ~5000 for N=2339, M=19.")

    # ── Part B: Efficient method ───────────────────────────────────────────────
    logger.info("--- Part B: Efficient eigenpairs ---")
    result_eff = eigenpairs_efficient(ret, calc_cor=True)
    logger.info(
        "Efficient method: elapsed=%.3f s, eigenvalues[:5]=%s",
        result_eff.elapsed_sec,
        np.round(result_eff.values[:5], 4).tolist(),
    )

    # ── Part C: Verify agreement between methods ───────────────────────────────
    logger.info("--- Part C: Agreement verification (run 0) ---")
    result_pow0 = eigenpairs_power_iterations(
        ret, calc_cor=True, k=M, prec=POWER_ITER_PREC, seed=0
    )
    cmp = compare_eigenpairs(result_eff, result_pow0, atol=5e-2)
    logger.info("Comparison: %s", cmp)

    # Also compare against numpy.linalg.eigh directly
    from src.risk_model import _sample_cov_eigh
    eigvals_ref, eigvecs_ref, _ = _sample_cov_eigh(ret, use_cor=True)
    from src.pca_methods import EigenpairsResult
    ref = EigenpairsResult(values=eigvals_ref[:M], vectors=eigvecs_ref[:, :M])
    cmp_ref = compare_eigenpairs(result_eff, ref, atol=1e-6)
    logger.info("Efficient vs numpy.eigh: %s", cmp_ref)

    # ── Save results ──────────────────────────────────────────────────────────
    iter_df = pd.DataFrame(
        iter_matrix,
        index=[f"Run {i+1}" for i in range(POWER_ITER_RUNS)],
        columns=[f"PC {j+1}" for j in range(M)],
    )
    iter_df["Total"] = iter_df.sum(axis=1)
    iter_df.to_csv(RESULTS_DIR / "exp2_iteration_counts.csv")

    summary = {
        "N": N,
        "M": M,
        "Mean total iterations": round(mean_total, 1),
        "Std total iterations": round(std_total, 1),
        "Paper value (approx)": 5000,
        "Efficient method elapsed (s)": round(result_eff.elapsed_sec, 4),
        "Max eigenvalue diff (eff vs numpy)": round(cmp_ref["max_eigenvalue_diff"], 8),
        "Max eigenvector diff (eff vs numpy)": round(cmp_ref["max_eigenvector_diff"], 8),
        "Methods agree": cmp_ref["pass"],
    }
    pd.Series(summary).to_csv(RESULTS_DIR / "exp2_summary.csv")
    logger.info("Results saved to %s", RESULTS_DIR)
    logger.info("Summary: %s", summary)

    return iter_df, summary


if __name__ == "__main__":
    run_exp2()
