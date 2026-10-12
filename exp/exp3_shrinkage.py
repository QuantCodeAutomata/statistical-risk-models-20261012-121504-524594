"""
Exp 3: Backtest of nontraditional shrinkage-based statistical risk models (Section 4.4–4.5).
Replicates Tables 5 and 6 of Kakushadze & Yu (2016).

Tests:
  - Diagonal-target shrinkage: q in {1e-6, 0.3, 0.6, 0.9}
  - Uniform-correlation target: q in {1e-6, 0.3, 0.6, 0.9}, rho in {0.1, 0.5, 0.9}
All with position bounds |H_i| <= 0.01 * ADDV_i.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest
from src.config import SHRINKAGE_Q_VALUES, SHRINKAGE_RHO_VALUES, USE_CORRELATION
from src.data_loader import build_price_arrays, download_all_bars, fetch_universe_tickers
from src.metrics import metrics_to_dict
from src.risk_model import RiskModelResult, build_shrinkage_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_shrinkage_fn(q: float, rho: Optional[float] = None):
    """Return a model function for the specified shrinkage parameters."""
    def _fn(ret: np.ndarray) -> RiskModelResult:
        return build_shrinkage_model(ret, q=q, rho=rho, use_cor=USE_CORRELATION)
    _fn.__name__ = f"shrinkage_q{q}_rho{rho}"
    return _fn


def run_exp3():
    """Run shrinkage model backtests and report Tables 5–6."""
    logger.info("=== Exp 3: Shrinkage risk models backtest ===")

    # ── Load shared data ───────────────────────────────────────────────────────
    logger.info("Loading data …")
    tickers = fetch_universe_tickers(cache=True)
    panel_unadj, panel_adj = download_all_bars(tickers, cache=True)
    price_arrays = build_price_arrays(panel_unadj, panel_adj)

    # ── Table 5: Diagonal-target shrinkage ────────────────────────────────────
    logger.info("--- Diagonal-target shrinkage (Table 5) ---")
    table5_rows = []

    for q in SHRINKAGE_Q_VALUES:
        label = f"q={q}"
        logger.info("Running bounded: %s", label)
        metrics, info = run_backtest(
            price_arrays=price_arrays,
            model_fn=make_shrinkage_fn(q, rho=None),
            bounded=True,
            verbose=True,
        )
        row = {"q": q, "rho": "—", "Model": "Diagonal shrinkage", **metrics_to_dict(metrics)}
        table5_rows.append(row)
        logger.info("  q=%s: %s", q, metrics)

    table5 = pd.DataFrame(table5_rows)
    logger.info("\nTable 5 (Diagonal shrinkage):\n%s", table5.to_string())

    # ── Table 6: Uniform-correlation shrinkage ────────────────────────────────
    logger.info("--- Uniform-correlation shrinkage (Table 6) ---")
    table6_rows = []

    for q in SHRINKAGE_Q_VALUES:
        for rho in SHRINKAGE_RHO_VALUES:
            label = f"q={q}, rho={rho}"
            logger.info("Running bounded: %s", label)
            metrics, info = run_backtest(
                price_arrays=price_arrays,
                model_fn=make_shrinkage_fn(q, rho=rho),
                bounded=True,
                verbose=True,
            )
            row = {"q": q, "rho": rho, "Model": "Uniform-corr shrinkage", **metrics_to_dict(metrics)}
            table6_rows.append(row)
            logger.info("  q=%s, rho=%s: %s", q, rho, metrics)

    table6 = pd.DataFrame(table6_rows)
    logger.info("\nTable 6 (Uniform-correlation shrinkage):\n%s", table6.to_string())

    # ── Save results ──────────────────────────────────────────────────────────
    table5.to_csv(RESULTS_DIR / "exp3_table5_diagonal_shrinkage.csv", index=False)
    table6.to_csv(RESULTS_DIR / "exp3_table6_uniform_shrinkage.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR)

    return table5, table6


if __name__ == "__main__":
    run_exp3()
