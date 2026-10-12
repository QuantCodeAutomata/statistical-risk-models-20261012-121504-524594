"""
Exp 1: Out-of-sample comparison of K-selection algorithms.
Replicates Tables 1, 2, 3 of Kakushadze & Yu (2016).

K-selection methods tested:
  (a) minimization ('min'),  (b) eRank-Round,  (c) eRank-Floor
  (d) K'+1 variation with first PC excluded (applied to each of a–c)
Both unbounded and bounded (|H| <= 0.01 * ADDV) backtests.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest
from src.config import USE_CORRELATION
from src.data_loader import build_price_arrays, download_all_bars, fetch_universe_tickers
from src.metrics import metrics_to_dict
from src.risk_model import RiskModelResult, fit_risk_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_model_fn(method: str, excl_first: bool = False):
    """Return a model function for the specified K-selection method."""
    def _fn(ret: np.ndarray) -> RiskModelResult:
        return fit_risk_model(ret, method=method, excl_first=excl_first,
                              use_cor=USE_CORRELATION)
    _fn.__name__ = f"model_{method}_excl{excl_first}"
    return _fn


def run_exp1():
    """Run all K-selection variants and report Tables 1–3."""
    logger.info("=== Exp 1: K-selection algorithm comparison ===")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Fetching universe tickers …")
    tickers = fetch_universe_tickers(cache=True)
    logger.info("Universe candidates: %d tickers", len(tickers))

    logger.info("Downloading/loading price bars …")
    panel_unadj, panel_adj = download_all_bars(tickers, cache=True)

    price_arrays = build_price_arrays(panel_unadj, panel_adj)
    logger.info(
        "Price arrays ready: %d dates, %d tickers.",
        price_arrays["unadj_close"].shape[0],
        price_arrays["unadj_close"].shape[1],
    )

    # ── Define model variants ─────────────────────────────────────────────────
    variants = [
        ("Minimization",            make_model_fn("min",          excl_first=False)),
        ("Minimization K'+1",       make_model_fn("min",          excl_first=True)),
        ("eRank Round",             make_model_fn("erank_round",  excl_first=False)),
        ("eRank Round K'+1",        make_model_fn("erank_round",  excl_first=True)),
        ("eRank Floor",             make_model_fn("erank_floor",  excl_first=False)),
        ("eRank Floor K'+1",        make_model_fn("erank_floor",  excl_first=True)),
    ]

    # ── Table 1: Unbounded backtest ───────────────────────────────────────────
    logger.info("--- Unbounded backtests (Table 1) ---")
    table1_rows = []
    table3_rows = []

    for name, model_fn in variants:
        logger.info("Running unbounded: %s", name)
        metrics, info = run_backtest(
            price_arrays=price_arrays,
            model_fn=model_fn,
            bounded=False,
            verbose=True,
        )
        row = {"Model": name, **metrics_to_dict(metrics)}
        table1_rows.append(row)
        table3_rows.append({
            "Model": name,
            "Mean K": round(info["mean_k"], 2),
            "Mean eRank": round(info["mean_erank"], 2),
            "Mean Avg Corr": round(info["mean_avg_corr"], 4),
            "Bounded": False,
        })
        logger.info("  %s | unbounded: %s", name, metrics)

    table1 = pd.DataFrame(table1_rows).set_index("Model")
    logger.info("\nTable 1 (Unbounded):\n%s", table1.to_string())

    # ── Table 2: Bounded backtest ─────────────────────────────────────────────
    logger.info("--- Bounded backtests (Table 2) ---")
    table2_rows = []

    for name, model_fn in variants:
        logger.info("Running bounded: %s", name)
        metrics, info = run_backtest(
            price_arrays=price_arrays,
            model_fn=model_fn,
            bounded=True,
            verbose=True,
        )
        row = {"Model": name, **metrics_to_dict(metrics)}
        table2_rows.append(row)
        table3_rows.append({
            "Model": name,
            "Mean K": round(info["mean_k"], 2),
            "Mean eRank": round(info["mean_erank"], 2),
            "Mean Avg Corr": round(info["mean_avg_corr"], 4),
            "Bounded": True,
        })
        logger.info("  %s | bounded: %s", name, metrics)

    table2 = pd.DataFrame(table2_rows).set_index("Model")
    logger.info("\nTable 2 (Bounded):\n%s", table2.to_string())

    table3 = pd.DataFrame(table3_rows)
    logger.info("\nTable 3 (K and avg corr summary):\n%s", table3.to_string())

    # ── Save results ──────────────────────────────────────────────────────────
    table1.to_csv(RESULTS_DIR / "exp1_table1_unbounded.csv")
    table2.to_csv(RESULTS_DIR / "exp1_table2_bounded.csv")
    table3.to_csv(RESULTS_DIR / "exp1_table3_summary.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR)

    return table1, table2, table3


if __name__ == "__main__":
    run_exp1()
