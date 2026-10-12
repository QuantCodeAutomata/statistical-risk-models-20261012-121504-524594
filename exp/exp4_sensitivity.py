"""
Exp 4: Sensitivity analysis — fixed K, inherited specific risk, and xi^2=1 (Section 5).
Replicates Tables 7–10 and Figures 1–2 of Kakushadze & Yu (2016).

Experiment A: Fixed K in {1..18}, unbounded and bounded.
Experiment B: K1=M factors with specific risks from lower-K model (bounded).
Experiment C: Varying K with xi_i^2 = 1 (bounded).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest
from src.config import M, USE_CORRELATION
from src.data_loader import build_price_arrays, download_all_bars, fetch_universe_tickers
from src.metrics import metrics_to_dict
from src.risk_model import RiskModelResult, _sample_cov_eigh, fit_risk_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_fixed_k_fn(k: int):
    """Model function with fixed K principal components."""
    def _fn(ret: np.ndarray) -> RiskModelResult:
        return fit_risk_model(ret, method="min", fixed_k=k, use_cor=USE_CORRELATION)
    _fn.__name__ = f"fixed_k_{k}"
    return _fn


def make_adhoc_b_fn(k_spec: int):
    """
    Exp B: Full M factors with specific risk inherited from a k_spec-factor model.
    Uses K1=M factor loadings but sets spec_risk from the K=k_spec model.
    """
    def _fn(ret: np.ndarray) -> RiskModelResult:
        # Step 1: fit K=k_spec model to get specific risk
        model_k = fit_risk_model(ret, method="min", fixed_k=k_spec, use_cor=USE_CORRELATION)
        # Step 2: full M-factor loading matrix
        eigvals, eigvecs, sigma = _sample_cov_eigh(ret, use_cor=USE_CORRELATION)
        full_fac_load_cor = eigvecs[:, :M] * np.sqrt(eigvals[:M])  # (N, M)
        # Use specific risk from k_spec model (in correlation space: spec_sq = model_k.spec_risk^2 / sigma^2)
        spec_risk_cor_sq = (model_k.spec_risk / sigma) ** 2
        spec_risk_out = sigma * np.sqrt(np.maximum(spec_risk_cor_sq, 0.0))
        fac_load_out = sigma[:, np.newaxis] * full_fac_load_cor
        return RiskModelResult(
            spec_risk=spec_risk_out,
            fac_load=fac_load_out,
            n_factors=M,
            sigma=sigma,
            erank=None,
        )
    _fn.__name__ = f"adhoc_b_kspec{k_spec}"
    return _fn


def make_adhoc_c_fn(k: int):
    """
    Exp C: K principal components with specific risk xi_i^2 = 1 in correlation space.
    (ad hoc: set in-sample total specific variance = 1 for all stocks)
    """
    def _fn(ret: np.ndarray) -> RiskModelResult:
        eigvals, eigvecs, sigma = _sample_cov_eigh(ret, use_cor=USE_CORRELATION)
        K_use = max(1, min(k, M))
        fac_load_cor = eigvecs[:, :K_use] * np.sqrt(eigvals[:K_use])
        # xi_i^2 = 1 in correlation space (Section 5 of the paper)
        spec_sq_override = np.ones(ret.shape[0])
        spec_risk_out = sigma * np.sqrt(spec_sq_override)
        fac_load_out = sigma[:, np.newaxis] * fac_load_cor
        return RiskModelResult(
            spec_risk=spec_risk_out,
            fac_load=fac_load_out,
            n_factors=K_use,
            sigma=sigma,
            erank=None,
        )
    _fn.__name__ = f"adhoc_c_k{k}"
    return _fn


def run_exp4():
    """Run all sensitivity experiments and report Tables 7–10 and Figures 1–2."""
    logger.info("=== Exp 4: Sensitivity analysis (fixed K and ad hoc specific risk) ===")

    # ── Load shared data ───────────────────────────────────────────────────────
    logger.info("Loading data …")
    tickers = fetch_universe_tickers(cache=True)
    panel_unadj, panel_adj = download_all_bars(tickers, cache=True)
    price_arrays = build_price_arrays(panel_unadj, panel_adj)

    # ── Exp A: Fixed-K, unbounded (Table 7) ──────────────────────────────────
    logger.info("--- Exp A: Fixed-K unbounded (Table 7) ---")
    table7_rows = []
    k_range = list(range(1, M))   # K = 1 .. 18

    for k in k_range:
        logger.info("Fixed K=%d unbounded", k)
        metrics, info = run_backtest(
            price_arrays=price_arrays,
            model_fn=make_fixed_k_fn(k),
            bounded=False,
            verbose=True,
        )
        row = {"K": k, **metrics_to_dict(metrics)}
        table7_rows.append(row)
        logger.info("  K=%d unbounded: %s", k, metrics)

    table7 = pd.DataFrame(table7_rows).set_index("K")
    logger.info("\nTable 7 (Fixed-K unbounded):\n%s", table7.to_string())

    # ── Exp A: Fixed-K, bounded (Table 8) ────────────────────────────────────
    logger.info("--- Exp A: Fixed-K bounded (Table 8) ---")
    table8_rows = []

    for k in k_range:
        logger.info("Fixed K=%d bounded", k)
        metrics, info = run_backtest(
            price_arrays=price_arrays,
            model_fn=make_fixed_k_fn(k),
            bounded=True,
            verbose=True,
        )
        row = {"K": k, **metrics_to_dict(metrics)}
        table8_rows.append(row)
        logger.info("  K=%d bounded: %s", k, metrics)

    table8 = pd.DataFrame(table8_rows).set_index("K")
    logger.info("\nTable 8 (Fixed-K bounded):\n%s", table8.to_string())

    # ── Exp B: K1=M with inherited specific risk (Table 9) ───────────────────
    logger.info("--- Exp B: K1=M with inherited specific risk (Table 9) ---")
    table9_rows = []

    for k_spec in k_range:
        logger.info("Adhoc B k_spec=%d bounded", k_spec)
        metrics, info = run_backtest(
            price_arrays=price_arrays,
            model_fn=make_adhoc_b_fn(k_spec),
            bounded=True,
            verbose=True,
        )
        row = {"K_spec": k_spec, **metrics_to_dict(metrics)}
        table9_rows.append(row)
        logger.info("  k_spec=%d bounded: %s", k_spec, metrics)

    table9 = pd.DataFrame(table9_rows).set_index("K_spec")
    logger.info("\nTable 9 (K1=M with inherited spec risk):\n%s", table9.to_string())

    # ── Exp C: Varying K with xi^2=1 (Table 10) ──────────────────────────────
    logger.info("--- Exp C: xi_i^2=1 varying K (Table 10) ---")
    table10_rows = []
    k_range_c = list(range(1, M + 2))  # K = 1 .. M+1 = 20

    for k in k_range_c:
        logger.info("Adhoc C K=%d xi^2=1 bounded", k)
        metrics, info = run_backtest(
            price_arrays=price_arrays,
            model_fn=make_adhoc_c_fn(k),
            bounded=True,
            verbose=True,
        )
        row = {"K": k, **metrics_to_dict(metrics)}
        table10_rows.append(row)
        logger.info("  K=%d xi^2=1: %s", k, metrics)

    table10 = pd.DataFrame(table10_rows).set_index("K")
    logger.info("\nTable 10 (xi^2=1, varying K):\n%s", table10.to_string())

    # ── Figures 1–2: SR vs K ──────────────────────────────────────────────────
    _plot_sr_vs_k(table7, table8, "Fixed-K Performance", "exp4_fig1_fixed_k.png")
    _plot_sr_vs_k(table9.rename_axis("K"), table10,
                  "Ad hoc Specific Risk Performance", "exp4_fig2_adhoc.png",
                  labels=("K1=M, spec from K-model", "xi^2=1"))

    # ── Save tables ────────────────────────────────────────────────────────────
    table7.to_csv(RESULTS_DIR / "exp4_table7_fixed_k_unbounded.csv")
    table8.to_csv(RESULTS_DIR / "exp4_table8_fixed_k_bounded.csv")
    table9.to_csv(RESULTS_DIR / "exp4_table9_adhoc_b.csv")
    table10.to_csv(RESULTS_DIR / "exp4_table10_adhoc_c.csv")
    logger.info("Results saved to %s", RESULTS_DIR)

    return table7, table8, table9, table10


def _plot_sr_vs_k(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    title: str,
    filename: str,
    labels: tuple[str, str] = ("Unbounded", "Bounded"),
) -> None:
    """Plot Sharpe ratio vs K for two configurations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if "SR" in df1.columns:
        ax.plot(df1.index, df1["SR"], marker="o", label=labels[0])
    if "SR" in df2.columns:
        ax.plot(df2.index, df2["SR"], marker="s", label=labels[1])
    ax.set_xlabel("K (number of factors)")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()
    logger.info("Saved %s", filename)


if __name__ == "__main__":
    run_exp4()
