# Statistical Risk Models

**Paper:** Zura Kakushadze and Willie Yu (2016). "Statistical Risk Models." *SSRN Working Paper 2732453.*
**Link:** https://ssrn.com/abstract=2732453

## Overview

This repository reproduces four experiments from the above paper. The paper introduces
statistical risk models built from the principal components of sample return correlation
matrices, proposes two K-selection algorithms (minimization and effective rank), and
evaluates them in an intraday mean-reversion backtest on 2000 U.S. stocks.

## Experiments

| Experiment | Title | Paper Reference |
|-----------|-------|-----------------|
| exp_1 | Out-of-sample comparison of K-selection algorithms | Tables 1–3 |
| exp_2 | Power-iterations vs efficient PCA comparison | Table 4 |
| exp_3 | Shrinkage and uniform-correlation risk models | Tables 5–6 |
| exp_4 | Fixed-K and ad hoc specific-risk sensitivity | Tables 7–10, Figures 1–2 |

## Repository Structure

```
├── data/           # Cached data (Parquet, not committed)
├── exp/            # Experiment scripts (call src/ only)
│   ├── exp1_k_selection.py
│   ├── exp2_pca_methods.py
│   ├── exp3_shrinkage.py
│   └── exp4_sensitivity.py
├── results/        # Output tables, plots, RESULTS.md
├── src/            # Core implementation modules
│   ├── config.py       # Paper-derived constants (Section 3.4)
│   ├── data_loader.py  # Massive API data fetch and caching
│   ├── risk_model.py   # PCA, K-selection, covariance construction
│   ├── optimizer.py    # Sharpe-ratio maximization (unbounded + bounded)
│   ├── metrics.py      # ROC, SR, CPS computations
│   ├── pca_methods.py  # Power-iterations and efficient eigenpair methods
│   └── backtest.py     # Shared intraday backtest engine
├── tests/          # pytest test suite
├── reproduce.sh    # Single-command reproduction entry point
└── .env.example    # API key configuration template
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure credentials:
   ```bash
   cp .env.example .env
   # Edit .env and set MASSIVE_API_KEY=<your_key>
   ```

3. Run all experiments:
   ```bash
   bash reproduce.sh
   ```

## Data

- Source: Massive REST API (U.S. equities daily OHLCV)
- Universe: top 2000 U.S.-listed common stocks and class shares by 21-day ADDV (Section 3.4.2)
- Period: 2008-08-01 to 2014-09-05 (1260 backtest trading days, Section 3.4.2)
- Fields: unadjusted open/close/volume and fully adjusted open/close

Data is cached locally to `data/` as Parquet files after the first download.

## Results

See `results/RESULTS.md` for detailed comparison with the paper's reported figures.
