"""
Paper-derived configuration constants for Statistical Risk Models.
Kakushadze & Yu (2016), SSRN:2732453.
All values taken directly from Section 3.4 of the paper.
"""

# --- Data parameters (Section 3.4.2) ---
BACKTEST_END_DATE = "2014-09-05"    # last trading day of backtest (inclusive)
BACKTEST_START_DATE = "2008-08-01"  # earliest price data needed
BACKTEST_DAYS = 1260                # trading days in backtest (Section 3.4.2)
UNIVERSE_SIZE = 2000                # top-N stocks by ADDV (Section 3.4.2)
REBAL_PERIOD = 21                   # rebalancing interval in trading days
ADDV_DAYS = 21                      # ADDV lookback window (same as rebal period)

# --- Risk model parameters (Section 3.4.3) ---
M_PLUS_1 = 20                       # number of return observations per estimation
M = M_PLUS_1 - 1                    # = 19, number of factors available
USE_CORRELATION = True              # use.cor=T in R code (paper Sections 2.4, A)

# --- Backtest parameters (Section 3.4.4) ---
INVESTMENT_LEVEL = 20_000_000       # gross portfolio value I = $20M
BOUNDS_MULTIPLIER = 0.01            # |H_is| <= 0.01 * ADDV_is (Section 3.4.4)

# --- K-selection: eRank (Section 3.2) ---
ERANK_ROUND = "round"               # K = round(eRank(Psi))
ERANK_FLOOR = "floor"               # K = floor(eRank(Psi))

# --- Minimization algorithm (Section 3.1 / Appendix A) ---
MIN_ALGO_PREC = 1e-10               # numerical tolerance for g-value comparison

# --- Power iterations (Section 4.1 / Appendix B) ---
POWER_ITER_PREC = 1e-3              # convergence precision prec (Appendix B)
POWER_ITER_RUNS = 10                # number of independent runs for Table 4

# --- Shrinkage (Section 4.4) ---
SHRINKAGE_Q_VALUES = [1e-6, 0.3, 0.6, 0.9]
SHRINKAGE_RHO_VALUES = [0.1, 0.5, 0.9]

# --- Data fetch range (extra buffer for ADDV lookback) ---
FETCH_START_DATE = "2008-05-01"     # earliest date to fetch (buffer for ADDV lookback)
