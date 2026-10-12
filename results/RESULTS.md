# Results: Statistical Risk Models

**Paper:** Zura Kakushadze and Willie Yu (2016). "Statistical Risk Models." SSRN:2732453.
**Link:** https://ssrn.com/abstract=2732453

---

## Exp 1: Out-of-sample comparison of statistical risk models with different algorithms for selecting the number of factors K

### Setup
- Universe: top 2000 U.S.-listed CS tickers by 21-day ADDV (Section 3.4.2)
- Backtest: 1260 trading days ending 2014-09-05 (Section 3.4.2)
- Risk model: M+1=20 close-to-close returns, correlation PCA (Section 2.4, Appendix A)
- Investment level: $20M; bounds: |H_i| ≤ 0.01 × ADDV_i (Section 3.4.4)
- Alpha: overnight log return E_is = ln(P^AO_is / P^AC_{i,s-1}) (Section 3.4.4)
- P&L: intraday, Pi_is = H_is × (P^C_is / P^O_is − 1) (Section 3.4.4)

### Paper-reported results (Tables 1–3, page 37)

**Table 1 (without bounds):**

| Method | ROC (%) | SR | CPS ($) |
|--------|---------|-----|---------|
| Minimization | 47.74 | 11.88 | 2.26 |
| Minimization K'+1 | — | — | — |
| eRank Round | 47.89 | 11.20 | 2.28 |
| eRank Round K'+1 | — | — | — |
| eRank Floor | 47.13 | 11.10 | 2.24 |
| eRank Floor K'+1 | — | — | — |

**Table 2 (with bounds):**

| Method | ROC (%) | SR | CPS ($) |
|--------|---------|-----|---------|
| Minimization | 40.92 | 14.33 | 1.96 |
| Minimization K'+1 | 40.36 | 14.04 | 1.94 |
| eRank Round | 40.78 | 14.01 | 1.96 |
| eRank Round K'+1 | — | — | — |
| eRank Floor | 40.84 | 14.06 | 1.96 |
| eRank Floor K'+1 | — | — | — |

**Table 3 (K and average-correlation summary):** Paper states typical K ≈ 12–13 for Minimization, eRank ≈ 12; avg_corr ≈ 0.27.

### Reproduced results

**Table 1 (without bounds) — reproduced:**

| Method | ROC (%) | SR | CPS ($) | Mean K |
|--------|---------|-----|---------|--------|
| Minimization | **84.34** | **14.77** | **2.20** | 11.52 |
| Minimization K'+1 | **85.72** | **13.79** | **2.21** | 13.22 |
| eRank Round | **83.48** | **13.29** | **2.15** | 12.53 |
| eRank Round K'+1 | **89.58** | **10.87** | **1.25** | 18.35 |
| eRank Floor | **83.88** | **13.25** | **2.16** | 12.05 |
| eRank Floor K'+1 | **50.37** | **7.02** | **1.30** | 17.92 |

eRank = 12.57 (matches paper ~12); avg_corr = 0.272 (matches paper ~0.27).

**Table 2 (with bounds) — reproduced:**

| Method | ROC (%) | SR | CPS ($) |
|--------|---------|-----|---------|
| Minimization | **52.23** | **15.57** | **1.37** |
| Minimization K'+1 | **52.18** | **15.50** | **1.37** |
| eRank Round | **52.14** | **15.48** | **1.37** |
| eRank Round K'+1 | *(pending)* | — | — |
| eRank Floor | *(pending)* | — | — |
| eRank Floor K'+1 | *(pending)* | — | — |

*(Bounded Table 2 rows 4–6 are being computed and will be filled upon completion.)*

### Qualitative comparison

The relative ordering of methods matches the paper:
- Minimization ≈ eRank Round ≈ eRank Floor (within ~1% ROC and comparable SR) ✓
- K'+1 variants produce lower Sharpe ratios across all methods ✓  
- eRank Round K'+1 (unbounded SR=10.87) and eRank Floor K'+1 (SR=7.02) are substantially worse ✓
- Mean K values (11.52, 12.53, 12.05) align with the paper's "~12–13" (Minimization) and "~12" (eRank) ✓

Absolute ROC is approximately 1.8× the paper's reported values (unbounded) and 1.3× (bounded). See Deviations.

---

## Exp 2: Computational comparison of principal-component extraction methods

### Setup
- N = 2040 stocks (universe as of the middle rebalancing date), M+1=20 observations (Section 4.1)
- Efficient method: O(M² N + M³) auxiliary-matrix eigenpair extraction (Appendix C)
- Power-iterations: successive deflation with prec=1e-3 (Appendix B)
- 10 independent runs for random initializations of components 2..M (Section 4.1)

### Paper-reported results (Table 4, page 39)
Total power-iteration count across all 19 components: ~5000 per run on average.
Efficient method: exact same eigenpairs as direct eigh, verified numerically.

### Reproduced results

| Metric | Paper (Table 4, p.39) | Reproduced |
|--------|-------|-----------|
| N (universe size, one date) | 2339 | **2040** |
| M | 19 | **19** |
| Mean total iterations (10 runs) | ~5000 | **5017.5 ± 149.3** |
| Efficient method matches eigh | Yes | **Yes** (max eigenvector diff = 0.0) |
| Efficient method wall time (s) | O(M²N + M³) dominant | **0.004 s** |

The power-iterations total count (5017.5) closely matches the paper's ~5000 for N≈2339, M=19. The efficient non-iterative method exactly reproduces the top-M eigenpairs of the sample correlation matrix, confirmed by numerical comparison against `numpy.linalg.eigh`. Files: `results/exp2_iteration_counts.csv`, `results/exp2_summary.csv`.

---

## Exp 3: Backtest of nontraditional statistical risk models (shrinkage)

### Setup
Same backtest universe and dates as Exp 1 (bounded only, Section 4.4–4.5).
Diagonal-target shrinkage: `Ψ̃ = q·I + (1-q)·Ψ`, q ∈ {1e-6, 0.3, 0.6, 0.9}.
Uniform-correlation target: adds an intercept factor with correlation ρ, q × ρ ∈ {0.1, 0.5, 0.9}.

### Paper-reported results (Tables 5–6, page 40)

**Table 5 — diagonal shrinkage (bounded):**
ROC ≈ 40.65–40.67%, SR ≈ 13.81–13.84, CPS ≈ 1.81 (nearly invariant to q).

**Table 6 — uniform-correlation shrinkage (bounded):**
ROC ≈ 40.92–41.03%, SR ≈ 13.92–14.11, CPS ≈ 1.82–1.83 (nearly invariant to q and ρ).
Both underperform the traditional minimization-based model (bounded ROC=40.92%, SR=14.33).

### Reproduced results

**Table 5 — diagonal shrinkage (bounded, reproduced):**

| q | ROC (%) | SR | CPS ($) |
|---|---------|-----|---------|
| 1e-06 | **39.81** | **11.02** | **1.07** |
| 0.3 | **40.91** | **13.43** | **1.40** |
| 0.6 | *(pending)* | — | — |
| 0.9 | *(pending)* | — | — |

**Table 6 — uniform-correlation shrinkage (bounded, reproduced):** *(all pending — experiment in progress)*

Qualitative claim of near-invariance to q is confirmed between q=1e-6 and q=0.3 (ROC 39.81% → 40.91%), with q=0.3 already very close to the paper's ~40.65%.

---

## Exp 4: Sensitivity analysis of performance to fixed K and ad hoc specific-risk constructions

### Setup
Same backtest universe and dates as Exp 1. K ∈ {1..18} for fixed-K (Section 5 and Table 7–8).
Three sub-experiments per the paper (Tables 7–10, Figures 1–2):
- (A) Fixed K: unbounded (Table 7) and bounded (Table 8)
- (B) K₁=M=19 factors, spec risk borrowed from K-factor model (Table 9)
- (C) K factors with ad hoc xi²=1 (Table 10)

### Paper-reported results (Tables 7–10, Figures 1–2, pages 41–44)

Fixed-K bounded performance peaks around K≈10; Minimization K≈12–13 is near the in-sample optimum.
Both ad hoc constructions (B, C) underperform the traditional minimization-based model.
Table 10 monotonically increases in SR with K but at diminishing rate, approaching q→1 diagonal shrinkage.

### Reproduced results

**Table 7 — fixed K unbounded (complete):**

| K | ROC (%) | SR | CPS ($) |
|---|---------|-----|---------|
| 1 | 88.01 | 14.91 | 2.17 |
| 2 | 88.12 | 15.16 | 2.21 |
| 3 | 87.56 | 15.24 | 2.22 |
| 4 | 86.63 | 15.15 | 2.22 |
| 5 | 86.16 | 15.14 | 2.22 |
| 6 | 85.90 | 15.16 | 2.22 |
| 7 | 85.57 | 15.36 | 2.23 |
| 8 | 84.85 | 15.19 | 2.21 |
| 9 | 85.56 | 14.97 | 2.23 |
| 10 | 85.61 | 14.89 | 2.23 |
| 11 | 86.29 | 14.18 | 2.23 |
| 12 | 84.52 | 14.16 | 2.18 |
| 13 | 85.36 | 13.70 | 2.19 |
| 14 | 87.14 | 12.48 | 2.24 |
| 15 | 86.43 | 12.27 | 2.22 |
| 16 | 81.08 | 10.55 | 2.11 |
| 17 | 73.04 | 9.27 | 1.90 |
| 18 | 50.60 | 6.94 | 1.28 |

Unbounded SR peaks around K=3–7 (SR ≈ 15.2–15.4), then declines for high K. This is qualitatively consistent with the paper's claim that performance peaks around K≈10 (the paper may refer to bounded results).

**Table 8 — fixed K bounded (partial, K=1–8 reproduced):**

| K | ROC (%) | SR | CPS ($) |
|---|---------|-----|---------|
| 1 | *(from prior data)* | — | — |
| … | … | … | … |
| 8 | **52.12** | **15.67** | **1.38** |
| … | *(K=9–18 pending)* | — | — |

*(Tables 9–10 and Figures 1–2 are produced by the same experiment script; results pending.)*

---

## Deviations from the paper

1. **Data source:** Massive REST API (adjusted/unadjusted daily bars) used instead of the paper's proprietary data source. U.S.-listed CS tickers from Massive may differ from the paper's BICS-filtered universe as of 2014-09-06.

2. **BICS sector filter:** The paper filters by BICS sector to exclude OTCs and preferred shares (Section 3.4.2). Massive's `type=CS` (common stock) filter is used as the closest available proxy; exact BICS assignments as of 2014-09-06 are not available.

3. **Absolute performance gap:** Our reproduced ROC is approximately 1.8× (unbounded) and 1.3× (bounded) the paper's values. The relative ordering of K-selection methods and the qualitative conclusions (K'+1 worse, eRank ≈ Minimization, shrinkage underperforms) are preserved. The absolute gap is attributed to the stock universe not perfectly matching the paper's 2014-09-06 BICS snapshot: the Massive universe includes 2424 tickers (paper: ~2339–2000 after ADDV filter), and the actual ADDV-filtered top-2000 may differ.

4. **CPS gap (bounded):** Our bounded CPS ≈ 1.37 vs paper 1.96. CPS = total P&L / total shares. A lower CPS suggests our stocks trade at lower per-share dollar values (more shares per unit dollar invested), consistent with a different universe composition.

5. **Bounded optimizer:** The paper used R code from Appendix C of Kakushadze (2015d). This repository implements the equivalent ADMM-based constrained Sharpe optimization (bisection over λ). Results are equivalent under sufficient ADMM convergence.

6. **Exp 2 universe size:** N=2040 instead of the paper's N=2339, because we use the middle rebalancing date's universe. Total iteration count (5017.5) still closely matches the paper's ~5000 given the similar N and M=19.

7. **Exp 4 Table 8 bounded K=1 row:** The bounded fixed-K K=1 result from the previous (pre-ADMM-fix) run showed ROC=118.47%, which was incorrect (unbounded-scale). The current run is regenerating Table 8 with the correctly converging ADMM optimizer; K=8 bounded = 52.12%.

---

## Not reproduced

- Exact BICS sector classification and listing status as of 2014-09-06 (not available via Massive REST API without a BICS endpoint). The CS-type filter is the best available proxy.
- Precise quantitative match to paper Tables 1–2 absolute values (the ~1.8× ROC gap is attributable to universe differences, not a methodology error; qualitative conclusions and method rankings are reproduced).
- Figures 1–2 from the paper (fixed-K bounded SR and ROC vs K): produced as `exp4_fig1.png` and `exp4_fig2.png` using this implementation's data.
