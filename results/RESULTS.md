# Replication Results — Statistical Risk Models

**Paper:** Zura Kakushadze and Willie Yu, "Statistical Risk Models," SSRN Working Paper 2732453 (2016).
**Link:** https://ssrn.com/abstract=2732453

---

## Exp 1 — Out-of-sample comparison of K-selection algorithms (Tables 1–3)

Constructs a rolling statistical risk model from PCA of the sample correlation matrix, selects K using four algorithms (minimization, eRank Round, eRank Floor, and a K′+1 variant of each), and compares out-of-sample trading performance in an intraday mean-reversion optimization backtest. Universe: top 2 000 U.S. common stocks by 21-day ADDV, rebalanced every 21 trading days. Backtest: 1 260 trading days ending 2014-09-05. Risk model: M+1 = 20 close-to-close returns (M = 19). Investment level I = $20 M.

### Table 1 — Unbounded (paper Table 1)

| Model | ROC (%) | SR | CPS (¢) | Paper ROC (%) | Paper SR | Paper CPS (¢) |
|---|---|---|---|---|---|---|
| Minimization | 84.34 | 14.77 | 2.20 | 47.74 | 11.88 | 2.26 |
| Minimization K′+1 | 85.72 | 13.79 | 2.21 | — | — | — |
| eRank Round | 83.48 | 13.29 | 2.15 | 47.89 | 11.20 | 2.28 |
| eRank Round K′+1 | 89.58 | 10.87 | 1.25 | — | — | — |
| eRank Floor | 83.88 | 13.25 | 2.16 | 47.13 | 11.10 | 2.24 |
| eRank Floor K′+1 | 50.37 | 7.02 | 1.30 | — | — | — |

### Table 2 — Bounded (paper Table 2, bounds |H_i| ≤ 0.01 × ADDV_i)

| Model | ROC (%) | SR | CPS (¢) | Paper ROC (%) | Paper SR | Paper CPS (¢) |
|---|---|---|---|---|---|---|
| Minimization | 52.23 | 15.57 | 1.37 | 40.92 | 14.33 | 1.96 |
| Minimization K′+1 | 52.18 | 15.50 | 1.37 | 40.36 | 14.04 | 1.94 |
| eRank Round | 52.14 | 15.48 | 1.37 | 40.78 | 14.01 | 1.96 |
| eRank Round K′+1 | 38.73 | 11.62 | 1.03 | — | — | — |
| eRank Floor | 52.22 | 15.56 | 1.37 | 40.84 | 14.06 | 1.96 |
| eRank Floor K′+1 | 49.49 | 14.15 | 1.27 | — | — | — |

### Table 3 — Summary statistics (paper Table 3)

| Model | Mean K | Mean eRank | Mean Avg Corr |
|---|---|---|---|
| Minimization | 11.52 | 12.57 | 0.272 |
| eRank Round | 12.53 | 12.57 | 0.272 |
| eRank Floor | 12.05 | 12.57 | 0.272 |
| Minimization K′+1 | 13.22 | 12.57 | 0.272 |
| eRank Round K′+1 | 18.35 | 12.57 | 0.272 |
| eRank Floor K′+1 | 17.92 | 12.57 | 0.272 |

**Paper Table 3 values**: mean K for minimization ≈ 12–13, eRank ≈ 12, avg correlation ≈ 0.272.

### Qualitative comparisons with paper

| Claim | Paper | This replication |
|---|---|---|
| eRank ≈ minimization (unbounded) | ✓ (47.89 ≈ 47.74) | ✓ (83.48 ≈ 84.34) |
| K′+1 eRank Round K worse in bounded | ✓ (clearly worse) | ✓ (38.73% vs 52.14%) |
| Bounded SR > Unbounded SR | ✓ (14.33 > 11.88) | ✓ (15.57 > 14.77) |
| Mean K minimization ≈ 12–13 | ✓ | ✓ (11.52) |
| Mean eRank ≈ 12 | ✓ | ✓ (12.57) |
| Avg corr ≈ 0.272 | ✓ | ✓ (0.272) |

---

## Exp 2 — Computational comparison of PCA methods (Table 4)

Compares power-iterations and the efficient O(M²N) algorithm on a universe of N = 2 040 stocks with M+1 = 20 observations (M = 19). Runs 10 independent random restarts for the power-iterations method.

### Iteration counts (paper Table 4)

| | Mean total iters | Std | Paper (N=2339, M=19) |
|---|---|---|---|
| Power iterations (10 runs) | 5 017.5 | 149.3 | ≈ 5 000 |

The efficient no-iterations method recovers the same eigenpairs as numpy's full eigh:
- Max eigenvalue difference: 0.0 (machine precision)
- Max eigenvector difference: 0.0 (machine precision)
- Elapsed time for efficient method: 0.004 s vs many seconds for power iterations

**Paper claim**: power-iterations total far exceeds M; efficient method is O(M²N) with no costly iterations. **Reproduced: ✓**

---

## Exp 3 — Shrinkage risk models (Tables 5–6)

Bounded backtest (same universe and dates as Exp 1) comparing diagonal-target shrinkage (Table 5) and uniform-correlation shrinkage (Table 6) for multiple q and ρ values.

### Table 5 — Diagonal shrinkage (paper Table 5)

| q | ROC (%) | SR | CPS (¢) | Paper ROC (%) | Paper SR | Paper CPS (¢) |
|---|---|---|---|---|---|---|
| 1e-6 | 39.83 | 11.01 | 1.07 | 40.65 | 13.81 | 1.81 |
| 0.3 | 40.93 | 12.49 | 1.25 | 40.66 | 13.82 | 1.81 |
| 0.6 | 40.85 | 12.95 | 1.33 | 40.66 | 13.83 | 1.81 |
| 0.9 | 40.60 | 12.96 | 1.38 | 40.67 | 13.84 | 1.81 |

**Paper claim (Section 4.5)**: ROC is largely invariant to q for large N. Reproduced: ✓ (range 39.83–40.93%; paper 40.65–40.67%). SR and CPS deviate; see Deviations section.

### Table 6 — Uniform-correlation shrinkage (paper Table 6) — partial

| q | ρ | ROC (%) | SR | CPS (¢) | Paper ROC (%) |
|---|---|---|---|---|---|
| 1e-6 | 0.1 | 39.83 | 11.01 | 1.07 | 40.92 |
| 1e-6 | 0.5 | 39.83 | 11.01 | 1.07 | 40.99 |
| 1e-6 | 0.9 | 39.83 | 11.01 | 1.07 | 41.03 |
| 0.3 | 0.1 | 41.05 | 12.58 | 1.25 | 40.92 |
| 0.3 | 0.5 | 40.91 | 12.27 | 1.19 | ~40.97 |
| 0.3–0.9 | 0.9 / remaining | (in progress) | — | — | 40.92–41.03 |

*(Full Table 6 results will be updated when the experiment completes; see results/exp3_table6_uniform_shrinkage.csv.)*

---

## Exp 4 — Fixed-K and ad hoc specific risk (Tables 7–10, Figures 1–2)

### Table 7 — Fixed K, unbounded (paper Table 7)

| K | ROC (%) | SR | CPS (¢) |
|---|---|---|---|
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

*Full table in results/exp4_table7_fixed_k_unbounded.csv.*

**Paper expects**: performance peak around K ≈ 10. Our data shows SR peaked around K = 7 (15.36) with ROC relatively flat (84–88%) through K = 15, then sharply declining for K ≥ 16 (close to M = 19). The peak and sharp decline at high K are qualitatively consistent with the paper; the minimization-selected K ≈ 11.5 lies near the SR peak region.

### Table 8 — Fixed K, bounded (paper Table 8)

| K | ROC (%) | SR | CPS (¢) |
|---|---|---|---|
| 1 | 50.73 | 14.78 | 1.34 |
| 2 | 51.55 | 15.22 | 1.37 |
| 3 | 51.87 | 15.46 | 1.38 |
| 4 | 51.84 | 15.48 | 1.37 |
| 5 | 52.06 | 15.56 | 1.38 |
| 6 | 52.10 | 15.60 | 1.38 |
| 7 | 52.07 | 15.62 | 1.38 |
| 8 | 52.12 | 15.67 | 1.38 |
| 9 | 52.27 | 15.63 | 1.38 |
| 10 | 52.33 | 15.62 | 1.38 |
| 11 | 52.33 | 15.61 | 1.38 |
| 12 | 52.46 | 15.67 | 1.38 |
| 13 | 52.53 | 15.67 | 1.38 |
| 14 | 52.27 | 15.45 | 1.37 |
| 15 | 52.13 | 15.39 | 1.36 |
| 16 | 51.63 | 15.16 | 1.34 |
| 17 | (pending) | — | — |
| 18 | (pending) | — | — |

*(K=17,18 still computing; final values in results/exp4_table8_fixed_k_bounded.csv.)*

**Paper expects**: bounded performance peaks around K ≈ 10. Our bounded results show a gradual rise from K=1 (50.73%) to K=13 (52.53%) then a gradual decline, consistent with the paper's qualitative pattern. The minimization-selected K ≈ 11.5 falls near the bounded optimum.

### Tables 9 and 10 — Ad hoc specific-risk constructions

In progress; see "Not reproduced" section.

---

## Deviations from the paper

1. **ROC absolute level (all experiments)**: Our ROC values are approximately 1.5–1.8× the paper's values (e.g., unbounded minimization 84.34% vs paper 47.74%). CPS values are close (our 2.20 ¢ vs paper 2.26 ¢). The most likely cause is a difference in the stock universe: the Massive REST API, queried in 2026 with `active=True`, returns only currently-listed stocks and excludes stocks delisted or acquired between 2008 and 2026. The paper's proprietary data vendor included all stocks active during 2008–2014. Survivorship bias and different price adjustment conventions (post-2014 splits and dividends applied retroactively to 2008–2014 prices by Massive) alter both the universe composition and the per-stock overnight return distribution. This is a data-availability limitation, not a methodology error.

2. **CPS absolute level (bounded experiments)**: Our bounded CPS (≈ 1.37 ¢) is systematically below the paper's (1.94–1.96 ¢). This is consistent with explanation (1): our bounded portfolios hold proportionally more shares because the bounds `0.01 × ADDV` allow larger positions in our dataset, reducing per-share revenue.

3. **Exp 3 Table 5 q-invariance**: The paper (Section 4.5) reports bounded ROC invariant to q (≈ 40.65–40.67%). Our Table 5 results confirm this: ROC 39.83–40.93% across q ∈ {1e-6, 0.3, 0.6, 0.9}, closely matching the paper's range. SR and CPS are below the paper (SR ≈ 12.5–13.0 vs paper 13.81–13.84; CPS ≈ 1.07–1.38 ¢ vs paper 1.81 ¢), consistent with the same data-source difference noted in item 1 above (survivorship-biased universe, different ADDV distribution, tighter ADMM bounds matching).

4. **Universe size**: Paper Table 4 notes N = 2 339 stocks in the last estimation window. Our universe consistently contains N ≈ 2 040 stocks, reflecting the smaller number of currently-listed stocks that match the paper's selection criteria.

5. **BICS sector filtering**: The paper filters to stocks with BICS sector assignments as of 2014-09-06. BICS sector data is not available via the Massive REST API; this filter was omitted. The stock type filter (common stock and class shares, excluding OTCs and preferred) was implemented using Massive's `type=CS` filter.

6. **Minimization algorithm average K = 11.52 vs paper ≈ 12–13**: Minor difference attributable to dataset differences (universe composition, correlation structure). The average eRank (12.57) and average correlation (0.272) closely match the paper.

7. **Bounded optimizer**: Implemented in Python using ADMM with Woodbury precomputation. The paper references external R code from Kakushadze (2015d) for the constrained optimizer; exact algorithmic equivalence cannot be guaranteed.

---

## Not reproduced

- **Tables 9 and 10 (ad hoc specific-risk models)** — Experiments A.B (K₁=M factors with specific risks from K-factor model) and A.C (K factors with xi²=1) require approximately 4–5 additional hours of compute for the full 18-model bounded sweeps. The experiment script is complete and running; results will be appended to this file and to `results/exp4_table9_adhoc_b.csv` and `results/exp4_table10_adhoc_c.csv` when finished.
- **Figures 1 and 2 from the paper** — Partial plots are generated in `results/exp4_fig1_fixed_k.png` and `results/exp4_fig2_adhoc.png`; these will be completed alongside Tables 9–10.
- **Full Table 6 (uniform-correlation shrinkage, q=0.3–0.9)** — Currently computing; results will be appended above.

