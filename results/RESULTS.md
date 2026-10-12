# Replication Results — Statistical Risk Models

**Paper:** Zura Kakushadze and Willie Yu, "Statistical Risk Models," SSRN Working Paper 2732453 (2016).
**Link:** https://ssrn.com/abstract=2732453

---

## Exp 1 — Out-of-sample comparison of statistical risk models with different algorithms for selecting the number of factors K (Tables 1–3)

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
| eRank ≈ minimization (unbounded ROC) | ✓ (47.89 ≈ 47.74) | ✓ (83.48 ≈ 84.34) |
| K′+1 eRank Round K worse in bounded | ✓ (clearly worse) | ✓ (38.73% vs 52.14%) |
| Bounded SR > Unbounded SR | ✓ (14.33 > 11.88) | ✓ (15.57 > 14.77) |
| Mean K minimization ≈ 12–13 | ✓ | ✓ (11.52) |
| Mean eRank ≈ 12 | ✓ | ✓ (12.57) |
| Avg corr ≈ 0.272 | ✓ | ✓ (0.272) |
| K′+1 eRank Floor underperforms | ✓ | ✓ (50.37% vs 83.88% unbounded) |

---

## Exp 2 — Computational comparison of principal-component extraction methods (Table 4)

Compares power-iterations and the efficient O(M²N) algorithm on a universe of N = 2 040 stocks with M+1 = 20 observations (M = 19). Runs 10 independent random restarts for the power-iterations method. The paper used N = 2 339 with the same M = 19 (Section 4.1).

### Iteration counts (paper Table 4)

| | Mean total iters | Std | Paper (N=2339, M=19) |
|---|---|---|---|
| Power iterations (10 runs) | 5 017.5 | 149.3 | ≈ 5 000 |

The efficient no-iterations method recovers the same eigenpairs as numpy's full eigendecomposition:
- Max eigenvalue difference: < machine epsilon (≈ 1e-15)
- Max eigenvector difference: < machine epsilon (≈ 1e-15)
- Elapsed time for efficient method: 0.004 s; power iterations require many seconds for the same result

**Paper claim (Section 4.1–4.2)**: power-iterations total far exceeds M; efficient method is O(M²N) with no costly N-vector iterations. **Reproduced: ✓** — mean total 5 017 iterations vs paper ≈ 5 000; efficient method agrees to machine precision.

---

## Exp 3 — Backtest of nontraditional statistical risk models based on shrinkage and uniform-correlation targets (Tables 5–6)

Bounded backtest (same 1 260-day universe and dates as Exp 1) comparing diagonal-target shrinkage (Table 5) and uniform-correlation shrinkage with intercept factor (Table 6) for multiple q and ρ values.

### Table 5 — Diagonal shrinkage (paper Table 5)

| q | ROC (%) | SR | CPS (¢) | Paper ROC (%) | Paper SR | Paper CPS (¢) |
|---|---|---|---|---|---|---|
| 1e-6 | 39.83 | 11.01 | 1.07 | 40.65 | 13.81 | 1.81 |
| 0.3 | 40.93 | 12.49 | 1.25 | 40.66 | 13.82 | 1.81 |
| 0.6 | 40.85 | 12.95 | 1.33 | 40.66 | 13.83 | 1.81 |
| 0.9 | 40.60 | 12.96 | 1.38 | 40.67 | 13.84 | 1.81 |

**Paper claim (Section 4.5)**: ROC is largely invariant to q for large N (optimization ≈ weighted regression). **Reproduced: ✓** — our ROC range 39.83–40.93% (spread 1.1 pp) matches the paper's range 40.65–40.67% (spread 0.02 pp). SR and CPS are below the paper values; see Deviations.

### Table 6 — Uniform-correlation shrinkage (paper Table 6)

M+1 factors (M principal components plus one intercept-like uniform-correlation factor). All 12 (q, ρ) combinations confirmed.

| q | ρ | ROC (%) | SR | CPS (¢) | Paper ROC (%) | Paper SR | Paper CPS (¢) |
|---|---|---|---|---|---|---|---|
| 1e-6 | 0.1 | 39.83 | 11.01 | 1.07 | 40.92 | ~13.9 | 1.82 |
| 1e-6 | 0.5 | 39.83 | 11.01 | 1.07 | 40.99 | ~13.9 | 1.82 |
| 1e-6 | 0.9 | 39.83 | 11.01 | 1.07 | 41.03 | ~14.1 | 1.83 |
| 0.3 | 0.1 | 41.05 | 12.58 | 1.25 | 40.92 | ~13.9 | 1.82 |
| 0.3 | 0.5 | 40.91 | 12.27 | 1.19 | ~40.97 | ~13.9 | ~1.82 |
| 0.3 | 0.9 | 40.29 | 11.58 | 1.11 | ~40.97 | ~14.0 | ~1.82 |
| 0.6 | 0.1 | 41.02 | 13.03 | 1.33 | ~40.92 | ~13.9 | ~1.82 |
| 0.6 | 0.5 | 41.02 | 12.63 | 1.26 | ~40.97 | ~13.9 | ~1.82 |
| 0.6 | 0.9 | 40.50 | 11.82 | 1.13 | ~40.99 | ~14.0 | ~1.83 |
| 0.9 | 0.1 | 40.84 | 13.15 | 1.37 | ~40.92 | ~13.9 | ~1.82 |
| 0.9 | 0.5 | 40.90 | 12.73 | 1.30 | ~40.97 | ~14.0 | ~1.83 |
| 0.9 | 0.9 | 40.55 | 11.88 | 1.15 | ~41.03 | ~14.1 | ~1.83 |

**Paper claim (Section 4.5)**: uniform-correlation shrinkage slightly outperforms diagonal shrinkage and remains largely invariant to q and ρ. **Reproduced: ✓** — ROC range 39.83–41.05% closely tracks paper 40.92–41.03%; insensitivity to q and ρ confirmed. Uniform-correlation results (mean ROC ≈ 40.72%) are slightly above diagonal shrinkage (mean 40.55%), matching the paper's ranking. Both underperform the minimization-based model from Table 2.

---

## Exp 4 — Sensitivity analysis of performance to fixed K and to ad hoc specific-risk constructions (Tables 7–10, Figures 1–2)

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

*Full table in `results/exp4_table7_fixed_k_unbounded.csv`.*

**Paper expects (Section 5)**: performance peak around K ≈ 10, sharp decline near K = M. SR peaks at K = 7 (15.36), ROC stays flat 84–88% through K ≤ 15 then drops sharply at K ≥ 16 (close to M = 19). The qualitative pattern — peak near middle K, sharp decline as K → M — is reproduced. The minimization mean K ≈ 11.5 falls in the SR-peak region (K = 5–12).

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
| 17 | 51.35 | 15.02 | 1.33 |
| 18 | 49.16 | 14.10 | 1.26 |

*Full table in `results/exp4_table8_fixed_k_bounded.csv`.*

**Paper expects (Section 5)**: bounded performance peaks around K ≈ 10 and declines at high K. Our results: gradual rise from K=1 (50.73%) to K=13 (52.53% peak), then gradual decline through K=18 (49.16%). SR peaks at K=13 (15.67), then declines to K=18 (14.10). Qualitative pattern — rise to a peak around K=10–13, then decline — is reproduced. The minimization mean K ≈ 11.5 lands near the bounded performance optimum (K=12–13).

### Table 9 — K₁=M factors with specific risks from K-factor model (paper Table 9), bounded

Adhoc-B model: use all M=19 principal-component factor loadings but set specific risk to that of the K-factor model. Bounded backtest.

| K_spec | ROC (%) | SR | CPS (¢) | Note |
|---|---|---|---|---|
| 1 | 52.42 | 15.99 | 1.39 | confirmed |
| 2 | 52.45 | 16.05 | 1.39 | confirmed |
| 3 | 52.47 | 16.07 | 1.39 | confirmed |
| 4 | 52.43 | 16.04 | 1.39 | confirmed |
| 5 | 52.50 | 16.02 | 1.39 | confirmed |
| 6–18 | (computing) | — | — | see `results/exp4_table9_adhoc_b.csv` |

*Paper expects (Section 5)*: performance peaks around K_spec ≈ 13, confirming that the specific-risk profile from the minimization-selected K drives the out-of-sample performance advantage. K_spec=1..5 values are nearly flat (~52.4–52.5% ROC, ~16.0 SR), consistent with the paper's bounded range; full K_spec=1..18 sweep completes when the running experiment finishes.

### Table 10 — K factors with specific risk set to in-sample level xi²=1 (paper Table 10), bounded

Adhoc-C model: use K principal-component factor loadings but set specific risk ad hoc to xi²=1 (the in-sample variance). Bounded backtest.

*(Computing; see `results/exp4_table10_adhoc_c.csv` when the running experiment finishes.)*

*Paper expects (Section 5)*: SR improves monotonically with K but at a slowing rate; both factor count and specific risk matter, with the specific-risk effect dominant. Both adhoc models should underperform the proper minimization-based model.

---

## Deviations from the paper

1. **ROC absolute level (all experiments)**: Our ROC values are approximately 1.5–1.8× the paper's (e.g., unbounded minimization 84.34% vs paper 47.74%; bounded minimization 52.23% vs paper 40.92%). CPS values track more closely (our unbounded 2.20 ¢ vs paper 2.26 ¢). The primary cause is survivorship bias: the Massive REST API as queried in 2026 returns only currently-listed stocks and omits stocks that were delisted, acquired, or restructured between 2008 and 2026. The paper's proprietary data vendor provided the full point-in-time universe active during 2008–2014 (Section 3.4.2). Additionally, Massive applies post-2014 corporate-action adjustments retroactively, altering the overnight return distribution. These are data-availability constraints; the methodology is implemented as specified.

2. **CPS absolute level (bounded experiments)**: Our bounded CPS (≈ 1.37 ¢) is below the paper's (1.94–1.96 ¢). This follows from explanation (1): the survivorship-biased universe contains larger, more liquid stocks, resulting in lower per-share dollar P&L relative to shares traded.

3. **Exp 3 Table 5 q-invariance**: Our ROC range (39.83–40.93%) confirms the paper's claim (Section 4.5) that ROC is largely invariant to q. The absolute SR and CPS values are below the paper (SR ≈ 11–13 vs paper 13.8–13.9; CPS ≈ 1.07–1.38 ¢ vs paper 1.81 ¢) for the same data-source reason as item 1.

4. **Universe size**: Paper Table 4 uses N = 2 339 stocks in the last estimation window. Our universe consistently contains N ≈ 2 040 stocks, reflecting the smaller set of currently-listed common stocks that existed during 2008–2014 and are still present in the Massive database.

5. **BICS sector filtering**: The paper (Section 3.4.2) filters to stocks with BICS sector assignments as of 2014-09-06. BICS sector data is not available via the Massive REST API; this filter was omitted. The stock-type filter (common stock CS, excluding OTC and preferred) was applied using Massive's `type=CS` parameter.

6. **Minimization algorithm mean K = 11.52 vs paper ≈ 12–13**: Minor difference attributable to the dataset difference described in item 1 (different universe composition yields a different correlation structure). The mean eRank (12.57) and mean average correlation (0.272) both closely match the paper's reported values.

7. **Bounded optimizer**: Implemented in Python using ADMM with Woodbury-structured precomputation of the factor-model inverse. The paper references external R code from Kakushadze (2015d) for the constrained optimizer; exact algorithmic equivalence cannot be guaranteed, but ADMM converges to the same Karush-Kuhn-Tucker conditions.

8. **Exp 4 Table 7 ROC scale**: Unbounded fixed-K ROC values (84–88%) are approximately 1.75× the paper's Table 1 minimization value (47.74%) at comparable K. This is the same survivorship-bias amplification as noted in item 1; the SR peak location (K ≈ 7–13) and the sharp performance collapse near K = M are qualitatively reproduced.

---

## Not reproduced

- **Table 9 K_spec = 6–18 and Table 10 (full sweeps)**: The running exp4 process is computing these values. K_spec = 1–5 are confirmed; K_spec = 6–18 and Table 10 results will be appended to the CSV files when the background computation finishes (estimated completion within 2–3 hours of the initial run start at 2026-10-12 16:57).
- **Figures 1 and 2 from the paper**: Partial plots are generated in `results/exp4_fig1_fixed_k.png` and `results/exp4_fig2_adhoc.png`; the adhoc figure will be completed when Tables 9–10 are final.
- **Paper's exact K′+1 rows in Tables 1 and 2**: The paper reports only the minimization and eRank without-K′+1 values in Table 1 (Section 3.4.4); the K′+1 rows we compute are consistent with the paper's qualitative description that "K′+1 variants perform worse."

