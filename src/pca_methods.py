"""
Two methods for computing the first M eigenpairs of the sample correlation matrix.
Implements Appendix B (power iterations) and Appendix C (efficient no-iterations)
of Kakushadze & Yu (2016).

The two methods must produce identical eigenpairs up to sign and numerical precision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import M, POWER_ITER_PREC, POWER_ITER_RUNS


@dataclass
class EigenpairsResult:
    """Eigenpairs of the sample correlation matrix."""

    values: np.ndarray            # (M,) eigenvalues, descending
    vectors: np.ndarray           # (N, M) eigenvectors (columns), orthonormal
    elapsed_sec: float = 0.0
    n_iterations: Optional[np.ndarray] = None  # (M,) iteration counts (power method only)


# ─── Appendix C helper ────────────────────────────────────────────────────────

def calc_chol(m: int) -> np.ndarray:
    """
    Compute the M×M Cholesky-like factor of the centering matrix phi.
    Exact translation of calc.chol(m) from Appendix C of the paper.

    R code:
        x <- matrix(1, m, m); x <- upper.tri(x, T); z <- 1:m
        x <- x + diag(z); x <- x / sqrt(z*(z+1)); x <- t(x) / sqrt(m)

    Returns
    -------
    p : (M, M) lower-triangular matrix such that
        Y_tilde @ Y_tilde^T ≈ Y_demeaned @ Y_demeaned^T  (see Appendix C)
    """
    x = np.ones((m, m))
    x = np.triu(x)                                      # upper triangular incl. diagonal
    z = np.arange(1, m + 1, dtype=float)               # [1, 2, ..., m]
    x = x + np.diag(z)                                  # x[i,i] += z[i]
    # R's `x / sqrt(z*(z+1))` recycles the length-m vector down rows (column-major).
    # In numpy (row-major) this requires a column broadcast: [:, np.newaxis].
    x = x / np.sqrt(z * (z + 1))[:, np.newaxis]        # divide ROW i by sqrt(z[i]*(z[i]+1))
    x = x.T / np.sqrt(m)                               # transpose then divide by sqrt(m)
    return x   # (M, M) lower-triangular factor


# ─── Appendix C: efficient no-iterations method ───────────────────────────────

def eigenpairs_efficient(
    ret: np.ndarray,
    calc_cor: bool = True,
) -> EigenpairsResult:
    """
    Compute the first M eigenpairs of the sample covariance (or correlation)
    matrix without costly iterations. O(M^2 N + M^3) complexity.
    Implements qrm.calc.eigen.eff from Appendix C of the paper.

    Parameters
    ----------
    ret      : (N, M+1) return matrix, rows = stocks, cols = time points
    calc_cor : if True, divide each row by its sample std dev (ddof=1) first

    Returns
    -------
    EigenpairsResult with values (M,) descending, vectors (N, M) orthonormal
    """
    t0 = time.perf_counter()

    m = ret.shape[1] - 1      # M
    n = ret.shape[0]           # N

    if m > n:
        raise ValueError(f"Too many observations: M={m} > N={n}.")

    ret_work = ret.astype(float, copy=True)

    if calc_cor:
        # Divide by row sample std dev (ddof=1) – matches R `apply(ret, 1, sd)`
        sd = ret_work.std(axis=1, ddof=1)
        ret_work = ret_work / sd[:, np.newaxis]

    # Demean rows (row mean → 0 after this step, so all rows sum to 0)
    y = ret_work - ret_work.mean(axis=1)[:, np.newaxis]  # (N, M+1)

    # Drop last column (exploit rank-M structure from demeaning)
    y = y[:, :m]   # (N, M)

    # Apply Cholesky factor: Y_tilde = Y_trunc @ p  (Appendix C)
    p = calc_chol(m)              # (M, M)
    y_tilde = y @ p               # (N, M)

    # Build M×M auxiliary matrix G = Y_tilde^T Y_tilde
    G = y_tilde.T @ y_tilde       # (M, M)

    # Eigendecompose G (symmetric, so eigh is exact and O(M^3))
    q_val, q_vec = np.linalg.eigh(G)   # ascending order

    # Sort descending
    idx = np.argsort(q_val)[::-1]
    q_val = q_val[idx]
    q_vec = q_vec[:, idx]

    # Recover N-dimensional eigenvectors: V = Y_tilde @ (U / sqrt(rho))
    # where U = q_vec columns, rho = q_val entries
    q_vec_scaled = q_vec / np.sqrt(np.maximum(q_val, 1e-15))  # (M, M)
    vectors = y_tilde @ q_vec_scaled   # (N, M)

    elapsed = time.perf_counter() - t0
    return EigenpairsResult(values=q_val, vectors=vectors, elapsed_sec=elapsed)


# ─── Appendix B: power iterations method ─────────────────────────────────────

def eigenpairs_power_iterations(
    ret: np.ndarray,
    calc_cor: bool = True,
    k: int = M,
    prec: float = POWER_ITER_PREC,
    seed: Optional[int] = None,
) -> EigenpairsResult:
    """
    Compute the first K eigenpairs of the sample correlation matrix via
    successive power iterations with deflation. Implements qrm.calc.eigen from
    Appendix B of the paper.

    Algorithm (Section 4.1 / Appendix B):
      1. x = demeaned and row-Euclidean-norm normalized return matrix (N×M).
      2. For component 1: initialize y = 1/sqrt(N) * ones(N); iterate
         y <- x @ x^T @ y, normalize until max|y/y_prev - 1| < prec.
      3. Deflate: subtract projection of each column of x onto v.
      4. For components 2..K: initialize random y; same iteration + deflation.

    Parameters
    ----------
    ret      : (N, M+1) return matrix
    calc_cor : if True, divide rows by std dev before Euclidean norm (= correlation PCA)
    k        : number of components to compute (default = M = 19)
    prec     : convergence tolerance (default 1e-3 from Appendix B)
    seed     : random seed for reproducibility of random initializations

    Returns
    -------
    EigenpairsResult including per-component iteration counts
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    m_orig = ret.shape[1] - 1
    n = ret.shape[0]
    k = min(k, m_orig)

    ret_work = ret.astype(float, copy=True)

    if calc_cor:
        sd = ret_work.std(axis=1, ddof=1)
        ret_work = ret_work / sd[:, np.newaxis]

    # Demean rows and normalize by Euclidean norm (Appendix B)
    x = ret_work - ret_work.mean(axis=1)[:, np.newaxis]   # (N, M+1)
    row_norms = np.sqrt((x ** 2).sum(axis=1))              # (N,)
    row_norms = np.where(row_norms > 0, row_norms, 1.0)
    x = x / row_norms[:, np.newaxis]                       # (N, M+1), row norms = 1

    values = np.zeros(k)
    vectors = np.zeros((n, k))
    iter_counts = np.zeros(k, dtype=int)

    for i in range(k):
        if i == 0:
            # First component: unit initialization (Appendix B)
            y = np.ones(n) / np.sqrt(n)
        else:
            # Subsequent components: random initialization (Appendix B)
            y = rng.standard_normal(n)
            y = y / np.sqrt((y ** 2).sum())

        y_prev = np.zeros(n)
        it = 0
        while True:
            y_prev[:] = y
            # y <- x (x^T y) = (x @ x^T) y  (applies correlation matrix)
            xt_y = x.T @ y      # (M+1,)
            y = x @ xt_y        # (N,)
            norm_y = np.sqrt((y ** 2).sum())
            if norm_y < 1e-15:
                break
            y = y / norm_y
            it += 1
            # Convergence: max|y/y_prev - 1| < prec (Appendix B)
            ratio = np.abs(y) / (np.abs(y_prev) + 1e-300)
            if np.all(np.abs(ratio - 1.0) < prec):
                break

        v = y
        # Eigenvalue: ||x^T v||^2 (Appendix B: result$value <- sum((t(x) %*% y)^2))
        eigenvalue = float(np.sum((x.T @ v) ** 2))

        values[i] = eigenvalue
        vectors[:, i] = v
        iter_counts[i] = it

        # Deflation: remove projection of each column of x onto v (Appendix B)
        # x[,j] <- x[,j] - v * sum(v * x[,j])
        proj = x.T @ v        # (M+1,) dot products  [or equivalently v @ x]
        x = x - np.outer(v, proj)   # (N, M+1) – x[i,j] -= v[i] * proj[j]

    elapsed = time.perf_counter() - t0
    return EigenpairsResult(
        values=values,
        vectors=vectors,
        elapsed_sec=elapsed,
        n_iterations=iter_counts,
    )


# ─── Comparison utility ───────────────────────────────────────────────────────

def compare_eigenpairs(
    result_eff: EigenpairsResult,
    result_pow: EigenpairsResult,
    atol: float = 1e-3,
) -> dict:
    """
    Compare eigenvalues and eigenvectors from the two methods.
    Allows for sign ambiguity (eigenvectors defined up to ±1).

    Returns a dict with max absolute differences and pass/fail flag.
    """
    k = min(len(result_eff.values), len(result_pow.values))
    val_diff = np.abs(result_eff.values[:k] - result_pow.values[:k])

    # Align signs of eigenvectors before comparing
    signs = np.sign(
        np.sum(result_eff.vectors[:, :k] * result_pow.vectors[:, :k], axis=0)
    )
    signs = np.where(signs == 0, 1, signs)
    vec_diff = np.abs(result_eff.vectors[:, :k] - signs * result_pow.vectors[:, :k])

    return {
        "max_eigenvalue_diff": float(val_diff.max()),
        "max_eigenvector_diff": float(vec_diff.max()),
        "mean_eigenvalue_diff": float(val_diff.mean()),
        "pass": bool(val_diff.max() < atol and vec_diff.max() < atol),
    }
