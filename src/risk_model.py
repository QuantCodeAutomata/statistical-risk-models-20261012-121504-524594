"""
Statistical risk model construction: PCA, K-selection algorithms, covariance matrix.
Implements qrm.cov.pc from Appendix A of Kakushadze & Yu (2016).

Key structures (Sections 2.4, 3.1–3.3):
  Gamma_hat_ij = sigma_i * sigma_j * (xi_i^2 * delta_ij + sum_A lambda_A * V_i^A * V_j^A)
  xi_i^2 = 1 - sum_A lambda_A * (V_i^A)^2       (in correlation space)
  Omega_iA = sqrt(lambda_A) * V_i^A              (factor loading in correlation space)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import M, USE_CORRELATION

logger = logging.getLogger(__name__)


@dataclass
class RiskModelResult:
    """Outputs of the statistical risk model (covariance form)."""

    spec_risk: np.ndarray    # (N,) specific risk sigma_i * xi_i in return space
    fac_load: np.ndarray     # (N, K) factor loadings sigma_i * Omega_iA in return space
    n_factors: int           # K (number of retained components)
    sigma: np.ndarray        # (N,) per-stock volatilities from sample data
    erank: Optional[float] = None  # effective rank of Psi, stored for reporting


def _sample_cov_eigh(ret: np.ndarray, use_cor: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficient eigendecomposition via the M×M auxiliary matrix (Appendix C, Section 4.2).

    Uses O(M² N) algorithm instead of O(N³) full N×N eigh, critical for N ≫ M.
    Returns the top M eigenpairs of the sample covariance or correlation matrix.

    Parameters
    ----------
    ret : (N, M+1) return matrix, rows = stocks, columns = time points
    use_cor : if True, normalize rows by sample std dev first (correlation matrix)

    Returns
    -------
    eigenvalues : (M,) descending order (top M eigenvalues only)
    eigenvectors : (N, M) columns = unit-length eigenvectors matching top M eigenvalues
    sigma : (N,) per-stock sample std devs (ddof=1); used to rescale to covariance units
    """
    from src.pca_methods import eigenpairs_efficient

    # Per-stock sample standard deviation for later rescaling to covariance units
    sigma = ret.std(axis=1, ddof=1)   # (N,)

    # Efficient M×M auxiliary-matrix eigendecomposition (Appendix C)
    result = eigenpairs_efficient(ret, calc_cor=use_cor)

    return result.values, result.vectors, sigma


def _k_select_minimization(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    tv: np.ndarray,
    k1: int,
    excl_first: bool,
    excl_fac_load: Optional[np.ndarray] = None,
    excl_tv_adj: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Minimization algorithm for K selection (Section 3.1 / Appendix A R code).

    Scans k from k1 to M (=d-1), at each step computing:
        Omega = sqrt(lambda_A) * V^A  for A = k1 .. k
        x_s   = tv - diag(Omega Omega^T)  (specific variance in cov-matrix diagonal units)
        z     = x_s / tv                  (fractional specific variance)
        g     = |sqrt(min(z)) + sqrt(max(z)) - 1|
    Stops when g worsens (increases) or becomes invalid; saves previous state.

    If excl_first=True: k1=2 (1-indexed) in the R code, meaning we skip the
    largest eigenpair and search from the 2nd onwards, then prepend first factor.
    The excl_fac_load is the first factor's contribution (to adjust tv beforehand).

    Parameters
    ----------
    eigvals  : (N,) or (M,) – full eigenvalue array (descending)
    eigvecs  : (N, N) or (N, M) – eigenvectors (columns)
    tv       : (N,) total variance per stock (diagonal of cov matrix)
    k1       : starting index (0-based Python) for the scan
    excl_first : whether we excluded the first PC (modifies final loading)
    excl_fac_load : (N, 1) factor loading of the first excluded PC
    excl_tv_adj : (N,) adjusted tv after removing first PC contribution (for g computation)

    Returns
    -------
    spec_risk_sq : (N,) specific variance (in cov-matrix units) at chosen K
    fac_load_out : (N, K) factor loading matrix at chosen K
    K            : number of factors
    """
    N = len(tv)
    d = len(eigvals)            # = M in the risk model (number of available eigenpairs)
    k_max = d - 1               # maximum index (0-based), corresponds to R's d-1

    # When excl_first=True, the diagonal variance used for g is tv adjusted for first PC
    tv_for_g = excl_tv_adj if (excl_first and excl_tv_adj is not None) else tv

    g_prev = np.inf
    best_spec_sq = tv.copy()    # fallback: no factors kept
    best_fac_load = np.zeros((N, 0))
    best_K = 0

    for k in range(k1, k_max + 1):  # inclusive [k1, k_max] in 0-based Python
        # Factor loadings: Omega_A = sqrt(lambda_A) * V^A  for A in k1..k
        u = eigvals[k1 : k + 1]   # eigenvalues for factors k1..k
        V = eigvecs[:, k1 : k + 1]  # eigenvectors
        omega = V * np.sqrt(u)       # (N, k-k1+1) factor loading matrix

        # Total factor contribution to diagonal variance
        xf = np.sum(omega ** 2, axis=1)    # (N,)
        xs = tv_for_g - xf                 # (N,) specific variance

        if np.any(xs <= 0) or np.any(~np.isfinite(xs)):
            break  # NA / invalid in R code → break

        z = xs / tv_for_g   # (N,) fractional specific variance
        g = abs(np.sqrt(z.min()) + np.sqrt(z.max()) - 1)

        if g > g_prev:
            break  # g worsened → keep previous best

        g_prev = g

        # Save state: actual xs must use original tv, not adjusted
        if excl_first and excl_fac_load is not None:
            # Combine first excluded factor with current search factors
            full_omega = np.hstack([excl_fac_load, omega])  # (N, 1 + k-k1+1)
            xf_full = np.sum(full_omega ** 2, axis=1)
            xs_full = tv - xf_full
            best_spec_sq = np.maximum(xs_full, 0.0)
            best_fac_load = full_omega
        else:
            best_spec_sq = np.maximum(xs, 0.0)
            best_fac_load = omega
        best_K = best_fac_load.shape[1]

    return best_spec_sq, best_fac_load, best_K


def _erank(eigenvalues: np.ndarray) -> float:
    """
    Effective Rank = exp(H) where H is the Shannon entropy of normalized eigenvalues.
    Section 3.2, Eq. (11) of the paper.
    eRank(Psi) = exp(-sum_A p_A * ln(p_A)) where p_A = lambda_A / sum_A lambda_A.
    """
    lam = eigenvalues[eigenvalues > 0]
    p = lam / lam.sum()
    # Shannon entropy of normalized eigenvalues
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def fit_risk_model(
    ret: np.ndarray,
    method: str = "min",
    excl_first: bool = False,
    use_cor: bool = USE_CORRELATION,
    fixed_k: Optional[int] = None,
    spec_risk_override: Optional[np.ndarray] = None,
) -> RiskModelResult:
    """
    Fit a statistical risk model to a (N, M+1) return matrix.

    Parameters
    ----------
    ret : (N, M+1) raw return matrix, rows = stocks, cols = time points
    method : one of 'min' (minimization), 'erank_round', 'erank_floor'
    excl_first : if True, exclude the first PC before K-selection and add K=K'+1
    use_cor : if True, use correlation matrix (recommended, Sections 2.4, 3.1)
    fixed_k : if set, override K-selection with this fixed number of factors
    spec_risk_override : if provided, replace computed xi^2 with this vector
                         (for ad hoc models in exp_4); shape (N,)

    Returns
    -------
    RiskModelResult with spec_risk and fac_load in RETURN space (sigma_i rescaled).

    Algorithm (Appendix A R code, qrm.cov.pc):
      1. Compute sigma_i (row std devs) for rescaling.
      2. If use_cor, normalize rows by sigma_i → correlation matrix.
      3. Compute N×N sample covariance (= correlation matrix if step 2) via np.cov.
      4. Eigendecompose descending.
      5. Select K via chosen method.
      6. Compute specific variance xi_i^2 = tv_i - sum_A lambda_A (V_i^A)^2.
      7. Rescale: spec_risk = sigma_i * sqrt(xi_i^2), fac_load = sigma_i * Omega_iA.
    """
    N, T = ret.shape
    M_data = T - 1  # number of available factors

    eigvals, eigvecs, sigma = _sample_cov_eigh(ret, use_cor=use_cor)

    # Diagonal of sample covariance / correlation matrix (= tv in R code).
    # For use_cor=True: tv = ones (diag of correlation matrix = 1 for each stock).
    # Use eigendecomposition reconstruction for generality: diag(C) = sum_A lambda_A * V_iA^2.
    tv = np.real(np.sum(eigvecs ** 2 * eigvals[np.newaxis, :], axis=1))

    # Effective rank over all M_data eigenpairs
    er = _erank(eigvals[:M_data])

    if fixed_k is not None:
        K = max(1, min(fixed_k, M_data))
        omega = eigvecs[:, :K] * np.sqrt(eigvals[:K])   # (N, K)
        xs = np.maximum(tv - np.sum(omega ** 2, axis=1), 0.0)
        spec_sq = xs
        fac_load_cor = omega

    elif method == "min":
        k1 = 0  # 0-based Python index → corresponds to k1=1 in R (1-indexed)
        excl_fac_load = None
        excl_tv_adj = None

        if excl_first:
            # Exclude first PC: compute its contribution and adjust tv
            v1 = eigvecs[:, 0:1] * np.sqrt(eigvals[0])  # (N, 1)
            excl_fac_load = v1
            excl_tv_adj = tv - np.sum(v1 ** 2, axis=1)
            k1 = 1  # start scanning from the second eigenvalue

        spec_sq, fac_load_cor, K = _k_select_minimization(
            eigvals[:M_data],
            eigvecs[:, :M_data],
            tv,
            k1=k1,
            excl_first=excl_first,
            excl_fac_load=excl_fac_load,
            excl_tv_adj=excl_tv_adj,
        )

    elif method in ("erank_round", "erank_floor"):
        k1 = 0
        excl_fac_load = None

        if excl_first:
            v1 = eigvecs[:, 0:1] * np.sqrt(eigvals[0])
            excl_fac_load = v1
            k1 = 1
            er_sub = _erank(eigvals[1:M_data])
            K_prime = (round(er_sub) if method == "erank_round" else int(np.floor(er_sub)))
            K_prime = max(1, min(K_prime, M_data - 1))
            omega_sub = eigvecs[:, 1 : 1 + K_prime] * np.sqrt(eigvals[1 : 1 + K_prime])
            fac_load_cor = np.hstack([excl_fac_load, omega_sub])  # (N, K'+1)
            K = fac_load_cor.shape[1]
        else:
            K = (round(er) if method == "erank_round" else int(np.floor(er)))
            K = max(1, min(K, M_data))
            fac_load_cor = eigvecs[:, :K] * np.sqrt(eigvals[:K])

        xs = np.maximum(tv - np.sum(fac_load_cor ** 2, axis=1), 0.0)
        spec_sq = xs

    else:
        raise ValueError(f"Unknown method: {method!r}")

    if spec_risk_override is not None:
        # Ad hoc: replace specific variance (exp_4 Experiment C)
        spec_sq = spec_risk_override

    # --- Rescale from correlation space to return space (Appendix A, end of qrm.cov.pc) ---
    if use_cor:
        # spec_risk in return space: sigma_i * sqrt(xi_i^2)
        spec_risk_out = sigma * np.sqrt(np.maximum(spec_sq, 0.0))
        # fac_load in return space: sigma_i * Omega_iA
        fac_load_out = sigma[:, np.newaxis] * fac_load_cor
    else:
        spec_risk_out = np.sqrt(np.maximum(spec_sq, 0.0))
        fac_load_out = fac_load_cor

    return RiskModelResult(
        spec_risk=spec_risk_out,
        fac_load=fac_load_out,
        n_factors=K,
        sigma=sigma,
        erank=er,
    )


def build_inv_cov(model: RiskModelResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute inverse covariance via Woodbury / Sherman-Morrison formula (Appendix A).
    Gamma^{-1} = D^{-1} - D^{-1} B (I + B^T D^{-1} B)^{-1} B^T D^{-1}
    where D = diag(spec_risk^2), B = fac_load.

    Returns
    -------
    inv_cov : (N, N) full inverse covariance (dense, only for small N)
    y_s     : (N,) = 1 / spec_risk^2   (the diagonal part of inv_cov)
    """
    spec_sq = model.spec_risk ** 2   # (N,)
    y_s = 1.0 / spec_sq              # (N,) = diag of D^{-1}
    B = model.fac_load               # (N, K)
    K = B.shape[1]

    v1 = y_s[:, np.newaxis] * B      # (N, K) = D^{-1} B
    M_mat = np.eye(K) + B.T @ v1    # (K, K) = I + B^T D^{-1} B
    M_inv = np.linalg.solve(M_mat, v1.T)  # (K, N) = M^{-1} B^T D^{-1}
    inv_cov = np.diag(y_s) - v1 @ M_inv  # (N, N)

    return inv_cov, y_s


def build_shrinkage_model(
    ret: np.ndarray,
    q: float,
    rho: Optional[float] = None,
    use_cor: bool = USE_CORRELATION,
) -> RiskModelResult:
    """
    Shrinkage-based risk model (Section 4.4).

    If rho is None: diagonal-target shrinkage.
        Gamma_ij = q * delta_ij + (1-q) * Psi_ij  (in correlation space)
        Factor form: K=M factors from Psi, specific variance xi_i^2 = q (all equal).

    If rho is given: uniform-correlation shrinkage target.
        Gamma_ij = xi_i^2 delta_ij + q*rho*nu_i*nu_j + (1-q)*sum Omega Omega^T
        where xi_i^2 = q*(1-rho), nu_i = 1.
        This yields M+1 factors: the M PCs plus the intercept (nu = ones/sqrt(N)).

    Both use the same M principal components of the sample correlation matrix Psi.
    """
    N, T = ret.shape
    M_data = T - 1

    eigvals, eigvecs, sigma = _sample_cov_eigh(ret, use_cor=use_cor)
    er = _erank(eigvals[:M_data])

    if rho is None:
        # Diagonal-target shrinkage: xi_i^2 = q in correlation space
        # Factor form: K=M factors with factor covariance (1-q)*I_K in PC basis
        # => fac_load_cor_A = sqrt(1-q) * sqrt(lambda_A) * V^A
        fac_load_cor = np.sqrt(1.0 - q) * eigvecs[:, :M_data] * np.sqrt(eigvals[:M_data])
        spec_sq = np.full(N, q)   # xi_i^2 = q for all i
    else:
        # Uniform-correlation shrinkage target: xi_i^2 = q*(1-rho), intercept factor
        # M PC factors with covariance (1-q)*I in PC basis, plus intercept factor nu=ones
        # Intercept factor loading: nu_i = sqrt(q*rho) (in correlation space)
        # -> adds one column to fac_load
        pc_load = np.sqrt(1.0 - q) * eigvecs[:, :M_data] * np.sqrt(eigvals[:M_data])  # (N, M)
        intercept_load = np.full((N, 1), np.sqrt(q * rho))    # (N, 1)
        fac_load_cor = np.hstack([pc_load, intercept_load])   # (N, M+1)
        spec_sq = np.full(N, q * (1.0 - rho))

    K = fac_load_cor.shape[1]

    if use_cor:
        spec_risk_out = sigma * np.sqrt(np.maximum(spec_sq, 0.0))
        fac_load_out = sigma[:, np.newaxis] * fac_load_cor
    else:
        spec_risk_out = np.sqrt(np.maximum(spec_sq, 0.0))
        fac_load_out = fac_load_cor

    return RiskModelResult(
        spec_risk=spec_risk_out,
        fac_load=fac_load_out,
        n_factors=K,
        sigma=sigma,
        erank=er,
    )
