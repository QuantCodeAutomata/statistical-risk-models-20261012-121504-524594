"""
Portfolio optimization for mean-reversion intraday strategy.
Implements Sharpe-ratio maximization under dollar neutrality (Section 3.4.4).

Unbounded: closed-form via Woodbury inverse (Appendix A R code).
Bounded:   ADMM with Woodbury structure — O(MN) per iteration,
           M factors, N stocks. No third-party QP solver required.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.config import INVESTMENT_LEVEL
from src.risk_model import RiskModelResult

logger = logging.getLogger(__name__)


class _BoundedADMMState:
    """
    Pre-computed Woodbury factors for the ADMM bounded optimizer.
    Shared across daily solves within one risk-model period (21 days).
    Follows the block structure of the Sherman-Morrison-Woodbury formula:

        M^{-1} = (D + rho*I + F F^T)^{-1} = D_rho_inv - D_F @ G_inv @ D_F.T

    where D_rho = diag(spec_sq + rho), D_F = D_rho^{-1} F, G = I_K + D_F.T @ F.
    """

    def __init__(self, spec_sq: np.ndarray, fac: np.ndarray, rho: float) -> None:
        N, K = fac.shape
        self.rho = float(rho)
        d = spec_sq + rho                           # (N,)
        self.d_inv = 1.0 / d                        # (N,)
        self.D_F = self.d_inv[:, np.newaxis] * fac  # (N, K): D_rho^{-1} F
        G = np.eye(K) + self.D_F.T @ fac           # (K, K): I + F^T D_rho^{-1} F
        self.G_inv = np.linalg.solve(G, np.eye(K)) # (K, K)
        ones = np.ones(N)
        self.M1 = self._apply_Minv(ones)            # (N,): M^{-1} 1
        self.den = self.M1.sum()                    # scalar: 1^T M^{-1} 1

    def _apply_Minv(self, v: np.ndarray) -> np.ndarray:
        """Apply (D_rho + F F^T)^{-1} to vector v using Woodbury formula."""
        w = self.D_F.T @ v                          # (K,)
        return self.d_inv * v - self.D_F @ (self.G_inv @ w)

    def solve_H_update(self, r: np.ndarray) -> np.ndarray:
        """
        Solve the dollar-neutral constrained H-update:
            min 0.5 H^T M H - r^T H  s.t.  1^T H = 0
        Returns H = M^{-1}(r - mu*1) with mu = (1^T M^{-1} r)/(1^T M^{-1} 1).
        """
        Mr = self._apply_Minv(r)
        mu = Mr.sum() / self.den
        return Mr - mu * self.M1


def _woodbury_apply(y_s: np.ndarray, B: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Apply Gamma^{-1} to a vector via Woodbury formula (Appendix A).
    Gamma^{-1} v = D^{-1} v - D^{-1} B (I + B^T D^{-1} B)^{-1} B^T D^{-1} v
    """
    K = B.shape[1]
    v1 = y_s[:, np.newaxis] * B              # (N, K) = D^{-1} B
    Dv = y_s * vec                            # (N,)  = D^{-1} v
    M_mat = np.eye(K) + B.T @ v1             # (K, K)
    sol = np.linalg.solve(M_mat, v1.T @ vec) # (K,)
    return Dv - v1 @ sol                      # (N,)


def optimize_unbounded(
    expected_returns: np.ndarray,
    model: RiskModelResult,
    investment_level: float = INVESTMENT_LEVEL,
) -> np.ndarray:
    """
    Unbounded Sharpe-ratio maximization under dollar neutrality.
    Section 3.4.4: H ∝ Gamma^{-1}(E - lambda*ones), lambda chosen for sum(H)=0.
    Holdings normalized so sum(|H|) = investment_level.

    Parameters
    ----------
    expected_returns : (N,) overnight alpha E_i
    model            : fitted RiskModelResult
    investment_level : gross portfolio value I (dollars)

    Returns
    -------
    H : (N,) dollar holdings
    """
    # Floor spec_risk to avoid 1/0 when K ≈ M factors absorb nearly all variance
    spec = np.maximum(model.spec_risk, 1e-8)
    y_s = 1.0 / (spec ** 2)              # (N,)
    B = model.fac_load                    # (N, K)
    E = expected_returns                  # (N,)

    # Gamma^{-1} E and Gamma^{-1} ones
    inv_cov_E = _woodbury_apply(y_s, B, E)
    inv_cov_1 = _woodbury_apply(y_s, B, np.ones(len(E)))

    # Dollar-neutral Lagrange multiplier: lambda = sum(Gamma^{-1} E) / sum(Gamma^{-1} 1)
    lam = inv_cov_E.sum() / inv_cov_1.sum()

    h = inv_cov_E - lam * inv_cov_1      # (N,) dollar-neutral signal

    gross = np.abs(h).sum()
    if gross < 1e-12:
        return np.zeros(len(E))

    return h / gross * investment_level


def make_admm_state(model: RiskModelResult, rho: Optional[float] = None) -> _BoundedADMMState:
    """
    Pre-compute Woodbury factors for the bounded ADMM solver from a RiskModelResult.
    Call once per risk-model update; reuse the state for all subsequent daily solves.

    Parameters
    ----------
    model : fitted RiskModelResult
    rho   : ADMM penalty (auto-set to mean(spec_sq) if None)
    """
    spec_sq = (model.spec_risk ** 2).astype(float)
    fac = model.fac_load.astype(float)
    if rho is None:
        # Set rho proportional to total portfolio variance so ADMM converges
        # even when specific risk is near zero (e.g., q≈0 shrinkage models).
        # Use at least 1% of mean total per-stock variance as the floor.
        total_var = float(np.mean(spec_sq + np.sum(fac ** 2, axis=1)))
        rho = max(float(np.mean(spec_sq)), total_var * 0.01)
        rho = max(rho, 1e-6)
    return _BoundedADMMState(spec_sq, fac, rho)


def optimize_bounded(
    expected_returns: np.ndarray,
    model: RiskModelResult,
    bounds: np.ndarray,
    investment_level: float = INVESTMENT_LEVEL,
    admm_state: Optional[_BoundedADMMState] = None,
    max_iter: int = 60,
    tol_abs: float = 1e-4,
    tol_rel: float = 1e-3,
) -> np.ndarray:
    """
    Bounded Sharpe-ratio maximization: max E^T H / sqrt(H^T Gamma H)
    s.t. sum(H) = 0, sum|H| = I, |H_i| <= B_i.   (Section 3.4.4)

    Works in fractional-holdings space h = H/I so that the box bounds
    b_i = B_i/I and the unit-gross constraint sum|h| = 1 are dimensionally
    consistent with the return-space covariance Gamma.

    For a given risk-aversion parameter lambda, solves:
        min (lambda/2) h^T Gamma h  -  E^T h
        s.t. sum(h) = 0,  |h_i| <= b_i
    via ADMM with Woodbury-structured H-update.  Then bisects lambda so that
    the gross sum|h*(lambda)| = 1 (enforcing sum|H| = I simultaneously with
    the box constraints).  Returns H = h*(lambda*) * I.

    Parameters
    ----------
    expected_returns : (N,) overnight alpha E_i  (return units)
    model            : fitted RiskModelResult
    bounds           : (N,) per-stock position bounds B_i in dollars
    investment_level : gross portfolio value I (dollars)
    admm_state       : unused (kept for API compatibility)
    max_iter         : ADMM iteration cap per lambda call
    tol_abs / tol_rel: ADMM convergence tolerances

    Returns
    -------
    H : (N,) dollar holdings satisfying |H_i| <= B_i and sum(|H|) = I
    """
    N = len(expected_returns)
    E = expected_returns.astype(float)

    # Fractional bounds b_i = B_i / I (dimensionless; same space as h = H/I)
    b = np.where(
        np.isfinite(bounds) & (bounds > 0),
        bounds.astype(float) / investment_level,
        1.0,   # 1.0 = no effective constraint
    )

    spec_sq = np.maximum(model.spec_risk ** 2, 1e-14)   # (N,) return^2
    F = model.fac_load                                    # (N, K)
    K = F.shape[1]
    ones = np.ones(N)
    # Use TOTAL model variance (specific + factor) as the ADMM regularization
    # constant, so rho = lam * total_var and the ADMM contraction ratio
    # rho/(lam·Γ_ii + rho) ≈ 0.5 regardless of the specific-risk level.
    # This ensures fast convergence even for shrinkage models with q→0,
    # which keep all M factors so spec_sq≈0 while factor_var≈sigma².
    factor_var_mean = float(np.mean(np.sum(F ** 2, axis=1)))
    mean_sq = float(np.mean(spec_sq)) + factor_var_mean   # total model variance
    mean_sq = max(mean_sq, 1e-12)

    # ------------------------------------------------------------------
    # Precompute constant Woodbury components.
    #
    # For any lam, use rho = lam * mean_sq so rho/lam = mean_sq = const.
    # Then (lam*Gamma + rho*I)^{-1} = (1/lam) * (Gamma + mean_sq*I)^{-1}.
    # The Woodbury factors of (Gamma + mean_sq*I) are INDEPENDENT OF lam,
    # computed once here, and reused across all bisection steps.
    # Paper Appendix A / Section 4.2 Woodbury identity.
    # ------------------------------------------------------------------
    c = spec_sq + mean_sq               # effective constant diagonal
    c_inv = 1.0 / c                     # (N,)
    D_cF = c_inv[:, np.newaxis] * F     # (N, K)  D_c^{-1} F
    G_c = np.eye(K) + F.T @ D_cF       # (K, K)
    G_c_inv = np.linalg.solve(G_c, np.eye(K))
    M1_c = c_inv * ones - D_cF @ (G_c_inv @ (D_cF.T @ ones))  # (N,)
    den_c = M1_c.sum()

    if abs(den_c) < 1e-30:
        return np.zeros(N)

    def h_update_base(r: np.ndarray) -> np.ndarray:
        """Apply (Gamma + mean_sq*I)^{-1}(r) with dollar-neutral projection."""
        Mr = c_inv * r - D_cF @ (G_c_inv @ (D_cF.T @ r))
        mu = Mr.sum() / den_c
        return Mr - mu * M1_c

    def admm_solve(lam: float) -> np.ndarray:
        """
        ADMM for  min (lam/2) h^T Gamma h - E^T h,  s.t. sum(h)=0, |h_i|<=b_i.

        H-update: h = (1/lam) * (Gamma + mean_sq*I)^{-1}(E + rho*(z-u))
                    = scale * h_update_base(E + rho*(z-u))
        where scale = 1/lam and rho = lam*mean_sq.
        Woodbury factors are precomputed and reused for all lam.
        """
        scale = 1.0 / lam
        rho = lam * mean_sq            # rho/lam = mean_sq = const → same factors

        # Warm start: unconstrained solution (no box) projected to bounds
        h = scale * h_update_base(E)
        z = np.clip(h, -b, b)
        u = np.zeros(N)

        for _ in range(max_iter):
            h_new = scale * h_update_base(E + rho * (z - u))
            z_new = np.clip(h_new + u, -b, b)
            res = h_new - z_new
            u += res
            p_norm = float(np.linalg.norm(res))
            eps_p = tol_abs * np.sqrt(N) + tol_rel * max(
                float(np.linalg.norm(h_new)), float(np.linalg.norm(z_new))
            )
            z = z_new
            h = h_new
            if p_norm < eps_p:
                break
        return h  # h is dollar-neutral by construction (h_update_base projects)

    # ------------------------------------------------------------------
    # Bisect lam so that gross(h*(lam)) = 1.
    # gross(lam) is monotone decreasing.  Bracket exploits the fact that
    # for the unconstrained problem gross ~ gross_ub / lam, so
    # lam_hi = gross_ub gives gross ≤ 1 (bounds only reduce it further).
    # ------------------------------------------------------------------
    def _finalize(h_raw: np.ndarray) -> np.ndarray:
        """Normalize to investment level, then project onto {sum=0, |H|<=bounds}.

        Uses water-filling (bisect on Lagrange multiplier nu for the equality
        constraint).  This satisfies both constraints exactly up to float eps.
        """
        g = float(np.abs(h_raw).sum())
        if g < 1e-12:
            return np.zeros(N)
        H = h_raw / g * investment_level
        # Bisect nu such that sum_i clip(H_i + nu, -B_i, B_i) = 0.
        # The function is monotone increasing in nu.
        radius = float(np.max(np.abs(H))) + float(np.max(bounds)) + 1.0
        nu_lo, nu_hi = -radius, radius
        for _ in range(60):
            nu_mid = 0.5 * (nu_lo + nu_hi)
            if float(np.clip(H + nu_mid, -bounds, bounds).sum()) > 0.0:
                nu_hi = nu_mid
            else:
                nu_lo = nu_mid
        nu = 0.5 * (nu_lo + nu_hi)
        return np.clip(H + nu, -bounds, bounds)

    h_lam1 = admm_solve(1.0)
    gross_ub = float(np.abs(h_lam1).sum())

    if gross_ub < 1e-9:
        return np.zeros(N)
    if gross_ub <= 1.0:
        return _finalize(h_lam1)

    # Upper bracket: lam_hi such that gross ≤ 1
    lam_hi = gross_ub
    h_hi = admm_solve(lam_hi)
    if float(np.abs(h_hi).sum()) > 1.0:
        for _ in range(15):
            lam_hi *= 3.0
            h_hi = admm_solve(lam_hi)
            if float(np.abs(h_hi).sum()) <= 1.0:
                break
        else:
            return _finalize(h_hi)

    # Lower bracket: lam_lo such that gross >> 1
    lam_lo = max(gross_ub / 1000.0, 1e-8)
    if float(np.abs(admm_solve(lam_lo)).sum()) <= 1.0:
        h_lo = admm_solve(lam_lo)
        return _finalize(h_lo)

    # Bisect 15 steps → ~0.003% precision on lam
    for _ in range(15):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        h_mid = admm_solve(lam_mid)
        if float(np.abs(h_mid).sum()) > 1.0:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
            h_hi = h_mid
        if (lam_hi - lam_lo) / (lam_hi + lam_lo + 1e-30) < 1e-3:
            break

    return _finalize(h_hi)


def compute_holdings(
    expected_returns: np.ndarray,
    model: RiskModelResult,
    bounds: Optional[np.ndarray] = None,
    investment_level: float = INVESTMENT_LEVEL,
) -> np.ndarray:
    """
    Dispatch to unbounded or bounded optimizer.
    Returns (N,) dollar holdings H_i.
    """
    if bounds is None:
        return optimize_unbounded(expected_returns, model, investment_level)
    return optimize_bounded(expected_returns, model, bounds, investment_level)
