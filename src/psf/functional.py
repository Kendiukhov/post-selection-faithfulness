"""Post-selection bands for faithfulness metrics that are *not* simple means.

The most widely reported circuit-faithfulness metric is the **normalised logit
difference recovered**

    F(c) = ( E[LD(c)] - E[LD(empty)] ) / ( E[LD(full)] - E[LD(empty)] ),

a ratio of differences of means (Wang et al., 2023; Conmy et al., 2023;
Zhang and Nanda, 2024).  It is a smooth functional of a mean vector, so the
same multiplier-bootstrap machinery applies once we replace per-instance scores
by per-instance **influence contributions**

    psi_i(c) = grad g(mu_hat_c) . ( u_i(c) - mu_hat_c ),

which satisfy ``F_hat(c) - F(c) = mean_i psi_i(c) + o_P(n^{-1/2})``.

This module provides
  * ``ratio_influence`` -- influence functions for the normalised-logit-difference
    family, and
  * ``influence_band``  -- a simultaneous one-sided band from any influence
    matrix,
plus a finite-sample (conservative, Bonferroni + Fieller-style) alternative for
the ratio metric.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from .bounds import _max_multiplier_draws, effective_num_hypotheses  # noqa: F401

__all__ = [
    "ratio_band",
    "ratio_point_estimate",
    "ratio_influence",
    "influence_band",
    "ratio_union_lcb",
]


def ratio_point_estimate(
    ld_circuit: np.ndarray, ld_empty: np.ndarray, ld_full: np.ndarray
) -> np.ndarray:
    """Point estimate of the normalised logit difference recovered.

    Parameters
    ----------
    ld_circuit : (n, m) per-instance metric with only circuit ``c`` preserved.
    ld_empty   : (n,)   per-instance metric with the whole circuit ablated.
    ld_full    : (n,)   per-instance metric of the unablated model.
    """
    a = np.asarray(ld_circuit, dtype=np.float64).mean(axis=0)
    b = float(np.asarray(ld_empty, dtype=np.float64).mean())
    f = float(np.asarray(ld_full, dtype=np.float64).mean())
    return (a - b) / (f - b)


def ratio_influence(
    ld_circuit: np.ndarray, ld_empty: np.ndarray, ld_full: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(F_hat, psi)`` with ``psi`` the (n, m) influence matrix.

    Writing ``a_c = E LD(c)``, ``b = E LD(empty)``, ``f = E LD(full)`` and
    ``F(c) = (a_c - b) / (f - b)``, the delta method gives

        psi_i(c) = [ (A_i(c) - a_c) - (B_i - b) ] / (f - b)
                   - F(c) * [ (F_i - f) - (B_i - b) ] / (f - b).
    """
    A = np.asarray(ld_circuit, dtype=np.float64)
    B = np.asarray(ld_empty, dtype=np.float64).ravel()
    Fu = np.asarray(ld_full, dtype=np.float64).ravel()
    n, m = A.shape
    a = A.mean(axis=0)
    b = B.mean()
    f = Fu.mean()
    denom = f - b
    if abs(denom) < 1e-12:
        raise ValueError("degenerate normalisation: E[LD(full)] == E[LD(empty)]")
    Fhat = (a - b) / denom
    cA = A - a  # (n, m)
    cB = (B - b)[:, None]
    cF = (Fu - f)[:, None]
    psi = (cA - cB) / denom - Fhat[None, :] * (cF - cB) / denom
    return Fhat, psi


def influence_band(
    point: np.ndarray,
    psi: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    clusters: Optional[np.ndarray] = None,
    multiplier: str = "gaussian",
    var_floor: float = 1e-12,
    small_g_correction: bool = True,
) -> tuple[np.ndarray, float]:
    """Simultaneous one-sided lower band for a smooth functional of means.

    ``psi`` is the (n, m) matrix of per-instance influence contributions
    (columns approximately mean zero).  When ``clusters`` is supplied the
    influence contributions are aggregated within cluster before resampling,
    giving a cluster-robust (wild cluster bootstrap) band.

    Returns ``(lcb, qhat)``.
    """
    psi = np.asarray(psi, dtype=np.float64)
    n, m = psi.shape
    psi = psi - psi.mean(axis=0, keepdims=True)

    if clusters is None:
        se = np.sqrt(np.maximum((psi ** 2).sum(axis=0) / (n * (n - 1)), var_floor))
        E = psi / (n * se)
        draws = _max_multiplier_draws(E, n_boot, multiplier, seed, False, 4096)
        qhat = float(np.quantile(draws, 1 - alpha))
    else:
        cl = np.asarray(clusters)
        uniq, inv = np.unique(cl, return_inverse=True)
        G = uniq.size
        Cs = np.zeros((G, m))
        np.add.at(Cs, inv, psi)
        se = np.sqrt(np.maximum((Cs ** 2).sum(axis=0) / (n ** 2), var_floor))
        E = Cs / (n * se)
        mult = "rademacher" if multiplier == "gaussian" else multiplier
        draws = _max_multiplier_draws(E, n_boot, mult, seed, False, 4096)
        qhat = float(np.quantile(draws, 1 - alpha))
        if small_g_correction and G > 1:
            infl = stats.t.ppf(1 - alpha, df=G - 1) / stats.norm.ppf(1 - alpha)
            qhat *= max(1.0, infl)
    return np.asarray(point) - qhat * se, qhat


def ratio_band(
    ld_circuit: np.ndarray,
    ld_empty: np.ndarray,
    ld_full: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
    batch: int = 256,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simultaneous lower band for the normalised logit difference recovered,
    by the **empirical bootstrap-t** -- the ratio, its influence function and its
    standard error are all recomputed inside every resample.

    Writing ``u_i(c) = LD_i(c) - LD_i(empty)`` and ``v_i = LD_i(full) -
    LD_i(empty)``, the estimator is ``F(c) = mean(u_c) / mean(v)`` and its
    influence function simplifies to ``psi_i(c) = (u_i(c) - F(c) v_i) / mean(v)``.
    Every bootstrap replicate therefore needs only three weighted sums of the
    ``(n, m)`` matrices ``u``, ``u^2`` and ``u v``, so the whole band costs three
    mat-muls per batch of replicates rather than a resample of the data.

    Returns ``(F_hat, lcb, qhat)``.
    """
    A = np.asarray(ld_circuit, dtype=np.float64)
    Bv = np.asarray(ld_empty, dtype=np.float64).ravel()
    Fv = np.asarray(ld_full, dtype=np.float64).ravel()
    n, m = A.shape
    U = A - Bv[:, None]  # (n, m)
    v = Fv - Bv  # (n,)
    ubar = U.mean(axis=0)
    vbar = float(v.mean())
    if abs(vbar) < 1e-12:
        raise ValueError("degenerate normalisation: E[LD(full)] == E[LD(empty)]")
    Fhat = ubar / vbar
    resid = U - Fhat[None, :] * v[:, None]
    s_hat = np.sqrt((resid ** 2).sum(axis=0) / (n - 1)) / abs(vbar)

    U2 = U * U
    Uv = U * v[:, None]
    v2 = v * v
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        # Efron resampling weights, drawn directly as multinomial counts
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)
        Sw = W.sum(axis=1, keepdims=True)  # (b, 1)
        su = W @ U  # (b, m)
        su2 = W @ U2
        suv = W @ Uv
        sv = (W @ v)[:, None]
        sv2 = (W @ v2)[:, None]
        ub = su / Sw
        vb = sv / Sw
        Fb = ub / vb  # (b, m)
        num = su2 - 2.0 * Fb * suv + (Fb ** 2) * sv2
        sb = np.sqrt(np.maximum(num, 0.0) / (Sw - 1.0)) / np.abs(vb)
        sb = np.maximum(sb, 1e-12)
        draws[done : done + b] = np.max(np.sqrt(n) * (Fb - Fhat[None, :]) / sb, axis=1)
        done += b
    qhat = float(np.quantile(draws, 1 - alpha))
    return Fhat, Fhat - qhat * s_hat / np.sqrt(n), qhat


def ratio_union_lcb(
    ld_circuit: np.ndarray,
    ld_empty: np.ndarray,
    ld_full: np.ndarray,
    alpha: float = 0.05,
    family_size: Optional[int] = None,
    lo: float = -20.0,
    hi: float = 20.0,
) -> np.ndarray:
    """Conservative finite-sample simultaneous bound for the ratio metric.

    We build simultaneous two-sided empirical-Bernstein intervals for the three
    ingredient means (numerator per circuit, and the two normalisers), splitting
    ``alpha`` across them, and then take the worst case of the ratio over the
    resulting box.  This requires the per-instance metric to be bounded in
    ``[lo, hi]``; for logit differences a generous range still gives a usable
    bound because the intervals shrink at rate ``1/sqrt(n)``.
    """
    from .bounds import empirical_bernstein_lcb

    A = np.asarray(ld_circuit, dtype=np.float64)
    B = np.asarray(ld_empty, dtype=np.float64).ravel()[:, None]
    Fu = np.asarray(ld_full, dtype=np.float64).ravel()[:, None]
    n, m = A.shape
    M = m if family_size is None else int(family_size)

    # two-sided EB intervals: run the one-sided bound on X and on -X
    def two_sided(X, fam, share):
        loB = empirical_bernstein_lcb(X, alpha=share / 2, family_size=fam, lo=lo, hi=hi)
        hiB = -empirical_bernstein_lcb(-X, alpha=share / 2, family_size=fam, lo=-hi, hi=-lo)
        return loB, hiB

    aL, aU = two_sided(A, M, alpha / 3)
    bL, bU = two_sided(B, 1, alpha / 3)
    fL, fU = two_sided(Fu, 1, alpha / 3)
    bL, bU, fL, fU = float(bL[0]), float(bU[0]), float(fL[0]), float(fU[0])

    # worst-case ratio over the box, assuming the normaliser stays positive
    if fL - bU <= 0:
        return np.full(m, -np.inf)
    num_lo = aL - bU
    den = np.where(num_lo >= 0, fU - bL, fL - bU)
    return num_lo / den
