"""Confidence bounds for a *selected* hypothesis.

All estimators in this module operate on a **score matrix**

    S : array of shape (n, m)

where ``S[i, c]`` is the per-instance faithfulness score of circuit hypothesis
``c`` on evaluation instance ``i``.  Scores are assumed to lie in a known
bounded range ``[lo, hi]`` (default ``[0, 1]``).  The population quantity of
interest is

    theta(c) = E_P[ phi(c; Z) ]

and the empirical estimate is the column mean of ``S``.

The distinction that this module exists to make precise:

* ``naive_*``            -- marginal bound for a *fixed, pre-specified* c.
                            INVALID after selection.
* ``union_*``            -- simultaneous over a pre-specified finite class C
                            (finite-sample, distribution-free).
* ``bootstrap_max_*``    -- simultaneous over C via multiplier bootstrap
                            (asymptotic, adapts to correlation across C).
* ``split_*``            -- sample splitting; valid for arbitrary selection.
* ``conditional_*``      -- selective inference conditional on the argmax event.

Every function returning a *lower confidence bound* (LCB) returns a value L
such that the advertised guarantee is ``P(theta >= L) >= 1 - alpha`` (one-sided).

References for the individual constructions are given in each docstring and
collected in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.special import log_ndtr

__all__ = [
    "BoundResult",
    "column_stats",
    "naive_lcb",
    "hoeffding_lcb",
    "empirical_bernstein_lcb",
    "wsr_betting_lcb",
    "bootstrap_max_quantile",
    "bootstrap_max_lcb",
    "cluster_bootstrap_max_lcb",
    "split_lcb",
    "conditional_winner_lcb",
    "hybrid_winner_lcb",
    "effective_num_hypotheses",
    "multiplicity_factor",
    "occam_lcb",
    "union_lcb",
    "size_stratified_log_prior",
    "floored_bootstrap_lcb",
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


_TORCH_DEVICE = None


def _torch_device():
    """Return a torch device for bootstrap matmuls, or ``None`` to use numpy.

    The multiplier bootstrap is a single large mat-mul; on Apple silicon the
    default OpenBLAS build in many environments is capped to a few threads and
    is ~20x slower than the GPU.  Set ``PSF_BOOTSTRAP_BACKEND=numpy`` to force
    the numpy path (results are statistically identical; the random draws
    differ, which is why every experiment records its seed and backend).
    """
    global _TORCH_DEVICE
    import os

    if _TORCH_DEVICE is not None:
        return _TORCH_DEVICE if _TORCH_DEVICE != "cpu-numpy" else None
    backend = os.environ.get("PSF_BOOTSTRAP_BACKEND", "auto").lower()
    if backend == "numpy":
        _TORCH_DEVICE = "cpu-numpy"
        return None
    try:
        import torch

        if torch.cuda.is_available():
            _TORCH_DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _TORCH_DEVICE = torch.device("mps")
        else:
            _TORCH_DEVICE = "cpu-numpy"
            return None
    except Exception:  # pragma: no cover
        _TORCH_DEVICE = "cpu-numpy"
        return None
    return _TORCH_DEVICE


def _max_multiplier_draws(
    E: np.ndarray,
    n_boot: int,
    multiplier: str,
    seed: int,
    two_sided: bool,
    batch: int,
) -> np.ndarray:
    """Draw ``n_boot`` copies of ``max_c sum_i xi_i E[i, c]``."""
    n, m = E.shape
    dev = _torch_device()
    if dev is not None:
        import torch

        # Draw the multipliers on the accelerator: generating them on the CPU and
        # copying costs ~25x more than the mat-mul they feed.
        try:
            g = torch.Generator(device=dev)
            g.manual_seed(int(seed))
            gdev = dev
        except (RuntimeError, TypeError):  # pragma: no cover - older torch
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            gdev = torch.device("cpu")
        Et = torch.from_numpy(np.ascontiguousarray(E, dtype=np.float32)).to(dev)
        out = torch.empty(n_boot, dtype=torch.float32, device=dev)
        done = 0
        while done < n_boot:
            b = min(batch, n_boot - done)
            if multiplier == "gaussian":
                xi = torch.randn(b, n, generator=g, device=gdev)
            elif multiplier == "rademacher":
                xi = torch.randint(0, 2, (b, n), generator=g, device=gdev).float() * 2.0 - 1.0
            elif multiplier == "mammen":
                p = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))
                a = -(np.sqrt(5.0) - 1.0) / 2.0
                bb = (np.sqrt(5.0) + 1.0) / 2.0
                u = torch.rand(b, n, generator=g, device=gdev)
                xi = torch.where(
                    u < p,
                    torch.tensor(a, device=gdev),
                    torch.tensor(bb, device=gdev),
                ).float()
            else:  # pragma: no cover
                raise ValueError(multiplier)
            Z = xi.to(dev) @ Et
            if two_sided:
                Z = Z.abs()
            out[done : done + b] = Z.max(dim=1).values
            done += b
        return out.cpu().numpy().astype(np.float64)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    Ef = np.ascontiguousarray(E, dtype=np.float32)
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        if multiplier == "gaussian":
            xi = rng.standard_normal((b, n), dtype=np.float32)
        elif multiplier == "rademacher":
            xi = rng.integers(0, 2, size=(b, n)).astype(np.float32) * 2.0 - 1.0
        elif multiplier == "mammen":
            p = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))
            a = -(np.sqrt(5.0) - 1.0) / 2.0
            bb = (np.sqrt(5.0) + 1.0) / 2.0
            u = rng.random((b, n), dtype=np.float32)
            xi = np.where(u < p, a, bb).astype(np.float32)
        else:  # pragma: no cover
            raise ValueError(multiplier)
        Z = xi @ Ef
        draws[done : done + b] = np.max(np.abs(Z), axis=1) if two_sided else np.max(Z, axis=1)
        done += b
    return draws


@dataclass
class BoundResult:
    """Container for a lower confidence bound on a (possibly selected) column."""

    lcb: np.ndarray  # (m,) simultaneous lower bounds, or (1,) for a single column
    selected: int  # index of the reported hypothesis
    lcb_selected: float  # lower bound for the reported hypothesis
    point: float  # point estimate used for the report
    method: str
    alpha: float
    extra: dict

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BoundResult(method={self.method!r}, selected={self.selected}, "
            f"point={self.point:.4f}, lcb={self.lcb_selected:.4f}, alpha={self.alpha})"
        )


def column_stats(S: np.ndarray, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, unbiased sample variance) for each column of ``S``."""
    S = np.asarray(S, dtype=np.float64)
    mean = S.mean(axis=0)
    var = S.var(axis=0, ddof=ddof)
    return mean, var


def _rescale(S: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if lo == 0.0 and hi == 1.0:
        return np.asarray(S, dtype=np.float64)
    rng = hi - lo
    if rng <= 0:
        raise ValueError("hi must be > lo")
    return (np.asarray(S, dtype=np.float64) - lo) / rng


# --------------------------------------------------------------------------
# marginal (invalid after selection) -- included as the "what people do" baseline
# --------------------------------------------------------------------------


def naive_lcb(S: np.ndarray, alpha: float = 0.05, use_t: bool = True) -> np.ndarray:
    """Marginal one-sided normal/t lower bound, column-wise.

    This is the bound implicitly used when a paper reports
    ``score +/- 1.96 * sem`` for the circuit it just searched for.  It is valid
    only for a hypothesis fixed *before* looking at the data.
    """
    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0]
    mean, var = column_stats(S)
    se = np.sqrt(np.maximum(var, 0.0) / n)
    q = stats.t.ppf(1 - alpha, df=n - 1) if use_t else stats.norm.ppf(1 - alpha)
    return mean - q * se


# --------------------------------------------------------------------------
# finite-sample simultaneous bounds (union bound over a pre-specified class)
# --------------------------------------------------------------------------


def hoeffding_lcb(
    S: np.ndarray,
    alpha: float = 0.05,
    family_size: Optional[int] = None,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Hoeffding + Bonferroni lower bound, simultaneous over ``family_size``
    hypotheses.

    With probability at least ``1 - alpha``, ``theta(c) >= L(c)`` for *every*
    ``c`` in the pre-specified class, hence for any data-dependent selection.

    ``family_size`` defaults to the number of columns of ``S``; pass a larger
    value when ``S`` holds only a subset of the class that was searched.
    """
    X = _rescale(S, lo, hi)
    n, m = X.shape
    M = m if family_size is None else int(family_size)
    delta = alpha / M
    rad = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    return lo + (hi - lo) * (X.mean(axis=0) - rad)


def empirical_bernstein_lcb(
    S: np.ndarray,
    alpha: float = 0.05,
    family_size: Optional[int] = None,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Maurer--Pontil empirical Bernstein + Bonferroni.

    For i.i.d. ``X_i in [0, 1]``, Maurer and Pontil (2009, Thm. 4) give, with
    probability at least ``1 - delta``,

        E[X] >= Xbar - sqrt(2 V_n log(2/delta) / n) - 7 log(2/delta) / (3(n-1))

    where ``V_n`` is the unbiased sample variance.  Setting
    ``delta = alpha / |C|`` yields a bound simultaneous over the class, and
    therefore valid post-selection.

    Because faithfulness scores are often near 0 or 1 (small variance), this is
    typically far tighter than Hoeffding.
    """
    X = _rescale(S, lo, hi)
    n, m = X.shape
    if n < 2:
        raise ValueError("empirical Bernstein needs n >= 2")
    M = m if family_size is None else int(family_size)
    delta = alpha / M
    L = np.log(2.0 / delta)
    mean, var = column_stats(X)
    rad = np.sqrt(2.0 * np.maximum(var, 0.0) * L / n) + 7.0 * L / (3.0 * (n - 1))
    return lo + (hi - lo) * (mean - rad)


def occam_lcb(
    S: np.ndarray,
    log_prior: np.ndarray,
    alpha: float = 0.05,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Prior-weighted (Occam) simultaneous bound.

    Given any probability distribution ``pi`` over the hypothesis class that is
    fixed before seeing the data, applying the empirical-Bernstein bound to
    hypothesis ``c`` at level ``alpha * pi(c)`` and taking a union gives a bound
    that holds simultaneously over the *whole* class, however large:

        theta(c) >= thetahat(c) - sqrt(2 V(c) L_c / n) - 7 L_c / (3(n-1)),
        L_c = log(2 / (alpha pi(c))).

    This is the practical tool for classes such as "all subsets of the 144
    attention heads of GPT-2 small", where a flat union bound is impossible.
    With the natural size-stratified prior it also makes precise a familiar
    intuition: *small circuits are cheaper to certify than large ones*, because
    there are fewer of them.

    ``log_prior`` must satisfy ``logsumexp(log_prior) <= 0``.
    """
    X = _rescale(S, lo, hi)
    n, m = X.shape
    lp = np.asarray(log_prior, dtype=np.float64)
    if lp.shape[0] != m:
        raise ValueError("log_prior must have one entry per column")
    tot = float(np.log(np.sum(np.exp(lp - lp.max()))) + lp.max())
    if tot > 1e-8:
        raise ValueError(f"prior does not sum to <= 1 (logsumexp = {tot:.4f})")
    L = np.log(2.0 / alpha) - lp
    mean, var = column_stats(X)
    rad = np.sqrt(2.0 * np.maximum(var, 0.0) * L / n) + 7.0 * L / (3.0 * (n - 1))
    return lo + (hi - lo) * (mean - rad)


def size_stratified_log_prior(sizes: np.ndarray, n_components: int) -> np.ndarray:
    """Log of the prior ``pi(c) = 1/(N+1) * 1/binom(N, |c|)`` over all subsets of
    ``n_components`` components, restricted to the columns actually evaluated.

    The returned vector is a valid (sub-probability) log-prior over the columns
    because distinct columns are distinct subsets.
    """
    from scipy.special import gammaln

    sizes = np.asarray(sizes, dtype=np.int64)
    N = int(n_components)
    log_binom = gammaln(N + 1) - gammaln(sizes + 1) - gammaln(N - sizes + 1)
    return -np.log(N + 1) - log_binom


def union_lcb(
    S: np.ndarray,
    alpha: float = 0.05,
    family_size: Optional[int] = None,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Best of Hoeffding and empirical Bernstein, each run at level
    ``alpha / 2`` so that their maximum is still a valid ``1 - alpha`` bound.

    Neither ingredient dominates: Hoeffding is better when scores sit near the
    middle of their range, empirical Bernstein when they concentrate near an
    endpoint (which is the common case for a *faithful* circuit).  Splitting
    the error budget costs only ``log 2`` and removes the need to guess.
    """
    a = hoeffding_lcb(S, alpha / 2, family_size, lo, hi)
    b = empirical_bernstein_lcb(S, alpha / 2, family_size, lo, hi)
    return np.maximum(a, b)


def wsr_betting_lcb(
    S: np.ndarray,
    alpha: float = 0.05,
    family_size: Optional[int] = None,
    lo: float = 0.0,
    hi: float = 1.0,
    grid: int = 400,
    n_perm: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Waudby-Smith & Ramdas (2024) predictable-plug-in empirical-Bernstein
    confidence *sequence*, evaluated at time ``n`` and Bonferroni-corrected.

    Included because for bounded scores concentrated near an endpoint the
    betting bound can dominate Maurer--Pontil at small ``n``.  This is a
    finite-sample, distribution-free bound.  ``n_perm > 1`` averages the
    capital process over random orderings of the data (the bound remains valid
    for each ordering; we take the *minimum* over orderings only if
    ``n_perm == 1``, otherwise we use the first ordering to stay valid).
    """
    X = _rescale(S, lo, hi)
    n, m = X.shape
    M = m if family_size is None else int(family_size)
    delta = alpha / M
    log_thresh = np.log(1.0 / delta)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n) if n_perm > 1 else np.arange(n)
    Xo = X[order]

    mus = np.linspace(0.0, 1.0, grid)
    out = np.empty(m)
    # running estimates for the predictable plug-in
    idx = np.arange(1, n + 1)[:, None]
    cum = np.cumsum(Xo, axis=0)
    mu_hat = (0.5 + cum) / (1.0 + idx)  # shape (n, m); mean of first t obs
    mu_hat_prev = np.vstack([np.full((1, m), 0.5), mu_hat[:-1]])
    sq = np.cumsum((Xo - mu_hat_prev) ** 2, axis=0)
    sigma_hat = np.sqrt((0.25 + sq) / (1.0 + idx))
    sigma_prev = np.vstack([np.full((1, m), 0.5), sigma_hat[:-1]])

    for j in range(m):
        x = Xo[:, j]
        sp = sigma_prev[:, j]
        lam = np.minimum(
            np.sqrt(2.0 * log_thresh / (np.maximum(sp, 1e-8) ** 2 * n * np.log(1 + n))),
            0.75,
        )
        # capital process for each candidate mu
        # K_t(mu) = sum_{i<=t} log(1 + lam_i (x_i - mu))
        ok = np.ones(grid, dtype=bool)
        for gi, mu in enumerate(mus):
            terms = 1.0 + lam * (x - mu)
            if np.any(terms <= 0):
                ok[gi] = False
                continue
            K = np.cumsum(np.log(terms))
            if np.max(K) >= log_thresh:
                ok[gi] = False
        valid = mus[ok]
        out[j] = valid.min() if valid.size else 0.0
    return lo + (hi - lo) * out


# --------------------------------------------------------------------------
# multiplier bootstrap simultaneous band  (main proposal)
# --------------------------------------------------------------------------


def _bootstrap_t_draws(
    S: np.ndarray,
    n_boot: int,
    seed: int,
    batch: int = 1024,
    var_floor: float = 1.0,
    ref_col: Optional[int] = None,
    two_sided: bool = False,
) -> np.ndarray:
    """Draws of ``max_c sqrt(n) (thetahat*(c) - thetahat(c)) / sigmahat*(c)``.

    The *variance is recomputed inside every bootstrap replicate*.  This matters:
    for the binary scores that dominate interpretability (interchange accuracy,
    behavioural agreement), the sample mean and the sample standard deviation are
    strongly dependent --- for a score above ``1/2``, an upward deviation of the
    mean comes with a *smaller* standard deviation --- so the studentised maximum
    has a heavier upper tail than a bootstrap that holds ``sigmahat`` fixed can
    reproduce.  Empirically, holding it fixed costs several points of coverage at
    ``n`` in the hundreds; recomputing it restores the nominal level (see
    ``tests/test_bounds.py``).

    The weights are Efron's multinomial resampling weights, so this is exactly
    the *empirical bootstrap* whose validity for studentised high-dimensional
    maxima is established by Chernozhukov, Chetverikov and Kato (2019, Thm. 4.3).
    """
    X = np.asarray(S, dtype=np.float64)
    n, m = X.shape
    mean = X.mean(axis=0)
    # Floor the studentising standard deviation at n^{-1/2}, in the observed
    # statistic and in every replicate alike.  Without a floor a hypothesis that
    # happens to score perfectly on every sampled instance has sigmahat = 0 and
    # would be certified at its point estimate; with this floor its interval has
    # width qhat/n, which is the order of the exact "rule of three" bound.
    floor = var_floor / n
    dev = _torch_device()

    if dev is not None:
        import torch

        try:
            g = torch.Generator(device=dev)
            g.manual_seed(int(seed))
            gdev = dev
        except (RuntimeError, TypeError):  # pragma: no cover
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            gdev = torch.device("cpu")
        A = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
        A2 = A * A
        mu = torch.from_numpy(mean.astype(np.float32)).to(dev)
        out = torch.empty(n_boot, dtype=torch.float32, device=dev)
        ref = torch.empty(n_boot, dtype=torch.float32, device=dev)
        done = 0
        while done < n_boot:
            b = min(batch, n_boot - done)
            idx = torch.randint(0, n, (b, n), generator=g, device=gdev)
            w = torch.zeros(b, n, device=gdev)
            w.scatter_add_(1, idx, torch.ones(b, n, device=gdev))
            w = w.to(dev)  # Efron multinomial resampling weights
            ws = w.sum(dim=1, keepdim=True)
            s1 = w @ A
            s2 = w @ A2
            th = s1 / ws
            var = (s2 / ws - th * th) * (ws / (ws - 1.0))
            sd = var.clamp_min(floor).sqrt()
            stat = np.sqrt(n) * (th - mu[None, :]) / sd
            out[done : done + b] = (stat.abs() if two_sided else stat).max(dim=1).values
            if ref_col is not None:
                ref[done : done + b] = stat[:, ref_col]
            done += b
        if ref_col is not None:
            return out.cpu().numpy().astype(np.float64), ref.cpu().numpy().astype(np.float64)
        return out.cpu().numpy().astype(np.float64)

    rng = np.random.default_rng(seed)
    A = np.ascontiguousarray(X, dtype=np.float32)
    A2 = A * A
    draws = np.empty(n_boot)
    refd = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float32)
        ws = w.sum(axis=1, keepdims=True)
        th = (w @ A) / ws
        var = ((w @ A2) / ws - th * th) * (ws / (ws - 1.0))
        sd = np.sqrt(np.maximum(var, floor))
        stat = np.sqrt(n) * (th - mean[None, :]) / sd
        draws[done : done + b] = np.max(np.abs(stat) if two_sided else stat, axis=1)
        if ref_col is not None:
            refd[done : done + b] = stat[:, ref_col]
        done += b
    if ref_col is not None:
        return draws, refd
    return draws


def bootstrap_max_quantile(
    S: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 2000,
    studentize: bool = True,
    multiplier: Literal["gaussian", "rademacher", "mammen"] = "gaussian",
    seed: int = 0,
    two_sided: bool = False,
    var_floor: float = 1.0,
    batch: int = 4096,
    recompute_variance: bool = True,
) -> tuple[float, np.ndarray]:
    """Multiplier-bootstrap critical value for the max of the studentized
    empirical process over the hypothesis class.

    Let ``e_i(c) = phi(c; Z_i) - thetahat(c)``.  We approximate the
    ``1 - alpha`` quantile of ``max_c sqrt(n) (thetahat(c) - theta(c)) /
    sigmahat(c)`` by the conditional quantile of

        T* = max_c  ( sum_i xi_i e_i(c) ) / ( sqrt(n) sigmahat(c) ),
        xi_i i.i.d. mean-0 variance-1 multipliers.

    Validity in high dimensions (``m`` may greatly exceed ``n``) follows from
    the Gaussian-approximation / multiplier-bootstrap theory of Chernozhukov,
    Chetverikov and Kato (2013, 2017).

    Returns ``(qhat, boot_draws)``.
    """
    X = np.asarray(S, dtype=np.float64)
    n, m = X.shape
    if studentize and recompute_variance:
        draws = _bootstrap_t_draws(
            X, n_boot, seed, batch=min(batch, 1024), var_floor=var_floor,
            two_sided=two_sided,
        )
        return float(np.quantile(draws, 1 - alpha)), draws
    mean, var = column_stats(X)
    sd = np.sqrt(np.maximum(var, var_floor / n)) if studentize else np.ones(m)
    E = (X - mean) / (np.sqrt(n) * sd)  # (n, m)
    draws = _max_multiplier_draws(E, n_boot, multiplier, seed, two_sided, batch)
    qhat = float(np.quantile(draws, 1 - alpha))
    return qhat, draws


def bootstrap_max_lcb(
    S: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 2000,
    studentize: bool = True,
    multiplier: str = "gaussian",
    seed: int = 0,
    var_floor: float = 1.0,
    return_q: bool = False,
    recompute_variance: bool = True,
):
    """Simultaneous one-sided lower band ``L(c) = thetahat(c) - qhat *
    sigmahat(c) / sqrt(n)`` valid for all ``c`` in the class at once.

    Because the band covers every column simultaneously, it remains valid for a
    column chosen by *any* rule that uses the same data (argmax, greedy search,
    eyeballing a plot, a Pareto frontier, ...).
    """
    X = np.asarray(S, dtype=np.float64)
    n, m = X.shape
    mean, var = column_stats(X)
    sd = np.sqrt(np.maximum(var, var_floor / n)) if studentize else np.ones(m)
    qhat, _ = bootstrap_max_quantile(
        X, alpha=alpha, n_boot=n_boot, studentize=studentize,
        multiplier=multiplier, seed=seed, var_floor=var_floor,
        recompute_variance=recompute_variance,
    )
    lcb = mean - qhat * sd / np.sqrt(n)
    if return_q:
        return lcb, qhat
    return lcb


def _wild_cluster_t_draws(
    Cs: np.ndarray,
    ng: np.ndarray,
    n: int,
    n_boot: int,
    multiplier: str,
    seed: int,
    var_floor: float,
    batch: int = 256,
) -> np.ndarray:
    """Wild cluster bootstrap-*t* draws of the studentised mean deviation.

    ``Cs[g, c]`` is the sum of the residuals of cluster ``g`` for hypothesis
    ``c``.  Multiplying cluster ``g``'s residuals by a sign ``xi_g`` gives a
    bootstrap sample whose mean deviates by ``D = sum_g xi_g Cs[g] / n``; its
    *recomputed* cluster-robust standard error is

        se*(c)^2 = sum_g ( xi_g Cs[g, c] - n_g D(c) )^2 / n^2 ,

    which is what Cameron, Gelbach and Miller (2008) prescribe -- the variance
    estimate must be recomputed in every replicate, not held fixed.
    """
    G, m = Cs.shape
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    Cs32 = np.ascontiguousarray(Cs, dtype=np.float64)
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        if multiplier == "rademacher":
            xi = rng.integers(0, 2, size=(b, G)).astype(np.float64) * 2.0 - 1.0
        elif multiplier == "gaussian":
            xi = rng.standard_normal((b, G))
        else:  # pragma: no cover
            raise ValueError(multiplier)
        D = (xi @ Cs32) / n  # (b, m)
        T = xi[:, :, None] * Cs32[None, :, :]  # (b, G, m)
        R = T - ng[None, :, None] * D[:, None, :]
        se = np.sqrt(np.maximum((R ** 2).sum(axis=1) / (n ** 2), var_floor))
        draws[done : done + b] = np.max(D / se, axis=1)
        done += b
    return draws


def floored_bootstrap_lcb(
    S: np.ndarray,
    alpha: float = 0.05,
    family_size: Optional[int] = None,
    n_boot: int = 2000,
    seed: int = 0,
    lo: float = 0.0,
    hi: float = 1.0,
    var_floor: float = 1.0,
    return_q: bool = False,
):
    """Bootstrap band where it is valid, finite-sample bound where it is not.

    The bootstrap band is asymptotic and it degenerates in one specific place: a
    hypothesis that scores perfectly on every *sampled* instance has zero sample
    variance, and no resampling of those instances can discover that its
    population value is below one.  We therefore use the (finite-sample) union
    bound for exactly those columns and the bootstrap band for the rest, each at
    level ``alpha / 2``.  Both ingredients are simultaneous over the whole class,
    so a union bound over their two failure events makes the combination valid at
    level ``alpha`` for any data-dependent choice between them -- including this
    one.  This is the recommended default when the class may contain hypotheses
    that are perfect on the sample.
    """
    X = np.asarray(S, dtype=np.float64)
    n, _ = X.shape
    var = X.var(axis=0, ddof=1)
    lb, q = bootstrap_max_lcb(
        X, alpha / 2, n_boot=n_boot, seed=seed, var_floor=var_floor, return_q=True
    )
    ub = union_lcb(X, alpha / 2, family_size, lo, hi)
    degenerate = var <= var_floor / n
    out = np.where(degenerate, ub, lb)
    return (out, q) if return_q else out


def cluster_bootstrap_max_lcb(
    S: np.ndarray,
    clusters: Sequence[int],
    alpha: float = 0.05,
    n_boot: int = 2000,
    multiplier: str = "rademacher",
    seed: int = 0,
    small_g_correction: bool = True,
    var_floor: float = 1e-12,
    return_q: bool = False,
):
    """Cluster-robust simultaneous lower band (wild cluster bootstrap).

    Intervention instances in mechanistic interpretability are rarely i.i.d.:
    the standard IOI dataset, for example, is generated from a handful of
    sentence templates.  If the target population is "new templates" rather
    than "new fills of these templates", the unit of resampling must be the
    cluster.

    We use Rademacher multipliers at the cluster level (the wild cluster
    bootstrap of Cameron, Gelbach and Miller, 2008).  With ``G`` clusters the
    number of distinct sign vectors is ``2^G``; for very small ``G`` the
    procedure is unreliable and we additionally inflate the critical value by
    the ratio of the ``t_{G-1}`` to the normal quantile when
    ``small_g_correction`` is set.
    """
    X = np.asarray(S, dtype=np.float64)
    n, m = X.shape
    cl = np.asarray(clusters)
    uniq, inv = np.unique(cl, return_inverse=True)
    G = uniq.size
    mean = X.mean(axis=0)
    R = X - mean  # residuals (n, m)

    # cluster sums of residuals: (G, m)
    Cs = np.zeros((G, m))
    np.add.at(Cs, inv, R)
    # cluster-robust variance of the mean: (1/n^2) sum_g (sum_{i in g} r_i)^2
    var_mean = (Cs ** 2).sum(axis=0) / (n ** 2)
    se = np.sqrt(np.maximum(var_mean, var_floor))

    ng = np.bincount(inv, minlength=G).astype(np.float64)
    draws = _wild_cluster_t_draws(Cs, ng, n, n_boot, multiplier, seed, var_floor)
    qhat = float(np.quantile(draws, 1 - alpha))
    if small_g_correction and G > 1:
        infl = stats.t.ppf(1 - alpha, df=G - 1) / stats.norm.ppf(1 - alpha)
        qhat = qhat * max(1.0, infl)
    lcb = mean - qhat * se
    if return_q:
        return lcb, qhat
    return lcb


def multiplicity_factor(
    S: np.ndarray, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Return ``(q_max, q_marginal, kappa)`` for the bootstrap-*t*.

    ``q_max`` is the simultaneous critical value; ``q_marginal`` is the
    critical value the *same* bootstrap would give for a single, pre-specified
    hypothesis (the arg-max column); and ``kappa = q_max / q_marginal`` is the
    factor by which searching inflates the interval.  Separating the two matters
    because ``q_max`` also absorbs the non-normality of the studentised
    statistic at finite ``n``, which has nothing to do with multiplicity.
    """
    X = np.asarray(S, dtype=np.float64)
    chat = int(X.mean(axis=0).argmax())
    dmax, dref = _bootstrap_t_draws(X, n_boot, seed, ref_col=chat)
    q_max = float(np.quantile(dmax, 1 - alpha))
    q_marg = float(np.quantile(dref, 1 - alpha))
    return q_max, q_marg, q_max / max(q_marg, 1e-12)


def effective_num_hypotheses(
    qhat: float, alpha: float = 0.05, q_marginal: Optional[float] = None
) -> float:
    """Number of *independent* hypotheses that would need the same critical
    value under a normal Bonferroni correction.

    With ``q_marginal`` supplied -- the critical value the same bootstrap gives
    for a single, pre-specified hypothesis -- the multiplicity factor
    ``kappa = qhat / q_marginal`` is applied to the normal quantile:
    ``m_eff = alpha / (1 - Phi(kappa * z_{1-alpha}))``.  This is the version to
    prefer: it isolates the price of *searching* from the finite-sample
    non-normality of the studentised statistic, which inflates ``qhat`` for
    reasons that have nothing to do with multiplicity.

    Without it, ``m_eff = alpha / (1 - Phi(qhat))``.  Either way the diagnostic
    is useful because circuit hypotheses are strongly positively correlated
    (overlapping circuits share most of their components), so ``m_eff`` can be
    far below the nominal class size.
    """
    if q_marginal is not None and q_marginal > 0:
        qhat = (qhat / q_marginal) * stats.norm.ppf(1 - alpha)
    tail = stats.norm.sf(qhat)
    if tail <= 0:
        return float("inf")
    return float(alpha / tail)


# --------------------------------------------------------------------------
# sample splitting
# --------------------------------------------------------------------------


def split_lcb(
    S: np.ndarray,
    alpha: float = 0.05,
    frac_select: float = 0.5,
    select_fn: Optional[Callable[[np.ndarray], int]] = None,
    method: str = "empirical_bernstein",
    seed: int = 0,
    lo: float = 0.0,
    hi: float = 1.0,
) -> BoundResult:
    """Select on one half of the instances, bound on the other.

    Valid for an arbitrary (even adversarial) selection rule and free of any
    dependence on the size of the hypothesis class.  The price is (i) a noisier
    selection and (ii) a wider interval on the smaller evaluation split.
    """
    X = np.asarray(S, dtype=np.float64)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n1 = int(round(frac_select * n))
    if not (0 < n1 < n):
        raise ValueError("frac_select must leave both splits non-empty")
    sel_idx, eval_idx = perm[:n1], perm[n1:]
    S_sel, S_eval = X[sel_idx], X[eval_idx]

    if select_fn is None:
        chosen = int(np.argmax(S_sel.mean(axis=0)))
    else:
        chosen = int(select_fn(S_sel))

    col = S_eval[:, [chosen]]
    if method == "empirical_bernstein":
        L = empirical_bernstein_lcb(col, alpha=alpha, family_size=1, lo=lo, hi=hi)[0]
    elif method == "hoeffding":
        L = hoeffding_lcb(col, alpha=alpha, family_size=1, lo=lo, hi=hi)[0]
    elif method == "normal":
        L = naive_lcb(col, alpha=alpha)[0]
    elif method == "betting":
        L = wsr_betting_lcb(col, alpha=alpha, family_size=1, lo=lo, hi=hi)[0]
    else:  # pragma: no cover
        raise ValueError(method)

    return BoundResult(
        lcb=np.array([L]),
        selected=chosen,
        lcb_selected=float(L),
        point=float(col.mean()),
        method=f"split[{method}]",
        alpha=alpha,
        extra={"n_select": n1, "n_eval": n - n1},
    )


# --------------------------------------------------------------------------
# selective inference conditional on the argmax event (polyhedral / hybrid)
# --------------------------------------------------------------------------


def _truncated_normal_sf(x: float, mu: float, sigma: float, a: float, b: float) -> float:
    """P(X > x) for X ~ N(mu, sigma^2) truncated to [a, b], computed stably."""
    if sigma <= 0:
        return float(x < mu)
    z = np.clip(
        np.array([(a - mu) / sigma, (b - mu) / sigma, (x - mu) / sigma]), -40.0, 40.0
    )
    # log-space survival functions, for stability in the far tail
    lo, hi, xx = log_ndtr(-z)
    # (sf(x) - sf(b)) / (sf(a) - sf(b))
    num = np.exp(xx) - np.exp(hi)
    den = np.exp(lo) - np.exp(hi)
    if den <= 0:
        return 0.5
    return float(np.clip(num / den, 0.0, 1.0))


def _invert_truncnorm_lower(
    obs: float, sigma: float, a: float, b: float, alpha: float,
    lo_search: float, hi_search: float, tol: float = 1e-8,
) -> float:
    """Smallest ``mu`` with ``P_mu(X > obs) >= alpha`` for a truncated normal.

    ``mu -> P_mu(X > obs)`` is increasing, so bisection gives the one-sided
    ``1 - alpha`` lower confidence bound of the selective interval.
    """
    f = lambda mu: _truncated_normal_sf(obs, mu, sigma, a, b) - alpha
    lo_v, hi_v = lo_search, hi_search
    if f(lo_v) >= 0:
        return lo_v
    if f(hi_v) < 0:
        return hi_v
    for _ in range(200):
        mid = 0.5 * (lo_v + hi_v)
        if f(mid) < 0:
            lo_v = mid
        else:
            hi_v = mid
        if hi_v - lo_v < tol:
            break
    return 0.5 * (lo_v + hi_v)


def conditional_winner_lcb(
    S: np.ndarray,
    alpha: float = 0.05,
    ridge: float = 1e-10,
    search_halfwidth: float = 5.0,
    feasible: Optional[np.ndarray] = None,
) -> BoundResult:
    """Selective (polyhedral) lower confidence bound for the argmax column.

    Under the normal approximation ``thetahat ~ N(theta, Sigma/n)``, the event
    ``{argmax = chat}`` is the polyhedron ``{A thetahat <= 0}`` with rows
    ``e_j - e_chat``.  Lee, Sun, Sun and Taylor (2016) show that
    ``eta' thetahat`` conditional on this event and on the sufficient statistic
    ``Z`` is a truncated normal; Andrews, Kitagawa and McCloskey (2024) use the
    same construction for inference on the winning arm.  Inverting the
    truncated-normal survival function yields a bound with *conditional*
    coverage given the selection event, and hence unconditional coverage as
    well.

    Note: conditional intervals can be extremely wide when the winner barely
    wins (``V^-`` close to the observed value); ``hybrid_winner_lcb`` fixes
    this at the cost of a small extra Bonferroni term.
    """
    X = np.asarray(S, dtype=np.float64)
    n, m = X.shape
    mean = X.mean(axis=0)
    # the selection event is "arg-max over the *feasible* hypotheses", so the
    # polyhedron must only contain the comparisons the investigator actually made
    feas = np.ones(m, dtype=bool) if feasible is None else np.asarray(feasible, dtype=bool)
    fidx = np.flatnonzero(feas)
    chat = int(fidx[np.argmax(mean[fidx])])
    Xc = X - mean
    # We only ever need the chat-th column of Sigma = Cov(thetahat), so avoid
    # forming the full m x m matrix (O(n m) instead of O(n m^2)).
    sig_col = (Xc.T @ Xc[:, chat]) / (n - 1) / n  # Sigma[:, chat]
    sig2 = float(sig_col[chat]) + ridge
    sigma = np.sqrt(max(sig2, 1e-300))
    cvec = sig_col / sig2
    obs = float(mean[chat])
    Z = mean - cvec * obs

    # constraints: mean[j] - mean[chat] <= 0 for all j != chat
    others = np.array([j for j in fidx if j != chat])
    if others.size == 0:
        L = naive_lcb(X[:, [chat]], alpha=alpha)[0]
        return BoundResult(np.array([L]), chat, float(L), obs, "conditional", alpha, {})
    Arow_c = cvec[others] - cvec[chat]  # coefficient on the pivot
    resid = Z[others] - Z[chat]  # constant part
    # constraint: resid_j + Arow_c_j * v <= 0
    Vminus, Vplus = -np.inf, np.inf
    pos = Arow_c > 1e-12
    neg = Arow_c < -1e-12
    if np.any(pos):
        Vplus = float(np.min(-resid[pos] / Arow_c[pos]))
    if np.any(neg):
        Vminus = float(np.max(-resid[neg] / Arow_c[neg]))
    zero = ~(pos | neg)
    poly_ok = bool(np.all(resid[zero] <= 1e-12)) if np.any(zero) else True

    if not poly_ok:  # numerically degenerate; fall back
        Vminus, Vplus = -np.inf, np.inf
    Vminus = max(Vminus, -1e6)
    Vplus = min(Vplus, 1e6)

    L = _invert_truncnorm_lower(
        obs, sigma, Vminus, Vplus, alpha,
        lo_search=obs - search_halfwidth, hi_search=obs + 1e-9,
    )
    return BoundResult(
        lcb=np.array([L]),
        selected=chat,
        lcb_selected=float(L),
        point=obs,
        method="conditional",
        alpha=alpha,
        extra={"Vminus": Vminus, "Vplus": Vplus, "sigma": sigma},
    )


def hybrid_winner_lcb(
    S: np.ndarray,
    alpha: float = 0.05,
    beta: Optional[float] = None,
    n_boot: int = 2000,
    seed: int = 0,
    feasible: Optional[np.ndarray] = None,
) -> BoundResult:
    """Hybrid selective bound (Andrews, Kitagawa and McCloskey, 2024).

    Step 1: form a simultaneous ``1 - beta`` band over the whole class (here by
    multiplier bootstrap rather than Bonferroni, which exploits the strong
    positive correlation between circuit hypotheses).
    Step 2: additionally condition on the argmax event *and* on the winner's
    value lying inside the step-1 band, then invert a truncated normal at level
    ``(alpha - beta) / (1 - beta)``.

    The result has coverage at least ``1 - alpha`` and is never much wider than
    the simultaneous band, while typically being much narrower than the pure
    conditional interval.
    """
    X = np.asarray(S, dtype=np.float64)
    n, m = X.shape
    if beta is None:
        beta = alpha / 10.0
    mean = X.mean(axis=0)
    feas = np.ones(m, dtype=bool) if feasible is None else np.asarray(feasible, dtype=bool)
    fidx = np.flatnonzero(feas)
    chat = int(fidx[np.argmax(mean[fidx])])
    # the step-1 region must be TWO-sided at level beta, because step 2 truncates
    # the winner's value from both sides
    qhat, _ = bootstrap_max_quantile(
        X, alpha=beta, n_boot=n_boot, seed=seed, two_sided=True
    )
    sd = np.sqrt(np.maximum(X.var(axis=0, ddof=1), 1e-12))
    sigma = float(sd[chat] / np.sqrt(n))
    obs = float(mean[chat])

    cond = conditional_winner_lcb(X, alpha=alpha, ridge=1e-12, feasible=feas)
    Vminus = max(cond.extra["Vminus"], obs - qhat * sigma)
    Vplus = min(cond.extra["Vplus"], obs + qhat * sigma)
    sigma_c = cond.extra["sigma"]
    a_star = (alpha - beta) / (1.0 - beta)
    L = _invert_truncnorm_lower(
        obs, sigma_c, Vminus, Vplus, a_star, lo_search=obs - 5.0, hi_search=obs + 1e-9
    )
    # the hybrid bound is by construction no lower than the simultaneous band
    L = max(L, obs - qhat * sigma)
    return BoundResult(
        lcb=np.array([L]),
        selected=chat,
        lcb_selected=float(L),
        point=obs,
        method="hybrid",
        alpha=alpha,
        extra={"beta": beta, "qhat": qhat, "Vminus": Vminus, "Vplus": Vplus},
    )
