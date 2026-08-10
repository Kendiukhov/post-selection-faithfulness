"""Replicate-based evaluation of post-selection bounds on a cached score matrix.

The design that makes every number in the paper exact:

    the pooled score matrix *is* the population.

We treat the ``N`` cached evaluation instances as a discrete distribution and
draw analysis samples of size ``n`` i.i.d. **with replacement** from it.  The
population faithfulness ``theta(c)`` is then the pooled column mean -- known
exactly, with no oracle noise -- so measured coverage is exact rather than an
estimate contaminated by uncertainty about the truth.

Two sampling schemes are supported:

``iid``      draw instances i.i.d.  (target population: new instances from the
             same generator, including new templates only in so far as the
             pool's template mix represents them);
``cluster``  draw *clusters* (e.g. prompt templates) i.i.d. and then instances
             within them (target population: new templates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from . import bounds as B
from .functional import influence_band, ratio_influence

__all__ = ["MethodSpec", "default_methods", "Selection", "run_replicates", "summarize"]


# ---------------------------------------------------------------------------
# selection rules
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    """How the practitioner picks a circuit to report."""

    kind: str = "argmax"  # "argmax" | "argmax_size_le" | "argmax_within"
    max_size: Optional[int] = None
    sizes: Optional[np.ndarray] = None

    def mask(self, m: int) -> np.ndarray:
        if self.kind == "argmax":
            return np.ones(m, dtype=bool)
        if self.kind == "argmax_size_le":
            assert self.sizes is not None and self.max_size is not None
            return self.sizes <= self.max_size
        raise ValueError(self.kind)

    def choose(self, point: np.ndarray) -> int:
        mk = self.mask(point.size)
        idx = np.flatnonzero(mk)
        return int(idx[np.argmax(point[idx])])


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


@dataclass
class MethodSpec:
    name: str
    fn: Callable  # (S, alpha, ctx) -> (lcb_vector, selected_or_None)
    internal_selection: bool = False
    valid: bool = True  # whether the method claims post-selection validity
    needs: tuple[str, ...] = ()


def default_methods(
    family_size: Optional[int] = None,
    n_boot: int = 2000,
    log_prior: Optional[np.ndarray] = None,
    lo: float = 0.0,
    hi: float = 1.0,
    include: Optional[Sequence[str]] = None,
) -> list[MethodSpec]:
    """The standard comparison set used throughout the paper."""

    def naive(S, alpha, ctx):
        return B.naive_lcb(S, alpha), None

    def union_hoeff(S, alpha, ctx):
        return B.hoeffding_lcb(S, alpha, family_size, lo, hi), None

    def union_eb(S, alpha, ctx):
        return B.empirical_bernstein_lcb(S, alpha, family_size, lo, hi), None

    def union_best(S, alpha, ctx):
        return B.union_lcb(S, alpha, family_size, lo, hi), None

    def occam(S, alpha, ctx):
        return B.occam_lcb(S, log_prior, alpha, lo, hi), None

    def boot_max(S, alpha, ctx):
        lcb, q = B.bootstrap_max_lcb(
            S, alpha, n_boot=n_boot, seed=ctx["seed"], return_q=True
        )
        ctx["qhat"] = q
        return lcb, None

    def boot_max_cluster(S, alpha, ctx):
        lcb, q = B.cluster_bootstrap_max_lcb(
            S, ctx["clusters"], alpha, n_boot=n_boot, seed=ctx["seed"], return_q=True
        )
        ctx["qhat_cluster"] = q
        return lcb, None

    def _split(S, alpha, ctx, method):
        sel = ctx["selection"]
        res = B.split_lcb(
            S,
            alpha,
            frac_select=0.5,
            select_fn=lambda Ssel: sel.choose(Ssel.mean(axis=0)),
            method=method,
            seed=ctx["seed"],
            lo=lo,
            hi=hi,
        )
        return res.lcb_selected, res.selected

    def split(S, alpha, ctx):
        # asymptotic t-interval on the held-out half: the like-for-like
        # comparison with the (also asymptotic) bootstrap band
        return _split(S, alpha, ctx, "normal")

    def split_eb(S, alpha, ctx):
        # finite-sample version of the same idea
        return _split(S, alpha, ctx, "empirical_bernstein")

    def floored(S, alpha, ctx):
        lcb, q = B.floored_bootstrap_lcb(
            S, alpha, family_size=family_size, n_boot=n_boot, seed=ctx["seed"],
            lo=lo, hi=hi, return_q=True,
        )
        return lcb, None

    def conditional(S, alpha, ctx):
        res = B.conditional_winner_lcb(
            S, alpha, feasible=ctx["selection"].mask(S.shape[1])
        )
        return res.lcb_selected, res.selected

    def hybrid(S, alpha, ctx):
        res = B.hybrid_winner_lcb(
            S, alpha, n_boot=max(500, n_boot // 2), seed=ctx["seed"],
            feasible=ctx["selection"].mask(S.shape[1]),
        )
        return res.lcb_selected, res.selected

    specs = [
        MethodSpec("naive", naive, valid=False),
        MethodSpec("union-hoeffding", union_hoeff),
        MethodSpec("union-eb", union_eb),
        MethodSpec("union-best", union_best),
        MethodSpec("occam", occam, needs=("log_prior",)),
        MethodSpec("boot-max", boot_max),
        MethodSpec("boot-floored", floored),
        MethodSpec("boot-max-cluster", boot_max_cluster, needs=("clusters",)),
        MethodSpec("split", split, internal_selection=True),
        MethodSpec("split-eb", split_eb, internal_selection=True),
        MethodSpec("conditional", conditional, internal_selection=True),
        MethodSpec("hybrid", hybrid, internal_selection=True),
    ]
    if log_prior is None:
        specs = [s for s in specs if s.name != "occam"]
    if include is None:
        # the cluster-robust band needs cluster labels, so it is opt-in
        specs = [s for s in specs if s.name != "boot-max-cluster"]
    else:
        specs = [s for s in specs if s.name in include]
    return specs


# ---------------------------------------------------------------------------
# replicate driver
# ---------------------------------------------------------------------------


def _draw_indices(
    N: int,
    n: int,
    rng: np.random.Generator,
    scheme: str,
    clusters: Optional[np.ndarray],
    n_clusters_draw: Optional[int],
) -> np.ndarray:
    if scheme == "iid":
        return rng.integers(0, N, size=n)
    if scheme == "cluster":
        assert clusters is not None
        uniq = np.unique(clusters)
        members = {u: np.flatnonzero(clusters == u) for u in uniq}
        G = n_clusters_draw or uniq.size
        chosen = rng.choice(uniq, size=G, replace=True)
        per = int(np.ceil(n / G))
        out = []
        for g in chosen:
            idx = members[g]
            out.append(rng.choice(idx, size=per, replace=True))
        return np.concatenate(out)[:n]
    raise ValueError(scheme)


def run_replicates(
    S_pool: Optional[np.ndarray],
    n: int,
    R: int,
    methods: Sequence[MethodSpec],
    selection: Selection,
    alpha: float = 0.05,
    seed: int = 0,
    scheme: str = "iid",
    clusters: Optional[np.ndarray] = None,
    n_clusters_draw: Optional[int] = None,
    theta: Optional[np.ndarray] = None,
    progress: bool = False,
    sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None,
) -> dict:
    """Run ``R`` independent analyses and record what each method reports.

    Either pass a cached ``S_pool`` (which is then treated as the population and
    sampled from i.i.d.) or a ``sampler(n, rng)`` that draws a fresh score matrix
    directly from a generative model; in the latter case ``theta`` must be given.
    Sampling on the fly avoids materialising a pool, which matters when the
    hypothesis class is large.

    Returns a dict of arrays, all of shape ``(R,)`` per method, plus the shared
    per-replicate quantities (selected index, point estimate, optimism).
    """
    if sampler is not None:
        if theta is None:
            raise ValueError("theta must be supplied when sampling on the fly")
        N, m = -1, np.asarray(theta).size
    else:
        S_pool = np.asarray(S_pool)
        N, m = S_pool.shape
        if theta is None:
            theta = S_pool.mean(axis=0).astype(np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    theta_star = float(theta[selection.mask(m)].max())

    rng = np.random.default_rng(seed)
    out: dict = {
        "selected": np.zeros(R, dtype=np.int64),
        "point": np.zeros(R),
        "theta_selected": np.zeros(R),
        "optimism": np.zeros(R),
        "regret": np.zeros(R),
        "qhat": np.full(R, np.nan),
        "qhat_cluster": np.full(R, np.nan),
    }
    for ms in methods:
        out[f"lcb::{ms.name}"] = np.full(R, np.nan)
        out[f"cover::{ms.name}"] = np.zeros(R, dtype=bool)
        out[f"sel::{ms.name}"] = np.zeros(R, dtype=np.int64)
        # each method's own point estimate, so that "width" always compares
        # like with like even for methods that do their own selection
        out[f"point::{ms.name}"] = np.full(R, np.nan)

    for r in range(R):
        if sampler is not None:
            S = np.asarray(sampler(n, rng), dtype=np.float64)
            cl = None
        else:
            idx = _draw_indices(N, n, rng, scheme, clusters, n_clusters_draw)
            S = S_pool[idx].astype(np.float64)
            cl = None if clusters is None else np.asarray(clusters)[idx]
        point = S.mean(axis=0)
        chat = selection.choose(point)
        out["selected"][r] = chat
        out["point"][r] = point[chat]
        out["theta_selected"][r] = theta[chat]
        out["optimism"][r] = point[chat] - theta[chat]
        out["regret"][r] = theta_star - theta[chat]

        ctx = {"seed": seed * 100003 + r, "clusters": cl, "selection": selection}
        for ms in methods:
            res, sel = ms.fn(S, alpha, ctx)
            if ms.internal_selection:
                c = int(sel)
                lcb = float(res)
            else:
                c = chat
                lcb = float(np.asarray(res)[chat])
            out[f"lcb::{ms.name}"][r] = lcb
            out[f"sel::{ms.name}"][r] = c
            out[f"point::{ms.name}"][r] = point[c]
            out[f"cover::{ms.name}"][r] = lcb <= theta[c] + 1e-12
        if "qhat" in ctx:
            out["qhat"][r] = ctx["qhat"]
        if "qhat_cluster" in ctx:
            out["qhat_cluster"][r] = ctx["qhat_cluster"]
        if progress and (r + 1) % max(1, R // 10) == 0:
            print(f"    replicate {r + 1}/{R}", flush=True)

    out["_meta"] = {
        "n": n,
        "sampled_on_the_fly": sampler is not None,
        "R": R,
        "alpha": alpha,
        "m": m,
        "N": N,
        "scheme": scheme,
        "theta_star": theta_star,
        "seed": seed,
    }
    return out


def summarize(res: dict, methods: Sequence[MethodSpec], alpha: float = 0.05) -> list[dict]:
    """Coverage, mean certified bound and mean width for each method."""
    rows = []
    R = res["_meta"]["R"]
    for ms in methods:
        lcb = res[f"lcb::{ms.name}"]
        cov = res[f"cover::{ms.name}"]
        pt = res[f"point::{ms.name}"]
        sel_idx = res[f"sel::{ms.name}"]
        rows.append(
            {
                "method": ms.name,
                "claims_validity": ms.valid,
                "coverage": float(np.mean(cov)),
                "coverage_se": float(np.sqrt(np.mean(cov) * (1 - np.mean(cov)) / R)),
                "mean_lcb": float(np.mean(lcb)),
                "median_lcb": float(np.median(lcb)),
                "mean_width": float(np.mean(pt - lcb)),
                "mean_point": float(np.mean(pt)),
                "frac_same_selection": float(np.mean(sel_idx == res["selected"])),
                "frac_uninformative": float(np.mean(lcb <= 0.0)),
            }
        )
    rows.append(
        {
            "method": "_selection",
            "claims_validity": None,
            "coverage": float("nan"),
            "coverage_se": float("nan"),
            "mean_lcb": float("nan"),
            "median_lcb": float("nan"),
            "mean_width": float("nan"),
            "frac_uninformative": float("nan"),
            "mean_optimism": float(np.mean(res["optimism"])),
            "mean_point": float(np.mean(res["point"])),
            "mean_theta_selected": float(np.mean(res["theta_selected"])),
            "mean_regret": float(np.mean(res["regret"])),
        }
    )
    return rows
