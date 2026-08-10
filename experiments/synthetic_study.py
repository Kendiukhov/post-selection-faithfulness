"""E1: the anatomy of selection optimism, in a fully controlled setting.

Three questions, each answered with both an exact calculation and a simulation:

A. How badly does the usual "score +/- 1.96 SE" interval under-cover once the
   circuit was selected on the same interventions?  (Closed form under an
   equicorrelated Gaussian model, plus simulation with Bernoulli scores.)
B. How large is the optimism, and does it follow the predicted
   sigma * sqrt(2 log m / n) law?
C. Which correction should one use?  Coverage and *certified* faithfulness for
   every method, across class size, sample size and correlation.
"""

from __future__ import annotations

import argparse
import json
import os

# BLAS thread pools spin-wait; on a busy machine that can make a small mat-mul
# a thousand times slower than the single-threaded version.  The heavy linear
# algebra here is either tiny or runs on the accelerator, so pin the pools to
# one thread each.  Must happen before numpy is imported.
for _v in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psf import bounds as B  # noqa: E402
from psf import simulate  # noqa: E402
from psf.evaluate import Selection, default_methods, run_replicates, summarize  # noqa: E402


# ---------------------------------------------------------------------------
# A. exact naive coverage under an equicorrelated Gaussian model
# ---------------------------------------------------------------------------


def naive_coverage_exact(m: int, rho: float, alpha: float = 0.05, n_quad: int = 20001) -> float:
    """P(max_c G_c <= z_{1-alpha}) for an equicorrelated standard Gaussian
    vector, i.e. the coverage of the marginal lower bound at the arg-max when
    every hypothesis has the same true value.

    Conditioning on the common factor ``W`` makes the coordinates independent:
        P = E_W [ Phi( (z - sqrt(rho) W) / sqrt(1 - rho) )^m ].
    """
    z = stats.norm.ppf(1 - alpha)
    if rho <= 0:
        return float(stats.norm.cdf(z) ** m)
    w = np.linspace(-9, 9, n_quad)
    dens = stats.norm.pdf(w)
    inner = stats.norm.cdf((z - np.sqrt(rho) * w) / np.sqrt(1 - rho)) ** m
    return float(np.trapz(inner * dens, w))


def part_a(out: dict, alpha: float, seed: int) -> None:
    print("\n[A] naive coverage vs class size")
    ms = np.unique(np.round(np.logspace(0, 3.5, 22)).astype(int))
    rhos = [0.0, 0.3, 0.6, 0.9]
    exact = {str(r): [naive_coverage_exact(int(m), r, alpha) for m in ms] for r in rhos}

    # simulation with Bernoulli scores (the actual IIA setting)
    rng = np.random.default_rng(seed)
    sim = {}
    n, R = 500, 3000
    for r in [0.0, 0.3, 0.6, 0.9]:
        cov = []
        for m in [3, 10, 30, 100, 300, 1000]:
            model = simulate.equicorrelated_bernoulli(np.full(m, 0.8), rho=r)
            c = 0
            for _ in range(R):
                S = model.sample(n, rng)
                pt = S.mean(0)
                chat = int(np.argmax(pt))
                lcb = B.naive_lcb(S, alpha)[chat]
                c += lcb <= model.theta[chat]
            cov.append(c / R)
        sim[str(r)] = cov
    out["A"] = {
        "ms": ms.tolist(),
        "rhos": rhos,
        "exact": exact,
        "sim_ms": [3, 10, 30, 100, 300, 1000],
        "sim": sim,
        "n": n,
        "R": R,
        "alpha": alpha,
    }
    for r in rhos:
        print(f"  rho={r}: exact coverage at m=100 -> {naive_coverage_exact(100, r, alpha):.3f}")


# ---------------------------------------------------------------------------
# B. optimism scaling
# ---------------------------------------------------------------------------


def part_b(out: dict, seed: int) -> None:
    print("\n[B] optimism scaling")
    rng = np.random.default_rng(seed + 1)
    ns = [100, 200, 400, 800, 1600, 3200]
    ms = [2, 10, 50, 250, 1250]
    R = 800
    rows = []
    for rho in [0.0, 0.5]:
        for m in ms:
            theta = np.full(m, 0.8)
            model = simulate.equicorrelated_bernoulli(theta, rho=rho)
            sigma = float(np.sqrt(0.8 * 0.2))
            for n in ns:
                opt = np.empty(R)
                for r in range(R):
                    S = model.sample(n, rng)
                    pt = S.mean(0)
                    chat = int(np.argmax(pt))
                    opt[r] = pt[chat] - theta[chat]
                pred = sigma * np.sqrt(2 * np.log(max(m, 2)) / n)
                rows.append(
                    {
                        "rho": rho,
                        "m": m,
                        "n": n,
                        "optimism": float(opt.mean()),
                        "optimism_se": float(opt.std(ddof=1) / np.sqrt(R)),
                        "predicted_iid": pred,
                    }
                )
        print(f"  rho={rho} done")
    out["B"] = {"rows": rows, "R": R}


# ---------------------------------------------------------------------------
# C. method comparison
# ---------------------------------------------------------------------------


def part_c(out: dict, alpha: float, seed: int) -> None:
    print("\n[C] method comparison across regimes")
    rows = []
    configs = []
    for m in [20, 200, 2000]:
        for rho in [0.0, 0.5, 0.9]:
            configs.append((m, rho, 1000, "flat"))
    # a regime with a genuinely better hypothesis (clear winner)
    for rho in [0.0, 0.5, 0.9]:
        configs.append((200, rho, 1000, "winner"))

    R = 400
    for (m, rho, n, shape) in configs:
        theta = np.full(m, 0.80)
        if shape == "winner":
            theta[0] = 0.90
        model = simulate.equicorrelated_bernoulli(theta, rho=rho)
        methods = default_methods(family_size=m, n_boot=1200)
        t0 = time.time()
        res = run_replicates(
            None, n=n, R=R, methods=methods, selection=Selection("argmax"),
            alpha=alpha, seed=seed + m + n, theta=theta,
            sampler=lambda nn, rr, _m=model: _m.sample(nn, rr),
        )
        for row in summarize(res, methods, alpha):
            row.update({"m": m, "rho": rho, "n": n, "shape": shape})
            rows.append(row)
        qh = float(np.nanmean(res["qhat"]))
        Sd = model.sample(n, np.random.default_rng(seed * 31 + m + n))
        q_max, q_marg, kappa = B.multiplicity_factor(Sd, alpha, n_boot=2000, seed=1)
        rows.append(
            {
                "method": "_diag", "m": m, "rho": rho, "n": n, "shape": shape,
                "qhat": qh, "q_marginal": q_marg, "kappa": kappa,
                "m_eff": B.effective_num_hypotheses(q_max, alpha, q_marginal=q_marg),
                "bonferroni_q": float(stats.norm.ppf(1 - alpha / m)),
                "mean_optimism": float(res["optimism"].mean()),
            }
        )
        print(
            f"  m={m:5d} rho={rho} n={n:5d} {shape:6s}  "
            f"qhat={qh:.2f} (bonf {stats.norm.ppf(1 - alpha / m):.2f})  "
            f"opt={res['optimism'].mean():.4f}  ({time.time() - t0:.0f}s)"
        )
    out["C"] = {"rows": rows, "R": R, "alpha": alpha}


# ---------------------------------------------------------------------------
# D. does the bootstrap need to recompute the variance?
# ---------------------------------------------------------------------------


def part_d(out: dict, alpha: float, seed: int) -> None:
    """Fixed-sigma multiplier bootstrap versus the empirical bootstrap-t.

    For binary scores the sample mean and sample standard deviation are
    dependent, so a bootstrap that holds sigma-hat fixed under-states the upper
    tail of the studentised maximum.  This part measures the cost.
    """
    print("\n[D] fixed-sigma multiplier bootstrap vs bootstrap-t")
    rows = []
    R = 500
    for (m, n, rho, p) in [
        (60, 400, 0.0, 0.75), (60, 400, 0.6, 0.75), (300, 400, 0.0, 0.75),
        (60, 1500, 0.0, 0.75), (60, 400, 0.0, 0.95), (60, 400, 0.0, 0.50),
    ]:
        model = simulate.equicorrelated_bernoulli(np.full(m, p), rho=rho)
        rng = np.random.default_rng(seed + m + n + int(100 * rho) + int(100 * p))
        cov_t = cov_f = 0
        w_t, w_f = [], []
        for r in range(R):
            S = model.sample(n, rng)
            chat = int(S.mean(0).argmax())
            lt = B.bootstrap_max_lcb(S, alpha, n_boot=1000, seed=r)[chat]
            lf = B.bootstrap_max_lcb(
                S, alpha, n_boot=1000, seed=r, recompute_variance=False
            )[chat]
            cov_t += lt <= p
            cov_f += lf <= p
            w_t.append(S.mean(0)[chat] - lt)
            w_f.append(S.mean(0)[chat] - lf)
        rows.append({
            "m": m, "n": n, "rho": rho, "theta": p,
            "coverage_bootstrap_t": cov_t / R, "coverage_fixed_sigma": cov_f / R,
            "width_bootstrap_t": float(np.mean(w_t)),
            "width_fixed_sigma": float(np.mean(w_f)),
        })
        print(f"  m={m:4d} n={n:5d} rho={rho} theta={p}: "
              f"bootstrap-t {cov_t / R:.3f}  fixed-sigma {cov_f / R:.3f}")
    out["D"] = {"rows": rows, "R": R, "alpha": alpha}


# ---------------------------------------------------------------------------
# E. how many clusters does template-level inference need?
# ---------------------------------------------------------------------------


def part_e(out: dict, alpha: float, seed: int) -> None:
    """Coverage of the cluster-robust band as a function of the number of
    clusters, under a one-way random-effects model with a strong cluster effect.

    This is the question a practitioner faces when deciding how many prompt
    templates to write."""
    print("\n[E] cluster-robust coverage vs number of clusters")
    rows = []
    R, per, m, p = 300, 40, 30, 0.7
    for G in [10, 15, 25, 50, 100]:
        n = G * per
        rng = np.random.default_rng(seed + G)
        cov_cl = cov_iid = 0
        for r in range(R):
            u = rng.normal(0.0, 1.0, size=(G, 1))
            eps = rng.normal(0.0, 1.0, size=(n, m))
            z = np.repeat(u, per, axis=0) + eps
            S = (z > stats.norm.ppf(1 - p) * np.sqrt(2.0)).astype(float)
            cl = np.repeat(np.arange(G), per)
            chat = int(S.mean(0).argmax())
            cov_iid += B.bootstrap_max_lcb(S, alpha, n_boot=600, seed=r)[chat] <= p
            cov_cl += (
                B.cluster_bootstrap_max_lcb(S, cl, alpha, n_boot=600, seed=r)[chat] <= p
            )
        rows.append({
            "G": G, "n": n, "per_cluster": per, "icc_design": 0.5,
            "coverage_cluster": cov_cl / R, "coverage_iid": cov_iid / R,
        })
        print(f"  G={G:4d} n={n:5d}: cluster band {cov_cl / R:.3f}   i.i.d. band {cov_iid / R:.3f}")
    out["E"] = {"rows": rows, "R": R, "alpha": alpha}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/synthetic")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", default="ABCDE")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "synthetic.json")
    out: dict = json.load(open(path)) if os.path.exists(path) else {}

    def flush():
        with open(path, "w") as f:
            json.dump(out, f, indent=1)

    if "A" in args.parts:
        part_a(out, args.alpha, args.seed)
        flush()
    if "B" in args.parts:
        part_b(out, args.seed)
        flush()
    if "C" in args.parts:
        part_c(out, args.alpha, args.seed)
        flush()
    if "D" in args.parts:
        part_d(out, args.alpha, args.seed)
        flush()
    if "E" in args.parts:
        part_e(out, args.alpha, args.seed)
        flush()
    print("\nwrote", path)


if __name__ == "__main__":
    main()
