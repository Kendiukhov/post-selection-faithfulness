"""E2: post-selection inference on the tiny transformers.

Every number here is computed from a cached score matrix, so the analysis is
CPU-only and exactly reproducible.  The cached pool is treated as the
population: analysis samples of size ``n`` are drawn i.i.d. from it, which makes
the true faithfulness of every hypothesis known exactly.
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
from scipy import stats as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psf import bounds as B  # noqa: E402
from psf.evaluate import Selection, default_methods, run_replicates, summarize  # noqa: E402
from psf.functional import influence_band, ratio_influence  # noqa: E402


def alignment_experiment(path: str, tag: str, ns, R, alpha, seed, planted=None,
                         max_dims=None):
    """``max_dims`` restricts the class to subspaces of at most that dimension,
    which is what an investigator searching for a *localised* representation
    would pre-register.  Restricting the class removes the trivially-best
    whole-residual-stream hypothesis and leaves a set of genuine near-ties, which
    is the regime in which selection actually bites."""
    d = np.load(path, allow_pickle=True)
    S_pool = d["S"].astype(np.float64)
    labels = d["labels"]
    if max_dims is not None:
        keep = np.flatnonzero(d["ndim"] <= max_dims)
        S_pool = np.ascontiguousarray(S_pool[:, keep])
        labels = labels[keep]
    N, m = S_pool.shape
    theta = S_pool.mean(axis=0)
    best = int(np.argmax(theta))
    print(f"\n[{tag}] N={N}, |C|={m}; best hypothesis {labels[best]} with theta={theta[best]:.4f}")
    print(f"        runner-up gap = {theta[best] - np.sort(theta)[-2]:.4f}; "
          f"#within 0.01 of best = {(theta >= theta[best] - 0.01).sum()}")

    out = {
        "tag": tag, "N": int(N), "m": int(m), "alpha": alpha,
        "theta_best": float(theta[best]), "best_label": str(labels[best]),
        "n_near_best": int((theta >= theta[best] - 0.01).sum()),
        "n_perfect_columns": int((theta >= 1.0 - 1e-12).sum()),
        "runner_up_gap": float(theta[best] - np.sort(theta)[-2]),
        "rows": [], "diag": [],
    }
    if planted is not None:
        out["planted_label"] = planted
        idx = [i for i, l in enumerate(labels) if l == planted]
        if idx:
            out["planted_theta"] = float(theta[idx[0]])
            out["planted_rank"] = int((theta > theta[idx[0]]).sum() + 1)

    methods = default_methods(family_size=m, n_boot=2000)
    for n in ns:
        t0 = time.time()
        res = run_replicates(
            S_pool, n=n, R=R, methods=methods, selection=Selection("argmax"),
            alpha=alpha, seed=seed + n, theta=theta,
        )
        for row in summarize(res, methods, alpha):
            row.update({"n": n, "tag": tag})
            out["rows"].append(row)
        qh = float(np.nanmean(res["qhat"]))
        sel_correct = float(np.mean(res["selected"] == best))
        rng_d = np.random.default_rng(seed * 31 + n)
        q_max, q_marg, kappa = B.multiplicity_factor(
            S_pool[rng_d.integers(0, N, size=n)], alpha, n_boot=4000, seed=1
        )
        out["diag"].append(
            {
                "n": n, "qhat": qh, "q_marginal": q_marg, "kappa": kappa,
                "m_eff": B.effective_num_hypotheses(q_max, alpha, q_marginal=q_marg),
                "bonferroni_q": float(st.norm.ppf(1 - alpha / m)),
                "mean_optimism": float(res["optimism"].mean()),
                "mean_point": float(res["point"].mean()),
                "mean_theta_selected": float(res["theta_selected"].mean()),
                "prob_select_argmax": sel_correct,
            }
        )
        print(
            f"  n={n:5d}  optimism={res['optimism'].mean():+.4f}  "
            f"qhat={qh:.2f}  P(pick true best)={sel_correct:.2f}  ({time.time() - t0:.0f}s)"
        )
    return out


def component_experiment(path: str, tag: str, ns, R, alpha, seed, n_components: int):
    d = np.load(path, allow_pickle=True)
    agree = d["agree"].astype(np.float64)
    sizes = d["sizes"]
    labels = d["labels"]
    N, m = agree.shape
    theta = agree.mean(axis=0)
    best = int(np.argmax(theta))
    print(f"\n[{tag}] N={N}, |C|={m}; best circuit {labels[best]} theta={theta[best]:.4f}")

    log_prior = B.size_stratified_log_prior(sizes, n_components)
    out = {
        "tag": tag, "N": int(N), "m": int(m), "alpha": alpha,
        "theta_best": float(theta[best]), "best_label": str(labels[best]),
        "n_perfect_columns": int((theta >= 1.0 - 1e-12).sum()),
        "n_near_best": int((theta >= theta[best] - 0.01).sum()),
        "rows": [], "diag": [], "size_rows": [],
    }
    methods = default_methods(family_size=m, n_boot=2000, log_prior=log_prior)

    for n in ns:
        res = run_replicates(
            agree, n=n, R=R, methods=methods, selection=Selection("argmax"),
            alpha=alpha, seed=seed + n, theta=theta,
        )
        for row in summarize(res, methods, alpha):
            row.update({"n": n, "tag": tag, "selection": "argmax"})
            out["rows"].append(row)
        qh = float(np.nanmean(res["qhat"]))
        rng_d = np.random.default_rng(seed * 31 + n)
        q_max, q_marg, kappa = B.multiplicity_factor(
            agree[rng_d.integers(0, N, size=n)], alpha, n_boot=4000, seed=1
        )
        out["diag"].append(
            {"n": n, "qhat": qh, "q_marginal": q_marg, "kappa": kappa,
             "m_eff": B.effective_num_hypotheses(q_max, alpha, q_marginal=q_marg),
             "mean_optimism": float(res["optimism"].mean())}
        )
        print(f"  argmax     n={n:5d}  optimism={res['optimism'].mean():+.4f} qhat={qh:.2f}")

    # size-constrained selection: "smallest circuit that still works"
    for kmax in sorted(set(int(x) for x in [2, 3, 4] if x <= n_components)):
        sel = Selection("argmax_size_le", max_size=kmax, sizes=sizes)
        for n in ns:
            res = run_replicates(
                agree, n=n, R=R, methods=methods, selection=sel,
                alpha=alpha, seed=seed + n + 7 * kmax, theta=theta,
            )
            for row in summarize(res, methods, alpha):
                row.update({"n": n, "tag": tag, "selection": f"size<={kmax}"})
                out["size_rows"].append(row)
        print(f"  size<={kmax}   done")

    # ratio metric (normalised logit difference recovered)
    ld, ld_full, ld_empty = d["ld"].astype(np.float64), d["ld_full"], d["ld_empty"]
    Fhat_pool, _ = ratio_influence(ld, ld_empty, ld_full)
    out["ratio"] = ratio_replicates(ld, ld_empty, ld_full, Fhat_pool, ns, R, alpha, seed, m)
    return out


def ratio_replicates(ld, ld_empty, ld_full, theta_pool, ns, R, alpha, seed, m):
    """Coverage of the influence-function bootstrap band for the ratio metric."""
    N = ld.shape[0]
    rows = []
    for n in ns:
        rng = np.random.default_rng(seed + n)
        cov_boot = cov_naive = 0
        w_boot = []
        w_naive = []
        opt = []
        for r in range(R):
            idx = rng.integers(0, N, size=n)
            F, psi = ratio_influence(ld[idx], ld_empty[idx], ld_full[idx])
            chat = int(np.argmax(F))
            opt.append(F[chat] - theta_pool[chat])
            lcb, q = influence_band(F, psi, alpha=alpha, n_boot=1000, seed=seed * 7 + r)
            cov_boot += lcb[chat] <= theta_pool[chat]
            w_boot.append(F[chat] - lcb[chat])
            se = np.sqrt((psi**2).sum(axis=0) / (n * (n - 1)))
            nl = F - st.norm.ppf(1 - alpha) * se
            cov_naive += nl[chat] <= theta_pool[chat]
            w_naive.append(F[chat] - nl[chat])
        rows.append(
            {
                "n": n,
                "coverage_boot": cov_boot / R,
                "coverage_naive": cov_naive / R,
                "width_boot": float(np.mean(w_boot)),
                "width_naive": float(np.mean(w_naive)),
                "mean_optimism": float(np.mean(opt)),
            }
        )
        print(f"  ratio      n={n:5d}  cov(boot)={cov_boot / R:.3f} cov(naive)={cov_naive / R:.3f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/tiny")
    ap.add_argument("--out", default="results/tiny")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--R", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ns = [125, 250, 500, 1000, 2000, 4000]
    res = {}

    configs = [
        # (output tag, score file, planted label, max subspace dimension)
        ("tt_a", "tt_a", None, None),
        ("tt_a_local", "tt_a", None, 16),
        ("tt_a_iit", "tt_a_iit", "resid_post.0@p2:std[0:16]", None),
        ("tt_c", "tt_c", None, None),
    ]
    for tag, fname, planted, md in configs:
        p = os.path.join(args.scores, f"{fname}_scores.npz")
        if os.path.exists(p):
            res[tag] = alignment_experiment(
                p, tag, ns, args.R, args.alpha, args.seed, planted=planted, max_dims=md
            )

    p = os.path.join(args.scores, "tt_b_scores.npz")
    if os.path.exists(p):
        res["tt_b"] = component_experiment(
            p, "tt_b", ns, args.R, args.alpha, args.seed, n_components=8
        )

    with open(os.path.join(args.out, "tiny_analysis.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", os.path.join(args.out, "tiny_analysis.json"))


if __name__ == "__main__":
    main()
