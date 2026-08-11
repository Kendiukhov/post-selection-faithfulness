"""E3a: post-selection inference for the GPT-2 small IOI circuit.

Runs on the cached score matrix produced by ``ioi_score_matrix.py``.  The 6000
cached prompts are treated as the population, so the true faithfulness of every
one of the 840 circuit hypotheses is known exactly.

Sections
  1. descriptives (does GPT-2 do IOI? how faithful is the published circuit?)
  2. i.i.d. instance sampling: optimism, coverage and certified value
  3. template-clustered sampling: what happens when the target population is
     "new templates" rather than "new fills of these templates"
  4. certifying the whole size/faithfulness Pareto frontier at once
  5. pre-registered versus searched hypotheses
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
from psf.functional import influence_band, ratio_band, ratio_influence  # noqa: E402


def icc(values: np.ndarray, clusters: np.ndarray) -> float:
    """One-way random-effects intra-cluster correlation of a score column."""
    uniq = np.unique(clusters)
    G = uniq.size
    groups = [values[clusters == g] for g in uniq]
    k = np.mean([g.size for g in groups])
    gm = values.mean()
    msb = sum(g.size * (g.mean() - gm) ** 2 for g in groups) / max(G - 1, 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / max(values.size - G, 1)
    if msb + (k - 1) * msw <= 0:
        return 0.0
    return float(max(0.0, (msb - msw) / (msb + (k - 1) * msw)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/ioi/ioi_scores.npz")
    ap.add_argument("--out", default="results/ioi")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.scores, allow_pickle=True)
    acc = d["acc"].astype(np.float64)
    ld, ld_full, ld_empty = d["ld"].astype(np.float64), d["ld_full"], d["ld_empty"]
    template = d["template"].astype(int)
    sizes = d["sizes"].astype(int)
    labels = d["labels"]
    N, m = acc.shape
    theta = acc.mean(axis=0)
    F_pool, _ = ratio_influence(ld, ld_empty, ld_full)
    out: dict = {"N": int(N), "m": int(m), "alpha": args.alpha, "R": args.R}

    # ---------------------------------------------------------------- 1
    # locate the published circuit and the full model by their head sets: the
    # class builder de-duplicates, so these may carry a different label
    circ = json.load(open(os.path.join(os.path.dirname(args.scores), "circuits.json")))
    head_sets = [frozenset(map(tuple, c["heads"])) for c in circ]
    from psf.ioi import ioi_circuit_heads

    pub_set = frozenset(ioi_circuit_heads())
    full_set = frozenset((l, h) for l in range(12) for h in range(12))
    pub = head_sets.index(pub_set)
    full = head_sets.index(full_set)
    out["descriptives"] = {
        "n_templates": int(np.unique(template).size),
        "ld_full_mean": float(ld_full.mean()),
        "ld_empty_mean": float(ld_empty.mean()),
        "full_model_accuracy": float((ld_full > 0).mean()),
        "published_circuit_size": int(sizes[pub]),
        "published_circuit_accuracy": float(theta[pub]),
        "published_circuit_nld": float(F_pool[pub]),
        "published_circuit_raw_ratio": float(ld[:, pub].mean() / ld_full.mean()),
        "best_accuracy_label": str(labels[int(np.argmax(theta))]),
        "best_accuracy": float(theta.max()),
        "best_nld_label": str(labels[int(np.argmax(F_pool))]),
        "best_nld": float(F_pool.max()),
        "sanity_all_heads_nld": float(F_pool[full]),
    }
    print(json.dumps(out["descriptives"], indent=1))

    # correlation structure of the hypothesis class
    sub = np.random.default_rng(0).choice(m, size=min(m, 200), replace=False)
    C = np.corrcoef(acc[:, sub].T)
    out["descriptives"]["median_pairwise_corr"] = float(
        np.median(C[np.triu_indices_from(C, k=1)])
    )
    print("median pairwise correlation of per-instance scores:",
          round(out["descriptives"]["median_pairwise_corr"], 3))

    # ---------------------------------------------------------------- 2
    print("\n[2] i.i.d. instance sampling")
    log_prior = B.size_stratified_log_prior(sizes, 144)
    methods = default_methods(family_size=m, n_boot=2000, log_prior=log_prior)
    ns = [250, 500, 1000, 2000]
    rows, diag = [], []
    for n in ns:
        t0 = time.time()
        res = run_replicates(
            acc, n=n, R=args.R, methods=methods, selection=Selection("argmax"),
            alpha=args.alpha, seed=args.seed + n, theta=theta,
        )
        for row in summarize(res, methods, args.alpha):
            row.update({"n": n, "scheme": "iid", "selection": "argmax"})
            rows.append(row)
        qh = float(np.nanmean(res["qhat"]))
        rng_d = np.random.default_rng(args.seed * 31 + n)
        q_max, q_marg, kappa = B.multiplicity_factor(
            acc[rng_d.integers(0, N, size=n)], args.alpha, n_boot=4000, seed=1
        )
        diag.append({
            "n": n, "scheme": "iid", "qhat": qh,
            "q_marginal": q_marg, "kappa": kappa,
            "m_eff": B.effective_num_hypotheses(q_max, args.alpha, q_marginal=q_marg),
            "bonferroni_q": float(st.norm.ppf(1 - args.alpha / m)),
            "mean_optimism": float(res["optimism"].mean()),
            "mean_point": float(res["point"].mean()),
            "mean_theta_selected": float(res["theta_selected"].mean()),
        })
        print(f"  n={n:5d} optimism={res['optimism'].mean():+.4f} qhat={qh:.2f} "
              f"kappa={kappa:.2f} "
              f"m_eff={B.effective_num_hypotheses(q_max, args.alpha, q_marginal=q_marg):.0f} "
              f"({time.time() - t0:.0f}s)")

    # size-constrained selection ("smallest circuit that still works")
    for kmax in [8, 16, 26]:
        sel = Selection("argmax_size_le", max_size=kmax, sizes=sizes)
        for n in [1000]:
            res = run_replicates(
                acc, n=n, R=args.R, methods=methods, selection=sel,
                alpha=args.alpha, seed=args.seed + n + kmax, theta=theta,
            )
            for row in summarize(res, methods, args.alpha):
                row.update({"n": n, "scheme": "iid", "selection": f"size<={kmax}"})
                rows.append(row)
        print(f"  size<={kmax} done")
    out["iid"] = {"rows": rows, "diag": diag}

    # ---------------------------------------------------------------- 3
    print("\n[3] template-clustered sampling")
    G = int(np.unique(template).size)
    iccs = [icc(acc[:, j], template) for j in range(0, m, 7)]
    out["cluster"] = {
        "n_templates": G,
        "median_icc_accuracy": float(np.median(iccs)),
        "icc_published": float(icc(acc[:, pub], template)),
        "rows": [], "diag": [],
    }
    sizes_per_template = np.bincount(template)
    balanced = bool(np.all(sizes_per_template[sizes_per_template > 0] ==
                           sizes_per_template[sizes_per_template > 0][0]))
    theta_cluster = np.stack(
        [acc[template == u].mean(axis=0) for u in np.unique(template)]
    ).mean(axis=0)
    out["cluster"]["balanced_templates"] = balanced
    out["cluster"]["max_abs_theta_shift"] = float(np.abs(theta_cluster - theta).max())
    print(f"  median ICC across circuits: {np.median(iccs):.3f} (G = {G} templates); "
          f"templates balanced: {balanced}")
    # The finite-sample union bounds assume i.i.d. instances, so they are not
    # valid under cluster sampling and are excluded here.  We also use the
    # size-constrained selection: the unconstrained arg-max is a circuit that is
    # perfect on every sampled prompt, and its near-degenerate variance would
    # dominate the comparison we are trying to make.
    cl_methods = default_methods(
        family_size=m, n_boot=2000, log_prior=log_prior,
        include=("naive", "boot-max", "boot-max-cluster"),
    )
    cl_sel = Selection("argmax_size_le", max_size=8, sizes=sizes)
    out["cluster"]["selection"] = "size<=8"
    for n in [500, 1000, 2000]:
        res = run_replicates(
            acc, n=n, R=args.R, methods=cl_methods, selection=cl_sel,
            alpha=args.alpha, seed=args.seed + 31 + n, theta=theta_cluster,
            scheme="cluster", clusters=template, n_clusters_draw=G,
        )
        for row in summarize(res, cl_methods, args.alpha):
            row.update({"n": n, "scheme": "cluster", "selection": "argmax"})
            out["cluster"]["rows"].append(row)
        out["cluster"]["diag"].append({
            "n": n,
            "qhat_iid": float(np.nanmean(res["qhat"])),
            "qhat_cluster": float(np.nanmean(res["qhat_cluster"])),
            "design_effect": float(1 + (n / G - 1) * np.median(iccs)),
            "mean_optimism": float(res["optimism"].mean()),
        })
        print(f"  n={n:5d} qhat_iid={np.nanmean(res['qhat']):.2f} "
              f"qhat_cluster={np.nanmean(res['qhat_cluster']):.2f} "
              f"DEFF={1 + (n / G - 1) * np.median(iccs):.1f}")

    # ---------------------------------------------------------------- 4
    print("\n[4] certifying the size/faithfulness frontier")
    n = 1000
    rng = np.random.default_rng(args.seed + 999)
    idx = rng.integers(0, N, size=n)
    S = acc[idx]
    point = S.mean(axis=0)
    lcb_boot, q = B.bootstrap_max_lcb(S, args.alpha, n_boot=4000, seed=1, return_q=True)
    lcb_occ = B.occam_lcb(S, log_prior, args.alpha)
    lcb_naive = B.naive_lcb(S, args.alpha)
    ks = sorted(set(int(x) for x in np.unique(sizes) if x <= 60))
    frontier = []
    for k in ks:
        mk = sizes <= k
        if not mk.any():
            continue
        j = int(np.flatnonzero(mk)[np.argmax(point[mk])])
        frontier.append({
            "k": k, "label": str(labels[j]), "size": int(sizes[j]),
            "naive_point": float(point[j]), "naive_lcb": float(lcb_naive[j]),
            "boot_lcb": float(lcb_boot[j]), "occam_lcb": float(lcb_occ[j]),
            "truth": float(theta[j]),
        })
    out["frontier"] = {"n": n, "qhat": q, "rows": frontier}
    bad = [r for r in frontier if r["naive_lcb"] > r["truth"]]
    print(f"  frontier points where the naive lower bound exceeds the truth: "
          f"{len(bad)}/{len(frontier)}")
    print(f"  frontier points where the simultaneous band fails: "
          f"{sum(r['boot_lcb'] > r['truth'] for r in frontier)}/{len(frontier)}")

    # how often does the *whole* frontier stay inside the band?
    cover_all = 0
    Rf = 200
    for r in range(Rf):
        ii = rng.integers(0, N, size=n)
        Sx = acc[ii]
        lb = B.bootstrap_max_lcb(Sx, args.alpha, n_boot=1500, seed=1000 + r)
        pt = Sx.mean(axis=0)
        ok = True
        for k in ks:
            mk = sizes <= k
            j = int(np.flatnonzero(mk)[np.argmax(pt[mk])])
            if lb[j] > theta[j]:
                ok = False
                break
        cover_all += ok
    out["frontier"]["simultaneous_frontier_coverage"] = cover_all / Rf
    print(f"  probability the entire certified frontier is valid: {cover_all / Rf:.3f}")

    # ---------------------------------------------------------------- 5
    print("\n[5] pre-registered vs searched")
    pre_rows = []
    for n in ns:
        cov_pre = cov_sea = 0
        lcb_pre, lcb_sea = [], []
        for r in range(args.R):
            ii = np.random.default_rng(args.seed * 13 + 7 * r + n).integers(0, N, size=n)
            Sx = acc[ii]
            pt = Sx.mean(axis=0)
            l1 = B.empirical_bernstein_lcb(Sx[:, [pub]], args.alpha, family_size=1)[0]
            cov_pre += l1 <= theta[pub]
            lcb_pre.append(l1)
            j = int(np.argmax(pt))
            l2 = B.empirical_bernstein_lcb(Sx, args.alpha, family_size=m)[j]
            cov_sea += l2 <= theta[j]
            lcb_sea.append(l2)
        pre_rows.append({
            "n": n,
            "pre_registered_coverage": cov_pre / args.R,
            "pre_registered_mean_lcb": float(np.mean(lcb_pre)),
            "searched_coverage": cov_sea / args.R,
            "searched_mean_lcb": float(np.mean(lcb_sea)),
        })
        print(f"  n={n:5d} pre-registered LCB={np.mean(lcb_pre):.4f} "
              f"searched LCB={np.mean(lcb_sea):.4f}")
    out["preregistration"] = pre_rows

    # ---------------------------------------------------------------- ratio metric
    print("\n[6] normalised logit difference recovered (ratio metric)")
    ratio_rows = []
    for n in ns:
        rng2 = np.random.default_rng(args.seed + 5000 + n)
        cb = cn = 0
        wb, wn, op = [], [], []
        for r in range(args.R):
            ii = rng2.integers(0, N, size=n)
            F, psi = ratio_influence(ld[ii], ld_empty[ii], ld_full[ii])
            j = int(np.argmax(F))
            op.append(F[j] - F_pool[j])
            _, lb, _ = ratio_band(
                ld[ii], ld_empty[ii], ld_full[ii], alpha=args.alpha, n_boot=600, seed=r
            )
            cb += lb[j] <= F_pool[j]
            wb.append(F[j] - lb[j])
            se = np.sqrt((psi**2).sum(axis=0) / (n * (n - 1)))
            nl = F - st.norm.ppf(1 - args.alpha) * se
            cn += nl[j] <= F_pool[j]
            wn.append(F[j] - nl[j])
        ratio_rows.append({
            "n": n, "coverage_boot": cb / args.R, "coverage_naive": cn / args.R,
            "width_boot": float(np.mean(wb)), "width_naive": float(np.mean(wn)),
            "mean_optimism": float(np.mean(op)),
        })
        print(f"  n={n:5d} cov(boot)={cb / args.R:.3f} cov(naive)={cn / args.R:.3f} "
              f"optimism={np.mean(op):+.4f}")
    out["ratio"] = ratio_rows

    with open(os.path.join(args.out, "ioi_analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", os.path.join(args.out, "ioi_analysis.json"))


if __name__ == "__main__":
    main()
