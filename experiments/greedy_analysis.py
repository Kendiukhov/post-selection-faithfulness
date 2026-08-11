"""Analysis of the GPT-2 greedy pruning run produced by ``ioi_greedy.py``."""

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

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psf.adaptive import reachable_log_size  # noqa: E402


def nld(ld_col, ld_empty, ld_full):
    return (ld_col.mean() - ld_empty.mean()) / (ld_full.mean() - ld_empty.mean())


def nld_influence(ld_col, ld_empty, ld_full):
    a, b, f = ld_col.mean(), ld_empty.mean(), ld_full.mean()
    F = (a - b) / (f - b)
    psi = ((ld_col - a) - (ld_empty - b)) / (f - b) - F * ((ld_full - f) - (ld_empty - b)) / (f - b)
    return float(F), psi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/ioi/ioi_greedy.npz")
    ap.add_argument("--meta", default="results/ioi/greedy_meta.json")
    ap.add_argument("--out", default="results/ioi/greedy_analysis.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target", type=float, default=0.90,
                    help="stop at the smallest circuit whose search score is at least this")
    args = ap.parse_args()

    d = np.load(args.path, allow_pickle=True)
    meta = json.load(open(args.meta))
    ld_s, ld_e = d["ld_search"].astype(np.float64), d["ld_eval"].astype(np.float64)
    fs, es = d["ld_full_search"].astype(np.float64), d["ld_empty_search"].astype(np.float64)
    fe, ee = d["ld_full_eval"].astype(np.float64), d["ld_empty_eval"].astype(np.float64)
    sizes = d["sizes"].astype(int)
    P = ld_s.shape[1]

    search_curve = np.array([nld(ld_s[:, p], es, fs) for p in range(P)])
    eval_curve = np.array([nld(ld_e[:, p], ee, fe) for p in range(P)])

    # a realistic stopping rule: the smallest circuit on the path that still
    # reaches the target score on the data the search itself used
    ok = np.flatnonzero(search_curve >= args.target)
    p_star = int(ok[-1]) if ok.size else int(np.argmax(search_curve))
    reported = float(search_curve[p_star])
    truth = float(eval_curve[p_star])

    n_s, n_e = ld_s.shape[0], ld_e.shape[0]
    z = st.norm.ppf(1 - args.alpha)

    F_s, psi_s = nld_influence(ld_s[:, p_star], es, fs)
    se_s = float(np.sqrt((psi_s**2).sum() / (n_s * (n_s - 1))))
    F_e, psi_e = nld_influence(ld_e[:, p_star], ee, fe)
    se_e = float(np.sqrt((psi_e**2).sum() / (n_e * (n_e - 1))))

    d = np.load(args.path, allow_pickle=True)
    n_pool = int(sizes[0])
    n_removed = int(sizes[0] - sizes[p_star])
    log_reach = reachable_log_size(n_pool, n_removed)
    tail = args.alpha * np.exp(-log_reach)
    q_reach = float(st.norm.isf(max(tail, 1e-300)))

    # a fresh holdout of the same size as the search set
    rng = np.random.default_rng(0)
    sub = rng.choice(n_e, size=min(n_s, n_e), replace=False)
    F_sub, psi_sub = nld_influence(ld_e[sub, p_star], ee[sub], fe[sub])
    se_sub = float(np.sqrt((psi_sub**2).sum() / (sub.size * (sub.size - 1))))

    bounds = [
        {"name": "naive bound on the search data", "lcb": F_s - z * se_s,
         "note": "\\textbf{invalid}: the circuit was chosen on these interventions",
         "color": "#B3282D"},
        {"name": ("uniform over the $10^{%d}$ circuits the search could have reached"
                  % round(log_reach / np.log(10))),
         "lcb": F_s - q_reach * se_s,
         "note": "valid, but vacuous from the search data alone", "color": "#8C8C8C"},
        {"name": f"held-out, {sub.size} fresh interventions", "lcb": F_sub - z * se_sub,
         "note": "valid", "color": "#4C72B0"},
        {"name": f"held-out, {n_e} fresh interventions", "lcb": F_e - z * se_e,
         "note": "valid", "color": "#55A868"},
    ]

    out = {
        "n_pool": n_pool,
        "sizes": sizes.tolist(),
        "search_curve": search_curve.tolist(),
        "eval_curve": eval_curve.tolist(),
        "p_star": p_star,
        "final_size": int(sizes[p_star]),
        "reported": reported,
        "truth": truth,
        "gap": reported - truth,
        "n_search": int(n_s),
        "n_eval": int(n_e),
        "n_queries": int(meta["n_queries"]),
        "log_reachable": float(log_reach),
        "q_reachable": q_reach,
        "bounds": bounds,
        "target": args.target,
        "max_gap_on_path": float(np.max(search_curve - eval_curve)),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("sizes", "search_curve", "eval_curve")}, indent=2))


if __name__ == "__main__":
    main()
