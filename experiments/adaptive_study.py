"""E4: adaptive circuit search, and whether a holdout can be reused.

A greedy pruner is run on a real (small) transformer over 36 ablatable
components, issuing hundreds of data-dependent queries.  Four ways of answering
those queries are compared, each repeated over many independent replications so
that bias and coverage are measured, not asserted:

  M1  same data          search on D_train, report the D_train score
  M2  holdout reused     search directly on D_holdout, report the D_holdout score
  M3  Thresholdout       search through the reusable-holdout mechanism of
                         Dwork et al. (2015), report the mechanism's answer
  M4  fresh holdout      search on D_train, report on an untouched D_holdout

The truth for the returned circuit is computed exactly on a large pool that
defines the population.
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
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psf import bounds as B  # noqa: E402
from psf.adaptive import Thresholdout, reachable_log_size  # noqa: E402
from psf.interventions import mean_ablation_cache  # noqa: E402
from psf.models import TinyConfig, TinyTransformer  # noqa: E402
from psf.tasks import InductionTask  # noqa: E402


class CircuitEvaluator:
    """Per-instance behavioural agreement of a component subset, on a fixed set
    of instances, for a tiny transformer."""

    def __init__(self, model, tokens, abl, eval_pos, device, batch=20000):
        self.model, self.device, self.batch = model, device, batch
        self.tokens = torch.tensor(tokens, device=device)
        self.pos = torch.tensor(eval_pos, device=device, dtype=torch.long)
        self.abl = abl
        self.L, self.H = model.cfg.n_layers, model.cfg.n_heads
        self.n_components = self.L * self.H + (self.L if model.cfg.d_mlp else 0)
        with torch.no_grad():
            self.pred_full = self._pred(np.ones(self.n_components, dtype=bool))

    def _hooks(self, mask):
        hooks = {}
        for l in range(self.L):
            drop = [h for h in range(self.H) if not mask[l * self.H + h]]
            if drop:
                idx = torch.tensor(drop, device=self.device, dtype=torch.long)
                base = self.abl[f"z.{l}"]

                def zh(name, t, idx=idx, base=base):
                    t = t.clone()
                    t[:, :, idx] = base[None, :, idx]
                    return t

                hooks[f"z.{l}"] = zh
            if self.model.cfg.d_mlp and not mask[self.L * self.H + l]:
                bm = self.abl[f"mlp_out.{l}"]

                def mh(name, t, bm=bm):
                    return bm[None].expand_as(t).clone()

                hooks[f"mlp_out.{l}"] = mh
        return hooks

    @torch.no_grad()
    def _pred(self, mask):
        outs = []
        for s0 in range(0, self.tokens.shape[0], self.batch):
            tt = self.tokens[s0 : s0 + self.batch]
            lg = self.model(tt, self._hooks(mask))[:, self.pos]
            outs.append(lg.argmax(-1))
        return torch.cat(outs)

    @torch.no_grad()
    def scores(self, mask) -> np.ndarray:
        """Per-instance fraction of evaluation positions agreeing with the full
        model."""
        pr = self._pred(mask)
        return (pr == self.pred_full).float().mean(dim=-1).cpu().numpy()


def greedy(score_fn, n_components, stop_at, start=None):
    mask = np.ones(n_components, dtype=bool) if start is None else start.copy()
    reported = [score_fn(mask)]
    n_q = 1
    while mask.sum() > stop_at:
        alive = np.flatnonzero(mask)
        best, bv = -1, -np.inf
        for j in alive:
            t = mask.copy()
            t[j] = False
            v = score_fn(t)
            n_q += 1
            if v > bv:
                best, bv = int(j), float(v)
        mask[best] = False
        reported.append(bv)
    return mask, reported, n_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/models/tt_d_induction_big.pt")
    ap.add_argument("--out", default="results/adaptive")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n_train", type=int, default=250)
    ap.add_argument("--n_holdout", type=int, default=250)
    ap.add_argument("--n_pool", type=int, default=40000)
    ap.add_argument("--stop_at", type=int, default=8)
    ap.add_argument("--R", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = TinyConfig(**ck["cfg"])
    model = TinyTransformer(cfg)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()

    task = InductionTask(n_symbols=40, block=8)
    rng = np.random.default_rng(args.seed)
    pool = task.sample(args.n_pool, rng)
    pos = task.eval_positions()
    abl = mean_ablation_cache(model, pool[:20000], device, batch_size=20000)
    n_comp = cfg.n_layers * cfg.n_heads + (cfg.n_layers if cfg.d_mlp else 0)
    d_removed = n_comp - args.stop_at
    log_reach = reachable_log_size(n_comp, d_removed)
    print(f"model: {cfg.n_layers}L x {cfg.n_heads}H + MLPs -> {n_comp} components; "
          f"greedy removes {d_removed}; log|reachable| = {log_reach:.1f} nats")

    pool_ev = CircuitEvaluator(model, pool, abl, pos, device)
    print(f"pool size {args.n_pool}; full-model self-agreement "
          f"{pool_ev.scores(np.ones(n_comp, dtype=bool)).mean():.4f}")

    rows = []
    t0 = time.time()
    for r in range(args.R):
        rr = np.random.default_rng(args.seed * 977 + r)
        itr = rr.integers(0, args.n_pool, size=args.n_train)
        iho = rr.integers(0, args.n_pool, size=args.n_holdout)
        ev_tr = CircuitEvaluator(model, pool[itr], abl, pos, device)
        ev_ho = CircuitEvaluator(model, pool[iho], abl, pos, device)
        cache_tr: dict[bytes, np.ndarray] = {}
        cache_ho: dict[bytes, np.ndarray] = {}

        def s_tr(mask):
            k = mask.tobytes()
            if k not in cache_tr:
                cache_tr[k] = ev_tr.scores(mask)
            return cache_tr[k]

        def s_ho(mask):
            k = mask.tobytes()
            if k not in cache_ho:
                cache_ho[k] = ev_ho.scores(mask)
            return cache_ho[k]

        # M1 / M4: search using the training answers
        m1, rep1, nq = greedy(lambda m: s_tr(m).mean(), n_comp, args.stop_at)
        theta1 = float(pool_ev.scores(m1).mean())
        tr1 = s_tr(m1)
        ho1 = s_ho(m1)
        naive_lcb = float(B.naive_lcb(tr1[:, None], args.alpha)[0])
        occam_lcb = float(
            B.empirical_bernstein_lcb(
                tr1[:, None], args.alpha * np.exp(-log_reach), family_size=1
            )[0]
        )
        split_lcb = float(B.empirical_bernstein_lcb(ho1[:, None], args.alpha, family_size=1)[0])

        # M2: search directly on the holdout (naive reuse)
        m2, rep2, _ = greedy(lambda m: s_ho(m).mean(), n_comp, args.stop_at)
        theta2 = float(pool_ev.scores(m2).mean())
        rep_ho2 = float(s_ho(m2).mean())

        # M3: search through Thresholdout
        th = Thresholdout(
            threshold=0.04, sigma=0.01, budget=30,
            rng=np.random.default_rng(args.seed * 13 + r),
        )
        m3, rep3, _ = greedy(lambda m: th.query(s_tr(m), s_ho(m)), n_comp, args.stop_at)
        theta3 = float(pool_ev.scores(m3).mean())
        # the point of a reusable holdout is that the HOLDOUT survives the search,
        # so the reported value is the holdout score of the returned circuit
        rep_th3 = float(s_ho(m3).mean())

        rows.append({
            "rep": r, "n_queries": nq,
            "m1_reported": float(rep1[-1]), "m1_theta": theta1,
            "m1_naive_lcb": naive_lcb, "m1_occam_lcb": occam_lcb,
            "m4_reported": float(ho1.mean()), "m4_split_lcb": split_lcb,
            "m2_reported": rep_ho2, "m2_theta": theta2,
            "m3_reported": rep_th3, "m3_theta": theta3,
            "th_spent": th.spent, "th_queries": th.n_queries,
        })
        if (r + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  replicate {r + 1}/{args.R}  ({el / 60:.1f} min, "
                  f"eta {el / (r + 1) * (args.R - r - 1) / 60:.1f} min)", flush=True)

    A = {k: np.array([row[k] for row in rows], dtype=float) for k in rows[0]}
    out = {
        "n_components": n_comp,
        "stop_at": args.stop_at,
        "log_reachable": log_reach,
        "n_queries": int(A["n_queries"][0]),
        "n_train": args.n_train,
        "n_holdout": args.n_holdout,
        "n_pool": args.n_pool,
        "R": args.R,
        "alpha": args.alpha,
        "bias": {
            "M1_same_data": float(np.mean(A["m1_reported"] - A["m1_theta"])),
            "M2_holdout_reused": float(np.mean(A["m2_reported"] - A["m2_theta"])),
            "M3_thresholdout": float(np.mean(A["m3_reported"] - A["m3_theta"])),
            "M4_fresh_holdout": float(np.mean(A["m4_reported"] - A["m1_theta"])),
        },
        "coverage": {
            "naive_on_search_data": float(np.mean(A["m1_naive_lcb"] <= A["m1_theta"])),
            "occam_reachable": float(np.mean(A["m1_occam_lcb"] <= A["m1_theta"])),
            "split": float(np.mean(A["m4_split_lcb"] <= A["m1_theta"])),
        },
        "certified": {
            "naive_on_search_data": float(np.mean(A["m1_naive_lcb"])),
            "occam_reachable": float(np.mean(A["m1_occam_lcb"])),
            "split": float(np.mean(A["m4_split_lcb"])),
        },
        "mean_truth": float(np.mean(A["m1_theta"])),
        "mean_reported_same_data": float(np.mean(A["m1_reported"])),
        "thresholdout_budget_spent": float(np.mean(A["th_spent"])),
        "rows": rows,
    }
    with open(os.path.join(args.out, "adaptive.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
