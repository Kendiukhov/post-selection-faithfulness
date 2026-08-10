"""E3b: adaptive circuit search on GPT-2 small, and what survives it.

A greedy pruner (the core of ACDC-style automated circuit discovery) removes one
attention head at a time, always dropping the head whose removal costs least on
the *search* set.  Each round asks the data many questions and the next question
depends on the previous answers, so a union bound over the questions actually
asked is not valid.

This script runs the search once, records the whole pruning path, and then
re-evaluates every circuit on the path on

  * the search set (what a same-data report would show), and
  * a large disjoint evaluation set (the truth, up to its own small noise),

so that the optimism of an adaptive search can be measured directly.

The search runs over a pre-registered pool of candidate heads (the 26 heads of
the published IOI circuit plus a fixed random sample of others), which keeps the
number of queries in the low thousands while leaving the reachable set far too
large for a union bound.
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

from psf.ioi import IOI_TEMPLATES, GPT2Patcher, build_ioi_dataset, ioi_circuit_heads  # noqa: E402


def pack_same_length(ds):
    """Group the dataset's templates by tokenised length and return the largest
    group as a single batch (all prompts then share the length, so they can be
    run together while each is still ablated towards its own template's mean)."""
    by_len: dict[int, list] = {}
    for g in ds.groups:
        by_len.setdefault(g["tokens"].shape[1], []).append(g)
    L = max(by_len, key=lambda k: sum(g["tokens"].shape[0] for g in by_len[k]))
    grp = by_len[L]
    return dict(
        tokens=np.concatenate([g["tokens"] for g in grp]),
        tokens_abc=np.concatenate([g["tokens_abc"] for g in grp]),
        io=np.concatenate([g["io_tok"] for g in grp]),
        s=np.concatenate([g["s_tok"] for g in grp]),
        template=np.concatenate(
            [np.full(g["tokens"].shape[0], g["template_id"]) for g in grp]
        ),
        length=L,
        n_templates=len(grp),
    )


class Evaluator:
    """Evaluates any head subset on a fixed set of prompts, returning the
    per-instance IO-minus-S logit difference."""

    def __init__(self, model, patch, data, device, batch=250):
        self.model, self.patch, self.device, self.batch = model, patch, device, batch
        self.toks = torch.tensor(data["tokens"], device=device)
        self.io = torch.tensor(data["io"], device=device)[:, None]
        self.s = torch.tensor(data["s"], device=device)[:, None]
        self.n = self.toks.shape[0]
        self.abc = torch.tensor(data["tokens_abc"], device=device)
        self.groups = data["template"]

    def prepare(self) -> None:
        self.patch.compute_means_grouped(self.abc, self.groups, batch_size=self.batch)

    @torch.no_grad()
    def per_instance(self, heads) -> np.ndarray:
        self.patch.set_circuit(heads)
        out = []
        for s0 in range(0, self.n, self.batch):
            sl = slice(s0, min(self.n, s0 + self.batch))
            self.patch.use_slice(sl)
            h = self.model.transformer(self.toks[sl]).last_hidden_state[:, -1]
            lg = self.model.lm_head(h)
            out.append((lg.gather(1, self.io[sl]) - lg.gather(1, self.s[sl])).squeeze(1))
        return torch.cat(out).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ioi")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n_search", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=1200)
    ap.add_argument("--pool", type=int, default=60, help="candidate heads in the search pool")
    ap.add_argument("--stop_at", type=int, default=10)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = torch.device(args.device)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    per_t = int(np.ceil((args.n_search + args.n_eval) / 8)) + 4
    ds = build_ioi_dataset(tok, n_per_template=per_t, templates=IOI_TEMPLATES, seed=args.seed)
    packed = pack_same_length(ds)
    ntot = packed["tokens"].shape[0]
    print(
        f"packed {ntot} prompts of length {packed['length']} "
        f"from {packed['n_templates']} templates",
        flush=True,
    )
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(ntot)
    i_search = perm[: args.n_search]
    i_eval = perm[args.n_search : args.n_search + args.n_eval]
    assert i_eval.size >= 200, f"only {i_eval.size} prompts left for evaluation"

    def subset(idx):
        out = {}
        for k, v in packed.items():
            out[k] = v[idx] if isinstance(v, np.ndarray) else v
        return out

    # pre-registered candidate pool: the published circuit plus a fixed random
    # sample of the remaining heads, chosen by seed and never by the data
    ioi_heads = list(ioi_circuit_heads())
    others = [(l, h) for l in range(12) for h in range(12) if (l, h) not in set(ioi_heads)]
    extra = [others[i] for i in rng.choice(len(others), size=args.pool - len(ioi_heads),
                                           replace=False)]
    pool = sorted(ioi_heads + extra)
    N = len(pool)
    print(f"candidate pool: {N} heads ({len(ioi_heads)} published + {len(extra)} random)")

    patch = GPT2Patcher(model)
    ev_search = Evaluator(model, patch, subset(i_search), device, batch=args.batch)
    ev_eval = Evaluator(model, patch, subset(i_eval), device, batch=args.batch)

    ev_search.prepare()
    ld_full_s = ev_search.per_instance(pool + [])  # only pool heads are ever kept
    all_heads = [(l, h) for l in range(12) for h in range(12)]
    ld_true_full_s = ev_search.per_instance(all_heads)
    ld_empty_s = ev_search.per_instance([])
    denom_s = ld_true_full_s.mean() - ld_empty_s.mean()
    print(
        f"search set: LD(all heads)={ld_true_full_s.mean():.3f}  "
        f"LD(pool only)={ld_full_s.mean():.3f}  LD(empty)={ld_empty_s.mean():.3f}",
        flush=True,
    )

    mask = np.ones(N, dtype=bool)
    path = [mask.copy()]
    reported = [(ld_full_s.mean() - ld_empty_s.mean()) / denom_s]
    removed: list[int] = []
    n_queries = 1
    t0 = time.time()
    while mask.sum() > args.stop_at:
        best, bv = -1, -np.inf
        for j in np.flatnonzero(mask):
            trial = mask.copy()
            trial[j] = False
            v = ev_search.per_instance([pool[i] for i in np.flatnonzero(trial)])
            n_queries += 1
            sc = (v.mean() - ld_empty_s.mean()) / denom_s
            if sc > bv:
                best, bv = int(j), float(sc)
        mask[best] = False
        removed.append(best)
        path.append(mask.copy())
        reported.append(bv)
        if len(removed) % 10 == 0:
            el = time.time() - t0
            print(f"  {int(mask.sum()):3d} heads left; reported {bv:.4f}; "
                  f"{n_queries} queries; {el / 60:.1f} min", flush=True)
    print(f"search finished: {n_queries} queries in {(time.time() - t0) / 60:.1f} min", flush=True)

    ev_eval.prepare()
    ld_full_e = ev_eval.per_instance(all_heads)
    ld_empty_e = ev_eval.per_instance([])
    P = len(path)
    ld_s = np.zeros((args.n_search, P), dtype=np.float32)
    ld_e = np.zeros((i_eval.size, P), dtype=np.float32)
    for p in range(P):
        heads = [pool[i] for i in np.flatnonzero(path[p])]
        ld_e[:, p] = ev_eval.per_instance(heads)
    ev_search.prepare()
    for p in range(P):
        heads = [pool[i] for i in np.flatnonzero(path[p])]
        ld_s[:, p] = ev_search.per_instance(heads)
    print("re-evaluation done", flush=True)

    np.savez_compressed(
        os.path.join(args.out, "ioi_greedy.npz"),
        path=np.array(path),
        pool=np.array(pool),
        reported=np.array(reported),
        removed=np.array(removed),
        ld_search=ld_s,
        ld_eval=ld_e,
        ld_full_search=ld_true_full_s,
        ld_empty_search=ld_empty_s,
        ld_full_eval=ld_full_e,
        ld_empty_eval=ld_empty_e,
        template_search=packed["template"][i_search],
        template_eval=packed["template"][i_eval],
        sizes=np.array([int(m.sum()) for m in path]),
    )
    meta = {
        "n_search": int(args.n_search),
        "n_eval": int(i_eval.size),
        "n_queries": int(n_queries),
        "pool_size": int(N),
        "path_len": int(P),
        "stop_at": args.stop_at,
        "minutes": (time.time() - t0) / 60,
        "n_templates_packed": int(packed["n_templates"]),
        "seq_len": int(packed["length"]),
    }
    with open(os.path.join(args.out, "greedy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
