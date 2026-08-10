"""Precompute the per-instance faithfulness score matrix for GPT-2 small on IOI.

This is the only expensive step of the GPT-2 experiment.  Everything after it --
selection, resampling, coverage simulation, every confidence bound -- operates
on the cached matrix and costs no GPU time at all.

Outputs (npz):
    acc      (n, m)  1 if the ablated model still ranks IO above S
    ld       (n, m)  logit difference (IO minus S) under circuit c
    ld_full  (n,)    logit difference of the unablated model
    ld_empty (n,)    logit difference with every attention head ablated
    template (n,)    template id of each instance  (cluster label)
    sizes    (m,)    number of heads in each circuit
    labels   (m,)    human-readable circuit labels
"""

from __future__ import annotations

import argparse
import itertools
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

from psf.ioi import (  # noqa: E402
    IOI_CIRCUIT_GROUPS,
    IOI_TEMPLATES,
    GPT2Patcher,
    build_ioi_dataset,
    ioi_circuit_heads,
)


def build_circuit_class(seed: int = 0, n_strat_ioi: int = 12, n_strat_all: int = 8):
    """A hypothesis class fixed *before* any data is seen.

    (a) all 2^7 unions of the seven functional head groups identified by
        Wang et al. (2023) -- this is the space an interpretability researcher
        actually reasons over when asking "which parts of the circuit are
        needed?";
    (b) size-stratified random subsets of the 26 published circuit heads;
    (c) size-stratified random subsets of all 144 heads.
    """
    rng = np.random.default_rng(seed)
    groups = list(IOI_CIRCUIT_GROUPS.items())
    circuits: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    seen: set[frozenset] = set()

    def add(label, heads):
        key = frozenset(heads)
        if key in seen:
            return
        seen.add(key)
        circuits.append((label, tuple(sorted(heads))))

    for r in range(len(groups) + 1):
        for combo in itertools.combinations(range(len(groups)), r):
            heads: list[tuple[int, int]] = []
            for i in combo:
                heads.extend(groups[i][1])
            add("grp:" + "+".join(groups[i][0] for i in combo) if combo else "grp:empty", heads)

    ioi_heads = list(ioi_circuit_heads())
    for s in range(1, len(ioi_heads)):
        for _ in range(n_strat_ioi):
            idx = rng.choice(len(ioi_heads), size=s, replace=False)
            add(f"rnd26:s{s}", [ioi_heads[i] for i in idx])

    all_heads = [(l, h) for l in range(12) for h in range(12)]
    for s in list(range(1, 40)) + list(range(40, 145, 8)):
        for _ in range(n_strat_all):
            idx = rng.choice(len(all_heads), size=s, replace=False)
            add(f"rnd144:s{s}", [all_heads[i] for i in idx])

    add("published_ioi_circuit", ioi_heads)
    add("all_heads", all_heads)
    return circuits


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ioi")
    ap.add_argument("--n_per_template", type=int, default=200)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_templates", type=int, default=0, help="0 = all")
    ap.add_argument("--max_circuits", type=int, default=0, help="0 = all (smoke test only)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = torch.device(args.device)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("building dataset ...")
    templates = IOI_TEMPLATES if args.n_templates == 0 else IOI_TEMPLATES[: args.n_templates]
    ds = build_ioi_dataset(
        tok, n_per_template=args.n_per_template, templates=templates, seed=args.seed
    )
    circuits = build_circuit_class(seed=args.seed)
    if args.max_circuits:
        circuits = circuits[: args.max_circuits]
    m = len(circuits)
    n = ds.n_total
    print(f"n = {n} instances over {len(ds.groups)} templates;  |C| = {m} circuits")

    patch = GPT2Patcher(model)
    acc = np.zeros((n, m), dtype=np.float32)
    ld = np.zeros((n, m), dtype=np.float32)
    ld_full = np.zeros(n, dtype=np.float32)
    ld_empty = np.zeros(n, dtype=np.float32)
    template = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    off = 0
    n_fwd = 0
    for gi, g in enumerate(ds.groups):
        K = g["tokens"].shape[0]
        toks = torch.tensor(g["tokens"], device=device)
        io = torch.tensor(g["io_tok"], device=device)[:, None]
        s = torch.tensor(g["s_tok"], device=device)[:, None]
        template[off : off + K] = g["template_id"]

        def ld_of(tok_slice):
            # Only the final-position logits matter, and only for two tokens, so
            # we skip the (expensive) full vocabulary projection at every
            # position.  ``model.transformer`` already applies the final
            # LayerNorm, so this is numerically identical to slicing
            # ``model(x).logits[:, -1]``.
            h = model.transformer(tok_slice).last_hidden_state[:, -1]
            lg = model.lm_head(h)
            return (lg.gather(1, io) - lg.gather(1, s)).squeeze(1)

        patch.set_full()
        chunks = [slice(i, min(K, i + args.batch)) for i in range(0, K, args.batch)]
        vals = []
        for sl in chunks:
            vals.append(ld_of(toks[sl]))
        ld_full[off : off + K] = torch.cat(vals).cpu().numpy()
        n_fwd += K

        patch.compute_means(torch.tensor(g["tokens_abc"], device=device), batch_size=args.batch)
        n_fwd += K

        patch.set_circuit([])
        vals = [ld_of(toks[sl]) for sl in chunks]
        ld_empty[off : off + K] = torch.cat(vals).cpu().numpy()
        n_fwd += K

        for j, (label, heads) in enumerate(circuits):
            patch.set_circuit(heads)
            vals = [ld_of(toks[sl]) for sl in chunks]
            v = torch.cat(vals)
            ld[off : off + K, j] = v.cpu().numpy()
            acc[off : off + K, j] = (v > 0).float().cpu().numpy()
            n_fwd += K
        off += K
        el = time.time() - t0
        print(
            f"  template {gi + 1}/{len(ds.groups)}  elapsed {el / 60:.1f} min  "
            f"({n_fwd / el:.0f} seq/s, eta {el / (gi + 1) * (len(ds.groups) - gi - 1) / 60:.1f} min)",
            flush=True,
        )

    sizes = np.array([len(h) for _, h in circuits], dtype=np.int32)
    labels = np.array([lab for lab, _ in circuits])
    np.savez_compressed(
        os.path.join(args.out, "ioi_scores.npz"),
        acc=acc,
        ld=ld,
        ld_full=ld_full,
        ld_empty=ld_empty,
        template=template,
        sizes=sizes,
        labels=labels,
    )
    with open(os.path.join(args.out, "circuits.json"), "w") as f:
        json.dump(
            [{"label": lab, "heads": [list(x) for x in h]} for lab, h in circuits], f, indent=1
        )
    meta = {
        "n": int(n),
        "m": int(m),
        "n_templates": len(ds.groups),
        "n_per_template": args.n_per_template,
        "forward_passes": int(n_fwd),
        "minutes": (time.time() - t0) / 60,
        "device": str(device),
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
