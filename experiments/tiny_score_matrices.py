"""Precompute per-instance faithfulness score matrices for the tiny models.

As for GPT-2, all statistical work afterwards runs on the cached matrices.
The cached pool *defines* the population: analysis samples are drawn i.i.d.
from it, so the true faithfulness of every hypothesis is known exactly and
coverage can be measured without oracle noise.
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

from psf.interventions import (  # noqa: E402
    build_alignment_class,
    build_component_class,
    component_scores,
    interchange_scores,
    mean_ablation_cache,
)
from psf.models import TinyConfig, TinyTransformer  # noqa: E402
from psf.tasks import ArithmeticTask, EqualityTask, InductionTask  # noqa: E402


def load(path: str, device) -> TinyTransformer:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TinyConfig(**ck["cfg"])
    model = TinyTransformer(cfg)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def run_alignment(model, task, variable, label_ids, N, device, seed, batch, tag, outdir):
    rng = np.random.default_rng(seed)
    base = task.sample(N, rng)
    source = task.sample(N, rng)
    cf = task.counterfactual_label(base, source, variable)
    sites = ["resid_pre.0"] + [f"resid_post.{l}" for l in range(model.cfg.n_layers)]
    hyps = build_alignment_class(
        sites=sites,
        positions=list(range(task.seq_len)),
        d_model=model.cfg.d_model,
        block_sizes=(8, 16, 32),
        include_full=True,
    )
    print(f"  [{tag}] N={N}  |C|={len(hyps)}  variable={variable}")
    t0 = time.time()
    S = interchange_scores(
        model, base, source, cf, hyps, label_ids, task.readout_positions(), device,
        batch_size=batch,
    )
    print(f"  [{tag}] done in {time.time() - t0:.0f}s;  theta range "
          f"[{S.mean(0).min():.3f}, {S.mean(0).max():.3f}]")
    np.savez_compressed(
        os.path.join(outdir, f"{tag}_scores.npz"),
        S=S.astype(np.uint8),
        labels=np.array([h.label() for h in hyps]),
        site=np.array([h.site for h in hyps]),
        position=np.array([h.position for h in hyps]),
        basis=np.array([h.basis for h in hyps]),
        dim0=np.array([h.dims[0] for h in hyps]),
        ndim=np.array([len(h.dims) for h in hyps]),
    )
    return S, hyps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="results/models")
    ap.add_argument("--out", default="results/tiny")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--N", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--which", default="ABC")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    meta = {}

    if "A" in args.which:
        print("[TT-A] hierarchical equality -- alignment search for V1")
        task = EqualityTask(n_symbols=10)
        for tag, fn in [("tt_a", "tt_a_equality.pt"), ("tt_a_iit", "tt_a_equality_iit.pt")]:
            model, ck = load(os.path.join(args.models, fn), device)
            S, hyps = run_alignment(
                model, task, "V1", list(task.label_tokens), args.N, device,
                seed=101, batch=args.batch, tag=tag, outdir=args.out,
            )
            meta[tag] = {
                "task": "equality", "variable": "V1", "N": args.N, "m": len(hyps),
                "model_acc": ck["acc"],
                "theta_max": float(S.mean(0).max()),
                "argmax_label": hyps[int(np.argmax(S.mean(0)))].label(),
            }
            if "planted" in ck:
                meta[tag]["planted"] = ck["planted"]
            del model

    if "B" in args.which:
        print("[TT-B] induction -- component-subset circuits")
        itask = InductionTask(n_symbols=40, block=8)
        model, ck = load(os.path.join(args.models, "tt_b_induction.pt"), device)
        rng = np.random.default_rng(202)
        toks = itask.sample(args.N, rng)
        pos = itask.eval_positions()
        targets = toks[:, pos + 1]
        # Distractors for the logit-difference metric: four symbols that appear
        # in the repeated block but are not the correct next token.  They are
        # the natural competitors, so the logit difference measures whether the
        # induction mechanism still points at the right one.
        blk = toks[:, : itask.block]
        dis = np.zeros((args.N, len(pos), 4), dtype=np.int64)
        for p in range(len(pos)):
            cand = blk.copy()
            mask = cand == targets[:, p : p + 1]
            cand = np.where(mask, -1, cand)
            order = np.argsort(-cand, axis=1)  # push -1 to the end
            dis[:, p] = np.take_along_axis(cand, order, axis=1)[:, :4]
        circuits = build_component_class(
            model.cfg.n_layers, model.cfg.n_heads, include_mlps=False, exhaustive=True
        )
        print(f"  |C| = {len(circuits)} (exhaustive power set of "
              f"{model.cfg.n_layers * model.cfg.n_heads} heads)")
        abl = mean_ablation_cache(model, toks, device, batch_size=args.batch)
        t0 = time.time()
        res = component_scores(
            model, toks, circuits, abl, pos, targets, dis, device, batch_size=args.batch
        )
        print(f"  done in {time.time() - t0:.0f}s")
        np.savez_compressed(
            os.path.join(args.out, "tt_b_scores.npz"),
            agree=res["agree"].astype(np.float32),
            ld=res["ld"].astype(np.float32),
            ld_full=res["ld_full"],
            ld_empty=res["ld_empty"],
            labels=np.array([c.label() for c in circuits]),
            sizes=np.array([c.size for c in circuits], dtype=np.int32),
        )
        meta["tt_b"] = {
            "task": "induction", "N": args.N, "m": len(circuits),
            "model_acc": ck["acc"],
            "ld_full": float(res["ld_full"].mean()),
            "ld_empty": float(res["ld_empty"].mean()),
            "agree_max": float(res["agree"].mean(0).max()),
        }
        del model

    if "C" in args.which:
        print("[TT-C] modular arithmetic -- alignment search for S1")
        atask = ArithmeticTask(modulus=10)
        for tag, fn in [("tt_c", "tt_c_arithmetic.pt"), ("tt_c_iit", "tt_c_arithmetic_iit.pt")]:
            path = os.path.join(args.models, fn)
            if not os.path.exists(path):
                print(f"  (skipping {tag}: {path} not found)")
                continue
            model, ck = load(path, device)
            S, hyps = run_alignment(
                model, atask, "S1", list(range(atask.modulus)), args.N, device,
                seed=303, batch=args.batch, tag=tag, outdir=args.out,
            )
            meta[tag] = {
                "task": "arithmetic", "variable": "S1", "N": args.N, "m": len(hyps),
                "model_acc": ck["acc"],
                "theta_max": float(S.mean(0).max()),
                "argmax_label": hyps[int(np.argmax(S.mean(0)))].label(),
            }
            if "planted" in ck:
                meta[tag]["planted"] = ck["planted"]
            del model

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
