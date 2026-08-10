"""Train the tiny transformers used in the paper.

Five models, all reaching 100% task accuracy:

  tt_a       hierarchical equality, ordinary training
  tt_a_iit   the same task with an alignment for V1 *planted* by interchange
             intervention training, giving a known-faithful hypothesis
  tt_b       induction, attention-only, ordinary training
  tt_c       three-term modular arithmetic, ordinary training
  tt_c_iit   the same task with the intermediate sum S1 planted

Usage:  python experiments/train_tiny_models.py --out results/models
        python experiments/train_tiny_models.py --only c_iit
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

from psf.models import TinyConfig  # noqa: E402
from psf.tasks import ArithmeticTask, EqualityTask, InductionTask  # noqa: E402
from psf.training import (  # noqa: E402
    PlantedAlignment,
    TrainConfig,
    train_induction,
    train_label_task,
)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def eval_label_task(model, task, label_ids, device, n=20000, seed=999):
    rng = np.random.default_rng(seed)
    toks = task.sample(n, rng)
    y = np.asarray(task.high_level(toks)["Y"], dtype=np.int64)
    tt = torch.tensor(toks, device=device)
    lg = model(tt)[:, task.readout_positions()][:, torch.tensor(label_ids, device=device)]
    return float((lg.argmax(-1).cpu().numpy() == y).mean())


@torch.no_grad()
def eval_induction(model, task, device, n=20000, seed=999):
    rng = np.random.default_rng(seed)
    toks = task.sample(n, rng)
    tt = torch.tensor(toks, device=device)
    pos = torch.tensor(task.eval_positions(), device=device, dtype=torch.long)
    sel = model(tt)[:, pos]
    return float((sel.argmax(-1) == tt[:, pos + 1]).float().mean().item())


def save(out, name, model, cfg, hist, acc, extra=None):
    payload = {"state_dict": model.state_dict(), "cfg": cfg.to_dict(), "hist": hist, "acc": acc}
    if extra:
        payload.update(extra)
    torch.save(payload, os.path.join(out, name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/models")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--only", default="", help="comma separated subset of a,a_iit,b,c,c_iit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    want = lambda k: (not only) or (k in only)  # noqa: E731
    print(f"device: {device}")

    summary_path = os.path.join(args.out, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    eq = EqualityTask(n_symbols=10)
    eq_cfg = TinyConfig(
        n_vocab=eq.n_vocab, n_ctx=eq.seq_len, d_model=64, n_layers=2, n_heads=4, d_mlp=256
    )
    ar = ArithmeticTask(modulus=10)
    ar_cfg = TinyConfig(
        n_vocab=ar.n_vocab, n_ctx=ar.seq_len, d_model=64, n_layers=3, n_heads=4, d_mlp=256
    )

    if want("a"):
        print("\n[TT-A] hierarchical equality (standard training)")
        tc = TrainConfig(steps=args.steps, batch_size=512, lr=3e-3, seed=0, device=str(device))
        t0 = time.time()
        model, hist = train_label_task(eq, eq_cfg, tc, list(eq.label_tokens))
        acc = eval_label_task(model, eq, list(eq.label_tokens), device)
        print(f"  held-out accuracy: {acc:.4f}   ({time.time() - t0:.0f}s)")
        save(args.out, "tt_a_equality.pt", model, eq_cfg, hist, acc)
        summary["tt_a_equality"] = {
            "acc": acc, "n_params": sum(p.numel() for p in model.parameters())
        }

    if want("a_iit"):
        print("\n[TT-A-IIT] hierarchical equality with a planted alignment")
        planted = [
            PlantedAlignment("V1", "resid_post.0", eq.positions["b"], tuple(range(0, 16))),
            PlantedAlignment("V2", "resid_post.0", eq.positions["d"], tuple(range(16, 32))),
        ]
        tc = TrainConfig(steps=args.steps, batch_size=512, lr=3e-3, seed=1, device=str(device))
        model, hist = train_label_task(
            eq, eq_cfg, tc, list(eq.label_tokens), iit=planted, iit_weight=1.0
        )
        acc = eval_label_task(model, eq, list(eq.label_tokens), device)
        print(f"  held-out accuracy: {acc:.4f}  final IIT accuracy: {hist['iit_acc'][-1]:.4f}")
        save(args.out, "tt_a_equality_iit.pt", model, eq_cfg, hist, acc,
             {"planted": [p.__dict__ for p in planted]})
        summary["tt_a_equality_iit"] = {"acc": acc, "iit_acc": hist["iit_acc"][-1]}

    if want("b"):
        print("\n[TT-B] induction (attention-only, 2 layers)")
        itask = InductionTask(n_symbols=40, block=8)
        icfg = TinyConfig(
            n_vocab=itask.n_vocab, n_ctx=itask.seq_len, d_model=64,
            n_layers=2, n_heads=4, d_mlp=None,
        )
        itc = TrainConfig(steps=args.steps, batch_size=512, lr=2e-3, seed=2, device=str(device))
        imodel, ihist = train_induction(itask, icfg, itc)
        iacc = eval_induction(imodel, itask, device)
        print(f"  held-out induction accuracy: {iacc:.4f}")
        save(args.out, "tt_b_induction.pt", imodel, icfg, ihist, iacc)
        summary["tt_b_induction"] = {"acc": iacc}

    if want("d"):
        print("\n[TT-D] induction, 4 layers x 8 heads + MLPs (36 ablatable components)")
        itask = InductionTask(n_symbols=40, block=8)
        dcfg = TinyConfig(
            n_vocab=itask.n_vocab, n_ctx=itask.seq_len, d_model=128,
            n_layers=4, n_heads=8, d_mlp=256,
        )
        dtc = TrainConfig(steps=args.steps, batch_size=512, lr=1e-3, seed=5, device=str(device))
        dmodel, dhist = train_induction(itask, dcfg, dtc)
        dacc = eval_induction(dmodel, itask, device)
        print(f"  held-out induction accuracy: {dacc:.4f}")
        save(args.out, "tt_d_induction_big.pt", dmodel, dcfg, dhist, dacc)
        summary["tt_d_induction_big"] = {"acc": dacc, "n_components": 4 * 8 + 4}

    if want("c"):
        print("\n[TT-C] three-term modular arithmetic (standard training)")
        atc = TrainConfig(
            steps=max(args.steps, 12000), batch_size=512, lr=3e-3, seed=3, device=str(device)
        )
        amodel, ahist = train_label_task(ar, ar_cfg, atc, list(range(ar.modulus)))
        aacc = eval_label_task(amodel, ar, list(range(ar.modulus)), device)
        print(f"  held-out accuracy: {aacc:.4f}")
        save(args.out, "tt_c_arithmetic.pt", amodel, ar_cfg, ahist, aacc)
        summary["tt_c_arithmetic"] = {"acc": aacc}

    if want("c_iit"):
        print("\n[TT-C-IIT] modular arithmetic with the intermediate sum planted")
        # S1 is planted at the *final* position after two of the three layers:
        # the network can then compute a+b there and add c in the last layer, which
        # is a natural algorithm for it to implement.
        planted = [PlantedAlignment("S1", "resid_post.1", 3, tuple(range(0, 32)))]
        atc = TrainConfig(
            steps=max(args.steps, 20000), batch_size=512, lr=3e-3, seed=4, device=str(device)
        )
        amodel, ahist = train_label_task(
            ar, ar_cfg, atc, list(range(ar.modulus)), iit=planted, iit_weight=2.0
        )
        aacc = eval_label_task(amodel, ar, list(range(ar.modulus)), device)
        print(f"  held-out accuracy: {aacc:.4f}  final IIT accuracy: {ahist['iit_acc'][-1]:.4f}")
        save(args.out, "tt_c_arithmetic_iit.pt", amodel, ar_cfg, ahist, aacc,
             {"planted": [p.__dict__ for p in planted]})
        summary["tt_c_arithmetic_iit"] = {"acc": aacc, "iit_acc": ahist["iit_acc"][-1]}

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved to", args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
