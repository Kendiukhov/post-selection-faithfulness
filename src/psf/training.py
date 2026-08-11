"""Training loops for the tiny transformers.

Two modes:

``standard``  ordinary next-token / label prediction.  The network is free to
              implement the task however it likes, which is the realistic
              setting for circuit discovery.

``iit``       interchange-intervention training (Geiger et al., 2022): in
              addition to the task loss, a randomly chosen high-level variable
              is interchanged between a base and a source input at a *planted*
              location in the network, and the output is trained to match the
              high-level model's counterfactual label.  At zero loss the
              planted alignment is a causal abstraction by construction, which
              gives us a known-correct hypothesis to check power against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .models import TinyConfig, TinyTransformer

__all__ = ["TrainConfig", "train_label_task", "train_induction", "PlantedAlignment"]


@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup: int = 200
    seed: int = 0
    log_every: int = 500
    device: str = "cpu"


@dataclass
class PlantedAlignment:
    """Location at which IIT plants a high-level variable."""

    variable: str
    site: str
    position: int
    dims: tuple[int, ...]


def _opt(model, cfg: TrainConfig):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup))
    )
    return opt, sched


def train_label_task(
    task,
    cfg_model: TinyConfig,
    cfg_train: TrainConfig,
    label_token_ids: list[int],
    iit: Optional[list[PlantedAlignment]] = None,
    iit_weight: float = 1.0,
    verbose: bool = True,
) -> tuple[TinyTransformer, dict]:
    """Train on a task whose label is a single token at the final position."""
    device = torch.device(cfg_train.device)
    torch.manual_seed(cfg_train.seed)
    model = TinyTransformer(cfg_model).to(device)
    opt, sched = _opt(model, cfg_train)
    rng = np.random.default_rng(cfg_train.seed)
    readout = task.readout_positions()
    label_ids = torch.tensor(label_token_ids, device=device)
    hist = {"step": [], "loss": [], "acc": [], "iit_acc": []}

    for step in range(cfg_train.steps):
        toks = task.sample(cfg_train.batch_size, rng)
        y = _label_index(task, toks, label_token_ids)
        tt = torch.tensor(toks, device=device)
        yy = torch.tensor(y, device=device)
        logits = model(tt)[:, readout][:, label_ids]
        loss = F.cross_entropy(logits, yy)
        acc = (logits.argmax(-1) == yy).float().mean().item()

        iit_acc = float("nan")
        if iit:
            pa = iit[int(rng.integers(0, len(iit)))]
            src = task.sample(cfg_train.batch_size, rng)
            cf = task.counterfactual_label(toks, src, pa.variable)
            cf_idx = _cf_index(task, cf, label_token_ids)
            st = torch.tensor(src, device=device)
            # Record the source activation *without* detaching: the network has to
            # receive gradient through the source run as well, otherwise it is
            # never asked to make the planted subspace encode the variable --
            # only to react to whatever happens to be there.
            store: dict = {}

            def _record(name, t, store=store):
                store[name] = t
                return t

            model(st, {pa.site: _record})
            srcact = store[pa.site]
            dims = torch.tensor(pa.dims, device=device, dtype=torch.long)

            def hook(name, t, srcact=srcact, dims=dims, p=pa.position):
                t = t.clone()
                t[:, p, dims] = srcact[:, p, dims]
                return t

            lg = model(tt, {pa.site: hook})[:, readout][:, label_ids]
            cfy = torch.tensor(cf_idx, device=device)
            iit_loss = F.cross_entropy(lg, cfy)
            iit_acc = (lg.argmax(-1) == cfy).float().mean().item()
            loss = loss + iit_weight * iit_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg_train.log_every == 0 or step == cfg_train.steps - 1:
            hist["step"].append(step)
            hist["loss"].append(float(loss.item()))
            hist["acc"].append(acc)
            hist["iit_acc"].append(iit_acc)
            if verbose:
                print(f"  step {step:5d}  loss {loss.item():.4f}  acc {acc:.4f}  iit {iit_acc:.4f}")
    return model, hist


def _label_index(task, toks: np.ndarray, label_token_ids: list[int]) -> np.ndarray:
    """Index into ``label_token_ids`` of the high-level model's output.

    Both label tasks are set up so that the high-level output ``Y`` is already
    the index (``Y in {0,1}`` for equality, ``Y in {0..M-1}`` for arithmetic).
    """
    y = np.asarray(task.high_level(toks)["Y"], dtype=np.int64)
    assert y.max(initial=0) < len(label_token_ids)
    return y


def _cf_index(task, cf: np.ndarray, label_token_ids: list[int]) -> np.ndarray:
    cf = np.asarray(cf, dtype=np.int64)
    assert cf.max(initial=0) < len(label_token_ids)
    return cf


def train_induction(
    task, cfg_model: TinyConfig, cfg_train: TrainConfig, verbose: bool = True
) -> tuple[TinyTransformer, dict]:
    device = torch.device(cfg_train.device)
    torch.manual_seed(cfg_train.seed)
    model = TinyTransformer(cfg_model).to(device)
    opt, sched = _opt(model, cfg_train)
    rng = np.random.default_rng(cfg_train.seed)
    pos = torch.tensor(task.eval_positions(), device=device, dtype=torch.long)
    hist = {"step": [], "loss": [], "acc": []}
    for step in range(cfg_train.steps):
        toks = task.sample(cfg_train.batch_size, rng)
        tt = torch.tensor(toks, device=device)
        logits = model(tt)
        sel = logits[:, pos]  # (B, P, V)
        tgt = tt[:, pos + 1]
        loss = F.cross_entropy(sel.reshape(-1, sel.shape[-1]), tgt.reshape(-1))
        acc = (sel.argmax(-1) == tgt).float().mean().item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg_train.log_every == 0 or step == cfg_train.steps - 1:
            hist["step"].append(step)
            hist["loss"].append(float(loss.item()))
            hist["acc"].append(acc)
            if verbose:
                print(f"  step {step:5d}  loss {loss.item():.4f}  acc {acc:.4f}")
    return model, hist
