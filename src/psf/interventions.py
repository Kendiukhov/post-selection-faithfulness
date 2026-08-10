"""Interchange interventions and component ablations for the tiny models.

Two families of circuit hypothesis are supported, matching the two dominant
practices in the literature.

**Alignment hypotheses** (causal abstraction / DAS).  A hypothesis names a
location ``(site, position)`` in the network and a linear subspace of the
residual stream there, and claims that this subspace carries a particular
high-level variable.  It is tested by an *interchange intervention*: copy the
subspace's value from a source input into a base run and check whether the
network's output matches the high-level model's counterfactual output.

**Component-subset hypotheses** (circuit discovery).  A hypothesis names a set
of model components (attention heads, MLPs); everything outside the set is
ablated, and faithfulness is measured by how much of the model's behaviour
survives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from .models import TinyTransformer

__all__ = [
    "AlignmentHypothesis",
    "build_alignment_class",
    "interchange_scores",
    "ComponentSet",
    "build_component_class",
    "mean_ablation_cache",
    "component_scores",
]


# ---------------------------------------------------------------------------
# alignment hypotheses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentHypothesis:
    site: str  # e.g. "resid_post.0"
    position: int  # token position that is intervened on
    basis: str  # "std" or "rot"
    dims: tuple[int, ...]  # column indices into the basis

    def label(self) -> str:
        return f"{self.site}@p{self.position}:{self.basis}[{self.dims[0]}:{self.dims[-1] + 1}]"


def build_alignment_class(
    sites: Sequence[str],
    positions: Sequence[int],
    d_model: int,
    block_sizes: Sequence[int] = (8, 16, 32),
    include_full: bool = True,
    bases: Sequence[str] = ("std", "rot"),
) -> list[AlignmentHypothesis]:
    """Enumerate a *pre-registered* class of alignment hypotheses.

    For each site and position we take contiguous blocks of coordinates of each
    requested size, in the standard basis and in one fixed random orthonormal
    basis (a stand-in for the rotated subspaces that DAS searches over).  The
    class is fixed before any data is seen, which is what the simultaneous
    bounds require.
    """
    hyps: list[AlignmentHypothesis] = []
    for site in sites:
        for p in positions:
            for basis in bases:
                for k in block_sizes:
                    for start in range(0, d_model - k + 1, k):
                        hyps.append(
                            AlignmentHypothesis(site, int(p), basis, tuple(range(start, start + k)))
                        )
            if include_full:
                # patching the *whole* residual stream is basis-independent, so it
                # is enumerated once per (site, position) rather than once per basis
                hyps.append(AlignmentHypothesis(site, int(p), "std", tuple(range(d_model))))
    return hyps


def _rotation(d_model: int, seed: int, device, dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d_model, d_model, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))[None, :]  # make it deterministic
    return Q.to(device=device, dtype=dtype)


@torch.no_grad()
def interchange_scores(
    model: TinyTransformer,
    base_tokens: np.ndarray,
    source_tokens: np.ndarray,
    cf_labels: np.ndarray,
    hypotheses: Sequence[AlignmentHypothesis],
    label_token_ids: Sequence[int],
    readout_pos: int,
    device: torch.device,
    rot_seed: int = 12345,
    batch_size: int = 8192,
) -> np.ndarray:
    """Per-instance interchange-intervention agreement, shape ``(n, |C|)``.

    ``cf_labels[i]`` is the index into ``label_token_ids`` that the high-level
    causal model outputs for instance ``i`` under the interchange intervention.
    The returned entry is 1 iff the network's arg-max over ``label_token_ids``
    at ``readout_pos`` equals that label.
    """
    model.eval()
    n = base_tokens.shape[0]
    m = len(hypotheses)
    out = np.zeros((n, m), dtype=np.float32)
    d_model = model.cfg.d_model
    label_ids = torch.tensor(label_token_ids, device=device, dtype=torch.long)

    sites = sorted({h.site for h in hypotheses})
    Q = _rotation(d_model, rot_seed, device, torch.float32)

    for s0 in range(0, n, batch_size):
        s1 = min(n, s0 + batch_size)
        bt = torch.tensor(base_tokens[s0:s1], device=device)
        st = torch.tensor(source_tokens[s0:s1], device=device)
        _, src_cache = model.run_with_cache(st, sites)

        for j, hyp in enumerate(hypotheses):
            src = src_cache[hyp.site]  # (B, T, d)
            dims = torch.tensor(hyp.dims, device=device, dtype=torch.long)
            pos = hyp.position

            if hyp.basis == "std":

                def hook(name, t, src=src, dims=dims, pos=pos):
                    t = t.clone()
                    t[:, pos, dims] = src[:, pos, dims]
                    return t

            elif hyp.basis == "rot":
                U = Q[:, dims]  # (d, k) orthonormal columns

                def hook(name, t, src=src, U=U, pos=pos):
                    t = t.clone()
                    delta = src[:, pos] - t[:, pos]
                    t[:, pos] = t[:, pos] + (delta @ U) @ U.T
                    return t

            else:  # pragma: no cover
                raise ValueError(hyp.basis)

            logits = model(bt, {hyp.site: hook})
            pred = logits[:, readout_pos][:, label_ids].argmax(dim=-1).cpu().numpy()
            out[s0:s1, j] = (pred == cf_labels[s0:s1]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# component-subset hypotheses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSet:
    """A circuit given as a set of attention heads (and optionally MLPs)."""

    heads: tuple[tuple[int, int], ...]  # (layer, head)
    mlps: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        return len(self.heads) + len(self.mlps)

    def label(self) -> str:
        h = ",".join(f"L{l}H{hh}" for l, hh in sorted(self.heads))
        m = ",".join(f"M{l}" for l in sorted(self.mlps))
        return "{" + "; ".join(x for x in (h, m) if x) + "}"


def build_component_class(
    n_layers: int,
    n_heads: int,
    include_mlps: bool = False,
    max_size: Optional[int] = None,
    exhaustive: bool = True,
    n_random: int = 0,
    seed: int = 0,
) -> list[ComponentSet]:
    """Enumerate circuits.  With ``exhaustive`` the full power set of heads is
    returned (feasible for the tiny models, where ``n_layers * n_heads <= 12``)."""
    from itertools import combinations

    heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    mlps = list(range(n_layers)) if include_mlps else []
    items = [("h", x) for x in heads] + [("m", x) for x in mlps]
    N = len(items)
    out: list[ComponentSet] = []
    if exhaustive:
        top = N if max_size is None else max_size
        for k in range(0, top + 1):
            for combo in combinations(range(N), k):
                hs = tuple(items[i][1] for i in combo if items[i][0] == "h")
                ms = tuple(items[i][1] for i in combo if items[i][0] == "m")
                out.append(ComponentSet(hs, ms))
    else:
        rng = np.random.default_rng(seed)
        seen = set()
        while len(out) < n_random:
            k = int(rng.integers(1, (max_size or N) + 1))
            combo = tuple(sorted(rng.choice(N, size=k, replace=False).tolist()))
            if combo in seen:
                continue
            seen.add(combo)
            hs = tuple(items[i][1] for i in combo if items[i][0] == "h")
            ms = tuple(items[i][1] for i in combo if items[i][0] == "m")
            out.append(ComponentSet(hs, ms))
    return out


@torch.no_grad()
def mean_ablation_cache(
    model: TinyTransformer, tokens: np.ndarray, device: torch.device, batch_size: int = 8192
) -> dict[str, torch.Tensor]:
    """Per-position means of every hookable component, used as ablation values."""
    names = [f"z.{l}" for l in range(model.cfg.n_layers)]
    if model.cfg.d_mlp:
        names += [f"mlp_out.{l}" for l in range(model.cfg.n_layers)]
    sums: dict[str, torch.Tensor] = {}
    count = 0
    for s0 in range(0, tokens.shape[0], batch_size):
        tt = torch.tensor(tokens[s0 : s0 + batch_size], device=device)
        _, cache = model.run_with_cache(tt, names)
        for k, v in cache.items():
            s = v.sum(dim=0)
            sums[k] = s if k not in sums else sums[k] + s
        count += tt.shape[0]
    return {k: v / count for k, v in sums.items()}


@torch.no_grad()
def component_scores(
    model: TinyTransformer,
    tokens: np.ndarray,
    circuits: Sequence[ComponentSet],
    abl_cache: dict[str, torch.Tensor],
    eval_positions: np.ndarray,
    targets: np.ndarray,
    distractors: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
    ablation: str = "mean",
    resample_tokens: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Evaluate every circuit by ablating all components outside it.

    Returns a dict with
      ``agree``  (n, m) fraction of evaluation positions where the ablated
                 model's arg-max token equals the full model's arg-max token;
      ``ld``     (n, m) mean logit difference (target minus mean distractor);
      ``ld_full`` (n,)  the same quantity for the unablated model;
      ``ld_empty`` (n,) the same quantity with *all* components ablated.
    """
    model.eval()
    n = tokens.shape[0]
    m = len(circuits)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    pos = torch.tensor(eval_positions, device=device, dtype=torch.long)
    agree = np.zeros((n, m), dtype=np.float32)
    ld = np.zeros((n, m), dtype=np.float32)
    ld_full = np.zeros(n, dtype=np.float32)
    ld_empty = np.zeros(n, dtype=np.float32)

    all_set = ComponentSet(
        tuple((l, h) for l in range(L) for h in range(H)),
        tuple(range(L)) if model.cfg.d_mlp else (),
    )
    empty_set = ComponentSet((), ())

    def make_hooks(cs: ComponentSet, B: int):
        hooks = {}
        keep_heads = set(cs.heads)
        keep_mlps = set(cs.mlps)
        for l in range(L):
            drop = [h for h in range(H) if (l, h) not in keep_heads]
            if drop:
                idx = torch.tensor(drop, device=device, dtype=torch.long)
                base = abl_cache[f"z.{l}"]  # (T, H, dh)

                def zhook(name, t, idx=idx, base=base):
                    t = t.clone()
                    t[:, :, idx] = base[None, :, idx]
                    return t

                hooks[f"z.{l}"] = zhook
            if model.cfg.d_mlp and l not in keep_mlps:
                base_m = abl_cache[f"mlp_out.{l}"]

                def mhook(name, t, base_m=base_m):
                    return base_m[None].expand_as(t).clone()

                hooks[f"mlp_out.{l}"] = mhook
        return hooks

    def metric(logits, tgt, dis):
        # logits: (B, T, V); tgt: (B, P); dis: (B, P, D)
        sel = logits[:, pos]  # (B, P, V)
        pred = sel.argmax(dim=-1)  # (B, P)
        lt = torch.gather(sel, 2, tgt[..., None]).squeeze(-1)  # (B, P)
        ldis = torch.gather(sel, 2, dis).mean(dim=-1)  # (B, P)
        return pred, (lt - ldis).mean(dim=-1)

    for s0 in range(0, n, batch_size):
        s1 = min(n, s0 + batch_size)
        tt = torch.tensor(tokens[s0:s1], device=device)
        tgt = torch.tensor(targets[s0:s1], device=device, dtype=torch.long)
        dis = torch.tensor(distractors[s0:s1], device=device, dtype=torch.long)

        logits = model(tt)
        pred_full, ldf = metric(logits, tgt, dis)
        ld_full[s0:s1] = ldf.cpu().numpy()

        lo = model(tt, make_hooks(empty_set, s1 - s0))
        _, lde = metric(lo, tgt, dis)
        ld_empty[s0:s1] = lde.cpu().numpy()

        for j, cs in enumerate(circuits):
            lg = model(tt, make_hooks(cs, s1 - s0))
            pr, l_ = metric(lg, tgt, dis)
            agree[s0:s1, j] = (pr == pred_full).float().mean(dim=-1).cpu().numpy()
            ld[s0:s1, j] = l_.cpu().numpy()

    return {"agree": agree, "ld": ld, "ld_full": ld_full, "ld_empty": ld_empty}
