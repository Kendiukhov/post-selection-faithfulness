"""Indirect Object Identification (IOI) harness for GPT-2 small.

Reproduces the standard evaluation of Wang, Variengien, Conmy, Shlegeris and
Steinhardt (2023): prompts of the form

    "Then, Mary and John went to the store. John gave a drink to"

where the correct continuation is the *indirect object* (" Mary") rather than
the repeated *subject* (" John").  Faithfulness of a circuit is measured by how
much of the model's logit difference survives when every attention head outside
the circuit is mean-ablated over the ABC distribution (three distinct names),
as in the original paper.

Design choices that matter for the statistics:

* prompts are generated from a fixed list of **templates**; every prompt records
  its template id so that cluster-robust (template-level) inference is possible;
* ablation means are computed **per template and per token position**, so that
  positions stay aligned (templates have different lengths);
* per-instance quantities are returned, never just their averages, so that any
  resampling scheme can be applied afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "IOI_TEMPLATES",
    "NAMES",
    "OBJECTS",
    "PLACES",
    "IOIDataset",
    "GPT2Patcher",
    "IOI_CIRCUIT_GROUPS",
    "ioi_circuit_heads",
]

# 15 base templates from the IOI paper's generator, used in both name orders
# (ABBA and BABA), giving 30 template clusters.
_BASE_TEMPLATES = [
    "Then, {n1} and {n2} went to the {place}. {s} gave a {obj} to",
    "Then, {n1} and {n2} had a lot of fun at the {place}. {s} gave a {obj} to",
    "Then, {n1} and {n2} were working at the {place}. {s} decided to give a {obj} to",
    "Then, {n1} and {n2} were thinking about going to the {place}. {s} wanted to give a {obj} to",
    "Then, {n1} and {n2} had a long argument, and afterwards {s} said to",
    "After {n1} and {n2} went to the {place}, {s} gave a {obj} to",
    "When {n1} and {n2} got a {obj} at the {place}, {s} decided to give it to",
    "When {n1} and {n2} got a {obj} at the {place}, {s} decided to give the {obj} to",
    "While {n1} and {n2} were working at the {place}, {s} gave a {obj} to",
    "While {n1} and {n2} were commuting to the {place}, {s} gave a {obj} to",
    "After the lunch, {n1} and {n2} went to the {place}. {s} gave a {obj} to",
    "Afterwards, {n1} and {n2} went to the {place}. {s} gave a {obj} to",
    "Then, {n1} and {n2} had a long argument. Afterwards {s} said to",
    "The {place} {n1} and {n2} went to had a {obj}. {s} gave it to",
    "Friends {n1} and {n2} found a {obj} at the {place}. {s} gave it to",
]

# (template_string, order) with order == "ABBA" meaning the subject appears
# second in the introduction clause, "BABA" meaning it appears first.
IOI_TEMPLATES = [(t, order) for t in _BASE_TEMPLATES for order in ("ABBA", "BABA")]

NAMES = [
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer", "Joshua",
    "Amanda", "Daniel", "David", "James", "Robert", "John", "Joseph", "Andrew",
    "Ryan", "Brandon", "Jason", "Justin", "Sarah", "William", "Jonathan", "Stephanie",
    "Brian", "Nicole", "Nicholas", "Anthony", "Heather", "Eric", "Elizabeth", "Adam",
    "Megan", "Melissa", "Kevin", "Steven", "Thomas", "Timothy", "Christina", "Kyle",
    "Rachel", "Laura", "Lauren", "Amber", "Brittany", "Danielle", "Richard", "Kimberly",
    "Jeffrey", "Amy", "Crystal", "Michelle", "Tiffany", "Jeremy", "Benjamin", "Mark",
    "Emily", "Aaron", "Charles", "Rebecca", "Jacob", "Stephen", "Patrick", "Sean",
    "Erin", "Zachary", "Jamie", "Kelly", "Samantha", "Nathan", "Sara", "Dustin",
    "Paul", "Angela", "Tyler", "Scott", "Katherine", "Andrea", "Gregory", "Erica",
    "Mary", "Travis", "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen", "Jose",
    "Alexander", "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney",
    "Christine", "Alicia", "Cody", "Allison", "Bradley", "Samuel",
]

OBJECTS = [
    "ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack",
    "book", "flower", "guitar", "camera", "letter", "ticket", "sandwich", "phone",
]

PLACES = [
    "store", "garden", "restaurant", "school", "hospital", "office", "house",
    "station", "market", "library", "museum", "park", "gym", "cafe", "theater",
]


@dataclass
class IOIDataset:
    """A batch of IOI prompts grouped by template.

    ``groups`` is a list of dicts, one per template, each holding
    ``tokens`` (K, L), ``tokens_abc`` (K, L), ``io_tok`` (K,), ``s_tok`` (K,)
    and ``template_id``.
    """

    groups: list[dict] = field(default_factory=list)
    n_total: int = 0

    def template_ids(self) -> np.ndarray:
        return np.concatenate(
            [np.full(g["tokens"].shape[0], g["template_id"]) for g in self.groups]
        )


def build_ioi_dataset(
    tokenizer,
    n_per_template: int = 300,
    templates: Sequence[tuple[str, str]] = tuple(IOI_TEMPLATES),
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> IOIDataset:
    """Generate the dataset.  Only names that are a single GPT-2 token with a
    leading space are used, so that the logit difference is well defined."""
    rng = np.random.default_rng(seed)
    good_names = [
        nm for nm in NAMES if len(tokenizer.encode(" " + nm)) == 1
    ]
    if len(good_names) < 10:  # pragma: no cover
        raise RuntimeError("too few single-token names")
    name_tok = {nm: tokenizer.encode(" " + nm)[0] for nm in good_names}

    ds = IOIDataset()
    for tid, (tmpl, order) in enumerate(templates):
        prompts, ios, subs, prompts_abc = [], [], [], []
        for _ in range(n_per_template):
            a, b = rng.choice(len(good_names), size=2, replace=False)
            io, s = good_names[a], good_names[b]
            obj = OBJECTS[rng.integers(len(OBJECTS))]
            place = PLACES[rng.integers(len(PLACES))]
            n1, n2 = (io, s) if order == "ABBA" else (s, io)
            prompts.append(tmpl.format(n1=n1, n2=n2, s=s, obj=obj, place=place))
            ios.append(name_tok[io])
            subs.append(name_tok[s])
            # ABC distribution: three distinct names, no repetition
            c1, c2, c3 = rng.choice(len(good_names), size=3, replace=False)
            prompts_abc.append(
                tmpl.format(
                    n1=good_names[c1], n2=good_names[c2], s=good_names[c3],
                    obj=obj, place=place,
                )
            )
        enc = tokenizer(prompts, return_tensors="np")["input_ids"]
        enc_abc = tokenizer(prompts_abc, return_tensors="np")["input_ids"]
        if enc.shape[1] != enc_abc.shape[1]:  # pragma: no cover
            raise RuntimeError(f"length mismatch for template {tid}")
        ds.groups.append(
            {
                "template_id": tid,
                "template": tmpl,
                "order": order,
                "tokens": enc.astype(np.int64),
                "tokens_abc": enc_abc.astype(np.int64),
                "io_tok": np.array(ios, dtype=np.int64),
                "s_tok": np.array(subs, dtype=np.int64),
            }
        )
        ds.n_total += enc.shape[0]
    return ds


# ---------------------------------------------------------------------------
# head-level patching for HuggingFace GPT-2
# ---------------------------------------------------------------------------


class GPT2Patcher:
    """Head-level mean ablation for ``transformers``' GPT-2.

    Registers a forward *pre*-hook on every block's ``attn.c_proj``.  Its input
    is exactly the concatenation of the per-head attention outputs ``z``, so
    replacing the slice belonging to head ``h`` with a fixed value ablates that
    head's contribution to the residual stream, leaving everything else intact.
    """

    def __init__(self, model, n_layers: int = 12, n_heads: int = 12):
        self.model = model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = model.config.n_embd // n_heads
        self.keep = torch.ones(n_layers, n_heads, dtype=torch.bool)
        self.means: dict[int, torch.Tensor] = {}  # layer -> (T, H, dh)
        self.enabled = False
        self._capture: Optional[dict[int, torch.Tensor]] = None
        self._handles = []
        for l in range(n_layers):
            mod = model.transformer.h[l].attn.c_proj
            self._handles.append(mod.register_forward_pre_hook(self._make_hook(l)))

    def _make_hook(self, layer: int):
        def hook(module, args):
            z = args[0]
            B, T, D = z.shape
            zz = z.view(B, T, self.n_heads, self.d_head)
            if self._capture is not None:
                self._capture[layer] = zz.detach().sum(dim=0)  # (T, H, dh)
            if not self.enabled:
                return None
            drop = ~self.keep[layer]
            if drop.any():
                mean = self.means[layer]  # (T, H, dh) or (B, T, H, dh)
                zz = zz.clone()
                if mean.dim() == 3:
                    zz[:, :, drop] = mean[None, :, drop].to(zz.dtype)
                else:
                    zz[:, :, drop] = mean[:, :, drop].to(zz.dtype)
                return (zz.view(B, T, D),) + tuple(args[1:])
            return None

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- API ---------------------------------------------------------------
    @torch.no_grad()
    def compute_means(self, tokens: torch.Tensor, batch_size: int = 256) -> None:
        """Set the ablation values to the per-position mean over ``tokens``."""
        self.enabled = False
        sums: dict[int, torch.Tensor] = {}
        n = tokens.shape[0]
        for s0 in range(0, n, batch_size):
            self._capture = {}
            self.model(tokens[s0 : s0 + batch_size])
            for l, v in self._capture.items():
                sums[l] = v if l not in sums else sums[l] + v
        self._capture = None
        self.means = {l: v / n for l, v in sums.items()}

    @torch.no_grad()
    def compute_means_grouped(
        self, tokens: torch.Tensor, groups: np.ndarray, batch_size: int = 256
    ) -> None:
        """Per-*group* (e.g. per-template) per-position ablation means, expanded
        to one value per instance and stored as ``means_full``.

        This lets prompts from several templates of the same token length share a
        single forward pass while each is still ablated towards the mean of its
        own template, which is what keeps token positions comparable.  Use
        ``use_slice`` to align the means with a mini-batch.
        """
        uniq, inv = np.unique(groups, return_inverse=True)
        self.means_full = self._grouped_means(tokens, inv, uniq.size, batch_size)
        self.means = self.means_full

    def use_slice(self, sl: slice) -> None:
        """Restrict the per-instance ablation means to a mini-batch."""
        self.means = {l: v[sl] for l, v in self.means_full.items()}

    @torch.no_grad()
    def _grouped_means(self, tokens, inv, n_groups, batch_size):
        store: dict[int, torch.Tensor] = {}
        n, T = tokens.shape
        H, dh = self.n_heads, self.d_head
        dev = tokens.device
        for l in range(self.n_layers):
            store[l] = torch.zeros(n_groups, T, H, dh, device=dev)
        counts = torch.zeros(n_groups, device=dev)
        cap: dict[int, torch.Tensor] = {}

        handles = []
        for l in range(self.n_layers):
            mod = self.model.transformer.h[l].attn.c_proj

            def mk(l):
                def hook(module, args):
                    z = args[0]
                    cap[l] = z.view(z.shape[0], z.shape[1], H, dh).detach()
                    return None

                return hook

            handles.append(mod.register_forward_pre_hook(mk(l)))
        prev_enabled, self.enabled = self.enabled, False
        try:
            for s0 in range(0, n, batch_size):
                sl = slice(s0, min(n, s0 + batch_size))
                cap.clear()
                self.model(tokens[sl])
                gi = torch.as_tensor(inv[sl], device=dev, dtype=torch.long)
                counts.index_add_(0, gi, torch.ones(gi.shape[0], device=dev))
                for l in range(self.n_layers):
                    store[l].index_add_(0, gi, cap[l].float())
        finally:
            for h in handles:
                h.remove()
            self.enabled = prev_enabled
        cnt = counts.clamp(min=1).view(-1, 1, 1, 1)
        inv_t = torch.as_tensor(inv, device=dev, dtype=torch.long)
        return {l: (store[l] / cnt)[inv_t] for l in range(self.n_layers)}

    def set_circuit(self, heads: Iterable[tuple[int, int]]) -> None:
        self.keep.zero_()
        for l, h in heads:
            self.keep[l, h] = True
        self.enabled = True

    def set_full(self) -> None:
        self.keep.fill_(True)
        self.enabled = False


# ---------------------------------------------------------------------------
# the published IOI circuit, grouped by the functional role assigned to each
# head by Wang et al. (2023), Figure 2
# ---------------------------------------------------------------------------

IOI_CIRCUIT_GROUPS: dict[str, tuple[tuple[int, int], ...]] = {
    "name_mover": ((9, 9), (10, 0), (9, 6)),
    "backup_name_mover": ((10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)),
    "negative_name_mover": ((10, 7), (11, 10)),
    "s_inhibition": ((7, 3), (7, 9), (8, 6), (8, 10)),
    "induction": ((5, 5), (5, 8), (5, 9), (6, 9)),
    "duplicate_token": ((0, 1), (0, 10), (3, 0)),
    "previous_token": ((2, 2), (4, 11)),
}


def ioi_circuit_heads() -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for v in IOI_CIRCUIT_GROUPS.values():
        out.extend(v)
    return tuple(sorted(set(out)))
