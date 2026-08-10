"""A small, fully hookable decoder-only transformer.

Hook names follow the ``TransformerLens`` convention closely enough to be
recognisable:

    ``resid_pre.{l}``   residual stream entering block ``l``      (B, T, d_model)
    ``z.{l}``           per-head attention output before W_O      (B, T, H, d_head)
    ``attn_out.{l}``    attention block output                    (B, T, d_model)
    ``resid_mid.{l}``   residual stream after attention           (B, T, d_model)
    ``mlp_out.{l}``     MLP block output                          (B, T, d_model)
    ``resid_post.{l}``  residual stream leaving block ``l``       (B, T, d_model)

A *hook* is a callable ``f(name, tensor) -> tensor``.  Returning a modified
tensor performs an intervention; returning the tensor unchanged makes the hook
a pure reader.  Hooking ``z`` rather than the merged attention output is what
lets us ablate or patch individual attention heads, since the block output is
``sum_h z[:, :, h] @ W_O[h]``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HookFn = Callable[[str, torch.Tensor], torch.Tensor]

__all__ = ["TinyConfig", "TinyTransformer"]


@dataclass
class TinyConfig:
    n_vocab: int = 32
    n_ctx: int = 16
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: Optional[int] = 256  # ``None`` gives an attention-only transformer
    use_ln: bool = True
    init_std: float = 0.1

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


class _Block(nn.Module):
    def __init__(self, cfg: TinyConfig, layer: int):
        super().__init__()
        self.cfg = cfg
        self.layer = layer
        d, h, dh = cfg.d_model, cfg.n_heads, cfg.d_head
        self.ln1 = nn.LayerNorm(d) if cfg.use_ln else nn.Identity()
        self.W_QKV = nn.Linear(d, 3 * d, bias=True)
        self.W_O = nn.Parameter(torch.randn(h, dh, d) * cfg.init_std)
        self.b_O = nn.Parameter(torch.zeros(d))
        if cfg.d_mlp:
            self.ln2 = nn.LayerNorm(d) if cfg.use_ln else nn.Identity()
            self.fc1 = nn.Linear(d, cfg.d_mlp)
            self.fc2 = nn.Linear(cfg.d_mlp, d)

    def forward(self, x: torch.Tensor, hooks: Dict[str, HookFn]) -> torch.Tensor:
        cfg = self.cfg
        B, T, D = x.shape
        H, dh = cfg.n_heads, cfg.d_head
        y = self.ln1(x)
        q, k, v = self.W_QKV(y).split(D, dim=-1)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B,H,T,dh)
        z = z.transpose(1, 2)  # (B,T,H,dh)
        z = _apply(hooks, f"z.{self.layer}", z)
        attn_out = torch.einsum("bthd,hde->bte", z, self.W_O) + self.b_O
        attn_out = _apply(hooks, f"attn_out.{self.layer}", attn_out)
        x = x + attn_out
        x = _apply(hooks, f"resid_mid.{self.layer}", x)
        if cfg.d_mlp:
            m = self.fc2(F.gelu(self.fc1(self.ln2(x))))
            m = _apply(hooks, f"mlp_out.{self.layer}", m)
            x = x + m
        return x


def _apply(hooks: Optional[Dict[str, HookFn]], name: str, t: torch.Tensor) -> torch.Tensor:
    if hooks and name in hooks:
        return hooks[name](name, t)
    return t


class TinyTransformer(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Embedding(cfg.n_vocab, cfg.d_model)
        self.W_pos = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList([_Block(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.use_ln else nn.Identity()
        self.W_U = nn.Linear(cfg.d_model, cfg.n_vocab, bias=False)
        self.apply(self._init)

    def _init(self, mod: nn.Module) -> None:
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            nn.init.normal_(mod.weight, std=self.cfg.init_std)
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                nn.init.zeros_(mod.bias)

    def forward(
        self, tokens: torch.Tensor, hooks: Optional[Dict[str, HookFn]] = None
    ) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.W_E(tokens) + self.W_pos(pos)[None]
        for l, blk in enumerate(self.blocks):
            x = _apply(hooks, f"resid_pre.{l}", x)
            x = blk(x, hooks or {})
            x = _apply(hooks, f"resid_post.{l}", x)
        x = _apply(hooks, "resid_final", x)
        return self.W_U(self.ln_f(x))

    # -- convenience ------------------------------------------------------
    def run_with_cache(
        self, tokens: torch.Tensor, names: list[str]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cache: Dict[str, torch.Tensor] = {}

        def make(nm: str) -> HookFn:
            def fn(name: str, t: torch.Tensor) -> torch.Tensor:
                cache[name] = t.detach()
                return t

            return fn

        hooks = {nm: make(nm) for nm in names}
        out = self(tokens, hooks)
        return out, cache

    def hook_names(self) -> list[str]:
        names = []
        for l in range(self.cfg.n_layers):
            names += [f"resid_pre.{l}", f"z.{l}", f"attn_out.{l}", f"resid_mid.{l}"]
            if self.cfg.d_mlp:
                names.append(f"mlp_out.{l}")
            names.append(f"resid_post.{l}")
        names.append("resid_final")
        return names
