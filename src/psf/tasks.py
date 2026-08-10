"""Algorithmic tasks with an explicit high-level causal model.

Each task supplies

  * a sampler for token sequences,
  * the high-level causal model ``H`` (a small program with named intermediate
    variables), and
  * a counterfactual evaluator: given a *base* input, a *source* input and a
    high-level variable ``V``, the label that ``H`` outputs when ``V`` is set to
    the value it takes on the source.

That last piece is exactly what an interchange intervention tests, so the tasks
give us ground truth for interchange-intervention accuracy (IIA) in the sense of
causal abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["EqualityTask", "ArithmeticTask", "InductionTask"]


# ---------------------------------------------------------------------------
# Task 1: hierarchical equality,  y = [ (a == b) == (c == d) ]
# ---------------------------------------------------------------------------


@dataclass
class EqualityTask:
    """Sequence ``[BOS, a, b, c, d]``; the label is read at the final position.

    High-level causal model::

        V1 = 1{a == b}
        V2 = 1{c == d}
        Y  = 1{V1 == V2}

    Token ids: symbols ``0 .. n_symbols-1``; ``BOS = n_symbols``; the two label
    tokens are ``n_symbols+1`` (FALSE) and ``n_symbols+2`` (TRUE).
    """

    n_symbols: int = 10

    @property
    def n_vocab(self) -> int:
        return self.n_symbols + 3

    @property
    def bos(self) -> int:
        return self.n_symbols

    @property
    def label_tokens(self) -> tuple[int, int]:
        return (self.n_symbols + 1, self.n_symbols + 2)

    seq_len: int = 5
    variables: tuple[str, ...] = ("V1", "V2")
    # position of each token in the sequence
    positions: dict = None  # filled in __post_init__

    def __post_init__(self):
        self.positions = {"BOS": 0, "a": 1, "b": 2, "c": 3, "d": 4}

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``(n, 5)`` int tokens with balanced V1, V2."""
        S = self.n_symbols
        a = rng.integers(0, S, n)
        c = rng.integers(0, S, n)
        eq1 = rng.random(n) < 0.5
        eq2 = rng.random(n) < 0.5
        b = np.where(eq1, a, (a + rng.integers(1, S, n)) % S)
        d = np.where(eq2, c, (c + rng.integers(1, S, n)) % S)
        toks = np.stack([np.full(n, self.bos), a, b, c, d], axis=1)
        return toks.astype(np.int64)

    def high_level(self, toks: np.ndarray) -> dict:
        a, b, c, d = toks[:, 1], toks[:, 2], toks[:, 3], toks[:, 4]
        V1 = (a == b).astype(np.int64)
        V2 = (c == d).astype(np.int64)
        return {"V1": V1, "V2": V2, "Y": (V1 == V2).astype(np.int64)}

    def counterfactual_label(
        self, base: np.ndarray, source: np.ndarray, variable: str
    ) -> np.ndarray:
        hb = self.high_level(base)
        hs = self.high_level(source)
        V1 = hs["V1"] if variable == "V1" else hb["V1"]
        V2 = hs["V2"] if variable == "V2" else hb["V2"]
        return (V1 == V2).astype(np.int64)

    def readout_positions(self) -> int:
        return self.seq_len - 1


# ---------------------------------------------------------------------------
# Task 2: three-term modular arithmetic with an intermediate sum
# ---------------------------------------------------------------------------


@dataclass
class ArithmeticTask:
    """Sequence ``[BOS, a, b, c]``; label ``(a + b + c) mod M`` at the last
    position.

    High-level causal model::

        S1 = (a + b) mod M
        Y  = (S1 + c) mod M

    ``S1`` is the interesting intermediate variable: a network that computes the
    sum left-to-right should represent it somewhere after token ``b``.
    """

    modulus: int = 10
    seq_len: int = 4
    variables: tuple[str, ...] = ("S1",)

    @property
    def n_vocab(self) -> int:
        # digits 0..M-1 are both inputs and outputs; plus BOS
        return self.modulus + 1

    @property
    def bos(self) -> int:
        return self.modulus

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        M = self.modulus
        a, b, c = (rng.integers(0, M, n) for _ in range(3))
        return np.stack([np.full(n, self.bos), a, b, c], axis=1).astype(np.int64)

    def high_level(self, toks: np.ndarray) -> dict:
        M = self.modulus
        a, b, c = toks[:, 1], toks[:, 2], toks[:, 3]
        S1 = (a + b) % M
        return {"S1": S1, "Y": (S1 + c) % M}

    def counterfactual_label(
        self, base: np.ndarray, source: np.ndarray, variable: str
    ) -> np.ndarray:
        M = self.modulus
        hb, hs = self.high_level(base), self.high_level(source)
        S1 = hs["S1"] if variable == "S1" else hb["S1"]
        return (S1 + base[:, 3]) % M

    def readout_positions(self) -> int:
        return self.seq_len - 1


# ---------------------------------------------------------------------------
# Task 3: induction / in-context copying
# ---------------------------------------------------------------------------


@dataclass
class InductionTask:
    """Random sequence of length ``block`` repeated twice.

    In the second copy the next token is fully determined by the first copy, so
    a two-layer attention-only transformer must implement the previous-token /
    induction-head algorithm (Elhage et al., 2021; Olsson et al., 2022).  We
    evaluate at the positions of the second copy.
    """

    n_symbols: int = 40
    block: int = 8

    @property
    def n_vocab(self) -> int:
        return self.n_symbols

    @property
    def seq_len(self) -> int:
        return 2 * self.block

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # sample blocks without repeated symbols so the induction target is unique
        blk = np.stack(
            [rng.choice(self.n_symbols, size=self.block, replace=False) for _ in range(n)]
        )
        return np.concatenate([blk, blk], axis=1).astype(np.int64)

    def eval_positions(self) -> np.ndarray:
        """Positions at which the next token is predictable by induction."""
        return np.arange(self.block, 2 * self.block - 1)

    def targets(self, toks: np.ndarray) -> np.ndarray:
        pos = self.eval_positions()
        return toks[:, pos + 1]
