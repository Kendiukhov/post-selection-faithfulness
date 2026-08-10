"""Synthetic generators of per-instance faithfulness score matrices.

These let us study the statistical behaviour of post-selection bounds under
*known* ground truth ``theta`` and *controlled* correlation between circuit
hypotheses, which is impossible with real models alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["ScoreModel", "equicorrelated_bernoulli", "nested_circuit_scores", "gaussian_scores"]


@dataclass
class ScoreModel:
    """A generator of ``(n, m)`` score matrices with known column means."""

    theta: np.ndarray  # (m,) true means
    sampler: object  # callable (n, rng) -> (n, m)
    name: str = ""

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sampler(n, rng)


def _gauss_copula_bernoulli(
    n: int, theta: np.ndarray, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Bernoulli scores with an equicorrelated Gaussian copula."""
    m = theta.size
    z_common = rng.standard_normal((n, 1))
    z_idio = rng.standard_normal((n, m))
    z = np.sqrt(rho) * z_common + np.sqrt(1.0 - rho) * z_idio
    from scipy import stats as _st

    thresh = _st.norm.ppf(1.0 - theta)[None, :]
    return (z > thresh).astype(np.float64)


def equicorrelated_bernoulli(
    theta: np.ndarray, rho: float = 0.0
) -> ScoreModel:
    """Binary (interchange-accuracy style) scores, equicorrelated across
    hypotheses through a Gaussian copula with correlation ``rho``."""
    theta = np.asarray(theta, dtype=np.float64)

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return _gauss_copula_bernoulli(n, theta, rho, rng)

    return ScoreModel(theta=theta, sampler=sampler, name=f"bern(rho={rho})")


def gaussian_scores(theta: np.ndarray, sigma: float = 0.25, rho: float = 0.0) -> ScoreModel:
    theta = np.asarray(theta, dtype=np.float64)
    m = theta.size

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        zc = rng.standard_normal((n, 1))
        zi = rng.standard_normal((n, m))
        z = np.sqrt(rho) * zc + np.sqrt(1 - rho) * zi
        return theta[None, :] + sigma * z

    return ScoreModel(theta=theta, sampler=sampler, name=f"gauss(rho={rho})")


def nested_circuit_scores(
    n_components: int = 12,
    max_size: Optional[int] = None,
    base_rate: float = 0.55,
    per_component: float = 0.03,
    noise_rho: float = 0.6,
    seed: int = 0,
) -> tuple[ScoreModel, list[tuple[int, ...]]]:
    """A stylised model of *structured* circuit hypotheses.

    Each hypothesis is a subset of ``n_components`` model components.  The true
    faithfulness of a subset increases with how many of the "truly important"
    components it contains.  Per-instance scores of two subsets are correlated
    through the components they share, which mimics the strong positive
    dependence between overlapping circuits in real interpretability searches.

    Returns the score model together with the list of subsets (as tuples).
    """
    from itertools import combinations

    rng0 = np.random.default_rng(seed)
    importance = np.sort(rng0.random(n_components))[::-1]
    importance = importance / importance.sum()

    max_size = n_components if max_size is None else max_size
    subsets: list[tuple[int, ...]] = []
    for k in range(1, max_size + 1):
        subsets.extend(combinations(range(n_components), k))
    m = len(subsets)
    masks = np.zeros((m, n_components))
    for j, s in enumerate(subsets):
        masks[j, list(s)] = 1.0
    theta = np.clip(base_rate + per_component * n_components * (masks @ importance), 0.0, 0.98)

    # unit-norm rows => each column of ``comp @ share.T`` has unit variance and
    # the correlation between two hypotheses equals their cosine overlap
    share = masks / np.sqrt(np.maximum(masks.sum(axis=1, keepdims=True), 1.0))

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        # a shared per-instance "difficulty" plus component-specific noise
        diff = rng.standard_normal((n, 1))
        comp = rng.standard_normal((n, n_components))
        z = np.sqrt(noise_rho) * diff + np.sqrt(1 - noise_rho) * (comp @ share.T)
        from scipy import stats as _st

        thresh = _st.norm.ppf(1 - theta)[None, :]
        return (z > thresh).astype(np.float64)

    return ScoreModel(theta=theta, sampler=sampler, name="nested"), subsets
