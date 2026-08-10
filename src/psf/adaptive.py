"""Adaptive circuit search and mechanisms that survive it.

Circuit discovery in practice is *sequential*: a greedy pruner asks thousands of
questions of the same evaluation set, and each question depends on the answers
to the previous ones.  A union bound over the questions actually asked is then
invalid, because the question set is itself random.  Three principled options
remain, all implemented here:

``greedy_prune``      the search itself (an ACDC-style pruner), instrumented so
                      that we can count queries and replay it under different
                      answering mechanisms;
``Thresholdout``      the reusable holdout of Dwork et al. (2015), which answers
                      adaptive queries from a holdout set while spending a
                      budget only on queries that actually overfit;
``reachable_log_size`` the log-cardinality of the set of circuits the search
                      could have returned, which is what a rigorous union bound
                      must pay for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.special import gammaln

__all__ = ["Thresholdout", "greedy_prune", "reachable_log_size", "GreedyResult"]


# ---------------------------------------------------------------------------
# reusable holdout
# ---------------------------------------------------------------------------


@dataclass
class Thresholdout:
    """The Thresholdout mechanism (Dwork, Feldman, Hardt, Pitassi, Reingold and
    Roth, *Science* 2015).

    ``train`` and ``holdout`` are per-instance score vectors for the query being
    asked.  The mechanism reports the training answer unless it differs from the
    holdout answer by more than a noisy threshold, in which case it reports a
    noisy holdout answer and spends one unit of budget.  Because the holdout is
    only ever touched through a differentially private channel, its statistics
    stay close to the population's over many adaptive queries.
    """

    threshold: float = 0.04
    sigma: float = 0.02
    budget: int = 40
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    spent: int = 0
    n_queries: int = 0

    def query(self, train_scores: np.ndarray, holdout_scores: np.ndarray) -> float:
        """Answer one adaptive query and return the reported value."""
        self.n_queries += 1
        a_t = float(np.mean(train_scores))
        a_h = float(np.mean(holdout_scores))
        if self.spent >= self.budget:
            return a_t
        gamma = self.rng.laplace(0.0, 2.0 * self.sigma)
        if abs(a_h - a_t) > self.threshold + gamma:
            self.spent += 1
            return a_h + self.rng.laplace(0.0, self.sigma)
        return a_t


# ---------------------------------------------------------------------------
# greedy pruning search
# ---------------------------------------------------------------------------


@dataclass
class GreedyResult:
    path: list[np.ndarray]  # boolean masks over components, one per round
    reported: list[float]  # answer used by the search at each round
    removed: list[int]
    n_queries: int


def greedy_prune(
    score_fn: Callable[[np.ndarray], float],
    n_components: int,
    n_rounds: Optional[int] = None,
    start_mask: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> GreedyResult:
    """Remove components one at a time, always dropping the one whose removal
    hurts the reported score least.

    ``score_fn(mask) -> float`` is the (possibly noisy, possibly mechanism-
    mediated) answer the searcher receives.  Every call is counted as one query.
    """
    mask = np.ones(n_components, dtype=bool) if start_mask is None else start_mask.copy()
    n_rounds = int(mask.sum()) if n_rounds is None else n_rounds
    path = [mask.copy()]
    reported = [score_fn(mask)]
    removed: list[int] = []
    n_q = 1
    for _ in range(n_rounds):
        alive = np.flatnonzero(mask)
        if alive.size == 0:
            break
        best, best_val = -1, -np.inf
        for j in alive:
            trial = mask.copy()
            trial[j] = False
            v = score_fn(trial)
            n_q += 1
            if v > best_val:
                best, best_val = int(j), v
        mask[best] = False
        removed.append(best)
        path.append(mask.copy())
        reported.append(best_val)
        if verbose:
            print(f"    removed {best}; reported score {best_val:.4f}; |c| = {mask.sum()}")
    return GreedyResult(path=path, reported=reported, removed=removed, n_queries=n_q)


def reachable_log_size(n_components: int, n_rounds: int) -> float:
    """``log |{ subsets reachable by removing at most n_rounds components }|``.

    A union bound that is valid for whatever the greedy search happens to return
    must cover this whole set, not just the queries that were issued.
    """
    N, d = int(n_components), int(min(n_rounds, n_components))
    ks = np.arange(0, d + 1)
    log_binom = gammaln(N + 1) - gammaln(ks + 1) - gammaln(N - ks + 1)
    mx = log_binom.max()
    return float(mx + np.log(np.exp(log_binom - mx).sum()))
