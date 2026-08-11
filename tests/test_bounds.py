"""Unit and Monte-Carlo coverage tests for every bound.

The coverage tests are the ones that matter: a bound whose implementation is
subtly wrong will still return plausible-looking numbers, but it will not cover.
Each test simulates many independent datasets, applies a data-dependent
selection rule, and checks the empirical coverage of the returned lower bound.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psf import bounds as B  # noqa: E402
from psf import simulate  # noqa: E402
from psf.adaptive import Thresholdout, greedy_prune, reachable_log_size  # noqa: E402
from psf.functional import influence_band, ratio_influence, ratio_point_estimate  # noqa: E402

ALPHA = 0.05
# Monte-Carlo tolerance: with R replications the standard error of an estimated
# coverage near 0.95 is about sqrt(.95*.05/R); we allow three of them.
def tol(R):
    return 3 * np.sqrt(0.95 * 0.05 / R)


# ---------------------------------------------------------------------------
# basic sanity
# ---------------------------------------------------------------------------


def test_bounds_are_below_the_point_estimate():
    rng = np.random.default_rng(0)
    S = (rng.random((300, 20)) < 0.7).astype(float)
    mean = S.mean(0)
    for L in [
        B.naive_lcb(S, ALPHA),
        B.hoeffding_lcb(S, ALPHA),
        B.empirical_bernstein_lcb(S, ALPHA),
        B.union_lcb(S, ALPHA),
        B.bootstrap_max_lcb(S, ALPHA, n_boot=500, seed=0),
    ]:
        assert np.all(L <= mean + 1e-12)


def test_union_bound_is_never_worse_than_its_ingredients_by_much():
    rng = np.random.default_rng(1)
    S = (rng.random((500, 50)) < 0.95).astype(float)
    h = B.hoeffding_lcb(S, ALPHA / 2)
    e = B.empirical_bernstein_lcb(S, ALPHA / 2)
    u = B.union_lcb(S, ALPHA)
    assert np.allclose(u, np.maximum(h, e))
    # for scores concentrated near 1, empirical Bernstein should win
    assert e.mean() > h.mean()


def test_hoeffding_matches_closed_form():
    rng = np.random.default_rng(2)
    S = (rng.random((200, 1)) < 0.5).astype(float)
    L = B.hoeffding_lcb(S, 0.05, family_size=10)[0]
    expected = S.mean() - np.sqrt(np.log(10 / 0.05) / (2 * 200))
    assert abs(L - expected) < 1e-12


def test_empirical_bernstein_matches_maurer_pontil():
    """Theorem 4 of Maurer & Pontil (2009), with the unbiased sample variance
    and ln(2/delta) in both terms."""
    rng = np.random.default_rng(3)
    x = rng.random(400)
    L = B.empirical_bernstein_lcb(x[:, None], 0.05, family_size=1)[0]
    n = 400
    V = x.var(ddof=1)
    lg = np.log(2 / 0.05)
    expected = x.mean() - np.sqrt(2 * V * lg / n) - 7 * lg / (3 * (n - 1))
    assert abs(L - expected) < 1e-12


def test_rescaling_is_consistent():
    rng = np.random.default_rng(4)
    x = rng.random((300, 3))
    a = B.empirical_bernstein_lcb(x, ALPHA)
    b = B.empirical_bernstein_lcb(3 * x - 1, ALPHA, lo=-1.0, hi=2.0)
    assert np.allclose(3 * a - 1, b)


def test_occam_prior_must_be_subprobability():
    rng = np.random.default_rng(5)
    S = (rng.random((100, 4)) < 0.5).astype(float)
    with pytest.raises(ValueError):
        B.occam_lcb(S, np.zeros(4), ALPHA)
    lp = np.full(4, -np.log(4))
    B.occam_lcb(S, lp, ALPHA)  # must not raise


def test_size_stratified_prior_sums_to_at_most_one():
    from scipy.special import logsumexp

    sizes = np.arange(0, 9)
    lp = B.size_stratified_log_prior(sizes, 8)
    # one representative per size: total mass = sum_k 1/((N+1) binom(N,k)) <= 1
    assert logsumexp(lp) <= 1e-9


def test_smaller_circuits_get_tighter_occam_bounds():
    rng = np.random.default_rng(6)
    S = (rng.random((2000, 2)) < 0.9).astype(float)
    sizes = np.array([5, 72])
    lp = B.size_stratified_log_prior(sizes, 144)
    L = B.occam_lcb(S, lp, ALPHA)
    assert L[0] > L[1]  # same data, smaller circuit certified higher


def test_effective_number_of_hypotheses_is_monotone():
    from scipy import stats

    q_bonf = stats.norm.ppf(1 - ALPHA / 200)
    assert B.effective_num_hypotheses(q_bonf, ALPHA) == pytest.approx(200, rel=1e-6)
    assert B.effective_num_hypotheses(1.645, ALPHA) < 2


def test_reachable_set_size():
    # removing every component from N reaches the whole power set
    assert reachable_log_size(10, 10) == pytest.approx(10 * np.log(2), rel=1e-9)
    assert reachable_log_size(10, 0) == pytest.approx(0.0, abs=1e-12)
    assert reachable_log_size(144, 26) < reachable_log_size(144, 72)


# ---------------------------------------------------------------------------
# Monte-Carlo coverage after selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, 0.6])
def test_valid_bounds_cover_after_argmax_selection(rho):
    R, n, m = 400, 400, 60
    theta = np.full(m, 0.75)
    model = simulate.equicorrelated_bernoulli(theta, rho=rho)
    rng = np.random.default_rng(11)
    cov = {k: 0 for k in ["union", "boot", "split", "hybrid"]}
    for r in range(R):
        S = model.sample(n, rng)
        chat = int(S.mean(0).argmax())
        cov["union"] += B.union_lcb(S, ALPHA)[chat] <= theta[chat]
        cov["boot"] += B.bootstrap_max_lcb(S, ALPHA, n_boot=800, seed=r)[chat] <= theta[chat]
        sp = B.split_lcb(S, ALPHA, seed=r)
        cov["split"] += sp.lcb_selected <= theta[sp.selected]
        hy = B.hybrid_winner_lcb(S, ALPHA, n_boot=400, seed=r)
        cov["hybrid"] += hy.lcb_selected <= theta[hy.selected]
    for k, c in cov.items():
        assert c / R >= 1 - ALPHA - tol(R), f"{k} under-covers: {c / R:.3f}"


def test_fixed_sigma_bootstrap_undercovers_for_binary_scores():
    """Documents why ``bootstrap_max_lcb`` recomputes the standard deviation:
    holding it fixed (the textbook multiplier bootstrap) loses coverage when the
    scores are binary, because then the sample mean and sample sd are dependent."""
    R, n, m = 300, 400, 60
    theta = np.full(m, 0.75)
    model = simulate.equicorrelated_bernoulli(theta, rho=0.0)
    rng = np.random.default_rng(17)
    cov_t = cov_f = 0
    for r in range(R):
        S = model.sample(n, rng)
        chat = int(S.mean(0).argmax())
        cov_t += B.bootstrap_max_lcb(S, ALPHA, n_boot=800, seed=r)[chat] <= theta[chat]
        cov_f += (
            B.bootstrap_max_lcb(S, ALPHA, n_boot=800, seed=r, recompute_variance=False)[chat]
            <= theta[chat]
        )
    assert cov_t / R >= 1 - ALPHA - tol(R)
    assert cov_f / R < cov_t / R - 0.02


def test_naive_bound_undercovers_this_is_the_point():
    R, n, m = 400, 400, 60
    theta = np.full(m, 0.75)
    model = simulate.equicorrelated_bernoulli(theta, rho=0.0)
    rng = np.random.default_rng(12)
    cov = 0
    for _ in range(R):
        S = model.sample(n, rng)
        chat = int(S.mean(0).argmax())
        cov += B.naive_lcb(S, ALPHA)[chat] <= theta[chat]
    assert cov / R < 0.5, "the naive bound should fail badly here"


def test_naive_bound_is_fine_without_selection():
    R, n = 600, 400
    theta = np.array([0.75])
    model = simulate.equicorrelated_bernoulli(theta)
    rng = np.random.default_rng(13)
    cov = sum(B.naive_lcb(model.sample(n, rng), ALPHA)[0] <= 0.75 for _ in range(R))
    assert cov / R >= 1 - ALPHA - tol(R)


def test_conditional_bound_covers():
    R, n, m = 400, 500, 12
    theta = np.linspace(0.6, 0.85, m)
    model = simulate.equicorrelated_bernoulli(theta, rho=0.3)
    rng = np.random.default_rng(14)
    cov = 0
    for _ in range(R):
        S = model.sample(n, rng)
        res = B.conditional_winner_lcb(S, ALPHA)
        cov += res.lcb_selected <= theta[res.selected]
    assert cov / R >= 1 - ALPHA - tol(R)


def test_cluster_bootstrap_covers_under_cluster_sampling():
    """i.i.d. resampling under-covers when the data are clustered; the wild
    cluster bootstrap does not."""
    R, G, per, m = 300, 25, 40, 30
    n = G * per
    rng = np.random.default_rng(15)
    theta = np.full(m, 0.7)
    cov_iid = cov_cl = 0
    for r in range(R):
        # cluster-level random effect shared by all instances in a template
        u = rng.normal(0, 1.0, size=(G, 1))
        eps = rng.normal(0, 1.0, size=(n, m))
        z = np.repeat(u, per, axis=0) + eps
        from scipy import stats as st

        thresh = st.norm.ppf(1 - theta) * np.sqrt(2.0)
        S = (z > thresh[None, :]).astype(float)
        clusters = np.repeat(np.arange(G), per)
        chat = int(S.mean(0).argmax())
        cov_iid += B.bootstrap_max_lcb(S, ALPHA, n_boot=600, seed=r)[chat] <= theta[chat]
        cov_cl += (
            B.cluster_bootstrap_max_lcb(S, clusters, ALPHA, n_boot=600, seed=r)[chat]
            <= theta[chat]
        )
    assert cov_cl / R >= 1 - ALPHA - tol(R), f"cluster bootstrap under-covers ({cov_cl / R:.3f})"
    assert cov_iid / R < cov_cl / R, "i.i.d. bootstrap should be the looser one here"


def test_betting_bound_covers_and_is_valid():
    R, n = 200, 100
    rng = np.random.default_rng(16)
    cov = 0
    for _ in range(R):
        x = (rng.random(n) < 0.9).astype(float)[:, None]
        L = B.wsr_betting_lcb(x, ALPHA, family_size=1, grid=200)[0]
        cov += L <= 0.9
    assert cov / R >= 1 - ALPHA - tol(R)


# ---------------------------------------------------------------------------
# ratio metric
# ---------------------------------------------------------------------------


def test_ratio_influence_matches_numerical_derivative():
    rng = np.random.default_rng(21)
    n, m = 500, 4
    ldc = rng.normal(2.0, 1.0, size=(n, m))
    lde = rng.normal(0.2, 1.0, size=n)
    ldf = rng.normal(3.0, 1.0, size=n)
    F, psi = ratio_influence(ldc, lde, ldf)
    assert np.allclose(F, ratio_point_estimate(ldc, lde, ldf))
    assert np.allclose(psi.mean(axis=0), 0.0, atol=1e-10)
    # the influence function should predict the effect of deleting one point
    i = 3
    keep = np.arange(n) != i
    F_del = ratio_point_estimate(ldc[keep], lde[keep], ldf[keep])
    predicted = F - psi[i] / (n - 1)
    assert np.allclose(F_del, predicted, atol=5e-3)


def test_ratio_band_covers_after_selection():
    R, n, m = 300, 800, 25
    rng = np.random.default_rng(22)
    truth_shift = np.linspace(0.0, 0.0, m)  # complete null
    cov_boot = cov_naive = 0
    for r in range(R):
        lde = rng.normal(0.2, 0.5, size=n)
        ldf = rng.normal(3.0, 0.5, size=n)
        ldc = rng.normal(2.4, 0.6, size=(n, m)) + truth_shift
        # population values of this generator
        F_true = (2.4 - 0.2) / (3.0 - 0.2)
        F, psi = ratio_influence(ldc, lde, ldf)
        chat = int(np.argmax(F))
        lb, _ = influence_band(F, psi, alpha=ALPHA, n_boot=600, seed=r)
        cov_boot += lb[chat] <= F_true
        from scipy import stats as st

        se = np.sqrt((psi**2).sum(0) / (n * (n - 1)))
        cov_naive += (F - st.norm.ppf(1 - ALPHA) * se)[chat] <= F_true
    assert cov_boot / R >= 1 - ALPHA - tol(R)
    assert cov_naive / R < cov_boot / R


# ---------------------------------------------------------------------------
# adaptive machinery
# ---------------------------------------------------------------------------


def test_greedy_prune_removes_the_least_useful_component():
    # component j contributes weight j; the greedy pruner should drop them in
    # increasing order of weight
    w = np.array([1.0, 2.0, 3.0, 4.0])

    def score(mask):
        return float((w * mask).sum())

    res = greedy_prune(score, 4, n_rounds=3)
    assert res.removed == [0, 1, 2]
    assert res.n_queries == 1 + 4 + 3 + 2


def test_thresholdout_spends_budget_only_when_answers_disagree():
    rng = np.random.default_rng(31)
    th = Thresholdout(threshold=0.1, sigma=0.01, budget=5, rng=rng)
    same = np.ones(100) * 0.5
    for _ in range(20):
        th.query(same, same)
    assert th.spent == 0
    th2 = Thresholdout(threshold=0.05, sigma=0.01, budget=5, rng=np.random.default_rng(32))
    for _ in range(20):
        th2.query(np.ones(100), np.zeros(100))
    assert th2.spent == 5  # budget exhausted, then it falls back to the train answer


def test_thresholdout_keeps_the_holdout_usable():
    """The classic adaptive-data-analysis demonstration, in miniature.

    Under the complete null every hypothesis has the same value.  An analyst who
    selects directly on the holdout burns it: the holdout score of the selected
    hypothesis is badly biased upwards.  The same analyst, whose queries are
    answered through Thresholdout, leaves the holdout almost unbiased --- which
    is the whole point of a reusable holdout.
    """
    rng = np.random.default_rng(33)
    n, m = 200, 400
    theta = 0.5
    bias_naive, bias_thr = [], []
    for _ in range(30):
        train = (rng.random((n, m)) < theta).astype(float)
        hold = (rng.random((n, m)) < theta).astype(float)
        # (a) select on the holdout, then report the holdout score
        j = int(hold.mean(0).argmax())
        bias_naive.append(hold.mean(0)[j] - theta)
        # (b) select through Thresholdout, then report the holdout score
        th = Thresholdout(threshold=0.05, sigma=0.02, budget=10, rng=rng)
        ans = np.array([th.query(train[:, k], hold[:, k]) for k in range(m)])
        k = int(ans.argmax())
        bias_thr.append(hold.mean(0)[k] - theta)
    assert np.mean(bias_thr) < 0.3 * np.mean(bias_naive)
    assert np.mean(bias_naive) > 0.05
