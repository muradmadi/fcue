"""Tests for the bandit -- the headline guarantee is that it recovers the correct
treat/skip policy from aggregate counts alone."""

import numpy as np

from fcue.bandit import ContextualThompsonBandit, aggregate_counts
from fcue.context import coarse_cluster, n_clusters
from fcue.world import World, WorldConfig


def _train_clean(seed=0, T=150, batch=2000):
    w = World(WorldConfig(seed=seed))
    K = n_clusters()
    b = ContextualThompsonBandit(K, w.cfg.margin, w.cfg.coupon_cost, seed=1)
    for _ in range(T):
        X = w.sample_users(batch)
        c = coarse_cluster(X)
        arms = b.assign(c)
        bought, _ = w.step(X, arms)
        exp, pur = aggregate_counts(c, arms, bought, K)
        b.update_from_counts(exp, pur)
    return w, b


def test_recovers_treat_skip_policy_on_clean_data():
    w, b = _train_clean()
    K = n_clusters()
    X = w.sample_users(100_000)
    c = coarse_cluster(X)
    inc = w.true_incremental_profit(X)
    truth = np.array([inc[c == k].mean() > 0 for k in range(K)])
    policy = b.greedy_policy().astype(bool)
    # allow one borderline (near-zero-signal) cluster to be wrong
    assert (policy == truth).sum() >= K - 1


def test_update_keeps_parameters_positive_under_heavy_noise():
    """Differential-privacy noise can make reported counts negative; the update
    must never drive a Beta parameter to <= 0."""
    b = ContextualThompsonBandit(4, margin=100, coupon_cost=15)
    b.update_from_counts(np.full((4, 2), -50.0), np.full((4, 2), -50.0))
    assert np.all(b.alpha > 0) and np.all(b.beta > 0)


def test_aggregate_counts_sums_correctly():
    clusters = np.array([0, 0, 1])
    arms = np.array([1, 1, 0])
    bought = np.array([1, 0, 1])
    exp, pur = aggregate_counts(clusters, arms, bought, n_clusters=2)
    assert exp[0, 1] == 2 and pur[0, 1] == 1   # two exposures in (c0, treat), one bought
    assert exp[1, 0] == 1 and pur[1, 0] == 1   # one exposure in (c1, control), bought


def test_greedy_policy_shape():
    b = ContextualThompsonBandit(n_clusters(), margin=100, coupon_cost=15)
    pol = b.greedy_policy()
    assert pol.shape == (n_clusters(),)
    assert set(np.unique(pol)).issubset({0, 1})
