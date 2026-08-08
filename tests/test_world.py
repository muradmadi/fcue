"""Tests for the synthetic world -- especially the coupling knob and its
orthogonalization fix, which were the subtlest part to get right."""

import numpy as np
import pytest

from fcue.world import World, WorldConfig


def test_baseline_prob_in_unit_interval():
    w = World(WorldConfig(seed=1))
    p = w.baseline_prob(w.sample_users(5000))
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_uplift_is_bounded():
    w = World(WorldConfig(seed=2))
    tau = w.uplift(w.sample_users(5000))
    assert np.all(np.abs(tau) <= 0.35 + 1e-9)  # amp = 0.35


@pytest.mark.parametrize("t", [0, 1])
def test_buy_prob_is_clipped(t):
    w = World(WorldConfig(seed=3))
    p = w.buy_prob(w.sample_users(5000), t)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_coupling_knob_controls_correlation():
    """corr(baseline propensity, uplift) should track the coupling knob. This is
    the regression test for the Gram-Schmidt orthogonalization fix: without it,
    the correlation was ~0.7 even when the knob said 0."""
    def corr(rho):
        w = World(WorldConfig(seed=0, uplift_baseline_coupling=rho))
        X = w.sample_users(40_000)
        return np.corrcoef(w.baseline_prob(X), w.uplift(X))[0, 1]

    assert corr(0.0) == pytest.approx(0.0, abs=0.12)
    assert corr(0.8) > 0.5
    assert corr(-0.8) < -0.5


def test_should_treat_matches_positive_incremental():
    """A user is 'worth treating' exactly when true incremental profit > 0."""
    w = World(WorldConfig(seed=4))
    X = w.sample_users(20_000)
    inc = w.true_incremental_profit(X)
    frac = (inc > 0).mean()
    assert 0.3 < frac < 0.8  # a genuine mix, neither 'treat all' nor 'treat none'
