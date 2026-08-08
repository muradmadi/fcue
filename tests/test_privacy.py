"""Tests for the privacy pipeline -- the important property is that the
down-sampling + inverse-propensity correction is UNBIASED, and that the DP noise
is zero-mean."""

import numpy as np

from fcue.bandit import aggregate_counts
from fcue.privacy import PrivacyConfig, PrivacyPipeline, split_budget


def _fixed_population(n=4000):
    rng = np.random.default_rng(0)
    clusters = rng.integers(0, 8, n)
    arms = rng.integers(0, 2, n)
    bought = (rng.random(n) < 0.3).astype(int)
    return clusters, arms, bought


def test_split_budget_is_sequential_composition():
    assert split_budget(1.0, 100) == 0.01
    assert split_budget(2.0, 1) == 2.0


def test_noise_scale_formula():
    p = PrivacyPipeline(PrivacyConfig(epsilon=0.5, sensitivity=2.0))
    assert p.laplace_scale == 4.0                       # sensitivity / epsilon
    assert np.isclose(p.noise_std, 4.0 * np.sqrt(2.0))  # Laplace std = b * sqrt(2)


def test_downsampling_ipw_is_unbiased():
    """With negligible noise, averaging many privatized exposure reports must
    recover the true exposure counts despite reporting only half the events."""
    clusters, arms, bought = _fixed_population()
    true_exp, _ = aggregate_counts(clusters, arms, bought, 8)
    accum = np.zeros((8, 2))
    trials = 400
    for s in range(trials):
        p = PrivacyPipeline(PrivacyConfig(epsilon=1e9, exposure_keep_rate=0.5, seed=s))
        exp, _ = p.privatize(clusters, arms, bought, 8)
        accum += exp
    mean_exp = accum / trials
    mask = true_exp > 50  # only judge buckets with enough mass to be stable
    rel_err = np.abs(mean_exp[mask] - true_exp[mask]) / true_exp[mask]
    assert rel_err.max() < 0.05


def test_dp_noise_is_zero_mean():
    """Averaging many noisy purchase reports must converge to the true counts."""
    clusters, arms, bought = _fixed_population()
    _, true_pur = aggregate_counts(clusters, arms, bought, 8)
    accum = np.zeros((8, 2))
    trials = 600
    for s in range(trials):
        p = PrivacyPipeline(PrivacyConfig(epsilon=1.0, exposure_keep_rate=1.0, seed=s))
        _, pur = p.privatize(clusters, arms, bought, 8)
        accum += pur
    mean_pur = accum / trials
    assert np.abs(mean_pur - true_pur).max() < 3.0


def test_negligible_noise_recovers_exact_counts():
    clusters, arms, bought = _fixed_population()
    true_exp, true_pur = aggregate_counts(clusters, arms, bought, 8)
    p = PrivacyPipeline(PrivacyConfig(epsilon=1e12, exposure_keep_rate=1.0, seed=0))
    exp, pur = p.privatize(clusters, arms, bought, 8)
    assert np.allclose(exp, true_exp, atol=1e-3)
    assert np.allclose(pur, true_pur, atol=1e-3)
