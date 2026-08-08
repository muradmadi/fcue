"""Tests for the evaluation tooling -- capture metrics and uplift curves."""

import numpy as np
import pytest

from fcue.baselines import best_capture, incremental_curve, policy_capture


def test_policy_capture_bounds():
    inc = np.array([5.0, -2.0, 3.0, -1.0])
    assert policy_capture(np.array([1, 0, 1, 0], dtype=bool), inc) == pytest.approx(1.0)  # oracle
    assert policy_capture(np.zeros(4, dtype=bool), inc) == 0.0                             # none
    assert policy_capture(np.ones(4, dtype=bool), inc) < 1.0                               # treats losers too


def test_best_capture_of_oracle_ranking_is_one():
    inc = np.random.default_rng(0).normal(0.0, 3.0, 5000)
    _, _, cap = best_capture(inc, inc)  # ranking users by true value
    assert cap == pytest.approx(1.0, abs=1e-9)


def test_incremental_curve_endpoints():
    inc = np.array([2.0, -1.0, 4.0])
    frac, curve = incremental_curve(np.array([0.1, 0.9, 0.5]), inc)
    assert frac[-1] == pytest.approx(1.0)
    assert curve[-1] == pytest.approx(inc.sum() / len(inc))  # treating everyone = mean(inc)


def test_random_ranking_underperforms_oracle():
    rng = np.random.default_rng(1)
    inc = rng.normal(0.0, 2.0, 20_000)
    _, _, cap_random = best_capture(rng.random(len(inc)), inc)
    _, _, cap_oracle = best_capture(inc, inc)
    assert cap_random < cap_oracle
