"""Experiment engine: runs the closed loop many times and returns tidy numbers.

Deliberately contains NO plotting -- it just returns arrays/dicts. The plotting
lives in experiments/make_figures.py. That separation means you can reuse these
runners in a notebook, a test, or a different chart without dragging matplotlib
around.

Every metric here is "fraction of the oracle prize captured": the incremental
profit a policy earns, divided by the incremental profit a perfect policy would
earn. 1.0 = perfect targeting, 0.0 = no better than treating nobody, negative =
actively destroying value. We score policies on EXPECTED incremental profit
(deterministic), so there's no Monte-Carlo wobble -- averaging over seeds then
reflects real variation across populations, not coin-flips.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .baselines import LogisticPropensity, best_capture, incremental_curve, policy_capture
from .bandit import ContextualThompsonBandit, aggregate_counts
from .context import coarse_cluster, n_clusters
from .privacy import PrivacyConfig, PrivacyPipeline, split_budget
from .world import World, WorldConfig


def _holdout(world: World, n: int = 100_000):
    """A fixed evaluation set: features, their clusters, and true incremental profit."""
    Xh = world.sample_users(n)
    return Xh, coarse_cluster(Xh), world.true_incremental_profit(Xh)


def _fit_propensity(world: World, n: int = 40_000) -> LogisticPropensity:
    """Train a lead-score on historical (no-coupon) purchase behavior."""
    Xtr = world.sample_users(n)
    y_hist, _ = world.step(Xtr, np.zeros(len(Xtr), dtype=int))
    return LogisticPropensity().fit(Xtr, y_hist)


def _run_bandit(world: World, T: int, batch: int, seed: int, epsilon: float | None,
                keep: float, holdout_clusters: np.ndarray, holdout_inc: np.ndarray):
    """Run the online loop for T rounds. Returns three things:

      cum_capture[r]  : CUMULATIVE fraction of achievable value actually earned
                        through round r -- the honest "online" score that pays
                        the price of early exploration. Smooth and rises as the
                        policy improves. This is the learning curve.
      policy_cap[r]   : the current GREEDY policy's capture on the holdout after
                        round r -- what the final deployed policy would be worth.
      bandit          : the trained model.

    epsilon=None -> clean aggregates; else the privacy pipeline runs with a
    per-release budget of split_budget(epsilon, T).
    """
    K = n_clusters()
    bandit = ContextualThompsonBandit(K, world.cfg.margin, world.cfg.coupon_cost, seed=seed)
    priv = None
    if epsilon is not None:
        priv = PrivacyPipeline(PrivacyConfig(epsilon=split_budget(epsilon, T),
                                             exposure_keep_rate=keep, seed=seed + 7))
    realized = np.empty(T)
    optimal = np.empty(T)
    policy_cap = np.empty(T)
    for r in range(T):
        X = world.sample_users(batch)
        c = coarse_cluster(X)
        arms = bandit.assign(c)
        # Bookkeeping (evaluation only, never fed to the learner): the expected
        # incremental value of the arms we actually chose this round vs. the best
        # possible for these users.
        inc_batch = world.true_incremental_profit(X)
        realized[r] = inc_batch[arms.astype(bool)].sum()
        optimal[r] = inc_batch[inc_batch > 0].sum()
        bought, _ = world.step(X, arms)
        if priv is None:
            exp, pur = aggregate_counts(c, arms, bought, K)
        else:
            exp, pur = priv.privatize(c, arms, bought, K)
        bandit.update_from_counts(exp, pur)
        policy_cap[r] = policy_capture(bandit.greedy_policy()[holdout_clusters].astype(bool),
                                       holdout_inc)
    cum_capture = np.cumsum(realized) / np.cumsum(optimal)
    return cum_capture, policy_cap, bandit


# --------------------------------------------------------------------------
# Experiment 1: learning curves (captured fraction vs round)
# --------------------------------------------------------------------------
def learning_curves(base_cfg: WorldConfig, T=200, batch=300, seeds=range(10)):
    # Small batch on purpose: with thousands of users per round the bandit
    # converges in a single step and there's no curve to see; ~300/round makes
    # the learning visible. This figure answers "does causal targeting learn the
    # right policy?" -- the cost of PRIVACY (which needs large batches to fight
    # the noise) is a separate question, owned by privacy_tradeoff().
    fcue_clean, prop_caps, all_caps = [], [], []
    for s in seeds:
        w = World(replace(base_cfg, seed=s))
        Xh, ch, inc = _holdout(w)
        prop = _fit_propensity(w)
        prop_caps.append(best_capture(prop.predict(Xh), inc)[2])
        all_caps.append(policy_capture(np.ones(len(inc), bool), inc))
        fcue_clean.append(_run_bandit(w, T, batch, 100 + s, None, 1.0, ch, inc)[0])
    return {
        "rounds": np.arange(1, T + 1),
        "FCUE (clean aggregates)": np.mean(fcue_clean, axis=0),
        "propensity (best cutoff)": float(np.mean(prop_caps)),
        "treat everyone": float(np.mean(all_caps)),
    }


# --------------------------------------------------------------------------
# Experiment 2: the privacy trade-off (final capture vs privacy budget)
# --------------------------------------------------------------------------
def privacy_tradeoff(base_cfg: WorldConfig, epsilons, T=150, batch=2000, seeds=range(6), keep=1.0):
    means, stds = [], []
    for te in epsilons:
        caps = []
        for s in seeds:
            w = World(replace(base_cfg, seed=s))
            _, ch, inc = _holdout(w)
            eps = None if np.isinf(te) else te
            caps.append(_run_bandit(w, T, batch, 400 + s, eps, keep, ch, inc)[1][-1])
        means.append(float(np.mean(caps)))
        stds.append(float(np.std(caps)))
    return {"epsilons": list(epsilons), "mean": np.array(means), "std": np.array(stds)}


# --------------------------------------------------------------------------
# Experiment 3: propensity vs uplift across the coupling knob
# --------------------------------------------------------------------------
def coupling_curve(base_cfg: WorldConfig, couplings, T=150, batch=2000, seeds=range(6)):
    prop_mean, band_mean = [], []
    for rho in couplings:
        pc, bc = [], []
        for s in seeds:
            w = World(replace(base_cfg, seed=s, uplift_baseline_coupling=rho))
            Xh, ch, inc = _holdout(w)
            prop = _fit_propensity(w)
            pc.append(best_capture(prop.predict(Xh), inc)[2])
            bc.append(_run_bandit(w, T, batch, 500 + s, None, 1.0, ch, inc)[1][-1])
        prop_mean.append(float(np.mean(pc)))
        band_mean.append(float(np.mean(bc)))
    return {"couplings": list(couplings),
            "propensity": np.array(prop_mean),
            "uplift_bandit": np.array(band_mean)}


# --------------------------------------------------------------------------
# Experiment 4: uplift curves (Qini-style) on one representative world
# --------------------------------------------------------------------------
def uplift_curve_data(base_cfg: WorldConfig, seed=0, n_eval=200_000):
    w = World(replace(base_cfg, seed=seed))
    prop = _fit_propensity(w, n=60_000)
    *_, bandit = _run_bandit(w, 150, 2000, 1, None, 1.0, *(_holdout(w)[1:]))
    Xh = w.sample_users(n_eval)
    inc = w.true_incremental_profit(Xh)
    ch = coarse_cluster(Xh)
    scores = {
        "oracle (rank by true uplift)": inc,
        "FCUE bandit (believed uplift)": bandit.believed_incremental_profit()[ch],
        "propensity (buy-likelihood)": prop.predict(Xh),
        "random": np.random.default_rng(0).random(len(Xh)),
    }
    return {name: incremental_curve(sc, inc) for name, sc in scores.items()}
