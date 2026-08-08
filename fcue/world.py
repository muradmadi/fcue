"""The simulated 'reality'.

This module holds the TRUE, hidden model of how users behave. In real life this
is unknowable -- and, post-cookies, un-loggable at the individual level. That
un-knowability is the entire reason this project is hard.

We get to know the truth ONLY because we invented it, so we can later *score*
how well our privacy-constrained learner did. The bandit, the privacy layer, and
the server are NEVER allowed to read these true parameters. They may only call
`World.step()` and observe sampled outcomes -- exactly as a real system would
only ever observe "this user did / didn't buy".

The 4 quadrants of every user (the heart of uplift modeling):
    - "sure thing"   : buys with or without the coupon   -> uplift ~= 0  (coupon wastes money)
    - "lost cause"   : never buys                        -> uplift ~= 0  (coupon wastes money)
    - "persuadable"  : buys ONLY if given the coupon      -> uplift  > 0  (THE ones worth treating)
    - "sleeping dog" : coupon annoys them into NOT buying -> uplift  < 0  (coupon actively harmful)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WorldConfig:
    n_features: int = 3       # dimensionality of each user's context vector
    margin: float = 100.0     # profit earned on a full-price purchase
    coupon_cost: float = 15.0 # extra cost incurred when a purchase happens under a coupon
    # THE crux assumption for the propensity-vs-uplift debate: how correlated is
    # "likely to buy" with "movable by a coupon"?  +1 => the eager buyers are
    # also the persuadable ones (propensity targeting works by luck); 0 =>
    # unrelated (propensity is no better than random); -1 => the eager buyers are
    # the LEAST movable "sure things" (propensity is actively wrong -- the report's
    # scenario). Making this explicit stops the result being a seed accident.
    uplift_baseline_coupling: float = 0.0
    seed: int = 0


class World:
    """Ground truth generator. The only 'legal' interface for learners is step()."""

    def __init__(self, config: WorldConfig | None = None):
        self.cfg = config or WorldConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        d = self.cfg.n_features
        # Two independent hidden directions in feature space:
        #   _w_base drives the baseline buy-probability (behavior with NO coupon)
        #   _w_ind  is an INDEPENDENT direction used to build uplift
        # We then blend _w_base and _w_ind to hit a *chosen* correlation (rho)
        # between propensity and uplift -- see uplift().
        self._w_base = self.rng.normal(0.0, 1.0, size=d)
        self._b_base = -1.0  # low baseline buy-rate: without a coupon, most don't buy
        # A second direction, Gram-Schmidt-orthogonalized against _w_base. Because
        # cov(X@a, X@b) is proportional to a.b for X ~ Uniform, making the weight
        # vectors orthogonal makes the two SCORES genuinely uncorrelated -- so the
        # coupling knob below sets the correlation exactly, not approximately.
        # (Without this, in low dimensions two random directions overlap a lot.)
        w_ind = self.rng.normal(0.0, 1.0, size=d)
        w_ind = w_ind - (w_ind @ self._w_base) / (self._w_base @ self._w_base) * self._w_base
        self._w_ind = w_ind
        self._rho = float(np.clip(self.cfg.uplift_baseline_coupling, -1.0, 1.0))

        self._uplift_amp = 0.35   # coupon can move buy-prob by up to +/-35 points
        self._uplift_bias = 0.30  # positive tilt: persuadable majority, sleeping-dog minority

        # Analytic mean/std of a linear score X @ w when X ~ Uniform[0,1]^d, so we
        # can standardize the two directions and mix them to an exact correlation.
        self._m_base, self._s_base = self._lin_moments(self._w_base)
        self._m_ind, self._s_ind = self._lin_moments(self._w_ind)

    # ----- user generation -------------------------------------------------
    def sample_users(self, n: int) -> np.ndarray:
        """Draw n users. Each row is a context feature vector in [0, 1]^d.

        Think of the features as things a browser could know locally: recency,
        frequency, on-device interest scores, etc.
        """
        return self.rng.random((n, self.cfg.n_features))

    # ----- the hidden truth (private to this module) -----------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _lin_moments(w: np.ndarray) -> tuple[float, float]:
        """Mean and std of X @ w for X ~ Uniform[0,1]^d (each feature: mean 1/2,
        var 1/12). Used to standardize the two hidden directions."""
        mean = 0.5 * w.sum()
        std = np.sqrt((w @ w) / 12.0)
        return float(mean), float(std)

    def _z_base(self, X: np.ndarray) -> np.ndarray:
        return (X @ self._w_base - self._m_base) / self._s_base

    def _z_ind(self, X: np.ndarray) -> np.ndarray:
        return (X @ self._w_ind - self._m_ind) / self._s_ind

    def baseline_prob(self, X: np.ndarray) -> np.ndarray:
        """p(buy | control / no coupon) for each user."""
        return self._sigmoid(X @ self._w_base + self._b_base)

    def uplift(self, X: np.ndarray) -> np.ndarray:
        """True treatment effect tau(x): the change in buy-probability caused by
        the coupon. This is the CATE at the individual level -- positive for
        persuadables, ~0 for sure-things/lost-causes, negative for sleeping dogs.

        We build the uplift's driving score as a blend of the baseline direction
        (`_z_base`) and an independent one (`_z_ind`), weighted so its correlation
        with baseline propensity equals `rho`:
            score = rho * z_base + sqrt(1 - rho^2) * z_ind
        Both z's are ~standard-normal, so `score` is too, and corr(score, z_base)
        = rho exactly. tanh bounds it to +/-amp and lets uplift take both signs.
        """
        rho = self._rho
        score = rho * self._z_base(X) + np.sqrt(1.0 - rho ** 2) * self._z_ind(X)
        return self._uplift_amp * np.tanh(score + self._uplift_bias)

    def buy_prob(self, X: np.ndarray, treatment: np.ndarray | int) -> np.ndarray:
        """p(buy) given a treatment (0 = control, 1 = coupon), clipped to [0,1]."""
        t = np.asarray(treatment)
        p = self.baseline_prob(X) + t * self.uplift(X)
        return np.clip(p, 0.0, 1.0)

    # ----- the ONLY legal channel for learners -----------------------------
    def step(self, X: np.ndarray, treatment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simulate what actually happens.

        Returns (bought, profit) as integer/float arrays, one entry per user.
        A learner is allowed to see these realized outcomes -- but NOT the
        probabilities or uplift that generated them.

        profit = margin on a purchase, minus the coupon_cost if that purchase
        happened under a coupon. So treating a user only pays off when the extra
        purchases the coupon *causes* outweigh the discount handed to everyone
        who would have bought anyway.
        """
        t = np.asarray(treatment)
        p = self.buy_prob(X, t)
        bought = (self.rng.random(len(X)) < p).astype(int)
        profit = bought * (self.cfg.margin - t * self.cfg.coupon_cost)
        return bought, profit.astype(float)

    # ----- oracle helpers: evaluation ONLY (never fed to any learner) ------
    def true_incremental_profit(self, X: np.ndarray) -> np.ndarray:
        """Expected [profit if treated] - [profit if control], per user.

        This is the 'right answer' the bandit is trying to approximate blind.
        We use it only to score methods after the fact. A user is worth treating
        exactly when this value is positive.
        """
        p1 = self.buy_prob(X, 1)
        p0 = self.buy_prob(X, 0)
        return p1 * (self.cfg.margin - self.cfg.coupon_cost) - p0 * self.cfg.margin

    def quadrant(self, X: np.ndarray) -> np.ndarray:
        """Label each user's quadrant from the true uplift. For inspection/plots."""
        tau = self.uplift(X)
        labels = np.where(tau > 0.02, "persuadable",
                 np.where(tau < -0.02, "sleeping_dog", "neutral"))
        return labels
