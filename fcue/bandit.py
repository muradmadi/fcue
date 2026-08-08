"""The brain: a per-cluster contextual Thompson-Sampling bandit.

For each context cluster k and each arm a in {0 = control, 1 = treat}, we hold a
Beta(alpha, beta) posterior over the buy-probability p(buy | cluster, arm).

Why Beta-Bernoulli? Two reasons, both worth being able to explain:

  1. Thompson Sampling. To decide, we draw ONE sample from each arm's posterior
     and pick the arm with the higher expected *profit*. Arms we're unsure about
     have wide posteriors, so their samples sometimes win -> automatic
     exploration. Arms we've nailed down have tight posteriors -> exploitation.
     No epsilon to tune; the uncertainty does the balancing. And because we draw
     a fresh sample per user, within a cluster we end up treating users roughly
     *in proportion to the probability that treating is the right call* -- the
     exact behavior the FCUE report describes.

  2. Aggregate-friendliness (the linchpin). The conjugate update for a Beta
     posterior needs only two SUMS per (cluster, arm): how many were exposed and
     how many bought. Those sums are precisely what the Private Aggregation API
     will later deliver (noisily). So this bandit can learn from aggregates
     instead of per-user rows -- which is the whole reason FCUE is possible under
     privacy. See `update_from_counts`.

Crucially, the bandit optimizes INCREMENTAL profit, not raw conversions. It
compares (buy-prob under treat) x (price under treat) against (buy-prob under
control) x (full price). That comparison is what steers it away from sure-things
(no lift, just a wasted discount) and sleeping-dogs (negative lift).
"""

from __future__ import annotations

import numpy as np


class ContextualThompsonBandit:
    def __init__(
        self,
        n_clusters: int,
        margin: float,
        coupon_cost: float,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: int = 0,
    ):
        self.K = int(n_clusters)
        self.margin = float(margin)
        self.coupon_cost = float(coupon_cost)
        # Posterior parameters per (cluster, arm). Start from a weak prior:
        # Beta(1,1) is uniform over [0,1] -- "we know nothing about buy-rates yet".
        self.alpha = np.full((self.K, 2), float(prior_alpha))
        self.beta = np.full((self.K, 2), float(prior_beta))
        self.rng = np.random.default_rng(seed)

    # price kept per purchase under each arm (business-known economics)
    def _price(self, arm: int) -> float:
        return self.margin - arm * self.coupon_cost

    def _prices(self) -> np.ndarray:
        return np.array([self._price(0), self._price(1)])  # shape (2,)

    # ----- decision: Thompson Sampling (runs "at the edge", per user) -------
    def assign(self, clusters: np.ndarray) -> np.ndarray:
        """Choose an arm (0 or 1) for each user, given their cluster id.

        One fresh posterior draw per user per arm, converted to expected profit,
        argmax. Returns an int array of chosen arms, same length as `clusters`.
        """
        clusters = np.asarray(clusters)
        a = self.alpha[clusters]  # (N, 2)
        b = self.beta[clusters]   # (N, 2)
        sampled_p = self.rng.beta(a, b)             # (N, 2): a plausible buy-prob per arm
        value = sampled_p * self._prices()          # (N, 2): expected profit per arm
        return np.argmax(value, axis=1)             # (N,): the chosen arm

    # ----- learning: the ONLY update path, consumes aggregate sums ----------
    def update_from_counts(self, exposures: np.ndarray, purchases: np.ndarray) -> None:
        """Update posteriors from aggregate sufficient statistics.

        exposures[k, a] = # users in cluster k assigned arm a in this batch
        purchases[k, a] = # of those who then bought

        These are exactly the (noisy) sums the Private Aggregation API delivers.
        Beta-Bernoulli conjugacy: successes add to alpha, failures add to beta.
        We clip so heavy noise can't push a count negative or a parameter to 0.
        """
        exposures = np.asarray(exposures, dtype=float)
        purchases = np.asarray(purchases, dtype=float)
        successes = np.clip(purchases, 0.0, None)
        failures = np.clip(exposures - purchases, 0.0, None)
        self.alpha += successes
        self.beta += failures
        np.clip(self.alpha, 1e-6, None, out=self.alpha)
        np.clip(self.beta, 1e-6, None, out=self.beta)

    # ----- inspection / evaluation ------------------------------------------
    def posterior_mean_p(self) -> np.ndarray:
        """Posterior-mean buy-probability per (cluster, arm)."""
        return self.alpha / (self.alpha + self.beta)

    def expected_profit(self) -> np.ndarray:
        """Posterior-mean expected profit per (cluster, arm), shape (K, 2)."""
        return self.posterior_mean_p() * self._prices()

    def believed_incremental_profit(self) -> np.ndarray:
        """What the bandit currently believes treat-minus-control is, per cluster."""
        ep = self.expected_profit()
        return ep[:, 1] - ep[:, 0]

    def greedy_policy(self) -> np.ndarray:
        """Current best arm per cluster (1 = treat, 0 = skip) under posterior means."""
        return np.argmax(self.expected_profit(), axis=1)


def aggregate_counts(
    clusters: np.ndarray,
    arms: np.ndarray,
    bought: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-user (cluster, arm, bought) into (K, 2) exposure & purchase sums.

    In the clean-data sim this is just bookkeeping. In the privacy version, this
    exact aggregation is what happens inside the browser before noise is added --
    which is why the bandit never needs to see a single individual row.
    """
    exposures = np.zeros((n_clusters, 2), dtype=float)
    purchases = np.zeros((n_clusters, 2), dtype=float)
    np.add.at(exposures, (clusters, arms), 1.0)
    np.add.at(purchases, (clusters, arms), bought.astype(float))
    return exposures, purchases
