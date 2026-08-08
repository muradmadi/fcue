"""Baselines: the alternatives FCUE has to beat, and the tools to compare them.

The headline comparison of the whole project is uplift-targeting vs. PROPENSITY
targeting. Propensity ("lead scoring") predicts who is most likely to buy and
targets them. The report's core accusation is that this misallocates budget: the
people most likely to buy are often the ones who'd buy *anyway* (sure-things),
so the coupon is pure waste. Persuadables -- who buy only WITH the nudge -- tend
to have LOW baseline buy-probability, so propensity ranks them last.

We demonstrate this with an "uplift curve": rank users by a score, then for each
fraction f, plot the true incremental profit you'd earn by treating the top-f.
A good score is concave and rises fast (captures value early); a useless score
tracks the diagonal (no better than treating a random subset).

Nothing here uses privacy noise -- these are clean-data reference points so we
can measure what the *targeting logic* is worth, separately from the privacy tax.
"""

from __future__ import annotations

import numpy as np


class LogisticPropensity:
    """A plain logistic-regression lead-score: predict p(buy) from features.

    Trained only on observed (features, bought) pairs -- no causal knowledge, no
    treatment awareness. This is what a large chunk of the industry actually
    ships. NumPy gradient descent so there are no hidden dependencies and you can
    explain every line.
    """

    def __init__(self, lr: float = 0.5, epochs: int = 500, l2: float = 1e-4):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticPropensity":
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        y = y.astype(float)
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            grad = p - y                      # gradient of logistic loss w.r.t. logits
            self.w -= self.lr * (X.T @ grad / n + self.l2 * self.w)
            self.b -= self.lr * grad.mean()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.w + self.b)


def incremental_curve(scores: np.ndarray, true_incremental: np.ndarray):
    """Uplift curve. Sort users by `scores` (desc); for each fraction treated,
    return the realized per-user incremental profit.

    Returns (fractions, per_user_profit), both length N.
    """
    order = np.argsort(-scores)
    per_user = np.cumsum(true_incremental[order]) / len(scores)
    fractions = np.arange(1, len(scores) + 1) / len(scores)
    return fractions, per_user


def best_capture(scores: np.ndarray, true_incremental: np.ndarray):
    """Steelman a ranking: let it treat the top-f that maximizes profit (best
    possible threshold), and report how much of the oracle prize that captures.

    Returns (peak_per_user_profit, fraction_treated, capture_ratio).
    """
    _, curve = incremental_curve(scores, true_incremental)
    prize = true_incremental[true_incremental > 0].sum() / len(true_incremental)
    peak_idx = int(curve.argmax())
    peak = float(curve[peak_idx])
    frac = (peak_idx + 1) / len(curve)
    return peak, frac, peak / prize


def policy_capture(treat_mask: np.ndarray, true_incremental: np.ndarray) -> float:
    """Capture ratio for a concrete yes/no policy (not a ranking).

    Deterministic (uses expected incremental profit), so it has none of the
    Monte-Carlo wobble of scoring a policy by simulating one stochastic outcome.
    """
    prize = true_incremental[true_incremental > 0].sum() / len(true_incremental)
    got = true_incremental[treat_mask.astype(bool)].sum() / len(true_incremental)
    return got / prize
