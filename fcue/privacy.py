"""The privacy tax: simulating the Privacy Sandbox measurement path.

The real pipeline is: the browser buckets each user's contribution into
aggregation keys under a per-user contribution budget (L1); the encrypted
reports are summed inside a Trusted Execution Environment (TEE) which injects
calibrated noise; your server receives only the noisy aggregate totals.

We simulate the parts that matter for learning:

  1. DOWN-SAMPLING (budget conservation). Exposure events are common and carry
     weak signal; purchases are rare and carry strong signal. Reporting every
     exposure would burn the contribution budget before you capture the rare,
     valuable conversions. So we report only a fraction `q` of exposures, then
     divide by `q` on the server to recover an unbiased total (inverse-propensity
     weighting). The catch, which we make visible: dividing by q *amplifies* the
     noise. That's the price of saving budget.

  2. DIFFERENTIAL-PRIVACY NOISE (the guarantee). The TEE adds Laplace noise to
     each aggregate count, scaled to sensitivity/epsilon. `sensitivity` is the
     most any single user can move the released histograms (1 exposure bucket +
     1 purchase bucket = 2 in L1). Smaller epsilon -> more noise -> more privacy.

  3. COMPOSITION (why the noise is actually large). See `split_budget`. Because
     each noisy release spends privacy budget, running T rounds on a fixed total
     budget forces each round to use epsilon/T. That is the real reason a causal
     signal is hard to see through the noise -- not a modeling artifact.

The output of `privatize` is a (exposures, purchases) pair of noisy (K, 2)
arrays that drop straight into `ContextualThompsonBandit.update_from_counts`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bandit import aggregate_counts


@dataclass
class PrivacyConfig:
    epsilon: float = 1.0            # DP budget SPENT PER RELEASE (see split_budget for why this is small)
    sensitivity: float = 2.0        # max L1 one user adds across all buckets (1 exposure + 1 purchase)
    exposure_keep_rate: float = 1.0 # q in (0, 1]: fraction of exposure events reported (budget saver)
    l1_budget_units: float = 5.0    # illustrative per-user contribution cap per window
    events_per_user: float = 1.0    # reportable interactions a user generates per window
    seed: int = 0


class PrivacyPipeline:
    def __init__(self, config: PrivacyConfig | None = None):
        self.cfg = config or PrivacyConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    @property
    def laplace_scale(self) -> float:
        """b in Laplace(0, b). ε-DP for an L1-sensitivity-`sensitivity` release."""
        return self.cfg.sensitivity / max(self.cfg.epsilon, 1e-12)

    @property
    def noise_std(self) -> float:
        """Std-dev of the per-bucket noise (Laplace std = b*sqrt(2)). For reporting."""
        return self.laplace_scale * np.sqrt(2.0)

    # ----- illustrative contribution-budget accounting ----------------------
    def budget_units_per_user(self) -> float:
        """Rough per-user contribution units. Exposures dominate and are what
        `exposure_keep_rate` controls; purchases are rare and always reported."""
        return self.cfg.events_per_user * self.cfg.exposure_keep_rate

    def within_budget(self) -> bool:
        return self.budget_units_per_user() <= self.cfg.l1_budget_units

    # ----- the measurement mechanism ----------------------------------------
    def privatize(
        self,
        clusters: np.ndarray,
        arms: np.ndarray,
        bought: np.ndarray,
        n_clusters: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Turn per-user outcomes into the noisy aggregate totals the server sees."""
        b = self.laplace_scale
        q = self.cfg.exposure_keep_rate

        # 1) DOWN-SAMPLE exposures -> conserve the contribution budget.
        keep = self.rng.random(len(clusters)) < q
        exp_kept, _ = aggregate_counts(clusters[keep], arms[keep], bought[keep], n_clusters)

        # 2) TEE adds Laplace noise to the reported (subsampled) counts.
        exp_noisy = exp_kept + self.rng.laplace(0.0, b, size=exp_kept.shape)

        # 3) SERVER inverse-propensity correction (÷ q): unbiased, but scales noise up too.
        exposures = exp_noisy / q

        # 4) PURCHASES are rare & valuable: report all of them, then add noise.
        _, pur_true = aggregate_counts(clusters, arms, bought, n_clusters)
        purchases = pur_true + self.rng.laplace(0.0, b, size=pur_true.shape)

        return exposures, purchases


def split_budget(total_epsilon: float, n_releases: int) -> float:
    """Basic sequential composition.

    T noisy releases that each satisfy (epsilon/T)-DP together satisfy
    epsilon-DP. So to keep a whole T-round campaign within a fixed privacy
    budget, every round may spend only total_epsilon / T. This is exactly why
    per-release epsilon is small and per-round noise is large -- the fundamental
    scarcity the FCUE report gestures at.
    """
    return total_epsilon / max(n_releases, 1)
