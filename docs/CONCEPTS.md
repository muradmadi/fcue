# Concepts, decoded

Every intimidating term from the source idea, in plain English, mapped to the code
that implements it. Read this once and the repo stops being scary.

The original write-up was dense and inflated. Mentally deflate the vocabulary by
~80% and a clear, buildable system is left. This file is that deflation.

---

## Part 1 — the causal / uplift side

**The core switch:** stop asking *"who is likely to buy?"* and start asking *"who
buys **because** we nudged them?"* The first question wastes money; the second is
the whole game.

**The four kinds of people** (every marketing intervention faces this split):

| Type | Behavior | True uplift `τ(x)` | Coupon effect |
|------|----------|--------------------|---------------|
| Sure thing | buys either way | ≈ 0 | wasted discount |
| Lost cause | never buys | ≈ 0 | wasted discount |
| **Persuadable** | buys **only if** nudged | **> 0** | **the whole point** |
| Sleeping dog | nudging makes them leave | **< 0** | actively harmful |

In code: [`World.uplift()`](../fcue/world.py) generates `τ(x)`, and
[`World.quadrant()`](../fcue/world.py) labels each user. In the default world about
58% are persuadable, ~39% sleeping dogs, ~3% neutral.

| Term (as written in the report) | Plain meaning | Where in the code |
|---|---|---|
| **Propensity / lead scoring** | Predicts who'll buy. Targets sure-things by mistake. The old way. | [`baselines.LogisticPropensity`](../fcue/baselines.py) |
| **Uplift / incremental / true-lift modeling** | Predicts who'll buy *because of* the treatment. Finds persuadables. | The bandit optimizes this: [`bandit.py`](../fcue/bandit.py) |
| **CATE** (Conditional Average Treatment Effect) | The uplift for a *type* of person ("conditional" = given their features/cluster). | `believed_incremental_profit()` per cluster in [`bandit.py`](../fcue/bandit.py); truth in `World.true_incremental_profit()` |
| **Counterfactual / "fundamental problem of causal inference"** | You can never see both outcomes (with-coupon *and* without) for the same person. | Why `World` is the only thing that "knows" the truth; learners see only `World.step()` |
| **T-learner, X-learner, uplift forests, Double ML** | Different recipes for estimating uplift. We don't need the fancy ones — the bandit's per-cluster treat-vs-control comparison *is* a simple CATE estimator. | conceptually, [`bandit.py`](../fcue/bandit.py) |
| **Contextual multi-armed bandit** | Learn *while* acting: try treatments, watch rewards, shift toward winners — decision depends on the user's context (cluster). | [`ContextualThompsonBandit`](../fcue/bandit.py) |
| **Thompson Sampling** | A principled explore/exploit rule: draw one sample from each arm's belief, pick the best. Uncertain arms sometimes win → automatic exploration. | `ContextualThompsonBandit.assign()` |
| **Explore/exploit dilemma** | Spend actions learning (explore) vs. cashing in on what you know (exploit). Thompson balances it with no tuning knob. | `assign()` (sampling) vs `greedy_policy()` (pure exploit) |
| **Regret** | Value lost vs. a perfect policy. Lower = better. | `1 − cum_capture` in [`simulation.py`](../fcue/simulation.py) |
| **Qini / uplift curve** | Rank users by a score; plot cumulative incremental profit as you treat more. Concave-and-high = good score. | `baselines.incremental_curve()`, figure 4 |

### Why not just use propensity?
Because a propensity model optimizes a *proxy* (buy-likelihood), and the proxy only
matches the goal (persuadability) when the two happen to be correlated — an accident
of the population you can't control. That correlation is an explicit knob here:
`WorldConfig.uplift_baseline_coupling` (−1 = eager buyers are the *least* movable
"sure-things"; 0 = unrelated; +1 = aligned). Figure 3 sweeps it: uplift wins
everywhere; propensity only when the knob is positive.

---

## Part 2 — the privacy side

**The problem:** third-party cookies are dying, so you can't follow one person across
the web and log "saw this → got that → bought this." The per-user data every
behavioral model normally eats is disappearing.

**Google Privacy Sandbox** is the replacement: APIs that let you still *measure*, but
only in aggregate, never per person.

| Term (as written) | Plain meaning | Simulated by |
|---|---|---|
| **Shared Storage** | A browser data store you can only *read* inside a sealed sandbox that can't phone home. | (conceptual) the edge portion of the loop |
| **Worklet** | That sealed JS sandbox. Sees the user's data; can't send it anywhere. | (conceptual) where `assign()` "runs" |
| **Topics API** | On-device interest categories replacing cross-site profiling. | the coarse features / clusters ([`context.py`](../fcue/context.py)) |
| **Private Aggregation API** | The *only* exit door: drop events into labeled buckets ("aggregation keys"); only summed totals ever come out. | [`PrivacyPipeline.privatize()`](../fcue/privacy.py) |
| **Aggregation key** | A bucket label packing dimensions (e.g. cluster + arm + metric). | the `(cluster, arm)` indices of the count matrices |
| **`contributeToHistogram()`** | The browser call that adds a value to a bucket. | `aggregate_counts()` in [`bandit.py`](../fcue/bandit.py) |
| **TEE / Trusted Execution Environment** (AWS Nitro Enclaves) | A locked server even *you* can't read; it sums encrypted reports and adds noise. | the noise + sum step inside `privatize()` |
| **Differential privacy (DP)** | A math guarantee that outputs can't be traced to any individual, enforced by adding calibrated random noise. | `np.random.laplace(...)` in `privatize()` |
| **ε (epsilon) / privacy budget** | How much privacy you spend. Smaller ε = more noise = more private. | `PrivacyConfig.epsilon`; `PrivacyPipeline.laplace_scale = sensitivity/ε` |
| **L1 contribution budget** | A cap on how much any one user can add to the histograms — a scarce resource. | `l1_budget_units` + the down-sampling in `privatize()` |
| **Sensitivity** | The most one user can move the released totals (here: 1 exposure + 1 purchase = 2 in L1). | `PrivacyConfig.sensitivity` |
| **Composition** | Every noisy release spends budget; T releases split one budget T ways → big per-release noise. | `privacy.split_budget()` |
| **Down-sampling + inverse-propensity weighting** | Report only a fraction `q` of common events to save budget, then divide by `q` to un-bias — at the cost of amplified noise. | the `exposure_keep_rate` path in `privatize()` |

---

## Part 3 — the synthesis (what FCUE actually is)

Run the "which coupon?" decision **at the edge** using downloaded model weights +
the user's private local features (Thompson Sampling picks the arm). Report the
result only as **noisy aggregate buckets** (Private Aggregation). The server sums the
noisy totals, updates a **global uplift-bandit model**, and rebroadcasts weights. The
loop repeats. It's **federated learning** (learn from everyone without collecting
anyone's raw data) **of a causal bandit** (optimizing uplift, not clicks).

**The linchpin** (say this in an interview): the bandit's belief about each cluster's
buy-rate is a **Beta-Bernoulli** posterior, updated by conjugacy. Its sufficient
statistics are just two **sums** — exposures and purchases. Aggregation APIs produce
exactly those sums. So learning from privacy-preserving aggregates is the *natural*
shape of the math, not a compromise. See
[`ContextualThompsonBandit.update_from_counts()`](../fcue/bandit.py).

---

## Part 4 — the one genuinely hard idea

CATE (uplift) is a **difference of two expectations**, so its variance is inherently
higher than a plain prediction's. DP then adds **independent noise to both** the
treatment and control aggregates, and those variances **compound**. If the true
causal signal for a cluster is small, the noise can flip the decision.

Watch it happen: in the privacy sweep, the *weakest-signal* clusters are always the
first to be mis-decided as ε shrinks. That's not a bug — it's the fundamental tension
the whole project is about.

The two mitigations, both in the code:

1. **Broad clusters** — pool users so each bucket's signal is large relative to the
   noise. Controlled by `n_bits` in [`context.py`](../fcue/context.py). Trade-off:
   more privacy-robust, less personalized.
2. **Budget-aware reporting** — down-sample cheap/common events, keep rare/valuable
   ones, and correct the bias on the server. In [`privacy.py`](../fcue/privacy.py).

And the reason the noise is big at all is **composition** (`split_budget`): a design
lever, not a law — reporting less often in bigger batches buys back precision.

---

## Part 5 — file-by-file, what concept lives where

- **[`world.py`](../fcue/world.py)** — the counterfactual reality. Holds the true
  baseline buy-probability, the true uplift `τ(x)`, and the economics (margin,
  coupon cost). The `uplift_baseline_coupling` knob sets the propensity↔uplift
  correlation. Learners may call only `step()`.
- **[`context.py`](../fcue/context.py)** — `coarse_cluster()`: the "privacy tax on
  granularity." Collapses features into `2**n_bits` broad buckets.
- **[`bandit.py`](../fcue/bandit.py)** — the brain. Per-cluster Beta-Bernoulli
  posteriors, Thompson `assign()`, aggregate-only `update_from_counts()`, and
  `aggregate_counts()` (the `contributeToHistogram` analog).
- **[`privacy.py`](../fcue/privacy.py)** — the privacy tax. `privatize()` =
  down-sample → noise → un-bias → report; `split_budget()` = composition.
- **[`baselines.py`](../fcue/baselines.py)** — `LogisticPropensity` (the industry
  baseline) plus uplift-curve tooling (`incremental_curve`, `best_capture`,
  `policy_capture`).
- **[`simulation.py`](../fcue/simulation.py)** — the four experiments; scores every
  policy by "fraction of the oracle prize captured."
- **[`experiments/make_figures.py`](../experiments/make_figures.py)** — renders the
  four charts.

---

## Further reading (real sources behind the buzzwords)

- Uplift modeling & CATE: Gutierrez & Gérardy, *Causal Inference and Uplift
  Modelling* (2017); the `causalml` and `EconML` libraries.
- Thompson Sampling: Russo et al., *A Tutorial on Thompson Sampling* (2018).
- Differential privacy: Dwork & Roth, *The Algorithmic Foundations of Differential
  Privacy* (2014).
- Privacy Sandbox: Google's Private Aggregation API and Aggregation Service docs
  (`privacysandbox.google.com`).

> Everything above is modeled here at a level meant for *understanding*, not for
> cryptographic or production fidelity. The goal is a system you can run, reason
> about, and explain.
