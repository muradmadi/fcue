// FCUE edge logic, written against the REAL Google Privacy Sandbox APIs.
//
// This file runs INSIDE a Shared Storage worklet: a sandboxed JavaScript context
// that can read cross-site data but CANNOT touch the network. It is the browser
// counterpart of the Python simulation's Phase 1-3:
//
//   * reads the user's private local context + downloaded model weights
//     ................................ mirrors fcue/bandit.py  (assign)
//   * picks treat/control via Thompson Sampling on expected profit
//     ................................ mirrors fcue/bandit.py  (assign)
//   * reports the result ONLY as an aggregate histogram contribution
//     ................................ mirrors fcue/privacy.py (privatize) +
//                                              fcue/bandit.py  (aggregate_counts)
//
// Nothing here sends per-user data anywhere. The only thing that ever leaves is a
// noisy, aggregated histogram, assembled by the browser + a Trusted Execution
// Environment via the Private Aggregation API.

const NUM_CLUSTERS = 8;
const MODEL_VERSION = 1;
const MARGIN = 100;
const COUPON_COST = 15;
const METRIC_EXPOSURE = 0; // "this arm was shown"
const METRIC_PURCHASE = 1; // "a purchase followed"

// --- Aggregation key encoding -------------------------------------------------
// Bit-pack a 128-bit bucket id (the report's "densely packed aggregation keys").
// Layout, from the low bits:  [metric:8][treatment:8][cluster:32][modelVersion:16]
// A bucket is just a BigInt; the server later decodes these bits to know which
// (cluster, arm, metric) each aggregated total belongs to.
function encodeBucket(modelVersion, cluster, treatment, metric) {
  return (BigInt(modelVersion) << 48n)
       | (BigInt(cluster) << 16n)
       | (BigInt(treatment) << 8n)
       | BigInt(metric);
}

// --- Coarse clustering (mirrors fcue/context.py) ------------------------------
// 3 bits from 3 features thresholded at 0.5 -> a cluster id in 0..7. Broad on
// purpose: big buckets survive differential-privacy noise.
function coarseCluster(features) {
  return ((features[0] > 0.5) ? 1 : 0)
       | ((features[1] > 0.5) ? 2 : 0)
       | ((features[2] > 0.5) ? 4 : 0);
}

// --- Thompson Sampling helpers ------------------------------------------------
function randn() { // standard normal via Box-Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
// Draw a plausible buy-probability from a Beta(a, b) posterior. The Python sim
// uses an exact Beta sampler; here we use a Gaussian approximation to keep the
// worklet compact (mean +/- sd of the Beta).
function sampleBuyProb(a, b) {
  const mean = a / (a + b);
  const varc = (a * b) / ((a + b) * (a + b) * (a + b + 1));
  return Math.min(1, Math.max(0, mean + Math.sqrt(varc) * randn()));
}

// --- Operation 1: the edge decision ------------------------------------------
class FcueDecision {
  async run(data) {
    // Makes debug reports visible in chrome://private-aggregation-internals.
    privateAggregation.enableDebugMode();

    // 1) CONTEXT: read private local features + the last broadcast global weights.
    const features = JSON.parse(
      (await this.sharedStorage.get('fcue_features')) || '[0.5,0.5,0.5]');
    const weights = JSON.parse(
      (await this.sharedStorage.get('fcue_weights')) || 'null'); // {alpha:[[..]], beta:[[..]]}
    const cluster = coarseCluster(features);

    // 2) DECIDE: Thompson Sampling over treat vs. control on EXPECTED PROFIT.
    //    (Falls back to control if no weights have been downloaded yet.)
    let arm = 0;
    if (weights) {
      const p0 = sampleBuyProb(weights.alpha[cluster][0], weights.beta[cluster][0]);
      const p1 = sampleBuyProb(weights.alpha[cluster][1], weights.beta[cluster][1]);
      const valueControl = p0 * MARGIN;
      const valueTreat = p1 * (MARGIN - COUPON_COST);
      arm = (valueTreat > valueControl) ? 1 : 0;
    }

    // 3) REPORT: one contribution to the EXPOSURE histogram for (cluster, arm).
    //    This is the browser equivalent of aggregate_counts() -> privatize():
    //    the value is bucketed; only a noisy AGGREGATE ever leaves the device.
    privateAggregation.contributeToHistogram({
      bucket: encodeBucket(MODEL_VERSION, cluster, arm, METRIC_EXPOSURE),
      value: 1,
    });

    // Remember the assignment locally so a later conversion can report the
    // matching purchase bucket. In a real integration the chosen arm would be
    // surfaced to the page via selectURL(); the decision itself never leaks.
    await this.sharedStorage.set('fcue_last_cluster', String(cluster));
    await this.sharedStorage.set('fcue_last_arm', String(arm));
  }
}

// --- Operation 2: the conversion report --------------------------------------
class FcueConversion {
  async run(data) {
    privateAggregation.enableDebugMode();
    const cluster = parseInt((await this.sharedStorage.get('fcue_last_cluster')) || '0', 10);
    const arm = parseInt((await this.sharedStorage.get('fcue_last_arm')) || '0', 10);

    // Contribute to the PURCHASE histogram for the same (cluster, arm). The
    // server later divides purchase-sums by exposure-sums per bucket to recover
    // buy-rates -> the Beta-Bernoulli update in fcue/bandit.py.
    privateAggregation.contributeToHistogram({
      bucket: encodeBucket(MODEL_VERSION, cluster, arm, METRIC_PURCHASE),
      value: 1,
    });
  }
}

register('fcue-decision', FcueDecision);
register('fcue-conversion', FcueConversion);
