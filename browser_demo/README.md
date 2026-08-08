# FCUE browser demo — real Privacy Sandbox edge logic

This is the *thin, real* counterpart to the Python simulation: the FCUE edge
decision written against the actual **Shared Storage** and **Private Aggregation**
browser APIs. It exists so the project isn't only a simulation — the same logic the
sim models is here in genuine Privacy Sandbox JavaScript.

## Files

- **`fcue-worklet.js`** — runs inside a Shared Storage worklet (sandboxed, no
  network). Reads local context + downloaded weights, picks treat/control via
  Thompson Sampling, and reports via `privateAggregation.contributeToHistogram()`.
- **`index.html`** — a page that loads the worklet, writes example context + global
  weights into Shared Storage, and triggers the decision and a conversion.

## How it maps to the simulation

| Browser (this folder) | Simulation (`../fcue/`) |
|---|---|
| `coarseCluster()` | `context.coarse_cluster()` |
| Thompson `sampleBuyProb()` + argmax on expected profit | `bandit.ContextualThompsonBandit.assign()` |
| `contributeToHistogram()` into exposure/purchase buckets | `bandit.aggregate_counts()` → `privacy.privatize()` |
| `encodeBucket()` bit-packing | the `(cluster, arm, metric)` indices of the count matrices |
| page `sharedStorage.set('fcue_weights', …)` | the server's broadcast of updated global weights |

## Running it (Chrome only, and gated)

These APIs are experimental and access-controlled. Rough steps — exact flags change
over Chrome versions, so treat this as a pointer, not a guarantee:

1. Use a recent Chrome/Chromium.
2. Enable the Privacy Sandbox Ads APIs: `chrome://settings/adPrivacy`, or launch with
   the relevant `--enable-features=PrivacySandboxAdsAPIsOverride,SharedStorageAPI,PrivateAggregationApi`
   flags.
3. Serve this folder over `http://localhost` (Shared Storage needs a secure context):
   ```bash
   cd browser_demo && python3 -m http.server 8000
   # then open http://localhost:8000/
   ```
4. Click **Run edge decision**, then **Simulate a conversion**.
5. Contributions are **batched and delayed** and encrypted for a Trusted Execution
   Environment — you won't see instant output. Inspect queued reports at
   `chrome://private-aggregation-internals`.

If the APIs aren't available, the page detects that and says so; the code is still
valid and readable as a reference for how the real integration looks.

## What this demo is and isn't

- **Is:** correct, idiomatic usage of the real Shared Storage + Private Aggregation
  APIs, structurally identical to the simulated pipeline.
- **Isn't:** a full deployment. There is no real coordinator enrollment, no TEE
  running Google's Aggregation Service, and no server loop here — those live
  (as a model) in the Python side. Production also requires Privacy Sandbox
  enrollment for your origin.
