"""Coarse contextualization -- the privacy tax on granularity.

In a normal system you'd feed a user's full, fine-grained feature vector to the
model. The Privacy Sandbox won't let you: everything that leaves the browser has
noise added, and noise destroys fine buckets that only hold a few users. So the
edge worklet must first collapse the rich context into a SMALL number of broad
"context clusters", each of which will collect lots of users -> lots of signal
to overpower the noise later.

This is the document's "highly structured, intentionally broad Contextual
Clusters" point, made concrete. Fewer, bigger clusters = more privacy-robust but
less personalized. That trade-off is a knob (n_bits) you can defend in an
interview.
"""

from __future__ import annotations

import numpy as np


def coarse_cluster(X: np.ndarray, n_bits: int = 3) -> np.ndarray:
    """Map each fine feature vector to one of 2**n_bits coarse cluster ids.

    We threshold the first `n_bits` features at 0.5 to get bits, then read the
    bits as a binary number. With n_bits=3 that's 8 clusters. Crude on purpose:
    it mimics an on-device hash into a handful of buckets, and it's trivial to
    explain.
    """
    bits = (X[:, :n_bits] > 0.5).astype(int)
    weights = 2 ** np.arange(n_bits)
    return bits @ weights


def n_clusters(n_bits: int = 3) -> int:
    return 2 ** n_bits
