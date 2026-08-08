"""Generate the project's headline figures as PNGs.

Run:  python -m experiments.make_figures     (from the repo root, venv active)

Produces four charts in experiments/figures/:
  1. learning_curves.png   - the bandit learns; baselines stay flat
  2. privacy_tradeoff.png   - % of prize kept vs privacy budget
  3. coupling_curve.png     - uplift beats propensity across all assumptions
  4. uplift_curves.png      - the Qini-style "who to target" comparison
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: render straight to files, no display needed
import matplotlib.pyplot as plt
import numpy as np

from fcue.simulation import (
    coupling_curve,
    learning_curves,
    privacy_tradeoff,
    uplift_curve_data,
)
from fcue.world import WorldConfig

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
C = {"clean": "#1f6feb", "priv": "#8957e5", "prop": "#d1242f",
     "all": "#9a6700", "oracle": "#1a7f37", "rand": "#8b949e"}


def fig_learning(base):
    d = learning_curves(base, seeds=range(10))
    r = d["rounds"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(r, d["FCUE (clean aggregates)"], color=C["clean"], lw=2.4,
            label="FCUE uplift bandit (learns online)")
    ax.axhline(d["propensity (best cutoff)"], color=C["prop"], ls="--", lw=1.6,
               label=f"propensity, best cutoff = {d['propensity (best cutoff)']:.0%}")
    ax.axhline(d["treat everyone"], color=C["all"], ls=":", lw=1.6,
               label=f"treat everyone = {d['treat everyone']:.0%}")
    ax.axhline(1.0, color=C["oracle"], ls="-", lw=1.0, alpha=0.6, label="oracle ceiling = 100%")
    ax.set(xlabel="round (each = one batch of ~300 users)",
           ylabel="cumulative fraction of prize captured",
           title="FCUE learns the right targeting policy from aggregates alone")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "learning_curves.png")); plt.close(fig)


def fig_privacy(base):
    eps = [np.inf, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1]
    d = privacy_tradeoff(base, eps, seeds=range(6))
    x = np.arange(len(eps))
    labels = ["clean"] + [f"ε={e:g}" for e in eps[1:]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(x, d["mean"], yerr=d["std"], marker="o", color=C["priv"], lw=2,
                capsize=4, label="FCUE (learned policy)")
    ax.axhline(0.0, color=C["rand"], ls=":", lw=1.2, label="treating nobody (0%)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set(xlabel="privacy budget  (left = no privacy, right = very private)",
           ylabel="fraction of prize captured (final policy)",
           title="The privacy tax: causal targeting degrades gracefully, doesn't collapse")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "privacy_tradeoff.png")); plt.close(fig)


def fig_coupling(base):
    rhos = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
    d = coupling_curve(base, rhos, seeds=range(6))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(d["couplings"], d["uplift_bandit"], marker="o", color=C["clean"], lw=2,
            label="FCUE uplift bandit")
    ax.plot(d["couplings"], d["propensity"], marker="s", color=C["prop"], lw=2,
            label="propensity (lead score)")
    ax.axvline(0.0, color=C["rand"], ls=":", lw=1.2)
    ax.annotate("propensity's whole value\nlives on this correlation\n(that you don't control)",
                xy=(-0.6, 0.05), xytext=(-0.78, 0.42), fontsize=8.5, color=C["prop"],
                arrowprops=dict(arrowstyle="->", color=C["prop"], lw=1))
    ax.set(xlabel="coupling: do eager buyers also respond to coupons?  (−1 = sure-things, +1 = aligned)",
           ylabel="fraction of prize captured",
           title="Uplift wins under every assumption; propensity only when it gets lucky")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "coupling_curve.png")); plt.close(fig)


def fig_uplift(base):
    curves = uplift_curve_data(base, seed=0)
    style = {"oracle (rank by true uplift)": (C["oracle"], "-"),
             "FCUE bandit (believed uplift)": (C["clean"], "-"),
             "propensity (buy-likelihood)": (C["prop"], "--"),
             "random": (C["rand"], ":")}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, (frac, curve) in curves.items():
        color, ls = style[name]
        ax.plot(frac, curve, color=color, ls=ls, lw=2, label=name)
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set(xlabel="fraction of users treated (highest-scored first)",
           ylabel="cumulative incremental profit per user",
           title="Uplift curve: how fast each score finds the users worth treating")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "uplift_curves.png")); plt.close(fig)


def main():
    base = WorldConfig(seed=0, uplift_baseline_coupling=0.0)
    print("1/4 learning curves ...");   fig_learning(base)
    print("2/4 privacy trade-off ..."); fig_privacy(base)
    print("3/4 coupling curve ...");    fig_coupling(base)
    print("4/4 uplift curves ...");     fig_uplift(base)
    print(f"done -> {FIGDIR}")


if __name__ == "__main__":
    main()
