#!/usr/bin/env python
"""Reward + its three components (bwd / delay / loss) for ONE STRIDE seed.

Top panel : total reward.
Bottom    : the three reward components, which sum to the total reward.

Point: reward valleys line up with the delay/loss components collapsing --
the capacity-threshold ("cliff") signature. bwd is the everyday differentiator;
delay/loss sit near their max until offered load crosses capacity, then drop.

Reads components.csv, which sits next to this file and is
built from the local training logs by make_curves_csv.py, which this script calls
for you, so there is no order to remember. No network needed.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/reward/make_reward_components_fig.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _figio import save_figure          # noqa: E402  (resolution + formats live there)

# ============================== CONFIG (edit here) ==========================
HERE         = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(HERE, "components.csv")
LINEWIDTH    = 1.2

# Output basename = the thesis caption. Topologies absent here fall back to an
# internal name; only the 32-node panel is a numbered figure.
FIG_NAME     = {"32node": "Figure 16. Reward components on 32-node"}
DIV          = 100           # each term is logged in [0,100]; /100 -> [0,1], total [0,3]

# one STRIDE seed per topology
TOPO_RUN = {"geant": "stride_geant_s17", "32node": "stride_32node_s17"}
# TOPO_RUN = {"geant": "stride_geant_s17", "32node": "stride_32node_v100"}

# per-topology display knobs
TOPO_CFG = {
    "geant":  {"ema": 0.85, "x_start": 0, "total_ylim": (1.5, 3.0), "comp_ylim": (0.4, 1.03)},
    "32node": {"ema": 0.90, "x_start": 0, "total_ylim": (0.5, 3.0), "comp_ylim": (0.0, 1.03)},
}

# (components.csv column, legend label, color)
COMPONENTS = [
    ("reward_bwd",   "bottleneck remaining bw", "#3b7dd8"),
    ("reward_delay", "cumulative delay",        "#f0a04b"),
    ("reward_loss",  "end-to-end loss",         "#d6456f"),
]
TOTAL_COLOR = "#479a5f"
# ===========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEYS = ["reward", "reward_bwd", "reward_delay", "reward_loss"]


def time_weighted_ema(x, y, smoothing):
    n = len(x)
    if n == 0 or smoothing <= 0:
        return list(y)
    rng = (x[-1] - x[0]) or 1.0
    mean_gap = rng / max(n - 1, 1)
    last = debias = 0.0
    out = []
    prev = x[0]
    for i in range(n):
        gap = (x[i] - prev) if i > 0 else mean_gap
        alpha = smoothing ** (gap / mean_gap) if mean_gap > 0 else smoothing
        last = last * alpha + (1 - alpha) * y[i]
        debias = debias * alpha + (1 - alpha)
        out.append(last / debias if debias > 0 else y[i])
        prev = x[i]
    return out


def load_data():
    """Rebuild components.csv from the training logs, then read it.

    This used to only check the file existed and tell you to go run
    make_curves_csv.py yourself. That was worth it while the rebuild parsed
    output_all.txt and took forty seconds; reading output.txt instead it takes a
    third of one, so redoing the work unconditionally is cheaper than deciding
    whether it needs redoing -- and it removes the quieter failure, where the
    figure plots a CSV older than the runs it claims to show and says nothing.

    make_curves_csv.py still runs standalone and still writes both CSVs. This
    calls the same builder and refreshes the file on the way past.
    """
    sys.path.insert(0, HERE)
    import make_curves_csv                      # sibling script, same folder
    df = make_curves_csv.build_components()
    df.to_csv(CSV_PATH, index=False)
    return df


def _style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="out", length=4, width=1.0, bottom=True, left=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_topology(df, topo, rid):
    cfg = TOPO_CFG[topo]
    d = df[df.run_id == rid].sort_values("step")
    xs = d.step.values

    fig, ax = plt.subplots(figsize=(7, 4.3))
    # total reward (= sum of the three components) + the components, all one axis
    yt = time_weighted_ema(xs, (d.reward / DIV).values, cfg["ema"])
    ax.plot(xs, yt, color=TOTAL_COLOR, lw=LINEWIDTH + 0.5, label="total reward")
    for col, label, color in COMPONENTS:
        yc = time_weighted_ema(xs, (d[col] / DIV).values, cfg["ema"])
        ax.plot(xs, yc, color=color, lw=LINEWIDTH, label=label)
    ax.set_ylim(0, 3.05)
    ax.set_xlim(left=cfg["x_start"])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward")
    _style(ax)
    ax.legend(loc="center right", frameon=False, fontsize=9, ncol=1)

    fig.tight_layout()
    save_figure(fig, HERE, FIG_NAME.get(topo, f"reward_components_{topo}"))
    return fig


def main():
    df = load_data()
    for topo, rid in TOPO_RUN.items():
        plot_topology(df, topo, rid)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
