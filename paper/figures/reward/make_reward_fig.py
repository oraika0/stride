#!/usr/bin/env python
"""Training-reward comparison figure (STRIDE vs MADQN vs LS2IC).

Builds train_curves.csv from the local training logs, reads it back, and writes
"Figure 13. Training reward on GÉANT" and "Figure 15. Training reward on
32-node" into this folder, one file per format.

Needs only pandas, numpy and matplotlib. No network, no wandb account.

train_curves.csv is built by the sibling script make_curves_csv.py, which this
one calls for you — there is no order to remember and no stale CSV to notice.
See its docstring for where the numbers come from, and why the local logs rather
than W&B are the original record.

WHAT YOU LIKELY WANT TO EDIT  ->  the CONFIG block right below.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/reward/make_reward_fig.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _figio import save_figure          # noqa: E402  (resolution + formats live there)

# ============================== CONFIG (edit here) ==========================
HERE         = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(HERE, "train_curves.csv")

REWARD_DIV   = 100           # reward is logged in [0,300]; /100 -> paper [0,3]

# per-topology knobs -- tune geant / 32node independently:
#   ema     = time-weighted EMA weight (0=raw, ->1 smoother)
#   x_start = cut the head; x-axis starts here (hide warmup steps before it)
#   y_lim   = reward axis range (cut below it, no info there)
TOPO = {
    "geant":  {"ema": 0.85, "x_start": 0,  "y_lim": (1.5, 3.0)},
    "32node": {"ema": 0.95, "x_start": 0, "y_lim": (0.5, 3.0)},
}
LINEWIDTH    = 1.0           # 1px line

# Output basename per topology = the thesis caption, so the file drops into the
# document without renaming. Keep in sync with the List of Figures.
FIG_NAME = {
    "geant":  "Figure 13. Training reward on GÉANT",
    "32node": "Figure 15. Training reward on 32-node",
}


# topology -> list of (legend label, run id, method)
# run ids index into train_curves.csv; make_curves_csv.py::RUN_MAP maps each one
# to the local training directory it is read from.
# Comment out any line you don't want plotted.
RUNS = {
    "geant": [
        ("STRIDE (seed 17)", "stride_geant_s17", "STRIDE"),
        ("STRIDE (seed 18)", "stride_geant_s18", "STRIDE"),   # holdout geant s18 (seed=18, pc0)
        ("LS2IC (seed 17)",  "ls2ic_geant_s17", "LS2IC"),
        ("LS2IC (seed 18)",  "ls2ic_geant_s18", "LS2IC"),
        ("MADQN (seed 17)",  "madqn_geant_s17", "MADQN"),
        ("MADQN (seed 18)",  "madqn_geant_s18", "MADQN"),
    ],
    "32node": [
        # ("STRIDE (seed 16)", "tzsabapp", "STRIDE"),
        ("STRIDE (seed 17)", "stride_32node_s17", "STRIDE"),
        ("STRIDE (seed 18)", "stride_32node_s18", "STRIDE"),   # holdout 32node s18 (seed=18, pc2)
        # ("STRIDE (seed 19)", "wyb98o6h", "STRIDE"),
        ("LS2IC (seed 17)",  "ls2ic_32node_s17", "LS2IC"),
        ("LS2IC (seed 18)",  "ls2ic_32node_s18", "LS2IC"),
        ("MADQN (seed 17)",  "madqn_32node_s17", "MADQN"),
        ("MADQN (seed 18)",  "madqn_32node_s18", "MADQN"),
    ],
}

METHOD_COLOR = {            # consistent color per method
    "STRIDE": "#479a5f", 
    "MADQN":  "#FFB83D",   
    "LS2IC":  "#C2327A",  
}
# ===========================================================================

from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def time_weighted_ema(x, y, smoothing):
    """wandb-style time-weighted EMA with debias. smoothing in [0,1)."""
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
    """Rebuild train_curves.csv from the training logs, then read it.

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
    df = make_curves_csv.build_train_curves()
    df.to_csv(CSV_PATH, index=False)
    return df


def plot_topology(df, topo, runs):
    cfg = TOPO.get(topo, {"ema": 0.85, "x_start": 0, "y_lim": (1.5, 3.0)})
    fig, ax = plt.subplots(figsize=(7, 4.3))
    # group runs by method: multi-seed methods merge into mean line + min-max shadow
    groups = OrderedDict()
    for label, rid, method in runs:
        groups.setdefault(method, []).append(rid)

    for method, rids in groups.items():
        color = METHOD_COLOR.get(method, "gray")
        series = []
        for rid in rids:
            d = df[df.run_id == rid].sort_values("step")
            if d.empty:
                print(f"  [skip] {rid} not in CSV (add it to make_curves_csv.py "
                      f"RUN_MAP and re-run that script)")
                continue
            xs = d.step.values
            ys = time_weighted_ema(xs, (d.reward / REWARD_DIV).values, cfg["ema"])
            series.append((xs, np.asarray(ys)))
        if not series:
            continue
        if len(series) == 1:                        # single run -> plain line
            xs, ys = series[0]
            ax.plot(xs, ys, color=color, lw=LINEWIDTH, label=method)
        else:                                       # multi-seed -> mean + min-max shadow
            lo_x = max(s[0].min() for s in series)
            hi_x = min(s[0].max() for s in series)
            grid = np.linspace(lo_x, hi_x, 1200)
            M = np.vstack([np.interp(grid, xs, ys) for xs, ys in series])
            ax.fill_between(grid, M.min(0), M.max(0), color=color, alpha=0.22, lw=0)
            ax.plot(grid, M.mean(0), color=color, lw=LINEWIDTH, label=method)

    ax.set_xlim(left=cfg["x_start"])
    ax.set_ylim(*cfg["y_lim"])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="out", length=4, width=1.0, bottom=True, left=True)
    ax.spines["top"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.spines["right"].set_visible(False)

    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=9, ncol=1)

    fig.tight_layout()
    save_figure(fig, HERE, FIG_NAME.get(topo, f"reward_{topo}"))
    return fig


def main():
    df = load_data()
    for topo, runs in RUNS.items():
        plot_topology(df, topo, runs)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
