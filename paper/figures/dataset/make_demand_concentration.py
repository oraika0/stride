#!/usr/bin/env python
"""Demand-concentration (Lorenz / Pareto) curve.

Verifies the paper claim "top 10% of source-destination pairs carry ~80% of
GÉANT demand" -- and contrasts it with the much flatter 32-node (gravity) TMs.

    x-axis: top fraction of OD pairs, sorted by demand DESCENDING (0..100%)
    y-axis: cumulative share of total demand (0..100%)
    diagonal y=x  -> perfectly uniform demand (no concentration)

The GÉANT curve is averaged over the 5 holdout test TMs; the 32-node curve is
averaged over a sample of its TMs. A dotted line at x=10% marks the value used
in the paper. Pure numpy + matplotlib.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/dataset/make_demand_concentration.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _figio import save_figure          # noqa: E402
import importlib.util
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ============================== CONFIG ======================================
REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
GEANT_TEST = [3, 10, 12, 14, 21]   # holdout test TMs (hour index), scale x3
MARK_X = 0.10                      # the "top 10%" annotation
LOG_X = False                      # log x-axis: expands the head (top few %)
LOG_Y = False                      # log y-axis: expands the low cumulative-demand region
GEANT_COLOR, N32_COLOR = "#2c7fb8", "#e6843c"
# ===========================================================================

# geant_tm(period, hour, scale) helper lives in make_dataset_figs.py
_spec = importlib.util.spec_from_file_location(
    "mdf", os.path.join(HERE, "make_dataset_figs.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def lorenz(v):
    """top-fraction of ALL OD pairs (desc) -> cumulative demand share. Both start 0.

    Includes zero-demand pairs (do NOT filter v>0): the routing problem spans all
    N(N-1) OD pairs, so "top 10% of OD pairs" is over the full set. Filtering zeros
    would rescale the x-axis to active pairs only and overstate 32-node's spread."""
    v = np.sort(np.asarray(v, float))[::-1]
    cum = np.cumsum(v) / v.sum()
    x = np.arange(1, len(v) + 1) / len(v)
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], cum])


def avg_curve(mats, grid):
    return np.mean([np.interp(grid, *lorenz(v)) for v in mats], axis=0)


def offdiag(Mx):
    Mx = np.asarray(Mx, float)
    return Mx[~np.eye(Mx.shape[0], dtype=bool)]


def main():
    grid = np.linspace(0, 1, 400)

    geant_mats = [offdiag(M.geant_tm("01-03", h, 3)) for h in GEANT_TEST]
    obj = pickle.load(open(os.path.join(
        REPO, "dataset/32node_traffic/traffic_generator/32node_tms_info_144tm.pkl"), "rb"))
    tms = obj[1] if isinstance(obj, (tuple, list)) and len(obj) == 2 else obj
    n32_mats = [offdiag(tms[i]) for i in range(0, len(tms), 10)]

    gy = avg_curve(geant_mats, grid)
    ny = avg_curve(n32_mats, grid)
    g_at, n_at = np.interp(MARK_X, grid, gy), np.interp(MARK_X, grid, ny)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    _ref = np.linspace(0.2, 100, 300)                       # y=x reference (a curve under a log axis)
    ax.plot(_ref, _ref, color="#cccccc", lw=1.2, ls="--", label="uniform (y=x)")
    ax.plot(grid * 100, gy * 100, color=GEANT_COLOR, lw=1.8, label="GÉANT")
    ax.plot(grid * 100, ny * 100, color=N32_COLOR, lw=1.8, label="32-node")
    if LOG_X:
        ax.set_xscale("log")
    if LOG_Y:
        ax.set_yscale("log")

    ax.axvline(MARK_X * 100, color="#bbbbbb", lw=0.8, ls=":")
    ax.scatter([MARK_X * 100, MARK_X * 100], [g_at * 100, n_at * 100],
               color=[GEANT_COLOR, N32_COLOR], zorder=5, s=28)
    ax.annotate(f"{g_at*100:.0f}%", (MARK_X * 100, g_at * 100),
                xytext=(14, g_at * 100 - 5), fontsize=11, color=GEANT_COLOR, fontweight="bold")
    ax.annotate(f"{n_at*100:.0f}%", (MARK_X * 100, n_at * 100),
                xytext=(14, n_at * 100 - 5), fontsize=11, color=N32_COLOR, fontweight="bold")

    ax.set_xlabel("Top src-dst pairs by demand (%)")
    ax.set_ylabel("Cumulative demand (%)")
    ax.set_xlim(0.15 if LOG_X else 0, 100)
    ax.set_ylim(2 if LOG_Y else 0, 105 if LOG_Y else 100)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.tight_layout()
    save_figure(fig, HERE, "Figure 12. Demand concentration comparison")
    print(f"GÉANT top {MARK_X*100:.0f}% pairs -> {g_at*100:.1f}% demand")
    print(f"32node top {MARK_X*100:.0f}% pairs -> {n_at*100:.1f}% demand")
    for f in (0.05, 0.10, 0.20):
        print(f"  top {f*100:>2.0f}%:  GÉANT {np.interp(f,grid,gy)*100:5.1f}%   "
              f"32node {np.interp(f,grid,ny)*100:5.1f}%")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
