#!/usr/bin/env python
"""32node holdout real-Mininet K-sufficiency ablation: bar chart + diff table.

STRIDE (canonical pma2 config, seed 17, greedy) trained/tested at K in
{10, 15, 20, 25, 30}. K<=20 use the frozen canonical k_paths.json prefix;
K=25/30 use k_paths_k30_ext.json (frozen 20 verbatim + 10 fresh hop-count Yen,
see dataset/extend_k_paths.py). Single seed -> no whiskers.

Writes:
    k_ablation_bars_32node.png -- 4-panel bar chart (per holdout TM, per K)
    k_ablation_diff_32node.md  -- MLU difference of each K against K=20 (pp)

Layout mirrors ../holdout/make_holdout_fig.py.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/k_ablation/make_k_fig.py

"""
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _figio import save_figure          # noqa: E402  (resolution + formats live there)


# ============================== CONFIG (edit here) ==========================
REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
# Output basename = the thesis caption (List of Tables). The bar chart itself
# is a diagnostic and is NOT in the thesis, so it keeps its descriptive name.
TBL_NAME = "Table 13. Candidate-Path Sufficiency Analysis on 32-node"

# K label -> [seed-17 session, seed-18 session] (greedy). 2026-07-18: upgraded
# to 2 seeds per cell; bar = seed mean, whisker = population std (|s17-s18|/2),
# mirroring make_holdout_fig's multi-seed convention.
METHODS = [
    ("K=10", ["results/stride/runs/k10_32node_s17_20260716_040318/test/20260716_223152",
              "results/stride/runs/k10_32node_s18_20260718_034831/test/20260718_165108"]),
    ("K=15", ["results/stride/runs/k15_32node_s17_20260716_122458/test/20260716_230351",
              "results/stride/runs/k15_32node_s18_20260718_121016/test/20260718_172306"]),
    ("K=20", ["results/stride/runs/base_32node_s17_20260605_114040/test/20260605_121029",
              "results/stride/runs/base_32node_s18_20260605_221156/test/20260606_001602",]),
    # ("K=20", ["results/stride/runs/base_32node_s17_20260605_114040/test/20260605_121029",
    #           "results/stride/runs/base_32node_s18_20260605_221156/test/20260606_001602",
    #           "results/stride/runs/base_32node_s17_20260824_063712/test/20260824_083547"]),
    ("K=25", ["results/stride/runs/k25_32node_s17_20260716_040416/test/20260716_223311",
              "results/stride/runs/k25_32node_s18_20260718_034925/test/20260718_165032"]),
    ("K=30", ["results/stride/runs/k30_32node_s17_20260716_122605/test/20260716_230513",
              "results/stride/runs/k30_32node_s18_20260718_121113/test/20260718_172234"]),
]
BASELINE = "K=20"                       # diff table reference (paper default)
DIFF_AGAINST = ["K=10", "K=15", "K=25", "K=30"]

# Sequential ramp so K order reads left->right; K=20 keeps the STRIDE green.
METHOD_COLOR = {
    "K=10": "#c7e9c0", "K=15": "#84c795", "K=20": "#479a5f",
    "K=25": "#2c6e46", "K=30": "#174a2f",
}

METRICS = [
    ("max_link_utilization", "(a) Maximum Link Utilization", "MLU (%)",           "linear", True),
    ("avg_throughput",       "(b) Average Link Throughput",  "Throughput (Mb/s)", "linear", False),
    ("avg_delay",            "(c) Average Link Delay",       "Delay (ms)",        "linear", True),
    ("avg_packet_loss",      "(d) Average Packet Loss",      "Packet Loss (%)",   "linear", True),
]
MET_COLS = [m[0] for m in METRICS]
# ===========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_one(label, sess_rel):
    """One session -> {tm: {metric: value}} from real_directed_test CSVs."""
    sess = os.path.join(REPO, sess_rel)
    out = {}
    for c in sorted(glob.glob(os.path.join(sess, "real/*/real_directed_test/*_eval_metrics.csv"))):
        tm = os.path.basename(c).split("_")[0]
        df = pd.read_csv(c)
        out[tm] = {
            "max_link_utilization": df["max_link_utilization"].mean(),
            "avg_delay":            df["avg_delay"].mean(),
            "avg_packet_loss":      df["avg_packet_loss"].mean(),   # directed loss already in %
            "avg_throughput":       df["avg_throughput"].mean(),
        }
    if not out:
        raise FileNotFoundError(f"{label}: no eval CSVs under {sess}")
    return out


def type_a_uncertainty(vals):
    """Type-A standard uncertainty of the MEAN: u(mean) = s / sqrt(n).

    s = np.std(vals, ddof=1) is the cross-seed SAMPLE std (denominator n-1); it
    answers "how far would one more seed land from the mean". Dividing by
    sqrt(n) converts it to the uncertainty of the reported MEAN, which is what
    an error bar on a mean should show.

    Keep this identical to the copies in holdout/, ablation/ and denoise_step/
    so all figures and tables share one definition.
    """
    n = len(vals)
    if n < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(n))


def load_method(label, sess_list):
    """Aggregate seeds -> (mean, err) dicts. err = Type-A u(mean) = s / sqrt(n)."""
    seeds = [load_one(label, s) for s in sess_list]
    tms = sorted(set.intersection(*[set(s) for s in seeds]), key=int)
    mean = {tm: {} for tm in tms}
    err = {tm: {} for tm in tms}
    for tm in tms:
        for m in MET_COLS:
            vals = [s[tm][m] for s in seeds]
            mean[tm][m] = float(np.mean(vals))
            err[tm][m] = type_a_uncertainty(vals)
    return mean, err


def _fmt(v):
    if v == 0:
        return "0"
    if v < 1:
        return f"{v:.2f}"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:.0f}"


def _style(ax):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="out", length=4, width=1.0, bottom=True, left=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def bar_chart(data, errs, tms, basename):
    n = len(data)
    x = np.arange(len(tms))
    width = 0.8 / n
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes = axes.flatten()
    for ax, (metric, title, ylabel, yscale, _lb) in zip(axes, METRICS):
        vmax = 0.0
        for j, (label, d) in enumerate(data.items()):
            vals = [d[tm][metric] for tm in tms]
            yerr = [errs[label][tm][metric] for tm in tms]
            vmax = max(vmax, max(v + e for v, e in zip(vals, yerr)))
            bars = ax.bar(x + (j - n / 2 + 0.5) * width, vals, width,
                          color=METHOD_COLOR.get(label, "gray"), label=label,
                          edgecolor="black", linewidth=0.4,
                          yerr=yerr, capsize=2,
                          error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor="#333333"))
            for b, v, e in zip(bars, vals, yerr):
                ax.annotate(_fmt(v), (b.get_x() + b.get_width() / 2, b.get_height() + e),
                            xytext=(0, 1.2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xlabel(title + (" ↓" if _lb else ""), fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"TM{t}" for t in tms], fontsize=10)
        ax.set_ylim(top=vmax * 1.22)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Diagnostic, not a thesis figure. The descriptive filename is what says so;
    # the formats are the same as everything else, because a plot being for us
    # rather than for the manuscript is no reason to have fewer ways to open it.
    save_figure(fig, HERE, basename)


def mlu_table(data, tms, out_md):
    """Raw holdout MLU (%) per TM per K -- K as columns (matches the oracle
    table layout), K=20 included. 2026-07-17: replaced the old diff-vs-K20
    table on user request (raw MLU% reads directly against the bar fig)."""
    labels = [lb for lb, _ in METHODS]
    lines = ["> **STRIDE K-sufficiency: real-Mininet holdout MLU (%), canonical pma2 "
             "config, mean of seeds 17/18, greedy (30 eval rows per TM per seed)**\n",
             "| | " + " | ".join(labels) + " |",
             "| :-- |" + " --: |" * len(labels)]
    for tm in tms:
        cells = [f"{data[lb][tm]['max_link_utilization']:.1f}%" for lb in labels]
        lines.append(f"| TM{tm} | " + " | ".join(cells) + " |")
    lines.append("| **平均** | " + " | ".join(
        f"**{np.mean([data[lb][tm]['max_link_utilization'] for tm in tms]):.1f}%**"
        for lb in labels) + " |")
    md = "\n".join(lines) + "\n"
    with open(out_md, "w") as f:
        f.write(md)
    print(f"saved {out_md}\n")
    print(md)


def main():
    data, errs = {}, {}
    for label, sess_list in METHODS:
        data[label], errs[label] = load_method(label, sess_list)
    tms = sorted(set.intersection(*[set(d.keys()) for d in data.values()]), key=int)
    bar_chart(data, errs, tms, "k_ablation_bars_32node")
    mlu_table(data, tms, os.path.join(HERE, f"{TBL_NAME}.md"))
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
