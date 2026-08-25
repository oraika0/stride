#!/usr/bin/env python
"""GEANT holdout real-Mininet test: bar chart + STRIDE-vs-baselines diff table.

Reads each method's directed eval metrics from its test session(s), then
writes:
    FIG_NAME[topo]  -- 4-panel bar chart (per holdout TM, per method), one
                       file per format in _figio.FIG_FORMATS
    TBL_NAME[topo]  -- MLU difference table (percentage points), markdown

A method may list MULTIPLE sessions (seeds); its bar shows the seed-mean with a
Type-A uncertainty whisker, u(mean) = s / sqrt(n), where s is the cross-seed
sample std (ddof=1). Single-seed methods get no whisker.

Pure pandas + matplotlib (no wandb). Edit the CONFIG block
to retarget. abla_test data lives under results/<alg>/test/<session>/real/<tm>/.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/holdout/make_holdout_fig.py

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
SHOW_ERRORBAR = True         # draw Type-A whisker on multi-seed methods (s17/s18).
                             # 2026-06-28: standardize STRIDE + LS2IC to seeds 17/18.

# method label -> list of test session globs (one per seed).
# First entry is the BASELINE used for the diff table (STRIDE).
# Output basename per topology = the thesis caption, so the file can be dropped
# into the docx without renaming. Keep in sync with the List of Figures.
FIG_NAME = {
    "geant":  "Figure 14. Performance evaluation in GÉANT",
    "32node": "Figure 17. Performance evaluation in 32-node",
}
TBL_NAME = {
    "geant":  "Table 8. MLU difference on GÉANT (percentage points)",
    "32node": "Table 9. MLU difference on 32-node (percentage points)",
}
_METHODS = {
  "geant": [
    ("STRIDE", ["results/stride/runs/base_geant_s17_20260603_214500/test/20260603_224428",            # seed 17
                "results/stride/runs/base_geant_s18_20260611_114456/test/20260611_132127",       # seed 18
                ]),
    ("LS2IC",  ["results/ls2ic_dd/runs/ls2ic_dd_geant_s17_20260524_125905/test/20260524_145702",                    # seed 17
                "results/ls2ic_dd/runs/ls2ic_dd_geant_s18_20260630_020836/test/20260630_032109"]),                  # seed 18 (2026-06-30)
    ("MADQN",  ["results/ps_dqn_dd/runs/ps_dqn_dd_geant_s17_20260618_235521/test/20260619_160141",   # seed 17 (fix)
                "results/ps_dqn_dd/runs/ps_dqn_dd_geant_s18_20260630_020924/test/20260630_032312"]),                 # seed 18 (2026-06-30)
    ("DRSIR",  ["results/drsir_dd/runs/drsir_dd_geant_s17_20260710_180907/test/20260710_180907",    # seed 17 (dd: STRIDE-aligned directed reward source, 2026-07-10; seed REAL since seed_torch wiring)
                "results/drsir_dd/runs/drsir_dd_geant_s18_20260710_181037/test/20260710_181037"]),  # seed 18 (dd)
    ("OSPF",   ["results/ospf/runs/ospf_geant_s17_20260524_203330/test/20260524_203330"]),
    ("ILP",    ["results/ilp/runs/ilp_geant_s17_20260528_132121/test/20260528_132121"]),
  ],
  "32node": [
    ("STRIDE", ["results/stride/runs/base_32node_s17_20260605_114040/test/20260605_121029",          # seed 17
                "results/stride/runs/base_32node_s18_20260605_221156/test/20260606_001602",   # seed 18
                ]),
    ("LS2IC",  ["results/ls2ic_dd/runs/ls2ic_dd_32node_s17_20260627_113519/test/20260627_161008",   # seed 17
                "results/ls2ic_dd/runs/ls2ic_dd_32node_s18_20260611_122711/test/20260611_132255"]),        # seed 18
    ("MADQN",  ["results/ps_dqn_dd/runs/ps_dqn_dd_32node_s17_20260618_233958/test/20260619_160116",  # seed 17 (fix)
                "results/ps_dqn_dd/runs/ps_dqn_dd_32node_s18_20260630_021144/test/20260630_032348"]),                # seed 18 (2026-06-30)
    ("DRSIR",  ["results/drsir_dd/runs/drsir_dd_32node_s17_20260710_180950/test/20260710_180950",    # seed 17 (dd: STRIDE-aligned directed reward source, 2026-07-10)
                "results/drsir_dd/runs/drsir_dd_32node_s18_20260710_184235/test/20260710_184235"]),  # seed 18 (dd)
    ("OSPF",   ["results/ospf/runs/ospf_32node_s17_20260605_020339/test/20260605_020339"]),
    ("ILP",    ["results/ilp/runs/ilp_32node_s17_20260605_020413/test/20260605_020413"]),
  ],
}
BASELINE = "STRIDE"
DIFF_AGAINST = ["LS2IC", "MADQN", "DRSIR", "OSPF", "ILP"]   # diff-table columns, match figure order (ILP = lower bound, shown for reference -> negative)

METHOD_COLOR = {
    "STRIDE": "#479a5f", "MADQN": "#FFB83D", "LS2IC": "#C2327A",
    "DRSIR": "#3b7dd8", "OSPF": "#9aa0a6", "ILP": "#444444",
}

# (metric column, panel title, y-label, y-scale, lower-is-better)
METRICS = [
    ("max_link_utilization", "(a) Maximum Link Utilization", "MLU (%)",            "linear", True),
    ("avg_throughput",       "(b) Average Link Throughput",  "Throughput (Mb/s)",  "linear", False),
    ("avg_delay",            "(c) Average Link Delay",       "Delay (ms)",         "linear",    True),
    ("avg_packet_loss",      "(d) Average Packet Loss",      "Packet Loss (%)",    "linear", True),
]
MET_COLS = [m[0] for m in METRICS]
# ===========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def type_a_uncertainty(vals):
    """Type-A standard uncertainty of the MEAN: u(mean) = s / sqrt(n).

    s = np.std(vals, ddof=1) is the cross-seed SAMPLE std (denominator n-1); it
    answers "how far would one more seed land from the mean". Dividing by
    sqrt(n) converts it to the uncertainty of the reported MEAN, which is what
    an error bar on a mean should show.

    n = 2 makes this numerically equal to |a-b|/2, i.e. to np.std(ddof=0) --
    a coincidence that does NOT hold for n >= 3, so always use this helper
    rather than either bare np.std form.
    """
    n = len(vals)
    if n < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(n))


def load_one(label, glob_pat):
    """One session -> {tm: {metric: value}}. DRSIR has a different file layout
    plus a directed-MLU column; everything else uses real_directed_test."""
    matches = sorted(glob.glob(os.path.join(REPO, glob_pat)))
    if not matches:
        raise FileNotFoundError(f"{label}: no session matches {glob_pat}")
    sess = matches[-1]
    is_drsir = label == "DRSIR"
    pat = "real/*/*_eval_metrics.csv" if is_drsir else "real/*/real_directed_test/*_eval_metrics.csv"
    out = {}
    for c in sorted(glob.glob(os.path.join(sess, pat))):
        tm = os.path.basename(c).split("_")[0]
        df = pd.read_csv(c)
        # NOTE on loss units: the DIRECTED loss (both DRSIR's avg_packet_loss_directed
        # and the others' real_directed_test avg_packet_loss) is ALREADY a percentage
        # (compute_network_metrics averages net_info_directed pkloss, which the
        # controller writes in %). Only the UNDIRECTED avg_packet_loss is a 0-1 ratio.
        # The loader used to *100 both, inflating the directed loss 100x (DRSIR TM141
        # 1.57% -> 157%). Do NOT *100 the directed loss.
        if is_drsir:
            # DRSIR runs on the undirected env but reports the directed ruler as
            # extra *_directed columns (added in the 2026-06-27 re-run). Use them so
            # DRSIR is apples-to-apples with the other methods' real_directed_test.
            out[tm] = {
                "max_link_utilization": df.get("max_link_utilization_directed", df["max_link_utilization"]).mean(),
                "avg_delay":            df.get("avg_delay_tc_directed",         df["avg_delay"]).mean(),
                "avg_packet_loss":      df["avg_packet_loss_directed"].mean(),   # already in %
                "avg_throughput":       df.get("avg_throughput_directed",       df["avg_throughput"]).mean(),
            }
        else:
            out[tm] = {
                "max_link_utilization": df["max_link_utilization"].mean(),
                "avg_delay":            df["avg_delay"].mean(),
                "avg_packet_loss":      df["avg_packet_loss"].mean(),   # real_directed_test loss already in %
                "avg_throughput":       df["avg_throughput"].mean(),
            }
    if not out:
        raise FileNotFoundError(f"{label}: no eval CSVs under {sess}")
    return out


def load_method(label, globs):
    """Aggregate seeds -> (mean, err) dicts. err = Type-A u(mean) = s / sqrt(n)."""
    seeds = [load_one(label, g) for g in globs]
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
            # Deterministic / single-seed methods (OSPF, ILP) have all-zero error;
            # draw NO whisker for them (else capsize=2 renders a stray 0-height cap).
            # Multi-seed methods (STRIDE/LS2IC/MADQN/DRSIR) keep their Type-A whisker.
            draw_err = SHOW_ERRORBAR and any(e > 0 for e in yerr)
            cap = [(e if draw_err else 0.0) for e in yerr]
            vmax = max(vmax, max(v + c for v, c in zip(vals, cap)))
            bars = ax.bar(x + (j - n / 2 + 0.5) * width, vals, width,
                          color=METHOD_COLOR.get(label, "gray"), label=label,
                          edgecolor="black", linewidth=0.4,
                          yerr=(yerr if draw_err else None),
                          capsize=(2 if draw_err else 0),
                          error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor="#333333"))
            for b, v, c in zip(bars, vals, cap):             # value at each bar apex
                ax.annotate(_fmt(v), (b.get_x() + b.get_width() / 2, b.get_height() + c),
                            xytext=(0, 1.2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xlabel(title + (" ↓" if _lb else ""), fontsize=12)   # panel name below
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"TM{t}" for t in tms], fontsize=10)
        if yscale == "log":
            ax.set_yscale("log")
            ax.set_ylim(top=vmax * 3.0)
        elif yscale == "symlog":
            ax.set_yscale("symlog", linthresh=0.5)
            ax.set_ylim(top=vmax * 3.0)
        else:
            ax.set_ylim(top=vmax * 1.22)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, HERE, basename)


def diff_table(data, tms, out_md):
    """BASELINE seed-mean MLU difference against each method in DIFF_AGAINST,
    per TM. MLU is in percent, so the difference is in percentage points:
    difference = other - base."""
    base = data[BASELINE]
    lines = ["> **STRIDE holdout MLU difference over baselines (percentage points)**\n",
             "| STRIDE vs | " + " | ".join(DIFF_AGAINST) + " |",
             "| :-- |" + " --: |" * len(DIFF_AGAINST)]
    acc = {b: [] for b in DIFF_AGAINST}
    for tm in tms:
        cells = []
        for b in DIFF_AGAINST:
            o, s = data[b][tm]["max_link_utilization"], base[tm]["max_link_utilization"]
            diff = o - s
            acc[b].append(diff)
            cells.append(f"{diff:.2f}")
        lines.append(f"| TM-{tm} | " + " | ".join(cells) + " |")
    lines.append("| **Avg** | " + " | ".join(f"**{np.mean(acc[b]):.2f}**" for b in DIFF_AGAINST) + " |")
    md = "\n".join(lines) + "\n"
    with open(out_md, "w") as f:
        f.write(md)
    print(f"saved {out_md}\n")
    print(md)


def main():
    # Both topologies in one run. This used to be a TOPO constant you edited
    # between runs, which meant regenerating "the holdout figures" quietly
    # produced half of them and said nothing about the other half.
    for topo, methods in _METHODS.items():
        data, errs = {}, {}
        for label, globs in methods:
            data[label], errs[label] = load_method(label, globs)
        tms = sorted(set.intersection(*[set(d.keys()) for d in data.values()]), key=int)
        bar_chart(data, errs, tms, FIG_NAME[topo])
        diff_table(data, tms, os.path.join(HERE, f"{TBL_NAME[topo]}.md"))
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
