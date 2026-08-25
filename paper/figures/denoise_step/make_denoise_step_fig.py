#!/usr/bin/env python
"""Denoise-step (M) sweep on 32-node: Figure 18 and Table 10.

Standard PMA-2 baseline (greedy real test) swept over M = 4 / 6 / 8 / 10 / 12. For
each M and each 32-node holdout TM we read the 30 per-step MLU values from its
real-test session (same source as make_holdout_fig) and average them, then write
the bar panels and the per-TM table.

Pure pandas + matplotlib (no wandb).

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/denoise_step/make_denoise_step_fig.py

"""
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _figio import save_figure          # noqa: E402  (resolution + formats live there)


# ============================== CONFIG ======================================
REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
TMS = ["06", "41", "73", "108", "141"]          # 32-node holdout TMs
SESS_DIR = ""          # session paths below are already repo-relative        # session globs in SESSIONS live here

# M -> {seed: session glob}. PMA-2 baseline, greedy. Table averages seeds 17+18;
# the per-step plot uses seed 17 (the headline baseline). M8-s18 is named
# "pma2seed18" (not "pma2_M8_s18") -- it is the plain PMA-2 baseline at seed 18.
SESSIONS = {
    4:  {17: "results/stride/runs/M4_32node_s17_20260607_080043/test/20260607_164306",  18: "results/stride/runs/M4_32node_s18_20260607_162228/test/20260607_171504"},  # s19 MISSING
    6:  {17: "results/stride/runs/M6_32node_s17_20260610_080844/test/20260610_093851",  18: "results/stride/runs/M6_32node_s18_20260611_032312/test/20260611_124930"},
    8:  {17: "results/stride/runs/base_32node_s17_20260605_114040/test/20260605_121029",  18: "results/stride/runs/base_32node_s18_20260605_221156/test/20260606_001602"},
    10: {17: "results/stride/runs/M10_32node_s17_20260610_080811/test/20260610_093804", 18: "results/stride/runs/M10_32node_s18_20260611_040040/test/20260611_125017"},
    12: {17: "results/stride/runs/M12_32node_s17_20260607_075829/test/20260608_094621", 18: "results/stride/runs/M12_32node_s18_20260609_002014/test/20260609_003659"},
}
# Output basename = the thesis caption, so the file can be dropped into the docx
# without renaming. Keep in sync with the List of Figures. The per-step line plot
# is a diagnostic that is NOT in the thesis, so it keeps its descriptive name.
FIG_NAME = "Figure 18. Denoise-step performance on 32-node"
TBL_NAME = "Table 10. Denoise-step performance on 32-node"

TABLE_SEEDS = (17, 18)                          # mean+-std table uses these seeds
BAR_SEEDS = (17, 18)                            # bar plot averages all available of these
ILP_32, OSPF_32 = 66.0, 99.4                    # 32node directed baselines (reference lines)
# M8 = STRIDE green (#479a5f, matches holdout) since M8 is the chosen config;
# the rest are high-contrast distinct hues (purple/blue/orange/red).
M_COLOR = {4: "#6A3D9A", 6: "#1F77B4", 8: "#479a5f", 10: "#FF7F0E", 12: "#D62728"}

# (metric column, panel title, y-label, lower-is-better) -- holdout 4-panel layout
METRICS = [
    ("max_link_utilization", "(a) Maximum Link Utilization", "MLU (%)",           True),
    ("avg_throughput",       "(b) Average Link Throughput",  "Throughput (Mb/s)", False),
    ("avg_delay",            "(c) Average Link Delay",       "Delay (ms)",        True),
    ("avg_packet_loss",      "(d) Average Packet Loss",      "Packet Loss (%)",   True),
]
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

    Keep this identical to the copy in holdout/make_holdout_fig.py and
    ablation/make_ablation_fig.py so all figures share one definition.
    """
    n = len(vals)
    if n < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(n))


def load(session_glob, col="max_link_utilization"):
    """{tm: per-step <col> array} for the latest session (under SESS_DIR) with all TMs."""
    matches = sorted(glob.glob(os.path.join(REPO, SESS_DIR, session_glob)))
    if not matches:
        raise FileNotFoundError(f"no session matches {session_glob}")
    # pick the latest session that has all 5 TMs (handles re-run duplicates).
    for sess in reversed(matches):
        out = {}
        for tm in TMS:
            cs = glob.glob(os.path.join(sess, "real", tm, "real_directed_test",
                                        f"{tm}_eval_metrics.csv"))
            if cs:
                out[tm] = pd.read_csv(cs[0])[col].to_numpy()   # MLU already %, throughput Mb/s
        if len(out) == len(TMS):
            return out
    raise FileNotFoundError(f"no complete (5-TM) session for {session_glob}")


def run_mean_mlu(session_glob):
    """Mean MLU (%) over steps and TMs for one session."""
    d = load(session_glob)
    return float(np.mean([d[tm].mean() for tm in TMS if tm in d]))


def _style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _per_M_metric(col):
    """{M: {tm: (mean, u_mean)}} over BAR_SEEDS for metric column `col`."""
    per_M = {}
    for M in sorted(SESSIONS):
        rows = []                                # one {tm: mean-over-steps} per seed
        for s in BAR_SEEDS:
            if s in SESSIONS[M]:
                d = load(SESSIONS[M][s], col)
                rows.append({tm: d[tm].mean() for tm in TMS if tm in d})
        per_M[M] = {tm: (float(np.mean([r[tm] for r in rows if tm in r])),
                         type_a_uncertainty([r[tm] for r in rows if tm in r]))
                    for tm in TMS}
    return per_M


def _fmt(v):
    if v == 0:
        return "0"
    if v < 1:
        return f"{v:.2f}"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:.0f}"


def make_panels(basename):
    """holdout-style 4-panel (MLU / throughput / delay / loss). Each panel: x = TM,
    one M-colored bar per denoise step, Type-A whisker = u(mean) = s / sqrt(n)."""
    Ms = sorted(SESSIONS)
    n = len(Ms)
    x = np.arange(len(TMS))
    width = 0.82 / n
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (col, title, ylabel, lb) in zip(axes.flatten(), METRICS):
        per_M = _per_M_metric(col)
        vmax = 0.0
        for j, M in enumerate(Ms):
            vals = [per_M[M][tm][0] for tm in TMS]
            yerr = [per_M[M][tm][1] for tm in TMS]
            vmax = max(vmax, max(v + e for v, e in zip(vals, yerr)))
            bars = ax.bar(x + (j - n / 2 + 0.5) * width, vals, width, color=M_COLOR[M],
                          edgecolor="black", linewidth=0.4, label=f"M={M}",
                          yerr=yerr, capsize=2, error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="#333"))
            for b, v, e in zip(bars, vals, yerr):
                ax.annotate(_fmt(v), (b.get_x() + b.get_width() / 2, b.get_height() + e),
                            xytext=(0, 1.2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([f"TM{tm}" for tm in TMS])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(title + (" ↓" if lb else ""), fontsize=12)
        ax.set_ylim(0, vmax * 1.18)
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=n, loc="upper center", frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save_figure(fig, HERE, basename, bbox_inches="tight")


def write_pertm_table(out):
    """Per-TM directed MLU (mean ± Type-A u(mean)) for every M over BAR_SEEDS.

    This is the table that goes into the thesis as 'Denoise-step performance on
    32-node' -- same numbers the bars in the FIG_NAME figure are annotated
    with. Emitted as markdown so it never has to be transcribed by hand.
    (denoise_step_table.md is a DIFFERENT table: 5-TM mean per M, per seed.)
    """
    per_M = _per_M_metric("max_link_utilization")
    Ms = sorted(per_M)
    lines = [f"> **Denoise-step (M) sweep on 32-node -- per-TM directed MLU "
             f"(mean ± Type-A u(mean), seeds {'+'.join(map(str, BAR_SEEDS))})**", "",
             "| TM | " + " | ".join(f"M = {M}" for M in Ms) + " |",
             "| :-- |" + " --: |" * len(Ms)]
    for tm in TMS:
        lines.append(f"| TM-{tm} | " + " | ".join(
            f"{per_M[M][tm][0]:.1f} ± {per_M[M][tm][1]:.1f}%" for M in Ms) + " |")
    lines.append("| **Avg** | " + " | ".join(
        f"**{np.mean([per_M[M][tm][0] for tm in TMS]):.1f}%**" for M in Ms) + " |")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"saved {out}")


def main():
    make_panels(FIG_NAME)
    write_pertm_table(os.path.join(HERE, f"{TBL_NAME}.md"))
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
