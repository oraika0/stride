#!/usr/bin/env python
"""32-node component ablation -- holdout-style 4-panel grouped bars (config = color).

Configs (each removes ONE design element from full STRIDE):
    STRIDE             grad both + diffusion + per-pair masked encoder
    w/o encoder        flatfc + global state (nomask)        -> per-pair encoder removed
    w/o diffusion      one-shot decode (nodiff)              -> iterative generation removed
    w/o actor grad     encoder_rl_grad_src = critic-only     -> actor->encoder gradient removed

Each config aggregates 2 seeds; the whisker is the Type-A standard uncertainty of
the mean, u(mean) = s / sqrt(n), where s is the cross-seed sample std (ddof=1).
Same data ruler as holdout (real_directed_test, 5 holdout TMs).
Edit CONFIGS to retarget. Pure pandas + matplotlib.
"""
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _figio import save_figure          # noqa: E402  (resolution + formats live there)


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
    denoise_step/make_denoise_step_fig.py so all figures share one definition.
    """
    n = len(vals)
    if n < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(n))


# ============================== CONFIG ======================================
REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
SESS_DIR = ""          # session paths below are already repo-relative
TMS = ["06", "41", "73", "108", "141"]
# Output basenames = the thesis captions, so the files can be dropped into the
# docx without renaming. Keep in sync with the List of Figures / List of Tables.
FIG_NAME = "Figure 19. Ablation performance on 32-node"
TBL_NAME = "Table 11. Ablation performance on 32-node"

# (label, color, [session glob per seed]).  STRIDE green = #479a5f (matches holdout).
CONFIGS = [
    ("STRIDE",   "#479a5f", ["results/stride/runs/base_32node_s17_20260605_114040/test/20260605_121029",        # s17
                             "results/stride/runs/base_32node_s18_20260605_221156/test/20260606_001602"]),# s18 (standardized to seeds 17/18)
    ("w/o encoder", "#3b7dd8", ["results/stride/runs/flatfc_nomask_32node_s17_20260628_050042/test/20260628_143532",       # s17 (standardized to seeds 17/18)
                                "results/stride/runs/flatfc_nomask_32node_s18_20260629_002232/test/20260629_142118"]),     # s18 (2026-06-29 rsync)
    ("w/o diffusion",   "#FF7F0E", ["results/stride/runs/nodiff_32node_s17_20260616_230840/test/20260616_235145",
                                    "results/stride/runs/nodiff_32node_s18_20260618_004911/test/20260618_025841"]),
    ("w/o actor gradient", "#D62728", ["results/stride/runs/critic_32node_s17_20260609_103919/test/20260609_105924",
                                    "results/stride/runs/critic_32node_s18_20260611_122225/test/20260611_132215"]),
]

METRICS = [
    ("max_link_utilization", "(a) Maximum Link Utilization", "MLU (%)",           True),
    ("avg_throughput",       "(b) Average Link Throughput",  "Throughput (Mb/s)", False),
    ("avg_delay",            "(c) Average Link Delay",       "Delay (ms)",        True),
    ("avg_packet_loss",      "(d) Average Packet Loss",      "Packet Loss (%)",   True),
]
MET_COLS = [m[0] for m in METRICS]
# ===========================================================================


def load_one(glob_pat):
    """Latest session (under SESS_DIR) with all 5 TMs -> {tm: {metric: mean}}."""
    cs = sorted(glob.glob(os.path.join(REPO, SESS_DIR, glob_pat)))
    if not cs:
        raise FileNotFoundError(f"no session matches {glob_pat}")
    for sess in reversed(cs):
        out = {}
        for tm in TMS:
            f = glob.glob(os.path.join(sess, "real", tm, "real_directed_test",
                                       f"{tm}_eval_metrics.csv"))
            if f:
                df = pd.read_csv(f[0])
                out[tm] = {c: df[c].mean() for c in MET_COLS}   # all already in test units
        if len(out) == len(TMS):
            return out
    raise FileNotFoundError(f"no complete (5-TM) session for {glob_pat}")


def aggregate(globs):
    """Seeds -> (mean, err) per tm/metric. err = Type-A u(mean) = s / sqrt(n)."""
    seeds = [load_one(g) for g in globs]
    mean, err = {tm: {} for tm in TMS}, {tm: {} for tm in TMS}
    for tm in TMS:
        for c in MET_COLS:
            v = [s[tm][c] for s in seeds]
            mean[tm][c] = float(np.mean(v))
            err[tm][c] = type_a_uncertainty(v)
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
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def write_pertm_table(out):
    """Per-TM directed MLU (mean ± Type-A u(mean)) for every config.

    This is the table that goes into the thesis as 'Ablation performance on
    32-node' -- same numbers the bars in the FIG_NAME figure are annotated
    with. Emitted as markdown so it never has to be transcribed by hand.
    """
    data = [(lab, *aggregate(globs)) for lab, _col, globs in CONFIGS]
    lines = ["> **32-node component ablation -- per-TM directed MLU (mean ± Type-A u(mean))**", "",
             "| TM | " + " | ".join(lab for lab, _m, _e in data) + " |",
             "| :-- |" + " --: |" * len(data)]
    for tm in TMS:
        cells = [f"{m[tm]['max_link_utilization']:.1f} ± {e[tm]['max_link_utilization']:.1f}%"
                 for _lab, m, e in data]
        lines.append(f"| TM-{tm} | " + " | ".join(cells) + " |")
    avg = [f"**{np.mean([m[tm]['max_link_utilization'] for tm in TMS]):.1f}%**"
           for _lab, m, _e in data]
    lines.append("| **Avg** | " + " | ".join(avg) + " |")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"saved {out}")


def main():
    write_pertm_table(os.path.join(HERE, f"{TBL_NAME}.md"))
    data = [(lab, col, *aggregate(g)) for lab, col, g in CONFIGS]
    n = len(CONFIGS)
    x = np.arange(len(TMS))
    width = 0.82 / n
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric, title, ylabel, lb) in zip(axes.flatten(), METRICS):
        vmax = 0.0
        for j, (lab, col, mean, err) in enumerate(data):
            vals = [mean[tm][metric] for tm in TMS]
            yerr = [err[tm][metric] for tm in TMS]
            vmax = max(vmax, max(v + e for v, e in zip(vals, yerr)))
            bars = ax.bar(x + (j - n / 2 + 0.5) * width, vals, width, color=col,
                          edgecolor="black", linewidth=0.4, label=lab,
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
    save_figure(fig, HERE, FIG_NAME, bbox_inches="tight")


if __name__ == "__main__":
    main()
