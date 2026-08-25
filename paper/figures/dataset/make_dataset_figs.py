#!/usr/bin/env python3
"""
make_dataset_figs.py -- paper dataset figures for STRIDE (GÉANT + 32-node).

Replaces the two legacy notebooks
    dataset/geant_traffic/traffic_generator/view.ipynb
    dataset/32node_traffic/traffic_generator/view.ipynb
and adapts them to OUR setup:

  * GÉANT traffic scaled x3 (was x5).  tm_scale=3 -> 23node_s3.
  * GÉANT x-axis = raw TM index 0..23 (NO 13-hour shift).  The legacy cell
    reordered the data with `tms[13:]+tms[:13]` while keeping 0..23 tick
    labels, so labels and data disagreed.  main.py executes TMs by their
    index (tm_list_train = ['13','15',...]); the index itself is never
    permuted, so the figure plots load vs raw index directly.
  * 32-node uses the 144-TM set only (test = 6/41/73/108/141).

Writes each figure into this folder in four formats through _figio, named after
its thesis caption.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/dataset/make_dataset_figs.py

"""

import os
import sys
import json
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _figio import save_figure          # noqa: E402  (resolution + formats live there)

# ------------------------------- CONFIG -------------------------------
REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUTDIR = os.path.join(REPO, "paper/figures/dataset")
GEANT_SCALE = 3                                    # was 5 (學長); we use s3

GEANT_GEN = os.path.join(REPO, "dataset/geant_traffic/traffic_generator")
GEANT_XML = os.path.join(GEANT_GEN, "traffic-matrices")   # the tracked day-01-03 subset; -all is the full 207 MB archive, not in the repo
GEANT_BW  = os.path.join(REPO, "dataset/geant_traffic/bw_r.txt")
GEANT_POS = os.path.join(GEANT_GEN, "pos.json")

N32_GEN   = os.path.join(REPO, "dataset/32node_traffic/traffic_generator")
N32_PKL   = os.path.join(N32_GEN, "32node_tms_info_144tm.pkl")
N32_TRAIN = os.path.join(N32_GEN, "train_indices.pkl")
N32_BW    = os.path.join(REPO, "dataset/32node_traffic/bw_r.txt")
N32_POS   = os.path.join(N32_GEN, "pos_32node.json")

# Train / test TM indices (match config/env/*_config.py).  No validation set.
# GÉANT: 24 hourly TMs of day 2005-01-03 (idx 0-23).
GEANT_TRAIN = [0, 1, 2, 4, 7, 8, 9, 11, 13, 15, 17, 19, 20, 22, 23]
GEANT_TEST  = [3, 10, 12, 14, 21]
GEANT_VAL   = []                                                # no validation set
N32_TEST    = [6, 41, 73, 108, 141]

DRAW_LINE   = True      # traffic-demand: connect ALL TMs (incl. unused) with a semi-transparent line

sns.set(style="whitegrid")
os.makedirs(OUTDIR, exist_ok=True)


def save(fig, name):
    """One figure, through the shared writer like every other generator.

    This used to be a private savefig loop writing png and svg at 500 dpi under
    a descriptive name. That is why the caption-named PDFs of Figures 6-12 had
    to be exported and renamed by hand: the generator could not produce them.
    """
    save_figure(fig, OUTDIR, name, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- GÉANT data -----------------------------
def geant_tm(day, hour, scale):
    """Parse one GÉANT IntraTM XML into a 24x24 demand matrix (scaled)."""
    fn = os.path.join(GEANT_XML, f"IntraTM-2005-{day}-{hour:02d}-00.xml")
    root = ET.parse(fn).getroot()
    tm = np.zeros((24, 24))
    for src in root[1]:
        for dst in src:
            tm[int(src.attrib["id"])][int(dst.attrib["id"])] = float(dst.text) * scale
    return tm


def geant_total_loads(scale):
    """24 hourly TMs of day 2005-01-03 (idx 0..23).  kbps."""
    return [geant_tm("01-03", h, scale).sum() / 100.0 for h in range(24)]


def parse_bw(path):
    """bw_r.txt: 'src,dst,_,capacity' per line -> (Graph, [(s,d,cap)])."""
    G, edges = nx.Graph(), []
    for line in open(path).read().strip().split("\n"):
        s, d, _, cap = map(float, line.split(","))
        G.add_edge(int(s), int(d), capacity=cap)
        edges.append((int(s), int(d), cap))
    return G, edges


def layout(G, pos_json, tweaks=None):
    """Load a saved tuned layout if present (keys json->int), else
    Kamada-Kawai + optional per-node manual tweaks (reproduces 學長 figs)."""
    if os.path.exists(pos_json):
        raw = json.load(open(pos_json))
        pos = {int(k): np.array(v, float) for k, v in raw.items()}
        if set(pos) >= set(G.nodes):
            return pos
    pos = nx.kamada_kawai_layout(G)
    for n, (dx, dy) in (tweaks or {}).items():
        if n in pos:
            pos[n] = (pos[n][0] + dx, pos[n][1] + dy)
    return pos


# ------------------------------- figures ------------------------------
def fig_traffic_demand(loads, train, test, val, name, xlabel="TM index", xticks=None):
    fig = plt.figure(figsize=(9, 5))
    if DRAW_LINE:
        plt.plot(range(len(loads)), loads, "-", color="gray", alpha=0.45, zorder=1)
    sc = dict(s=80, alpha=0.85, edgecolors="black", zorder=3)
    if val:
        plt.scatter(val, [loads[i] for i in val], color="tab:green", label="validation", **sc)
    plt.scatter(train, [loads[i] for i in train], color="tab:blue", label="training", **sc)
    plt.scatter(test, [loads[i] for i in test], color="tab:red", label="testing", **sc)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Total load (kbps)", fontsize=16)
    ax = plt.gca()
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(13)
    ax.tick_params(labelsize=13)
    plt.legend(prop={"size": 13}, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)                # above the axes -> never covers data
    plt.tight_layout()
    save(fig, name)


def fig_heatmap(mats, titles, name):
    """All sub-heatmaps share one color scale (vmin=0, common vmax) so the
    same colour means the same demand across TMs, with one shared colorbar."""
    mats = [np.array(m) for m in mats]
    vmax = max(m.max() for m in mats)
    n = mats[0].shape[0]
    step = max(1, n // 4)                            # ~4 sparse ticks, not every node
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(13, 7.2),
                             gridspec_kw={"wspace": 0.25, "hspace": 0.25})
    axes = axes.flatten()
    for i, (m, t) in enumerate(zip(mats, titles)):
        left   = (i % cols == 0)                     # left column -> src label
        bottom = (i // cols == rows - 1)             # bottom row  -> dst label
        sns.heatmap(m, cmap="inferno", vmin=0, vmax=vmax, cbar=False, square=True,
                    xticklabels=step, yticklabels=step, ax=axes[i])   # sparse ticks on every subplot
        axes[i].set_title(t, fontsize=15)
        axes[i].set_xlabel("dst" if bottom else "", fontsize=12)
        axes[i].set_ylabel("src" if left else "", fontsize=12)
        axes[i].tick_params(labelsize=10, length=4, width=1.2, direction="out",
                            bottom=True, left=True, color="black")        # short tick marks
    used = [axes[i] for i in range(len(mats))]
    for j in range(len(mats), rows * cols):
        fig.delaxes(axes[j])
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm, ax=used, shrink=0.85, label="traffic demand")
    save(fig, name)


def fig_topology(G, edges, pos, name, three_cap=True):
    fig = plt.figure(figsize=(11, 8))
    node_c = "dodgerblue"                            # bright blue nodes, white labels
    if three_cap:                                    # 100 black / 25 yellow / 1.55 red
        tier = {100: "black", 25: "orange"}
        colors = [tier.get(int(c), "red") for *_, c in edges]
        legend = [mpatches.Patch(color="black", label="100 Mbps"),
                  mpatches.Patch(color="orange", label="25 Mbps"),
                  mpatches.Patch(color="red", label="1.55 Mbps")]
    else:                                            # uniform 100 Mbps -> black
        colors = ["black"] * len(edges)
        legend = [mpatches.Patch(color="black", label="100 Mbps")]
    # edges first, nodes on top (white outline) for a clean look
    nx.draw_networkx_edges(G, pos, edgelist=[(s, d) for s, d, _ in edges],
                           edge_color=colors, width=2.2)
    nx.draw_networkx_nodes(G, pos, node_size=680, node_color=node_c,
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold")
    plt.legend(handles=legend, loc="upper right", prop={"size": 13}, framealpha=0.9)
    plt.axis("off")
    plt.tight_layout()
    save(fig, name)


def main():
    # ---- GÉANT ----
    loads = geant_total_loads(GEANT_SCALE)
    fig_traffic_demand(loads, GEANT_TRAIN, GEANT_TEST, GEANT_VAL, "Figure 8. GÉANT traffic demand",
                       xticks=list(range(0, len(loads) + 1, 4)))
    fig_heatmap([geant_tm("01-03", h, GEANT_SCALE) for h in GEANT_TEST],
                [f"TM {h}" for h in GEANT_TEST], "Figure 10. Traffic heatmap of the GÉANT test set")
    G, edges = parse_bw(GEANT_BW)
    fig_topology(G, edges, layout(G, GEANT_POS, {2: (0.05, 0.05)}),
                 "Figure 6. GÉANT topology", three_cap=True)

    # ---- 32-node (144 TM) ----
    _, tms = pickle.load(open(N32_PKL, "rb"))
    train_idx = pickle.load(open(N32_TRAIN, "rb"))
    loads32 = [np.array(tm).sum() for tm in tms]
    fig_traffic_demand(loads32, list(train_idx), N32_TEST, [], "Figure 9. 32-node traffic demand",
                       xticks=list(range(0, len(loads32) + 1, 24)))
    fig_heatmap([tms[i] for i in N32_TEST], [f"TM {i}" for i in N32_TEST], "Figure 11. Traffic heatmap of the 32-node test set")
    G32, edges32 = parse_bw(N32_BW)
    tweaks32 = {1: (0.10, 0), 2: (-0.25, 0.05), 7: (0, -0.10), 8: (0, 0.05),
                9: (-0.05, 0), 10: (0.10, 0), 11: (-0.05, 0.075), 13: (0.05, 0.03),
                15: (0, 0.20), 16: (-0.10, -0.10), 26: (0.10, 0), 28: (0, -0.15)}
    fig_topology(G32, edges32, layout(G32, N32_POS, tweaks32),
                 "Figure 7. 32-node topology", three_cap=False)

    print(f"\nAll figures written to {OUTDIR}")


if __name__ == "__main__":
    main()
