#!/usr/bin/env python
"""Rebuild train_curves.csv and components.csv from LOCAL training logs.

No network, no wandb, no credentials.

Why this exists
---------------
The reward figures used to pull their numbers from W&B and cache them in CSVs.
That was backwards: for these runs the local `results/<alg>/train/<dir>/output_*.txt`
files are the ORIGINAL record and W&B was always a copy, in some cases a copy of
a copy. The `ls2ic` family had no live W&B logging at all until 2026-05-26 —
those runs were replayed into W&B afterwards from exactly the files read here.

One of those W&B copies was also lossy. `train/reward` for the geant LS2IC
seed-17 curve survived as 999 of ~3000 points, because the directed-MLU
correction pass re-read its source run through a sampled history endpoint. The
local log never lost anything.

Data layout
-----------
`output.txt`      one line per training step, REWARD_SCALE * the sum over pairs.
                  Divided by num_agents this is the per-pair mean * 100 that
                  reproduces W&B `train/reward`, and it is what the curves read.
`output_all.txt`  the same thing before summing: one line per step, a Python
                  list of per-pair rewards. 48 MB per run against output.txt's
                  55 KB, and identical to 1e-12 once averaged, so the figures do
                  not read it. Keep it -- it is the only per-pair record there is.
`output_bwd.txt`  one line per training step, the SUM over pairs of that reward
`output_delay.txt`  component, written as an integer. Divide by num_agents for
`output_loss.txt`   the per-pair value. The rounding caps agreement with the
                  W&B copy at ~0.006% relative — invisible after the /100 the
                  figures apply, but it is not exact, unlike output_all.txt.

Step numbering
--------------
`train_loader.save_stepwise_log` opens with mode 'w' when step == 1, so the
first line of every file is step 1 and line index i (0-based) is step i+1.
That holds for every algorithm.

The post-hoc uploader (`tools/upload_senior_logs_to_wandb.py`, since removed)
logged with `for step in range(max_len)` — 0-based. So the `ls2ic` and
`ps_dqn_dd` curves as they existed in W&B sat one step to the LEFT of the
live-logged `stride` curves. Reading everything locally with the i+1 rule
removes that skew; there is no per-family special case.

Provenance
----------
run_id keys below are the W&B ids the figures have always used, kept as opaque
labels so nothing downstream has to change. The local directory each one maps
to was established two independent ways that agree: by content (matching the
reward series against all 121 local run dirs, best match ~1e-13, unambiguous)
and by the RUNS table that used to live in tools/upload_senior_logs_to_wandb.py.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/reward/make_curves_csv.py

"""
import os

# ============================== CONFIG (edit here) ==========================
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Every curve is cut here, so all of them cover the same training budget.
#
# The 32-node env used to ask for 3010 steps -- 35 per traffic matrix across its
# 86 training matrices, one full cycle -- while STRIDE's own config pinned 3000,
# and the algorithm wins the merge. STRIDE therefore trained for 3000 steps on
# 32-node and every baseline for 3010, which showed up in the figure as the
# STRIDE line stopping ten steps before the others. The env is 3000 for everyone
# now, so runs from here match; this cut is what makes the archived ones match
# too, and it costs the baselines ten steps they had and STRIDE never did.
#
# GÉANT was always 3000 on both sides and is unaffected.
STEP_BUDGET = 3000

# run_id -> (topology, local run dir relative to REPO, num_agents)
# Keys are <method>_<topology>_s<seed>. They used to be W&B run ids -- eight
# random characters that named nothing, so reading RUNS in make_reward_fig.py
# meant looking every one of them up here to find out what was being plotted.
# The local run directory on each line is the record that matters; W&B was only
# ever a mirror of it.
# num_agents = number of OD pairs: geant 23 nodes -> 506, 32node -> 992.
# 2026-08-18: the 32node LS2IC seed-17 config was trained twice with identical
# settings (06-04 and 06-27). Replaying the archived per-step link state through
# both checkpoints and comparing against the routing decisions recorded in
# results/ls2ic_dd/test/20260627_161008_ls2ic_dd_32node_s17_pc0/real/06/drl_paths_snapshots/
# identified the 06-27 run: 992/992 pairs matched at steps 5 and 15, versus ~31%
# for 06-04. This entry therefore reads 06-27, the same run the Fig 17 bar is
# scored from.
RUN_MAP = {
    # --- geant ---
    "stride_geant_s17":    ("geant",  "results/stride/runs/base_geant_s17_20260603_214500/train", 506),
    "stride_geant_s18":    ("geant",  "results/stride/runs/base_geant_s18_20260611_114456/train", 506),
    "ls2ic_geant_s17":     ("geant",  "results/ls2ic_dd/runs/ls2ic_dd_geant_s17_20260524_125905/train", 506),
    "ls2ic_geant_s18":     ("geant",  "results/ls2ic_dd/runs/ls2ic_dd_geant_s18_20260630_020836/train", 506),
    "madqn_geant_s17":     ("geant",  "results/ps_dqn_dd/runs/ps_dqn_dd_geant_s17_20260618_235521/train", 506),
    "madqn_geant_s18":     ("geant",  "results/ps_dqn_dd/runs/ps_dqn_dd_geant_s18_20260630_020924/train", 506),
    # --- 32node ---
    "stride_32node_s17":   ("32node", "results/stride/runs/base_32node_s17_20260605_114040/train", 992),
    "stride_32node_s18":   ("32node", "results/stride/runs/base_32node_s18_20260605_221156/train", 992),
    "ls2ic_32node_s17":    ("32node", "results/ls2ic_dd/runs/ls2ic_dd_32node_s17_20260627_113519/train", 992),
    "ls2ic_32node_s18":    ("32node", "results/ls2ic_dd/runs/ls2ic_dd_32node_s18_20260611_122711/train", 992),
    "madqn_32node_s17":    ("32node", "results/ps_dqn_dd/runs/ps_dqn_dd_32node_s17_20260618_233958/train", 992),
    "madqn_32node_s18":    ("32node", "results/ps_dqn_dd/runs/ps_dqn_dd_32node_s18_20260630_021144/train", 992),
    "stride_32node_v100":   ("32node", "results/stride/runs/base_32node_s17_20260824_063712/train", 992),
}

REWARD_SCALE = 100.0   # per-pair mean -> the [0,300] scale the figures divide by
COMPONENTS = {"reward_bwd": "output_bwd.txt",
              "reward_delay": "output_delay.txt",
              "reward_loss": "output_loss.txt"}
# ===========================================================================

import pandas as pd


def read_reward(run_dir, num_agents):
    """output.txt -> {step: per-pair mean * REWARD_SCALE}.

    output.txt holds REWARD_SCALE times the sum over pairs, so dividing by
    num_agents gives the per-pair mean * REWARD_SCALE these figures plot -- the
    same number output_all.txt gives. Checked over all twelve runs in RUN_MAP
    and every one of their ~36000 steps: the two agree to 1e-12, which is the
    order of a float summation reassociating, not a difference.

    That matters because output_all.txt carries one value per OD pair per step,
    992 of them on 32-node, and runs to 48 MB per run. Reading it to recover a
    per-step mean costs 576 MB and forty seconds; this file is 55 KB.

    Warmup steps, which output_all.txt writes as '[]', are 0.0 here, so they are
    skipped as a leading run of zeros rather than as empty lists -- dropped, not
    plotted as zero, so every curve begins at step 3 where the first reward
    exists. Zero occurs at
    steps 1 and 2 and nowhere else in any of the twelve runs -- a real reward of
    exactly 0.0 mid-run would be dropped, and has never happened.
    """
    path = os.path.join(REPO, run_dir, "output.txt")
    out = {}
    warmup = True
    with open(path) as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            try:
                v = float(ln)
            except ValueError:
                continue
            if warmup and v == 0.0:
                continue
            warmup = False
            step = i + 1
            if step > STEP_BUDGET:
                break
            out[step] = v / num_agents
    return out


def read_component(run_dir, fname, num_agents):
    """output_{bwd,delay,loss}.txt -> {step: sum-over-pairs / num_agents}."""
    path = os.path.join(REPO, run_dir, fname)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            if i + 1 > STEP_BUDGET:
                break
            try:
                out[i + 1] = float(ln) / num_agents
            except ValueError:
                continue
    return out


def build_train_curves():
    rows = []
    for rid, (topo, run_dir, na) in RUN_MAP.items():
        pts = read_reward(run_dir, na)
        rows += [{"topo": topo, "run_id": rid, "step": s, "reward": r}
                 for s, r in sorted(pts.items())]
        print(f"  {topo:7s} {rid:18s} {len(pts):5d} pts  <- {run_dir}")
    return pd.DataFrame(rows)


def build_components():
    rows = []
    for rid, (topo, run_dir, na) in RUN_MAP.items():
        total = read_reward(run_dir, na)
        comps = {k: read_component(run_dir, f, na) for k, f in COMPONENTS.items()}
        if not all(comps.values()):
            print(f"  {topo:7s} {rid:18s} skipped (no component logs)")
            continue
        steps = sorted(set(total) & set.intersection(*(set(c) for c in comps.values())))
        for s in steps:
            row = {"topo": topo, "run_id": rid, "step": s, "reward": total[s]}
            row.update({k: comps[k][s] for k in COMPONENTS})
            rows.append(row)
        print(f"  {topo:7s} {rid:18s} {len(steps):5d} pts")
    return pd.DataFrame(rows)


def main():
    print("train_curves.csv:")
    tc = build_train_curves()
    tc_path = os.path.join(HERE, "train_curves.csv")
    tc.to_csv(tc_path, index=False)
    print(f"wrote {tc_path}  ({len(tc)} rows)\n")

    print("components.csv:")
    cp = build_components()
    cp_path = os.path.join(HERE, "components.csv")
    cp.to_csv(cp_path, index=False)
    print(f"wrote {cp_path}  ({len(cp)} rows)")


if __name__ == "__main__":
    main()
