#!/usr/bin/env python
"""GPU inference latency of ONE full routing decision (§V-D).

The training loop never times inference on its own -- the per-step logs hold
step_time_sec (whole loop) and training_time_sec (update only), whose
difference (~0.22 s on 32-node) is env interaction + inference + logging
together. This script isolates the model call so the number quoted in §V-D
has a reproducible source.

WHAT IS TIMED
    StrideAgent.get_action(state) end to end, i.e. encoder_forward + the full
    M-step chain_rollout + the argmax/log-prob bookkeeping -- exactly what the
    controller calls once per monitoring period. CUDA is synchronised around
    every call, so the number is device time, not queue time.

WHY THE INPUT VALUES DO NOT MATTER
    Decoding is greedy with a fixed number of reverse steps and no
    data-dependent control flow, so latency depends on shapes (N pairs, K
    candidates, num_link, M) and not on the measured link state. We still
    replay a REAL archived net_info_directed.csv from the holdout test session
    so the shapes and the code path are the production ones.

WHAT IS REPORTED
    per-seed median and mean over REPEATS calls after WARMUP discarded calls,
    then the unweighted mean of the two per-seed medians (seeds 17, 18), which
    is the number to quote.

Run from anywhere; the script chdir's to the repo root because the config
holds repo-relative dataset paths.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/timing/inference/make_inference_bench.py

"""
import os
import sys

# ============================== CONFIG (edit here) ==========================
HERE = os.path.dirname(os.path.abspath(__file__))
# paper/figures/timing/<pipeline>/ -> repo root, four levels up.
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TBL_NAME = "timing_inference_bench"      # diagnostic table, not a thesis table

ENV, ALG, CTRL = "32node_144tm_directed", "stride", "simple_monitor"

# The variant the checkpoints were trained under. It must be set BEFORE
# `import config`, because config/algs/stride_config.py resolves
# STRIDE_VARIANT at import time. Seed is not part of the architecture, so one
# variant reconstructs both models.
VARIANT = "base"

# seed -> checkpoint dir written by save_model, one per training run
CKPTS = {
    17: "results/stride/runs/base_32node_s17_20260605_114040/train/model",
    18: "results/stride/runs/base_32node_s18_20260605_221156/train/model",
}

# A real link-state snapshot from the seed-17 holdout session (TM-06, step 15).
# get_mask_directed / get_state_directed both read
# ./results/<algs_name>/net_info_directed.csv, so we hand them a config copy
# whose algs_name points at a scratch directory -- the live results/stride/
# file is never touched.
SNAPSHOT_CSV = ("results/stride/runs/base_32node_s17_20260605_114040/test/"
                "20260605_121029/real/06/"
                "real_directed_link_data/06_15_net_info_directed.csv")
IO_ALIAS = "stride_infbench"             # scratch results/<alias>/ for the CSV

WARMUP = 20                              # discarded calls (CUDA/cuDNN autotune)
REPEATS = 200                            # timed calls per seed
# ===========================================================================

import shutil
from types import SimpleNamespace

import numpy as np
import torch

os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ["STRIDE_VARIANT"] = VARIANT          # must precede `import config`

import config as config_pkg                                    # noqa: E402
from algs import REGISTRY as algs_REGISTRY                     # noqa: E402
from loader.train_loader import (get_mask_directed,            # noqa: E402
                                 get_state_directed)


def build_config():
    env_cfg, alg_cfg, ctrl_cfg = config_pkg.get(ENV, ALG, CTRL)
    return {**env_cfg, **alg_cfg, **ctrl_cfg}


def build_state(cfg):
    """Replay the archived snapshot -> (masked state (N, num_link, 3), masks)."""
    io_cfg = dict(cfg)
    io_cfg["algs_name"] = IO_ALIAS
    dst_dir = os.path.join("results", IO_ALIAS)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(SNAPSHOT_CSV, os.path.join(dst_dir, "net_info_directed.csv"))
    mask, link_indices = get_mask_directed(io_cfg)
    state, mlu, _flat = get_state_directed(io_cfg, mask, link_indices)
    return state, mask, mlu


def build_agent(cfg, ckpt_dir):
    agent = algs_REGISTRY[cfg["algs_name"]](SimpleNamespace(**cfg))
    agent.load_model(ckpt_dir)
    if cfg.get("rnn", False):
        agent.hidden_states = agent.init_hidden(agent.actor, 1)
    return agent


def time_agent(agent, state):
    """-> (median_ms, mean_ms, p95_ms) over REPEATS synchronised get_action calls."""
    cuda = torch.cuda.is_available()
    for _ in range(WARMUP):
        agent.get_action([state])
    if cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        t0 = torch.cuda.Event(enable_timing=True) if cuda else None
        if cuda:
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            agent.get_action([state])
            t1.record()
            torch.cuda.synchronize()
            samples.append(t0.elapsed_time(t1))          # ms
        else:
            import time
            s = time.perf_counter()
            agent.get_action([state])
            samples.append((time.perf_counter() - s) * 1e3)
    a = np.array(samples)
    return float(np.median(a)), float(a.mean()), float(np.percentile(a, 95))


def write_md(cfg, rows, out_md, device, mlu):
    med = float(np.mean([r["median"] for r in rows]))
    L = ["> **GPU inference latency of one full routing decision "
         "(32-node, M = 8, 992 pairs, K = 20), seeds 17+18**", "",
         f"Timed call: `StrideAgent.get_action()` (encoder + full {cfg['iter_steps']}-step "
         f"chain rollout). Device: {device}. "
         f"{WARMUP} warmup calls discarded, {REPEATS} timed calls per seed, CUDA-synchronised. "
         f"Input is a real archived link-state snapshot "
         f"(`{os.path.basename(SNAPSHOT_CSV)}`, MLU = {mlu * 100:.1f}%); decoding is greedy with a "
         "fixed step count, so latency is shape-determined and not input-dependent.", "",
         "| seed | checkpoint | median (ms) | mean (ms) | p95 (ms) |",
         "| --: | :-- | --: | --: | --: |"]
    for r in rows:
        L.append(f"| {r['seed']} | `{r['ckpt']}` | {r['median']:.2f} | "
                 f"{r['mean']:.2f} | {r['p95']:.2f} |")
    L += [f"| **mean** | | **{med:.2f}** | | |", "",
          f"**Measured latency: ~{med:.1f} ms**", "",
          "Regenerate with `python make_inference_bench.py`."]
    md = "\n".join(L) + "\n"
    with open(out_md, "w") as f:
        f.write(md)
    print(f"\nsaved {out_md}\n")
    print(md)


def main():
    cfg = build_config()
    state, mask, mlu = build_state(cfg)
    device = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
              else "CPU (no CUDA)")
    print(f"device={device}  state={state.shape}  MLU={mlu * 100:.1f}%")

    rows = []
    for seed, ckpt in sorted(CKPTS.items()):
        agent = build_agent(cfg, ckpt)
        median, mean, p95 = time_agent(agent, state)
        rows.append(dict(seed=seed, ckpt=ckpt,
                         median=median, mean=mean, p95=p95))
        print(f"  seed{seed}: median={median:.2f} ms  "
              f"mean={mean:.2f} ms  p95={p95:.2f} ms")
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_md(cfg, rows, os.path.join(HERE, f"{TBL_NAME}.md"), device, mlu)


if __name__ == "__main__":
    main()
