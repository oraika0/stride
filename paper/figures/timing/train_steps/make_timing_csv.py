#!/usr/bin/env python
"""Append per-step timing from local training logs into `timing_steps.csv`.

No network.

Two kinds of row live in that CSV and they are not equally replaceable:

    source = wandb_cache   The 12 runs the paper quotes, finished before
                           2026-08-17. Back then `step_time` was printed to the
                           agent terminal and sent to W&B, and nothing was
                           written to disk — the terminal was never captured.
                           These rows were dumped out of W&B once, while the
                           project was still reachable. They CANNOT be rebuilt.
                           This script never touches them.

    source = local         Runs finished after 2026-08-17, read back from
                           `step_time.txt` / `training_time.txt` in the run
                           directory. Add the run to LOCAL_RUNS below and re-run
                           this script; rows are matched on run_id, so re-running
                           refreshes a run in place instead of duplicating it.

Line index i in those files is training step i+1, matching every other log in
the run directory. `training_time.txt` records 0.0 on warmup steps where
`agents.update()` did not fire — those rows are kept here and filtered by the
reader, so the warmup count stays visible in the table.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/timing/train_steps/make_timing_csv.py

"""
import json
import os

# ============================== CONFIG (edit here) ==========================
HERE = os.path.dirname(os.path.abspath(__file__))
# paper/figures/timing/<pipeline>/ -> repo root, four levels up.
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CSV_PATH = os.path.join(HERE, "timing_steps.csv")

# label -> {seed: train dir relative to REPO}
# `label` groups the seeds into one table row; keep it in the same vocabulary as
# the existing rows ("GÉANT M=8", "32node M=4", ...) or the table gains a line.
# The CSV's run_id/host columns are filled from the run directory name and the
# archived config.json, so a run is identified the same way everywhere.
LOCAL_RUNS = {
    # "32node M=8": {
    #     17: "results/stride/runs/base_32node_s17_<date>_<time>/train",
    # },
}
# ===========================================================================

import pandas as pd


def read_series(run_dir, fname):
    path = os.path.join(REPO, run_dir, fname)
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing.\n"
            f"Runs finished before 2026-08-17 never wrote it; their timing "
            f"exists only as source=wandb_cache rows already in the CSV.")
    out = []
    with open(path) as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if ln:
                out.append((i + 1, float(ln)))
    return dict(out)


def run_identity(train_dir):
    """(run_id, host) for a run directory: its name, and the host it ran on.

    The run directory name is the identity now -- it already carries variant,
    topology, seed and start time. `_host` is written into the archived
    config.json at the end of the run; older archives may not have it.
    """
    run_dir = os.path.dirname(os.path.normpath(os.path.join(REPO, train_dir)))
    host = "?"
    try:
        with open(os.path.join(REPO, train_dir, "config.json")) as f:
            host = json.load(f).get("_host", "?")
    except OSError:
        pass
    return os.path.basename(run_dir), host


def main():
    base = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
    if not LOCAL_RUNS:
        n = len(base)
        by = base.groupby("source").run_id.nunique().to_dict() if n else {}
        print(f"LOCAL_RUNS is empty — nothing to append.\n"
              f"{CSV_PATH}: {n} rows, runs by source: {by}")
        return

    rows = []
    for label, seeds in LOCAL_RUNS.items():
        for seed, run_dir in sorted(seeds.items()):
            rid, host = run_identity(run_dir)
            st = read_series(run_dir, "step_time.txt")
            tt = read_series(run_dir, "training_time.txt")
            steps = sorted(set(st) & set(tt))
            rows += [{"label": label, "seed": seed, "run_id": rid, "host": host,
                      "step": s, "step_time_sec": st[s],
                      "training_time_sec": tt[s], "source": "local"}
                     for s in steps]
            print(f"  {label:<12} seed{seed} {rid} {host}  {len(steps)} steps  <- {run_dir}")

    new = pd.DataFrame(rows)
    if len(base):
        base = base[~base.run_id.isin(set(new.run_id))]      # refresh in place
    out = pd.concat([base, new], ignore_index=True)
    out.to_csv(CSV_PATH, index=False)
    print(f"wrote {CSV_PATH}  ({len(out)} rows, {out.run_id.nunique()} runs)")


if __name__ == "__main__":
    main()
