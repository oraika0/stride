"""Per-step TRAINING time quoted in the paper (§V-D update cost, §V-G M ladder).

Reads `timing_steps.csv` next to this file. No network, no wandb.

WHERE THE NUMBERS COME FROM
    `step_time_sec` and `training_time_sec` are per-step wall clock, printed by
    loader/train_loader.py once per training step.

    Runs finished BEFORE 2026-08-17 only ever sent them to W&B — the training
    loop printed them to the agent terminal and wrote nothing to disk, and that
    terminal was never captured. `timing_steps.csv` is a one-time dump of those
    12 runs, pulled while the W&B project was still reachable. Those rows carry
    `source = wandb_cache` and CANNOT be regenerated; treat the file as data.

    Runs finished AFTER that date write `step_time.txt` and `training_time.txt`
    into their own `results/<alg>/train/<dir>/`. Add such a run to
    `make_timing_csv.py::LOCAL_RUNS` and re-run it to append the rows here.

WHAT THE TWO SERIES MEAN  (loader/train_loader.py)
    step_time_sec      time_end - time_in, i.e. the WHOLE while-loop iteration:
                       env interaction + inference + update (it does NOT include
                       the idle wait for the next monitoring period). Every step.
    training_time_sec  duration of agents.update() ONLY. The guard is
                           if len(agents.memory) > agents.batch_size:
                       so warmup steps record an explicit 0.0 rather than a
                       missing value — 66 steps in every run. They MUST be
                       excluded from a "per-step update cost" mean or the
                       average is dragged down by ~2%.

AGGREGATION (stated so the paper number is reproducible)
    1. per run: mean over steps. step_time uses every step; update uses only
       steps where training_time_sec > 0.
    2. seed average = unweighted mean of the two per-run means. Both runs log
       the same number of steps in most cells, so this equals pooling; where a
       run logged slightly fewer steps (M=6 s17, M=10 s17) the difference is
       below the reported precision.

The seed-18 sessions are the ones the figure scripts treat as canonical --
keep the CSV in sync with denoise_step/make_denoise_step_fig.py::SESSIONS.



Run:

    cd ~/stride
    conda activate stride
    python paper/figures/timing/train_steps/make_timing_table.py

"""
import os
import sys

# ============================== CONFIG (edit here) ==========================
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "timing_steps.csv")
TBL_NAME = "timing_train_steps"          # diagnostic table, not a thesis table

# Row order in the table.
#
# host is not provenance only, and the ladder is not machine-balanced. It reads:
# M=4 pc1+pc1, M=6 pc0+pc2, M=8 pc1+pc2, M=10 pc1+pc1, M=12 pc1+pc2. M=6 is the
# only rung with a pc0 run and the only one without a pc1 run, and pc0 is the
# slow machine: GÉANT M=8 is the one configuration measured on two hosts, and it
# comes out 2.783 s on pc2 against 3.182 s on pc0, 14.3% apart.
#
# That is exactly the rung that sits off the line. Fitting the other four gives
# 0.378 s per denoise step at R^2 = 0.9994 and predicts 3.49 s at M=6 against
# the 3.69 s measured -- and a rung whose mean is half a pc0 run should sit
# about 7% high, which is 3.73 s. The deviation is the host, not M.
#
# The linear claim survives this; it is cleaner without the confound than with
# it. What does not survive is reading the per-rung numbers as if one machine
# had produced them.
LABEL_ORDER = ["GÉANT M=8", "32node M=4", "32node M=6",
               "32node M=8", "32node M=10", "32node M=12"]
# ===========================================================================

import numpy as np
import pandas as pd


def collect():
    """-> {label: (per_run rows, (step_mean, update_mean, other_mean))}."""
    if not os.path.exists(CSV_PATH):
        raise SystemExit(
            f"{CSV_PATH} is missing. It holds the only surviving copy of the "
            f"pre-2026-08 timing history — restore it from version control "
            f"rather than trying to regenerate it.")
    df = pd.read_csv(CSV_PATH)
    out = {}
    for label in [l for l in LABEL_ORDER if l in set(df.label)]:
        rows = []
        for (seed, rid, host), g in sorted(
                df[df.label == label].groupby(["seed", "run_id", "host"]),
                key=lambda kv: kv[0][0]):
            st = g.step_time_sec.to_numpy(float)
            tt = g.training_time_sec.to_numpy(float)
            fired = tt > 0                       # update() actually ran
            rows.append(dict(seed=seed, run=rid, host=host, n=len(st),
                             warm=int((~fired).sum()), step=st.mean(),
                             update=tt[fired].mean(), other=(st - tt).mean()))
            print(f"  {label:<12} seed{seed} {rid} {host}  n={rows[-1]['n']} "
                  f"warm={rows[-1]['warm']}  step={rows[-1]['step']:.3f} "
                  f"update={rows[-1]['update']:.3f}")
        mean = tuple(float(np.mean([r[k] for r in rows]))
                     for k in ("step", "update", "other"))
        out[label] = (rows, mean)
    return out


def write_md(data, out_md):
    L = ["> **Per-step training time (`step_time_sec` and "
         "`training_time_sec`, seeds 17+18)**", "",
         "`step_time` = whole loop iteration (env + inference + update), every step. "
         "`update` = `agents.update()` only, averaged over steps where it fired "
         "(the first 66 warmup steps log 0.0 and are excluded). "
         "`env+inference` = `step_time - update`. "
         "Seed mean = unweighted mean of the two per-run means.", "",
         "| config | seed | run | host | steps | warmup | step_time (s) | update (s) | env+inference (s) |",
         "| :-- | --: | :-- | :-- | --: | --: | --: | --: | --: |"]
    for label, (rows, mean) in data.items():
        for r in rows:
            L.append(f"| {label} | {r['seed']} | `{r['run']}` | {r['host']} | "
                     f"{r['n']} | {r['warm']} | {r['step']:.3f} | {r['update']:.3f} | "
                     f"{r['other']:.3f} |")
        L.append(f"| **{label}** | **mean** | | | | | **{mean[0]:.2f}** | "
                 f"**{mean[1]:.2f}** | **{mean[2]:.3f}** |")
    md = "\n".join(L) + "\n"
    with open(out_md, "w") as f:
        f.write(md)
    print(f"\nsaved {out_md}\n")
    print(md)


def _fit(Ms, ys):
    slope, icept = np.polyfit(Ms, ys, 1)
    r2 = 1 - ((ys - (slope * Ms + icept)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    return slope, icept, r2


def report_paper_numbers(data):
    """Print what the manuscript quotes. Deliberately not written into the .md.

    The table is measurements. Which section quotes which of them, and a fit
    through five of them, are conclusions drawn from the table -- keeping them
    out of the file means a diff on it is a change in what was measured.

    total/M is not the linearity check: it falls with M whatever happens,
    because of the M-independent fixed cost. The marginal cost per denoise step
    is the slope.
    """
    ladder = sorted((int(lb.split("M=")[1]), mn[0])
                    for lb, (_, mn) in data.items() if lb.startswith("32node"))
    Ms = np.array([m for m, _ in ladder], float)
    ys = np.array([v for _, v in ladder], float)

    g8, n8 = data["GÉANT M=8"][1], data["32node M=8"][1]
    print("numbers the manuscript quotes")
    print(f"  V-D  per-step cost   {g8[0]:.2f} s GÉANT / {n8[0]:.2f} s 32-node, M=8")
    print(f"  V-D  update only     {g8[1]:.2f} s GÉANT / {n8[1]:.2f} s 32-node; "
          f"the other {n8[2] * 1000:.0f} ms on 32-node is env interaction, state "
          f"assembly and logging")
    print("  V-G  M ladder        " + " / ".join(f"M={m} {v:.2f}s" for m, v in ladder))

    slope, icept, r2 = _fit(Ms, ys)
    print(f"  V-G  fit, all rungs  {slope:.3f} s per denoise step, "
          f"intercept {icept:.2f} s, R2 = {r2:.4f}")
    keep = Ms != 6
    slope6, icept6, r26 = _fit(Ms[keep], ys[keep])
    print(f"       without M=6     {slope6:.3f} s per step, intercept {icept6:.2f} s, "
          f"R2 = {r26:.4f}")
    print(f"       M=6 is the only rung averaging a pc0 run and sits "
          f"{ys[1] - (slope6 * 6 + icept6):+.2f} s off that line -- see the note "
          f"above LABEL_ORDER. The linear claim holds either way.")


def main():
    # Refresh timing_steps.csv first, so there is no order to remember. This is
    # safe in a way rebuilding train_curves.csv is not: the collector appends and
    # refreshes, it never rebuilds. With LOCAL_RUNS empty it returns before
    # touching the file at all, and when it is not empty it drops only rows whose
    # run_id it is about to rewrite from disk. Those ids are run directory names;
    # the source=wandb_cache rows are keyed by W&B ids and cannot collide with
    # them, which is what keeps the twelve unrebuildable runs out of reach.
    sys.path.insert(0, HERE)
    import make_timing_csv
    make_timing_csv.main()
    print()

    data = collect()
    write_md(data, os.path.join(HERE, f"{TBL_NAME}.md"))
    report_paper_numbers(data)


if __name__ == "__main__":
    main()
