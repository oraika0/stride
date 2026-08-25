# AGENTS.md

The single instruction file for coding agents working in this repository.
There is deliberately no per-tool variant of it — one file, every agent.
[`AGENTS.zh-TW.md`](AGENTS.zh-TW.md) is a translation for human readers and has
to be updated alongside this one.

## Ground rules

- This repository is code-and-experiment first. Prefer code changes, experiments,
  debugging and runtime investigation over documentation work.
- `results/` and `dataset/` hold experimental data and inputs. Do not modify or
  delete anything under them without being asked.
- `A-Traffic-Engineering-.../` is vendored upstream code. Only `SAC_PL_KP` (the gym
  environment) and `Enero_datasets` (traffic data) are used. Do not edit it.
- `paper/` regenerates manuscript figures and tables from
  `results/<alg>/runs/<run>/test/`.
  Changing a path there can silently produce an empty plot rather than an error —
  verify that globs still resolve after any move.
- There is no test suite. `diagnostics/` holds standalone methodology experiments,
  not unit tests, and nothing runs automatically.

## What this project is

STRIDE picks one path per OD pair from a frozen K=20 candidate set. All pairs are
decided jointly by a discrete-diffusion decoder (masked start, M denoise steps),
trained with actor–critic RL on a real Mininet + Ryu testbed.

**`algs/stride.py` is the mainline method.** Everything else in `algs/` is a
baseline for comparison. An earlier implementation named `maskgit_routenet` was
removed in 2026-08 along with its results; references to it in comments describe
where a piece of logic came from, not live code.

## Repo layout

```text
main.py test_single_tm.py
                    what you run (see "Running")
run_drl.py          what those two spawn in the agent process, in its own
                    terminal under the user account; never invoked by hand
test_sim_only.py test_single_tm_udp.py
                    unmaintained, off the paper path — see "Running"
algs/               agents; algs/__init__.py REGISTRY maps --alg to a class
config/             env/ algs/ controller/ — three independent layers
loader/             env_loader.py (topology, traffic), train_loader.py (train/eval loops)
utils/              Ryu app (simple_monitor.py, manager.py), iperf3 drivers, measurement
dataset/            topologies, TMs, candidate path sets + the scripts that build
                    them, prepare_dataset.py included
results/            experiment output — see "Results layout"
paper/              figure/table generators + LP/ILP bounds; paper/README.md is the index
docs/               methodology notes; docs/README.md is the index
scripts/            cleanup and dataset-setup shell helpers
diagnostics/        standalone experiments backing methodology claims
A-Traffic-.../      vendored Enero/RouteNet. Only SAC_PL_KP (gym env) and
                    Enero_datasets (traffic data) are used; nothing else is imported.
```

## Running

Set the interpreter once — the system `python3` has no torch and `sudo` does not
inherit conda activation:

```bash
PY="$HOME/miniconda3/envs/stride/bin/python"
```

Mode is decided by `sim_training` in the algorithm config, **not** by a CLI flag.
`main.py` treats an unset value as `False`, so almost everything runs real:
`ls2ic_nx` is the only algorithm with `sim_training=True`; every STRIDE variant
is `False`.

**The simulation-training path is unmaintained** — stale code, only `ls2ic_nx`
reaches it, and it has not been kept in step with the rest of the repo. Assume real
Mininet for everything.

**Simulation at evaluation time is unmaintained too.** `test_single_tm.py --auto`
used to finish by calling `test_sim_only.py`, giving every session a `sim/` half
beside `real/`; that call was removed 2026-08-19. The sim side had drifted to
covering a single TM, so the two halves stopped measuring the same thing, and no
figure or table ever read the sim half. `test_sim_only.py` and
`test_single_tm_udp.py` still run standalone — treat their output as unvalidated
and do not put it in a comparison.

```bash
# train — variant and seed are separate knobs; both default (base, 17) if unset
sudo -E "STRIDE_VARIANT=nodiff" "$PY" main.py \
    --env 32node_144tm_directed --alg stride --seed 18 train
#   -> results/stride/runs/nodiff_32node_s18_<date>_<time>/train/

# test — --model names the checkpoint, and the session lands inside that run
sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg stride --auto \
    --model results/stride/runs/nodiff_32node_s18_<date>_<time>/train/model
#   -> .../runs/nodiff_32node_s18_<date>_<time>/test/<date>_<time>/
```

**Without `--model` the test reads `results/<alg>/model`**, the live directory
every training run overwrites — so the checkpoint under test is whichever
training finished last. Pass `--model`.

**`sudo -E` alone does not reliably forward variables.** Use the
`sudo -E "VAR=value" "$PY" ...` form. A dropped `STRIDE_VARIANT` falls back to
`base` silently, so the run looks fine and trains the wrong thing.

A real run prints `Building topology ...` then `Controller spawned, wait 30 s ...`
and starts the controller and the agent, by default as two windows in a
detached tmux session (`tmux attach -t stride`). `--terminal gnome|inline`
picks a different backend; only gnome needs a DISPLAY and the sudo env
forwarding.

Cleanup — use `./scripts/clean.sh`, never `killall` + `mn -c`. Ryu runs as
a `python` process so `killall ryu-manager` never matches it, and a `main.py` or
`run_drl.py` stuck in `net.stop()` keeps holding the controller port. The next run
then attaches to a stale frozen controller and yields a plausible but worthless
curve. `clean.sh` kills the stuck orchestrator first, hard-gates on ports
6633/6653 being released, restores `results/` ownership, and clears `.drl_done`.

## Config system

`config/__init__.py` scans `config/{env,algs,controller}/*_config.py` at import.
Each file exports a dict named `config`; the basename minus `_config` is the CLI
key. `main.py` merges the three dicts before handing them to `run_drl.py`.

`config/algs/stride_config.py` is structured differently from the others: `_BASE`
is the mainline configuration the paper reports, and each `VARIANTS` entry lists
only what its ablation changes — one line each. `build_config()` reads
`STRIDE_VARIANT` (default `base`). An unknown variant name raises. The seed is
not a config knob: it is `--seed` on main.py / test_single_tm.py, honoured by
every algorithm, defaulting to 17.

A variant is an architecture, nothing more. Topology comes from `--env` and the
seed from `--seed`, so there is no `..._32node_seed18` variant: that
combination is a run, not a configuration.

Two test-time overrides are orthogonal to the variant, so one checkpoint can be
evaluated several ways: `STRIDE_EVAL_SAMPLE` (greedy vs sampled decoding) and
`STRIDE_ATTN_KERNEL`.

## Results layout

```text
results/<alg>/
├── runs/<variant>_<topology>_s<seed>_<date>_<time>/
│   ├── train/   model/ config.json output_*.txt step_time.txt ...
│   └── test/<date>_<time>[_sampled]/
│       ├── ckpt.txt          checkpoint path + sha256, decode, variant/topo/seed
│       └── real/<tm_id>/     Mininet measurements (a sim/ half used to sit
│                              beside it; deleted 2026-08-19)
└── Metrics/ model/ net_info*.csv drl_paths.json ...
           live scratch — the file-based channel between the Ryu controller and
           the agent while a run is in progress. Not an archive.
```

A run owns its tests, so a result carries its own provenance: the checkpoint is
one directory up from the session that was scored with it. One training run can
hold several tests (a re-run, a different decode); `ckpt.txt` records which
checkpoint each actually loaded, since a session may legitimately evaluate one
from elsewhere.

The run name is also the command that reproduces it: variant to `STRIDE_VARIANT`,
topology to `--env`, seed to `--seed`. The host is not in the name; it is in
the archived `config.json` alongside the start and finish times.

The controller writes link statistics into the scratch area and the agent reads
them; there is no socket between the two.

Paper figure scripts name the sessions they read outright, so it is visible from
the source which run each figure comes from.

### What is in git, and what to do with a new run

`results/*/runs/` is tracked. Every run behind the paper is in the repository --
training logs, test sessions, checkpoints -- so a clone can rebuild every figure
and re-evaluate every checkpoint without reproducing the experiments first. The
live scratch beside it (`results/<alg>/Metrics/`, `net_info*.csv`, ...) is not:
every run overwrites it, so which run it belongs to is unknowable.

New runs are **not** pushed by default. Nothing ignores them, so `git status`
shows each one as a single untracked directory line; leave them there unless the
run is worth publishing.

When one is, **commit the run directory whole**:

```bash
git add results/stride/runs/<name>          # the directory, not files inside it
```

Never a run's logs without its checkpoint. A result you cannot re-evaluate is not
evidence, and the two halves drifting apart is how an archive stops being one.
The cost is real -- a checkpoint is 12-32 MB and replacing it later leaves both
copies in history forever -- so decide once, per run, deliberately.

## Evaluation metrics

A test session writes four metric sets per TM under `real/<tm_id>/`:
`real_directed_test/` and `real_test/` (Mininet `tc` counters, per direction and
both-directions-summed) and `sim_directed_test/` / `sim_test/` (the same numbers
estimated offline with NetworkX from the same chosen paths).

**Read the directed ones.** Undirected aggregation sums a link's capacity across
both directions, so one saturated direction averages away. Everything in
`paper/figures/` reads `real_directed_test/<tm_id>_eval_metrics.csv`.

"NX" in these names means NetworkX — the offline estimator, a cross-check, never
a reported number. Older notes tell you to read `eval/*_NX_directed`; those were
wandb keys, and the function that produced them
(`quick_sim_eval_with_action`) was deleted with wandb. Read the CSVs.

Reference points, Geant `tm_scale=3`, directed: ILP ≈ 0.67, LP ≈ 0.67, OSPF ≈ 0.87
maximum link utilisation. A trained policy should land in that band; 1.0 means it
is worse than shortest path.

When comparing runs, never call a mean difference smaller than one baseline
standard deviation an improvement. Compute the baseline spread first.

## Environment

- conda env `stride`, Python 3.8.20, torch 2.0.1+cu118.
- Mininet lives in the **system** `dist-packages` and is bridged into the conda env
  by `site-packages/system_dist_packages.pth`. If the env is rebuilt, recreate that
  file or every real run dies at `from mininet.net import Mininet`.
- Ryu 4.34 needs a local patch to `ryu/topology/switches.py`, applied by
  `scripts/patch_ryu.py` (idempotent, `--check` verifies, `--revert` undoes). A
  missing patch does not raise — link delays just read 0.

- Real experiments need `sudo` and a kernel with `openvswitch`, `veth`, `sch_netem`.
- Killing a real run with `SIGKILL` leaves `results/` root-owned; a clean exit
  chowns it back.

## Couplings that are not visible locally

These are the traps where the code you are editing looks self-contained and is
not. None of them raise; all of them produce a plausible wrong number.

**Config layers merge as `{**env, **alg, **ctrl}` — but not everything reads the
merged dict.** `main.py` calls `init_paths(env_cfg, alg_cfg)` *before* it builds
`merged_cfg`, and `utils/init_path.py` reads `env_config["k_paths_file"]`
straight out of the unmerged env layer. So an env-layer key set from a STRIDE
variant wins in `merged_cfg` and is still ignored by `init_paths`. The switches
get flow rules from one candidate set while the model decides over another, with
no error. Env-layer keys belong in `config/env/`.

**`dataset/32node_traffic/k_paths.json` holds exactly 20 paths per pair, and its
order is frozen.** K=10/15 are prefixes of it, so they need no new file. K=25/30
cannot be — they use `k_paths_k30_ext.json`, whose first 20 entries are that file
verbatim (verified per pair) with 10 Yen extensions appended, and which is
selected by the separate `32node_144tm_directed_k30` env. Regenerating
`k_paths.json` with a larger K instead would reorder ties and silently redefine
what K=20 means, invalidating every run in the paper. Extend, never regenerate.

**A `VARIANTS` entry must pin anything it depends on, not inherit it.** Each
entry lists only its own diff, so a value it needs but does not name comes from
`_BASE` — and moving a `_BASE` default then changes that ablation's architecture
without touching its line. `flatfc_nomask` pins `encoder_pool: "flatten"` for
exactly this reason: it is the without-encoder ablation, and it used to rely on
`_BASE` defaulting to `flatten`.

**Checkpoints are shape-bound to `action_dim`.** A K=30 checkpoint cannot be
loaded by a K=20 config; the action head differs. Loading across them is a
loud failure, but *evaluating* a K=30 checkpoint on the plain env is quiet — the
env truncates to the canonical 20-path prefix and it degenerates to a K=20 run.

## Things that look wrong but are not

- **A run's reward curve says nothing about whether the network was real.** The
  reward follows the action, so it keeps moving even when the link measurements
  behind it have not changed for hours. Check `measurement.txt` in the run
  archive: `stale_seconds` is how long before training ended the controller last
  wrote a measurement, and anything past a few monitoring periods means the agent
  was reading a frozen file. A run found this way had 2988 of its 3009 steps on a
  snapshot ten hours old, with a checkpoint saved and a curve that looked fine.
  MLU pinned at 1.0 is a hint rather than proof -- a policy that never learns to
  relieve saturation looks the same.
- **A monitor cycle that reports no port stats stops updating the metrics files.**
  The controller keeps printing `[Statistics Module Ok]` and `[Flow Installation
  Ok]` -- both come before the check -- while `net_info*.csv` holds the previous
  cycle and the agent, which reads those files, trains against a network frozen in
  time. Look for the `[monitor] cycle N: 0/32 switches returned port stats`
  warning, and check that `results/<alg>/net_info_directed.csv` has a recent
  mtime. It surfaces on loaded machines: the replies are asynchronous, and before
  `PORT_STATS_WAIT` existed the only thing giving them time to arrive was
  flow installation happening to take a few seconds.
- **`kpath_init`, `kpath_reset`, `get_link_features`** are local additions to the
  vendored gym environment (`A-Traffic-.../SAC_PL_KP/gym-graph/.../environment16.py`),
  not upstream code. They are the per-OD K-candidate interface every algorithm
  goes through — `train_loader` calls `kpath_init` for all of them, and
  `algs/widest_path.py` calls `get_link_features` — so do not treat them as
  STRIDE-only. They were named `maskgit_*` until 2026-08-19, after the removed
  algorithm; the name was misleading in both halves.
- **`test_single_tm.py` vs `diagnostics/`** — the former is an experiment entry
  point, the latter holds standalone methodology experiments. Neither is a unit
  test suite; there is no pytest setup in this repo.
- **ELP / candidate-frame / transferable-head code in `stride.py`** is dormant.
  Those knobs default to off and the paper does not use them.

## Training logs

**Weights & Biases was removed on 2026-08-17.** No code imports it, no config
enables it, and it is not in `requirements.txt`. Every training log and every
manuscript figure now comes from `results/`. The remaining mentions are
docstrings explaining where historical data came from, plus the vendored
`A-Traffic-.../SAC_PL_KP` files, which are not ours and are never imported.

**The local logs were always the original record, and wandb the copy.**
`results/<alg>/runs/<run>/train/*.txt` is written directly by `train_loader`. Several
runs were never logged live and were replayed into wandb afterwards from exactly
those files, so a wandb copy was at best equal and sometimes worse — one lost two
thirds of its points to a sampled-history read and sat in a paper figure unnoticed
for weeks.

| file | contents |
| --- | --- |
| `output_all.txt` | full per-pair float vector per step; per-pair mean × 100 is the reward curve |
| `output_{bwd,delay,loss}.txt` | reward components, **sum over pairs rounded to an integer** — divide by the OD-pair count (geant 506, 32node 992); rounding caps agreement at ~0.006% |
| `output.txt`, `training_mlu.txt` | reward sum, MLU |
| `step_time.txt`, `training_time.txt` | per-step wall clock |
| `action_k_{entropy,topfrac,mode}.txt` | cross-pair action diversity; entropy ≈ 0 with topfrac ≈ 1 means every pair collapsed onto one candidate |
| `kdiv.txt`, `top1.txt`, `entropy_px0.txt` | per-denoise-step diagnostics, one list per line |
| `epsilon.txt` | only written when `epsilon_ini > 0`, so STRIDE's canonical runs have no such file |
| `<loss name>.txt` | one per key returned by `agents.update()` |

Everything below the reward rows was added 2026-08-17; runs older than that have
none of it, and their step timings survive only as the `source = wandb_cache`
rows in `paper/figures/timing/train_steps/timing_steps.csv`, which cannot be regenerated.

Line index `i` is training step `i+1` in every one of these files, for every
algorithm.

Provenance is structural now: a test session lives inside the run whose
checkpoint it scored, and `ckpt.txt` records the sha256 of what it actually
loaded. Before the `runs/` layout (2026-08-19) the association lived only in a
directory name, so for anything predating it, check `ckpt.txt` exists before
claiming provenance for a number. `paper/figures/reward/make_curves_csv.py::RUN_MAP`
maps the training runs behind the reward curves.

## Before claiming something works

Several failure modes in this project are silent:

- a missing Ryu patch makes every link delay read 0
- a dropped `STRIDE_VARIANT` falls back to `base` (`--seed` is an argument, so
  it cannot be dropped the same way; STRIDE_SEED is gone and raises if still set)
- a test without `--model` scores whichever checkpoint training last left in
  `results/<alg>/model`
- an unresolved figure glob yields an empty plot
- undirected metrics hide per-direction link saturation

Check the actual output, not just the exit code.
