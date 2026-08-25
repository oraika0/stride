# Quick start

> 中文版見 [QUICKSTART.zh-TW.md](QUICKSTART.zh-TW.md)。

Commands only, install included. Every step says in a line or two what it does
and what it leaves behind. For why any of it is shaped this way, see
[`README.md`](README.md).

A fresh clone can run all of it. The run archive under `results/*/runs/` is in the
repository -- training logs, test sessions and checkpoints -- so the figures and
tables rebuild without having to reproduce the experiments first.

> **1-8 are once per machine; 9-12 are the loop you repeat per experiment.**
> Steps 2 and 3 are alternatives -- pick the one matching your Ubuntu. Full
> explanations, and what lands where on a shared server, are in
> [`README.md` §2](README.md).

## 1. Get the repository

**Run the whole thing inside tmux.** An SSH drop then does not kill the install,
scrollback survives, and when output needs to go to someone else -- or to an AI --
it can be dumped to a file rather than copied out of a terminal by hand:

```bash
tmux new -s setup
```

`Ctrl-b d` detaches, `tmux attach -t setup` comes back. To save a pane's output:

```bash
tmux capture-pane -p -t setup -S -5000 > /tmp/setup.log
```

`-S -5000` reaches five thousand lines back. **This is the command to reach for
whenever a step below goes wrong.**


```bash
git clone https://github.com/oraika0/stride.git ~/stride
cd ~/stride
```

Commands here and throughout this file assume the repository root as the working
directory. Only the Mininet source install leaves it, and that one walks back by
itself. `apt`, `conda` and `ln -s` do not care where you are; everything else does.

## 2. Mininet + Open vSwitch (Ubuntu 24.04 and later)

Distribution packages; do
not run Mininet's `install.sh` here. **On a shared machine, look first** -- if both
are present, skip these two lines: `apt install` will also upgrade an
already-installed Open vSwitch and restart the service, cutting off anyone's
running experiment.

```bash
dpkg -l mininet openvswitch-switch 2>/dev/null | grep ^ii   # both listed = already there, skip
sudo apt install mininet openvswitch-switch
sudo systemctl enable --now openvswitch-switch
```

## 3. Mininet + Open vSwitch (Ubuntu 20.04)

From source. `-s` puts the dependency
clones in `src/` alongside it rather than loose in `$HOME`, and has to come
before `-n` and `-v`.

```bash
git clone https://github.com/mininet/mininet src/mininet
SRC="$PWD/src"
(cd src/mininet/util && ./install.sh -s "$SRC" -nv)    # 20-40 min
```

The parentheses are deliberate: they keep the `cd` inside a subshell, so you are
still at the repository root afterwards. `src/` is gitignored, so none of this
gets committed.

## 4. Python environment

Both routes.

```bash
conda create -n stride python=3.8 -y
conda activate stride
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 5. Let the env see Mininet

apt installed it for the system Python 3.12, and a
source install put it in the system `dist-packages` — either way the conda
interpreter cannot see it. Pick the line matching your route:

```bash
# 24.04 (apt): link the one package in
ln -sfn /usr/lib/python3/dist-packages/mininet "$CONDA_PREFIX/lib/python3.8/site-packages/mininet"

# 20.04 (source): put the system path on sys.path
echo /usr/local/lib/python3.8/dist-packages > "$CONDA_PREFIX/lib/python3.8/site-packages/system_dist_packages.pth"
```

## 6. Ryu, and the delay patch

**Install from git, not `pip install ryu`.** The last
PyPI release (4.34, 2020) predates eventlet 0.30.3's breaking change, so `import
ryu` dies on `ALREADY_HANDLED`. Upstream master fixed it and never released again.

```bash
git clone https://github.com/faucetsdn/ryu src/ryu
pip install "setuptools==58.0.0" wheel
pip install ./src/ryu --no-build-isolation
python scripts/patch_ryu.py
```

The setuptools pin is not optional either: ryu's `setup.py` calls
`easy_install.get_script_args`, removed in setuptools 58. `--no-build-isolation`
is what makes the build use that pinned copy instead of a fresh latest one pip
fetches for itself.

The delay patch is what makes per-link latency measurable; **without it every link
delay reads 0 and nothing raises.**

## 7. Build the traffic scripts

Turns the pickled traffic matrices into the
per-host iperf3 scripts Mininet replays, under
`dataset/<topo>_traffic/<tm dir>/TM-<id>/{Clients,Servers}/`. Skips what exists.

```bash
python dataset/prepare_dataset.py --topology 32node --tms 144tm
python dataset/prepare_dataset.py --topology geant  --tms 24tm --tm_scale 3
```

## 8. Check

All three must pass.

```bash
python -c "import torch, mininet; print(torch.__version__, torch.cuda.is_available())"
python scripts/patch_ryu.py --check
sudo mn --test pingall
```

The first must print a torch version and `True`, the second `PATCHED`. **Do not
start training if the second fails** — without the patch every link delay reads 0
and nothing tells you.

## 9. Train

**Give it a named window in the session from step 1.** The three processes then
take a window each and the shell you started from stays free to look things up:

```bash
PY="$HOME/miniconda3/envs/stride/bin/python"
tmux new-window -n main "sudo -E $PY main.py --env 32node_144tm_directed --alg stride train; exec bash"
```

The window is called `main`, and the controller and the agent appear beside it.
The trailing `exec bash` keeps it open when training ends or dies, so the error
stays readable.

If tying up the current shell does not matter, run it directly -- the window is
just named `bash` then:

```bash
sudo -E "$PY" main.py --env 32node_144tm_directed --alg stride train
```

`sudo -E` carries `$TMUX` through, and the launcher, seeing it, opens the
controller and the agent as windows of **this** session instead of making a root
one. All three are then reachable with a plain `tmux attach`; no `sudo tmux`.

**Outside tmux**, the orchestrator stays where it is and the two children go into
a detached **root** session, which needs `sudo tmux attach -t stride` to reach --
it belongs to root, whose socket is not yours (`/tmp/tmux-0/` vs
`/tmp/tmux-1000/`).

The absolute path in `$PY` is required and `python` will not do: `sudo` replaces
`PATH` with its own `secure_path`, so a bare `python` under it is the system
Python, which has no torch.

Builds the topology, spawns the Ryu controller and the agent, then runs 3000
monitoring periods (~8 h at 10 s each). A healthy start prints
`Building topology ...` then `Controller spawned, wait 30 s ...`.

### Working inside tmux

```
Ctrl-b d      detach (training keeps running, nothing is interrupted)
Ctrl-b n / p  next / previous window
Ctrl-b w      pick from a list
Ctrl-b [      scroll back (q to leave)
```

The windows are named `controller` and `drl`. A window stays after its process
exits (`remain-on-exit`), so **a crash stays on screen** instead of vanishing with
the terminal -- the status line reads `Pane is dead (status N)`. Scroll the
traceback with `Ctrl-b [`, then let step 10's `clean.sh` take the session away.

To save a window's output to a file, for someone else or for an AI to read:

```bash
sudo tmux capture-pane -p -t stride:drl -S -5000 > /tmp/drl.log
```

Without tmux, `--terminal gnome` opens a GUI window each and `--terminal inline`
interleaves both into the current terminal.

### Output

```
results/stride/runs/base_32node_s17_<date>_<time>/train/
├── model/            the checkpoint
├── config.json       every setting, plus host and start/finish times
├── output_*.txt      reward and its components, one line per step
└── step_time.txt     wall clock per step
```

### Variations

```bash
# a different architecture (see README §5 for the list)
sudo -E "STRIDE_VARIANT=nodiff" "$PY" main.py --env 32node_144tm_directed --alg stride train

# a second seed, for a cross-seed error bar
sudo -E "$PY" main.py --env 32node_144tm_directed --alg stride --seed 18 train

# a baseline instead
sudo -E "$PY" main.py --env 32node_144tm_directed --alg ls2ic_dd train
```

`STRIDE_VARIANT` has to be an environment variable and has to be written as an
explicit assignment — `sudo -E` alone drops it, and a dropped value silently
trains the default. `--seed` is an ordinary argument and cannot be dropped.

## 10. Clean up between runs

```bash
./scripts/clean.sh
```

Run this after every run, and after any crash. `killall` + `mn -c` by hand
misses the stuck orchestrator and does not verify the controller port was
released — the next run then attaches to a stale frozen controller and produces a
plausible but worthless curve, with no error anywhere.

## 11. Test

```bash
PY="$HOME/miniconda3/envs/stride/bin/python"
sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg stride --auto \
    --model results/stride/runs/base_32node_s17_<date>_<time>/train/model
```

Evaluates that checkpoint on the held-out traffic matrices, 30 monitoring periods
each. `--auto` runs all of them; drop it to be prompted for one. Watching it works
the same way as training, `sudo tmux attach -t stride`.

Given `--model`, the session is written **inside the run that owns the
checkpoint**:

```
results/stride/runs/base_32node_s17_<date>_<time>/test/<date>_<time>/
├── ckpt.txt                 which checkpoint, with a sha256 per file
└── real/<tm_id>/
    ├── real_directed_test/  ← the metrics the paper reads
    ├── real_test/           same measurement, both link directions summed
    └── sim_*/               NetworkX estimate, a cross-check
```

Read `real_directed_test/<tm_id>_eval_metrics.csv`. The undirected variants sum a
link's capacity across both directions and hide one-way saturation.

### What happens without `--model`

It reads `results/stride/model`, the live directory every training run
overwrites — you would be measuring whichever training finished last. And since
there is no run to attach to, **the session opens a run directory of its own**:

```
results/stride/runs/base_32node_s17_<date>_<time>/
└── test/<date>_<time>/          ← a test/ with no train/ beside it
```

A directory under `runs/` holding only `test/` is exactly this case, and which
checkpoint it scored is knowable only from the sha256 in its `ckpt.txt`.

Baselines with nothing to load (`ospf`, `ilp`, `widest_path`, `drsir`) are run
without `--model` by design, and that directory shape is normal for them.

## 12. Figures and tables

```bash
PY="$HOME/miniconda3/envs/stride/bin/python"
for f in paper/figures/*/make_*.py; do (cd "$(dirname "$f")" && "$PY" "$(basename "$f")"); done
"$PY" paper/tables/build_paper_table.py
```

Rebuilds everything from `results/` — no network, no GPU needed. Files are named
after their thesis captions and overwrite in place.

Each generator names the sessions it reads at the top of the file. After a new
run, point the relevant one at it and re-run just that script:

```bash
cd paper/figures/holdout && "$PY" make_holdout_fig.py
```

The reward curves are the one group that reads training logs rather than test
sessions; they go through `paper/figures/reward/make_curves_csv.py`, whose
`RUN_MAP` lists which run each curve comes from.

## clean, train and test in one command

```bash
./scripts/run_chain.sh                                  # 32node + stride
./scripts/run_chain.sh geant_directed stride            # another env
./scripts/run_chain.sh 32node_144tm_directed stride 18  # another seed
STRIDE_CHUNK=4 ./scripts/run_chain.sh                   # with an override
```

`clean.sh`, train, `clean.sh`, test the checkpoint that training just produced,
`clean.sh`.

**It finds the checkpoint itself.** A run directory's name carries a timestamp
nobody knows in advance, so the script lists `runs/` before training and takes the
difference afterwards. Taking "the newest" would quietly test the wrong
checkpoint whenever something else had touched an older directory, and a test
against the wrong checkpoint looks exactly like a test against the right one. If
the difference is not exactly one directory it stops rather than guessing.

sudo is asked for once at the start and refreshed in the background, so an
eight-hour run does not halt at a password prompt hours after you left. The
`STRIDE_*` variables are passed as explicit `VAR=value` assignments, since
`sudo -E` does not carry them reliably.

Run it inside tmux as usual; the controller and the agent open windows beside it.

## If the update does not fit in 10 s

The monitoring period is 10 s and both inference and the update have to finish
inside it. Watch `training_time` in the drl window; exceeding it means this card
cannot hold the default settings.

The first knob is `update_window_chunk` in `config/algs/stride_config.py`, which
sets how many windows an update stacks into one GPU pass. **It does not change
what training produces** -- the gradients are identical bit for bit -- only
parallelism and memory. The default `2` was chosen for an 8 GB card; on a larger
one, raise it a step and measure again. Too high fails with an OOM rather than
quietly. The reasoning is in [`README.md` §5](README.md).

## Where things live

```
dataset/     topologies, traffic matrices, frozen candidate paths — inputs, never written to
results/     everything a run produces; runs/<name>/{train,test} are the archives,
             the rest of results/<alg>/ is live scratch the controller and agent
             exchange files through
paper/       figure and table generators (the thesis itself is not published -- paper/README.md)
docs/        methodology notes, including the dead ends and why they failed
```

A run directory name is the command that reproduces it: `base_32node_s17_...` is
`STRIDE_VARIANT=base`, `--env 32node_144tm_directed`, `--seed 17`.

---

## Every run the paper reports

One command per row. Each is the whole chain — clean, train, clean, test, clean
— so a row is a result, not a step. Prepend `STRIDE_CHUNK=<n>` to any of them if
the update does not fit in 10 s on your card (see above); it changes nothing
about what is produced.

Every row below is **seed 17**, the default. For the second seed, spell all three
positional arguments out and put `18` last — the seed is the third argument, so
it cannot be given without the two before it:

```bash
STRIDE_VARIANT=M4 ./scripts/run_chain.sh 32node_144tm_directed stride 18
```

### STRIDE and the methods it is compared against

Figures 14 and 17, Tables 8 and 9.

Three of them train, so they take the chain:

| Method | 32-node | GÉANT |
| --- | --- | --- |
| STRIDE | `./scripts/run_chain.sh` | `./scripts/run_chain.sh geant_directed stride` |
| LS2IC | `./scripts/run_chain.sh 32node_144tm_directed ls2ic_dd` | `./scripts/run_chain.sh geant_directed ls2ic_dd` |
| MADQN | `./scripts/run_chain.sh 32node_144tm_directed ps_dqn_dd` | `./scripts/run_chain.sh geant_directed ps_dqn_dd` |

**The other three have no training phase at all**, so there is no checkpoint for
a test to point at and the chain is the wrong tool — it would stop at its
`train/model` check. Run the evaluation directly, without `--model`, and it
becomes a run of its own:

| Method | 32-node | GÉANT |
| --- | --- | --- |
| DRSIR | `sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg drsir_dd --auto` | `sudo -E "$PY" test_single_tm.py --env geant_directed --alg drsir_dd --auto` |
| OSPF | `sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg ospf --auto` | `sudo -E "$PY" test_single_tm.py --env geant_directed --alg ospf --auto` |
| ILP | `sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg ilp --auto` | `sudo -E "$PY" test_single_tm.py --env geant_directed --alg ilp --auto` |

Their archived runs show it: `results/ospf/runs/<run>/` contains `test` and
nothing else, where a STRIDE run contains `test` and `train`. Run `clean.sh`
before each, since nothing else will.

OSPF and ILP are deterministic and were run at one seed only. DRSIR learns
during the evaluation itself, which is why it has no separate training phase.

Use `_dd` for LS2IC, MADQN and DRSIR. The undirected variants (`ls2ic`,
`ps_dqn`, `drsir`) read link metrics aggregated across both directions, which
hides per-direction saturation and is not what the paper compares against.

### Denoise steps M

Figure 18, Table 10. M=8 is the mainline, so it is the STRIDE row above rather
than an entry here.

| M | Command |
| --- | --- |
| 4 | `STRIDE_VARIANT=M4 ./scripts/run_chain.sh` |
| 6 | `STRIDE_VARIANT=M6 ./scripts/run_chain.sh` |
| 10 | `STRIDE_VARIANT=M10 ./scripts/run_chain.sh` |
| 12 | `STRIDE_VARIANT=M12 ./scripts/run_chain.sh` |

### Candidate-set size K

Table 13. K=20 is the mainline. **K=25 and K=30 need a different `--env`**:
`dataset/32node_traffic/k_paths.json` holds exactly 20 paths per pair, so K=10
and K=15 are prefixes of it and need nothing new, while K=25 and K=30 ask for
more paths than the file contains and read `k_paths_k30_ext.json` instead.
Regenerating `k_paths.json` at a larger K would reorder ties and silently
redefine what K=20 means, so the extension is a separate file behind a separate
env.

| K | Command |
| --- | --- |
| 10 | `STRIDE_VARIANT=k10 ./scripts/run_chain.sh` |
| 15 | `STRIDE_VARIANT=k15 ./scripts/run_chain.sh` |
| 25 | `STRIDE_VARIANT=k25 ./scripts/run_chain.sh 32node_144tm_directed_k30 stride` |
| 30 | `STRIDE_VARIANT=k30 ./scripts/run_chain.sh 32node_144tm_directed_k30 stride` |

### Component ablations

Figure 19, Table 11. Each removes one part of STRIDE and changes nothing else.

| Removed | Command | What it does |
| --- | --- | --- |
| encoder | `STRIDE_VARIANT=flatfc_nomask ./scripts/run_chain.sh` | flat `Linear` instead of attention + PMA, and an all-ones pair mask, so every pair sees the full global link state and only the per-pair head can tell them apart |
| diffusion | `STRIDE_VARIANT=nodiff ./scripts/run_chain.sh` | one decode pass, no denoise-step conditioning |
| actor gradient | `STRIDE_VARIANT=critic ./scripts/run_chain.sh` | the encoder is shaped by the critic alone |
