# STRIDE — Diffusion-Based Routing for SDN Traffic Engineering

> 中文版說明見 [README.zh-TW.md](README.zh-TW.md)。
>
> In a hurry? [`QUICKSTART.md`](QUICKSTART.md) is the same workflow as commands
> with one line of explanation each.

STRIDE selects, for every origin–destination (OD) pair, one path out of a frozen
set of K=20 candidates. All pairs are decided **jointly and non-autoregressively**
by a discrete-diffusion decoder that starts from a fully masked configuration and
refines it over M denoise steps. The policy is trained with actor–critic RL against
either a fluid-queue simulator or a real Mininet + Ryu testbed.

The repository also contains the baselines the paper compares against (LS2IC,
MADQN/PS-DQN, DRSIR, OSPF, widest-path, adaptive Dijkstra, mean-field, and a static
ILP oracle), the evaluation pipeline, and the scripts that generate every figure and
table in the paper.

---

## 1. Requirements

Verified on the development machine:

| Component | Version | Notes |
| --- | --- | --- |
| OS | Ubuntu 20.04 | Python 3.8 is the system default, which Mininet needs |
| Python | 3.8.20 | conda env, see §2.2 |
| Mininet | 2.3.1b1 | system-wide install, **not** in conda |
| Open vSwitch | 2.13.8 | system daemon |
| Ryu | 4.34 | pip, **requires a local patch**, see §2.3 |
| PyTorch | 2.0.1+cu118 | CUDA 11.8 |
| GPU | NVIDIA RTX 3060 Ti, 8 GB | the machine every reported number was measured on |

Real Mininet experiments additionally need `sudo`, and a kernel providing the
`openvswitch`, `veth` and `sch_netem` modules (standard on desktop Linux
distributions; **not** available under WSL2 or minimal cloud kernels).

A real run has a hard per-step budget: the agent must observe, decide and update
inside one monitoring period (10 s), or it falls behind the traffic it is
supposed to be routing. On the GPU above, one routing decision takes ~22 ms and a
32-node step ~4.2 s end to end. Do not assume a CPU-only machine fits in the
budget — verify it before trusting a curve produced on one.

---

## 2. Installation

### 2.1 Mininet + Open vSwitch

Two ways, and the distribution decides which. Pick one.

#### Ubuntu 24.04 and later — distribution packages

```bash
sudo add-apt-repository universe          # if mininet is not found
sudo apt install mininet openvswitch-switch
sudo systemctl enable --now openvswitch-switch
mn --version                              # 2.3.0
sudo mn --test pingall                    # verify
```

Do **not** run Mininet's `install.sh` here: it fails on pep8/pycodestyle and then
on PEP 668, and every workaround is worse than the packages. 2.3.0 is enough —
this repository only calls `Mininet(controller=RemoteController, link=TCLink)`
and `addLink(bw=, max_queue_size=)`.

apt installs Mininet for the **system** Python, 3.12 on 24.04, while the conda
env in §2.2 is on 3.8, so §2.2's `.pth` recipe points at the wrong directory.
Link the package in directly instead — this also exposes Mininet without exposing
everything else installed for 3.12:

```bash
conda activate stride
ln -sfn /usr/lib/python3/dist-packages/mininet \
      "$CONDA_PREFIX/lib/python3.8/site-packages/mininet"
python -c "from mininet.net import Mininet; from mininet.link import TCLink; \
           from mininet.node import RemoteController; print('ok')"
```

Mininet's Python side is pure Python and targets 3.6+, so a 3.8 interpreter reads
the 3.12 install without trouble. The compiled part, `mnexec`, is a binary on
`PATH` and never imported. Skip §2.2's `.pth` step; everything else is the same.

#### Ubuntu 20.04 — from source

```bash
git clone https://github.com/mininet/mininet src/mininet
cd src/mininet && git checkout -b 2.3.1b1
cd util && ./install.sh -a
sudo mn --test pingall
```

`install.sh -a` installs Open vSwitch too, and takes 20–40 minutes. Then do
§2.2's `.pth` step so the conda env can see it.

#### Where all of this ends up

Worth knowing on a shared machine:

| | where | yours or everyone's |
| --- | --- | --- |
| Ryu, and its patch | `$CONDA_PREFIX/lib/python3.8/site-packages/ryu/` | **yours** — inside the env, nothing outside it is touched |
| Mininet package | `/usr/local/lib/python3.8/dist-packages` (source) or `/usr/lib/python3/dist-packages` (apt) | shared |
| `mnexec`, `ovs-*` | `/usr/bin` | shared |
| the Mininet source tree | wherever you cloned it | yours, and **deletable once installed** |

Only the clone location is up to you — the command above puts it in the
repository's gitignored `src/`, so everything you added to the machine except the
system packages sits in one directory. The installation itself is system-wide either way: Mininet creates
network namespaces and attaches to Open vSwitch, so it needs root regardless, and
nothing in this repository can change that. On a shared server, check that
Mininet and OVS are not already installed before adding a second copy.

### 2.2 Python environment

```bash
conda create -n stride python=3.8 -y
conda activate stride
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Bridge Mininet into the conda env — source installs only, once.** On 24.04 the
symlink in §2.1 has already done this; skip to §2.3.

A source install puts Mininet in the *system* `dist-packages`, so the conda
interpreter cannot see it. Create a `.pth` file that appends the system path. Python reads it
automatically on every interpreter start from then on; nothing has to be repeated
per run:

```bash
echo /usr/local/lib/python3.8/dist-packages \
  > "$CONDA_PREFIX/lib/python3.8/site-packages/system_dist_packages.pth"
python -c "import mininet; print(mininet.__file__)"   # must succeed
```

conda's own numpy/torch stay higher priority on `sys.path`; only Mininet falls
through to the system copy. **If the env is ever rebuilt, recreate this file** —
otherwise every real-environment run dies at `from mininet.net import Mininet`.

### 2.3 Ryu + delay patch

The controller measures per-link latency from LLDP round-trip timing, which
upstream Ryu records nowhere: `PortData` timestamps when a frame was *sent*, and
`lldp_packet_in_handler` never notes when the reply came back.

```bash
git clone https://github.com/faucetsdn/ryu src/ryu
pip install "setuptools==58.0.0" wheel "pbr==5.11.1"
pip install ./src/ryu --no-build-isolation
python scripts/patch_ryu.py
```

**Not `pip install ryu`.** The last PyPI release is 4.34 from 2020, which predates
eventlet 0.30.3 removing `ALREADY_HANDLED`; installing it gives you a package that
imports and then dies. Upstream master carries the compatibility fix and has never
been released, so the git tree is the only working source.

The two pins are separate and both required. `setuptools==58.0.0`: ryu's
`setup.py` calls `easy_install.get_script_args`, gone since setuptools 58, and
`--no-build-isolation` is what makes the build see the pin instead of a fresh
latest setuptools pip fetches for itself. `pbr==5.11.1`: ryu declares
`setup_requires=['pbr']` unversioned, and `setup_requires` is setuptools' own
mechanism rather than pip's -- it downloads the newest pbr into `src/ryu/.eggs/`
mid-build, out of `--no-build-isolation`'s reach, and current pbr imports
`setuptools.extern.tomli`, which setuptools only began vendoring at 61. Having
pbr installed already satisfies the requirement and nothing is fetched.

The script makes three edits inside the installed package — a `delay` field on
`PortData`, a receive timestamp at the top of the handler, and the subtraction
that fills the field — then re-imports the module to confirm they took. It is
idempotent, keeps the untouched file as `switches.py.orig`, and `--revert`
restores it. If the file does not look like the 4.34 it was written against, it
refuses rather than guessing, and §2.3 of the git history has the edits to make
by hand.

Verify at any time:

```bash
python scripts/patch_ryu.py --check      # exit 0 = patched
```

> **This is the failure worth guarding against.** A missing patch raises nothing.
> Every link delay reads 0, training optimises against a constant, and the run
> looks healthy from start to finish. Check before a long run, not after.

Because Ryu is a pip package, the patch lives inside your conda env — nothing is
written outside it, and rebuilding the env means running the script again.

### 2.4 Controller path

`config/controller/simple_monitor_config.py` spawns `ryu-manager` inside a conda
env, and the same file supplies the env the DRL agent runs in. **`conda_env` and
`conda_sh` are the only two machine-specific values in the repository** — set
them to the env you built in §2.2, whatever it is called. Everything else is
derived from the file's own location and needs no editing.

The env named there has to hold torch *and* the patched Ryu, since both children
are started inside it.

### 2.5 Datasets

```bash
python dataset/prepare_dataset.py --topology 32node --tms 144tm
python dataset/prepare_dataset.py --topology 32node --tms 24tm
python dataset/prepare_dataset.py --topology geant  --tms 24tm --tm_scale 3
```

---

## 3. Repository layout

```text
main.py                  Entry point: builds topology, spawns controller + agent
run_drl.py               Launched by main.py inside the agent process
test_single_tm.py        Evaluate one traffic matrix (real Mininet)

                         --- not on the paper path, kept for reference ---
test_sim_only.py         Simulator-side evaluation. UNMAINTAINED, see below.
test_single_tm_udp.py    UDP-probe measurement experiment, not the delay/loss
                         method the paper uses. One-off, never rerun.

algs/                    Agents. stride.py is the main method; the rest are baselines.
                         algs/__init__.py holds the REGISTRY that maps --alg to a class.
config/                  Three independent layers: env/ algs/ controller/
loader/                  env_loader.py (topology + traffic), train_loader.py (train/eval loops)
utils/                   Ryu controller app (simple_monitor.py, manager.py), iperf3 drivers,
                         delay/loss measurement
dataset/                 Topologies, traffic matrices, candidate path sets, and the
                         scripts that build them, including prepare_dataset.py
results/                 Experiment output, see §6
paper/                   Figure and table generators, LP/ILP bounds, see §7
docs/                    Methodology notes; docs/README.md indexes them
scripts/                 Cleanup and dataset-setup shell helpers
diagnostics/             Standalone methodology experiments. Nothing in the paper
                         pipeline runs them; they exist to justify a decision.
A-Traffic-.../           Vendored Enero/RouteNet code. Only SAC_PL_KP (the gym
                         environment) and Enero_datasets (traffic data) are used.
```

---

## 4. Running

### Interpreter

Every command below names the conda interpreter by absolute path. This is not
optional and it is not avoidable by activating the env first: `sudo` replaces
`PATH` with its own `secure_path`, so under `sudo` a bare `python` is the system
Python 3.8, which has no torch.

A shell variable shortens it to `"$PY"`, but a shell variable dies with the
shell, so put the line in `~/.bashrc` — then it is set in every terminal you
open from then on, and you never type it again:

```bash
echo 'PY="$HOME/miniconda3/envs/stride/bin/python"' >> ~/.bashrc
``` The repository scripts
resolve the interpreter from `$HOME` the same way, so they work unchanged on any
machine and user account.

### Real Mininet is the supported path

Training mode is decided by `sim_training` in the algorithm config — **not** by a
command-line flag. `main.py` treats an unset value as `False`, so:

| Algorithm | `sim_training` | Mode |
| --- | --- | --- |
| `ls2ic_nx` | `True` | fluid-queue simulator |
| `stride` | `False` | real Mininet, every variant |
| everything else | unset -> `False` | real Mininet |

> **The simulation-training path is not maintained.** `ls2ic_nx` is the only
> algorithm still wired to it, and that code has not been kept in step with the
> rest of the repository — expect to fix things yourself before it runs. **STRIDE
> has no simulation-training variant at all.** Everything below assumes real
> Mininet.

**Evaluation-time simulation is not maintained either.** `test_sim_only.py`
scores routing decisions against the fluid-queue model. It used to be invoked
automatically at the end of every `--auto` test session, writing a `sim/` half
next to the real measurements; that call was removed, because the sim half had
drifted to covering a single TM and was no longer comparable to the real half
beside it. No figure or table in the paper ever read it. The script is still
here and still runs standalone, but treat its output as unvalidated.

### Training

```bash
sudo -E "STRIDE_VARIANT=nodiff" "$PY" main.py \
    --env 32node_144tm_directed --alg stride --seed 18 train
```

The archive lands in
`results/stride/runs/nodiff_32node_s18_<date>_<time>/train/`. Both variables are
optional: they default to `base` and `17`.

Every part of that line is load-bearing:

| Part | Why |
| --- | --- |
| `sudo` | Mininet creates network namespaces and veth pairs and attaches to Open vSwitch. All of that needs root. |
| `-E` | sudo's `env_reset` default discards your environment. Only the `gnome` terminal backend needs it — `DISPLAY`, `XAUTHORITY` and `DBUS_SESSION_BUS_ADDRESS` have to survive or gnome-terminal cannot find its session bus. The default `tmux` backend needs none of them. |
| `"$PY"` (absolute path) | `-E` does **not** preserve `PATH` — sudo's `secure_path` replaces it with a fixed system list. So a bare `python` under sudo resolves to the system Python 3.8, which has no torch, no numpy and no ryu. This is also why running `conda activate stride` first does not help: activation only edits `PATH`, and sudo throws `PATH` away. |
| `"STRIDE_VARIANT=..."` | Picks the architecture. Write it as an explicit `VAR=value` assignment rather than trusting `-E`: if it gets dropped the run silently falls back to `base` and trains the wrong thing with no error. |
| `--seed` | Picks the seed, for any algorithm. It is an ordinary argument, so `sudo` cannot drop it. Omit it for 17. |

A real run prints `Building topology ...` followed by
`Controller spawned, wait 30 s ...` and opens a terminal each for the controller
and the agent.

Where those two go is `--terminal`:

| | |
| --- | --- |
| `tmux` | a window each in a detached `stride` session — `tmux attach -t stride` to watch. No DISPLAY needed, so this works over SSH, and a window outlives the process it ran so a crash stays readable. |
| `gnome` | a gnome-terminal window each. Needs a graphical session, and under `sudo` needs the variables above. |
| `inline` | both in the current terminal, output interleaved. The fallback when neither of the others is available. |
| `auto` | tmux if installed, else gnome if there is a DISPLAY, else inline. The default. |
If neither appears, the setup is wrong.

### Testing

```bash
sudo -E "$PY" test_single_tm.py --env 32node_144tm_directed --alg stride --auto \
    --model results/stride/runs/nodiff_32node_s18_<date>_<time>/train/model
```

`--model` names the checkpoint to evaluate, and the session is written inside
that run: `.../runs/nodiff_32node_s18_<date>_<time>/test/<date>_<time>/`. One
argument decides both, so the results cannot end up filed under a different run
than the one they came from.

**Without `--model` the test reads `results/<alg>/model`** — the live directory
that every training run overwrites. The checkpoint under test is then whichever
training happened to finish last, and the session records no way to tell which.
Either pass `--model`, or read the `ckpt.txt` the session writes: it holds the
resolved path and a sha256 per checkpoint file.

OSPF, ILP, widest-path and DRSIR have nothing to evaluate from, so they are run
without `--model` and become runs of their own.

### The whole chain in one command

```bash
./scripts/run_chain.sh                                  # 32node + stride, seed 17
./scripts/run_chain.sh geant_directed stride            # another topology
./scripts/run_chain.sh 32node_144tm_directed stride 18  # another seed
STRIDE_VARIANT=M4 ./scripts/run_chain.sh                # another architecture
```

Arguments are positional -- env, alg, seed -- and each defaults to the mainline
choice, so the bare command is 32-node, `stride`, seed 17. `STRIDE_*` overrides
go in front as environment assignments; the script forwards them to `sudo` as
explicit `VAR=value` pairs, because `sudo -E` does not carry them reliably.

It runs `clean.sh`, trains, `clean.sh`, tests the checkpoint that training just
produced, `clean.sh`. About eight and a half hours for 3000 steps on 32-node,
plus half an hour for the five test traffic matrices.

**It finds the checkpoint itself.** A run directory's name carries a timestamp
nobody knows in advance, so the script lists `runs/` before training and takes
the difference afterwards. Taking "the newest" would quietly test the wrong
checkpoint whenever anything else had touched an older directory, and a test
against the wrong checkpoint looks exactly like a test against the right one. If
the difference is not exactly one directory it stops rather than guessing.

sudo is asked for once at the start and refreshed in the background, so an
eight-hour run does not halt at a password prompt hours after you left. Run it
inside tmux; the controller and the agent open windows beside it.

Three things say whether what came back is usable:

```bash
cat results/<alg>/runs/<run>/train/measurement.txt   # stale_seconds should be single digits
ls  results/<alg>/runs/<run>/test/<session>/real/    # five traffic matrices, 154 files each
ls  results/_terminal_logs/                          # controller and agent output, kept on disk
```

`measurement.txt` is the one that matters. A run whose controller stopped
measuring partway still produces a full step count, a saved checkpoint and a
reward curve that keeps moving, because the reward follows the action rather
than the network -- `stale_seconds` is how long before the end the last
measurement landed, and anything past a monitoring period means the agent spent
that time training against a frozen file. See
[`docs/controller_stops_measuring.md`](docs/controller_stops_measuring.md).

### Shutdown

```bash
./scripts/clean.sh            # or: ./scripts/clean.sh <alg>
```

`clean.sh` does `mn -c` and the process kills in the order that actually works,
then checks the result:

1. kills a stuck `main.py` / `run_drl.py` **first**, before the daemons
2. `pkill -f ryu-manager`, iperf3, `mn -c`
3. verifies ports 6633/6653 were released, and aborts loudly if not
4. chowns `results/` back (a sudo run leaves it root-owned) and removes
   the `.drl_done` sentinel

Doing it by hand with `killall` + `mn -c` misses steps 1 and 3, and those are the
ones that matter. Ryu runs under the process name `python`, so `killall
ryu-manager` never matches it, and neither does anything stuck in `net.stop()`.
A survivor keeps holding the controller port, and the *next* run then connects to
a stale frozen controller and produces a plausible-looking training curve that is
actually garbage — with no error anywhere.

---

## 5. Configuration

A run is configured by three independent choices, one per command-line flag:

```bash
"$PY" main.py --env geant --alg stride --ctrl simple_monitor train
```

| flag | answers | reads |
| --- | --- | --- |
| `--env` | which topology and traffic | `config/env/geant_config.py` |
| `--alg` | which algorithm | `config/algs/stride_config.py` |
| `--ctrl` | which Ryu controller app | `config/controller/simple_monitor_config.py` |

**The filename is the flag value.** `config/__init__.py` walks those three
directories at import time, and for every file exporting a dict named `config` it
registers the basename minus `_config` as an available key. There is no list of
names to keep in sync — dropping `config/env/mytopo_config.py` into the tree is
what makes `--env mytopo` work, and deleting the file is what removes it.
`main.py` then merges the three dicts into one and hands it to `run_drl.py`.

So the keys available are exactly the files present:

```bash
ls config/env config/algs config/controller | sed 's/_config\.py//'
```

At the time of writing:

- **env**: `geant`, `geant_directed`, `32node_24tm`, `32node_144tm`,
  `32node_144tm_directed`, `32node_144tm_directed_k30`
- **alg**: `stride`, `ls2ic`, `ls2ic_dd`, `ls2ic_nx`, `ps_dqn`, `ps_dqn_a`,
  `ps_dqn_dd`, `drsir`, `drsir_dd`, `ospf`, `widest_path`, `adaptive_dijkstra`,
  `meanfield`, `ilp`
- **ctrl**: `simple_monitor`

The seed is not part of any of these. It is `--seed` on the command line, for
every algorithm alike. There used to be four `*_seed18` config files whose only
content was `"seed": 18`, because the baselines had no override of their own.

### STRIDE variants

`config/algs/stride_config.py` is a two-layer config: `_BASE` is the mainline
configuration the paper reports, and each `VARIANTS` entry lists only what its
ablation changes:

```python
"base":          {},                       # mainline STRIDE
"M4":            {"iter_steps": 4},        # denoise-step ladder
"k10":           {"action_dim": 10},       # candidate-set size
"nodiff":        {"iter_steps": 1, "decision_token": "tau_only"},
"critic":        {"encoder_rl_grad_src": "critic"},
```

A variant is an architecture and nothing else. Topology comes from `--env` and
the seed from `--seed`, so there is no `..._32node_seed18` variant — that
combination is a run, not a configuration.

```bash
STRIDE_VARIANT=M4 "$PY" main.py --env 32node_144tm_directed --alg stride --seed 18 train
```

Two orthogonal test-time overrides exist so a single checkpoint can be evaluated
under different inference modes without defining a new variant:
`STRIDE_EVAL_SAMPLE` (greedy vs sampled decoding) and `STRIDE_ATTN_KERNEL`.

---

### Fitting the update inside 10 s: `update_window_chunk`

An update draws `mini_batch_seq` windows, each `time_seq` **consecutive**
transitions. Consecutive is required -- the encoder's GRU rolls its hidden state
forward, so the timesteps *within* a window can only run in sequence. Windows,
though, are independent of each other: each starts from `init_hidden` zeros, so
they can be stacked into a batch and run in one forward.

`update_window_chunk` (C) is how many get stacked:

```
C=1   8 chunks of batch 1     <- most sequential
C=2   4 chunks of batch 2     <- the default, chosen for an 8 GB card
C=4   2 chunks of batch 4
C=8   1 chunk  of batch 8
```

**Changing C does not change training.** The two paths weight their losses to the
same total (`/n_w` against `*B/n_w`), `.backward()` accumulates into `.grad` with
no `zero_grad()` between, so `opt.step()` sees the same gradients bit for bit,
give or take floating-point associativity. It is a parallelism knob, not a
hyperparameter -- `mini_batch_seq` and `time_seq` are those.

What it costs is memory: B times the batch is B times the activations held for
backward. Splitting `update()` into per-chunk backwards exists for exactly this
reason -- the whole graph at once is 512 forwards and OOMs on 8 GB.

No file edit needed: `STRIDE_CHUNK=4 sudo -E ...` overrides it, through the same
mechanism as `STRIDE_VARIANT`. A value that depends on the machine does not
belong in a tracked default -- 2 suits 8 GB, a 32 GB card wants more.

**So C should follow the card.** Watch `nvidia-smi` once a run is going, and if
there is plenty of headroom, raise C one step and measure again. Too high fails
loudly with an OOM rather than quietly.

## 6. Results layout

```text
results/<alg>/
├── runs/<variant>_<topology>_s<seed>_<YYYYMMDD>_<HHMMSS>/
│   ├── train/          model/ config.json output*.txt *_loss.txt
│   │                   training_mlu.txt step_time.txt
│   └── test/<YYYYMMDD>_<HHMMSS>[_sampled]/
│       ├── ckpt.txt    checkpoint path + sha256, decode mode, variant/topology/seed
│       └── real/<tm_id>/   Mininet measurements. A sim/ half used to sit
│                           beside this one; removed 2026-08-19, see §4
└── Metrics/ model/ net_info*.csv drl_paths.json ...
                        live scratch area — the file-based channel between the Ryu
                        controller and the agent during a run. Not an archive.
```

A run owns its tests. The checkpoint that produced a result is one directory up
from the session, so no naming convention has to carry that relationship. A run
may hold several tests — a re-run, or a different decoding mode — and each
records in `ckpt.txt` which checkpoint it actually loaded, because a session may
legitimately evaluate one from another run.

The run directory name is the command that reproduces it: variant to
`STRIDE_VARIANT`, topology to `--env`, seed to `--seed`. Only a non-default
decoding mode is marked in a session name; the host is not in either name, it is
in the archived `config.json` with the start and finish times.

The controller writes link statistics into the scratch area; the agent reads them.
There is no socket between the two. `runs/` is written at the end of a run or
session; everything beside it is scratch.

---

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

## 7. Paper figures and tables

```text
paper/figures/<topic>/make_*.py     regenerate one figure group
paper/figures/reward/*.csv          cached training curves (see below)
paper/tables/build_paper_table.py   LP/ILP comparison tables
paper/figures/algo/                 algorithm pseudocode rendering
```

Every generator resolves the repository root from its own file location, so the
tree can be moved or renamed without editing paths.

Each generator names the sessions it reads outright, so which run a figure comes
from is visible in the source. They read
`results/<alg>/runs/<run>/test/<session>/real/<tm_id>/` — evaluation output, not
training logs. The reward figures are the exception: they read training logs
instead, via two CSVs generated by `paper/figures/reward/make_curves_csv.py`.

That script rebuilds `train_curves.csv` and `components.csv` from the
`output_*.txt` files under `results/*/runs/*/train/`, which are the original
record of every training run. Re-run it if a CSV goes missing or you add a run;
its `RUN_MAP` holds the run-to-directory mapping. Nothing in `paper/` needs
network access.

To rebuild every figure and table:

```bash
for f in paper/figures/*/make_*.py; do (cd "$(dirname "$f")" && "$PY" "$(basename "$f")"); done
"$PY" paper/tables/build_paper_table.py
```

Each generator resolves paths from its own location, so it must be run from its
own directory — hence the `cd`. Outputs are named after their thesis captions
and overwrite in place.

[`paper/README.md`](paper/README.md) maps every numbered figure and table to the
generator that produces it, and carries a notation table from the thesis's
symbols to the code's identifiers.

---

## 8. Evaluation metrics

Every test session writes four sets of metrics per traffic matrix, under
`real/<tm_id>/`:

| directory | measured how | aggregation |
| --- | --- | --- |
| `real_directed_test/` | Mininet, `tc` counters on the switches | per direction |
| `real_test/` | same measurement | both directions of a link summed |
| `sim_directed_test/` | NetworkX, offline from the same chosen paths | per direction |
| `sim_test/` | same estimate | both directions summed |

**Read the directed ones.** The undirected aggregation sums a link's capacity
across both of its directions, so a link saturated one way and idle the other
averages out to something that looks fine. The paper reads
`real_directed_test/<tm_id>_eval_metrics.csv` throughout, and every
`paper/figures/` script has that path in it.

`sim_*` is the NetworkX estimate — "NX" in metric names means NetworkX, not
anything network-specific. It shares the routing decision with the real
measurement but models the queues instead of measuring them, so it is a
cross-check, never a reported number.

Reference points on Geant at `tm_scale=3`, directed: ILP ≈ 0.67, LP ≈ 0.67,
shortest-path/OSPF ≈ 0.87 maximum link utilisation. A trained policy should land
inside that band; reaching 1.0 means it is worse than plain shortest path.

---

## 9. Analysis tools

```text
paper/bounds/check_min_mlu.py         per-TM theoretical minimum MLU (LP relaxation or single-path ILP)
paper/bounds/edge_lp_bound.py         edge-based multicommodity LP lower bound (no candidate-set restriction)
paper/bounds/k_oracle_curve.py        optimal MLU as a function of candidate-set size K
dataset/extend_k_paths.py        build a K=30 candidate file while preserving the frozen K=20 prefix
```

---

## 10. Known pitfalls

- **`kpath_init`, `kpath_reset`, `get_link_features`** are methods added to the
  vendored gym environment, not upstream code. They are the per-OD K-candidate
  interface that *every* algorithm here goes through, STRIDE and baselines alike,
  so nothing about them is STRIDE-specific.
- **Killing a real run with `SIGKILL`** leaves `results/` owned by root. Fix with
  `chown -R`. A clean exit chowns it back automatically.
- **A slow machine** can make the controller miss its 30-second startup window,
  after which the agent process errors out. Shut everything down (§4) and retry.
- **`sudo -E` does not reliably forward variables.** Always use the
  `sudo -E "VAR=value" python ...` form.
