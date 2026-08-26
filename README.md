# STRIDE — Diffusion-Based Routing for SDN Traffic Engineering

> 中文版說明見 [README.zh-TW.md](README.zh-TW.md)。
>
> In a hurry? [`QUICKSTART.md`](QUICKSTART.md) is the same workflow as commands,
> briefly explained.

STRIDE selects, for every origin–destination (OD) pair, one path out of a fixed
set of K=20 candidates. All pairs are decided **jointly and non-autoregressively**
by a discrete-diffusion decoder that starts from a fully masked configuration and
refines it over M denoise steps. The policy is trained with actor–critic RL on a
real Mininet + Ryu testbed.

The repository also contains the baselines the paper compares against (LS2IC,
MADQN/PS-DQN, DRSIR, OSPF, and a static ILP oracle), the evaluation pipeline, and
the scripts that generate every figure and table in the paper.

---

## 1. Requirements

Verified on the development machine:

| Component | Version |
| --- | --- |
| OS | Ubuntu 20.04 |
| Python | 3.8.20 |
| Mininet | 2.3.1b1 |
| Open vSwitch | 2.13.8 |
| Ryu | 4.34 |
| PyTorch | 2.0.1+cu118 |
| GPU | NVIDIA RTX 3060 Ti, 8 GB |

Python and PyTorch live in the conda environment (§2.2), Mininet and Open vSwitch
at system level. Ryu is installed into the conda environment and **requires a
manual patch** (§2.3).

Real Mininet experiments additionally need `sudo`, and a kernel providing the
`openvswitch`, `veth` and `sch_netem` modules. Desktop Linux distributions have
them. Virtualised or trimmed kernels (WSL, some cloud images, some container
environments) may not, so check before installing.

```bash
lsmod   | grep -E 'openvswitch|veth|sch_netem'   # already loaded
modinfo openvswitch veth sch_netem               # available to load
```

A real run has a hard per-step budget: the agent must observe, decide and update
inside one monitoring period (10 s), or it falls behind the traffic it is
supposed to be routing. On an RTX 3060 Ti, one routing decision takes ~22 ms and a
32-node step ~4.2 s end to end. On a different machine, run a training session and
watch the update time the `drl` pane prints to see whether it fits.

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

`universe` is not an external source. It is one of the four components of the
official Ubuntu archive (`main`, `restricted`, `universe`, `multiverse`) and holds
community-maintained packages, Mininet among them. Desktop installations enable it
by default, some server and cloud images do not, and `apt install mininet` then
reports the package as missing. This line switches that component on.

`systemctl` is systemd's control command, systemd being what Ubuntu now uses to
manage background services. `enable` sets the service to start at boot, `--now`
starts it immediately as well.

**Why the daemon has to be running rather than started on demand.** Open vSwitch
is two resident processes, `ovsdb-server` (the configuration database) and
`ovs-vswitchd` (the one that actually forwards packets). What Mininet runs when it
creates a switch is `ovs-vsctl`, a **client** that issues commands to
`ovsdb-server` over `/var/run/openvswitch/db.sock`. It does not start the daemons
and has no way to. With them down, `ovs-vsctl` cannot connect and Mininet aborts
with `Error connecting to ovs-db with ovs-vsctl`.

So it is not started on demand — it has to be up beforehand. If you would rather
it did not start at boot, replace `enable --now` with
`sudo systemctl start openvswitch-switch` and run that yourself after each
reboot. `enable` only saves you from remembering.

These packages cover both Mininet and Open vSwitch, so Mininet's own `install.sh`
is **not** needed here.

**Why there is a further step.** Mininet installs only into the system Python,
never into a conda environment, while this repository's main process runs under
the **conda** Python (§4) and needs to `import mininet`. So the conda interpreter
has to be able to find the system copy. This is not moving Mininet into conda —
the `mn` command still runs under the system Python. Only the import search path
is at stake.

apt installs Mininet for the system Python, 3.12 on 24.04, while the conda
environment is on 3.8. The directories differ, so §2.2's `.pth` recipe would point
at the wrong one. Create a link instead.

```bash
conda activate stride
ln -sfn /usr/lib/python3/dist-packages/mininet \
      "$CONDA_PREFIX/lib/python3.8/site-packages/mininet"
python -c "from mininet.net import Mininet; from mininet.link import TCLink; \
           from mininet.node import RemoteController; print('ok')"
```

`ln` creates links. `-s` makes a symbolic link (a pointer to another path), `-f`
overwrites an existing target, and `-n` replaces a target that is itself a link to
a directory rather than descending into it. So the line places a link named
`mininet` in the conda environment's `site-packages/` pointing at the system copy,
and `import mininet` under the conda Python then resolves to the real files.

Only this one package is linked in. Nothing else installed for 3.12 becomes
visible to conda. The second command is the check — it must print `ok`.

**On this route, skip §2.2's `.pth` step.** Everything else is the same.

#### Ubuntu 20.04 — from source

```bash
git clone https://github.com/mininet/mininet src/mininet
cd src/mininet && git checkout -b 2.3.1b1
cd util && ./install.sh -a
sudo mn --test pingall
```

`install.sh -a` installs Open vSwitch too. Then do
§2.2's `.pth` step so the conda env can see it.

### 2.2 Python environment

```bash
conda create -n stride python=3.8 -y
conda activate stride
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Which Python runs what, and where each finds Mininet

**There is only one copy of the Mininet package**, in the system
`dist-packages`. Nothing is copied and nothing is installed into conda. The only
question is whether the conda interpreter can find that copy.

| Who | Which Python | How it finds Mininet |
| --- | --- | --- |
| `mn`, `sudo mn -c` and the other CLI tools | the system `/usr/bin/python3` | already on its search path, nothing to do |
| this repository's `main.py` | the **conda** `envs/stride/bin/python` | needs the `.pth` or the symlink |
| the Ryu controller | the conda Python | does not import mininet at all |

`main.py` builds the topology itself with `from mininet.net import Mininet`, and
it runs under the conda interpreter (§4), so without this bridge it stops on that
line.

**Bridge Mininet into the conda environment — source installs only, once.** The
apt route already did this with a symlink in §2.1, so skip to §2.3.

```bash
echo /usr/local/lib/python3.8/dist-packages \
  > "$CONDA_PREFIX/lib/python3.8/site-packages/system_dist_packages.pth"
python -c "import mininet; print(mininet.__file__)"   # must succeed
```

A `.pth` file is plain text holding one path per line. On every start, Python
reads every `.pth` under `site-packages/` and **appends** those paths to
`sys.path`. Appended, not prepended, so conda's own numpy and torch keep
priority and only what conda lacks — Mininet — falls through to the system copy.

**Why the two routes differ.** A source install targets 3.8, the same version as
the conda environment, so the whole directory can be added safely. apt installs
for 3.12, and that directory holds many other packages built for 3.12, which
must not be added to a 3.8 path — hence a symlink for `mininet` alone.

> **This file lives inside the conda environment.** Rebuilding the environment
> after `conda env remove -n stride` removes it along with everything else, and
> the two commands above have to be run again. The system Mininet is unaffected
> and does not need reinstalling. Without the file, every real run stops at
> `from mininet.net import Mininet`.

### 2.3 Ryu + delay patch

The controller measures per-link latency from LLDP round-trip timing, which
upstream Ryu records nowhere. `PortData` timestamps when a frame was *sent*, and
`lldp_packet_in_handler` never notes when the reply came back.

> **The per-link delay the paper reports does not come from LLDP.** LLDP
> underestimates systematically, reading about a third of the truth in steady
> state, and tc backlog is used as ground truth instead — see
> [`docs/delay_measurement_issues.md`](docs/delay_measurement_issues.md). The
> patch is still required, because the controller computes the LLDP delay either
> way and reads 0 without it.

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
restores it.

Verify at any time:

```bash
python scripts/patch_ryu.py --check      # exit 0 = patched
```

> **This is the failure worth guarding against.** A missing patch raises nothing.
> Every link delay reads 0, training optimises against a constant, and the run
> looks healthy from start to finish. Check before a long run, not after.


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

### 2.6 Where the installed files land

Three things are installed by now, and they land in different places. Worth
knowing on a shared machine.

| Component | Path | Installed into |
| --- | --- | --- |
| Ryu, and its patch | `$CONDA_PREFIX/lib/python3.8/site-packages/ryu/` | conda env, nothing outside it is touched |
| Mininet package | `/usr/local/lib/python3.8/dist-packages` (source) or `/usr/lib/python3/dist-packages` (apt) | system |
| `mnexec`, `ovs-*` | `/usr/bin` | system |
| the Mininet source tree | wherever you cloned it | a local directory, **deletable once installed** |

Only the clone location is up to you. The commands in §2.1 put it in the
repository's gitignored `src/`, so everything you added to this machine except the
system packages sits in one directory, and those 5 MB can go once the install is
done.

**Installing Mininet is a system-level act either way.** It creates network
namespaces and drives Open vSwitch, both of which need root, and this repository
cannot change that. On a shared server, check whether Mininet and Open vSwitch are
already installed rather than installing a second copy.

---

## 3. Architecture and layout

### A run is three processes

A real run opens three tmux windows. You start `main`, and it opens the other two.

```text
                            main.py   (sudo)
            builds the topology, drives the iperf3 traffic,
                      tears everything down
                     |                                  |
              spawns |                                  | spawns
                     v                                  v
          +-------------------+              +-------------------+
          |    controller     |              |        drl        |
          |    ryu-manager    |              |     run_drl.py    |
          +-------------------+              +-------------------+
                     |                                  ^
                     |    net_info_directed.csv         |
                     |    paths_metrics.json            |
                     +--------------------------------->+
                     |                                  |
                     ^           drl_paths.json         |
                     +----------------------------------+
```

| Window | Process | Responsible for |
| --- | --- | --- |
| `main` | `main.py`, needs sudo | building the topology, spawning the other two, driving the iperf3 traffic, and on exit tearing down Mininet and `ryu-manager` and chowning `results/` back to you |
| `controller` | `ryu-manager --observe-link` with the apps under `utils/` | measuring the network (utilisation, delay, loss), maintaining the topology, installing the agent's chosen paths as flow rules |
| `drl` | `run_drl.py`, the agent itself | reading link state, choosing one candidate path per source-destination pair, training |

**There is no socket between the three. It is all files.** The controller writes
its measurements to `results/<alg>/net_info_directed.csv` and
`paths_metrics.json`, the agent writes its decisions to `drl_paths.json`, and the
controller installs those.

The order.

1. `main` builds the Mininet topology
2. it spawns `controller`, then **waits 30 seconds** — so topology discovery and
   the first round of port stats complete first
3. it spawns `drl`
4. `main` starts the iperf3 traffic. Only now does the agent have anything to
   measure, so training effectively begins here, and the loop runs until the
   agent writes `.drl_done`
5. `main` tears down Mininet and `ryu-manager` and chowns `results/` back to you

**The three panes fail independently, and all three need watching.** None of them
reports on the state of the other two. If the controller stops, `main` and `drl`
print nothing and continue. If the agent stops, `main` keeps driving traffic and
waits for a `.drl_done` that will not arrive. A pane producing no output is not
evidence that the process is still running.

### Layout

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

Read `measurement.txt` first. It decides whether the run is usable.

```bash
cat results/<alg>/runs/<run>/train/measurement.txt   # stale_seconds should be single digits
```

A run whose controller stopped
measuring partway still produces a full step count, a saved checkpoint and a
reward curve that keeps moving, because the reward follows the action rather
than the network -- `stale_seconds` is how long before the end the last
measurement landed, and anything past a monitoring period means the agent spent
that time training against a file that stopped updating. See
[`docs/controller_stops_measuring.md`](docs/controller_stops_measuring.md).

### Shutdown

```bash
./scripts/clean.sh            # or: ./scripts/clean.sh <alg>
```

`clean.sh` does `mn -c` and the process kills in the order that actually works,
then checks the result:

1. kills a stuck `main.py` / `run_drl.py` **first**, before the daemons
2. `pkill -f ryu-manager`, iperf3, `mn -c`
3. verifies ports 6633/6653 were released, and aborts if not
4. chowns `results/` back (a sudo run leaves it root-owned) and removes
   the `.drl_done` sentinel


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
every algorithm alike.

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
the seed from `--seed`.

```bash
STRIDE_VARIANT=M4 "$PY" main.py --env 32node_144tm_directed --alg stride --seed 18 train
```


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
there is plenty of headroom, raise C one step and measure again. Too high fails with an OOM rather than producing
wrong results silently.

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

## 7. Paper figures and tables

```text
paper/figures/<topic>/make_*.py     regenerate one figure group
paper/figures/reward/*.csv          cached training curves (see below)
paper/tables/build_paper_table.py   LP/ILP comparison tables
paper/figures/algo/                 algorithm pseudocode rendering
```


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

`sim_*` is the NetworkX estimate — "NX" in metric names means NetworkX. It
shares the routing decision with the real
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
dataset/extend_k_paths.py        build a K=30 candidate file while preserving the fixed K=20 prefix
```

---

## 10. Known pitfalls

- **A slow machine** can make the controller miss its 30-second startup window,
  after which the agent process errors out. Shut everything down (§4) and retry.
- **`sudo -E` does not reliably forward variables.** Always use the
  `sudo -E "VAR=value" python ...` form.

---

## 11. Complete uninstall

The order runs from the **conda environment** to the **system packages**. The
first three steps are safe on a shared machine. The fourth is not.

Against the table in §2.1: Ryu and its patch live inside the conda environment,
so removing the environment takes them with it. Mininet and Open vSwitch are
system-level, and everyone else on the machine uses the same copy.

### 11.1 Confirm the experiment data is backed up

`results/*/runs/` holds the complete archive of every run — training logs, test
sessions, checkpoints. It is tracked in git, but **only what has been pushed
exists anywhere else**.

```bash
cd ~/stride && git status --short && git log --oneline origin/main..HEAD
```

Continue only if both print nothing. Any output means something still exists
only on this machine.

### 11.2 Stop the running processes

```bash
sudo mn -c                                  # clear leftover namespaces and bridges
sudo killall -q iperf3 ryu-manager
tmux ls; sudo tmux ls                       # check both — the controller may be in root's session
```

`sudo tmux ls` is a separate check because, when the run was not started inside
tmux, the controller and the agent are put in a detached session owned by root
whose socket is separate from yours. Once you know which sessions to remove, use
`tmux kill-session -t <name>` rather than `kill-server`, which would take your
other sessions with it.

### 11.3 The repository and the conda environment

```bash
rm -rf ~/stride                             # including the Mininet source tree under src/
conda env remove -n stride
```

Ryu, the delay-measurement patch and the Mininet symlink created by the apt route
in §2.1 all live inside the environment and disappear with it. Other conda
environments on the machine are untouched.

### 11.4 Mininet and Open vSwitch — shared, affects other people

> **Check that nobody else is using them first.** These are system packages and
> belong to no single user. If you simply do not want to run experiments any
> more, stopping after 11.3 is enough.

apt route (Ubuntu 24.04 and later)

```bash
sudo systemctl disable --now openvswitch-switch
sudo apt purge mininet openvswitch-switch openvswitch-common \
               openvswitch-pki openvswitch-testcontroller
sudo apt autoremove
```

Source route (Ubuntu 20.04). `install.sh` has **no** uninstall target, so this is
manual.

```bash
sudo rm -rf /usr/local/lib/python3.8/dist-packages/mininet \
            /usr/local/lib/python3.8/dist-packages/mininet-*.dist-info
sudo rm -f  /usr/local/bin/mn /usr/bin/mnexec
```

The source route installs Open vSwitch **through apt as well** (`install.sh -a`
calls apt internally), so purge it exactly as above.

### 11.5 Open vSwitch leftover state

purge does not remove the database. It holds the bridge definitions, so delete it
only if Open vSwitch is not wanted at all.

```bash
sudo rm -rf /etc/openvswitch /var/lib/openvswitch /var/log/openvswitch
```

### 11.6 Verify

```bash
command -v mn mnexec ovs-vsctl     # all three should print nothing
conda env list | grep stride       # should print nothing
ls ~/stride                        # No such file or directory
ip -br link | grep -E "^(s[0-9]|ovs)"   # no leftover interfaces
```

All four clear means the machine is clean. Interfaces like `s1` or `s2` still in
`ip -br link` mean `sudo mn -c` in 11.2 did not run or did not succeed — run it
again.
