# Training helpers

## `patch_ryu.py`

```bash
python scripts/patch_ryu.py            # apply, then verify by re-importing
python scripts/patch_ryu.py --check    # verify only; exit 1 if unpatched
python scripts/patch_ryu.py --revert   # restore switches.py.orig
```

Ryu records when an LLDP frame was sent but not when the reply arrived, so it
cannot report link latency. This makes three edits inside the installed package
to close that: a `delay` field on `PortData`, a receive timestamp at the top of
`lldp_packet_in_handler`, and the subtraction that fills the field.

It exists because doing this by hand is how it went wrong before. **A missing
edit raises nothing** — every link delay reads 0, training optimises against a
constant, and the run looks healthy from start to finish. So the script verifies
by importing the patched module rather than trusting that the write happened,
and refuses to patch a file whose anchors do not appear exactly once instead of
guessing at a different Ryu version.

Ryu is a pip package, so this only ever writes inside the conda env. Rebuilding
the env means running it again.

## `clean.sh`

Shuts down everything a real Mininet run leaves behind. Run it after any
interrupted or killed training, and before relaunching.

```bash
./scripts/clean.sh              # default alg
./scripts/clean.sh <alg>        # specific results/<alg> to chown back
```

Auto-detects the repository root from its own location, so it works unchanged on
any machine and any user account.

### Why not `killall` + `mn -c`

Both of the obvious commands miss the case that actually breaks the next run.

- **Ryu runs as a `python` process**, not `ryu-manager` — the launcher is a shell
  script and the real binary is the interpreter. `killall ryu-manager` matches
  nothing and exits 0, so it looks like it worked.
- **`main.py` can outlive its own training.** It reaches the final step, writes
  its results, and then hangs — usually in `net.stop()` waiting on an OVS or
  controller socket that will never answer, sometimes for many hours. `mn -c`
  does not touch it.

Either survivor keeps holding the controller port. The next run then attaches to
a stale, frozen controller instead of a fresh one and produces a training curve
that looks completely plausible and is entirely garbage. There is no error
message anywhere in this failure.

### What it does, in order

1. `SIGKILL` the stuck `main.py` / `run_drl.py` orchestrator **first** — before
   touching Mininet or Ryu, so a process ignoring `SIGTERM` cannot survive the
   rest of the sequence.
2. Kill Ryu, iperf3, and Mininet.
3. Verify ports 6633 and 6653 were actually released, and abort loudly if not.
   This is the gate that turns a silent bad run into a visible failure.
4. Restore ownership of `results/` — a run under `sudo` leaves root-owned files.
5. Clear the `.drl_done` sentinel.

### Troubleshooting

**`main.py` exits 1 immediately after launch** — Mininet residue was not fully
cleared. Run `clean.sh` again, then confirm the ports are free before relaunching:

```bash
sudo netstat -tlnp | grep -E ':6633|:6653'      # must print nothing
```

**`PYTHON not found`** — `$HOME/miniconda3/envs/stride/bin/python` does not exist.
See the main [README](../README.md) §2.2 for creating the `stride` environment.

## Launching a run

See [README](../README.md) §4. The short version:

```bash
sudo -v
sudo -E "STRIDE_VARIANT=<name>" "$HOME/miniconda3/envs/stride/bin/python" \
    main.py --env geant --alg stride train
```

Write `STRIDE_VARIANT` as a positional `NAME=VALUE` assignment inside the `sudo`
invocation, as shown. The `STRIDE_VARIANT=... sudo ...` form places it in the
caller's environment, where `env_reset` discards it — and a missing
`STRIDE_VARIANT` does not raise. It falls back to `ACTIVE_VARIANT` in
`config/algs/stride_config.py` and trains a different variant than you asked for,
with no indication that anything went wrong.
