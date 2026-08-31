# main.py
"""Entry point for training and evaluation.

Launching this file correctly is non-obvious. The canonical form is:

    PY="$HOME/miniconda3/envs/stride/bin/python"
    sudo -E "STRIDE_VARIANT=nodiff" "$PY" main.py \
        --env 32node_144tm_directed --alg stride --seed 18 train

  sudo
      Mininet creates network namespaces and veth pairs, and attaches to
      Open vSwitch. All of that needs root.

  -E
      sudo's `env_reset` default discards the caller's environment. We need
      DISPLAY / XAUTHORITY / DBUS_SESSION_BUS_ADDRESS to survive, because
      the launcher opens two gnome-terminal windows (Ryu controller and DRL
      agent) and gnome-terminal cannot find its session bus without them.
      `--terminal inline` needs none of this and runs both children in the
      current terminal instead.

  absolute path to the conda interpreter
      -E does NOT preserve PATH; sudo's `secure_path` overrides it with a
      fixed system list. So `python` under sudo resolves to the system
      Python 3.8, which has no torch, no numpy and no ryu. Naming the
      interpreter by full path sidesteps PATH entirely. This is also why
      `conda activate stride` before sudo does not help — activation only
      edits PATH, and sudo throws PATH away.

  STRIDE_VARIANT
      Picks the architecture; optional, defaults to "base". It has to be an
      environment variable because the config package resolves it at import,
      before argparse runs. Pass it as an explicit "VAR=value" assignment
      rather than relying on -E — if it gets dropped, the run silently falls
      back to the default and trains the wrong thing without any error.

  --seed
      Picks the seed, for every algorithm, defaulting to 17. It is an ordinary
      argument precisely so sudo cannot drop it.

      Architecture, topology and seed are three separate knobs on purpose: a
      variant describes an architecture only. The run archive is named from
      all three:
      results/<alg>/runs/<variant>_<topology>_s<seed>_<date>_<time>/train/

A healthy real run prints "Building topology ..." then
"Controller spawned, wait 30 s ..." and opens two terminal windows.
"""
import argparse, config, os, time, subprocess, signal, atexit, termios
from utils.init_path import init_paths
from utils.process_launcher import (TERMINAL_CHOICES, launch, resolve_terminal,
                                    write_child_config)
import pwd

def _tty_guard():
    """Save this terminal's settings and hand back a restore function.

    Building the topology puts the controlling terminal into raw mode -- Mininet
    talks to a shell per switch -- and leaves it there. With -opost the terminal
    stops turning newlines into CR+NL, so from that point every line of output
    starts where the previous one ended, walking off the right edge of the
    screen. The run is fine; only reading it is not.

    Returns a no-op when stdin is not a terminal, which is the case whenever the
    orchestrator itself was launched into a pipe.
    """
    try:
        saved = termios.tcgetattr(1)          # stdout: stdin is /dev/null by now
    except (termios.error, ValueError, OSError):
        return lambda: None

    def restore():
        try:
            termios.tcsetattr(1, termios.TCSADRAIN, saved)
        except (termios.error, ValueError, OSError):
            pass
    return restore


def _detach_stdin():
    """Point stdin at /dev/null so no child can reach this terminal through it.

    Mininet's Node.popen sets stdout and stderr but leaves stdin alone, so every
    process it starts -- a shell per switch, then one iperf3 per flow, hundreds of
    them -- inherits ours. One of them puts the terminal into raw mode and does
    not put it back, after which output stops getting CR with its NL and each
    line starts where the last one ended. Restoring the settings once is not
    enough, because traffic keeps starting new processes for the whole run.

    main.py never reads stdin. test_single_tm.py does, when it asks which matrix
    to test, so it only does this under --auto.
    """
    try:
        fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(fd, 0)
        os.close(fd)
    except OSError:
        pass


def spawn_controller(ctrl_cfg, terminal="auto"):
    cmd = (
        f"source {ctrl_cfg['conda_sh']} && "
        f"conda activate {ctrl_cfg['conda_env']} && "
        f"cd {ctrl_cfg['controller_dir']} && "
        f"ryu-manager --observe-link {ctrl_cfg['controller_entry']}"
    )
    return launch(cmd, "controller", terminal)

def spawn_drl(merged_cfg, mode, terminal="auto"):
    cfg_path = write_child_config(merged_cfg)
    cmd = (
        f"source {merged_cfg['conda_sh']} && "
        f"conda activate {merged_cfg['conda_env']} && "
        f"python {os.path.join(merged_cfg.get('project_root', os.getcwd()), 'run_drl.py')} "
        f"--merged_cfg {cfg_path} --mode {mode}"
    )
    return launch(cmd, "drl", terminal)


def _chown_recursive(path, uid, gid):
    """遞迴 chown path 底下所有檔案和資料夾給 (uid, gid)."""
    for root, dirs, files in os.walk(path):
        try:
            os.chown(root, uid, gid)
        except OSError:
            pass
        for name in files:
            try:
                os.chown(os.path.join(root, name), uid, gid)
            except OSError:
                pass


def fix_results_ownership(alg_name=None):
    """把 results/ 下的檔案 owner 從 root 改回真正的使用者."""
    sudo_uid = os.environ.get("SUDO_UID")
    if not sudo_uid:
        return
    try:
        uid = int(sudo_uid)
        gid = pwd.getpwuid(uid).pw_gid
    except (ValueError, KeyError):
        return

    if alg_name:
        target = os.path.join("results", alg_name)
    else:
        target = "results"

    if os.path.isdir(target):
        _chown_recursive(target, uid, gid)


def init_result_dirs(config):
    algs_name = config["algs_name"]
    base_dir = os.path.join("results", algs_name)
    model_dir = os.path.join(base_dir, "model")
    metrics_dir = os.path.join(base_dir, "Metrics")

    if config["algs_name"].startswith('drsir'):
        stretch_dir = os.path.join(base_dir, "stretch")
        os.makedirs(stretch_dir, exist_ok=True)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    fix_results_ownership(algs_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,  help="32node_24tm")
    parser.add_argument("--alg", required=True,  help="ls2ic / meanfield ...")
    parser.add_argument("--ctrl", default="simple_monitor", help="controller config key")
    parser.add_argument("--terminal", choices=TERMINAL_CHOICES, default="auto",
                        help="where the controller and agent run: tmux windows in a "
                             "detached 'stride' session, a gnome-terminal window each, "
                             "or inline in this terminal. 'auto' prefers tmux.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for training. Overrides the algorithm config; "
                             "omit to use its own value (17 everywhere). The run "
                             "archive is named after it, so this is what tells two "
                             "runs of one configuration apart.")
    parser.add_argument("mode", choices=["train", "test"])
    args = parser.parse_args()

    terminal = resolve_terminal(args.terminal)
    print(f"[main] terminal backend: {terminal}")
    env_cfg, alg_cfg, ctrl_cfg = config.get(args.env, args.alg, args.ctrl)
    print(env_cfg)
    print(alg_cfg)
    print(ctrl_cfg)

    init_result_dirs(alg_cfg)
    init_paths(env_cfg, alg_cfg)

    # 結束時自動把 results/ 的檔案 owner 改回使用者
    atexit.register(fix_results_ownership, alg_cfg["algs_name"])
        
    os.environ["ALG_NAME"] = alg_cfg["algs_name"]
    os.environ["ENV_NAME"] = env_cfg["topology"]
    os.environ["NUM_LINK"] = str(env_cfg.get("num_link", 0))  # controller LLDP-completeness audit
    # 2026-07-10: DRSIR reward mode from config (drsir_dd sets 2). Config wins
    # over any shell-set DRSIR_REWARD_CORRECTED — no silent mode-A fallback.
    if "drsir_reward_mode" in alg_cfg:
        os.environ["DRSIR_REWARD_CORRECTED"] = str(alg_cfg["drsir_reward_mode"])
        print(f"DRSIR_REWARD_CORRECTED={os.environ['DRSIR_REWARD_CORRECTED']} (from alg config)")
    if (alg_cfg.get("sim_training", False) == False):
        from loader import env_loader
        # --- 1. 先建拓樸 ---------------------------------------------------
        print("Building topology ...")
        _detach_stdin()
        restore_tty = _tty_guard()
        atexit.register(restore_tty)
        net = env_loader.build_topo(env_cfg)
        restore_tty()          # Mininet leaves it raw; the rest of the run is readable again

        # --- 2. 開 Ryu controller -----------------------------------------
        ctrl_proc = spawn_controller(ctrl_cfg, terminal)

        # Pre-DRL wait for the controller (topology discovery + first port-stats).
        # The monitor greenlet additionally waits setting.MONITOR_START_DELAY before
        # its first metric cycle, so it doesn't run on a partial topology. Bump BOTH
        # this and MONITOR_START_DELAY if a slow host reintroduces the frozen-MLU race.
        print("Controller spawned, wait 30 s ...")
        time.sleep(30)

        # --- 3. 啟動 DRL 訓練 ---------------------------------------------
        print("Real training mode enabled.")
        print("Start DRL ...")
        merged_cfg = {**env_cfg, **alg_cfg, **ctrl_cfg}   # SimpleNamespace 給 training()
        if args.seed is not None:
            merged_cfg["seed"] = args.seed
        drl_proc = spawn_drl(merged_cfg, args.mode, terminal)

        # --- 4. 啟動流量 (循環直到 DRL 寫 .drl_done) -----------------------
        print("Start traffic ...")
        env_loader.start_traffic(net, env_cfg, mode="train", alg_name=alg_cfg["algs_name"])

        # --- 5. 收尾 -------------------------------------------------------
        # ctrl_proc.terminate() 只殺 gnome-terminal launcher，內層的 ryu-manager
        # 因為 cmd 用了 `; exec bash` 而被 bash 接管 process slot，不會跟著死。
        # 用 pkill -f 對 cmdline regex 才抓得到（ryu-manager 實際 binary 是
        # python，killall 找不到）。對 Popen 本身的 wait() 也加 timeout 防止
        # gnome-terminal hang。
        # start_traffic only returns once .drl_done exists, and the abort path
        # fills that file with its reason. The agent's traceback lands in the
        # drl window, but this is the pane anyone actually watches, so the
        # reason has to be readable here too.
        abort_reason = ""
        try:
            with open(f"./results/{alg_cfg['algs_name']}/.drl_done") as fh:
                first = fh.read().strip().splitlines()[0] if fh else ""
            if first.startswith("aborted"):
                abort_reason = first
        except (OSError, IndexError):
            pass

        if abort_reason:
            print("=" * 78)
            print(f"[main] TRAINING ABORTED -- {abort_reason}")
            print("[main] The controller stopped writing measurements, so the agent "
                  "stopped rather than train on a frozen file for the rest of the run.")
            print("[main] Full traceback: the drl window, or "
                  "results/_terminal_logs/drl_*.log")
            print("[main] What to check next: docs/controller_stops_measuring.md")
            print("=" * 78)
        else:
            print("Training finished, clean up.")
        net.stop()
        # The agent writes .drl_done and exits on its own; a non-zero status
        # means it died instead, which used to pass unnoticed because nothing
        # looked at it. wait() rather than poll(): the agent can still be
        # printing its traceback when start_traffic notices the sentinel, and a
        # poll() there reads None and loses the status.
        try:
            drl_status = drl_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            drl_status = None
        if drl_status not in (None, 0):
            print(f"[main] WARNING: the DRL agent exited with status {drl_status} "
                  f"-- this run's archive may be incomplete.")
        if drl_status is None:
            drl_proc.terminate()
        try:
            drl_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("[main] drl_proc didn't exit in 10s, force kill")
            drl_proc.kill()
        ctrl_proc.terminate()
        try:
            ctrl_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("[main] ctrl_proc (gnome-terminal) didn't exit, ignoring")
        # The actual ryu-manager process inside the terminal — kill explicitly.
        subprocess.run(["sudo", "pkill", "-f", "ryu-manager"], check=False)
        print("[main] Cleanup done (Mininet stopped, ryu-manager killed).")
        # Cleanup first, exit status second: scripts/run_chain.sh reads this to
        # decide whether to go on and test the checkpoint, and a run the agent
        # aborted must not be tested.
        # abort_reason is the authority: it says the agent decided to stop, and
        # it survives whatever the exit status did.
        if abort_reason or drl_status not in (None, 0):
            raise SystemExit(drl_status if drl_status not in (None, 0) else 1)
    else:
        # SimpleNamespace 給 training()
        # drl_proc = spawn_drl(merged_cfg, args.mode)
        print("Simulated training mode enabled.")
        print("Start DRL  ...")

        import tempfile, json
        # subprocess intentionally NOT re-imported here — module-level import
        # at line 2 already provides it. A local `import subprocess` shadows
        # the module-level binding across the ENTIRE main() function (Python
        # scoping rule: any local assignment makes the name local everywhere
        # in the function), which triggers UnboundLocalError at lines 195/201/204
        # in the real-env cleanup branch even though those lines never reach
        # this `else` branch. Real-env crash trace observed 2026-05-17.

        merged_cfg = {**env_cfg, **alg_cfg, **ctrl_cfg}
        if args.seed is not None:
            merged_cfg["seed"] = args.seed

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as fp:
            json.dump(merged_cfg, fp)
            cfg_path = fp.name

        cmd = (
            f"source {merged_cfg['conda_sh']} && "
            f"conda activate {merged_cfg['conda_env']} && "
            f"python run_drl.py --merged_cfg {cfg_path} --mode {args.mode}"
        )

        subprocess.run(["bash", "-c", cmd])

if __name__ == "__main__":
    main()