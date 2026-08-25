# test_single_tm.py
import argparse, config, os, time, subprocess, signal
from datetime import datetime
from loader import env_loader
from utils.init_path import init_paths
from utils.process_launcher import (TERMINAL_CHOICES, launch, resolve_terminal,
                                    write_child_config)
import pwd

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


def resolve_session_dir(model_arg, algs_name, cfg, timestamp, decode):
    """Where this session's output goes -> (session_dir, its parent).

    A test belongs to the training run whose checkpoint it evaluates, so when
    --model names one, the session is written inside that run:

        results/<alg>/runs/<run>/train/model   (--model)
        results/<alg>/runs/<run>/test/<ts>_<decode>   (session)

    One training run therefore holds its own tests, and the relationship needs
    no naming convention to survive.

    Without --model there is no run to attach to -- the checkpoint is whatever
    results/<alg>/model happens to hold -- so the session becomes a run of its
    own, named the same way build_run_archive_dir names training runs. That is
    also the normal case for OSPF, ILP, widest-path and DRSIR, which have no
    training to attach to at all. ckpt.txt still records the checkpoint hash, so
    such a session can be attributed afterwards if the need arises.
    """
    # greedy is the default, so only a sampled run says so in its name -- a field
    # that always reads the same carries nothing.
    leaf = timestamp if decode == "greedy" else f"{timestamp}_{decode}"

    if model_arg:
        m = os.path.abspath(model_arg).rstrip(os.sep).split(os.sep)
        # .../runs/<run>/train/model  ->  .../runs/<run>
        if len(m) >= 3 and m[-1] == "model" and m[-2] == "train":
            run_dir = os.sep.join(m[:-2])
            parent = os.path.join(run_dir, "test")
            return os.path.join(parent, leaf), parent

    import re
    core = re.sub(r"_?(32node|geant)", "", str(cfg.get("_experiment", algs_name)))
    core = re.sub(r"_?seed\d+", "", core).strip("_") or algs_name
    run = f"{core}_{cfg.get('topology', 'unknown')}_s{int(cfg.get('seed', 17) or 17)}_{timestamp}"
    parent = os.path.join(".", "results", algs_name, "runs", run, "test")
    return os.path.join(parent, leaf), parent


def write_ckpt_record(session_dir, model_dir, decode="greedy", cfg=None):
    """Record what a session was scored from, as <session>/ckpt.txt.

    The sha256 is the point: it identifies the checkpoint no matter how the
    directories are later renamed or moved, and it is what settles questions
    like "which of these two runs produced this result" without replaying the
    model against the recorded decisions.

    decode is written here rather than into the directory name, which only
    marks the non-default case.
    """
    import hashlib
    cfg = cfg or {}
    lines = [f"model_dir: {os.path.abspath(model_dir)}"]
    if os.path.isdir(model_dir):
        for f in sorted(os.listdir(model_dir)):
            path = os.path.join(model_dir, f)
            if not os.path.isfile(path):
                continue
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            lines.append(f"sha256 {h.hexdigest()}  {f}")
    else:
        lines.append("(directory not found at test time)")
    lines.append(f"decode: {decode}")
    for k, label in (("_experiment", "variant"), ("topology", "topology"), ("seed", "seed")):
        if cfg.get(k) is not None:
            lines.append(f"{label}: {cfg[k]}")
    lines.append(f"loaded_at: {datetime.now().isoformat(timespec='seconds')}")
    with open(os.path.join(session_dir, "ckpt.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  ckpt record -> {session_dir}/ckpt.txt")


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

    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid:
        try:
            uid = int(sudo_uid)
            user_info = pwd.getpwuid(uid)
            gid = user_info.pw_gid

            os.chown(base_dir, uid, gid)
            os.chown(model_dir, uid, gid)
            os.chown(metrics_dir, uid, gid)

            if algs_name.startswith('drsir'):
                os.chown(stretch_dir, uid, gid)

            # 2026-05-28: reclaim ownership of the WHOLE base_dir + drop a stale
            # drl_paths.json. test_single_tm runs as root (sudo) but spawns
            # run_drl in a gnome-terminal that drops to SUDO_USER; run_eval there
            # writes results/<alg>/drl_paths.json. If a prior interrupted run
            # left that file root-owned, the user-context writer hits
            # PermissionError -> header-only CSVs, no metrics (ILP real-test
            # incident). Recursive chown + removing the stale file makes every
            # launch self-healing regardless of prior ownership contamination.
            import subprocess
            subprocess.run(["chown", "-R", f"{uid}:{gid}", base_dir], check=False)
            stale_drl_paths = os.path.join(base_dir, "drl_paths.json")
            if os.path.exists(stale_drl_paths):
                os.remove(stale_drl_paths)
        except Exception as e:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,  help="32node_24tm")
    parser.add_argument("--alg", required=True,  help="ls2ic / meanfield ...")
    parser.add_argument("--ctrl", default="simple_monitor", help="controller config key")
    parser.add_argument("--terminal", choices=TERMINAL_CHOICES, default="auto",
                        help="where the controller and agent run: tmux windows "
                             "in a detached 'stride' session, a gnome window each, "
                             "or inline here. 'auto' prefers tmux.")
    parser.add_argument("--auto", action="store_true",
                        help="Auto mode: run test TMs [3,10,12,14,21] sequentially then exit")
    parser.add_argument("--delay-trace-links", default="",
                        help="Enable simple_delay trace for directed links, e.g. '13->14,14->13'")
    parser.add_argument("--delay-trace-file", default="",
                        help="Optional delay trace csv path (default: artifacts/delay_trace_<alg>.csv)")
    parser.add_argument("--trace-13-14", action="store_true",
                        help="Shortcut: set delay trace links to '13->14,14->13'")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed. Only matters for algorithms that train during "
                             "the test (DRSIR); everything else loads a checkpoint.")
    parser.add_argument("--greedy", dest="greedy", action="store_true", default=True,
                        help="Force greedy (argmax) inference (default). Real eval "
                             "uses argmax to match sweep selection criterion.")
    parser.add_argument("--stochastic", dest="greedy", action="store_false",
                        help="Force stochastic (sample) inference; overrides --greedy.")
    parser.add_argument("--model", default=None,
                        help="Checkpoint directory to evaluate, e.g. "
                             "results/stride/train/<run>/model. Defaults to "
                             "results/<alg>/model, the LIVE directory the most recent "
                             "training overwrote -- so without --model the checkpoint "
                             "under test is whatever trained last.")
    args = parser.parse_args()
    terminal = resolve_terminal(args.terminal)
    print(f"[test] terminal backend: {terminal}")

    env_cfg, alg_cfg, ctrl_cfg = config.get(args.env, args.alg, args.ctrl)
    
    init_result_dirs(alg_cfg)
    init_paths(env_cfg, alg_cfg)
        
    os.environ["ALG_NAME"] = alg_cfg["algs_name"]
    os.environ["ENV_NAME"] = env_cfg["topology"]
    os.environ["NUM_LINK"] = str(env_cfg.get("num_link", 0))  # controller LLDP-completeness audit
    # 2026-07-10: DRSIR reward mode from config (drsir_dd sets 2). Config wins
    # over any shell-set DRSIR_REWARD_CORRECTED so a forgotten export can't
    # silently fall back to mode A (cf. sudo -E EXP silent-fallback pitfall).
    if "drsir_reward_mode" in alg_cfg:
        os.environ["DRSIR_REWARD_CORRECTED"] = str(alg_cfg["drsir_reward_mode"])
        print(f"DRSIR_REWARD_CORRECTED={os.environ['DRSIR_REWARD_CORRECTED']} (from alg config)")

    trace_links = args.delay_trace_links.strip() or os.environ.get("DELAY_TRACE_LINKS", "").strip()
    if args.trace_13_14 and not trace_links:
        trace_links = "13->14,14->13"

    if trace_links:
        os.environ["DELAY_TRACE_LINKS"] = trace_links
        trace_file = args.delay_trace_file.strip() or os.environ.get("DELAY_TRACE_FILE", "").strip()
        if not trace_file:
            project_root = os.path.dirname(os.path.abspath(__file__))
            trace_dir = os.path.join(project_root, "artifacts")
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(trace_dir, f"delay_trace_{alg_cfg['algs_name']}.csv")
        trace_file = os.path.abspath(trace_file)
        os.environ["DELAY_TRACE_FILE"] = trace_file
        raw_trace_file = os.environ.get("LLDP_RAW_TRACE_FILE", "").strip()
        if not raw_trace_file:
            project_root = os.path.dirname(os.path.abspath(__file__))
            trace_dir = os.path.join(project_root, "artifacts")
            os.makedirs(trace_dir, exist_ok=True)
            raw_trace_file = os.path.join(trace_dir, f"lldp_raw_trace_{alg_cfg['algs_name']}.csv")
        raw_trace_file = os.path.abspath(raw_trace_file)
        os.environ["LLDP_RAW_TRACE_FILE"] = raw_trace_file
        print(f"Delay trace enabled for links: {trace_links}")
        print(f"Delay trace file: {trace_file}")
        print(f"LLDP raw trace file: {raw_trace_file}")

    # --- 1. 先建拓樸 ---------------------------------------------------
    print("Building topology ...")
    if args.auto:
        # Same reason as main.py: Mininet's children inherit stdin and one of
        # them leaves the terminal in raw mode. Only safe under --auto, since
        # without it this script asks which traffic matrix to run.
        _fd = os.open(os.devnull, os.O_RDONLY); os.dup2(_fd, 0); os.close(_fd)
    net = env_loader.build_topo(env_cfg)

    # --- 2. 開 Ryu controller -----------------------------------------
    ctrl_proc = spawn_controller(ctrl_cfg, terminal)

    # Pre-DRL wait; monitor additionally waits setting.MONITOR_START_DELAY. See main.py note.
    print("Controller spawned, wait 30 s ...")
    time.sleep(30)

    # Session folder: one timestamp per program launch
    original_algs_name = alg_cfg["algs_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    decode = "greedy" if args.greedy else "sampled"
    session_dir, test_dir = resolve_session_dir(
        args.model, original_algs_name, {**env_cfg, **alg_cfg}, timestamp, decode)
    os.makedirs(session_dir, exist_ok=True)
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid:
        try:
            uid = int(sudo_uid)
            gid = pwd.getpwuid(uid).pw_gid
            os.chown(test_dir, uid, gid)
            os.chown(session_dir, uid, gid)
        except Exception:
            pass
    print(f"Session dir: {session_dir}")

    # Which checkpoint this session was scored from. The directory layout cannot
    # carry it -- a session may legitimately evaluate a checkpoint from another
    # run (transfer tests do exactly that) -- so write it down. The sha256 is what
    # lets anyone confirm provenance later instead of inferring it from timestamps.
    model_dir = args.model or f"./results/{original_algs_name}/model"
    write_ckpt_record(session_dir, model_dir, decode, {**env_cfg, **alg_cfg})

    # --- Build TM iterator: auto mode vs interactive mode ---
    # AUTO_TMS reads from env_cfg.tm_list_test so it adapts per topology:
    #   geant: [3, 10, 12, 14, 21]   (string IDs in cfg → cast to int)
    #   32node: [2, 4, 6, 8, 10]
    # Old behaviour was hard-coded Geant list, broke 32node transfer eval.
    AUTO_TMS = [int(t) for t in env_cfg.get("tm_list_test", [3, 10, 12, 14, 21])]

    if args.auto:
        tm_iter = iter(AUTO_TMS)
        print(f"Auto mode: will run TMs {AUTO_TMS} sequentially "
              f"(from env_cfg['tm_list_test']={env_cfg.get('tm_list_test')})")
    else:
        tm_iter = None  # use interactive input

    while True:
        # --- Get next TM ID ---
        if tm_iter is not None:
            tm_id = next(tm_iter, None)
            if tm_id is None:
                break
            formatted_input = f"{tm_id:02}"
            print(f"\n[auto] Next TM: {formatted_input}")
        else:
            input_ = input("Enter the traffic matrix ID to be tested (e.g., 06) or type QUIT to exit: ").strip()
            if input_.upper() == 'QUIT':
                break
            if not input_:
                print("Input cannot be empty. Please enter a valid ID or QUIT.")
                continue
            try:
                tm_id = int(input_)
                if tm_id < 0:
                    print("Traffic matrix ID must be non-negative.")
                    continue
                formatted_input = f"{tm_id:02}"
            except ValueError:
                print("Invalid input. Please enter a numeric traffic matrix ID (e.g., 06).")
                continue

        # --- 3. 啟動 DRL 訓練 ---------------------------------------------
        print(f"Start DRL for traffic matrix {formatted_input} ...")
        merged_cfg = {**env_cfg, **alg_cfg, **ctrl_cfg}   # SimpleNamespace 給 training()
        if args.seed is not None:
            merged_cfg["seed"] = args.seed
        merged_cfg["tm_id"] = formatted_input
        # eval_sample is what StrideActor actually reads (stochastic = training
        # or eval_sample), so --greedy has to set THIS key. It used to set
        # force_greedy_infer, which nothing reads -- a test then decoded
        # stochastically while its session was named greedy.
        merged_cfg["eval_sample"] = not args.greedy
        if args.model:
            merged_cfg["model_dir"] = args.model

        # Output dir: session_dir/real/{tm_id}
        test_output_dir = f"{session_dir}/real/{formatted_input}"
        os.makedirs(test_output_dir, exist_ok=True)
        if sudo_uid:
            try:
                uid = int(sudo_uid)
                gid = pwd.getpwuid(uid).pw_gid
                os.chown(test_output_dir, uid, gid)
            except Exception:
                pass
        merged_cfg["test_output_dir"] = test_output_dir

        drl_proc = spawn_drl(merged_cfg, 'test_single', terminal)

        # --- 4. 啟動流量 ---------------------------------------------------
        print("Start traffic ...")
        env_loader.start_single_traffic(net, env_cfg, formatted_input)

        # 關閉 DRL 程序
        drl_proc.terminate()
        drl_proc.wait()

    # --- 5. 收尾 -------------------------------------------------------
    # 同 main.py 的 fix：ctrl_proc.terminate() 只殺 gnome-terminal 外殼，
    # 內層 ryu-manager 不會跟著死（`; exec bash` + python binary 名問題）。
    # 用 pkill -f 對 cmdline regex 才抓得到。
    print("Training finished, clean up.")
    net.stop()
    ctrl_proc.terminate()
    try:
        ctrl_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("[test_single_tm] ctrl_proc didn't exit, ignoring")
    subprocess.run(["sudo", "pkill", "-f", "ryu-manager"], check=False)

    # --- 6. Sim-only eval: no longer run from here ------------------------
    # `--auto` used to end by invoking test_sim_only.py with --session-dir, so
    # every session got a sim/<tm>/ half beside real/<tm>/. That call is gone.
    # The sim path is unmaintained and had drifted to covering a single TM, so
    # the two halves of a session were no longer measuring the same thing, and
    # nothing in paper/ ever read the sim half. Run test_sim_only.py by hand if
    # you want it, and treat what it produces as unvalidated.

if __name__ == "__main__":
    main()
