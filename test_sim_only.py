# test_sim_only.py
# Sim-only ablation testing — no Mininet, no Ryu controller.
# Each step: kpath_reset(last_actions) → infer → sim eval via NX metrics.
# State comes entirely from sim graph (rollout_full), never from real net_info.

import argparse, config, os, sys, json, csv, time
import numpy as np
import torch
from datetime import datetime
from types import SimpleNamespace
from utils.init_path import init_paths
import pwd


def init_result_dirs(alg_cfg):
    algs_name = alg_cfg["algs_name"]
    base_dir = os.path.join("results", algs_name)
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid:
        try:
            uid = int(sudo_uid)
            gid = pwd.getpwuid(uid).pw_gid
            os.chown(base_dir, uid, gid)
            os.chown(model_dir, uid, gid)
        except Exception:
            pass


def build_env(merged_cfg):
    """Build GraphEnv-v16 for sim evaluation (same setup as train_loader.testing)."""
    import gym
    from gym.envs.registration import register

    REFRACTOR_ROOT = os.path.abspath(os.path.dirname(__file__))
    SACD_ROOT = os.path.join(
        REFRACTOR_ROOT,
        "A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main",
        "SAC_PL_KP",
    )
    GYM_GRAPH_ROOT = os.path.join(SACD_ROOT, "gym-graph")

    if SACD_ROOT not in sys.path:
        sys.path.insert(0, SACD_ROOT)
    if GYM_GRAPH_ROOT not in sys.path:
        sys.path.insert(0, GYM_GRAPH_ROOT)

    import gym_graph  # noqa: F401

    try:
        gym.spec("GraphEnv-v16")
    except gym.error.UnregisteredEnv:
        register(
            id="GraphEnv-v16",
            entry_point="gym_graph.envs.environment16:GraphEnv",
        )

    env_eval = gym.make("GraphEnv-v16")
    env_eval.seed(9)
    env_eval.use_K_path = True

    dataset_root = os.path.join(
        os.getcwd(),
        "A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main",
        "Enero_datasets", "dataset_sing_top", "data",
        "results_my_3_tops_unif_05-1",
    )

    # Topology-aware dataset selection
    topology = merged_cfg.get("topology", "geant")
    tm_name = merged_cfg.get("tm_name", "")

    if topology == "geant":
        tm_scale = int(merged_cfg.get("tm_scale", 3))
        dataset_folder = "NEW_Geant" if tm_scale == 5 else f"NEW_Geant_s{tm_scale}"
        graph_name = "Geant"
    elif topology == "32node":
        # tm_name: "32nodes_144tm" or "32nodes_24tm"
        if "144" in tm_name:
            dataset_folder = "NEW_32node_144tm"
        else:
            dataset_folder = "NEW_32node_24tm"
        graph_name = "32node"
    else:
        raise ValueError(f"Unknown topology: {topology}")

    dataset_path = os.path.join(dataset_root, dataset_folder, "EVALUATE")
    print(f"[dataset] topology={topology}, graph={graph_name} -> {dataset_path}")

    K = merged_cfg["action_dim"]
    # Override k_paths_file so generate_environment loads the correct topology's paths
    env_eval.k_paths_file_override = merged_cfg["k_paths_file"]
    env_eval.generate_environment(
        dataset_path, graph_name, EPISODE_LENGTH=0, K=K, X=15,
    )
    env_eval.precompute_path_structures_for_routenet()
    env_eval.kpath_init(merged_cfg)

    return env_eval


def chown_if_sudo(*paths):
    sudo_uid = os.environ.get("SUDO_UID")
    if not sudo_uid:
        return
    try:
        uid = int(sudo_uid)
        gid = pwd.getpwuid(uid).pw_gid
        for p in paths:
            if os.path.exists(p):
                os.chown(p, uid, gid)
    except Exception:
        pass


def run_sim_eval(merged_cfg, drl_paths, env_eval, step, out_base,
                 skip_queue_update=False):
    """Sim-only eval: only NX metrics, no real MN metrics.

    skip_queue_update: if True, skip write_load_to_graph and _update_queues.
        Used by --ospf mode where queue is advanced externally with correct
        timeline (1 advance/step instead of 2).
    """
    from loader.train_loader import compute_network_metrics_nx, write_load_to_graph

    tm_id = merged_cfg["tm_id"]
    step_tag = f"_{step}" if step is not None else ""

    # --- transient mode: write load once, advance queue 1 step, read twice ---
    sim_link_csv = f"{out_base}/sim_link_data/{tm_id}{step_tag}_sim_link.csv"
    sim_dir_csv = f"{out_base}/sim_directed_link_data/{tm_id}{step_tag}_sim_link_directed.csv"

    if not skip_queue_update:
        write_load_to_graph(env_eval, drl_paths, merged_cfg)
        if hasattr(env_eval, '_update_queues'):
            env_eval._update_queues()

    avg_delay, avg_loss, avg_tput, mlu, ds = compute_network_metrics_nx(
        env_eval, drl_paths, merged_cfg, directed=False, out_link_csv=sim_link_csv, skip_queue=True,
    )
    avg_delay_d, avg_loss_d, avg_tput_d, mlu_d, ds_d = compute_network_metrics_nx(
        env_eval, drl_paths, merged_cfg, directed=True, out_link_csv=sim_dir_csv, skip_queue=True,
    )

    # --- append to CSVs ---
    sim_csv = f"{out_base}/sim_test/{tm_id}_eval_metrics.csv"
    with open(sim_csv, "a", newline="") as f:
        csv.writer(f).writerow([avg_delay, avg_loss, avg_tput, mlu, ds])

    sim_dir_test_csv = f"{out_base}/sim_directed_test/{tm_id}_eval_metrics_directed.csv"
    with open(sim_dir_test_csv, "a", newline="") as f:
        csv.writer(f).writerow([avg_delay_d, avg_loss_d, avg_tput_d, mlu_d, ds_d])

    # --- snapshot drl_paths ---
    snap_dir = f"{out_base}/drl_paths_snapshots"
    os.makedirs(snap_dir, exist_ok=True)
    with open(os.path.join(snap_dir, f"{tm_id}_{step}_drl_paths.json"), "w") as f:
        json.dump(drl_paths, f, indent=2)

    print(f"  [step {step}] tput={avg_tput_d:.3f}  mlu={mlu_d:.2f}%  ds={ds_d:.4f}")

    return mlu, ds


def main():
    parser = argparse.ArgumentParser(
        description="Sim-only testing (no Mininet / controller)")
    parser.add_argument("--env", required=True, help="e.g. geant, 32node_144tm")
    parser.add_argument("--alg", required=True, help="e.g. stride")
    parser.add_argument("--model", type=str, default=None,
                        help="model dir override (default: results/{alg}/model). "
                             "For transfer test: --model results/stride/model")
    parser.add_argument("--steps", type=int, default=30,
                        help="number of sim steps per TM (default 30)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto mode: run test TMs [3,10,12,14,21] sequentially then exit")
    parser.add_argument("--tms", nargs="+", type=int, default=None,
                        help="Custom TM list (e.g. --tms 0 1 2 3 10 12 14 21). Overrides --auto.")
    parser.add_argument("--session-dir", type=str, default=None,
                        help="Override session output dir (e.g. to write sim/ into an existing "
                             "test_single_tm session for analyze_sim_vs_real)")
    parser.add_argument("--ospf", action="store_true",
                        help="OSPF mode: skip model loading, use shortest paths, and fix "
                             "queue timeline to 2 pre-fill + 1 advance/step (matching real env)")
    parser.add_argument("--chain-log", action="store_true",
                        help="Per-step chain MLU logging: argmax each diffusion-chain logit, "
                             "compute env.compute_metrics_fast(action) to get GT MLU per step. "
                             "Writes chain_mlu CSV alongside sim_test CSV.")
    parser.add_argument("--greedy", dest="greedy", action="store_true", default=None,
                        help="Force greedy (argmax) inference. Useful for "
                             "deterministic eval sweeps.")
    parser.add_argument("--stochastic", dest="greedy", action="store_false",
                        help="Force stochastic (sample) inference. Overrides config.")
    parser.add_argument("--tm-scale-factor", type=float, default=1.0,
                        help="Multiplier applied to env.TM after each TM load. "
                             "Use < 1.0 to scale demands DOWN (e.g. 32node 144tm "
                             "is too saturated; 0.6 brings load closer to Geant). "
                             "Default 1.0 = no scaling.")
    args = parser.parse_args()

    env_cfg, alg_cfg, _ = config.get(args.env, args.alg, "simple_monitor")
    merged_cfg = {**env_cfg, **alg_cfg}

    init_result_dirs(alg_cfg)
    init_paths(env_cfg, alg_cfg)

    # --- Build sim env ---
    print("Building sim environment ...")
    env_eval = build_env(merged_cfg)

    # --- TM scaling hook (intercept clean_and_generate_tm) ---
    # kpath_reset → clean_and_generate_tm only when tm_id changes; intercept
    # so every fresh TM load gets scaled in-place. Idempotent: if scale=1.0,
    # no monkey-patch.
    if args.tm_scale_factor != 1.0:
        _orig_cgtm = env_eval.clean_and_generate_tm
        def _scaled_cgtm(tm_id_):
            r = _orig_cgtm(tm_id_)
            env_eval.TM = env_eval.TM * args.tm_scale_factor
            print(f"[tm-scale] TM {tm_id_} × {args.tm_scale_factor:.3f}")
            return r
        env_eval.clean_and_generate_tm = _scaled_cgtm

    # --- Load agent ---
    from algs import REGISTRY as algs_REGISTRY
    from loader.train_loader import state_to_action, write_load_to_graph

    agents = algs_REGISTRY[merged_cfg["algs_name"]](SimpleNamespace(**merged_cfg))
    if not args.ospf:
        model_path = args.model or f'./results/{merged_cfg["algs_name"]}/model'
        try:
            agents.load_model(model_path)
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            return
        print(f"Model loaded from {model_path}")
    else:
        print("OSPF mode: using shortest paths (no model needed)")

    all_path_list = state_to_action(merged_cfg)
    size = merged_cfg["num_node"] + 1

    # --- Session folder ---
    if args.session_dir:
        session_dir = args.session_dir.rstrip("/")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = f"./results/{alg_cfg['algs_name']}/test/{timestamp}"
    test_dir = os.path.dirname(session_dir)
    os.makedirs(session_dir, exist_ok=True)
    chown_if_sudo(test_dir, session_dir)
    print(f"Session dir: {session_dir}")

    # --- Build TM iterator: auto mode vs interactive mode ---
    # Same 32node/geant fix as test_single_tm.py — read from env_cfg.tm_list_test.
    AUTO_TMS = [int(t) for t in env_cfg.get("tm_list_test", [3, 10, 12, 14, 21])]

    if args.tms:
        tm_iter = iter(args.tms)
        print(f"Custom TM mode: will run TMs {args.tms} sequentially.")
    elif args.auto:
        tm_iter = iter(AUTO_TMS)
        print(f"Auto mode: will run TMs {AUTO_TMS} sequentially.")
    else:
        tm_iter = None  # use interactive input

    while True:
        # --- Get next TM ID ---
        if tm_iter is not None:
            tm_id = next(tm_iter, None)
            if tm_id is None:
                break
            formatted_id = f"{tm_id:02}"
            print(f"\n[auto] Next TM: {formatted_id}")
        else:
            input_ = input(
                "Enter TM ID (e.g. 06) or QUIT: "
            ).strip()

            if input_.upper() == "QUIT":
                break
            if not input_:
                print("Input cannot be empty.")
                continue

            try:
                tm_id = int(input_)
                if tm_id < 0:
                    print("TM ID must be non-negative.")
                    continue
                formatted_id = f"{tm_id:02}"
            except ValueError:
                print("Invalid input. Please enter a numeric TM ID.")
                continue

        merged_cfg["tm_id"] = formatted_id

        # --- Output dir: session_dir/sim/{tm_id} ---
        out_base = f"{session_dir}/sim/{formatted_id}"
        for d in ["sim_test", "sim_directed_test",
                   "sim_link_data", "sim_directed_link_data"]:
            os.makedirs(f"{out_base}/{d}", exist_ok=True)
        chown_if_sudo(out_base)

        # --- CSV headers ---
        sim_csv = f"{out_base}/sim_test/{formatted_id}_eval_metrics.csv"
        sim_dir_csv = f"{out_base}/sim_directed_test/{formatted_id}_eval_metrics_directed.csv"
        for path in [sim_csv, sim_dir_csv]:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "avg_delay", "avg_packet_loss", "avg_throughput",
                    "max_link_utilization", "demand_satisfaction",
                ])

        # =============================================
        # Sim-only test loop
        # =============================================
        print(f"\n=== Sim-only test: TM={formatted_id}, steps={args.steps} ===")

        if args.ospf:
            # ============================================================
            # OSPF mode: correct queue timeline
            # ============================================================
            #
            # Real env timeline (test_single_tm):
            #   t=40  iperf3 starts → SP traffic begins
            #   t=50  Ryu met_1 (1 step SP traffic, but iperf3 ramp-up means
            #         effective excess < 1 step)
            #   t=60  DRL step 0 reads state (~1.6 steps SP effectively accumulated,
            #         not full 2 steps — confirmed by tc backlog 118 pkts vs 146
            #         predicted by 2-step model)
            #   ...each subsequent step: 1 step of traffic
            #
            # Correct pure sim (1-step pre-fill matches empirical data):
            #   clean_and_generate_tm → queue = 0
            #   write_load (SP) → set utilization
            #   _update_queues × 1  → pre-fill (models 1.6-step real ramp-up)
            #   for each step:
            #       _update_queues × 1  → 1 step advance
            #       write CSV (skip_queue_update=True)
            #
            # Total: 1 (pre-fill) + N (steps) queue advances
            #
            # Why default DRL loop is WRONG for OSPF:
            #   kpath_reset → rollout_full → _update_queues (+1)
            #   run_sim_eval → write_load + _update_queues (+1)
            #   = 2 advances/step → queue fills at 2× real speed
            # ============================================================

            # Build OSPF paths (all index 0 = shortest path)
            ospf_paths = {}
            for i in range(1, size):
                ospf_paths[str(i)] = {}
                for j in range(1, size):
                    if i == j:
                        continue
                    ospf_paths[str(i)][str(j)] = [all_path_list[i][j][0]]

            # Reset: load TM, clear queues
            env_eval.clean_and_generate_tm(int(formatted_id))

            # Write SP utilization to graph (constant for all OSPF steps)
            write_load_to_graph(env_eval, ospf_paths, merged_cfg)

            # Pre-fill: 1 step matching real env ramp-up (~1.6 effective steps,
            # but 1 step matches empirical tc backlog at DRL step 0 best)
            env_eval._update_queues()
            print(f"  [pre-fill] 1 step SP queue accumulated")

            for step in range(args.steps):
                # Advance queue 1 step (utilization unchanged for OSPF)
                env_eval._update_queues()

                # Write metrics CSV (skip_queue_update=True: no double-advance)
                run_sim_eval(merged_cfg, ospf_paths, env_eval, step, out_base,
                             skip_queue_update=True)

        else:
            # ============================================================
            # DRL mode: original loop (kpath_reset + infer + run_sim_eval)
            # Note: 2 queue advances/step (rollout_full + run_sim_eval).
            # For DRL this partially models control delay (old + new routing).
            # ============================================================
            chain_csv = None
            if args.chain_log:
                chain_csv = f"{out_base}/sim_test/{formatted_id}_chain_mlu.csv"
                with open(chain_csv, "w", newline="") as f:
                    csv.writer(f).writerow(["env_step", "chain_step", "mlu_argmax"])

            last_actions = None
            for step in range(args.steps):
                env_eval.kpath_reset(
                    tm_id, merged_cfg, last_actions=last_actions,
                )

                info = {
                    "env": env_eval,
                    "mode": "infer",
                    "temperature": merged_cfg.get("eval_temperature", 1.0),
                    "score_mode": merged_cfg.get("score_mode", "conf_x_impact"),
                    "return_chain": args.chain_log,
                }
                if args.greedy is not None:
                    info["stochastic"] = not args.greedy
                with torch.inference_mode():
                    action, info_out = agents.get_action(None, 0.0, **info)

                # --- chain step MLU: argmax each mid-chain logit, compute_metrics_fast ---
                if args.chain_log and info_out.get("chain_logits"):
                    chain_logits = info_out["chain_logits"]
                    prev_delay_snap = (dict(env_eval.prev_link_delay)
                                       if hasattr(env_eval, "prev_link_delay") else None)
                    try:
                        with open(chain_csv, "a", newline="") as f:
                            writer = csv.writer(f)
                            for cs, lg in enumerate(chain_logits):
                                acts_t = lg[0].argmax(dim=-1).cpu().numpy().astype(np.int64)
                                m_t = env_eval.compute_metrics_fast(acts_t, return_vectors=False)
                                mlu_t = float(m_t.get("mlu_gt",
                                                      m_t.get("mlu_clamped",
                                                              m_t.get("mlu", 0.0))))
                                writer.writerow([step, cs, mlu_t])
                    finally:
                        if prev_delay_snap is not None:
                            env_eval.prev_link_delay.clear()
                            env_eval.prev_link_delay.update(prev_delay_snap)

                drl_paths = {}
                agent_index = 0
                for i in range(1, size):
                    drl_paths.setdefault(str(i), {})
                    for j in range(1, size):
                        if i == j:
                            continue
                        chosen = action[agent_index]
                        drl_paths[str(i)][str(j)] = [all_path_list[i][j][chosen]]
                        agent_index += 1

                run_sim_eval(merged_cfg, drl_paths, env_eval, step, out_base)
                last_actions = action

        print(f"=== Done TM={formatted_id}. Results: {out_base} ===\n")


if __name__ == "__main__":
    main()
