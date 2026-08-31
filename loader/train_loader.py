# train_loader.py
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json,ast
from statistics import mean
import torch
from utils import setting
from utils import atomic_io
import copy
import hashlib
import os
import time
import csv
from algs import REGISTRY as algs_REGISTRY
from numpy import random
from types import SimpleNamespace
from datetime import datetime
from functools import reduce
from collections import deque

# --- Reward floors (paper eq. 20-21) --------------------------------------
# The delay and loss components are inverted (1/v) so that "smaller is better"
# becomes "larger is better", which needs a floor on v.
#
# DELAY_FLOOR_MS is not a divide-by-zero guard, it is the measurement's noise
# floor: below about 1.5 ms a difference in reported delay is jitter rather than
# queueing, so treating one path as better than another on that basis would be
# reading noise. Flooring there makes every uncongested path score the same,
# which is the honest answer. It bites often -- roughly 97% of 32-node link
# samples fall under it, with a median of 0 ms -- so this constant is what a
# healthy path's delay component is worth, not an edge case. The congested paths
# the reward exists to separate are far above it.
#
# LOSS_FLOOR_PCT is a guard: loss is a percentage and a clean path gives 0.
DELAY_FLOOR_MS = 1.5
LOSS_FLOOR_PCT = 0.001


random.seed(17)
np.random.seed(17)
paths_metrics_minmax_dict = {}
link_index = {}


def _resolve_dataset_and_topo(config, dataset_root_folder):
    """Resolve (dataset_folder_path, topo_name) from env config.

    Selects the Enero-format graph dataset + topology name that environment16.py
    will load via `<dataset>/TRAIN/<topo>.graph`. Env-flag-driven so --env
    switches the entire dataset/topology, not just Mininet builder + iperf scripts.

    Geant: NEW_Geant_s{tm_scale} (or NEW_Geant if scale=5), topo="Geant".
    32node: NEW_32node_{144tm|24tm} (selected by tm_name), topo="32node".

    Added 2026-05-13 to unblock 32node native training. Previously train_loader
    hardcoded NEW_Geant_s{tm_scale} + topo="Geant" regardless of --env, causing
    numPairs=506 (Geant) when running --env 32node_144tm (which has 992 pairs)
    → IndexError at train_loader.py:907 first forward pass.
    """
    topology = str(config.get("topology", "geant")).lower()
    if topology == "geant":
        tm_scale = int(config.get("tm_scale", 3))
        sub = "NEW_Geant" if tm_scale == 5 else f"NEW_Geant_s{tm_scale}"
        return os.path.join(dataset_root_folder, sub), "Geant"
    if topology == "32node":
        tm_name = str(config.get("tm_name", "32nodes_144tm"))
        if "144tm" in tm_name:
            sub = "NEW_32node_144tm"
        elif "24tm" in tm_name:
            sub = "NEW_32node_24tm"
        else:
            raise ValueError(f"Unknown 32node tm_name: {tm_name!r}")
        return os.path.join(dataset_root_folder, sub), "32node"
    raise ValueError(f"Unknown topology: {topology!r}")

def seed_torch(seed):
    import random as _py_random
    os.environ['PYTHONHASHSEED'] = str(seed)
    _py_random.seed(seed)            # Python stdlib random (replay buffer sampling)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def training(config):
    print("===  TRAINING by train_loader.training===")
    # --- Build GraphEnv (used in ALL modes: sim as simulator, real as struct holder) ---
    import gym
    from gym.envs.registration import register

    REFRACTOR_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    SACD_ROOT = os.path.join(
        REFRACTOR_ROOT,
        "A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main",
        "SAC_PL_KP"
    )

    GYM_GRAPH_ROOT = os.path.join(SACD_ROOT, "gym-graph")

    if SACD_ROOT not in sys.path:
        sys.path.insert(0, SACD_ROOT)
    if GYM_GRAPH_ROOT not in sys.path:
        sys.path.insert(0, GYM_GRAPH_ROOT)

    import gym_graph

    try:
        gym.spec("GraphEnv-v16")
    except gym.error.UnregisteredEnv:
        register(
            id="GraphEnv-v16",
            entry_point="gym_graph.envs.environment16:GraphEnv",
        )

    dataset_root_folder = os.path.join(os.getcwd(), "A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main","Enero_datasets", "dataset_sing_top", "data", "results_my_3_tops_unif_05-1")
    dataset_folder_name, topo_name = _resolve_dataset_and_topo(config, dataset_root_folder)
    print(f"[dataset] topology={config.get('topology')} → {dataset_folder_name} (topo={topo_name})")

    env_train = gym.make("GraphEnv-v16")
    env_train.seed(int(config.get("seed", 17)))
    env_train.use_K_path = True
    # Override k_paths default (which hardcodes dataset/geant_traffic/k_paths.json
    # inside environment16.py:856-859) so 32node/non-Geant topologies pick up
    # their own k_paths.json. Mirrors test_sim_only.py:94 pattern.
    env_train.k_paths_file_override = config["k_paths_file"]

    K = config["action_dim"]
    percentage_demands = 15

    env_train.generate_environment(
        dataset_folder_name + "/TRAIN",
        topo_name,
        EPISODE_LENGTH=0,
        K=K,
        X=percentage_demands
    )
    env_train.precompute_path_structures_for_routenet()
    env_train.kpath_init(config)

    if config.get("sim_training", False) == True:
        print("sim_training mode: using GraphEnv as simulator")

    _started = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_torch(int(config.get("seed", 17)))
    agents = algs_REGISTRY[config["algs_name"]](SimpleNamespace(**config))
    gen_link_index(config)
    print("regular ma training...")
    print("init get mask")
    mask, link_indices = get_mask(config)

    init_minmax_dic(config)
    all_path_list = state_to_action(config)

    size = config["num_node"] + 1
    
    step = 0
    epsilon_ini = config["epsilon_ini"]
    epsilon_final = config["epsilon_final"]
    epsilon = config["epsilon"]
    agent_info_memory = []
    reward_memory = []
    mlu_list = []

    reward_list = []
    reward_list_bwd = []
    reward_list_delay = []
    reward_list_loss = []
    # 2026-08-17: per-step diagnostics, written to results/<alg>/train/<dir>/.
    # These used to exist only in W&B, which made the timing table in
    # paper/figures/timing/ the one artefact that could not be rebuilt from
    # results/ alone. Runs finished before this date have none of these files;
    # their step timings survive only as the cached CSV in that directory.
    step_time_list = []
    training_time_list = []
    epsilon_list = []
    action_k_entropy_list = []
    action_k_topfrac_list = []
    action_k_mode_list = []
    kdiv_list = []
    top1_list = []
    entropy_px0_list = []
    loss_logs = {}
    
    if config.get("rnn", False):
        agents.hidden_states = agents.init_hidden(agents.actor, 1)
    
    print(config)

    if (config.get("sim_training", False) == False):
        waiting_time = 30
        print("waiting ",waiting_time," second, then start training")
        time.sleep(waiting_time)
    

    tm_duration_steps =  config["tm_duration_training"] // 10

    # 把 regular ma 跟 ls2ic_nx training 流程切開
    alg_name = config.get("algs_name", "ls2ic")

    if alg_name == "ls2ic_nx":
        # ================================================================
        # ls2ic_nx: sim training with real-Mininet-style 2-step delay
        #   (s_{t-1}, a_{t-2}, r_t, s_t) — identical to real Mininet
        #   only difference: env is GraphEnv sim instead of real Mininet
        # ================================================================
        print("====== ls2ic_nx sim training (2-step delay) ======")
        out_dir = f"./results/{config['algs_name']}"
        os.makedirs(out_dir, exist_ok=True)


        # Build initial default drl_paths (k=0 for all pairs, used before
        # any agent action is available)
        initial_drl_paths = {}
        for i in range(1, size):
            initial_drl_paths[str(i)] = {}
            for j in range(1, size):
                if i == j:
                    continue
                initial_drl_paths[str(i)][str(j)] = [all_path_list[i][j][0]]

        # Write initial drl_paths so first reset_and_get_state_by_NX works
        model_result_dir = os.path.join(os.getcwd(), "results", config["algs_name"])
        drl_paths_file = os.path.join(model_result_dir, "drl_paths.json")
        atomic_io.dump_json(drl_paths_file, initial_drl_paths)

        # Action buffer: stores drl_paths dicts from each step
        # At step t, env is allocated with drl_paths_buffer[t-2] (2-step delay)
        drl_paths_buffer = []

        while True:
            time_in = time.time()
            step += 1

            # ---- TM rotation ----
            tm_index = (step // tm_duration_steps) % len(config["tm_list_train"])
            tm_id = config["tm_list_train"][tm_index]

            # ---- Write DELAYED drl_paths (a_{t-2}) to file ----
            # step 1,2: no agent action from 2 steps ago → use initial routing
            # step 3+:  use drl_paths_buffer[-2]
            if len(drl_paths_buffer) >= 2:
                delayed_paths = drl_paths_buffer[-2]
            else:
                delayed_paths = initial_drl_paths
            # Atomic: the controller reads this file on its own schedule, and a
            # plain open('w') leaves it truncated for the length of the write.
            # See utils/atomic_io.py -- this is what killed a 32-node run.
            atomic_io.dump_json(drl_paths_file, delayed_paths)

            # ---- Get state (reflects delayed action a_{t-2}) ----
            state, mlu, global_state = env_train.reset_and_get_state_by_NX(
                config, mask, model_result_dir, tm_id
            )

            # ---- Get reward (also reflects a_{t-2}) ----
            all_reward, all_reward_indicator, loss_value_path, delay_value_path = \
                path_metrics_to_reward_sim(env_train, config)

            # ---- Agent forward ----
            if config.get("use_global_state", False):
                input_state = global_state
            else:
                input_state = state

            drl_paths = {}
            info = build_info(agents, input_state, epsilon, config, drl_paths)
            action, output_info = agents.get_action([input_state], epsilon, **info)

            # ---- loop_pairs: build current drl_paths + extract delayed reward ----
            (reward_all, reward_bwd, reward_delay, reward_loss, agent_reward_list) = loop_pairs(
                config, size, action, step,
                all_reward, all_reward_indicator,
                all_path_list, drl_paths,
                agent_info_memory, reward_memory
            )

            # ---- Store agent info ----
            agent_info = {**info, **output_info}
            agent_info["input_state"] = input_state
            agent_info["action"] = action
            agent_info_memory.append(agent_info)
            reward_memory = agent_reward_list

            # ---- Buffer current drl_paths (DO NOT write to file — delay by 2) ----
            drl_paths_buffer.append(copy.deepcopy(drl_paths))
            if len(drl_paths_buffer) > 4:
                drl_paths_buffer.pop(0)  # keep only recent entries

            # ---- Training (same delay alignment as real Mininet) ----
            # At step >= 3: buffer = (s_{t-1}, a_{t-2}, r_t, s_t)
            if step >= 3:
                agents.append_sample(agent_info_memory, input_state, agent_reward_list)
                agent_info_memory.pop(0)

            if len(agents.memory) > agents.batch_size:
                loss_dict = agents.update()

                if step % config["softupdate_freq"] == 0:
                    agents.update_target()

                for name in loss_logs:
                    if name not in loss_dict:
                        loss_dict[name] = None
                for name in loss_dict:
                    if name not in loss_logs:
                        loss_logs[name] = [None] * step
                for name, value in loss_dict.items():
                    loss_logs[name].append(float(value) if value is not None else None)
                flush_logs(out_dir, {f"{name}.txt": values for name, values in loss_logs.items()})


            # ---- Logging ----
            # Logged before the finish check below, not after it. The other way
            # round, the final step trained and was never written down: a run of
            # total_timestep=3000 left 2999 lines, because step 3000 reached the
            # check and returned before it reached this block. Every algorithm
            # lost its last step the same way, so nothing was unfair -- it was
            # just a step of training with no record of it.
            output_all_path = os.path.join(out_dir, "output_all.txt")
            save_stepwise_log(output_all_path, agent_reward_list, step)

            reward_list.append(float(reward_all))
            mlu_list.append(float(mlu))
            reward_list_bwd.append(int(reward_bwd))
            reward_list_delay.append(int(reward_delay))
            reward_list_loss.append(int(reward_loss))

            flush_logs(out_dir, {
                "output.txt":          reward_list,
                "training_mlu.txt":    mlu_list,
                "output_bwd.txt":      reward_list_bwd,
                "output_delay.txt":    reward_list_delay,
                "output_loss.txt":     reward_list_loss,
            })

            # ---- Save model ----
            if step == config["total_timestep"]:
                model_path = f'./results/{config["algs_name"]}/model'
                agents.save_model(model_path)
                archive_run_outputs(out_dir, config, started=_started)
                return

            print(f"step {step} | tm={tm_id} | mlu={mlu:.4f} | reward={reward_all:.1f} | "
                  f"eps={epsilon:.4f} | delay_buf={len(drl_paths_buffer)} | "
                  f"time={time.time()-time_in:.2f}s")

            # ---- Epsilon decay ----
            if epsilon > 0.1:
                epsilon -= (epsilon_ini - 0.1) / config["epsilon_first_phase"]
            elif epsilon > epsilon_final:
                epsilon -= (0.1 - epsilon_final) / config["epsilon_second_phase"]

    else:
        # regular ma training (original ls2ic, real or sim)
        print("======regular ma training======")
        out_dir = f"./results/{config['algs_name']}"


        while True:
            time_in = time.time()
            step += 1
            
            # ==== get state ====
            if (config.get("sim_training", False) == False):
                state, mlu, global_state = get_state(config, mask, link_indices, step=step)
                check_controller_alive(config, step)
            else:
                tm_index = (step // tm_duration_steps) % len(config["tm_list_train"])
                # mod 是因為 tm_duration_step 可能會不是整數被裁小 如果不用 mod 循環可能會爆出 tm_list_train
                tm_id = config["tm_list_train"][tm_index]
                print(f"=== TM ID: {tm_id} ===")
                env_start_time = time.time()
                model_result_dir = os.path.join(os.getcwd(), "results", config["algs_name"])
                state, mlu, global_state = env_train.reset_and_get_state_by_NX(config, mask, model_result_dir, tm_id)
                print(f"env time:{time.time() - env_start_time}")



            if (config.get("sim_training", False) == False):
                # 2026-05-24: dispatch to directed-reward variant when knob set.
                # path_metrics_to_reward_directed reads net_info_directed.csv (tc_delay
                # + per-direction bw/loss) instead of paths_metrics.json (LLDP delay +
                # undirected).  State path independently respects state_directed knob.
                if config.get("reward_directed", False):
                    print("real_training (reward_directed)")
                    all_reward,all_reward_indicator, loss_value_path, delay_value_path = path_metrics_to_reward_directed(config)
                else:
                    print("real_training")
                    all_reward,all_reward_indicator, loss_value_path, delay_value_path = path_metrics_to_reward(config)
            else:
                print("sim_training")
                all_reward,all_reward_indicator, loss_value_path, delay_value_path = path_metrics_to_reward_sim(env_train, config)
                
            drl_paths = {}
            agent_info = {}
            start_time = time.time()
            
            if config.get("use_global_state", False):
                input_state = global_state
            else:
                input_state = state
            #print(input_state)
            info = build_info(agents, input_state, epsilon, config, drl_paths)
            action, output_info = agents.get_action([input_state], epsilon, **info)

            (reward_all, reward_bwd, reward_delay, reward_loss, agent_reward_list) = loop_pairs(
                config, size, action, step,
                all_reward, all_reward_indicator,
                all_path_list, drl_paths,
                agent_info_memory, reward_memory
            )
            
            print("get action:", time.time()-start_time)

            agent_info = {**info, **output_info}
            agent_info["input_state"] = input_state
            agent_info["action"] = action

            agent_info_memory.append(agent_info)
            reward_memory = agent_reward_list
            out_dir = f"./results/{config['algs_name']}"
            os.makedirs(out_dir, exist_ok=True)
            
            # Renaming a fresh file over the target needs write permission on the
            # directory, not on the file, so this also heals the root-owned
            # drl_paths.json a prior sudo run leaves behind -- which used to need
            # an explicit remove first (2026-06-27).
            _drl_paths_file = os.path.join(out_dir, "drl_paths.json")
            atomic_io.dump_json(_drl_paths_file, drl_paths)

            if step >= 3:
                agents.append_sample(agent_info_memory, input_state, agent_reward_list)
                # agent_info_memory 裡面存的是step-1 step-2 的 2筆歷史資料 (s,a)
                # append_sample 裡面還要對齊  state_tm1,action_tm2,input_state(就是t),agent_reward_list(t)
                # 有 action delay、reward delay 的長 （s1,a0,r2,s2）
                # sim 的就是                          (s1,a1,r2,s2)
                # 正常來講sim 可以從 (s0,a0,r1,s1) 但懶的搞了 直接在 agent append_sample裡面把s跟a對齊就好 就是從(s1,a1,r2,s2)開始
                agent_info_memory.pop(0)
                
            # 2026-05-26: track training_time separately so we can log it
            # whether or not update() fires this step (warmup vs steady state).
            training_time = 0.0
            loss_dict = {}
            if len(agents.memory) > agents.batch_size:
                start_time = time.time()
                loss_dict = agents.update()
                training_time = time.time() - start_time
                print("update time:", training_time)

                if step % config["softupdate_freq"] == 0:
                    agents.update_target()

                for name in loss_logs:
                    if name not in loss_dict:
                        loss_dict[name] = None

                for name in loss_dict:
                    if name not in loss_logs:
                        loss_logs[name] = [None] * step

                for name, value in loss_dict.items():
                    loss_logs[name].append(float(value) if value is not None else None)

                flush_logs(out_dir, {f"{name}.txt": values for name, values in loss_logs.items()})

            # Logged before the finish check below, for the same reason as the
            # loop above: step total_timestep used to train and then return
            # without ever being written, leaving N-1 lines for N steps.
            output_all_path = os.path.join(out_dir, "output_all.txt")
            save_stepwise_log(output_all_path, agent_reward_list, step)
            
            reward_list.append(float(reward_all))
            mlu_list.append(float(mlu))
            reward_list_bwd.append(int(reward_bwd))
            reward_list_delay.append(int(reward_delay))
            reward_list_loss.append(int(reward_loss))
            
            flush_logs(out_dir, {
                "output.txt":          reward_list,
                "training_mlu.txt":    mlu_list,
                "output_bwd.txt":      reward_list_bwd,
                "output_delay.txt":    reward_list_delay,
                "output_loss.txt":     reward_list_loss,
            })

            if step == config["total_timestep"]:
                model_path = f'./results/{config["algs_name"]}/model'
                agents.save_model(model_path)
                archive_run_outputs(out_dir, config, started=_started)
                # Write the .drl_done sentinel so env_loader.start_traffic stops
                # cycling TMs -- it polls for this file at every TM boundary and
                # every 5 s mid-TM. Without it the DRL side finishes and returns
                # while the gnome-terminal traffic generator loops forever, and
                # main.py hangs waiting
                # for child gnome-terminal -> chain stuck (root cause of PC1 b8
                # ugovavzw hang and PC0 b8_pma round 2 cycle observed 2026-05-28).
                sentinel = f'./results/{config["algs_name"]}/.drl_done'
                with open(sentinel, "w") as f:
                    f.write(str(step))
                return
            if (config.get("sim_training", False) == True):
                print(f"episode time : {time.time()-time_in} sec")
            print("--------------------------------------- step %d --------------------------------------" % step)
            print("--------------------------------------  epsilon  %f ----------------------------------" % epsilon)
            time_end = time.time()
            step_time = time_end - time_in
            print(f"step_time: {step_time:.3f}s  training_time: {training_time:.3f}s")

            # --- per-step diagnostics -------------------------------------
            # Flushed separately from the block above because step_time is only
            # known here, after it. Line index i is step i+1, same as every other
            # log in this directory.
            #
            # 2026-08-17: these used to go to W&B only. Everything worth keeping
            # is now a plain .txt beside output_all.txt, so a run is fully
            # readable from results/ alone. Scalars are one value per line;
            # per-denoise-step series are one list per line, like output_all.txt.
            #
            # training_time is 0.0 on warmup steps where agents.update() did not
            # fire — those rows are kept, the reader filters them.
            step_time_list.append(round(step_time, 6))
            training_time_list.append(round(training_time, 6))
            diag_logs = {
                "step_time.txt":     step_time_list,
                "training_time.txt": training_time_list,
            }

            # epsilon only when the run actually explores. STRIDE's _BASE pins it
            # to 0.0, so writing a file of zeros for every canonical run would
            # just be misleading; the ls2ic/ps_dqn baselines and the two
            # *_eps_both variants decay it for real and do get the file.
            if float(config.get("epsilon_ini", 0.0)) > 0.0:
                epsilon_list.append(round(float(epsilon), 6))
                diag_logs["epsilon.txt"] = epsilon_list

            # Cross-pair action diversity (2026-05-28) -- detects the degenerate
            # mode where every pair collapses to one candidate k. entropy/ln(K) ~ 0
            # with topfrac ~ 1 means a rigid, state-insensitive policy; a healthy
            # one sits above ~0.5.
            try:
                _act = np.asarray(action).ravel()
                if _act.size and np.issubdtype(_act.dtype, np.integer):
                    _K = int(config.get("action_dim", 20))
                    _hist = np.bincount(_act, minlength=_K).astype(np.float64)
                    _p = _hist / max(_hist.sum(), 1.0)
                    _nz = _p[_p > 0]
                    _maxent = np.log(_K) if _K > 1 else 1.0
                    action_k_entropy_list.append(float(-(_nz * np.log(_nz)).sum() / _maxent))
                    action_k_topfrac_list.append(float(_p.max()))
                    action_k_mode_list.append(int(_hist.argmax()))
                    diag_logs["action_k_entropy.txt"] = action_k_entropy_list
                    diag_logs["action_k_topfrac.txt"] = action_k_topfrac_list
                    diag_logs["action_k_mode.txt"]    = action_k_mode_list
            except (TypeError, ValueError):
                pass  # adaptive_dijkstra returns a dict, not an int array -- skip

            # Per-denoise-step series, one list per line (M entries each):
            #   kdiv       entropy of chosen-k at each denoise step -- shows the
            #              collapse trajectory, whether the policy starts diverse
            #              at m0 and collapses by m7 or is anchor-dominated from
            #              the start.
            #   top1       mean per-pair top-1 mass of p_x0 (peakedness; low means
            #              multi-modal, where soft-vs-argmax self_cond matters).
            #   entropy    mean per-pair Shannon entropy of p_x0 / ln(K) (spread
            #              of the whole distribution, complements top1).
            if isinstance(output_info, dict):
                for _key, _lst, _fname in (
                        ("step_kdiv",    kdiv_list,       "kdiv.txt"),
                        ("step_top1",    top1_list,       "top1.txt"),
                        ("step_entropy", entropy_px0_list, "entropy_px0.txt")):
                    _series = output_info.get(_key)
                    if _series:
                        _lst.append([round(float(_e), 6) for _e in _series])
                        diag_logs[_fname] = _lst

            flush_logs(out_dir, diag_logs)

            if epsilon > 0.1:
                epsilon -= (epsilon_ini - 0.1)/config["epsilon_first_phase"]
            elif epsilon > epsilon_final:
                epsilon -= (0.1 - epsilon_final)/config["epsilon_second_phase"]
            if (config.get("sim_training", False) == False):
                if time_end - time_in < 10 :
                    time.sleep(10 - (time_end - time_in))
 
def testing(config):
    # test single TM
    # 只會從 test_single_tm.py 呼叫
    # main.py 裡面的 testing 會跑到 test_anime 那個現在沒在用
    # 目前裡面串進來的dataset 沒用config 裡面的而是直接寫死 
    print(config)
    dataset_root_folder = os.path.join(os.getcwd(), "A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main","Enero_datasets", "dataset_sing_top", "data", "results_my_3_tops_unif_05-1")
    dataset_folder_name, topo_name = _resolve_dataset_and_topo(config, dataset_root_folder)
    print(f"[dataset] topology={config.get('topology')} → {dataset_folder_name} (topo={topo_name})")

    # ---------- 1. 建 SACD 專用的執行決策用的 graph env 來給定 action 以在 MN 中 eval----------
    print("sim_training:importing gym and SACD...")
    import gym
    from gym.envs.registration import register
    REFRACTOR_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    SACD_ROOT = os.path.join(
        REFRACTOR_ROOT,
        "A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main",
        "SAC_PL_KP"
    )
    GYM_GRAPH_ROOT = os.path.join(SACD_ROOT, "gym-graph")
    # 1. add SACD root
    if SACD_ROOT not in sys.path:
        sys.path.insert(0, SACD_ROOT)
    # 2. add gym-graph root (THIS WAS MISSING)
    if GYM_GRAPH_ROOT not in sys.path:
        sys.path.insert(0, GYM_GRAPH_ROOT)
    import gym_graph
    # 3. register env
    try:
        gym.spec("GraphEnv-v16")
    except gym.error.UnregisteredEnv:
        register(
            id="GraphEnv-v16",
            entry_point="gym_graph.envs.environment16:GraphEnv",
        )
    env_eval = gym.make("GraphEnv-v16")
    env_eval.seed(int(config.get("seed", 17)) + 1000)
    env_eval.use_K_path = True
    env_eval.k_paths_file_override = config["k_paths_file"]

    K = config["action_dim"]
    percentage_demands = 15

    env_eval.generate_environment(
        dataset_folder_name + "/EVALUATE",
        topo_name,
        EPISODE_LENGTH=0,
        K=K,
        X=percentage_demands
    )
    env_eval.precompute_path_structures_for_routenet()
    env_eval.kpath_init(config)

    if config["algs_name"] == "sacd_nx":
        return testing_sacd_nx(config, env_eval)
    else:
        return testing_ma(config, env_eval)

def testing_sacd_nx(config, env_eval):
    # SACD lives in the vendored SAC_PL_KP tree, which this repository does not
    # ship -- there is no sacd_nx config, so nothing reaches here. The import
    # stays inside the one function that uses it, so a missing SACD breaks that
    # path and not every training and test run.
    from SACD import SACD

    test_step=config.get("test_step", 30)
    # ---------- 2. load 已訓練好的 SACD model ----------
    hyper_parameter = {
        'feature_size': 20,
        't': 4,
        'readout_units': 20,
        'episode': 20,
        'a_lr': 0.0003,
        'c_lr': 0.0015,
        'gamma': 0.99,
        'alpha': 0.2,
        'batch_size': 55,
        'buffer_size': 10000,
        'update_freq': 100,
        'update_times': 10,
        'max_a_dim': 20,
        'avg_a_dim': 20
    }
    SACD_Agent = SACD(hyper_parameter)
    SACD_Agent.K_path = config["action_dim"]
    SACD_Agent.target_entropy = 0.5 * (-np.log(1 / K))
    model_dir = os.path.join("A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main",
                              "SAC_PL_KP", "models")
    
    
    # 切換model的位置
    # model_dir = os.path.join(model_dir, "Enero_3top_15_B_SAC2025-12-31_01-31-35trained_with_Geant")
    model_dir = os.path.join(model_dir, "Enero_3top_15_B_SAC2026-01-05_04-06-35")
    model_file_name = "actor_23.pt"

    SACD_Agent.actor.load_state_dict(torch.load(os.path.join(model_dir, model_file_name)))
    SACD_Agent.actor.eval()

    print("load model success....")

    waiting_time = 30
    step = 0
    print("waiting ", waiting_time, " second, then start testing")
    time.sleep(waiting_time)
    
    while True:
        time_in = time.time()
        # ---------- 3. reset TM ----------
        tm_id = int(config["tm_id"])
        demand, source, destination = env_eval.reset(tm_id=tm_id)

        # ---------- 4. 跑「完整 episode」，收集 routing ----------
        print("Running SACD episode to collect routing plan ...")
        routing_plan = {}  # (src, dst) -> chosen path index
        while True:
            with torch.no_grad():
                action_dist, _ = SACD_Agent.predict(env_eval, source, destination, demand)
                action = torch.argmax(action_dist).item()

            # 記錄這個 flow 的決策
            # 0-based
            routing_plan[(source, destination)] = action

            # SACD env step（只在 graph env 裡）
            reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_eval.step(
                action, demand, source, destination)
            if done:
                break
            
        # ---------- 5. 轉成 Mininet 要的 drl_paths ----------
        # 1-based
        drl_paths = convert_sacd_plan_to_drl_paths(routing_plan, env_eval)


        # ---------- 6. test ----------
        run_eval(
            config=config,
            drl_paths=drl_paths,
            env_eval=env_eval,
            step=step,
            transient=True,
        )
        step += 1
        if step == test_step:
            return
        time_end = time.time()
        if time_end - time_in < setting.MONITOR_PERIOD:
            time.sleep(setting.MONITOR_PERIOD - (time_end - time_in))

# 原本的 testing 都是MA類型 現在被拆成 SA MA
def testing_ma(config, env_eval):
    test_step=config.get("test_step", 30)
    size = config["num_node"] + 1

    agents = algs_REGISTRY[config["algs_name"]](SimpleNamespace(**config))
    print("load model...")
    try:
        # model_dir lets the caller name the checkpoint explicitly. Without it
        # this reads results/<alg>/model, the live directory every training run
        # overwrites -- so the checkpoint under test is whatever finished last.
        model_path = config.get("model_dir") or f'./results/{config["algs_name"]}/model'
        print(f"  ckpt: {model_path}")
        agents.load_model(model_path)
    except Exception as e:
        print(model_path)
        print("No model, have to train model first")
        print(e)
        return
                
    print("load model success....")
    all_path_list = state_to_action(config) 
    waiting_time = 30
    print("waiting ",waiting_time," second, then start testing (testing_ma)")
    time.sleep(waiting_time)

    step = 0
    if config.get("rnn", False):
        agents.hidden_states = agents.init_hidden(agents.actor, 1)

    # --- make env_eval.TM available for NX metric ---
    tm_id = int(config["tm_id"])

    # =========================
    # ===== REGULAR MA =========
    # =========================
    # regular MA needs mask/state from net_info.csv
    mask, link_indices = get_mask(config)

    # prepare TM once (or you can refresh each step if you want)
    env_eval.clean_and_generate_tm(tm_id=tm_id)

    while True:
        time_in = time.time()
        state, _, global_state = get_state(config, mask, link_indices)
        drl_paths = {}
        agent_index = 0

        if config.get("use_global_state", False):
            input_state = global_state
        else:
            input_state = state
        # path-feat head extras (mirror training-loop logic at line 413-421).
        info = build_info(agents, input_state, 0.0, config, drl_paths={})
        action, _ = agents.get_action([input_state], 0.0, **info)
        for i in range(1, size):
            drl_paths.setdefault(str(i), {})
            for j in range(1, size):
                if i != j:
                    if config['algs_name'] == 'adaptive_dijkstra':
                        drl_paths[str(i)][str(j)] = action[i][j]
                    else:
                        chosen = action[agent_index]
                        drl_paths[str(i)][str(j)] = [all_path_list[i][j][chosen]]
                    agent_index += 1
        # (4) eval — transient mode: queue advances 1 step per call, not reset+30
        run_eval(config=config, drl_paths=drl_paths, env_eval=env_eval, step=step, transient=True)
        step += 1
        if step == test_step:
            return
        time_end = time.time()
        if time_end - time_in < setting.MONITOR_PERIOD:
            time.sleep(setting.MONITOR_PERIOD - (time_end - time_in))

def run_eval(
    config,
    drl_paths,
    env_eval=None,
    step=None,
    transient=False,
):
    print(">>> writing metrics for TM =", config["tm_id"])

    # Output base directory: use test_output_dir if set (ablation testing), else default
    out_base = config.get("test_output_dir", f"./results/{config['algs_name']}")

    # ----- ensure folders -----
    os.makedirs(f"{out_base}/real_test", exist_ok=True)
    os.makedirs(f"{out_base}/real_directed_test", exist_ok=True)
    os.makedirs(f"{out_base}/sim_test", exist_ok=True)
    os.makedirs(f"{out_base}/sim_directed_test", exist_ok=True)
    os.makedirs(f"{out_base}/sim_link_data", exist_ok=True)
    os.makedirs(f"{out_base}/sim_directed_link_data", exist_ok=True)
    
    # ---------- 0. metrics CSV (mininet)----------
    # ---------- 0. metrics CSV (networkX)----------
    metrics_csv = f"{out_base}/real_test/{config['tm_id']}_eval_metrics.csv"
    metrics_directed_csv = f"{out_base}/real_directed_test/{config['tm_id']}_eval_metrics.csv"
    metrics_NX_csv = f"{out_base}/sim_test/{config['tm_id']}_eval_metrics.csv"
    metrics_NX_directed_csv = f"{out_base}/sim_directed_test/{config['tm_id']}_eval_metrics_directed.csv"
    
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["avg_delay", "avg_packet_loss", "avg_throughput", "max_link_utilization", "demand_satisfaction"])
    if not os.path.exists(metrics_directed_csv):
        with open(metrics_directed_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # avg_delay = tc-based queue truth; avg_delay_lldp kept as diagnostic.
            # demand_satisfaction left blank in real (only NX has oracle TM).
            writer.writerow(["avg_delay", "avg_packet_loss", "avg_throughput",
                             "max_link_utilization", "demand_satisfaction", "avg_delay_lldp"])
    if not os.path.exists(metrics_NX_csv):
        with open(metrics_NX_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["avg_delay", "avg_packet_loss", "avg_throughput", "max_link_utilization", "demand_satisfaction"])
    if not os.path.exists(metrics_NX_directed_csv):
        with open(metrics_NX_directed_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["avg_delay", "avg_packet_loss", "avg_throughput", "max_link_utilization", "demand_satisfaction"])

    # dump routing: always write to original location (controller reads it)
    orig_base = f"./results/{config['algs_name']}"
    drl_paths_path = f"{orig_base}/drl_paths.json"
    # 2026-05-28: a root-context process can leave drl_paths.json root-owned
    # between init and this user-context (gnome SUDO_USER) write -> open('w')
    # PermissionError -> eval aborts after the CSV header (header-only CSVs,
    # ILP real-test incident). orig_base is user-owned, so removing the file
    # first (needs dir write perm, NOT file ownership) lets us recreate it
    # fresh regardless of who last wrote it.
    atomic_io.dump_json(drl_paths_path, drl_paths)
    if step is not None:
        drl_snap_dir = f"{out_base}/drl_paths_snapshots"
        os.makedirs(drl_snap_dir, exist_ok=True)
        atomic_io.dump_json(
            os.path.join(drl_snap_dir, f"{config['tm_id']}_{step}_drl_paths.json"),
            drl_paths)
    print("Dumped DRL paths")

    step_tag = f"_{step}" if step is not None else ""

    # ----- snapshot real net_info (跟 sim 同一時刻的 real 資料) -----
    # net_info.csv 來源還是在原始 results/{algs_name}/ 下（controller 寫的）
    import shutil
    orig_base = f"./results/{config['algs_name']}"
    for src_name, dst_folder in [
        ("net_info.csv", "real_link_data"),
        ("net_info_directed.csv", "real_directed_link_data"),
    ]:
        src_path = f"{orig_base}/{src_name}"
        dst_dir = f"{out_base}/{dst_folder}"
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src_path):
            dst_path = os.path.join(dst_dir, f"{config['tm_id']}{step_tag}_{src_name}")
            shutil.copy2(src_path, dst_path)

    # ----- real -----
    avg_delay, avg_packet_loss, avg_throughput, max_link_utilization = \
        compute_network_metrics(config,directed=False)
    # ----- real directed -----
    avg_delay_directed, avg_delay_lldp_directed, avg_packet_loss_directed, avg_throughput_directed, max_link_utilization_directed = \
        compute_network_metrics(config, directed=True)
    # ----- sim  -----
    sim_link_csv = f"{out_base}/sim_link_data/{config['tm_id']}{step_tag}_sim_link.csv"
    sim_dir_link_csv = f"{out_base}/sim_directed_link_data/{config['tm_id']}{step_tag}_sim_link_directed.csv"

    if transient and env_eval is not None:
        # Transient mode (test_single_tm): write load once, advance queue 1 step,
        # then read metrics twice (undirected + directed) from the same graph state.
        write_load_to_graph(env_eval, drl_paths, config)
        if hasattr(env_eval, '_update_queues'):
            env_eval._update_queues()
        avg_delay_NX, avg_loss_NX, avg_tput_NX, mlu_NX, ds_NX = \
            compute_network_metrics_nx(env_eval, drl_paths, config, directed=False, out_link_csv=sim_link_csv, skip_queue=True)
        avg_delay_NX_d, avg_loss_NX_d, avg_tput_NX_d, mlu_NX_d, ds_NX_d = \
            compute_network_metrics_nx(env_eval, drl_paths, config, directed=True, out_link_csv=sim_dir_link_csv, skip_queue=True)
    else:
        # Steady-state mode (training eval): reset queue + 30 steps inside each call.
        avg_delay_NX, avg_loss_NX, avg_tput_NX, mlu_NX, ds_NX = \
            compute_network_metrics_nx(env_eval, drl_paths, config, directed=False, out_link_csv=sim_link_csv)
        avg_delay_NX_d, avg_loss_NX_d, avg_tput_NX_d, mlu_NX_d, ds_NX_d = \
            compute_network_metrics_nx(env_eval, drl_paths, config, directed=True, out_link_csv=sim_dir_link_csv)

    # ----- log -----
    with open(metrics_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            avg_delay, avg_packet_loss, avg_throughput, max_link_utilization, None
        ])
    with open(metrics_directed_csv, 'a', newline='') as csvfile_directed:
        writer = csv.writer(csvfile_directed)
        writer.writerow([
            avg_delay_directed, avg_packet_loss_directed, avg_throughput_directed,
            max_link_utilization_directed, None, avg_delay_lldp_directed
        ])
    with open(metrics_NX_csv, 'a', newline='') as csvfileNX:
        writerNX = csv.writer(csvfileNX)
        writerNX.writerow([
            avg_delay_NX, avg_loss_NX, avg_tput_NX, mlu_NX, ds_NX
        ])
    with open(metrics_NX_directed_csv, 'a', newline='') as csvfileNX_d:
        writerNX_d = csv.writer(csvfileNX_d)
        writerNX_d.writerow([
            avg_delay_NX_d, avg_loss_NX_d, avg_tput_NX_d, mlu_NX_d, ds_NX_d
        ])  

    print(
        "Eval metrics: delay {:.3f}, loss {:.3f}, throughput {:.3f}, max util {:.3f}%"
        .format(avg_delay, avg_packet_loss, avg_throughput, max_link_utilization)
    )
    print(
        "Eval metrics directed: delay_tc {:.3f} (lldp {:.3f}), loss {:.3f}, throughput {:.3f}, max util {:.3f}%"
        .format(avg_delay_directed, avg_delay_lldp_directed, avg_packet_loss_directed,
                avg_throughput_directed, max_link_utilization_directed)
    )
    print(
        "Eval metrics NX: delay {:.3f}, loss {:.3f}, throughput {:.3f}, max util {:.3f}%, ds {:.3f}"
        .format(avg_delay_NX, avg_loss_NX, avg_tput_NX, mlu_NX, ds_NX)
    )
    print(
        "Eval metrics NX directed: delay {:.3f}, loss {:.3f}, throughput {:.3f}, max util {:.3f}%, ds {:.3f}"
        .format(avg_delay_NX_d, avg_loss_NX_d, avg_tput_NX_d, mlu_NX_d, ds_NX_d)
    )
    
def testing_anime(config):
    
    size = config["num_node"] + 1

    agents = algs_REGISTRY[config["algs_name"]](SimpleNamespace(**config))
    print("load model...")
    try:
        # model_dir lets the caller name the checkpoint explicitly. Without it
        # this reads results/<alg>/model, the live directory every training run
        # overwrites -- so the checkpoint under test is whatever finished last.
        model_path = config.get("model_dir") or f'./results/{config["algs_name"]}/model'
        print(f"  ckpt: {model_path}")
        agents.load_model(model_path)
    except Exception as e:
        print("No model, have to train model first")
        return
                
    print("load model success....")
    all_path_list = state_to_action(config) 
    print("Start eval")

    mask, link_indices = get_mask(config)

    # path-feat head setup (2026-05-26): mirror testing_ma path.

    metrics_csv = f"./results/{config['algs_name']}/anime_eval_metrics.csv"
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["avg_delay", "avg_packet_loss", "avg_throughput", "max_link_utilization"])
    
    step = 0
    if config["rnn"]:
        agents.hidden_states = agents.init_hidden(agents.actor, 1)
    
    waiting_time = 30
    print("waiting ",waiting_time," second, then start testing")
    time.sleep(waiting_time)
    
    while True:
        time_in = time.time()
        state, _, global_state = get_state(config, mask, link_indices)
        drl_paths = {}
        agent_index = 0

        if config.get("use_global_state", False):
            input_state = global_state
        else:
            input_state = state

        # path-feat head extras (mirror training-loop logic at line 413-421).

        info = build_info(agents, input_state, 0.0, config, drl_paths={})
        action, _ = agents.get_action([input_state], 0.0, **info)

        for i in range(1, size):
            drl_paths.setdefault(str(i), {})
            for j in range(1, size):
                if i != j:
                    if config['algs_name'] == 'adaptive_dijkstra':
                        drl_paths[str(i)][str(j)] = action[i][j]
                    else:
                        chosen = action[agent_index]
                        drl_paths[str(i)][str(j)] = [all_path_list[i][j][chosen]]
                    agent_index += 1

        drl_paths_path = f"./results/{config['algs_name']}/drl_paths.json"
        
        atomic_io.dump_json(drl_paths_path, drl_paths)
        print("Dumped DRL paths")
        
        file = f"./results/{config['algs_name']}/net_info.csv"
        net_metrics = pd.read_csv(file)

        net_metrics['step'] = step

        path = f"./results/{config['algs_name']}/net_metrics.csv"

        if step == 0 or not os.path.exists(path):
            net_metrics.to_csv(path, index=False)
        else:
            net_metrics.to_csv(path, mode='a', header=False, index=False)
        
        path = f"./results/{config['algs_name']}/drl_paths_list.txt"
        save_stepwise_log(path, drl_paths, step)
        
        avg_delay, avg_packet_loss, avg_throughput, max_link_utilization = compute_network_metrics(config)
        
        with open(metrics_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([avg_delay, avg_packet_loss, avg_throughput, max_link_utilization])
        print("Eval metrics: delay {:.3f}, loss {:.3f}, throughput {:.3f}, max util {:.3f}%".format(
            avg_delay, avg_packet_loss, avg_throughput, max_link_utilization))
        
        step += 1
        if step == 100:
            return
        
        time_end = time.time()
        if time_end - time_in < setting.MONITOR_PERIOD:
            time.sleep(setting.MONITOR_PERIOD - (time_end - time_in))

def convert_sacd_plan_to_drl_paths(routing_plan, env):
    num_node = env.numNodes
    drl_paths = {}

    for i in range(num_node):
        drl_paths[str(i+1)] = {}
        for j in range(num_node):
            if i == j:
                continue

            key = (i, j)

            # SACD 有決策的 pair
            if key in routing_plan:
                action = routing_plan[key]
                path = env.allPaths[f"{i}:{j}"][action]

            # SACD 沒碰到的 pair → shortest paths
            else:
                path = env.shortest_paths[i][j]
            
            # 0-based to 1-based
            path_1based = [node + 1 for node in path]

            drl_paths[str(i+1)][str(j+1)] = [path_1based]
    return drl_paths

def flush_logs(out_dir, data_dict):
    os.makedirs(out_dir, exist_ok=True)
    for fname, data in data_dict.items():
        fname = fname.replace("/", "_")  # avoid subdirectory in filename
        with open(os.path.join(out_dir, fname), "w") as f:
            for line in data:
                f.write(f"{line}\n")


def build_run_archive_dir(config):
    """Directory for one run's archive: ./results/<alg>/runs/<run_name>/train

    run_name is <variant>_<topology>_s<seed>_<timestamp>, and every field appears
    exactly once. The variant may already spell out a topology or a seed (older
    configs did), so those are stripped before the canonical ones are appended --
    otherwise a run reads ..._32node_seed18_32node_s18_...

    The name is what tells you how to reproduce the run: variant goes in
    STRIDE_VARIANT, topology in --env, seed in --seed. The host does not appear;
    it is recorded in the archived config instead, because it identifies where a
    run happened, not what it was.
    """
    import re

    name = config.get("run_name", None)
    if name is None:
        core = str(config.get("_experiment", "run"))
        core = re.sub(r"_?(32node|geant)", "", core)
        core = re.sub(r"_?seed\d+", "", core)
        core = core.strip("_") or config["algs_name"]
        topo = str(config.get("topology", "unknown"))
        seed = int(config.get("seed", 17) or 17)
        name = f"{core}_{topo}_s{seed}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("./results", config["algs_name"], "runs",
                        f"{name}_{timestamp}", "train")


def archive_run_outputs(out_dir, config, started=None):
    """
    Copy all training outputs (logs, model, config) to a per-run archive directory.
    Called at the end of training so each run's files are preserved.
    """
    import shutil

    run_dir = build_run_archive_dir(config)
    if run_dir is None:
        return
    os.makedirs(run_dir, exist_ok=True)

    # 1) Copy all .txt log files
    for f in os.listdir(out_dir):
        if f.endswith(".txt"):
            src = os.path.join(out_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(run_dir, f))

    # 2) Copy the checkpoint directory. save_model writes results/<alg>/model,
    # which the next training run overwrites -- archiving it here is what ties a
    # checkpoint to the run that produced it.
    src = os.path.join(out_dir, "model")
    if os.path.isdir(src):
        dst = os.path.join(run_dir, "model")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # 3) Evidence that the controller was actually measuring while this ran.
    #
    # The agent reads results/<alg>/net_info_directed.csv every step. When the
    # controller stops writing it -- a loaded machine starves its single-threaded
    # event loop until port-stats replies stop arriving -- the file simply stays
    # as it was, and training continues against a snapshot. Nothing raises, the
    # step count is right, a checkpoint is written, and the reward curve keeps
    # moving, because the reward follows the action even when the network behind
    # it does not. One run found this way had 2988 of its 3009 steps reading a
    # file last written ten hours earlier.
    #
    # The per-cycle measurements under results/<alg>/Metrics are far too large to
    # archive, so what goes in is the part that settles the question: how many
    # cycles were recorded, and when the last one landed. A gap between that and
    # the end of training is the whole story.
    try:
        metrics_dir = os.path.join(out_dir, "Metrics")
        cycles = len([f for f in os.listdir(metrics_dir)]) if os.path.isdir(metrics_dir) else 0
        newest = 0.0
        for name in ("net_info_directed.csv", "net_info.csv"):
            path = os.path.join(out_dir, name)
            if os.path.isfile(path):
                newest = max(newest, os.path.getmtime(path))
        finished = time.time()
        with open(os.path.join(run_dir, "measurement.txt"), "w") as fh:
            fh.write(f"metrics_files      {cycles}\n")
            fh.write(f"last_measurement   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest)) if newest else 'never'}\n")
            fh.write(f"training_finished  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finished))}\n")
            fh.write(f"stale_seconds      {int(finished - newest) if newest else -1}\n")
            fh.write("\n"
                     "stale_seconds is how long before training ended the controller last\n"
                     "wrote a measurement. More than a few monitoring periods means the agent\n"
                     "spent that time reading a frozen file, and the run is not usable however\n"
                     "healthy its reward curve looks.\n")
        stale = int(finished - newest) if newest else -1
        if stale < 0 or stale > 60:
            print(f"[archive] WARNING: last measurement was {stale}s before training ended "
                  f"-- see {run_dir}/measurement.txt")
    except OSError as exc:
        print(f"[archive] could not record measurement evidence: {exc}")

    # 4) Save config snapshot as JSON. _host is added here rather than to the
    #    directory name: it says where a run happened, which is provenance, not
    #    identity, and putting it in the name made every name machine-specific.
    try:
        import json as _json, socket
        snapshot = {k: v for k, v in config.items()
                    if isinstance(v, (str, int, float, bool, list, type(None)))}
        snapshot["_host"] = socket.gethostname()
        snapshot["_started"] = started
        snapshot["_finished"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            _json.dump(snapshot, f, indent=2)
    except Exception:
        pass

    print(f"[archive] Run outputs saved to {run_dir}")

def save_stepwise_log(path, data, step):
    mode = 'w' if step == 1 else 'a'
    with open(path, mode) as f:
        f.write(str(data) + '\n')

def build_info(agents, state, epsilon, config, drl_paths):
    info = {}

    if config.get("encode_path", False):
        path_vector  = gen_path_vector(config, drl_paths)
        att_vector   = agents.dqn_model.cal_attention_v(path_vector)[0].detach()
        info["path_vector"] = path_vector
        info["att_vector"] = att_vector

    if config.get("mean_action", False):
        mean_field = agents.prepare_step(state, epsilon)
        info["mean_field"] = [mean_field]

    return info

def loop_pairs(config, size, action, step,
               all_reward, all_reward_indicator,
               all_path_list, drl_paths,
               agent_info_memory, reward_memory):
    """
    return：
       reward_sum, bwd_sum, delay_sum, loss_sum,
       agent_reward_list, agent_delta_list
    """
    r_sum = r_bwd = r_delay = r_loss = 0
    agent_rewards = []
    idx = 0
    
    if step >= 3:
        action_memory = agent_info_memory[0].get("action")

    for i in range(1, size):
        drl_paths.setdefault(str(i), {})
        for j in range(1, size):
            if i == j: continue
            if step >= 3:
                r = all_reward[str(i)][str(j)][action_memory[idx]]
                if config["use_delta_reward"] and step >= 4:
                    agent_rewards.append((r-reward_memory[idx]))
                else:
                    agent_rewards.append(r/100.0)
                r_sum   += r
                r_bwd   += all_reward_indicator[str(i)][str(j)][action_memory[idx]][0]
                r_delay += all_reward_indicator[str(i)][str(j)][action_memory[idx]][1]
                r_loss  += all_reward_indicator[str(i)][str(j)][action_memory[idx]][2]

            if config['algs_name'] == 'adaptive_dijkstra':
                drl_paths[str(i)][str(j)] = action[i][j]
            else:
                chosen = action[idx]
                drl_paths[str(i)][str(j)] = [all_path_list[i][j][chosen]]
            idx += 1
            
    return (r_sum, r_bwd, r_delay, r_loss,
            agent_rewards)

def init_minmax_dic(config):
    #paths_metrics_minmax_dict.setdefault(i, {})
    metrics = ['bwd_paths','delay_paths','loss_paths']
    size = config["num_node"] + 1
    for i in range(1, size):
        paths_metrics_minmax_dict.setdefault(str(i), {})
        for j in range(1, size):
            paths_metrics_minmax_dict[str(i)].setdefault(str(j), {})
            for m in metrics:
                paths_metrics_minmax_dict[str(i)][str(j)].setdefault(m,{})
                paths_metrics_minmax_dict[str(i)][str(j)][m]['min']=100000000
                paths_metrics_minmax_dict[str(i)][str(j)][m]['max']= -1

def path_metrics_to_reward(config):
        
    # read path metrices file
    file = f"./results/{config['algs_name']}/paths_metrics.json"
    rewards_dic = {}
    rewards_indicator = {}
    loss_value = {}
    delay_value = {}
    metrics = ['bwd_paths','delay_paths','loss_paths']
    try:
        with open(file,'r') as json_file:
            paths_metrics_dict = json.load(json_file)
            paths_metrics_dict = ast.literal_eval(json.dumps(paths_metrics_dict))
    except:
        time.sleep(0.35) # wait until file is ok
        with open(file,'r') as json_file:
            paths_metrics_dict = json.load(json_file)
            paths_metrics_dict = ast.literal_eval(json.dumps(paths_metrics_dict))


    for i in paths_metrics_dict:
        rewards_dic.setdefault(i,{})
        rewards_indicator.setdefault(i,{})
        loss_value.setdefault(i,{})
        delay_value.setdefault(i,{})
        for j in paths_metrics_dict[i]:
            rewards_dic.setdefault(j,{})
            rewards_indicator.setdefault(j,{})
            loss_value.setdefault(j,{})
            delay_value.setdefault(i,{})
            loss_value[i][j] = paths_metrics_dict[str(i)][str(j)]['loss_paths'][0]
            delay_value[i][j] = paths_metrics_dict[str(i)][str(j)]['delay_paths'][0]
            for m in metrics:
                if m == metrics[0]:
                    bwd_cost = []
                    for val in paths_metrics_dict[str(i)][str(j)][m][0]:
                        bwd_cost.append(round(val, 15))
                    paths_metrics_dict[str(i)][str(j)][m][0] = bwd_cost
                    paths_metrics_minmax_dict[i][j][m]['max'] = max(paths_metrics_minmax_dict[i][j][m]['max'],max(paths_metrics_dict[str(i)][str(j)][m][0]))
                    paths_metrics_minmax_dict[i][j][m]['min'] = min(paths_metrics_minmax_dict[i][j][m]['min'],min(paths_metrics_dict[str(i)][str(j)][m][0]))
                    met_norm = [normalize(met_val, 0,100, paths_metrics_minmax_dict[i][j][m]['min'], max(paths_metrics_dict[str(i)][str(j)][m][0])) for met_val in paths_metrics_dict[str(i)][str(j)][m][0]]
                elif m == metrics[1]:
                    cost = [] 
                    for val in paths_metrics_dict[str(i)][str(j)][m][0]:
                        if val > DELAY_FLOOR_MS: 
                            temp = 1/val
                            cost.append(round(temp, 15))
                        else:
                            cost.append(round(1/DELAY_FLOOR_MS, 15))
                        #cost.append(round(-val - 1e-6, 15))
                    paths_metrics_dict[str(i)][str(j)][m][0] = cost
                    paths_metrics_minmax_dict[i][j][m]['max'] = max(paths_metrics_minmax_dict[i][j][m]['max'],max(cost))
                    paths_metrics_minmax_dict[i][j][m]['min'] = min(paths_metrics_minmax_dict[i][j][m]['min'],min(cost))
                    met_norm = [normalize(met_val, 0, 100, paths_metrics_minmax_dict[i][j][m]['min'], paths_metrics_minmax_dict[i][j][m]['max']) for met_val in paths_metrics_dict[str(i)][str(j)][m][0]]
                elif m == metrics[2]:    
                    cost = [] 
                    for val in paths_metrics_dict[str(i)][str(j)][m][0]:
                        if val > LOSS_FLOOR_PCT:
                            temp = 1/val
                            cost.append(round(temp, 15))
                        else:
                            cost.append(1/LOSS_FLOOR_PCT)
                        #cost.append(round(-val - 1e-6, 15))
                    paths_metrics_dict[str(i)][str(j)][m][0] = cost
                    paths_metrics_minmax_dict[i][j][m]['max'] = max(paths_metrics_minmax_dict[i][j][m]['max'],max(cost))
                    paths_metrics_minmax_dict[i][j][m]['min'] = min(paths_metrics_minmax_dict[i][j][m]['min'],min(cost))
                    met_norm = [normalize(met_val, 0, 100, paths_metrics_minmax_dict[i][j][m]['min'], paths_metrics_minmax_dict[i][j][m]['max']) for met_val in paths_metrics_dict[str(i)][str(j)][m][0]]
                paths_metrics_dict[str(i)][str(j)][m].append(met_norm)
    
    # 2026-07-15: range(20) -> action_dim (K=30 ablation support). Real-env
    # paths_metrics.json carries as many entries as the controller's k_paths;
    # a K larger than that fails loudly (IndexError) instead of silently
    # truncating the action space.
    _K_act = int(config.get("action_dim", 20))
    for i in paths_metrics_dict:
        for j in paths_metrics_dict[i]:
            rewards_actions = []
            rewards_actions_indicator = []
            for act in range(_K_act):
                rewards_actions.append(reward(i,j,paths_metrics_dict,act,metrics,config))
                rewards_actions_indicator.append(rewards_indicator_fun(i,j,paths_metrics_dict,act,metrics))
                rewards_dic[i][j] = rewards_actions
                rewards_indicator[i][j] = rewards_actions_indicator
    return rewards_dic, rewards_indicator, loss_value,delay_value

def path_metrics_to_reward_sim(env, config):
    # 目前 delay loss 是寫死在裡面的 但目前應該用不到
    # 只給 regular ma 用
    """
    Sim version of path_metrics_to_reward
    - 不讀檔
    - 直接用 env.graph + env.allPaths
    - 完全對齊原本 reward / minmax / normalize 邏輯
    """
    paths_metrics_dict = {}
    rewards_dic = {}
    rewards_indicator = {}
    loss_value = {}
    delay_value = {}

    metrics = ['bwd_paths', 'delay_paths', 'loss_paths']
    size = config["num_node"] + 1   # 1-based


    for i in range(1, size):
        paths_metrics_dict.setdefault(str(i), {})
        for j in range(1, size):
            if i != j:
                paths_metrics_dict[str(i)].setdefault(str(j), {
                    'bwd_paths': [],
                    'delay_paths': [],
                    'loss_paths': []
                })

    # ---------- 準備結構 ----------
    for i in range(1, size):
        rewards_dic.setdefault(str(i), {})
        rewards_indicator.setdefault(str(i), {})
        loss_value.setdefault(str(i), {})
        delay_value.setdefault(str(i), {})
        for j in range(1, size):
            rewards_dic[str(i)].setdefault(str(j), {})
            rewards_indicator[str(i)].setdefault(str(j), {})
            loss_value[str(i)].setdefault(str(j), {})
            delay_value[str(i)].setdefault(str(j), {})

    # ---------- 主計算 ----------
    for src in range(1, size):
        for dst in range(1, size):
            if src == dst:
                continue

            key = f"{src-1}:{dst-1}"
            if key not in env.allPaths:
                continue

            k_paths = env.allPaths[key]   # list of paths (0-based nodes)

            bwd_paths = []
            delay_paths = []
            loss_paths = []

            # === 算每一條 path 的 raw metrics ===
            for path in k_paths:
                bwd_links = []
                delay_links = []
                loss_links = []

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = env.graph[u][v][0]

                    capacity = edge['capacity']
                    used = edge['utilization']
                    bwd = round(capacity - used, 6)
                    # ---- bwd = 剩餘頻寬 ----
                    bwd_links.append(bwd)

                    # ---- delay / loss：目前 sim 沒有就給常數 ----
                    delay_links.append(1e-6)
                    loss_links.append(0)

                # path aggregation（完全照你原本）
                bwd_paths.append(calc_bwd_path(bwd_links))
                delay_paths.append(calc_delay_path(delay_links))
                loss_paths.append(calc_loss_path(loss_links))

            # ---------- 存 raw（for indicator） ----------
            loss_value[str(src)][str(dst)] = loss_paths
            delay_value[str(src)][str(dst)] = delay_paths

            # ---------- normalization（完全對齊） ----------
            paths_metrics_dict[str(src)][str(dst)]['bwd_paths'].append(bwd_paths)
            paths_metrics_dict[str(src)][str(dst)]['delay_paths'].append(delay_paths)
            paths_metrics_dict[str(src)][str(dst)]['loss_paths'].append(loss_paths)

    # for i in paths_metrics_dict:
    #     rewards_dic.setdefault(i,{})
    #     rewards_indicator.setdefault(i,{})
    #     loss_value.setdefault(i,{})
    #     delay_value.setdefault(i,{})
    #     for j in paths_metrics_dict[i]:
    #         rewards_dic.setdefault(j,{})
    #         rewards_indicator.setdefault(j,{})
    #         loss_value.setdefault(j,{})
    #         delay_value.setdefault(i,{})
    #         loss_value[i][j] = paths_metrics_dict[str(i)][str(j)]['loss_paths'][0]
    #         delay_value[i][j] = paths_metrics_dict[str(i)][str(j)]['delay_paths'][0]
    for i in paths_metrics_dict:
        rewards_dic.setdefault(i, {})
        rewards_indicator.setdefault(i, {})
        loss_value.setdefault(i, {})
        delay_value.setdefault(i, {})

        for j in paths_metrics_dict[i]:
            rewards_dic[i].setdefault(j, {})
            rewards_indicator[i].setdefault(j, {})
            loss_value[i].setdefault(j, {})
            delay_value[i].setdefault(j, {})

            loss_value[i][j]  = paths_metrics_dict[str(i)][str(j)]['loss_paths'][0]
            delay_value[i][j] = paths_metrics_dict[str(i)][str(j)]['delay_paths'][0]

            for m in metrics:
                if m == metrics[0]:
                    bwd_cost = []
                    for val in paths_metrics_dict[str(i)][str(j)][m][0]:
                        bwd_cost.append(round(val, 15))
                    paths_metrics_dict[str(i)][str(j)][m][0] = bwd_cost
                    paths_metrics_minmax_dict[i][j][m]['max'] = max(paths_metrics_minmax_dict[i][j][m]['max'],max(paths_metrics_dict[str(i)][str(j)][m][0]))
                    paths_metrics_minmax_dict[i][j][m]['min'] = min(paths_metrics_minmax_dict[i][j][m]['min'],min(paths_metrics_dict[str(i)][str(j)][m][0]))
                    met_norm = [normalize(met_val, 0,100, paths_metrics_minmax_dict[i][j][m]['min'], max(paths_metrics_dict[str(i)][str(j)][m][0])) for met_val in paths_metrics_dict[str(i)][str(j)][m][0]]
                elif m == metrics[1]:
                    cost = [] 
                    for val in paths_metrics_dict[str(i)][str(j)][m][0]:
                        if val > DELAY_FLOOR_MS: 
                            temp = 1/val
                            cost.append(round(temp, 15))
                        else:
                            cost.append(round(1/DELAY_FLOOR_MS, 15))
                        #cost.append(round(-val - 1e-6, 15))
                    paths_metrics_dict[str(i)][str(j)][m][0] = cost
                    paths_metrics_minmax_dict[i][j][m]['max'] = max(paths_metrics_minmax_dict[i][j][m]['max'],max(cost))
                    paths_metrics_minmax_dict[i][j][m]['min'] = min(paths_metrics_minmax_dict[i][j][m]['min'],min(cost))
                    met_norm = [normalize(met_val, 0, 100, paths_metrics_minmax_dict[i][j][m]['min'], paths_metrics_minmax_dict[i][j][m]['max']) for met_val in paths_metrics_dict[str(i)][str(j)][m][0]]
                elif m == metrics[2]:    
                    cost = [] 
                    for val in paths_metrics_dict[str(i)][str(j)][m][0]:
                        if val > LOSS_FLOOR_PCT:
                            temp = 1/val
                            cost.append(round(temp, 15))
                        else:
                            cost.append(1/LOSS_FLOOR_PCT)
                        #cost.append(round(-val - 1e-6, 15))
                    paths_metrics_dict[str(i)][str(j)][m][0] = cost
                    paths_metrics_minmax_dict[i][j][m]['max'] = max(paths_metrics_minmax_dict[i][j][m]['max'],max(cost))
                    paths_metrics_minmax_dict[i][j][m]['min'] = min(paths_metrics_minmax_dict[i][j][m]['min'],min(cost))
                    met_norm = [normalize(met_val, 0, 100, paths_metrics_minmax_dict[i][j][m]['min'], paths_metrics_minmax_dict[i][j][m]['max']) for met_val in paths_metrics_dict[str(i)][str(j)][m][0]]
                paths_metrics_dict[str(i)][str(j)][m].append(met_norm)
    
    # 2026-07-15: range(20) -> action_dim (K=30 ablation support). Real-env
    # paths_metrics.json carries as many entries as the controller's k_paths;
    # a K larger than that fails loudly (IndexError) instead of silently
    # truncating the action space.
    _K_act = int(config.get("action_dim", 20))
    for i in paths_metrics_dict:
        for j in paths_metrics_dict[i]:
            rewards_actions = []
            rewards_actions_indicator = []
            for act in range(_K_act):
                rewards_actions.append(reward(i,j,paths_metrics_dict,act,metrics,config))
                rewards_actions_indicator.append(rewards_indicator_fun(i,j,paths_metrics_dict,act,metrics))
                rewards_dic[i][j] = rewards_actions
                rewards_indicator[i][j] = rewards_actions_indicator
    return rewards_dic, rewards_indicator, loss_value,delay_value

def calc_bwd_path(bwd_links_path):
    '''
    path = [link1, link2, link3]
    path_bwd = min(bwd of all links)
    '''
    bwd_path = min(bwd_links_path)
    return round(bwd_path,6)

def calc_delay_path(delay_links_path):
    '''
    path = [link1, link2, link3]
    path_ldelay = sum(delay of all links)
    '''
    delay_path = sum(delay_links_path)
    return round(delay_path,6)

def calc_loss_path(loss_links_path): 
    '''
    path = [link1, link2, link3]
    path_loss = 1-[(1-loss_link1)*(1-loss_link2)*(1-loss_link3)]
    '''
    loss_links_path_ = [1-(i/100.0) for i in loss_links_path]
    result_multi = reduce((lambda x, y: x * y), loss_links_path_)
    loss_path = 1.0 - result_multi
    return round(loss_path*100.0,6)

def normalize(value, minD, maxD, min_val, max_val):
    if max_val == min_val:
        value_n = (maxD + minD) / 2 
    else:
        value_n = (maxD - minD) * (value - min_val) / (max_val - min_val) + minD
    return round(value_n,15)
                    
def get_mask_directed(config):
    """Directed variant of get_mask: each (u, v) directed edge gets unique index.

    Reads net_info_directed.csv (controller writes both .csv and _directed.csv
    concurrently — see utils/manager.py). num_links = #directed_edges (= 2×
    #undirected for symmetric topologies like Geant: 37→74).

    Mask building reuses the existing path iteration `(path[i], path[i+1])`
    which is naturally directed — caller code unchanged.
    """
    net_info_path = f"./results/{config['algs_name']}/net_info_directed.csv"
    net_info = pd.read_csv(net_info_path).dropna(subset=['node1', 'node2'])
    link_indices = {}  # (u, v) -> unique idx per direction
    link_index_counter = 0
    num_nodes = config["num_node"]

    for _, row in net_info.iterrows():
        node1, node2 = int(row['node1']), int(row['node2'])
        if (node1, node2) not in link_indices:
            link_indices[(node1, node2)] = link_index_counter
            link_index_counter += 1

    num_links = len(link_indices)  # no /2 dedup

    with open(config["k_paths_file"], 'r') as f:
        k_paths = json.load(f)

    K = config.get("action_dim", 20)
    num_agents = num_nodes * (num_nodes - 1)
    mask = np.zeros((num_agents, num_links * 3))

    agent_id = 0
    for src in range(1, num_nodes + 1):
        for dst in range(1, num_nodes + 1):
            if src == dst:
                continue
            if str(src) in k_paths and str(dst) in k_paths[str(src)]:
                paths = k_paths[str(src)][str(dst)][:K]
                for path in paths:
                    for i in range(len(path) - 1):
                        link = (path[i], path[i + 1])
                        if link in link_indices:
                            index = link_indices[link]
                            mask[agent_id, index * 3:(index + 1) * 3] = 1
            agent_id += 1
    mask_3d = mask.reshape(num_agents, num_links, 3)
    if config.get("encoder_input_global", False):
        # nomask ablation: all-ones masks => local_state = masks * global_state =
        # the full network for EVERY pair. With encoder_spatial='flat' this is the
        # clean "no per-pair encoder" ablation (only the per-pair head differentiates).
        mask_3d = np.ones_like(mask_3d)
    return mask_3d, link_indices


def get_state_directed(config, masks, link_indices, step=None):
    """Directed variant of get_state: reads net_info_directed.csv, uses tc_delay_ms
    for delay channel (vs LLDP delay in undirected mode), per-direction bw/pkloss.

    Same 3-ch state structure as undirected: [bwd_ratio, delay_norm, pkloss_pct].
    delay_norm = (cur_delay_ms + 1e-6) / delay_norm_div (default 200).

    """
    num_links = len(link_indices)  # all directed edges
    global_state_2d = np.zeros((num_links, 3), dtype=float)

    bwd = load_bwd_table(config["bw_file"], link_indices, bidirectional=False)
    # 2026-05-27 bugfix: was `bwd * 2000.0` copied from undirected path -- WRONG for
    # directed mode. Undirected net_info.csv stores 2C - (tx+rx) [kbps], so the
    # *2 was needed to match 2C accounting. Directed net_info_directed.csv stores
    # C - tx [kbps] (single direction); only *1000 (Mbps -> kbps) is needed.
    # Old formula gave mlu = (2C - (C-tx)) / 2C = 0.5 + tx/(2C), inflating reports
    # by an offset of 0.5 (0% util -> reported 0.5; 100% -> reported 1.0). Agent
    # state's bwd channel was also halved to [0, 0.5] instead of [0, 1.0]. Existing
    # ckpts trained on the halved scale still work for inference (Linear layer
    # absorbs the constant factor) but post-fix mlu logs are now true MLU. Runs
    # from before the fix were corrected retroactively; the affected curves in
    # paper/figures/reward/ are rebuilt from local logs and are unaffected.
    bwd = bwd * 1000.0  # Mbps -> kbps (single-direction; matches net_info_directed.csv)

    net_info_file = f"./results/{config['algs_name']}/net_info_directed.csv"
    mlu = 0.0
    try:
        net_info = pd.read_csv(net_info_file)
    except Exception:
        time.sleep(0.35)
        net_info = pd.read_csv(net_info_file)
    net_info = net_info.dropna(subset=['node1', 'node2', 'bwd'])

    for _, row in net_info.iterrows():
        node1, node2 = int(row['node1']), int(row['node2'])
        if (node1, node2) not in link_indices:
            continue
        index = link_indices[(node1, node2)]

        cur_bwd = row['bwd']
        # Prefer tc_delay_ms (queue truth) over LLDP delay column
        if 'delay_tc_ms' in row and pd.notna(row['delay_tc_ms']):
            cur_delay = float(row['delay_tc_ms']) + 1e-6
        elif 'delay_lldp' in row and pd.notna(row['delay_lldp']):
            cur_delay = float(row['delay_lldp']) + 1e-6
        else:
            cur_delay = float(row.get('delay', 0.0)) + 1e-6
        cur_pkloss = float(row['pkloss'])

        mlu = max(mlu, (bwd[index] - cur_bwd) / bwd[index])

        global_state_2d[index, 0] = cur_bwd / bwd[index]
        global_state_2d[index, 1] = cur_delay / config.get("delay_norm_div", 200.0)
        global_state_2d[index, 2] = cur_pkloss

    if config.get("use_bwd_only", False):
        global_state_2d = global_state_2d[:, 0:1]
        global_state_2d_expanded = np.expand_dims(global_state_2d, axis=0)
        local_state = masks[:, :, :global_state_2d.shape[1]] * global_state_2d_expanded
    else:
        global_state_2d_expanded = np.expand_dims(global_state_2d, axis=0)
        local_state = masks * global_state_2d_expanded

    global_state_2d = global_state_2d.flatten()
    return local_state, mlu, global_state_2d


def path_metrics_to_reward_directed(config):
    """Directed variant: reads net_info_directed.csv (controller writes this in
    parallel with undirected net_info.csv), builds per-directed-edge lookup,
    traverses K-paths with directed edge data. Aggregation primitives (min /
    sum / 1-Π(1-l)) and normalize/invert/min-max logic match the original
    path_metrics_to_reward exactly — only the data source differs.

    Per-link in path uses:
      bwd:   directed free bw (kbps) from row['bwd']
      delay: tc_delay_ms (queue truth) preferred, fallback to lldp_fwd_ms
      loss:  per-direction pkloss (percent) from row['pkloss']
    """
    file = f"./results/{config['algs_name']}/net_info_directed.csv"
    metrics = ['bwd_paths', 'delay_paths', 'loss_paths']
    rewards_dic = {}
    rewards_indicator = {}
    loss_value = {}
    delay_value = {}

    try:
        net_info = pd.read_csv(file)
    except Exception:
        time.sleep(0.35)
        net_info = pd.read_csv(file)
    net_info = net_info.dropna(subset=['node1', 'node2', 'bwd'])

    # Build per-(u, v) directed link lookup
    link_data = {}
    for _, row in net_info.iterrows():
        u, v = int(row['node1']), int(row['node2'])
        bwd = float(row['bwd'])
        if 'delay_tc_ms' in row and pd.notna(row['delay_tc_ms']):
            delay_ms = float(row['delay_tc_ms'])
        elif 'delay_lldp' in row and pd.notna(row['delay_lldp']):
            delay_ms = float(row['delay_lldp'])
        else:
            delay_ms = 1e-6
        pkloss = float(row['pkloss'])  # already in percent per controller convention
        link_data[(u, v)] = (bwd, delay_ms, pkloss)

    # Build per-(src,dst,K-path) raw metrics from directed link data
    with open(config["k_paths_file"], 'r') as fp:
        k_paths_dict = json.load(fp)

    K = config.get("action_dim", 20)
    paths_metrics_dict = {}
    for s_str, dst_dict in k_paths_dict.items():
        paths_metrics_dict[s_str] = {}
        for d_str, paths in dst_dict.items():
            bwd_paths_, delay_paths_, loss_paths_ = [], [], []
            for path in paths[:K]:
                bwd_links, delay_links, loss_links = [], [], []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    bw, dl, ls_ = link_data.get((u, v), (0.0, 1e-6, 0.0))
                    bwd_links.append(bw)
                    delay_links.append(dl)
                    loss_links.append(ls_)
                bwd_paths_.append(calc_bwd_path(bwd_links))      # min
                delay_paths_.append(calc_delay_path(delay_links))  # sum
                loss_paths_.append(calc_loss_path(loss_links))    # 1 - Π(1-l/100)*100
            paths_metrics_dict[s_str][d_str] = {
                'bwd_paths':   [bwd_paths_],
                'delay_paths': [delay_paths_],
                'loss_paths':  [loss_paths_],
            }

    # Apply EXACT normalize + invert + min-max logic from path_metrics_to_reward.
    # (Mirrors lines 1804-1862; kept verbatim for byte-for-byte semantic parity.)
    for i in paths_metrics_dict:
        rewards_dic.setdefault(i, {})
        rewards_indicator.setdefault(i, {})
        loss_value.setdefault(i, {})
        delay_value.setdefault(i, {})
        for j in paths_metrics_dict[i]:
            rewards_dic.setdefault(j, {})
            rewards_indicator.setdefault(j, {})
            loss_value.setdefault(j, {})
            delay_value.setdefault(i, {})
            loss_value[i][j] = paths_metrics_dict[str(i)][str(j)]['loss_paths'][0]
            delay_value[i][j] = paths_metrics_dict[str(i)][str(j)]['delay_paths'][0]
            for m in metrics:
                if m == metrics[0]:
                    bwd_cost = [round(v, 15) for v in paths_metrics_dict[str(i)][str(j)][m][0]]
                    paths_metrics_dict[str(i)][str(j)][m][0] = bwd_cost
                    paths_metrics_minmax_dict[i][j][m]['max'] = max(paths_metrics_minmax_dict[i][j][m]['max'], max(bwd_cost))
                    paths_metrics_minmax_dict[i][j][m]['min'] = min(paths_metrics_minmax_dict[i][j][m]['min'], min(bwd_cost))
                    met_norm = [normalize(v, 0, 100, paths_metrics_minmax_dict[i][j][m]['min'], max(bwd_cost)) for v in bwd_cost]
                elif m == metrics[1]:
                    cost = []
                    for v in paths_metrics_dict[str(i)][str(j)][m][0]:
                        cost.append(round(1.0 / v, 15) if v > DELAY_FLOOR_MS
                                    else round(1.0 / DELAY_FLOOR_MS, 15))
                    paths_metrics_dict[str(i)][str(j)][m][0] = cost
                    paths_metrics_minmax_dict[i][j][m]['max'] = max(paths_metrics_minmax_dict[i][j][m]['max'], max(cost))
                    paths_metrics_minmax_dict[i][j][m]['min'] = min(paths_metrics_minmax_dict[i][j][m]['min'], min(cost))
                    met_norm = [normalize(v, 0, 100, paths_metrics_minmax_dict[i][j][m]['min'], paths_metrics_minmax_dict[i][j][m]['max']) for v in cost]
                elif m == metrics[2]:
                    cost = []
                    for v in paths_metrics_dict[str(i)][str(j)][m][0]:
                        cost.append(round(1.0 / v, 15) if v > LOSS_FLOOR_PCT
                                    else 1.0 / LOSS_FLOOR_PCT)
                    paths_metrics_dict[str(i)][str(j)][m][0] = cost
                    paths_metrics_minmax_dict[i][j][m]['max'] = max(paths_metrics_minmax_dict[i][j][m]['max'], max(cost))
                    paths_metrics_minmax_dict[i][j][m]['min'] = min(paths_metrics_minmax_dict[i][j][m]['min'], min(cost))
                    met_norm = [normalize(v, 0, 100, paths_metrics_minmax_dict[i][j][m]['min'], paths_metrics_minmax_dict[i][j][m]['max']) for v in cost]
                paths_metrics_dict[str(i)][str(j)][m].append(met_norm)

    # Build per-action reward dict (matches original path_metrics_to_reward output shape)
    # 2026-07-15: range(20) -> K (already action_dim-driven above; K=30 ablation)
    for i in paths_metrics_dict:
        for j in paths_metrics_dict[i]:
            rewards_actions = []
            rewards_actions_indicator = []
            for act in range(K):
                rewards_actions.append(reward(i, j, paths_metrics_dict, act, metrics, config))
                rewards_actions_indicator.append(rewards_indicator_fun(i, j, paths_metrics_dict, act, metrics))
                rewards_dic[i][j] = rewards_actions
                rewards_indicator[i][j] = rewards_actions_indicator
    return rewards_dic, rewards_indicator, loss_value, delay_value


def get_mask(config):
    # 2026-05-24: dispatch to directed variant when `state_directed=True` config knob set.
    # Directed variant reads net_info_directed.csv, builds per-direction link_indices,
    # mask iterates same (path[i] → path[i+1]) traversal but indexes directed edge.
    # Caller / agent unchanged — only link_indices semantics differ.
    if config.get("state_directed", False):
        return get_mask_directed(config)

    net_info_path = f"./results/{config['algs_name']}/net_info.csv"
    net_info = pd.read_csv(net_info_path).dropna(subset=['node1', 'node2'])
    link_indices = {}  # (node1, node2) -> index
    link_index_counter = 0
    num_nodes = config["num_node"]

    for _, row in net_info.iterrows():
        node1, node2 = int(row['node1']), int(row['node2'])
        if (node1, node2) not in link_indices and (node2, node1) not in link_indices:
            link_indices[(node1, node2)] = link_index_counter
            link_indices[(node2, node1)] = link_index_counter
            link_index_counter += 1

    num_links = int(len(link_indices)/2)

    with open(config["k_paths_file"], 'r') as f:
        k_paths = json.load(f)

    K = config.get("action_dim", 20)
    num_agents = num_nodes * (num_nodes - 1)
    mask = np.zeros((num_agents, num_links * 3))

    agent_id = 0
    for src in range(1, num_nodes + 1):
        for dst in range(1, num_nodes + 1):
            if src == dst:
                continue

            if str(src) in k_paths and str(dst) in k_paths[str(src)]:
                paths = k_paths[str(src)][str(dst)][:K]
                for path in paths:
                    for i in range(len(path) - 1):
                        link = (path[i], path[i + 1])
                        if link in link_indices:
                            index = link_indices[link]
                            mask[agent_id, index * 3:(index + 1) * 3] = 1
            agent_id += 1
    mask_3d = mask.reshape(num_agents, num_links, 3)
    if config.get("encoder_input_global", False):
        # nomask ablation: all-ones masks => local_state = masks * global_state =
        # the full network for EVERY pair. With encoder_spatial='flat' this is the
        # clean "no per-pair encoder" ablation (only the per-pair head differentiates).
        mask_3d = np.ones_like(mask_3d)
    return mask_3d, link_indices

def load_bwd_table(bw_file_path, link_indices, bidirectional=True):
    num_links = int(len(link_indices) / (2 if bidirectional else 1))
    bwd = np.zeros(num_links)
    link_bwd_map = {}

    with open(bw_file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            src, dst, _, bw = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            link_bwd_map[(src, dst)] = bw
            if bidirectional:
                link_bwd_map[(dst, src)] = bw

    for (src, dst), idx in link_indices.items():
        bwd[idx] = link_bwd_map.get((src, dst), 100000.0)  # default safety value

    return bwd

def update_env_graph_from_net_info(env, config):
    """
    Read net_info from Mininet and update env.graph edge attributes.
    This bridges real Mininet measurements → GNN input for the GNN-based agents.

    Units: env.graph stores capacity/utilization in kbps (matching TM units).
           net_info_directed.csv 'bwd' column is in kbps.

    Returns: mlu (max directed utilization ratio, float)
    """
    net_info_mode = config.get("real_net_info_mode", "directed")
    alg_name = config["algs_name"]

    if net_info_mode == "directed":
        path = f"./results/{alg_name}/net_info_directed.csv"
    else:
        path = f"./results/{alg_name}/net_info.csv"

    try:
        net_info = pd.read_csv(path)
    except:
        time.sleep(0.35)
        net_info = pd.read_csv(path)

    # Drop malformed rows: Ryu controller may emit NaN under link flap /
    # partition. One bad row was the cause of PC2 R11 ga86orjd crash at step
    # 2023 (2026-05-22). One row missing != whole step lost — env.graph for
    # that link just retains last value until next valid CSV.
    net_info = net_info.dropna(subset=['node1', 'node2', 'bwd'])

    mlu = 0.0
    # Directed CSV columns: node1,node2,bwd,lldp_fwd_ms,delay_lldp,delay_tc_ms,pkloss,queue_pkts,queue_bytes
    # Undirected CSV columns: node1,node2,bwd,delay,pkloss
    # Directed: prefer tc-backlog delay (ground truth for queue delay; LLDP underestimates ~3x
    # per docs/delay_measurement_issues.md). CSV stores ms; env expects seconds.
    for _, row in net_info.iterrows():
        node1, node2 = int(row['node1']), int(row['node2'])
        u, v = node1 - 1, node2 - 1  # env graph is 0-based

        free_bw_kbps = float(row['bwd'])
        if net_info_mode == "directed":
            if "delay_tc_ms" in row and pd.notna(row["delay_tc_ms"]):
                delay = float(row["delay_tc_ms"]) / 1000.0
            else:
                delay = float(row["lldp_fwd_ms"]) / 1000.0  # fallback
        else:
            delay = float(row["delay"])  # undirected path: already seconds
        # CSV pkloss is in PERCENT (manager.py:113 link_loss_dir = ratio*100,
        # simple_tc_loss.py:114 link_loss_tc = loss*100). Env contract
        # ([env16.py:1097, 1634]) is RATIO [0,1] — get_link_features
        # line 1685 clamps to [0,1] so any loss >1% binarises to 1.0 without
        # this divide. Train-path twin of the eval-path fix at
        # environment16.py:3108 (commit 75663ef).
        pkloss = float(row['pkloss']) / 100.0

        if net_info_mode == "directed":
            # Direct mapping: one CSV row = one directed edge
            if u in env.graph and v in env.graph[u]:
                cap_kbps = float(env.graph[u][v][0].get('capacity', 100000.0))
                used_bw_kbps = max(0.0, cap_kbps - free_bw_kbps)

                env.graph[u][v][0]['utilization'] = used_bw_kbps
                env.graph[u][v][0]['delay'] = delay
                env.graph[u][v][0]['pkloss'] = pkloss

                ratio = used_bw_kbps / (cap_kbps + 1e-8)
                mlu = max(mlu, ratio)
        else:
            # Undirected: one CSV row = physical link, split equally to both directions
            if u in env.graph and v in env.graph[u]:
                cap_kbps = float(env.graph[u][v][0].get('capacity', 100000.0))
                total_used_kbps = max(0.0, 2.0 * cap_kbps - free_bw_kbps)
                per_dir_used = total_used_kbps / 2.0

                env.graph[u][v][0]['utilization'] = per_dir_used
                env.graph[u][v][0]['delay'] = delay
                env.graph[u][v][0]['pkloss'] = pkloss

                ratio = per_dir_used / (cap_kbps + 1e-8)
                mlu = max(mlu, ratio)

            if v in env.graph and u in env.graph[v]:
                cap_kbps_rev = float(env.graph[v][u][0].get('capacity', 100000.0))
                env.graph[v][u][0]['utilization'] = per_dir_used
                env.graph[v][u][0]['delay'] = delay
                env.graph[v][u][0]['pkloss'] = pkloss

                ratio_rev = per_dir_used / (cap_kbps_rev + 1e-8)
                mlu = max(mlu, ratio_rev)

    return mlu


def extract_per_pair_reward(all_reward, actions, config, all_reward_indicator=None):
    """
    Extract per-pair reward vector from path_metrics_to_reward() output.
    Matches the agent ordering: (1,2), (1,3), ..., (1,N), (2,1), (2,3), ...

    Args:
        all_reward: dict from path_metrics_to_reward(), all_reward[src][dst][action_idx]
        actions: numpy array of per-agent action indices (from 2 steps ago for delay alignment)
        config: config dict
        all_reward_indicator: optional, if provided also extract per-component (bwd, delay, loss)

    Returns:
        r_vec: numpy array of shape (num_agents,) with per-pair rewards
        components: dict {"bwd": array, "delay": array, "loss": array} or None
    """
    size = config["num_node"] + 1
    r_vec = []
    bwd_vec, delay_vec, loss_vec = [], [], []
    has_indicator = (all_reward_indicator is not None)
    idx = 0
    for i in range(1, size):
        for j in range(1, size):
            if i == j:
                continue
            act = int(actions[idx])
            r = all_reward[str(i)][str(j)][act]
            r_vec.append(float(r) / 100.0)  # normalize [0,100] → [0,1]
            if has_indicator:
                ind = all_reward_indicator[str(i)][str(j)][act]
                bwd_vec.append(float(ind[0]) / 100.0)    # [0,100] → [0,1]
                delay_vec.append(float(ind[1]) / 100.0)
                loss_vec.append(float(ind[2]) / 100.0)
            idx += 1
    components = None
    if has_indicator:
        components = {
            "bwd": np.array(bwd_vec, dtype=np.float32),
            "delay": np.array(delay_vec, dtype=np.float32),
            "loss": np.array(loss_vec, dtype=np.float32),
        }
    return np.array(r_vec, dtype=np.float32), components


def check_controller_alive(config, step):
    """Stop the run when the controller's measurement file stops changing.

    The agent reads results/<alg>/net_info_directed.csv once per step. If the
    monitor greenthread dies the file keeps its last contents, nothing raises,
    the step count stays right and the reward curve keeps moving, because the
    reward follows the action rather than the network. One run found this way
    spent 2988 of 3009 steps reading a file written ten hours earlier. See
    docs/controller_stops_measuring.md.

    Under live traffic two consecutive cycles never write a byte-identical file:
    every directed link carries a remaining-bandwidth, queue-delay and loss
    figure, and all of them would have to repeat at once. A repeat is therefore
    the controller, not the network.

    Two knobs, both on the env config:
      stall_abort_steps  identical reads that end the run (0 disables the check)
      stall_grace_steps  steps skipped at the start, before traffic ramps up --
                         an idle network does produce identical files
    """
    limit = int(config.get("stall_abort_steps", 5))
    if limit <= 0 or step <= int(config.get("stall_grace_steps", 10)):
        return
    name = ("net_info_directed.csv" if config.get("state_directed", False)
            else "net_info.csv")
    path = f"./results/{config['algs_name']}/{name}"
    try:
        with open(path, "rb") as fh:
            digest = hashlib.md5(fh.read()).hexdigest()
    except OSError:
        return                      # mid-write; get_state already retries reads
    if digest == getattr(check_controller_alive, "digest", None):
        check_controller_alive.same = getattr(check_controller_alive, "same", 0) + 1
    else:
        check_controller_alive.digest = digest
        check_controller_alive.same = 0
    if check_controller_alive.same >= limit:
        raise RuntimeError(
            f"[monitor] {name} has been byte-identical for "
            f"{check_controller_alive.same + 1} reads in a row, ending at step "
            f"{step}. The controller has stopped measuring and every step from "
            f"here would train against a frozen snapshot. Stopping now rather "
            f"than at the end of the run. See docs/controller_stops_measuring.md"
        )


def get_state(config, masks, link_indices, step=None): # get the current network state
    # 2026-05-24: dispatch to directed variant when `state_directed=True`.
    if config.get("state_directed", False):
        return get_state_directed(config, masks, link_indices, step=step)

    num_links = int(len(link_indices) / 2)
    global_state_2d = np.zeros((num_links, 3), dtype=float)

    bwd = load_bwd_table(config["bw_file"], link_indices, bidirectional=True)
    bwd = bwd * 2000.0

    net_info_file = f"./results/{config['algs_name']}/net_info.csv"
    mlu = 0.0

    try:
        net_info = pd.read_csv(net_info_file)
    except:
        time.sleep(0.35)
        net_info = pd.read_csv(net_info_file)
        
    for i, (_, row) in enumerate(net_info.iterrows()):
        node1, node2 = int(row['node1']), int(row['node2'])
        if (node1, node2) in link_indices:
            index = link_indices[(node1, node2)]

            # cur_bwd 是剩餘容量
            cur_bwd = row['bwd']
            cur_delay = row['delay'] + 1e-6
            cur_pkloss = row['pkloss']

            mlu = max(mlu, (bwd[index] - cur_bwd) / bwd[index])

            global_state_2d[index, 0] = cur_bwd / bwd[index]  # normalized throughput
            global_state_2d[index, 1] = cur_delay / config.get("delay_norm_div", 200.0)  # configurable
            global_state_2d[index, 2] = cur_pkloss

    if config.get("use_bwd_only", False):
        print("state使用bwd_only")
        global_state_2d = global_state_2d[:, 0:1]   # (num_links, 1)
        global_state_2d_expanded = np.expand_dims(global_state_2d, axis=0)  # (1, num_links, 3)
        local_state = masks[:, :, :global_state_2d.shape[1]] * global_state_2d_expanded # 如果只用 bwd mask 也跟著縮惟度
    else:
        print("state使用全部特徵")
        global_state_2d_expanded = np.expand_dims(global_state_2d, axis=0)  # (1, num_links, 3)
        local_state = masks * global_state_2d_expanded

    global_state_2d = global_state_2d.flatten()
    return local_state, mlu, global_state_2d

def reward(src, dst, paths_metrics_dict, act, metrics, config):
    """Weighted sum of the three path-quality components (paper eq. 22).

    r_i = lambda_bwd * r_i^bwd + lambda_delay * r_i^delay + lambda_pkl * r_i^pkl

    Each component is already min-max normalised to [0, 100] by the caller;
    loop_pairs divides the total by 100 before it reaches the agent, so the
    reward the policy sees is on the [0, 1] scale the paper describes.

    reward_mode="bwd_only" zeroes the delay and loss weights whatever the
    lambdas say -- it is a switch, not a weighting.
    """
    if config.get("reward_mode", "all") == "bwd_only":
        w_bwd, w_delay, w_pkl = 1.0, 0.0, 0.0
    else:
        w_bwd   = float(config.get("lambda_bwd", 1.0))
        w_delay = float(config.get("lambda_delay", 1.0))
        w_pkl   = float(config.get("lambda_pkl", 1.0))

    per_pair = paths_metrics_dict[str(src)][str(dst)]
    reward = (w_bwd   * per_pair[metrics[0]][1][act]
              + w_delay * per_pair[metrics[1]][1][act]
              + w_pkl   * per_pair[metrics[2]][1][act])
    return round(reward, 15)

def rewards_indicator_fun(src, dst, paths_metrics_dict, act, metrics):

    return (paths_metrics_dict[str(src)][str(dst)][metrics[0]][1][act],paths_metrics_dict[str(src)][str(dst)][metrics[1]][1][act],paths_metrics_dict[str(src)][str(dst)][metrics[2]][1][act])


def state_to_action(config): # K paths according src,dst (truncated to action_dim)
    file = config["k_paths_file"]
    size = config["num_node"] + 1
    K = config.get("action_dim", 20)
    paths = []
    with open(file,'r') as json_file:
        paths = json.load(json_file)
    column, row = size, size
    paths_20 = [[0]*row for _ in range(column)]
    print(f"k_paths loaded from {file} (truncated to K={K})")
    for i in range(1, size):
        for j in range(1, size):
            if i != j:
                paths_20[i][j] = paths[str(i)][str(j)][:K]
    return paths_20
    
def compute_network_metrics(config, directed=False):
    """Aggregate per-link metrics into scalars for one step.

    Returns:
        directed=False: (avg_delay, avg_loss, avg_throughput, max_util_pct)
        directed=True:  (avg_delay_tc, avg_delay_lldp, avg_loss, avg_throughput, max_util_pct)
                        avg_delay_tc is the queue-based ground truth (delay_tc_ms);
                        avg_delay_lldp is the LLDP-derived value retained for
                        diagnostic logging (LLDP underestimates by ~20-35%).
                        Falls back to delay_lldp when delay_tc_ms is missing
                        (pre-Phase-2 CSVs).
    """
    try:
        if not directed:
            path = f"./results/{config['algs_name']}/net_info.csv"
        else:
            path = f"./results/{config['algs_name']}/net_info_directed.csv"
        net_info = pd.read_csv(path)
    except Exception as e:
        print("Error reading net_info.csv:", e)
        return (0, 0, 0, 0, 0) if directed else (0, 0, 0, 0)

    capacity_dict = {}
    try:
        with open(config["bw_file"], 'r') as file:
            for line in file:
                data = line.strip().split(',')
                if len(data) < 4:
                    continue
                src, dst, _, bw = int(data[0]), int(data[1]), data[2], float(data[3])
                capacity_dict[(src, dst)] = bw
                capacity_dict[(dst, src)] = bw
    except Exception as e:
        print("Error reading bw_r.txt:", e)
        capacity_dict = {}

    has_tc   = directed and ('delay_tc_ms' in net_info.columns)
    has_lldp = directed and ('delay_lldp'   in net_info.columns)

    delays      = []
    delays_lldp = []
    packet_losses = []
    throughputs   = []
    utilizations  = []
    for _, row in net_info.iterrows():
        try:
            node1 = int(row['node1'])
            node2 = int(row['node2'])
        except Exception:
            continue

        if directed:
            # Main delay: tc-based queue truth (matches NX fluid queue magnitude).
            if has_tc and pd.notna(row['delay_tc_ms']):
                delay = float(row['delay_tc_ms'])
            elif has_lldp:
                delay = float(row['delay_lldp'])  # legacy fallback
            else:
                delay = 0.0
            if has_lldp and pd.notna(row['delay_lldp']):
                delays_lldp.append(float(row['delay_lldp']))
        else:
            delay = float(row['delay'])

        pkloss = row['pkloss']
        free_bw = row['bwd'] / 1000.0  # Kbps -> Mbps
        cap = capacity_dict.get((node1, node2), 200)
        if not directed:
            throughput = (2 * cap) - free_bw
            utilization = (2 * cap - free_bw) / (2 * cap)
        else:
            throughput = cap - free_bw
            utilization = (cap - free_bw) / cap

        delays.append(delay)
        packet_losses.append(pkloss)
        throughputs.append(throughput)
        utilizations.append(utilization)

    if len(delays) == 0:
        return (0, 0, 0, 0, 0) if directed else (0, 0, 0, 0)

    avg_delay = np.mean(delays)
    avg_packet_loss = np.mean(packet_losses)
    avg_link_throughput = np.mean(throughputs)
    max_link_utilization = max(utilizations) * 100.0

    if directed:
        avg_delay_lldp = float(np.mean(delays_lldp)) if delays_lldp else 0.0
        return avg_delay, avg_delay_lldp, avg_packet_loss, avg_link_throughput, max_link_utilization
    return avg_delay, avg_packet_loss, avg_link_throughput, max_link_utilization

def write_load_to_graph(env_eval, drl_paths, config):
    """Write TM demand along drl_paths to graph edges as directed utilization (kbps).

    Returns load dict {(u,v): used_kbps} for downstream metric computation.
    Used by transient queue callers (run_eval / run_sim_eval) to separate
    load-writing from queue update, so queue is only advanced once per step.
    """
    load = {}
    for u, v in env_eval.graph.edges():
        load[(u, v)] = 0.0
        if (v, u) not in load:
            load[(v, u)] = 0.0

    for src in drl_paths:
        for dst in drl_paths[src]:
            if src == dst:
                continue
            path_1based = drl_paths[src][dst][0]
            path_0based = [n - 1 for n in path_1based]
            demand_bw_kbps = float(env_eval.TM[int(src) - 1][int(dst) - 1])
            for i in range(len(path_0based) - 1):
                u, v = path_0based[i], path_0based[i + 1]
                load[(u, v)] = load.get((u, v), 0.0) + demand_bw_kbps

    for (u, v), used_kbps in load.items():
        if env_eval.graph.has_edge(u, v):
            env_eval.graph[u][v][0]['utilization'] = used_kbps

    return load


def compute_network_metrics_nx(env_eval, drl_paths, config, directed=False, out_link_csv=None, skip_queue=False):
    # 這裡輸出是 Mbps

    # 只拿 env TM + topology + drl_paths
    # 不拿 graph.util
    # 自己 maintain load 累加器，最後算 throughput / utilization
    # 完全不改、不動裡面的 util等
    """
    用 NetworkX 的 topology + env_eval.TM + 你已經算好的 drl_paths 來「離線」估計網路 metrics，
    目標是跟 compute_network_metrics() 的單位與計算方式對齊（至少 throughput / utilization 對齊）。

    參數
    ----
    env_eval:
        GraphEnv-v16 instance（NX graph 在 env_eval.graph）
        必須已經 reset / kpath_reset 過，確保 env_eval.TM 是當前 tm 的 demand matrix。

    drl_paths:
        dict，key 用字串 "1".."N"（1-based node id）
        drl_paths[src][dst] = [[path_nodes_1based]]
        例如 drl_paths["1"]["3"] = [[1, 7, 3]]

    directed:
        False: 把 (u,v) 當作「物理無向 link」，兩個方向的 traffic 都加總在同一條物理 link 上
               並且 capacity 用 2*cap（對齊你 compute_network_metrics() 對 undirected 的處理）
        True : 把 (u->v) 當作「有向 link」，每個方向各算各的，capacity 用 cap
    """

    # ============================================================
    # (0) 防呆：確認 TM 有存在
    # ============================================================
    if not hasattr(env_eval, "TM") or env_eval.TM is None:
        raise ValueError(
            "[compute_network_metrics_nx] env_eval.TM 不存在。"
            "請先做 env_eval.reset(tm_id=...) 或 env_eval.kpath_reset(tm_id)。"
        )

    # ============================================================
    # (1) 讀 bw_file 建 capacity map
    #     - bw_file 的 node id 是 1-based
    #     - env_eval.graph 的 node id 是 0-based
    #
    # cap_dir:  (u,v) 0-based -> cap（單方向容量, 單位跟 bw_file 一樣，後面會和 used_mbps 對齊）
    # cap_und:  frozenset({u,v}) -> cap（物理無向 link 的單方向 cap）
    # ============================================================
    cap_dir = {}
    cap_und = {}

    DEFAULT_CAP = float(config.get("link_bw_default", 200.0))  # 沒讀到 bw_file 時的保底值

    try:
        with open(config["bw_file"], "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue

                u1, v1, _, bw = int(parts[0]), int(parts[1]), parts[2], float(parts[3])
                u = u1 - 1
                v = v1 - 1

                # 你的 bw_file 通常是一條 link 一行，但你在 real metric 那邊視為雙向同 cap
                cap_dir[(u, v)] = bw
                cap_dir[(v, u)] = bw
                cap_und[frozenset((u, v))] = bw
    except Exception as e:
        # 不要直接 pass，至少留 log，避免 cap=None 造成後面爆炸
        print("[compute_network_metrics_nx] read bw_file failed, fallback to DEFAULT_CAP. err =", e)
        raise e
    # ============================================================
    # (2) 初始化 per-direction load 累加器
    #     Always track per-direction to support per-direction clamping
    #     (matches real Mininet: each direction capped independently)
    # ============================================================
    load = {}  # (u,v) -> demand_kbps, always per-direction
    for u, v in env_eval.graph.edges():
        load[(u, v)] = 0.0
        if (v, u) not in load:
            load[(v, u)] = 0.0

    # ============================================================
    # (3) 把每一對 (src,dst) 的 demand 加到它選到的 path 的每一條 link 上
    # ============================================================
    for src in drl_paths:
        for dst in drl_paths[src]:
            if src == dst:
                continue
            path_1based = drl_paths[src][dst][0]
            path_0based = [n - 1 for n in path_1based]
            demand_bw_kbps = float(env_eval.TM[int(src) - 1][int(dst) - 1])
            for i in range(len(path_0based) - 1):
                u, v = path_0based[i], path_0based[i + 1]
                load[(u, v)] = load.get((u, v), 0.0) + demand_bw_kbps

    # ============================================================
    # (3b) Run fluid queue model to steady state (30 steps) so delay/loss are available.
    #      Save/restore queue_bits so training state is not disturbed.
    #
    #      skip_queue=True: caller already updated graph utilization + queue state
    #      (transient mode for test_single_tm / test_sim_only).  Just read metrics.
    # ============================================================
    _has_queue = hasattr(env_eval, '_update_queues') and hasattr(env_eval, 'queue_bits')
    _saved_queue = None
    if not skip_queue:
        if _has_queue:
            _saved_queue = dict(env_eval.queue_bits)        # save training state
        for (u, v), used_kbps in load.items():
            if env_eval.graph.has_edge(u, v):
                env_eval.graph[u][v][0]['utilization'] = used_kbps
        if _has_queue:
            env_eval.reset_queues()
            for _ in range(30):                             # run to steady state
                env_eval._update_queues()

    # ============================================================
    # (4) Per-direction clamped metrics (matches real Mininet physics)
    #
    #     Real MN: each direction's throughput is physically capped.
    #     So we clamp each direction independently, THEN aggregate
    #     for undirected mode.
    # ============================================================
    total_used = 0.0
    total_over = 0.0

    if directed:
        throughputs = []
        utilizations = []
        for (u, v), used_bw_kbps in load.items():
            cap = float(cap_dir.get((u, v), DEFAULT_CAP))
            used_mbps_raw = max(0.0, float(used_bw_kbps) / 1000.0)
            over = max(0.0, used_mbps_raw - cap)
            used_mbps = min(used_mbps_raw, cap)
            util = used_mbps / cap if cap > 0 else 0.0
            throughputs.append(used_mbps)
            utilizations.append(util)
            total_used += used_mbps_raw
            total_over += over
    else:
        # Per-direction clamp, then aggregate by undirected link
        # This matches real MN where rx/tx are each bounded by cap
        throughputs = []
        utilizations = []
        seen = set()
        for u, v in env_eval.graph.edges():
            key = frozenset((u, v))
            if key in seen:
                continue
            seen.add(key)

            cap = float(cap_und.get(key, DEFAULT_CAP))  # per-direction cap

            # Direction u->v
            raw_uv = max(0.0, float(load.get((u, v), 0.0)) / 1000.0)
            clamped_uv = min(raw_uv, cap)
            over_uv = max(0.0, raw_uv - cap)

            # Direction v->u
            raw_vu = max(0.0, float(load.get((v, u), 0.0)) / 1000.0)
            clamped_vu = min(raw_vu, cap)
            over_vu = max(0.0, raw_vu - cap)

            # Undirected: sum of clamped directions / total capacity
            total_cap = 2.0 * cap
            used_clamped = clamped_uv + clamped_vu
            util = used_clamped / total_cap if total_cap > 0 else 0.0

            throughputs.append(used_clamped)
            utilizations.append(util)

            total_used += (raw_uv + raw_vu)
            total_over += (over_uv + over_vu)

    demand_satisfaction = 1.0 if total_used <= 1e-12 else float(1.0 - (total_over / total_used))
    demand_satisfaction = max(0.0, min(1.0, demand_satisfaction))

    # ============================================================
    # (5) 聚合輸出
    # ============================================================
    if not throughputs:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_throughput = float(np.mean(throughputs))
    max_link_utilization = float(max(utilizations) * 100.0)

    # ============================================================
    # (5b) Read delay/loss from graph edges (fluid queue model)
    # ============================================================
    delays_list = []
    losses_list = []
    if directed:
        for (u, v) in load:
            d = float(env_eval.graph[u][v][0].get("delay", 0.0)) * 1000.0  # s → ms
            l = float(env_eval.graph[u][v][0].get("pkloss", 0.0))
            delays_list.append(d)
            losses_list.append(l)
    else:
        seen_dl = set()
        for u, v in env_eval.graph.edges():
            key = frozenset((u, v))
            if key in seen_dl:
                continue
            seen_dl.add(key)
            d_uv = float(env_eval.graph[u][v][0].get("delay", 0.0)) * 1000.0
            d_vu = float(env_eval.graph[v][u][0].get("delay", 0.0)) * 1000.0
            l_uv = float(env_eval.graph[u][v][0].get("pkloss", 0.0))
            l_vu = float(env_eval.graph[v][u][0].get("pkloss", 0.0))
            delays_list.append(max(d_uv, d_vu))
            losses_list.append(max(l_uv, l_vu))

    avg_delay = float(np.mean(delays_list)) if delays_list else 0.0
    avg_packet_loss = float(np.mean(losses_list)) if losses_list else 0.0

    # ============================================================
    # (6) 可選：寫 per-link CSV（給 sim vs real 分析用）
    #     格式對齊 real 的 Metrics/*_net_metrics_directed.csv
    #     directed mode: 每個 (u->v) 一行
    #     undirected mode: 每個 link 一行（兩個方向合併）
    # ============================================================
    if out_link_csv is not None:
        os.makedirs(os.path.dirname(out_link_csv) or ".", exist_ok=True)
        with open(out_link_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "node1", "node2", "free_bw", "used_bw",
                "delay", "pkloss", "queue_pkts", "capacity_mbps", "utilization",
            ])
            if directed:
                for (u, v), used_bw_kbps in sorted(load.items()):
                    cap = float(cap_dir.get((u, v), DEFAULT_CAP))
                    cap_kbps = cap * 1000.0
                    used_kbps = float(used_bw_kbps)
                    clamped_kbps = min(used_kbps, cap_kbps)
                    free_kbps = max(0.0, cap_kbps - clamped_kbps)
                    util = clamped_kbps / cap_kbps if cap_kbps > 0 else 0.0
                    d_ms = float(env_eval.graph[u][v][0].get("delay", 0.0)) * 1000.0
                    l_pct = float(env_eval.graph[u][v][0].get("pkloss", 0.0))
                    q_pkts = int(env_eval.graph[u][v][0].get("queue_pkts", 0))
                    writer.writerow([
                        u + 1, v + 1,
                        round(free_kbps, 6), round(used_kbps, 6),
                        round(d_ms, 6), round(l_pct, 6),
                        q_pkts, cap, round(util, 6),
                    ])
            else:
                seen_csv = set()
                for u, v in env_eval.graph.edges():
                    key = frozenset((u, v))
                    if key in seen_csv:
                        continue
                    seen_csv.add(key)
                    cap = float(cap_und.get(key, DEFAULT_CAP))
                    cap_kbps = cap * 1000.0
                    used_uv = float(load.get((u, v), 0.0))
                    used_vu = float(load.get((v, u), 0.0))
                    clamped_uv = min(used_uv, cap_kbps)
                    clamped_vu = min(used_vu, cap_kbps)
                    total_used_kbps = clamped_uv + clamped_vu
                    total_cap_kbps = 2.0 * cap_kbps
                    free_kbps = max(0.0, total_cap_kbps - total_used_kbps)
                    util = total_used_kbps / total_cap_kbps if total_cap_kbps > 0 else 0.0
                    d_uv = float(env_eval.graph[u][v][0].get("delay", 0.0)) * 1000.0
                    d_vu = float(env_eval.graph[v][u][0].get("delay", 0.0)) * 1000.0
                    l_uv = float(env_eval.graph[u][v][0].get("pkloss", 0.0))
                    l_vu = float(env_eval.graph[v][u][0].get("pkloss", 0.0))
                    q_uv = int(env_eval.graph[u][v][0].get("queue_pkts", 0))
                    q_vu = int(env_eval.graph[v][u][0].get("queue_pkts", 0))
                    writer.writerow([
                        u + 1, v + 1,
                        round(free_kbps, 6), round(total_used_kbps, 6),
                        round(max(d_uv, d_vu), 6), round(max(l_uv, l_vu), 6),
                        max(q_uv, q_vu), cap, round(util, 6),
                    ])

    # restore training queue state (only if we saved it in steady-state mode)
    if _saved_queue is not None:
        env_eval.queue_bits = _saved_queue

    return avg_delay, avg_packet_loss, avg_throughput, max_link_utilization, demand_satisfaction

def gen_path_vector(config, drl_paths):
    size = config["num_node"] + 1
    path_vector = np.zeros((((size-1)*(size-2)), config["num_link"]))
    agent_idx = 0
    if not drl_paths:
        return path_vector
    for i in range(1, size):
        for j in range(1, size):
            if i != j:
                for k in range(1,len(drl_paths[str(i)][str(j)][0])):
                    link=(drl_paths[str(i)][str(j)][0][k-1],drl_paths[str(i)][str(j)][0][k])
                    reversed_link=(drl_paths[str(i)][str(j)][0][k],drl_paths[str(i)][str(j)][0][k-1])
                    if link in link_index:
                        index = link_index[link]
                    else:
                        index = link_index[reversed_link]
                    path_vector[agent_idx][index] =  100
                agent_idx=agent_idx+1
    return path_vector

def gen_link_index(config):
    with open(config["bw_file"], 'r') as file:
        for line in file:
            data = line.strip().split(',')
            src, dst, _, _ = int(data[0]), int(data[1]), data[2], float(data[3])
            add_link(src,dst)

def add_link(node1, node2):
    link = (node1, node2)
    reversed_link = (node2, node1)

    if link in link_index:
        index = link_index[link]
    elif reversed_link in link_index:
        index = link_index[reversed_link]
    else:
        index = len(link_index) 
        link_index[link] = index
