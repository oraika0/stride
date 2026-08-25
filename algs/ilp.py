import os
import json
import numpy as np


class ilp_agent:
    """Static ILP single-path optimal baseline (lookup stub, not a learner).

    Returns the per-OD-pair path index precomputed by
    dataset/compute_ilp_actions.py for the active test TM. Mirrors ospf_agent's
    minimal interface so it runs through the standard real-Mininet eval
    (testing_ma -> run_eval -> Ryu drl_paths.json) with NO pipeline changes.

    The action array follows testing_ma's agent_index order exactly:
    nested (i, j) for i in 1..num_node, j in 1..num_node, i != j. Each value
    is chosen_k in [0, K), consumed as all_path_list[i][j][chosen_k]. Pairs
    with no demand default to path index 0 (shortest); they carry no load so
    the choice does not affect MLU.
    """

    def __init__(self, args):
        self.args = args
        self.num_agents = args.num_agents
        self.num_node = args.num_node
        self.action_size = args.action_dim
        self.tm_id = int(args.tm_id)

        # 2026-06-04: pick the precomputed-actions file by topology so the same
        # ilp alg works on geant + 32node (the env's `topology` is 'geant'/'32node';
        # geant_directed/32node_144tm_directed inherit it). Falls back to the
        # config's ilp_actions_file for any other topology.
        _by_topo = {
            "geant":  "dataset/geant_traffic/ilp_actions_s3.json",
            "32node": "dataset/32node_traffic/ilp_actions_144tm.json",
        }
        actions_file = (_by_topo.get(getattr(args, "topology", None))
                        or getattr(args, "ilp_actions_file",
                                   "dataset/geant_traffic/ilp_actions_s3.json"))
        if not os.path.isabs(actions_file):
            # Resolve relative to the repo root (parent of algs/), NOT os.getcwd():
            # run_drl.py is spawned in a gnome-terminal whose cwd is the user's
            # home, not the repo, so a cwd-relative path would FileNotFoundError
            # -> __init__ raises -> gnome closes on error (2026-05-28 incident,
            # which then left the eval running a STALE all-shortest drl_paths.json).
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            actions_file = os.path.join(repo_root, actions_file)
        if not os.path.isfile(actions_file):
            raise FileNotFoundError(
                f"[ilp] actions file not found: {actions_file}. Run "
                f"dataset/compute_ilp_actions.py first.")
        with open(actions_file) as f:
            data = json.load(f)

        key = f"{self.tm_id:02d}"
        if key not in data:
            raise KeyError(
                f"[ilp] no precomputed actions for TM {key} in {actions_file}; "
                f"available={list(data.keys())}. Run dataset/compute_ilp_actions.py "
                f"--tm_ids {key} first.")
        self.actions = data[key]["actions"]          # {"src:dst": chosen_k}
        self.ilp_mlu = data[key].get("ilp_mlu")
        print(f"[ilp] TM {key}: {len(self.actions)} pair actions loaded, "
              f"analytical MLU={self.ilp_mlu:.4f} (src={actions_file})")

    def get_action(self, state, epsilon, **info):
        action = np.zeros(self.num_agents, dtype=int)
        idx = 0
        for i in range(1, self.num_node + 1):
            for j in range(1, self.num_node + 1):
                if i != j:
                    action[idx] = int(self.actions.get(f"{i}:{j}", 0))
                    idx += 1
        return action, {}

    def append_sample(self, info, next_state, reward):
        pass

    def update_target(self):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass

    def update(self):
        return {}
