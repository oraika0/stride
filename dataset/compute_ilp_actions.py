"""Compute ILP single-path optimal routing ACTIONS for real-Mininet test TMs.

Unlike paper/bounds/check_min_mlu.py (which returns only the scalar optimal MLU), this
script extracts the per-OD-pair chosen path INDEX (0..K-1) so the result can be
fed into the real Mininet eval pipeline as a drl_paths-style action.

Demand source: parsed directly from the iperf3 client scripts the real network
actually injects (dataset/geant_traffic/23node_s<scale>/TM-<id>/Clients/*.sh),
NOT the ENERO .demands files -- so the ILP is solved on exactly the traffic the
Mininet eval sees. Restricted to the SAME K=20 candidate paths the DRL agent
chooses from (dataset/geant_traffic/k_paths.json) -> fair apples-to-apples
"optimal single path within the K-set" oracle baseline.

Step 1 of the ILP-real-test plan: just compute + save the actions. Conversion to
drl_paths.json + running the real eval are later steps.

Run:
  cd <repo root>
  python dataset/compute_ilp_actions.py \
      --tm_ids 03 10 12 14 21 --scale 3 --time-limit 300
"""
import os
import re
import sys
import glob
import json
import time
import argparse

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "paper", "bounds"))
from check_min_mlu import load_bw_file, load_k_paths, path_to_directed_edges

# Per-topology dataset paths -- the ILP solves on exactly the same bw / k_paths
# the DRL uses (apples-to-apples baseline). {scale} substituted for geant only.
TOPO_PATHS = {
    "geant": {
        "bw": "dataset/geant_traffic/bw_r.txt",
        "kp": "dataset/geant_traffic/k_paths.json",
        "tm_root": "dataset/geant_traffic/23node_s{scale}",
        "out": "dataset/geant_traffic/ilp_actions_s{scale}.json",
        "default_tms": ["03", "10", "12", "14", "21"],
    },
    "32node_144tm": {
        "bw": "dataset/32node_traffic/bw_r.txt",
        "kp": "dataset/32node_traffic/k_paths.json",
        "tm_root": "dataset/32node_traffic/32node_144tm",
        "out": "dataset/32node_traffic/ilp_actions_144tm.json",
        "default_tms": ["06", "41", "73", "108", "141"],
    },
}


def parse_demands_from_clients(tm_dir):
    """Parse iperf3 client scripts -> demands[(src,dst)] in Mbps.

    client_<NN>.sh holds host NN's outgoing flows; each line is
      iperf3 -c 10.0.0.<dst> -p <port> -u -b <rate><unit> ...
    IP 10.0.0.<d> maps to host d. Multiple flows to same dst are summed.
    """
    demands = {}
    client_files = glob.glob(os.path.join(tm_dir, "Clients", "client_*.sh"))
    if not client_files:
        raise FileNotFoundError(f"No client scripts under {tm_dir}/Clients/")
    for f in client_files:
        m = re.search(r"client_(\d+)\.sh", os.path.basename(f))
        src = int(m.group(1))
        with open(f) as fh:
            for line in fh:
                cm = re.search(r"-c\s+10\.0\.0\.(\d+).*?-b\s+([\d.]+)([kKmMgG])", line)
                if not cm:
                    continue
                dst = int(cm.group(1))
                val = float(cm.group(2))
                unit = cm.group(3).lower()
                mbps = {"k": val / 1000.0, "m": val, "g": val * 1000.0}[unit]
                if src != dst and mbps > 0:
                    demands[(src, dst)] = demands.get((src, dst), 0.0) + mbps
    return demands


def solve_ilp_actions(demands, k_paths, cap_dict, time_limit=300, max_k=20):
    """ILP single-path min-MLU, returns (opt_mlu, chosen_k, proved_optimal).

    chosen_k: dict[(src,dst)] = path index 0..K-1 (only for pairs with demand).
    Mirrors check_min_mlu.solve_min_mlu_kpath_nosplit but extracts the path
    assignment from result.x instead of discarding it.
    """
    pairs = sorted(demands.keys())
    N = len(pairs)
    K = min(20, max_k)
    edge_fn = path_to_directed_edges
    eff_cap = dict(cap_dict)

    # Edge set spanning all candidate paths + capacities.
    edge_set = set()
    for pair in pairs:
        for path in k_paths.get(pair, [])[:K]:
            for e in edge_fn(path):
                edge_set.add(e)
    for e in eff_cap:
        edge_set.add(e)
    edges = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    E = len(edges)

    num_x = N * K
    num_vars = num_x + 1               # + t (MLU)
    c = np.zeros(num_vars); c[-1] = 1.0
    integrality = np.ones(num_vars, dtype=int); integrality[-1] = 0
    lb = np.zeros(num_vars)
    ub = np.full(num_vars, 1.0); ub[-1] = 1e6

    # sum_k x_{i,k} = 1
    A_eq = np.zeros((N, num_vars))
    for i, pair in enumerate(pairs):
        for k in range(min(len(k_paths.get(pair, [])), K)):
            A_eq[i, i * K + k] = 1.0

    # sum d_i x_{i,k} - C_e t <= 0
    A_ub = np.zeros((E, num_vars))
    for i, pair in enumerate(pairs):
        d_i = demands[pair]
        for k in range(min(len(k_paths.get(pair, [])), K)):
            for e in edge_fn(k_paths[pair][k]):
                if e in edge_to_idx:
                    A_ub[edge_to_idx[e], i * K + k] += d_i
    for eidx, e in enumerate(edges):
        A_ub[eidx, -1] = -eff_cap.get(e, 1e-6)

    constraints = [
        LinearConstraint(csc_matrix(A_eq), lb=np.ones(N), ub=np.ones(N)),
        LinearConstraint(csc_matrix(A_ub), ub=np.zeros(E)),
    ]
    bounds = Bounds(lb=lb, ub=ub)

    t0 = time.time()
    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraints, options={"time_limit": time_limit})
    elapsed = time.time() - t0

    has_sol = res.success or (res.x is not None and np.isfinite(res.x[-1]))
    if not has_sol:
        return None, {}, False, elapsed

    x_sol = res.x[:num_x]
    chosen_k = {}
    for i, pair in enumerate(pairs):
        kk = int(np.argmax(x_sol[i * K:(i + 1) * K]))
        chosen_k[pair] = kk
    return float(res.x[-1]), chosen_k, bool(res.success), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="geant", choices=list(TOPO_PATHS))
    ap.add_argument("--tm_ids", nargs="+", default=None,
                    help="default per topology (geant 03/10/12/14/21, 32node 06/41/73/108/141)")
    ap.add_argument("--scale", type=int, default=3)   # geant dataset folder only
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--out", default=None, help="default per topology")
    args = ap.parse_args()

    p = TOPO_PATHS[args.topology]
    bw_file = os.path.join(_REPO, p["bw"])
    kp_file = os.path.join(_REPO, p["kp"])
    tm_root = os.path.join(_REPO, p["tm_root"].format(scale=args.scale))
    out_rel = (args.out or p["out"]).format(scale=args.scale)
    tm_ids  = args.tm_ids or p["default_tms"]
    cap_dict = load_bw_file(bw_file)
    k_paths = load_k_paths(kp_file)
    print(f"[setup] topology={args.topology}: {len(cap_dict)} directed links, "
          f"{len(k_paths)} candidate pairs (K<=20)")

    out = {}
    for tm in tm_ids:
        tm_dir = os.path.join(tm_root, f"TM-{tm}")
        demands = parse_demands_from_clients(tm_dir)
        tot = sum(demands.values())
        opt_mlu, chosen_k, proved, elapsed = solve_ilp_actions(
            demands, k_paths, cap_dict, time_limit=args.time_limit)
        if opt_mlu is None:
            print(f"TM-{tm}: ILP FAILED ({elapsed:.1f}s)")
            continue
        tag = "" if proved else " [incumbent, not proved optimal]"
        # Path-index histogram (how spread across the K candidates).
        hist = np.bincount([chosen_k[p] for p in chosen_k], minlength=20)
        nonzero_k = {int(k): int(v) for k, v in enumerate(hist) if v > 0}
        print(f"TM-{tm}: ILP MLU = {opt_mlu:.4f}{tag}  "
              f"| {len(demands)} demand pairs, total {tot:.1f} Mbps  "
              f"| {elapsed:.1f}s")
        print(f"         chosen-path-index histogram (k:count): {nonzero_k}")
        # Store as {"src:dst": chosen_k} for all demand pairs.
        out[tm] = {
            "ilp_mlu": opt_mlu,
            "proved_optimal": proved,
            "total_demand_mbps": tot,
            "n_demand_pairs": len(demands),
            "actions": {f"{s}:{d}": chosen_k[(s, d)] for (s, d) in chosen_k},
        }

    out_path = os.path.join(_REPO, out_rel)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
