"""
Compute the theoretical minimum directed MLU for each TM using LP or ILP.

Modes:
  1) LP relaxation (flow splitting allowed) → tightest lower bound
  2) ILP / --no-split (single-path only)   → true single-path optimum

Topologies:
  --topo geant    (23-node, default)
  --topo 32node   (32-node, requires --variant 24tm|144tm)

Usage:
    python paper/bounds/check_min_mlu.py                       # LP, all TMs, Geant
    python paper/bounds/check_min_mlu.py --tm_ids 0 3          # specific TMs
    python paper/bounds/check_min_mlu.py --no-split --tm_ids 12 14  # ILP, single-path
    python paper/bounds/check_min_mlu.py --topo 32node --variant 24tm --tm-scale 1.0
    python paper/bounds/check_min_mlu.py --topo 32node --variant 24tm --tm-scale 3.0
"""
import os, sys, json, argparse, time, io, contextlib, multiprocessing
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

# ---- paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# repo root is two levels up: paper/bounds/ -> paper/ -> repo
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ENERO_BASE = os.path.join(
    PROJECT_DIR,
    "A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main",
    "Enero_datasets", "dataset_sing_top", "data",
    "results_my_3_tops_unif_05-1",
)

# Geant defaults (kept for backward compat)
GRAPH_FILE = os.path.join(ENERO_BASE, "NEW_Geant", "TRAIN", "Geant.graph")
K_PATHS_FILE = os.path.join(PROJECT_DIR, "dataset", "geant_traffic", "k_paths.json")
BW_FILE = os.path.join(PROJECT_DIR, "dataset", "geant_traffic", "bw_r.txt")


def load_graph(graph_file):
    """Parse DEFO .graph file → (num_nodes, directed_edges[(src,dst)], capacities[])"""
    with open(graph_file) as f:
        lines = f.readlines()
    # First line: NODES <n>
    # Then: EDGES <m>
    # Then m lines: label src dst weight bw delay
    idx = 0
    while idx < len(lines) and "NODES" not in lines[idx]:
        idx += 1
    num_nodes = int(lines[idx].strip().split()[1])
    idx += 1
    # skip node lines
    while idx < len(lines) and "EDGES" not in lines[idx]:
        idx += 1
    num_edges = int(lines[idx].strip().split()[1])
    idx += 1
    # skip "label src dst weight bw delay" header
    if idx < len(lines) and "label" in lines[idx]:
        idx += 1

    edges = []
    caps = []
    for i in range(num_edges):
        parts = lines[idx + i].strip().split()
        src, dst = int(parts[1]), int(parts[2])
        bw = float(parts[4])
        edges.append((src, dst))
        caps.append(bw)
    return num_nodes, edges, caps


def load_bw_file(bw_file):
    """Parse bw_r.txt → dict[(src,dst)] = capacity_Mbps"""
    cap_dict = {}
    with open(bw_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            s, d = int(parts[0]), int(parts[1])
            bw = float(parts[3])
            cap_dict[(s, d)] = bw
    return cap_dict


def load_demands(tm_file):
    """Parse DEFO .demands file → demands[(src,dst)] = bw_kbps"""
    demands = {}
    with open(tm_file) as f:
        lines = f.readlines()
    for line in lines[2:]:  # skip header lines
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        src, dst = int(parts[1]), int(parts[2])
        bw = float(parts[3])
        if bw > 0:
            demands[(src, dst)] = bw
    return demands


def load_k_paths(k_paths_file):
    """Load K shortest paths per (src, dst) pair.
    Handles two formats:
      - Nested: {"src": {"dst": [[path], ...]}}  (1-indexed, Geant/32node original)
      - Flat:   {"src:dst": [[path], ...]}        (0-indexed, Enero converted)
    """
    with open(k_paths_file) as f:
        data = json.load(f)
    paths = {}
    sample_key = next(iter(data))
    if ":" in sample_key:
        # Flat format: "src:dst" → [[path], ...]
        for key, path_list in data.items():
            src, dst = key.split(":")
            paths[(int(src), int(dst))] = path_list
    else:
        # Nested format: {src: {dst: [[path], ...]}}
        for src_s, dsts in data.items():
            for dst_s, path_list in dsts.items():
                src, dst = int(src_s), int(dst_s)
                paths[(src, dst)] = path_list
    return paths


def path_to_directed_edges(path):
    """Convert node path [1, 3, 2] → directed edges [(1,3), (3,2)]"""
    return [(path[i], path[i+1]) for i in range(len(path) - 1)]


def make_undirected_cap(cap_dict):
    """Merge directed edges into undirected: cap(a,b) = cap(a→b) + cap(b→a)"""
    undir = {}
    for (s, d), bw in cap_dict.items():
        key = (min(s, d), max(s, d))
        undir[key] = undir.get(key, 0.0) + bw
    return undir


def path_to_undirected_edges(path):
    """Convert node path → undirected edges [(min,max), ...]"""
    return [(min(path[i], path[i+1]), max(path[i], path[i+1]))
            for i in range(len(path) - 1)]


def solve_min_mlu_kpath(demands, k_paths, cap_dict, verbose=False,
                         undirected=False, max_k=None):
    """
    K-path restricted LP:
      minimize t
      s.t. sum_k x_{i,k} = 1   for each pair i
           sum_{i,k: e in path(i,k)} d_i * x_{i,k} <= t * C_e   for each edge e
           x_{i,k} >= 0, t >= 0

    If undirected=True, edges are merged: cap(a,b) = cap(a→b) + cap(b→a),
    and both directions of traffic through (a,b) count toward the same capacity.
    """
    pairs = sorted(demands.keys())
    N = len(pairs)
    K_raw = min(len(k_paths.get(pairs[0], [])), 20) if pairs else 20
    K = min(K_raw, max_k) if max_k else K_raw

    # Edge conversion functions
    if undirected:
        edge_fn = path_to_undirected_edges
        eff_cap = make_undirected_cap(cap_dict)
    else:
        edge_fn = path_to_directed_edges
        eff_cap = dict(cap_dict)

    # Build edge set from all paths
    edge_set = set()
    for pair in pairs:
        for path in k_paths.get(pair, [])[:K]:
            for e in edge_fn(path):
                edge_set.add(e)
    # Also include all edges from eff_cap
    for e in eff_cap:
        edge_set.add(e)
    edges = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    E = len(edges)

    # Variables: [x_0_0, x_0_1, ..., x_{N-1}_{K-1}, t]
    num_x = N * K
    num_vars = num_x + 1  # +1 for t

    # Objective: minimize t
    c = np.zeros(num_vars)
    c[-1] = 1.0  # minimize t

    # Equality: sum_k x_{i,k} = 1
    A_eq = np.zeros((N, num_vars))
    b_eq = np.ones(N)
    for i, pair in enumerate(pairs):
        paths_for_pair = k_paths.get(pair, [])
        for k in range(min(len(paths_for_pair), K)):
            A_eq[i, i * K + k] = 1.0

    # Inequality: sum d_i * x_{i,k} - C_e * t <= 0
    A_ub = np.zeros((E, num_vars))
    b_ub = np.zeros(E)
    for i, pair in enumerate(pairs):
        d_i = demands[pair]
        paths_for_pair = k_paths.get(pair, [])
        for k in range(min(len(paths_for_pair), K)):
            for e in edge_fn(paths_for_pair[k]):
                if e in edge_to_idx:
                    eidx = edge_to_idx[e]
                    A_ub[eidx, i * K + k] += d_i

    for eidx, e in enumerate(edges):
        cap = eff_cap.get(e, 1e-6)
        A_ub[eidx, -1] = -cap  # -C_e * t

    # Bounds
    bounds = [(0, None)] * num_x + [(0, None)]  # t >= 0

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        opt_mlu = result.x[-1]
        if verbose:
            # Find bottleneck edges
            x_sol = result.x[:num_x]
            link_util = np.zeros(E)
            for i, pair in enumerate(pairs):
                d_i = demands[pair]
                paths_for_pair = k_paths.get(pair, [])
                for k in range(min(len(paths_for_pair), K)):
                    frac = x_sol[i * K + k]
                    if frac > 1e-8:
                        for e in path_to_directed_edges(paths_for_pair[k]):
                            if e in edge_to_idx:
                                link_util[edge_to_idx[e]] += d_i * frac

            # Top 5 utilized edges
            util_ratio = np.array([
                link_util[j] / eff_cap.get(edges[j], 1e-6)
                for j in range(E)
            ])
            top5 = np.argsort(util_ratio)[-5:][::-1]
            mode_str = "undirected" if undirected else "directed"
            print(f"  Top 5 bottleneck edges ({mode_str}):")
            for j in top5:
                e = edges[j]
                cap = eff_cap.get(e, 0)
                print(f"    {e[0]:>2} → {e[1]:>2}  util={link_util[j]:.1f}  cap={cap:.1f}  ratio={util_ratio[j]:.4f}")

            # How many pairs use flow splitting
            n_split = 0
            for i in range(N):
                active = sum(1 for k in range(K) if x_sol[i*K+k] > 0.01)
                if active > 1:
                    n_split += 1
            print(f"  Pairs using flow splitting: {n_split}/{N}")

        return opt_mlu
    else:
        print(f"  LP failed: {result.message}")
        return None


def solve_min_mlu_kpath_nosplit(demands, k_paths, cap_dict, verbose=False,
                                undirected=False, time_limit=300, max_k=None):
    """
    K-path restricted ILP (NO flow splitting):
      minimize t
      s.t. sum_k x_{i,k} = 1   for each pair i     (pick exactly one path)
           sum_{i,k: e in path(i,k)} d_i * x_{i,k} <= t * C_e
           x_{i,k} ∈ {0, 1},  t >= 0
    """
    pairs = sorted(demands.keys())
    N = len(pairs)
    K_raw = min(len(k_paths.get(pairs[0], [])), 20) if pairs else 20
    K = min(K_raw, max_k) if max_k else K_raw

    if undirected:
        edge_fn = path_to_undirected_edges
        eff_cap = make_undirected_cap(cap_dict)
    else:
        edge_fn = path_to_directed_edges
        eff_cap = dict(cap_dict)

    # Build edge set
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
    num_vars = num_x + 1  # +1 for t

    # Objective: minimize t
    c = np.zeros(num_vars)
    c[-1] = 1.0

    # Integrality: x_{i,k} = integer (binary via bounds), t = continuous
    integrality = np.ones(num_vars, dtype=int)
    integrality[-1] = 0

    # Bounds: x in [0,1], t in [0, inf)
    lb = np.zeros(num_vars)
    ub = np.full(num_vars, 1.0)
    ub[-1] = 1e6  # large upper bound for t

    # Equality: sum_k x_{i,k} = 1
    A_eq = np.zeros((N, num_vars))
    for i, pair in enumerate(pairs):
        paths_for_pair = k_paths.get(pair, [])
        for k in range(min(len(paths_for_pair), K)):
            A_eq[i, i * K + k] = 1.0

    # Inequality: sum d_i * x_{i,k} - C_e * t <= 0
    A_ub = np.zeros((E, num_vars))
    for i, pair in enumerate(pairs):
        d_i = demands[pair]
        paths_for_pair = k_paths.get(pair, [])
        for k in range(min(len(paths_for_pair), K)):
            for e in edge_fn(paths_for_pair[k]):
                if e in edge_to_idx:
                    eidx = edge_to_idx[e]
                    A_ub[eidx, i * K + k] += d_i
    for eidx, e in enumerate(edges):
        cap = eff_cap.get(e, 1e-6)
        A_ub[eidx, -1] = -cap

    # Combine constraints
    constraints = [
        LinearConstraint(csc_matrix(A_eq), lb=np.ones(N), ub=np.ones(N)),
        LinearConstraint(csc_matrix(A_ub), ub=np.zeros(E)),
    ]
    bounds = Bounds(lb=lb, ub=ub)

    options = {"time_limit": time_limit}
    t0 = time.time()
    result = milp(c, integrality=integrality, bounds=bounds,
                  constraints=constraints, options=options)
    elapsed = time.time() - t0

    has_solution = result.success or (result.x is not None and np.isfinite(result.x[-1]))
    if has_solution:
        opt_mlu = result.x[-1]
        tag = "" if result.success else " [incumbent, not proved optimal]"
        if verbose:
            x_sol = result.x[:num_x]
            link_util = np.zeros(E)
            for i, pair in enumerate(pairs):
                d_i = demands[pair]
                paths_for_pair = k_paths.get(pair, [])
                for k in range(min(len(paths_for_pair), K)):
                    if x_sol[i * K + k] > 0.5:  # binary → round
                        for e in edge_fn(paths_for_pair[k]):
                            if e in edge_to_idx:
                                link_util[edge_to_idx[e]] += d_i

            util_ratio = np.array([
                link_util[j] / eff_cap.get(edges[j], 1e-6) for j in range(E)
            ])
            top5 = np.argsort(util_ratio)[-5:][::-1]
            mode_str = "undirected" if undirected else "directed"
            print(f"  Top 5 bottleneck edges ({mode_str}, no-split):{tag}")
            for j in top5:
                e = edges[j]
                cap = eff_cap.get(e, 0)
                print(f"    {e[0]:>2} -> {e[1]:>2}  util={link_util[j]:.1f}"
                      f"  cap={cap:.1f}  ratio={util_ratio[j]:.4f}")
            print(f"  Solve time: {elapsed:.1f}s")

        return opt_mlu
    else:
        print(f"  ILP failed: {result.message} ({elapsed:.1f}s)")
        return None


def enumerate_near_optimal(demands, k_paths, cap_dict, undirected=False,
                           margin=0.01, max_solutions=50, time_limit=60,
                           max_k=None):
    """
    Find how many distinct single-path solutions have MLU within
    (1 + margin) of optimal, using iterative no-good cuts.

    Returns (opt_mlu, solutions_list) where each solution is
    (mlu, chosen_path_indices).
    """
    pairs = sorted(demands.keys())
    N = len(pairs)
    K_raw = min(len(k_paths.get(pairs[0], [])), 20) if pairs else 20
    K = min(K_raw, max_k) if max_k else K_raw

    if undirected:
        edge_fn = path_to_undirected_edges
        eff_cap = make_undirected_cap(cap_dict)
    else:
        edge_fn = path_to_directed_edges
        eff_cap = dict(cap_dict)

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
    num_vars = num_x + 1

    # Objective: minimize t
    c = np.zeros(num_vars)
    c[-1] = 1.0

    integrality = np.ones(num_vars, dtype=int)
    integrality[-1] = 0

    lb = np.zeros(num_vars)
    ub = np.full(num_vars, 1.0)
    ub[-1] = 1e6

    # Base constraints: flow conservation + capacity
    A_eq = np.zeros((N, num_vars))
    for i, pair in enumerate(pairs):
        paths_for_pair = k_paths.get(pair, [])
        for k in range(min(len(paths_for_pair), K)):
            A_eq[i, i * K + k] = 1.0

    A_ub = np.zeros((E, num_vars))
    for i, pair in enumerate(pairs):
        d_i = demands[pair]
        paths_for_pair = k_paths.get(pair, [])
        for k in range(min(len(paths_for_pair), K)):
            for e in edge_fn(paths_for_pair[k]):
                if e in edge_to_idx:
                    A_ub[edge_to_idx[e], i * K + k] += d_i
    for eidx, e in enumerate(edges):
        A_ub[eidx, -1] = -eff_cap.get(e, 1e-6)

    base_constraints = [
        LinearConstraint(csc_matrix(A_eq), lb=np.ones(N), ub=np.ones(N)),
        LinearConstraint(csc_matrix(A_ub), ub=np.zeros(E)),
    ]
    bounds_obj = Bounds(lb=lb, ub=ub)

    solutions = []
    nogood_rows = []  # accumulate no-good cut rows
    t_total = time.time()

    for sol_idx in range(max_solutions):
        # Build constraints: base + all no-good cuts so far
        constraints = list(base_constraints)
        if nogood_rows:
            A_ng = np.array(nogood_rows)
            # Each row: sum of x_{i,k_i} <= N-1 (at least one must differ)
            constraints.append(
                LinearConstraint(csc_matrix(A_ng),
                                 ub=np.full(len(nogood_rows), N - 1))
            )

        result = milp(c, integrality=integrality, bounds=bounds_obj,
                      constraints=constraints,
                      options={"time_limit": time_limit})

        if not result.success:
            break

        mlu = result.x[-1]
        # Check if within margin of first (optimal) solution
        if solutions and mlu > solutions[0][0] * (1 + margin):
            break

        x_sol = result.x[:num_x]
        # Extract chosen path index per pair
        chosen = []
        for i in range(N):
            best_k = max(range(K), key=lambda k: x_sol[i * K + k])
            chosen.append(best_k)

        solutions.append((mlu, tuple(chosen)))

        # Add no-good cut: sum_{i} x_{i, chosen[i]} <= N - 1
        ng_row = np.zeros(num_vars)
        for i, k in enumerate(chosen):
            ng_row[i * K + k] = 1.0
        nogood_rows.append(ng_row)

    elapsed = time.time() - t_total
    return solutions, elapsed


def get_topo_config(topo, variant, scale):
    """Return (bw_file, k_paths_file, dataset_dir, topo_name, num_nodes, train_tms, test_tms)."""
    if topo == "geant":
        bw_file = os.path.join(PROJECT_DIR, "dataset", "geant_traffic", "bw_r.txt")
        k_paths_file = os.path.join(PROJECT_DIR, "dataset", "geant_traffic", "k_paths.json")
        if scale == 5:
            dataset_subdir = "NEW_Geant"
        else:
            dataset_subdir = f"NEW_Geant_s{scale}"
        dataset_dir = os.path.join(ENERO_BASE, dataset_subdir, "TRAIN")
        topo_name = "Geant"
        num_nodes = 23
        train_tms = [13, 15, 17, 19, 20, 22, 23, 0, 1, 2, 4, 7, 8, 9, 11]
        test_tms = [3, 10, 12, 14, 21]
    elif topo == "32node":
        bw_file = os.path.join(PROJECT_DIR, "dataset", "32node_traffic", "bw_r.txt")
        k_paths_file = os.path.join(PROJECT_DIR, "dataset", "32node_traffic", "k_paths.json")
        dataset_dir = os.path.join(ENERO_BASE, f"NEW_32node_{variant}", "TRAIN")
        topo_name = "32node"
        num_nodes = 32
        n_tms = 144 if variant == "144tm" else 24
        train_tms = list(range(n_tms))
        test_tms = []
    else:
        raise ValueError(f"Unknown topology: {topo}")
    return bw_file, k_paths_file, dataset_dir, topo_name, num_nodes, train_tms, test_tms


# ---- multiprocessing worker ----
_worker_k_paths = None
_worker_cap_dict = None
_worker_opts = None


def _init_worker(k_paths, cap_dict, opts):
    global _worker_k_paths, _worker_cap_dict, _worker_opts
    _worker_k_paths = k_paths
    _worker_cap_dict = cap_dict
    _worker_opts = opts


def _solve_one_tm(task):
    tm_id, demands_adj, demands_raw_mbps, label = task
    k_paths = _worker_k_paths
    cap_dict = _worker_cap_dict
    opts = _worker_opts
    buf = io.StringIO()
    opt_mlu = None
    total_demand = sum(demands_raw_mbps.values())

    with contextlib.redirect_stdout(buf):
        if opts['enumerate']:
            solutions, elapsed = enumerate_near_optimal(
                demands_adj, k_paths, cap_dict,
                undirected=opts['undirected'], margin=opts['margin'],
                max_solutions=opts['max_solutions'],
                time_limit=opts['time_limit'], max_k=opts['max_k'])
            if solutions:
                opt_mlu = solutions[0][0]
                if len(solutions) > 1:
                    diffs = [sum(1 for a, b in zip(solutions[0][1], s[1]) if a != b)
                             for s in solutions[1:]]
                    diff_str = f", diffs from opt: {min(diffs)}~{max(diffs)} pairs"
                else:
                    diff_str = ""
                print(f"TM-{tm_id:02d} [{label}]: opt MLU = {opt_mlu:.4f}  |  "
                      f"{len(solutions)} solutions within {opts['margin']*100:.0f}%  "
                      f"(MLU range: {solutions[0][0]:.4f}~{solutions[-1][0]:.4f}"
                      f"{diff_str})  [{elapsed:.1f}s]")
            else:
                print(f"TM-{tm_id:02d} [{label}]: ILP failed")
        elif opts['no_split']:
            opt_mlu = solve_min_mlu_kpath_nosplit(
                demands_adj, k_paths, cap_dict,
                verbose=opts['verbose'], undirected=opts['undirected'],
                time_limit=opts['time_limit'], max_k=opts['max_k'])
        else:
            opt_mlu = solve_min_mlu_kpath(
                demands_adj, k_paths, cap_dict,
                verbose=opts['verbose'], undirected=opts['undirected'],
                max_k=opts['max_k'])

        if not opts['enumerate'] and opt_mlu is not None:
            print(f"TM-{tm_id:02d} [{label}]: min {opts['mode_str'].lower()} MLU = "
                  f"{opt_mlu:.4f}  (total demand = {total_demand:.1f} Mbps)")

    return (tm_id, opt_mlu, total_demand, buf.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topo", choices=["geant", "32node"], default="geant",
                        help="Topology (default: geant)")
    parser.add_argument("--variant", choices=["24tm", "144tm"], default="24tm",
                        help="32-node TM variant (default: 24tm)")
    parser.add_argument("--tm_ids", nargs="+", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--undirected", action="store_true",
                        help="Merge bidirectional capacity (undirected MLU)")
    parser.add_argument("--no-split", action="store_true",
                        help="ILP: force single-path per pair (no flow splitting)")
    parser.add_argument("--enumerate", action="store_true",
                        help="Count near-optimal single-path solutions")
    parser.add_argument("--margin", type=float, default=0.01,
                        help="Margin above optimal for enumeration (default: 0.01 = 1%%)")
    parser.add_argument("--max-solutions", type=int, default=50,
                        help="Max solutions to enumerate (default: 50)")
    parser.add_argument("--time-limit", type=int, default=300,
                        help="ILP time limit per TM in seconds (default: 300)")
    parser.add_argument("--scale", type=int, default=5,
                        help="Geant TM scale version: 3/4/5 (default: 5)")
    parser.add_argument("--tm-scale", type=float, default=1.0,
                        help="Multiply all demands by this factor (default: 1.0)")
    parser.add_argument("--max-k", type=int, default=None,
                        help="Limit K paths per pair (default: use all). Smaller = faster ILP")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--all-tms", action="store_true",
                        help="Run over every .demands file in dataset_dir/TM "
                             "(ignores --tm_ids; overrides default train+test).")
    parser.add_argument("--save-json", default=None,
                        help="Write {tm_id_str: mlu} dict to this JSON path. "
                             "Appends/updates key inferred from --topo / --scale / split mode.")
    parser.add_argument("--kp-file", default=None,
                        help="Override the candidate-path file (default: the topo's "
                             "canonical k_paths.json). Use with the K-extension file "
                             "from dataset/extend_k_paths.py, e.g. "
                             "dataset/32node_traffic/k_paths_k30_ext.json + --max-k 25/30.")
    args = parser.parse_args()

    bw_file, k_paths_file, dataset_dir, topo_name, num_nodes, \
        train_tms, test_tms = get_topo_config(args.topo, args.variant, args.scale)
    if args.kp_file:
        k_paths_file = os.path.abspath(args.kp_file)
        print(f"[--kp-file] candidate paths overridden: {k_paths_file}")

    all_tms = train_tms + test_tms
    if args.tm_ids is not None:
        all_tms = args.tm_ids
    if args.all_tms:
        tm_dir = os.path.join(dataset_dir, "TM")
        demand_files = sorted(
            f for f in os.listdir(tm_dir)
            if f.startswith(f"{topo_name}.") and f.endswith(".demands")
        )
        all_tms = sorted(int(f.split(".")[1]) for f in demand_files)
        print(f"[--all-tms] discovered {len(all_tms)} .demands files in {tm_dir}")

    # Load topology
    cap_dict = load_bw_file(bw_file)
    k_paths = load_k_paths(k_paths_file)

    split_str = "NO-SPLIT (ILP)" if args.no_split else "SPLIT (LP)"
    mode_str = "UNDIRECTED" if args.undirected else "DIRECTED"
    topo_label = f"{topo_name} {num_nodes}-node"
    if args.topo == "32node":
        topo_label += f" ({args.variant})"
    scale_label = f"tm_scale={args.tm_scale}" if args.tm_scale != 1.0 else f"scale={args.scale}"
    print(f"Topology: {topo_label}, {len(cap_dict)} directed edges, "
          f"mode={mode_str}, {split_str}, {scale_label}")
    K_avail = len(next(iter(k_paths.values())))
    K_eff = min(K_avail, args.max_k) if args.max_k else K_avail
    print(f"K-paths: {len(k_paths)} pairs, K={K_eff} (of {K_avail})")
    print(f"Capacity range: {min(cap_dict.values()):.2f} ~ {max(cap_dict.values()):.2f} Mbps")
    print()

    # Check node indexing consistency
    max_node_in_paths = max(max(p) for paths in k_paths.values() for p in paths)
    cap_max_node = max(max(s, d) for s, d in cap_dict.keys())
    print(f"Path node range: 0 ~ {max_node_in_paths} (1-based={max_node_in_paths == num_nodes})")
    print(f"Cap dict node range: up to {cap_max_node}")
    print()

    # Prepare per-TM tasks
    path_nodes = set()
    for pair in k_paths:
        path_nodes.update(pair)

    tasks = []
    for tm_id in sorted(all_tms):
        tm_file = os.path.join(dataset_dir, "TM", f"{topo_name}.{tm_id}.demands")
        if not os.path.exists(tm_file):
            print(f"TM-{tm_id:02d}: file not found, skipping")
            continue

        demands_raw = load_demands(tm_file)
        demands = {k: v / 1000.0 * args.tm_scale for k, v in demands_raw.items()}

        # Adjust demand keys if needed (0-based demands → 1-based paths)
        demand_nodes = set()
        for s, d in demands:
            demand_nodes.update([s, d])
        if 0 in demand_nodes and 0 not in path_nodes:
            demands_adj = {(s+1, d+1): bw for (s, d), bw in demands.items()}
        else:
            demands_adj = demands

        is_train = tm_id in train_tms
        label = "TRAIN" if is_train else "TEST "
        tasks.append((tm_id, demands_adj, demands, label))

    # Solver options shared across workers
    solve_opts = dict(
        enumerate=args.enumerate, no_split=args.no_split,
        verbose=args.verbose, undirected=args.undirected,
        margin=args.margin, max_solutions=args.max_solutions,
        time_limit=args.time_limit, max_k=args.max_k,
        mode_str=mode_str,
    )

    # Run tasks (parallel or sequential)
    if args.workers > 1:
        print(f"Running {len(tasks)} TMs with {args.workers} workers ...\n")
        with multiprocessing.Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(k_paths, cap_dict, solve_opts),
        ) as pool:
            raw_results = pool.map(_solve_one_tm, tasks)
    else:
        _init_worker(k_paths, cap_dict, solve_opts)
        raw_results = [_solve_one_tm(t) for t in tasks]

    # Print output in TM order and collect results
    results = []
    for tm_id, opt_mlu, total_demand, output in sorted(raw_results, key=lambda x: x[0]):
        if output:
            print(output, end="" if output.endswith("\n") else "\n")
        if opt_mlu is not None:
            results.append((tm_id, opt_mlu, total_demand))

    if results:
        mlus = [r[1] for r in results]
        print(f"\n{'='*60}")
        print(f"Summary: min MLU range = [{min(mlus):.4f}, {max(mlus):.4f}]")
        print(f"         mean = {np.mean(mlus):.4f}")
        print(f"         mode: {mode_str}")
        if args.tm_scale != 1.0:
            print(f"         tm_scale: {args.tm_scale}")
        print(f"         TMs with MLU > 1.0: {sum(1 for m in mlus if m > 1.0)}/{len(mlus)}")
        print(f"{'='*60}")

        if args.save_json:
            # Infer key: scale_{s}_{k}{split}_{mode}
            split_tag = "k4_ilp" if args.no_split else f"k{K_eff}_lp"
            mode_tag = "undirected" if args.undirected else "directed"
            key = f"scale_{args.scale}_{split_tag}_{mode_tag}"
            width = 4 if (args.all_tms and max(r[0] for r in results) > 99) else 2
            tm_to_mlu = {str(r[0]).zfill(width): round(r[1], 4) for r in results}

            # Merge with existing file if present.
            out_path = args.save_json
            payload = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path) as f:
                        payload = json.load(f)
                except Exception:
                    payload = {}
            meta = payload.setdefault("_meta", {})
            meta.setdefault("keys", {})
            meta["keys"][key] = {
                "topology": topo_name, "scale": args.scale,
                "k_paths": K_eff, "split": "ilp_no_split" if args.no_split else "lp_split",
                "mode": mode_tag, "n_tms": len(tm_to_mlu),
            }
            payload[key] = tm_to_mlu
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[save-json] wrote key='{key}' ({len(tm_to_mlu)} TMs) -> {out_path}")


if __name__ == "__main__":
    main()
