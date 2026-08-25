"""Unrestricted (edge-based) multicommodity-flow LP lower bound on min MLU.

2026-07-17, K-sufficiency analysis: closes the "is the frozen 20-candidate set
as good as ALL possible paths" question. Result on the 5 32node test TMs: the
best single-path assignment within the canonical 20 candidates sits 0.00-0.76%
(avg 0.2%) above this bound -> the candidate set, single-path restriction and
K>20 all have <1% headroom. Referenced by paper_fig/k_ablation/k_oracle_curve.

No candidate-set restriction, splitting allowed: the true floor over ALL
possible paths. Commodities aggregated by destination (standard, exact for
min-MLU). Same demand prep / units as build_paper_table.
"""
import os
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_min_mlu import load_bw_file, load_k_paths, load_demands, get_topo_config

def prep_demands(tm_file, path_nodes):
    demands = {k: v / 1000.0 for k, v in load_demands(tm_file).items()}
    dn = set()
    for s, d in demands:
        dn.update((s, d))
    if 0 in dn and 0 not in path_nodes:
        demands = {(s + 1, d + 1): bw for (s, d), bw in demands.items()}
    return demands

def edge_lp_min_mlu(demands, cap):
    edges = sorted(cap)
    E = len(edges)
    eidx = {e: i for i, e in enumerate(edges)}
    nodes = sorted({n for e in edges for n in e})
    nidx = {n: i for i, n in enumerate(nodes)}
    V = len(nodes)
    dests = sorted({d for (_, d) in demands})
    C = len(dests)
    nv = C * E + 1                      # f[c,e] >= 0, plus t

    # conservation: for commodity c (dest D), node v != D:
    #   out(v) - in(v) = demand(v -> D)
    rows = []
    rhs = []
    A = lil_matrix(((V - 1) * C, nv))
    r = 0
    for ci, D in enumerate(dests):
        for v in nodes:
            if v == D:
                continue
            for e in edges:
                if e[0] == v:
                    A[r, ci * E + eidx[e]] = 1.0
                elif e[1] == v:
                    A[r, ci * E + eidx[e]] = -1.0
            rhs.append(demands.get((v, D), 0.0))
            r += 1
    # capacity: sum_c f[c,e] - C_e * t <= 0
    Au = lil_matrix((E, nv))
    for ei in range(E):
        for ci in range(C):
            Au[ei, ci * E + ei] = 1.0
        Au[ei, -1] = -cap[edges[ei]]
    c_obj = np.zeros(nv)
    c_obj[-1] = 1.0
    res = linprog(c_obj, A_ub=Au.tocsr(), b_ub=np.zeros(E),
                  A_eq=A.tocsr(), b_eq=np.array(rhs),
                  bounds=[(0, None)] * nv, method="highs")
    assert res.status == 0, res.message
    return res.x[-1]

bw_file, kp_file, dataset_dir, topo_name, *_ = get_topo_config("32node", "144tm", 5)
cap = load_bw_file(bw_file)
kp = load_k_paths(kp_file)
path_nodes = {n for pair in kp for n in pair}

BEST = {6: 0.8043, 41: 0.6103, 73: 0.5193, 108: 0.4685, 141: 0.4570}   # per-TM best single-path in K=20 set
K20LP = {6: 0.8043, 41: 0.6103, 73: 0.5190, 108: 0.4680, 141: 0.4540}  # K=20-restricted split LP

print(f"{'TM':>6} {'edgeLP(all paths)':>18} {'K20-LP(split)':>14} {'best 1-path in K20':>19} {'K20set vs edgeLP':>17}")
for tm in [6, 41, 73, 108, 141]:
    tm_file = os.path.join(dataset_dir, "TM", f"{topo_name}.{tm}.demands")
    demands = prep_demands(tm_file, path_nodes)
    v = edge_lp_min_mlu(demands, cap)
    print(f"{tm:>6} {v:>18.4f} {K20LP[tm]:>14.4f} {BEST[tm]:>19.4f} {100*(BEST[tm]/v-1):>16.2f}%")
