"""Oracle K-prefix curve: ILP single-path min-MLU vs K on the 32node test TMs.

Same measurement conventions as raw/inbox/paper_table (build_paper_table.py):
same .demands files (tm_scale=1 -- the 144tm dataset is already at experiment
intensity), same canonical frozen k_paths.json, same directed ILP solver
(check_min_mlu.solve_min_mlu_kpath_nosplit, 300 s limit). Endpoints reuse the
paper-table values: K=1 == SP (candidate index 0, closed form) and K=20 ==
the frozen incumbent in paper_table/data/ilp_32node_k20.json, so the curve is
directly comparable with the published lp_ilp tables. K=2..19 solved here.

Answers: how small can K be before the CANDIDATE SET (not the learner)
becomes the bottleneck -- per TM, the smallest K whose optimum is within
REL_TOL of the K=20 optimum, plus each K's quality (K20/K) in %. The RL-side
answer lives in paper_fig/k_ablation (trained K in {10,15,20,25,30}).

Run:
  cd <repo root>
  python paper/bounds/k_oracle_curve.py
Writes k_oracle_curve_32node.csv and "Table 12. ...md" beside this script.
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_min_mlu import (load_bw_file, load_k_paths, load_demands,
                           get_topo_config, solve_min_mlu_kpath_nosplit,
                           path_to_directed_edges)

# Beside this script, like every other generator in paper/. This used to
# write into figures/k_ablation/, which put Table 12 in the folder that
# make_k_fig.py owns and made that one folder the only place two scripts
# wrote to -- one of them from a different directory.
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ILP_K20_JSON = os.path.join(_REPO, "paper", "tables", "data", "ilp_32node_k20.json")
LP_CSV = os.path.join(_REPO, "paper", "tables", "32node_lp_ilp.csv")
REL_TOL = 0.005   # "reaches the optimum" = within 0.5% relative.
# Reference = per-TM BEST value found across all solved K (the true optimum is
# non-increasing in K, so the min over solves is its best estimate) -- NOT the
# frozen K=20 incumbent: larger K makes the MILP harder, so at a fixed 300 s
# limit big-K incumbents can sit ABOVE small-K proven optima (observed: TM06
# K=4 hits the LP bound 0.8043 proven in 22 s, while the frozen K=20 incumbent
# is 0.8082 and K=19 times out at 0.8078). The LP column shows how close the
# best is to the flow-split lower bound (equal => proven single-path optimal).


def prep_demands(tm_file, path_nodes):
    """build_paper_table's demand prep: kbps->Mbps, 0-based -> 1-based keys."""
    demands = {k: v / 1000.0 for k, v in load_demands(tm_file).items()}
    dn = set()
    for s, d in demands:
        dn.update((s, d))
    if 0 in dn and 0 not in path_nodes:
        demands = {(s + 1, d + 1): bw for (s, d), bw in demands.items()}
    return demands


def sp_mlu(demands, k_paths, cap_dict):
    """K=1 oracle == SP routing (candidate index 0 for every pair)."""
    link = {}
    for pair, d in demands.items():
        for e in path_to_directed_edges(k_paths[pair][0]):
            link[e] = link.get(e, 0.0) + d
    return max(link[e] / cap_dict[e] for e in link)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm_ids", nargs="+", type=int, default=[6, 41, 73, 108, 141])
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--k-values", nargs="+", type=int, default=list(range(2, 20)))
    args = ap.parse_args()

    bw_file, k_paths_file, dataset_dir, topo_name, *_ = get_topo_config("32node", "144tm", 5)
    cap = load_bw_file(bw_file)
    kp = load_k_paths(k_paths_file)
    path_nodes = {n for pair in kp for n in pair}
    ilp20 = json.load(open(ILP_K20_JSON))
    ilp20 = next(v for k, v in ilp20.items() if k != "_meta")

    tms = args.tm_ids
    ks = [1] + sorted(set(args.k_values)) + [20]
    mlu = {tm: {} for tm in tms}
    for tm in tms:
        tm_file = os.path.join(dataset_dir, "TM", f"{topo_name}.{tm}.demands")
        demands = prep_demands(tm_file, path_nodes)
        mlu[tm][1] = sp_mlu(demands, kp, cap)
        mlu[tm][20] = float(ilp20[f"{tm:02d}"])   # frozen json keys are zero-padded
        print(f"TM-{tm}: K=1 (SP) {mlu[tm][1]:.4f} | K=20 (frozen) {mlu[tm][20]:.4f}", flush=True)
        for K in sorted(set(args.k_values)):
            t0 = time.time()
            v = solve_min_mlu_kpath_nosplit(demands, kp, cap, undirected=False,
                                            time_limit=args.time_limit, max_k=K)
            mlu[tm][K] = float(v) if v is not None else None
            print(f"TM-{tm} K={K:2d}: opt_mlu={v if v is None else round(float(v), 4)} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    # ---- reference = per-TM best across all solved K; LP bound for context ----
    ref = {tm: min(v for v in mlu[tm].values() if v is not None) for tm in tms}
    lp = {}
    with open(LP_CSV) as f:
        for r in csv.DictReader(f):
            lp[int(r["tm"])] = float(r["lp_mlu"])

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "k_oracle_curve_32node.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K"] + [f"TM{tm}" for tm in tms] + ["avg_mlu", "avg_quality_pct_of_best"])
        for K in ks:
            row = [mlu[tm][K] for tm in tms]
            qual = [100.0 * ref[tm] / mlu[tm][K] for tm in tms]
            w.writerow([K] + [f"{v:.4f}" for v in row] +
                       [f"{np.mean(row):.4f}", f"{np.mean(qual):.2f}"])
        w.writerow(["best"] + [f"{ref[tm]:.4f}" for tm in tms] +
                   [f"{np.mean(list(ref.values())):.4f}", "100.00"])
        w.writerow(["LP_bound"] + [f"{lp[tm]:.4f}" for tm in tms] +
                   [f"{np.mean([lp[tm] for tm in tms]):.4f}", ""])
    print(f"saved {csv_path}")

    # ---- markdown ----
    md = ["> **Oracle K-prefix curve (32node 144tm test TMs, directed single-path ILP, "
          "canonical k_paths prefix; K=1 = SP closed-form, K=20 = paper_table frozen "
          "incumbent, K=2..19 solved at 300 s). Reference = per-TM best across all K "
          "(large-K incumbents can exceed small-K proven optima at a fixed time "
          "limit); LP row = flow-split lower bound (equal => proven optimal).**\n",
          "| K | " + " | ".join(f"TM{tm}" for tm in tms) + " | avg MLU | % of best |",
          "| --: |" + " --: |" * (len(tms) + 2)]
    first_hit = {tm: None for tm in tms}
    for K in ks:
        cells = []
        for tm in tms:
            v = mlu[tm][K]
            hit = v is not None and v <= ref[tm] * (1 + REL_TOL)
            if hit and first_hit[tm] is None:
                first_hit[tm] = K
            cells.append(f"{v:.3f}" + ("  ✓" if hit else ""))
        avg = np.mean([mlu[tm][K] for tm in tms])
        qual = np.mean([100.0 * ref[tm] / mlu[tm][K] for tm in tms])
        md.append(f"| {K} | " + " | ".join(cells) + f" | {avg:.3f} | {qual:.1f}% |")
    md.append("| **LP bound** | " + " | ".join(f"{lp[tm]:.3f}" for tm in tms) +
              f" | {np.mean([lp[tm] for tm in tms]):.3f} | — |")
    md.append(f"\n**Smallest K within {REL_TOL*100:.1f}% of the best optimum, per TM**: " +
              ", ".join(f"TM{tm}: K={first_hit[tm]}" for tm in tms) +
              f" → overall K* = {max(v for v in first_hit.values() if v)}")
    # The thesis caption, like every other generated table, so the file drops
    # into the document without renaming. The CSV keeps its descriptive name:
    # it is the data behind the table, not the table.
    md_path = os.path.join(
        OUT_DIR,
        "Table 12. Candidate-path ILP and full-graph LP theoretical MLU on 32-node.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
