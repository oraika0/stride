#!/usr/bin/env python
"""Build the theoretical-MLU (offered-load) paper tables for GÉANT / 32-node.

Single source of truth for Appendix A / §5.7.2. Takes no arguments.

Data flow (clear provenance):
  data/ilp_{geant,32node}_k20.json   ILP single-path min-MLU at K=20, directed,
                                     time-limit 300 s. EXPENSIVE -- precomputed by
                                     paper/bounds/check_min_mlu.py, NOT recomputed here.
                                     On 32-node it is a time-limited incumbent
                                     (not proven optimal), so re-solving may shift
                                     it slightly; that is why it is frozen as data.
  paper/bounds/check_min_mlu  (imported)    SP and LP are recomputed here every run --
                                     both are K-independent / cheap and fully
                                     reproducible, so they need no frozen data.
  config/env/*_config.py             train+test TM scope used for the summary.

Outputs (this folder):
  geant_lp_ilp.csv    per-TM  tm,split,total_demand_mbps,sp_mlu,lp_mlu,ilp_mlu (20)
  32node_lp_ilp.csv   per-TM  (all 144 rows; summary uses the 91 train+test subset)
  lp_ilp_summary.csv  SP + ILP stats per topo  (LP lives only in the per-TM CSVs)
  lp_ilp_summary.md   human-readable summary

ILP K=20 == the same candidate set the DRL agent chooses from (apples-to-apples).
To regenerate the ILP data itself (slow), see README.md.

Run:

    cd ~/stride
    conda activate stride
    python paper/tables/build_paper_table.py

"""
import os
import sys
import csv
import json
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "paper", "bounds"))
from check_min_mlu import (load_bw_file, load_k_paths, load_demands,      # noqa: E402
                           get_topo_config, solve_min_mlu_kpath,
                           path_to_directed_edges)

MAX_K = 20

# (name, topo, variant, scale, ilp_json, env_cfg)
TOPOS = [
    ("GÉANT (s3)",      "geant",  "24tm",  3, "data/ilp_geant_k20.json",
     "config/env/geant_config.py"),
    ("32-node (144tm)", "32node", "144tm", 5, "data/ilp_32node_k20.json",
     "config/env/32node_144tm_config.py"),
]


def env_scope(env_cfg):
    """train+test TM ids (ints) from the experiment's env config."""
    spec = importlib.util.spec_from_file_location("envcfg", os.path.join(REPO, env_cfg))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    c = m.config
    train = {int(x) for x in c.get("tm_list_train", [])}
    test = {int(x) for x in c.get("tm_list_test", [])}
    return train, test


def prep_demands(tm_file, path_nodes, tm_scale=1.0):
    """check_min_mlu's per-TM demand prep: kbps->Mbps, 0-based -> 1-based keys."""
    demands = {k: v / 1000.0 * tm_scale for k, v in load_demands(tm_file).items()}
    dn = set()
    for s, d in demands:
        dn.update((s, d))
    if 0 in dn and 0 not in path_nodes:
        demands = {(s + 1, d + 1): bw for (s, d), bw in demands.items()}
    return demands


def sp_mlu(demands, k_paths, cap_dict):
    """Shortest-path routing (candidate index 0 for every pair) -> directed MLU."""
    link = {}
    for pair, d in demands.items():
        for e in path_to_directed_edges(k_paths[pair][0]):
            link[e] = link.get(e, 0.0) + d
    return max(link[e] / cap_dict[e] for e in link)


def load_ilp_map(ilp_json):
    """data json is {'_meta':..., '<key>': {tm_str: mlu}} -> return the inner map."""
    d = json.load(open(os.path.join(HERE, ilp_json)))
    return next(v for k, v in d.items() if k != "_meta")


def build_topo(topo, variant, scale, ilp_json, env_cfg):
    bw_file, k_paths_file, dataset_dir, topo_name, _n, _tr, _te = \
        get_topo_config(topo, variant, scale)
    cap_dict = load_bw_file(bw_file)
    k_paths = load_k_paths(k_paths_file)
    path_nodes = {n for pair in k_paths for n in pair}
    ilp_map = load_ilp_map(ilp_json)
    train, test = env_scope(env_cfg)

    rows = []
    for tm_str in sorted(ilp_map, key=int):
        tm_id = int(tm_str)
        tm_file = os.path.join(dataset_dir, "TM", f"{topo_name}.{tm_id}.demands")
        if not os.path.exists(tm_file):
            print(f"  [warn] {topo_name} TM-{tm_id} demands missing, skipped")
            continue
        demands = prep_demands(tm_file, path_nodes)
        split = "train" if tm_id in train else ("test" if tm_id in test else "other")
        rows.append({
            "tm": tm_id,
            "split": split,
            "total_demand_mbps": round(sum(demands.values()), 1),
            "sp_mlu": round(sp_mlu(demands, k_paths, cap_dict), 4),
            "lp_mlu": round(solve_min_mlu_kpath(demands, k_paths, cap_dict,
                                                undirected=False, max_k=MAX_K), 4),
            "ilp_mlu": round(ilp_map[tm_str], 4),
        })
    return rows


def write_per_tm_csv(rows, path):
    cols = ["tm", "split", "total_demand_mbps", "sp_mlu", "lp_mlu", "ilp_mlu"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def summarize(name, rows):
    """SP + ILP stats over the train+test subset (LP dropped from the summary)."""
    sub = [r for r in rows if r["split"] in ("train", "test")]
    sp = np.array([r["sp_mlu"] for r in sub])
    ilp = np.array([r["ilp_mlu"] for r in sub])
    gap = (sp - ilp) / ilp * 100.0
    sd = lambda a: float(np.std(a, ddof=1))
    n_gt1 = int((sp > 1.0).sum())
    return {
        "testbed": name, "N": len(sub),
        "sp_mean": float(sp.mean()), "sp_std": sd(sp),
        "ilp_mean": float(ilp.mean()), "ilp_std": sd(ilp),
        "gap_mean": float(gap.mean()), "gap_std": sd(gap),
        "sp_gt1": n_gt1, "sp_gt1_frac": round(100.0 * n_gt1 / len(sub), 1),
    }


def write_summary(stats):
    with open(os.path.join(HERE, "lp_ilp_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    lines = [
        "# Testbed difficulty — offered-load SP / ILP (train+test)",
        "",
        "| Testbed | N | SP (OSPF) MLU | ILP MLU | SP–ILP gap (%) | SP MLU > 1.0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in stats:
        lines.append(
            f"| {s['testbed']} | {s['N']} | {s['sp_mean']:.2f} ± {s['sp_std']:.2f} "
            f"| {s['ilp_mean']:.2f} ± {s['ilp_std']:.2f} "
            f"| {s['gap_mean']:.0f} ± {s['gap_std']:.0f} "
            f"| {s['sp_gt1']}/{s['N']} ({s['sp_gt1_frac']:.0f}%) |")
    lines += [
        "",
        "> Offered-load (theoretical) SP / ILP MLU over the experiment TMs "
        "(train+test only); can exceed 1.0 and is **not** comparable to the "
        "real-Mininet measured MLU. ILP is single-path at K=20 (the agent's "
        "candidate set), directed, 300 s time limit. mean ± sample std (ddof=1). "
        "Per-TM LP (flow-split lower bound) is in the *_lp_ilp.csv files.",
        "",
    ]
    with open(os.path.join(HERE, "lp_ilp_summary.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    stats = []
    for name, topo, variant, scale, ilp_json, env_cfg in TOPOS:
        print(f"[{name}] building ...")
        rows = build_topo(topo, variant, scale, ilp_json, env_cfg)
        out_csv = os.path.join(HERE, f"{topo}_lp_ilp.csv")
        write_per_tm_csv(rows, out_csv)
        s = summarize(name, rows)
        stats.append(s)
        print(f"  {len(rows)} TMs -> {os.path.basename(out_csv)}  "
              f"| summary N={s['N']} SP={s['sp_mean']:.3f} ILP={s['ilp_mean']:.3f} "
              f"gap={s['gap_mean']:.0f}%")
    write_summary(stats)
    print("wrote lp_ilp_summary.{csv,md}")


if __name__ == "__main__":
    main()
