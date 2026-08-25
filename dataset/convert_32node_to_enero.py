"""
Convert 32-node dataset (bw_r.txt + pkl TMs + k_paths.json) → Enero sim format.

Source:
  - bw_r.txt:     1-indexed, "src,dst,delay,cap_Mbps" per directed link
  - *.pkl:        [od_bin, tms], tms[i] is 32x32 matrix in kbps (0-indexed)
  - k_paths.json: 1-indexed nested dict {"src": {"dst": [[path], ...]}}

Target (Enero):
  - {topo}.graph:          0-indexed, "label src dst weight bw_kbps delay"
  - {topo}.{idx}.demands:  0-indexed, "label src dst bw_kbps"
  - k_shortest_paths.json: 0-indexed, {"src:dst": [[path], ...]}
  - Directory: TRAIN/ and EVALUATE/ (symlink ALL/ → TRAIN/)

Usage:
  python dataset/convert_32node_to_enero.py --variant 24tm
  python dataset/convert_32node_to_enero.py --variant 144tm

WHO CONSUMES THE OUTPUT
  The .demands files feed two unrelated consumers:
    1. The fluid-queue simulator (the vendored Enero gym env). That path is NOT
       maintained -- no STRIDE variant trains in simulation -- so nothing in the
       current workflow regenerates .demands for it.
    2. paper/bounds/check_min_mlu.py::load_demands, and through it every LP/ILP
       bound quoted in the paper tables. That path is live.
  So the files matter even though the simulator does not. Do not delete them
  because the simulator is unused.
"""

import argparse
import json
import os
import pickle
import numpy as np

# Repo root, derived from this file's location so the script survives a move.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_topology(bw_r_path):
    """Parse bw_r.txt → list of (src_0idx, dst_0idx, delay, cap_Mbps)."""
    links = []
    with open(bw_r_path) as f:
        for line in f:
            parts = line.strip().split(",")
            src = int(parts[0]) - 1   # 1-indexed → 0-indexed
            dst = int(parts[1]) - 1
            delay = int(parts[2])
            cap = int(parts[3])       # Mbps
            links.append((src, dst, delay, cap))
    nodes = set()
    for s, d, _, _ in links:
        nodes.add(s)
        nodes.add(d)
    return sorted(nodes), links


def write_graph(nodes, links, out_path, topo_name):
    """Write Enero-style .graph file."""
    with open(out_path, "w") as f:
        f.write(f"NODES {len(nodes)}\n")
        f.write("label x y\n")
        for n in nodes:
            f.write(f"N{n} 0.0 0.0\n")
        f.write(f"\nEDGES {len(links)}\n")
        f.write("label src dest weight bw delay\n")
        for i, (src, dst, delay, cap) in enumerate(links):
            bw_kbps = cap * 1000.0  # Mbps → kbps
            f.write(f"Link_{i} {src} {dst} 8840 {bw_kbps:.1f} {delay}\n")
    print(f"[OK] {out_path} ({len(nodes)} nodes, {len(links)} edges)")


def load_tms(pkl_path):
    """Load TMs from pkl: [od_bin, tms], tms[i] is NxN matrix (kbps)."""
    with open(pkl_path, "rb") as f:
        od_bin, tms = pickle.load(f)
    return tms


def write_demands(tms, out_dir, topo_name):
    """Write Enero-style .demands files (0-indexed, kbps)."""
    os.makedirs(out_dir, exist_ok=True)
    for idx, tm in enumerate(tms):
        tm = np.array(tm)
        demands = []
        N = tm.shape[0]
        for src in range(N):
            for dst in range(N):
                if src == dst:
                    continue
                bw = tm[src, dst]
                demands.append((src, dst, bw))

        out_path = os.path.join(out_dir, f"{topo_name}.{idx}.demands")
        with open(out_path, "w") as f:
            f.write(f"DEMANDS {len(demands)}\n")
            f.write("label src dest bw\n")
            for i, (src, dst, bw) in enumerate(demands):
                f.write(f"demand_{i} {src} {dst} {bw:.6f}\n")
    print(f"[OK] {len(tms)} demand files → {out_dir}")


def convert_kpaths(kpaths_1idx, out_path):
    """Convert 1-indexed nested dict → 0-indexed flat dict {"src:dst": [[path]]}."""
    ksp_0idx = {}
    for src_str, dst_dict in kpaths_1idx.items():
        for dst_str, paths in dst_dict.items():
            src_0 = int(src_str) - 1
            dst_0 = int(dst_str) - 1
            key = f"{src_0}:{dst_0}"
            paths_0 = [[node - 1 for node in path] for path in paths]
            ksp_0idx[key] = paths_0
    with open(out_path, "w") as f:
        json.dump(ksp_0idx, f, indent=2)
    print(f"[OK] k_shortest_paths.json: {len(ksp_0idx)} pairs → {out_path}")
    return ksp_0idx


def verify_kpaths_vs_topo(ksp, links):
    """Verify all path edges exist in topology."""
    link_set = set((s, d) for s, d, _, _ in links)
    bad = 0
    for key, paths in ksp.items():
        for path in paths:
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge not in link_set:
                    print(f"  [WARN] {key}: edge {edge[0]}→{edge[1]} not in topology!")
                    bad += 1
    if bad == 0:
        print("[OK] All k-path edges exist in topology")
    else:
        print(f"[WARN] {bad} invalid edges found!")


def analyze_demands(tms, links, tm_scale=1.0):
    """Quick feasibility analysis."""
    total_cap = sum(c for _, _, _, c in links) * 1000  # kbps
    print(f"\n=== Demand analysis (tm_scale={tm_scale}) ===")
    print(f"Total link capacity: {total_cap/1000:.0f} Mbps ({len(links)} directed links)")
    for i, tm in enumerate(tms):
        tm = np.array(tm) * tm_scale
        total = tm.sum()
        nonzero = np.count_nonzero(tm) - tm.shape[0]  # exclude diagonal
        lb_mlu = total / total_cap * 100
        print(f"  TM-{i:02d}: total={total/1000:8.1f} Mbps, pairs={nonzero}, "
              f"LB_MLU={lb_mlu:5.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Convert 32-node dataset to Enero format")
    parser.add_argument("--variant", choices=["24tm", "144tm"], default="24tm")
    parser.add_argument("--tm-scale", type=float, default=1.0,
                        help="Scale factor to preview demand feasibility (not applied to output)")
    parser.add_argument("--src-dir",
                        default=os.path.join(_REPO, "dataset", "32node_traffic"))
    parser.add_argument("--kpaths-src", default=None,
                        help="Source k_paths.json (default: {src-dir}/k_paths.json)")
    parser.add_argument("--dst-dir", default=None,
                        help="Output dir (default: Enero dataset_sing_top path)")
    args = parser.parse_args()

    topo_name = "32node"
    src_dir = args.src_dir
    bw_r_path = os.path.join(src_dir, "bw_r.txt")
    pkl_name = f"32node_tms_info_{args.variant}.pkl"
    pkl_path = os.path.join(src_dir, "traffic_generator", pkl_name)
    kpaths_path = args.kpaths_src or os.path.join(src_dir, "k_paths.json")

    if args.dst_dir:
        dst_base = args.dst_dir
    else:
        dst_base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main",
            "Enero_datasets", "dataset_sing_top", "data",
            "results_my_3_tops_unif_05-1",
            f"NEW_32node_{args.variant}",
        )

    print(f"=== Converting 32-node {args.variant} ===")
    print(f"Source: {src_dir}")
    print(f"Output: {dst_base}")

    # 1. Topology
    nodes, links = load_topology(bw_r_path)
    print(f"\nTopology: {len(nodes)} nodes, {len(links)} directed links")

    # 2. TMs
    tms = load_tms(pkl_path)
    print(f"TMs: {len(tms)} matrices, {np.array(tms[0]).shape}")

    # 3. Feasibility analysis
    analyze_demands(tms, links, tm_scale=1.0)
    if args.tm_scale != 1.0:
        analyze_demands(tms, links, tm_scale=args.tm_scale)

    # 4. k-paths
    with open(kpaths_path) as f:
        kpaths_raw = json.load(f)
    print(f"k_paths source: {kpaths_path}")

    # 5. Write output
    for split in ["TRAIN", "EVALUATE"]:
        split_dir = os.path.join(dst_base, split)
        os.makedirs(split_dir, exist_ok=True)

        # Graph
        write_graph(nodes, links, os.path.join(split_dir, f"{topo_name}.graph"), topo_name)

        # k_shortest_paths
        ksp = convert_kpaths(kpaths_raw, os.path.join(split_dir, "k_shortest_paths.json"))

        # Verify
        verify_kpaths_vs_topo(ksp, links)

        # Demands (same TMs in both TRAIN and EVALUATE for now — split via config)
        write_demands(tms, os.path.join(split_dir, "TM"), topo_name)

    # ALL → symlink to TRAIN
    all_dir = os.path.join(dst_base, "ALL")
    if os.path.exists(all_dir):
        if os.path.islink(all_dir):
            os.remove(all_dir)
        else:
            import shutil
            shutil.rmtree(all_dir)
    os.symlink("TRAIN", all_dir)
    print(f"[OK] ALL → TRAIN (symlink)")

    print(f"\n=== Done! Output at: {dst_base} ===")


if __name__ == "__main__":
    main()
