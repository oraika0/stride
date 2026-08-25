"""Build the K candidate paths per OD pair from a topology's bw_r.txt.

    python dataset/get_k_paths.py --topology 32node

Writes <topology>_traffic/k_paths.json. Only use this for a NEW topology: the
shipped k_paths.json files are frozen and cannot be regenerated -- see this
directory's README.
"""
import argparse
import json
import os
from itertools import islice

import networkx as nx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def k_shortest_paths(graph, src, dst, k, weight='weight'):
    try:
        return list(islice(nx.shortest_simple_paths(graph, src, dst, weight=weight), k))
    except nx.NetworkXNoPath:
        return [] 

def all_k_shortest_paths(graph, k=20):
    """
    For all pairs of nodes in the graph where src ≠ dst, calculate the first k shortest paths
    The return format is dict[src][dst] = [path1, path2, ..., pathk]
    """
    paths = {}
    for src in graph.nodes():
        paths[str(src)] = {}
        for dst in graph.nodes():
            if src != dst:
                path_list = k_shortest_paths(graph, src, dst, k)
                if path_list:
                    paths[str(src)][str(dst)] = path_list
    return paths

def build_graph_from_bw_r(file_path):
    G = nx.Graph()

    # Step 1: Read and sort all wires
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            node1, node2 = int(parts[0]), int(parts[1])
            bwd_cap = float(parts[3])
            edge = tuple(sorted((node1, node2)))
            edges.append((edge[0], edge[1], bwd_cap))

    # Step 2: Sort by (node1, node2)
    seen = set()
    unique_edges = []
    for n1, n2, bwd in sorted(edges):
        if (n1, n2) not in seen:
            seen.add((n1, n2))
            unique_edges.append((n1, n2, bwd))
    # Step 3: Add to the graph
    for node1, node2, bwd_cap in unique_edges:
        if bwd_cap > 0:
            G.add_edge(node1, node2, weight=1.0 / bwd_cap)
        else:
            print(f"[Warning] Link ({node1}, {node2}) has zero bwd_cap, skipped.")

    return G

def generate_k_paths(bw_r_path, output_path, k=20):
    G = build_graph_from_bw_r(bw_r_path)
    k_paths = all_k_shortest_paths(G, k=k)
    
    sorted_k_paths = {
        str(src): {
            str(dst): k_paths[str(src)][str(dst)]
            for dst in sorted(map(int, k_paths[str(src)].keys()))
        }
        for src in sorted(map(int, k_paths.keys()))
    }
    
    with open(output_path, 'w') as f:
        json.dump(sorted_k_paths, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True,
                        help="folder prefix, e.g. 32node for 32node_traffic/")
    parser.add_argument("--k", type=int, default=20, help="paths per OD pair")
    args = parser.parse_args()

    folder = os.path.join(SCRIPT_DIR, f"{args.topology}_traffic")
    bw_r_file = os.path.join(folder, "bw_r.txt")
    output_file = os.path.join(folder, "k_paths.json")
    if os.path.exists(output_file):
        raise SystemExit(
            f"{output_file} already exists. The shipped candidate sets are frozen "
            f"and every reported result indexes into their ordering -- delete the "
            f"file by hand if you really mean to replace it.")
    generate_k_paths(bw_r_file, output_file, k=args.k)
    print(f"wrote {output_file}")
