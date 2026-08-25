"""Build a K=30 EXTENDED candidate-path file WITHOUT touching the canonical
k_paths.json.

Why this exists (2026-07-15, K-sufficiency analysis): the canonical
dataset/<topo>_traffic/k_paths.json is the DRSIR authors' original artifact
(md5-identical to their repo; geant 494bc301..., 32node 8a018388...). It was
generated with a hop-count Yen recipe whose tie-break ordering depended on the
authors' live-controller adjacency order -- that ordering is lost, so the file
is NOT regenerable (best local replay: 107/992 exact on 32node). K>20 paths
consistent with the historical ordering therefore cannot exist.

This script sidesteps that: positions 0..19 are the canonical file VERBATIM
(prefix-preserving -> every K<=20 analysis point stays bit-identical to all
published experiments), positions 20..K_EXT-1 are fresh hop-count Yen paths
(nx.shortest_simple_paths, weight=None) on the same topology, deduped against
the frozen prefix, appended in generation order. Deterministic given the
networkx version (edges inserted in sorted order; nx 2.5 and 3.1 verified to
agree on this recipe).

The extension is ANALYSIS-ONLY (oracle K-curve headroom probe). There is no
historical K=30 artifact to be faithful to. Do NOT point training/real-test
runs at the extended file: action indices >= 20 have no meaning to any
existing checkpoint or baseline.

Run:
  cd <repo root>
  python dataset/extend_k_paths.py \
      --topo 32node --k-ext 30
Output: dataset/<topo>_traffic/k_paths_k<K_EXT>_ext.json (canonical untouched)
"""
import os
import sys
import json
import argparse

import networkx as nx

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOPO_FILES = {
    "geant":  ("dataset/geant_traffic/bw_r.txt",  "dataset/geant_traffic/k_paths.json"),
    "32node": ("dataset/32node_traffic/bw_r.txt", "dataset/32node_traffic/k_paths.json"),
}

# Safety cap on candidates enumerated per pair while hunting for new paths.
# Yen's gets slow on dense tie groups; if a pair can't fill K_EXT within this
# budget it is reported (not silently padded).
CANDIDATE_CAP = 2000


def build_graph(bw_file):
    """Same canonical construction as dataset/get_k_paths.py: normalized,
    deduped, sorted edge insertion -> adjacency order independent of file
    order. Weight attr unused (extension recipe is hop count, weight=None)."""
    edges = set()
    with open(bw_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            n1, n2 = int(parts[0]), int(parts[1])
            edges.add(tuple(sorted((n1, n2))))
    G = nx.Graph()
    for n1, n2 in sorted(edges):
        G.add_edge(n1, n2)
    return G


def extend_pair(G, src, dst, frozen, k_ext):
    """frozen: list of canonical paths (verbatim). Returns frozen + fresh
    hop-count Yen paths not already present, in generation order."""
    have = {tuple(p) for p in frozen}
    out = list(frozen)
    gen = nx.shortest_simple_paths(G, src, dst, weight=None)
    n_seen = 0
    for cand in gen:
        n_seen += 1
        if len(out) >= k_ext or n_seen > CANDIDATE_CAP:
            break
        t = tuple(cand)
        if t not in have:
            have.add(t)
            out.append(list(cand))
    return out


def validate(G, canon, ext, k_ext):
    """Hard checks; raises on any violation. Returns summary stats."""
    n_short = 0
    ext_hops, frozen_maxhops = [], []
    for s in canon:
        for d in canon[s]:
            c, e = canon[s][d], ext[s][d]
            assert e[:len(c)] == c, f"prefix broken at {s}->{d}"
            seen = set()
            for p in e:
                t = tuple(p)
                assert t not in seen, f"dup path {s}->{d}: {p}"
                seen.add(t)
                assert p[0] == int(s) and p[-1] == int(d), f"endpoint mismatch {s}->{d}: {p}"
                assert len(set(p)) == len(p), f"non-simple path {s}->{d}: {p}"
                for u, v in zip(p, p[1:]):
                    assert G.has_edge(u, v), f"nonexistent edge ({u},{v}) in {s}->{d}: {p}"
            if len(e) < k_ext:
                n_short += 1
            frozen_maxhops.append(max(len(p) - 1 for p in c))
            ext_hops.extend(len(p) - 1 for p in e[len(c):])
    return n_short, frozen_maxhops, ext_hops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", choices=list(TOPO_FILES), default="32node")
    ap.add_argument("--k-ext", type=int, default=30)
    args = ap.parse_args()

    bw_rel, kp_rel = TOPO_FILES[args.topo]
    bw_file = os.path.join(_REPO, bw_rel)
    kp_file = os.path.join(_REPO, kp_rel)
    out_file = kp_file.replace("k_paths.json", f"k_paths_k{args.k_ext}_ext.json")
    assert out_file != kp_file
    if os.path.exists(out_file):
        print(f"[abort] {out_file} already exists -- delete it explicitly to regenerate.")
        sys.exit(1)

    print(f"networkx {nx.__version__}")
    G = build_graph(bw_file)
    canon = json.load(open(kp_file))

    ext = {}
    for s in canon:
        ext[s] = {}
        for d in canon[s]:
            ext[s][d] = extend_pair(G, int(s), int(d), canon[s][d], args.k_ext)

    n_short, frozen_maxhops, ext_hops = validate(G, canon, ext, args.k_ext)
    n_pairs = sum(len(v) for v in canon.values())
    print(f"pairs: {n_pairs}, target K={args.k_ext}, pairs short of target: {n_short}")
    if ext_hops:
        import statistics
        print(f"extension paths: {len(ext_hops)}, hops min/med/max = "
              f"{min(ext_hops)}/{statistics.median(ext_hops)}/{max(ext_hops)} "
              f"(frozen prefix max-hop median {statistics.median(frozen_maxhops)})")

    with open(out_file, "w") as f:
        json.dump(ext, f, indent=2)
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
