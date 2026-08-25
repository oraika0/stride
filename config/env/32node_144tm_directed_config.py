"""32node 144tm (scale=1) env for the STRIDE directed pipeline.

Identical to 32node_144tm_config but num_link declares the DIRECTED count
(120 = 60 undirected x 2) for the state_directed / reward_directed pipeline.
The base 32node_144tm_config keeps num_link=60 (undirected); build_topo reads
bw_r.txt (120 directed rows) and dedups to 60 Mininet links regardless.

2026-06-04: created alongside geant_directed when num_link moved out of the
stride alg config (env now owns the link count). num_node=32 -> 992 pairs.
"""
import importlib

config = {**importlib.import_module("config.env.32node_144tm_config").config,
          "num_link": 120}
