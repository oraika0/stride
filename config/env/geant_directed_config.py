"""Geant env for the STRIDE directed pipeline.

Identical to geant_config but num_link declares the DIRECTED count
(74 = 37 undirected x 2) that the state_directed / reward_directed pipeline
needs. The base geant_config keeps num_link=37 (undirected) for undirected-
pipeline consumers; build_topo reads bw_r.txt and dedups to 37 Mininet links
regardless of this value.

2026-06-04: created when num_link moved OUT of the stride alg config (the merge
is {**env, **alg, **ctrl} so the alg used to override the env's num_link; now
the env owns it as the single source of truth). DRY via importlib so the geant
params stay single-source -- only num_link diverges.
"""
import importlib

config = {**importlib.import_module("config.env.geant_config").config,
          "num_link": 74}
