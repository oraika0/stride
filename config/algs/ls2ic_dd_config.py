# ls2ic_dd_config.py — ls2ic with FULL directed (state + reward both directed)
#
# Both state and reward read net_info_directed.csv (tc_delay_ms + per-direction
# bw/loss).  Most aligned with STRIDE's measurement basis — fair head-to-head
# comparison with STRIDE eval.
#
# num_link is the DIRECTED count (Geant 74 = 37×2, 32node 120 = 60×2); NN input
# dim scales with it. 2026-06-04: NO LONGER hard-coded here -- the env owns it
# (geant_directed=74 / 32node_144tm_directed=120), same as the stride refactor.
# main.py merges {**env, **alg}, so a num_link in this alg dict would OVERRIDE the
# env's and silently mis-shape on 32node. => ALWAYS run ls2ic_dd on a *_directed
# env (plain undirected `geant` would not supply the directed num_link).
from config.algs.ls2ic_config import config as _base

config = {
    **_base,
    "algs_name":       "ls2ic_dd",
    "reward_directed": True,
    "state_directed":  True,
    "_experiment":     "ls2ic_dd",
}
