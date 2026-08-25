# ps_dqn_dd_config.py -- ps_dqn baseline with DIRECTED state + DIRECTED reward
# (tc-queue delay + per-direction packet loss), matching ls2ic_dd's measurement
# basis for a fair head-to-head baseline (like ls2ic -> ls2ic_dd).
#
# The two directed knobs are alg-agnostic in train_loader:
#   reward_directed=True -> path_metrics_to_reward_directed (tc_delay_ms preferred,
#                           per-direction pkloss; reads net_info_directed.csv)
#   state_directed=True  -> get_state_directed (directed 3-ch global_state, also tc)
# ps_dqn keeps use_global_state=True, so it consumes the DIRECTED global_state.
#
# MUST run on a *_directed env (32node_144tm_directed: num_link=120 directed) so
# the net input dim (num_link*num_link_fea) and net_info_directed.csv line up.
from config.algs.ps_dqn_config import config as _base

config = {
    **_base,
    "algs_name":       "ps_dqn_dd",
    "reward_directed": True,
    "state_directed":  True,
    "_experiment":     "ps_dqn_dd",
}
