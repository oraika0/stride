# drsir_dd_config.py — DRSIR on the STRIDE-aligned DIRECTED link-data source
# (mode C, DRSIR_REWARD_CORRECTED=2). Naming mirrors the ls2ic -> ls2ic_dd
# precedent: algs_name changes, so results land under results/drsir_dd and the
# published-undirected A/B sessions under results/drsir stay distinct.
#
# Reward link-data source (controller side, simple_monitor mode C branch):
#   bwd   per-direction free bw (C - tx, kbps)   [net_info_directed col 0]
#   delay tc queue delay ms (backlog*8/C)        [net_info_directed col 3]
#   loss  tc drop % per direction                [net_info_directed col 4]
# i.e. the same three columns STRIDE's reward/state read. DRSIR's own DQN
# formulation (pair-index state, per-pair argmin cost, k=20 paths) is unchanged;
# only the measurements feeding paths_metrics.json differ, and (u,v) / (v,u)
# now get their own per-direction path metrics.
#
# "drsir_reward_mode" below is exported to DRSIR_REWARD_CORRECTED by
# test_single_tm.py / main.py before the controller spawns — config wins over
# any shell-set value, so a forgotten export can't silently fall back to mode A
# (cf. the sudo -E EXP silent-fallback pitfall).
#
# Unlike drsir_seed18 (config KEY differs, algs_name stays "drsir"), algs_name
# DOES change here — run_drl.py dispatches on startswith("drsir") and the
# ./results/drsir/ paths in DRL_paths_threading / environment_test_* are
# parameterized by algs_name (2026-07-10).
#
# Run:
#   run_real_test drsir_dd "drsir_dd_geant_s17"  NONE NONE default geant
#   run_real_test drsir_dd "drsir_dd_32node_s17" NONE NONE default 32node_144tm
from config.algs.drsir_config import config as _drsir

config = {
    **_drsir,
    "algs_name": "drsir_dd",
    "drsir_reward_mode": 2,
}
