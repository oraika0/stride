config = {
    "algs_name": "ps_dqn",
    "action_dim": 20,
    "lr": 0.001,
    "gamma": 0.9,   # SAME as ls2ic / stride (kept consistent for a fair comparison).
    "tau": 0.005,
    "batch_size": 32,
    "buffer_size": 2000,
    "hidden_dim": 32,
    "hidden_dim2": 32,
    "epsilon_ini": 0.5,
    "epsilon_final": 0.01,
    "softupdate_freq": 30,
    "epsilon": 0.5,
    "use_global_state": True,
    "epsilon_first_phase": 1000,
    "epsilon_second_phase": 1000,
    # 2026-06-18 DIVERGENCE FIX (was True -> dqn_loss ~2M, MLU 0.66->1.0 collapse).
    # The delta branch in loop_pairs SKIPS the /100 normalization that ls2ic/stride
    # get, so ps_dqn was fed raw-scale (~100s) rewards -> Q exploded. False -> r/100
    # (the SAME reward ls2ic uses) -> fair + stable. (gamma stays 0.9 like everyone.)
    "use_delta_reward": False,
    "tm_scale": 3,
}
