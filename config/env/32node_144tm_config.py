config = {
    "topology": "32node",
    "tm_name": "32nodes_144tm",
    "tm_prefix": "dataset/32node_traffic/32node_144tm/TM-{tm_id}/",
    "num_link": 60,
    "num_link_fea": 3,
    "num_agents": 992,
    "num_node": 32,
    "num_tm": 144,
    # tm_duration_training × 10 = MONITOR_PERIOD = wall-clock seconds per TM
    # on Mininet side. DRL side does tm_duration_steps = tm_duration_training //
    # 10 logical-steps per TM (train_loader.py:199). Alignment requires both
    # sides to consume `tm_duration_training` wall-seconds per TM:
    #     DRL wall: (tm_duration_training // 10) × 10s_padded_step = tm_duration_training
    #     Mininet wall: tm_duration_training
    # 350 → 35 step/TM × 86 train TMs = 3010 step for one full cycle through all
    # 86 TMs (was 339 → 33 step/TM → 2838 step cycle + 162 step partial 2nd
    # cycle, uneven coverage). total_timestep below stops at 3000 rather than
    # 3010, so the 86th TM gets 25 of its 35 steps -- see the note there.
    "tm_duration_training": 350,
    "tm_duration_test": 30,
    "tm_list_train": ['00', '01', '02', '03', '04', '07', '09', '11', '12', '13', '14', '16', '17', '19', '22', '23', '25', '26', '28', '29', '30', '31', '33', '35', '36', '38', '40', '42', '43', '44', '48', '49', '50', '53', '54', '55', '57', '58', '59', '61', '64', '67', '68', '69', '72', '74', '75', '76', '77', '78', '79', '80', '81', '83', '84', '85', '86', '87', '90', '91', '93', '95', '97', '102', '103', '104', '106', '109', '111', '114', '115', '116', '117', '118', '123', '124', '125', '126', '128', '129', '130', '132', '133', '135', '139', '140'],
    "tm_list_test": ['06', '41', '73', '108', '141'],
    "k_paths_file": "dataset/32node_traffic/k_paths.json",
    "bw_file": "dataset/32node_traffic/bw_r.txt",
    "link_bw_default": 100,
    "delay_norm_div": 200,
    # 3000, not the 3010 a full TM cycle would take. STRIDE's alg config pins
    # 3000 and alg beats env in main.py's {**env_cfg, **alg_cfg, **ctrl_cfg},
    # so STRIDE has always trained for 3000 steps here while every baseline
    # inherited 3010 from this line and trained for ten more -- a difference
    # nobody chose and nothing recorded. Matching STRIDE is the side that keeps
    # the comparison honest, and it costs the 86th TM ten of its thirty-five
    # steps. GÉANT never had the split: that env is 3000 for everyone.
    #
    # The archived baseline runs were produced at 3010 and their config.json
    # still says so, so the published numbers stand and remain traceable. What
    # changes is that re-running a baseline now gives 3000 steps and will not
    # reproduce those archives to the step.
    "total_timestep": 3000,
}
