config = {
    "algs_name": "ilp",
    "action_dim": 20,
    "net_info_file": "./results/ilp/net_info.csv",
    "tm_scale": 3,
    # Precomputed single-path ILP optimal actions (dataset/compute_ilp_actions.py).
    # Path relative to the repository root, which is run_drl.py's working dir.
    "ilp_actions_file": "dataset/geant_traffic/ilp_actions_s3.json",
}
