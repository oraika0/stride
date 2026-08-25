_tm_scale_real = 3   # real Mininet traffic scale: 3 → 23node_s3/; 5 → 23node/
_tm_folder = "23node" if _tm_scale_real == 5 else f"23node_s{_tm_scale_real}"

config = {
    "topology": "geant",
    "tm_name": "23nodes",
    "tm_scale_real": _tm_scale_real,
    "tm_prefix": f"dataset/geant_traffic/{_tm_folder}/TM-{{tm_id}}/",
    "num_link": 37,
    "num_link_fea": 3,
    "num_agents": 506,
    "num_node": 23,
    "num_tm": 24,
    "tm_duration_training": 2000,
    "tm_duration_test": 30,
    "tm_list_train": ['13','15','17','19','20','22','23','00','01','02','04','07','08','09','11'],
    "tm_list_test": ['03', '10', '12', '14', '21'],
    "k_paths_file": "dataset/geant_traffic/k_paths.json",
    "bw_file": "dataset/geant_traffic/bw_r.txt",
    "link_bw_default": 100,
    "delay_norm_div": 2500,
    "total_timestep": 3000,
}
