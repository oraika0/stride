"""32-node directed env pointing at the K=30 extended candidate-path file.

Used by the K-sufficiency ablation, which asks whether 20 candidate paths per OD
pair are enough for the policy to express a good routing configuration.

WHY THIS EXISTS AS A SEPARATE ENV

`dataset/32node_traffic/k_paths.json` holds exactly 20 paths per pair. K=10 and
K=15 are prefixes of it and need nothing new. K=25 and K=30 ask for more paths
than the file contains, so they read `k_paths_k30_ext.json` instead: positions
0-19 are the canonical file VERBATIM, 20-29 are hop-count Yen extensions
appended by `dataset/extend_k_paths.py`.

That prefix property is what makes the ablation a controlled comparison -- every
K is the same frozen candidate set truncated at a different point. Regenerating
`k_paths.json` with a larger K would reorder ties and redefine what K=20 means,
invalidating every other run.

The file is selected here, in the env layer, rather than by the K variants,
because `main.py` calls `init_paths(env_cfg, alg_cfg)` before merging the config
layers, and `utils/init_path.py` reads `k_paths_file` out of the unmerged env
dict. A `k_paths_file` set from a STRIDE variant would win in the merged config
and still be ignored when the switches are given their flow rules -- the model
would decide over 30 candidates while the network was built for 20, with no
error raised.

USAGE

Pair with a variant that sets `action_dim` to 25 or 30. With the default 20 the
env truncates back to the canonical prefix and the run silently degenerates to a
plain K=20 run. K=30 checkpoints are shape-incompatible with K=20 ones (the
action head differs); never swap checkpoints across the two.

    STRIDE_VARIANT=k30 python main.py --env 32node_144tm_directed_k30 \
        --alg stride train
"""
import importlib

config = {**importlib.import_module("config.env.32node_144tm_directed_config").config,
          "k_paths_file": "dataset/32node_traffic/k_paths_k30_ext.json"}
