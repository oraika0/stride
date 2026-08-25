import numpy as np


class widest_path_agent:
    """Per-pair greedy widest-path heuristic (informed baseline).

    For each OD pair, picks the K-candidate path with the maximum bottleneck
    residual bandwidth (= min over links: rem_ratio = (cap - util) / cap),
    using the link observation visible at the current step. No coordination
    across pairs; no learning. Parameter-free.
    """

    def __init__(self, args):
        self.args = args
        self.action_size = int(args.action_dim)

    def get_action(self, state, epsilon=0.0, **info):
        env = info.get("env", None)
        if env is None:
            raise RuntimeError("[widest_path.get_action] info['env'] is required.")

        N = int(env.numPairs)
        K = int(env.K)
        if K != self.action_size:
            raise RuntimeError(
                f"[widest_path.get_action] env.K={K} != action_dim={self.action_size}"
            )

        link_feat = env.get_link_features()      # (E, 4): cap_norm, util_ratio, rem_ratio, betweenness
        rem_ratio = link_feat[:, 2].astype(np.float64)   # (E,) higher == more residual bw

        if not np.isfinite(rem_ratio).all() or rem_ratio.sum() <= 0.0:
            return np.zeros(N, dtype=np.int64), {}

        path_link_id = env.path_link_id                  # (T,) link eid per (path, hop) incidence
        path_id      = env.path_id                       # (T,) path pid per incidence
        P            = N * K

        bottleneck = np.full((P,), np.inf, dtype=np.float64)
        np.minimum.at(bottleneck, path_id, rem_ratio[path_link_id])

        bottleneck_NK = bottleneck.reshape(N, K)
        action = bottleneck_NK.argmax(axis=1).astype(np.int64)
        return action, {}

    def append_sample(self, *a, **k):
        pass

    def update(self):
        return {}

    def save_model(self, *a, **k):
        pass

    def load_model(self, *a, **k):
        pass

    def update_target(self):
        pass
