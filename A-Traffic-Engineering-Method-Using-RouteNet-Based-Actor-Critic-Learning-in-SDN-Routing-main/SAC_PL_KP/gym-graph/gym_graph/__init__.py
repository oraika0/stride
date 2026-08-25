from gym.envs.registration import register


register(
    id='GraphEnv-v16',
    entry_point='gym_graph.envs:Env16',
)
