REGISTRY = {}

from .ls2ic import ls2ic_agent
from .meanfield import meanfield_agent
from .ps_dqn_a import ps_dqn_a_agent
from .ps_dqn import ps_dqn_agent
from .ospf import ospf_agent
from .adaptive_dijkstra import dijkstraAgent
from .widest_path import widest_path_agent
from .stride import StrideAgent
from .ilp import ilp_agent

REGISTRY["ls2ic"] = ls2ic_agent
REGISTRY["ls2ic_nx"] = ls2ic_agent
REGISTRY["ls2ic_dd"] = ls2ic_agent  # state+reward both directed (2026-05-24)
REGISTRY["meanfield"] = meanfield_agent
REGISTRY["ps_dqn_a"] = ps_dqn_a_agent
REGISTRY["ps_dqn"] = ps_dqn_agent
REGISTRY["ps_dqn_dd"] = ps_dqn_agent  # ps_dqn + directed state/reward (tc-delay), like ls2ic_dd
REGISTRY["ospf"] = ospf_agent
REGISTRY["adaptive_dijkstra"] = dijkstraAgent
REGISTRY["widest_path"] = widest_path_agent
REGISTRY["stride"] = StrideAgent  # 2026-05-25: post-H3 clean rewrite
REGISTRY["ilp"] = ilp_agent  # 2026-05-28: static ILP single-path optimal baseline