DISCOVERY_PERIOD = 5# #5For discovering topology.

MONITOR_PERIOD = 10# #10For monitoring traffic

# 2026-06-08: the monitor greenlet used to start computing link metrics
# IMMEDIATELY at controller boot, while LLDP topology discovery was still
# climbing (e.g. 115->120 links) and port-stats were not yet populated ->
# KeyError on partial dicts crashed the whole monitor greenlet -> all metrics
# froze (degenerate run). This is a single startup wait before the FIRST monitor
# cycle so it only runs once topology + stats are ready. Constraint: keep it
# ABOVE topology-discovery time (~30-40 s for 32node/120 links) and BELOW the
# caller's pre-DRL wait minus stats-fill time (main.py sleeps 120 s -> DRL ~boot
# +114 s, stats need ~20 s -> upper bound ~90 s). Bump BOTH this and main.py's
# 120 s together if a slower host needs more topology-discovery headroom.
MONITOR_START_DELAY = 20  # seconds; wait for LLDP topology + port-stats before metrics

DELAY_DETECTING_PERIOD = 8 #8

# How long a monitor cycle waits for port-stats replies before giving up on
# them. The requests are asynchronous, and until this existed the only thing
# giving the replies time to land was flow_install_monitor happening to take a
# few seconds. That holds on an idle machine and stops holding on a loaded one,
# where the cycle then finds no stats, skips the metrics write, and leaves the
# agent reading the previous cycle's file with nothing reported.
PORT_STATS_WAIT = 3.0   # seconds

# How often the awareness worker may rebuild the topology. The handler only
# marks the topology dirty (blocking in a handler wedges the controller, see
# docs/ryu_controller_deadlock.md), so this also coalesces the event burst
# that discovery produces into one rebuild.
TOPO_REBUILD_PERIOD = 0.5   # seconds

TOSHOW = True   # For showing information in terminal
SHOW_DIRECTED = True  # True: show per-direction (src->dst); False: show undirected (legacy)


'''
para 64 nodos intente correr con discover 10, monitor 15, delay 13  pero cuparece que el monitor no es suficiente para el drl, si toca aumenta rmuco pues paila, entones savoy a ahcer con 48nodos
'''
