from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.log import info, setLogLevel
import time
import os
import subprocess

def build_topo(env_config):
    net = Mininet(controller=RemoteController, link=TCLink)
    setLogLevel("info")
    info("*** Add Controller ***\n")
    net.addController("c0", controller=RemoteController, ip='127.0.0.1')

    info("*** Add Switches and host ***\n")
    for i in range(1, env_config["num_node"] + 1):
        net.addSwitch(f"s{i}")
        net.addHost(f"h{i}", mac=f"00:00:00:00:00:{i:02d}")
        net.addLink(f"s{i}", f"h{i}")

    info("*** Add Inter-Switch Links from BW file ***\n")
    added_links = set()
    switch_links = []  # [(intf1_name, intf2_name, bw)]
    queue_size = env_config.get("max_queue_size", 1000)
    with open(env_config["bw_file"], 'r') as f:
        for line in f:
            if not line.strip():
                continue
            src, dst, _, bw = line.strip().split(",")
            src, dst = int(src), int(dst)
            bw = float(bw)
            link_key = tuple(sorted((src, dst)))
            if link_key not in added_links:
                link = net.addLink(f"s{src}", f"s{dst}", bw=bw, max_queue_size=queue_size)
                switch_links.append((str(link.intf1), str(link.intf2), bw))
                added_links.add(link_key)

    info("*** Network Start ***\n")
    net.start()

    # Replace HTB+netem with netem-only for accurate loss measurement.
    # HTB absorbs excess traffic via backpressure, making netem drops invisible.
    # netem-only exposes all drops directly — closer to real ASIC output buffer.
    info("*** Replacing tc HTB+netem with netem-only ***\n")
    for intf1, intf2, bw in switch_links:
        for intf in (intf1, intf2):
            subprocess.run(['tc', 'qdisc', 'del', 'dev', intf, 'root'],
                           capture_output=True)
            subprocess.run(['tc', 'qdisc', 'add', 'dev', intf, 'root',
                           'netem', 'rate', f'{bw}mbit', 'limit', str(queue_size)],
                           check=True)

    return net

def _drl_done_path(alg_name):
    return f"./results/{alg_name}/.drl_done"

def start_traffic(net, env_config, mode="train", alg_name=None):
    """
    Start the traffic generation process, mode supports 'train' / 'test'.
    If alg_name is provided, loops through tm_list repeatedly until
    .drl_done sentinel file appears (written by DRL when training ends).
    If alg_name is None, runs exactly one round (backward compatible).
    """
    tm_ids = env_config["tm_list_train"] if mode == "train" else env_config["tm_list_test"]
    tm_duration = env_config["tm_duration_training"] if mode == "train" else env_config["tm_duration_test"]
    tm_prefix = env_config["tm_prefix"]  # e.g. 32nodos_24tm/TM-{tm_id}/
    num_hosts = env_config["num_node"]

    # Clean up stale sentinel from previous run
    if alg_name:
        sentinel = _drl_done_path(alg_name)
        if os.path.exists(sentinel):
            os.remove(sentinel)

    # 2026-05-27: kill iperf3 reliably. The old `killall -p iperf3` had a
    # bogus `-p` (not a valid killall signal — silently treated as nothing,
    # so killall sent the default SIGTERM which iperf3 sometimes ignored,
    # leaving zombie iperf3 processes long after training finished). Use
    # SIGKILL via `pkill -9 -f iperf3` so the kill is unconditional.
    def _kill_iperf3():
        os.system("sudo pkill -9 -f iperf3 2>/dev/null")

    # 2026-05-27: sentinel-aware sleep so iperf3 dies promptly when DRL
    # finishes mid-TM (rather than waiting up to tm_duration_training=2000s
    # for the next TM iter to check sentinel).
    def _sleep_until_sentinel(seconds, poll_interval=5):
        if alg_name is None:
            time.sleep(seconds)
            return False  # no sentinel possible
        sentinel_path = _drl_done_path(alg_name)
        deadline = time.time() + seconds
        while time.time() < deadline:
            if os.path.exists(sentinel_path):
                return True
            time.sleep(min(poll_interval, deadline - time.time()))
        return False

    round_num = 0
    while True:
        round_num += 1
        print(f"========== Traffic round {round_num} ==========", flush=True)

        for tm_id in tm_ids:
            # Check if DRL finished via sentinel file
            if alg_name and os.path.exists(_drl_done_path(alg_name)):
                print(f"DRL done sentinel found at TM boundary, stopping traffic.", flush=True)
                _kill_iperf3()
                return

            print(f"--- TM {tm_id} (round {round_num}) ---", flush=True)

            for i in range(1, num_hosts + 1):
                hname = f"h{i}"
                suffix = f"0{i}" if i < 10 else str(i)
                server_sh = os.path.join(tm_prefix.format(tm_id=tm_id), f"Servers/server_{suffix}.sh")
                net.get(hname).popen(f"sh {server_sh}")

            if _sleep_until_sentinel(10):
                print(f"DRL done sentinel found during server warmup, stopping traffic.", flush=True)
                _kill_iperf3()
                return

            for i in range(1, num_hosts + 1):
                hname = f"h{i}"
                suffix = f"0{i}" if i < 10 else str(i)
                client_sh = os.path.join(tm_prefix.format(tm_id=tm_id), f"Clients/client_{suffix}.sh")
                net.get(hname).popen(f"sh {client_sh}")

            if _sleep_until_sentinel(tm_duration):
                print(f"DRL done sentinel found mid-TM {tm_id}, stopping traffic early.", flush=True)
                _kill_iperf3()
                return

            _kill_iperf3()
            print(f"next TM (TM {tm_id} done)\n", flush=True)

        # No alg_name → single round only (backward compatible)
        if alg_name is None:
            return

def start_single_traffic(net, env_config, input_):
    
    tm_prefix = env_config["tm_prefix"]
    num_hosts = env_config["num_node"]
    
    print("################################################")
    for i in range(1, num_hosts + 1):
        hname = f"h{i}"
        suffix = f"0{i}" if i < 10 else str(i)
        server_sh = os.path.join(tm_prefix.format(tm_id=input_), f"Servers/server_{suffix}.sh")
        net.get(hname).popen(f"sh {server_sh}")
    time.sleep(10)
    for i in range(1, num_hosts + 1):
        hname = f"h{i}"
        suffix = f"0{i}" if i < 10 else str(i)
        client_sh = os.path.join(tm_prefix.format(tm_id=input_), f"Clients/client_{suffix}.sh")
        net.get(hname).popen(f"sh {client_sh}")

    time.sleep(360)
    os.system("sudo pkill -9 -f iperf3 2>/dev/null")   # 2026-05-27: -p was bogus, use SIGKILL
    time.sleep(2)