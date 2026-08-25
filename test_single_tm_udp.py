#!/usr/bin/env python3
"""
test_single_tm with per-step UDP probing + tc backlog measurement.

Same lifecycle as test_single_tm.py (builds MN topo, spawns Ryu + DRL,
runs iperf traffic), but the main process also samples every MONITOR_PERIOD:

    tc_delay_ms   = backlog_bytes * 8 / C   (ground truth, from tc qdisc)
    probe_fwd_ms  = UDP one-way forward     (independent witness)
    lldp_fwd_ms   = raw LLDP forward        (snapshot from controller)
    delay_lldp_ms = echo-corrected LLDP     (snapshot from controller)

Output per TM:
    {session_dir}/real/{tm_id}/probe_vs_tc.csv

Usage:
    sudo python3 test_single_tm_udp.py --env geant --alg ospf --auto
    sudo python3 test_single_tm_udp.py --env geant --alg ospf --tm 03
"""

import argparse
import config as config_module
import csv
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

from loader import env_loader
from utils.init_path import init_paths
from utils.process_launcher import (TERMINAL_CHOICES, launch, resolve_terminal,
                                    write_child_config)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UDP_PROBE = os.path.join(THIS_DIR, "diagnostics", "udp_probe_echo.py")

MONITOR_PERIOD = 10          # same cadence as DRL controller
PROBE_PORT_BASE = 5400
NUM_NODES = 23
QUEUE_LIMIT = 1000

STATS_RE = re.compile(r"Sent\s+\d+\s+bytes\s+(\d+)\s+pkt\s+\(dropped\s+(\d+),")
BACKLOG_RE = re.compile(r"backlog\s+(\d+)b\s+(\d+)p\s")

# ---------------------------------------------------------------------------
# Helpers (from test_all_links_probe_vs_tc.py, kept inline)
# ---------------------------------------------------------------------------

def spawn_controller(ctrl_cfg, terminal="auto"):
    cmd = (
        f"source {ctrl_cfg['conda_sh']} && "
        f"conda activate {ctrl_cfg['conda_env']} && "
        f"cd {ctrl_cfg['controller_dir']} && "
        f"ryu-manager --observe-link {ctrl_cfg['controller_entry']}"
    )
    return launch(cmd, "controller", terminal)


def spawn_drl(merged_cfg, mode, terminal="auto"):
    cfg_path = write_child_config(merged_cfg)
    cmd = (
        f"source {merged_cfg['conda_sh']} && "
        f"conda activate {merged_cfg['conda_env']} && "
        f"python {os.path.join(merged_cfg.get('project_root', os.getcwd()), 'run_drl.py')} "
        f"--merged_cfg {cfg_path} --mode {mode}"
    )
    return launch(cmd, "drl", terminal)


def init_result_dirs(alg_cfg):
    algs_name = alg_cfg["algs_name"]
    base_dir = os.path.join("results", algs_name)
    model_dir = os.path.join(base_dir, "model")
    metrics_dir = os.path.join(base_dir, "Metrics")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid:
        try:
            uid = int(sudo_uid)
            gid = pwd.getpwuid(uid).pw_gid
            os.chown(base_dir, uid, gid)
            os.chown(model_dir, uid, gid)
            os.chown(metrics_dir, uid, gid)
        except Exception:
            pass


def parse_bw_file(bw_file):
    """Parse bw_r.txt → directed_links [(src,dst,bw)], bw_dict {(s,d):bw}.

    bw_r.txt already has both directions (74 lines for 37 undirected links),
    so we do NOT duplicate them here.
    """
    directed_links = []
    bw_dict = {}
    seen = set()
    with open(bw_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            s, d, _, bw = int(parts[0]), int(parts[1]), parts[2], float(parts[3])
            if (s, d) not in seen:
                directed_links.append((s, d, bw))
                bw_dict[(s, d)] = bw
                seen.add((s, d))
    return directed_links, bw_dict


def get_port_map(net):
    """Build {(src,dst): dev_name} for switch-to-switch interfaces."""
    dev_map = {}
    for link in net.links:
        intf1, intf2 = link.intf1, link.intf2
        n1, n2 = intf1.node.name, intf2.node.name
        if n1.startswith("s") and n2.startswith("s"):
            dpid1, dpid2 = int(n1[1:]), int(n2[1:])
            dev_map[(dpid1, dpid2)] = str(intf1)
            dev_map[(dpid2, dpid1)] = str(intf2)
    return dev_map


def _tc_stats_dev_once(dev):
    """Single tc read."""
    try:
        raw = subprocess.check_output(
            ["tc", "-s", "qdisc", "show", "dev", dev], text=True, timeout=3
        )
    except Exception:
        return 0, 0, 0, 0
    sent = dropped = backlog_bytes = backlog_pkts = 0
    for line in raw.splitlines():
        m = STATS_RE.search(line)
        if m:
            sent, dropped = int(m.group(1)), int(m.group(2))
        m = BACKLOG_RE.search(line)
        if m:
            backlog_bytes, backlog_pkts = int(m.group(1)), int(m.group(2))
    return sent, dropped, backlog_bytes, backlog_pkts


def tc_stats_dev(dev, retries=3):
    """Read tc netem stats, retry on zero backlog to avoid kernel race.

    netem's internal counter can read 0 during a dequeue batch (~1.3% chance).
    Retrying 2x with subprocess overhead as natural spacing eliminates this.
    """
    for _ in range(1 + retries):
        sent, dropped, backlog_bytes, backlog_pkts = _tc_stats_dev_once(dev)
        if backlog_bytes > 0:
            return sent, dropped, backlog_bytes, backlog_pkts
    return sent, dropped, backlog_bytes, backlog_pkts


def snapshot_lldp(alg_name):
    """Read net_info_directed.csv → {(src,dst): (lldp_fwd_ms, delay_lldp)}."""
    path = os.path.join("results", alg_name, "net_info_directed.csv")
    out = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    k = (int(r["node1"]), int(r["node2"]))
                    lf = float(r.get("lldp_fwd_ms", -1.0))
                    dl = float(r.get("delay_lldp", -1.0))
                    out[k] = (lf, dl)
                except (KeyError, ValueError):
                    continue
    except Exception:
        pass
    return out


def snapshot_undirected(alg_name):
    """Read net_info.csv → {(src,dst): delay} (undirected, what eval_metrics uses)."""
    path = os.path.join("results", alg_name, "net_info.csv")
    out = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    k = (int(r["node1"]), int(r["node2"]))
                    d = float(r.get("delay", -1.0))
                    out[k] = d
                    out[(k[1], k[0])] = d  # same for both directions
                except (KeyError, ValueError):
                    continue
    except Exception:
        pass
    return out

# ---------------------------------------------------------------------------
# Probe servers
# ---------------------------------------------------------------------------

def start_probe_servers(net):
    """Start UDP echo servers on all hosts. Returns dict of Popen handles."""
    servers = {}
    for i in range(1, NUM_NODES + 1):
        h = net.get(f"h{i}")
        port = PROBE_PORT_BASE + i
        srv = h.popen(
            ["python3", UDP_PROBE, "server", "--host", "0.0.0.0",
             "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        servers[i] = srv
    return servers


def stop_probe_servers(servers):
    for srv in servers.values():
        try:
            srv.kill()
            srv.wait(timeout=2)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Traffic (non-blocking start / explicit stop)
# ---------------------------------------------------------------------------

def start_iperf_traffic(net, tm_prefix, tm_id, num_hosts):
    """Start iperf servers then clients (non-blocking). Returns popen list."""
    tm_dir = tm_prefix.format(tm_id=tm_id)
    procs = []
    for i in range(1, num_hosts + 1):
        h = net.get(f"h{i}")
        suffix = f"0{i}" if i < 10 else str(i)
        srv_sh = os.path.join(tm_dir, "Servers", f"server_{suffix}.sh")
        if os.path.exists(srv_sh):
            procs.append(h.popen(f"sh {srv_sh}", shell=True))
    print("[traffic] servers started, waiting 10s ...")
    time.sleep(10)
    for i in range(1, num_hosts + 1):
        h = net.get(f"h{i}")
        suffix = f"0{i}" if i < 10 else str(i)
        cli_sh = os.path.join(tm_dir, "Clients", f"client_{suffix}.sh")
        if os.path.exists(cli_sh):
            procs.append(h.popen(f"sh {cli_sh}", shell=True))
    print("[traffic] clients started")
    return procs


def stop_iperf_traffic():
    subprocess.run(["pkill", "-9", "-f", "iperf3"], capture_output=True)
    time.sleep(2)

# ---------------------------------------------------------------------------
# Sampling round
# ---------------------------------------------------------------------------

def sample_all_links(net, directed_links, dev_map, bw_dict,
                     seq_counter, probe_timeout, lldp_snap, undirected_snap):
    """One round: tc stats → parallel UDP probes → collect results."""
    # Phase 1: tc stats (fast, sequential)
    tc_data = {}
    for src, dst, bw in directed_links:
        dev = dev_map.get((src, dst))
        if dev is None:
            continue
        _, _, backlog_bytes, backlog_pkts = tc_stats_dev(dev)
        tc_delay = backlog_bytes * 8.0 / (bw * 1e6) * 1000.0 if bw > 0 else 0.0
        tc_data[(src, dst)] = (backlog_bytes, backlog_pkts, tc_delay)

    # Phase 2: launch probes in parallel
    probe_procs = {}
    for src, dst, bw in directed_links:
        if (src, dst) not in tc_data:
            continue
        h_src = net.get(f"h{src}")
        dst_ip = f"10.0.0.{dst}"
        port = PROBE_PORT_BASE + dst
        seq_counter[0] += 1
        proc = h_src.popen(
            ["python3", UDP_PROBE, "client",
             "--host", dst_ip, "--port", str(port),
             "--size", "64", "--timeout", str(probe_timeout),
             "--seq", str(seq_counter[0])],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        probe_procs[(src, dst)] = proc

    # Phase 3: collect
    deadline = time.time() + probe_timeout + 3.0
    rows = []
    for src, dst, bw in directed_links:
        if (src, dst) not in tc_data:
            continue
        backlog_bytes, backlog_pkts, tc_delay_ms = tc_data[(src, dst)]

        proc = probe_procs.get((src, dst))
        probe = {"ok": False}
        if proc:
            remaining = max(0.5, deadline - time.time())
            try:
                stdout, _ = proc.communicate(timeout=remaining)
                probe = json.loads(stdout.decode().strip())
            except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
                proc.kill()
                try:
                    proc.communicate(timeout=1)
                except Exception:
                    pass

        probe_fwd = probe.get("fwd_ms", -1.0) if probe.get("ok") else -1.0
        probe_rtt = probe.get("rtt_ms", -1.0) if probe.get("ok") else -1.0
        lldp_fwd, delay_lldp = lldp_snap.get((src, dst), (-1.0, -1.0))
        delay_und = undirected_snap.get((src, dst), -1.0)

        rows.append({
            "src": src,
            "dst": dst,
            "bw_mbps": bw,
            "backlog_bytes": backlog_bytes,
            "backlog_pkts": backlog_pkts,
            "tc_delay_ms": round(tc_delay_ms, 3),
            "probe_fwd_ms": round(probe_fwd, 3),
            "probe_rtt_ms": round(probe_rtt, 3),
            "probe_ok": probe.get("ok", False),
            "lldp_fwd_ms": round(lldp_fwd, 3),
            "delay_lldp_ms": round(delay_lldp, 3),
            "delay_undirected_ms": round(delay_und, 3),
        })
    return rows

# ---------------------------------------------------------------------------
# Sampling loop for one TM
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "tm_id", "step", "wall_time",
    "src", "dst", "bw_mbps",
    "backlog_bytes", "backlog_pkts", "tc_delay_ms",
    "probe_fwd_ms", "probe_rtt_ms", "probe_ok",
    "lldp_fwd_ms", "delay_lldp_ms", "delay_undirected_ms",
]


def run_sampling_loop(net, directed_links, dev_map, bw_dict,
                      tm_id, test_step, probe_timeout, alg_name,
                      csv_path, warmup_s):
    """Sample tc + probe for test_step iterations at MONITOR_PERIOD cadence."""
    seq_counter = [0]

    need_header = not os.path.exists(csv_path)
    csvfile = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
    if need_header:
        writer.writeheader()

    if warmup_s > 0:
        print(f"[probe] warmup {warmup_s}s ...")
        time.sleep(warmup_s)

    t0 = time.time()
    print(f"[probe] sampling {test_step} steps at {MONITOR_PERIOD}s interval")
    for step in range(test_step):
        step_start = time.time()

        # Snapshot LLDP + undirected from controller files
        lldp_snap = snapshot_lldp(alg_name)
        undirected_snap = snapshot_undirected(alg_name)

        rows = sample_all_links(
            net, directed_links, dev_map, bw_dict,
            seq_counter, probe_timeout, lldp_snap, undirected_snap,
        )

        wall = time.time()
        for r in rows:
            r["tm_id"] = tm_id
            r["step"] = step
            r["wall_time"] = f"{wall:.3f}"
            writer.writerow(r)
        csvfile.flush()

        # Print summary
        n_ok = sum(1 for r in rows if r["probe_ok"])
        max_tc = max((r["tc_delay_ms"] for r in rows), default=0)
        elapsed = time.time() - t0
        print(f"  step {step:3d}  probe_ok={n_ok}/{len(rows)}  "
              f"max_tc={max_tc:.0f} ms  elapsed={elapsed:.0f}s")

        # Pace to MONITOR_PERIOD
        used = time.time() - step_start
        if used < MONITOR_PERIOD:
            time.sleep(MONITOR_PERIOD - used)

    csvfile.close()
    print(f"[probe] wrote {csv_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="test_single_tm + per-step UDP probing & tc measurement")
    parser.add_argument("--env", required=True, help="e.g. geant")
    parser.add_argument("--alg", required=True, help="e.g. ospf / ls2ic")
    parser.add_argument("--ctrl", default="simple_monitor")
    parser.add_argument("--terminal", choices=TERMINAL_CHOICES, default="auto",
                        help="where the controller and agent run: tmux windows "
                             "in a detached 'stride' session, a gnome window each, "
                             "or inline here. 'auto' prefers tmux.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed. Only matters for algorithms that train during "
                             "the test (DRSIR); everything else loads a checkpoint.")
    parser.add_argument("--auto", action="store_true",
                        help="Run test TMs [3,10,12,14,21] sequentially")
    parser.add_argument("--tms", nargs="+", type=int, default=None,
                        help="Custom TM list (e.g. --tms 0 1 2 3 10 12 14 21). Overrides --auto.")
    parser.add_argument("--tm", default="",
                        help="Single TM id to test (e.g. 03). Overrides --auto and --tms.")
    parser.add_argument("--probe-timeout", type=float, default=8.0,
                        help="UDP probe socket timeout (s)")
    parser.add_argument("--warmup", type=float, default=0.0,
                        help="Seconds to wait after iperf clients start before sampling")
    parser.add_argument("--test-step", type=int, default=None,
                        help="Override number of DRL steps (default: from config or 30)")

    args = parser.parse_args()
    terminal = resolve_terminal(args.terminal)
    print(f"[test] terminal backend: {terminal}")

    env_cfg, alg_cfg, ctrl_cfg = config_module.get(args.env, args.alg, args.ctrl)
    init_result_dirs(alg_cfg)
    init_paths(env_cfg, alg_cfg)

    alg_name = alg_cfg["algs_name"]
    os.environ["ALG_NAME"] = alg_name
    os.environ["ENV_NAME"] = env_cfg["topology"]
    os.environ["NUM_LINK"] = str(env_cfg.get("num_link", 0))  # controller LLDP-completeness audit

    test_step = args.test_step or alg_cfg.get("test_step", 30)
    bw_file = env_cfg["bw_file"]
    tm_prefix = env_cfg["tm_prefix"]
    num_hosts = env_cfg["num_node"]

    # --- 1. Parse topology ---
    directed_links, bw_dict = parse_bw_file(bw_file)
    print(f"Topology: {len(directed_links)} directed links")

    # --- 2. Build Mininet ---
    print("Building topology ...")
    net = env_loader.build_topo(env_cfg)

    # --- 3. Port → device mapping ---
    dev_map = get_port_map(net)
    print(f"dev_map: {len(dev_map)} entries")

    # --- 4. Spawn Ryu controller ---
    ctrl_proc = spawn_controller(ctrl_cfg, terminal)
    # Pre-DRL wait; monitor additionally waits setting.MONITOR_START_DELAY. See main.py note.
    print("Controller spawned, wait 30s ...")
    time.sleep(30)

    # --- 5. Start probe servers ---
    probe_servers = start_probe_servers(net)
    print(f"Probe servers started on {len(probe_servers)} hosts")

    # --- 6. Session directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("results", alg_name, "test", timestamp)
    test_dir = os.path.join("results", alg_name, "test")
    os.makedirs(session_dir, exist_ok=True)
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid:
        try:
            uid = int(sudo_uid)
            gid = pwd.getpwuid(uid).pw_gid
            os.chown(test_dir, uid, gid)
            os.chown(session_dir, uid, gid)
        except Exception:
            pass
    print(f"Session dir: {session_dir}")

    # --- 7. TM iterator ---
    AUTO_TMS = [3, 10, 12, 14, 21]
    if args.tm:
        tm_list = [args.tm]
    elif args.tms:
        tm_list = [f"{t:02d}" for t in args.tms]
    elif args.auto:
        tm_list = [f"{t:02d}" for t in AUTO_TMS]
    else:
        tm_list = None  # interactive

    def iter_tms():
        if tm_list is not None:
            yield from tm_list
        else:
            while True:
                inp = input("Enter TM ID (e.g. 03) or QUIT: ").strip()
                if inp.upper() == "QUIT":
                    return
                try:
                    yield f"{int(inp):02d}"
                except ValueError:
                    print("Invalid input.")

    # --- 8. Per-TM loop ---
    for formatted_input in iter_tms():
        print(f"\n{'='*72}")
        print(f"[TM {formatted_input}] starting")
        print(f"{'='*72}")

        test_output_dir = os.path.join(session_dir, "real", formatted_input)
        os.makedirs(test_output_dir, exist_ok=True)
        if sudo_uid:
            try:
                uid = int(sudo_uid)
                gid = pwd.getpwuid(uid).pw_gid
                os.chown(test_output_dir, uid, gid)
            except Exception:
                pass

        # Spawn DRL agent (runs in separate terminal, waits 30s then steps)
        merged_cfg = {**env_cfg, **alg_cfg, **ctrl_cfg}
        if args.seed is not None:
            merged_cfg["seed"] = args.seed
        merged_cfg["tm_id"] = formatted_input
        merged_cfg["test_output_dir"] = test_output_dir
        merged_cfg["test_step"] = test_step

        drl_proc = spawn_drl(merged_cfg, "test_single", terminal)

        # Start iperf traffic (non-blocking)
        iperf_procs = start_iperf_traffic(net, tm_prefix, formatted_input, num_hosts)

        # DRL waits 30s internally (for controller).
        # iperf: servers (0s) + wait(10s) + clients → clients running at ~t+10s.
        # We add DRL-aligned warmup: DRL starts stepping at ~t+30s from its spawn.
        # So from here (after iperf clients started), wait ~20s to align with DRL step 0.
        drl_warmup = 20.0 + args.warmup
        print(f"[probe] waiting {drl_warmup:.0f}s to align with DRL first step ...")
        time.sleep(drl_warmup)

        # Run sampling
        csv_path = os.path.join(test_output_dir, "probe_vs_tc.csv")
        run_sampling_loop(
            net, directed_links, dev_map, bw_dict,
            tm_id=formatted_input,
            test_step=test_step,
            probe_timeout=args.probe_timeout,
            alg_name=alg_name,
            csv_path=csv_path,
            warmup_s=0,  # already waited above
        )

        # Stop iperf
        stop_iperf_traffic()
        print(f"[TM {formatted_input}] done, draining 10s ...")
        time.sleep(10)

        # DRL should have finished by now; terminate just in case
        drl_proc.terminate()
        try:
            drl_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            drl_proc.kill()

    # --- 9. Cleanup ---
    print("Cleaning up ...")
    stop_probe_servers(probe_servers)
    net.stop()
    ctrl_proc.terminate()
    try:
        ctrl_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        ctrl_proc.kill()

    print(f"\nDone. Results in {session_dir}")


if __name__ == "__main__":
    main()
