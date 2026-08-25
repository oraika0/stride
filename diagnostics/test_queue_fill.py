#!/usr/bin/env python3
"""
Test: does netem queue actually fill up and drop packets
when arrival rate > link capacity with UDP fixed rate?

Topology: h1 --(100M)-- s1 --(1M, queue=Q)-- s2 --(100M)-- h2
Traffic:  h1 sends UDP at 5 Mbps → 5x overload on bottleneck

Modes:
  htb+netem (default Mininet): HTB rate-limits, netem buffers
  netem-only: netem does rate+buffer+drop in one qdisc
"""
import subprocess, time, re, sys, os
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import OVSBridge

BOTTLENECK_BW = 1        # Mbps
SEND_RATE     = 5        # Mbps (5x overload)
DURATION      = 30       # seconds to run traffic
PKT_SIZE      = 1460     # bytes (UDP payload). Aligned with GEANT experiment:
                         # utils/iperf3_geant.py 不帶 -l 參數，iperf3 預設 UDP
                         # datagram = 1460B。tc backlog 計量為 L3 skb size =
                         # IP(20) + UDP(8) + payload(1460) = 1488B（與
                         # docs/sim_fluid_queue_calibration.md 校正參數一致）。

def get_netem_stats(dev='s1-eth2'):
    """Parse tc -s qdisc show for netem entries on given device."""
    raw = subprocess.check_output(['tc', '-s', 'qdisc', 'show', 'dev', dev]).decode()
    sent = dropped = backlog_pkts = 0
    lines = raw.split('\n')
    for i, line in enumerate(lines):
        if 'netem' in line:
            for j in range(i+1, min(i+4, len(lines))):
                m = re.search(r'Sent\s+\d+\s+bytes\s+(\d+)\s+pkt\s+\(dropped\s+(\d+),', lines[j])
                if m:
                    sent = int(m.group(1))
                    dropped = int(m.group(2))
                bm = re.search(r'backlog\s+(\d+)b\s+(\d+)p', lines[j])
                if bm:
                    backlog_pkts = int(bm.group(2))
    return sent, dropped, backlog_pkts

def get_all_qdisc_stats(dev='s1-eth2'):
    """Get total dropped from ALL qdiscs (htb + netem)."""
    raw = subprocess.check_output(['tc', '-s', 'qdisc', 'show', 'dev', dev]).decode()
    total_dropped = 0
    for line in raw.split('\n'):
        m = re.search(r'\(dropped\s+(\d+),', line)
        if m:
            total_dropped = max(total_dropped, int(m.group(1)))
    return total_dropped

def dump_tc(dev='s1-eth2'):
    raw = subprocess.check_output(['tc', '-s', 'qdisc', 'show', 'dev', dev]).decode()
    print(f"\n  tc -s qdisc show dev {dev}:")
    for line in raw.strip().split('\n'):
        print(f"    {line}")

def get_iface_tx_dropped(dev='s1-eth2'):
    """Read tx_dropped from /proc/net/dev."""
    with open('/proc/net/dev') as f:
        for line in f:
            if dev in line:
                # Format: iface: rx_bytes rx_pkt rx_err rx_drop ... tx_bytes tx_pkt tx_err tx_drop ...
                parts = line.split()
                # tx_drop is the 12th field (0-indexed: 11)
                return int(parts[11])
    return 0

def replace_with_netem_only(dev, bw_mbit, limit):
    """Replace HTB+netem with netem-only on device."""
    os.system(f'tc qdisc del dev {dev} root 2>/dev/null')
    cmd = f'tc qdisc add dev {dev} root netem rate {bw_mbit}mbit limit {limit}'
    print(f"  Running: {cmd}")
    os.system(cmd)

def run_test(queue_size, mode='htb_netem'):
    print(f"\n{'='*60}")
    print(f"  MODE: {mode}")
    print(f"  Queue limit = {queue_size}, BW = {BOTTLENECK_BW}Mbps, Send = {SEND_RATE}Mbps")
    print(f"  Expected max queuing delay = {queue_size * PKT_SIZE * 8 / (BOTTLENECK_BW * 1e6):.2f}s")
    print(f"{'='*60}")

    net = Mininet(link=TCLink, switch=OVSBridge)
    s1 = net.addSwitch('s1')
    s2 = net.addSwitch('s2')
    h1 = net.addHost('h1')
    h2 = net.addHost('h2')
    net.addLink(h1, s1, bw=100)
    net.addLink(s1, s2, bw=BOTTLENECK_BW, max_queue_size=queue_size)
    net.addLink(s2, h2, bw=100)
    net.start()
    time.sleep(2)

    # Replace tc qdisc if netem-only mode
    if mode == 'netem_only':
        print("\n  Replacing HTB+netem with netem-only...")
        replace_with_netem_only('s1-eth2', BOTTLENECK_BW, queue_size)
        # Also do the reverse direction
        replace_with_netem_only('s2-eth1', BOTTLENECK_BW, queue_size)
        time.sleep(1)

    net.pingAll()

    # Show tc qdisc config
    dump_tc('s1-eth2')

    # Baseline
    sent0, dropped0, _ = get_netem_stats()
    tx_drop0 = get_iface_tx_dropped('s1-eth2')
    print(f"\n  Baseline: netem sent={sent0}, dropped={dropped0}, iface tx_drop={tx_drop0}")

    # Start iperf3
    h2_obj = net.get('h2')
    h1_obj = net.get('h1')
    server_proc = h2_obj.popen(['iperf3', '-s', '-p', '5001'])
    time.sleep(1)
    client_proc = h1_obj.popen(['iperf3', '-c', h2_obj.IP(), '-u',
                                 '-b', f'{SEND_RATE}M', '-p', '5001',
                                 '-t', str(DURATION), '-l', str(PKT_SIZE)])
    time.sleep(1)

    # Verify processes
    ps_out = h1_obj.cmd('ps aux | grep "iperf3 -c" | grep -v grep')
    print(f"  Client: {ps_out.strip()}")

    # Monitor
    print(f"\n  {'time':>5s}  {'sent':>8s}  {'dropped':>8s}  {'backlog':>8s}  {'drop%':>8s}  {'iface_drop':>10s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    for t in range(0, DURATION + 2, 2):
        time.sleep(2)
        sent, dropped, backlog = get_netem_stats()
        tx_drop = get_iface_tx_dropped('s1-eth2')
        d_sent = sent - sent0
        d_dropped = dropped - dropped0
        d_tx_drop = tx_drop - tx_drop0
        total = d_sent + d_dropped
        rate = f"{d_dropped/total*100:.1f}%" if total > 0 else "n/a"
        print(f"  {t+2:>5d}s  {d_sent:>8d}  {d_dropped:>8d}  {backlog:>8d}  {rate:>8s}  {d_tx_drop:>10d}")

    # Final summary
    sent_f, dropped_f, backlog_f = get_netem_stats()
    tx_drop_f = get_iface_tx_dropped('s1-eth2')
    total_arrived = (sent_f - sent0) + (dropped_f - dropped0)
    total_dropped = dropped_f - dropped0
    total_sent = sent_f - sent0
    total_iface_drop = tx_drop_f - tx_drop0

    print(f"\n  SUMMARY:")
    print(f"    Total arrived at netem:  {total_arrived}")
    print(f"    Total sent (through):    {total_sent}")
    print(f"    netem dropped:           {total_dropped}")
    print(f"    iface tx_dropped:        {total_iface_drop}")
    print(f"    Final backlog:           {backlog_f} packets")
    if total_arrived > 0:
        print(f"    netem loss rate:         {total_dropped/total_arrived*100:.1f}%")
    expected_sent = BOTTLENECK_BW * 1e6 / 8 * DURATION / PKT_SIZE
    print(f"    Expected sent (C×t):     {expected_sent:.0f}")

    dump_tc('s1-eth2')

    client_proc.terminate()
    server_proc.terminate()
    net.stop()

if __name__ == '__main__':
    # Usage: python test_queue_fill.py [queue_size] [mode]
    # mode: htb_netem (default) or netem_only
    queue_size = 1000
    mode = 'htb_netem'
    if len(sys.argv) > 1:
        queue_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        mode = sys.argv[2]

    if mode == 'both':
        # Run both modes for comparison
        run_test(queue_size, 'htb_netem')
        run_test(queue_size, 'netem_only')
    else:
        run_test(queue_size, mode)
