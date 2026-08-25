from operator import attrgetter

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.base.app_manager import lookup_service_brick
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.topology import event, switches
from ryu.ofproto.ether import ETH_TYPE_IP
from ryu.topology.api import get_switch, get_link
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import arp

from functools import reduce

import time

import simple_awareness
import simple_delay
import simple_monitor
import json, ast
import setting
import csv
import atomic_io
import time
import os

class Manager(app_manager.RyuApp):
    """
    Goal:
      1) Keep undirected pipeline IDENTICAL to original Manager:
         - read monitor.free_bandwidth (legacy)
         - read monitor.port_speed     (legacy)
         - use monitor.get_sw_dst() exactly like original
         - write net_info.csv and Metrics/'{count}_net_metrics.csv exactly like original (including the weird quote)
         - write paths_metrics.json exactly like original (json structure)
      2) Optionally add directed outputs (tx-only) from monitor.free_bandwidth_dir / port_speed_dir,
         but NEVER touch/override the undirected dicts or files.
    """

    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__(*args, **kwargs)
        self.name = "manager"
        self.awareness = lookup_service_brick("awareness")
        self.delay = lookup_service_brick("delay")
        self.monitor = lookup_service_brick("monitor")
        self.tc_loss = lookup_service_brick("tc_loss")

        self.link_loss = {}
        self.net_info = {}
        self.net_metrics= {}
        self.link_free_bw = {}
        self.link_used_bw = {}
        self.paths_metrics = {}

        # --- extra (new) directed outputs; does NOT affect legacy ---
        self.net_info_directed = {}
        self.net_metrics_directed = {}
        self.link_free_bw_dir = {}
        self.link_used_bw_dir = {}
        self.link_loss_dir = {}
        self.link_bw_capacity = {}  # {(src,dst): Mbps} lazily loaded from bw_r.txt
        # DRSIR mode C (build_directed_metrics): persistent per-directed-link
        # last-known [bwd_kbps, delay_tc_ms, loss_pct]; seeded from bw_r.txt.
        self._drsir_dir_state = {}

    # -----------------------------
    # legacy: port loss (unchanged)
    # -----------------------------
    def get_port_loss(self):
        #Get loss_port
        i = time.time()
        try:
            bodies = self.monitor.stats['port']
        except:
            self.monitor = lookup_service_brick('monitor')
            bodies = self.monitor.stats['port']

        for dp in sorted(bodies.keys()):
            for stat in sorted(bodies[dp], key=attrgetter('port_no')):
                if self.awareness.link_to_port and stat.port_no != 1 and stat.port_no != ofproto_v1_3.OFPP_LOCAL: #get loss form ports of network
                    key1 = (dp, stat.port_no)
                    tmp1 = self.monitor.port_stats[key1]
                    tx_bytes_src = tmp1[-1][0]
                    tx_pkts_src = tmp1[-1][8]
                    tx_pkts_src_period = tmp1[-1][8] - tmp1[0][8]

                    key2 = self.monitor.get_sw_dst(dp, stat.port_no)
                    tmp2 = self.monitor.port_stats.get(key2)
                    if tmp2 is None:  # peer side not populated yet on first cycles
                        continue
                    rx_bytes_dst = tmp2[-1][1]
                    rx_pkts_dst = tmp2[-1][9]
                    rx_pkts_dst_period = tmp2[-1][9] - tmp2[0][9]
                    loss_port = 0
                    if tx_pkts_src_period!=0:
                        loss_port = float(tx_pkts_src_period - rx_pkts_dst_period) / tx_pkts_src_period #loss rate
                    values = (loss_port, key2)
                    self.monitor.save_stats(self.monitor.port_loss[dp], key1, values, 5)

        #Calculates the total link loss and save it in self.link_loss[(node1,node2)]:loss
        _miss = _tot = 0   # links whose reverse-direction port is unmeasured this cycle
        for dp in self.monitor.port_loss.keys():
            for port in self.monitor.port_loss[dp]:
                key2 = self.monitor.port_loss[dp][port][-1][1]
                loss_src = self.monitor.port_loss[dp][port][-1][0]
                # tx_src = self.port_loss[dp][port][-1][1]
                # 2026-06-05: the reverse-direction port may be unmeasured when link
                # discovery is incomplete (32node LLDP race -> "Link X to Y is not in
                # links"). An unguarded port_loss[key2[0]][key2] then raised KeyError,
                # which killed the WHOLE monitor greenlet ("hub: uncaught exception")
                # -> every metric froze at its last value = the bit-identical
                # frozen-eval (noselfcond). Guard so the monitor survives, but COUNT
                # the fallbacks + warn below: a real/widespread discovery failure stays
                # VISIBLE (not silently masked); a couple = a transient to ignore.
                _tot += 1
                rev = self.monitor.port_loss.get(key2[0], {}).get(key2)
                if rev is None:
                    _miss += 1
                    loss_dst = loss_src   # directed link_loss_dir uses loss_src only
                else:
                    loss_dst = rev[-1][0]
                # tx_dst = self.port_loss[key2[0]][key2][-1][1]
                loss_l = max(abs(loss_src),abs(loss_dst)) #para DRL estoy cambiando cual es el loss del link... ahora es el max de los dos puertos, el peor de los casos, no el promedio
                link = (dp, key2[0])
                self.link_loss[link] = loss_l*100.0     #link loss ration in %

                # directed: per-direction loss, no max
                self.link_loss_dir[link] = abs(loss_src) * 100.0
        if _miss:
            self.logger.warning("get_port_loss: %d/%d links missing reverse-direction port "
                                "(incomplete discovery) -> loss fell back to src. "
                                "A few = transient; many = this eval's loss is DEGRADED.",
                                _miss, _tot)
        # print(self.link_loss)
        # print('Time get_port_loss', time.time()-i)

    # ---------------------------------------------------------
    # legacy: undirected free bw (IDENTICAL to original Manager)
    # ---------------------------------------------------------
    def get_link_free_bw(self):
        # Calculates total free bw of link and save it in self.link_free_bw[(u,v)]
        i = time.time()
        for dp in self.monitor.free_bandwidth.keys():
            for port in self.monitor.free_bandwidth[dp]:
                free_bw1 = self.monitor.free_bandwidth[dp][port]
                key2 = self.monitor.get_sw_dst(dp, port) #key2 = (dp,port)
                free_bw2= self.monitor.free_bandwidth[key2[0]][key2[1]]
                link_free_bw = min(free_bw1,free_bw2) #para DRL estoy cambiando cual es el bw del link... es el min de ambos, el peor de los caso, no el promedio
                link = (dp, key2[0])
                self.link_free_bw[link] = link_free_bw
        # print(self.free_bandwidth)
        # print('- - - - -  - - - - - - - - ')
        # print(self.link_free_bw)
        # print('Time to get link_free_bw', time.time()-i)

    # ---------------------------------------------------------
    # legacy: undirected used bw (IDENTICAL to original Manager)
    # ---------------------------------------------------------
    def get_link_used_bw(self):
        #Calculates the total free bw of link and save it in self.link_free_bw[(node1,node2)]:link_free_bw
        i = time.time()
        for key in self.monitor.port_speed.keys():
            used_bw1 = self.monitor.port_speed[key][-1]
            key2 = self.monitor.get_sw_dst(key[0], key[1]) #key2 = (dp,port)
            used_bw2 = self.monitor.port_speed[key2][-1]
            #print(used_bw1,used_bw2)
            link_used_bw = (used_bw1 + used_bw2)/2
            link = (key[0], key2[0])
            self.link_used_bw[link] = link_used_bw
        # print(self.link_free_bw)
        # print('Time to get link_used_bw', time.time()-i)

    # ---------------------------------------------------------
    # extra: directed (tx-only) link bw from monitor *_dir dicts
    # ---------------------------------------------------------
    def get_link_free_bw_dir(self):
        self.link_free_bw_dir = {}
        link_to_port = getattr(self.awareness, "link_to_port", None)
        fb_dir = getattr(self.monitor, "free_bandwidth_dir", None)
        if not link_to_port or fb_dir is None:
            return
        for (u, v), (u_port, v_port) in link_to_port.items():
            if u in fb_dir and u_port in fb_dir[u]:
                self.link_free_bw_dir[(u, v)] = fb_dir[u][u_port]

    def get_link_used_bw_dir(self):
        self.link_used_bw_dir = {}
        link_to_port = getattr(self.awareness, "link_to_port", None)
        ps_dir = getattr(self.monitor, "port_speed_dir", None)
        if not link_to_port or ps_dir is None:
            return
        for (u, v), (u_port, v_port) in link_to_port.items():
            key = (u, u_port)
            if key in ps_dir and len(ps_dir[key]) > 0:
                self.link_used_bw_dir[(u, v)] = ps_dir[key][-1]

    # ---------------------------------------------------------
    # tc-based queueing delay: backlog_bytes * 8 / C
    # ---------------------------------------------------------
    def _ensure_link_bw_capacity(self):
        """Lazily load per-link BW capacity (Mbps) from bw_r.txt once."""
        if self.link_bw_capacity:
            return
        env_name = os.environ.get("ENV_NAME", "geant")
        bw_file = f"../dataset/{env_name}_traffic/bw_r.txt"
        if not os.path.exists(bw_file):
            return
        with open(bw_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                src, dst = int(parts[0]), int(parts[1])
                bw_mbps = float(parts[3])
                self.link_bw_capacity[(src, dst)] = bw_mbps

    def get_tc_delay_ms(self, link, tc_queue_bytes):
        """Compute queueing delay (ms) = backlog_bytes * 8 / (C_bps)."""
        self._ensure_link_bw_capacity()
        queue_bytes = tc_queue_bytes.get(link, 0)
        bw_mbps = self.link_bw_capacity.get(link, 0)
        if bw_mbps <= 0 or queue_bytes <= 0:
            return 0.0
        return queue_bytes * 8.0 / (bw_mbps * 1e6) * 1000.0  # ms

    def build_undirected_corrected_metrics(self):
        """DRSIR 'corrected-reward' mode (env DRSIR_REWARD_CORRECTED=1).

        Build undirected delay/loss dicts from the CORRECTED directed measurement
        (tc queue-based delay_tc_ms + per-direction tc loss) that write_values()
        already populated into self.net_info_directed ([3]=delay_tc, [4]=loss_dir).
        Each undirected link takes the MEAN of its two directions and is keyed by
        BOTH (u,v) and (v,u) -- so metrics_links_kpaths(), which looks up directed
        consecutive-node tuples along a path, finds either orientation. Missing
        directions fall back to the LLDP/undirected value (degrades toward A).

        Returns (delay_corr, loss_corr) drop-in for self.delay.link_delay /
        self.link_loss at the get_k_paths_metrics_dic call site.
        """
        nid = self.net_info_directed  # {(u,v): [free_bw, lldp_fwd, delay_lldp, delay_tc, loss_dir, qp, qb]}
        delay_corr, loss_corr = {}, {}
        seen = set()
        for (u, v) in list(self.link_free_bw.keys()):
            if (u, v) in seen:
                continue
            d_uv = nid[(u, v)][3] if (u, v) in nid else self.delay.link_delay.get((u, v), 0.0)
            d_vu = nid[(v, u)][3] if (v, u) in nid else self.delay.link_delay.get((v, u), 0.0)
            d_mean = (d_uv + d_vu) / 2.0
            l_uv = nid[(u, v)][4] if (u, v) in nid else self.link_loss.get((u, v), 0.0)
            l_vu = nid[(v, u)][4] if (v, u) in nid else self.link_loss.get((v, u), 0.0)
            l_mean = (l_uv + l_vu) / 2.0
            delay_corr[(u, v)] = delay_corr[(v, u)] = d_mean
            loss_corr[(u, v)] = loss_corr[(v, u)] = l_mean
            seen.add((u, v)); seen.add((v, u))
        return delay_corr, loss_corr

    def build_directed_metrics(self):
        """DRSIR mode C (env DRSIR_REWARD_CORRECTED=2): STRIDE-aligned DIRECTED source.

        Returns (bwd_dir, delay_dir, loss_dir) dicts keyed by directed (u, v),
        drop-in for get_k_paths_metrics_dic's bwd/delay/loss args -- per-direction
        free bw (C - tx, kbps), tc queue delay (ms) and tc loss (%), i.e. the SAME
        three columns STRIDE's reward/state read from net_info_directed
        ([0]=bwd, [3]=delay_tc_ms, [4]=pkloss). write_values() runs earlier in the
        monitor cycle, so self.net_info_directed is this cycle's measurement.

        Missing-link policy: keep the LAST-KNOWN value per link across cycles.
        First call seeds EVERY directed link from bw_r.txt as an idle prior
        (bwd=C, delay=0, loss=0); bw_r.txt lists both directions (geant 74 /
        32node 120), so any k_paths hop always resolves -- a transient LLDP gap
        degrades to a stale value instead of a KeyError that would kill the
        monitor greenlet (frozen-metrics failure mode, see get_port_loss note).
        """
        self._ensure_link_bw_capacity()
        if not self._drsir_dir_state and self.link_bw_capacity:
            for link, bw_mbps in self.link_bw_capacity.items():
                self._drsir_dir_state[link] = [bw_mbps * 1000.0, 0.0, 0.0]
        measured = 0
        for link, row in self.net_info_directed.items():
            self._drsir_dir_state[link] = [row[0], row[3], row[4]]
            measured += 1
        stale = len(self._drsir_dir_state) - measured
        if stale:
            self.logger.warning("build_directed_metrics: %d/%d directed links "
                                "unmeasured this cycle -> kept last-known value "
                                "(idle prior if never measured).",
                                stale, len(self._drsir_dir_state))
        bwd_dir = {l: v[0] for l, v in self._drsir_dir_state.items()}
        delay_dir = {l: v[1] for l, v in self._drsir_dir_state.items()}
        loss_dir = {l: v[2] for l, v in self._drsir_dir_state.items()}
        return bwd_dir, delay_dir, loss_dir

    # ---------------------------------------------------------
    # legacy write (IDENTICAL content/format to original)
    # + add directed csvs in parallel (optional)
    # ---------------------------------------------------------
    def write_values(self, alg_name):
        a = time.time()
        # self.delay = lookup_service_brick('delay')
        # print('\nwriting file............')
        # print(self.free_bandwidth[1][2] , self.free_bandwidth[7][4] )
        # print('- - - - -  - - - - - - - - ')
        # print(self.link_free_bw[(1, 7)], self.link_free_bw[(7, 1)])
        # print('- - - - -  - - - - - - - - ')
        # if self.delay is None:
        #     self.delay = app_manager.lookup_service_brick('delay')
        # else:
        if self.delay is not None:
            for link in self.link_free_bw:
                # print('loss_links', self.link_loss)
                self.net_info[link] = [round(self.link_free_bw[link],6) , round(self.delay.link_delay[link],6), round(self.link_loss[link],6)]
                self.net_metrics[link] = [round(self.link_free_bw[link],6), round(self.link_used_bw[link],6), round(self.delay.link_delay[link],6), round(self.link_loss[link],6)]
                
            # print(self.net_info[(1, 7)])
            with atomic_io.replacing(f"../results/{alg_name}/net_info.csv") as csvfile:
                header_names = ['node1','node2','bwd','delay','pkloss']
                file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                links_in = []
                file.writerow(header_names)
                for link, values in sorted(self.net_info.items()):
                    links_in.append(link)
                    tup = (link[1], link[0])
                    if tup not in links_in:
                        file.writerow([link[0],link[1], values[0],values[1],values[2]])

            file_metrics = f"../results/{alg_name}/Metrics/'{str(self.monitor.count_monitor)}_net_metrics.csv"
            with open(file_metrics,'w') as csvfile:
                header_ = ['node1','node2','free_bw','used_bw','delay','pkloss']
                file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                links_in = []
                file.writerow(header_)
                for link, values in sorted(self.net_metrics.items()):
                    links_in.append(link)
                    tup = (link[1], link[0])
                    if tup not in links_in:
                        file.writerow([link[0],link[1],values[0],values[1],values[2],values[3]])

            # directed outputs (guarded: never crash the monitor cycle)
            try:
                self.get_link_free_bw_dir()
                self.get_link_used_bw_dir()
                if self.tc_loss is None:
                    self.tc_loss = lookup_service_brick('tc_loss')
                tc_loss_data = getattr(self.tc_loss, 'link_loss_tc', {}) if self.tc_loss else {}
                tc_queue_data = getattr(self.tc_loss, 'link_queue_pkts', {}) if self.tc_loss else {}
                tc_queue_bytes = getattr(self.tc_loss, 'link_queue_bytes', {}) if self.tc_loss else {}
                if self.link_free_bw_dir:
                    self.net_info_directed = {}
                    self.net_metrics_directed = {}
                    delay_dir = getattr(self.delay, 'link_delay_dir', {})
                    lldp_fwd_dir = getattr(self.delay, 'link_lldp_fwd_ms', {})
                    for link in self.link_free_bw_dir:
                        loss_dir = tc_loss_data.get(link, self.link_loss_dir.get(link, 0.0))
                        queue_pkts = tc_queue_data.get(link, 0)
                        queue_bytes = tc_queue_bytes.get(link, 0)
                        delay_tc = self.get_tc_delay_ms(link, tc_queue_bytes)
                        lldp_fwd_ms = lldp_fwd_dir.get(link, 0.0)
                        self.net_info_directed[link] = [round(self.link_free_bw_dir[link],6), round(lldp_fwd_ms,6), round(delay_dir.get(link,0.0),6), round(delay_tc,6), round(loss_dir,6), queue_pkts, queue_bytes]
                        self.net_metrics_directed[link] = [round(self.link_free_bw_dir[link],6), round(self.link_used_bw_dir.get(link,0.0),6), round(lldp_fwd_ms,6), round(delay_dir.get(link,0.0),6), round(delay_tc,6), round(loss_dir,6), queue_pkts, queue_bytes]

                    with atomic_io.replacing(f"../results/{alg_name}/net_info_directed.csv") as csvfile:
                        header_names = ['node1','node2','bwd','lldp_fwd_ms','delay_lldp','delay_tc_ms','pkloss','queue_pkts','queue_bytes']
                        file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        file.writerow(header_names)
                        for link, values in sorted(self.net_info_directed.items()):
                            file.writerow([link[0],link[1], values[0],values[1],values[2],values[3],values[4],values[5],values[6]])

                    file_metrics_dir = f"../results/{alg_name}/Metrics/'{str(self.monitor.count_monitor)}_net_metrics_directed.csv"
                    with open(file_metrics_dir,'w') as csvfile:
                        header_ = ['node1','node2','free_bw','used_bw','lldp_fwd_ms','delay_lldp','delay_tc_ms','pkloss','queue_pkts','queue_bytes']
                        file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        file.writerow(header_)
                        for link, values in sorted(self.net_metrics_directed.items()):
                            file.writerow([link[0],link[1],values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7]])
            except Exception as e:
                print("[manager] directed output error (ignored): %s" % e)

            b = time.time()
            # print('total writing time: {0}'.format(b-a))
            return
        else:
            self.delay = lookup_service_brick('delay')
            # if self.delay.link_delay:
            for link in self.link_free_bw:
                # print('fre_links', self.link_free_bw)
                # print('loss_links', self.link_loss)
                self.net_info[link] = [round(self.link_free_bw[link],6) , round(self.delay.link_delay[link],6), round(self.link_loss[link],6)]
                self.net_metrics[link] = [round(self.link_free_bw[link],6), round(self.link_used_bw[link],6), round(self.delay.link_delay[link],6), round(self.link_loss[link],6)]

            # print(self.net_info[(1, 7)])
            with atomic_io.replacing(f"../results/{alg_name}/net_info.csv") as csvfile:
                header_names = ['node1','node2','bwd','delay','pkloss']
                file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                links_in = []
                file.writerow(header_names)
                for link, values in sorted(self.net_info.items()):
                    links_in.append(link)
                    tup = (link[1], link[0])
                    if tup not in links_in:
                        file.writerow([link[0],link[1], values[0],values[1],values[2]])

            file_metrics = f"../results/{alg_name}/Metrics/'{str(self.monitor.count_monitor)}_net_metrics.csv"
            with open(file_metrics,'w') as csvfile:
                header_ = ['node1','node2','free_bw','used_bw','delay','pkloss']
                file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                links_in = []
                file.writerow(header_)
                for link, values in sorted(self.net_metrics.items()):
                    links_in.append(link)
                    tup = (link[1], link[0])
                    if tup not in links_in:
                        file.writerow([link[0],link[1],values[0],values[1],values[2],values[3]])

            # directed outputs (guarded: never crash the monitor cycle)
            try:
                self.get_link_free_bw_dir()
                self.get_link_used_bw_dir()
                if self.tc_loss is None:
                    self.tc_loss = lookup_service_brick('tc_loss')
                tc_loss_data = getattr(self.tc_loss, 'link_loss_tc', {}) if self.tc_loss else {}
                tc_queue_data = getattr(self.tc_loss, 'link_queue_pkts', {}) if self.tc_loss else {}
                tc_queue_bytes = getattr(self.tc_loss, 'link_queue_bytes', {}) if self.tc_loss else {}
                if self.link_free_bw_dir:
                    self.net_info_directed = {}
                    self.net_metrics_directed = {}
                    delay_dir = getattr(self.delay, 'link_delay_dir', {})
                    lldp_fwd_dir = getattr(self.delay, 'link_lldp_fwd_ms', {})
                    for link in self.link_free_bw_dir:
                        loss_dir = tc_loss_data.get(link, self.link_loss_dir.get(link, 0.0))
                        queue_pkts = tc_queue_data.get(link, 0)
                        queue_bytes = tc_queue_bytes.get(link, 0)
                        delay_tc = self.get_tc_delay_ms(link, tc_queue_bytes)
                        lldp_fwd_ms = lldp_fwd_dir.get(link, 0.0)
                        self.net_info_directed[link] = [round(self.link_free_bw_dir[link],6), round(lldp_fwd_ms,6), round(delay_dir.get(link,0.0),6), round(delay_tc,6), round(loss_dir,6), queue_pkts, queue_bytes]
                        self.net_metrics_directed[link] = [round(self.link_free_bw_dir[link],6), round(self.link_used_bw_dir.get(link,0.0),6), round(lldp_fwd_ms,6), round(delay_dir.get(link,0.0),6), round(delay_tc,6), round(loss_dir,6), queue_pkts, queue_bytes]

                    with atomic_io.replacing(f"../results/{alg_name}/net_info_directed.csv") as csvfile:
                        header_names = ['node1','node2','bwd','lldp_fwd_ms','delay_lldp','delay_tc_ms','pkloss','queue_pkts','queue_bytes']
                        file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        file.writerow(header_names)
                        for link, values in sorted(self.net_info_directed.items()):
                            file.writerow([link[0],link[1], values[0],values[1],values[2],values[3],values[4],values[5],values[6]])

                    file_metrics_dir = f"../results/{alg_name}/Metrics/'{str(self.monitor.count_monitor)}_net_metrics_directed.csv"
                    with open(file_metrics_dir,'w') as csvfile:
                        header_ = ['node1','node2','free_bw','used_bw','lldp_fwd_ms','delay_lldp','delay_tc_ms','pkloss','queue_pkts','queue_bytes']
                        file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        file.writerow(header_)
                        for link, values in sorted(self.net_metrics_directed.items()):
                            file.writerow([link[0],link[1],values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7]])
            except Exception as e:
                print("[manager] directed output error (ignored): %s" % e)

            b = time.time()
            # print('total writing time: {0}'.format(b-a))
            return

    # ----------Path metrics -------- 
    def get_k_paths_nodes(self,shortest_paths,src,dst):
        k_paths = shortest_paths[src][dst]
        return k_paths

    def calc_bwd_path(self,bwd_links_path):
        '''
        path = [link1, link2, link3]
        path_bwd = min(bwd of all links)
        '''
        bwd_path = min(bwd_links_path)
        return round(bwd_path,6)

    def calc_delay_path(self,delay_links_path):
        '''
        path = [link1, link2, link3]
        path_ldelay = sum(delay of all links)
        '''
        delay_path = sum(delay_links_path)
        return round(delay_path,6)

    def calc_loss_path(self,loss_links_path): 
        '''
        path = [link1, link2, link3]
        path_loss = 1-[(1-loss_link1)*(1-loss_link2)*(1-loss_link3)]
        '''
        loss_links_path_ = [1-(i/100.0) for i in loss_links_path]
        result_multi = reduce((lambda x, y: x * y), loss_links_path_)
        loss_path = 1.0 - result_multi
        return round(loss_path*100.0,6)

    def metrics_links_kpaths(self,k_paths,bwd_links,delay_links,loss_links):
        '''
        Calculates the metrics for k_paths of a pair of nodes src - dst
        k_paths = [path1, path2, ..., pathk]

        '''
        bwd_paths_nodes = []
        delay_paths_nodes = []
        loss_paths_nodes = []

        # print('------****',src,dst)
        for path in k_paths:
            # print('------',src,dst,path)
            bwd_links_path = []
            delay_links_path = []
            loss_links_path = []
            for i in range(len(path)-1):
                link_ = (path[i],path[i+1])

                bwd = round(bwd_links[link_],6)
                delay = round(delay_links[link_],6)
                loss = round(loss_links[link_],6)

                bwd_links_path.append(bwd)
                delay_links_path.append(delay)
                loss_links_path.append(loss)

            bwd_path = self.calc_bwd_path(bwd_links_path)
            bwd_paths_nodes.append(bwd_path)

            delay_path = self.calc_delay_path(delay_links_path)
            delay_paths_nodes.append(delay_path)

            loss_path = self.calc_loss_path(loss_links_path)
            loss_paths_nodes.append(loss_path)

        # bwd_paths[src][dst] = bwd_paths_nodes
        # delay_paths[src][dst] = delay_paths_nodes
        # loss_paths[src][dst] = loss_paths_nodes

        return bwd_paths_nodes,delay_paths_nodes,loss_paths_nodes

    def get_k_paths_metrics_dic(self,shortest_paths,bwd_links,delay_links,loss_links, alg_name):

        i = time.time()
        metrics = ['bwd_paths','delay_paths','loss_paths']
        # print('------switches',self.awareness.switches)
        for sw in shortest_paths.keys():
            self.paths_metrics.setdefault(sw,{})
            for sw2 in shortest_paths.keys():
                if sw != sw2:
                    self.paths_metrics[sw].setdefault(sw2,{})
                    for m in metrics:
                        self.paths_metrics[sw][sw2].setdefault(m,)

            # if shortest_paths is not None:
         
        for src in shortest_paths.keys():
            for dst in shortest_paths[src].keys():
                if src != dst:
                    k_paths = self.get_k_paths_nodes(shortest_paths,src,dst)
                    bwd_paths_nodes, delay_paths_nodes, loss_paths_nodes = self.metrics_links_kpaths(k_paths,bwd_links,delay_links,loss_links)      
                    # print('---',src,dst,bwd_paths_nodes, delay_paths_nodes, loss_paths_nodes)
                    self.paths_metrics[src][dst][metrics[0]] = [bwd_paths_nodes]
                    self.paths_metrics[src][dst][metrics[1]] = [delay_paths_nodes]
                    self.paths_metrics[src][dst][metrics[2]] = [loss_paths_nodes]
        print('writing paths_metrics')
        
        with atomic_io.replacing(f"../results/{alg_name}/paths_metrics.json") as json_file:
            json.dump(self.paths_metrics, json_file, indent=2) 
        
        print('------****metrics k_paths', time.time()-i)

    def get_k_paths_metrics(self,shortest_paths,bwd_links,delay_links,loss_links):
        ''' escribe las metricas en un diccionario por separado
            bwd_paths [src][dst]:[bwd1,bwd1,bwd3...,bwdk]''' 
        for sw in self.awareness.switches:
            self.bwd_paths.setdefault(sw,{})
            self.delay_paths.setdefault(sw,{})
            self.loss_paths.setdefault(sw,{})
            for sw2 in self.awareness.switches:
                if sw != sw2:
                    self.bwd_paths[sw].setdefault(sw2,[])
                    self.delay_paths[sw].setdefault(sw2,[])
                    self.loss_paths[sw].setdefault(sw2,[])

        if shortest_paths is not None:
            for src in shortest_paths.keys():
                for dst in shortest_paths[src].keys():
                    if src != dst:
                        k_paths = self.get_k_paths_nodes(shortest_paths,src,dst)
                        bwd_paths_nodes, delay_paths_nodes, loss_paths_nodes = self.metrics_links_kpaths(k_paths,bwd_links,delay_links,loss_links)      
                        self.bwd_paths[src][dst] = bwd_paths_nodes
                        self.delay_paths[src][dst] = delay_paths_nodes
                        self.loss_paths[src][dst] = loss_paths_nodes
            # print('bwd_paths',self.bwd_paths) 
            # print('delay_paths',self.delay_paths)
            # print('loss_paths',self.loss_paths)
            
            return 

    # def write_values_paths(self):
    #     a = time.time()
           
    #     if self.delay is not None:
    #         for link in self.link_free_bw:
    #             self.net_info[link] = [round(self.link_free_bw[link],6), round(self.delay.link_delay[link],6), round(self.link_loss[link],6)]
    #             self.net_metrics[link] = [round(self.link_free_bw[link],6), round(self.link_used_bw[link],6), round(self.delay.link_delay[link],6), round(self.link_loss[link],6)]
                
    #         # print(self.net_info[(1, 7)])
    #         with open('./net_info.csv','wb') as csvfile:
    #             header_names = ['node1','node2','bwd','delay','pkloss']
    #             file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #             links_in = []
    #             file.writerow(header_names)
    #             for link, values in sorted(self.net_info.items()):
    #                 links_in.append(link)
    #                 tup = (link[1], link[0])
    #                 if tup not in links_in:
    #                     file.writerow([link[0],link[1], values[0],values[1],values[2]])

    #         file_metrics = './Metrics/'+str(self.monitor.count_monitor)+'_net_metrics.csv'
    #         with open(file_metrics,'wb') as csvfile:
    #             header_ = ['node1','node2','free_bw','used_bw','delay','pkloss']
    #             file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #             links_in = []
    #             file.writerow(header_)
    #             for link, values in sorted(self.net_metrics.items()):
    #                 links_in.append(link)
    #                 tup = (link[1], link[0])
    #                 if tup not in links_in:
    #                     file.writerow([link[0],link[1],values[0],values[1],values[2],values[3]]) 
    #         b = time.time()            
    #         # print('total writing time: {0}'.format(b-a))
    #         return
    #     else:
    #         self.delay = lookup_service_brick('delay')
    #         # if self.delay.link_delay:
    #         for link in self.link_free_bw:
    #             # print('fre_links', self.link_free_bw)
    #             print('loss_links', self.link_loss)
    #             self.net_info[link] = [round(self.link_free_bw[link],6) , round(self.delay.link_delay[link],6), round(self.link_loss[link],6)]
    #             self.net_metrics[link] = [round(self.link_free_bw[link],6), round(self.link_used_bw[link],6), round(self.delay.link_delay[link],6), round(self.link_loss[link],6)]
        
    #         # print(self.net_info[(1, 7)])
    #         with open('./net_info.csv','wb') as csvfile:
    #             header_names = ['node1','node2','bwd','delay','pkloss']
    #             file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #             links_in = []
    #             file.writerow(header_names)
    #             for link, values in sorted(self.net_info.items()):
    #                 links_in.append(link)
    #                 tup = (link[1], link[0])
    #                 if tup not in links_in:
    #                     file.writerow([link[0],link[1], values[0],values[1],values[2]])

    #         file_metrics = './Metrics/'+str(self.monitor.count_monitor)+'_net_metrics.csv'
    #         with open(file_metrics,'wb') as csvfile:
    #             header_ = ['node1','node2','free_bw','used_bw','delay','pkloss']
    #             file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #             links_in = []
    #             file.writerow(header_)
    #             for link, values in sorted(self.net_metrics.items()):
    #                 links_in.append(link)
    #                 tup = (link[1], link[0])
    #                 if tup not in links_in:
    #                     file.writerow([link[0],link[1],values[0],values[1],values[2],values[3]]) 
    #         b = time.time()            
    #         # print('total writing time: {0}'.format(b-a))
    #         return