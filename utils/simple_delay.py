from __future__ import division
from ryu import cfg
from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_0, ofproto_v1_2, ofproto_v1_3
from ryu.lib import hub
from ryu.topology.switches import Switches
from ryu.topology.switches import LLDPPacket
from ryu.app import simple_switch_13
import networkx as nx
import csv
import os
import time
import setting

CONF = cfg.CONF


class simple_Delay(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    # _CONTEXTS = {"simple_awareness": simple_awareness.simple_Awareness}

    def __init__(self, *args, **kwargs):
        super(simple_Delay, self).__init__(*args, **kwargs)
        self.name = "delay"
        self.sending_echo_request_interval = 0.01
        self.sw_module = lookup_service_brick('switches')
        self.awareness = lookup_service_brick('awareness')
        self.datapaths = {}
        self.echo_latency = {}
        self.link_delay = {}
        self.link_delay_dir = {}   # directed: per-direction delay (ms)
        self.link_lldp_fwd_ms = {}  # raw LLDP forward delay (ms), no echo correction
        self.delay_trace_seq = 0
        self.delay_trace_links = self._parse_trace_links(os.environ.get("DELAY_TRACE_LINKS", ""))
        self.delay_trace_file = self._build_trace_file("DELAY_TRACE_FILE", self.delay_trace_links, "delay_trace")
        raw_links_env = os.environ.get("LLDP_RAW_TRACE_LINKS", "").strip()
        if not raw_links_env and self.delay_trace_links:
            raw_links_env = os.environ.get("DELAY_TRACE_LINKS", "")
        self.lldp_raw_trace_links = self._parse_trace_links(raw_links_env)
        self.lldp_raw_trace_file = self._build_trace_file(
            "LLDP_RAW_TRACE_FILE", self.lldp_raw_trace_links, "lldp_raw_trace"
        )
        # Get initiation delay.
        self.initiation_delay = 10#setting.DISCOVERY_PERIOD #wait topology discovery is done
        self.start_time = time.time()

        self.measure_thread = hub.spawn(self._detector)

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if not datapath.id in self.datapaths:
                self.logger.debug('Register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('Unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]


    def _detector(self):
        """
            Delay detecting functon.
            Send echo request and calculate link delay periodically
        """
        while True:
            self.delay_trace_seq += 1
            self._send_echo_request()
            self.create_link_delay()
            # try:
            #     self.awareness.shortest_paths = {}
            #     self.logger.debug("Refresh the shortest_paths")
            # except:
            #     self.awareness = lookup_service_brick('awareness')
            if self.awareness is not None:
                self.get_link_delay()
                self.get_link_delay_dir()
                self.show_delay_statis()
            hub.sleep(setting.DELAY_DETECTING_PERIOD)# + (setting.MONITOR_PERIOD - MONITOR_PERIOD))

    def _parse_trace_links(self, raw):
        raw = (raw or "").strip()
        links = set()
        if not raw:
            return links
        for token in raw.split(","):
            token = token.strip()
            if not token or "->" not in token:
                continue
            try:
                src_s, dst_s = token.split("->", 1)
                links.add((int(src_s), int(dst_s)))
            except Exception:
                continue
        return links

    def _build_trace_file(self, env_name, links, stem):
        if not links:
            return None
        trace_path = os.environ.get(env_name, "").strip()
        if trace_path:
            return trace_path
        alg_name = os.environ.get("ALG_NAME", "ls2ic")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(project_root, "artifacts", f"{stem}_{alg_name}.csv")

    def _append_delay_trace(self, src, dst, lldp_fwd, echo_src, echo_dst, d_dir_ms):
        if not self.delay_trace_links or (src, dst) not in self.delay_trace_links or not self.delay_trace_file:
            return
        monitor = lookup_service_brick('monitor')
        monitor_count = getattr(monitor, 'count_monitor', '')
        need_header = not os.path.exists(self.delay_trace_file)
        os.makedirs(os.path.dirname(self.delay_trace_file), exist_ok=True)
        with open(self.delay_trace_file, "a", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow([
                    "wall_time", "delay_seq", "monitor_count", "src", "dst",
                    "lldp_fwd_ms", "echo_src_ms", "echo_dst_ms", "d_dir_ms"
                ])
            writer.writerow([
                f"{time.time():.6f}",
                self.delay_trace_seq,
                monitor_count,
                src,
                dst,
                f"{lldp_fwd * 1000.0:.6f}",
                f"{echo_src * 1000.0:.6f}",
                f"{echo_dst * 1000.0:.6f}",
                f"{d_dir_ms:.6f}",
            ])

    def _append_lldp_raw_trace(
        self,
        src,
        src_port,
        dst,
        dst_port,
        send_timestamp,
        recv_timestamp,
        raw_lldp_ms,
        port_delay_ms,
    ):
        if not self.lldp_raw_trace_links or (src, dst) not in self.lldp_raw_trace_links or not self.lldp_raw_trace_file:
            return
        monitor = lookup_service_brick('monitor')
        monitor_count = getattr(monitor, 'count_monitor', '')
        need_header = not os.path.exists(self.lldp_raw_trace_file)
        os.makedirs(os.path.dirname(self.lldp_raw_trace_file), exist_ok=True)
        with open(self.lldp_raw_trace_file, "a", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow([
                    "wall_time",
                    "delay_seq",
                    "monitor_count",
                    "src",
                    "src_port",
                    "dst",
                    "dst_port",
                    "send_timestamp",
                    "recv_timestamp",
                    "raw_lldp_ms",
                    "port_delay_ms",
                ])
            writer.writerow([
                f"{time.time():.6f}",
                self.delay_trace_seq,
                monitor_count,
                src,
                src_port,
                dst,
                dst_port,
                f"{send_timestamp:.6f}" if send_timestamp is not None else "",
                f"{recv_timestamp:.6f}",
                f"{raw_lldp_ms:.6f}" if raw_lldp_ms is not None else "",
                f"{port_delay_ms:.6f}" if port_delay_ms is not None else "",
            ])

    def _send_echo_request(self):
        """
            Seng echo request msg to datapath.
        """
        for datapath in self.datapaths.values():
            parser = datapath.ofproto_parser
            s = "%.12f" % time.time()
            encoded = s.encode('utf-8')
            array = bytearray(encoded)
            echo_req = parser.OFPEchoRequest(datapath,
                                             data=array)  # data=bytes("%.12f" % time.time(),encoding = "utf-8")
            datapath.send_msg(echo_req)
            # Important! Don't send echo request together, Because it will
            # generate a lot of echo reply almost in the same time.
            # which will generate a lot of delay of waiting in queue
            # when processing echo reply in echo_reply_handler.
            hub.sleep(self.sending_echo_request_interval)

    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def echo_reply_handler(self, ev):
        """
            Handle the echo reply msg, and get the latency of link.
        """
        now_timestamp = time.time()
        try:
            latency = now_timestamp - eval(ev.msg.data)
            self.echo_latency[ev.msg.datapath.id] = latency
        except:
            return

    def get_delay(self, src, dst):
        """
            Get link delay.
                        Controller
                        |        |
        src echo latency|        |dst echo latency
                        |        |
                   SwitchA-------SwitchB
                        
                    fwd_delay--->
                        <----reply_delay
            delay = (forward delay + reply delay - src datapath's echo latency
        """
        try:
            fwd_delay = self.awareness.graph[src][dst]['lldpdelay']
            re_delay = self.awareness.graph[dst][src]['lldpdelay']
            src_latency = self.echo_latency[src]
            dst_latency = self.echo_latency[dst]
            # print('link {0}-{1} --> ')
            delay = (fwd_delay + re_delay - src_latency - dst_latency)/2
            return max(delay, 0)
        except:
            return float(0)

    def _save_lldp_delay(self, src=0, dst=0, lldpdelay=0):
        try:
            self.awareness.graph[src][dst]['lldpdelay'] = lldpdelay
        except:
            if self.awareness is None:
                self.awareness = lookup_service_brick('awareness')
            return

    def create_link_delay(self):
        """
            Create link delay data, and save it into graph object.
        """
        present_time = time.time()
        if present_time - self.start_time < self.initiation_delay: #Set to 30s
            return
        try:
            for src in self.awareness.graph:
                for dst in self.awareness.graph[src]:
                    if src == dst:
                        self.awareness.graph[src][dst]['delay'] = 0
                        continue
                    delay = self.get_delay(src, dst)
                    self.awareness.graph[src][dst]['delay'] = delay
            if self.awareness is not None:
                self.get_link_delay()
        except:
            if self.awareness is None:
                self.awareness = lookup_service_brick('awareness')
            return
           

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
            Explore LLDP packet and get the delay of link (fwd and reply).
        """
        msg = ev.msg
        recv_timestamp = time.time()
        try:
            src_dpid, src_port_no = LLDPPacket.lldp_parse(msg.data)
            dpid = msg.datapath.id
            if msg.datapath.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
                dst_port_no = msg.in_port
            elif msg.datapath.ofproto.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
                dst_port_no = msg.match['in_port']
            else:
                dst_port_no = ""
            if self.sw_module is None:
                self.sw_module = lookup_service_brick('switches')

            for port in self.sw_module.ports.keys():
                if src_dpid == port.dpid and src_port_no == port.port_no:
                    send_timestamp = self.sw_module.ports[port].timestamp
                    raw_lldp_ms = None
                    if send_timestamp:
                        raw_lldp_ms = (recv_timestamp - send_timestamp) * 1000.0
                    delay = self.sw_module.ports[port].delay
                    port_delay_ms = delay * 1000.0 if delay is not None else None
                    self._append_lldp_raw_trace(
                        src=src_dpid,
                        src_port=src_port_no,
                        dst=dpid,
                        dst_port=dst_port_no,
                        send_timestamp=send_timestamp,
                        recv_timestamp=recv_timestamp,
                        raw_lldp_ms=raw_lldp_ms,
                        port_delay_ms=port_delay_ms,
                    )
                    self._save_lldp_delay(src=src_dpid, dst=dpid,
                                          lldpdelay=delay)
        except LLDPPacket.LLDPUnknownFormat as e:
            return

    def get_link_delay(self):
        '''
        Calculates total link dealy and save it in self.link_delay[(node1,node2)]: link_delay
        '''
        i = time.time()
        for src in self.awareness.graph:
            for dst in self.awareness.graph[src]:
                if src != dst:
                    delay1 = self.awareness.graph[src][dst]['delay']
                    delay2 = self.awareness.graph[dst][src]['delay']
                    link_delay = ((delay1 + delay2)*1000.0)/2 #saves in ms
                    link = (src, dst)
                    self.link_delay[link] = link_delay

        
    def get_link_delay_dir(self):
        '''
        Directed link delay: extract per-direction delay from LLDP timestamps.
        lldpdelay(A→B) ≈ echo_A/2 + d(A→B) + echo_B/2
        => d(A→B) ≈ lldpdelay(A→B) - echo_A/2 - echo_B/2
        '''
        for src in self.awareness.graph:
            for dst in self.awareness.graph[src]:
                if src != dst:
                    lldp_fwd = self.awareness.graph[src][dst].get('lldpdelay', 0)
                    echo_src = self.echo_latency.get(src, 0)
                    echo_dst = self.echo_latency.get(dst, 0)
                    d_dir = max(lldp_fwd - echo_src / 2.0 - echo_dst / 2.0, 0)
                    d_dir_ms = d_dir * 1000.0
                    self.link_delay_dir[(src, dst)] = d_dir_ms
                    self.link_lldp_fwd_ms[(src, dst)] = lldp_fwd * 1000.0
                    self._append_delay_trace(src, dst, lldp_fwd, echo_src, echo_dst, d_dir_ms)

    def show_delay_statis(self):
        if self.awareness is None:
            print("Not doing nothing, awarness none")
        # else:
        #     print("Latency ok")
        # if setting.TOSHOW and self.awareness is not None:
        #     self.logger.info("\nsrc   dst      delay")
        #     self.logger.info("---------------------------")
        #     for src in self.awareness.graph:
        #         for dst in self.awareness.graph[src]:
        #             delay = self.awareness.graph[src][dst]['delay']
        #             self.logger.info("%s   <-->   %s :   %s" % (src, dst, delay))
