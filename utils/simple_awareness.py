import networkx as nx
from itertools import islice
import matplotlib.pyplot as plt
import time

from ryu import cfg
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv4
from ryu.lib.packet import arp
from ryu.lib import hub
from ryu.topology import event
from ryu.topology.api import get_switch, get_link

import setting
import json,ast,os


CONF = cfg.CONF


class simple_Awareness(app_manager.RyuApp):
    """
        NetworkAwareness is a Ryu app for discovering topology information.
        This App can provide many data services for other App, such as
        link_to_port, access_table, switch_port_table, access_ports,
        interior_ports, topology graph and shortest paths.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    # List the event list should be listened.
    events = [event.EventSwitchEnter,
              event.EventSwitchLeave, event.EventPortAdd,
              event.EventPortDelete, event.EventPortModify,
              event.EventLinkAdd, event.EventLinkDelete]

    def __init__(self, *args, **kwargs):
        super(simple_Awareness, self).__init__(*args, **kwargs)
        self.topology_api_app = self
        self.name = "awareness"
        self.link_to_port = {}                # {(src_dpid,dst_dpid):(src_port,dst_port),}
        self.access_table = {}                # {(sw,port):(ip, mac),}
        self.switch_port_table = {}           # {dpid:set(port_num,),}
        self.access_ports = {}                # {dpid:set(port_num,),}
        self.interior_ports = {}              # {dpid:set(port_num,),}
        self.switches = []                    # self.switches = [dpid,]
        
        self.pre_link_to_port = {}
        self.pre_access_table = {}
        self.graph = nx.DiGraph()
        self.initiation_delay = 10 # # Get initiation delay.
        self.start_time = time.time()

        # 2026-06-08: LLDP-completeness diagnostic. NUM_LINK is the expected
        # directed link count (set by the caller via env); 0 => audit disabled.
        self.expected_links = int(os.environ.get("NUM_LINK", 0))
        self._last_link_n = -1

        self._topo_dirty = False
        self.rebuild_thread = hub.spawn(self._topology_worker)
        self.discover_thread = hub.spawn(self._discover)
        

    def _discover(self):

        time.sleep(self.initiation_delay)
        self._rebuild_topology()
        # 2026-06-08: one-shot completeness audit ~90s after the first build,
        # just before the caller's 120s start deadline. Makes a residual
        # discovery failure LOUD instead of a silent frozen-MLU run.
        if self.expected_links > 0:
            time.sleep(90)
            n = len(self.link_to_port)
            if n >= self.expected_links:
                self.logger.info("[topo] complete: %d/%d directed links", n, self.expected_links)
            else:
                self.logger.warning(
                    "[topo] INCOMPLETE: %d/%d directed links discovered -- routing WILL "
                    "fail on missing links (frozen MLU). LLDP hard-fail, not slowness.",
                    n, self.expected_links)
        #time.sleep(40000)

# ------------------------------------table-miss----------------------------------------
# --------------------------------------------------------------------------------------
#
#     Install table-miss flow entry to datapaths.
# 
    def add_flow(self, dp, priority, match, actions, idle_timeout=0, hard_timeout=0):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=priority,
                                idle_timeout=idle_timeout,
                                hard_timeout=hard_timeout,
                                match=match, instructions=inst)
        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):    
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.logger.info("switch:%s connected", datapath.id)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------


    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
            Handle the packet_in packet, and register the access info.
        """
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        eth_type = pkt.get_protocols(ethernet.ethernet)[0].ethertype #delay
        arp_pkt = pkt.get_protocol(arp.arp)
        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        if arp_pkt:
            arp_src_ip = arp_pkt.src_ip
            arp_dst_ip = arp_pkt.dst_ip #delay
            mac = arp_pkt.src_mac
            # Record the access infomation.
            self.register_access_info(datapath.id, in_port, arp_src_ip, mac)

        elif ip_pkt:
            ip_src_ip = ip_pkt.src
            eth = pkt.get_protocols(ethernet.ethernet)[0]
            mac = eth.src
            # Record the access infomation.
            self.register_access_info(datapath.id, in_port, ip_src_ip, mac)
        else:
            pass
            
    def is_topology_ready(self):
        switch_list = get_switch(self.topology_api_app, None)
        links = get_link(self.topology_api_app, None)
        return len(switch_list) > 0 and len(links) > 0
        
    @set_ev_cls(events)
    def get_topology(self, ev):
        # Handler contract, from ryu/doc/source/ryu_app_api.rst: "Because the
        # event handler is called in the context of the event processing
        # thread, it should be careful when blocking. While an event handler
        # is blocked, no further events for the Ryu application will be
        # processed."
        #
        # The rebuild below calls get_switch()/get_link(), which are synchronous
        # requests into the Switches app and block until it replies. Running
        # that here stalls this app's event loop, and because every app's queue
        # is bounded (hub.Queue(128)) a stalled loop eventually blocks the
        # per-datapath receive loops that feed it -- at which point the
        # controller stops reading every switch socket and never recovers.
        # See docs/ryu_controller_deadlock.md.
        #
        # Marking dirty also coalesces the burst of LinkAdd/PortAdd events that
        # topology discovery produces into a single rebuild.
        self._topo_dirty = True

    def _topology_worker(self):
        """
            Rebuild the topology outside the event loop, where blocking is safe.
        """
        while True:
            if self._topo_dirty:
                self._topo_dirty = False
                self._rebuild_topology()
            hub.sleep(setting.TOPO_REBUILD_PERIOD)

    def _rebuild_topology(self):
        """
            Blocking. Only ever call this from a greenthread of our own, never
            from an event handler.
        """
        present_time = time.time()
        if present_time - self.start_time < self.initiation_delay: #Set to 30s
            return
        self.logger.info("[Topology Discovery Ok]")
        
        switch_list = get_switch(self.topology_api_app, None)
        self.create_port_map(switch_list)
        
        retry_count = 0
        while not self.is_topology_ready():
            self.logger.info("Waiting for topology discovery to complete...")
            hub.sleep(0.1)  # 小幅等待並重試
            retry_count += 1
            if retry_count > 50:  # 最多重試 5 秒
                self.logger.warning("Topology discovery timeout! Proceeding with current data.")
                break

        switch_list = get_switch(self.topology_api_app, None)
        self.create_port_map(switch_list)
        
        self.switches = [sw.dp.id for sw in switch_list]
        links = get_link(self.topology_api_app, None)
        self.create_interior_links(links)
        # 2026-06-08: log the cumulative discovered count only when it changes,
        # so the controller terminal shows the climb (.. -> N/expected) without
        # per-event spam. A plateau below expected == the freeze cause.
        n = len(self.link_to_port)
        if n != self._last_link_n:
            self.logger.info("[topo] discovered %d/%d directed links", n, self.expected_links)
            self._last_link_n = n
        self.create_access_ports()
        self.graph = self.get_graph(self.link_to_port.keys())

        # get this once for topology and no more
        # graph_dict = nx.to_dict_of_dicts(self.graph)

        # with open('./graph_'+str(len(self.switches))+'Nodes.json','w') as json_file:
        #     json.dump(graph_dict, json_file, indent=2)

        # # print('topology',graph_dict)

        # self.shortest_paths = self.get_k_paths() 
        # k shorthest paths for drl--> removed from C0 since huge CPU consumptio
        # Now I calculate k_spaths outside, the agent just know it 
        # self.shortest_paths = self.all_k_shortest_paths(
        #     self.graph, weight='weight', k=1)

    def get_host_location(self, host_ip):
        """
            Get host location info ((datapath, port)) according to the host ip.
            self.access_table = {(sw,port):(ip, mac),}
        """
        # print('Access table: \n{0}'.format(self.access_table))
        # print(host_ip)
        for key in self.access_table.keys():
            if self.access_table[key][0] == host_ip:
                return key
        self.logger.info("%s location is not found." % host_ip)
        return None

    def get_graph(self, link_list):
        """
            Get Adjacency matrix from link_to_port.
        """
        _graph = self.graph.copy()
        for src in self.switches:
            for dst in self.switches:
                if src == dst:
                    _graph.add_edge(src, dst, weight=0)
                elif (src, dst) in link_list:
                    _graph.add_edge(src, dst, weight=1)
                else:
                    pass
        return _graph

    def create_port_map(self, switch_list):
        """
            Create interior_port table and access_port table.
        """
        for sw in switch_list:
            dpid = sw.dp.id
            self.switch_port_table.setdefault(dpid, set())
            # switch_port_table is equal to interior_ports plus access_ports.
            self.interior_ports.setdefault(dpid, set())
            self.access_ports.setdefault(dpid, set())
            for port in sw.ports:
                # switch_port_table = {dpid:set(port_num,),}
                self.switch_port_table[dpid].add(port.port_no)

    def create_interior_links(self, link_list):
        """
            Get links' srouce port to dst port  from link_list.
            link_to_port = {(src_dpid,dst_dpid):(src_port,dst_port),}
        """
        for link in link_list:
            src = link.src
            dst = link.dst
            self.link_to_port[(src.dpid, dst.dpid)] = (src.port_no, dst.port_no)
            # Find the access ports and interior ports.
            if link.src.dpid in self.switches:
                self.interior_ports[link.src.dpid].add(link.src.port_no)
            if link.dst.dpid in self.switches:
                self.interior_ports[link.dst.dpid].add(link.dst.port_no)

    def create_access_ports(self):
        """
            Get ports without link into access_ports.
        """
        for sw in self.switch_port_table:
            all_port_table = self.switch_port_table[sw]
            interior_port = self.interior_ports[sw]
            # That comes the access port of the switch.
            self.access_ports[sw] = all_port_table - interior_port

    def register_access_info(self, dpid, in_port, ip, mac):
        """
            Register access host info into access table.
        """
        if in_port in self.access_ports[dpid]:
            if (dpid, in_port) in self.access_table:
                if self.access_table[(dpid, in_port)] == (ip, mac):
                    return
                else:
                    self.access_table[(dpid, in_port)] = (ip, mac)
                    return
            else:
                self.access_table.setdefault((dpid, in_port), None)
                self.access_table[(dpid, in_port)] = (ip, mac)
                return
