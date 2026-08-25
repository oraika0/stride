from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.lib import hub
import subprocess
import re
import time
import setting


class TcLossMonitor(app_manager.RyuApp):
    """
    Reads tc qdisc stats to get real drop counts and queue depth.
    Exposes:
      self.link_loss_tc[(src, dst)]      = loss %
      self.link_queue_pkts[(src, dst)]   = backlog packets
      self.link_queue_bytes[(src, dst)]  = backlog bytes
    """

    def __init__(self, *args, **kwargs):
        super(TcLossMonitor, self).__init__(*args, **kwargs)
        self.name = "tc_loss"
        self.awareness = lookup_service_brick('awareness')
        self.tc_stats_prev = {}       # {dev: (sent_pkts, dropped_pkts)}
        self.link_loss_tc = {}        # {(src, dst): loss %}
        self.link_queue_pkts = {}     # {(src, dst): backlog packets}
        self.link_queue_bytes = {}    # {(src, dst): backlog bytes}
        self.initiation_delay = 10
        self.start_time = time.time()
        self.monitor_thread = hub.spawn(self._monitor_loop)

    def _monitor_loop(self):
        while True:
            hub.sleep(setting.MONITOR_PERIOD)
            if time.time() - self.start_time < self.initiation_delay:
                continue
            if self.awareness is None:
                self.awareness = lookup_service_brick('awareness')
                continue
            try:
                raw = subprocess.check_output(
                    ['tc', '-s', 'qdisc', 'show'],
                    timeout=5
                ).decode('utf-8')
                parsed = self._parse_tc_output(raw)
                self._compute_loss(parsed)
            except Exception as e:
                self.logger.debug("tc_loss poll error: %s" % e)

    def _parse_tc_output(self, raw):
        """
        Parse `tc -s qdisc show` output.
        For each switch dev (s\\d+-eth\\d+), extract (sent_pkts, dropped_pkts, backlog_bytes, backlog_pkts)
        from the leaf qdisc (netem if present, otherwise htb).

        Returns: {dev: (sent_pkts, dropped_pkts, backlog_bytes, backlog_pkts)}
        """
        result = {}
        dev_pattern = re.compile(r'^qdisc\s+(\w+)\s+\S+\s+dev\s+(s\d+-eth\d+)\s')
        stats_pattern = re.compile(r'Sent\s+\d+\s+bytes\s+(\d+)\s+pkt\s+\(dropped\s+(\d+),')
        backlog_pattern = re.compile(r'backlog\s+(\d+)b\s+(\d+)p\s')

        lines = raw.split('\n')
        i = 0
        while i < len(lines):
            m = dev_pattern.match(lines[i])
            if m:
                qdisc_type = m.group(1)   # htb or netem
                dev = m.group(2)
                sent = dropped = backlog_bytes = backlog_pkts = 0
                # parse next 2-3 lines for Sent and backlog
                for j in range(i + 1, min(i + 4, len(lines))):
                    sm = stats_pattern.search(lines[j])
                    if sm:
                        sent = int(sm.group(1))
                        dropped = int(sm.group(2))
                    bm = backlog_pattern.search(lines[j])
                    if bm:
                        backlog_bytes = int(bm.group(1))
                        backlog_pkts = int(bm.group(2))
                # netem overwrites htb for same dev (leaf wins)
                if qdisc_type == 'netem' or dev not in result:
                    result[dev] = (sent, dropped, backlog_bytes, backlog_pkts)
            i += 1
        return result

    def _dev_to_link(self, dev):
        """
        Map 's1-eth2' -> link (1, dst_dpid) using awareness.link_to_port.
        Returns (src_dpid, dst_dpid) or None.
        """
        m = re.match(r's(\d+)-eth(\d+)', dev)
        if not m:
            return None
        dpid = int(m.group(1))
        port_no = int(m.group(2))

        if not self.awareness or not self.awareness.link_to_port:
            return None

        for (src, dst), (src_port, dst_port) in self.awareness.link_to_port.items():
            if src == dpid and src_port == port_no:
                return (src, dst)
        return None

    def _compute_loss(self, parsed):
        for dev, (sent, dropped, backlog_bytes, backlog_pkts) in parsed.items():
            prev_sent, prev_dropped = self.tc_stats_prev.get(dev, (0, 0))
            d_sent = sent - prev_sent
            d_dropped = dropped - prev_dropped
            total = d_sent + d_dropped
            loss = d_dropped / total if total > 0 else 0.0
            link = self._dev_to_link(dev)
            if link:
                self.link_loss_tc[link] = loss * 100.0
                self.link_queue_pkts[link] = backlog_pkts
                self.link_queue_bytes[link] = backlog_bytes
            self.tc_stats_prev[dev] = (sent, dropped)
