#!/usr/bin/env python3
"""
Small UDP echo helper for Mininet namespace experiments.

Because Mininet hosts are network namespaces on the same kernel,
`time.time()` is on the same clock across h1/h2. That lets us estimate
forward and reverse one-way delay from echoed timestamps.
"""

import argparse
import json
import socket
import struct
import sys
import time


MAGIC = b"QPRB"
REQ_HDR = struct.Struct("!4sId")
RESP_HDR = struct.Struct("!4sIddd")


def run_server(host: str, port: int) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    while True:
        data, addr = sock.recvfrom(65535)
        recv_ts = time.time()
        if len(data) < REQ_HDR.size:
            continue
        magic, seq, client_send_ts = REQ_HDR.unpack_from(data)
        if magic != MAGIC:
            continue
        send_ts = time.time()
        reply = RESP_HDR.pack(MAGIC, seq, client_send_ts, recv_ts, send_ts)
        sock.sendto(reply, addr)


def run_client(host: str, port: int, size: int, timeout: float, seq: int) -> int:
    size = max(size, REQ_HDR.size)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)

    client_send_ts = time.time()
    payload = b"\x00" * (size - REQ_HDR.size)
    packet = REQ_HDR.pack(MAGIC, seq, client_send_ts) + payload

    result = {
        "ok": False,
        "seq": seq,
        "req_bytes": len(packet),
        "target_host": host,
        "target_port": port,
    }

    try:
        sock.sendto(packet, (host, port))
        reply, _ = sock.recvfrom(65535)
        client_recv_ts = time.time()
        if len(reply) < RESP_HDR.size:
            result["error"] = f"short_reply:{len(reply)}"
        else:
            magic, seq_r, client_send_echo, server_recv_ts, server_send_ts = RESP_HDR.unpack_from(reply)
            if magic != MAGIC:
                result["error"] = "bad_magic"
            else:
                result.update({
                    "ok": True,
                    "reply_bytes": len(reply),
                    "seq_reply": seq_r,
                    "client_send_ts": client_send_echo,
                    "server_recv_ts": server_recv_ts,
                    "server_send_ts": server_send_ts,
                    "client_recv_ts": client_recv_ts,
                    "fwd_ms": (server_recv_ts - client_send_echo) * 1000.0,
                    "rev_ms": (client_recv_ts - server_send_ts) * 1000.0,
                    "rtt_ms": (client_recv_ts - client_send_echo) * 1000.0,
                })
    except socket.timeout:
        result["error"] = "timeout"
    finally:
        sock.close()

    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_server = sub.add_parser("server")
    p_server.add_argument("--host", default="0.0.0.0")
    p_server.add_argument("--port", type=int, required=True)

    p_client = sub.add_parser("client")
    p_client.add_argument("--host", required=True)
    p_client.add_argument("--port", type=int, required=True)
    p_client.add_argument("--size", type=int, default=64)
    p_client.add_argument("--timeout", type=float, default=1.0)
    p_client.add_argument("--seq", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "server":
        return run_server(args.host, args.port)
    return run_client(args.host, args.port, args.size, args.timeout, args.seq)


if __name__ == "__main__":
    sys.exit(main())
