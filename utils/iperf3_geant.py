import argparse
import json
import os
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# =============================================================================
# Shared helpers (pure functions — no side effects on module state)
# =============================================================================

def _parse_xml_to_tm(xml_path: str, tm_scale: int, n_nodes: int = 24) -> np.ndarray:
    """Parse one Géant IntraTM XML → (n_nodes × n_nodes) kbps matrix * tm_scale.

    Node IDs in the XML are 1-based; we keep the 0-row/0-col unused to match
    the original script (which indexes src/dst from 1..23).
    """
    tm = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Original file structure: root[1] is the <IntraTM> block.
    for src in root[1]:
        for dst in src:
            s = int(src.attrib.get("id"))
            d = int(dst.attrib.get("id"))
            tm[s][d] = float(dst.text) * tm_scale
    # Zero the diagonal defensively.
    for s in range(1, n_nodes):
        tm[s][s] = 0.0
    return tm


def _write_tm_iperf_dir(tm: np.ndarray, tm_dir: Path, duration: int) -> int:
    """Write TM-XXXX/{Clients,Servers}/*.sh. Returns number of active flows."""
    clients_dir = tm_dir / "Clients"
    servers_dir = tm_dir / "Servers"
    clients_dir.mkdir(parents=True, exist_ok=True)
    servers_dir.mkdir(parents=True, exist_ok=True)

    # ---- Clients ----
    for src in range(1, 24):
        fname = f"client_{src:02d}.sh"
        with open(clients_dir / fname, "w") as fc:
            for dst in range(1, 24):
                throughput = float(tm[src][dst])
                # /100: scale TM to mininet link capacities (matches original)
                throughput_g = throughput / 100.0
                if throughput_g < 0.001:
                    continue
                # Port encoding: src-prefix + dst-3digit
                if dst > 9:
                    port = f"{src}0{dst}"
                else:
                    port = f"{src}00{dst}"
                fc.write("\n")
                fc.write(f"stdbuf -o0 iperf3 -c 10.0.0.{dst} -p {port} "
                         f"-u -b {throughput_g:.3f}k -w 256k -t {duration} -i 0 &\n")
                fc.write("sleep 0.4")

    # ---- Servers ----
    for dst in range(1, 24):
        fname = f"server_{dst:02d}.sh"
        with open(servers_dir / fname, "w") as fs:
            for src in range(1, 24):
                if dst > 9:
                    port = f"{src}0{dst}"
                else:
                    port = f"{src}00{dst}"
                fs.write("\n")
                fs.write(f"iperf3 -s -p {port} -1 & \nsleep 0.3")

    # ---- Count active flows for summary ----
    n_flows = int(np.count_nonzero(tm[1:, 1:]))
    return n_flows


# =============================================================================
# Legacy entry point — preserved so existing callers keep working.
# Generates 24 hourly TMs from 2005-01-03 (hardcoded).
# =============================================================================

def generate_traffic(data_file: str, out_dir: str, duration: int = 80000, tm_scale: int = 5):
    for i in range(24):
        tag = f"0{i}" if i < 10 else f"{i}"
        xml_name = f"IntraTM-2005-01-03-{tag}-00.xml"
        xml_path = os.path.join(data_file, "traffic-matrices", xml_name)
        tm = _parse_xml_to_tm(xml_path, tm_scale=tm_scale)

        tm_dir = Path(out_dir) / (f"TM-0{i}" if i < 10 else f"TM-{i}")
        tm_dir.mkdir(parents=True, exist_ok=True)
        _write_tm_iperf_dir(tm, tm_dir, duration=duration)

        # Original script printed total + mean load per TM.
        loads = tm[1:, 1:][tm[1:, 1:] > 0.0].tolist()
        if loads:
            print(sum(loads))
            print(statistics.mean(loads))


# =============================================================================
# New entry point — generate iperf scripts for EVERY hourly TM in a directory.
# Output TM indices match convert_geant_tm_to_enero_tm.py so TM-XXXX aligns 1:1
# with Geant.XXXX.demands in NEW_Geant_s3/TRAIN/TM/.
# =============================================================================

def generate_traffic_from_xmls(
    xml_files,
    out_root: str,
    duration: int = 80000,
    tm_scale: int = 3,
    name_fmt: str = "TM-{:04d}",
    index_width: int = 4,
) -> dict:
    """Generate one TM-XXXX/{Clients,Servers} directory per XML.

    Args:
        xml_files: iterable of absolute XML paths, already filtered & sorted.
        out_root: destination root (e.g. dataset/geant_traffic/NEW_Geant_s3_perhour)
        duration: iperf3 -t seconds
        tm_scale: bandwidth multiplier (should match convert_geant_tm_to_enero_tm scale)
        name_fmt: python format string for the TM-index directory name.
    """
    out_root_p = Path(out_root)
    out_root_p.mkdir(parents=True, exist_ok=True)

    manifest = {}
    loads_global = []

    for idx, xml_path in enumerate(xml_files):
        tm_name = name_fmt.format(idx)
        tm_dir = out_root_p / tm_name
        tm = _parse_xml_to_tm(xml_path, tm_scale=tm_scale)
        n_flows = _write_tm_iperf_dir(tm, tm_dir, duration=duration)

        manifest[tm_name] = {
            "source_xml": os.path.basename(xml_path),
            "n_flows": n_flows,
            "tm_scale": tm_scale,
        }

        flows = tm[1:, 1:][tm[1:, 1:] > 0.0]
        if flows.size:
            loads_global.extend(flows.tolist())

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[iperf3_geant] {idx + 1}/{len(xml_files)}  "
                  f"TM={tm_name}  src={manifest[tm_name]['source_xml']}  "
                  f"flows={n_flows}")

    # Manifest + README
    with open(out_root_p / "manifest.json", "w") as f:
        json.dump({
            "_meta": {
                "tm_scale": tm_scale,
                "duration": duration,
                "count": len(manifest),
                "index_width": index_width,
            },
            "tms": manifest,
        }, f, indent=2)

    if loads_global:
        print(f"[iperf3_geant] DONE.  {len(manifest)} TMs written to {out_root_p}")
        print(f"[iperf3_geant] flows/TM: total={sum(loads_global):.1f} kbps  "
              f"mean={statistics.mean(loads_global):.3f} kbps")
    return manifest


def _collect_perhour_xmls(input_dir: str, start_xml: str = "IntraTM-2005-01-03-00-00.xml"):
    """List all hourly (mm==00) XMLs from input_dir, sorted, starting at start_xml.

    Mirrors convert_geant_tm_to_enero_tm.py so TM indices match Geant.<idx>.demands.
    """
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.startswith("IntraTM") and f.endswith("-00.xml")
    )
    files = [f for f in files if f >= start_xml]
    return [os.path.join(input_dir, f) for f in files]


def main():
    ap = argparse.ArgumentParser(
        description="Generate iperf3 client/server shell scripts for Géant TMs."
    )
    ap.add_argument(
        "--input-dir", required=True,
        help="Directory containing IntraTM-*.xml files "
             "(e.g. dataset/geant_traffic/traffic_generator/traffic-matrices-all)",
    )
    ap.add_argument(
        "--out-dir", required=True,
        help="Output root (e.g. dataset/geant_traffic/NEW_Geant_s3_perhour)",
    )
    ap.add_argument("--tm-scale", type=int, default=3)
    ap.add_argument("--duration", type=int, default=80000, help="iperf3 -t seconds")
    ap.add_argument(
        "--start-xml", default="IntraTM-2005-01-03-00-00.xml",
        help="First XML to include (sorted); matches Enero convert script default",
    )
    ap.add_argument(
        "--max-tm", type=int, default=None,
        help="Limit to first N TMs (handy for smoke test).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="List XMLs that would be processed and exit.",
    )
    args = ap.parse_args()

    xml_files = _collect_perhour_xmls(args.input_dir, start_xml=args.start_xml)
    if args.max_tm:
        xml_files = xml_files[:args.max_tm]

    print(f"[iperf3_geant] input_dir={args.input_dir}")
    print(f"[iperf3_geant] collected {len(xml_files)} hourly XMLs")
    if xml_files:
        print(f"[iperf3_geant] first: {os.path.basename(xml_files[0])}")
        print(f"[iperf3_geant] last : {os.path.basename(xml_files[-1])}")

    if args.dry_run:
        print("[iperf3_geant] --dry-run set; not writing anything.")
        return

    generate_traffic_from_xmls(
        xml_files, args.out_dir,
        duration=args.duration, tm_scale=args.tm_scale,
    )


if __name__ == "__main__":
    main()
