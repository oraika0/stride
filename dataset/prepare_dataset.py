"""Generate the iperf3 client/server scripts Mininet replays as traffic.

Reads the pickled traffic matrices under <topo>_traffic/traffic_generator/ and
writes one directory per TM, each holding the per-host shell scripts the real
environment runs. Everything it touches lives under dataset/.

    python dataset/prepare_dataset.py --topology 32node --tms 144tm
    python dataset/prepare_dataset.py --topology 32node --tms 24tm
    python dataset/prepare_dataset.py --topology geant  --tms 24tm --tm_scale 3

--tm_scale multiplies every demand and only applies to geant. The default 5 is
the original scale and writes 23node/; any other value writes 23node_s<scale>/,
which is what the env configs point at (the paper uses 3).
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from utils.iperf3_scripts import generate_traffic as generate_32node   # noqa: E402
from utils.iperf3_geant import generate_traffic as generate_geant      # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', type=str, required=True, help='Input topology name')
    parser.add_argument('--tms', type=str, required=True, help='Output number of tms')
    parser.add_argument('--duration', type=int, default=80000, help='Time duration per flow')
    parser.add_argument('--tm_scale', type=int, default=5,
                        help='Traffic matrix scale multiplier (geant only; default 5, paper uses 3)')
    args = parser.parse_args()

    topo_dir = os.path.join(SCRIPT_DIR, f"{args.topology}_traffic")

    if args.topology == "32node":
        pkl_file = os.path.join(topo_dir, "traffic_generator",
                                f"{args.topology}_tms_info_{args.tms}.pkl")
        out_path = os.path.join(topo_dir, f"{args.topology}_{args.tms}")
        generate_32node(pkl_file, out_path, args.duration)
    elif args.topology == "geant":
        data_file = os.path.join(topo_dir, "traffic_generator")
        suffix = "" if args.tm_scale == 5 else f"_s{args.tm_scale}"
        out_path = os.path.join(topo_dir, f"23node{suffix}")
        print(f"[prepare_dataset] tm_scale={args.tm_scale} -> {out_path}")
        generate_geant(data_file, out_path, args.duration, tm_scale=args.tm_scale)
    else:
        raise SystemExit(f"Unknown topology {args.topology!r}; expected '32node' or 'geant'.")
