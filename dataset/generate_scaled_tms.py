"""
Generate scaled TM versions from existing NEW_Geant (scale=5) demands.

Usage:
    python dataset/generate_scaled_tms.py --scales 3 4
    python dataset/generate_scaled_tms.py --scales 3    # just scale=3
"""
import os, sys, argparse, shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

BASE_DIR = os.path.join(
    PROJECT_DIR,
    "A-Traffic-Engineering-Method-Using-RouteNet-Based-Actor-Critic-Learning-in-SDN-Routing-main",
    "Enero_datasets", "dataset_sing_top", "data",
    "results_my_3_tops_unif_05-1"
)
SOURCE_NAME = "NEW_Geant"       # scale=5 (original)
SOURCE_SCALE = 5
SUBFOLDERS = ["TRAIN", "EVALUATE", "ALL"]


def scale_demands_file(src_path, dst_path, ratio):
    """Read a .demands file, scale all bw values by ratio, write to dst."""
    with open(src_path) as f:
        lines = f.readlines()

    with open(dst_path, "w") as f:
        for i, line in enumerate(lines):
            if i < 2:  # header lines (DEMANDS N, label src dest bw)
                f.write(line)
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                f.write(line)
                continue
            # parts: [label, src, dst, bw]
            bw = float(parts[3]) * ratio
            f.write(f"{parts[0]} {parts[1]} {parts[2]} {bw:.6f}\n")


def create_scaled_dataset(target_scale):
    ratio = target_scale / SOURCE_SCALE
    target_name = f"NEW_Geant_s{target_scale}"
    source_dir = os.path.join(BASE_DIR, SOURCE_NAME)
    target_dir = os.path.join(BASE_DIR, target_name)

    print(f"\n{'='*60}")
    print(f"Creating {target_name} (scale={target_scale}, ratio={ratio:.2f})")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"{'='*60}")

    for subfolder in SUBFOLDERS:
        src_sub = os.path.join(source_dir, subfolder)
        dst_sub = os.path.join(target_dir, subfolder)

        if not os.path.exists(src_sub):
            print(f"  [SKIP] {subfolder}/ not found in source")
            continue

        os.makedirs(dst_sub, exist_ok=True)

        # Symlink topology files (graph, k_shortest_paths)
        for fname in ["Geant.graph", "k_shortest_paths.json"]:
            src_file = os.path.join(src_sub, fname)
            dst_file = os.path.join(dst_sub, fname)
            if os.path.exists(src_file) and not os.path.exists(dst_file):
                os.symlink(src_file, dst_file)
                print(f"  [LINK] {subfolder}/{fname}")

        # Scale TM files
        src_tm = os.path.join(src_sub, "TM")
        dst_tm = os.path.join(dst_sub, "TM")
        if not os.path.exists(src_tm):
            print(f"  [SKIP] {subfolder}/TM/ not found")
            continue

        os.makedirs(dst_tm, exist_ok=True)
        tm_files = [f for f in os.listdir(src_tm) if f.endswith(".demands")]
        print(f"  [SCALE] {subfolder}/TM/: {len(tm_files)} files x {ratio:.2f}")

        for tm_file in tm_files:
            scale_demands_file(
                os.path.join(src_tm, tm_file),
                os.path.join(dst_tm, tm_file),
                ratio
            )

    print(f"  [DONE] {target_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", type=int, required=True,
                        help="Target scale values (e.g., 3 4)")
    args = parser.parse_args()

    for s in args.scales:
        if s == SOURCE_SCALE:
            print(f"Scale={s} is the source scale, skipping")
            continue
        create_scaled_dataset(s)

    print(f"\nAll done. Available datasets:")
    for name in sorted(os.listdir(BASE_DIR)):
        if name.startswith("NEW_Geant"):
            full = os.path.join(BASE_DIR, name)
            print(f"  {name}/")


if __name__ == "__main__":
    main()
