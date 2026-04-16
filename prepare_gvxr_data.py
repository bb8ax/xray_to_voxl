"""
prepare_gvxr_data.py  -  Auto-discover all views from gvxr directory structure
                          and convert to SAX-NeRF .pickle format.

Supports:
  - Auto-discover all available views (360 total)
  - Manual angle selection (e.g. --angles 0 45 -90)
  - Train/val split by ratio or explicit held-out angles
  - GT volume from .npy or voxelized .stl (requires stl_to_volume.py)

Usage:
    # Use all 360 views (300 train, 60 val)
    python prepare_gvxr_data.py --all

    # Use every N-th view (e.g. every 2nd = 180 views)
    python prepare_gvxr_data.py --all --step 2

    # Specific angles only
    python prepare_gvxr_data.py --angles 0 45 -90

    # With STL ground truth
    python prepare_gvxr_data.py --all --gt_stl path/to/model.stl

    # List available views
    python prepare_gvxr_data.py --list_available
"""

import os
import sys
import re
import glob
import pickle
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ============================================================
#  CONFIG  -  Edit these to match your setup
# ============================================================

# Base directory - adjust to your WSL mount or Linux path
BASE_DIR = "./Temp/TripoSR_0.06"

DEFAULT_SAMPLE = "000001_gvxr_processed"

# Scanner geometry (mm)
DSD = 80.0
DSO = 80.0

IMG_SIZE   = 256
OUTPUT_DIR = "data/custom"

# ============================================================
#  HELPERS
# ============================================================

def angle_to_dirname(angle: int) -> str:
    sign = "-" if angle < 0 else "+"
    return f"view_{sign}{abs(angle)}"


def dirname_to_angle(dirname: str):
    """Parse 'view_+45' or 'view_-90' -> 45 or -90, returns None if not parseable."""
    m = re.match(r"view_([+-])(\d+)$", dirname)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        return sign * int(m.group(2))
    return None


def discover_all_angles(base_dir: str, sample: str):
    """Return sorted list of (angle_int, full_image_path) for all valid view folders."""
    pattern = os.path.join(base_dir, sample, "view_*", "0","input.png")
    found = []
    for p in glob.glob(pattern):
        dirname = Path(p).parent.parent.name
        angle = dirname_to_angle(dirname)
        if angle is not None:
            angle += 180
            found.append((angle, p))
    found.sort(key=lambda x: x[0])
    return found


def load_png(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.rot90(arr, k=1)
    arr = 1.0 - arr   # INVERT: white background -> 0, dark bones -> bright
    return arr


def make_geometry(angles_rad, dsd, dso, n_det, d_det):
    return [{
        "DSD":         dsd,
        "DSO":         dso,
        "nDetector":   np.array([n_det, n_det]),
        "dDetector":   np.array([d_det, d_det]),
        "offDetector": np.array([0.0, 0.0]),
        "offOrigin":   np.array([0.0, 0.0, 0.0]),
        "nVoxel":      np.array([n_det, n_det, n_det]),
        "dVoxel":      np.array([d_det, d_det, d_det]),
        "sVoxel":      np.array([n_det * d_det] * 3),
        "angles":      np.array([a]),
    } for a in angles_rad]


def load_gt(gt_npy=None, gt_stl=None, dso=1000.0, img_size=256):
    """Load GT volume from .npy or voxelize from .stl."""
    if gt_npy and os.path.exists(gt_npy):
        print(f"Loading GT from {gt_npy} ...")
        gt = np.load(gt_npy).astype(np.float32)
    elif gt_stl and os.path.exists(gt_stl):
        print(f"Voxelizing STL from {gt_stl} ...")
        try:
            from stl_to_volume import stl_to_volume
            gt = stl_to_volume(gt_stl, volume_size=img_size, dso=dso)
        except ImportError:
            print("ERROR: stl_to_volume.py not found. Place it in ~/SAX-NeRF/ first.")
            return None
    else:
        return None

    # Normalize to [0, 1]
    if gt.min() < -0.5 or gt.max() > 1.5:
        gt = (gt - gt.min()) / (gt.max() - gt.min())
    print(f"  GT volume shape: {gt.shape},  range [{gt.min():.3f}, {gt.max():.3f}]")
    return gt


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # View selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true",
                       help="Use all discovered views")
    group.add_argument("--angles", type=int, nargs="+",
                       help="Specific angles in degrees, e.g. --angles 0 45 -90")

    parser.add_argument("--step", type=int, default=1,
                        help="When using --all, take every N-th view (e.g. --step 2 = 180 views)")
    parser.add_argument("--sample_id", type=str, default=DEFAULT_SAMPLE)
    parser.add_argument("--val_ratio", type=float, default=1/6,
                        help="Fraction of views reserved for validation (default: 1/6 ~ 60 views)")
    parser.add_argument("--list_available", action="store_true")

    # Ground truth
    parser.add_argument("--gt_npy", type=str, default=None,
                        help="Path to GT volume as .npy file")
    parser.add_argument("--gt_stl", type=str, default=None,
                        help="Path to GT mesh as .stl file")

    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    sample = args.sample_id

    # --- List mode ---
    if args.list_available:
        views = discover_all_angles(BASE_DIR, sample)
        print(f"\nAvailable views in '{sample}': {len(views)} total")
        for ang, path in views:
            print(f"  {angle_to_dirname(ang):12s}  {path}")
        return

    # --- Discover views ---
    all_views = discover_all_angles(BASE_DIR, sample)
    if not all_views:
        print(f"ERROR: No views found under {BASE_DIR}/{sample}/")
        print("Check BASE_DIR in the CONFIG section and run --list_available")
        sys.exit(1)

    if args.angles:
        # Manual selection
        angle_map = {a: p for a, p in all_views}
        selected = []
        for a in args.angles:
            if a not in angle_map:
                print(f"ERROR: Angle {a} not found. Run --list_available to see what exists.")
                sys.exit(1)
            selected.append((a, angle_map[a]))
    else:
        # Auto-select with optional step
        selected = all_views[::args.step]

    print(f"\nSample:      {sample}")
    print(f"Total views: {len(selected)}  (step={args.step})")

    # --- Train / val split ---
    n_total = len(selected)
    n_val   = max(1, int(round(n_total * args.val_ratio)))
    n_train = n_total - n_val

    # Spread val views evenly across the full angular range
    val_indices   = np.linspace(0, n_total - 1, n_val, dtype=int).tolist()
    train_indices = [i for i in range(n_total) if i not in val_indices]

    print(f"Train views: {n_train},  Val views: {n_val}")

    # --- Load images ---
    print("\nLoading images ...")
    angles_deg = np.array([a for a, _ in selected])
    paths      = [p for _, p in selected]
    projs      = np.stack([load_png(p, IMG_SIZE) for p in paths], axis=0)
    print(f"  projections shape: {projs.shape}  (N x H x W)")

    angles_rad = np.deg2rad(angles_deg)
    d_det = DSD / IMG_SIZE
    geos  = make_geometry(angles_rad, DSD, DSO, IMG_SIZE, d_det)

    def pack(indices):
        idx = np.array(indices)
        return {
            "projections": projs[idx],
            "angles":      angles_rad[idx],
        }


    # --- Ground truth ---
    gt = load_gt(gt_npy=args.gt_npy, gt_stl=args.gt_stl, dso=DSO, img_size=IMG_SIZE)
    if gt is not None:
        image = gt
    else:
        print("WARNING: No GT volume provided. Using blank volume - metrics will be meaningless.")
        image = np.zeros((IMG_SIZE, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    data = {
    # Geometry (top-level)
    "DSD":         DSD,
    "DSO":         DSO,
    "nDetector":   np.array([IMG_SIZE, IMG_SIZE]),
    "dDetector":   np.array([d_det, d_det]),
    "nVoxel":      np.array([IMG_SIZE, IMG_SIZE, IMG_SIZE]),
    "dVoxel":      np.array([d_det, d_det, d_det]),
    "offOrigin":   np.array([0.0, 0.0, 0.0]),
    "offDetector": np.array([0.0, 0.0]),
    "accuracy":    0.5,
    "mode":        "cone",
    "filter":      None,
    # Required count fields
    "numTrain":    n_train,
    "numVal":      n_val,
    # GT volume (required by both train and val dataset loaders)
    "image":       image,
    # Splits
    "train": pack(train_indices),
    "val":   pack(val_indices),
    }

    # --- Save pickle ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"all_step{args.step}" if (args.all or not args.angles) else \
          "_".join(str(a) for a in args.angles)
    out_path = args.output or os.path.join(OUTPUT_DIR, f"{sample}_{tag}.pickle")

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nSaved pickle: {out_path}")
    print()
    print("Suggested config changes (config/Lineformer/custom_360.yaml):")
    print(f"  datadir: {out_path}")
    print(f"  n_train: {n_train}")
    print(f"  epoch:   5000       # more views = needs more epochs")
    print(f"  lrate_step: 5000")


if __name__ == "__main__":
    main()
