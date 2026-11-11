"""Quick look: ZED depth (several modes) vs. DA3 vs. ground-plane depth on one frame per bag.

Usage: python scripts/compare_zed.py data/frames/<bag_a> data/frames/<bag_b> ... [--frame 5]
(run extract_rosbag.py on the bags first). Writes results/compare_zed.png and prints per bag
how much of the image / of the field the ZED fills and its error against the ground plane.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from depth_anything_3.api import DepthAnything3
from PIL import Image

from eval_rosbag import ground_depth, predict
from gpu import pick_device

ROOT = Path(__file__).resolve().parents[1]
DA3 = "depth-anything/DA3METRIC-LARGE"
VMAX = 8.0  # colour scale in metres
WALL = 4.775  # measured 04.11.: wall opposite the robot, from the point below the camera


def abs_rel(pred, gt, mask):
    m = mask & np.isfinite(pred) & np.isfinite(gt) & (pred > 0)
    return float(np.mean(np.abs(pred[m] - gt[m]) / gt[m])) if m.any() else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path, nargs="+")
    parser.add_argument("--frame", type=int, default=5)
    args = parser.parse_args()

    model = DepthAnything3.from_pretrained(DA3).to(pick_device()).eval()
    fig, axes = plt.subplots(len(args.frames), 4, figsize=(16, 2.6 * len(args.frames)), squeeze=False)
    for row, folder in enumerate(args.frames):
        K = json.loads((folder / "intrinsics.json").read_text())
        f = json.loads((folder / "frames.json").read_text())[args.frame]
        rgb = np.array(Image.open(folder / f["file"]).convert("RGB"))
        zed = np.load(folder / f["zed_depth"])
        gt = ground_depth(np.array(f["T_ground_camera"]), K)
        da3 = predict(model, rgb, K, is_metric=True)
        # floor pixels in front of the wall (colour mask from the simulation does not fit the darker real carpet)
        field = np.isfinite(gt) & (gt < WALL - 0.3)
        wall_rows = np.isfinite(gt) & (gt >= WALL)  # rays that would hit the floor behind the wall -> wall
        wall = np.nanmedian(zed[wall_rows]) if wall_rows.any() else np.nan
        wall_da3 = np.nanmedian(da3[wall_rows]) if wall_rows.any() else np.nan

        valid = np.isfinite(zed) & (zed > 0)
        print(f"{folder.name:34s} ZED gültig: Bild {valid.mean():5.1%}, Rasen {valid[field].mean():5.1%} | "
              f"AbsRel Rasen: ZED {abs_rel(zed, gt, field):.3f}, DA3-Metric {abs_rel(da3, gt, field):.3f} | "
              f"Wand (gemessen {WALL} m): ZED {wall:.2f}, DA3 {wall_da3:.2f}")

        for col, (img, title) in enumerate([(rgb, folder.name), (zed, "ZED"), (da3, "DA3 Metric-Large"),
                                            (np.where(gt < WALL, gt, np.nan), "Boden aus Kamerapose (bis zur Wand)")]):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(img)
            else:
                ax.imshow(np.where(np.isfinite(img), img, np.nan), cmap="turbo", vmin=0, vmax=VMAX)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, VMAX))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Tiefe [m] (weiß = kein Wert)")
    out = ROOT / "results/compare_zed.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
