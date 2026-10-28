"""Run DA3 models on the simulated views and compare with ground-truth depth.

Usage: KMP_DUPLICATE_LIB_OK=TRUE python scripts/eval_da3.py
Relative models are median-scaled to the ground truth; DA3METRIC uses depth = focal * out / 300.
"""

import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from depth_anything_3.api import DepthAnything3
from gpu import free_memory, pick_device, sync

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "data/sim"
OUT = ROOT / "results"
VIEWS = ["near", "mid", "far"]
MODELS = ["depth-anything/DA3-SMALL", "depth-anything/DA3-BASE", "depth-anything/DA3METRIC-LARGE"]
MAX_DEPTH = 20.0  # ignore sky / background beyond the field


def metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    ratio = np.maximum(p / g, g / p)
    return {
        "AbsRel": float(np.mean(np.abs(p - g) / g)),
        "RMSE": float(np.sqrt(np.mean((p - g) ** 2))),
        "d1": float(np.mean(ratio < 1.25)),
    }


def main():
    OUT.mkdir(exist_ok=True)
    K = json.loads((SIM / "intrinsics.json").read_text())
    device = pick_device()
    images = [cv2.cvtColor(cv2.imread(str(SIM / f"{v}_rgb.png")), cv2.COLOR_BGR2RGB) for v in VIEWS]
    gts = [np.load(SIM / f"{v}_depth.npy") for v in VIEWS]
    rows, preds = [], {}

    for name in MODELS:
        model = DepthAnything3.from_pretrained(name).to(device).eval()
        is_metric = "METRIC" in name
        model.inference([images[0]])  # warm-up
        for view, img, gt in zip(VIEWS, images, gts):
            sync(device)
            t0 = time.perf_counter()
            pred = model.inference([img])
            sync(device)
            ms = (time.perf_counter() - t0) * 1000
            raw = pred.depth[0]
            ph, pw = raw.shape
            if is_metric:
                focal = (K["fx"] + K["fy"]) / 2 * (pw / K["w"])  # focal at processed resolution
                raw = focal * raw / 300.0
            depth = cv2.resize(raw, (K["w"], K["h"]), interpolation=cv2.INTER_LINEAR)
            mask = (gt > 0) & (gt < MAX_DEPTH)
            scale = 1.0
            if not is_metric:
                scale = float(np.median(gt[mask]) / np.median(depth[mask]))
            m = metrics(depth * scale, gt, mask)
            rows.append({"model": name.split("/")[1], "view": view, "ms": round(ms), "proc": f"{pw}x{ph}",
                         "scale": round(scale, 3), **{k: round(v, 3) for k, v in m.items()}})
            preds[(name, view)] = depth * scale
            print(rows[-1])
        del model
        free_memory(device)

    (OUT / "sim_metrics.json").write_text(json.dumps(rows, indent=2))

    # Figure: RGB, GT, and each model per view
    fig, axes = plt.subplots(len(VIEWS), 2 + len(MODELS), figsize=(4 * (2 + len(MODELS)), 2.4 * len(VIEWS)))
    for r, (view, img, gt) in enumerate(zip(VIEWS, images, gts)):
        vmax = min(np.percentile(gt[gt < MAX_DEPTH], 99), MAX_DEPTH)
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(np.where(gt < MAX_DEPTH, gt, np.nan), cmap="turbo", vmin=0, vmax=vmax)
        for c, name in enumerate(MODELS):
            d = np.where(gt < MAX_DEPTH, preds[(name, view)], np.nan)
            axes[r, 2 + c].imshow(d, cmap="turbo", vmin=0, vmax=vmax)
        for c, title in enumerate(["RGB", "Ground truth"] + [n.split("/")[1] for n in MODELS]):
            axes[0, c].set_title(title)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "sim_comparison.png", dpi=110)
    print("saved", OUT / "sim_comparison.png")


if __name__ == "__main__":
    main()
