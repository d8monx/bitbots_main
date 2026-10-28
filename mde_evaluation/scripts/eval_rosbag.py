"""Run DA3 on frames extracted from a rosbag and compare with ground-plane depth on the field.

Usage: KMP_DUPLICATE_LIB_OK=TRUE python scripts/eval_rosbag.py data/frames/<bag>
Ground truth: for every field pixel, intersect its viewing ray with the ground (z = 0 in
base_footprint), using the camera pose from TF. Only green field pixels are evaluated, since
lines, ball, robots and background do not lie on (or are not) the ground plane.
Relative models are median-scaled on the field pixels (the ground-plane scale correction);
the metric model is reported raw and with the same correction.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from depth_anything_3.api import DepthAnything3
from gpu import free_memory, pick_device, sync
from eval_da3 import MAX_DEPTH, MODELS, metrics

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0, 1), (1, 3), (3, 6), (6, MAX_DEPTH)]


def ground_depth(T: np.ndarray, K: dict) -> np.ndarray:
    """z-depth of the ground plane for every pixel (NaN where the ray does not hit the ground)."""
    u, v = np.meshgrid(np.arange(K["w"]), np.arange(K["h"]))
    rays = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones_like(u, float)], -1)
    dz = rays @ T[:3, :3].T[:, 2]  # z component of each ray in the ground frame
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -T[2, 3] / dz  # ray length factor; equals z-depth because rays have z = 1
    return np.where((dz < 0) & (s < MAX_DEPTH), s, np.nan)


def field_mask(rgb: np.ndarray) -> np.ndarray:
    """Green field pixels, shrunk a little so edges of lines and objects are left out."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, (35, 80, 40), (85, 255, 255))
    return cv2.erode(green, np.ones((5, 5), np.uint8)) > 0


def predict(model, img, K, is_metric):
    pred = model.inference([img])
    raw = pred.depth[0]
    ph, pw = raw.shape
    if is_metric:
        focal = (K["fx"] + K["fy"]) / 2 * (pw / K["w"])  # focal at processed resolution
        raw = focal * raw / 300.0
    return cv2.resize(raw, (K["w"], K["h"]), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    args = parser.parse_args()
    out = ROOT / "results" / args.frames.name
    out.mkdir(parents=True, exist_ok=True)
    K = json.loads((args.frames / "intrinsics.json").read_text())
    frames = json.loads((args.frames / "frames.json").read_text())
    device = pick_device()

    images, gts, pitches = [], [], []
    for f in frames:
        img = cv2.cvtColor(cv2.imread(str(args.frames / f["file"])), cv2.COLOR_BGR2RGB)
        T = np.array(f["T_ground_camera"])
        gt = ground_depth(T, K)
        images.append(img)
        gts.append(np.where(field_mask(img), gt, np.nan))
        pitches.append(float(np.degrees(np.arcsin(-T[2, 2]))))  # how far the camera looks down

    rows, preds = [], {}
    for name in MODELS:
        short = name.split("/")[1]
        is_metric = "METRIC" in name
        model = DepthAnything3.from_pretrained(name).to(device).eval()
        model.inference([images[0]])  # warm-up
        times = []
        for i, (img, gt) in enumerate(zip(images, gts)):
            t0 = time.perf_counter()
            depth = predict(model, img, K, is_metric)
            times.append(time.perf_counter() - t0)
            mask = np.isfinite(gt)
            if mask.sum() < 500:
                continue  # hardly any field visible
            scale = float(np.median(gt[mask]) / np.median(depth[mask]))
            variants = {"ground-scaled": scale}
            if is_metric:
                variants["raw"] = 1.0
            for variant, s in variants.items():
                row = {"model": short, "variant": variant, "frame": frames[i]["file"], "pitch": round(pitches[i], 1),
                       "scale": round(scale, 3), "pixels": int(mask.sum()), **metrics(depth * s, gt, mask)}
                for lo, hi in BINS:
                    m = mask & (gt >= lo) & (gt < hi)
                    row[f"AbsRel_{lo}-{hi:g}m"] = metrics(depth * s, gt, m)["AbsRel"] if m.sum() > 100 else None
                rows.append(row)
            preds[(short, i)] = depth * scale
        print(f"{short}: {1000 * np.mean(times):.0f} ms per frame")
        del model
        free_memory(device)

    (out / "metrics.json").write_text(json.dumps(rows, indent=1))

    # Summary table: mean over frames per model and variant
    keys = ["AbsRel", "RMSE", "d1"] + [f"AbsRel_{lo}-{hi:g}m" for lo, hi in BINS]
    print(f"\n{'model':16} {'variant':14} " + " ".join(f"{k:>13}" for k in keys))
    for short in [n.split("/")[1] for n in MODELS]:
        for variant in ["raw", "ground-scaled"]:
            sel = [r for r in rows if r["model"] == short and r["variant"] == variant]
            if not sel:
                continue
            vals = [np.mean([r[k] for r in sel if r[k] is not None]) if any(r[k] is not None for r in sel) else np.nan
                    for k in keys]
            print(f"{short:16} {variant:14} " + " ".join(f"{v:13.3f}" for v in vals))

    # Figure 1: a few frames with RGB, ground-plane depth and each model (ground-scaled)
    shown = [i for i in np.linspace(0, len(frames) - 1, 4).astype(int) if (MODELS[0].split("/")[1], i) in preds]
    shorts = [n.split("/")[1] for n in MODELS]
    fig, axes = plt.subplots(len(shown), 2 + len(MODELS), figsize=(4 * (2 + len(MODELS)), 2.4 * len(shown)))
    for r, i in enumerate(shown):
        gt = gts[i]
        vmax = np.nanpercentile(gt, 99)
        axes[r, 0].imshow(images[i])
        axes[r, 0].set_ylabel(frames[i]["file"])
        axes[r, 1].imshow(gt, cmap="turbo", vmin=0, vmax=vmax)
        for c, short in enumerate(shorts):
            axes[r, 2 + c].imshow(np.where(np.isfinite(gt), preds[(short, i)], np.nan), cmap="turbo", vmin=0, vmax=vmax)
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])
    for c, title in enumerate(["RGB", "Ground plane (field only)"] + [s + " (ground-scaled)" for s in shorts]):
        axes[0, c].set_title(title)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=100)

    # Figure 2: signed relative error per pixel, (model - ground plane) / ground plane.
    # Red = model too far, blue = model too near, white = correct or not evaluated (not field).
    fig, axes = plt.subplots(len(shown), 1 + len(MODELS), figsize=(4 * (1 + len(MODELS)) + 1.5, 2.4 * len(shown)),
                             layout="constrained")
    fig.suptitle("Error vs. ground plane (ground-scaled, field pixels only)")
    for r, i in enumerate(shown):
        axes[r, 0].imshow(images[i])
        axes[r, 0].set_ylabel(frames[i]["file"])
        for c, short in enumerate(shorts):
            err = (preds[(short, i)] - gts[i]) / gts[i]
            im = axes[r, 1 + c].imshow(err, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])
    for c, title in enumerate(["RGB"] + shorts):
        axes[0, c].set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label="red: too far, blue: too near", format="{x:+.0%}")
    fig.savefig(out / "error_maps.png", dpi=100)

    # Figure 3: signed relative error over distance, all frames and field pixels.
    # A constant offset means a pure scale error; a slope means the error grows with distance.
    edges = np.arange(0.0, MAX_DEPTH + 0.5, 0.5)
    fig, ax = plt.subplots(figsize=(7, 4))
    for short in shorts:
        gt_all = np.concatenate([gts[i][np.isfinite(gts[i])] for i in range(len(gts)) if (short, i) in preds])
        pr_all = np.concatenate([preds[(short, i)][np.isfinite(gts[i])] for i in range(len(gts)) if (short, i) in preds])
        err = (pr_all - gt_all) / gt_all
        idx = np.digitize(gt_all, edges)
        centers, med, q25, q75 = [], [], [], []
        for b in range(1, len(edges)):
            e = err[idx == b]
            if e.size > 1000:  # skip distances with too few pixels
                centers.append((edges[b - 1] + edges[b]) / 2)
                med.append(np.median(e)), q25.append(np.percentile(e, 25)), q75.append(np.percentile(e, 75))
        line, = ax.plot(centers, med, marker="o", ms=3, label=short)
        ax.fill_between(centers, q25, q75, color=line.get_color(), alpha=0.15)
    ax.axhline(0.0, color="gray", lw=1)
    ax.set_xlabel("ground-plane depth [m]")
    ax.set_ylabel("relative error (median, band: middle 50 %)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_title("Ground-scaled models vs. ground plane")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_vs_distance.png", dpi=100)

    # Figure 4: how far off is the raw metric model, depending on how steeply the camera looks down
    raw = [r for r in rows if r["variant"] == "raw"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r["pitch"] for r in raw], [1 / r["scale"] for r in raw])
    ax.axhline(1.0, color="gray", lw=1)
    ax.set_xlabel("camera looks down [deg]")
    ax.set_ylabel("metric depth / ground-plane depth (median)")
    ax.set_title(raw[0]["model"] if raw else "")
    fig.tight_layout()
    fig.savefig(out / "metric_scale_vs_pitch.png", dpi=100)
    print("saved", out)


if __name__ == "__main__":
    main()
